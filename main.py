"""
Консольный биддер для Яндекс Директа.

Стратегии (config.yaml → strategy):
  * scalping — адаптивный скальпинг: поднимаем ставку шагами,
               которые растут, пока current < entry (не выиграли аукцион).
               Как только current >= entry — замораживаем.
               Если entry вырастет выше current — возобновляем скальпинг.
  * rules    — обычные правила из config.yaml.

CLI:
  python bidder.py            — обычный запуск
  python bidder.py --reset    — сбросить ставки к scalping.start_bid
  python bidder.py --status   — показать текущие ставки и состояние
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# =============================================================================
# Константы
# =============================================================================

BID_MULTIPLIER = 1_000_000
PROD_BASE = "https://api.direct.yandex.com/json/v501"
SANDBOX_BASE = "https://api-sandbox.direct.yandex.com/json/v501"
CONFIG_FILE_DEFAULT = "config.yaml"

AUTH_ERROR_CODES = {53, 54, 55, 56, 58}
AUTH_HINTS = {
    53: "Недействительный OAuth-токен — проверьте YANDEX_DIRECT_TOKEN в .env",
    54: "Токен отозван — выпустите новый",
    55: "Недостаточно прав — добавьте скоуп «Яндекс Директ»",
    56: "Доступ к API не выдан — подайте заявку в кабинете Директа",
    58: "Токен не подходит для режима (sandbox/prod) — проверьте mode.sandbox",
}
RATE_LIMIT_STATUS_CODE = 429
PLACEHOLDER_TOKENS = {
    "", "your_oauth_token_here", "changeme", "token", "put_your_token_here",
}


# =============================================================================
# Ошибки
# =============================================================================

class DirectApiError(Exception):
    def __init__(
        self,
        code: int,
        message: str,
        details: str = "",
        request_id: str | None = None,
        retry_after: int | None = None,
    ):
        super().__init__(f"[{code}] {message} {details}".strip())
        self.code = code
        self.message = message
        self.details = details
        self.request_id = request_id
        self.retry_after = retry_after

    @property
    def is_auth_error(self) -> bool:
        return self.code in AUTH_ERROR_CODES

    @property
    def is_rate_limit_error(self) -> bool:
        return self.code == RATE_LIMIT_STATUS_CODE

    @property
    def hint(self) -> str | None:
        return AUTH_HINTS.get(self.code)


class ConfigError(Exception): ...
class RulesError(Exception): ...


# =============================================================================
# ДВИЖОК ПРАВИЛ (для strategy: rules)
# =============================================================================

def _to_decimal(x: Any) -> Decimal:
    return Decimal(str(x))


OPERATORS: dict[str, Any] = {
    "lt": lambda a, b: a < _to_decimal(b),
    "lte": lambda a, b: a <= _to_decimal(b),
    "gt": lambda a, b: a > _to_decimal(b),
    "gte": lambda a, b: a >= _to_decimal(b),
    "eq": lambda a, b: a == _to_decimal(b),
    "ne": lambda a, b: a != _to_decimal(b),
    "between": lambda a, b: _to_decimal(b[0]) <= a <= _to_decimal(b[1]),
}


@dataclass
class Condition:
    field: str
    op: str
    value: Any

    def check(self, ctx: dict) -> bool:
        actual = ctx.get(self.field)
        if self.op == "is_null":
            return actual is None
        if self.op == "not_null":
            return actual is not None
        if actual is None:
            return False
        if not isinstance(actual, Decimal):
            try:
                actual = _to_decimal(actual)
            except Exception:
                return False
        try:
            return OPERATORS[self.op](actual, self.value)
        except Exception as exc:  # noqa: BLE001
            raise RulesError(
                f"Ошибка оператора '{self.op}' для поля '{self.field}': {exc}"
            ) from exc


@dataclass
class Rule:
    name: str
    enabled: bool
    conditions: list[Condition]
    action_type: str
    action_params: dict

    def matches(self, ctx: dict) -> bool:
        return self.enabled and all(c.check(ctx) for c in self.conditions)

    def apply(self, ctx: dict) -> tuple[Decimal, str]:
        entry = ctx.get("premium_entry_price")
        current = ctx.get("current_bid")
        p = self.action_params

        if self.action_type == "fixed":
            return _to_decimal(p["value"]), f"{self.name}: fixed={p['value']}"

        if self.action_type == "entry_plus":
            if entry is None:
                return _to_decimal(p.get("fallback", 100)), f"{self.name}: entry=None → fallback"
            return entry + _to_decimal(p["add"]), f"{self.name}: entry({entry}) + {p['add']}"

        if self.action_type == "entry_minus":
            if entry is None:
                return _to_decimal(p.get("fallback", 100)), f"{self.name}: entry=None → fallback"
            return entry - _to_decimal(p["subtract"]), f"{self.name}: entry({entry}) − {p['subtract']}"

        if self.action_type == "entry_multiply":
            if entry is None:
                return _to_decimal(p.get("fallback", 100)), f"{self.name}: entry=None → fallback"
            return entry * _to_decimal(p["factor"]), f"{self.name}: entry({entry}) × {p['factor']}"

        if self.action_type == "current_plus":
            if current is None:
                return _to_decimal(p.get("fallback", 100)), f"{self.name}: current=None → fallback"
            return current + _to_decimal(p["add"]), f"{self.name}: current({current}) + {p['add']}"

        raise RulesError(f"Правило '{self.name}': неизвестный тип действия '{self.action_type}'")


class RulesEngine:
    def __init__(self, rules: list[Rule], defaults: dict, source: str = ""):
        self.rules = rules
        self.defaults = defaults
        self.source = source

    @classmethod
    def from_dict(cls, raw: dict | None, source: str = "") -> "RulesEngine":
        raw = raw or {}
        rules_raw = raw.get("rules") or []
        if not isinstance(rules_raw, list) or not rules_raw:
            raise RulesError("В секции 'rules.rules' нет ни одного правила")

        rules: list[Rule] = []
        seen: set[str] = set()
        for i, r in enumerate(rules_raw):
            if not isinstance(r, dict):
                raise RulesError(f"Правило #{i}: ожидался объект")
            name = r.get("name")
            if not name:
                raise RulesError(f"Правило #{i}: не указано 'name'")
            if name in seen:
                raise RulesError(f"Дубликат имени правила: '{name}'")
            seen.add(name)

            then = r.get("then")
            if not isinstance(then, dict) or "type" not in then:
                raise RulesError(f"Правило '{name}': не указано 'then.type'")

            conditions: list[Condition] = []
            when = r.get("when") or {}
            if not isinstance(when, dict):
                raise RulesError(f"Правило '{name}': 'when' должен быть объектом")
            for field_name, cond in when.items():
                if cond is None:
                    conditions.append(Condition(field_name, "is_null", None))
                elif isinstance(cond, dict):
                    for op, value in cond.items():
                        if op not in OPERATORS:
                            raise RulesError(
                                f"Правило '{name}': неизвестный оператор '{op}'. "
                                f"Допустимые: {sorted(OPERATORS.keys())}"
                            )
                        conditions.append(Condition(field_name, op, value))
                else:
                    conditions.append(Condition(field_name, "eq", cond))

            action_type = then["type"]
            action_params = {k: v for k, v in then.items() if k != "type"}
            rules.append(Rule(name, bool(r.get("enabled", True)),
                              conditions, action_type, action_params))

        defaults = raw.get("defaults") or {"bid": 100, "reason": "default"}
        if "bid" not in defaults:
            raise RulesError("В 'rules.defaults' должен быть ключ 'bid'")
        return cls(rules, defaults, source=source)

    def compute(self, *, premium_entry_price: Decimal | None,
                current_bid: Decimal | None, **extra: Any) -> tuple[Decimal, str]:
        ctx: dict[str, Any] = {
            "premium_entry_price": premium_entry_price,
            "current_bid": current_bid,
            **extra,
        }
        for rule in self.rules:
            if rule.matches(ctx):
                return rule.apply(ctx)
        return _to_decimal(self.defaults["bid"]), self.defaults.get("reason", "default")

    def describe(self) -> str:
        lines = [f"Правила из {self.source or '<dict>'}:"]
        for r in self.rules:
            flag = "✓" if r.enabled else "✗"
            lines.append(f"  {flag} {r.name} → {r.action_type} {r.action_params}")
        return "\n".join(lines)


# =============================================================================
# ХРАНИЛИЩЕ СОСТОЯНИЯ
# =============================================================================

class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "campaigns": {}}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("корень не объект")
            raw.setdefault("version", 1)
            raw.setdefault("campaigns", {})
            return raw
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning(f"Не удалось прочитать {self.path}: {exc}. Начинаем с пустого.")
            return {"version": 1, "campaigns": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def get_step_n(self, campaign_id: int, keyword_id: int) -> int:
        try:
            return int(self.data["campaigns"][str(campaign_id)]["keywords"][str(keyword_id)]["step_n"])
        except (KeyError, TypeError, ValueError):
            return 0

    def set_step_n(self, campaign_id: int, keyword_id: int, step_n: int) -> None:
        c = self.data.setdefault("campaigns", {}).setdefault(str(campaign_id), {"keywords": {}})
        kw = c.setdefault("keywords", {}).setdefault(str(keyword_id), {})
        kw["step_n"] = step_n
        kw["updated_at"] = datetime.now().isoformat(timespec="seconds")

    def reset_campaign(self, campaign_id: int) -> None:
        self.data.setdefault("campaigns", {}).pop(str(campaign_id), None)


# =============================================================================
# СТРАТЕГИЯ SCALPING
# =============================================================================

@dataclass
class ScalpingConfig:
    start_bid: Decimal
    base_step: Decimal
    step_increment: Decimal
    max_step: Decimal


@dataclass
class Limits:
    max_bid: Decimal
    min_bid: Decimal | None
    no_downgrade: bool


def scalp_step(
    current: Decimal | None,
    entry: Decimal | None,
    step_n: int,
    cfg: ScalpingConfig,
    limits: Limits,
) -> tuple[Decimal | None, str, int]:
    if entry is None:
        return current, "нет цены входа → пропускаем", step_n

    if current is None:
        target = cfg.start_bid
        if target > limits.max_bid:
            target = limits.max_bid
        return target, f"нет ставки → start_bid={cfg.start_bid}", 0

    if current >= entry:
        return current, f"уже выигрываем (current={current} ≥ entry={entry})", step_n

    if current >= limits.max_bid:
        return (
            current,
            f"current={current} ≥ max_bid={limits.max_bid}; entry={entry} недостижим — стоим",
            step_n,
        )

    step = cfg.base_step + Decimal(step_n) * cfg.step_increment
    if step > cfg.max_step:
        step = cfg.max_step
    if step <= 0:
        step = cfg.base_step

    target = current + step
    reason = f"шаг #{step_n}: current={current} + step={step}"

    if target > entry:
        target = entry
        reason += f" → cap entry={entry}"
    if target > limits.max_bid:
        target = limits.max_bid
        reason += f" → cap max_bid={limits.max_bid}"

    if target <= current:
        return current, f"{reason} → не двигаем (target ≤ current)", step_n + 1

    return target, f"{reason} → {target}", step_n + 1


# =============================================================================
# НАСТРОЙКИ
# =============================================================================
#
# ВАЖНО: поля с дефолтами (default=...) идут СТРОГО ПОСЛЕ полей без дефолтов.
# Поэтому секреты и опциональные структуры (telegram, scalping, rules_engine)
# вынесены в самый низ dataclass-а.
# =============================================================================

@dataclass(frozen=True)
class Settings:
    # --- обязательные, без дефолтов ---
    token: str = field(repr=False)
    api_base: str
    dry_run: bool
    campaign_id: int
    client_login: str | None
    max_concurrent: int
    max_keywords_per_batch: int
    request_timeout: int
    retry_count: int
    log_file: str
    log_level: str
    state_file: str
    strategy: str
    limits: Limits

    # --- опциональные, с дефолтами (только в конце) ---
    telegram_bot_token: str | None = field(default=None, repr=False)
    telegram_chat_id: str | None = field(default=None, repr=False)
    scalping: ScalpingConfig | None = field(default=None, repr=False)
    rules_engine: RulesEngine | None = field(default=None, repr=False)

    @classmethod
    def load(cls, config_path: str = CONFIG_FILE_DEFAULT) -> "Settings":
        load_dotenv()
        token = os.environ.get("YANDEX_DIRECT_TOKEN", "").strip()
        tg_bot = os.environ.get("TELEGRAM_BOT_TOKEN") or None
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID") or None

        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Не найден файл конфигурации: {path.resolve()}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"Синтаксическая ошибка YAML в {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: ожидался объект верхнего уровня")

        mode = raw.get("mode") or {}
        campaign = raw.get("campaign") or {}
        api = raw.get("api") or {}
        logging_cfg = raw.get("logging") or {}
        state_cfg = raw.get("state") or {}
        limits_cfg = raw.get("limits") or {}
        strategy = str(raw.get("strategy", "rules")).strip().lower()

        if strategy not in ("scalping", "rules"):
            raise ConfigError(f"strategy должна быть 'scalping' или 'rules', а не '{strategy}'")
        if "id" not in campaign:
            raise ConfigError(f"{path}: не указан campaign.id")

        limits = Limits(
            max_bid=_to_decimal(limits_cfg.get("max_bid", 150)),
            min_bid=_to_decimal(limits_cfg["min_bid"]) if limits_cfg.get("min_bid") is not None else None,
            no_downgrade=bool(limits_cfg.get("no_downgrade", True)),
        )

        scalping: ScalpingConfig | None = None
        rules_engine: RulesEngine | None = None

        if strategy == "scalping":
            sc = raw.get("scalping") or {}
            if "start_bid" not in sc:
                raise ConfigError("strategy=scalping, но в config.yaml нет 'scalping.start_bid'")
            scalping = ScalpingConfig(
                start_bid=_to_decimal(sc["start_bid"]),
                base_step=_to_decimal(sc.get("base_step", 5)),
                step_increment=_to_decimal(sc.get("step_increment", 0)),
                max_step=_to_decimal(sc.get("max_step", 1000)),
            )
            if scalping.base_step <= 0:
                raise ConfigError("scalping.base_step должен быть > 0")
            if scalping.start_bid > limits.max_bid:
                raise ConfigError(
                    f"scalping.start_bid ({scalping.start_bid}) > limits.max_bid ({limits.max_bid})"
                )
        else:
            rules_engine = RulesEngine.from_dict(raw.get("rules"), source=str(path))

        sandbox = bool(mode.get("sandbox", True))
        api_base = SANDBOX_BASE if sandbox else PROD_BASE

        return cls(
            token=token,
            api_base=api_base,
            dry_run=bool(mode.get("dry_run", True)),
            campaign_id=int(campaign["id"]),
            client_login=campaign.get("client_login") or None,
            max_concurrent=int(api.get("max_concurrent", 5)),
            max_keywords_per_batch=int(api.get("max_keywords_per_batch", 50)),
            request_timeout=int(api.get("timeout", 30)),
            retry_count=int(api.get("retry_count", 3)),
            log_file=logging_cfg.get("file", "logs/bidder.log"),
            log_level=str(logging_cfg.get("level", "INFO")).upper(),
            state_file=state_cfg.get("file", "state.json"),
            strategy=strategy,
            limits=limits,
            telegram_bot_token=tg_bot,
            telegram_chat_id=tg_chat,
            scalping=scalping,
            rules_engine=rules_engine,
        )

    def describe(self) -> str:
        mode = "sandbox" if "sandbox" in self.api_base else "PROD"
        return (
            f"Режим: {mode}, dry_run={self.dry_run}, "
            f"strategy={self.strategy}, campaign_id={self.campaign_id}"
        )


# =============================================================================
# ЛОГИРОВАНИЕ
# =============================================================================

def setup_logger(log_file: str, level: str) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr, level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    logger.add(
        log_file, level=level, rotation="10 MB", retention="14 days",
        compression="zip", encoding="utf-8", enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


# =============================================================================
# УВЕДОМЛЕНИЯ
# =============================================================================

class TelegramNotifier:
    def __init__(self, bot_token: str | None, chat_id: str | None, timeout: int = 10):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, text: str) -> None:
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            requests.post(
                url,
                json={"chat_id": self.chat_id, "text": text[:4000], "parse_mode": "HTML"},
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Telegram-уведомление не отправлено: {exc}")


# =============================================================================
# КЛИЕНТ API
# =============================================================================

class DirectClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {settings.token}",
            "Accept-Language": "ru",
            "Content-Type": "application/json; charset=utf-8",
        })
        if settings.client_login:
            self.session.headers["Client-Login"] = settings.client_login
        self._semaphore = threading.Semaphore(settings.max_concurrent)
        self.last_units: dict[str, int] = {}
        self._units_limit_reached = False

    def check_units_limit(self) -> tuple[bool, int | None]:
        """
        Проверяет, не достигнут ли лимит единиц API.
        Возвращает (is_limit_reached, units_rest).
        """
        units_limit = self.last_units.get("Units-Limit")
        units_rest = self.last_units.get("Units-Rest")
        
        if units_limit is None or units_rest is None:
            return False, None
        
        # Если осталось меньше 10% единиц, считаем что лимит близок
        threshold = max(1, units_limit // 10)
        is_near_limit = units_rest <= threshold
        
        return is_near_limit, units_rest

    def _do_post(self, service: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.api_base}/{service}"
        logger.debug(f"POST {url} → {json.dumps(payload, ensure_ascii=False)[:500]}")
        with self._semaphore:
            response = self.session.post(url, json=payload, timeout=self.settings.request_timeout)

        # Чтение заголовков лимитов API
        for header in ("Units", "Units-Used", "Units-Rest", "Units-Limit"):
            value = response.headers.get(header)
            if value is None:
                continue
            try:
                self.last_units[header] = int(value)
            except ValueError:
                pass

        # Проверка на превышение лимита запросов (429 Too Many Requests)
        if response.status_code == RATE_LIMIT_STATUS_CODE:
            retry_after = None
            retry_after_header = response.headers.get("Retry-After")
            if retry_after_header:
                try:
                    retry_after = int(retry_after_header)
                except ValueError:
                    pass
            raise DirectApiError(
                code=RATE_LIMIT_STATUS_CODE,
                message="Превышен лимит запросов (Too Many Requests)",
                details=f"Retry-After: {retry_after}",
                retry_after=retry_after,
            )

        if response.status_code >= 500:
            raise requests.RequestException(
                f"Сервер вернул {response.status_code}: {response.text[:200]}"
            )
        if response.status_code >= 400:
            raise DirectApiError(
                code=response.status_code,
                message=f"HTTP-ошибка {response.status_code}",
                details=response.text[:300],
            )
        data = response.json()
        if "error" in data:
            err = data["error"]
            raise DirectApiError(
                code=int(err.get("error_code", 0)),
                message=err.get("error_string", "Unknown error"),
                details=err.get("error_detail", ""),
                request_id=err.get("request_id"),
            )
        return data

    def _request(self, service: str, payload: dict[str, Any]) -> dict[str, Any]:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.settings.retry_count),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((requests.RequestException, DirectApiError)),
        )
        def _wrapped() -> dict[str, Any]:
            return self._do_post(service, payload)
        return _wrapped()

    def _request_with_pagination(
        self,
        service: str,
        method: str,
        selection_criteria: dict[str, Any],
        field_names: list[str],
    ) -> list[dict[str, Any]]:
        """Выполняет запрос с поддержкой пагинации (Offset/Limit)."""
        all_results: list[dict[str, Any]] = []
        limit = self.settings.max_keywords_per_batch
        offset = 0

        while True:
            payload = {
                "method": method,
                "params": {
                    "SelectionCriteria": selection_criteria,
                    "FieldNames": field_names,
                    "Limit": limit,
                    "Offset": offset,
                },
            }
            data = self._request(service, payload)
            items = data.get("result", {}).get(method == "get" and service in ("keywords", "bids", "ads", "adgroups", "campaigns") and {"keywords": "Keywords", "bids": "Bids", "ads": "Ads", "adgroups": "AdGroups", "campaigns": "Campaigns"}.get(service, "Items") or "Items", [])
            
            if not items:
                break
            
            all_results.extend(items)
            
            if len(items) < limit:
                break
            
            offset += limit
            logger.debug(f"Пагинация: получено {len(items)}, продолжаем с offset={offset}")

        return all_results

    def check_campaigns(self, timestamp: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"method": "checkCampaigns", "params": {}}
        if timestamp:
            payload["params"]["Timestamp"] = timestamp
        return self._request("changes", payload)

    def get_bids(self, campaign_id: int) -> list[dict[str, Any]]:
        """Получает ставки для кампании с поддержкой пагинации."""
        selection_criteria = {"CampaignIds": [campaign_id]}
        field_names = ["KeywordId", "Bid", "ContextBid"]
        
        # Используем пагинацию для получения всех ставок
        return self._request_with_pagination("bids", "get", selection_criteria, field_names)

    def get_auction_prices(self, keyword_ids: list[int]) -> dict[int, Decimal | None]:
        result: dict[int, Decimal | None] = {}
        batch = self.settings.max_keywords_per_batch
        
        for start in range(0, len(keyword_ids), batch):
            chunk = keyword_ids[start:start + batch]
            payload = {
                "method": "get",
                "params": {
                    "SelectionCriteria": {"Ids": chunk},
                    "FieldNames": ["Id", "PremiumEntryPrice", "PremiumBid"],
                },
            }
            data = self._request("keywords", payload)
            
            # Валидация ответа API
            result_data = data.get("result")
            if result_data is None:
                logger.warning(f"get_auction_prices: пустой результат для чанка {chunk[:5]}...")
                continue
                
            keywords = result_data.get("Keywords", [])
            for item in keywords:
                if "Id" not in item:
                    logger.warning(f"get_auction_prices: отсутствует поле Id в элементе {item}")
                    continue
                kw_id = int(item["Id"])
                raw = item.get("PremiumEntryPrice")
                result[kw_id] = None if raw is None else Decimal(raw) / BID_MULTIPLIER
        return result

    def set_bids(self, bids: list[dict[str, int]]) -> dict[str, Any]:
        payload = {"method": "set", "params": {"Bids": bids}}
        return self._request("bids", payload)


# =============================================================================
# БИДДЕР
# =============================================================================

@dataclass
class BidChange:
    keyword_id: int
    old_bid_micro: int | None
    new_bid_micro: int
    reason: str

    @property
    def old_rub(self) -> Decimal | None:
        return None if self.old_bid_micro is None else Decimal(self.old_bid_micro) / BID_MULTIPLIER

    @property
    def new_rub(self) -> Decimal:
        return Decimal(self.new_bid_micro) / BID_MULTIPLIER


class Bidder:
    def __init__(
        self,
        settings: Settings,
        client: DirectClient,
        notifier: TelegramNotifier,
        state: StateStore,
    ):
        self.settings = settings
        self.client = client
        self.notifier = notifier
        self.state = state

    def run(self) -> list[BidChange]:
        logger.info(f"Старт биддера. {self.settings.describe()}")

        # Проверка лимитов API перед началом работы
        is_near_limit, units_rest = self.client.check_units_limit()
        if is_near_limit:
            logger.warning(f"Близко к лимиту API: осталось {units_rest} единиц из {self.client.last_units.get('Units-Limit')}")

        try:
            resp = self.client.check_campaigns()
            logger.info(f"Changes.checkCampaigns → Timestamp={resp.get('result', {}).get('Timestamp')}")
        except DirectApiError as exc:
            if exc.is_rate_limit_error:
                retry_after = exc.retry_after
                if retry_after:
                    logger.warning(f"Превышен лимит запросов. Ожидание {retry_after} секунд...")
                    import time
                    time.sleep(retry_after)
                    # Повторная попытка после ожидания
                    try:
                        resp = self.client.check_campaigns()
                        logger.info(f"Changes.checkCampaigns (повтор) → Timestamp={resp.get('result', {}).get('Timestamp')}")
                    except Exception as retry_exc:
                        logger.warning(f"Повторная checkCampaigns не удалась: {retry_exc}")
                else:
                    logger.warning("Превышен лимит запросов (без Retry-After)")
            elif exc.is_auth_error:
                logger.error(f"Авторизация не прошла: {exc}")
                if exc.hint:
                    logger.error(f"Подсказка: {exc.hint}")
                raise
            else:
                logger.warning(f"checkCampaigns недоступен ({exc}); продолжаем без кэша")
        except requests.RequestException as exc:
            logger.warning(f"checkCampaigns: сетевая ошибка ({exc}); продолжаем без кэша")

        raw_bids = self.client.get_bids(self.settings.campaign_id)
        logger.info(f"Получено ставок: {len(raw_bids)}. Units: {self.client.last_units}")
        if not raw_bids:
            logger.info("Ставок нет — выходим.")
            return []

        keyword_ids = [int(item["KeywordId"]) for item in raw_bids]
        try:
            auction_prices = self.client.get_auction_prices(keyword_ids)
        except DirectApiError as exc:
            if exc.is_auth_error or exc.is_rate_limit_error:
                raise
            logger.warning(f"Не удалось получить цены аукциона ({exc}); используем None")
            auction_prices = {kw: None for kw in keyword_ids}
        except requests.RequestException as exc:
            logger.warning(f"Цены аукциона: сетевая ошибка ({exc}); используем None")
            auction_prices = {kw: None for kw in keyword_ids}

        if self.settings.strategy == "scalping":
            changes = self._run_scalping(raw_bids, auction_prices)
        else:
            changes = self._run_rules(raw_bids, auction_prices)

        if not changes:
            logger.info("Изменений нет.")
            return []

        if self.settings.dry_run:
            logger.warning(f"DRY-RUN: пропускаю отправку {len(changes)} изменений")
            return changes

        payload = [{"KeywordId": c.keyword_id, "Bid": c.new_bid_micro} for c in changes]
        try:
            response = self.client.set_bids(payload)
        except DirectApiError as exc:
            if exc.is_auth_error and exc.hint:
                logger.error(f"Bids.set: {exc.hint}")
            logger.exception("Ошибка установки ставок")
            self.notifier.send(f"❌ Биддер: ошибка Bids.set\n<code>{exc}</code>")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка установки ставок")
            self.notifier.send(f"❌ Биддер: ошибка Bids.set\n<code>{exc}</code>")
            raise

        results = response.get("result", {}).get("SetResults", [])
        failed = [r for r in results if r.get("Errors")]
        for f in failed:
            logger.error(f"Bids.set: ошибка для объекта {f}")
        if failed:
            self.notifier.send(
                f"⚠️ Bids.set: {len(failed)} из {len(changes)} ставок не установлены."
            )
        logger.info(f"Успешно установлено: {len(changes) - len(failed)} из {len(changes)}")
        return changes

    def _run_scalping(
        self,
        raw_bids: list[dict[str, Any]],
        auction_prices: dict[int, Decimal | None],
    ) -> list[BidChange]:
        cfg = self.settings.scalping
        limits = self.settings.limits
        logger.info(
            f"Стратегия: scalping. start_bid={cfg.start_bid}, base_step={cfg.base_step}, "
            f"step_increment={cfg.step_increment}, max_step={cfg.max_step}"
        )

        changes: list[BidChange] = []
        won = 0
        stuck = 0

        for raw in raw_bids:
            kw_id = int(raw["KeywordId"])
            old_micro = raw.get("Bid")
            old_rub = None if old_micro is None else Decimal(old_micro) / BID_MULTIPLIER
            entry = auction_prices.get(kw_id)
            step_n = self.state.get_step_n(self.settings.campaign_id, kw_id)

            new_rub, reason, new_step_n = scalp_step(old_rub, entry, step_n, cfg, limits)

            if entry is not None and old_rub is not None and old_rub >= entry:
                won += 1
            # Исправлена логическая ошибка: старое условие было всегда ложным
            # Было: old_rub >= limits.max_bid < entry (цепочка сравнений)
            # Стало: old_rub >= limits.max_bid и old_rub < entry
            if entry is not None and old_rub is not None and old_rub >= limits.max_bid and old_rub < entry:
                stuck += 1

            if new_rub is None or old_micro == int(new_rub * BID_MULTIPLIER):
                if new_step_n != step_n:
                    self.state.set_step_n(self.settings.campaign_id, kw_id, new_step_n)
                logger.debug(f"KW {kw_id}: без изменений ({reason})")
                continue

            new_micro = int(new_rub * BID_MULTIPLIER)
            changes.append(BidChange(kw_id, old_micro, new_micro, reason))
            logger.info(f"KW {kw_id}: {old_rub} → {new_rub} ({reason})")
            self.state.set_step_n(self.settings.campaign_id, kw_id, new_step_n)

        self.state.save()

        if won:
            logger.info(f"Уже выигрывают аукцион: {won} фраз")
        if stuck:
            logger.warning(f"Упёрлись в max_bid без победы: {stuck} фраз")
        return changes

    def _run_rules(
        self,
        raw_bids: list[dict[str, Any]],
        auction_prices: dict[int, Decimal | None],
    ) -> list[BidChange]:
        engine = self.settings.rules_engine
        limits = self.settings.limits
        logger.info(f"Стратегия: rules.\n{engine.describe()}")

        changes: list[BidChange] = []
        for raw in raw_bids:
            kw_id = int(raw["KeywordId"])
            old_micro = raw.get("Bid")
            old_rub = None if old_micro is None else Decimal(old_micro) / BID_MULTIPLIER
            entry = auction_prices.get(kw_id)

            new_rub, reason = engine.compute(
                premium_entry_price=entry,
                current_bid=old_rub,
                keyword_id=kw_id,
            )

            if new_rub > limits.max_bid:
                new_rub = limits.max_bid
                reason += f" | cap max_bid={limits.max_bid}"
            if limits.min_bid is not None and new_rub < limits.min_bid:
                new_rub = limits.min_bid
                reason += f" | floor min_bid={limits.min_bid}"
            if limits.no_downgrade and old_rub is not None and old_rub > new_rub:
                logger.debug(f"KW {kw_id}: no_downgrade, оставляем {old_rub}")
                continue

            new_micro = int(new_rub * BID_MULTIPLIER)
            if old_micro == new_micro:
                logger.debug(f"KW {kw_id}: без изменений ({new_rub} руб.)")
                continue

            changes.append(BidChange(kw_id, old_micro, new_micro, reason))
            logger.info(f"KW {kw_id}: {old_rub} → {new_rub} руб. ({reason})")

        return changes


# =============================================================================
# КОМАНДЫ CLI
# =============================================================================

def cmd_status(settings: Settings, client: DirectClient, state: StateStore) -> int:
    raw_bids = client.get_bids(settings.campaign_id)
    if not raw_bids:
        print("Ставок нет.")
        return 0

    try:
        auction_prices = client.get_auction_prices([int(b["KeywordId"]) for b in raw_bids])
    except Exception as exc:  # noqa: BLE001
        print(f"Не удалось получить цены аукциона: {exc}")
        auction_prices = {}

    print(f"{'KeywordId':>12} | {'current':>10} | {'entry':>10} | {'step_n':>6} | фаза")
    print("-" * 65)
    for b in raw_bids:
        kw_id = int(b["KeywordId"])
        cur_micro = b.get("Bid")
        cur = "—" if cur_micro is None else str(Decimal(cur_micro) / BID_MULTIPLIER)
        entry_d = auction_prices.get(kw_id)
        entry = "—" if entry_d is None else str(entry_d)
        step_n = state.get_step_n(settings.campaign_id, kw_id)

        phase = "—"
        if entry_d is not None and cur_micro is not None:
            cur_d = Decimal(cur_micro) / BID_MULTIPLIER
            phase = "выигрываем" if cur_d >= entry_d else "скальпинг"
        print(f"{kw_id:>12} | {cur:>10} | {entry:>10} | {step_n:>6} | {phase}")
    return 0


def cmd_reset(settings: Settings, client: DirectClient, state: StateStore) -> int:
    if settings.scalping is None:
        logger.error("--reset работает только при strategy: scalping")
        return 2

    target = settings.scalping.start_bid
    raw_bids = client.get_bids(settings.campaign_id)
    if not raw_bids:
        logger.info("Ставок нет — сбрасывать нечего.")
        state.reset_campaign(settings.campaign_id)
        state.save()
        return 0

    payload = [
        {"KeywordId": int(b["KeywordId"]), "Bid": int(target * BID_MULTIPLIER)}
        for b in raw_bids
    ]

    if settings.dry_run:
        logger.warning(f"DRY-RUN: пропускаю сброс {len(payload)} ставок к {target} ₽")
        logger.warning("Состояние НЕ сброшено (чтобы не разойтись с реальностью).")
        return 0

    response = client.set_bids(payload)
    results = response.get("result", {}).get("SetResults", [])
    failed = [r for r in results if r.get("Errors")]
    if failed:
        for f in failed:
            logger.error(f"Сброс: ошибка для {f}")
        logger.warning(f"Сброшено {len(payload) - len(failed)} из {len(payload)} ставок.")
    else:
        logger.info(f"Сброшено {len(payload)} ставок к {target} ₽.")

    state.reset_campaign(settings.campaign_id)
    state.save()
    logger.info("Состояние обнулено.")
    return 0


# =============================================================================
# ВАЛИДАЦИЯ
# =============================================================================

def validate_settings(settings: Settings) -> None:
    token = settings.token.strip()
    if token in PLACEHOLDER_TOKENS:
        raise ConfigError(
            "YANDEX_DIRECT_TOKEN не заполнен в .env. "
            "Впишите реальный OAuth-токен (или sandbox-токен при mode.sandbox=true)."
        )
    if len(token) < 20:
        raise ConfigError(f"YANDEX_DIRECT_TOKEN подозрительно короткий ({len(token)} симв.).")


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yandex Direct bidder")
    parser.add_argument("--config", default=CONFIG_FILE_DEFAULT,
                        help="Путь к config.yaml (по умолчанию config.yaml)")
    parser.add_argument("--reset", action="store_true",
                        help="Сбросить все ставки к scalping.start_bid и обнулить состояние")
    parser.add_argument("--status", action="store_true",
                        help="Показать текущие ставки, цены входа и step_n")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        settings = Settings.load(args.config)
    except ConfigError as exc:
        print(f"[FATAL] Конфигурация: {exc}", file=sys.stderr)
        return 2
    except RulesError as exc:
        print(f"[FATAL] Ошибка в правилах: {exc}", file=sys.stderr)
        return 3

    setup_logger(settings.log_file, settings.log_level)
    notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)

    try:
        validate_settings(settings)
    except ConfigError as exc:
        logger.error(f"Конфигурация не готова: {exc}")
        notifier.send(f"❌ Биддер: конфигурация не готова\n<code>{exc}</code>")
        return 2

    client = DirectClient(settings)
    state = StateStore(settings.state_file)

    try:
        if args.status:
            return cmd_status(settings, client, state)
        if args.reset:
            return cmd_reset(settings, client, state)

        bidder = Bidder(settings, client, notifier, state)
        changes = bidder.run()

    except DirectApiError as exc:
        if exc.is_auth_error and exc.hint:
            logger.error(f"Подсказка: {exc.hint}")
            notifier.send(f"❌ Биддер: ошибка авторизации [{exc.code}]\n<code>{exc.hint}</code>")
        else:
            logger.exception("Ошибка API")
            notifier.send(f"❌ Биддер: ошибка API\n<code>{exc}</code>")
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Критическая ошибка биддера")
        notifier.send(f"❌ Биддер упал: <code>{exc}</code>")
        return 1

    if changes:
        prefix = "🧪 DRY-RUN" if settings.dry_run else "✅ Применено"
        summary = "\n".join(
            f"• KW {c.keyword_id}: {c.old_rub or '—'} → {c.new_rub} руб. ({c.reason})"
            for c in changes
        )
        notifier.send(f"{prefix}: изменений {len(changes)}\n{summary}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
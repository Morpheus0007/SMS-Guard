"""Configuration, constants, and domain whitelist helpers for SMS Guard."""

from __future__ import annotations

from pathlib import Path
from typing import Final

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
MODELS_DIR: Final[Path] = BASE_DIR / "models"

DEFAULT_TRAIN_CSV: Final[Path] = DATA_DIR / "sms_train.csv"
DEFAULT_DEV_CSV: Final[Path] = DATA_DIR / "sms_dev.csv"
DEFAULT_MODEL_PATH: Final[Path] = MODELS_DIR / "model.pkl"

OFFICIAL_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "beeline.kg",
        "balance.kg",
        "nbkr.kg",
        "mbank.kg",
    }
)

SHORTENER_HOSTS: Final[frozenset[str]] = frozenset(
    {
        "bit.ly",
        "bit-ly.example",
        "clck.ru",
        "t.me",
        "tinyurl.com",
        "goo.gl",
        "cutt.ly",
        "is.gd",
    }
)

BRAND_SPOOF_TOKENS: Final[tuple[str, ...]] = (
    "beeline",
    "mbank",
    "optima",
    "kicb",
    "demir",
    "bakai",
    "rsk",
    "eldik",
    "megacom",
    "balance",
    "nbkr",
    "o-kg",
    "bit-ly",
    "bitly",
    "verify",
    "secure",
    "login",
    "bonus",
    "prize",
    "gift",
    "card",
    "bank",
    "online",
    "check",
    "tunduk",
    "pochta",
    "refund",
    "fine",
    "cert",
    "pass",
    "auth",
)

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "has_link",
    "asks_code_or_password",
    "has_urgency",
    "threatens_block",
    "promises_prize",
    "mentions_bank_or_operator",
    "asks_money_transfer",
    "asks_callback",
    "has_official_domain",
    "has_brand_spoof_domain",
    "has_safe_warning",
    "has_phone_number",
)

PROTECTIVE_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "has_official_domain",
        "has_safe_warning",
    }
)

RISK_MEDIUM_MIN: Final[int] = 30
RISK_HIGH_MIN: Final[int] = 65


def is_official_host(host: str) -> bool:
    """Return True strictly if host matches OFFICIAL_DOMAINS or one of their subdomains."""
    if not isinstance(host, str):
        return False
    normalized = host.strip().lower()
    if ":" in normalized:
        normalized = normalized.split(":", 1)[0]
    normalized = normalized.rstrip(".")
    if not normalized:
        return False
    if normalized in OFFICIAL_DOMAINS:
        return True
    return any(normalized.endswith("." + domain) for domain in OFFICIAL_DOMAINS)

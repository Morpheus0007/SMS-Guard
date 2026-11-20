"""Canonical 12-feature extraction for SMS Guard (RU / KG / Translit)."""

from __future__ import annotations

import re

from src.config import FEATURE_NAMES
from src.preprocessing import normalize, strip_urls_for_text_features, unify_kyrgyz

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "has_link": "содержит ссылку",
    "asks_code_or_password": "просьба сообщить, ввести или обновить код или пароль",
    "has_urgency": "создание срочности или ограничения по времени",
    "threatens_block": "угроза блокировки или отключения карты, аккаунта или SIM",
    "promises_prize": "обещание выигрыша, приза, подарка или бесплатного бонуса",
    "mentions_bank_or_operator": "упоминание банка, платежного сервиса или мобильного оператора",
    "asks_money_transfer": "просьба перевести, оплатить или вернуть деньги",
    "asks_callback": "просьба перезвонить или позвонить по указанному номеру",
    "has_official_domain": "ссылка ведет на официальный домен из белого списка",
    "has_brand_spoof_domain": "ссылка на неофициальный домен с имитацией бренда или сокращателя",
    "has_safe_warning": "предупреждение о безопасности («никому не сообщайте»)",
    "has_phone_number": "указан контактный номер телефона",
}

_SAFE_WARNING_RE = re.compile(
    r"(?:"
    r"никому\s+не\s+сообщайте"
    r"|не\s+сообщайте\s+(?:его\s+|код\s+)?никому"
    r"|ничего\s+подтверждать\s+не\s+нужно"
    r"|не\s+переводите\s+деньги\s+незнакомым"
    r"|эч\s+кимге\s+айтпа"
    r"|эч\s+кимге\s+бербе"
    r"|алдамчылардан\s+сак\s+бол"
    r"|акча\s+которбо"
    r"|утуш\s+учун\s+акча\s+которбо"
    r")",
    re.IGNORECASE,
)

# Guarded code/password patterns (avoid false match on negated safe warnings)
_ASKS_CODE_RE = re.compile(
    r"(?:"
    r"(?:сооб[щш]ите|назовите|отправьте|пришлите|передайте|введите|диктуйте|подтвердите)\b[^.!?\n]{0,40}\b(?:код|парол|пин|pin|cvv|данные)"
    r"|(?:код(?:ом)?|парол[ьяе]?|пин|pin|cvv)\b[^.!?\n]{0,30}\b(?:сооб[щш]ите|назовите|отправьте|пришлите|передайте|введите|обновите|подтвердите|истекает)"
    r"|(?:код\s+из\s+sms\s+(?:нашему\s+сотруднику|по\s+номеру))"
    r"|обновите\s+(?:его|пароль)"
    r"|подтвердите\s+(?:данные|личность|аккаунт|карту)"
    r"|кодду\w*\s+(?:жибер|жонот|айты)"
    r"|активдештир\w*[^.!?\n]{0,30}(?:шилтеме|от)"
    r"|(?:ырастоо|подтверждения)\s+(?:учун|үчүн)[^.!?\n]{0,40}(?:отунуз|өтүңүз|чалыныз|звоните)"
    r")",
    re.IGNORECASE,
)

_URGENCY_RE = re.compile(
    r"(?:"
    r"\bсрочно\b"
    r"|\bнемедленно\b"
    r"|\bэкстренно\b"
    r"|тез\s+арада"
    r"|через\s+\d+\s*(?:час|мин)"
    r"|\d+\s*(?:сааттан|муноттон|часа|минут)"
    r"|до\s+конца\s+дня"
    r"|бугун\s+ичинде"
    r"|шашылыш"
    r")",
    re.IGNORECASE,
)

_BLOCK_RE = re.compile(
    r"(?:"
    r"заблокирован\w*"
    r"|будет\s+отключен\w*"
    r"|будет\s+заблокирован\w*"
    r"|приостановлен\w*"
    r"|подозрительн\w+\s+операци\w*"
    r"|боготтол\w*"
    r"|очурул\w*"
    r"|разблокировк\w*"
    r"|истекает"
    r")",
    re.IGNORECASE,
)

_PRIZE_RE = re.compile(
    r"(?:"
    r"выиграл\w*"
    r"|победител\w*"
    r"|розыгрыш\w*"
    r"|\bприз\w*\b"
    r"|\bподарок\b|\bподарк\w*"
    r"|\bдарит\b"
    r"|бонус\w*"
    r"|уттунуз|уттун\w*"
    r"|белек"
    r"|байге"
    r"|куттуктай\w*"
    r"|поздравляем"
    r"|активируйте\s+(?:подарок|бонус)"
    r"|активдештир\w*"
    r")",
    re.IGNORECASE,
)

_BANK_OP_RE = re.compile(
    r"(?:"
    r"\bмбанк\b|\bmbank\b"
    r"|\bоптима\b|\boptima\b"
    r"|\bдемир\b|\bdemir\b"
    r"|\bбакай\b|\bbakai\b"
    r"|\bkicb\b|\bкикб\b"
    r"|\bрск\b|\brsk\b"
    r"|\bэлдик\b|\beldik\b"
    r"|\bнбкр\b|\bnbkr\b"
    r"|\bbeeline\b|\bбилайн\b"
    r"|\bmegacom\b|\bмегаком\b|\bmega\b"
    r"|\bbalance\b|\bбаланс\.кж\b"
    r"|\b[oо]!(?!\w)"
    r"|\bбанк\w*\b|\bbank\w*\b|\bванк\w*\b"
    r"|\bsim[\-\s]?карт\w*"
    r"|служба\s+безопасности"
    r")",
    re.IGNORECASE,
)

_TRANSFER_RE = re.compile(
    r"(?:"
    r"переведи(?:те)?\s+(?:\d+|деньги|средств|сумм)"
    r"|оплатите\s+(?:доставку|комиссию|пошлину|налог)"
    r"|скин[ьи]\w*\s+\d+"
    r"|ошибочно\s+зачислен\w*"
    r"|верните\s+(?:на\s+номер|деньги|средств)"
    r"|которуп\s+жибер\w*"
    r"|(?<!не\s)акча\s+котор(?!бо)"
    r"|пополните\s+(?:баланс|счет)\s+(?:на\s+номер|для)"
    r")",
    re.IGNORECASE,
)

_CALLBACK_RE = re.compile(
    r"(?:"
    r"перезвоните\s+(?:на|по)"
    r"|позвоните\s+(?:на|по)"
    r"|чалыныз\w*"
    r"|номерине\s+чалын\w*"
    r"|сообщите\s+[^.!?\n]{0,25}\s+по\s+номеру"
    r"|свяжитесь\s+(?:по\s+номеру|с\s+нами)"
    r")",
    re.IGNORECASE,
)


def extract_features(norm: dict[str, object]) -> tuple[dict[str, float], dict[str, str]]:
    """Extract strictly ordered 12 canonical features and Russian descriptions from normalize() output."""
    if not isinstance(norm, dict):
        norm = normalize("")

    domains: list[dict[str, object]] = list(norm.get("domains") or [])  # type: ignore[arg-type]
    phones: list[str] = list(norm.get("phones") or [])  # type: ignore[arg-type]

    clean_no_urls = strip_urls_for_text_features(str(norm.get("clean_text") or ""), domains)
    translit_no_urls = strip_urls_for_text_features(str(norm.get("translit_text") or ""), domains)
    # Unified matching text combining normalized Cyrillic + transliterated Latin
    combined = f"{unify_kyrgyz(clean_no_urls)} | {unify_kyrgyz(translit_no_urls)}"

    safe_warning = bool(_SAFE_WARNING_RE.search(combined))

    # Strip explicit negated safe-warning clauses before checking asks_code / asks_money_transfer
    # so phrases like "не сообщайте код никому" or "утуш учун акча которбонуз" do not false-trigger.
    scrubbed = _SAFE_WARNING_RE.sub(" ", combined)

    values: dict[str, float] = {
        "has_link": 1.0 if len(domains) > 0 else 0.0,
        "asks_code_or_password": 1.0 if _ASKS_CODE_RE.search(scrubbed) else 0.0,
        "has_urgency": 1.0 if _URGENCY_RE.search(combined) else 0.0,
        "threatens_block": 1.0 if _BLOCK_RE.search(combined) else 0.0,
        "promises_prize": 1.0 if _PRIZE_RE.search(scrubbed) else 0.0,
        "mentions_bank_or_operator": 1.0 if _BANK_OP_RE.search(combined) else 0.0,
        "asks_money_transfer": 1.0 if _TRANSFER_RE.search(scrubbed) else 0.0,
        "asks_callback": 1.0 if _CALLBACK_RE.search(combined) else 0.0,
        "has_official_domain": 1.0 if any(bool(d.get("is_official")) for d in domains) else 0.0,
        "has_brand_spoof_domain": 1.0 if any(bool(d.get("is_brand_spoof")) for d in domains) else 0.0,
        "has_safe_warning": 1.0 if safe_warning else 0.0,
        "has_phone_number": 1.0 if len(phones) > 0 else 0.0,
    }

    # Guarantee exact canonical key order
    ordered_values = {name: float(values[name]) for name in FEATURE_NAMES}
    triggered_reasons = {
        name: FEATURE_DESCRIPTIONS[name]
        for name in FEATURE_NAMES
        if ordered_values[name] > 0.0
    }
    return ordered_values, triggered_reasons

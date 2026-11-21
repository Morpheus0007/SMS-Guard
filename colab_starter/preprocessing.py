"""Colab-standalone copy of src/preprocessing.py (do not diverge from src/preprocessing.py)."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from config import BRAND_SPOOF_TOKENS, SHORTENER_HOSTS, is_official_host

_EXPLICIT_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_BARE_URL_RE = re.compile(
    r"(?<![@\w/.-])(?:[A-Za-zА-Яа-яЁё0-9-]{2,}\.)+"
    r"(?:kg|example|com|ru|net|org|xyz|top|site|click|info|me|io|uz|kz)"
    r"(?:/[^\s<>\"']*)?",
    re.IGNORECASE,
)

_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+\d{1,3}[\s\-]?(?:\(\d{2,4}\)[\s\-]?)?\d(?:[\s\-]?\d){6,12}"
    r"|0(?:22\d|312|50\d|55\d|70\d|77\d|99\d)[\s\-]?\d{3}[\s\-]?\d{3})"
    r"(?!\d)"
)

_CYRILLIC_CHAR_RE = re.compile(r"[А-Яа-яЁёҢңӨөҮү]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_DIGIT_UNIT_RE = re.compile(
    r"^(\d+)(до|дo|ч|саат|гб|мб|кб|сом|мин|сек|gb|mb|kb|som|min|sec)$",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_WHITESPACE_RE = re.compile(r"\s+")

_URL_HOMOGLYPHS = str.maketrans(
    {
        "а": "a", "А": "a", "е": "e", "Е": "e", "о": "o", "О": "o",
        "р": "p", "Р": "p", "с": "c", "С": "c", "х": "x", "Х": "x",
        "у": "y", "У": "y", "к": "k", "К": "k", "м": "m", "М": "m",
        "т": "t", "Т": "t", "н": "h", "Н": "h", "в": "b", "В": "b",
        "і": "i", "І": "i",
    }
)

_MIXED_HOMOGLYPHS = {
    "a": "а", "A": "А", "e": "е", "E": "Е", "o": "о", "O": "О",
    "p": "р", "P": "Р", "c": "с", "C": "С", "x": "х", "X": "Х",
    "y": "у", "Y": "У", "k": "к", "K": "К", "m": "м", "M": "М",
    "t": "т", "T": "Т", "h": "н", "H": "Н", "u": "и", "U": "И",
    "n": "п", "N": "П", "B": "В", "b": "б",
    "@": "а", "$": "с", "0": "о", "3": "з", "6": "б", "4": "ч", "1": "и",
}

_STRICT_HOMOGLYPH_LATIN = frozenset("aeopcxymtkhABCEHKMOPTX")
_LATIN_KEEP_WORDS = frozenset(
    {
        "O", "OK", "SMS", "SIM", "MBANK", "KICB", "RSK", "OPT", "OTP", "PIN",
        "CVV", "API", "URL", "ID", "KG", "RU", "VIP", "USD", "EUR", "SOM",
        "ONLINE", "BEELINE", "MEGACOM", "BALANCE", "IPHONE", "SAMSUNG",
        "BONUS", "AK", "TV", "PC", "ATM",
    }
)
_SHORT_LOOKALIKE_WORDS = frozenset({"he", "ha", "bo", "co", "po", "ko", "mo", "to", "oт", "c"})
_KYRGYZ_UNIFY_TABLE = str.maketrans({"ң": "н", "ө": "о", "ү": "у", "ё": "е"})

_TRANSLIT_MULTI: tuple[tuple[str, str], ...] = (
    ("shch", "щ"), ("sch", "щ"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"),
    ("kh", "х"), ("ts", "ц"), ("yu", "ю"), ("ya", "я"), ("yo", "ё"),
    ("ye", "е"), ("ng", "нг"),
)

_TRANSLIT_SINGLE: dict[str, str] = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф",
    "g": "г", "h": "х", "i": "и", "j": "ж", "k": "к", "l": "л",
    "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р",
    "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
    "y": "ы", "z": "з",
}


def unify_kyrgyz(text: str) -> str:
    """Fold Kyrgyz-specific letters and yo into base Cyrillic for uniform matching."""
    return text.lower().translate(_KYRGYZ_UNIFY_TABLE)


def transliterate_latin_to_cyrillic(text: str) -> str:
    """Convert Latin/translit SMS text to approximate Cyrillic for unified matching."""
    if not text:
        return ""
    lower = text.lower()
    out: list[str] = []
    i = 0
    n = len(lower)
    while i < n:
        ch = lower[i]
        if "a" <= ch <= "z":
            matched = False
            for src_seq, dst_seq in _TRANSLIT_MULTI:
                if lower.startswith(src_seq, i):
                    out.append(dst_seq)
                    i += len(src_seq)
                    matched = True
                    break
            if not matched:
                out.append(_TRANSLIT_SINGLE.get(ch, ch))
                i += 1
        else:
            out.append(ch)
            i += 1
    result = "".join(out)
    result = re.sub(r"у{2,}", "у", result)
    return result


_HOST_LEET_L = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"})
_HOST_LEET_I = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})


def _analyze_url(raw_url: str) -> dict[str, object]:
    cleaned = raw_url.rstrip(".,!?;:)]}'\"»")
    candidate = cleaned if "://" in cleaned else f"http://{cleaned}"
    parsed = urlparse(candidate)
    raw_host = (parsed.hostname or "").lower().rstrip(".")
    has_cyr_host = bool(_CYRILLIC_CHAR_RE.search(raw_host))
    ascii_host = raw_host.translate(_URL_HOMOGLYPHS) if has_cyr_host else raw_host
    is_official = (not has_cyr_host) and is_official_host(ascii_host)

    path_lower = (parsed.path or "").lower()
    is_spoof = False
    if not is_official:
        host_variants = (
            ascii_host,
            ascii_host.translate(_HOST_LEET_L),
            ascii_host.translate(_HOST_LEET_I),
        )
        if has_cyr_host or ascii_host in SHORTENER_HOSTS:
            is_spoof = True
        elif any(
            tok in hv or tok in path_lower
            for hv in host_variants
            for tok in BRAND_SPOOF_TOKENS
        ):
            is_spoof = True
        elif any(is_official_host(hv) for hv in host_variants[1:]):
            is_spoof = True

    return {
        "url": cleaned,
        "domain": ascii_host,
        "is_official": is_official,
        "is_brand_spoof": is_spoof,
    }


def _extract_urls_and_mask(text: str) -> tuple[str, list[dict[str, object]], list[str]]:
    domains: list[dict[str, object]] = []
    raw_urls: list[str] = []

    def _replace_explicit(match: re.Match[str]) -> str:
        raw = match.group(0)
        trimmed = raw.rstrip(".,!?;:)]}'\"»")
        suffix = raw[len(trimmed):]
        idx = len(raw_urls)
        raw_urls.append(trimmed)
        domains.append(_analyze_url(trimmed))
        return f" __URL_{idx}__ {suffix}".strip()

    masked = _EXPLICIT_URL_RE.sub(_replace_explicit, text)

    def _replace_bare(match: re.Match[str]) -> str:
        raw = match.group(0)
        if raw.startswith("__URL_"):
            return raw
        trimmed = raw.rstrip(".,!?;:)]}'\"»")
        suffix = raw[len(trimmed):]
        idx = len(raw_urls)
        raw_urls.append(trimmed)
        domains.append(_analyze_url(trimmed))
        return f" __URL_{idx}__ {suffix}".strip()

    masked = _BARE_URL_RE.sub(_replace_bare, masked)
    return masked, domains, raw_urls


def _extract_phones_and_mask(text: str) -> tuple[str, list[str], list[str]]:
    phones: list[str] = []

    def _replace_phone(match: re.Match[str]) -> str:
        raw = match.group(0).strip()
        idx = len(phones)
        phones.append(raw)
        return f" __PHONE_{idx}__ "

    masked = _PHONE_RE.sub(_replace_phone, text)
    return masked, phones, list(phones)


_RU_SINGLE_WORDS = frozenset({"а", "в", "и", "к", "о", "с", "у", "я"})


def _is_single_letter_token(tok: str) -> bool:
    return len(tok) == 1 and tok.isalpha()


def _merge_spaced_letters(tokens: list[str]) -> tuple[list[str], bool]:
    merged: list[str] = []
    was_merged = False
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]
        if "-" in tok and not tok.startswith("__"):
            parts = tok.split("-")
            if len(parts) >= 3 and all(_is_single_letter_token(p) for p in parts):
                merged.append("".join(parts))
                was_merged = True
                i += 1
                continue

        lead = ""
        first_core = tok
        if len(tok) >= 2 and tok[0] in "\"'«(" and _is_single_letter_token(tok[1:]):
            lead = tok[0]
            first_core = tok[1:]

        if _is_single_letter_token(first_core) and (lead or len(tok) == 1):
            run = [first_core]
            trail = ""
            j = i + 1
            while j < n:
                nxt = tokens[j]
                if _is_single_letter_token(nxt):
                    run.append(nxt)
                    j += 1
                elif nxt in ("@", "$") and j + 1 < n and _is_single_letter_token(tokens[j + 1]):
                    run.append(nxt)
                    j += 1
                elif len(nxt) >= 2 and _is_single_letter_token(nxt[0]) and all(
                    c in ".,!?;:»\"')" for c in nxt[1:]
                ):
                    run.append(nxt[0])
                    trail = nxt[1:]
                    j += 1
                    break
                else:
                    break
            alpha_chars = [ch.lower() for ch in run if ch.isalpha()]
            only_ru_prepositions = (
                len(alpha_chars) <= 4
                and not trail
                and not lead
                and all(ch in _RU_SINGLE_WORDS for ch in alpha_chars)
            )
            if len(alpha_chars) >= 3 and not only_ru_prepositions:
                merged.append(lead + "".join(run) + trail)
                was_merged = True
                i = j
                continue

        merged.append(tok)
        i += 1

    return merged, was_merged


def _deobfuscate_subtoken(sub: str, sentence_has_cyr: bool) -> tuple[str, bool]:
    if not sub or sub.startswith("__") or _EMAIL_RE.match(sub):
        return sub, False

    unit_match = _DIGIT_UNIT_RE.match(sub)
    if unit_match:
        digits, unit = unit_match.groups()
        has_lat_u = bool(_LATIN_CHAR_RE.search(unit))
        has_cyr_u = bool(_CYRILLIC_CHAR_RE.search(unit))
        if has_lat_u and has_cyr_u:
            fixed_unit = "".join(_MIXED_HOMOGLYPHS.get(c, c) for c in unit)
            return digits + fixed_unit, fixed_unit != unit
        return sub, False

    cleaned = sub
    obf = False
    if ("*" in cleaned or "_" in cleaned) and any(c.isalpha() for c in cleaned):
        stripped = re.sub(
            r"(?<=[A-Za-zА-Яа-яЁёҢңӨөҮү@$\d])[*_]+(?=[A-Za-zА-Яа-яЁёҢңӨөҮү@$\d])",
            "",
            cleaned,
        )
        if stripped != cleaned:
            cleaned = stripped
            obf = True

    if not any(c.isalpha() or c in "@$" for c in cleaned):
        return cleaned, obf

    has_cyr = bool(_CYRILLIC_CHAR_RE.search(cleaned))
    has_lat = bool(_LATIN_CHAR_RE.search(cleaned))
    has_dig = any(c.isdigit() for c in cleaned)
    has_leet = "@" in cleaned or "$" in cleaned

    if has_leet and has_lat and not has_cyr:
        latin_fixed = cleaned.replace("@", "a").replace("$", "s")
        return latin_fixed, obf or (latin_fixed != cleaned)

    should_convert = False
    if has_leet:
        should_convert = True
    elif has_cyr and (has_lat or has_dig):
        should_convert = True
    elif has_lat and has_dig and all(
        (c in _STRICT_HOMOGLYPH_LATIN or c in "uU0346@$") for c in cleaned
    ):
        should_convert = True
    elif sentence_has_cyr and has_lat and not has_dig and not has_cyr:
        upper = cleaned.upper()
        lower = cleaned.lower()
        if upper not in _LATIN_KEEP_WORDS and (
            lower in _SHORT_LOOKALIKE_WORDS
            or (len(cleaned) >= 2 and all(c in _STRICT_HOMOGLYPH_LATIN for c in cleaned))
        ):
            should_convert = True

    if not should_convert:
        return cleaned, obf

    converted = "".join(_MIXED_HOMOGLYPHS.get(c, c) for c in cleaned)
    if converted != cleaned:
        obf = True
    return converted, obf


def _deobfuscate_token(tok: str, sentence_has_cyr: bool) -> tuple[str, bool]:
    if not tok or tok.startswith("__URL_") or tok.startswith("__PHONE_"):
        return tok, False

    start = 0
    end = len(tok)
    while start < end and not (tok[start].isalnum() or tok[start] in "@$"):
        start += 1
    while end > start and not (tok[end - 1].isalnum() or tok[end - 1] in "@$"):
        end -= 1
    if start >= end:
        return tok, False

    lead = tok[:start]
    core = tok[start:end]
    trail = tok[end:]

    if _EMAIL_RE.match(core):
        return tok, False

    parts = re.split(r"([-'])", core)
    new_parts: list[str] = []
    any_obf = False
    for part in parts:
        if part in ("-", "'"):
            new_parts.append(part)
        else:
            fixed, obf = _deobfuscate_subtoken(part, sentence_has_cyr)
            new_parts.append(fixed)
            if obf:
                any_obf = True
    return lead + "".join(new_parts) + trail, any_obf


def normalize(text: object) -> dict[str, object]:
    """Normalize SMS text: extract URLs/phones, deobfuscate, and prepare RU/KG/translit views."""
    if text is None:
        original = ""
    elif isinstance(text, str):
        original = text
    else:
        original = str(text)

    if not original.strip():
        return {
            "original_text": original,
            "clean_text": "",
            "translit_text": "",
            "domains": [],
            "phones": [],
            "was_obfuscated": False,
        }

    masked_urls, domains, raw_urls = _extract_urls_and_mask(original)
    masked_all, phones, raw_phones = _extract_phones_and_mask(masked_urls)

    raw_tokens = masked_all.split()
    merged_tokens, spaced_obf = _merge_spaced_letters(raw_tokens)

    sentence_has_cyr = any(_CYRILLIC_CHAR_RE.search(t) for t in merged_tokens if not t.startswith("__"))
    clean_tokens: list[str] = []
    homoglyph_obf = False
    for tok in merged_tokens:
        fixed, tok_obf = _deobfuscate_token(tok, sentence_has_cyr)
        clean_tokens.append(fixed)
        if tok_obf:
            homoglyph_obf = True

    was_obfuscated = spaced_obf or homoglyph_obf

    restored_tokens: list[str] = []
    translit_source_tokens: list[str] = []
    for tok in clean_tokens:
        if tok.startswith("__URL_") and tok.endswith("__"):
            try:
                idx = int(tok[6:-2])
                restored_tokens.append(raw_urls[idx])
                continue
            except (ValueError, IndexError):
                pass
        if tok.startswith("__PHONE_") and tok.endswith("__"):
            try:
                idx = int(tok[8:-2])
                restored_tokens.append(raw_phones[idx])
                translit_source_tokens.append(raw_phones[idx])
                continue
            except (ValueError, IndexError):
                pass
        restored_tokens.append(tok)
        translit_source_tokens.append(tok)

    clean_text = _WHITESPACE_RE.sub(" ", " ".join(restored_tokens)).strip().lower()
    translit_raw = _WHITESPACE_RE.sub(" ", " ".join(translit_source_tokens)).strip()
    translit_text = unify_kyrgyz(transliterate_latin_to_cyrillic(translit_raw))

    return {
        "original_text": original,
        "clean_text": clean_text,
        "translit_text": translit_text,
        "domains": domains,
        "phones": phones,
        "was_obfuscated": bool(was_obfuscated),
    }


def strip_urls_for_text_features(clean_text: str, domains: list[dict[str, object]]) -> str:
    """Remove extracted URLs from text before text/TF-IDF matching."""
    text = clean_text
    for dom in domains:
        url = str(dom.get("url", "")).lower()
        if url:
            text = text.replace(url, " ")
    text = _EXPLICIT_URL_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()

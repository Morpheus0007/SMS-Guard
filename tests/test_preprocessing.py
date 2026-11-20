"""Unit tests for text normalization, deobfuscation, domains, phones, and speed."""

from __future__ import annotations

import time

from src.config import is_official_host
from src.preprocessing import normalize


def test_homoglyphs_and_leetspeak_deobfuscation() -> None:
    norm = normalize("П03др@вляем! Вы выиграли в Ванk. 3а6ерuте п0дтвердuте кредuт од0брен")
    clean = str(norm["clean_text"])
    assert norm["was_obfuscated"] is True
    assert "поздравляем" in clean
    assert "банк" in clean or "ванк" in clean
    assert "заберите" in clean
    assert "подтвердите" in clean
    assert "кредит" in clean
    assert "одобрен" in clean


def test_spaced_letters_merged_without_breaking_prepositions() -> None:
    norm = normalize("С р о ч н о ! Я и в банке получил к о д, п р и з а, B e e l i n e.")
    clean = str(norm["clean_text"])
    assert norm["was_obfuscated"] is True
    assert "срочно" in clean
    assert "код" in clean
    assert "приза" in clean
    assert "beeline" in clean
    assert "я и в банке" in clean


def test_hyphen_spaced_letters_merged_without_breaking_sim_karta() -> None:
    norm = normalize("Ваш п-а-р-о-л-ь от SIM-карты")
    assert norm["was_obfuscated"] is True
    assert "пароль" in str(norm["clean_text"])
    assert "sim-карты" in str(norm["clean_text"])


def test_kyrgyz_obfuscation_and_translit() -> None:
    norm1 = normalize("Beeline: сизге 3 ГБ белек! Шилтеме @ркылуу актuвдештuриңu3")
    assert norm1["was_obfuscated"] is True
    assert "аркылуу" in str(norm1["clean_text"])
    assert "активдештириңиз" in str(norm1["clean_text"])

    norm2 = normalize("РСК Банк: kartanyz bogottoldu. Tez arada kodduu jiberiniz 0555 538030")
    tr = str(norm2["translit_text"])
    assert norm2["was_obfuscated"] is False
    assert "картаныз" in tr
    assert "боготтолду" in tr
    assert "кодду" in tr
    assert "жибериниз" in tr


def test_digits_dates_codes_phones_not_corrupted() -> None:
    sms = (
        "Напоминаем: 0пл@тите домашний и н т е р н е т д0 03.12.2026. "
        "Код: 605225, карта *0497, саат 18дo, звоните +996 770 044186"
    )
    norm = normalize(sms)
    clean = str(norm["clean_text"])
    assert "03.12.2026" in clean
    assert "605225" in clean
    assert "*0497" in clean
    assert "18до" in clean
    assert "+996 770 044186" in clean
    assert norm["phones"] == ["+996 770 044186"]


def test_sms_tegi_and_sim_karta_not_false_obfuscated() -> None:
    norm = normalize("РСК Банк: картаңыз бөгөттөлдү. SMS'теги кодду жибериңиз. SIM-карта ок")
    assert norm["was_obfuscated"] is False


def test_official_domain_whitelist_strictness() -> None:
    assert is_official_host("beeline.kg") is True
    assert is_official_host("my.beeline.kg") is True
    assert is_official_host("balance.kg") is True
    assert is_official_host("nbkr.kg") is True
    assert is_official_host("mbank.kg") is True

    assert is_official_host("my-beeline.kg.example") is False
    assert is_official_host("mbank.kg-verify.example") is False
    assert is_official_host("fakebeeline.kg") is False
    assert is_official_host("univer.example") is False


def test_domains_classification_official_vs_spoof_vs_benign() -> None:
    off = normalize("Подробности на https://my.beeline.kg")
    assert len(off["domains"]) == 1
    assert off["domains"][0]["is_official"] is True
    assert off["domains"][0]["is_brand_spoof"] is False

    spoof = normalize("Срочно: https://mbank.kg-verify.example и https://bit-ly.example/3xKp9q")
    assert len(spoof["domains"]) == 2
    assert all(d["is_official"] is False and d["is_brand_spoof"] is True for d in spoof["domains"])

    benign = normalize("Расписание: https://univer.example/raspisanie и https://cafe-navat.example/menu")
    assert len(benign["domains"]) == 2
    assert all(d["is_official"] is False and d["is_brand_spoof"] is False for d in benign["domains"])


def test_cyrillic_homoglyph_inside_domain_flagged_as_spoof() -> None:
    # 'а' in mbаnk.kg is Cyrillic U+0430
    norm = normalize("Войдите срочно: https://mb\u0430nk.kg/login")
    assert len(norm["domains"]) == 1
    assert norm["domains"][0]["is_official"] is False
    assert norm["domains"][0]["is_brand_spoof"] is True


def test_edge_inputs_and_5000_chars_performance() -> None:
    for bad in (None, "", "   ", 12345, "🔥🔥🔥"):
        res = normalize(bad)
        assert set(res.keys()) == {
            "original_text",
            "clean_text",
            "translit_text",
            "domains",
            "phones",
            "was_obfuscated",
        }

    long_text = ("к о д б@нк https://univer.example/test +996 770 123456 " * 120)[:5500]
    t0 = time.perf_counter()
    res_long = normalize(long_text)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert res_long["was_obfuscated"] is True
    assert elapsed_ms < 50.0

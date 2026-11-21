"""Tests for bot.py HTML formatting, run.py launcher, and newly hardened edge patterns."""

from __future__ import annotations

import os
import subprocess
import sys

import bot
from src.config import BASE_DIR
from src.detector import check
from src.features import extract_features
from src.preprocessing import normalize


def test_bot_html_report_escapes_user_input_and_shows_risk() -> None:
    malicious_sms = "<script>alert(1)</script> Срочно сообщите код из SMS по ссылке https://mbank.kg-verify.example"
    report = bot.format_sms_report_html(malicious_sms)
    assert "<script>" not in report
    assert "ВЫСОКИЙ РИСК" in report
    assert "mbank.kg-verify.example" in report
    assert "Что делать:" in report


def test_bot_html_report_safe_sms_and_obfuscation_badge() -> None:
    safe_sms = "Заказ готов, 3а6ерuте в пункте выд@чu. Подробнее: https://univer.example/raspisanie"
    report = bot.format_sms_report_html(safe_sms)
    assert "ВЫСОКИЙ РИСК" not in report
    assert "деобфускация" in report
    assert "univer.example" in report


def test_bot_main_exits_cleanly_without_token() -> None:
    env = dict(os.environ)
    env["TELEGRAM_BOT_TOKEN"] = ""
    env["SMS_GUARD_SKIP_DOTENV"] = "1"
    proc = subprocess.run(
        [sys.executable, str(BASE_DIR / "bot.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=True,
    )
    assert "TELEGRAM_BOT_TOKEN" in proc.stdout


def test_run_py_eval_and_check_modes() -> None:
    proc_eval = subprocess.run(
        [sys.executable, str(BASE_DIR / "run.py"), "--eval", "data/sms_dev.csv"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert "Accuracy" in proc_eval.stdout

    proc_check = subprocess.run(
        [
            sys.executable,
            str(BASE_DIR / "run.py"),
            "--check",
            "Ваш аккаунт MBANK Online будет заблокирован. Назовите код из SMS",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert '"risk": "high"' in proc_check.stdout


def test_hardened_vank_bank_o_bang_and_soobshite_translit() -> None:
    # 1. Literal "Ванk" from jury specification
    vals1, _ = extract_features(normalize("Ваш счет в Ванk заблокирован"))
    assert vals1["mentions_bank_or_operator"] == 1.0
    assert vals1["threatens_block"] == 1.0

    # 2. Latin leetspeak "B@nk"
    vals2, _ = extract_features(normalize("Vash schet v B@nk zablokirovan"))
    assert vals2["mentions_bank_or_operator"] == 1.0

    # 3. Standalone operator "O!"
    vals3, _ = extract_features(normalize("O! дарит вам 3 ГБ! Активируйте: https://gift-megacom.example"))
    assert vals3["mentions_bank_or_operator"] == 1.0
    assert vals3["promises_prize"] == 1.0

    # 4. Translit "Soobshite kod iz SMS"
    vals4, _ = extract_features(normalize("Vasha karta zablokirovana. Soobshite kod iz SMS po nomeru +996 700 265844"))
    assert vals4["asks_code_or_password"] == 1.0
    assert check("Vasha karta zablokirovana. Soobshite kod iz SMS po nomeru +996 700 265844")["risk"] == "high"


def test_hard_stress_patterns_domain_leet_kyrgyz_and_antifraud_ok() -> None:
    # Digit-homoglyph brand spoofing in hosts
    n1 = normalize("Beeline: вам доступен бесплатный пакет 30 ГБ. Активация: https://beel1ne-kg.example/gift")
    assert n1["domains"][0]["is_brand_spoof"] is True
    assert check(n1["original_text"])["risk"] == "high"

    n2 = normalize("Получите возврат комиссии 1800 сом на кошелёк: https://ba1ance-kg.example/refund")
    assert n2["domains"][0]["is_brand_spoof"] is True
    assert check(n2["original_text"])["risk"] == "high"

    # Colloquial RU/KG money requests & threats
    hard_frauds = [
        "Түндүк: сизге 3000 сом айып пул жазылды. Бүгүн төлөбөсөңүз эсеп жабылат: https://tunduk-pay.example/fine",
        "Извините пожалуйста, я по ошибке перевёл вам 1200 сом на баланс. Будьте добры, верните обратно на номер 0550 882144",
        "Байке, телефонум өчүп калды, 2500 сом салып турчу 0708 331920, кечинде кайтарам",
        "Братан, застрял на трассе, деньги на карте кончились. Закинь 2000 сом на 0772 614098, вечером отдам",
        "Salam dosum, jany nomerden jazyp jatam. 500 som salyp turchu 0703 441200, erten berem",
        "Balance.kg: ваш кошелёк временно ограничен. Пройдите проверку: https://balance.kg.verify.example/sms",
        "Демир Банк: необходимо обновить сертификат безопасности в течение часа, иначе вход будет закрыт: https://demir-cert.example",
        "Урматтуу абонент! Номериңизди сактап калуу үчүн 0771 445012 номерине чалып, келген сандарды айтыңыз",
    ]
    for sms in hard_frauds:
        res = check(sms)
        assert res["risk"] == "high", f"Missed hard fraud ({res['score']}): {sms}"

    # Everyday prize mentions & anti-fraud warnings must stay ok (< 65)
    hard_oks = [
        "Забрал приз за олимпиаду в универе! Вечером покажу фото",
        "Привет! Мы выиграли хакатон в INAI, приходи вечером отмечать!",
        "Осторожно, мошенники! Beeline никогда не просит продиктовать код из SMS или перевести деньги за приз",
        "Внимание! Не переходите по подозрительным ссылкам о выигрыше призов. Официальный сайт: https://beeline.kg",
    ]
    for sms in hard_oks:
        res = check(sms)
        assert res["risk"] != "high", f"False positive ({res['score']}): {sms}"


def test_bot_presentation_menus_and_file_handlers() -> None:
    import io
    import zipfile

    welcome = bot.build_welcome_html()
    assert "SMS Guard" in welcome
    assert ".csv" in welcome and ".docx" in welcome

    model_html = bot.format_model_info_html()
    assert "linear_features_tfidf" in model_html
    assert "score_shift" in model_html

    domains_html = bot.format_domains_html()
    assert "beeline.kg" in domains_html and "mbank.kg" in domains_html

    # 1. Sample CSV batch evaluation
    sample_csv = bot.build_sample_csv_bytes()
    summary_csv, out_csv = bot.analyze_csv_bytes(sample_csv, filename="sample_sms.csv")
    assert "Accuracy" in summary_csv
    assert "100.00%" in summary_csv
    assert b"D01,fraud,high" in out_csv

    # 2. Word (.docx) extraction & batch evaluation
    docx_buf = io.BytesIO()
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:r><w:t>Срочно! Ваша карта MBANK заблокирована, сообщите код из SMS по номеру 0555 123456</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Ваш код подтверждения: 451200. Никому не сообщайте его</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(docx_buf, "w") as zf:
        zf.writestr("word/document.xml", doc_xml)
    extracted = bot.extract_texts_from_docx_bytes(docx_buf.getvalue())
    assert len(extracted) == 2
    summary_docx, out_docx = bot.analyze_lines_batch(extracted, filename="demo.docx")
    assert "Проверено строк/абзацев: <b>2</b>" in summary_docx
    assert out_docx is not None and b"LINE_001,fraud,high" in out_docx

    # 3. TXT extraction
    txt_lines = bot.extract_texts_from_txt_bytes("Привет, как дела?\n".encode("utf-8"))
    summary_txt, out_txt = bot.analyze_lines_batch(txt_lines, filename="one.txt")
    assert "НИЗКИЙ РИСК" in summary_txt
    assert out_txt is None

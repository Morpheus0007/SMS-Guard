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
    env.pop("TELEGRAM_BOT_TOKEN", None)
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

"""Single-command launcher for SMS Guard (Streamlit Web UI + Telegram Bot, Evaluation, or CLI)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from src.config import BASE_DIR, DEFAULT_MODEL_PATH


def ensure_model_exists() -> None:
    """Train models/model.pkl on data/sms_train.csv if missing."""
    if not DEFAULT_MODEL_PATH.is_file():
        print("models/model.pkl not found — training from data/sms_train.csv...")
        from src.train import main as train_main

        train_main()


def _has_telegram_token() -> bool:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(BASE_DIR / ".env")
    except ImportError:
        pass
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    return bool(token) and token != "your_telegram_bot_token_here"


def main() -> int:
    parser = argparse.ArgumentParser(description="SMS Guard single-command launcher.")
    parser.add_argument("--eval", nargs="?", const="data/sms_dev.csv", help="Run evaluate.py on CSV")
    parser.add_argument("--bot", action="store_true", help="Launch ONLY Telegram bot")
    parser.add_argument("--no-bot", action="store_true", help="Launch ONLY Streamlit without Telegram bot")
    parser.add_argument("--check", help="Analyze single SMS text and print JSON")
    args = parser.parse_args()

    ensure_model_exists()

    if args.check is not None:
        return subprocess.call([sys.executable, str(BASE_DIR / "main.py"), args.check])
    if args.eval is not None:
        return subprocess.call([sys.executable, str(BASE_DIR / "evaluate.py"), args.eval])
    if args.bot:
        return subprocess.call([sys.executable, str(BASE_DIR / "bot.py")])

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "Streamlit is not installed in the current environment.\n"
            "Install dependencies via: pip install -r requirements.txt\n"
            "Or run CLI/evaluation directly without Streamlit:\n"
            "  python evaluate.py data/sms_dev.csv\n"
            '  python main.py "Срочно сообщите код из SMS"'
        )
        return 1

    bot_proc: subprocess.Popen[bytes] | None = None
    if not args.no_bot and _has_telegram_token():
        try:
            import aiogram  # noqa: F401

            print("Starting Telegram bot (bot.py) alongside Streamlit...")
            bot_proc = subprocess.Popen([sys.executable, str(BASE_DIR / "bot.py")])
        except ImportError:
            print("Note: aiogram is not installed — skipping Telegram bot startup.")
    elif not args.no_bot:
        print("Note: TELEGRAM_BOT_TOKEN is not set — launching Streamlit only.")

    try:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "app.py")])
    except KeyboardInterrupt:
        return 0
    finally:
        if bot_proc is not None and bot_proc.poll() is None:
            bot_proc.terminate()
            try:
                bot_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bot_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())

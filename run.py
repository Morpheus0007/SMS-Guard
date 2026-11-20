"""Single-command launcher for SMS Guard (Web UI, Evaluation, CLI, or Telegram Bot)."""

from __future__ import annotations

import argparse
import subprocess
import sys

from src.config import BASE_DIR, DEFAULT_MODEL_PATH


def ensure_model_exists() -> None:
    """Train models/model.pkl on data/sms_train.csv if missing."""
    if not DEFAULT_MODEL_PATH.is_file():
        print("models/model.pkl not found — training from data/sms_train.csv...")
        from src.train import main as train_main

        train_main()


def main() -> int:
    parser = argparse.ArgumentParser(description="SMS Guard single-command launcher.")
    parser.add_argument("--eval", nargs="?", const="data/sms_dev.csv", help="Run evaluate.py on CSV")
    parser.add_argument("--bot", action="store_true", help="Launch Telegram bot (requires TELEGRAM_BOT_TOKEN)")
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

    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "app.py")])


if __name__ == "__main__":
    raise SystemExit(main())

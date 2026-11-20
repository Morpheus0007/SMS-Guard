"""CLI entry point for SMS Guard (pure JSON mode for args/stdin, interactive TTY mode)."""

from __future__ import annotations

import argparse
import json
import sys

from src.detector import check, check_detailed


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _run_interactive() -> int:
    print("SMS Guard — детектор фишинговых SMS (введите пустую строку или 'exit' для выхода)")
    while True:
        try:
            text = input("\nВведите текст SMS: ")
        except EOFError:
            break
        if not text.strip() or text.strip().lower() in {"exit", "quit", "q"}:
            break
        res = check_detailed(text)
        category = "fraud" if res["risk"] == "high" else "ok"
        print(f"Исходный текст : {text}")
        print(f"Категория      : {category}")
        print(f"Уровень риска  : {res['risk'].upper()}")
        print(f"Оценка         : {res['score']} / 100")
        print(f"Причины        : {', '.join(res['reasons']) if res['reasons'] else 'нет'}")
        if res["advice"]:
            print("Рекомендация   :")
            for tip in res["advice"]:
                print(f"  - {tip}")
    return 0


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="SMS Guard CLI: pass SMS text as argument or stdin for pure JSON output."
    )
    parser.add_argument("text", nargs="*", help="SMS text to analyze")
    args = parser.parse_args()

    if args.text:
        sms_text = " ".join(args.text)
        print(json.dumps(check(sms_text), ensure_ascii=False))
        return 0

    if not sys.stdin.isatty():
        sms_text = sys.stdin.read().strip()
        print(json.dumps(check(sms_text), ensure_ascii=False))
        return 0

    return _run_interactive()


if __name__ == "__main__":
    raise SystemExit(main())

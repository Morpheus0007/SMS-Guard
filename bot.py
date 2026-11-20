"""Bonus Telegram bot for SMS Guard using aiogram 3.x."""

from __future__ import annotations

import asyncio
import html
import logging
import os
import sys

from src.config import BASE_DIR, OFFICIAL_DOMAINS
from src.detector import check_detailed

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sms_guard_bot")

MAX_SMS_LENGTH = 4000

SAMPLE_SMS: dict[str, tuple[str, str]] = {
    "demo_ru": (
        "🚨 Фишинг RU + обфускация",
        "С р о ч н о ! Ваша SIM-к@рт@ будет заблокирована. Перезвоните на 0770 553707 для подтверждения",
    ),
    "demo_kg": (
        "🚨 Фишинг KG / транслит",
        "РСК Банк: kartanyz bogottoldu. Tez arada kodduu jiberiniz 0555 538030",
    ),
    "demo_ok": (
        "✅ Обычное OTP от банка",
        "Код для входа в Бакай Банк Online: 457469. Не сообщайте его никому",
    ),
}


def format_sms_report_html(raw_text: str) -> str:
    """Format check_detailed(raw_text) into an HTML-escaped Telegram message."""
    res = check_detailed(raw_text)
    risk = res["risk"]
    score = res["score"]

    if risk == "high":
        header = "🔴 <b>ВЫСОКИЙ РИСК (Мошенничество)</b>"
    elif risk == "medium":
        header = "🟡 <b>СРЕДНИЙ РИСК (Подозрительно)</b>"
    else:
        header = "🟢 <b>НИЗКИЙ РИСК (Безопасно)</b>"

    lines = [header, f"Оценка риска: <b>{score}/100</b>"]

    if res["was_obfuscated"]:
        lines.append("⚠️ <i>Обнаружена маскировка текста (выполнена деобфускация).</i>")
        lines.append(f"Очищенный текст: <code>{html.escape(res['clean_text'])}</code>")

    if res["reasons"]:
        lines.append("\n<b>Причины:</b>")
        for reason in res["reasons"]:
            lines.append(f"• {html.escape(str(reason))}")

    if res["domains"]:
        lines.append("\n<b>Ссылки:</b>")
        for dom in res["domains"]:
            domain_escaped = html.escape(str(dom.get("domain", "")))
            if dom.get("is_official"):
                lines.append(f"✅ <code>{domain_escaped}</code> — официальный домен")
            elif dom.get("is_brand_spoof"):
                lines.append(f"⚠️ <code>{domain_escaped}</code> — подозрительный домен (имитация бренда)")
            else:
                lines.append(f"ℹ️ <code>{domain_escaped}</code> — внешний домен (не в белом списке)")

    if res["advice"]:
        lines.append("\n<b>Что делать:</b>")
        for tip in res["advice"]:
            lines.append(f"• {html.escape(str(tip))}")

    return "\n".join(lines)


async def run_bot(token: str) -> None:
    try:
        from aiogram import Bot, Dispatcher, F  # type: ignore
        from aiogram.client.default import DefaultBotProperties  # type: ignore
        from aiogram.enums import ParseMode  # type: ignore
        from aiogram.filters import Command  # type: ignore
        from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message  # type: ignore
    except ImportError as exc:
        print(f"Ошибка: пакет aiogram 3.x не установлен ({exc}). Установите: pip install aiogram")
        sys.exit(1)

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()

    def _build_demo_keyboard() -> InlineKeyboardMarkup:
        buttons = [
            [InlineKeyboardButton(text=label, callback_data=key)]
            for key, (label, _text) in SAMPLE_SMS.items()
        ]
        return InlineKeyboardMarkup(inline_keyboard=buttons)

    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        await message.answer(
            "🛡️ <b>SMS Guard</b>\n"
            "Перешлите или отправьте мне текст SMS-сообщения на русском, кыргызском или транслите — "
            "я оценю риск мошенничества от 0 до 100 и объясню причины.\n\n"
            "Или нажмите на готовый пример ниже:",
            reply_markup=_build_demo_keyboard(),
        )

    @dp.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        domains_list = ", ".join(f"<code>{d}</code>" for d in sorted(OFFICIAL_DOMAINS))
        await message.answer(
            "<b>Как пользоваться SMS Guard:</b>\n"
            "1. Просто отправьте или перешлите текст подозрительного SMS.\n"
            "2. Бот очистит маскировку («к о д», «б@нк»), проверит ссылки и оценит риск.\n\n"
            f"<b>Официальный белый список доменов:</b> {domains_list}\n"
            "Пороги риска: <code>low &lt; 30</code>, <code>medium 30–64</code>, <code>high &gt;= 65</code>."
        )

    @dp.callback_query(F.data.in_(SAMPLE_SMS.keys()))
    async def cb_sample(callback: CallbackQuery) -> None:
        _, sample_text = SAMPLE_SMS[str(callback.data)]
        report = format_sms_report_html(sample_text)
        if callback.message:
            await callback.message.answer(
                f"<b>Пример SMS:</b>\n<code>{html.escape(sample_text)}</code>\n\n{report}"
            )
        await callback.answer()

    @dp.message(F.text)
    async def handle_sms(message: Message) -> None:
        raw_text = message.text or ""
        if len(raw_text) > MAX_SMS_LENGTH:
            await message.answer(
                f"Сообщение слишком длинное (максимум {MAX_SMS_LENGTH} символов). Сократите текст и повторите попытку."
            )
            return
        try:
            report = format_sms_report_html(raw_text)
            await message.answer(report)
        except Exception as exc:
            logger.exception("Failed to analyze message: %s", exc)
            await message.answer("Произошла ошибка при анализе сообщения. Попробуйте позже.")

    logger.info("Starting SMS Guard Telegram bot...")
    await dp.start_polling(bot)


def main() -> int:
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        print(
            "TELEGRAM_BOT_TOKEN не задан. Укажите токен в переменной окружения или в файле .env, "
            "например:\n  TELEGRAM_BOT_TOKEN=123456:ABC... python bot.py"
        )
        return 0
    asyncio.run(run_bot(token))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

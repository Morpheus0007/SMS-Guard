"""Presentation-ready Telegram bot for SMS Guard using aiogram 3.x.

Features:
- Rich welcome message explaining purpose, capabilities, and quick-action menus.
- Single SMS inspection (RU / KG / Mixed / Translit / Obfuscated) with visual risk bar,
  deobfuscation "X-ray", domain verification, reasons, and safety advice.
- Batch document analysis directly in Telegram: .csv (labeled or unlabeled), .docx (Word),
  and .txt files, returning summary metrics and a downloadable results CSV.
- Interactive demo presets, model weights/metrics viewer, official whitelist reference,
  and downloadable sample CSV for live jury demonstrations.
"""

from __future__ import annotations

import asyncio
import csv
import html
import io
import logging
import os
import sys
import xml.etree.ElementTree as ET
import zipfile
from typing import Any

from src.config import BASE_DIR, OFFICIAL_DOMAINS
from src.detector import check, check_detailed, load_model_info
from src.scoring import compute_binary_metrics

try:
    from dotenv import load_dotenv  # type: ignore

    if not os.environ.get("SMS_GUARD_SKIP_DOTENV"):
        load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sms_guard_bot")

MAX_SMS_LENGTH = 4000
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_BATCH_ROWS = 2000

# Reply keyboard button labels
BTN_CHECK_SMS = "🔍 Проверить SMS"
BTN_UPLOAD_FILE = "📂 Проверить файл (CSV / Word / TXT)"
BTN_DEMO = "🎯 Демо-примеры"
BTN_MODEL = "🧠 О модели"
BTN_DOMAINS = "🌐 Белый список"
BTN_HELP = "ℹ️ Помощь"

SAMPLE_SMS: dict[str, tuple[str, str]] = {
    "demo_ru": (
        "🚨 Фишинг RU + обфускация",
        "С р о ч н о ! Ваша SIM-к@рт@ будет заблокирована. Перезвоните на 0770 553707 для подтверждения",
    ),
    "demo_kg": (
        "🚨 Фишинг KG / транслит",
        "РСК Банк: kartanyz bogottoldu. Tez arada kodduu jiberiniz 0555 538030",
    ),
    "demo_spoof": (
        "🚨 Подделка домена (beel1ne)",
        "Beeline: вам доступен бесплатный пакет 30 ГБ. Активация: https://beel1ne-kg.example/gift",
    ),
    "demo_ok_obf": (
        "✅ Обычное SMS + обфускация",
        "Заказ готов, 3а6ерuте в пункте выд@чu. Подробнее: https://univer.example/raspisanie",
    ),
    "demo_ok_otp": (
        "✅ Официальный банк (OTP)",
        "Код для входа в Бакай Банк Online: 457469. Не сообщайте его никому",
    ),
}


def _risk_bar(score: int, risk: str) -> str:
    filled = max(0, min(10, int(round(score / 10.0))))
    empty = 10 - filled
    block = "🟥" if risk == "high" else ("🟨" if risk == "medium" else "🟩")
    return f"{block * filled}{'⬜' * empty} <b>{score}/100</b>"


def _decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def build_welcome_html() -> str:
    """Rich structured welcome message for presentation & first-time users."""
    return (
        "🛡️ <b>SMS Guard — ИИ-защитник от фишинговых SMS</b>\n"
        "<i>Проект команды для Осеннего хакатона INAI 2026</i>\n\n"
        "<b>Для чего создан бот:</b>\n"
        "Помогает мгновенно распознавать SMS-мошенничество, поддельные ссылки "
        "банков/операторов, просьбы сообщить одноразовые коды и социальную инженерию.\n\n"
        "<b>Основные возможности:</b>\n"
        "• 🔍 <b>Анализ SMS за доли секунды</b> — русский, кыргызский (<code>ң/ө/ү</code>), "
        "смешанный язык и латинский транслит.\n"
        "• 🧪 <b>«Рентген» обфускации</b> — очищает замаскированный текст "
        "(<code>к о д</code>, <code>б@нк</code>, <code>Ванk</code>, <code>beel1ne-kg.example</code>).\n"
        "• 📂 <b>Проверка файлов</b> — прикрепите <code>.csv</code>, <code>.docx</code> (Word) "
        "или <code>.txt</code> и получите сводку + готовый отчёт <code>.csv</code>.\n"
        "• 🌐 <b>Проверка ссылок</b> по официальному белому списку КР.\n"
        "• 💡 <b>Объяснение причин и советы «Что делать»</b>.\n\n"
        "👇 <b>Выберите действие в меню ниже</b> или просто отправьте/перешлите текст SMS в чат:"
    )


def format_upload_guide_html() -> str:
    return (
        "📂 <b>Пакетная проверка файлов (CSV / Word / TXT)</b>\n\n"
        "Просто <b>прикрепите и отправьте файл</b> в этот чат — бот сам определит формат:\n\n"
        "• 📊 <b>CSV (<code>.csv</code>)</b> — колонки <code>id,text</code> или <code>id,text,label</code>.\n"
        "  Если есть колонка <code>label</code> (<code>fraud</code>/<code>ok</code>), бот сразу "
        "посчитает <b>Accuracy, Precision, Recall, F1</b> и матрицу ошибок!\n"
        "• 📝 <b>Word (<code>.docx</code>)</b> — извлечёт абзацы и таблицы документа и проверит каждое SMS.\n"
        "• 📄 <b>Текст (<code>.txt</code>)</b> — проверит каждую непустую строку как отдельное SMS.\n\n"
        "В ответ вы получите аккуратную сводку в чат и готовый файл результатов <code>.csv</code>.\n"
        "<i>Лимит: до 5 МБ и до 2 000 сообщений за раз.</i>"
    )


def format_domains_html() -> str:
    domains_lines = "\n".join(f"• ✅ <code>{d}</code> (и поддомены, например <code>my.{d}</code>)" for d in sorted(OFFICIAL_DOMAINS))
    return (
        "🌐 <b>Официальный белый список доменов</b>\n\n"
        f"{domains_lines}\n\n"
        "<b>Как работает проверка ссылок:</b>\n"
        "• Поддомены официальных сайтов разрешены (<code>my.beeline.kg</code> ✅).\n"
        "• Похожие суффиксы, кириллические и цифровые подмены "
        "(<code>mbank.kg-verify.example</code>, <code>beel1ne-kg.example</code>, "
        "<code>ba1ance-kg.example</code>) помечаются как <b>⚠️ имитация бренда</b>.\n"
        "• Обычные домены <code>.example</code> (доставка, расписание, билеты) сами по себе не считаются мошенничеством."
    )


def format_model_info_html() -> str:
    info = load_model_info()
    meta = info.get("meta") or {}
    weights = info.get("feature_weights") or {}
    sorted_w = sorted(weights.items(), key=lambda kv: float(kv[1]), reverse=True)
    top_pos = [(k, float(v)) for k, v in sorted_w if float(v) > 0][:5]
    top_neg = [(k, float(v)) for k, v in reversed(sorted_w) if float(v) < 0][:3]

    lines = [
        "🧠 <b>Как устроена модель SMS Guard</b>\n",
        f"• <b>Архитектура:</b> <code>{html.escape(str(info['model_type']))}</code> "
        "(12 канонических признаков + Word/Char TF-IDF)",
        f"• <b>Обучение:</b> L2 Logistic Regression на <code>sms_train.csv</code> "
        f"(<code>n={meta.get('n_train', 200)}</code>, без ручного задания весов)",
        f"• <b>Калибровка под порог 65:</b> <code>score_shift = {info['score_shift']:+.2f}</code> (логиты)",
    ]

    cv_m = meta.get("cv_metrics") or {}
    dev_m = meta.get("dev_metrics") or {}
    if cv_m:
        lines.append(
            f"• <b>5-fold CV (train):</b> Accuracy <b>{float(cv_m.get('accuracy', 0))*100:.1f}%</b>, "
            f"Recall <b>{float(cv_m.get('recall', 0))*100:.1f}%</b>"
        )
    if dev_m:
        lines.append(
            f"• <b>Открытый dev (40 SMS):</b> Accuracy <b>{float(dev_m.get('accuracy', 0))*100:.1f}%</b>, "
            f"Precision <b>{float(dev_m.get('precision', 0))*100:.1f}%</b>, "
            f"Recall <b>{float(dev_m.get('recall', 0))*100:.1f}%</b>"
        )

    if top_pos:
        lines.append("\n<b>Топ признаков риска (обученные веса):</b>")
        for k, v in top_pos:
            lines.append(f"• 🔺 <code>{html.escape(k)}</code>: <b>{v:+.2f}</b>")

    if top_neg:
        lines.append("\n<b>Защитные признаки (снижают риск):</b>")
        for k, v in top_neg:
            lines.append(f"• 🛡️ <code>{html.escape(k)}</code>: <b>{v:+.2f}</b>")

    lines.append("\n<b>Пороги:</b> 🟢 <code>low &lt; 30</code> | 🟡 <code>medium 30–64</code> | 🔴 <code>high &gt;= 65</code>")
    return "\n".join(lines)


def format_sms_report_html(raw_text: str) -> str:
    """Format check_detailed(raw_text) into an HTML-escaped Telegram message."""
    res = check_detailed(raw_text)
    risk = res["risk"]
    score = int(res["score"])

    if risk == "high":
        header = "🔴 <b>ВЫСОКИЙ РИСК (Мошенничество)</b>"
    elif risk == "medium":
        header = "🟡 <b>СРЕДНИЙ РИСК (Подозрительно)</b>"
    else:
        header = "🟢 <b>НИЗКИЙ РИСК (Безопасно)</b>"

    lines = [header, f"Шкала риска: {_risk_bar(score, risk)}"]

    if res["was_obfuscated"]:
        lines.append("\n🧪 <b>Рентген текста (выполнена деобфускация):</b>")
        lines.append(f"<code>{html.escape(res['clean_text'])}</code>")

    if res["reasons"]:
        lines.append("\n<b>Обнаруженные признаки:</b>")
        for reason in res["reasons"]:
            lines.append(f"• {html.escape(str(reason))}")

    if res["domains"]:
        lines.append("\n<b>Проверка ссылок:</b>")
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


def extract_texts_from_docx_bytes(raw: bytes) -> list[str]:
    """Extract non-empty paragraph/table rows from a .docx file using pure Python stdlib."""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        if "word/document.xml" not in zf.namelist():
            raise ValueError("Некорректный Word-файл: отсутствует word/document.xml")
        xml_bytes = zf.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    body = root.find("w:body", ns)
    if body is None:
        return []

    items: list[str] = []
    for child in body:
        tag = child.tag.split("}")[-1]
        if tag == "p":
            txt = "".join(t.text for t in child.findall(".//w:t", ns) if t.text).strip()
            if txt:
                items.append(txt)
        elif tag == "tbl":
            for tr in child.findall(".//w:tr", ns):
                cells = []
                for tc in tr.findall("./w:tc", ns):
                    c_txt = " ".join(
                        "".join(t.text for t in p.findall(".//w:t", ns) if t.text).strip()
                        for p in tc.findall("./w:p", ns)
                    ).strip()
                    if c_txt:
                        cells.append(c_txt)
                line = " | ".join(cells).strip()
                if line:
                    items.append(line)
    return items[:MAX_BATCH_ROWS]


def extract_texts_from_txt_bytes(raw: bytes) -> list[str]:
    """Extract non-empty lines from a .txt file."""
    text = _decode_bytes(raw)
    return [line.strip() for line in text.splitlines() if line.strip()][:MAX_BATCH_ROWS]


def _rows_to_results_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "label", "risk", "score", "reasons", "text"])
    for item in rows:
        writer.writerow(
            [
                item["id"],
                item["pred"],
                item["risk"],
                item["score"],
                "; ".join(item["reasons"]),
                item["text"],
            ]
        )
    return buf.getvalue().encode("utf-8-sig")


def analyze_csv_bytes(raw: bytes, filename: str = "file.csv") -> tuple[str, bytes]:
    """Analyze uploaded .csv (`id,text` or `id,text,label`) and return (HTML summary, results CSV bytes)."""
    decoded = _decode_bytes(raw)
    reader = csv.DictReader(io.StringIO(decoded))
    fieldnames = list(reader.fieldnames or [])
    if "text" not in fieldnames:
        raise ValueError("В CSV-файле не найдена обязательная колонка <code>text</code>.")

    rows_in: list[dict[str, str]] = []
    for idx, row in enumerate(reader, start=1):
        if len(rows_in) >= MAX_BATCH_ROWS:
            break
        if not row or not (row.get("text") or "").strip():
            continue
        rows_in.append(
            {
                "id": str(row.get("id") or f"ROW_{idx:03d}").strip(),
                "text": str(row.get("text") or ""),
                "label": str(row.get("label") or "").strip().lower(),
            }
        )

    if not rows_in:
        raise ValueError("Файл пуст или не содержит текстовых SMS.")

    has_label = "label" in fieldnames and all(r["label"] in ("fraud", "ok") for r in rows_in)

    results: list[dict[str, Any]] = []
    counts = {"high": 0, "medium": 0, "low": 0}
    for row in rows_in:
        res = check(row["text"])
        pred = "fraud" if res["risk"] == "high" else "ok"
        counts[res["risk"]] += 1
        results.append(
            {
                "id": row["id"],
                "text": row["text"],
                "truth": row["label"],
                "pred": pred,
                "risk": res["risk"],
                "score": res["score"],
                "reasons": res["reasons"],
            }
        )

    lines = [
        f"📊 <b>Отчёт по CSV:</b> <code>{html.escape(filename)}</code>",
        f"Проверено сообщений: <b>{len(results)}</b>",
        f"🔴 Высокий риск (fraud): <b>{counts['high']}</b> | "
        f"🟡 Средний: <b>{counts['medium']}</b> | "
        f"🟢 Низкий: <b>{counts['low']}</b>",
    ]

    if has_label:
        m = compute_binary_metrics([r["truth"] for r in results], [r["pred"] for r in results])
        lines.append("\n<b>Метрики качества (high == fraud):</b>")
        lines.append(
            f"• Accuracy: <b>{float(m['accuracy']) * 100:.2f}%</b>\n"
            f"• Precision (fraud): <b>{float(m['precision']) * 100:.2f}%</b>\n"
            f"• Recall (fraud): <b>{float(m['recall']) * 100:.2f}%</b>\n"
            f"• F1: <b>{float(m['f1']):.3f}</b>\n"
            f"• Матрица ошибок: <code>TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}</code>"
        )
        errors = [r for r in results if r["truth"] != r["pred"]]
        if errors:
            lines.append(f"\n<b>Ошибки ({len(errors)}, показаны первые 5):</b>")
            for err in errors[:5]:
                tag = "FP" if err["pred"] == "fraud" else "FN"
                preview = html.escape(err["text"][:80])
                lines.append(f"• [{tag}] <code>{html.escape(err['id'])}</code> ({err['score']}/100): {preview}")
    else:
        flagged = sorted(
            [r for r in results if r["risk"] == "high"],
            key=lambda x: x["score"],
            reverse=True,
        )
        if flagged:
            lines.append(f"\n<b>Топ опасных SMS (показаны {min(5, len(flagged))} из {len(flagged)}):</b>")
            for item in flagged[:5]:
                preview = html.escape(item["text"][:80])
                lines.append(f"• 🔴 <code>{html.escape(item['id'])}</code> ({item['score']}/100): {preview}")

    lines.append("\n📎 Полная таблица результатов прикреплена ниже в файле <code>.csv</code>.")
    return "\n".join(lines), _rows_to_results_csv_bytes(results)


def analyze_lines_batch(texts: list[str], filename: str) -> tuple[str, bytes | None]:
    """Analyze extracted lines from a .docx or .txt file."""
    if not texts:
        raise ValueError("В документе не найдено текста для проверки.")
    if len(texts) == 1:
        single_header = f"📄 <b>Файл:</b> <code>{html.escape(filename)}</code>\n\n"
        return single_header + format_sms_report_html(texts[0]), None

    results: list[dict[str, Any]] = []
    counts = {"high": 0, "medium": 0, "low": 0}
    for idx, txt in enumerate(texts, start=1):
        res = check(txt)
        pred = "fraud" if res["risk"] == "high" else "ok"
        counts[res["risk"]] += 1
        results.append(
            {
                "id": f"LINE_{idx:03d}",
                "text": txt,
                "pred": pred,
                "risk": res["risk"],
                "score": res["score"],
                "reasons": res["reasons"],
            }
        )

    lines = [
        f"📝 <b>Отчёт по документу:</b> <code>{html.escape(filename)}</code>",
        f"Проверено строк/абзацев: <b>{len(results)}</b>",
        f"🔴 Высокий риск (fraud): <b>{counts['high']}</b> | "
        f"🟡 Средний: <b>{counts['medium']}</b> | "
        f"🟢 Низкий: <b>{counts['low']}</b>",
    ]
    flagged = sorted(
        [r for r in results if r["risk"] == "high"],
        key=lambda x: x["score"],
        reverse=True,
    )
    if flagged:
        lines.append(f"\n<b>Обнаруженные фишинговые сообщения (первые {min(5, len(flagged))}):</b>")
        for item in flagged[:5]:
            preview = html.escape(item["text"][:85])
            lines.append(f"• 🔴 <b>{item['score']}/100</b>: {preview}")
    else:
        lines.append("\n✅ Мошеннических сообщений высокого риска не обнаружено.")

    lines.append("\n📎 Полный отчёт по всем строкам прикреплён ниже в файле <code>.csv</code>.")
    return "\n".join(lines), _rows_to_results_csv_bytes(results)


def build_sample_csv_bytes() -> bytes:
    """Generate a ready-to-demo labeled CSV so jury/users can immediately test file upload."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "text", "label"])
    writer.writerow(["D01", "С р о ч н о ! Ваша SIM-к@рт@ будет заблокирована. Перезвоните на 0770 553707", "fraud"])
    writer.writerow(["D02", "РСК Банк: kartanyz bogottoldu. Tez arada kodduu jiberiniz 0555 538030", "fraud"])
    writer.writerow(["D03", "Beeline: вам доступен бесплатный пакет 30 ГБ. Активация: https://beel1ne-kg.example/gift", "fraud"])
    writer.writerow(["D04", "Код для входа в Бакай Банк Online: 457469. Не сообщайте его никому", "ok"])
    writer.writerow(["D05", "Заказ готов, 3а6ерuте в пункте выд@чu. Подробнее: https://univer.example/raspisanie", "ok"])
    writer.writerow(["D06", "Алдамчылардан сак болуңуз: утуш үчүн акча которбоңуз. MegaCom", "ok"])
    return buf.getvalue().encode("utf-8-sig")


async def run_bot(token: str) -> None:
    try:
        from aiogram import Bot, Dispatcher, F  # type: ignore
        from aiogram.client.default import DefaultBotProperties  # type: ignore
        from aiogram.enums import ParseMode  # type: ignore
        from aiogram.filters import Command  # type: ignore
        from aiogram.types import (  # type: ignore
            BotCommand,
            BufferedInputFile,
            CallbackQuery,
            InlineKeyboardButton,
            InlineKeyboardMarkup,
            KeyboardButton,
            Message,
            ReplyKeyboardMarkup,
        )
    except ImportError as exc:
        print(f"Ошибка: пакет aiogram 3.x не установлен ({exc}). Установите: pip install aiogram")
        sys.exit(1)

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()

    await bot.set_my_commands(
        [
            BotCommand(command="start", description="Главное меню и описание SMS Guard"),
            BotCommand(command="demo", description="Готовые примеры SMS для демонстрации"),
            BotCommand(command="file", description="Проверка файлов CSV / Word (.docx) / TXT"),
            BotCommand(command="model", description="Архитектура, веса и метрики модели"),
            BotCommand(command="domains", description="Официальный белый список доменов"),
            BotCommand(command="help", description="Справка по работе бота"),
        ]
    )

    def _main_reply_keyboard() -> ReplyKeyboardMarkup:
        return ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text=BTN_CHECK_SMS), KeyboardButton(text=BTN_UPLOAD_FILE)],
                [KeyboardButton(text=BTN_DEMO), KeyboardButton(text=BTN_MODEL)],
                [KeyboardButton(text=BTN_DOMAINS), KeyboardButton(text=BTN_HELP)],
            ],
            resize_keyboard=True,
            input_field_placeholder="Отправьте SMS или прикрепите CSV / Word / TXT...",
        )

    def _start_inline_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🎯 Запустить демо-примеры", callback_data="menu_demo")],
                [InlineKeyboardButton(text="📂 Проверка CSV / Word / TXT", callback_data="menu_file")],
                [
                    InlineKeyboardButton(text="🧠 О модели", callback_data="menu_model"),
                    InlineKeyboardButton(text="🌐 Домены", callback_data="menu_domains"),
                ],
            ]
        )

    def _demo_inline_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text=label, callback_data=key)]
                for key, (label, _text) in SAMPLE_SMS.items()
            ]
        )

    def _file_inline_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="📥 Скачать шаблон sample_sms.csv", callback_data="get_sample_csv")]
            ]
        )

    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        await message.answer(
            build_welcome_html(),
            reply_markup=_main_reply_keyboard(),
        )
        await message.answer(
            "⚡ <b>Быстрый старт для презентации:</b>",
            reply_markup=_start_inline_keyboard(),
        )

    @dp.message(Command("demo"))
    @dp.message(F.text == BTN_DEMO)
    async def cmd_demo(message: Message) -> None:
        await message.answer(
            "🎯 <b>Выберите готовый пример SMS для мгновенной проверки:</b>\n"
            "<i>Нажмите на любую кнопку ниже, чтобы увидеть деобфускацию, проверку ссылок и советы.</i>",
            reply_markup=_demo_inline_keyboard(),
        )

    @dp.message(Command("file"))
    @dp.message(F.text == BTN_UPLOAD_FILE)
    async def cmd_file_help(message: Message) -> None:
        await message.answer(
            format_upload_guide_html(),
            reply_markup=_file_inline_keyboard(),
        )

    @dp.message(Command("model"))
    @dp.message(F.text == BTN_MODEL)
    async def cmd_model(message: Message) -> None:
        await message.answer(format_model_info_html())

    @dp.message(Command("domains"))
    @dp.message(F.text == BTN_DOMAINS)
    async def cmd_domains(message: Message) -> None:
        await message.answer(format_domains_html())

    @dp.message(Command("help"))
    @dp.message(F.text == BTN_HELP)
    async def cmd_help(message: Message) -> None:
        await message.answer(
            build_welcome_html(),
            reply_markup=_main_reply_keyboard(),
        )

    @dp.message(F.text == BTN_CHECK_SMS)
    async def cmd_prompt_sms(message: Message) -> None:
        await message.answer(
            "🔍 <b>Отправьте или перешлите текст SMS прямо в этот чат.</b>\n"
            "Поддерживаются русский, кыргызский (<code>ң/ө/ү</code>), латинский транслит и замаскированный текст.",
            reply_markup=_demo_inline_keyboard(),
        )

    @dp.callback_query(F.data == "menu_demo")
    async def cb_menu_demo(callback: CallbackQuery) -> None:
        if callback.message:
            await callback.message.answer(
                "🎯 <b>Выберите пример SMS:</b>",
                reply_markup=_demo_inline_keyboard(),
            )
        await callback.answer()

    @dp.callback_query(F.data == "menu_file")
    async def cb_menu_file(callback: CallbackQuery) -> None:
        if callback.message:
            await callback.message.answer(
                format_upload_guide_html(),
                reply_markup=_file_inline_keyboard(),
            )
        await callback.answer()

    @dp.callback_query(F.data == "menu_model")
    async def cb_menu_model(callback: CallbackQuery) -> None:
        if callback.message:
            await callback.message.answer(format_model_info_html())
        await callback.answer()

    @dp.callback_query(F.data == "menu_domains")
    async def cb_menu_domains(callback: CallbackQuery) -> None:
        if callback.message:
            await callback.message.answer(format_domains_html())
        await callback.answer()

    @dp.callback_query(F.data == "get_sample_csv")
    async def cb_sample_csv(callback: CallbackQuery) -> None:
        if callback.message:
            sample_file = BufferedInputFile(build_sample_csv_bytes(), filename="sample_sms.csv")
            await callback.message.answer_document(
                sample_file,
                caption=(
                    "📥 Готовый тестовый файл <code>sample_sms.csv</code> (6 SMS с метками).\n"
                    "Перешлите его обратно боту, чтобы увидеть пакетный расчёт метрик и отчёт!"
                ),
            )
        await callback.answer()

    @dp.callback_query(F.data.in_(SAMPLE_SMS.keys()))
    async def cb_sample(callback: CallbackQuery) -> None:
        _, sample_text = SAMPLE_SMS[str(callback.data)]
        report = format_sms_report_html(sample_text)
        if callback.message:
            await callback.message.answer(
                f"📩 <b>Проверяемое SMS:</b>\n<code>{html.escape(sample_text)}</code>\n\n{report}"
            )
        await callback.answer()

    @dp.message(F.document)
    async def handle_document(message: Message) -> None:
        doc = message.document
        if doc is None:
            return
        filename = doc.file_name or "document"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in {".csv", ".docx", ".txt"}:
            await message.answer(
                "⚠️ Неподдерживаемый формат файла.\n"
                "Отправьте файл в одном из форматов: <code>.csv</code>, <code>.docx</code> (Word) или <code>.txt</code>."
            )
            return
        if doc.file_size and doc.file_size > MAX_FILE_SIZE_BYTES:
            await message.answer("⚠️ Файл слишком большой (максимум 5 МБ).")
            return

        status_msg = await message.answer(f"⏳ Анализирую файл <code>{html.escape(filename)}</code>...")
        try:
            tg_file = await bot.get_file(doc.file_id)
            buf = io.BytesIO()
            await bot.download_file(tg_file.file_path, destination=buf)
            raw_bytes = buf.getvalue()

            if ext == ".csv":
                summary_html, report_csv = analyze_csv_bytes(raw_bytes, filename=filename)
                out_name = f"results_{os.path.splitext(filename)[0]}.csv"
            elif ext == ".docx":
                lines_list = extract_texts_from_docx_bytes(raw_bytes)
                summary_html, report_csv = analyze_lines_batch(lines_list, filename=filename)
                out_name = f"results_{os.path.splitext(filename)[0]}.csv"
            else:
                lines_list = extract_texts_from_txt_bytes(raw_bytes)
                summary_html, report_csv = analyze_lines_batch(lines_list, filename=filename)
                out_name = f"results_{os.path.splitext(filename)[0]}.csv"

            await status_msg.edit_text(summary_html)
            if report_csv is not None:
                await message.answer_document(
                    BufferedInputFile(report_csv, filename=out_name),
                    caption="📊 Подробный отчёт SMS Guard по всем проверенным сообщениям.",
                )
        except ValueError as ve:
            await status_msg.edit_text(f"⚠️ {ve}")
        except Exception as exc:
            logger.exception("Failed to process document %s: %s", filename, exc)
            await status_msg.edit_text("⚠️ Не удалось прочитать файл. Проверьте формат и кодировку (UTF-8).")

    @dp.message(F.text)
    async def handle_sms(message: Message) -> None:
        raw_text = message.text or ""
        if len(raw_text) > MAX_SMS_LENGTH:
            await message.answer(
                f"⚠️ Сообщение слишком длинное (максимум {MAX_SMS_LENGTH} символов). Сократите текст и повторите попытку."
            )
            return
        try:
            report = format_sms_report_html(raw_text)
            await message.answer(report)
        except Exception as exc:
            logger.exception("Failed to analyze message: %s", exc)
            await message.answer("⚠️ Произошла ошибка при анализе сообщения. Попробуйте позже.")

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

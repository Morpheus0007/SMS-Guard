"""Streamlit web dashboard for SMS Guard."""

from __future__ import annotations

import csv
import io
from typing import Any

import streamlit as st

from src.config import OFFICIAL_DOMAINS
from src.detector import check, check_detailed, load_model_info
from src.scoring import compute_binary_metrics

st.set_page_config(page_title="SMS Guard", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_cached_model_info() -> dict[str, Any]:
    return load_model_info()


def _init_state() -> None:
    if "sms_input" not in st.session_state:
        st.session_state.sms_input = ""
    if "history" not in st.session_state:
        st.session_state.history = []


def _render_risk_badge(risk: str, score: int) -> None:
    col1, col2 = st.columns([1, 3])
    col1.metric("Оценка риска", f"{score} / 100")
    if risk == "high":
        col2.error("🔴 ВЫСОКИЙ РИСК (Мошенничество / Fraud)")
    elif risk == "medium":
        col2.warning("🟡 СРЕДНИЙ РИСК (Подозрительно)")
    else:
        col2.success("🟢 НИЗКИЙ РИСК (Безопасно / OK)")
    st.progress(max(0, min(100, int(score))))


def _render_tab_single() -> None:
    st.subheader("Быстрые примеры для демонстрации")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("🚨 Фишинг + обфускация", use_container_width=True):
        st.session_state.sms_input = (
            "С р о ч н о ! Ваша SIM-к@рт@ будет заблокирована. Перезвоните на 0770 553707 для подтверждения"
        )
    if c2.button("🚨 Фишинг KG / транслит", use_container_width=True):
        st.session_state.sms_input = (
            "РСК Банк: kartanyz bogottoldu. Tez arada kodduu jiberiniz 0555 538030"
        )
    if c3.button("✅ Официальный банк (OTP)", use_container_width=True):
        st.session_state.sms_input = (
            "Код для входа в Бакай Банк Online: 457469. Не сообщайте его никому"
        )
    if c4.button("✅ Обычная ссылка (.example)", use_container_width=True):
        st.session_state.sms_input = (
            "Заказ готов, 3а6ерuте в пункте выд@чu. Подробнее: https://univer.example/raspisanie"
        )

    sms_text = st.text_area("Текст SMS-сообщения:", key="sms_input", height=120)
    if st.button("Проверить сообщение", type="primary", use_container_width=True):
        if not sms_text.strip():
            st.warning("Введите текст SMS для проверки.")
        else:
            res = check_detailed(sms_text)
            st.divider()
            _render_risk_badge(res["risk"], res["score"])

            if res["was_obfuscated"]:
                st.info("Обнаружена маскировка текста — выполнен «Рентген» деобфускации:")
                col_raw, col_clean = st.columns(2)
                col_raw.text_area("Исходный текст", value=sms_text, height=90, disabled=True)
                col_clean.text_area(
                    "После деобфускации", value=res["clean_text"], height=90, disabled=True
                )

            if res["domains"]:
                st.subheader("Проверка ссылок и доменов")
                for d in res["domains"]:
                    if d["is_official"]:
                        st.success(f"✅ Официальный домен: `{d['domain']}` ({d['url']})")
                    elif d["is_brand_spoof"]:
                        st.error(
                            f"⚠️ Подозрительный домен (имитация бренда/сокращатель): `{d['domain']}` ({d['url']})"
                        )
                    else:
                        st.info(f"ℹ️ Внешний домен (не в белом списке): `{d['domain']}` ({d['url']})")

            col_r, col_a = st.columns(2)
            with col_r:
                st.subheader("Обнаруженные признаки")
                if res["reasons"]:
                    for reason in res["reasons"]:
                        st.markdown(f"- {reason}")
                else:
                    st.write("Подозрительные признаки не выявлены.")

            with col_a:
                st.subheader("Что делать абоненту")
                for tip in res["advice"]:
                    st.markdown(f"- {tip}")

            active_contribs = [
                {"Признак": k, "Вклад (логит)": v}
                for k, v in res["feature_contributions"].items()
                if res["feature_values"].get(k, 0.0) > 0.0
            ]
            if active_contribs:
                with st.expander("Разбор вклада сработавших признаков"):
                    st.dataframe(active_contribs, use_container_width=True)

            st.session_state.history.insert(
                0,
                {
                    "text": sms_text[:80] + ("..." if len(sms_text) > 80 else ""),
                    "risk": res["risk"],
                    "score": res["score"],
                    "reasons": ", ".join(res["reasons"]) or "—",
                },
            )
            st.session_state.history = st.session_state.history[:10]

    if st.session_state.history:
        st.divider()
        st.subheader("История проверок сессии")
        st.dataframe(st.session_state.history, use_container_width=True)


def _render_tab_csv() -> None:
    st.subheader("Пакетная проверка CSV (`id,text` или `id,text,label`)")
    uploaded = st.file_uploader("Загрузите CSV-файл", type=["csv"])
    if uploaded is None:
        return

    raw_text = uploaded.getvalue().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(raw_text))
    fieldnames = list(reader.fieldnames or [])
    if "text" not in fieldnames:
        st.error("В CSV отсутствует обязательная колонка `text`.")
        return

    rows = [r for r in reader if r and (r.get("text") or "").strip()]
    has_label = "label" in fieldnames and all(
        (r.get("label") or "").strip().lower() in ("fraud", "ok") for r in rows
    ) and len(rows) > 0

    output_rows: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []

    for idx, row in enumerate(rows, start=1):
        sms_id = (row.get("id") or f"ROW_{idx:03d}").strip()
        text = str(row.get("text") or "")
        res = check(text)
        pred = "fraud" if res["risk"] == "high" else "ok"
        item: dict[str, Any] = {
            "id": sms_id,
            "risk": res["risk"],
            "score": res["score"],
            "pred": pred,
            "reasons": ", ".join(res["reasons"]),
            "text": text,
        }
        if has_label:
            truth = str(row.get("label") or "").strip().lower()
            item["label"] = truth
            item["is_error"] = truth != pred
            y_true.append(truth)
            y_pred.append(pred)
        output_rows.append(item)

    if has_label:
        m = compute_binary_metrics(y_true, y_pred)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{m['accuracy'] * 100:.2f}%")
        m2.metric("Precision (fraud)", f"{m['precision'] * 100:.2f}%")
        m3.metric("Recall (fraud)", f"{m['recall'] * 100:.2f}%")
        m4.metric("F1", f"{m['f1']:.3f}")

        st.write(
            f"**Confusion Matrix:** TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}"
        )
        only_errors = st.checkbox("Показать только ошибки (FP/FN)")
        shown = [r for r in output_rows if not only_errors or r["is_error"]]
    else:
        shown = output_rows

    st.dataframe(shown, use_container_width=True)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "label", "risk", "score"])
    for r in output_rows:
        writer.writerow([r["id"], r["pred"], r["risk"], r["score"]])
    st.download_button(
        "Скачать predictions.csv",
        data=buf.getvalue().encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )


def _render_tab_model(info: dict[str, Any]) -> None:
    st.subheader("Архитектура и веса обученной модели (`models/model.pkl`)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Тип модели", info["model_type"])
    c2.metric("Версия контракта", info["checkpoint_version"])
    c3.metric("Intercept", f"{info['intercept']:.3f}")
    c4.metric("Score Shift (логиты)", f"{info['score_shift']:+.3f}")

    meta = info.get("meta") or {}
    if meta:
        ml_src = info.get("ml_team_source")
        src_note = f", `ML-модель={ml_src}`" if ml_src else ""
        st.write(
            f"**Обучена на:** `n_train={meta.get('n_train', '?')}`, "
            f"`used_dev_for_training={meta.get('used_dev_for_training', False)}`, "
            f"`TF-IDF={'да' if info.get('has_tfidf') else 'нет'}`{src_note}"
        )
        if meta.get("cv_metrics"):
            st.write("**5-fold CV (train):**", meta["cv_metrics"])
        if meta.get("dev_metrics"):
            st.write("**Dev metrics (`sms_dev.csv`):**", meta["dev_metrics"])

    weights = info.get("feature_weights") or {}
    chart_rows = [
        {"Признак": k, "Вес": float(v)}
        for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)
    ]
    st.bar_chart(data=chart_rows, x="Признак", y="Вес", use_container_width=True)

    st.subheader("Официальный белый список доменов и пороги риска")
    st.write("**Разрешённые официальные домены (и их поддомены):**", sorted(OFFICIAL_DOMAINS))
    st.write("**Пороги риска:** `low < 30`, `medium 30–64`, `high >= 65` (`high` считается `fraud`).")


def main() -> None:
    _init_state()
    info = get_cached_model_info()
    st.title("🛡️ SMS Guard")
    st.caption(
        "Детектор фишинговых SMS (Русский / Кыргызский / Латинский транслит / Деобфускация)"
    )

    tab1, tab2, tab3 = st.tabs(
        ["Проверка SMS", "Пакетная проверка CSV", "Как думает модель (.pkl)"]
    )
    with tab1:
        _render_tab_single()
    with tab2:
        _render_tab_csv()
    with tab3:
        _render_tab_model(info)


if __name__ == "__main__":
    main()

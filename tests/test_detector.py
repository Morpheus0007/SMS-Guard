"""Unit tests for detector.py, advisor.py, risk thresholds, and Format B sklearn loading."""

from __future__ import annotations

import pytest

from src.advisor import get_advice
from src.config import FEATURE_NAMES
from src.detector import check, check_detailed, load_model_info
from src.model_loader import LoadedSMSModel


def _assert_contract(res: dict) -> None:
    assert set(res.keys()) == {"risk", "score", "reasons"}
    assert res["risk"] in {"low", "medium", "high"}
    assert isinstance(res["score"], int)
    assert 0 <= res["score"] <= 100
    assert isinstance(res["reasons"], list)
    assert all(isinstance(r, str) and r.strip() for r in res["reasons"])
    if res["score"] >= 65:
        assert res["risk"] == "high"
    elif res["score"] >= 30:
        assert res["risk"] == "medium"
    else:
        assert res["risk"] == "low"


def test_check_strict_schema_and_boundaries() -> None:
    for sample in (
        "С р о ч н о ! Ваша SIM-к@рт@ будет заблокирована. Перезвоните на 0770 553707",
        "Код для входа в Бакай Банк Online: 457469. Не сообщайте его никому",
        "Привет! Во сколько завтра встречаемся?",
        "",
    ):
        res = check(sample)
        _assert_contract(res)


@pytest.mark.parametrize(
    "text",
    [
        "Bonus 500 s o m sizge! O! aktivdeshtiruu: https://kicb-online.example/card",
        "Привет, это я с нового номера. Скинь 150 сом срочно на 0557 743693, верну завтра",
        "РСК Банк: kartanyz bogottoldu. Tez arada kodduu jiberiniz 0555 538030",
        "Сиздин KICB эсебиңизге кирүү бөгөттөлдү, ырастоо үчүн https://secure-optima.example/login өтүңүз",
        "П03др@вляем! Вы в ы и г р а л и iPhone 15. Для получения прu3а перейдите по ссылке https://secure-optima.example/login",
        "Ваш аккаунт Демир Банк Online будет заблокирован. Назовите код из SMS нашему сотруднику",
        "Вам ошибочно зачислено 2000 сом. Пожалуйста, верните на номер 0705 537451",
    ],
)
def test_real_fraud_cases_detected_as_high(text: str) -> None:
    res = check(text)
    _assert_contract(res)
    assert res["risk"] == "high", f"Missed fraud ({res['score']}): {text}"
    assert len(res["reasons"]) >= 1


@pytest.mark.parametrize(
    "text",
    [
        "Код для входа в Бакай Банк Online: 457469. Не сообщайте его никому",
        "Заказ готов, 3а6ерuте в пункте выд@чu. Подробнее: https://univer.example/raspisanie",
        "KICB: в а ш кредuт од0брен, подробности в приложении или на https://mbank.kg",
        "Плановые работы с 01:00 до 03:00, ничего подтверждать не нужно. MegaCom",
        "Алдамчылардан сак болуңуз: утуш үчүн акча которбоңуз. MegaCom",
        "Срочно перезвони, когда освободишься, по поводу дачи",
        "Сиздин 10 ГБ пакетиңиз uйгuлuктүү тут@штырылды. B e e l i n e",
        "Напоминаем: 0пл@тите домашний и н т е р н е т д0 03.12.2026. Оплата: https://balance.kg",
    ],
)
def test_safe_and_obfuscated_ok_cases_not_high(text: str) -> None:
    res = check(text)
    _assert_contract(res)
    assert res["risk"] != "high", f"False positive ({res['score']}): {text}"


@pytest.mark.parametrize("val", [None, "", "   ", "🔥🔥🔥", "a" * 5200, 999])
def test_edge_inputs_no_crash(val: object) -> None:
    res = check(val)
    _assert_contract(res)


def test_empty_string_exact_response() -> None:
    assert check("") == {"risk": "low", "score": 0, "reasons": []}


def test_detailed_advisor_and_model_info() -> None:
    det = check_detailed("Ваш пароль от РСК Банк Online истекает. Обновите его по ссылке https://nbkr-check.example")
    assert det["risk"] == "high"
    assert len(det["advice"]) >= 1
    assert det["domains"][0]["is_brand_spoof"] is True

    info = load_model_info()
    assert tuple(info["feature_names"]) == FEATURE_NAMES
    assert set(info["feature_weights"].keys()) == set(FEATURE_NAMES)

    tips = get_advice("high", ["asks_code_or_password"], [])
    assert any("SMS-коды" in t for t in tips)


def test_format_b_sklearn_compatible_loader() -> None:
    class MockEstimator:
        n_features_in_ = 12
        coef_ = [[0.5, 2.5, 1.0, 1.5, 1.5, 0.0, 1.8, 0.8, -1.5, 2.0, -2.0, 1.2]]
        intercept_ = [-2.0]

        def decision_function(self, X):
            row = X[0]
            return [self.intercept_[0] + sum(c * v for c, v in zip(self.coef_[0], row))]

    wrapped = {
        "checkpoint_version": "3.1-sklearn",
        "model": MockEstimator(),
        "input_mode": "features_12",
        "score_shift": 0.6,
    }
    loaded = LoadedSMSModel(wrapped, source_path="<mock>")
    assert loaded.format_kind == "sklearn_wrapped"
    from src.preprocessing import normalize
    from src.features import extract_features

    nm = normalize("Срочно сообщите код из SMS по номеру 0555 123456")
    vals, _ = extract_features(nm)
    z_raw, contribs = loaded.predict_logit(nm, vals)
    assert z_raw > 1.0
    assert contribs["asks_code_or_password"] == 2.5

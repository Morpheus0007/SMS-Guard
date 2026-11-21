"""Contract v3.1 acceptance tests for models/model.pkl and end-to-end dev metrics."""

from __future__ import annotations

import csv
import math
import pickle
import time

from src.config import DEFAULT_DEV_CSV, DEFAULT_MODEL_PATH, FEATURE_NAMES
from src.detector import check, reload_model
from src.scoring import compute_binary_metrics


def _load_dev_rows() -> list[dict[str, str]]:
    with open(DEFAULT_DEV_CSV, "r", encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r and r.get("text")]


def test_model_pkl_exists_and_size_limit() -> None:
    assert DEFAULT_MODEL_PATH.is_file(), f"Missing {DEFAULT_MODEL_PATH}"
    size_bytes = DEFAULT_MODEL_PATH.stat().st_size
    assert 100 < size_bytes <= 95 * 1024 * 1024


def test_model_pkl_schema_and_weights_valid() -> None:
    with open(DEFAULT_MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    assert isinstance(payload, dict)
    assert payload.get("checkpoint_version") == "3.1"
    assert payload.get("model_type") in {"linear_features", "linear_features_tfidf", "ml_team_hybrid"}
    assert tuple(payload["feature_names"]) == tuple(FEATURE_NAMES)

    fw = payload["feature_weights"]
    assert set(fw.keys()) == set(FEATURE_NAMES)
    for fname in FEATURE_NAMES:
        val = float(fw[fname])
        assert not math.isnan(val) and not math.isinf(val)

    # Protective signals must not increase fraud risk
    assert float(fw["has_official_domain"]) <= 0.0
    assert float(fw["has_safe_warning"]) <= 0.0
    assert payload["meta"]["used_dev_for_training"] is False


def test_dev_metrics_match_and_meet_kpi() -> None:
    reload_model(DEFAULT_MODEL_PATH)
    with open(DEFAULT_MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    dev_rows = _load_dev_rows()
    y_true = [r["label"].strip().lower() for r in dev_rows]
    y_pred = ["fraud" if check(r["text"])["risk"] == "high" else "ok" for r in dev_rows]
    m = compute_binary_metrics(y_true, y_pred)

    assert m["accuracy"] >= 0.95
    assert m["recall"] >= 0.90

    saved_dm = payload["meta"].get("dev_metrics")
    if saved_dm:
        assert abs(float(saved_dm["accuracy"]) - float(m["accuracy"])) < 1e-6
        assert abs(float(saved_dm["recall"]) - float(m["recall"])) < 1e-6


def test_inference_speed_80_sms_on_cpu() -> None:
    dev_rows = _load_dev_rows()
    texts = [r["text"] for r in dev_rows] * 2  # 80 SMS like hidden jury set
    start = time.perf_counter()
    for t in texts:
        res = check(t)
        assert res["risk"] in {"low", "medium", "high"}
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0


def test_ml_team_artifact_preserved_and_baseline_selected() -> None:
    from src.config import BASE_DIR

    with open(DEFAULT_MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    # Compact generalizable model selected for release (avoids 45k-feature Colab overfitting)
    assert payload["model_type"] == "linear_features_tfidf"
    assert (BASE_DIR / "models" / "sms_fraud_model.pkl").is_file()
    assert (BASE_DIR / "notebooks" / "sms_fraud_model.ipynb").is_file()

"""Main inference entry points check(text), check_detailed(text), and load_model_info()."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.advisor import get_advice
from src.config import FEATURE_NAMES, PROTECTIVE_FEATURES
from src.features import extract_features
from src.model_loader import LoadedSMSModel, load_model
from src.preprocessing import normalize
from src.scoring import logit_to_score_and_risk

_MODEL_CACHE: LoadedSMSModel | None = None
_MODEL_PATH_CACHE: str | None = None


def get_model(model_path: Path | None = None) -> LoadedSMSModel:
    """Return cached LoadedSMSModel (reloads if path changed)."""
    global _MODEL_CACHE, _MODEL_PATH_CACHE
    requested = str(model_path) if model_path is not None else None
    if _MODEL_CACHE is None or (requested is not None and requested != _MODEL_PATH_CACHE):
        _MODEL_CACHE = load_model(model_path)
        _MODEL_PATH_CACHE = requested
    return _MODEL_CACHE


def reload_model(model_path: Path | None = None) -> LoadedSMSModel:
    """Force-reload model.pkl from disk."""
    global _MODEL_CACHE, _MODEL_PATH_CACHE
    _MODEL_CACHE = load_model(model_path)
    _MODEL_PATH_CACHE = str(model_path) if model_path is not None else None
    return _MODEL_CACHE


def load_model_info(model_path: Path | None = None) -> dict[str, Any]:
    """Return model metadata and feature weights for inspection and UI visualization."""
    model = get_model(model_path)
    return {
        "checkpoint_version": model.checkpoint_version,
        "format_kind": model.format_kind,
        "model_type": model.model_type,
        "source_path": model.source_path,
        "feature_names": list(model.feature_names),
        "feature_weights": dict(model.feature_weights),
        "intercept": model.intercept,
        "score_shift": model.score_shift,
        "has_tfidf": bool(model.tfidf),
        "meta": dict(model.meta),
    }


def check_detailed(text: object, model_path: Path | None = None) -> dict[str, Any]:
    """Run full SMS phishing analysis with normalized text, domains, feature breakdown, and advice."""
    model = get_model(model_path)
    norm = normalize(text)
    feature_values, reason_map = extract_features(norm)

    if not str(norm.get("clean_text") or "").strip():
        return {
            "risk": "low",
            "score": 0,
            "reasons": [],
            "advice": get_advice("low", [], []),
            "clean_text": "",
            "translit_text": "",
            "was_obfuscated": False,
            "domains": [],
            "phones": [],
            "triggered_features": [],
            "feature_values": {f: 0.0 for f in FEATURE_NAMES},
            "feature_contributions": {f: 0.0 for f in FEATURE_NAMES},
        }

    z_raw, contributions = model.predict_logit(norm, feature_values)
    score, risk = logit_to_score_and_risk(z_raw, model.score_shift)

    # Sort triggered risk features strictly by positive contribution w*x > 0
    positive_risk_keys = [
        f
        for f in FEATURE_NAMES
        if feature_values[f] > 0.0
        and f not in PROTECTIVE_FEATURES
        and contributions.get(f, 0.0) > 0.0
    ]
    positive_risk_keys.sort(key=lambda f: contributions.get(f, 0.0), reverse=True)
    reasons = [reason_map[k] for k in positive_risk_keys if k in reason_map]

    # Do not clutter safe messages with minor positive noise when risk == "low" and score < 15
    if risk == "low" and score < 15:
        reasons = []

    domains: list[dict[str, object]] = list(norm.get("domains") or [])  # type: ignore[arg-type]
    advice = get_advice(risk, positive_risk_keys, domains)

    return {
        "risk": risk,
        "score": score,
        "reasons": reasons,
        "advice": advice,
        "clean_text": str(norm.get("clean_text") or ""),
        "translit_text": str(norm.get("translit_text") or ""),
        "was_obfuscated": bool(norm.get("was_obfuscated", False)),
        "domains": domains,
        "phones": list(norm.get("phones") or []),  # type: ignore[arg-type]
        "triggered_features": [f for f in FEATURE_NAMES if feature_values[f] > 0.0],
        "feature_values": feature_values,
        "feature_contributions": {
            f: round(float(contributions.get(f, 0.0)), 6) for f in FEATURE_NAMES
        },
    }


def check(text: object, model_path: Path | None = None) -> dict[str, Any]:
    """Strict jury contract entry point returning only {"risk", "score", "reasons"}."""
    detailed = check_detailed(text, model_path=model_path)
    return {
        "risk": detailed["risk"],
        "score": int(detailed["score"]),
        "reasons": list(detailed["reasons"]),
    }

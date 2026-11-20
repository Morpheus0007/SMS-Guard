"""Export helper converting Colab sklearn LogisticRegression (+ TF-IDF) into portable Contract v3.1 model.pkl."""

from __future__ import annotations

import hashlib
import math
import pickle
import platform
from pathlib import Path
from typing import Any

from config import FEATURE_NAMES


def _extract_vectorizer_block(vectorizer: Any, coef_slice: list[float]) -> dict[str, Any]:
    vocab_raw = dict(vectorizer.vocabulary_)
    idf_arr = [float(x) for x in vectorizer.idf_]
    n_terms = len(vocab_raw)
    if len(coef_slice) != n_terms:
        raise ValueError(
            f"TF-IDF coef slice length {len(coef_slice)} != vocab size {n_terms}"
        )

    ordered_terms = sorted(vocab_raw.items(), key=lambda item: int(item[1]))
    vocab: dict[str, int] = {}
    idf: list[float] = []
    weights: list[float] = []
    for new_idx, (term, old_idx_raw) in enumerate(ordered_terms):
        old_idx = int(old_idx_raw)
        vocab[str(term)] = new_idx
        idf.append(float(idf_arr[old_idx]))
        weights.append(float(coef_slice[old_idx]))

    return {
        "ngram_range": [int(vectorizer.ngram_range[0]), int(vectorizer.ngram_range[1])],
        "vocab": vocab,
        "idf": idf,
        "weights": weights,
    }


def export_sklearn_to_model_pkl(
    clf: Any,
    out_path: str | Path = "model.pkl",
    word_vectorizer: Any = None,
    char_vectorizer: Any = None,
    score_shift: float = 0.0,
    scaler: Any = None,
    input_field: str = "clean_and_translit",
    strip_urls: bool = True,
    unify_kyrgyz_flag: bool = True,
    cv_metrics: dict[str, Any] | None = None,
    dev_metrics: dict[str, Any] | None = None,
    train_csv_path: str | Path | None = None,
    n_train: int = 200,
    used_dev_for_training: bool = False,
    seed: int = 42,
) -> Path:
    """Export fitted sklearn LogisticRegression (+ optional TF-IDF) to pure-Python dict model.pkl."""
    coef = [float(x) for x in clf.coef_[0]]
    intercept = float(clf.intercept_[0])

    n_feat = len(FEATURE_NAMES)
    n_word = len(word_vectorizer.vocabulary_) if word_vectorizer is not None else 0
    n_char = len(char_vectorizer.vocabulary_) if char_vectorizer is not None else 0
    expected_len = n_feat + n_word + n_char
    if len(coef) != expected_len:
        raise ValueError(
            f"Expected {expected_len} coefficients (12 + {n_word} word + {n_char} char), got {len(coef)}"
        )

    feature_weights = {fname: float(coef[i]) for i, fname in enumerate(FEATURE_NAMES)}

    feature_scaler: dict[str, list[float]] | None = None
    if scaler is not None:
        means = [float(x) for x in getattr(scaler, "mean_")]
        stds = [float(x) for x in getattr(scaler, "scale_")]
        if len(means) != n_feat or len(stds) != n_feat:
            raise ValueError("Scaler mean_/scale_ must have length 12")
        feature_scaler = {"mean": means, "std": stds}

    tfidf_block: dict[str, Any] | None = None
    if word_vectorizer is not None or char_vectorizer is not None:
        sublinear = bool(
            getattr(word_vectorizer or char_vectorizer, "sublinear_tf", True)
        )
        word_block = (
            _extract_vectorizer_block(word_vectorizer, coef[n_feat : n_feat + n_word])
            if word_vectorizer is not None
            else None
        )
        char_block = (
            _extract_vectorizer_block(char_vectorizer, coef[n_feat + n_word :])
            if char_vectorizer is not None
            else None
        )
        tfidf_block = {
            "input_field": input_field,
            "strip_urls": bool(strip_urls),
            "unify_kyrgyz": bool(unify_kyrgyz_flag),
            "sublinear_tf": sublinear,
            "l2_norm": True,
            "word": word_block,
            "char_wb": char_block,
        }

    train_sha = ""
    if train_csv_path is not None and Path(train_csv_path).is_file():
        train_sha = hashlib.sha256(Path(train_csv_path).read_bytes()).hexdigest()

    sklearn_ver = "unknown"
    try:
        import sklearn  # type: ignore

        sklearn_ver = str(sklearn.__version__)
    except Exception:
        pass

    payload = {
        "checkpoint_version": "3.1",
        "model_type": "linear_features_tfidf" if tfidf_block is not None else "linear_features",
        "feature_names": list(FEATURE_NAMES),
        "feature_weights": feature_weights,
        "intercept": intercept,
        "score_shift": float(score_shift),
        "feature_scaler": feature_scaler,
        "tfidf": tfidf_block,
        "meta": {
            "trained_on_sha256": train_sha,
            "n_train": int(n_train),
            "used_dev_for_training": bool(used_dev_for_training),
            "seed": int(seed),
            "sklearn_version": sklearn_ver,
            "python_version": platform.python_version(),
            "cv_metrics": dict(cv_metrics or {}),
            "dev_metrics": dict(dev_metrics) if dev_metrics is not None else None,
        },
    }

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    return target


def verify_model_pkl(path: str | Path = "model.pkl") -> dict[str, Any]:
    """Validate Contract v3.1 invariants on a saved model.pkl file."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Missing model file: {target}")
    size_bytes = target.stat().st_size
    if size_bytes > 95 * 1024 * 1024:
        raise ValueError(f"Model file exceeds 95 MB limit: {size_bytes} bytes")

    with open(target, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict pickle for Contract v3.1 Format A, got {type(payload)}")

    required = {
        "checkpoint_version",
        "model_type",
        "feature_names",
        "feature_weights",
        "intercept",
        "score_shift",
        "meta",
    }
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"Missing required keys in model.pkl: {sorted(missing)}")

    if tuple(payload["feature_names"]) != tuple(FEATURE_NAMES):
        raise ValueError("feature_names does not match canonical 12 FEATURE_NAMES")

    fw = payload["feature_weights"]
    for name in FEATURE_NAMES:
        val = float(fw[name])
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Invalid NaN/Inf weight for {name}")

    if float(fw["has_official_domain"]) > 0.0:
        print("WARNING: feature_weights['has_official_domain'] > 0 (expected <= 0)")
    if float(fw["has_safe_warning"]) > 0.0:
        print("WARNING: feature_weights['has_safe_warning'] > 0 (expected <= 0)")

    print(f"OK: {target} ({size_bytes} bytes), type={payload['model_type']}, shift={payload['score_shift']}")
    return payload

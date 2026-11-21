"""Universal Pickle loader and pure-Python/sklearn inference for SMS Guard."""

from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Any

from src.config import DEFAULT_MODEL_PATH, FEATURE_NAMES
from src.preprocessing import strip_urls_for_text_features, unify_kyrgyz
from src.scoring import prob_to_logit

_WORD_TOKEN_RE = re.compile(r"\b\w\w+\b")
_MULTI_SPACE_RE = re.compile(r"\s\s+")


def _extract_word_ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    tokens = _WORD_TOKEN_RE.findall(text.lower())
    min_n, max_n = int(ngram_range[0]), int(ngram_range[1])
    if max_n == 1:
        return [t for t in tokens] if min_n <= 1 else []
    ngrams: list[str] = []
    n_tok = len(tokens)
    for k in range(min_n, max_n + 1):
        if k <= 0 or k > n_tok:
            continue
        if k == 1:
            ngrams.extend(tokens)
        else:
            for i in range(n_tok - k + 1):
                ngrams.append(" ".join(tokens[i : i + k]))
    return ngrams


def _extract_char_wb_ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    cleaned = _MULTI_SPACE_RE.sub(" ", text.lower()).strip()
    min_n, max_n = int(ngram_range[0]), int(ngram_range[1])
    ngrams: list[str] = []
    for word in cleaned.split():
        padded = f" {word} "
        w_len = len(padded)
        for k in range(min_n, max_n + 1):
            if k <= 0 or k > w_len:
                continue
            for i in range(w_len - k + 1):
                ngrams.append(padded[i : i + k])
    return ngrams


def _extract_char_ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    """Match sklearn TfidfVectorizer(analyzer='char') sliding windows."""
    cleaned = _MULTI_SPACE_RE.sub(" ", text.lower())
    min_n, max_n = int(ngram_range[0]), int(ngram_range[1])
    n_len = len(cleaned)
    ngrams: list[str] = []
    for k in range(min_n, max_n + 1):
        if k <= 0 or k > n_len:
            continue
        for i in range(n_len - k + 1):
            ngrams.append(cleaned[i : i + k])
    return ngrams


def compute_tfidf_dot(
    text: str,
    spec: dict[str, Any],
    analyzer: str,
    sublinear_tf: bool = True,
    l2_norm: bool = True,
) -> float:
    """Compute TF-IDF weighted dot product matching sklearn TfidfVectorizer."""
    vocab: dict[str, int] = spec.get("vocab") or {}
    idf: list[float] = spec.get("idf") or []
    weights: list[float] = spec.get("weights") or []
    if not vocab or not idf or not weights:
        return 0.0

    ngram_range = tuple(spec.get("ngram_range") or ((1, 2) if analyzer == "word" else (3, 5)))
    if analyzer == "word":
        grams = _extract_word_ngrams(text, ngram_range)  # type: ignore[arg-type]
    elif analyzer == "char":
        grams = _extract_char_ngrams(text, ngram_range)  # type: ignore[arg-type]
    else:
        grams = _extract_char_wb_ngrams(text, ngram_range)  # type: ignore[arg-type]

    counts: dict[int, int] = {}
    for gram in grams:
        idx = vocab.get(gram)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1

    if not counts:
        return 0.0

    vals: dict[int, float] = {}
    norm_sq = 0.0
    for idx, cnt in counts.items():
        tf = (1.0 + math.log(cnt)) if sublinear_tf else float(cnt)
        val = tf * float(idf[idx])
        vals[idx] = val
        norm_sq += val * val

    inv_norm = (1.0 / math.sqrt(norm_sq)) if (l2_norm and norm_sq > 0.0) else 1.0
    dot = 0.0
    for idx, val in vals.items():
        dot += (val * inv_norm) * float(weights[idx])
    return dot


class LoadedSMSModel:
    """Unified wrapper around Format A (dict_pickle_v3 / ml_team_hybrid) and Format B (sklearn/joblib)."""

    def __init__(self, payload: Any, source_path: str = "<memory>") -> None:
        self.source_path = source_path
        self.raw_payload = payload
        self.ml_team_model: dict[str, Any] | None = None

        if isinstance(payload, dict) and "feature_names" in payload and "feature_weights" in payload and "model" not in payload:
            self.format_kind = "dict_pickle_v3"
            self.checkpoint_version = str(payload.get("checkpoint_version", "3.1"))
            self.model_type = str(payload.get("model_type", "linear_features"))
            self.feature_names: tuple[str, ...] = tuple(payload["feature_names"])
            self.feature_weights: dict[str, float] = {
                str(k): float(v) for k, v in payload["feature_weights"].items()
            }
            self.intercept: float = float(payload.get("intercept", 0.0))
            self.score_shift: float = float(payload.get("score_shift", 0.0))
            self.feature_scaler: dict[str, list[float]] | None = payload.get("feature_scaler")
            self.tfidf: dict[str, Any] | None = payload.get("tfidf")
            self.ml_team_model = payload.get("ml_team_model")
            self.meta: dict[str, Any] = dict(payload.get("meta") or {})
            self.estimator = None
            self.input_mode = "dict_v3"
            self._validate_dict_v3()
        elif isinstance(payload, dict) and "model" in payload:
            self.format_kind = "sklearn_wrapped"
            self.checkpoint_version = str(payload.get("checkpoint_version", "3.1-sklearn"))
            self.estimator = payload["model"]
            self.input_mode = str(payload.get("input_mode", "features_12"))
            self.model_type = f"sklearn_{self.input_mode}"
            self.feature_names = tuple(payload.get("feature_names") or FEATURE_NAMES)
            self.score_shift = float(payload.get("score_shift", 0.0))
            self.feature_scaler = None
            self.tfidf = None
            self.meta = dict(payload.get("meta") or {})
            self.feature_weights = self._resolve_sklearn_weights(
                payload.get("feature_weights"), self.estimator
            )
            self.intercept = float(payload.get("intercept", self._resolve_sklearn_intercept(self.estimator)))
        else:
            self.format_kind = "sklearn_raw"
            self.checkpoint_version = "raw-sklearn"
            self.estimator = payload
            self.input_mode = self._detect_raw_input_mode(payload)
            self.model_type = f"sklearn_raw_{self.input_mode}"
            self.feature_names = FEATURE_NAMES
            self.score_shift = 0.60 if self.input_mode == "clean_text" else 0.0
            self.feature_scaler = None
            self.tfidf = None
            self.meta = {}
            self.feature_weights = self._resolve_sklearn_weights(None, self.estimator)
            self.intercept = self._resolve_sklearn_intercept(self.estimator)

    def _validate_dict_v3(self) -> None:
        if tuple(self.feature_names) != tuple(FEATURE_NAMES):
            raise ValueError(
                f"Model feature_names mismatch: expected {FEATURE_NAMES}, got {self.feature_names}"
            )
        for fname in FEATURE_NAMES:
            if fname not in self.feature_weights:
                raise ValueError(f"Missing weight for feature '{fname}' in model.pkl")
            val = self.feature_weights[fname]
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Invalid NaN/Inf weight for feature '{fname}'")

    @staticmethod
    def _resolve_sklearn_intercept(estimator: Any) -> float:
        if hasattr(estimator, "intercept_"):
            try:
                return float(estimator.intercept_[0])
            except Exception:
                return 0.0
        return 0.0

    @staticmethod
    def _resolve_sklearn_weights(
        explicit: dict[str, Any] | None, estimator: Any
    ) -> dict[str, float]:
        if isinstance(explicit, dict) and all(f in explicit for f in FEATURE_NAMES):
            return {f: float(explicit[f]) for f in FEATURE_NAMES}
        if hasattr(estimator, "coef_"):
            try:
                coef = estimator.coef_[0]
                if len(coef) == len(FEATURE_NAMES):
                    return {f: float(coef[i]) for i, f in enumerate(FEATURE_NAMES)}
            except Exception:
                pass
        return {
            f: (-1.0 if f in ("has_official_domain", "has_safe_warning") else 1.0)
            for f in FEATURE_NAMES
        }

    @staticmethod
    def _detect_raw_input_mode(estimator: Any) -> str:
        if hasattr(estimator, "n_features_in_") and getattr(estimator, "n_features_in_", None) == len(
            FEATURE_NAMES
        ):
            return "features_12"
        if hasattr(estimator, "named_steps"):
            return "clean_text"
        return "features_12"

    def _prepare_tfidf_text(self, norm: dict[str, Any]) -> str:
        tfidf_cfg = self.tfidf or {}
        input_field = str(tfidf_cfg.get("input_field", "clean_and_translit"))
        do_strip_urls = bool(tfidf_cfg.get("strip_urls", True))
        do_unify = bool(tfidf_cfg.get("unify_kyrgyz", True))
        domains = list(norm.get("domains") or [])

        clean = str(norm.get("clean_text") or "")
        translit = str(norm.get("translit_text") or "")
        if do_strip_urls:
            clean = strip_urls_for_text_features(clean, domains)
            translit = strip_urls_for_text_features(translit, domains)

        if input_field == "clean_text":
            text = clean
        else:
            text = f"{clean} {translit}".strip()

        return unify_kyrgyz(text) if do_unify else text

    def _predict_ml_team_logit(self, clean_text: str) -> float:
        """Exact pure-Python inference of ML team's Colab Pipeline (char + word TF-IDF + 11 numeric)."""
        from src.ml_team_adapter import extract_ml_team_features

        cfg = self.ml_team_model or {}
        sublinear = bool(cfg.get("sublinear_tf", True))
        l2_norm = bool(cfg.get("l2_norm", True))

        z_ml = float(cfg.get("intercept", 0.0))
        if cfg.get("char"):
            z_ml += compute_tfidf_dot(clean_text, cfg["char"], "char", sublinear, l2_norm)
        if cfg.get("word"):
            z_ml += compute_tfidf_dot(clean_text, cfg["word"], "word", sublinear, l2_norm)

        num_weights: dict[str, float] = cfg.get("numeric_weights") or {}
        if num_weights:
            ml_feats = extract_ml_team_features(clean_text)
            for fname, weight in num_weights.items():
                val = float(ml_feats.get(fname, 0))
                if val != 0.0:
                    z_ml += val * float(weight)
        return z_ml

    def _resolve_fraud_class_sign_and_index(self) -> tuple[float, int]:
        classes = getattr(self.estimator, "classes_", None)
        if classes is not None:
            cls_list = [str(c).lower() for c in classes]
            if "fraud" in cls_list:
                idx = cls_list.index("fraud")
                # In binary sklearn, decision_function > 0 corresponds to classes_[1]
                return (1.0 if idx == 1 else -1.0), idx
        return 1.0, 1

    def predict_logit(
        self, norm: dict[str, Any], feature_values: dict[str, float]
    ) -> tuple[float, dict[str, float]]:
        """Return (raw_logit_before_shift, per_feature_contributions_dict)."""
        if self.format_kind == "dict_pickle_v3":
            means = (self.feature_scaler or {}).get("mean")
            stds = (self.feature_scaler or {}).get("std")
            z = self.intercept
            contributions: dict[str, float] = {}

            for idx, fname in enumerate(self.feature_names):
                raw_x = float(feature_values.get(fname, 0.0))
                w = self.feature_weights[fname]
                if means is not None and stds is not None:
                    std = float(stds[idx]) if float(stds[idx]) > 1e-12 else 1.0
                    mean = float(means[idx])
                    scaled_x = (raw_x - mean) / std
                    z += w * scaled_x
                    eff_w = w / std
                    contributions[fname] = eff_w * raw_x if raw_x > 0.0 else 0.0
                else:
                    contrib = w * raw_x
                    z += contrib
                    contributions[fname] = contrib

            if self.tfidf:
                tfidf_text = self._prepare_tfidf_text(norm)
                sublinear = bool(self.tfidf.get("sublinear_tf", True))
                l2_norm = bool(self.tfidf.get("l2_norm", True))
                if self.tfidf.get("word"):
                    z += compute_tfidf_dot(
                        tfidf_text, self.tfidf["word"], "word", sublinear, l2_norm
                    )
                if self.tfidf.get("char"):
                    z += compute_tfidf_dot(
                        tfidf_text, self.tfidf["char"], "char", sublinear, l2_norm
                    )
                if self.tfidf.get("char_wb"):
                    z += compute_tfidf_dot(
                        tfidf_text, self.tfidf["char_wb"], "char_wb", sublinear, l2_norm
                    )

            if self.ml_team_model:
                alpha = float(self.ml_team_model.get("alpha", 0.5))
                z_ml = self._predict_ml_team_logit(str(norm.get("clean_text") or ""))
                z += alpha * z_ml

            return z, contributions

        # Format B: sklearn / joblib estimator or pipeline
        if self.input_mode == "clean_text":
            X = [str(norm.get("clean_text") or "")]
        else:
            X = [[float(feature_values.get(f, 0.0)) for f in self.feature_names]]

        sign, fraud_idx = self._resolve_fraud_class_sign_and_index()
        if hasattr(self.estimator, "decision_function"):
            z_raw = sign * float(self.estimator.decision_function(X)[0])
        elif hasattr(self.estimator, "predict_proba"):
            prob = float(self.estimator.predict_proba(X)[0][fraud_idx])
            z_raw = prob_to_logit(prob)
        else:
            pred = str(self.estimator.predict(X)[0]).lower()
            z_raw = 3.0 if pred in ("1", "fraud", "true") else -3.0

        contributions = {
            f: self.feature_weights.get(f, 0.0) * float(feature_values.get(f, 0.0))
            for f in self.feature_names
        }
        return z_raw, contributions


def _read_pickle_any(path: Path) -> Any:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as pickle_err:
        try:
            from src.ml_team_adapter import load_raw_ml_team_joblib

            return load_raw_ml_team_joblib(path)
        except Exception:
            raise pickle_err


def load_model(model_path: Path | None = None) -> LoadedSMSModel:
    """Load model.pkl from disk, falling back to in-memory training if missing."""
    target = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if target.is_file():
        payload = _read_pickle_any(target)
        return LoadedSMSModel(payload, source_path=str(target))

    from src.train import train_model_payload

    payload = train_model_payload()
    return LoadedSMSModel(payload, source_path="<in-memory-fallback>")

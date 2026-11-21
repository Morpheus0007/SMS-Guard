"""Adapter and portable converter for the ML team's Colab model (sms_fraud_model.pkl)."""

from __future__ import annotations

import csv
import hashlib
import pickle
import platform
import re
import sys
import unicodedata
import warnings
from pathlib import Path
from typing import Any

from src.config import (
    BASE_DIR,
    DEFAULT_DEV_CSV,
    DEFAULT_MODEL_PATH,
    DEFAULT_TRAIN_CSV,
    FEATURE_NAMES,
)
from src.features import extract_features as extract_canonical_features
from src.preprocessing import normalize as canonical_normalize
from src.scoring import compute_binary_metrics, logit_to_score_and_risk

DEFAULT_RAW_ML_PKL = BASE_DIR / "models" / "sms_fraud_model.pkl"

# --- Exact feature extraction from ML team's sms_fraud_model.ipynb (Cells 3, 5, 9) ---
_ML_PROTECTED_WORDS = frozenset({"sim", "sms", "url", "pin", "megacom", "beeline", "mbank", "kicb"})
_ML_CYR_TO_LAT = str.maketrans(
    {"а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y", "к": "k", "м": "m", "т": "t", "в": "b", "н": "h"}
)
_ML_CYR_OBF = str.maketrans({"0": "о", "3": "з", "4": "ч", "6": "б", "@": "а", "$": "с"})


def _ml_normalize_word(word: str) -> str:
    if word in _ML_PROTECTED_WORDS:
        return word
    has_latin = bool(re.search(r"[a-z]", word))
    has_cyrillic = bool(re.search(r"[а-яёөүң]", word))
    if has_latin:
        word = word.translate(_ML_CYR_TO_LAT).replace("@", "a").replace("3", "e").replace("1", "i")
    elif has_cyrillic:
        word = word.translate(_ML_CYR_OBF)
    return word


def ml_normalize_text(text: str) -> str:
    """Normalization logic from ML team's sms_fraud_model.ipynb."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    urls: list[str] = []
    domains: list[str] = []
    phones: list[str] = []

    def _save_url(m: re.Match[str]) -> str:
        urls.append(m.group(0))
        return f" URLTOKEN{len(urls) - 1}END "

    def _save_domain(m: re.Match[str]) -> str:
        domains.append(m.group(0))
        return f" DOMAINTOKEN{len(domains) - 1}END "

    def _save_phone(m: re.Match[str]) -> str:
        phones.append(m.group(0))
        return f" PHONETOKEN{len(phones) - 1}END "

    text = re.sub(r"(?:https?://|www\.)[^\s]+", _save_url, text)
    text = re.sub(r"(?<![\w@])(?:[a-z0-9-]+\.)+[a-z]{2,}(?![\w-])", _save_domain, text)
    text = re.sub(r"(?<!\w)(?:\+?\d[\d\s()\-]{6,}\d)(?!\w)", _save_phone, text)
    text = re.sub(r"(?<=[а-яёөүңa-z])[._*]+(?=[а-яёөүңa-z])", "", text)
    text = re.sub(r"\b(sim|sms|url|pin)-(?=[а-яёөүң])", r"\1 ", text)
    text = re.sub(r"(?<=[а-яёөүңa-z])-(?=[а-яёөүңa-z])", "", text)
    text = re.sub(
        r"(?<!\S)(?:[а-яёөүңa-z]\s+){2,}[а-яёөүңa-z](?=\s+[а-яёөүңa-z]{2,}|[^\w\s]|$)",
        lambda m: re.sub(r"\s+", "", m.group(0)),
        text,
    )
    text = re.sub(
        r"[а-яёөүңa-z0-9@$]+",
        lambda m: m.group(0) if m.group(0) in _ML_PROTECTED_WORDS else _ml_normalize_word(m.group(0)),
        text,
    )
    for i, u in enumerate(urls):
        text = text.replace(f"URLTOKEN{i}END", u)
    for i, d in enumerate(domains):
        text = text.replace(f"DOMAINTOKEN{i}END", d)
    for i, p in enumerate(phones):
        text = text.replace(f"PHONETOKEN{i}END", p)
    return re.sub(r"\s+", " ", text).strip()


def extract_ml_team_features(text: str) -> dict[str, int]:
    """Exact 11 numeric features from ML team's sms_fraud_model.ipynb."""
    t = ml_normalize_text(text)
    return {
        "has_url": int(bool(re.search(r"(https?://|www\.|[a-z0-9-]+\.[a-z]{2,})", t))),
        "asks_code_password": int(
            bool(re.search(r"(код|кодду|кодун|парол[ья]|сырсөз|сырсөзүңүз|pin|sms.?код)", t))
        ),
        "urgency": int(bool(re.search(r"(срочно|немедленно|быстро|сейчас|тез|дароо|азыр)", t))),
        "blocking_threat": int(
            bool(re.search(r"(заблокирован|блокировка|блоктолот|блоктолду|бөгөттөлдү|бөгөттөлөт|жабылат|жабылды)", t))
        ),
        "prize": int(
            bool(
                re.search(
                    r"(bonus|бонус|выиграл|выиграли|победител\w*|приз|подарок|байге|уттуңуз|утту|жеңүүчү|утуп алдыңыз)",
                    t,
                )
            )
        ),
        "bank_operator": int(
            bool(re.search(r"(банк|bank|mbank|kicb|рск|оптима|megacom|beeline|элкарт)", t))
        ),
        "money_transfer": int(
            bool(re.search(r"(переведите|перевод|отправьте деньги|пополните счет|акча котор|акча жөнөт|акчаңызды)", t))
        ),
        "callback_request": int(
            bool(re.search(r"(перезвоните|позвоните|обратитесь по номеру|чалгыла|кайра чалыңыз)", t))
        ),
        "send_request": int(
            bool(
                re.search(
                    r"(отправьте|сообщите|введите|подтвердите|подтвердить|активируйте|активировать|активдештир|aktivdeshtir|aktivirovat|жөнөтүңүз|жибериңиз|жазыңыз|бериңиз|киргизиңиз)",
                    t,
                )
            )
        ),
        "account_card": int(
            bool(re.search(r"(аккаунт|сч[её]т|карта|card|эсебиңиз|эсеп|картаңыз|пароль|парол[ья]|сырсөз|кабинет)", t))
        ),
        "link_action": int(
            bool(re.search(r"(перейдите по ссылке|перейти по ссылке|шилтемеге өт|шилтемени бас|ссылка боюнча өт)", t))
        ),
    }


def extract_features_for_model(texts: Any) -> list[dict[str, int]]:
    """FunctionTransformer callback expected when unpickling raw sms_fraud_model.pkl."""
    return [extract_ml_team_features(t) for t in texts]


def load_raw_ml_team_joblib(pkl_path: Path = DEFAULT_RAW_ML_PKL) -> Any:
    """Safely unpickle ML team's raw Colab joblib Pipeline without __main__ AttributeError."""
    import joblib  # type: ignore

    main_mod = sys.modules.get("__main__")
    had_attr = hasattr(main_mod, "extract_features_for_model") if main_mod is not None else True
    if main_mod is not None and not had_attr:
        setattr(main_mod, "extract_features_for_model", extract_features_for_model)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(pkl_path)
    finally:
        if main_mod is not None and not had_attr:
            try:
                delattr(main_mod, "extract_features_for_model")
            except Exception:
                pass


def _extract_vectorizer_pure(vec: Any, coef_slice: list[float]) -> dict[str, Any]:
    vocab_raw = dict(vec.vocabulary_)
    idf_arr = [float(x) for x in vec.idf_]
    ordered = sorted(vocab_raw.items(), key=lambda item: int(item[1]))
    vocab: dict[str, int] = {}
    idf: list[float] = []
    weights: list[float] = []
    for new_idx, (term, old_idx_raw) in enumerate(ordered):
        old_idx = int(old_idx_raw)
        vocab[str(term)] = new_idx
        idf.append(round(float(idf_arr[old_idx]), 8))
        weights.append(round(float(coef_slice[old_idx]), 8))
    return {
        "ngram_range": [int(vec.ngram_range[0]), int(vec.ngram_range[1])],
        "vocab": vocab,
        "idf": idf,
        "weights": weights,
    }


def _load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [
            {
                "id": str(r.get("id") or "").strip(),
                "text": str(r.get("text") or ""),
                "label": str(r.get("label") or "").strip().lower(),
            }
            for r in csv.DictReader(f)
            if r and (r.get("text") or "").strip()
        ]


def build_integrated_model_pkl(
    raw_ml_pkl: Path = DEFAULT_RAW_ML_PKL,
    out_path: Path = DEFAULT_MODEL_PATH,
    train_csv: Path = DEFAULT_TRAIN_CSV,
    dev_csv: Path = DEFAULT_DEV_CSV,
    alpha_ml: float = 0.50,
    seed: int = 42,
) -> dict[str, Any]:
    """Extract ML team's sms_fraud_model.pkl weights + calibrate with 12-feature LR -> portable model.pkl."""
    from src.model_loader import LoadedSMSModel
    from src.train import (
        _calibrate_shift_on_logits,
        _fit_l2_logistic_regression,
        _stratified_kfold_indices,
    )

    pipeline = load_raw_ml_team_joblib(raw_ml_pkl)
    union = pipeline.named_steps["features"]
    clf = pipeline.named_steps["classifier"]

    classes = [str(c).lower() for c in clf.classes_]
    # In sklearn binary LogisticRegression, decision_function > 0 predicts classes_[1].
    sign = 1.0 if classes[1] == "fraud" else -1.0
    full_coef = [sign * float(x) for x in clf.coef_[0]]
    ml_intercept = sign * float(clf.intercept_[0])

    transformers = dict(union.transformer_list)
    char_vec = transformers["char"]
    word_vec = transformers["word"]
    dv = transformers["numeric"].named_steps["vectorize"]

    n_char = len(char_vec.vocabulary_)
    n_word = len(word_vec.vocabulary_)
    ml_char_block = _extract_vectorizer_pure(char_vec, full_coef[:n_char])
    ml_word_block = _extract_vectorizer_pure(word_vec, full_coef[n_char : n_char + n_word])
    ml_numeric_weights = {
        str(name): round(float(w), 8)
        for name, w in zip(dv.feature_names_, full_coef[n_char + n_word :])
    }

    # Fit companion 12-feature L2 Logistic Regression strictly on sms_train.csv
    train_rows = _load_csv(train_csv)
    y_train = [1 if r["label"] == "fraud" else 0 for r in train_rows]
    train_norms = [canonical_normalize(r["text"]) for r in train_rows]
    train_fdicts = [extract_canonical_features(nm)[0] for nm in train_norms]
    X12_train = [
        [(i, float(fd[f])) for i, f in enumerate(FEATURE_NAMES) if fd[f] != 0.0]
        for fd in train_fdicts
    ]

    w12, b12 = _fit_l2_logistic_regression(X12_train, y_train, len(FEATURE_NAMES), c_reg=2.0)

    # Calibrate score_shift strictly via 5-fold CV on sms_train.csv for the 12-feature branch
    splits = _stratified_kfold_indices(y_train, n_splits=5, seed=seed)
    oof_12 = [0.0] * len(y_train)
    for tr_idx, va_idx in splits:
        wf, bf = _fit_l2_logistic_regression(
            [X12_train[i] for i in tr_idx], [y_train[i] for i in tr_idx], len(FEATURE_NAMES), c_reg=2.0
        )
        for idx in va_idx:
            oof_12[idx] = bf + sum(wf[j] * v for j, v in X12_train[idx])

    # Compute ML team logits on train clean_text to calibrate blended shift on train
    clean_train = [str(nm["clean_text"]) for nm in train_norms]
    z_ml_train = (sign * pipeline.decision_function(clean_train)).tolist()
    blended_cv_logits = [
        alpha_ml * float(zm) + (1.0 - alpha_ml) * float(z12)
        for zm, z12 in zip(z_ml_train, oof_12)
    ]
    best_shift, cv_m = _calibrate_shift_on_logits(blended_cv_logits, y_train)
    # Keep shift moderate (avoid pushing everyday ok messages over 65)
    best_shift = round(min(max(best_shift, 0.35), 0.50), 4)

    # Effective 12-feature weights (scaled by 1 - alpha_ml for transparent contribution display)
    domain_weight = 1.0 - alpha_ml
    feature_weights = {
        fname: round(domain_weight * float(w12[i]), 8)
        for i, fname in enumerate(FEATURE_NAMES)
    }

    ml_team_sha = hashlib.sha256(raw_ml_pkl.read_bytes()).hexdigest()
    train_sha = hashlib.sha256(train_csv.read_bytes()).hexdigest()

    payload: dict[str, Any] = {
        "checkpoint_version": "3.1",
        "model_type": "ml_team_hybrid",
        "feature_names": list(FEATURE_NAMES),
        "feature_weights": feature_weights,
        "intercept": round(domain_weight * float(b12), 8),
        "score_shift": float(best_shift),
        "feature_scaler": None,
        "tfidf": None,
        "ml_team_model": {
            "source_file": raw_ml_pkl.name,
            "source_sha256": ml_team_sha,
            "alpha": float(alpha_ml),
            "intercept": round(float(ml_intercept), 8),
            "sublinear_tf": True,
            "l2_norm": True,
            "char": ml_char_block,
            "word": ml_word_block,
            "numeric_weights": ml_numeric_weights,
        },
        "meta": {
            "trained_on_sha256": train_sha,
            "n_train": len(train_rows),
            "used_dev_for_training": False,
            "seed": seed,
            "trainer": "ml_team_colab_sms_fraud_model_pkl + domain_12_lr",
            "python_version": platform.python_version(),
            "cv_metrics": {
                "accuracy": round(float(cv_m["accuracy"]), 6),
                "precision": round(float(cv_m["precision"]), 6),
                "recall": round(float(cv_m["recall"]), 6),
                "f1": round(float(cv_m["f1"]), 6),
            },
            "dev_metrics": None,
        },
    }

    # Evaluate independently on sms_dev.csv using exact LoadedSMSModel path
    if dev_csv.is_file():
        loaded = LoadedSMSModel(payload, source_path="<integrated-eval>")
        dev_rows = _load_csv(dev_csv)
        y_true: list[str] = []
        y_pred: list[str] = []
        for row in dev_rows:
            nm = canonical_normalize(row["text"])
            fv, _ = extract_canonical_features(nm)
            z_raw, _ = loaded.predict_logit(nm, fv)
            _, rk = logit_to_score_and_risk(z_raw, loaded.score_shift)
            y_true.append(row["label"])
            y_pred.append("fraud" if rk == "high" else "ok")
        dm = compute_binary_metrics(y_true, y_pred)
        payload["meta"]["dev_metrics"] = {
            "accuracy": round(float(dm["accuracy"]), 6),
            "precision": round(float(dm["precision"]), 6),
            "recall": round(float(dm["recall"]), 6),
            "f1": round(float(dm["f1"]), 6),
            "tp": int(dm["tp"]),
            "fp": int(dm["fp"]),
            "fn": int(dm["fn"]),
            "tn": int(dm["tn"]),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    return payload


def main() -> None:
    payload = build_integrated_model_pkl()
    size = DEFAULT_MODEL_PATH.stat().st_size
    ml_meta = payload["ml_team_model"]
    print(f"Integrated ML team model -> {DEFAULT_MODEL_PATH} ({size} bytes)")
    print(
        f"Source: {ml_meta['source_file']} (alpha={ml_meta['alpha']}, "
        f"char_vocab={len(ml_meta['char']['vocab'])}, word_vocab={len(ml_meta['word']['vocab'])}, "
        f"ml_numeric={len(ml_meta['numeric_weights'])})"
    )
    print(f"score_shift={payload['score_shift']}, domain_intercept={payload['intercept']}")
    print("CV metrics:", payload["meta"]["cv_metrics"])
    print("Dev metrics:", payload["meta"]["dev_metrics"])


if __name__ == "__main__":
    main()

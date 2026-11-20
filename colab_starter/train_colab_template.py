"""Ready-to-run Google Colab training template for SMS Guard (Contract v3.1 Pickle)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from config import FEATURE_NAMES
from export_pkl import export_sklearn_to_model_pkl, verify_model_pkl
from features import extract_features
from preprocessing import normalize, strip_urls_for_text_features, unify_kyrgyz
from scoring import compute_binary_metrics, logit_to_score_and_risk, prob_to_logit


def prepare_dataset(texts: list[str]) -> tuple[csr_matrix, list[str], list[dict[str, float]]]:
    feat_rows: list[list[float]] = []
    feat_dicts: list[dict[str, float]] = []
    tfidf_texts: list[str] = []

    for text in texts:
        norm = normalize(text)
        vals, _ = extract_features(norm)
        feat_dicts.append(vals)
        feat_rows.append([vals[f] for f in FEATURE_NAMES])
        clean = strip_urls_for_text_features(str(norm["clean_text"]), list(norm["domains"]))
        tr = strip_urls_for_text_features(str(norm["translit_text"]), list(norm["domains"]))
        tfidf_texts.append(unify_kyrgyz(f"{clean} {tr}".strip()))

    return csr_matrix(feat_rows), tfidf_texts, feat_dicts


def calibrate_shift(oof_logits: list[float], y_bin: np.ndarray) -> tuple[float, dict]:
    y_true = ["fraud" if int(yi) == 1 else "ok" for yi in y_bin]
    best_shift = 0.0
    best_key = (-1.0, -1.0, -1.0, -10.0)
    best_metrics: dict = {}

    for step in range(-10, 41):
        shift = round(step * 0.05, 4)
        y_pred = [
            "fraud" if logit_to_score_and_risk(z, shift)[1] == "high" else "ok"
            for z in oof_logits
        ]
        m = compute_binary_metrics(y_true, y_pred)
        key = (
            1.0 if float(m["recall"]) >= 0.92 else 0.0,
            float(m["accuracy"]),
            float(m["f1"]),
            -abs(shift - 0.6),
        )
        if key > best_key:
            best_key = key
            best_shift = shift
            best_metrics = m
    return best_shift, best_metrics


def run_colab_pipeline(
    train_csv: str = "sms_train.csv",
    dev_csv: str = "sms_dev.csv",
    out_pkl: str = "model.pkl",
    seed: int = 42,
) -> None:
    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv) if Path(dev_csv).is_file() else None

    y_train = (train_df["label"].str.strip().str.lower() == "fraud").astype(int).to_numpy()
    X12_tr, texts_tr, feat_dicts_tr = prepare_dataset(train_df["text"].tolist())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = [
        ("linear_features_tfidf", 2.0, None),
        ("linear_features_tfidf", 4.0, None),
        ("linear_features_tfidf", 2.0, {0: 1.0, 1: 1.2}),
        ("linear_features", 2.0, None),
        ("linear_features", 5.0, None),
    ]

    best_score = (-1.0, -1.0, -1.0)
    best_cfg = candidates[0]
    best_shift = 0.6
    best_cv: dict = {}

    for mtype, c_val, cw in candidates:
        use_tfidf = mtype == "linear_features_tfidf"
        oof_logits = [0.0] * len(y_train)
        for tr_idx, va_idx in skf.split(X12_tr, y_train):
            if use_tfidf:
                w_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=250, sublinear_tf=True)
                c_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=350, sublinear_tf=True)
                X_tr_fold = hstack([
                    X12_tr[tr_idx],
                    w_vec.fit_transform([texts_tr[i] for i in tr_idx]),
                    c_vec.fit_transform([texts_tr[i] for i in tr_idx]),
                ])
                X_va_fold = hstack([
                    X12_tr[va_idx],
                    w_vec.transform([texts_tr[i] for i in va_idx]),
                    c_vec.transform([texts_tr[i] for i in va_idx]),
                ])
            else:
                X_tr_fold = X12_tr[tr_idx]
                X_va_fold = X12_tr[va_idx]

            clf = LogisticRegression(C=c_val, class_weight=cw, max_iter=1000, random_state=seed)
            clf.fit(X_tr_fold, y_train[tr_idx])
            logits = clf.decision_function(X_va_fold)
            for pos, orig_i in enumerate(va_idx):
                oof_logits[orig_i] = float(logits[pos])

        shift, cv_m = calibrate_shift(oof_logits, y_train)
        score = (
            1.0 if float(cv_m["recall"]) >= 0.92 else 0.0,
            float(cv_m["accuracy"]),
            float(cv_m["f1"]),
        )
        if score > best_score:
            best_score = score
            best_cfg = (mtype, c_val, cw)
            best_shift = shift
            best_cv = {
                "accuracy": round(float(cv_m["accuracy"]), 6),
                "precision": round(float(cv_m["precision"]), 6),
                "recall": round(float(cv_m["recall"]), 6),
                "f1": round(float(cv_m["f1"]), 6),
                "c_reg": c_val,
                "class_weight": cw,
            }

    final_type, final_c, final_cw = best_cfg
    if final_type == "linear_features_tfidf":
        word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=250, sublinear_tf=True)
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=350, sublinear_tf=True)
        X_full = hstack([X12_tr, word_vec.fit_transform(texts_tr), char_vec.fit_transform(texts_tr)])
    else:
        word_vec, char_vec = None, None
        X_full = X12_tr

    final_clf = LogisticRegression(C=final_c, class_weight=final_cw, max_iter=1000, random_state=seed)
    final_clf.fit(X_full, y_train)

    dev_metrics = None
    if dev_df is not None:
        X12_dev, texts_dev, _ = prepare_dataset(dev_df["text"].tolist())
        if final_type == "linear_features_tfidf":
            X_dev = hstack([X12_dev, word_vec.transform(texts_dev), char_vec.transform(texts_dev)])
        else:
            X_dev = X12_dev
        dev_logits = final_clf.decision_function(X_dev)
        y_dev_true = dev_df["label"].str.strip().str.lower().tolist()
        y_dev_pred = [
            "fraud" if logit_to_score_and_risk(float(z), best_shift)[1] == "high" else "ok"
            for z in dev_logits
        ]
        dm = compute_binary_metrics(y_dev_true, y_dev_pred)
        dev_metrics = {
            "accuracy": round(float(dm["accuracy"]), 6),
            "precision": round(float(dm["precision"]), 6),
            "recall": round(float(dm["recall"]), 6),
            "f1": round(float(dm["f1"]), 6),
            "tp": int(dm["tp"]),
            "fp": int(dm["fp"]),
            "fn": int(dm["fn"]),
            "tn": int(dm["tn"]),
        }
        print("Dev metrics:", dev_metrics)
        for idx, (t, p) in enumerate(zip(y_dev_true, y_dev_pred)):
            if t != p:
                sc, rk = logit_to_score_and_risk(float(dev_logits[idx]), best_shift)
                print(f"ERROR {dev_df.iloc[idx]['id']} true={t} pred={p} score={sc} ({rk}): {dev_df.iloc[idx]['text']}")

    export_sklearn_to_model_pkl(
        clf=final_clf,
        out_path=out_pkl,
        word_vectorizer=word_vec,
        char_vectorizer=char_vec,
        score_shift=best_shift,
        cv_metrics=best_cv,
        dev_metrics=dev_metrics,
        train_csv_path=train_csv,
        n_train=len(train_df),
        used_dev_for_training=False,
        seed=seed,
    )
    payload = verify_model_pkl(out_pkl)

    print("\n12 Feature Report:")
    for fname in FEATURE_NAMES:
        f_hits = sum(1 for fd, yi in zip(feat_dicts_tr, y_train) if yi == 1 and fd[fname] > 0)
        o_hits = sum(1 for fd, yi in zip(feat_dicts_tr, y_train) if yi == 0 and fd[fname] > 0)
        w = payload["feature_weights"][fname]
        print(f"{fname:<26} | fraud={f_hits:>3} | ok={o_hits:>3} | weight={w:>8.4f}")


if __name__ == "__main__":
    run_colab_pipeline()

"""Reproducible L2 Logistic Regression + TF-IDF trainer producing models/model.pkl."""

from __future__ import annotations

import csv
import hashlib
import math
import pickle
import platform
import random
from pathlib import Path
from typing import Any

from src.config import DEFAULT_DEV_CSV, DEFAULT_MODEL_PATH, DEFAULT_TRAIN_CSV, FEATURE_NAMES
from src.features import extract_features
from src.model_loader import LoadedSMSModel, _extract_char_wb_ngrams, _extract_word_ngrams
from src.preprocessing import normalize, strip_urls_for_text_features, unify_kyrgyz
from src.scoring import compute_binary_metrics, logit_to_score_and_risk


def _load_labeled_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if not row or not (row.get("text") or "").strip():
                continue
            rows.append(
                {
                    "id": str(row.get("id") or "").strip(),
                    "text": str(row.get("text") or ""),
                    "label": str(row.get("label") or "").strip().lower(),
                }
            )
        return rows


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_tfidf_string(norm: dict[str, Any]) -> str:
    domains = list(norm.get("domains") or [])
    clean = strip_urls_for_text_features(str(norm.get("clean_text") or ""), domains)
    translit = strip_urls_for_text_features(str(norm.get("translit_text") or ""), domains)
    return unify_kyrgyz(f"{clean} {translit}".strip())


def _fit_tfidf_vocab(
    texts: list[str],
    analyzer: str,
    ngram_range: tuple[int, int],
    min_df: int = 2,
    max_features: int = 250,
) -> dict[str, Any]:
    """Fit smoothed TF-IDF vocabulary matching scikit-learn smooth_idf=True."""
    n_docs = len(texts)
    df_counts: dict[str, int] = {}
    tf_totals: dict[str, int] = {}

    for text in texts:
        grams = (
            _extract_word_ngrams(text, ngram_range)
            if analyzer == "word"
            else _extract_char_wb_ngrams(text, ngram_range)
        )
        seen_in_doc: set[str] = set()
        for g in grams:
            tf_totals[g] = tf_totals.get(g, 0) + 1
            if g not in seen_in_doc:
                df_counts[g] = df_counts.get(g, 0) + 1
                seen_in_doc.add(g)

    eligible = [
        (g, df)
        for g, df in df_counts.items()
        if df >= min_df and df <= int(0.95 * n_docs)
    ]
    eligible.sort(key=lambda item: (-tf_totals[item[0]], item[0]))
    if len(eligible) > max_features:
        eligible = eligible[:max_features]
    # Sort selected terms lexicographically for deterministic indices
    eligible.sort(key=lambda item: item[0])

    vocab: dict[str, int] = {}
    idf: list[float] = []
    for idx, (term, df) in enumerate(eligible):
        vocab[term] = idx
        # sklearn TfidfTransformer(smooth_idf=True) formula
        idf.append(math.log((1.0 + n_docs) / (1.0 + df)) + 1.0)

    return {
        "ngram_range": [int(ngram_range[0]), int(ngram_range[1])],
        "vocab": vocab,
        "idf": idf,
    }


def _transform_tfidf_sparse(
    texts: list[str],
    spec: dict[str, Any],
    analyzer: str,
    offset: int,
    sublinear_tf: bool = True,
    l2_norm: bool = True,
) -> list[list[tuple[int, float]]]:
    vocab: dict[str, int] = spec["vocab"]
    idf: list[float] = spec["idf"]
    ngram_range = tuple(spec["ngram_range"])
    rows: list[list[tuple[int, float]]] = []

    for text in texts:
        grams = (
            _extract_word_ngrams(text, ngram_range)  # type: ignore[arg-type]
            if analyzer == "word"
            else _extract_char_wb_ngrams(text, ngram_range)  # type: ignore[arg-type]
        )
        counts: dict[int, int] = {}
        for g in grams:
            idx = vocab.get(g)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            rows.append([])
            continue

        vals: list[tuple[int, float]] = []
        norm_sq = 0.0
        for idx, cnt in counts.items():
            tf = (1.0 + math.log(cnt)) if sublinear_tf else float(cnt)
            val = tf * float(idf[idx])
            vals.append((idx, val))
            norm_sq += val * val

        inv_norm = (1.0 / math.sqrt(norm_sq)) if (l2_norm and norm_sq > 0.0) else 1.0
        rows.append([(offset + idx, val * inv_norm) for idx, val in vals])

    return rows


def _fit_l2_logistic_regression(
    sparse_X: list[list[tuple[int, float]]],
    y: list[int],
    dim: int,
    c_reg: float = 2.0,
    pos_weight: float = 1.0,
    max_iter: int = 250,
) -> tuple[list[float], float]:
    """Fit exact convex L2 Logistic Regression using L-BFGS + Armijo line search in pure Python."""
    n = len(sparse_X)
    if n == 0:
        return [0.0] * dim, 0.0

    # Match sklearn penalty convention: sum_i loss_i + (0.5 / C) * ||w||^2
    inv_c = 1.0 / max(1e-6, float(c_reg))
    sample_weights = [float(pos_weight) if yi == 1 else 1.0 for yi in y]

    def loss_and_grad(params: list[float]) -> tuple[float, list[float]]:
        b = params[dim]
        grad = [0.0] * (dim + 1)
        total_loss = 0.0

        for i in range(n):
            z = b
            row = sparse_X[i]
            for j, v in row:
                z += params[j] * v
            z = max(-35.0, min(35.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            yi = float(y[i])
            sw = sample_weights[i]

            if yi == 1.0:
                total_loss += sw * math.log1p(math.exp(-z))
            else:
                total_loss += sw * math.log1p(math.exp(z))

            diff = sw * (p - yi)
            for j, v in row:
                grad[j] += diff * v
            grad[dim] += diff

        # L2 regularization ONLY on weights (never on intercept)
        reg_loss = 0.0
        for j in range(dim):
            wj = params[j]
            reg_loss += 0.5 * inv_c * wj * wj
            grad[j] += inv_c * wj

        return total_loss + reg_loss, grad

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y_ for x, y_ in zip(a, b))

    params = [0.0] * (dim + 1)
    loss, grad = loss_and_grad(params)
    m_hist = 10
    s_hist: list[list[float]] = []
    y_hist: list[list[float]] = []
    rho_hist: list[float] = []

    for _ in range(max_iter):
        grad_norm = math.sqrt(dot(grad, grad))
        if grad_norm < 1e-6:
            break

        # L-BFGS two-loop recursion
        q = grad[:]
        k = len(s_hist)
        alphas = [0.0] * k
        for idx in range(k - 1, -1, -1):
            a = rho_hist[idx] * dot(s_hist[idx], q)
            alphas[idx] = a
            si_yi = y_hist[idx]
            for j in range(dim + 1):
                q[j] -= a * si_yi[j]

        if k > 0:
            yy = dot(y_hist[-1], y_hist[-1])
            gamma = dot(s_hist[-1], y_hist[-1]) / yy if yy > 1e-12 else 1.0
        else:
            gamma = 1.0 / max(1.0, grad_norm)

        r = [gamma * qj for qj in q]
        for idx in range(k):
            b_val = rho_hist[idx] * dot(y_hist[idx], r)
            coef = alphas[idx] - b_val
            si = s_hist[idx]
            for j in range(dim + 1):
                r[j] += coef * si[j]

        direction = [-rj for rj in r]
        slope = dot(grad, direction)
        if slope >= 0.0:
            direction = [-gj for gj in grad]
            slope = -grad_norm * grad_norm

        step = 1.0
        accepted = False
        new_params = params
        new_loss = loss
        new_grad = grad
        for _ls in range(25):
            cand = [params[j] + step * direction[j] for j in range(dim + 1)]
            cand_loss, cand_grad = loss_and_grad(cand)
            if cand_loss <= loss + 1e-4 * step * slope:
                new_params = cand
                new_loss = cand_loss
                new_grad = cand_grad
                accepted = True
                break
            step *= 0.5

        if not accepted:
            break

        s_vec = [new_params[j] - params[j] for j in range(dim + 1)]
        y_vec = [new_grad[j] - grad[j] for j in range(dim + 1)]
        sy = dot(s_vec, y_vec)
        if sy > 1e-12:
            if len(s_hist) == m_hist:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(s_vec)
            y_hist.append(y_vec)
            rho_hist.append(1.0 / sy)

        params = new_params
        loss = new_loss
        grad = new_grad

    return params[:dim], params[dim]


def _stratified_kfold_indices(y: list[int], n_splits: int = 5, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    rng = random.Random(seed)
    pos_idx = [i for i, label in enumerate(y) if label == 1]
    neg_idx = [i for i, label in enumerate(y) if label == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for i, idx in enumerate(pos_idx):
        folds[i % n_splits].append(idx)
    for i, idx in enumerate(neg_idx):
        folds[i % n_splits].append(idx)

    splits: list[tuple[list[int], list[int]]] = []
    for k in range(n_splits):
        val = sorted(folds[k])
        val_set = set(val)
        tr = [i for i in range(len(y)) if i not in val_set]
        splits.append((tr, val))
    return splits


def _build_design_matrix(
    norms: list[dict[str, Any]],
    feat_dicts: list[dict[str, float]],
    use_tfidf: bool,
    word_spec: dict[str, Any] | None = None,
    char_spec: dict[str, Any] | None = None,
    fit_tfidf: bool = False,
) -> tuple[list[list[tuple[int, float]]], int, dict[str, Any] | None, dict[str, Any] | None]:
    n_feat = len(FEATURE_NAMES)
    dense_sparse: list[list[tuple[int, float]]] = []
    for fd in feat_dicts:
        row = [(i, float(fd[fname])) for i, fname in enumerate(FEATURE_NAMES) if fd[fname] != 0.0]
        dense_sparse.append(row)

    if not use_tfidf:
        return dense_sparse, n_feat, None, None

    tfidf_texts = [_prepare_tfidf_string(nm) for nm in norms]
    if fit_tfidf:
        word_spec = _fit_tfidf_vocab(tfidf_texts, "word", (1, 2), min_df=2, max_features=250)
        char_spec = _fit_tfidf_vocab(tfidf_texts, "char_wb", (3, 5), min_df=2, max_features=350)

    assert word_spec is not None and char_spec is not None
    n_word = len(word_spec["vocab"])
    n_char = len(char_spec["vocab"])
    word_rows = _transform_tfidf_sparse(tfidf_texts, word_spec, "word", offset=n_feat)
    char_rows = _transform_tfidf_sparse(tfidf_texts, char_spec, "char_wb", offset=n_feat + n_word)

    combined = [
        dense_sparse[i] + word_rows[i] + char_rows[i]
        for i in range(len(norms))
    ]
    return combined, n_feat + n_word + n_char, word_spec, char_spec


def _calibrate_shift_on_logits(
    logits: list[float], y: list[int]
) -> tuple[float, dict[str, float | int]]:
    """Select logit-space score_shift maximizing Accuracy with Recall >= 0.92 at score >= 65."""
    y_true_str = ["fraud" if yi == 1 else "ok" for yi in y]
    best_shift = 0.0
    best_key = (-1.0, -1.0, -10.0)
    best_metrics: dict[str, float | int] = {}

    # Scan [-0.50 .. +2.00] in 0.05 logit increments
    for step in range(-10, 41):
        shift = round(step * 0.05, 4)
        preds = [
            "fraud" if logit_to_score_and_risk(z, shift)[1] == "high" else "ok"
            for z in logits
        ]
        m = compute_binary_metrics(y_true_str, preds)
        acc = float(m["accuracy"])
        rec = float(m["recall"])
        meets_rec = 1.0 if rec >= 0.92 else 0.0
        # Prefer meeting recall target, highest accuracy, highest f1, smallest |shift - 0.6|
        key = (meets_rec, acc, float(m["f1"]), rec, -abs(shift - 0.6))
        if key > best_key:
            best_key = key  # type: ignore[assignment]
            best_shift = shift
            best_metrics = m

    return best_shift, best_metrics


def _build_payload_from_fit(
    model_type: str,
    weights: list[float],
    intercept: float,
    score_shift: float,
    word_spec: dict[str, Any] | None,
    char_spec: dict[str, Any] | None,
    train_sha: str,
    n_train: int,
    seed: int,
    cv_metrics: dict[str, Any],
    dev_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n_feat = len(FEATURE_NAMES)
    feature_weights = {fname: round(float(weights[i]), 8) for i, fname in enumerate(FEATURE_NAMES)}

    tfidf_block: dict[str, Any] | None = None
    if model_type == "linear_features_tfidf" and word_spec is not None and char_spec is not None:
        n_word = len(word_spec["vocab"])
        n_char = len(char_spec["vocab"])
        word_weights = [round(float(w), 8) for w in weights[n_feat : n_feat + n_word]]
        char_weights = [round(float(w), 8) for w in weights[n_feat + n_word : n_feat + n_word + n_char]]
        tfidf_block = {
            "input_field": "clean_and_translit",
            "strip_urls": True,
            "unify_kyrgyz": True,
            "sublinear_tf": True,
            "l2_norm": True,
            "word": {
                "ngram_range": word_spec["ngram_range"],
                "vocab": word_spec["vocab"],
                "idf": [round(float(x), 8) for x in word_spec["idf"]],
                "weights": word_weights,
            },
            "char_wb": {
                "ngram_range": char_spec["ngram_range"],
                "vocab": char_spec["vocab"],
                "idf": [round(float(x), 8) for x in char_spec["idf"]],
                "weights": char_weights,
            },
        }

    return {
        "checkpoint_version": "3.1",
        "model_type": model_type,
        "feature_names": list(FEATURE_NAMES),
        "feature_weights": feature_weights,
        "intercept": round(float(intercept), 8),
        "score_shift": round(float(score_shift), 6),
        "feature_scaler": None,
        "tfidf": tfidf_block,
        "meta": {
            "trained_on_sha256": train_sha,
            "n_train": n_train,
            "used_dev_for_training": False,
            "seed": seed,
            "trainer": "pure_python_l2_logreg",
            "python_version": platform.python_version(),
            "cv_metrics": cv_metrics,
            "dev_metrics": dev_metrics,
        },
    }


def train_model_payload(
    train_csv: Path = DEFAULT_TRAIN_CSV,
    dev_csv: Path | None = DEFAULT_DEV_CSV,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and calibrate L2 Logistic Regression strictly on sms_train.csv via 5-fold CV."""
    train_rows = _load_labeled_csv(train_csv)
    train_sha = _file_sha256(train_csv)
    y = [1 if r["label"] == "fraud" else 0 for r in train_rows]
    norms = [normalize(r["text"]) for r in train_rows]
    feat_dicts = [extract_features(nm)[0] for nm in norms]

    splits = _stratified_kfold_indices(y, n_splits=5, seed=seed)

    candidates: list[tuple[str, float, float]] = [
        ("linear_features_tfidf", 2.0, 1.0),
        ("linear_features_tfidf", 4.0, 1.0),
        ("linear_features_tfidf", 2.0, 1.2),
        ("linear_features", 2.0, 1.0),
        ("linear_features", 5.0, 1.0),
    ]

    best_score = (-1.0, -1.0, -1.0)
    best_cfg: tuple[str, float, float] = candidates[0]
    best_shift = 0.6
    best_cv_metrics: dict[str, Any] = {}

    for mtype, c_reg, pos_w in candidates:
        use_tfidf = mtype == "linear_features_tfidf"
        oof_logits = [0.0] * len(y)
        for tr_idx, val_idx in splits:
            tr_norms = [norms[i] for i in tr_idx]
            tr_feats = [feat_dicts[i] for i in tr_idx]
            tr_y = [y[i] for i in tr_idx]
            X_tr, dim, w_spec, c_spec = _build_design_matrix(
                tr_norms, tr_feats, use_tfidf=use_tfidf, fit_tfidf=True
            )
            w, b = _fit_l2_logistic_regression(X_tr, tr_y, dim, c_reg=c_reg, pos_weight=pos_w)

            val_norms = [norms[i] for i in val_idx]
            val_feats = [feat_dicts[i] for i in val_idx]
            X_val, _, _, _ = _build_design_matrix(
                val_norms, val_feats, use_tfidf=use_tfidf, word_spec=w_spec, char_spec=c_spec, fit_tfidf=False
            )
            for local_i, orig_i in enumerate(val_idx):
                z = b
                for j, val in X_val[local_i]:
                    z += w[j] * val
                oof_logits[orig_i] = z

        shift, cv_m = _calibrate_shift_on_logits(oof_logits, y)
        score_tuple = (
            1.0 if float(cv_m["recall"]) >= 0.92 else 0.0,
            float(cv_m["accuracy"]),
            float(cv_m["f1"]),
        )
        if score_tuple > best_score:
            best_score = score_tuple
            best_cfg = (mtype, c_reg, pos_w)
            best_shift = shift
            best_cv_metrics = {
                "accuracy": round(float(cv_m["accuracy"]), 6),
                "precision": round(float(cv_m["precision"]), 6),
                "recall": round(float(cv_m["recall"]), 6),
                "f1": round(float(cv_m["f1"]), 6),
                "c_reg": c_reg,
                "pos_weight": pos_w,
            }

    # Final refit on all 200 rows of sms_train.csv using CV-selected hyperparameters
    final_mtype, final_c, final_pw = best_cfg
    use_tfidf = final_mtype == "linear_features_tfidf"
    X_full, full_dim, word_spec, char_spec = _build_design_matrix(
        norms, feat_dicts, use_tfidf=use_tfidf, fit_tfidf=True
    )
    final_w, final_b = _fit_l2_logistic_regression(
        X_full, y, full_dim, c_reg=final_c, pos_weight=final_pw
    )

    payload = _build_payload_from_fit(
        model_type=final_mtype,
        weights=final_w,
        intercept=final_b,
        score_shift=best_shift,
        word_spec=word_spec,
        char_spec=char_spec,
        train_sha=train_sha,
        n_train=len(train_rows),
        seed=seed,
        cv_metrics=best_cv_metrics,
        dev_metrics=None,
    )

    # Independent evaluation on sms_dev.csv using exact LoadedSMSModel path
    if dev_csv is not None and dev_csv.is_file():
        loaded = LoadedSMSModel(payload, source_path="<train-eval>")
        dev_rows = _load_labeled_csv(dev_csv)
        dev_true: list[str] = []
        dev_pred: list[str] = []
        for row in dev_rows:
            nm = normalize(row["text"])
            f_vals, _ = extract_features(nm)
            z_raw, _ = loaded.predict_logit(nm, f_vals)
            _, risk = logit_to_score_and_risk(z_raw, loaded.score_shift)
            dev_true.append(row["label"])
            dev_pred.append("fraud" if risk == "high" else "ok")
        dm = compute_binary_metrics(dev_true, dev_pred)
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

    return payload


def save_model_pickle(payload: dict[str, Any], out_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist payload as portable protocol-4 Pickle file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    return out_path


def main() -> None:
    payload = train_model_payload()
    saved = save_model_pickle(payload, DEFAULT_MODEL_PATH)

    train_rows = _load_labeled_csv(DEFAULT_TRAIN_CSV)
    fraud_hits = {f: 0 for f in FEATURE_NAMES}
    ok_hits = {f: 0 for f in FEATURE_NAMES}
    for row in train_rows:
        vals, _ = extract_features(normalize(row["text"]))
        bucket = fraud_hits if row["label"] == "fraud" else ok_hits
        for f in FEATURE_NAMES:
            if vals[f] > 0.0:
                bucket[f] += 1

    print(f"Saved model: {saved} ({saved.stat().st_size} bytes)")
    print(f"Model type: {payload['model_type']}, intercept={payload['intercept']}, score_shift={payload['score_shift']}")
    print("CV metrics:", payload["meta"]["cv_metrics"])
    print("Dev metrics:", payload["meta"]["dev_metrics"])
    print("\nFeature | Fraud(100) | OK(100) | Weight | Sign")
    print("-" * 58)
    for f in FEATURE_NAMES:
        w = payload["feature_weights"][f]
        sign = "+" if w > 0 else ("-" if w < 0 else "0")
        print(f"{f:<26} | {fraud_hits[f]:>10} | {ok_hits[f]:>7} | {w:>8.4f} | {sign}")


if __name__ == "__main__":
    main()

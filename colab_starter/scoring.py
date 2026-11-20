"""Colab-standalone copy of src/scoring.py (do not diverge from src/scoring.py)."""

from __future__ import annotations

import math
from typing import Iterable

from config import RISK_HIGH_MIN, RISK_MEDIUM_MIN


def logit_to_score_and_risk(z_raw: float, score_shift: float = 0.0) -> tuple[int, str]:
    """Convert raw model logit plus logit-space score_shift into (score 0..100, risk level)."""
    z = float(z_raw) + float(score_shift)
    if math.isnan(z):
        z = 0.0
    z = max(-30.0, min(30.0, z))
    prob = 1.0 / (1.0 + math.exp(-z))
    score = int(round(prob * 100.0))
    score = max(0, min(100, score))
    if score >= RISK_HIGH_MIN:
        risk = "high"
    elif score >= RISK_MEDIUM_MIN:
        risk = "medium"
    else:
        risk = "low"
    return score, risk


def prob_to_logit(prob: float) -> float:
    """Convert probability in (0, 1) to raw logit safely."""
    p = max(1e-7, min(1.0 - 1e-7, float(prob)))
    return math.log(p / (1.0 - p))


def compute_binary_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, float | int]:
    """Compute Accuracy, Precision, Recall, F1 and Confusion Matrix for positive class 'fraud'."""
    tp = fp = fn = tn = 0
    for truth, pred in zip(y_true, y_pred):
        t_pos = truth == "fraud"
        p_pos = pred == "fraud"
        if t_pos and p_pos:
            tp += 1
        elif not t_pos and p_pos:
            fp += 1
        elif t_pos and not p_pos:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
    }

"""Mandatory jury evaluation script for SMS Guard (`id,text,label` and `id,text`)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from src.config import DEFAULT_DEV_CSV
from src.detector import check, reload_model
from src.scoring import compute_binary_metrics


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _read_csv_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows: list[dict[str, str]] = []
        for idx, row in enumerate(reader, start=1):
            if not row:
                continue
            text = row.get("text")
            if text is None or (not str(text).strip() and not (row.get("id") or "").strip()):
                continue
            row_id = str(row.get("id") or f"ROW_{idx:03d}").strip()
            clean_row: dict[str, str] = {"id": row_id, "text": str(text)}
            if "label" in fieldnames:
                clean_row["label"] = str(row.get("label") or "").strip().lower()
            rows.append(clean_row)
    return rows, fieldnames


def evaluate_csv(
    csv_path: Path,
    model_path: Path | None = None,
    out_csv: Path | None = None,
    show_errors: bool = False,
) -> int:
    """Evaluate SMS Guard on labeled (`id,text,label`) or unlabeled (`id,text`) CSV."""
    if model_path is not None:
        reload_model(model_path)

    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    rows, fieldnames = _read_csv_rows(csv_path)
    has_label = "label" in fieldnames and all(r.get("label") in ("fraud", "ok") for r in rows) and len(rows) > 0

    results: list[dict[str, object]] = []
    risk_counts = {"high": 0, "medium": 0, "low": 0}

    for row in rows:
        res = check(row["text"])
        pred_label = "fraud" if res["risk"] == "high" else "ok"
        risk_counts[res["risk"]] = risk_counts.get(res["risk"], 0) + 1
        results.append(
            {
                "id": row["id"],
                "text": row["text"],
                "true_label": row.get("label", ""),
                "pred_label": pred_label,
                "risk": res["risk"],
                "score": res["score"],
                "reasons": res["reasons"],
            }
        )

    if out_csv is not None or not has_label:
        target_out = out_csv if out_csv is not None else Path("predictions.csv")
        target_out.parent.mkdir(parents=True, exist_ok=True)
        with open(target_out, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "risk", "score"])
            for item in results:
                writer.writerow([item["id"], item["pred_label"], item["risk"], item["score"]])
        print(f"Saved predictions ({len(results)} rows) -> {target_out}")

    if not has_label:
        print(
            f"Unlabeled dataset summary ({len(results)} SMS): "
            f"high(fraud)={risk_counts['high']}, "
            f"medium(ok)={risk_counts['medium']}, "
            f"low(ok)={risk_counts['low']}"
        )
        return 0

    y_true = [str(r["true_label"]) for r in results]
    y_pred = [str(r["pred_label"]) for r in results]
    m = compute_binary_metrics(y_true, y_pred)

    print(f"Evaluated: {csv_path} ({m['total']} SMS)")
    print(f"Accuracy : {m['accuracy']:.4f} ({m['accuracy'] * 100:.2f}%)")
    print(f"Precision (fraud): {m['precision']:.4f} ({m['precision'] * 100:.2f}%)")
    print(f"Recall    (fraud): {m['recall']:.4f} ({m['recall'] * 100:.2f}%)")
    print(f"F1        (fraud): {m['f1']:.4f}")
    print("\nConfusion Matrix (positive='fraud', high==fraud):")
    print("                Pred fraud | Pred ok")
    print(f"Actual fraud  : {m['tp']:>10} | {m['fn']:>7}   (TP={m['tp']}, FN={m['fn']})")
    print(f"Actual ok     : {m['fp']:>10} | {m['tn']:>7}   (FP={m['fp']}, TN={m['tn']})")

    if show_errors:
        errors = [r for r in results if r["true_label"] != r["pred_label"]]
        print(f"\nErrors ({len(errors)}):")
        if not errors:
            print("  None (0 FP, 0 FN)")
        for err in errors:
            kind = "FP" if err["pred_label"] == "fraud" else "FN"
            print(
                f"  [{kind}] {err['id']} score={err['score']} risk={err['risk']} "
                f"reasons={err['reasons']} | {err['text']}"
            )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SMS Guard on CSV dataset.")
    parser.add_argument("csv_pos", nargs="?", help="Path to CSV file (default: data/sms_dev.csv)")
    parser.add_argument("--input", "-i", dest="csv_flag", help="Path to CSV file")
    parser.add_argument("--model", "-m", dest="model_path", help="Optional path to model.pkl")
    parser.add_argument("--out", "-o", dest="out_csv", help="Optional output CSV path for predictions")
    parser.add_argument("--errors", action="store_true", help="Print misclassified FP/FN rows")
    return parser


def main() -> int:
    _configure_utf8_stdio()
    args = build_parser().parse_args()
    csv_arg = args.csv_flag or args.csv_pos
    csv_path = Path(csv_arg) if csv_arg else DEFAULT_DEV_CSV
    model_path = Path(args.model_path) if args.model_path else None
    out_csv = Path(args.out_csv) if args.out_csv else None
    return evaluate_csv(csv_path, model_path=model_path, out_csv=out_csv, show_errors=args.errors)


if __name__ == "__main__":
    raise SystemExit(main())

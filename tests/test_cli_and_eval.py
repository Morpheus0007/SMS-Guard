"""Integration tests for main.py CLI, evaluate.py, app.py/bot.py syntax, and export_pkl.py."""

from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

from src.config import BASE_DIR, FEATURE_NAMES


def test_main_cli_outputs_strict_json() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(BASE_DIR / "main.py"),
            "С р о ч н о ! Ваша SIM-к@рт@ будет заблокирована. Перезвоните на 0770 553707",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    payload = json.loads(proc.stdout.strip())
    assert set(payload.keys()) == {"risk", "score", "reasons"}
    assert payload["risk"] == "high"
    assert payload["score"] >= 65


def test_main_cli_stdin_outputs_strict_json() -> None:
    proc = subprocess.run(
        [sys.executable, str(BASE_DIR / "main.py")],
        input="Код подтверждения: 123456. Никому не сообщайте его",
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    payload = json.loads(proc.stdout.strip())
    assert payload["risk"] != "high"


def test_evaluate_labeled_and_unlabeled_csv(tmp_path: Path) -> None:
    # 1. Labeled CSV (dev)
    proc_labeled = subprocess.run(
        [sys.executable, str(BASE_DIR / "evaluate.py"), str(BASE_DIR / "data" / "sms_dev.csv")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert "Accuracy" in proc_labeled.stdout
    assert "Recall" in proc_labeled.stdout
    assert "Confusion Matrix" in proc_labeled.stdout

    # 2. Unlabeled CSV (id,text only, matching hidden jury submission mode)
    unlabeled_csv = tmp_path / "hidden_like.csv"
    out_csv = tmp_path / "predictions.csv"
    with open(unlabeled_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text"])
        w.writerow(["H01", "Срочно сообщите код из SMS по номеру 0555 123456"])
        w.writerow(["H02", "Привет, увидимся завтра в 10:00"])

    proc_unlabeled = subprocess.run(
        [
            sys.executable,
            str(BASE_DIR / "evaluate.py"),
            "--input",
            str(unlabeled_csv),
            "--out",
            str(out_csv),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert "Unlabeled dataset summary" in proc_unlabeled.stdout
    assert out_csv.is_file()
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["label"] == "fraud"
    assert rows[1]["label"] == "ok"


def test_evaluate_blind_test_generalization() -> None:
    blind_csv = BASE_DIR / "tests" / "blind_test.csv"
    proc = subprocess.run(
        [sys.executable, str(BASE_DIR / "evaluate.py"), str(blind_csv)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert "Accuracy" in proc.stdout


def test_streamlit_and_bot_syntax_and_colab_export(tmp_path: Path) -> None:
    for fname in ("app.py", "bot.py"):
        src_text = (BASE_DIR / fname).read_text(encoding="utf-8")
        ast.parse(src_text, filename=fname)

    # Round-trip test of colab_starter/export_pkl.py
    colab_dir = str(BASE_DIR / "colab_starter")
    if colab_dir not in sys.path:
        sys.path.insert(0, colab_dir)
    import export_pkl  # type: ignore

    class DummyClf:
        coef_ = [[0.4, 2.0, 1.0, 1.5, 1.5, -0.2, 1.5, 0.6, -1.0, 2.0, -1.2, 1.0]]
        intercept_ = [-2.0]

    out_pkl = tmp_path / "colab_out.pkl"
    export_pkl.export_sklearn_to_model_pkl(
        clf=DummyClf(),
        out_path=out_pkl,
        score_shift=0.6,
        train_csv_path=BASE_DIR / "data" / "sms_train.csv",
    )
    loaded_dict = export_pkl.verify_model_pkl(out_pkl)
    assert tuple(loaded_dict["feature_names"]) == FEATURE_NAMES

    from src.model_loader import LoadedSMSModel

    model = LoadedSMSModel(loaded_dict, source_path=str(out_pkl))
    assert model.format_kind == "dict_pickle_v3"

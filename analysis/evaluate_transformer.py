"""Regenerate summary plots/text from a transformer run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from analysis.train_transformer import plot_confusion, write_csv


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    y_true = np.asarray([int(r["true_label"]) for r in rows], dtype=np.int64)
    y_pred = np.asarray([int(r["pred_tuned"]) for r in rows], dtype=np.int64)
    return y_true, y_pred


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    metrics_path = args.run_dir / "metrics.json"
    preds_path = args.run_dir / "fold_predictions.csv"
    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}")
    if not preds_path.exists():
        raise SystemExit(f"Missing {preds_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    class_names = metrics["class_names"]
    y_true, y_pred = _read_predictions(preds_path)
    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plot_confusion(cm, class_names, args.run_dir / "confusion_matrix.png", "Aggregate LOPO confusion matrix")
    (args.run_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
        encoding="utf-8",
    )

    fold_rows = []
    for fold in metrics["folds"]:
        m = fold["metrics_tuned_threshold"]
        fold_rows.append({
            "fold": fold["fold"],
            "held_out_participant": fold["held_out_participant"],
            "n_train": fold["n_train"],
            "n_val": fold["n_val"],
            "n_test": fold["n_test"],
            "accuracy": m.get("accuracy"),
            "balanced_accuracy": m.get("balanced_accuracy"),
            "f1_macro": m.get("f1_macro"),
            "roc_auc": m.get("roc_auc", m.get("roc_auc_ovr_macro", "")),
            "tuned_threshold": fold["tuned_threshold"],
        })
    write_csv(args.run_dir / "per_fold_metrics.csv", fold_rows)
    print(f"[evaluate] regenerated evaluation artefacts in {args.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Train classical baselines on preprocessed sequence windows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from analysis.make_splits import make_lopo_splits, save_splits
from analysis.sequence_dataset import SequenceData, load_sequence_data
from analysis.train_transformer import plot_confusion, write_csv


BASELINE_FEATURES = [
    "mean_angular_speed",
    "median_angular_speed",
    "max_angular_speed",
    "p95_angular_speed",
    "total_abs_yaw_movement",
    "total_abs_pitch_movement",
    "yaw_range",
    "pitch_range",
    "yaw_direction_changes",
    "pitch_direction_changes",
    "mean_player_speed",
    "max_player_speed",
    "visible_enemy_proportion",
    "mean_target_error",
    "target_error_at_fire",
]


def _feature_index(names: list[str], name: str) -> int | None:
    try:
        return names.index(name)
    except ValueError:
        return None


def _col(X: np.ndarray, names: list[str], name: str) -> np.ndarray:
    idx = _feature_index(names, name)
    if idx is None:
        return np.zeros(X.shape[:2], dtype=np.float32)
    return X[:, :, idx]


def _direction_changes(values: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    signs = np.sign(values)
    signs[np.abs(values) < eps] = 0
    out = np.zeros(values.shape[0], dtype=np.float32)
    for i, row in enumerate(signs):
        nz = row[row != 0]
        out[i] = float(np.sum(nz[1:] != nz[:-1])) if nz.size > 1 else 0.0
    return out


def engineered_window_features(data: SequenceData) -> tuple[np.ndarray, list[str]]:
    X = data.X
    names = data.feature_names
    angular = _col(X, names, "angular_speed")
    yaw_delta = _col(X, names, "yaw_delta")
    pitch_delta = _col(X, names, "pitch_delta")
    player_speed = _col(X, names, "player_speed")
    target_error = _col(X, names, "target_angular_error")
    target_visible = _col(X, names, "target_visible")

    feats = np.column_stack([
        np.mean(angular, axis=1),
        np.median(angular, axis=1),
        np.max(angular, axis=1),
        np.percentile(angular, 95, axis=1),
        np.sum(np.abs(yaw_delta), axis=1),
        np.sum(np.abs(pitch_delta), axis=1),
        np.max(np.cumsum(yaw_delta, axis=1), axis=1) - np.min(np.cumsum(yaw_delta, axis=1), axis=1),
        np.max(np.cumsum(pitch_delta, axis=1), axis=1) - np.min(np.cumsum(pitch_delta, axis=1), axis=1),
        _direction_changes(yaw_delta),
        _direction_changes(pitch_delta),
        np.mean(player_speed, axis=1),
        np.max(player_speed, axis=1),
        np.mean(target_visible, axis=1),
        np.mean(np.nan_to_num(target_error, nan=0.0), axis=1),
        np.nan_to_num(target_error[:, -1], nan=0.0),
    ]).astype(np.float32)
    return feats, list(BASELINE_FEATURES)


def make_model(name: str):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ])
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1, class_weight="balanced")
    if name == "linear_svm":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearSVC(class_weight="balanced", random_state=0, max_iter=10000)),
        ])
    if name == "gb":
        return GradientBoostingClassifier(random_state=0)
    raise ValueError(f"Unknown baseline model: {name}")


def predict_scores(model: Any, X: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    pred = model.predict(X).astype(np.int64)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        if n_classes == 2:
            raw = raw.reshape(-1)
            prob1 = 1.0 / (1.0 + np.exp(-raw))
            scores = np.column_stack([1.0 - prob1, prob1])
        else:
            raw = np.asarray(raw)
            exp = np.exp(raw - raw.max(axis=1, keepdims=True))
            scores = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    else:
        scores = np.zeros((X.shape[0], n_classes), dtype=np.float32)
        scores[np.arange(X.shape[0]), pred] = 1.0
    return pred, scores


def metrics(y_true: np.ndarray, pred: np.ndarray, scores: np.ndarray, n_classes: int, class_names: list[str]) -> dict[str, Any]:
    p, r, f, support = precision_recall_fscore_support(y_true, pred, labels=list(range(n_classes)), zero_division=0)
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(n_classes))).tolist(),
        "per_class": {
            class_names[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(support[i]),
            }
            for i in range(n_classes)
        },
    }
    try:
        if n_classes == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, scores[:, 1]))
        else:
            out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro", labels=list(range(n_classes))))
    except ValueError:
        pass
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_sequence_data(args.data, args.metadata, args.task)
    X, feature_names = engineered_window_features(data)
    splits = make_lopo_splits(data.participants, data.sessions, seed=args.seed)
    save_splits(splits, out_dir / "fold_splits.json")
    n_classes = len(data.class_names)

    all_results: dict[str, Any] = {}
    all_predictions: list[dict[str, Any]] = []
    for model_name in args.models:
        fold_results = []
        aggregate_cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for split in splits:
            tr = np.asarray(split["train_idx"], dtype=np.int64)
            te = np.asarray(split["test_idx"], dtype=np.int64)
            model = make_model(model_name)
            model.fit(X[tr], data.y[tr])
            pred, scores = predict_scores(model, X[te], n_classes)
            m = metrics(data.y[te], pred, scores, n_classes, data.class_names)
            aggregate_cm += np.asarray(m["confusion_matrix"], dtype=np.int64)
            fold_results.append({
                "fold": split["fold"],
                "held_out_participant": split["held_out_participant"],
                "n_train": int(tr.size),
                "n_test": int(te.size),
                "class_counts_train": dict(Counter(data.y[tr].tolist())),
                "class_counts_test": dict(Counter(data.y[te].tolist())),
                "metrics": m,
            })
            for local_i, global_i in enumerate(te.tolist()):
                row = dict(data.metadata[global_i])
                row.update({
                    "model": model_name,
                    "fold": split["fold"],
                    "true_label": int(data.y[global_i]),
                    "pred_label": int(pred[local_i]),
                })
                for class_idx, class_name in enumerate(data.class_names):
                    row[f"score_{class_name}"] = float(scores[local_i, class_idx])
                all_predictions.append(row)

        summary = {}
        for metric_name in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc", "roc_auc_ovr_macro"]:
            vals = [float(fr["metrics"][metric_name]) for fr in fold_results if metric_name in fr["metrics"]]
            if vals:
                summary[f"{metric_name}_mean"] = float(np.mean(vals))
                summary[f"{metric_name}_std"] = float(np.std(vals))
        all_results[model_name] = {
            "folds": fold_results,
            "summary": summary,
            "aggregate_confusion_matrix": aggregate_cm.tolist(),
        }
        plot_confusion(aggregate_cm, data.class_names, out_dir / f"confusion_matrix_{model_name}.png", f"{model_name} aggregate confusion matrix")

    payload = {
        "task": args.task,
        "class_names": data.class_names,
        "source_data": str(args.data),
        "source_metadata": str(args.metadata),
        "engineered_feature_names": feature_names,
        "models": all_results,
    }
    (out_dir / "baseline_metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    write_csv(out_dir / "baseline_predictions.csv", all_predictions)
    print(f"[baselines] wrote metrics and predictions to {out_dir}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--task", choices=("binary_all", "binary_smooth", "binary_humanised", "multiclass"), default="binary_all")
    parser.add_argument("--models", nargs="+", default=["majority", "logreg", "rf", "linear_svm", "gb"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/results/baselines"))
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

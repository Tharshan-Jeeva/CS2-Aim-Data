"""Train a small PyTorch Transformer Encoder on preprocessed aim windows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

from analysis.make_splits import make_lopo_splits, save_splits
from analysis.sequence_dataset import AimSequenceDataset, SequenceData, load_sequence_data
from analysis.transformer_model import AimTransformerEncoder, TransformerConfig


@dataclass
class TrainConfig:
    data: str
    metadata: str
    task: str
    splitter: str
    out_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    seed: int
    device: str
    require_cuda: bool
    amp: bool
    grad_clip: float
    weighted_sampler: bool
    num_workers: int
    d_model: int
    n_heads: int
    n_layers: int
    dim_feedforward: int
    dropout: float
    pooling: str
    conv_kernel: int
    conv_stride: int


@dataclass
class ChannelScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "ChannelScaler":
        flat = X.reshape(-1, X.shape[-1])
        mean = flat.mean(axis=0).astype(np.float32)
        std = flat.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(spec: str, require_cuda: bool) -> torch.device:
    if spec == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(spec)
    if require_cuda and device.type != "cuda":
        raise SystemExit("CUDA was required (--require-cuda) but torch.cuda.is_available() is false.")
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
    return device


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    weighted_sampler: bool = False,
) -> DataLoader:
    sampler = None
    if weighted_sampler:
        counts = np.bincount(y.astype(np.int64), minlength=int(y.max()) + 1).astype(np.float64)
        weights = 1.0 / np.maximum(counts, 1.0)
        sample_weights = weights[y.astype(np.int64)]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    return DataLoader(
        AimSequenceDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def class_weights(y: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y.astype(np.int64), minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    grad_clip: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(xb)
                loss = loss_fn(logits, yb)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        losses.append(float(loss.detach().cpu()) * xb.shape[0])
        ys.append(yb.detach().cpu().numpy())
        probs.append(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())

    n = sum(len(x) for x in ys)
    return (
        float(sum(losses) / max(1, n)),
        np.concatenate(ys) if ys else np.zeros(0, dtype=np.int64),
        np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=np.float32),
    )


def predict_from_probs(probs: np.ndarray, n_classes: int, threshold: float = 0.5) -> np.ndarray:
    if n_classes == 2:
        return (probs[:, 1] >= threshold).astype(np.int64)
    return np.argmax(probs, axis=1).astype(np.int64)


def tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = predict_from_probs(probs, 2, float(threshold))
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def fold_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    n_classes = len(class_names)
    pred = predict_from_probs(probs, n_classes, threshold)
    labels = list(range(n_classes))
    p, r, f, support = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
        "per_class": {
            class_names[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
    }
    if n_classes == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        except ValueError:
            out["roc_auc"] = float("nan")
        out["human_recall"] = out["per_class"][class_names[0]]["recall"]
        out["aimbot_recall"] = out["per_class"][class_names[1]]["recall"]
    else:
        try:
            out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels))
        except ValueError:
            out["roc_auc_ovr_macro"] = float("nan")
    return out


def plot_confusion(cm: np.ndarray, class_names: list[str], out_path: Path, title: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(histories: list[dict[str, list[float]]], out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for hist in histories:
        ax.plot(hist["train_loss"], color="tab:blue", alpha=0.25)
        ax.plot(hist["val_f1_macro"], color="tab:orange", alpha=0.25)
    ax.set_xlabel("Epoch")
    ax.set_title("Training loss and validation macro F1 per fold")
    ax.plot([], [], color="tab:blue", label="train loss")
    ax.plot([], [], color="tab:orange", label="val macro F1")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def train_fold(
    data: SequenceData,
    split: dict[str, Any],
    cfg: TrainConfig,
    model_cfg: TransformerConfig,
    device: torch.device,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[float]], np.ndarray]:
    fold = int(split["fold"])
    tr = np.asarray(split["train_idx"], dtype=np.int64)
    va = np.asarray(split["val_idx"], dtype=np.int64)
    te = np.asarray(split["test_idx"], dtype=np.int64)
    if tr.size == 0 or va.size == 0 or te.size == 0:
        raise RuntimeError(f"Fold {fold} has an empty split.")

    scaler = ChannelScaler.fit(data.X[tr])
    X_tr = scaler.transform(data.X[tr])
    X_va = scaler.transform(data.X[va])
    X_te = scaler.transform(data.X[te])

    n_classes = len(data.class_names)
    train_loader = make_loader(X_tr, data.y[tr], cfg.batch_size, True, cfg.num_workers, cfg.weighted_sampler)
    val_loader = make_loader(X_va, data.y[va], cfg.batch_size, False, cfg.num_workers)
    test_loader = make_loader(X_te, data.y[te], cfg.batch_size, False, cfg.num_workers)

    model = AimTransformerEncoder(model_cfg).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(data.y[tr], n_classes, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = 0.5
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": [], "val_f1_macro": [], "val_threshold": []}

    for epoch in range(cfg.epochs):
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer, grad_scaler, amp_enabled, cfg.grad_clip)
        val_loss, y_val, p_val = run_epoch(model, val_loader, loss_fn, device, None, None, amp_enabled, cfg.grad_clip)
        threshold = 0.5
        if n_classes == 2:
            threshold, _ = tune_threshold(y_val, p_val)
        val_pred = predict_from_probs(p_val, n_classes, threshold)
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1_macro"].append(val_f1)
        history["val_threshold"].append(threshold)
        print(
            f"fold {fold:02d} epoch {epoch + 1:03d}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f} threshold={threshold:.2f}"
        )
        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            best_threshold = threshold
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= cfg.patience:
            print(f"fold {fold:02d} early stopping after {epoch + 1} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, y_test, p_test = run_epoch(model, test_loader, loss_fn, device, None, None, amp_enabled, cfg.grad_clip)
    default_metrics = fold_metrics(y_test, p_test, data.class_names, threshold=0.5)
    tuned_metrics = fold_metrics(y_test, p_test, data.class_names, threshold=best_threshold)
    primary_pred = predict_from_probs(p_test, n_classes, best_threshold if n_classes == 2 else 0.5)
    default_pred = predict_from_probs(p_test, n_classes, 0.5)

    rows: list[dict[str, Any]] = []
    for local_i, global_i in enumerate(te.tolist()):
        row = dict(data.metadata[global_i])
        row.update({
            "fold": fold,
            "true_label": int(y_test[local_i]),
            "pred_default": int(default_pred[local_i]),
            "pred_tuned": int(primary_pred[local_i]),
            "threshold_tuned": best_threshold,
        })
        for class_idx, class_name in enumerate(data.class_names):
            row[f"prob_{class_name}"] = float(p_test[local_i, class_idx])
        rows.append(row)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fold": fold,
            "held_out_participant": split["held_out_participant"],
            "model_config": asdict(model_cfg),
            "train_config": asdict(cfg),
            "feature_names": data.feature_names,
            "class_names": data.class_names,
            "scaler_mean": scaler.mean.tolist(),
            "scaler_std": scaler.std.tolist(),
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "best_val_f1_macro": best_f1,
            "best_threshold": best_threshold,
        },
        ckpt_dir / f"fold_{fold:02d}_{split['held_out_participant']}.pt",
    )

    fold_summary = {
        "fold": fold,
        "held_out_participant": split["held_out_participant"],
        "validation_participant": split["validation_participant"],
        "n_train": int(tr.size),
        "n_val": int(va.size),
        "n_test": int(te.size),
        "class_counts_train": dict(Counter(data.y[tr].tolist())),
        "class_counts_val": dict(Counter(data.y[va].tolist())),
        "class_counts_test": dict(Counter(data.y[te].tolist())),
        "best_val_f1_macro": best_f1,
        "default_threshold": 0.5,
        "tuned_threshold": best_threshold,
        "metrics_default_threshold": default_metrics,
        "metrics_tuned_threshold": tuned_metrics,
    }
    return fold_summary, rows, history, np.asarray(tuned_metrics["confusion_matrix"], dtype=np.int64)


def aggregate_metrics(folds: list[dict[str, Any]], key: str) -> dict[str, float]:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc",
        "roc_auc_ovr_macro",
        "human_recall",
        "aimbot_recall",
    ]
    out: dict[str, float] = {}
    for name in metric_names:
        vals = [float(fold[key].get(name)) for fold in folds if name in fold[key] and np.isfinite(float(fold[key].get(name)))]
        if vals:
            out[f"{name}_mean"] = float(np.mean(vals))
            out[f"{name}_std"] = float(np.std(vals))
    return out


def copy_preprocess_report(data_path: Path, out_dir: Path) -> None:
    name = data_path.name
    if name.startswith("windows_") and name.endswith(".npz"):
        config_name = name[len("windows_") : -len(".npz")]
        src = data_path.parent / f"preprocess_report_{config_name}.json"
        if src.exists():
            shutil.copyfile(src, out_dir / "preprocessing_report.json")


def run_training(cfg: TrainConfig) -> dict[str, Any]:
    seed_everything(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_sequence_data(Path(cfg.data), Path(cfg.metadata), cfg.task)
    print(data.summary())
    device = resolve_device(cfg.device, cfg.require_cuda)

    if cfg.splitter != "lopo":
        raise SystemExit("Only --splitter lopo is currently implemented.")
    splits = make_lopo_splits(data.participants, data.sessions, seed=cfg.seed)
    save_splits(splits, out_dir / "fold_splits.json")

    model_cfg = TransformerConfig(
        n_channels=data.X.shape[2],
        n_classes=len(data.class_names),
        seq_len=data.X.shape[1],
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
        conv_kernel=cfg.conv_kernel,
        conv_stride=cfg.conv_stride,
    )
    model_preview = AimTransformerEncoder(model_cfg)
    print(f"Model parameters: {model_preview.n_params():,}")
    del model_preview

    all_rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    histories: list[dict[str, list[float]]] = []
    aggregate_cm = np.zeros((len(data.class_names), len(data.class_names)), dtype=np.int64)

    for split in splits:
        fold_summary, pred_rows, history, cm = train_fold(data, split, cfg, model_cfg, device, out_dir)
        folds.append(fold_summary)
        all_rows.extend(pred_rows)
        histories.append(history)
        aggregate_cm += cm

    write_csv(out_dir / "fold_predictions.csv", all_rows)
    included_sessions = sorted({(row["participant_id"], row["session_name"], row["condition"], row["source_path"]) for row in data.metadata})
    write_csv(
        out_dir / "included_sessions.csv",
        [
            {"participant_id": pid, "session_name": sess, "condition": cond, "source_path": path}
            for pid, sess, cond, path in included_sessions
        ],
    )
    write_csv(out_dir / "excluded_sessions.csv", [])
    copy_preprocess_report(Path(cfg.data), out_dir)

    y_true = np.asarray([int(r["true_label"]) for r in all_rows], dtype=np.int64)
    y_pred = np.asarray([int(r["pred_tuned"]) for r in all_rows], dtype=np.int64)
    (out_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=data.class_names, zero_division=0),
        encoding="utf-8",
    )
    plot_confusion(aggregate_cm, data.class_names, out_dir / "confusion_matrix.png", "Aggregate LOPO confusion matrix")
    plot_training_curves(histories, out_dir / "training_curves.png")

    metrics = {
        "task": cfg.task,
        "class_names": data.class_names,
        "feature_names": data.feature_names,
        "n_samples": int(data.X.shape[0]),
        "input_shape": list(data.X.shape),
        "participants": sorted(set(data.participants.tolist())),
        "condition_counts": dict(Counter(data.conditions.tolist())),
        "label_counts": dict(Counter(data.y.tolist())),
        "model_config": asdict(model_cfg),
        "train_config": asdict(cfg),
        "folds": folds,
        "summary_default_threshold": aggregate_metrics(folds, "metrics_default_threshold"),
        "summary_tuned_threshold": aggregate_metrics(folds, "metrics_tuned_threshold"),
        "aggregate_confusion_matrix_tuned": aggregate_cm.tolist(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    (out_dir / "config_used.json").write_text(json.dumps({"train": asdict(cfg), "model": asdict(model_cfg)}, indent=2), encoding="utf-8")
    print(f"[train] wrote metrics and predictions to {out_dir}")
    return metrics


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--task", choices=("binary_all", "binary_smooth", "binary_humanised", "multiclass"), required=True)
    parser.add_argument("--splitter", choices=("lopo",), default="lopo")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument("--conv-kernel", type=int, default=0)
    parser.add_argument("--conv-stride", type=int, default=1)
    args = parser.parse_args(argv)
    return TrainConfig(
        data=str(args.data),
        metadata=str(args.metadata),
        task=args.task,
        splitter=args.splitter,
        out_dir=str(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        require_cuda=args.require_cuda,
        amp=args.amp,
        grad_clip=args.grad_clip,
        weighted_sampler=args.weighted_sampler,
        num_workers=args.num_workers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pooling=args.pooling,
        conv_kernel=args.conv_kernel,
        conv_stride=args.conv_stride,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    run_training(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Utilities for loading preprocessed sequence windows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from analysis.labels import label_mapping, labels_for_task


@dataclass
class SequenceData:
    X: np.ndarray
    y: np.ndarray
    metadata: list[dict[str, str]]
    feature_names: list[str]
    task: str
    class_names: list[str]
    sample_ids: np.ndarray

    @property
    def participants(self) -> np.ndarray:
        return np.asarray([row["participant_id"] for row in self.metadata], dtype=object)

    @property
    def sessions(self) -> np.ndarray:
        return np.asarray([row["session_name"] for row in self.metadata], dtype=object)

    @property
    def conditions(self) -> np.ndarray:
        return np.asarray([row["condition"] for row in self.metadata], dtype=object)

    def summary(self) -> str:
        from collections import Counter

        counts = Counter(self.y.tolist())
        participants = sorted(set(self.participants.tolist()))
        return (
            f"SequenceData: X={self.X.shape}, task={self.task}, "
            f"participants={len(participants)}, class_counts={dict(counts)}"
        )


class AimSequenceDataset(Dataset):
    """Torch dataset for (sequence, class label)."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _task_class_names(task: str) -> list[str]:
    if task == "binary":
        task = "binary_all"
    if task in {"binary_all", "binary_smooth", "binary_humanised"}:
        return ["human", "aimbot"]
    if task == "multiclass":
        return ["human", "sm_native_smooth", "sm_native_humanised_high"]
    label_mapping(task)
    raise AssertionError("unreachable")


def load_sequence_data(data_path: Path, metadata_path: Path, task: str) -> SequenceData:
    """Load a ``windows_*.npz`` plus metadata CSV and filter for ``task``."""
    payload = np.load(data_path, allow_pickle=True)
    X = payload["X"].astype(np.float32)
    feature_names = [str(x) for x in payload["feature_names"].tolist()]
    rows = read_metadata(metadata_path)
    if X.shape[0] != len(rows):
        raise ValueError(f"metadata rows ({len(rows)}) do not match X rows ({X.shape[0]})")

    labels_allowed = set(labels_for_task(task))
    label_map = label_mapping(task)
    keep: list[int] = []
    y: list[int] = []
    for idx, row in enumerate(rows):
        condition = row["condition"]
        if condition not in labels_allowed:
            continue
        keep.append(idx)
        y.append(label_map[condition])

    if not keep:
        raise RuntimeError(f"No samples remain for task {task!r}.")

    keep_arr = np.asarray(keep, dtype=np.int64)
    sample_ids = payload["sample_ids"][keep_arr] if "sample_ids" in payload else keep_arr.astype(object)
    return SequenceData(
        X=X[keep_arr],
        y=np.asarray(y, dtype=np.int64),
        metadata=[rows[i] for i in keep],
        feature_names=feature_names,
        task=task,
        class_names=_task_class_names(task),
        sample_ids=sample_ids,
    )


def subset_by_indices(data: SequenceData, indices: Iterable[int]) -> SequenceData:
    idx = np.asarray(list(indices), dtype=np.int64)
    return SequenceData(
        X=data.X[idx],
        y=data.y[idx],
        metadata=[data.metadata[int(i)] for i in idx],
        feature_names=data.feature_names,
        task=data.task,
        class_names=data.class_names,
        sample_ids=data.sample_ids[idx],
    )

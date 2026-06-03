"""Assemble a feature matrix from a directory of recorded sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.features import (DEFAULT_STRIDE_TICKS, DEFAULT_WINDOW_TICKS,
                               FEATURE_NAMES, windows_from_events)
from analysis.labels import SessionMeta, discover_sessions


@dataclass
class FeatureMatrix:
    X: np.ndarray              # (n_windows, n_features)
    y: np.ndarray              # (n_windows,) binary 0/1
    label: np.ndarray          # (n_windows,) string labels (object dtype)
    participant: np.ndarray    # (n_windows,) participant IDs
    session: np.ndarray        # (n_windows,) session file stems
    feature_names: list[str]

    def __len__(self) -> int:
        return self.X.shape[0]

    def summary(self) -> str:
        from collections import Counter
        lc = Counter(self.label.tolist())
        pc = Counter(self.participant.tolist())
        return (f"FeatureMatrix: {len(self)} windows, "
                f"{self.X.shape[1]} features, "
                f"{len(pc)} participants, "
                f"label counts={dict(lc)}")


def _load_events(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_feature_matrix(sessions_dir: Path,
                         window: int = DEFAULT_WINDOW_TICKS,
                         stride: int = DEFAULT_STRIDE_TICKS,
                         metas: list[SessionMeta] | None = None) -> FeatureMatrix:
    """Walk `sessions_dir` and assemble a feature matrix.

    If `metas` is provided, those are used instead of re-scanning the directory
    — useful when the caller wants to filter (e.g. exclude pilot data).
    """
    if metas is None:
        metas = discover_sessions(sessions_dir)
    if not metas:
        raise RuntimeError(f"No recognised *_events.json files in {sessions_dir}")

    Xs: list[np.ndarray] = []
    ys: list[int] = []
    labels: list[str] = []
    pids: list[str] = []
    sess: list[str] = []

    for meta in metas:
        events = _load_events(meta.path)
        wins = windows_from_events(events, window=window, stride=stride)
        if not wins:
            # Session too short to produce a single window — skip but warn.
            print(f"[dataset] {meta.path.name}: no full window, skipped")
            continue
        stem = meta.path.name.replace("_events.json", "")
        for _, feats in wins:
            Xs.append(feats)
            ys.append(meta.binary_target)
            labels.append(meta.label)
            pids.append(meta.participant_id)
            sess.append(stem)

    if not Xs:
        raise RuntimeError("No windows produced from any session.")

    return FeatureMatrix(
        X=np.vstack(Xs),
        y=np.asarray(ys, dtype=np.int64),
        label=np.asarray(labels, dtype=object),
        participant=np.asarray(pids, dtype=object),
        session=np.asarray(sess, dtype=object),
        feature_names=list(FEATURE_NAMES),
    )

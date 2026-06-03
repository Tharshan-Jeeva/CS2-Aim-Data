"""Create Leave-One-Participant-Out splits for preprocessed windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from analysis.sequence_dataset import load_sequence_data


def make_lopo_splits(participants: np.ndarray, sessions: np.ndarray, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    unique_participants = sorted(set(str(p) for p in participants.tolist()))
    if len(unique_participants) < 2:
        raise RuntimeError("LOPO requires at least two participants.")

    splits: list[dict] = []
    for fold, test_pid in enumerate(unique_participants):
        test_idx = np.flatnonzero(participants == test_pid)
        train_val_pids = [p for p in unique_participants if p != test_pid]
        val_pid = train_val_pids[int(rng.integers(0, len(train_val_pids)))]
        val_idx = np.flatnonzero(participants == val_pid)
        train_idx = np.flatnonzero((participants != test_pid) & (participants != val_pid))

        train_pids = set(participants[train_idx].tolist())
        val_pids = set(participants[val_idx].tolist())
        test_pids = set(participants[test_idx].tolist())
        assert train_pids.isdisjoint(test_pids)
        assert val_pids.isdisjoint(test_pids)
        assert train_pids.isdisjoint(val_pids)

        splits.append({
            "fold": fold,
            "held_out_participant": test_pid,
            "validation_participant": val_pid,
            "train_participants": sorted(train_pids),
            "val_participants": sorted(val_pids),
            "test_participants": sorted(test_pids),
            "train_idx": train_idx.astype(int).tolist(),
            "val_idx": val_idx.astype(int).tolist(),
            "test_idx": test_idx.astype(int).tolist(),
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
            "n_test": int(test_idx.size),
            "train_sessions": sorted(set(sessions[train_idx].tolist())),
            "val_sessions": sorted(set(sessions[val_idx].tolist())),
            "test_sessions": sorted(set(sessions[test_idx].tolist())),
        })
    return splits


def save_splits(splits: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"splitter": "lopo", "folds": splits}, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--task", choices=("binary_all", "binary_smooth", "binary_humanised", "multiclass"), default="binary_all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("analysis/results/fold_splits.json"))
    args = parser.parse_args(argv)

    data = load_sequence_data(args.data, args.metadata, args.task)
    splits = make_lopo_splits(data.participants, data.sessions, seed=args.seed)
    save_splits(splits, args.out)
    print(f"[splits] wrote {len(splits)} folds to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

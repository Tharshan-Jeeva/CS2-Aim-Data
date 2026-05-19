"""Visual sanity check: plot yaw / pitch / angular-speed for each condition.

The classifier can only learn what's visible in the data. If the trajectories
for `human` and `bot_humanised_high` look identical on this plot, either the
bot is too good (a finding) or the telemetry is too coarse (a problem).

Usage:
    python -m analysis.plot_trajectories --sessions sessions
    python -m analysis.plot_trajectories --sessions sessions --duration 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.features import _unwrap_yaw, ticks_to_arrays
from analysis.labels import SessionMeta, discover_sessions


def _pick_one_per_label(metas: list[SessionMeta]) -> dict[str, SessionMeta]:
    """Choose the largest events file per label for plotting."""
    by_label: dict[str, list[SessionMeta]] = defaultdict(list)
    for m in metas:
        by_label[m.label].append(m)
    out: dict[str, SessionMeta] = {}
    for label, group in by_label.items():
        out[label] = max(group, key=lambda m: m.path.stat().st_size)
    return out


def _trim_to_duration(arrs: dict, duration_s: float | None) -> dict:
    if duration_s is None or arrs["t"].size == 0:
        return arrs
    t0 = arrs["t"][0]
    mask = arrs["t"] <= t0 + duration_s
    return {k: v[mask] for k, v in arrs.items()}


def plot_trajectories(sessions_dir: Path, out_path: Path,
                      duration_s: float | None = 15.0) -> None:
    metas = discover_sessions(sessions_dir)
    if not metas:
        raise SystemExit(f"No recognised sessions in {sessions_dir}")

    picks = _pick_one_per_label(metas)
    labels = sorted(picks.keys())
    n = len(labels)

    fig, axes = plt.subplots(n, 3, figsize=(14, 2.4 * n + 0.6),
                             squeeze=False, sharex="col")
    fig.suptitle("Per-condition trajectory inspection "
                 f"({duration_s or '∞'}s window)", fontsize=12)

    for row, label in enumerate(labels):
        meta = picks[label]
        with meta.path.open("r", encoding="utf-8") as f:
            events = json.load(f)
        arrs = _trim_to_duration(ticks_to_arrays(events), duration_s)
        t = arrs["t"] - (arrs["t"][0] if arrs["t"].size else 0.0)
        yaw = _unwrap_yaw(arrs["yaw"])
        pitch = arrs["pitch"]

        # Column 1: yaw vs time
        ax = axes[row, 0]
        ax.plot(t, yaw, lw=0.7)
        ax.set_ylabel(f"{label}\nyaw (°)", fontsize=8)
        ax.tick_params(labelsize=7)

        # Column 2: pitch vs time
        ax = axes[row, 1]
        ax.plot(t, pitch, lw=0.7, color="tab:orange")
        ax.set_ylabel("pitch (°)", fontsize=8)
        ax.tick_params(labelsize=7)

        # Column 3: angular speed vs time (the discriminative thing)
        if t.size > 1:
            dt = np.diff(t)
            dt = np.where(dt > 1e-6, dt, np.median(dt) if np.median(dt) > 0 else 0.01)
            ang_speed = np.sqrt((np.diff(yaw) / dt) ** 2 +
                                (np.diff(pitch) / dt) ** 2)
            ax = axes[row, 2]
            ax.plot(t[1:], ang_speed, lw=0.5, color="tab:green")
            ax.set_ylabel("|ω| (°/s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_yscale("symlog")

    for col, title in enumerate(("yaw", "pitch", "angular speed")):
        axes[0, col].set_title(title, fontsize=9)
    for col in range(3):
        axes[-1, col].set_xlabel("time (s)", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_path}  ({n} conditions)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("figures/trajectories.png"))
    ap.add_argument("--duration", type=float, default=15.0,
                    help="Trim each session to N seconds for plotting "
                         "(None = full session)")
    args = ap.parse_args(argv)
    plot_trajectories(args.sessions, args.out,
                      duration_s=args.duration if args.duration > 0 else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Visual sanity check: plot yaw / pitch / angular-speed for each condition.

The classifier can only learn what's visible in the data. If the trajectories
for `human` and `bot_humanised_high` look identical on this plot, either the
bot is too good (a finding) or the telemetry is too coarse (a problem).

Two viewing modes:

* Continuous (default) — one window of `--duration` seconds. Use
  `--from-first-fire` to anchor t=0 at the participant's first weapon_fire,
  stripping spawn / orientation time.
* Per-fire overlay — `--fire-window N` overlays one N-second slice per
  weapon_fire event, centred at the fire moment. Reveals the flick-and-settle
  signature that distinguishes aim modes; the continuous plot averages it out.

Usage:
    python -m analysis.plot_trajectories --sessions sessions
    python -m analysis.plot_trajectories --sessions sessions --from-first-fire
    python -m analysis.plot_trajectories --sessions sessions --fire-window 1.0
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


def _weapon_fire_times(events: list[dict]) -> np.ndarray:
    return np.asarray(
        [float(e["timestamp_server"]) for e in events
         if e.get("type") == "weapon_fire"],
        dtype=np.float64,
    )


def _trim_from_first_fire(arrs: dict, events: list[dict]) -> dict:
    """Drop ticks before the first weapon_fire event; no-op if none."""
    fires = _weapon_fire_times(events)
    if fires.size == 0 or arrs["t"].size == 0:
        return arrs
    first = float(fires[0])
    mask = arrs["t"] >= first
    return {k: v[mask] for k, v in arrs.items()}


def _fire_segments(arrs: dict, events: list[dict],
                   window_s: float) -> list[dict]:
    """One slice per weapon_fire, total width `window_s`, re-zeroed at fire.

    Returns segments with `t` already shifted so the fire moment is at t=0.
    """
    fires = _weapon_fire_times(events)
    if fires.size == 0 or arrs["t"].size == 0 or window_s <= 0:
        return []
    half = window_s / 2.0
    segments: list[dict] = []
    for ft in fires:
        mask = (arrs["t"] >= ft - half) & (arrs["t"] <= ft + half)
        if mask.sum() < 2:
            continue
        seg = {k: v[mask] for k, v in arrs.items()}
        seg["t"] = seg["t"] - ft
        segments.append(seg)
    return segments


def _draw_continuous(ax_yaw, ax_pitch, ax_speed, arrs: dict) -> None:
    """One continuous trajectory; t already re-zeroed by caller."""
    t = arrs["t"]
    yaw = _unwrap_yaw(arrs["yaw"])
    pitch = arrs["pitch"]
    ax_yaw.plot(t, yaw, lw=0.7)
    ax_pitch.plot(t, pitch, lw=0.7, color="tab:orange")
    if t.size > 1:
        dt = np.diff(t)
        med = np.median(dt) if np.median(dt) > 0 else 0.01
        dt = np.where(dt > 1e-6, dt, med)
        ang = np.sqrt((np.diff(yaw) / dt) ** 2 + (np.diff(pitch) / dt) ** 2)
        ax_speed.plot(t[1:], ang, lw=0.5, color="tab:green")
        ax_speed.set_yscale("symlog")


def _draw_fire_overlay(ax_yaw, ax_pitch, ax_speed,
                       segments: list[dict]) -> None:
    """Per-fire segments overlaid. Yaw re-zeroed to start of each segment
    so envelope shape (not absolute heading) is what's compared."""
    if not segments:
        return
    alpha = float(np.clip(1.0 / np.sqrt(len(segments)), 0.08, 0.4))
    for seg in segments:
        t = seg["t"]
        yaw = _unwrap_yaw(seg["yaw"])
        yaw = yaw - yaw[0]
        pitch = seg["pitch"] - seg["pitch"][0]
        ax_yaw.plot(t, yaw, lw=0.6, color="tab:blue", alpha=alpha)
        ax_pitch.plot(t, pitch, lw=0.6, color="tab:orange", alpha=alpha)
        if t.size > 1:
            dt = np.diff(t)
            med = np.median(dt) if np.median(dt) > 0 else 0.01
            dt = np.where(dt > 1e-6, dt, med)
            ang = np.sqrt((np.diff(yaw) / dt) ** 2 +
                          (np.diff(pitch) / dt) ** 2)
            ax_speed.plot(t[1:], ang, lw=0.5, color="tab:green", alpha=alpha)
    ax_speed.set_yscale("symlog")
    # Mark the fire moment.
    for ax in (ax_yaw, ax_pitch, ax_speed):
        ax.axvline(0.0, color="red", lw=0.6, ls="--", alpha=0.5)


def plot_trajectories(sessions_dir: Path, out_path: Path,
                      duration_s: float | None = 15.0,
                      from_first_fire: bool = False,
                      fire_window_s: float = 0.0) -> None:
    metas = discover_sessions(sessions_dir)
    if not metas:
        raise SystemExit(f"No recognised sessions in {sessions_dir}")

    picks = _pick_one_per_label(metas)
    labels = sorted(picks.keys())
    n = len(labels)

    overlay_mode = fire_window_s > 0
    if overlay_mode:
        title = (f"Per-fire trajectory overlay ({fire_window_s:.2f}s window, "
                 "t=0 at weapon_fire)")
    else:
        anchor = "from first fire" if from_first_fire else "from session start"
        title = (f"Per-condition trajectory inspection ("
                 f"{duration_s or '∞'}s, {anchor})")

    fig, axes = plt.subplots(n, 3, figsize=(14, 2.4 * n + 0.6),
                             squeeze=False, sharex="col")
    fig.suptitle(title, fontsize=12)

    for row, label in enumerate(labels):
        meta = picks[label]
        with meta.path.open("r", encoding="utf-8") as f:
            events = json.load(f)
        arrs = ticks_to_arrays(events)

        ax_yaw, ax_pitch, ax_speed = axes[row]
        ax_yaw.set_ylabel(f"{label}\nyaw (°)", fontsize=8)
        ax_pitch.set_ylabel("pitch (°)", fontsize=8)
        ax_speed.set_ylabel("|ω| (°/s)", fontsize=8)
        for ax in (ax_yaw, ax_pitch, ax_speed):
            ax.tick_params(labelsize=7)

        if overlay_mode:
            segments = _fire_segments(arrs, events, fire_window_s)
            _draw_fire_overlay(ax_yaw, ax_pitch, ax_speed, segments)
            ax_yaw.text(0.02, 0.92, f"n={len(segments)} fires",
                        transform=ax_yaw.transAxes, fontsize=7,
                        color="gray", va="top")
        else:
            if from_first_fire:
                arrs = _trim_from_first_fire(arrs, events)
            arrs = _trim_to_duration(arrs, duration_s)
            if arrs["t"].size:
                arrs = {k: v.copy() for k, v in arrs.items()}
                arrs["t"] = arrs["t"] - arrs["t"][0]
            _draw_continuous(ax_yaw, ax_pitch, ax_speed, arrs)

    for col, title_col in enumerate(("yaw", "pitch", "angular speed")):
        axes[0, col].set_title(title_col, fontsize=9)
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
                         "(0 or negative = full session). Ignored in "
                         "--fire-window mode.")
    ap.add_argument("--from-first-fire", action="store_true",
                    help="Anchor t=0 to the first weapon_fire in each "
                         "session, stripping spawn / orientation idle.")
    ap.add_argument("--fire-window", type=float, default=0.0,
                    help="Switch to per-fire overlay mode: plot a window "
                         "of this many seconds centred on each weapon_fire "
                         "event, overlaid translucent. Try 1.0.")
    args = ap.parse_args(argv)
    plot_trajectories(
        args.sessions, args.out,
        duration_s=args.duration if args.duration > 0 else None,
        from_first_fire=args.from_first_fire,
        fire_window_s=max(0.0, args.fire_window),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

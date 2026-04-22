"""
trajectory_extractor.py
────────────────────────
Builds labelled aim trajectory sequences from two sources:
  1. GSI JSON  — kill event timestamps and metadata
  2. Mouse CSV — raw hardware mouse input at ~1000Hz

No demo file required. CS2 offline demos are missing entity
data and cannot be parsed by awpy or demoparser2.

Output: one JSON file of trajectory sequences ready for the transformer.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# How far before each kill to capture (microseconds)
WINDOW_US = 2_000_000   # 2 seconds

# Minimum number of mouse events required to keep a trajectory
MIN_EVENTS = 20

# Quality filters — trajectories recorded while these conditions are
# true are discarded as they represent non-deliberate aim movement
FILTER_FLASHED  = True   # discard if player was flashed at kill time
FILTER_SMOKED   = True   # discard if player was smoked at kill time
FILTER_BURNING  = True   # discard if player was burning at kill time


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────────────────────

def extract_trajectories(gsi_path: str, mouse_path: str,
                          output_path: str, label: int,
                          visualise: bool = True) -> list:
    """
    Parameters
    ----------
    gsi_path    : path to _gsi.json from your session
    mouse_path  : path to _mouse_*.csv from your session
    output_path : where to save the trajectory JSON
    label       : 0 = human, 1 = aimbot
    visualise   : plot a sample of trajectories for sanity checking

    Returns
    -------
    List of trajectory dicts ready for transformer training
    """

    print(f"\n[Extractor] GSI:   {gsi_path}")
    print(f"[Extractor] Mouse: {mouse_path}")
    print(f"[Extractor] Label: {'human' if label == 0 else 'aimbot'}\n")

    # ── Load data ─────────────────────────────────────────────────────────
    gsi_df   = _load_gsi(gsi_path)
    mouse_df = _load_mouse(mouse_path)

    if gsi_df.empty:
        print("[Extractor] ERROR: GSI file empty or missing kill events.")
        return []
    if mouse_df.empty:
        print("[Extractor] ERROR: Mouse CSV empty.")
        return []

    print(f"[Extractor] GSI events:        {len(gsi_df)}")
    print(f"[Extractor] Mouse events:      {len(mouse_df)}")

    # ── Extract kill events ───────────────────────────────────────────────
    kill_events = _get_kill_events(gsi_df)
    print(f"[Extractor] Kill events found: {len(kill_events)}")

    if len(kill_events) == 0:
        print("[Extractor] ERROR: No kill events in GSI log.")
        print("[Extractor] Check that GSI config has 'player_match_stats' = 1")
        return []

    # ── Slice trajectories ────────────────────────────────────────────────
    trajectories = []
    skipped_quality  = 0
    skipped_tooshort = 0

    for i, kill in enumerate(kill_events):
        kill_ts  = kill["timestamp_us"]
        win_start = kill_ts - WINDOW_US

        # Slice raw mouse window
        window = mouse_df[
            (mouse_df["timestamp_us"] >= win_start) &
            (mouse_df["timestamp_us"] <= kill_ts)
        ].copy()

        # Skip if not enough data
        if len(window) < MIN_EVENTS:
            skipped_tooshort += 1
            continue

        # Quality filter — skip compromised aim events
        if FILTER_FLASHED and kill.get("player_flashed", 0) > 50:
            skipped_quality += 1
            continue
        if FILTER_SMOKED and kill.get("player_smoked", 0) > 0:
            skipped_quality += 1
            continue
        if FILTER_BURNING and kill.get("player_burning", 0) > 0:
            skipped_quality += 1
            continue

        # Compute features
        features = _compute_features(window)

        trajectories.append({
            "id":           i,
            "label":        label,
            "label_str":    "human" if label == 0 else "aimbot",
            "kill_ts_us":   kill_ts,
            "event_count":  len(window),
            "duration_us":  int(window["elapsed_us"].max()
                                - window["elapsed_us"].min()),

            # Scalar summary features (useful for baseline classifiers)
            "summary": {
                "mean_velocity":     float(features["velocity"].mean()),
                "max_velocity":      float(features["velocity"].max()),
                "mean_acceleration": float(features["acceleration"].abs().mean()),
                "max_acceleration":  float(features["acceleration"].abs().max()),
                "total_dx":          float(window["dx"].abs().sum()),
                "total_dy":          float(window["dy"].abs().sum()),
                "overshoot_count":   int(_count_overshoots(features)),
                "jitter_score":      float(_jitter_score(features)),
                "reaction_time_us":  int(_reaction_time(window)),
            },

            # Full sequence for transformer
            # Columns: [dx, dy, velocity, acceleration, elapsed_us_norm]
            "sequence":     features.values.tolist(),
            "seq_columns":  list(features.columns),
        })

    print(f"\n[Extractor] ── Results ──────────────────────────────────────")
    print(f"  Kill events:          {len(kill_events)}")
    print(f"  Trajectories kept:    {len(trajectories)}")
    print(f"  Skipped (too short):  {skipped_tooshort}")
    print(f"  Skipped (quality):    {skipped_quality}")
    print(f"────────────────────────────────────────────────────────────")

    # ── Save ──────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"\n[Extractor] Saved {len(trajectories)} trajectories → {output_path}")

    # ── Visualise ─────────────────────────────────────────────────────────
    if visualise and len(trajectories) > 0:
        _plot_sample(trajectories, output_path)

    return trajectories


# ─────────────────────────────────────────────────────────────────────────────
# GSI parsing
# ─────────────────────────────────────────────────────────────────────────────

def _load_gsi(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        print(f"[Extractor] WARNING: GSI file not found: {path}")
        return pd.DataFrame()

    with open(path) as f:
        events = json.load(f)

    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)
    print(f"[Extractor] GSI columns: {list(df.columns)}")
    return df


def _get_kill_events(gsi_df: pd.DataFrame) -> list:
    """
    Extract kill events from GSI log.
    A kill is detected when the 'kills' counter increments.
    Returns list of dicts with timestamp and contextual metadata.
    """
    if "kills" not in gsi_df.columns:
        print("[Extractor] WARNING: No 'kills' column in GSI data.")
        print(f"[Extractor] Available columns: {list(gsi_df.columns)}")
        return []

    gsi_df = gsi_df.sort_values("timestamp_us").reset_index(drop=True)

    kill_events = []
    prev_kills  = None

    for _, row in gsi_df.iterrows():
        current_kills = row.get("kills")
        if pd.isna(current_kills):
            continue

        current_kills = int(current_kills)

        if prev_kills is not None and current_kills > prev_kills:
            # Kill detected — capture full context at this timestamp
            kill_events.append({
                "timestamp_us":   int(row["timestamp_us"]),
                "kills_total":    current_kills,
                "player_health":  row.get("player_health"),
                "player_flashed": row.get("player_flashed", 0),
                "player_smoked":  row.get("player_smoked",  0),
                "player_burning": row.get("player_burning", 0),
                "round_phase":    row.get("round_phase"),
            })

        prev_kills = current_kills

    return kill_events


# ─────────────────────────────────────────────────────────────────────────────
# Mouse data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_mouse(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        print(f"[Extractor] WARNING: Mouse file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df.sort_values("timestamp_us").reset_index(drop=True)

    # Sanity check
    duration_s = (df["timestamp_us"].max() - df["timestamp_us"].min()) / 1e6
    rate_hz    = len(df) / duration_s if duration_s > 0 else 0
    print(f"[Extractor] Mouse duration: {duration_s:.1f}s  |  "
          f"Effective rate: {rate_hz:.0f} Hz")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _compute_features(window: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-event features from raw dx/dy stream.
    Returns a DataFrame where each row is one timestep for the transformer.
    """
    w = window.copy().reset_index(drop=True)

    # Time delta between consecutive events (microseconds)
    w["dt_us"] = w["timestamp_us"].diff().fillna(0)

    # Velocity = displacement magnitude per unit time
    # Units: raw counts per microsecond (proportional to deg/sec)
    displacement = np.sqrt(w["dx"] ** 2 + w["dy"] ** 2)
    dt_safe      = w["dt_us"].replace(0, 1)   # avoid division by zero
    w["velocity"] = displacement / dt_safe

    # Acceleration = change in velocity
    w["acceleration"] = w["velocity"].diff().fillna(0)

    # Direction angle of movement (radians, -π to π)
    w["direction"] = np.arctan2(w["dy"], w["dx"])

    # Normalise elapsed time to [0, 1] within this window
    t_min = w["elapsed_us"].min()
    t_max = w["elapsed_us"].max()
    t_range = t_max - t_min if t_max > t_min else 1
    w["time_norm"] = (w["elapsed_us"] - t_min) / t_range

    return w[[
        "dx", "dy",
        "velocity", "acceleration",
        "direction", "time_norm"
    ]].astype(float)


def _count_overshoots(features: pd.DataFrame) -> int:
    """
    Count direction reversals — a proxy for overshoot/correction behaviour.
    Humans typically overshoot a target and micro-correct; smooth aimbots don't.
    """
    dirs   = features["direction"].values
    deltas = np.diff(dirs)
    # Wrap to [-π, π]
    deltas = (deltas + np.pi) % (2 * np.pi) - np.pi
    # Count sign changes (direction reversal)
    reversals = np.sum(np.diff(np.sign(deltas)) != 0)
    return int(reversals)


def _jitter_score(features: pd.DataFrame) -> float:
    """
    Measure of micro-jitter in the trajectory.
    High jitter = human-like. Very low jitter = bot-like (too smooth).
    Computed as std deviation of acceleration.
    """
    return float(features["acceleration"].std())


def _reaction_time(window: pd.DataFrame) -> int:
    """
    Estimate reaction time as the time from the start of the window
    to the first significant mouse movement.
    'Significant' = displacement > 3 raw counts in a single event.
    """
    displacement = np.sqrt(window["dx"] ** 2 + window["dy"] ** 2)
    significant  = window[displacement > 3]
    if significant.empty:
        return int(window["elapsed_us"].max() - window["elapsed_us"].min())
    first_move_us = significant.iloc[0]["elapsed_us"]
    start_us      = window.iloc[0]["elapsed_us"]
    return int(first_move_us - start_us)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _plot_sample(trajectories: list, output_path: str, n: int = 6):
    """
    Plot dx/dy trajectory paths and velocity curves for the first n samples.
    Saves as a PNG next to the output JSON.
    """
    n      = min(n, len(trajectories))
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6))
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    for i, traj in enumerate(trajectories[:n]):
        seq = np.array(traj["sequence"])
        dx, dy   = seq[:, 0], seq[:, 1]
        velocity = seq[:, 2]

        # Cumulative path
        ax_path = axes[0][i]
        ax_path.plot(np.cumsum(dx), np.cumsum(dy),
                     linewidth=0.8, alpha=0.8)
        ax_path.scatter([0], [0], color="green", s=30, zorder=5, label="start")
        ax_path.scatter([np.cumsum(dx)[-1]], [np.cumsum(dy)[-1]],
                        color="red", s=30, zorder=5, label="kill")
        ax_path.set_title(f"#{i} path | {traj['event_count']} events",
                          fontsize=8)
        ax_path.set_xlabel("cumulative dx")
        ax_path.set_ylabel("cumulative dy")
        ax_path.legend(fontsize=6)

        # Velocity over time
        ax_vel = axes[1][i]
        ax_vel.plot(velocity, linewidth=0.8, color="darkorange")
        ax_vel.set_title(f"#{i} velocity", fontsize=8)
        ax_vel.set_xlabel("event index")
        ax_vel.set_ylabel("velocity (counts/μs)")

    label_str = trajectories[0]["label_str"]
    fig.suptitle(f"Sample trajectories — {label_str}", fontsize=12)
    plt.tight_layout()

    plot_path = str(output_path).replace(".json", "_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Extractor] Plot saved → {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Edit these to match your session ─────────────────────────────────
    SESSION    = "TEST1_human_session_1776783434"
    LABEL      = 0     # 0 = human, 1 = aimbot

    GSI_PATH   = f"sessions/{SESSION}_gsi.json"
    MOUSE_PATH = f"sessions/{SESSION}_mouse_{SESSION}.csv"
    OUT_PATH   = f"sessions/{SESSION}_trajectories.json"
    # ─────────────────────────────────────────────────────────────────────

    trajs = extract_trajectories(
        gsi_path    = GSI_PATH,
        mouse_path  = MOUSE_PATH,
        output_path = OUT_PATH,
        label       = LABEL,
        visualise   = True,
    )
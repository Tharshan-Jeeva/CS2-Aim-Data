"""Per-window feature extraction from CS:Source telemetry tick streams.

Inputs are the dicts produced by `capture.telemetry_server`. Only `tick`
events are used here — heartbeats / kills / fires are ignored for the
trajectory-classification baseline.

Output is a numpy feature matrix where each row is one fixed-length window
of consecutive ticks. The transformer model will consume the raw tick stream
directly; baselines consume these aggregated windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# CS:Source server tickrate. The capture is downsampled to whatever
# `cs_aim_telemetry.sp` emits; this is the *expected* nominal rate.
TICK_HZ = 100.0
DEFAULT_WINDOW_TICKS = 100   # 1.0 s windows
DEFAULT_STRIDE_TICKS = 50    # 50% overlap

# Names of the columns produced by `window_features` — keep in sync.
FEATURE_NAMES: list[str] = [
    "yaw_speed_mean", "yaw_speed_std", "yaw_speed_max", "yaw_speed_p95",
    "pitch_speed_mean", "pitch_speed_std", "pitch_speed_max", "pitch_speed_p95",
    "ang_speed_mean", "ang_speed_std", "ang_speed_max", "ang_speed_p95",
    "yaw_accel_mean", "yaw_accel_std", "yaw_accel_max",
    "pitch_accel_mean", "pitch_accel_std", "pitch_accel_max",
    "yaw_jerk_std", "pitch_jerk_std",
    "yaw_total_path", "pitch_total_path",
    "yaw_direction_changes", "pitch_direction_changes",
    "stationary_frac", "yaw_acf_lag1", "yaw_acf_lag5",
    "yaw_spectral_centroid_hz", "yaw_spectral_entropy",
    "settle_ratio", "player_speed_mean", "player_speed_max",
    "aim_speed_corr_player_speed",
]


@dataclass
class TrajectorySample:
    """One window's worth of aim trajectory + the label it belongs to."""
    features: np.ndarray            # shape (len(FEATURE_NAMES),)
    label: str                       # "human" / "bot_smooth" / ...
    binary_target: int               # 0 = human, 1 = bot
    participant_id: str
    session_id: str                  # filename stem
    window_start_tick: int


# ----------------------------------------------------------------------------
# Tick-stream extraction
# ----------------------------------------------------------------------------

def ticks_to_arrays(events: list[dict]) -> dict[str, np.ndarray]:
    """Extract aligned per-tick arrays from a raw event list.

    Returns dict with keys: tick, t (seconds), yaw, pitch, vx, vy, vz.
    Only events of type 'tick' contribute. View angles are stored as
    [pitch, yaw] in the telemetry, matching Source-engine convention.
    """
    ticks, ts, yaws, pitches = [], [], [], []
    vxs, vys, vzs = [], [], []
    for e in events:
        if e.get("type") != "tick":
            continue
        va = e.get("view_angles")
        if not va or len(va) < 2:
            continue
        ticks.append(int(e["tick"]))
        ts.append(float(e["timestamp_server"]))
        pitches.append(float(va[0]))
        yaws.append(float(va[1]))
        vel = e.get("velocity") or [0.0, 0.0, 0.0]
        vxs.append(float(vel[0]))
        vys.append(float(vel[1]))
        vzs.append(float(vel[2]) if len(vel) > 2 else 0.0)

    return {
        "tick": np.asarray(ticks, dtype=np.int64),
        "t":    np.asarray(ts, dtype=np.float64),
        "yaw":  np.asarray(yaws, dtype=np.float64),
        "pitch": np.asarray(pitches, dtype=np.float64),
        "vx":   np.asarray(vxs, dtype=np.float64),
        "vy":   np.asarray(vys, dtype=np.float64),
        "vz":   np.asarray(vzs, dtype=np.float64),
    }


def _unwrap_yaw(yaw: np.ndarray) -> np.ndarray:
    """Yaw is reported in degrees in [-180, 180]; unwrap so derivatives
    don't see a 360° jump when the player crosses ±180°."""
    if yaw.size == 0:
        return yaw
    return np.degrees(np.unwrap(np.radians(yaw)))


# ----------------------------------------------------------------------------
# Window-level features
# ----------------------------------------------------------------------------

def _spectral_stats(signal: np.ndarray, fs: float) -> tuple[float, float]:
    """Return (spectral_centroid_hz, spectral_entropy_normalised).

    The signal is mean-removed first so DC doesn't dominate the centroid.
    """
    if signal.size < 8:
        return 0.0, 0.0
    x = signal - signal.mean()
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    total = spec.sum()
    if total <= 1e-12:
        return 0.0, 0.0
    p = spec / total
    centroid = float((freqs * p).sum())
    # Normalised entropy in [0, 1]: H / log(N)
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    H_max = float(np.log(p.size))
    entropy = H / H_max if H_max > 0 else 0.0
    return centroid, entropy


def _acf(signal: np.ndarray, lag: int) -> float:
    """Lag-`lag` autocorrelation of `signal`. Robust to short windows."""
    n = signal.size
    if n <= lag + 1:
        return 0.0
    x = signal - signal.mean()
    denom = float((x * x).sum())
    if denom <= 1e-12:
        return 0.0
    num = float((x[:-lag] * x[lag:]).sum())
    return num / denom


def _direction_changes(diff: np.ndarray, eps: float = 1e-3) -> int:
    """Count sign flips in `diff`, ignoring near-zero noise."""
    sig = np.sign(diff)
    sig[np.abs(diff) < eps] = 0
    # Drop zeros so back-to-back positive ticks don't get split by a zero.
    sig = sig[sig != 0]
    if sig.size < 2:
        return 0
    return int(np.sum(sig[1:] != sig[:-1]))


def window_features(arrs: dict[str, np.ndarray],
                    i0: int,
                    i1: int,
                    fs: float = TICK_HZ) -> np.ndarray:
    """Compute the feature vector for samples [i0:i1] of a tick stream.

    Caller guarantees i1 - i0 == window length; we don't enforce here so the
    same routine can be reused for the synthetic and pilot data alike.
    """
    yaw   = _unwrap_yaw(arrs["yaw"][i0:i1])
    pitch = arrs["pitch"][i0:i1]
    t     = arrs["t"][i0:i1]

    if t.size < 2:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    dt = np.diff(t)
    # Guard against zero-duration ticks (shouldn't happen but the server
    # occasionally emits two ticks at the same timestamp on round restarts).
    dt = np.where(dt > 1e-6, dt, 1.0 / fs)

    dyaw  = np.diff(yaw)
    dpitch = np.diff(pitch)
    yaw_speed   = dyaw / dt          # deg/s
    pitch_speed = dpitch / dt
    ang_speed = np.sqrt(yaw_speed ** 2 + pitch_speed ** 2)

    yaw_accel = np.diff(yaw_speed) / dt[:-1] if yaw_speed.size > 1 else np.zeros(1)
    pitch_accel = np.diff(pitch_speed) / dt[:-1] if pitch_speed.size > 1 else np.zeros(1)
    yaw_jerk = np.diff(yaw_accel) / dt[:-2] if yaw_accel.size > 1 else np.zeros(1)
    pitch_jerk = np.diff(pitch_accel) / dt[:-2] if pitch_accel.size > 1 else np.zeros(1)

    # Player movement
    vx = arrs["vx"][i0:i1]
    vy = arrs["vy"][i0:i1]
    player_speed = np.sqrt(vx ** 2 + vy ** 2)

    # Aim-vs-movement coupling: do they aim faster when moving?
    if player_speed.size > 1 and ang_speed.size > 0:
        ps = player_speed[1:]  # align with diffs
        if np.std(ps) > 1e-6 and np.std(ang_speed) > 1e-6:
            corr = float(np.corrcoef(ps, ang_speed)[0, 1])
        else:
            corr = 0.0
    else:
        corr = 0.0

    # Settle ratio: variance in the last quarter vs first quarter of yaw_speed.
    if yaw_speed.size >= 8:
        q = max(2, yaw_speed.size // 4)
        var_first = float(np.var(yaw_speed[:q]))
        var_last  = float(np.var(yaw_speed[-q:]))
        settle = var_last / (var_first + 1e-6)
    else:
        settle = 1.0

    stationary_frac = float(np.mean(np.abs(yaw_speed) < 1.0))  # <1 deg/s
    yaw_centroid, yaw_entropy = _spectral_stats(yaw_speed, fs)

    feats = np.array([
        float(np.mean(np.abs(yaw_speed))),
        float(np.std(yaw_speed)),
        float(np.max(np.abs(yaw_speed))) if yaw_speed.size else 0.0,
        float(np.percentile(np.abs(yaw_speed), 95)) if yaw_speed.size else 0.0,

        float(np.mean(np.abs(pitch_speed))),
        float(np.std(pitch_speed)),
        float(np.max(np.abs(pitch_speed))) if pitch_speed.size else 0.0,
        float(np.percentile(np.abs(pitch_speed), 95)) if pitch_speed.size else 0.0,

        float(np.mean(ang_speed)),
        float(np.std(ang_speed)),
        float(np.max(ang_speed)) if ang_speed.size else 0.0,
        float(np.percentile(ang_speed, 95)) if ang_speed.size else 0.0,

        float(np.mean(np.abs(yaw_accel))),
        float(np.std(yaw_accel)),
        float(np.max(np.abs(yaw_accel))) if yaw_accel.size else 0.0,

        float(np.mean(np.abs(pitch_accel))),
        float(np.std(pitch_accel)),
        float(np.max(np.abs(pitch_accel))) if pitch_accel.size else 0.0,

        float(np.std(yaw_jerk)),
        float(np.std(pitch_jerk)),

        float(np.sum(np.abs(dyaw))),
        float(np.sum(np.abs(dpitch))),

        float(_direction_changes(dyaw)),
        float(_direction_changes(dpitch)),

        stationary_frac,
        _acf(yaw_speed, 1),
        _acf(yaw_speed, 5),

        yaw_centroid,
        yaw_entropy,

        settle,
        float(np.mean(player_speed)),
        float(np.max(player_speed)) if player_speed.size else 0.0,
        corr,
    ], dtype=np.float64)

    assert feats.size == len(FEATURE_NAMES), \
        f"feature length mismatch: {feats.size} vs {len(FEATURE_NAMES)}"
    return feats


def windows_from_events(events: list[dict],
                        window: int = DEFAULT_WINDOW_TICKS,
                        stride: int = DEFAULT_STRIDE_TICKS) -> list[tuple[int, np.ndarray]]:
    """Yield (window_start_tick, feature_vector) for one events file."""
    arrs = ticks_to_arrays(events)
    n = arrs["yaw"].size
    out: list[tuple[int, np.ndarray]] = []
    for i0 in range(0, n - window + 1, stride):
        i1 = i0 + window
        feats = window_features(arrs, i0, i1)
        out.append((int(arrs["tick"][i0]), feats))
    return out

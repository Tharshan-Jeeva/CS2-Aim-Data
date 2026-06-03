"""Preprocess CS:S telemetry into weapon-fire anchored sequence windows.

This script is intentionally separate from training. It scans participant
``events.json`` files, extracts fixed-length windows around ``weapon_fire``
events, writes model-ready arrays to ``analysis/processed``, and writes data
quality reports to ``analysis/results``.

Run from the repository root, for example:

    python -m analysis.preprocess_sequences \
      --sessions-dir sessions \
      --out-dir analysis/processed \
      --window-start -2.0 \
      --window-end 0.0 \
      --seq-len 200 \
      --feature-set aim_plus_movement \
      --config-name prefire_2s_aim_movement
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analysis.labels import (
    FINAL_STUDY_LABELS,
    MULTICLASS_LABEL_TO_ID,
    SessionMeta,
    discover_sessions,
)


AIM_FEATURES = [
    "yaw_delta",
    "pitch_delta",
    "yaw_speed",
    "pitch_speed",
    "angular_speed",
    "yaw_accel",
    "pitch_accel",
    "angular_accel",
    "yaw_jerk",
    "pitch_jerk",
]

MOVEMENT_FEATURES = [
    "player_speed",
    "horizontal_speed",
    "vertical_speed",
    "player_accel",
    "is_moving",
    "is_airborne_or_falling",
    "button_forward",
    "button_back",
    "button_left",
    "button_right",
    "button_jump",
    "button_duck",
    "button_fire",
]

TARGET_FEATURES = [
    "visible_enemy_count",
    "enemy_count",
    "closest_visible_enemy_distance",
    "closest_to_crosshair_enemy_distance",
    "target_yaw_error",
    "target_pitch_error",
    "target_angular_error",
    "target_distance",
    "target_visible",
    "target_health",
    "target_speed",
    "target_lateral_speed",
    "target_id_change",
    "target_switch_count",
]

SENSITIVITY_FEATURES = ["log_edpi"]

FEATURE_SETS = {
    "aim_only": AIM_FEATURES,
    "aim_plus_movement": AIM_FEATURES + MOVEMENT_FEATURES,
    "target_aware": AIM_FEATURES + MOVEMENT_FEATURES + TARGET_FEATURES,
    "full_context": AIM_FEATURES + MOVEMENT_FEATURES + TARGET_FEATURES,
    "sensitivity_conditioned": AIM_FEATURES + MOVEMENT_FEATURES + SENSITIVITY_FEATURES,
}


@dataclass
class PreprocessConfig:
    sessions_dir: str
    out_dir: str
    results_dir: str
    config_name: str
    window_start: float
    window_end: float
    seq_len: int
    feature_set: str
    min_coverage: float
    gap_threshold: float
    min_ticks: int
    min_duration: float
    min_tickrate: float
    max_angular_speed: float
    weapon: str | None
    require_visible_enemy_at_fire: bool
    per_session_standardised: bool
    include_old: bool


def angle_delta(from_angle: float, to_angle: float) -> float:
    """Smallest signed angle delta in degrees."""
    delta = to_angle - from_angle
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _vec3(value: Any, default: float = np.nan) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)):
        return (default, default, default)
    vals = list(value)[:3] + [default, default, default]
    return (_safe_float(vals[0], default), _safe_float(vals[1], default), _safe_float(vals[2], default))


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _schema_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, Counter] = defaultdict(Counter)
    nested: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    counts: Counter = Counter()
    for event in events:
        typ = str(event.get("type", "unknown"))
        counts[typ] += 1
        by_type[typ].update(event.keys())
        if typ == "tick":
            buttons = event.get("buttons")
            if isinstance(buttons, dict):
                nested[typ]["buttons"].update(buttons.keys())
            for enemy in event.get("enemies") or []:
                if isinstance(enemy, dict):
                    nested[typ]["enemies"].update(enemy.keys())
    out: dict[str, Any] = {"event_counts": dict(counts), "keys_by_type": {}}
    for typ, keys in by_type.items():
        out["keys_by_type"][typ] = sorted(keys)
    out["nested_keys"] = {
        typ: {name: sorted(keys) for name, keys in groups.items()}
        for typ, groups in nested.items()
    }
    return out


def _find_scalar(data: Any, candidate_keys: tuple[str, ...]) -> Any:
    if isinstance(data, dict):
        lowered = {str(k).lower(): v for k, v in data.items()}
        for key in candidate_keys:
            if key.lower() in lowered:
                return lowered[key.lower()]
        for value in data.values():
            found = _find_scalar(value, candidate_keys)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _find_scalar(value, candidate_keys)
            if found is not None:
                return found
    return None


def _metadata_for_participant(sessions_dir: Path, participant_id: str) -> dict[str, Any]:
    roots = [sessions_dir / participant_id, sessions_dir]
    payloads: list[Any] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob(f"{participant_id}*Demographics*.json")):
            try:
                payloads.append(_load_json(path))
            except Exception:
                pass
        for path in sorted(root.glob(f"{participant_id}*questionnaire*.json")):
            try:
                payloads.append(_load_json(path))
            except Exception:
                pass

    merged = {"participant_id": participant_id}
    for data in payloads:
        if isinstance(data, dict):
            merged.update(data)

    dpi = _safe_float(_find_scalar(payloads, ("mouse_dpi", "dpi")), np.nan)
    sens = _safe_float(_find_scalar(payloads, ("in_game_sensitivity", "sensitivity")), np.nan)
    m_yaw = _safe_float(_find_scalar(payloads, ("m_yaw",)), 0.022)
    m_pitch = _safe_float(_find_scalar(payloads, ("m_pitch",)), 0.022)
    edpi = dpi * sens if math.isfinite(dpi) and math.isfinite(sens) else np.nan
    cm_per_360 = np.nan
    if math.isfinite(dpi) and dpi > 0 and math.isfinite(sens) and sens > 0 and math.isfinite(m_yaw) and m_yaw > 0:
        cm_per_360 = (360.0 / (sens * m_yaw)) / dpi * 2.54

    return {
        "participant_id": participant_id,
        "dpi": dpi,
        "sensitivity": sens,
        "m_yaw": m_yaw if math.isfinite(m_yaw) else np.nan,
        "m_pitch": m_pitch if math.isfinite(m_pitch) else np.nan,
        "eDPI": edpi,
        "cm_per_360": cm_per_360,
        "hours_fps_per_week": _safe_float(merged.get("hours_fps_per_week"), np.nan),
        "self_rated_aim_skill_1to7": _safe_float(merged.get("self_rated_aim_skill_1to7"), np.nan),
    }


def _manifest_for_session(events_path: Path) -> dict[str, Any]:
    path = events_path.with_name(events_path.name.replace("_events.json", "_manifest.json"))
    if not path.exists():
        return {}
    try:
        data = _load_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {"manifest_load_error": True}


def _target_angles(src: tuple[float, float, float], dst: tuple[float, float, float]) -> tuple[float, float]:
    dx, dy, dz = dst[0] - src[0], dst[1] - src[1], dst[2] - src[2]
    yaw = math.degrees(math.atan2(dy, dx))
    hyp = math.hypot(dx, dy)
    pitch = -math.degrees(math.atan2(dz, max(hyp, 1e-9)))
    return pitch, yaw


def _enemy_position(enemy: dict[str, Any]) -> tuple[float, float, float]:
    return _vec3(enemy.get("aim_position") or enemy.get("origin") or enemy.get("position"))


def _extract_ticks(events: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    rows: list[dict[str, float]] = []
    previous: dict[str, float] | None = None
    switch_count = 0
    last_target_id: Any = None

    for event in events:
        if event.get("type") != "tick":
            continue
        view_angles = event.get("view_angles")
        if not isinstance(view_angles, (list, tuple)) or len(view_angles) < 2:
            continue
        t = _safe_float(event.get("timestamp_server"))
        tick = _safe_float(event.get("tick"))
        pitch = _safe_float(view_angles[0])
        yaw = _safe_float(view_angles[1])
        if not all(math.isfinite(x) for x in (t, tick, pitch, yaw)):
            continue

        vx, vy, vz = _vec3(event.get("velocity"), 0.0)
        buttons = event.get("buttons") if isinstance(event.get("buttons"), dict) else {}
        eye = _vec3(event.get("eye_position") or event.get("position"))
        enemies = [e for e in (event.get("enemies") or []) if isinstance(e, dict)]
        visible_enemies = [e for e in enemies if bool(e.get("visible"))]

        target_id = None
        target_visible = 0.0
        target_yaw_error = np.nan
        target_pitch_error = np.nan
        target_angular_error = np.nan
        target_distance = np.nan
        target_health = np.nan
        target_speed = np.nan
        target_lateral_speed = np.nan
        closest_visible_distance = np.nan
        closest_crosshair_distance = np.nan

        candidates = visible_enemies or enemies
        best: dict[str, Any] | None = None
        best_err = float("inf")
        closest_visible = float("inf")
        for enemy in candidates:
            pos = _enemy_position(enemy)
            if any(not math.isfinite(x) for x in (*eye, *pos)):
                continue
            epitch, eyaw = _target_angles(eye, pos)
            ye = angle_delta(yaw, eyaw)
            pe = epitch - pitch
            angular = math.hypot(ye, pe)
            dist = math.dist(eye, pos)
            if bool(enemy.get("visible")) and dist < closest_visible:
                closest_visible = dist
            if angular < best_err:
                best_err = angular
                best = enemy
                closest_crosshair_distance = dist
                target_yaw_error = ye
                target_pitch_error = pe
                target_angular_error = angular
                target_distance = dist

        if best is not None:
            target_id = best.get("id")
            target_visible = 1.0 if bool(best.get("visible")) else 0.0
            target_health = _safe_float(best.get("health"), np.nan)
            evx, evy, evz = _vec3(best.get("velocity"), 0.0)
            target_speed = math.sqrt(evx * evx + evy * evy + evz * evz)
            target_lateral_speed = math.sqrt(evx * evx + evy * evy)
        if math.isfinite(closest_visible):
            closest_visible_distance = closest_visible

        target_id_change = 0.0
        if target_id is not None:
            if last_target_id is not None and target_id != last_target_id:
                switch_count += 1
                target_id_change = 1.0
            last_target_id = target_id

        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        horiz = math.sqrt(vx * vx + vy * vy)
        row = {
            "tick": tick,
            "t": t,
            "pitch": pitch,
            "yaw": yaw,
            "player_speed": speed,
            "horizontal_speed": horiz,
            "vertical_speed": vz,
            "is_moving": 1.0 if horiz > 5.0 else 0.0,
            "is_airborne_or_falling": 1.0 if abs(vz) > 5.0 else 0.0,
            "button_forward": float(bool(buttons.get("forward"))),
            "button_back": float(bool(buttons.get("back"))),
            "button_left": float(bool(buttons.get("left"))),
            "button_right": float(bool(buttons.get("right"))),
            "button_jump": float(bool(buttons.get("jump"))),
            "button_duck": float(bool(buttons.get("duck"))),
            "button_fire": float(bool(buttons.get("fire"))),
            "visible_enemy_count": float(len(visible_enemies)),
            "enemy_count": float(len(enemies)),
            "closest_visible_enemy_distance": closest_visible_distance,
            "closest_to_crosshair_enemy_distance": closest_crosshair_distance,
            "target_yaw_error": target_yaw_error,
            "target_pitch_error": target_pitch_error,
            "target_angular_error": target_angular_error,
            "target_distance": target_distance,
            "target_visible": target_visible,
            "target_health": target_health,
            "target_speed": target_speed,
            "target_lateral_speed": target_lateral_speed,
            "target_id_change": target_id_change,
            "target_switch_count": float(switch_count),
        }

        if previous is None:
            dt = np.nan
            yaw_delta = 0.0
            pitch_delta = 0.0
            yaw_speed = 0.0
            pitch_speed = 0.0
            yaw_accel = 0.0
            pitch_accel = 0.0
            angular_accel = 0.0
            yaw_jerk = 0.0
            pitch_jerk = 0.0
            player_accel = 0.0
        else:
            dt = t - previous["t"]
            if math.isfinite(dt) and dt > 1e-6:
                yaw_delta = angle_delta(previous["yaw"], yaw)
                pitch_delta = pitch - previous["pitch"]
                yaw_speed = yaw_delta / dt
                pitch_speed = pitch_delta / dt
                last_ang_speed = previous["angular_speed"]
                yaw_accel = (yaw_speed - previous["yaw_speed"]) / dt
                pitch_accel = (pitch_speed - previous["pitch_speed"]) / dt
                angular_speed = math.hypot(yaw_speed, pitch_speed)
                angular_accel = (angular_speed - last_ang_speed) / dt
                yaw_jerk = (yaw_accel - previous["yaw_accel"]) / dt
                pitch_jerk = (pitch_accel - previous["pitch_accel"]) / dt
                player_accel = (speed - previous["player_speed"]) / dt
            else:
                yaw_delta = pitch_delta = yaw_speed = pitch_speed = 0.0
                yaw_accel = pitch_accel = angular_accel = yaw_jerk = pitch_jerk = player_accel = 0.0
        angular_speed = math.hypot(yaw_speed, pitch_speed)
        row.update({
            "dt": dt,
            "yaw_delta": yaw_delta,
            "pitch_delta": pitch_delta,
            "yaw_speed": yaw_speed,
            "pitch_speed": pitch_speed,
            "angular_speed": angular_speed,
            "yaw_accel": yaw_accel,
            "pitch_accel": pitch_accel,
            "angular_accel": angular_accel,
            "yaw_jerk": yaw_jerk,
            "pitch_jerk": pitch_jerk,
            "player_accel": player_accel,
        })
        rows.append(row)
        previous = row

    if not rows:
        return {"n": np.asarray([], dtype=np.float64)}
    keys = list(rows[0].keys())
    return {key: np.asarray([r[key] for r in rows], dtype=np.float64) for key in keys}


def _session_diagnostics(arr: dict[str, np.ndarray], events: list[dict[str, Any]]) -> dict[str, Any]:
    t = arr.get("t", np.asarray([], dtype=np.float64))
    dts = np.diff(t) if t.size > 1 else np.asarray([], dtype=np.float64)
    valid_dt = dts[dts > 0]
    duration = float(t[-1] - t[0]) if t.size > 1 else 0.0
    round_events = sum(1 for e in events if e.get("type") in {"round_start", "round_end"})
    weapon_fires = sum(1 for e in events if e.get("type") == "weapon_fire")
    return {
        "n_ticks": int(t.size),
        "duration_s": duration,
        "mean_dt": float(np.mean(valid_dt)) if valid_dt.size else np.nan,
        "median_dt": float(np.median(valid_dt)) if valid_dt.size else np.nan,
        "effective_tickrate": float(1.0 / np.median(valid_dt)) if valid_dt.size else np.nan,
        "n_gaps_gt_0_05": int(np.sum(valid_dt > 0.05)) if valid_dt.size else 0,
        "max_gap": float(np.max(valid_dt)) if valid_dt.size else np.nan,
        "max_angular_speed": float(np.nanmax(np.abs(arr.get("angular_speed", [np.nan])))),
        "weapon_fires": int(weapon_fires),
        "round_events": int(round_events),
        "event_count": int(len(events)),
    }


def _events_between(events: list[dict[str, Any]], start: float, end: float, types: set[str]) -> bool:
    for event in events:
        if event.get("type") not in types:
            continue
        ts = _safe_float(event.get("timestamp_server"))
        if math.isfinite(ts) and start <= ts <= end:
            return True
    return False


def _standardise_session_features(arr: dict[str, np.ndarray], feature_names: list[str]) -> None:
    for name in feature_names:
        if name not in arr or name.startswith("button_") or name.startswith("is_") or name in {"target_visible"}:
            continue
        values = arr[name]
        finite = np.isfinite(values)
        if np.sum(finite) < 2:
            continue
        mean = float(np.nanmean(values[finite]))
        std = float(np.nanstd(values[finite]))
        if std > 1e-6:
            arr[name] = (values - mean) / std


def _interpolate_window(
    arr: dict[str, np.ndarray],
    feature_names: list[str],
    anchor_time: float,
    window_start: float,
    window_end: float,
    seq_len: int,
    fill_values: dict[str, float],
) -> np.ndarray:
    t = arr["t"]
    rel = t - anchor_time
    grid = np.linspace(window_start, window_end, seq_len, dtype=np.float64)
    X = np.zeros((seq_len, len(feature_names)), dtype=np.float32)
    for j, name in enumerate(feature_names):
        values = arr.get(name)
        if values is None:
            X[:, j] = fill_values.get(name, 0.0)
            continue
        y = values.astype(np.float64, copy=False)
        finite = np.isfinite(y)
        if np.sum(finite) < 2:
            X[:, j] = fill_values.get(name, 0.0)
            continue
        interp = np.interp(grid, rel[finite], y[finite])
        X[:, j] = interp.astype(np.float32)
    return X


def _target_at_time(arr: dict[str, np.ndarray], anchor_time: float, name: str) -> float:
    if name not in arr or arr["t"].size == 0:
        return np.nan
    idx = int(np.argmin(np.abs(arr["t"] - anchor_time)))
    return float(arr[name][idx])


def _window_fill_values(feature_names: list[str], participant_meta: dict[str, Any]) -> dict[str, float]:
    values = {name: 0.0 for name in feature_names}
    for name in TARGET_FEATURES:
        if name in values:
            values[name] = 0.0
    for name in (
        "closest_visible_enemy_distance",
        "closest_to_crosshair_enemy_distance",
        "target_distance",
        "target_health",
        "target_speed",
        "target_lateral_speed",
        "target_yaw_error",
        "target_pitch_error",
        "target_angular_error",
    ):
        if name in values:
            values[name] = 0.0
    if "log_edpi" in values:
        edpi = _safe_float(participant_meta.get("eDPI"), np.nan)
        values["log_edpi"] = float(math.log1p(edpi)) if math.isfinite(edpi) else 0.0
    return values


def _append_sensitivity(arr: dict[str, np.ndarray], participant_meta: dict[str, Any]) -> None:
    n = arr["t"].size
    edpi = _safe_float(participant_meta.get("eDPI"), np.nan)
    value = float(math.log1p(edpi)) if math.isfinite(edpi) else 0.0
    arr["log_edpi"] = np.full(n, value, dtype=np.float64)


def _session_exclusion_reason(meta: SessionMeta, manifest: dict[str, Any], diag: dict[str, Any], cfg: PreprocessConfig) -> str | None:
    if meta.label not in FINAL_STUDY_LABELS:
        return "condition_label_not_final_study"
    flags = manifest.get("flags")
    if isinstance(flags, list) and any(str(flag).lower() in {"excluded", "exclude"} for flag in flags):
        return "manifest_marks_excluded"
    if int(diag["n_ticks"]) < cfg.min_ticks:
        return f"tick_count_too_low:{diag['n_ticks']}"
    if float(diag["duration_s"]) < cfg.min_duration:
        return f"duration_too_short:{diag['duration_s']:.3f}"
    if not math.isfinite(float(diag["effective_tickrate"])) or float(diag["effective_tickrate"]) < cfg.min_tickrate:
        return f"effective_tickrate_too_low:{diag['effective_tickrate']}"
    if int(diag["weapon_fires"]) == 0:
        return "no_weapon_fire_events"
    return None


def preprocess(cfg: PreprocessConfig) -> dict[str, Any]:
    sessions_dir = Path(cfg.sessions_dir)
    out_dir = Path(cfg.out_dir)
    results_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if cfg.feature_set not in FEATURE_SETS:
        raise SystemExit(f"Unknown feature set {cfg.feature_set!r}. Expected one of {sorted(FEATURE_SETS)}")

    excluded_dirs = () if cfg.include_old else ("OLD", "Questionnaire", "Deception", "Deception questions")
    metas = discover_sessions(
        sessions_dir,
        recursive=True,
        exclude_dirs=excluded_dirs,
        allowed_labels=set(FINAL_STUDY_LABELS),
    )

    feature_names = list(FEATURE_SETS[cfg.feature_set])
    all_X: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    schema: dict[str, Any] = {}
    session_reports: dict[str, Any] = {}
    sensitivity_by_pid: dict[str, dict[str, Any]] = {}
    windows_per_condition: dict[str, Counter] = defaultdict(Counter)
    sample_idx = 0

    for meta in metas:
        participant_meta = sensitivity_by_pid.setdefault(
            meta.participant_id,
            _metadata_for_participant(sessions_dir, meta.participant_id),
        )
        manifest = _manifest_for_session(meta.path)
        session_name = meta.path.name.replace("_events.json", "")
        try:
            events = _load_json(meta.path)
            if not isinstance(events, list):
                raise ValueError("events JSON root is not a list")
        except Exception as exc:
            excluded_rows.append({
                "path": str(meta.path),
                "participant_id": meta.participant_id,
                "condition": meta.label,
                "reason": f"unreadable_events:{exc}",
            })
            continue

        schema[session_name] = _schema_summary(events)
        arr = _extract_ticks(events)
        if "t" not in arr:
            diag = {"n_ticks": 0, "duration_s": 0.0, "weapon_fires": 0, "effective_tickrate": np.nan}
        else:
            _append_sensitivity(arr, participant_meta)
            if cfg.per_session_standardised:
                _standardise_session_features(arr, feature_names)
            diag = _session_diagnostics(arr, events)

        reason = _session_exclusion_reason(meta, manifest, diag, cfg)
        inventory_rows.append({
            "participant_id": meta.participant_id,
            "session_name": session_name,
            "condition": meta.label,
            "path": str(meta.path),
            "n_ticks": diag.get("n_ticks", 0),
            "duration_s": diag.get("duration_s", 0.0),
            "effective_tickrate": diag.get("effective_tickrate", ""),
            "weapon_fires": diag.get("weapon_fires", 0),
            "excluded": bool(reason),
            "exclusion_reason": reason or "",
        })
        if reason is not None:
            excluded_rows.append({
                "path": str(meta.path),
                "participant_id": meta.participant_id,
                "condition": meta.label,
                "reason": reason,
            })
            session_reports[session_name] = {**diag, "excluded": True, "exclusion_reason": reason}
            continue

        weapon_fires = [e for e in events if e.get("type") == "weapon_fire"]
        session_rejections = Counter()
        session_windows = 0
        t = arr["t"]
        dt = np.diff(t)
        fill_values = _window_fill_values(feature_names, participant_meta)

        for fire in weapon_fires:
            weapon = str(fire.get("weapon", "")).lower()
            if cfg.weapon and weapon != cfg.weapon.lower():
                session_rejections["weapon_filter"] += 1
                continue
            anchor_time = _safe_float(fire.get("timestamp_server"))
            if not math.isfinite(anchor_time):
                session_rejections["missing_anchor_timestamp"] += 1
                continue
            start = anchor_time + cfg.window_start
            end = anchor_time + cfg.window_end
            if end <= start:
                session_rejections["invalid_window_bounds"] += 1
                continue
            mask = (t >= start) & (t <= end)
            idx = np.flatnonzero(mask)
            expected_ticks = max(1, int(round((end - start) * 100.0)))
            coverage = float(idx.size / expected_ticks)
            if idx.size < 2 or coverage < cfg.min_coverage:
                reason = f"insufficient_tick_coverage:{coverage:.3f}"
            elif idx[0] > 0 and t[idx[0]] - t[idx[0] - 1] > cfg.gap_threshold:
                reason = "large_gap_at_window_start"
            elif idx.size > 1 and np.any(np.diff(t[idx]) <= 0):
                reason = "non_monotonic_timestamp_sequence"
            elif idx.size > 1 and np.any(np.diff(t[idx]) > cfg.gap_threshold):
                reason = "large_gap_inside_window"
            elif _events_between(events, start, end, {"round_start", "round_end"}):
                reason = "round_boundary_inside_window"
            elif np.nanmax(np.abs(arr["angular_speed"][idx])) > cfg.max_angular_speed:
                reason = "impossible_angular_spike"
            else:
                reason = None

            visible_at_fire = _target_at_time(arr, anchor_time, "target_visible")
            if reason is None and cfg.require_visible_enemy_at_fire and visible_at_fire < 0.5:
                reason = "no_visible_enemy_at_fire"
            if reason is not None:
                session_rejections[reason.split(":")[0]] += 1
                rejected_rows.append({
                    "participant_id": meta.participant_id,
                    "session_name": session_name,
                    "condition": meta.label,
                    "anchor_tick": fire.get("tick", ""),
                    "anchor_timestamp": anchor_time,
                    "weapon": weapon,
                    "reason": reason,
                })
                continue

            X = _interpolate_window(
                arr,
                feature_names,
                anchor_time,
                cfg.window_start,
                cfg.window_end,
                cfg.seq_len,
                fill_values,
            )
            if not np.all(np.isfinite(X)):
                rejected_rows.append({
                    "participant_id": meta.participant_id,
                    "session_name": session_name,
                    "condition": meta.label,
                    "anchor_tick": fire.get("tick", ""),
                    "anchor_timestamp": anchor_time,
                    "weapon": weapon,
                    "reason": "nan_or_inf_after_interpolation",
                })
                session_rejections["nan_or_inf_after_interpolation"] += 1
                continue

            sample_id = f"{cfg.config_name}_{sample_idx:06d}"
            all_X.append(X)
            metadata_rows.append({
                "sample_id": sample_id,
                "participant_id": meta.participant_id,
                "session_name": session_name,
                "condition": meta.label,
                "binary_label": 0 if meta.label == "human" else 1,
                "multiclass_label": MULTICLASS_LABEL_TO_ID[meta.label],
                "source_path": str(meta.path),
                "anchor_timestamp": anchor_time,
                "anchor_tick": fire.get("tick", ""),
                "weapon": weapon,
                "window_start": cfg.window_start,
                "window_end": cfg.window_end,
                "seq_len": cfg.seq_len,
                "feature_set": cfg.feature_set,
                "dpi": participant_meta.get("dpi", np.nan),
                "sensitivity": participant_meta.get("sensitivity", np.nan),
                "m_yaw": participant_meta.get("m_yaw", np.nan),
                "m_pitch": participant_meta.get("m_pitch", np.nan),
                "eDPI": participant_meta.get("eDPI", np.nan),
                "cm_per_360": participant_meta.get("cm_per_360", np.nan),
                "valid": True,
                "rejection_reason": "",
                "visible_enemy_at_fire": visible_at_fire,
                "target_error_at_fire": _target_at_time(arr, anchor_time, "target_angular_error"),
                "player_speed_at_fire": _target_at_time(arr, anchor_time, "player_speed"),
            })
            sample_idx += 1
            session_windows += 1
            windows_per_condition[meta.participant_id][meta.label] += 1

        session_reports[session_name] = {
            **diag,
            "excluded": False,
            "accepted_windows": int(session_windows),
            "rejected_windows": int(sum(session_rejections.values())),
            "window_rejection_reasons": dict(session_rejections),
        }

    if not all_X:
        _write_csv(results_dir / "excluded_sessions.csv", excluded_rows)
        _write_csv(results_dir / "rejected_windows.csv", rejected_rows)
        raise RuntimeError("No valid sequence windows were extracted.")

    X_arr = np.stack(all_X, axis=0).astype(np.float32)
    y_binary = np.asarray([int(row["binary_label"]) for row in metadata_rows], dtype=np.int64)
    y_multiclass = np.asarray([int(row["multiclass_label"]) for row in metadata_rows], dtype=np.int64)
    sample_ids = np.asarray([row["sample_id"] for row in metadata_rows], dtype=object)

    npz_path = out_dir / f"windows_{cfg.config_name}.npz"
    np.savez_compressed(
        npz_path,
        X=X_arr,
        y_binary=y_binary,
        y_multiclass=y_multiclass,
        sample_ids=sample_ids,
        feature_names=np.asarray(feature_names, dtype=object),
        config=json.dumps(asdict(cfg)),
    )

    metadata_path = out_dir / f"windows_{cfg.config_name}_metadata.csv"
    _write_csv(metadata_path, metadata_rows)

    report = {
        "config": asdict(cfg),
        "n_windows": int(X_arr.shape[0]),
        "shape": list(X_arr.shape),
        "feature_names": feature_names,
        "class_counts_binary": dict(Counter(y_binary.tolist())),
        "class_counts_multiclass": dict(Counter(y_multiclass.tolist())),
        "condition_counts": dict(Counter(row["condition"] for row in metadata_rows)),
        "session_reports": session_reports,
        "excluded_sessions": excluded_rows,
        "n_rejected_windows": len(rejected_rows),
    }
    _write_json(out_dir / f"preprocess_report_{cfg.config_name}.json", report)
    _write_json(results_dir / "schema_summary.json", schema)
    _write_csv(results_dir / "session_inventory.csv", inventory_rows)
    _write_csv(results_dir / "excluded_sessions.csv", excluded_rows)
    _write_csv(results_dir / "rejected_windows.csv", rejected_rows)

    sensitivity_rows: list[dict[str, Any]] = []
    for pid, meta in sorted(sensitivity_by_pid.items()):
        row = dict(meta)
        row["sessions_found"] = sum(1 for inv in inventory_rows if inv["participant_id"] == pid)
        counts = windows_per_condition.get(pid, Counter())
        row["windows_per_condition"] = json.dumps(dict(counts), sort_keys=True)
        missing = []
        if not math.isfinite(_safe_float(row.get("dpi"), np.nan)):
            missing.append("dpi")
        if not math.isfinite(_safe_float(row.get("sensitivity"), np.nan)):
            missing.append("sensitivity")
        row["notes"] = "missing:" + ",".join(missing) if missing else ""
        sensitivity_rows.append(row)
    _write_csv(results_dir / "sensitivity_report.csv", sensitivity_rows)
    summary = {
        "participants": len(sensitivity_rows),
        "missing_dpi": [r["participant_id"] for r in sensitivity_rows if "dpi" in str(r.get("notes", ""))],
        "missing_sensitivity": [r["participant_id"] for r in sensitivity_rows if "sensitivity" in str(r.get("notes", ""))],
        "expected_conditions": list(FINAL_STUDY_LABELS),
        "participants_missing_conditions": {
            pid: [c for c in FINAL_STUDY_LABELS if c not in counts]
            for pid, counts in windows_per_condition.items()
            if any(c not in counts for c in FINAL_STUDY_LABELS)
        },
    }
    _write_json(results_dir / "sensitivity_summary.json", summary)

    print(f"[preprocess] wrote {npz_path} shape={X_arr.shape}")
    print(f"[preprocess] wrote {metadata_path}")
    print(f"[preprocess] wrote reports under {results_dir}")
    return report


def _build_config(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        sessions_dir=str(args.sessions_dir),
        out_dir=str(args.out_dir),
        results_dir=str(args.results_dir),
        config_name=args.config_name,
        window_start=args.window_start,
        window_end=args.window_end,
        seq_len=args.seq_len,
        feature_set=args.feature_set,
        min_coverage=args.min_coverage,
        gap_threshold=args.gap_threshold,
        min_ticks=args.min_ticks,
        min_duration=args.min_duration,
        min_tickrate=args.min_tickrate,
        max_angular_speed=args.max_angular_speed,
        weapon=args.weapon,
        require_visible_enemy_at_fire=args.require_visible_enemy_at_fire,
        per_session_standardised=args.per_session_standardised,
        include_old=args.include_old,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions-dir", type=Path, default=Path("sessions"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("analysis/results"))
    parser.add_argument("--config-name", default="prefire_2s_aim_movement")
    parser.add_argument("--window-start", type=float, default=-2.0)
    parser.add_argument("--window-end", type=float, default=0.0)
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="aim_plus_movement")
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--gap-threshold", type=float, default=0.10)
    parser.add_argument("--min-ticks", type=int, default=100)
    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--min-tickrate", type=float, default=20.0)
    parser.add_argument("--max-angular-speed", type=float, default=5000.0)
    parser.add_argument("--weapon", default=None, help="Optional weapon filter, e.g. ak47.")
    parser.add_argument("--require-visible-enemy-at-fire", action="store_true")
    parser.add_argument("--per-session-standardised", action="store_true")
    parser.add_argument("--include-old", action="store_true", help="Include sessions in OLD folders.")
    args = parser.parse_args(argv)

    preprocess(_build_config(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())

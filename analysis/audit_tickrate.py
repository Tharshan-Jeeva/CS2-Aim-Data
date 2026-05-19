"""Audit the tick stream from each recorded session.

Reports, per session: tick count, duration, mean / median Hz, inter-tick
interval std, count of gaps > 30 ms, count of duplicate timestamps, and
whether timestamps are strictly monotonic.

A session is FLAGGED when:
  - mean Hz < 95 (expected 100)
  - more than 1% of inter-tick intervals exceed 30 ms
  - any duplicate timestamps exist
  - timestamps are non-monotonic

Run before training. If any session is flagged, exclude it (or rerecord that
condition for that participant).

Usage:
    python -m analysis.audit_tickrate --sessions sessions
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from analysis.labels import discover_sessions

EXPECTED_HZ = 100.0
MIN_OK_HZ = 95.0
GAP_THRESHOLD_S = 0.030
MAX_GAP_FRACTION = 0.01

# OnPlayerRunCmd fires at the client's cl_cmdrate. Source's stock value
# is 66 (cl_cmdrate 66, period 15.15 ms) which the audit sees as a tight
# cluster around 66 Hz. When the median lands in this window we can be
# fairly confident the participant forgot to raise cl_cmdrate.
CMDRATE_BOTTLENECK_HZ = (60.0, 72.0)


@dataclass
class AuditRow:
    session: str
    participant: str
    label: str
    n_ticks: int
    duration_s: float
    mean_hz: float
    median_hz: float
    iti_std_ms: float
    gap_count: int
    gap_fraction: float
    duplicate_ts_count: int
    monotonic: bool
    flagged: bool
    flag_reason: str


def _active_intervals(events: list[dict]) -> list[tuple[float, float]]:
    """Build `(start_ts, end_ts)` windows where the player is alive.

    Derived from `round_start` / `round_end` markers as a state machine.
    Consecutive `round_start`s without an intervening `round_end` are ignored
    (the first one wins). If the session ends mid-round, the final interval is
    closed at the last observed tick.

    Returns an empty list if no round markers exist (caller should fall back
    to the full-session path).
    """
    open_start: float | None = None
    intervals: list[tuple[float, float]] = []
    for e in events:
        etype = e.get("type")
        if etype not in ("round_start", "round_end"):
            continue
        ts = float(e["timestamp_server"])
        if etype == "round_start":
            if open_start is None:
                open_start = ts
        else:  # round_end
            if open_start is not None and ts > open_start:
                intervals.append((open_start, ts))
            open_start = None
    if open_start is not None:
        last_tick_ts = max(
            (float(e["timestamp_server"]) for e in events if e.get("type") == "tick"),
            default=open_start,
        )
        if last_tick_ts > open_start:
            intervals.append((open_start, last_tick_ts))
    return intervals


def audit_one(
    events: list[dict],
    *,
    active_only: bool = False,
) -> tuple[int, float, float, float, float, int, int, bool]:
    """Return raw stats for one session's events list.

    Uses the engine `tick` counter (authoritative, monotonic at source) for
    duplicate and monotonicity detection. `timestamp_server` is only 3-decimal
    precision so consecutive ticks can round to the same float without being
    real duplicates — counting those would produce false positives.

    When `active_only=True`, ticks falling outside `round_start`/`round_end`
    windows are excluded and inter-tick intervals are never computed across a
    round boundary. This strips legitimate idle (freeze time, respawn waits)
    from the rate metric so the audit reflects *capture quality*, not whether
    the participant was alive.
    """
    ticks_all = [(int(e["tick"]), float(e["timestamp_server"]))
                 for e in events if e.get("type") == "tick"]
    if len(ticks_all) < 2:
        return 0, 0.0, 0.0, 0.0, 0.0, 0, 0, True

    intervals = _active_intervals(events) if active_only else []
    if active_only and intervals:
        groups: list[list[tuple[int, float]]] = []
        for lo, hi in intervals:
            g = [tp for tp in ticks_all if lo <= tp[1] <= hi]
            if len(g) >= 2:
                groups.append(g)
        if not groups:
            # No ticks fell inside any active interval — degrade to full-session
            # to avoid reporting all-zeros.
            groups = [ticks_all]
    else:
        groups = [ticks_all]

    iti_parts: list[np.ndarray] = []
    tick_diff_parts: list[np.ndarray] = []
    duration = 0.0
    n_ticks = 0
    for g in groups:
        tick_arr = np.asarray([t for t, _ in g], dtype=np.int64)
        ts_arr   = np.asarray([s for _, s in g], dtype=np.float64)
        duration += float(ts_arr[-1] - ts_arr[0])
        n_ticks  += len(ts_arr)
        iti_parts.append(np.diff(ts_arr))
        tick_diff_parts.append(np.diff(tick_arr))

    iti       = np.concatenate(iti_parts)
    tick_diff = np.concatenate(tick_diff_parts)
    monotonic = bool(np.all(tick_diff >= 0))
    duplicate_count = int(np.sum(tick_diff == 0))
    nonzero = iti[iti > 0]
    mean_hz   = float(1.0 / np.mean(nonzero)) if nonzero.size else 0.0
    median_hz = float(1.0 / np.median(nonzero)) if nonzero.size else 0.0
    iti_std_ms = float(np.std(iti) * 1000.0)
    gap_count = int(np.sum(iti > GAP_THRESHOLD_S))
    return (n_ticks, duration, mean_hz, median_hz, iti_std_ms,
            gap_count, duplicate_count, monotonic)


def audit_directory(sessions_dir: Path, *, active_only: bool = False) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for meta in discover_sessions(sessions_dir):
        with meta.path.open("r", encoding="utf-8") as f:
            events = json.load(f)
        (n, dur, mhz, medhz, iti_std, gaps, dupes, mono) = audit_one(
            events, active_only=active_only
        )
        gap_frac = gaps / max(1, n - 1)
        reasons: list[str] = []
        if mhz and mhz < MIN_OK_HZ:
            lo, hi = CMDRATE_BOTTLENECK_HZ
            if lo <= medhz <= hi:
                # Specific, actionable hint rather than the generic "low Hz".
                reasons.append("cl_cmdrate~66 (raise to 100)")
            else:
                reasons.append(f"mean_hz<{MIN_OK_HZ}")
        if gap_frac > MAX_GAP_FRACTION:
            reasons.append(f"gaps>{MAX_GAP_FRACTION*100:.0f}%")
        if dupes > 0:
            reasons.append("duplicate_ts")
        if not mono:
            reasons.append("non_monotonic")
        rows.append(AuditRow(
            session=meta.path.name.replace("_events.json", ""),
            participant=meta.participant_id,
            label=meta.label,
            n_ticks=n,
            duration_s=round(dur, 2),
            mean_hz=round(mhz, 2),
            median_hz=round(medhz, 2),
            iti_std_ms=round(iti_std, 2),
            gap_count=gaps,
            gap_fraction=round(gap_frac, 4),
            duplicate_ts_count=dupes,
            monotonic=mono,
            flagged=bool(reasons),
            flag_reason=",".join(reasons) if reasons else "",
        ))
    return rows


def _print_table(rows: list[AuditRow]) -> None:
    if not rows:
        print("No sessions found.")
        return
    fmt = ("{flag:>2}  {pid:<5} {label:<28} ticks={n:>6}  dur={dur:>6.1f}s  "
           "Hz={mhz:>6.2f}/{medhz:<6.2f}  itiσ={iti:>5.2f}ms  "
           "gaps={g:>3}({gf:>5.2%})  dupes={d:>3}  "
           "{reason}")
    for r in rows:
        flag = "❌" if r.flagged else "✓"
        print(fmt.format(
            flag=flag, pid=r.participant, label=r.label, n=r.n_ticks,
            dur=r.duration_s, mhz=r.mean_hz, medhz=r.median_hz,
            iti=r.iti_std_ms, g=r.gap_count, gf=r.gap_fraction,
            d=r.duplicate_ts_count,
            reason=("[" + r.flag_reason + "]") if r.flag_reason else "",
        ))
    flagged = sum(r.flagged for r in rows)
    print(f"\n{len(rows)} sessions, {flagged} flagged.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--csv", type=Path,
                    help="Optional: also write the audit table to this CSV.")
    ap.add_argument("--active-only", action="store_true",
                    help="Compute Hz only inside round_start..round_end windows, "
                         "stripping freeze time and respawn idle from the metric.")
    args = ap.parse_args(argv)
    rows = audit_directory(args.sessions, active_only=args.active_only)
    if args.active_only:
        print("[audit] active-only mode: stats computed over alive intervals only\n")
    _print_table(rows)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(asdict(r))
        print(f"[audit] wrote {args.csv}")
    return 1 if any(r.flagged for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())

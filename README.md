# CS:Source Aim-Trajectory Data Collection Pipeline

BSc Critical Project 6 — Can transformer-based classifiers distinguish human aim
from raw, smooth, and humanised aimbot trajectories?

## Architecture

```
CS:Source dedicated server (srcds_linux + Tickrate_Enabler → 100 Hz)
    ├── Tickrate_Enabler.so              Patches the engine 66.7 Hz cap
    ├── cs_aim_telemetry.smx             HTTP POST every tick → Flask :3000
    └── cs_aim_live_controller.smx       Primary aimbot — runs in-process
                                         on OnPlayerRunCmd, zero round-trip

Python pipeline (capture/)
    ├── telemetry_server.py              Flask receiver → sort/dedupe → JSON
    ├── session_orchestrator.py          Session runner (labels, console hints)
    └── bot_profiles/                    YAML parameter sets (kept for the
                                         dissertation — documents the bot
                                         design space and the legacy
                                         Python-TCP aim path)

Data
    └── sessions/                        Per-session events JSON

Analysis (analysis/)
    ├── audit_tickrate.py                Capture-quality gate (--active-only)
    ├── plot_trajectories.py             Visual inspection (--fire-window)
    ├── features.py                      33-dim handcrafted feature extractor
    └── labels.py                        Filename → SessionMeta parser
```

The **SourceMod-native controller** (`cs_aim_live_controller.smx`) is the active
aimbot for the study. It runs inside the server's `OnPlayerRunCmd` hook, so the
aim correction lands on the same tick as the position read — no Python
round-trip latency. The legacy Python TCP loop (`cs_aim_override.smx` +
`bot_aim_generator.py`) is kept in the repo for reference and parameter
documentation but is **not** the path used for participant recordings.

## Replication

The canonical replication guide is **[SETUP.md](SETUP.md)**. It is structured
as numbered steps with a **Verify** block after each one — if your output
doesn't match the expected output, stop and fix it before continuing.

A clean run produces sessions that pass:

```bash
python -m analysis.audit_tickrate --sessions sessions --active-only
```

with no flags (mean ≥ 95 Hz, zero duplicates, zero non-monotonic).

## Quick Start (already set up)

```bash
# 1. Start server
./start_css_server.sh

# 2. Start telemetry receiver
source .venv/bin/activate
python -m capture.session_orchestrator

# 3. In-game console (per round)
sm_nativeaim_me                          // register the participant
sm_nativeaim_mode humanised_high         // pick a mode
sm_nativeaim_active 0                    // bind to MOUSE4 (see SETUP.md §8)
```

## Conditions

| Label                          | Native mode       | Notes                                                  |
|--------------------------------|-------------------|--------------------------------------------------------|
| `human`                        | (assist off)      | Subject plays manually, no assist active               |
| `sm_native_raw`                | raw               | Zero-latency instant snap                              |
| `sm_native_smooth`             | smooth            | Distance-scaled gain, linear settle                    |
| `sm_native_humanised_med`      | humanised         | Reaction delay + light jitter                          |
| `sm_native_humanised_high`     | humanised\_high   | Distance-scaled jitter, drift, stochastic overshoot    |

Legacy Python-TCP labels (`bot_raw`, `bot_smooth`, `bot_humanised_*`) remain
recognised by `session_orchestrator.py` but are not the default. Their
parameters in `capture/bot_profiles/` document the bot design space used for
the dissertation discussion.

## Repository Structure

```
sourcemod/              SourceMod plugins (.sp + .smx) and Tickrate_Enabler
capture/                Python telemetry receiver, orchestrator, bot profiles
analysis/               Audit + plot tooling (audit_tickrate, plot_trajectories)
cssource_server/        Server install (gitignored — install via SteamCMD)
start_css_server.sh     Server launch script (forces 100 Hz, restricts bots)
requirements.txt        Python dependencies
SETUP.md                Full replication guide
```

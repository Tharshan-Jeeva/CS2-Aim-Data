# CS:Source Aim-Trajectory Data Collection Pipeline

BSc Critical Project 6 — Can transformer-based classifiers distinguish human aim
from raw, smooth, and humanised aimbot trajectories?

## Architecture

```
CS:Source dedicated server (srcds_linux, 100 Hz)
    └── cs_aim_telemetry.smx   HTTP POST every tick → Flask :3000
    └── cs_aim_override.smx    TCP server :27020 ← Python angle commands

Python pipeline (capture/)
    ├── telemetry_server.py    Flask receiver → tick queue
    ├── bot_aim_generator.py   Aim computation + world-space prediction
    ├── evdev_keyboard_capture.py  Raw keyboard labels (human sessions)
    └── session_orchestrator.py    Top-level session runner

Data
    └── sessions/              Raw events JSON + keyboard CSV per session

Analysis
    └── parse/                 Trajectory extractor → labelled CSV for model
```

## Quick Start

See **[SETUP.md](SETUP.md)** for full installation instructions.

```bash
# 1. Start server
./start_css_server.sh

# 2. Start pipeline
source .venv/bin/activate
python -m capture.session_orchestrator

# 3. In-game console
sm_override_me
sm_override_active 1
```

## Bot Profiles

| Profile | Mode | Reaction | Notes |
|---|---|---|---|
| `bot_raw` | raw | 0 ms | Instant snap, no smoothing |
| `bot_smooth` | smooth | 20 ms | Linear interpolation |
| `bot_humanised_low` | humanised | 50 ms | Sigmoid, no noise |
| `bot_humanised_med` | humanised | 150 ms | Sigmoid + light jitter |
| `bot_humanised_high` | humanised | 250 ms | Sigmoid + overshoot + jitter |

## Repository Structure

```
capture/            Python data collection pipeline
sourcemod/          SourceMod plugin source (.sp) + compiled binaries (.smx)
server_config/      server.cfg template
parse/              Trajectory extraction and analysis
start_css_server.sh Server launch script (100 Hz tickrate)
requirements.txt    Python dependencies
SETUP.md            Full replication guide
```

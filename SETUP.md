# CS:Source Aim-Trajectory Pipeline — Setup Guide

Complete setup to replicate the data collection pipeline on a fresh Linux machine.

## System Requirements

- Linux (tested on Arch Linux; Ubuntu/Debian also supported)
- 32-bit library support (`lib32-glibc` on Arch, `lib32gcc-s1` on Ubuntu)
- Python 3.10+
- ~5 GB disk space for the CS:Source dedicated server

---

## 1. Install the CS:Source Dedicated Server

```bash
# Arch Linux
sudo pacman -S steamcmd

# Ubuntu / Debian
sudo apt install steamcmd

# Install CS:Source dedicated server (App ID 232330)
steamcmd +force_install_dir ~/cssource-server \
         +login anonymous \
         +app_update 232330 validate \
         +quit
```

---

## 2. Install Metamod:Source (32-bit)

Download Metamod:Source **1.11.x stable** (Linux build) from:
`https://www.sourcemm.net/downloads.php?branch=stable`

```bash
tar -xzf mmsource-*.tar.gz -C ~/cssource-server/cstrike
```

Edit `~/cssource-server/cstrike/addons/metamod.vdf` so the path points to the
32-bit binary (not `linux64`):

```
"Plugin"
{
    "file"    "../cstrike/addons/metamod/bin/server"
}
```

---

## 3. Install SourceMod

Download SourceMod **1.11.x stable** (Linux build) from:
`https://www.sourcemod.net/downloads.php?branch=stable`

```bash
tar -xzf sourcemod-*.tar.gz -C ~/cssource-server/cstrike
```

---

## 4. Install SourceMod Extensions

### SteamWorks (HTTP POST from plugins)

Download from: `https://github.com/KyleSanderson/SteamWorks/releases`

```bash
cp SteamWorks.ext.so ~/cssource-server/cstrike/addons/sourcemod/extensions/
cp SteamWorks.inc    ~/cssource-server/cstrike/addons/sourcemod/scripting/include/
```

### Socket (TCP server in override plugin)

Download from: `https://forums.alliedmods.net/showthread.php?t=67640`

```bash
cp socket.ext.so ~/cssource-server/cstrike/addons/sourcemod/extensions/
cp socket.inc    ~/cssource-server/cstrike/addons/sourcemod/scripting/include/
```

> **Important:** Older `socket.inc` files use deprecated `funcenum` syntax and
> will fail to compile. If compilation errors mention `funcenum`, replace the
> `funcenum` blocks with `typeset` blocks (see the AlliedMods thread for a
> patched version).

---

## 5. Install the Pipeline Plugins

The compiled `.smx` binaries are included in this repo under `sourcemod/`.
Copy them directly — no compilation needed unless you modify the `.sp` sources.

```bash
cp sourcemod/cs_aim_telemetry.smx \
   ~/cssource-server/cstrike/addons/sourcemod/plugins/

cp sourcemod/cs_aim_override.smx \
   ~/cssource-server/cstrike/addons/sourcemod/plugins/
```

### Recompiling from source (optional)

Only needed if you edit the `.sp` files:

```bash
cd ~/cssource-server/cstrike/addons/sourcemod/scripting
cp <repo>/sourcemod/cs_aim_telemetry.sp .
cp <repo>/sourcemod/cs_aim_override.sp .
./spcomp cs_aim_telemetry.sp  && mv cs_aim_telemetry.smx ../plugins/
./spcomp cs_aim_override.sp   && mv cs_aim_override.smx  ../plugins/
```

---

## 6. Configure the Server

Copy `server_config/server.cfg` from this repo into the server:

```bash
cp server_config/server.cfg ~/cssource-server/cstrike/cfg/server.cfg
```

Key settings already set:
- `sv_maxupdaterate 100` / `sv_minupdaterate 100` — maximum client update rate
- `bot_quota 5`, `bot_join_team CT` — 5 CT bots for the subject to fight
- `mp_freezetime 0`, `mp_roundtime 60` — no freeze, 60-second rounds

---

## 7. Python Environment

```bash
cd <repo>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Allow evdev keyboard capture (log out and back in after this)
sudo usermod -aG input $USER
```

---

## 8. Running the Pipeline

### Terminal 1 — CS:Source dedicated server

```bash
./start_css_server.sh
```

This launches `srcds_linux` with `-tickrate 100` and 5 bots on `de_dust2`.

Wait until you see:
```
[AimOverride] Listening on TCP 127.0.0.1:27020
[CS Aim Telemetry] Telemetry active — posting to http://127.0.0.1:3000/tick
```

### Terminal 2 — Python orchestrator

```bash
source .venv/bin/activate
python -m capture.session_orchestrator
```

Enter a participant ID (e.g. `P01`) and a label:

| Label | Description |
|---|---|
| `human` | Subject plays manually — keyboard captured for labels |
| `bot_raw` | Instant snap aimbot (0 ms reaction, no smoothing) |
| `bot_smooth` | Linear-interpolated smooth aimbot |
| `bot_humanised_low` | Sigmoid curve, 50 ms reaction, no overshoot |
| `bot_humanised_med` | Sigmoid curve, 150 ms reaction, light jitter |
| `bot_humanised_high` | Sigmoid curve, 250 ms reaction, 40% overshoot, strong jitter |

### Terminal 3 — CS:Source client

Connect to the server (`connect 127.0.0.1`) and run in console:

```
sm_override_me          // register yourself as the aim-override target
sm_override_active 1    // enable the aimbot (toggle with sm_override_toggle)
```

Press **Ctrl+C** in Terminal 2 to end the session. Events JSON and keyboard
CSV are saved to `sessions/`.

---

## 9. Verify Everything Works

```
[ ] sm plugins list  →  shows "CS Aim Telemetry" and "CS Aim Override"
[ ] Server console shows "[AimOverride] Listening on TCP 127.0.0.1:27020"
[ ] Orchestrator console shows "Telemetry server started on port 3000"
[ ] groups $USER  →  includes "input"  (keyboard capture)
[ ] After connecting: "[BotAim] Connected to override at 127.0.0.1:27020"
[ ] sm_override_active 1 + look at enemy  →  crosshair snaps to head
```

---

## 10. Bot Profile Tuning Parameters

All profiles live in `capture/bot_profiles/`. Key parameters:

| Parameter | Description |
|---|---|
| `mode` | `raw` / `smooth` / `humanised` |
| `reaction_ms` | Delay before aim starts moving (ms) |
| `tracking_ms` | Time to reach target (ms, smooth/humanised) |
| `prediction_ticks` | How many measured ticks to extrapolate enemy position |
| `lateral_offset_units` | World-space head-centre calibration offset (positive = left) |
| `overshoot_prob` | Probability of overshooting past target |
| `jitter_amp_deg` | Random noise amplitude on aim (degrees) |
| `fov_deg` | Only lock onto enemies within this cone |

---

## 11. Extracting Trajectories

```bash
python -m parse.extract_trajectories --session sessions/<session_name>_events.json
```

Outputs a labelled trajectory CSV ready for model training.

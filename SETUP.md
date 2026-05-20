# Replication Guide

Step-by-step setup to reproduce the CS:Source aim-trajectory dataset on a fresh
Linux machine. Every step has a **verify** block — if your output doesn't match,
stop and fix it before continuing.

The pipeline as documented produces sessions that:

- capture at **100 Hz** (≥ 95 Hz mean once idle / freeze time is excluded);
- contain **zero duplicate ticks** and **zero non-monotonic samples**;
- pass `python -m analysis.audit_tickrate --sessions sessions --active-only`
  with no flags on a clean run.

If your audit doesn't look like that, the data isn't comparable to the
dissertation results.

---

## Contents

1. [System requirements](#1-system-requirements)
2. [Clone the repo and create the Python env](#2-clone-the-repo-and-create-the-python-env)
3. [Install the CS:Source dedicated server](#3-install-the-cssource-dedicated-server)
4. [Install Metamod:Source](#4-install-metamodsource)
5. [Install SourceMod](#5-install-sourcemod)
6. [Install the SteamWorks extension](#6-install-the-steamworks-extension-http-from-plugins)
7. [Install Tickrate Enabler — required for 100 Hz](#7-install-tickrate-enabler--required-for-100hz)
8. [Deploy the pipeline plugins](#8-deploy-the-pipeline-plugins)
9. [Server config](#9-server-config)
10. [Client config (the participant's machine)](#10-client-config-the-participants-machine)
11. [Smoke test — run one session](#11-smoke-test--run-one-session)
12. [Audit the capture](#12-audit-the-capture)
13. [Inspect trajectories](#13-inspect-trajectories)
14. [Troubleshooting](#14-troubleshooting)
15. [Appendix A — Rebuilding Tickrate Enabler from source](#appendix-a--rebuilding-tickrate-enabler-from-source)
16. [Appendix B — Legacy Python-TCP aim path](#appendix-b--legacy-python-tcp-aim-path-optional)
17. [Appendix C — Bot profile parameter reference](#appendix-c--bot-profile-parameter-reference)

---

## 1. System requirements

| Item            | Tested                  | Notes                                                   |
|-----------------|-------------------------|---------------------------------------------------------|
| OS              | Arch Linux (`7.0.7-arch2-1`)        | Debian / Ubuntu also work               |
| 32-bit libs     | `lib32-glibc` (Arch) or `lib32gcc-s1` (Debian/Ubuntu) | `srcds_linux` is 32-bit |
| Python          | 3.10+                   | 3.14 confirmed working                                  |
| Disk            | ~5 GB                   | dedicated server install + sessions                     |
| Network         | LAN-only                | the pipeline binds to `127.0.0.1`                       |

A real Steam-keyboard mouse and a CRT or 144 Hz+ monitor are not required to
*replicate*, but participant data collected on different hardware should be
treated as a separate condition.

---

## 2. Clone the repo and create the Python env

```bash
git clone https://github.com/<your-fork>/CS2-Aim-Data.git
cd CS2-Aim-Data
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Verify:**

```bash
groups | tr ' ' '\n' | grep -x input    # → input
python -c "import flask, numpy, sklearn; print('ok')"   # → ok
```

---

## 3. Install the CS:Source dedicated server

Install `steamcmd` for your distribution, then:

```bash
steamcmd +force_install_dir "$PWD/cssource_server" \
         +login anonymous \
         +app_update 232330 validate \
         +quit
```

App ID `232330` is **Counter-Strike: Source Dedicated Server**. The install
target `cssource_server/` is the path the launch script expects and is
**gitignored** so it won't pollute the repo.

**Verify:**

```bash
ls cssource_server/srcds_linux
```

The binary must exist. (`server.cfg` is copied in §9 — it won't be here yet.)

---

## 4. Install Metamod:Source

Download Metamod:Source **1.11.x stable, Linux** from
<https://www.sourcemm.net/downloads.php?branch=stable>.

```bash
tar -xzf mmsource-1.11.*-linux.tar.gz -C cssource_server/cstrike
```

Edit `cssource_server/cstrike/addons/metamod.vdf` so it points at the 32-bit
binary (not `linux64`):

```
"Plugin"
{
    "file"  "../cstrike/addons/metamod/bin/server"
}
```

---

## 5. Install SourceMod

Download SourceMod **1.11.x stable, Linux** from
<https://www.sourcemod.net/downloads.php?branch=stable>.

```bash
tar -xzf sourcemod-1.11.*-linux.tar.gz -C cssource_server/cstrike
```

---

## 6. Install the SteamWorks extension (HTTP from plugins)

The telemetry plugin uses SteamWorks to POST to the local Flask receiver.

Download from <https://github.com/KyleSanderson/SteamWorks/releases> (Linux, CS:S/Source-2007 build).

```bash
cp SteamWorks.ext.so cssource_server/cstrike/addons/sourcemod/extensions/
cp SteamWorks.inc    cssource_server/cstrike/addons/sourcemod/scripting/include/
```

---

## 7. Install Tickrate Enabler — required for 100 Hz

**The single most important step.** Vanilla CS:Source srcds is hard-capped at
66.7 Hz regardless of `-tickrate 100`. Without this plugin your audit will
flag every session.

A prebuilt 32-bit Linux binary is committed in this repo (built against the
modern `ServerGameDLL012` interface — see Appendix A if you need to rebuild).

```bash
cp sourcemod/Tickrate_Enabler.so  cssource_server/cstrike/addons/
cp sourcemod/Tickrate_Enabler.vdf cssource_server/cstrike/addons/
```

**Verify (after starting the server in §11):** the console should print

```
SV_ActivateServer: setting tickrate to 100
```

If it says `66.7` instead, the plugin didn't load — check the troubleshooting
section.

---

## 8. Deploy the pipeline plugins

Both `.smx` binaries are committed under `sourcemod/`. No SourceMod compile
needed unless you modify the `.sp` sources.

```bash
cp sourcemod/cs_aim_telemetry.smx        cssource_server/cstrike/addons/sourcemod/plugins/
cp sourcemod/cs_aim_live_controller.smx  cssource_server/cstrike/addons/sourcemod/plugins/
```

`cs_aim_live_controller` is the **primary aimbot** — it runs in-process inside
`OnPlayerRunCmd`, so the aim correction lands on the same tick as the position
read (zero round-trip latency). It exposes:

| Command                                | What it does                                                       |
|----------------------------------------|--------------------------------------------------------------------|
| `sm_nativeaim_me`                      | Register the calling client as the assist target                   |
| `sm_nativeaim_mode <mode>`             | `raw` / `smooth` / `humanised` / `humanised_high`                  |
| `sm_nativeaim_active <0\|1>`           | Toggle the assist (bind to a key — see §10)                        |
| `sm_nativeaim_status`                  | Dump every ConVar value to the console (for reproducibility logs)  |

The legacy Python-TCP override (`cs_aim_override.smx`) is **not** deployed by
default — see Appendix B if you want to compare it against the native path.

---

## 9. Server config

The repo ships an audited `server.cfg`. Copy it into the install:

```bash
cp -v cssource_server/cstrike/cfg/server.cfg \
      cssource_server/cstrike/cfg/server.cfg.bak 2>/dev/null || true

# Then write the canonical study config:
cat > cssource_server/cstrike/cfg/server.cfg <<'EOF'
hostname "Aim Data Collection"

sv_lan 1
sv_pure 0
sv_maxrate 0
sv_minrate 100000
sv_maxupdaterate 100
sv_minupdaterate 100

bot_quota 5
bot_difficulty 2
bot_quota_mode normal
bot_join_after_player 1
bot_join_team CT

// Weapon restrictions for the study.
bot_allow_grenades 0   // no HE / flash / smoke from bots
bot_allow_snipers 0    // no AWP, Scout, G3SG1, SG550

mp_autoteambalance 0
mp_limitteams 0
mp_freezetime 0
mp_roundtime 60
mp_startmoney 16000
mp_buytime 9999
mp_timelimit 0
mp_round_restart_delay 0

mp_restartgame 1
EOF
```

Why these settings:

- `bot_allow_grenades 0` and `bot_allow_snipers 0` — keep engagements rifle /
  pistol / SMG only, so the aim signal isn't contaminated by smoke pollution
  or one-shot AWP kills.
- `sv_*rate` block forces the server to permit 100 Hz client rates (it would
  otherwise clamp them to its own defaults).
- `mp_freezetime 0` so participants don't sit idle between rounds — the
  `--active-only` audit handles round transitions, but less idle is better.

The launch script `start_css_server.sh` is already configured:

```bash
cat start_css_server.sh
```

It launches `srcds_run` with `-tickrate 100` plus an explicit
`+sv_maxcmdrate 128 +sv_minupdaterate 100` block so the server-side rate
clamps line up with 100 Hz client cmdrates.

---

## 10. Client config (the participant's machine)

The participant connects with a regular CS:Source client. In their console,
**before joining**, paste:

```
cl_cmdrate 100        // OnPlayerRunCmd fires at the client's cmdrate
cl_updaterate 100
rate 1000000
fps_max 300
sensitivity 2.0       // fixed across all participants
m_rawinput 1
```

Then connect to the LAN server and bind the aim toggle to MOUSE4 so it is
physically identical for every participant and every condition:

```
bind MOUSE4 +nativeaim
alias +nativeaim "sm_nativeaim_active 1"
alias -nativeaim "sm_nativeaim_active 0"
```

The hold-to-engage bind matters: in the `human` condition the participant
will press MOUSE4 too (it just does nothing), so finger movement is held
constant across conditions.

---

## 11. Smoke test — run one session

Three terminals.

### Terminal 1 — the server

```bash
./start_css_server.sh
```

Expected lines (in order):

```
Loading Tickrate_Enabler version 0.5.0...
SV_ActivateServer: setting tickrate to 100
[CS Aim Telemetry] Telemetry active — posting to http://127.0.0.1:3000/tick
[NativeAim] Loaded
```

If `setting tickrate to 100` is missing or reads `66.7`, **stop** — fix
Tickrate Enabler (§7) before going further.

### Terminal 2 — the telemetry receiver

```bash
source .venv/bin/activate
python -m capture.session_orchestrator
```

Enter:

- Participant ID: `SMOKE01`
- Label: `sm_native_humanised_high`

The orchestrator will print the exact console lines to paste into the
client to set up the bind for this condition.

### Terminal 3 — the CS:Source client

Connect (`connect 127.0.0.1`) and in console:

```
sm_nativeaim_me                          // register yourself
sm_nativeaim_mode humanised_high
sm_nativeaim_active 0
```

(Then use the MOUSE4 bind from §10 to toggle the assist on while tracking.)

Play one or two rounds, then press **Ctrl+C** in Terminal 2.

**Verify:**

```bash
ls -la sessions/ | tail -3
# → SMOKE01_sm_native_humanised_high_<unix_ts>_events.json
```

File size for a 60-second smoke test should be on the order of 1–2 MB JSON.

---

## 12. Audit the capture

This is the hard gate. If this fails the data is not comparable.

```bash
source .venv/bin/activate
python -m analysis.audit_tickrate --sessions sessions --active-only
```

**Pass output:**

```
 ✓  SMOKE01 sm_native_humanised_high  ticks= 6000  dur= 60.0s  Hz=100.00/100.00  itiσ= 0.7ms  gaps=  0(0.00%)  dupes=  0
```

The `--active-only` flag strips round freeze / respawn idle (those are real
in-game pauses, not capture loss). Without it, sessions with multiple rounds
will appear to dip below 95 Hz mean — that is expected and is **not** a
capture problem.

If anything is flagged:

| Symptom                                | Likely cause                                                  |
|----------------------------------------|---------------------------------------------------------------|
| `mean_hz<95` and median Hz ≈ 100       | Idle time leaked in (try `--active-only`).                    |
| `cl_cmdrate~66` flag                   | Client forgot `cl_cmdrate 100` — see §10.                     |
| `mean_hz<95` and median Hz ≈ 66        | Tickrate Enabler didn't load — see §7.                        |
| `duplicate_ts` > 0                     | telemetry_server.py `_sort_and_dedupe` not running — re-pull. |
| `non_monotonic`                        | Same as above — Flask appended out-of-order; should be fixed. |

---

## 13. Inspect trajectories

Quick visual sanity check that the conditions actually differ:

```bash
python -m analysis.plot_trajectories --sessions sessions \
    --fire-window 1.0 --out figures/smoke_fire_overlay.png
```

This overlays a one-second window centred on every `weapon_fire`, one row per
condition. The flick-and-settle shape should visibly differ between
`sm_native_smooth` and `sm_native_humanised_high`. If they look identical,
either the controller didn't switch modes (check `sm_nativeaim_status`) or
the participant wasn't actually pressing MOUSE4.

Other useful flags:

- `--from-first-fire --duration 30` — continuous plot, anchored at first
  trigger pull (strips the spawn plateau).
- `--duration 0` — full session.

---

## 14. Troubleshooting

**`SV_ActivateServer: setting tickrate to 66.7`** — Tickrate Enabler failed
to load. In the server console run `meta list`; if Tickrate Enabler isn't
listed, check the `.vdf` path and that the `.so` is 32-bit:
`file cssource_server/cstrike/addons/Tickrate_Enabler.so` should say
`ELF 32-bit LSB`. If it's 64-bit you have the wrong build — rebuild from
source (Appendix A).

**`Failed to get a pointer on ServerGameDLL`** — your srcds exposes a
different ServerGameDLL interface version than the prebuilt binary expects.
Rebuild from `sourcemod/tickrate_enabler_src/` after editing the version
string in `serverplugin_empty.cpp` (Appendix A).

**Telemetry rate sits at 66 Hz exactly** — the *client* is running
`cl_cmdrate 66` even though the server runs at 100. Have the participant
paste the §10 block and reconnect.

**Telemetry rate sits at ~30 Hz on a fresh install** — Flask is single-
threaded by default. The repo's `telemetry_server.py` runs in threaded mode;
make sure you haven't replaced it with a stock Flask example.

**`[BotAim] Connected to override at 127.0.0.1:27020` is missing** — that's
expected if you're using the native controller. Only the legacy Python-TCP
path (Appendix B) prints this line.

**Participant says the assist feels different in the same condition twice in
a row** — `sm_nativeaim_status` dumps every ConVar; diff two captures to
prove they were identical. ConVar values are not in `events.json`; log them
separately if you change them mid-study.

---

## Appendix A — Rebuilding Tickrate Enabler from source

The committed binary was built against `ServerGameDLL012` (modern srcds).
If your srcds is older, edit and rebuild:

```bash
cd sourcemod/tickrate_enabler_src
# See BUILD.md in this directory for the toolchain prerequisites
# (gcc-multilib, 32-bit libs, hl2sdk-css from AlliedModders, Metamod headers).
make ENGINE=css
cp Tickrate_Enabler.so ../Tickrate_Enabler.so
```

The one-line patch that distinguishes this build from upstream is documented
in `BUILD.md`: it tries `ServerGameDLL012` first and falls back to `010`, so
the same source works against both old and new srcds binaries.

---

## Appendix B — Legacy Python-TCP aim path (optional)

The original pipeline drove aim from Python over TCP. It is kept for
methodology comparison only — the native controller is the canonical path.

```bash
cp sourcemod/cs_aim_override.smx \
   cssource_server/cstrike/addons/sourcemod/plugins/
```

The orchestrator labels `bot_raw`, `bot_smooth`, `bot_humanised_low`,
`bot_humanised_med`, `bot_humanised_high` route through this path and read
their parameters from `capture/bot_profiles/*.yaml`. The trade-off is a
~2-tick round-trip latency at 100 Hz; the native controller's whole reason
for existing is to eliminate that.

---

## Appendix C — Bot profile parameter reference

The YAML profiles in `capture/bot_profiles/` document the bot design space
used for the dissertation discussion. Even when using the native controller
(which reads its parameters from `sm_nativeaim_*` ConVars, not YAML), these
files are the canonical record of "what each tier means".

| Parameter            | Description                                                       |
|----------------------|-------------------------------------------------------------------|
| `mode`               | `raw` / `smooth` / `humanised` / `humanised_high`                 |
| `reaction_ms`        | Delay before aim begins moving                                    |
| `tracking_ms`        | Time to traverse to the target (smooth / humanised)               |
| `prediction_ticks`   | Number of past ticks used to extrapolate the enemy's position     |
| `lateral_offset_units` | Head-centre calibration in world units                          |
| `overshoot_prob`     | Probability of overshooting the target on a flick                 |
| `jitter_amp_deg`     | Random-walk amplitude added to the aim signal (degrees)           |
| `fov_deg`            | Only lock on to enemies within this cone                          |

The native controller's equivalents are `sm_nativeaim_reaction_ms`,
`sm_nativeaim_smooth_gain`, `sm_nativeaim_jitter_deg`,
`sm_nativeaim_hh_overshoot_prob`, etc. — run `sm_nativeaim_status` in the
server console to dump them all.

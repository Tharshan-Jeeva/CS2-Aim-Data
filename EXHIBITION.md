# Exhibition Kiosk Mode

A self-healing "walk up and play" setup for public exhibitions. A visitor sits
down, reads the on-screen message, and plays Counter-Strike: Source while
feeling the four aim-assist styles (raw → smooth → humanised → humanised-high).
**No telemetry is recorded** — this mode is purely for show and is completely
separate from the dissertation data pipeline.

The only aimbot used is the **SM-native in-engine controller**
(`cs_aim_live_controller.smx`). The Python TCP bot path is never involved.

> **This is additive.** Nothing here changes `session_orchestrator.py`,
> `preprocess_sequences.py`, the model, the splits, or your recorded sessions.
> All new files live in `exhibition/`, `sourcemod/cs_aim_kiosk.*`, and the two
> `*_exhibition.sh` scripts. To go back to normal recording, just use
> `start_css_server.sh` + `session_orchestrator.py` as before.

---

## What it does

- **One command** brings up the dedicated server (supervised, auto-restarts on
  crash) and the game client (auto-connected to the server).
- **MOTD panel** on join explains the controls and what the project is.
- Visitor spawns **already holding an AK-47** on the T side, with **CT bots** to
  shoot and **instant respawn** — they can never get stuck on a buy menu, death
  screen, scoreboard, or map vote.
- **HOLD MOUSE4** = aim assist takes over. Release = human baseline.
- **Tap F** = cycle the assist style; the **current style is shown on screen**
  at all times, plus a chat reminder every 30 s.
- Match never ends (`mp_timelimit 0`, `mp_maxrounds 0`,
  `mp_ignore_round_win_conditions 1`), so it runs unattended indefinitely.

## Controls (the real ones, from the plugins)

| Action | Input | Under the hood |
|---|---|---|
| Engage aim assist | **HOLD MOUSE4** | `+nativeaim` → `sm_nativeaim_active 1/0` |
| Change assist style | **Tap F** | `sm_kiosk_cycle` → `sm_nativeaim_mode <style>` |
| (Styles, in order) | — | `raw → smooth → humanised → humanised_high` |

---

## Configure (optional)

Edit the block at the top of **`start_exhibition.sh`**:

```bash
MAP="de_dust2"            # map to run
SERVER_IP="127.0.0.1"     # single box → localhost
AUTO_LAUNCH_CLIENT=1      # 1 = also start + connect the game client
BOT_QUOTA=5               # number of CT bots to shoot
```

Two-box setup later? Set `SERVER_IP` to the server box's LAN IP on the **client**
machine, set `AUTO_LAUNCH_CLIENT=0` on the server box, and copy
`exhibition/cfg/autoexec_exhibition.cfg` into the play box's `cstrike/cfg/`.

## Start

```bash
cd ~/Documents/CS2-Aim-Data
./start_exhibition.sh
```

Leave that terminal open. The script:
1. compiles + deploys the kiosk plugin and copies the cfg/MOTD into the server,
2. deploys the client autoexec into your Steam CS:S install,
3. starts the supervised server, then launches + connects the game client.

**Requirements:** the CS:Source dedicated server installed under
`cssource_server/` (see `SETUP.md`), and **Steam running and logged in** if
`AUTO_LAUNCH_CLIENT=1`.

## Stop

```bash
./stop_exhibition.sh         # stop server + supervisors, leave the game open
./stop_exhibition.sh --all   # also close the CS:Source game client
```

Or press **Ctrl+C** in the launch terminal.

## Logs

Everything is timestamped under `logs/`:

| File | What |
|---|---|
| `logs/srcds.log` | Dedicated server output (restarts are marked with `=====`) |
| `logs/client.log` | Game-client launch output |
| `logs/plugin_compile.log` | Result of compiling `cs_aim_kiosk.sp` |
| `logs/exhibition.pids` | Supervisor PIDs (used by the stop script) |
| `logs/exhibition.run` | Run flag; deleting it tells the supervisors to stop |

---

## 60-second "it's frozen when I get back" triage

Work top to bottom; stop as soon as one fixes it.

1. **Is the server alive?**
   `pgrep -f srcds_linux` → nothing means it died. The supervisor should restart
   it within ~3 s; check `tail -n 40 logs/srcds.log` for the reason. If the
   supervisor itself is gone, just re-run `./start_exhibition.sh`.

2. **Is the client connected?** If the game shows the main menu, the visitor
   disconnected. The client supervisor relaunches/reconnects within ~30–40 s
   (single box). To force it: `./stop_exhibition.sh --all` then
   `./start_exhibition.sh`.

3. **No bots to shoot?** In the server console (or via `tail logs/srcds.log`)
   confirm `bot_quota 5`. If bots are missing, the kiosk respawns them on death,
   but you can nudge with `bot_add_ct` in the server console.

4. **Aim assist does nothing?** The visitor must be **alive and holding MOUSE4**,
   with an enemy inside the FOV cone and in line of sight. Confirm the plugins
   loaded: server console `sm plugins list` should show **CS Aim Live
   Controller** and **CS Aim Kiosk**.

5. **MOTD / on-screen text missing?** The text MOTD needs `cl_disablehtmlmotd 1`
   (set by the client autoexec). The on-screen style hint comes from the kiosk
   plugin — if it's absent, the kiosk plugin didn't load (see step 4) and the
   AK/respawn won't work either.

6. **Nuclear option:** `./stop_exhibition.sh --all && ./start_exhibition.sh`.
   The whole stack is stateless; a clean relaunch fixes almost everything.

---

## Files in this mode

```
start_exhibition.sh                     one-command supervised launcher
stop_exhibition.sh                      clean shutdown
EXHIBITION.md                           this file
EXHIBITION_CARD.md                      printable one-page visitor card
sourcemod/cs_aim_kiosk.sp / .smx        kiosk plugin (AK spawn, respawn, cycle, hints)
exhibition/cfg/exhibition_server.cfg    server config (never-ending match, bots, unload telemetry)
exhibition/cfg/autoexec_exhibition.cfg  client config (binds, rates, plain-text MOTD)
exhibition/motd_text.txt                plain-text MOTD (what actually shows)
exhibition/motd.txt                     HTML MOTD (backup)
```

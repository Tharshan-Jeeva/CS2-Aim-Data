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
- Visitor spawns **already holding an AK-47** (with full **armor + helmet** and a
  maxed wallet) on the T side, with **CT bots** to shoot. On death the round
  resets; every 30 deaths the map reloads — they can never get stuck on a buy
  menu, death screen, scoreboard, or map vote.
- **HOLD V** = aim assist takes over. Release = human baseline.
- **Tap F** = cycle the assist style; the **current style is shown on screen**
  at all times, plus a chat reminder every 30 s.
- Match never ends (`mp_timelimit 0`, `mp_maxrounds 0`,
  `mp_ignore_round_win_conditions 1`), so it runs unattended indefinitely.
- If a visitor wanders off, after `sm_kiosk_afk_seconds` of no input the server
  **resets (reloads the map)** so the next person gets a clean slate. (Default is
  currently **15 s for testing** — set back to ~120 for the real event.)

## Controls (the real ones, from the plugins)

| Action | Input | Under the hood |
|---|---|---|
| Engage aim assist | **HOLD V** | `+nativeaim` → `sm_nativeaim_active 1/0` |
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
   `./start_exhibition.sh`. **`Connection failed after 4 retries` to
   `127.0.0.1`?** On this host (docker bridges present) loopback connects often
   fail — connect to the **LAN IP** instead. The launcher auto-detects it from
   the IP srcds prints (`Network: IP <ip>`), but to do it by hand, read that IP
   from `logs/srcds.log` and in the client console run `connect <ip>:27015`
   (currently `10.97.72.94:27015`).

3. **No bots to shoot?** In the server console (or via `tail logs/srcds.log`)
   confirm `bot_quota 5`. If bots are missing, the kiosk respawns them on death,
   but you can nudge with `bot_add_ct` in the server console.

4. **Aim assist does nothing?** The visitor must be **alive and holding V**,
   with an enemy inside the FOV cone and in line of sight. Confirm the plugins
   loaded: server console `sm plugins list` should show **CS Aim Live
   Controller** and **CS Aim Kiosk**.

5. **MOTD / on-screen text missing?** The text MOTD needs `cl_disablehtmlmotd 1`
   (set by the client autoexec). The on-screen style hint comes from the kiosk
   plugin — if it's absent, the kiosk plugin didn't load (see step 4) and the
   AK/respawn won't work either.

6. **Nuclear option:** `./stop_exhibition.sh --all && ./start_exhibition.sh`.
   The whole stack is stateless; a clean relaunch fixes almost everything.

### Why the launcher runs srcds under `script`

srcds with `-console` **hangs at `SteamAPI_Init` when it has no controlling
terminal** (the symptom is the boot log stopping at
`[S_API FAIL] ... SteamUtils010 before SteamAPI_Init succeeded` and never
loading the map). Because the supervisor backgrounds the server, the launch
script runs it inside a `script` pseudo-TTY (util-linux), which is what lets it
finish booting. If you ever start srcds by hand in the background, wrap it the
same way or it will stall. The launcher waits for the boot-complete markers in
`logs/srcds.log` before launching the client.

An **empty** CS:Source server hibernates (you'll see `Server is hibernating`)
and the 5 CT bots don't spawn until the first client connects — there is no
`sv_hibernate_when_empty` cvar in CS:S. This is expected: the launcher
auto-connects the game client on boot, which wakes the server and spawns the
bots before any visitor arrives.

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

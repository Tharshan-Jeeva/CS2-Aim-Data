# Per-Session Recording Checklist (CS:Source / Linux)

Follow this top-to-bottom **every** session.

---

## 0. Before you sit down

- [ ] Volunteer has signed the consent form (file the signed copy)
- [ ] Consent form ID: `_______________`
- [ ] Participant ID (e.g. `P03`): `_______________`
- [ ] Label: `human` / `bot_raw` / `bot_smooth` / `bot_humanised_low` / `bot_humanised_med` / `bot_humanised_high`
- [ ] Target kill count (≥ 30 recommended): `_______________`

---

## 1. Machine prep

- [ ] Close Discord / OBS / browser / anything that polls input
- [ ] Confirm no package updates running
- [ ] Verify user is in `input` group: `groups $USER` should show `input`
- [ ] Confirm `xdotool` is installed: `which xdotool`
- [ ] CS:Source server is running with SourceMod loaded

---

## 2. Hardware / settings snapshot

- [ ] Mouse model: `_______________`
- [ ] Mouse DPI: `_______________`
- [ ] Mouse polling rate (Hz): `_______________`
- [ ] Monitor refresh rate (Hz): `_______________`
- [ ] CS:Source `sensitivity`: `_______________`
- [ ] CS:Source `zoom_sensitivity_ratio`: `_______________`
- [ ] Map: `_______________`
- [ ] Weapon restriction (if any): `_______________`

If `bot_*` label: also record bot profile name and any parameter overrides.

---

## 3. Launch order

1. [ ] CS:Source server is running with bots
2. [ ] Connect to server as participant
3. [ ] Verify SourceMod plugins loaded: `sm plugins list` in server console
4. [ ] Run: `source .venv/bin/activate && python -m capture.session_orchestrator`
5. [ ] Enter participant ID + label
6. [ ] Wait for:
   - `[Session] Telemetry server running on port 3000`
   - `[Session] Keyboard capture started`
   - (If bot session) `[Session] Bot aim generator started`
7. [ ] In CS:Source console: `sm_telemetry_target <your_userid>`
8. [ ] (If bot session) `sm_aim_override_active 1` and `sm_override_target <your_userid>`
9. [ ] In CS:Source console: `record <session_name>` (printed by orchestrator)

---

## 4. Pre-roll sanity test (≤ 30 seconds)

- [ ] Move mouse in a circle
- [ ] Tap W, A, S, D once each
- [ ] Fire one shot (LMB)
- [ ] Get one kill

Check orchestrator console output:
- [ ] Telemetry shows tick events arriving (no errors)
- [ ] Kill event printed after the kill

---

## 5. During the session

- [ ] Keep CS:Source in foreground at all times
- [ ] No alt-tabbing or pausing
- [ ] If participant takes a real break: end session and start a new one
- [ ] Note anomalies: `_______________`

---

## 6. Ending the session

1. [ ] Press Ctrl+C in the orchestrator terminal
2. [ ] In CS:Source console: `stop`
3. [ ] Move `.dem` from CS:Source demo folder to `demos/<session_name>.dem`
4. [ ] Note kill count from orchestrator output: `_______________`

---

## 7. Post-session verification

In `sessions/` you should see:
- [ ] `<session_name>_events.json` — size > 10 KB for a real session
- [ ] `<session_name>_keyboard_<session_name>.csv` — non-empty

Open events.json and confirm:
- [ ] Contains `"type": "tick"` entries
- [ ] Contains at least one `"type": "weapon_fire"` entry
- [ ] Contains at least one `"type": "kill"` entry (if kills happened)
- [ ] Tick entries have `view_angles` with changing values

---

## 8. Write the session manifest

Create `sessions/<session_name>_manifest.json`:

```json
{
  "session_name": "<session_name>",
  "participant_id": "P03",
  "consent_form_id": "...",
  "label": "human",
  "bot_profile": null,
  "mouse_model": "...",
  "mouse_dpi": 800,
  "polling_rate_hz": 1000,
  "monitor_hz": 144,
  "cs_sensitivity": 1.5,
  "zoom_sens_ratio": 1.0,
  "map": "de_dust2",
  "date_iso": "2026-05-03",
  "duration_s": null,
  "kill_count": null,
  "notes": ""
}
```

- [ ] Manifest written
- [ ] kill_count and duration_s filled in

---

## 9. Run extraction (smoke test)

```bash
source .venv/bin/activate
python -c "
from parse.Trajectory_extractor import extract_trajectories_v2
trajs = extract_trajectories_v2('sessions/<session_name>_events.json', label=0)
print(f'{len(trajs)} trajectories extracted')
if trajs:
    print(f'First trajectory: {len(trajs[0][\"timesteps\"])} timesteps')
"
```

- [ ] Confirm at least 1 trajectory extracted
- [ ] Spot check: timesteps have non-zero dyaw/dpitch values

---

## 10. Backup

- [ ] Copy session files to backup location
- [ ] Tick this session off the collection plan

---

## Common failure modes

| Symptom | Likely cause |
|---|---|
| events.json empty / tiny | SourceMod plugin not loaded, or sm_telemetry_target not set |
| Keyboard CSV empty | CS:Source not in foreground, or user not in `input` group |
| No weapon_fire events | Plugin only tracks the target player — check userid |
| Extractor finds 0 trajectories | No fire events, or session too short for window |
| Socket error in override plugin | bot_aim_generator not running, or wrong UDP port |

# gsi_server.py
from flask import Flask, request
import json
import time
import os

app = Flask(__name__)

session_name = None
event_log = []

def set_session(name):
    global session_name, event_log
    session_name = name
    event_log = []

def save_events():
    if not session_name:
        return
    path = f"sessions/{session_name}_gsi.json"
    with open(path, "w") as f:
        json.dump(event_log, f, indent=2)
    print(f"[GSI] Saved {len(event_log)} events to {path}")

@app.route("/gsi", methods=["POST"])
def gsi():
    data = request.json
    if not data:
        return "OK", 200

    # High resolution timestamp in microseconds
    ts = time.perf_counter_ns() // 1000

    event = {"timestamp_us": ts}

    # Map info
    if "map" in data:
        event["map_name"] = data["map"].get("name")
        event["map_phase"] = data["map"].get("phase")
        event["round_wins"] = data["map"].get("round_wins")

    # Round phase changes — critical for segmenting sessions
    if "round" in data:
        event["round_phase"] = data["round"].get("phase")
        event["round_bomb"] = data["round"].get("bomb")
        event["type"] = "round_event"

    # Player state
    if "player" in data:
        p = data["player"]
        state = p.get("state", {})
        event["player_health"] = state.get("health")
        event["player_armor"] = state.get("armor")
        event["player_flashed"] = state.get("flashed")
        event["player_smoked"] = state.get("smoked")
        event["player_burning"] = state.get("burning")
        event["player_money"] = state.get("money")

        # Match stats — track kills incrementally
        stats = p.get("match_stats", {})
        event["kills"] = stats.get("kills")
        event["deaths"] = stats.get("deaths")
        event["mvps"] = stats.get("mvps")
        event["score"] = stats.get("score")

        if "activity" in p:
            event["activity"] = p["activity"]
            event["type"] = "activity_change"

    # Kill detection — compare kill count to previous
    if event.get("kills") is not None and event_log:
        prev_kills = next(
            (e.get("kills") for e in reversed(event_log)
             if e.get("kills") is not None), None
        )
        if prev_kills is not None and event["kills"] > prev_kills:
            event["type"] = "kill"
            print(f"[GSI] KILL detected at {ts}us | "
                  f"Total kills: {event['kills']}")

    event_log.append(event)
    return "OK", 200


def run_gsi_server():
    # Suppress Flask's default logging noise
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(port=3000, debug=False, use_reloader=False)
import queue as _queue
from flask import Flask, request
import json
import os
import logging


def put_latest(q: _queue.Queue, item: dict) -> None:
    """Drain the queue then insert only the newest tick.

    Ensures the bot loop always acts on the freshest game state and
    never accumulates a backlog of stale frames that would appear as
    lag or overshoot.
    """
    try:
        while True:
            q.get_nowait()
    except _queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except _queue.Full:
        try:
            q.get_nowait()
        except _queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except _queue.Full:
            pass


def create_app(session_name: str, bot_queue=None):
    app = Flask(__name__)
    app.config["SESSION_NAME"] = session_name
    app.config["EVENTS"] = []
    app.config["BOT_QUEUE"] = bot_queue

    @app.route("/event", methods=["POST"])
    def event():
        data = request.json
        if not data:
            return "OK", 200

        # Always store every event — full history is required for dataset export.
        app.config["EVENTS"].append(data)

        # For live bot control only keep the newest tick; older ticks are discarded.
        if bot_queue and data.get("type") == "tick":
            put_latest(bot_queue, data)

        return "OK", 200

    return app


def get_events(app):
    return app.config["EVENTS"]


def _sort_and_dedupe(events):
    """Restore chronological order and drop duplicates.

    Flask's threaded request handler appends events in HTTP-arrival order,
    not server-time order — under load the SourceMod SteamWorks HTTP client
    can deliver a later request before an earlier one completes, producing
    non-monotonic timestamps and duplicate ticks in the captured log.

    The engine's `tick` counter (GetGameTickCount) is authoritative. We sort
    by (tick, timestamp_server, type) so events with identical ticks but
    different event types (e.g. tick + weapon_fire on the same engine tick)
    retain a stable order, and drop exact duplicate payloads.
    """
    def key(e):
        return (
            int(e.get("tick", 0)),
            float(e.get("timestamp_server", 0.0)),
            str(e.get("type", "")),
        )
    sorted_events = sorted(events, key=key)
    # Per-tick events: dedupe by (tick, type) — Source's prediction can call
    # OnPlayerRunCmd more than once per usercmd, producing two emits for the
    # same engine tick with slightly different payloads. Keep the *last* seen
    # for a given (tick, type) because that's the one the engine committed.
    # Discrete events (kill / fire / hurt / round_*) are deduped on full JSON
    # so legitimately distinct events on the same tick are preserved.
    PER_TICK_TYPES = {"tick", "heartbeat"}
    last_per_tick: dict[tuple, dict] = {}
    discrete: list[dict] = []
    seen_discrete: set[str] = set()
    for e in sorted_events:
        etype = e.get("type")
        if etype in PER_TICK_TYPES:
            last_per_tick[(int(e.get("tick", 0)), etype)] = e
        else:
            sig = json.dumps(e, sort_keys=True)
            if sig in seen_discrete:
                continue
            seen_discrete.add(sig)
            discrete.append(e)
    # Re-merge in chronological order.
    deduped = list(last_per_tick.values()) + discrete
    deduped.sort(key=key)
    return deduped


def save_events(app, output_dir="sessions"):
    session_name = app.config["SESSION_NAME"]
    raw = app.config["EVENTS"]
    events = _sort_and_dedupe(raw)
    dropped = len(raw) - len(events)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{session_name}_events.json")
    with open(path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"[Telemetry] Saved {len(events)} events to {path}"
          + (f" (dropped {dropped} duplicates / reorders)" if dropped else ""))
    return path


def run_server(app, port=3000):
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)

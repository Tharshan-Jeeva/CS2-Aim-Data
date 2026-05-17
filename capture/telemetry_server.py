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


def save_events(app, output_dir="sessions"):
    session_name = app.config["SESSION_NAME"]
    events = app.config["EVENTS"]
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{session_name}_events.json")
    with open(path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"[Telemetry] Saved {len(events)} events to {path}")
    return path


def run_server(app, port=3000):
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)

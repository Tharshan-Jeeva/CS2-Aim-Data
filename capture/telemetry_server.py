from flask import Flask, request
import json
import os


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

        app.config["EVENTS"].append(data)

        if bot_queue and data.get("type") == "tick":
            bot_queue.put_nowait(data)

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
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)

import math
import socket
import threading
import time
import queue
from collections import deque
from typing import Optional

import numpy as np
import yaml


def angle_delta(from_angle: float, to_angle: float) -> float:
    delta = to_angle - from_angle
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def angle_to_target(player_pos: list, player_angles: list, target_pos: list) -> tuple:
    dx = target_pos[0] - player_pos[0]
    dy = target_pos[1] - player_pos[1]
    dz = target_pos[2] - player_pos[2]
    dist_h = math.sqrt(dx * dx + dy * dy)

    target_yaw = math.degrees(math.atan2(dy, dx))
    target_pitch = -math.degrees(math.atan2(dz, dist_h))

    return target_yaw, target_pitch


def is_in_fov(angular_distance: float, fov_deg: float) -> bool:
    return angular_distance <= fov_deg / 2.0


def select_target(enemies: list, player_pos: list, player_angles: list,
                  priority: str) -> Optional[dict]:
    visible = [e for e in enemies if e.get("visible") and e.get("health", 0) > 0]
    if not visible:
        return None

    if priority == "nearest":
        def dist(e):
            p = e["position"]
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(player_pos, p)))
        return min(visible, key=dist)

    elif priority == "closest_to_crosshair":
        def ang_dist(e):
            ty, tp = angle_to_target(player_pos, player_angles, e["position"])
            dy = abs(angle_delta(player_angles[1], ty))
            dp = abs(angle_delta(player_angles[0], tp))
            return math.sqrt(dy * dy + dp * dp)
        return min(visible, key=ang_dist)

    elif priority == "lowest_health":
        return min(visible, key=lambda e: e["health"])

    return visible[0]


class BotAimGenerator:

    def __init__(self, config: dict, udp_host: str = "127.0.0.1",
                 udp_port: int = 27020):
        self.config = config
        self.mode = config["mode"]
        self.reaction_ms = config["reaction_ms"]
        self.tracking_ms = config["tracking_ms"]
        self.overshoot_prob = config.get("overshoot_prob", 0.0)
        self.overshoot_deg = config.get("overshoot_deg", 0.0)
        self.jitter_amp_deg = config.get("jitter_amp_deg", 0.0)
        self.fov_deg = config.get("fov_deg", 180)
        self.target_priority = config.get("target_priority", "nearest")
        self.send_rate_hz = config.get("send_rate_hz", 64)
        self.target_z_offset = config.get("target_z_offset", 0.0)
        self.prediction_ticks = config.get("prediction_ticks", 1.0)
        # World-space lateral offset applied to target position before angle computation.
        # Positive = shifts aim point LEFT (from player's perspective).
        # Scales correctly with distance — use this for head-centre calibration.
        self.lateral_offset_units = config.get("lateral_offset_units", 0.0)
        # Fixed angular fine-tune applied after geometry (normally 0.0).
        self.yaw_offset_deg = config.get("yaw_offset_deg", 0.0)

        self.udp_host = udp_host
        self.udp_port = udp_port
        self._sock = None
        self._connected = False

        self._engagement_start_ms = None
        self._engagement_start_yaw = 0.0
        self._engagement_start_pitch = 0.0
        self._current_target_id = None
        self._tick_count = 0
        self._send_count = 0
        self._last_status_ms = 0.0
        self._last_send_ms = 0.0

        # Rolling position history for smoothed velocity prediction (last 4 samples)
        self._enemy_history: dict = {}  # id -> deque[(pos, ts), ...] maxlen=4

    def compute_aim_angles(self, current_yaw: float, current_pitch: float,
                           target_yaw: float, target_pitch: float,
                           engagement_time_ms: float) -> tuple:
        if self.mode == "raw":
            return (target_yaw, target_pitch)

        elif self.mode == "smooth":
            t = min(engagement_time_ms / max(self.tracking_ms, 1), 1.0)
            yaw = current_yaw + angle_delta(current_yaw, target_yaw) * t
            pitch = current_pitch + angle_delta(current_pitch, target_pitch) * t
            return (yaw, pitch)

        elif self.mode == "humanised":
            t = min(engagement_time_ms / max(self.tracking_ms, 1), 1.0)
            s = 1.0 / (1.0 + math.exp(-12.0 * (t - 0.5)))

            yaw_delta = angle_delta(self._engagement_start_yaw, target_yaw)
            pitch_delta = angle_delta(self._engagement_start_pitch, target_pitch)

            yaw = self._engagement_start_yaw + yaw_delta * s
            pitch = self._engagement_start_pitch + pitch_delta * s

            if t > 0.85 and np.random.random() < self.overshoot_prob:
                overshoot = np.random.uniform(0, self.overshoot_deg)
                yaw += overshoot * np.sign(yaw_delta)

            if self.jitter_amp_deg > 0:
                jitter_yaw = np.random.normal(0, self.jitter_amp_deg * 0.5)
                jitter_pitch = np.random.normal(0, self.jitter_amp_deg * 0.3)
                yaw += jitter_yaw
                pitch += jitter_pitch

            return (yaw, pitch)

        return (current_yaw, current_pitch)

    def send_angles(self, yaw: float, pitch: float):
        if not self._connected:
            try:
                self._sock = socket.create_connection(
                    (self.udp_host, self.udp_port), timeout=1.0)
                self._sock.settimeout(1.0)
                self._connected = True
                print(f"[BotAim] Connected to override at {self.udp_host}:{self.udp_port}")
            except OSError as exc:
                self._print_status(f"[BotAim] Override connection failed: {exc}")
                return

        data = f"{yaw:.4f} {pitch:.4f}\n".encode("ascii")
        try:
            self._sock.sendall(data)
        except OSError as exc:
            self._print_status(f"[BotAim] Override send failed: {exc}")
            self._connected = False
            if self._sock is not None:
                self._sock.close()
                self._sock = None
            return
        self._send_count += 1

    def _print_status(self, message: str, min_interval_ms: float = 1000.0):
        now_ms = time.time() * 1000
        if now_ms - self._last_status_ms >= min_interval_ms:
            print(message)
            self._last_status_ms = now_ms

    def run(self, tick_queue: queue.Queue, stop_event: threading.Event):
        print(
            f"[BotAim] Started — mode={self.mode}, fov={self.fov_deg}, "
            f"override={self.udp_host}:{self.udp_port}")

        while not stop_event.is_set():
            try:
                tick = tick_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._tick_count += 1
            player_pos = tick.get("eye_position", tick.get("position", [0, 0, 0]))
            player_angles = tick.get("view_angles", [0, 0])
            server_ts = tick.get("timestamp_server", 0.0)
            enemies = tick.get("enemies", [])
            visible_enemies = [
                e for e in enemies if e.get("visible") and e.get("health", 0) > 0
            ]

            target = select_target(enemies, player_pos, player_angles,
                                   self.target_priority)

            if target is None:
                self._engagement_start_ms = None
                self._current_target_id = None
                self._enemy_history.clear()
                self._print_status(
                    "[BotAim] No visible target "
                    f"(ticks={self._tick_count}, enemies={len(enemies)}, "
                    f"visible={len(visible_enemies)}, sent={self._send_count})",
                    min_interval_ms=2000.0)
                continue

            # Raw position from telemetry (enemy eye position)
            raw_target_pos = list(target["position"])

            # --- Rolling-average velocity prediction ---------------------
            # Keep the last 4 (pos, ts) samples per enemy. Average the
            # velocity across consecutive pairs so single-frame noise from
            # strafes or packet jitter doesn't spike the prediction.
            target_id = target["id"]
            history = self._enemy_history.setdefault(
                target_id, deque(maxlen=4))
            history.append((raw_target_pos[:], server_ts))

            predicted_pos = raw_target_pos[:]
            if len(history) >= 2:
                velocities = []
                for j in range(1, len(history)):
                    p_new, t_new = history[j]
                    p_old, t_old = history[j - 1]
                    dt = t_new - t_old
                    if 0.001 < dt < 0.5:
                        velocities.append(
                            [(p_new[k] - p_old[k]) / dt for k in range(3)])
                if velocities:
                    avg_vel = [
                        sum(v[k] for v in velocities) / len(velocities)
                        for k in range(3)]
                    dt_last = history[-1][1] - history[-2][1]
                    if 0.001 < dt_last < 0.5:
                        predict_s = self.prediction_ticks * dt_last
                        predicted_pos = [
                            raw_target_pos[k] + avg_vel[k] * predict_s
                            for k in range(3)]
            # --------------------------------------------------------------

            target_pos = [predicted_pos[0], predicted_pos[1],
                          predicted_pos[2] + self.target_z_offset]

            # Apply world-space lateral offset before computing aim angles so
            # the correction scales naturally with distance.  Positive = left.
            if self.lateral_offset_units != 0.0:
                dx = target_pos[0] - player_pos[0]
                dy = target_pos[1] - player_pos[1]
                dist_h = math.sqrt(dx * dx + dy * dy)
                if dist_h > 0.001:
                    left_x = -dy / dist_h
                    left_y = dx / dist_h
                    target_pos[0] += left_x * self.lateral_offset_units
                    target_pos[1] += left_y * self.lateral_offset_units

            target_yaw, target_pitch = angle_to_target(
                player_pos, player_angles, target_pos)
            target_yaw += self.yaw_offset_deg
            angular_dist = math.sqrt(
                angle_delta(player_angles[1], target_yaw) ** 2 +
                angle_delta(player_angles[0], target_pitch) ** 2)

            if not is_in_fov(angular_dist, self.fov_deg):
                self._engagement_start_ms = None
                self._current_target_id = None
                self._print_status(
                    "[BotAim] Target outside FOV "
                    f"(target={target['id']}, angular_dist={angular_dist:.2f}, "
                    f"fov={self.fov_deg}, sent={self._send_count})",
                    min_interval_ms=2000.0)
                continue

            now_ms = time.time() * 1000
            if self._current_target_id != target["id"]:
                self._current_target_id = target["id"]
                self._engagement_start_ms = now_ms
                self._engagement_start_yaw = player_angles[1]
                self._engagement_start_pitch = player_angles[0]
                print(
                    "[BotAim] Target acquired "
                    f"id={target['id']} dist={angular_dist:.2f} "
                    f"target_yaw={target_yaw:.2f} target_pitch={target_pitch:.2f}")

            elapsed_ms = now_ms - self._engagement_start_ms
            actual_reaction = np.random.normal(
                self.reaction_ms, self.reaction_ms * 0.3) if self.reaction_ms > 0 else 0

            if elapsed_ms < actual_reaction:
                continue

            engagement_time = elapsed_ms - actual_reaction
            yaw, pitch = self.compute_aim_angles(
                self._engagement_start_yaw, self._engagement_start_pitch,
                target_yaw, target_pitch, engagement_time)

            # No wall-clock throttle: the tick queue is the natural rate limiter.
            # Sending on every received tick gives the override the freshest
            # possible angle at each game frame.
            self.send_angles(yaw, pitch)
            self._print_status(
                "[BotAim] Sent aim "
                f"yaw={yaw:.2f} pitch={pitch:.2f} "
                f"target={target['id']} sent={self._send_count}")

        if self._sock is not None:
            self._sock.close()
        print(f"[BotAim] Stopped (ticks={self._tick_count}, sent={self._send_count})")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

import math
import socket
import threading
import time
import queue
from collections import deque
from typing import Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_yaw(yaw: float) -> float:
    while yaw > 180.0:
        yaw -= 360.0
    while yaw < -180.0:
        yaw += 360.0
    return yaw


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
    target_yaw = normalize_yaw(math.degrees(math.atan2(dy, dx)))
    target_pitch = clamp(-math.degrees(math.atan2(dz, dist_h)), -89.0, 89.0)
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


# ---------------------------------------------------------------------------
# BotAimGenerator
# ---------------------------------------------------------------------------

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
        self.lateral_offset_units = config.get("lateral_offset_units", 0.0)
        self.yaw_offset_deg = config.get("yaw_offset_deg", 0.0)
        self.follow_gain = float(config.get("follow_gain", 0.55))

        # Conservative prediction — cap prediction_ticks so old high values
        # (e.g. 1.0) cannot silently produce an aggressive full-tick lead.
        self.prediction_enabled = bool(config.get("prediction_enabled", True))
        self.max_prediction_ticks = float(config.get("max_prediction_ticks", 0.35))
        self.prediction_ticks = clamp(
            float(config.get("prediction_ticks", 0.25)),
            0.0,
            self.max_prediction_ticks,
        )
        self.max_prediction_seconds = float(config.get("max_prediction_seconds", 0.025))
        self.max_prediction_units = float(config.get("max_prediction_units", 6.0))
        self.max_target_speed_units = float(config.get("max_target_speed_units", 320.0))
        self.velocity_smoothing = float(config.get("velocity_smoothing", 0.35))

        self.udp_host = udp_host
        self.udp_port = udp_port
        self._sock = None
        self._connected = False

        self._engagement_start_ms: Optional[float] = None
        self._engagement_start_yaw = 0.0
        self._engagement_start_pitch = 0.0
        self._current_target_id: Optional[int] = None
        self._reaction_delay_ms = 0.0
        self._tick_count = 0
        self._dropped_tick_count = 0
        self._send_count = 0
        self._last_status_ms = 0.0

        # Per-enemy rolling position history for velocity estimation
        self._enemy_history: dict = {}        # id -> deque[(pos, ts)]
        # Per-enemy EMA velocity (XY only, Z not predicted)
        self._enemy_velocity_ema: dict = {}   # id -> [vx, vy]

    # ------------------------------------------------------------------
    # Queue helpers
    # ------------------------------------------------------------------

    def _drain_to_latest_tick(self, tick_queue: queue.Queue,
                               current_tick: dict) -> dict:
        """Return the newest available tick, discarding older ones."""
        latest = current_tick
        drained = 0
        try:
            while True:
                latest = tick_queue.get_nowait()
                drained += 1
        except queue.Empty:
            pass
        self._dropped_tick_count += drained
        return latest

    # ------------------------------------------------------------------
    # Engagement helpers
    # ------------------------------------------------------------------

    def _sample_reaction_delay(self) -> float:
        if self.reaction_ms <= 0:
            return 0.0
        return max(0.0, np.random.normal(self.reaction_ms, self.reaction_ms * 0.3))

    def _start_engagement(self, target_id: int, angular_dist: float,
                           target_yaw: float, target_pitch: float,
                           current_yaw: float, current_pitch: float) -> None:
        self._current_target_id = target_id
        self._engagement_start_ms = time.time() * 1000
        self._engagement_start_yaw = current_yaw
        self._engagement_start_pitch = current_pitch
        # Sample reaction delay once per engagement, not every tick
        self._reaction_delay_ms = self._sample_reaction_delay()
        print(
            f"[BotAim] Target acquired id={target_id} dist={angular_dist:.2f} "
            f"yaw={target_yaw:.2f} pitch={target_pitch:.2f} "
            f"reaction_ms={self._reaction_delay_ms:.1f}")

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _clamp_xy_velocity(self, vx: float, vy: float) -> tuple:
        speed = math.sqrt(vx * vx + vy * vy)
        if speed > self.max_target_speed_units and speed > 0:
            scale = self.max_target_speed_units / speed
            return vx * scale, vy * scale
        return vx, vy

    def _estimate_velocity_from_history(self, target_id: int) -> Optional[list]:
        history = self._enemy_history.get(target_id)
        if history is None or len(history) < 2:
            return None
        velocities = []
        for j in range(1, len(history)):
            p_new, t_new = history[j]
            p_old, t_old = history[j - 1]
            dt = t_new - t_old
            if 0.001 < dt < 0.5:
                velocities.append(
                    [(p_new[k] - p_old[k]) / dt for k in range(3)])
        if not velocities:
            return None
        return [sum(v[k] for v in velocities) / len(velocities) for k in range(3)]

    def _smooth_target_velocity(self, target_id: int,
                                 vx: float, vy: float) -> tuple:
        alpha = self.velocity_smoothing
        prev = self._enemy_velocity_ema.get(target_id)
        if prev is None:
            smoothed = [vx, vy]
        else:
            smoothed = [
                alpha * vx + (1.0 - alpha) * prev[0],
                alpha * vy + (1.0 - alpha) * prev[1],
            ]
        self._enemy_velocity_ema[target_id] = smoothed
        return smoothed[0], smoothed[1]

    def _predict_target_position(self, target_id: int, raw_pos: list,
                                  target: dict, server_ts: float) -> list:
        if not self.prediction_enabled or self.prediction_ticks <= 0.0:
            return raw_pos[:]

        # Always update position history for fallback estimation
        history = self._enemy_history.setdefault(target_id, deque(maxlen=4))
        history.append((raw_pos[:], server_ts))

        # Prefer server-reported velocity (added by updated telemetry plugin)
        measured_vel = target.get("velocity")
        if measured_vel and len(measured_vel) == 3:
            vx, vy = float(measured_vel[0]), float(measured_vel[1])
        else:
            est = self._estimate_velocity_from_history(target_id)
            if est is None:
                return raw_pos[:]
            vx, vy = est[0], est[1]

        # Clamp to max plausible CS:S movement speed
        vx, vy = self._clamp_xy_velocity(vx, vy)

        # EMA smoothing reduces noise from jittery position samples
        vx, vy = self._smooth_target_velocity(target_id, vx, vy)

        # Lead time: capped by both max_prediction_seconds and measured tick dt
        if len(history) >= 2:
            tick_dt = history[-1][1] - history[-2][1]
            if not (0.001 < tick_dt < 0.5):
                tick_dt = 1.0 / max(self.send_rate_hz, 1)
        else:
            tick_dt = 1.0 / max(self.send_rate_hz, 1)

        lead_s = min(self.prediction_ticks * tick_dt, self.max_prediction_seconds)

        lead_x = vx * lead_s
        lead_y = vy * lead_s

        # Clamp total XY displacement to max_prediction_units
        lead_dist = math.sqrt(lead_x * lead_x + lead_y * lead_y)
        if lead_dist > self.max_prediction_units and lead_dist > 0:
            scale = self.max_prediction_units / lead_dist
            lead_x *= scale
            lead_y *= scale

        # XY only — do not predict Z.  Head height is stable and Z prediction
        # causes misses on crouch/jump transitions.
        return [raw_pos[0] + lead_x, raw_pos[1] + lead_y, raw_pos[2]]

    # ------------------------------------------------------------------
    # Aim computation
    # ------------------------------------------------------------------

    def compute_aim_angles(self, current_yaw: float, current_pitch: float,
                           target_yaw: float, target_pitch: float,
                           engagement_time_ms: float) -> tuple:
        if self.mode == "raw":
            return (target_yaw, target_pitch)

        elif self.mode == "smooth":
            # Always track from current view angle so a moving target
            # does not cause the bot to hold stale engagement-start aim.
            t = clamp(engagement_time_ms / max(self.tracking_ms, 1.0), 0.05, 1.0)
            yaw = current_yaw + angle_delta(current_yaw, target_yaw) * t
            pitch = current_pitch + angle_delta(current_pitch, target_pitch) * t
            return (yaw, pitch)

        elif self.mode == "humanised":
            t = clamp(engagement_time_ms / max(self.tracking_ms, 1.0), 0.0, 1.0)

            if t < 0.95:
                # Initial acquisition: S-curve sweep from engagement-start angle
                s = 1.0 / (1.0 + math.exp(-12.0 * (t - 0.5)))
                yaw_delta = angle_delta(self._engagement_start_yaw, target_yaw)
                pitch_delta = angle_delta(self._engagement_start_pitch, target_pitch)
                yaw = self._engagement_start_yaw + yaw_delta * s
                pitch = self._engagement_start_pitch + pitch_delta * s

                if t > 0.85 and np.random.random() < self.overshoot_prob:
                    overshoot = np.random.uniform(0, self.overshoot_deg)
                    yaw += overshoot * np.sign(yaw_delta)
            else:
                # Tracking phase: follow from current angle using follow_gain
                # so stale engagement-start angles don't fight a moving target.
                yaw = current_yaw + angle_delta(current_yaw, target_yaw) * self.follow_gain
                pitch = current_pitch + angle_delta(current_pitch, target_pitch) * self.follow_gain

            if self.jitter_amp_deg > 0:
                yaw += np.random.normal(0, self.jitter_amp_deg * 0.5)
                pitch += np.random.normal(0, self.jitter_amp_deg * 0.3)

            return (yaw, pitch)

        return (current_yaw, current_pitch)

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def send_angles(self, yaw: float, pitch: float):
        if not self._connected:
            try:
                self._sock = socket.create_connection(
                    (self.udp_host, self.udp_port), timeout=1.0)
                self._sock.settimeout(1.0)
                self._connected = True
                print(f"[BotAim] Connected to override at "
                      f"{self.udp_host}:{self.udp_port}")
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, tick_queue: queue.Queue, stop_event: threading.Event):
        print(
            f"[BotAim] Started — mode={self.mode}, fov={self.fov_deg}, "
            f"override={self.udp_host}:{self.udp_port}, "
            f"prediction_ticks={self.prediction_ticks:.3f} "
            f"(max={self.max_prediction_ticks:.3f}), "
            f"max_prediction_units={self.max_prediction_units:.1f}")

        while not stop_event.is_set():
            try:
                tick = tick_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Discard any queued ticks older than this one
            tick = self._drain_to_latest_tick(tick_queue, tick)
            self._tick_count += 1

            player_pos = tick.get("eye_position", tick.get("position", [0, 0, 0]))
            player_angles = tick.get("view_angles", [0, 0])
            server_ts = float(tick.get("timestamp_server", 0.0))
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
                self._enemy_velocity_ema.clear()
                self._print_status(
                    "[BotAim] No visible target "
                    f"(ticks={self._tick_count}, enemies={len(enemies)}, "
                    f"visible={len(visible_enemies)}, sent={self._send_count})",
                    min_interval_ms=2000.0)
                continue

            raw_target_pos = list(target.get("aim_position", target["position"]))
            target_id = int(target.get("id", -1))

            predicted_pos = self._predict_target_position(
                target_id, raw_target_pos, target, server_ts)

            target_pos = [predicted_pos[0], predicted_pos[1],
                          predicted_pos[2] + self.target_z_offset]

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
            target_yaw = normalize_yaw(target_yaw + self.yaw_offset_deg)

            angular_dist = math.sqrt(
                angle_delta(player_angles[1], target_yaw) ** 2 +
                angle_delta(player_angles[0], target_pitch) ** 2)

            if not is_in_fov(angular_dist, self.fov_deg):
                self._engagement_start_ms = None
                self._current_target_id = None
                self._print_status(
                    "[BotAim] Target outside FOV "
                    f"(target={target_id}, angular_dist={angular_dist:.2f}, "
                    f"fov={self.fov_deg}, sent={self._send_count})",
                    min_interval_ms=2000.0)
                continue

            now_ms = time.time() * 1000
            if self._current_target_id != target_id:
                self._start_engagement(
                    target_id, angular_dist, target_yaw, target_pitch,
                    player_angles[1], player_angles[0])

            elapsed_ms = now_ms - self._engagement_start_ms
            if elapsed_ms < self._reaction_delay_ms:
                continue

            engagement_time = elapsed_ms - self._reaction_delay_ms
            # Pass current player angles so smooth/humanised modes track from
            # the live view position, not stale engagement-start angles.
            yaw, pitch = self.compute_aim_angles(
                player_angles[1], player_angles[0],
                target_yaw, target_pitch, engagement_time)

            self.send_angles(yaw, pitch)
            self._print_status(
                "[BotAim] Sent aim "
                f"yaw={yaw:.2f} pitch={pitch:.2f} "
                f"target={target_id} sent={self._send_count}")

        if self._sock is not None:
            self._sock.close()
        print(
            f"[BotAim] Stopped (ticks={self._tick_count}, "
            f"dropped_stale={self._dropped_tick_count}, sent={self._send_count})")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

from __future__ import annotations

from typing import Any


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class GimbalController:
    LOCK_STATES = {"LOCKED", "TRACKING_XY", "TRACKING_XYZ"}

    def __init__(self, cfg: dict[str, Any]):
        self.yaw_min = float(cfg.get("yaw_min", -1.4))
        self.yaw_max = float(cfg.get("yaw_max", 1.4))
        self.pitch_min = float(cfg.get("pitch_min", -0.7))
        self.pitch_max = float(cfg.get("pitch_max", 0.7))

        self.kp_yaw = float(cfg.get("kp_yaw", 1.1))
        self.kp_pitch = float(cfg.get("kp_pitch", 0.9))
        self.pitch_sign = float(cfg.get("pitch_sign", 1.0))

        self.deadzone_x = float(cfg.get("deadzone_x", 0.03))
        self.deadzone_y = float(cfg.get("deadzone_y", 0.04))

        self.max_yaw_rate = float(cfg.get("max_yaw_rate", 0.9))
        self.max_pitch_rate = float(cfg.get("max_pitch_rate", 0.7))

        self.ema_alpha_x = float(cfg.get("ema_alpha_x", 0.25))
        self.ema_alpha_y = float(cfg.get("ema_alpha_y", 0.25))
        self.rate_smoothing_alpha = float(cfg.get("rate_smoothing_alpha", 0.35))

        self.yaw_cmd = 0.0
        self.pitch_cmd = 0.0
        self._err_x_ema = 0.0
        self._err_y_ema = 0.0
        self._yaw_rate_smoothed = 0.0
        self._pitch_rate_smoothed = 0.0

    def _compute_centering_rate(self, err: float, deadzone: float, kp: float, max_rate: float) -> float:
        if abs(err) <= deadzone:
            return 0.0
        raw = kp * err
        return _clamp(raw, -max_rate, max_rate)

    def _smooth_rate(self, new_rate: float, prev_rate: float) -> float:
        a = self.rate_smoothing_alpha
        return a * new_rate + (1.0 - a) * prev_rate

    def compute(
        self,
        state: str,
        primary_track: dict[str, Any] | None,
        err_x: float | None,
        err_y: float | None,
        search_cmd: dict[str, Any] | None,
        dt: float,
    ) -> tuple[float, float, dict[str, Any]]:
        yaw_rate = 0.0
        pitch_rate = 0.0

        if state in self.LOCK_STATES and primary_track is not None and err_x is not None and err_y is not None:
            self._err_x_ema = self.ema_alpha_x * err_x + (1.0 - self.ema_alpha_x) * self._err_x_ema
            self._err_y_ema = self.ema_alpha_y * err_y + (1.0 - self.ema_alpha_y) * self._err_y_ema

            yaw_rate = -self._compute_centering_rate(
                err=self._err_x_ema,
                deadzone=self.deadzone_x,
                kp=self.kp_yaw,
                max_rate=self.max_yaw_rate,
            )
            pitch_rate = self.pitch_sign * self._compute_centering_rate(
                err=self._err_y_ema,
                deadzone=self.deadzone_y,
                kp=self.kp_pitch,
                max_rate=self.max_pitch_rate,
            )
        elif state == "SEARCHING" and search_cmd is not None:
            yaw_rate = _clamp(float(search_cmd.get("yaw_rate", 0.0)), -self.max_yaw_rate, self.max_yaw_rate)
            pitch_rate = _clamp(float(search_cmd.get("pitch_rate", 0.0)), -self.max_pitch_rate, self.max_pitch_rate)

        self._yaw_rate_smoothed = self._smooth_rate(yaw_rate, self._yaw_rate_smoothed)
        self._pitch_rate_smoothed = self._smooth_rate(pitch_rate, self._pitch_rate_smoothed)

        self.yaw_cmd = _clamp(self.yaw_cmd + self._yaw_rate_smoothed * dt, self.yaw_min, self.yaw_max)
        self.pitch_cmd = _clamp(self.pitch_cmd + self._pitch_rate_smoothed * dt, self.pitch_min, self.pitch_max)

        debug = {
            "yaw_rate": self._yaw_rate_smoothed,
            "pitch_rate": self._pitch_rate_smoothed,
            "yaw_saturated": abs(self.yaw_cmd - self.yaw_min) < 1e-9 or abs(self.yaw_cmd - self.yaw_max) < 1e-9,
            "pitch_saturated": abs(self.pitch_cmd - self.pitch_min) < 1e-9 or abs(self.pitch_cmd - self.pitch_max) < 1e-9,
            "err_x_ema": self._err_x_ema,
            "err_y_ema": self._err_y_ema,
        }
        return self.yaw_cmd, self.pitch_cmd, debug

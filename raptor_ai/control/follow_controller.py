from __future__ import annotations

from typing import Any


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class FollowController:
    """
    Lightweight visual-follow controller for iris platform.

    Inputs are image-space errors and target bbox area ratio.
    Outputs are body-frame velocity commands (x/y), vertical speed (z) and yaw_rate.
    """

    ACTIVE_STATES = {"TRACKING_XY", "TRACKING_XYZ"}

    def __init__(self, cfg: dict[str, Any]):
        self.enabled = bool(cfg.get("enabled", False))
        mode = str(cfg.get("mode", "xy")).strip().lower()
        if mode not in ("xy", "xyz"):
            raise ValueError(f"Unsupported follow.mode: {mode}")
        self.mode = mode
        xy_strategy = str(cfg.get("xy_strategy", "full_xy")).strip().lower()
        if xy_strategy not in ("yaw_only", "full_xy", "forward_lock", "forward_track"):
            raise ValueError(f"Unsupported follow.xy_strategy: {xy_strategy}")
        self.xy_strategy = xy_strategy

        # Tracking target in image-space.
        self.target_area_ratio = float(cfg.get("target_area_ratio", 0.045))
        self.min_distance_area_ratio = float(cfg.get("min_distance_area_ratio", 0.12))
        self.deadzone_x = float(cfg.get("deadzone_x", 0.03))
        self.deadzone_y = float(cfg.get("deadzone_y", 0.04))
        self.deadzone_area = float(cfg.get("deadzone_area", 0.004))

        # Controller gains.
        self.kp_forward = float(cfg.get("kp_forward", 2.2))
        self.kp_lateral = float(cfg.get("kp_lateral", 1.1))
        self.kp_vertical = float(cfg.get("kp_vertical", 1.0))
        self.kp_yaw = float(cfg.get("kp_yaw", 1.2))
        self.lateral_error_sign = float(cfg.get("lateral_error_sign", -1.0))
        self.yaw_error_sign = float(cfg.get("yaw_error_sign", -1.0))

        # Safety limits.
        self.max_vx = float(cfg.get("max_vx", 1.2))
        self.max_vy = float(cfg.get("max_vy", 1.0))
        self.max_vz = float(cfg.get("max_vz", 0.7))
        self.max_yaw_rate = float(cfg.get("max_yaw_rate", 0.7))

        # forward_lock strategy: center target by yaw, then move forward.
        self.forward_lock_speed = float(cfg.get("forward_lock_speed", 0.35))
        self.forward_lock_required_frames = max(1, int(cfg.get("forward_lock_required_frames", 5)))
        self.forward_lock_deadzone_x = float(cfg.get("forward_lock_deadzone_x", self.deadzone_x))
        self.forward_lock_use_area_gate = bool(cfg.get("forward_lock_use_area_gate", True))
        self.forward_lock_use_lateral = bool(cfg.get("forward_lock_use_lateral", True))
        # forward_track strategy: move forward while aligning by yaw/lateral.
        self.forward_track_speed = float(cfg.get("forward_track_speed", self.forward_lock_speed))
        self.forward_track_min_speed = float(cfg.get("forward_track_min_speed", 0.10))
        self.forward_track_align_gate_x = float(cfg.get("forward_track_align_gate_x", 0.12))
        self.forward_track_stop_gate_x = float(cfg.get("forward_track_stop_gate_x", 0.30))
        self.forward_track_use_area_gate = bool(cfg.get("forward_track_use_area_gate", True))
        self.forward_track_use_lateral = bool(cfg.get("forward_track_use_lateral", self.forward_lock_use_lateral))
        self.forward_track_lateral_scale = float(cfg.get("forward_track_lateral_scale", 0.6))
        self.forward_track_yaw_scale = float(cfg.get("forward_track_yaw_scale", 1.0))

        self.smoothing_alpha = float(cfg.get("smoothing_alpha", 0.35))
        self._vx_smoothed = 0.0
        self._vy_smoothed = 0.0
        self._vz_smoothed = 0.0
        self._yaw_rate_smoothed = 0.0
        self._center_lock_frames = 0

    def _smooth(self, value: float, prev: float) -> float:
        a = self.smoothing_alpha
        return a * value + (1.0 - a) * prev

    def _target_area_error(self, primary_track: dict[str, Any]) -> tuple[float, float]:
        frame_w = max(1, int(primary_track.get("frame_w", 1)))
        frame_h = max(1, int(primary_track.get("frame_h", 1)))
        area = float(primary_track.get("area", 0.0))
        area_ratio = area / float(frame_w * frame_h)
        area_err = self.target_area_ratio - area_ratio
        return area_ratio, area_err

    @staticmethod
    def _in_deadzone(value: float, dz: float) -> bool:
        return abs(value) <= dz

    def _zero(self, state: str, reason: str) -> dict[str, Any]:
        self._vx_smoothed = self._smooth(0.0, self._vx_smoothed)
        self._vy_smoothed = self._smooth(0.0, self._vy_smoothed)
        self._vz_smoothed = self._smooth(0.0, self._vz_smoothed)
        self._yaw_rate_smoothed = self._smooth(0.0, self._yaw_rate_smoothed)
        self._center_lock_frames = 0
        return {
            "active": False,
            "mode": self.mode,
            "xy_strategy": self.xy_strategy,
            "state": state,
            "reason": reason,
            "vx_body": 0.0,
            "vy_body": 0.0,
            "vz": 0.0,
            "yaw_rate": 0.0,
            "area_ratio": 0.0,
            "area_error": 0.0,
            "center_lock_frames": 0,
        }

    def compute(
        self,
        *,
        state: str,
        primary_track: dict[str, Any] | None,
        err_x: float | None,
        err_y: float | None,
        dt: float,
    ) -> dict[str, Any]:
        del dt
        if not self.enabled:
            return self._zero(state, "disabled")
        if state not in self.ACTIVE_STATES:
            return self._zero(state, "inactive_state")
        if primary_track is None or err_x is None:
            return self._zero(state, "no_primary")

        area_ratio, area_err = self._target_area_error(primary_track)

        if self._in_deadzone(area_err, self.deadzone_area):
            vx = 0.0
        else:
            vx = self.kp_forward * area_err

        # Do not move forward when already too close.
        if area_ratio >= self.min_distance_area_ratio and vx > 0.0:
            vx = 0.0

        vy = (
            0.0
            if self._in_deadzone(err_x, self.deadzone_x)
            else (self.lateral_error_sign * self.kp_lateral * float(err_x))
        )
        yaw_rate = (
            0.0
            if self._in_deadzone(err_x, self.deadzone_x)
            else (self.yaw_error_sign * self.kp_yaw * float(err_x))
        )
        if self.mode == "xy" and self.xy_strategy == "yaw_only":
            vx = 0.0
            vy = 0.0
        elif self.mode == "xy" and self.xy_strategy == "forward_lock":
            ex = float(err_x)
            centered_x = self._in_deadzone(ex, self.forward_lock_deadzone_x)
            if centered_x:
                self._center_lock_frames += 1
            else:
                self._center_lock_frames = 0

            if centered_x:
                vy = 0.0
            elif self.forward_lock_use_lateral:
                vy = self.lateral_error_sign * self.kp_lateral * ex
            else:
                vy = 0.0

            yaw_rate = 0.0 if centered_x else (self.yaw_error_sign * self.kp_yaw * ex)
            vx = 0.0
            if self._center_lock_frames >= self.forward_lock_required_frames:
                can_move_by_area = True
                if self.forward_lock_use_area_gate:
                    can_move_by_area = area_ratio < self.min_distance_area_ratio
                if can_move_by_area:
                    vx = self.forward_lock_speed
        elif self.mode == "xy" and self.xy_strategy == "forward_track":
            ex = float(err_x)
            abs_ex = abs(ex)

            centered_x = self._in_deadzone(ex, self.forward_lock_deadzone_x)
            if centered_x:
                self._center_lock_frames += 1
            else:
                self._center_lock_frames = 0

            yaw_rate = 0.0 if self._in_deadzone(ex, self.deadzone_x) else (
                self.yaw_error_sign * self.kp_yaw * ex * self.forward_track_yaw_scale
            )

            if self._in_deadzone(ex, self.deadzone_x):
                vy = 0.0
            elif self.forward_track_use_lateral:
                vy = self.lateral_error_sign * self.kp_lateral * ex * self.forward_track_lateral_scale
            else:
                vy = 0.0

            too_close = self.forward_track_use_area_gate and (area_ratio >= self.min_distance_area_ratio)
            if too_close:
                vx = 0.0
            else:
                align_gate = max(1e-3, self.forward_track_align_gate_x)
                stop_gate = max(align_gate + 1e-3, self.forward_track_stop_gate_x)

                if abs_ex >= stop_gate:
                    blend = 0.0
                elif abs_ex <= align_gate:
                    blend = 1.0
                else:
                    blend = 1.0 - ((abs_ex - align_gate) / (stop_gate - align_gate))

                vmax = max(self.forward_track_speed, 0.0)
                vmin = max(0.0, min(self.forward_track_min_speed, vmax))
                if blend <= 0.0:
                    vx = 0.0
                else:
                    vx = vmin + (vmax - vmin) * blend

        vz = 0.0
        if self.mode == "xyz" and err_y is not None and not self._in_deadzone(float(err_y), self.deadzone_y):
            # Positive err_y means target is below image center -> descend.
            vz = -self.kp_vertical * float(err_y)

        vx = _clamp(vx, -self.max_vx, self.max_vx)
        vy = _clamp(vy, -self.max_vy, self.max_vy)
        vz = _clamp(vz, -self.max_vz, self.max_vz)
        yaw_rate = _clamp(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        self._vx_smoothed = self._smooth(vx, self._vx_smoothed)
        self._vy_smoothed = self._smooth(vy, self._vy_smoothed)
        self._vz_smoothed = self._smooth(vz, self._vz_smoothed)
        self._yaw_rate_smoothed = self._smooth(yaw_rate, self._yaw_rate_smoothed)

        return {
            "active": True,
            "mode": self.mode,
            "xy_strategy": self.xy_strategy,
            "state": state,
            "reason": (
                "forward_lock_align"
                if (self.mode == "xy" and self.xy_strategy == "forward_lock" and abs(float(err_x)) > self.forward_lock_deadzone_x)
                else (
                    "forward_lock_forward"
                    if (self.mode == "xy" and self.xy_strategy == "forward_lock" and abs(self._vx_smoothed) > 1e-4)
                    else (
                        "forward_lock_wait"
                        if (self.mode == "xy" and self.xy_strategy == "forward_lock")
                        else (
                            "forward_track_hold_area"
                            if (self.mode == "xy" and self.xy_strategy == "forward_track" and area_ratio >= self.min_distance_area_ratio and self.forward_track_use_area_gate)
                            else (
                                "forward_track_align_hard"
                                if (self.mode == "xy" and self.xy_strategy == "forward_track" and abs(float(err_x)) >= self.forward_track_stop_gate_x)
                                else (
                                    "forward_track_align"
                                    if (self.mode == "xy" and self.xy_strategy == "forward_track" and abs(float(err_x)) > self.forward_track_align_gate_x)
                                    else (
                                        "forward_track_cruise"
                                        if (self.mode == "xy" and self.xy_strategy == "forward_track" and abs(self._vx_smoothed) > 1e-4)
                                        else (
                                            "forward_track_wait"
                                            if (self.mode == "xy" and self.xy_strategy == "forward_track")
                                            else "tracking"
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            "vx_body": self._vx_smoothed,
            "vy_body": self._vy_smoothed,
            "vz": self._vz_smoothed,
            "yaw_rate": self._yaw_rate_smoothed,
            "area_ratio": area_ratio,
            "area_error": area_err,
            "center_lock_frames": int(self._center_lock_frames),
        }

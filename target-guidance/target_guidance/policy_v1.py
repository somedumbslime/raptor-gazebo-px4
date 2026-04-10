from __future__ import annotations

from typing import Any

from .contracts import GuidanceInput, GuidanceOutput


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class TargetGuidancePolicyV1:
    """
    Baseline visual-follow policy extracted from RuntimeV2 follow logic.

    Input contract: GuidanceInput (primary track + normalized image errors).
    Output contract: GuidanceOutput (body-frame command + telemetry fields).
    """

    ACTIVE_STATES = {"TRACKING_XY", "TRACKING_XYZ"}

    def __init__(self, cfg: dict[str, Any]):
        self.enabled = bool(cfg.get("enabled", False))
        mode = str(cfg.get("mode", "xy")).strip().lower()
        if mode not in ("xy", "xyz"):
            raise ValueError(f"Unsupported follow.mode: {mode}")
        self.mode = mode
        xy_strategy = str(cfg.get("xy_strategy", "zone_track")).strip().lower()
        if xy_strategy != "zone_track":
            raise ValueError("TargetGuidancePolicyV1 supports only follow.xy_strategy='zone_track'")
        self.xy_strategy = xy_strategy

        self.target_area_ratio = float(cfg.get("target_area_ratio", 0.045))
        self.min_distance_area_ratio = max(0.0, float(cfg.get("min_distance_area_ratio", 0.12)))
        self.deadzone_x = float(cfg.get("deadzone_x", 0.03))
        self.deadzone_y = float(cfg.get("deadzone_y", 0.04))
        self.deadzone_area = float(cfg.get("deadzone_area", 0.004))

        self.kp_forward = float(cfg.get("kp_forward", 2.2))
        self.kp_lateral = float(cfg.get("kp_lateral", 1.1))
        self.kp_vertical = float(cfg.get("kp_vertical", 1.0))
        self.kp_yaw = float(cfg.get("kp_yaw", 1.2))
        self.lateral_error_sign = float(cfg.get("lateral_error_sign", -1.0))
        self.yaw_error_sign = float(cfg.get("yaw_error_sign", -1.0))

        self.max_vx = float(cfg.get("max_vx", 1.2))
        self.max_vy = float(cfg.get("max_vy", 1.0))
        self.max_vz = float(cfg.get("max_vz", 0.7))
        self.max_yaw_rate = float(cfg.get("max_yaw_rate", 0.7))

        # zone_track strategy: continuous quadrant-aware steering with non-zero forward speed.
        self.zone_track_box_scale_x = max(1.0, float(cfg.get("zone_track_box_scale_x", 2.2)))
        self.zone_track_box_scale_y = max(1.0, float(cfg.get("zone_track_box_scale_y", 2.0)))
        self.zone_track_axis_hysteresis = _clamp(float(cfg.get("zone_track_axis_hysteresis", 0.06)), 0.0, 0.5)
        self.zone_track_forward_speed_min = max(
            0.0,
            float(cfg.get("zone_track_forward_speed_min", 0.20)),
        )
        self.zone_track_forward_speed_max = max(
            self.zone_track_forward_speed_min,
            float(cfg.get("zone_track_forward_speed_max", max(0.65, self.zone_track_forward_speed_min))),
        )
        self.zone_track_forward_speed_exponent = max(
            0.2,
            float(cfg.get("zone_track_forward_speed_exponent", 1.20)),
        )
        self.zone_track_steer_exponent = max(
            0.5,
            float(cfg.get("zone_track_steer_exponent", 1.35)),
        )
        self.zone_track_forward_align_weight_x = _clamp(
            float(cfg.get("zone_track_forward_align_weight_x", 0.85)),
            0.0,
            1.0,
        )
        self.zone_track_forward_steer_brake = _clamp(
            float(cfg.get("zone_track_forward_steer_brake", 0.45)),
            0.0,
            0.95,
        )
        self.zone_track_lateral_scale = max(
            0.0,
            float(cfg.get("zone_track_lateral_scale", 0.80)),
        )
        self.zone_track_yaw_scale = max(
            0.0,
            float(cfg.get("zone_track_yaw_scale", 1.10)),
        )
        self.zone_track_center_deadzone_x = _clamp(
            float(cfg.get("zone_track_center_deadzone_x", max(self.deadzone_x, 0.06))),
            0.0,
            0.95,
        )
        self.zone_track_center_deadzone_y = _clamp(
            float(cfg.get("zone_track_center_deadzone_y", max(self.deadzone_y, 0.08))),
            0.0,
            0.95,
        )
        self.zone_track_use_area_gate = bool(cfg.get("zone_track_use_area_gate", False))
        # Terminal charge: close-range "ring pass" mode (do not brake near target).
        self.terminal_charge_enabled = bool(cfg.get("terminal_charge_enabled", True))
        self.terminal_charge_enter_area_ratio = float(cfg.get("terminal_charge_enter_area_ratio", 0.070))
        self.terminal_charge_exit_area_ratio = float(
            cfg.get("terminal_charge_exit_area_ratio", self.terminal_charge_enter_area_ratio * 0.80)
        )
        self.terminal_charge_align_gate_x = float(
            cfg.get("terminal_charge_align_gate_x", max(self.deadzone_x, self.zone_track_center_deadzone_x * 0.80))
        )
        self.terminal_charge_exit_align_gate_x = float(
            cfg.get("terminal_charge_exit_align_gate_x", max(self.terminal_charge_align_gate_x * 1.6, self.deadzone_x))
        )
        self.terminal_charge_required_frames = max(1, int(cfg.get("terminal_charge_required_frames", 4)))
        self.terminal_charge_speed = float(
            cfg.get("terminal_charge_speed", max(self.zone_track_forward_speed_max * 1.10, 0.85))
        )
        self.terminal_charge_misaligned_speed = float(
            cfg.get("terminal_charge_misaligned_speed", max(0.0, self.zone_track_forward_speed_min * 0.70))
        )
        self.terminal_charge_yaw_scale = float(cfg.get("terminal_charge_yaw_scale", 0.85))
        self.terminal_charge_lateral_scale = float(cfg.get("terminal_charge_lateral_scale", 0.85))

        # XY altitude-hold assist (runs in target-guidance, not in bridge).
        self.xy_alt_hold_enabled = bool(cfg.get("xy_alt_hold_enabled", True))
        self.xy_alt_hold_min_forward_m_s = max(0.0, float(cfg.get("xy_alt_hold_min_forward_m_s", 0.10)))
        self.xy_alt_hold_min_active_alt_m = max(0.0, float(cfg.get("xy_alt_hold_min_active_alt_m", 0.8)))
        self.xy_alt_hold_kp = max(0.0, float(cfg.get("xy_alt_hold_kp", 0.75)))
        self.xy_alt_hold_kd = max(0.0, float(cfg.get("xy_alt_hold_kd", 0.30)))
        self.xy_alt_hold_max_vz = max(0.0, float(cfg.get("xy_alt_hold_max_vz", 0.20)))
        self.xy_alt_hold_ref_alpha = _clamp(float(cfg.get("xy_alt_hold_ref_alpha", 0.06)), 0.0, 1.0)
        self.xy_alt_hold_release_s = max(0.0, float(cfg.get("xy_alt_hold_release_s", 0.8)))
        self.xy_alt_hold_disable_floor_in_terminal_charge = bool(
            cfg.get("xy_alt_hold_disable_floor_in_terminal_charge", True)
        )
        self.xy_alt_hold_terminal_ref_alpha = _clamp(float(cfg.get("xy_alt_hold_terminal_ref_alpha", 0.30)), 0.0, 1.0)
        self.xy_alt_hold_hard_emergency_enabled = bool(cfg.get("xy_alt_hold_hard_emergency_enabled", True))
        self.xy_alt_hold_hard_emergency_floor_m = max(
            0.0,
            float(cfg.get("xy_alt_hold_hard_emergency_floor_m", 0.28)),
        )
        self.xy_alt_hold_hard_emergency_release_m = max(
            self.xy_alt_hold_hard_emergency_floor_m,
            float(cfg.get("xy_alt_hold_hard_emergency_release_m", 0.40)),
        )
        self.xy_alt_hold_hard_emergency_vz_up = max(
            0.0,
            float(cfg.get("xy_alt_hold_hard_emergency_vz_up", 0.30)),
        )

        # PX4 lifecycle policy (arm/takeoff/offboard) moved from px4 bridge.
        self.px4_lifecycle_enabled = bool(cfg.get("px4_lifecycle_enabled", True))
        self.px4_auto_arm = bool(cfg.get("px4_auto_arm", False))
        self.px4_auto_takeoff = bool(cfg.get("px4_auto_takeoff", False))
        self.px4_auto_offboard = bool(cfg.get("px4_auto_offboard", True))
        self.px4_auto_arm_require_armable = bool(cfg.get("px4_auto_arm_require_armable", True))
        self.px4_auto_arm_require_local_position = bool(cfg.get("px4_auto_arm_require_local_position", False))
        self.px4_offboard_min_relative_alt_m = max(0.0, float(cfg.get("px4_offboard_min_relative_alt_m", 0.6)))
        self.px4_takeoff_confirm_alt_m = max(
            0.0,
            float(cfg.get("px4_takeoff_confirm_alt_m", self.px4_offboard_min_relative_alt_m)),
        )
        self.px4_arm_retry_s = max(0.1, float(cfg.get("px4_arm_retry_s", 2.0)))
        self.px4_takeoff_retry_s = max(0.1, float(cfg.get("px4_takeoff_retry_s", 4.0)))
        self.px4_takeoff_liftoff_timeout_s = max(1.0, float(cfg.get("px4_takeoff_liftoff_timeout_s", 20.0)))
        self.px4_offboard_start_delay_after_liftoff_s = max(
            0.0,
            float(cfg.get("px4_offboard_start_delay_after_liftoff_s", 1.2)),
        )
        self.px4_offboard_retry_s = max(0.1, float(cfg.get("px4_offboard_retry_s", 1.0)))

        self.smoothing_alpha = float(cfg.get("smoothing_alpha", 0.35))
        self._vx_smoothed = 0.0
        self._vy_smoothed = 0.0
        self._vz_smoothed = 0.0
        self._yaw_rate_smoothed = 0.0
        self._center_lock_frames = 0
        self._xy_alt_ref_m: float | None = None
        self._xy_alt_last_forward_ts = 0.0
        self._xy_alt_emergency_active = False
        self._xy_alt_emergency_latched = False
        self._terminal_charge_active = False
        self._terminal_charge_ready_frames = 0
        self._zone_track_sign_x = 0
        self._zone_track_sign_y = 0
        self._zone_track_last_label = "center"
        self._next_arm_ts = 0.0
        self._next_takeoff_ts = 0.0
        self._next_offboard_ts = 0.0
        self._liftoff_since_ts = 0.0
        self._takeoff_started_ts = 0.0

    def _smooth(self, value: float, prev: float) -> float:
        a = self.smoothing_alpha
        return a * value + (1.0 - a) * prev

    @staticmethod
    def _in_deadzone(value: float, dz: float) -> bool:
        return abs(value) <= dz

    def _target_area_error(self, primary_track: dict[str, Any]) -> tuple[float, float]:
        frame_w = max(1, int(primary_track.get("frame_w", 1)))
        frame_h = max(1, int(primary_track.get("frame_h", 1)))
        area = float(primary_track.get("area", 0.0))
        area_ratio = area / float(frame_w * frame_h)
        area_err = self.target_area_ratio - area_ratio
        return area_ratio, area_err

    def _area_gate_enabled(self) -> bool:
        # Explicitly treat non-positive threshold as "disabled".
        # This prevents accidental freezes when users set 0.0 expecting "no gate".
        return self.min_distance_area_ratio > 1e-6

    @staticmethod
    def _bbox_dims(primary_track: dict[str, Any]) -> tuple[float, float]:
        bbox = primary_track.get("bbox_xyxy")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return 0.0, 0.0
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return 0.0, 0.0
        return max(1.0, abs(x2 - x1)), max(1.0, abs(y2 - y1))

    def _zone_track_normalized_error(
        self,
        *,
        primary_track: dict[str, Any],
        err_x: float,
        err_y: float,
    ) -> tuple[float, float]:
        frame_w = max(1, int(primary_track.get("frame_w", 1)))
        frame_h = max(1, int(primary_track.get("frame_h", 1)))
        bbox_w, bbox_h = self._bbox_dims(primary_track)
        guide_w = max(1.0, bbox_w * self.zone_track_box_scale_x)
        guide_h = max(1.0, bbox_h * self.zone_track_box_scale_y)
        norm_x = max(1e-6, guide_w / float(frame_w))
        norm_y = max(1e-6, guide_h / float(frame_h))
        ux = _clamp(float(err_x) / norm_x, -1.0, 1.0)
        uy = _clamp(float(err_y) / norm_y, -1.0, 1.0)
        return ux, uy

    def _sign_with_hysteresis(self, value: float, prev: int) -> int:
        h = self.zone_track_axis_hysteresis
        if value > h:
            return 1
        if value < -h:
            return -1
        if abs(value) <= (h * 0.5):
            return 0
        return int(prev)

    @staticmethod
    def _zone_label(sign_x: int, sign_y: int) -> str:
        if sign_x == 0 and sign_y == 0:
            return "center"
        if sign_x == 0:
            return "down" if sign_y > 0 else "up"
        if sign_y == 0:
            return "right" if sign_x > 0 else "left"
        if sign_x > 0 and sign_y > 0:
            return "down_right"
        if sign_x > 0 and sign_y < 0:
            return "up_right"
        if sign_x < 0 and sign_y > 0:
            return "down_left"
        return "up_left"

    def _zero(
        self,
        *,
        state: str,
        reason: str,
        platform_action: str,
        platform_action_payload: dict[str, Any],
    ) -> GuidanceOutput:
        self._vx_smoothed = self._smooth(0.0, self._vx_smoothed)
        self._vy_smoothed = self._smooth(0.0, self._vy_smoothed)
        self._vz_smoothed = self._smooth(0.0, self._vz_smoothed)
        self._yaw_rate_smoothed = self._smooth(0.0, self._yaw_rate_smoothed)
        self._center_lock_frames = 0
        self._xy_alt_ref_m = None
        self._xy_alt_last_forward_ts = 0.0
        self._xy_alt_emergency_active = False
        self._xy_alt_emergency_latched = False
        self._terminal_charge_active = False
        self._terminal_charge_ready_frames = 0
        self._zone_track_sign_x = 0
        self._zone_track_sign_y = 0
        self._zone_track_last_label = "center"
        return GuidanceOutput(
            active=False,
            mode=self.mode,
            xy_strategy=self.xy_strategy,
            state=state,
            reason=reason,
            vx_body=0.0,
            vy_body=0.0,
            vz=0.0,
            yaw_rate=0.0,
            area_ratio=0.0,
            area_error=0.0,
            center_lock_frames=0,
            platform_action=platform_action,
            platform_action_payload=dict(platform_action_payload),
        )

    def _compute_px4_lifecycle_action(
        self,
        *,
        ts: float,
        platform_meta: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        if not self.px4_lifecycle_enabled:
            return "none", {"reason": "lifecycle_disabled"}
        if not isinstance(platform_meta, dict):
            return "none", {"reason": "no_platform_meta"}
        if str(platform_meta.get("platform_type", "")).strip().lower() != "px4":
            return "none", {"reason": "platform_not_px4"}

        connected = bool(platform_meta.get("connected", False))
        armed = bool(platform_meta.get("armed", False))
        in_air_reported = bool(platform_meta.get("in_air", False))
        is_armable = bool(platform_meta.get("is_armable", False))
        is_local_position_ok = bool(platform_meta.get("is_local_position_ok", False))
        flight_mode = str(platform_meta.get("flight_mode", "")).upper()
        rel_alt = float(platform_meta.get("relative_altitude_m", 0.0) or 0.0)
        offboard_active = bool(
            platform_meta.get("offboard_started", False)
            or platform_meta.get("offboard_mode_active", False)
            or ("OFFBOARD" in flight_mode)
        )
        liftoff_confirmed = rel_alt >= self.px4_takeoff_confirm_alt_m

        if liftoff_confirmed:
            if self._liftoff_since_ts <= 0.0:
                self._liftoff_since_ts = float(ts)
        else:
            self._liftoff_since_ts = 0.0

        if not connected:
            return "none", {"reason": "px4_not_connected"}

        if not armed:
            self._next_takeoff_ts = 0.0
            self._next_offboard_ts = 0.0
            self._takeoff_started_ts = 0.0
            self._liftoff_since_ts = 0.0
            if not self.px4_auto_arm:
                return "none", {"reason": "auto_arm_disabled"}
            if self.px4_auto_arm_require_armable and not is_armable:
                return "none", {"reason": "wait_armable"}
            if self.px4_auto_arm_require_local_position and not is_local_position_ok:
                return "none", {"reason": "wait_local_position"}
            if float(ts) >= self._next_arm_ts:
                self._next_arm_ts = float(ts) + self.px4_arm_retry_s
                return "arm", {"reason": "auto_arm"}
            return "none", {"reason": "arm_retry_wait"}

        self._next_arm_ts = 0.0

        if not liftoff_confirmed:
            self._next_offboard_ts = 0.0
            if not self.px4_auto_takeoff:
                return "none", {"reason": "auto_takeoff_disabled_wait_liftoff"}
            if self._takeoff_started_ts <= 0.0:
                self._takeoff_started_ts = float(ts)
            takeoff_timed_out = (float(ts) - self._takeoff_started_ts) >= self.px4_takeoff_liftoff_timeout_s
            if float(ts) >= self._next_takeoff_ts:
                self._next_takeoff_ts = float(ts) + self.px4_takeoff_retry_s
                reason = "auto_takeoff"
                if takeoff_timed_out:
                    reason = "auto_takeoff_retry_liftoff_timeout"
                return "takeoff", {
                    "reason": reason,
                    "rel_alt": rel_alt,
                    "in_air_reported": in_air_reported,
                }
            wait_reason = "takeoff_retry_wait"
            if takeoff_timed_out:
                wait_reason = "takeoff_retry_wait_liftoff_timeout"
            return "none", {
                "reason": wait_reason,
                "rel_alt": rel_alt,
                "in_air_reported": in_air_reported,
            }

        self._next_takeoff_ts = 0.0
        if self._takeoff_started_ts <= 0.0:
            self._takeoff_started_ts = float(ts)

        if not self.px4_auto_offboard:
            return "none", {"reason": "auto_offboard_disabled"}

        if offboard_active:
            self._next_offboard_ts = 0.0
            return "none", {"reason": "offboard_active"}

        liftoff_age_s = 0.0
        if self._liftoff_since_ts > 0.0:
            liftoff_age_s = max(0.0, float(ts) - self._liftoff_since_ts)
        if liftoff_age_s < self.px4_offboard_start_delay_after_liftoff_s:
            return "none", {"reason": "wait_offboard_delay_after_liftoff", "liftoff_age_s": liftoff_age_s}

        # Start OFFBOARD only after verified safe altitude.
        # Do not use TAKEOFF-mode timeout fallback here: it can force OFFBOARD
        # at near-zero altitude and lead to ground "crawling".
        if rel_alt < self.px4_offboard_min_relative_alt_m:
            return "none", {"reason": "wait_offboard_altitude", "rel_alt": rel_alt}

        if float(ts) >= self._next_offboard_ts:
            self._next_offboard_ts = float(ts) + self.px4_offboard_retry_s
            return "ensure_offboard", {
                "reason": "auto_offboard",
                "rel_alt": rel_alt,
                "liftoff_age_s": liftoff_age_s,
            }
        return "none", {"reason": "offboard_retry_wait"}

    def _update_terminal_charge_state(self, *, area_ratio: float, err_x: float) -> bool:
        if not (self.mode == "xy" and self.xy_strategy == "zone_track" and self.terminal_charge_enabled):
            self._terminal_charge_active = False
            self._terminal_charge_ready_frames = 0
            return False

        abs_ex = abs(float(err_x))
        can_enter = (
            area_ratio >= self.terminal_charge_enter_area_ratio
            and abs_ex <= self.terminal_charge_align_gate_x
        )
        if can_enter:
            self._terminal_charge_ready_frames += 1
        else:
            self._terminal_charge_ready_frames = 0

        if (not self._terminal_charge_active) and (
            self._terminal_charge_ready_frames >= self.terminal_charge_required_frames
        ):
            self._terminal_charge_active = True

        if self._terminal_charge_active:
            if area_ratio < self.terminal_charge_exit_area_ratio:
                self._terminal_charge_active = False
                self._terminal_charge_ready_frames = 0
            elif abs_ex > self.terminal_charge_exit_align_gate_x:
                self._terminal_charge_active = False
                self._terminal_charge_ready_frames = 0

        return self._terminal_charge_active

    def _compute_xy_alt_hold_vz(
        self,
        *,
        ts: float,
        platform_meta: dict[str, Any] | None,
        vx_cmd: float,
        terminal_charge_active: bool,
    ) -> float:
        self._xy_alt_emergency_active = False
        if not self.xy_alt_hold_enabled:
            self._xy_alt_ref_m = None
            self._xy_alt_last_forward_ts = 0.0
            self._xy_alt_emergency_latched = False
            return 0.0
        if not isinstance(platform_meta, dict):
            self._xy_alt_ref_m = None
            self._xy_alt_last_forward_ts = 0.0
            self._xy_alt_emergency_latched = False
            return 0.0
        if not bool(platform_meta.get("in_air", False)):
            self._xy_alt_ref_m = None
            self._xy_alt_last_forward_ts = 0.0
            self._xy_alt_emergency_latched = False
            return 0.0

        rel_alt = float(platform_meta.get("relative_altitude_m", 0.0) or 0.0)
        vel_down = float(platform_meta.get("vel_down_m_s", 0.0) or 0.0)

        if self.xy_alt_hold_hard_emergency_enabled:
            if rel_alt <= self.xy_alt_hold_hard_emergency_floor_m:
                self._xy_alt_emergency_latched = True
            elif rel_alt >= self.xy_alt_hold_hard_emergency_release_m:
                self._xy_alt_emergency_latched = False
            if self._xy_alt_emergency_latched:
                self._xy_alt_emergency_active = True
                if self._xy_alt_ref_m is None:
                    self._xy_alt_ref_m = rel_alt
                self._xy_alt_last_forward_ts = float(ts)
                return _clamp(self.xy_alt_hold_hard_emergency_vz_up, 0.0, self.xy_alt_hold_max_vz)

        floor_disabled = terminal_charge_active and self.xy_alt_hold_disable_floor_in_terminal_charge
        if floor_disabled and self.xy_alt_hold_terminal_ref_alpha >= 0.999:
            self._xy_alt_ref_m = rel_alt
        if self._xy_alt_ref_m is None:
            # Establish a floor-level reference once airborne.
            # This prevents "ground crawling": if OFFBOARD starts too low,
            # guidance still asks for a climb to safe XY operating altitude.
            if floor_disabled:
                self._xy_alt_ref_m = rel_alt
            else:
                self._xy_alt_ref_m = max(rel_alt, self.xy_alt_hold_min_active_alt_m)

        abs_vx = abs(float(vx_cmd))
        below_floor = (not floor_disabled) and (rel_alt < self.xy_alt_hold_min_active_alt_m)
        if abs_vx >= self.xy_alt_hold_min_forward_m_s or below_floor:
            alt_err = float(self._xy_alt_ref_m) - rel_alt
            # Positive vel_down means descending, so it contributes to upward correction (+vz).
            vz = (self.xy_alt_hold_kp * alt_err) + (self.xy_alt_hold_kd * vel_down)
            self._xy_alt_last_forward_ts = float(ts)
            return _clamp(vz, -self.xy_alt_hold_max_vz, self.xy_alt_hold_max_vz)

        if self._xy_alt_last_forward_ts <= 0.0 or (float(ts) - self._xy_alt_last_forward_ts) >= self.xy_alt_hold_release_s:
            a = self.xy_alt_hold_terminal_ref_alpha if floor_disabled else self.xy_alt_hold_ref_alpha
            if a >= 0.999:
                self._xy_alt_ref_m = rel_alt
            elif a > 1e-6:
                self._xy_alt_ref_m = (1.0 - a) * float(self._xy_alt_ref_m) + a * rel_alt
        return 0.0

    def compute(self, command_input: GuidanceInput) -> GuidanceOutput:
        state = str(command_input.state)
        primary_track = command_input.primary_track
        err_x = command_input.err_x
        err_y = command_input.err_y
        ts = float(command_input.ts or 0.0)
        platform_meta = command_input.platform_meta if isinstance(command_input.platform_meta, dict) else {}

        platform_action, platform_action_payload = self._compute_px4_lifecycle_action(
            ts=ts,
            platform_meta=platform_meta,
        )

        if not self.enabled:
            return self._zero(
                state=state,
                reason="disabled",
                platform_action=platform_action,
                platform_action_payload=platform_action_payload,
            )
        if state not in self.ACTIVE_STATES:
            return self._zero(
                state=state,
                reason="inactive_state",
                platform_action=platform_action,
                platform_action_payload=platform_action_payload,
            )
        if primary_track is None or err_x is None:
            return self._zero(
                state=state,
                reason="no_primary",
                platform_action=platform_action,
                platform_action_payload=platform_action_payload,
            )

        area_ratio, area_err = self._target_area_error(primary_track)
        area_gate_enabled = self._area_gate_enabled()
        vx = 0.0
        vy = 0.0
        yaw_rate = 0.0
        if self.mode == "xy" and self.xy_strategy == "zone_track":
            ex = float(err_x)
            ey = float(err_y or 0.0)
            ux, uy = self._zone_track_normalized_error(
                primary_track=primary_track,
                err_x=ex,
                err_y=ey,
            )
            abs_ux = abs(ux)
            abs_uy = abs(uy)
            centered_x = abs_ux <= self.zone_track_center_deadzone_x
            centered_y = abs_uy <= self.zone_track_center_deadzone_y
            if centered_x and centered_y:
                self._center_lock_frames += 1
            else:
                self._center_lock_frames = 0

            self._zone_track_sign_x = self._sign_with_hysteresis(ux, self._zone_track_sign_x)
            self._zone_track_sign_y = self._sign_with_hysteresis(uy, self._zone_track_sign_y)
            self._zone_track_last_label = self._zone_label(self._zone_track_sign_x, self._zone_track_sign_y)

            h = max(1e-6, self.zone_track_axis_hysteresis)
            # Commands should react to current continuous error, not to zone-label hysteresis.
            # This avoids delayed sign flips and reduces overshoot around center.
            mag_x = max(0.0, abs_ux - h) / max(1e-6, 1.0 - h)
            steer_mag = mag_x ** self.zone_track_steer_exponent
            signed_ux = 0.0
            if steer_mag > 1e-9:
                signed_ux = steer_mag if ux >= 0.0 else -steer_mag
            if centered_x:
                signed_ux = 0.0

            yaw_rate = 0.0 if centered_x else (
                self.yaw_error_sign * self.kp_yaw * signed_ux * self.zone_track_yaw_scale
            )
            vy = 0.0 if centered_x else (
                self.lateral_error_sign * self.kp_lateral * signed_ux * self.zone_track_lateral_scale
            )

            terminal_prev = self._terminal_charge_active
            terminal_charge_active = self._update_terminal_charge_state(
                area_ratio=area_ratio,
                err_x=ex,
            )
            if terminal_charge_active and (not terminal_prev) and isinstance(platform_meta, dict):
                rel_alt = float(platform_meta.get("relative_altitude_m", 0.0) or 0.0)
                self._xy_alt_ref_m = rel_alt

            if terminal_charge_active:
                align_mag = _clamp(1.0 - abs_ux, 0.0, 1.0)
                vx = self.terminal_charge_misaligned_speed + (
                    (self.terminal_charge_speed - self.terminal_charge_misaligned_speed) * align_mag
                )
                yaw_rate *= self.terminal_charge_yaw_scale
                vy *= self.terminal_charge_lateral_scale
            else:
                if self.zone_track_use_area_gate and area_gate_enabled and (area_ratio >= self.min_distance_area_ratio):
                    vx = 0.0
                else:
                    align_x = _clamp(1.0 - abs_ux, 0.0, 1.0)
                    align_y = _clamp(1.0 - abs_uy, 0.0, 1.0)
                    w_x = self.zone_track_forward_align_weight_x
                    w_y = 1.0 - w_x
                    # Keep approach fast, but reduce forward pressure while steering hard.
                    align = _clamp((w_x * align_x) + (w_y * align_y), 0.0, 1.0)
                    steer_brake = 1.0 - (self.zone_track_forward_steer_brake * _clamp(abs(signed_ux), 0.0, 1.0))
                    align = _clamp(align * steer_brake, 0.0, 1.0)
                    speed_span = self.zone_track_forward_speed_max - self.zone_track_forward_speed_min
                    vx = self.zone_track_forward_speed_min + speed_span * (
                        align ** self.zone_track_forward_speed_exponent
                    )
        else:
            self._center_lock_frames = 0
            self._terminal_charge_active = False
            self._terminal_charge_ready_frames = 0
            self._zone_track_sign_x = 0
            self._zone_track_sign_y = 0
            self._zone_track_last_label = "center"

        vz = 0.0
        if self.mode == "xyz" and err_y is not None and not self._in_deadzone(float(err_y), self.deadzone_y):
            vz = -self.kp_vertical * float(err_y)
        elif self.mode == "xy":
            vz = self._compute_xy_alt_hold_vz(
                ts=ts,
                platform_meta=platform_meta,
                vx_cmd=float(vx),
                terminal_charge_active=bool(self._terminal_charge_active),
            )

        vx = _clamp(vx, -self.max_vx, self.max_vx)
        vy = _clamp(vy, -self.max_vy, self.max_vy)
        vz = _clamp(vz, -self.max_vz, self.max_vz)
        yaw_rate = _clamp(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        self._vx_smoothed = self._smooth(vx, self._vx_smoothed)
        self._vy_smoothed = self._smooth(vy, self._vy_smoothed)
        self._vz_smoothed = self._smooth(vz, self._vz_smoothed)
        self._yaw_rate_smoothed = self._smooth(yaw_rate, self._yaw_rate_smoothed)

        reason = "tracking"
        if self.mode == "xy" and self.xy_strategy == "zone_track":
            if self._xy_alt_emergency_active:
                reason = "emergency_floor_climb"
            elif self._terminal_charge_active:
                reason = f"zone_track_terminal_{self._zone_track_last_label}"
            elif area_gate_enabled and area_ratio >= self.min_distance_area_ratio and self.zone_track_use_area_gate:
                reason = "zone_track_hold_area"
            else:
                reason = f"zone_track_{self._zone_track_last_label}"

        return GuidanceOutput(
            active=True,
            mode=self.mode,
            xy_strategy=self.xy_strategy,
            state=state,
            reason=reason,
            vx_body=float(self._vx_smoothed),
            vy_body=float(self._vy_smoothed),
            vz=float(self._vz_smoothed),
            yaw_rate=float(self._yaw_rate_smoothed),
            area_ratio=float(area_ratio),
            area_error=float(area_err),
            center_lock_frames=int(self._center_lock_frames),
            platform_action=platform_action,
            platform_action_payload=dict(platform_action_payload),
        )

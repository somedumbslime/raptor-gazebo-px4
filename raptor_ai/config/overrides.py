from __future__ import annotations

import copy
from typing import Any


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = dict(dst)
    for key, value in src.items():
        prev = out.get(key)
        if isinstance(prev, dict) and isinstance(value, dict):
            out[key] = _deep_update(prev, value)
        else:
            out[key] = value
    return out


def _parse_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    parts = [p.strip() for p in str(raw).split(",")]
    return [p for p in parts if p]


_FOLLOW_PROFILES: dict[str, dict[str, Any]] = {
    "safe": {
        "mode": "xy",
        "xy_strategy": "zone_track",
        "deadzone_x": 0.032,
        "kp_lateral": 0.85,
        "kp_yaw": 0.95,
        "max_vx": 0.60,
        "max_vy": 0.30,
        "max_yaw_rate": 0.34,
        "zone_track_box_scale_x": 2.3,
        "zone_track_box_scale_y": 2.0,
        "zone_track_axis_hysteresis": 0.08,
        "zone_track_forward_speed_min": 0.20,
        "zone_track_forward_speed_max": 0.45,
        "zone_track_forward_speed_exponent": 1.05,
        "zone_track_steer_exponent": 1.50,
        "zone_track_forward_align_weight_x": 0.82,
        "zone_track_forward_steer_brake": 0.55,
        "zone_track_lateral_scale": 0.65,
        "zone_track_yaw_scale": 0.95,
        "zone_track_center_deadzone_x": 0.10,
        "zone_track_center_deadzone_y": 0.12,
        "zone_track_use_area_gate": False,
        "terminal_charge_enter_area_ratio": 0.075,
        "terminal_charge_exit_area_ratio": 0.060,
        "terminal_charge_required_frames": 5,
        "terminal_charge_speed": 0.52,
        "terminal_charge_misaligned_speed": 0.16,
        "xy_alt_hold_min_active_alt_m": 1.40,
        "xy_alt_hold_kp": 0.85,
        "xy_alt_hold_max_vz": 0.25,
        "smoothing_alpha": 0.22,
    },
    "balanced": {
        "mode": "xy",
        "xy_strategy": "zone_track",
        "deadzone_x": 0.026,
        "kp_lateral": 1.00,
        "kp_yaw": 1.15,
        "max_vx": 1.00,
        "max_vy": 0.48,
        "max_yaw_rate": 0.55,
        "zone_track_box_scale_x": 2.2,
        "zone_track_box_scale_y": 1.9,
        "zone_track_axis_hysteresis": 0.06,
        "zone_track_forward_speed_min": 0.34,
        "zone_track_forward_speed_max": 0.85,
        "zone_track_forward_speed_exponent": 0.85,
        "zone_track_steer_exponent": 1.35,
        "zone_track_forward_align_weight_x": 0.85,
        "zone_track_forward_steer_brake": 0.50,
        "zone_track_lateral_scale": 0.85,
        "zone_track_yaw_scale": 1.15,
        "zone_track_center_deadzone_x": 0.08,
        "zone_track_center_deadzone_y": 0.10,
        "zone_track_use_area_gate": False,
        "terminal_charge_enter_area_ratio": 0.070,
        "terminal_charge_exit_area_ratio": 0.055,
        "terminal_charge_required_frames": 4,
        "terminal_charge_speed": 0.95,
        "terminal_charge_misaligned_speed": 0.26,
        "xy_alt_hold_min_active_alt_m": 1.20,
        "xy_alt_hold_kp": 0.75,
        "xy_alt_hold_max_vz": 0.30,
        "smoothing_alpha": 0.27,
    },
    "aggressive": {
        "mode": "xy",
        "xy_strategy": "zone_track",
        "deadzone_x": 0.020,
        "kp_lateral": 1.15,
        "kp_yaw": 1.30,
        "max_vx": 1.50,
        "max_vy": 0.55,
        "max_yaw_rate": 0.62,
        "zone_track_box_scale_x": 2.0,
        "zone_track_box_scale_y": 1.8,
        "zone_track_axis_hysteresis": 0.04,
        "zone_track_forward_speed_min": 0.45,
        "zone_track_forward_speed_max": 1.15,
        "zone_track_forward_speed_exponent": 0.95,
        "zone_track_steer_exponent": 1.45,
        "zone_track_forward_align_weight_x": 0.88,
        "zone_track_forward_steer_brake": 0.62,
        "zone_track_lateral_scale": 1.00,
        "zone_track_yaw_scale": 1.35,
        "zone_track_center_deadzone_x": 0.06,
        "zone_track_center_deadzone_y": 0.08,
        "zone_track_use_area_gate": False,
        "terminal_charge_enter_area_ratio": 0.065,
        "terminal_charge_exit_area_ratio": 0.050,
        "terminal_charge_required_frames": 2,
        "terminal_charge_speed": 1.20,
        "terminal_charge_misaligned_speed": 0.30,
        "xy_alt_hold_min_active_alt_m": 1.10,
        "xy_alt_hold_kp": 0.65,
        "xy_alt_hold_max_vz": 0.34,
        "smoothing_alpha": 0.30,
    },
}

_FOLLOW_PROFILE_PLATFORM_PX4: dict[str, dict[str, Any]] = {
    # Lower smoothing lag for safer turn-in without oscillation.
    "safe": {"cmd_smoothing_alpha": 0.28},
    "balanced": {"cmd_smoothing_alpha": 0.35},
    "aggressive": {"cmd_smoothing_alpha": 0.42},
}


def apply_runtime_overrides(
    cfg: dict[str, Any],
    *,
    platform_type: str | None = None,
    detector_type: str | None = None,
    selector_backend: str | None = None,
    guidance_backend: str | None = None,
    yolo_model_path: str | None = None,
    yolo_target_classes_csv: str | None = None,
    follow_profile: str | None = None,
    follow_mode: str | None = None,
    state_lost_frame_threshold: int | None = None,
    state_reacquire_threshold: int | None = None,
    follow_enabled: bool | None = None,
    follow_xy_strategy: str | None = None,
    follow_yaw_error_sign: float | None = None,
    follow_lateral_error_sign: float | None = None,
    px4_cv_only: bool | None = None,
    px4_auto_arm: bool | None = None,
    px4_auto_takeoff: bool | None = None,
    px4_auto_arm_require_armable: bool | None = None,
    px4_auto_arm_require_local_position: bool | None = None,
    px4_takeoff_altitude_m: float | None = None,
    px4_takeoff_confirm_alt_m: float | None = None,
    px4_offboard_min_relative_alt_m: float | None = None,
    px4_offboard_start_delay_after_liftoff_s: float | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)

    if follow_profile:
        profile = str(follow_profile).strip().lower()
        patch = _FOLLOW_PROFILES.get(profile)
        if patch is None:
            supported = ", ".join(sorted(_FOLLOW_PROFILES))
            raise ValueError(f"Unsupported follow_profile: {profile}. Supported: {supported}")
        profile_patch = dict(patch)
        profile_patch["profile"] = profile
        out = _deep_update(out, {"follow": profile_patch})
        px4_patch = _FOLLOW_PROFILE_PLATFORM_PX4.get(profile)
        if px4_patch:
            out = _deep_update(out, {"platform": {"px4": dict(px4_patch)}})

    if platform_type:
        out = _deep_update(out, {"platform": {"type": str(platform_type).strip().lower()}})

    if detector_type:
        out = _deep_update(out, {"detector": {"type": str(detector_type).strip().lower()}})

    if selector_backend:
        out = _deep_update(out, {"selector": {"backend": str(selector_backend).strip().lower()}})

    if guidance_backend:
        out = _deep_update(out, {"guidance": {"backend": str(guidance_backend).strip().lower()}})

    if yolo_model_path:
        out = _deep_update(out, {"detector": {"yolo_onnx": {"model_path": str(yolo_model_path)}}})

    target_classes = _parse_list(yolo_target_classes_csv)
    if target_classes:
        out = _deep_update(out, {"detector": {"yolo_onnx": {"target_classes": target_classes}}})

    if follow_mode:
        out = _deep_update(out, {"state_machine": {"follow_mode": str(follow_mode).strip().lower()}})

    if state_lost_frame_threshold is not None:
        out = _deep_update(out, {"state_machine": {"lost_frame_threshold": int(state_lost_frame_threshold)}})

    if state_reacquire_threshold is not None:
        out = _deep_update(out, {"state_machine": {"reacquire_threshold": int(state_reacquire_threshold)}})

    if follow_enabled is not None:
        out = _deep_update(out, {"follow": {"enabled": bool(follow_enabled)}})

    if follow_xy_strategy:
        strategy = str(follow_xy_strategy).strip().lower()
        if strategy != "zone_track":
            raise ValueError("Only follow.xy_strategy='zone_track' is supported")
        out = _deep_update(out, {"follow": {"xy_strategy": strategy}})

    if follow_yaw_error_sign is not None:
        out = _deep_update(out, {"follow": {"yaw_error_sign": float(follow_yaw_error_sign)}})

    if follow_lateral_error_sign is not None:
        out = _deep_update(out, {"follow": {"lateral_error_sign": float(follow_lateral_error_sign)}})

    if px4_cv_only is not None:
        out = _deep_update(out, {"platform": {"px4": {"cv_only": bool(px4_cv_only)}}})

    if px4_auto_arm is not None:
        out = _deep_update(out, {"platform": {"px4": {"auto_arm": bool(px4_auto_arm)}}})

    if px4_auto_takeoff is not None:
        out = _deep_update(out, {"platform": {"px4": {"auto_takeoff": bool(px4_auto_takeoff)}}})

    if px4_auto_arm_require_armable is not None:
        out = _deep_update(out, {"platform": {"px4": {"auto_arm_require_armable": bool(px4_auto_arm_require_armable)}}})

    if px4_auto_arm_require_local_position is not None:
        out = _deep_update(
            out,
            {"platform": {"px4": {"auto_arm_require_local_position": bool(px4_auto_arm_require_local_position)}}},
        )

    if px4_takeoff_altitude_m is not None:
        out = _deep_update(out, {"platform": {"px4": {"takeoff_altitude_m": float(px4_takeoff_altitude_m)}}})

    if px4_takeoff_confirm_alt_m is not None:
        out = _deep_update(out, {"platform": {"px4": {"takeoff_confirm_alt_m": float(px4_takeoff_confirm_alt_m)}}})

    if px4_offboard_min_relative_alt_m is not None:
        out = _deep_update(
            out,
            {"platform": {"px4": {"offboard_min_relative_alt_m": float(px4_offboard_min_relative_alt_m)}}},
        )

    if px4_offboard_start_delay_after_liftoff_s is not None:
        out = _deep_update(
            out,
            {
                "platform": {
                    "px4": {
                        "offboard_start_delay_after_liftoff_s": float(
                            px4_offboard_start_delay_after_liftoff_s
                        )
                    }
                }
            },
        )

    return out

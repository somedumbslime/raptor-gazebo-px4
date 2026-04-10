from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TG_PATH = ROOT / "target-guidance"
if str(TG_PATH) not in sys.path:
    sys.path.insert(0, str(TG_PATH))

from target_guidance.contracts import GuidanceInput
from target_guidance.policy_v1 import TargetGuidancePolicyV1


def _track(area: float = 1000.0) -> dict:
    return {
        "track_id": 1,
        "bbox_xyxy": [100.0, 100.0, 180.0, 220.0],
        "center": [140.0, 160.0],
        "area": area,
        "frame_w": 640,
        "frame_h": 480,
    }


def _mk_policy(**overrides: object) -> TargetGuidancePolicyV1:
    cfg = {
        "enabled": True,
        "mode": "xy",
        "xy_strategy": "zone_track",
        "smoothing_alpha": 1.0,
        "deadzone_x": 0.01,
        "deadzone_area": 0.0,
        "target_area_ratio": 0.2,
        "min_distance_area_ratio": 0.12,
        "zone_track_box_scale_x": 2.2,
        "zone_track_box_scale_y": 2.0,
        "zone_track_axis_hysteresis": 0.06,
        "zone_track_forward_speed_min": 0.20,
        "zone_track_forward_speed_max": 0.60,
        "zone_track_forward_speed_exponent": 0.9,
        "zone_track_center_deadzone_x": 0.08,
        "zone_track_center_deadzone_y": 0.10,
        "zone_track_use_area_gate": False,
        "terminal_charge_enabled": True,
        "terminal_charge_enter_area_ratio": 0.07,
        "terminal_charge_exit_area_ratio": 0.05,
        "terminal_charge_align_gate_x": 0.06,
        "terminal_charge_exit_align_gate_x": 0.12,
        "terminal_charge_required_frames": 4,
        "terminal_charge_speed": 0.8,
        "terminal_charge_misaligned_speed": 0.2,
        "kp_lateral": 1.0,
        "kp_yaw": 1.0,
        "yaw_error_sign": 1.0,
        "lateral_error_sign": 1.0,
        "max_vx": 2.0,
        "max_vy": 2.0,
        "max_vz": 1.0,
        "max_yaw_rate": 2.0,
        "xy_alt_hold_enabled": True,
        "xy_alt_hold_min_forward_m_s": 0.1,
        "xy_alt_hold_min_active_alt_m": 0.8,
        "xy_alt_hold_kp": 1.0,
        "xy_alt_hold_kd": 0.5,
        "xy_alt_hold_max_vz": 0.4,
        "xy_alt_hold_hard_emergency_enabled": True,
        "xy_alt_hold_hard_emergency_floor_m": 0.28,
        "xy_alt_hold_hard_emergency_release_m": 0.40,
        "xy_alt_hold_hard_emergency_vz_up": 0.3,
        "px4_lifecycle_enabled": True,
        "px4_auto_arm": True,
        "px4_auto_takeoff": True,
        "px4_auto_offboard": True,
        "px4_auto_arm_require_armable": True,
        "px4_auto_arm_require_local_position": False,
        "px4_offboard_min_relative_alt_m": 0.6,
        "px4_takeoff_confirm_alt_m": 0.6,
        "px4_arm_retry_s": 0.01,
        "px4_takeoff_retry_s": 0.01,
        "px4_takeoff_liftoff_timeout_s": 2.0,
        "px4_offboard_start_delay_after_liftoff_s": 0.0,
        "px4_offboard_retry_s": 0.01,
    }
    cfg.update(overrides)
    return TargetGuidancePolicyV1(cfg)


def test_lifecycle_requests_arm_when_ready() -> None:
    policy = _mk_policy()
    out = policy.compute(
        GuidanceInput(
            state="SEARCHING",
            primary_track=None,
            err_x=None,
            err_y=None,
            dt=0.05,
            ts=10.0,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": False,
                "in_air": False,
                "is_armable": True,
                "is_local_position_ok": True,
                "flight_mode": "HOLD",
            },
        )
    )
    assert out.platform_action == "arm"


def test_lifecycle_requests_takeoff_when_armed_but_grounded() -> None:
    policy = _mk_policy()
    out = policy.compute(
        GuidanceInput(
            state="SEARCHING",
            primary_track=None,
            err_x=None,
            err_y=None,
            dt=0.05,
            ts=20.0,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": True,
                "in_air": False,
                "is_armable": True,
                "is_local_position_ok": True,
                "flight_mode": "HOLD",
            },
        )
    )
    assert out.platform_action == "takeoff"


def test_lifecycle_requests_offboard_after_takeoff() -> None:
    policy = _mk_policy()
    out = policy.compute(
        GuidanceInput(
            state="SEARCHING",
            primary_track=None,
            err_x=None,
            err_y=None,
            dt=0.05,
            ts=30.0,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": True,
                "in_air": True,
                "is_armable": True,
                "is_local_position_ok": True,
                "flight_mode": "TAKEOFF",
                "relative_altitude_m": 1.0,
                "offboard_started": False,
                "offboard_mode_active": False,
            },
        )
    )
    assert out.platform_action == "ensure_offboard"


def test_lifecycle_retries_takeoff_until_altitude_confirmed_even_if_in_air_reported_true() -> None:
    policy = _mk_policy()
    out = policy.compute(
        GuidanceInput(
            state="SEARCHING",
            primary_track=None,
            err_x=None,
            err_y=None,
            dt=0.05,
            ts=30.0,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": True,
                "in_air": True,
                "is_armable": True,
                "is_local_position_ok": True,
                "flight_mode": "TAKEOFF",
                "relative_altitude_m": 0.02,
                "offboard_started": False,
                "offboard_mode_active": False,
            },
        )
    )
    assert out.platform_action == "takeoff"


def test_xy_alt_hold_adds_positive_vz_when_altitude_drops_while_moving_forward() -> None:
    policy = _mk_policy()
    # First frame establishes altitude reference.
    _ = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=1000.0),
            err_x=0.0,
            err_y=0.0,
            dt=0.05,
            ts=40.0,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": True,
                "in_air": True,
                "flight_mode": "OFFBOARD",
                "offboard_started": True,
                "offboard_mode_active": True,
                "relative_altitude_m": 2.0,
                "vel_down_m_s": 0.0,
            },
        )
    )
    # Second frame drops altitude and adds descent velocity.
    out = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=1000.0),
            err_x=0.0,
            err_y=0.0,
            dt=0.05,
            ts=40.1,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": True,
                "in_air": True,
                "flight_mode": "OFFBOARD",
                "offboard_started": True,
                "offboard_mode_active": True,
                "relative_altitude_m": 1.7,
                "vel_down_m_s": 0.2,
            },
        )
    )

    assert out.active is True
    assert float(out.vx_body) > 0.0
    assert float(out.vz) > 0.0


def test_xy_alt_hold_climbs_when_below_min_operating_altitude() -> None:
    policy = _mk_policy()

    out = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=1000.0),
            err_x=0.0,
            err_y=0.0,
            dt=0.05,
            ts=50.0,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": True,
                "in_air": True,
                "flight_mode": "OFFBOARD",
                "offboard_started": True,
                "offboard_mode_active": True,
                "relative_altitude_m": 0.45,
                "vel_down_m_s": 0.0,
            },
        )
    )

    assert out.active is True
    # Safety floor: guidance must request upward motion even if horizontal speed is low.
    assert float(out.vz) > 0.0


def test_terminal_charge_keeps_forward_when_area_gate_would_hold() -> None:
    policy = _mk_policy(
        terminal_charge_required_frames=1,
        zone_track_use_area_gate=True,
    )
    frame_area = 640.0 * 480.0

    out = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=frame_area * 0.18),
            err_x=0.01,
            err_y=0.0,
            dt=0.05,
            ts=60.0,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": True,
                "in_air": True,
                "flight_mode": "OFFBOARD",
                "offboard_started": True,
                "offboard_mode_active": True,
                "relative_altitude_m": 1.4,
                "vel_down_m_s": 0.0,
            },
        )
    )

    assert out.reason.startswith("zone_track_terminal_")
    assert float(out.vx_body) > 0.0


def test_terminal_charge_exits_on_large_alignment_error() -> None:
    policy = _mk_policy(
        terminal_charge_required_frames=1,
        zone_track_use_area_gate=True,
    )
    frame_area = 640.0 * 480.0

    _ = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=frame_area * 0.18),
            err_x=0.01,
            err_y=0.0,
            dt=0.05,
            ts=61.0,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": True,
                "in_air": True,
                "flight_mode": "OFFBOARD",
                "offboard_started": True,
                "offboard_mode_active": True,
                "relative_altitude_m": 1.4,
                "vel_down_m_s": 0.0,
            },
        )
    )

    out = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=frame_area * 0.18),
            err_x=0.22,
            err_y=0.0,
            dt=0.05,
            ts=61.05,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": True,
                "in_air": True,
                "flight_mode": "OFFBOARD",
                "offboard_started": True,
                "offboard_mode_active": True,
                "relative_altitude_m": 1.4,
                "vel_down_m_s": 0.0,
            },
        )
    )

    assert out.reason == "zone_track_hold_area"
    assert float(out.vx_body) == 0.0


def test_hard_emergency_floor_forces_upward_velocity() -> None:
    policy = _mk_policy(
        terminal_charge_required_frames=1,
        xy_alt_hold_hard_emergency_floor_m=0.30,
        xy_alt_hold_hard_emergency_release_m=0.45,
        xy_alt_hold_hard_emergency_vz_up=0.35,
    )
    frame_area = 640.0 * 480.0

    out = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=frame_area * 0.20),
            err_x=0.0,
            err_y=0.0,
            dt=0.05,
            ts=62.0,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": True,
                "in_air": True,
                "flight_mode": "OFFBOARD",
                "offboard_started": True,
                "offboard_mode_active": True,
                "relative_altitude_m": 0.22,
                "vel_down_m_s": 0.0,
            },
        )
    )

    assert out.reason == "emergency_floor_climb"
    assert float(out.vz) > 0.0


def test_terminal_charge_disables_min_active_floor_pull_up() -> None:
    policy = _mk_policy(
        terminal_charge_required_frames=1,
        xy_alt_hold_min_active_alt_m=1.2,
        xy_alt_hold_disable_floor_in_terminal_charge=True,
        xy_alt_hold_hard_emergency_floor_m=0.05,
        xy_alt_hold_hard_emergency_release_m=0.08,
    )
    frame_area = 640.0 * 480.0

    out = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=frame_area * 0.20),
            err_x=0.0,
            err_y=0.0,
            dt=0.05,
            ts=63.0,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": True,
                "in_air": True,
                "flight_mode": "OFFBOARD",
                "offboard_started": True,
                "offboard_mode_active": True,
                "relative_altitude_m": 0.70,
                "vel_down_m_s": 0.0,
            },
        )
    )

    assert out.reason.startswith("zone_track_terminal_")
    assert abs(float(out.vz)) < 1e-6


def test_min_distance_area_zero_disables_area_gate_not_freeze() -> None:
    policy = _mk_policy(
        min_distance_area_ratio=0.0,
        zone_track_use_area_gate=True,
        terminal_charge_enabled=False,
    )
    frame_area = 640.0 * 480.0

    out = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=frame_area * 0.20),  # very close target
            err_x=0.03,
            err_y=0.0,
            dt=0.05,
            ts=64.0,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": True,
                "in_air": True,
                "flight_mode": "OFFBOARD",
                "offboard_started": True,
                "offboard_mode_active": True,
                "relative_altitude_m": 1.2,
                "vel_down_m_s": 0.0,
            },
        )
    )

    assert out.reason != "zone_track_hold_area"
    assert float(out.vx_body) > 0.0


def test_zone_track_moves_forward_while_steering() -> None:
    policy = _mk_policy(
        xy_strategy="zone_track",
        terminal_charge_enabled=False,
        zone_track_box_scale_x=2.2,
        zone_track_box_scale_y=2.0,
        zone_track_forward_speed_min=0.14,
        zone_track_forward_speed_max=0.50,
        zone_track_center_deadzone_x=0.08,
        zone_track_center_deadzone_y=0.10,
    )

    out = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=1000.0),
            err_x=0.18,
            err_y=0.0,
            dt=0.05,
            ts=65.0,
            platform_meta={
                "platform_type": "px4",
                "connected": True,
                "armed": True,
                "in_air": True,
                "flight_mode": "OFFBOARD",
                "offboard_started": True,
                "offboard_mode_active": True,
                "relative_altitude_m": 1.4,
                "vel_down_m_s": 0.0,
            },
        )
    )

    assert out.reason == "zone_track_right"
    assert float(out.vx_body) > 0.0
    assert float(out.vy_body) > 0.0
    assert float(out.yaw_rate) > 0.0


def test_zone_track_hysteresis_keeps_zone_near_axis_boundary() -> None:
    policy = _mk_policy(
        xy_strategy="zone_track",
        terminal_charge_enabled=False,
        zone_track_axis_hysteresis=0.06,
        zone_track_box_scale_x=2.2,
        zone_track_box_scale_y=2.0,
        zone_track_center_deadzone_x=0.08,
        zone_track_center_deadzone_y=0.10,
    )
    platform_meta = {
        "platform_type": "px4",
        "connected": True,
        "armed": True,
        "in_air": True,
        "flight_mode": "OFFBOARD",
        "offboard_started": True,
        "offboard_mode_active": True,
        "relative_altitude_m": 1.4,
        "vel_down_m_s": 0.0,
    }

    _ = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=1000.0),
            err_x=0.20,
            err_y=0.0,
            dt=0.05,
            ts=66.0,
            platform_meta=platform_meta,
        )
    )
    out = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=1000.0),
            err_x=-0.01,
            err_y=0.0,
            dt=0.05,
            ts=66.05,
            platform_meta=platform_meta,
        )
    )

    assert out.reason == "zone_track_right"
    assert "left" not in out.reason


def test_zone_track_steering_softens_near_center() -> None:
    policy = _mk_policy(
        terminal_charge_enabled=False,
        smoothing_alpha=1.0,
        zone_track_axis_hysteresis=0.06,
        zone_track_steer_exponent=1.45,
    )
    platform_meta = {
        "platform_type": "px4",
        "connected": True,
        "armed": True,
        "in_air": True,
        "flight_mode": "OFFBOARD",
        "offboard_started": True,
        "offboard_mode_active": True,
        "relative_altitude_m": 1.4,
        "vel_down_m_s": 0.0,
    }

    out_far = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=1000.0),
            err_x=0.22,
            err_y=0.0,
            dt=0.05,
            ts=67.0,
            platform_meta=platform_meta,
        )
    )
    out_near = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=1000.0),
            err_x=0.05,
            err_y=0.0,
            dt=0.05,
            ts=67.05,
            platform_meta=platform_meta,
        )
    )

    assert abs(float(out_near.yaw_rate)) < abs(float(out_far.yaw_rate))
    assert abs(float(out_near.vy_body)) < abs(float(out_far.vy_body))


def test_zone_track_forward_reduces_when_turning_hard() -> None:
    policy = _mk_policy(
        terminal_charge_enabled=False,
        smoothing_alpha=1.0,
        zone_track_forward_speed_min=0.30,
        zone_track_forward_speed_max=0.90,
        zone_track_forward_steer_brake=0.60,
    )
    platform_meta = {
        "platform_type": "px4",
        "connected": True,
        "armed": True,
        "in_air": True,
        "flight_mode": "OFFBOARD",
        "offboard_started": True,
        "offboard_mode_active": True,
        "relative_altitude_m": 1.4,
        "vel_down_m_s": 0.0,
    }

    out_aligned = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=1000.0),
            err_x=0.01,
            err_y=0.0,
            dt=0.05,
            ts=68.0,
            platform_meta=platform_meta,
        )
    )
    out_turning = policy.compute(
        GuidanceInput(
            state="TRACKING_XY",
            primary_track=_track(area=1000.0),
            err_x=0.25,
            err_y=0.0,
            dt=0.05,
            ts=68.05,
            platform_meta=platform_meta,
        )
    )

    assert float(out_turning.vx_body) < float(out_aligned.vx_body)

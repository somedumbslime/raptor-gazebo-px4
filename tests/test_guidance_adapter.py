from __future__ import annotations

from raptor_ai.control.follow_controller import FollowController
from raptor_ai.control.guidance_adapter import GuidanceAdapter


def _track(area: float = 1000.0) -> dict:
    return {
        "track_id": 1,
        "bbox_xyxy": [100.0, 100.0, 180.0, 220.0],
        "center": [140.0, 160.0],
        "area": area,
        "frame_w": 640,
        "frame_h": 480,
    }


def test_guidance_adapter_legacy_internal_matches_follow_controller() -> None:
    follow_cfg = {
        "enabled": True,
        "mode": "xy",
        "xy_strategy": "forward_lock",
        "forward_lock_speed": 0.3,
        "forward_lock_required_frames": 2,
        "forward_lock_deadzone_x": 0.05,
        "kp_yaw": 1.0,
        "kp_lateral": 1.0,
        "smoothing_alpha": 1.0,
        "target_area_ratio": 0.2,
        "min_distance_area_ratio": 0.12,
        "deadzone_x": 0.01,
        "deadzone_area": 0.0,
    }
    legacy = FollowController(follow_cfg)
    adapter = GuidanceAdapter(follow_cfg=follow_cfg, guidance_cfg={"backend": "legacy_internal"})

    for ex in (0.2, 0.04, 0.02):
        expected = legacy.compute(
            state="TRACKING_XY",
            primary_track=_track(),
            err_x=ex,
            err_y=0.0,
            dt=0.05,
        )
        got = adapter.compute(
            state="TRACKING_XY",
            primary_track=_track(),
            err_x=ex,
            err_y=0.0,
            dt=0.05,
        )
        assert got == expected


def test_guidance_adapter_target_guidance_backend_computes() -> None:
    follow_cfg = {
        "enabled": True,
        "mode": "xy",
        "xy_strategy": "zone_track",
        "zone_track_box_scale_x": 2.2,
        "zone_track_box_scale_y": 2.0,
        "zone_track_forward_speed_min": 0.25,
        "zone_track_forward_speed_max": 0.60,
        "zone_track_center_deadzone_x": 0.08,
        "zone_track_center_deadzone_y": 0.10,
        "kp_yaw": 1.0,
        "kp_lateral": 1.0,
        "yaw_error_sign": 1.0,
        "smoothing_alpha": 1.0,
        "min_distance_area_ratio": 0.12,
        "zone_track_use_area_gate": False,
        "terminal_charge_enabled": False,
    }
    guidance_cfg = {
        "backend": "target_guidance",
        "external_callable": "target_guidance.entrypoint:create_policy",
        "external_pythonpath": ["target-guidance"],
    }
    adapter = GuidanceAdapter(follow_cfg=follow_cfg, guidance_cfg=guidance_cfg)
    out = adapter.compute(
        state="TRACKING_XY",
        primary_track=_track(),
        err_x=0.2,
        err_y=0.0,
        dt=0.05,
    )

    assert out["active"] is True
    assert out["mode"] == "xy"
    assert out["xy_strategy"] == "zone_track"
    assert str(out["reason"]).startswith("zone_track_")
    assert float(out["vx_body"]) > 0.0
    assert float(out["yaw_rate"]) > 0.0


def test_guidance_adapter_target_guidance_zone_track_area_gate_hold() -> None:
    follow_cfg = {
        "enabled": True,
        "mode": "xy",
        "xy_strategy": "zone_track",
        "zone_track_box_scale_x": 2.2,
        "zone_track_box_scale_y": 2.0,
        "zone_track_forward_speed_min": 0.22,
        "zone_track_forward_speed_max": 0.60,
        "zone_track_center_deadzone_x": 0.08,
        "zone_track_center_deadzone_y": 0.10,
        "zone_track_use_area_gate": True,
        "kp_yaw": 1.0,
        "kp_lateral": 1.0,
        "yaw_error_sign": 1.0,
        "lateral_error_sign": 1.0,
        "smoothing_alpha": 1.0,
        "min_distance_area_ratio": 0.12,
        "terminal_charge_enabled": False,
    }
    guidance_cfg = {
        "backend": "target_guidance",
        "external_callable": "target_guidance.entrypoint:create_policy",
        "external_pythonpath": ["target-guidance"],
    }
    adapter = GuidanceAdapter(follow_cfg=follow_cfg, guidance_cfg=guidance_cfg)
    out = adapter.compute(
        state="TRACKING_XY",
        primary_track=_track(area=(640 * 480 * 0.20)),
        err_x=0.03,
        err_y=0.0,
        dt=0.05,
    )

    assert out["active"] is True
    assert out["xy_strategy"] == "zone_track"
    assert out["reason"] == "zone_track_hold_area"
    assert float(out["vx_body"]) == 0.0


def test_guidance_adapter_target_guidance_zone_track() -> None:
    follow_cfg = {
        "enabled": True,
        "mode": "xy",
        "xy_strategy": "zone_track",
        "zone_track_box_scale_x": 2.2,
        "zone_track_box_scale_y": 2.0,
        "zone_track_forward_speed_min": 0.14,
        "zone_track_forward_speed_max": 0.50,
        "zone_track_center_deadzone_x": 0.08,
        "zone_track_center_deadzone_y": 0.10,
        "kp_yaw": 1.0,
        "kp_lateral": 1.0,
        "yaw_error_sign": 1.0,
        "lateral_error_sign": 1.0,
        "smoothing_alpha": 1.0,
        "target_area_ratio": 0.2,
        "min_distance_area_ratio": 0.12,
        "deadzone_x": 0.01,
        "deadzone_area": 0.0,
        "terminal_charge_enabled": False,
    }
    guidance_cfg = {
        "backend": "target_guidance",
        "external_callable": "target_guidance.entrypoint:create_policy",
        "external_pythonpath": ["target-guidance"],
    }
    adapter = GuidanceAdapter(follow_cfg=follow_cfg, guidance_cfg=guidance_cfg)
    out = adapter.compute(
        state="TRACKING_XY",
        primary_track=_track(),
        err_x=0.18,
        err_y=0.0,
        dt=0.05,
    )

    assert out["active"] is True
    assert out["xy_strategy"] == "zone_track"
    assert str(out["reason"]).startswith("zone_track_")
    assert float(out["vx_body"]) > 0.0
    assert float(out["vy_body"]) > 0.0
    assert float(out["yaw_rate"]) > 0.0

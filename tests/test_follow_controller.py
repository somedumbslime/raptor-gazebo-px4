from __future__ import annotations

from raptor_ai.control.follow_controller import FollowController


def _track(area: float = 1000.0) -> dict:
    return {
        "track_id": 1,
        "bbox_xyxy": [100.0, 100.0, 180.0, 220.0],
        "center": [140.0, 160.0],
        "area": area,
        "frame_w": 640,
        "frame_h": 480,
    }


def test_follow_controller_yaw_only_xy_strategy() -> None:
    ctrl = FollowController(
        {
            "enabled": True,
            "mode": "xy",
            "xy_strategy": "yaw_only",
            "kp_yaw": 1.0,
            "smoothing_alpha": 1.0,
            "deadzone_x": 0.01,
            "max_yaw_rate": 2.0,
        }
    )
    out = ctrl.compute(
        state="TRACKING_XY",
        primary_track=_track(),
        err_x=0.2,
        err_y=0.0,
        dt=0.05,
    )

    assert out["active"] is True
    assert out["mode"] == "xy"
    assert out["xy_strategy"] == "yaw_only"
    assert abs(float(out["vx_body"])) < 1e-6
    assert abs(float(out["vy_body"])) < 1e-6
    assert float(out["yaw_rate"]) < 0.0


def test_follow_controller_yaw_sign_can_be_flipped() -> None:
    ctrl = FollowController(
        {
            "enabled": True,
            "mode": "xy",
            "xy_strategy": "yaw_only",
            "kp_yaw": 1.0,
            "yaw_error_sign": 1.0,
            "smoothing_alpha": 1.0,
            "deadzone_x": 0.01,
            "max_yaw_rate": 2.0,
        }
    )
    out = ctrl.compute(
        state="TRACKING_XY",
        primary_track=_track(),
        err_x=0.2,
        err_y=0.0,
        dt=0.05,
    )
    assert float(out["yaw_rate"]) > 0.0


def test_follow_controller_full_xy_moves_forward_when_far() -> None:
    ctrl = FollowController(
        {
            "enabled": True,
            "mode": "xy",
            "xy_strategy": "full_xy",
            "target_area_ratio": 0.2,
            "kp_forward": 1.0,
            "kp_lateral": 1.0,
            "kp_yaw": 1.0,
            "smoothing_alpha": 1.0,
            "deadzone_x": 0.0,
            "deadzone_area": 0.0,
            "max_vx": 2.0,
            "max_vy": 2.0,
            "max_yaw_rate": 2.0,
        }
    )
    out = ctrl.compute(
        state="TRACKING_XY",
        primary_track=_track(area=1000.0),  # far target => small area ratio
        err_x=0.1,
        err_y=0.0,
        dt=0.05,
    )

    assert out["active"] is True
    assert float(out["vx_body"]) > 0.0
    assert float(out["vy_body"]) < 0.0
    assert float(out["yaw_rate"]) < 0.0


def test_follow_controller_forward_lock_aligns_then_moves_forward() -> None:
    ctrl = FollowController(
        {
            "enabled": True,
            "mode": "xy",
            "xy_strategy": "forward_lock",
            "forward_lock_speed": 0.3,
            "forward_lock_required_frames": 3,
            "forward_lock_deadzone_x": 0.05,
            "smoothing_alpha": 1.0,
            "kp_yaw": 1.0,
            "max_yaw_rate": 2.0,
            "target_area_ratio": 0.2,
            "min_distance_area_ratio": 0.12,
            "deadzone_x": 0.01,
            "deadzone_area": 0.0,
        }
    )

    out1 = ctrl.compute(state="TRACKING_XY", primary_track=_track(area=1000.0), err_x=0.2, err_y=0.0, dt=0.05)
    out2 = ctrl.compute(state="TRACKING_XY", primary_track=_track(area=1000.0), err_x=0.04, err_y=0.0, dt=0.05)
    out3 = ctrl.compute(state="TRACKING_XY", primary_track=_track(area=1000.0), err_x=0.03, err_y=0.0, dt=0.05)
    out4 = ctrl.compute(state="TRACKING_XY", primary_track=_track(area=1000.0), err_x=0.02, err_y=0.0, dt=0.05)

    assert out1["reason"] == "forward_lock_align"
    assert float(out1["vx_body"]) == 0.0
    assert float(out1["yaw_rate"]) < 0.0

    assert out2["reason"] == "forward_lock_wait"
    assert out3["reason"] == "forward_lock_wait"
    assert float(out3["vx_body"]) == 0.0

    assert out4["reason"] == "forward_lock_forward"
    assert float(out4["vx_body"]) > 0.0
    assert abs(float(out4["yaw_rate"])) < 1e-6
    assert int(out4["center_lock_frames"]) >= 3


def test_follow_controller_forward_lock_stops_when_too_close() -> None:
    ctrl = FollowController(
        {
            "enabled": True,
            "mode": "xy",
            "xy_strategy": "forward_lock",
            "forward_lock_speed": 0.3,
            "forward_lock_required_frames": 1,
            "forward_lock_deadzone_x": 0.05,
            "forward_lock_use_area_gate": True,
            "smoothing_alpha": 1.0,
            "target_area_ratio": 0.05,
            "min_distance_area_ratio": 0.12,
            "deadzone_area": 0.0,
        }
    )

    frame_area = 640.0 * 480.0
    area_ratio_close = 0.20
    out = ctrl.compute(
        state="TRACKING_XY",
        primary_track=_track(area=frame_area * area_ratio_close),
        err_x=0.0,
        err_y=0.0,
        dt=0.05,
    )
    assert out["reason"] == "forward_lock_wait"
    assert abs(float(out["vx_body"])) < 1e-6


def test_follow_controller_forward_track_moves_while_aligning() -> None:
    ctrl = FollowController(
        {
            "enabled": True,
            "mode": "xy",
            "xy_strategy": "forward_track",
            "forward_track_speed": 0.5,
            "forward_track_min_speed": 0.1,
            "forward_track_align_gate_x": 0.08,
            "forward_track_stop_gate_x": 0.30,
            "forward_track_lateral_scale": 0.8,
            "forward_track_use_lateral": True,
            "kp_lateral": 1.0,
            "kp_yaw": 1.0,
            "yaw_error_sign": 1.0,
            "lateral_error_sign": 1.0,
            "smoothing_alpha": 1.0,
            "target_area_ratio": 0.2,
            "min_distance_area_ratio": 0.12,
            "deadzone_x": 0.01,
            "deadzone_area": 0.0,
            "max_vx": 2.0,
            "max_vy": 2.0,
            "max_yaw_rate": 2.0,
        }
    )
    out = ctrl.compute(
        state="TRACKING_XY",
        primary_track=_track(area=1000.0),
        err_x=0.16,
        err_y=0.0,
        dt=0.05,
    )

    assert out["reason"] == "forward_track_align"
    assert float(out["vx_body"]) > 0.0
    assert float(out["vy_body"]) > 0.0
    assert float(out["yaw_rate"]) > 0.0


def test_follow_controller_forward_track_holds_when_too_close() -> None:
    ctrl = FollowController(
        {
            "enabled": True,
            "mode": "xy",
            "xy_strategy": "forward_track",
            "forward_track_speed": 0.5,
            "forward_track_use_area_gate": True,
            "target_area_ratio": 0.05,
            "min_distance_area_ratio": 0.12,
            "smoothing_alpha": 1.0,
        }
    )
    frame_area = 640.0 * 480.0
    out = ctrl.compute(
        state="TRACKING_XY",
        primary_track=_track(area=frame_area * 0.20),
        err_x=0.0,
        err_y=0.0,
        dt=0.05,
    )
    assert out["reason"] == "forward_track_hold_area"
    assert abs(float(out["vx_body"])) < 1e-6

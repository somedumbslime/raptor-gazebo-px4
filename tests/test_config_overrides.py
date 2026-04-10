from __future__ import annotations

import pytest

from raptor_ai.config.overrides import apply_runtime_overrides


def _base_cfg() -> dict:
    return {
        "platform": {"type": "gimbal", "px4": {"cv_only": False}},
        "detector": {
            "type": "red",
            "yolo_onnx": {"model_path": "models/old.onnx", "target_classes": ["person"]},
        },
        "selector": {"backend": "stub"},
        "guidance": {"backend": "legacy_internal"},
        "state_machine": {"follow_mode": "off"},
        "follow": {"enabled": False, "xy_strategy": "zone_track"},
    }


def test_apply_runtime_overrides_basic() -> None:
    cfg = apply_runtime_overrides(
        _base_cfg(),
        platform_type="px4",
        detector_type="yolo_onnx",
        selector_backend="external",
        guidance_backend="target_guidance",
        yolo_model_path="models/new.onnx",
        yolo_target_classes_csv="person,car",
        follow_mode="xy",
        follow_enabled=True,
        follow_xy_strategy="zone_track",
        follow_yaw_error_sign=1.0,
        follow_lateral_error_sign=1.0,
        px4_cv_only=True,
    )

    assert cfg["platform"]["type"] == "px4"
    assert cfg["platform"]["px4"]["cv_only"] is True
    assert cfg["detector"]["type"] == "yolo_onnx"
    assert cfg["detector"]["yolo_onnx"]["model_path"] == "models/new.onnx"
    assert cfg["detector"]["yolo_onnx"]["target_classes"] == ["person", "car"]
    assert cfg["selector"]["backend"] == "external"
    assert cfg["guidance"]["backend"] == "target_guidance"
    assert cfg["state_machine"]["follow_mode"] == "xy"
    assert cfg["follow"]["enabled"] is True
    assert cfg["follow"]["xy_strategy"] == "zone_track"
    assert cfg["follow"]["yaw_error_sign"] == 1.0
    assert cfg["follow"]["lateral_error_sign"] == 1.0


def test_apply_runtime_overrides_keeps_unrelated_fields() -> None:
    base = _base_cfg()
    cfg = apply_runtime_overrides(base, platform_type="iris")

    assert cfg["platform"]["type"] == "iris"
    assert cfg["detector"]["type"] == "red"
    assert cfg["selector"]["backend"] == "stub"
    assert cfg["detector"]["yolo_onnx"]["model_path"] == "models/old.onnx"


def test_apply_runtime_overrides_rejects_legacy_strategy() -> None:
    with pytest.raises(ValueError):
        _ = apply_runtime_overrides(_base_cfg(), follow_xy_strategy="forward_lock")


def test_apply_runtime_overrides_accepts_zone_track_strategy() -> None:
    cfg = apply_runtime_overrides(_base_cfg(), follow_xy_strategy="zone_track")
    assert cfg["follow"]["xy_strategy"] == "zone_track"


def test_apply_runtime_overrides_px4_altitude_gates() -> None:
    cfg = apply_runtime_overrides(
        _base_cfg(),
        state_lost_frame_threshold=14,
        state_reacquire_threshold=3,
        px4_auto_arm_require_armable=False,
        px4_auto_arm_require_local_position=True,
        px4_takeoff_confirm_alt_m=0.7,
        px4_offboard_min_relative_alt_m=0.75,
        px4_offboard_start_delay_after_liftoff_s=1.5,
    )
    sm = cfg["state_machine"]
    assert sm["lost_frame_threshold"] == 14
    assert sm["reacquire_threshold"] == 3
    px4 = cfg["platform"]["px4"]
    assert px4["auto_arm_require_armable"] is False
    assert px4["auto_arm_require_local_position"] is True
    assert px4["takeoff_confirm_alt_m"] == 0.7
    assert px4["offboard_min_relative_alt_m"] == 0.75
    assert px4["offboard_start_delay_after_liftoff_s"] == 1.5


def test_apply_runtime_overrides_follow_profile_safe() -> None:
    cfg = apply_runtime_overrides(_base_cfg(), follow_profile="safe")
    follow = cfg["follow"]
    px4 = cfg["platform"]["px4"]
    assert follow["profile"] == "safe"
    assert follow["xy_strategy"] == "zone_track"
    assert follow["max_vx"] == 0.60
    assert follow["max_vy"] == 0.30
    assert follow["zone_track_forward_speed_min"] == 0.20
    assert follow["terminal_charge_speed"] == 0.52
    assert px4["cmd_smoothing_alpha"] == 0.28


def test_apply_runtime_overrides_follow_profile_keeps_explicit_overrides_priority() -> None:
    cfg = apply_runtime_overrides(
        _base_cfg(),
        follow_profile="aggressive",
        follow_xy_strategy="zone_track",
        follow_yaw_error_sign=0.9,
    )
    follow = cfg["follow"]
    assert follow["max_vx"] == 1.50
    assert follow["xy_strategy"] == "zone_track"
    assert follow["yaw_error_sign"] == 0.9


def test_apply_runtime_overrides_follow_profile_unknown_raises() -> None:
    with pytest.raises(ValueError):
        _ = apply_runtime_overrides(_base_cfg(), follow_profile="turbo")

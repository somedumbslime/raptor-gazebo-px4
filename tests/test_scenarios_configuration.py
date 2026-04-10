from __future__ import annotations

import pytest

from raptor_ai.scenarios.configuration import resolve_scenarios_config


def _new_style_cfg() -> dict:
    return {
        "target_mode": "synthetic",
        "set_pose_timeout_ms": 1000,
        "target_modes": {
            "synthetic": {
                "world_name": "raptor_mvp",
                "model_name": "target_stub",
                "profiles": {"slow_circle": {"mode": "circle", "duration_s": 10}},
            },
            "actor": {
                "world_name": "raptor_mvp_actor",
                "target_name": "target_actor",
                "platform_overrides": {
                    "iris": {
                        "world_name": "raptor_mvp_iris_actor",
                    },
                    "px4": {
                        "world_name": "default",
                        "model_name": "target_actor",
                        "spawn_if_missing": True,
                    }
                },
                "profiles": {"slow_circle": {"mode": "circle", "duration_s": 10}},
            },
        },
    }


def test_resolve_target_modes_default() -> None:
    mode, active_cfg, profiles = resolve_scenarios_config(_new_style_cfg())

    assert mode == "synthetic"
    assert active_cfg["world_name"] == "raptor_mvp"
    assert active_cfg["model_name"] == "target_stub"
    assert active_cfg["target_mode"] == "synthetic"
    assert "target_modes" not in active_cfg
    assert list(profiles.keys()) == ["slow_circle"]


def test_resolve_target_modes_override_and_alias() -> None:
    mode, active_cfg, profiles = resolve_scenarios_config(_new_style_cfg(), target_mode_override="actor")

    assert mode == "actor"
    assert active_cfg["world_name"] == "raptor_mvp_actor"
    assert active_cfg["model_name"] == "target_actor"
    assert active_cfg["target_mode"] == "actor"
    assert list(profiles.keys()) == ["slow_circle"]


def test_resolve_legacy_profiles() -> None:
    mode, active_cfg, profiles = resolve_scenarios_config(
        {
            "world_name": "legacy_world",
            "model_name": "legacy_target",
            "profiles": {"legacy": {"mode": "circle", "duration_s": 5}},
        }
    )

    assert mode == "synthetic"
    assert active_cfg["world_name"] == "legacy_world"
    assert active_cfg["model_name"] == "legacy_target"
    assert active_cfg["target_mode"] == "synthetic"
    assert list(profiles.keys()) == ["legacy"]


def test_resolve_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown target_mode"):
        resolve_scenarios_config(_new_style_cfg(), target_mode_override="unknown")


def test_resolve_target_modes_platform_override() -> None:
    mode, active_cfg, profiles = resolve_scenarios_config(
        _new_style_cfg(),
        target_mode_override="actor",
        platform_type="iris",
    )

    assert mode == "actor"
    assert active_cfg["world_name"] == "raptor_mvp_iris_actor"
    assert active_cfg["model_name"] == "target_actor"
    assert active_cfg["platform_type"] == "iris"
    assert list(profiles.keys()) == ["slow_circle"]


def test_resolve_target_modes_platform_override_px4() -> None:
    mode, active_cfg, profiles = resolve_scenarios_config(
        _new_style_cfg(),
        target_mode_override="actor",
        platform_type="px4",
    )

    assert mode == "actor"
    assert active_cfg["world_name"] == "default"
    assert active_cfg["model_name"] == "target_actor"
    assert active_cfg["spawn_if_missing"] is True
    assert active_cfg["platform_type"] == "px4"
    assert list(profiles.keys()) == ["slow_circle"]

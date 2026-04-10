from __future__ import annotations

import sys
import types

import pytest

from raptor_ai.platform.factory import build_platform


def test_build_platform_gimbal_with_controller_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = types.ModuleType("raptor_ai.platform.gimbal_platform")

    class FakeGimbalPlatform:
        def __init__(self, cfg):
            self.cfg = dict(cfg)
            self.platform_type = "gimbal"

        def metadata(self):
            return {"platform_type": "gimbal", **self.cfg}

    fake_mod.GimbalPlatform = FakeGimbalPlatform
    monkeypatch.setitem(sys.modules, "raptor_ai.platform.gimbal_platform", fake_mod)

    platform = build_platform(
        platform_cfg={"type": "gimbal", "gimbal": {}},
        controller_cfg={"yaw_topic": "/y", "pitch_topic": "/p"},
    )

    assert isinstance(platform, FakeGimbalPlatform)
    assert platform.cfg["yaw_topic"] == "/y"
    assert platform.cfg["pitch_topic"] == "/p"


def test_build_platform_iris(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = types.ModuleType("raptor_ai.platform.iris_platform")

    class FakeIrisPlatform:
        def __init__(self, cfg):
            self.cfg = dict(cfg)
            self.platform_type = "iris"

        def metadata(self):
            return {"platform_type": "iris", **self.cfg}

    fake_mod.IrisPlatform = FakeIrisPlatform
    monkeypatch.setitem(sys.modules, "raptor_ai.platform.iris_platform", fake_mod)

    platform = build_platform(
        platform_cfg={"type": "iris", "iris": {"cmd_twist_topic": "/cmd"}},
        controller_cfg=None,
    )

    assert isinstance(platform, FakeIrisPlatform)
    assert platform.cfg["cmd_twist_topic"] == "/cmd"


def test_build_platform_px4_with_top_level_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = types.ModuleType("raptor_ai.platform.px4_bridge")

    class FakePx4Bridge:
        def __init__(self, cfg):
            self.cfg = dict(cfg)
            self.platform_type = "px4"

        def metadata(self):
            return {"platform_type": "px4", **self.cfg}

    fake_mod.Px4Bridge = FakePx4Bridge
    monkeypatch.setitem(sys.modules, "raptor_ai.platform.px4_bridge", fake_mod)

    platform = build_platform(
        platform_cfg={"type": "px4", "px4": {"command_hz": 30.0}},
        controller_cfg=None,
        px4_cfg={"system_address": "udpin://0.0.0.0:14540"},
    )

    assert isinstance(platform, FakePx4Bridge)
    assert platform.cfg["command_hz"] == 30.0
    assert platform.cfg["system_address"] == "udpin://0.0.0.0:14540"


def test_build_platform_unsupported_type() -> None:
    with pytest.raises(ValueError, match="Unsupported platform.type"):
        build_platform(platform_cfg={"type": "unknown"}, controller_cfg={})

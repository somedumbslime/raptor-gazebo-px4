from __future__ import annotations

import sys
import types
from pathlib import Path

import yaml


def _install_fake_gz_modules() -> None:
    if "gz" in sys.modules:
        return

    gz = types.ModuleType("gz")
    msgs = types.ModuleType("gz.msgs")
    transport = types.ModuleType("gz.transport")

    class _Pub:
        def publish(self, _msg) -> bool:
            return True

    class _Node:
        def subscribe(self, *_args, **_kwargs) -> bool:
            return True

        def advertise(self, *_args, **_kwargs):
            return _Pub()

        def request(self, *_args, **_kwargs):
            class _Rep:
                data = True

            return True, _Rep()

    transport.Node = _Node

    def _add_module(name: str, attrs: dict[str, object]) -> None:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod

    class _Image:
        def __init__(self):
            self.pixel_format_type = 3
            self.height = 0
            self.width = 0
            self.data = b""

    class _Double:
        def __init__(self):
            self.data = 0.0

    class _Boolean:
        def __init__(self):
            self.data = False

    class _Vec3:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class _Twist:
        def __init__(self):
            self.linear = _Vec3()
            self.angular = _Vec3()

    class _Position:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class _Orientation:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.w = 1.0

    class _Pose:
        def __init__(self):
            self.name = ""
            self.position = _Position()
            self.orientation = _Orientation()

    sys.modules["gz"] = gz
    sys.modules["gz.msgs"] = msgs
    sys.modules["gz.transport"] = transport
    _add_module("gz.msgs.image_pb2", {"Image": _Image})
    _add_module("gz.msgs.double_pb2", {"Double": _Double})
    _add_module("gz.msgs.boolean_pb2", {"Boolean": _Boolean})
    _add_module("gz.msgs.twist_pb2", {"Twist": _Twist})
    _add_module("gz.msgs.pose_pb2", {"Pose": _Pose})


def _write_runtime_config(path: Path) -> None:
    cfg = {
        "camera": {
            "topic": "auto",
            "topics": {"px4": "/front_camera/image_raw"},
        },
        "platform": {
            "type": "px4",
            "px4": {
                "camera_topic": "/front_camera/image_raw",
                "offboard_enabled": True,
                "auto_takeoff": True,
            },
        },
        "controller": {
            "control_hz": 20.0,
        },
        "follow": {
            "enabled": True,
            "mode": "xy",
            "xy_strategy": "forward_track",
            "px4_auto_takeoff": True,
            "px4_offboard_min_relative_alt_m": 0.75,
        },
        "guidance": {
            "backend": "legacy_internal",
        },
        "detector": {
            "type": "red",
            "red": {
                "hsv_ranges": [],
                "min_area": 100,
            },
        },
        "tracking": {},
        "selector": {"backend": "stub"},
        "memory": {},
        "search_policy": {},
        "state_machine": {},
        "runtime": {
            "px4_skip_inference_until_offboard": True,
        },
        "logging": {
            "output_dir": "runs/latest",
            "events_file": "events.jsonl",
            "metrics_file": "metrics_summary.json",
            "run_meta_file": "run_meta.json",
            "verbosity": 0,
        },
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def test_px4_ready_gate_uses_follow_policy_threshold(tmp_path: Path) -> None:
    _install_fake_gz_modules()
    from raptor_ai.runtime.runtime_v2 import RuntimeV2

    cfg_path = tmp_path / "runtime_px4_gate.yaml"
    out_dir = tmp_path / "out_px4_gate"
    _write_runtime_config(cfg_path)
    runtime = RuntimeV2(config_path=str(cfg_path), output_dir=str(out_dir))

    # No lifecycle hints in platform metadata: gate must still use follow config.
    runtime.platform.metadata = lambda: {  # type: ignore[method-assign]
        "platform_type": "px4",
        "offboard_started": False,
        "offboard_mode_active": False,
        "in_air": True,
        "relative_altitude_m": 0.70,
    }
    ready, _meta = runtime._is_px4_runtime_ready()
    assert ready is False

    runtime.platform.metadata = lambda: {  # type: ignore[method-assign]
        "platform_type": "px4",
        "offboard_started": True,
        "offboard_mode_active": True,
        "in_air": True,
        "relative_altitude_m": 0.80,
    }
    ready, _meta = runtime._is_px4_runtime_ready()
    assert ready is True


def test_px4_ready_gate_requires_offboard_when_auto_offboard_enabled(tmp_path: Path) -> None:
    _install_fake_gz_modules()
    from raptor_ai.runtime.runtime_v2 import RuntimeV2

    cfg_path = tmp_path / "runtime_px4_gate_offboard.yaml"
    out_dir = tmp_path / "out_px4_gate_offboard"
    _write_runtime_config(cfg_path)
    runtime = RuntimeV2(config_path=str(cfg_path), output_dir=str(out_dir))

    runtime.platform.metadata = lambda: {  # type: ignore[method-assign]
        "platform_type": "px4",
        "offboard_started": False,
        "offboard_mode_active": False,
        "in_air": True,
        "relative_altitude_m": 1.20,
    }
    ready, _meta = runtime._is_px4_runtime_ready()
    assert ready is False

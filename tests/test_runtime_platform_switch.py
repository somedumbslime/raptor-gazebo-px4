from __future__ import annotations

import json
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

    class _Double:
        def __init__(self):
            self.data = 0.0

    class _Image:
        def __init__(self):
            self.pixel_format_type = 3
            self.height = 0
            self.width = 0
            self.data = b""

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
    _add_module("gz.msgs.double_pb2", {"Double": _Double})
    _add_module("gz.msgs.image_pb2", {"Image": _Image})
    _add_module("gz.msgs.boolean_pb2", {"Boolean": _Boolean})
    _add_module("gz.msgs.twist_pb2", {"Twist": _Twist})
    _add_module("gz.msgs.pose_pb2", {"Pose": _Pose})


def _write_runtime_config(path: Path, platform_type: str) -> None:
    cfg = {
        "camera": {
            "topic": "auto",
            "topics": {
                "gimbal": "/raptor/camera",
                "iris": "/raptor/iris/camera",
            },
        },
        "platform": {
            "type": platform_type,
            "gimbal": {
                "camera_topic": "/raptor/camera",
                "yaw_topic": "/raptor/gimbal/yaw_cmd",
                "pitch_topic": "/raptor/gimbal/pitch_cmd",
            },
            "iris": {
                "camera_topic": "/raptor/iris/camera",
                "cmd_twist_topic": "/raptor/iris/cmd_vel",
                "enable_topic": "/raptor/iris/enable",
                "publish_enable": False,
                "control_mode": "hold",
            },
        },
        "controller": {
            "control_hz": 20.0,
            "yaw_topic": "/raptor/gimbal/yaw_cmd",
            "pitch_topic": "/raptor/gimbal/pitch_cmd",
        },
        "detector": {
            "type": "red",
            "red": {
                "hsv_ranges": [],
                "min_area": 100,
            },
        },
        "tracking": {},
        "selector": {
            "backend": "stub",
        },
        "memory": {},
        "search_policy": {},
        "state_machine": {},
        "runtime": {},
        "logging": {
            "output_dir": "runs/latest",
            "events_file": "events.jsonl",
            "metrics_file": "metrics_summary.json",
            "run_meta_file": "run_meta.json",
            "verbosity": 0,
        },
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _build_runtime(tmp_path: Path, platform_type: str):
    _install_fake_gz_modules()
    from raptor_ai.runtime.runtime_v2 import RuntimeV2

    cfg_path = tmp_path / f"runtime_{platform_type}.yaml"
    out_dir = tmp_path / f"out_{platform_type}"
    _write_runtime_config(cfg_path, platform_type)
    return RuntimeV2(config_path=str(cfg_path), output_dir=str(out_dir))


def test_runtime_platform_switch_gimbal(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path, "gimbal")
    summary = runtime.run(duration_s=0.0)

    assert summary["stop_reason"] == "duration"
    run_meta = json.loads((Path(runtime.run_meta_path)).read_text(encoding="utf-8"))
    assert run_meta["platform_type"] == "gimbal"
    assert run_meta["camera_topic"] == "/raptor/camera"
    assert run_meta["yaw_topic"] == "/raptor/gimbal/yaw_cmd"
    assert run_meta["pitch_topic"] == "/raptor/gimbal/pitch_cmd"


def test_runtime_platform_switch_iris(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path, "iris")
    summary = runtime.run(duration_s=0.0)

    assert summary["stop_reason"] == "duration"
    run_meta = json.loads((Path(runtime.run_meta_path)).read_text(encoding="utf-8"))
    assert run_meta["platform_type"] == "iris"
    assert run_meta["camera_topic"] == "/raptor/iris/camera"
    assert run_meta["platform_meta"]["cmd_twist_topic"] == "/raptor/iris/cmd_vel"


from __future__ import annotations

import pytest

from raptor_ai.config.loader import resolve_camera_topic, resolve_platform_type


def test_resolve_platform_type_default_gimbal() -> None:
    assert resolve_platform_type({}) == "gimbal"


def test_resolve_platform_type_px4() -> None:
    assert resolve_platform_type({"platform": {"type": "px4"}}) == "px4"


def test_resolve_platform_type_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported platform.type"):
        resolve_platform_type({"platform": {"type": "unknown"}})


def test_resolve_camera_topic_explicit_legacy_priority() -> None:
    cfg = {
        "platform": {"type": "iris"},
        "camera": {
            "topic": "/legacy/manual/topic",
            "topics": {"gimbal": "/raptor/camera", "iris": "/raptor/iris/camera"},
        },
    }
    assert resolve_camera_topic(cfg) == "/legacy/manual/topic"


def test_resolve_camera_topic_auto_from_camera_topics() -> None:
    cfg = {
        "platform": {"type": "iris"},
        "camera": {
            "topic": "auto",
            "topics": {"gimbal": "/raptor/camera", "iris": "/raptor/iris/camera"},
        },
    }
    assert resolve_camera_topic(cfg) == "/raptor/iris/camera"


def test_resolve_camera_topic_auto_from_platform_section_fallback() -> None:
    cfg = {
        "platform": {
            "type": "iris",
            "iris": {"camera_topic": "/custom/iris/camera"},
        },
        "camera": {
            "topic": "auto",
        },
    }
    assert resolve_camera_topic(cfg) == "/custom/iris/camera"


def test_resolve_camera_topic_legacy_default_when_missing() -> None:
    cfg = {
        "platform": {"type": "gimbal"},
        "camera": {"topic": "auto"},
    }
    assert resolve_camera_topic(cfg) == "/raptor/camera"


def test_resolve_camera_topic_px4_legacy_default_when_missing() -> None:
    cfg = {
        "platform": {"type": "px4"},
        "camera": {"topic": "auto"},
    }
    assert resolve_camera_topic(cfg) == "/front_camera/image_raw"

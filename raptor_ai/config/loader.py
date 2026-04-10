from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_SUPPORTED_PLATFORM_TYPES = {"gimbal", "iris", "px4"}


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load runtime YAML config into a plain dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")

    return data


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_platform_type(cfg: dict[str, Any]) -> str:
    platform_cfg = cfg.get("platform", {})
    if not isinstance(platform_cfg, dict):
        return "gimbal"

    platform_type = str(platform_cfg.get("type", "gimbal")).strip().lower()
    if platform_type not in _SUPPORTED_PLATFORM_TYPES:
        raise ValueError(f"Unsupported platform.type: {platform_type}")
    return platform_type


def resolve_camera_topic(cfg: dict[str, Any], platform_type: str | None = None) -> str:
    ptype = platform_type or resolve_platform_type(cfg)
    camera_cfg = cfg.get("camera", {})
    platform_cfg = cfg.get("platform", {})

    if not isinstance(camera_cfg, dict):
        camera_cfg = {}
    if not isinstance(platform_cfg, dict):
        platform_cfg = {}

    explicit_topic = str(camera_cfg.get("topic", "")).strip()
    if explicit_topic and explicit_topic.lower() not in ("auto", "platform", "default"):
        return explicit_topic

    topics = camera_cfg.get("topics", {})
    if isinstance(topics, dict):
        from_map = str(topics.get(ptype, "")).strip()
        if from_map:
            return from_map

    ptype_cfg = platform_cfg.get(ptype, {})
    if isinstance(ptype_cfg, dict):
        from_platform = str(ptype_cfg.get("camera_topic", "")).strip()
        if from_platform:
            return from_platform

    legacy_defaults = {
        "gimbal": "/raptor/camera",
        "iris": "/raptor/iris/camera",
        "px4": "/front_camera/image_raw",
    }
    return legacy_defaults.get(ptype, "/raptor/camera")

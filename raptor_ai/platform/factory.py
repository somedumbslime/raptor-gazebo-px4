from __future__ import annotations

from typing import Any

from raptor_ai.platform.platform_types import PlatformController


def build_platform(
    platform_cfg: dict[str, Any] | None,
    controller_cfg: dict[str, Any] | None = None,
    px4_cfg: dict[str, Any] | None = None,
) -> PlatformController:
    cfg = dict(platform_cfg or {})
    ctrl = dict(controller_cfg or {})
    platform_type = str(cfg.get("type", "gimbal")).strip().lower()

    if platform_type == "gimbal":
        from raptor_ai.platform.gimbal_platform import GimbalPlatform

        gimbal_cfg = dict(cfg.get("gimbal", {}))
        gimbal_cfg.setdefault("yaw_topic", ctrl.get("yaw_topic", "/raptor/gimbal/yaw_cmd"))
        gimbal_cfg.setdefault("pitch_topic", ctrl.get("pitch_topic", "/raptor/gimbal/pitch_cmd"))
        return GimbalPlatform(gimbal_cfg)

    if platform_type == "iris":
        from raptor_ai.platform.iris_platform import IrisPlatform

        iris_cfg = dict(cfg.get("iris", {}))
        return IrisPlatform(iris_cfg)

    if platform_type == "px4":
        from raptor_ai.platform.px4_bridge import Px4Bridge

        merged_px4_cfg = dict(px4_cfg or {})
        merged_px4_cfg.update(dict(cfg.get("px4", {})))
        return Px4Bridge(merged_px4_cfg)

    raise ValueError(f"Unsupported platform.type: {platform_type}")

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GuidanceInput:
    state: str
    primary_track: dict[str, Any] | None
    err_x: float | None
    err_y: float | None
    dt: float
    ts: float = 0.0
    platform_meta: dict[str, Any] | None = None


@dataclass(slots=True)
class GuidanceOutput:
    active: bool
    mode: str
    xy_strategy: str
    state: str
    reason: str
    vx_body: float
    vy_body: float
    vz: float
    yaw_rate: float
    area_ratio: float
    area_error: float
    center_lock_frames: int
    platform_action: str
    platform_action_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": bool(self.active),
            "mode": str(self.mode),
            "xy_strategy": str(self.xy_strategy),
            "state": str(self.state),
            "reason": str(self.reason),
            "vx_body": float(self.vx_body),
            "vy_body": float(self.vy_body),
            "vz": float(self.vz),
            "yaw_rate": float(self.yaw_rate),
            "area_ratio": float(self.area_ratio),
            "area_error": float(self.area_error),
            "center_lock_frames": int(self.center_lock_frames),
            "platform_action": str(self.platform_action),
            "platform_action_payload": dict(self.platform_action_payload),
        }

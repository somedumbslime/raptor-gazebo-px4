from __future__ import annotations

from typing import Any, Protocol


class PlatformController(Protocol):
    platform_type: str

    def publish_commands(
        self,
        yaw_cmd: float,
        pitch_cmd: float,
        *,
        state: str | None = None,
        follow_cmd: dict[str, Any] | None = None,
    ) -> None:
        ...

    def metadata(self) -> dict[str, Any]:
        ...

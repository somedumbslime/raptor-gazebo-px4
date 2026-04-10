from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from raptor_ai.tracking.track_types import TrackInfo


@dataclass
class MemorySnapshot:
    last_seen_timestamp: float | None = None
    last_seen_bbox: list[float] | None = None
    last_seen_center: list[float] | None = None
    last_seen_err_x: float | None = None
    last_seen_err_y: float | None = None
    last_seen_side_horizontal: str = "center"
    last_seen_side_vertical: str = "center"
    last_seen_yaw: float | None = None
    last_seen_pitch: float | None = None
    last_known_track_id: int | None = None


class TargetMemory:
    def __init__(self, cfg: dict[str, Any]):
        self.side_center_threshold_x = float(cfg.get("side_center_threshold_x", 0.05))
        self.side_center_threshold_y = float(cfg.get("side_center_threshold_y", 0.05))
        self._snapshot = MemorySnapshot()

    def _side_horizontal(self, err_x: float | None) -> str:
        if err_x is None:
            return self._snapshot.last_seen_side_horizontal
        if abs(err_x) <= self.side_center_threshold_x:
            return "center"
        return "right" if err_x > 0.0 else "left"

    def _side_vertical(self, err_y: float | None) -> str:
        if err_y is None:
            return self._snapshot.last_seen_side_vertical
        if abs(err_y) <= self.side_center_threshold_y:
            return "center"
        return "down" if err_y > 0.0 else "up"

    def update(
        self,
        selected_track: TrackInfo,
        control_state: str,
        timestamp: float,
        err_x: float | None = None,
        err_y: float | None = None,
        yaw: float | None = None,
        pitch: float | None = None,
    ) -> None:
        if selected_track is None:
            return

        self._snapshot.last_seen_timestamp = float(timestamp)
        self._snapshot.last_seen_bbox = list(selected_track.get("bbox_xyxy", []))
        self._snapshot.last_seen_center = list(selected_track.get("center", []))
        self._snapshot.last_seen_err_x = None if err_x is None else float(err_x)
        self._snapshot.last_seen_err_y = None if err_y is None else float(err_y)
        self._snapshot.last_seen_side_horizontal = self._side_horizontal(err_x)
        self._snapshot.last_seen_side_vertical = self._side_vertical(err_y)
        self._snapshot.last_seen_yaw = None if yaw is None else float(yaw)
        self._snapshot.last_seen_pitch = None if pitch is None else float(pitch)
        self._snapshot.last_known_track_id = int(selected_track.get("track_id", -1))

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(**asdict(self._snapshot))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self._snapshot)

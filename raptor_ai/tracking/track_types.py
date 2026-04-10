from __future__ import annotations

from typing import Any, TypedDict


class Detection(TypedDict, total=False):
    bbox_xyxy: list[float]
    center: list[float]
    area: float
    conf: float
    cls_id: int
    cls_name: str
    source: str


class TrackInfo(TypedDict, total=False):
    track_id: int
    bbox_xyxy: list[float]
    bbox_xywh: list[float]
    center: list[float]
    conf: float
    cls_id: int
    cls_name: str
    area: float
    frame_w: int
    frame_h: int
    timestamp: float
    source: str
    visible: bool
    selector_payload: dict[str, Any]


class SelectionResult(TypedDict, total=False):
    selected_primary_target: TrackInfo | None
    selection_state: str
    selection_reason: str
    selector_events: list[dict[str, Any]]

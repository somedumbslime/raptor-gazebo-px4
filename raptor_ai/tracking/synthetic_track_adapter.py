from __future__ import annotations

from typing import Any

from raptor_ai.tracking.iou_tracker import IouTracker


class SyntheticTrackAdapter(IouTracker):
    """
    Backward-compatible alias for the legacy synthetic adapter name.
    Internally uses the generic IoU tracker layer.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)

from __future__ import annotations

from raptor_ai.tracking.synthetic_track_adapter import SyntheticTrackAdapter


def det(x1: float, y1: float, x2: float, y2: float, area: float) -> dict:
    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "center": [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
        "area": area,
        "conf": 1.0,
        "cls_id": 0,
        "cls_name": "synthetic_target",
        "source": "red_detector",
    }


def test_track_id_continuity_and_reassignment() -> None:
    adapter = SyntheticTrackAdapter({"iou_match_threshold": 0.3, "max_missing_frames": 10})

    tracks1 = adapter.to_tracks([det(10, 10, 50, 50, 1600)], (480, 640, 3), 1.0)
    tracks2 = adapter.to_tracks([det(12, 12, 52, 52, 1600)], (480, 640, 3), 2.0)
    tracks3 = adapter.to_tracks([det(120, 120, 170, 170, 2500)], (480, 640, 3), 3.0)

    assert len(tracks1) == 1
    assert len(tracks2) == 1
    assert len(tracks3) == 1

    id1 = tracks1[0]["track_id"]
    id2 = tracks2[0]["track_id"]
    id3 = tracks3[0]["track_id"]

    assert id1 == id2
    assert id3 != id2


def test_xywh_and_frame_fields() -> None:
    adapter = SyntheticTrackAdapter({})
    tracks = adapter.to_tracks([det(20, 30, 70, 90, 3000)], (480, 640, 3), 10.0)
    tr = tracks[0]

    assert tr["bbox_xywh"] == [20.0, 30.0, 50.0, 60.0]
    assert tr["frame_w"] == 640
    assert tr["frame_h"] == 480
    assert tr["timestamp"] == 10.0

from __future__ import annotations

from pathlib import Path

import pytest

from raptor_ai.tracking.primary_selector_adapter import PrimaryTargetSelectorAdapter
from raptor_ai.tracking.pts_external_selector import reset_selector

PTS_ROOT = Path(__file__).resolve().parents[1] / "primary-target-selection"

pytestmark = pytest.mark.skipif(not PTS_ROOT.exists(), reason="PTS repository is not available")


def _track(track_id: int = 11) -> dict:
    return {
        "track_id": track_id,
        "bbox_xyxy": [220.0, 120.0, 420.0, 380.0],
        "bbox_xywh": [220.0, 120.0, 200.0, 260.0],
        "center": [320.0, 250.0],
        "conf": 0.9,
        "cls_id": 0,
        "cls_name": "soldier",
        "area": 52000.0,
        "frame_w": 640,
        "frame_h": 480,
        "timestamp": 0.0,
        "source": "test",
    }


def test_external_pts_selector_eventually_locks_primary() -> None:
    reset_selector()

    adapter = PrimaryTargetSelectorAdapter(
        {
            "backend": "external",
            "external_callable": "raptor_ai.tracking.pts_external_selector:select_primary",
            "external_pythonpath": ["primary-target-selection"],
            "required_fields": [
                "track_id",
                "bbox_xyxy",
                "confidence",
                "class_id",
                "class_name",
                "visible",
            ],
            "field_map": {
                "confidence": "conf",
                "class_id": "cls_id",
                "class_name": "cls_name",
            },
            "external_context": {
                "pts_config_path": "primary-target-selection/pts/resources/target_selection.yaml",
                "pts_save_events_jsonl": False,
            },
        }
    )

    selected_ids: list[int] = []
    saw_selector_event = False

    for i in range(12):
        tr = _track(track_id=11)
        tr["timestamp"] = i / 20.0
        result = adapter.select_primary(
            [tr],
            context={
                "timestamp": i / 20.0,
                "frame_index": i,
                "frame_size": (640, 480),
            },
        )
        sel = result.get("selected_primary_target")
        if sel is not None:
            selected_ids.append(int(sel["track_id"]))
        saw_selector_event = saw_selector_event or bool(result.get("selector_events"))

    assert selected_ids, "PTS never selected a primary target"
    assert selected_ids[-1] == 11
    assert saw_selector_event


def test_stub_backend_still_available_with_pts_config_present() -> None:
    adapter = PrimaryTargetSelectorAdapter(
        {
            "backend": "stub",
            "stub_policy": "max_area",
            "external_callable": "raptor_ai.tracking.pts_external_selector:select_primary",
            "external_pythonpath": ["primary-target-selection"],
        }
    )

    tracks = [
        {
            "track_id": 1,
            "bbox_xyxy": [0.0, 0.0, 20.0, 20.0],
            "center": [10.0, 10.0],
            "area": 400.0,
            "conf": 0.9,
            "cls_id": 0,
            "cls_name": "synthetic_target",
        },
        {
            "track_id": 2,
            "bbox_xyxy": [0.0, 0.0, 50.0, 50.0],
            "center": [25.0, 25.0],
            "area": 2500.0,
            "conf": 0.8,
            "cls_id": 0,
            "cls_name": "synthetic_target",
        },
    ]

    result = adapter.select_primary(tracks, context={"frame_size": (640, 480)})
    assert result["selected_primary_target"] is not None
    assert result["selected_primary_target"]["track_id"] == 2
    assert result["selection_state"] == "selected"

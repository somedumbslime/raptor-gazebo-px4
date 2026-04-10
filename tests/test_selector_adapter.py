from __future__ import annotations

from raptor_ai.tracking.primary_selector_adapter import PrimaryTargetSelectorAdapter


def test_stub_selects_max_area_and_maps_payload() -> None:
    adapter = PrimaryTargetSelectorAdapter(
        {
            "backend": "stub",
            "stub_policy": "max_area",
            "required_fields": ["track_id", "xyxy", "score"],
            "field_map": {
                "xyxy": "bbox_xyxy",
                "score": "conf",
            },
        }
    )

    tracks = [
        {
            "track_id": 1,
            "bbox_xyxy": [0, 0, 20, 20],
            "conf": 0.8,
            "area": 400,
            "center": [10, 10],
        },
        {
            "track_id": 2,
            "bbox_xyxy": [0, 0, 40, 40],
            "conf": 0.6,
            "area": 1600,
            "center": [20, 20],
        },
    ]

    result = adapter.select_primary(tracks, context={})

    assert result["selection_state"] == "selected"
    assert result["selected_primary_target"]["track_id"] == 2
    payload = result["selected_primary_target"]["selector_payload"]
    assert payload["track_id"] == 2
    assert payload["xyxy"] == [0, 0, 40, 40]
    assert payload["score"] == 0.6

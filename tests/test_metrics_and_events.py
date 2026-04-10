from __future__ import annotations

import json

from raptor_ai.metrics.event_logger import EventLogger
from raptor_ai.metrics.metrics_logger import MetricsLogger


def test_metrics_summary_and_events_file(tmp_path) -> None:
    metrics = MetricsLogger(deadzone_x=0.05, deadzone_y=0.05)

    metrics.record_event("target_lost")
    metrics.record_event("target_reacquired")

    metrics.record_frame(
        has_detection=True,
        has_primary=True,
        state="LOCKED",
        err_x=0.01,
        err_y=0.02,
        yaw_cmd=0.1,
        pitch_cmd=0.2,
        yaw_saturated=False,
        pitch_saturated=False,
    )
    metrics.record_frame(
        has_detection=False,
        has_primary=False,
        state="SEARCHING",
        err_x=None,
        err_y=None,
        yaw_cmd=0.0,
        pitch_cmd=0.0,
        yaw_saturated=True,
        pitch_saturated=False,
    )

    summary = metrics.summary()
    assert summary["frames_total"] == 2
    assert summary["frames_with_primary"] == 1
    assert summary["lost_target_count"] == 1
    assert summary["reacquire_count"] == 1
    assert summary["search_frames"] == 1

    out_path = tmp_path / "metrics.json"
    metrics.write_summary(out_path)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["frames_total"] == 2

    ev_path = tmp_path / "events.jsonl"
    ev = EventLogger(ev_path)
    ev.log(1.0, "target_acquired", track_id=1)
    ev.log(2.0, "target_lost", last_seen_side="right")
    ev.close()

    lines = ev_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    row1 = json.loads(lines[1])
    assert row0["event"] == "target_acquired"
    assert row1["event"] == "target_lost"

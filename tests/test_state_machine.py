from __future__ import annotations

from raptor_ai.runtime.state_machine import LOCKED, LOST, SEARCHING, RuntimeStateMachine


def test_state_machine_lost_search_reacquire_cycle() -> None:
    sm = RuntimeStateMachine({"lost_frame_threshold": 3, "reacquire_threshold": 2})

    up = sm.update(True)
    assert up.state == LOCKED
    assert any(e["event"] == "target_acquired" for e in up.events)

    up = sm.update(False)
    assert up.state == LOST
    assert any(e["event"] == "target_lost" for e in up.events)

    up = sm.update(False)
    assert up.state == LOST

    up = sm.update(False)
    assert up.state == SEARCHING
    assert any(e["event"] == "search_started" for e in up.events)

    up = sm.update(True)
    assert up.state == SEARCHING

    up = sm.update(True)
    assert up.state == LOCKED
    names = [e["event"] for e in up.events]
    assert "target_reacquired" in names
    assert "search_stopped" in names

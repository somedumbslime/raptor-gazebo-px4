from __future__ import annotations

from raptor_ai.control.search_policy_last_seen import SearchPolicyLastSeen
from raptor_ai.memory.target_memory import MemorySnapshot


def test_last_seen_direction_is_used() -> None:
    policy = SearchPolicyLastSeen({"yaw_scan_speed": 0.5, "pitch_scan_speed": 0.2, "fallback_sweep_period_s": 2.0})
    snap = MemorySnapshot(last_seen_side_horizontal="right", last_seen_side_vertical="up")

    cmd = policy.compute(snap, ts=1.0)
    assert cmd["mode"] == "last_seen_right"
    assert cmd["yaw_rate"] > 0.0
    assert cmd["pitch_rate"] < 0.0


def test_fallback_sweep_changes_direction() -> None:
    policy = SearchPolicyLastSeen({"yaw_scan_speed": 0.5, "fallback_sweep_period_s": 1.0})
    snap = MemorySnapshot(last_seen_side_horizontal="center", last_seen_side_vertical="center")

    cmd_a = policy.compute(snap, ts=0.2)
    cmd_b = policy.compute(snap, ts=1.2)
    assert cmd_a["mode"] == "fallback_sweep"
    assert cmd_b["mode"] == "fallback_sweep"
    assert cmd_a["yaw_rate"] == -cmd_b["yaw_rate"]

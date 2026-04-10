from __future__ import annotations

import math

from raptor_ai.platform.px4_bridge import Px4Bridge


def _make_stub_bridge() -> tuple[Px4Bridge, list[tuple[float, float, float, float]], list[str]]:
    bridge = Px4Bridge.__new__(Px4Bridge)
    bridge.cv_only = False
    bridge.legacy_gimbal_mapping = True
    bridge.active_states = set(Px4Bridge.DEFAULT_ACTIVE_STATES)
    bridge.yaw_to_yaw_rate_gain = 20.0
    bridge.pitch_to_vertical_gain = 0.4

    set_calls: list[tuple[float, float, float, float]] = []
    stop_calls: list[str] = []

    def _set_velocity_xyz(*, forward_m_s: float, right_m_s: float, down_m_s: float, yaw_rate_deg_s: float) -> None:
        set_calls.append((float(forward_m_s), float(right_m_s), float(down_m_s), float(yaw_rate_deg_s)))

    def _stop_motion() -> None:
        stop_calls.append("stop")

    bridge.set_velocity_xyz = _set_velocity_xyz  # type: ignore[method-assign]
    bridge.stop_motion = _stop_motion  # type: ignore[method-assign]
    return bridge, set_calls, stop_calls


def test_follow_contract_disables_legacy_mapping_when_follow_inactive() -> None:
    bridge, set_calls, stop_calls = _make_stub_bridge()

    bridge.publish_commands(
        yaw_cmd=0.5,
        pitch_cmd=0.5,
        state="SEARCHING",
        follow_cmd={"active": False, "vx_body": 0.0, "vy_body": 0.0, "vz": 0.0, "yaw_rate": 0.0},
    )

    assert stop_calls == ["stop"]
    assert set_calls == []


def test_follow_contract_applies_follow_velocity_in_tracking() -> None:
    bridge, set_calls, stop_calls = _make_stub_bridge()

    bridge.publish_commands(
        yaw_cmd=0.0,
        pitch_cmd=0.0,
        state="TRACKING_XY",
        follow_cmd={"active": True, "vx_body": 0.3, "vy_body": -0.1, "vz": 0.2, "yaw_rate": 0.25},
    )

    assert stop_calls == []
    assert len(set_calls) == 1
    fwd, right, down, yaw_deg_s = set_calls[0]
    assert fwd == 0.3
    assert right == -0.1
    assert down == -0.2
    assert abs(yaw_deg_s - math.degrees(0.25)) < 1e-6


def test_legacy_mapping_kept_when_follow_contract_absent() -> None:
    bridge, set_calls, stop_calls = _make_stub_bridge()

    bridge.publish_commands(
        yaw_cmd=0.4,
        pitch_cmd=-0.5,
        state="SEARCHING",
        follow_cmd=None,
    )

    assert stop_calls == []
    assert len(set_calls) == 1
    fwd, right, down, yaw_deg_s = set_calls[0]
    assert fwd == 0.0
    assert right == 0.0
    assert down == (-0.5 * 0.4)
    assert yaw_deg_s == (0.4 * 20.0)


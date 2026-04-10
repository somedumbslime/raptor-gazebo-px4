from __future__ import annotations

from raptor_ai.runtime.runtime_v2 import RuntimeV2


def test_primary_color_follows_selection_state() -> None:
    colors = {
        "primary_default": (0, 255, 0),
        "primary_locked": (0, 165, 255),
        "primary_switch_pending": (0, 255, 255),
        "primary_lost": (0, 0, 255),
    }

    assert RuntimeV2._resolve_primary_color("locked", colors) == (0, 165, 255)
    assert RuntimeV2._resolve_primary_color("switch_pending", colors) == (0, 255, 255)
    assert RuntimeV2._resolve_primary_color("lost", colors) == (0, 0, 255)
    assert RuntimeV2._resolve_primary_color("selected", colors) == (0, 255, 0)
    assert RuntimeV2._resolve_primary_color("none", colors) == (0, 255, 0)


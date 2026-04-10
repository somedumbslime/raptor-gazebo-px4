from __future__ import annotations

from typing import Any

from raptor_ai.memory.target_memory import MemorySnapshot


class SearchPolicyLastSeen:
    def __init__(self, cfg: dict[str, Any]):
        self.yaw_scan_speed = float(cfg.get("yaw_scan_speed", 0.45))
        self.pitch_scan_speed = float(cfg.get("pitch_scan_speed", 0.2))
        self.fallback_sweep_period_s = float(cfg.get("fallback_sweep_period_s", 2.0))

    def _fallback_yaw_direction(self, ts: float) -> float:
        phase = int(ts / max(self.fallback_sweep_period_s, 0.001))
        return 1.0 if phase % 2 == 0 else -1.0

    def compute(self, memory_snapshot: MemorySnapshot, ts: float) -> dict[str, Any]:
        side_h = memory_snapshot.last_seen_side_horizontal
        side_v = memory_snapshot.last_seen_side_vertical

        if side_h == "right":
            yaw_rate = +self.yaw_scan_speed
            mode = "last_seen_right"
        elif side_h == "left":
            yaw_rate = -self.yaw_scan_speed
            mode = "last_seen_left"
        else:
            yaw_rate = self._fallback_yaw_direction(ts) * self.yaw_scan_speed
            mode = "fallback_sweep"

        if side_v == "up":
            pitch_rate = -self.pitch_scan_speed
        elif side_v == "down":
            pitch_rate = +self.pitch_scan_speed
        else:
            pitch_rate = 0.0

        return {
            "yaw_rate": float(yaw_rate),
            "pitch_rate": float(pitch_rate),
            "mode": mode,
        }

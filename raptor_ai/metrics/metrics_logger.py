from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricsLogger:
    LOCK_EQUIVALENT_STATES = {"LOCKED", "TRACKING_XY", "TRACKING_XYZ"}

    def __init__(self, deadzone_x: float, deadzone_y: float):
        self.deadzone_x = float(deadzone_x)
        self.deadzone_y = float(deadzone_y)

        self.frames_total = 0
        self.frames_with_detection = 0
        self.frames_with_primary = 0
        self.search_frames = 0
        self.locked_frames = 0

        self._abs_err_x_sum = 0.0
        self._abs_err_y_sum = 0.0
        self.max_abs_err_x = 0.0
        self.max_abs_err_y = 0.0

        self.frames_in_deadzone_x = 0
        self.frames_in_deadzone_y = 0

        self._yaw_cmd_sum = 0.0
        self._pitch_cmd_sum = 0.0
        self._cmd_samples = 0

        self._sat_count = 0

        self.lost_target_count = 0
        self.reacquire_count = 0

        self._lock_duration_acc = 0
        self._lock_periods = 0
        self._lock_run = 0

    def record_event(self, event_name: str) -> None:
        if event_name == "target_lost":
            self.lost_target_count += 1
        elif event_name == "target_reacquired":
            self.reacquire_count += 1

    def record_frame(
        self,
        has_detection: bool,
        has_primary: bool,
        state: str,
        err_x: float | None,
        err_y: float | None,
        yaw_cmd: float,
        pitch_cmd: float,
        yaw_saturated: bool,
        pitch_saturated: bool,
    ) -> None:
        self.frames_total += 1

        if has_detection:
            self.frames_with_detection += 1
        if has_primary:
            self.frames_with_primary += 1
        if state == "SEARCHING":
            self.search_frames += 1
        if state in self.LOCK_EQUIVALENT_STATES:
            self.locked_frames += 1

        if state in self.LOCK_EQUIVALENT_STATES:
            self._lock_run += 1
        elif self._lock_run > 0:
            self._lock_duration_acc += self._lock_run
            self._lock_periods += 1
            self._lock_run = 0

        if has_primary and err_x is not None and err_y is not None:
            abs_x = abs(float(err_x))
            abs_y = abs(float(err_y))
            self._abs_err_x_sum += abs_x
            self._abs_err_y_sum += abs_y
            self.max_abs_err_x = max(self.max_abs_err_x, abs_x)
            self.max_abs_err_y = max(self.max_abs_err_y, abs_y)

            if abs_x <= self.deadzone_x:
                self.frames_in_deadzone_x += 1
            if abs_y <= self.deadzone_y:
                self.frames_in_deadzone_y += 1

        self._yaw_cmd_sum += float(yaw_cmd)
        self._pitch_cmd_sum += float(pitch_cmd)
        self._cmd_samples += 1

        if yaw_saturated or pitch_saturated:
            self._sat_count += 1

    def summary(self) -> dict[str, Any]:
        if self._lock_run > 0:
            self._lock_duration_acc += self._lock_run
            self._lock_periods += 1
            self._lock_run = 0

        primary_den = max(1, self.frames_with_primary)
        total_den = max(1, self.frames_total)
        cmd_den = max(1, self._cmd_samples)
        lock_den = max(1, self._lock_periods)

        return {
            "frames_total": self.frames_total,
            "frames_with_detection": self.frames_with_detection,
            "frames_with_primary": self.frames_with_primary,
            "presence_ratio": self.frames_with_primary / total_den,
            "lost_target_count": self.lost_target_count,
            "reacquire_count": self.reacquire_count,
            "mean_abs_err_x": self._abs_err_x_sum / primary_den,
            "mean_abs_err_y": self._abs_err_y_sum / primary_den,
            "max_abs_err_x": self.max_abs_err_x,
            "max_abs_err_y": self.max_abs_err_y,
            "frames_in_deadzone_x": self.frames_in_deadzone_x,
            "frames_in_deadzone_y": self.frames_in_deadzone_y,
            "deadzone_ratio_x": self.frames_in_deadzone_x / primary_den,
            "deadzone_ratio_y": self.frames_in_deadzone_y / primary_den,
            "search_frames": self.search_frames,
            "locked_frames": self.locked_frames,
            "mean_yaw_cmd": self._yaw_cmd_sum / cmd_den,
            "mean_pitch_cmd": self._pitch_cmd_sum / cmd_den,
            "command_saturation_ratio": self._sat_count / total_den,
            "avg_lock_duration_frames": self._lock_duration_acc / lock_den,
        }

    def write_summary(self, path: str | Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        out = self.summary()
        if extra:
            out.update(extra)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=True)
        return out

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


NO_TARGET = "NO_TARGET"
LOCKED = "LOCKED"
LOST = "LOST"
SEARCHING = "SEARCHING"
TRACKING_XY = "TRACKING_XY"
TRACKING_XYZ = "TRACKING_XYZ"


@dataclass
class StateUpdate:
    state: str
    events: list[dict[str, Any]]


class RuntimeStateMachine:
    def __init__(self, cfg: dict[str, Any]):
        self.state = NO_TARGET
        self.lost_frame_threshold = int(cfg.get("lost_frame_threshold", 8))
        self.reacquire_threshold = int(cfg.get("reacquire_threshold", 2))
        follow_mode = str(cfg.get("follow_mode", "off")).strip().lower()
        if follow_mode not in ("off", "xy", "xyz"):
            raise ValueError(f"Unsupported state_machine.follow_mode: {follow_mode}")
        self.follow_mode = follow_mode

        self._lost_count = 0
        self._reacquire_count = 0

    @property
    def tracking_state(self) -> str:
        if self.follow_mode == "xy":
            return TRACKING_XY
        if self.follow_mode == "xyz":
            return TRACKING_XYZ
        return LOCKED

    @staticmethod
    def _is_tracking_or_locked(state: str) -> bool:
        return state in (LOCKED, TRACKING_XY, TRACKING_XYZ)

    def _change_state(self, new_state: str, events: list[dict[str, Any]]) -> None:
        if self.state != new_state:
            events.append(
                {
                    "event": "state_changed",
                    "from": self.state,
                    "to": new_state,
                }
            )
            self.state = new_state

    def update(self, has_primary_target: bool) -> StateUpdate:
        events: list[dict[str, Any]] = []
        tracked_state = self.tracking_state

        if self.state == NO_TARGET:
            if has_primary_target:
                events.append({"event": "target_acquired"})
                self._lost_count = 0
                self._reacquire_count = 0
                self._change_state(tracked_state, events)
            return StateUpdate(state=self.state, events=events)

        if self._is_tracking_or_locked(self.state):
            if has_primary_target:
                self._lost_count = 0
                self._reacquire_count = 0
            else:
                self._lost_count = 1
                self._reacquire_count = 0
                events.append({"event": "target_lost"})
                self._change_state(LOST, events)
            return StateUpdate(state=self.state, events=events)

        if self.state == LOST:
            if has_primary_target:
                self._reacquire_count += 1
                if self._reacquire_count >= self.reacquire_threshold:
                    events.append({"event": "target_reacquired"})
                    self._lost_count = 0
                    self._reacquire_count = 0
                    self._change_state(tracked_state, events)
            else:
                self._reacquire_count = 0
                self._lost_count += 1
                if self._lost_count >= self.lost_frame_threshold:
                    events.append({"event": "search_started"})
                    self._change_state(SEARCHING, events)
            return StateUpdate(state=self.state, events=events)

        if self.state == SEARCHING:
            if has_primary_target:
                self._reacquire_count += 1
                if self._reacquire_count >= self.reacquire_threshold:
                    events.append({"event": "target_reacquired"})
                    events.append({"event": "search_stopped"})
                    self._lost_count = 0
                    self._reacquire_count = 0
                    self._change_state(tracked_state, events)
            else:
                self._reacquire_count = 0
            return StateUpdate(state=self.state, events=events)

        self._change_state(NO_TARGET, events)
        return StateUpdate(state=self.state, events=events)

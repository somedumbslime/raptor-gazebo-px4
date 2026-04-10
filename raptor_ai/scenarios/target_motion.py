from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

from gz.transport import Node

from raptor_ai.scenarios.gazebo_world import (
    compute_pose_ahead,
    create_model_from_uri,
    set_model_pose,
    wait_for_model_pose,
)


class TargetMotionThread(threading.Thread):
    def __init__(
        self,
        scenarios_cfg: dict[str, Any],
        profile_name: str,
        profile_cfg: dict[str, Any],
        motion_trace_path: str | None = None,
        verbosity: int = 1,
    ):
        super().__init__(daemon=True)
        self.scenarios_cfg = scenarios_cfg
        self.profile_name = profile_name
        self.profile_cfg = profile_cfg
        self.verbosity = verbosity
        self.motion_trace_path = motion_trace_path

        self.target_mode = str(scenarios_cfg.get("target_mode", "synthetic"))
        self.world_name = str(scenarios_cfg.get("world_name", "raptor_mvp"))
        self.model_name = str(scenarios_cfg.get("model_name", scenarios_cfg.get("target_name", "target_stub")))
        self.update_hz = float(scenarios_cfg.get("update_hz", 15.0))
        self.timeout_ms = int(scenarios_cfg.get("set_pose_timeout_ms", 1000))
        center = scenarios_cfg.get("center", [0.0, 0.0])
        self.center_x = float(center[0])
        self.center_y = float(center[1])
        self.base_z = float(scenarios_cfg.get("z", 0.85))
        self.spawn_if_missing = bool(scenarios_cfg.get("spawn_if_missing", False))
        self.spawn_uri = str(scenarios_cfg.get("spawn_uri", "")).strip()
        self.reference_model_name = str(scenarios_cfg.get("reference_model_name", "")).strip()
        self.reference_ahead_m = float(scenarios_cfg.get("reference_ahead_m", 8.0))
        self.reference_right_m = float(scenarios_cfg.get("reference_right_m", 0.0))
        self.reference_z = float(scenarios_cfg.get("reference_z", self.base_z))
        self.reference_yaw_mode = str(scenarios_cfg.get("reference_yaw_mode", "face_reference")).strip()
        self.reference_fixed_yaw = float(scenarios_cfg.get("reference_fixed_yaw", 0.0))
        self.reference_timeout_s = float(scenarios_cfg.get("reference_timeout_s", 3.0))
        self.center_from_reference = bool(scenarios_cfg.get("center_from_reference", True))

        self._stop_evt = threading.Event()
        self._ready_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def wait_ready(self, timeout: float = 2.0) -> bool:
        return self._ready_evt.wait(timeout)

    def _circle_pose(self, t: float, radius: float, speed: float) -> tuple[float, float, float, float]:
        omega = speed / max(radius, 1e-6)
        x = self.center_x + radius * math.cos(omega * t)
        y = self.center_y + radius * math.sin(omega * t)
        yaw = omega * t + math.pi / 2.0
        return x, y, self.base_z, yaw

    def _zigzag_pose(self, t: float, amplitude: float, period_s: float) -> tuple[float, float, float, float]:
        period = max(period_s, 0.5)
        x = self.center_x + amplitude * math.sin((2.0 * math.pi / period) * t)
        y = self.center_y + (0.6 * amplitude) * math.sin((4.0 * math.pi / period) * t)

        dx = (2.0 * math.pi / period) * amplitude * math.cos((2.0 * math.pi / period) * t)
        dy = (4.0 * math.pi / period) * 0.6 * amplitude * math.cos((4.0 * math.pi / period) * t)
        yaw = math.atan2(dy, dx)
        return x, y, self.base_z, yaw

    def _visibility_pose(self, t: float, hide_start_s: float, hide_duration_s: float) -> tuple[float, float, float] | None:
        hide_end = hide_start_s + hide_duration_s
        if hide_start_s <= t <= hide_end:
            hide_pose = self.profile_cfg.get("hide_pose", [20.0, 20.0, self.base_z])
            return float(hide_pose[0]), float(hide_pose[1]), float(hide_pose[2])
        return None

    def _pose_for_time(self, t: float) -> tuple[float, float, float, float]:
        mode = str(self.profile_cfg.get("mode", "circle"))

        if mode == "circle":
            radius = float(self.profile_cfg.get("radius", 1.8))
            speed = float(self.profile_cfg.get("speed", 0.7))
            return self._circle_pose(t, radius, speed)

        if mode == "zigzag":
            amplitude = float(self.profile_cfg.get("amplitude", 1.8))
            period_s = float(self.profile_cfg.get("period_s", 4.0))
            return self._zigzag_pose(t, amplitude, period_s)

        if mode in ("temporary_disappearance", "long_absence"):
            hidden = self._visibility_pose(
                t,
                hide_start_s=float(self.profile_cfg.get("hide_start_s", 8.0)),
                hide_duration_s=float(self.profile_cfg.get("hide_duration_s", 5.0)),
            )
            if hidden is not None:
                hx, hy, hz = hidden
                return hx, hy, hz, 0.0

            radius = float(self.profile_cfg.get("radius", 1.6))
            speed = float(self.profile_cfg.get("speed", 0.9))
            return self._circle_pose(t, radius, speed)

        radius = float(self.profile_cfg.get("radius", 1.8))
        speed = float(self.profile_cfg.get("speed", 0.7))
        return self._circle_pose(t, radius, speed)

    def _send_pose(self, node: Node, x: float, y: float, z: float, yaw: float) -> tuple[bool, bool]:
        ok, rep_data = set_model_pose(
            node=node,
            world_name=self.world_name,
            model_name=self.model_name,
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            timeout_ms=self.timeout_ms,
        )
        if self.verbosity > 1 and (not ok or not rep_data):
            print(f"[SCENARIO] set_pose failed | ok={ok} rep={rep_data}")
        return bool(ok), rep_data

    def _resolve_spawn_pose(self, node: Node) -> tuple[float, float, float, float]:
        x, y, z, yaw = self.center_x, self.center_y, self.base_z, 0.0
        if not self.reference_model_name:
            return x, y, z, yaw

        ref = wait_for_model_pose(
            node=node,
            world_name=self.world_name,
            model_name=self.reference_model_name,
            timeout_s=self.reference_timeout_s,
        )
        if ref is None:
            if self.verbosity:
                print(
                    f"[SCENARIO] reference pose not found: model={self.reference_model_name} "
                    f"world={self.world_name}; fallback center=({x:.2f},{y:.2f})"
                )
            return x, y, z, yaw

        sx, sy, sz, syaw = compute_pose_ahead(
            reference_pose=ref,
            ahead_m=self.reference_ahead_m,
            right_m=self.reference_right_m,
            z=self.reference_z,
            yaw_mode=self.reference_yaw_mode,
            fixed_yaw=self.reference_fixed_yaw,
        )

        if self.center_from_reference:
            self.center_x = sx
            self.center_y = sy
        if self.verbosity:
            print(
                f"[SCENARIO] spawn anchor: ref={ref.name} "
                f"-> x={sx:+.2f} y={sy:+.2f} z={sz:.2f} yaw={syaw:+.2f}"
            )
        return sx, sy, sz, syaw

    def _ensure_target_exists(self, node: Node) -> None:
        if not self.spawn_if_missing:
            return

        x, y, z, yaw = self._resolve_spawn_pose(node)

        # Fast path: model already exists.
        existing = wait_for_model_pose(
            node=node,
            world_name=self.world_name,
            model_name=self.model_name,
            timeout_s=max(0.2, min(self.reference_timeout_s, 1.0)),
        )
        if existing is not None:
            req_ok, rep_ok = self._send_pose(node, x, y, z, yaw)
            if req_ok and rep_ok:
                if self.verbosity:
                    print(f"[SCENARIO] target exists -> repositioned: {self.model_name}")
            else:
                if self.verbosity:
                    print(
                        f"[SCENARIO] target exists but pose update failed; "
                        f"continuing with current pose: {self.model_name}"
                    )
            return

        if not self.spawn_uri:
            if self.verbosity:
                print(
                    f"[SCENARIO] spawn_if_missing enabled but spawn_uri is empty "
                    f"(model={self.model_name})"
                )
            return

        create_ok, create_rep = create_model_from_uri(
            node=node,
            world_name=self.world_name,
            model_name=self.model_name,
            model_uri=self.spawn_uri,
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            timeout_ms=max(2000, self.timeout_ms),
        )
        if not (create_ok and create_rep):
            # Some worlds may already contain the actor (or reject pose updates for actor entities).
            # Re-check presence before declaring failure.
            existing_after_fail = wait_for_model_pose(
                node=node,
                world_name=self.world_name,
                model_name=self.model_name,
                timeout_s=0.5,
            )
            if existing_after_fail is not None:
                if self.verbosity:
                    print(f"[SCENARIO] target detected after create fail -> continuing: {self.model_name}")
                return
            if self.verbosity:
                print(
                    f"[SCENARIO] create failed | ok={create_ok} rep={create_rep} "
                    f"uri={self.spawn_uri}"
                )
            return

        time.sleep(0.15)
        self._send_pose(node, x, y, z, yaw)
        if self.verbosity:
            print(f"[SCENARIO] target spawned: {self.model_name} uri={self.spawn_uri}")

    def run(self) -> None:
        node = Node()
        t0 = time.time()
        dt = 1.0 / max(self.update_hz, 1e-6)

        self._ensure_target_exists(node)
        self._ready_evt.set()
        trace_file = None
        if self.motion_trace_path:
            trace_path = Path(self.motion_trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_file = trace_path.open("w", encoding="utf-8")

        if self.verbosity:
            print(
                f"[SCENARIO] {self.profile_name} started"
                f" | mode={self.target_mode} world={self.world_name} target={self.model_name}"
            )

        try:
            while not self._stop_evt.is_set():
                now = time.time()
                t = now - t0
                x, y, z, yaw = self._pose_for_time(t)
                req_ok, rep_ok = self._send_pose(node, x, y, z, yaw)

                if trace_file is not None:
                    trace_file.write(
                        json.dumps(
                            {
                                "ts": now,
                                "t": t,
                                "profile_name": self.profile_name,
                                "profile_mode": str(self.profile_cfg.get("mode", "circle")),
                                "target_mode": self.target_mode,
                                "world_name": self.world_name,
                                "model_name": self.model_name,
                                "x": x,
                                "y": y,
                                "z": z,
                                "yaw": yaw,
                                "set_pose_ok": req_ok,
                                "set_pose_response": rep_ok,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                    trace_file.flush()

                time.sleep(dt)
        finally:
            if trace_file is not None:
                trace_file.close()

        if self.verbosity:
            print(f"[SCENARIO] {self.profile_name} stopped")

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2

from raptor_ai.camera.gazebo_camera_source import GazeboCameraSource
from raptor_ai.config.loader import ensure_dir, load_config, resolve_camera_topic, resolve_platform_type
from raptor_ai.control.guidance_adapter import GuidanceAdapter
from raptor_ai.control.gimbal_controller import GimbalController
from raptor_ai.control.search_policy_last_seen import SearchPolicyLastSeen
from raptor_ai.detection.factory import build_detector
from raptor_ai.memory.target_memory import TargetMemory
from raptor_ai.metrics.event_logger import EventLogger
from raptor_ai.metrics.metrics_logger import MetricsLogger
from raptor_ai.platform.factory import build_platform
from raptor_ai.runtime.state_machine import RuntimeStateMachine
from raptor_ai.tracking.iou_tracker import IouTracker
from raptor_ai.tracking.primary_selector_adapter import PrimaryTargetSelectorAdapter


class RuntimeV2:
    @staticmethod
    def _parse_bgr_color(value: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                b = int(float(value[0]))
                g = int(float(value[1]))
                r = int(float(value[2]))
                return (
                    max(0, min(255, b)),
                    max(0, min(255, g)),
                    max(0, min(255, r)),
                )
            except (TypeError, ValueError):
                return fallback
        return fallback

    @staticmethod
    def _resolve_primary_color(
        selection_state: str | None,
        colors: dict[str, tuple[int, int, int]],
    ) -> tuple[int, int, int]:
        state = str(selection_state or "").strip().lower()
        if state == "locked":
            return colors["primary_locked"]
        if state == "switch_pending":
            return colors["primary_switch_pending"]
        if state == "lost":
            return colors["primary_lost"]
        return colors["primary_default"]

    def __init__(self, config_path: str, output_dir: str | None = None):
        self.config_path = str(config_path)
        self.cfg = load_config(config_path)

        logging_cfg = self.cfg.get("logging", {})
        default_output_dir = logging_cfg.get("output_dir", "runs/latest")
        self.output_dir = ensure_dir(output_dir or default_output_dir)

        self.events_path = self.output_dir / str(logging_cfg.get("events_file", "events.jsonl"))
        self.metrics_path = self.output_dir / str(logging_cfg.get("metrics_file", "metrics_summary.json"))
        self.run_meta_path = self.output_dir / str(logging_cfg.get("run_meta_file", "run_meta.json"))
        self.verbosity = int(logging_cfg.get("verbosity", 1))

        detector_cfg = self.cfg.get("detector", {})
        tracking_cfg = self.cfg.get("tracking", {})
        selector_cfg = self.cfg.get("selector", {})
        memory_cfg = self.cfg.get("memory", {})
        search_cfg = self.cfg.get("search_policy", {})
        controller_cfg = self.cfg.get("controller", {})
        follow_cfg = dict(self.cfg.get("follow", {}))
        guidance_cfg = self.cfg.get("guidance", {})
        platform_cfg = self.cfg.get("platform", {})
        state_cfg = self.cfg.get("state_machine", {})
        runtime_cfg = self.cfg.get("runtime", {})
        viz_cfg = self.cfg.get("viz", {})
        overlay_cfg = viz_cfg.get("overlay", {})

        self.control_hz = float(controller_cfg.get("control_hz", 20.0))
        self.dt = 1.0 / max(1e-6, self.control_hz)
        self.no_frame_sleep_s = float(runtime_cfg.get("no_frame_sleep_s", 0.002))
        self.no_frame_warn_interval_s = float(runtime_cfg.get("no_frame_warn_interval_s", 2.0))
        self.px4_skip_inference_until_offboard = bool(runtime_cfg.get("px4_skip_inference_until_offboard", True))
        self.px4_ready_warn_interval_s = float(runtime_cfg.get("px4_ready_warn_interval_s", 2.0))

        self.platform_type = resolve_platform_type(self.cfg)
        self._px4_auto_takeoff_for_ready_gate = False
        self._px4_auto_offboard_for_ready_gate = True
        self._px4_ready_min_alt_m = 0.6
        if self.platform_type == "px4":
            px4_cfg_merged: dict[str, Any] = dict(self.cfg.get("px4", {}))
            px4_cfg_merged.update(dict(platform_cfg.get("px4", {})))
            follow_cfg.setdefault("px4_lifecycle_enabled", True)
            follow_cfg.setdefault("px4_auto_arm", bool(px4_cfg_merged.get("auto_arm", False)))
            follow_cfg.setdefault("px4_auto_takeoff", bool(px4_cfg_merged.get("auto_takeoff", False)))
            follow_cfg.setdefault("px4_auto_offboard", bool(px4_cfg_merged.get("offboard_enabled", True)))
            follow_cfg.setdefault(
                "px4_auto_arm_require_armable",
                bool(px4_cfg_merged.get("auto_arm_require_armable", True)),
            )
            follow_cfg.setdefault(
                "px4_auto_arm_require_local_position",
                bool(px4_cfg_merged.get("auto_arm_require_local_position", False)),
            )
            follow_cfg.setdefault(
                "px4_offboard_min_relative_alt_m",
                float(px4_cfg_merged.get("offboard_min_relative_alt_m", 0.6)),
            )
            follow_cfg.setdefault(
                "px4_takeoff_confirm_alt_m",
                float(
                    px4_cfg_merged.get(
                        "takeoff_confirm_alt_m",
                        px4_cfg_merged.get("offboard_min_relative_alt_m", 0.6),
                    )
                ),
            )
            follow_cfg.setdefault(
                "px4_takeoff_liftoff_timeout_s",
                float(px4_cfg_merged.get("auto_takeoff_liftoff_timeout_s", 20.0)),
            )
            follow_cfg.setdefault(
                "px4_offboard_start_delay_after_liftoff_s",
                float(px4_cfg_merged.get("offboard_start_delay_after_liftoff_s", 1.2)),
            )
            follow_cfg.setdefault("px4_arm_retry_s", float(px4_cfg_merged.get("auto_arm_retry_s", 2.0)))
            follow_cfg.setdefault("px4_takeoff_retry_s", float(px4_cfg_merged.get("auto_takeoff_retry_s", 4.0)))
            follow_cfg.setdefault("px4_offboard_retry_s", 1.0)
            self._px4_auto_takeoff_for_ready_gate = bool(follow_cfg.get("px4_auto_takeoff", False))
            self._px4_auto_offboard_for_ready_gate = bool(follow_cfg.get("px4_auto_offboard", True))
            self._px4_ready_min_alt_m = float(follow_cfg.get("px4_offboard_min_relative_alt_m", 0.6))
        self.camera_topic = resolve_camera_topic(self.cfg, platform_type=self.platform_type)
        self.camera = GazeboCameraSource(topic=self.camera_topic)
        self.detector = build_detector(detector_cfg)
        self.detector_type = str(detector_cfg.get("type", "red"))
        self.detector_backend = str(getattr(self.detector, "backend", "n/a"))
        self.detector_model_path = str(getattr(self.detector, "model_path", ""))
        self.track_adapter = IouTracker(tracking_cfg)
        self.selector = PrimaryTargetSelectorAdapter(selector_cfg)
        self.selector_backend = str(selector_cfg.get("backend", "stub"))
        self.memory = TargetMemory(memory_cfg)
        self.search_policy = SearchPolicyLastSeen(search_cfg)
        self.controller = GimbalController(controller_cfg)
        self.guidance = GuidanceAdapter(follow_cfg=follow_cfg, guidance_cfg=guidance_cfg)
        self.guidance_backend = str(getattr(self.guidance, "backend", "legacy_internal"))
        self.follow_profile = str(follow_cfg.get("profile", "")).strip().lower()
        self._zone_track_box_scale_x = max(1.0, float(follow_cfg.get("zone_track_box_scale_x", 2.2)))
        self._zone_track_box_scale_y = max(1.0, float(follow_cfg.get("zone_track_box_scale_y", 2.0)))
        self._zone_track_center_deadzone_x = max(
            0.0,
            min(0.95, float(follow_cfg.get("zone_track_center_deadzone_x", 0.08))),
        )
        self._zone_track_center_deadzone_y = max(
            0.0,
            min(0.95, float(follow_cfg.get("zone_track_center_deadzone_y", 0.10))),
        )
        self.platform = build_platform(
            platform_cfg=platform_cfg,
            controller_cfg=controller_cfg,
            px4_cfg=self.cfg.get("px4", {}),
        )
        self.platform_meta = self.platform.metadata()
        self.platform_type = str(self.platform_meta.get("platform_type", self.platform_type))
        self.state_machine = RuntimeStateMachine(state_cfg)

        self.metrics = MetricsLogger(
            deadzone_x=float(controller_cfg.get("deadzone_x", 0.03)),
            deadzone_y=float(controller_cfg.get("deadzone_y", 0.04)),
        )
        self.event_logger = EventLogger(self.events_path)
        self.viz_colors = {
            "track_default": self._parse_bgr_color(overlay_cfg.get("track_default_color"), (80, 80, 80)),
            "primary_default": self._parse_bgr_color(overlay_cfg.get("primary_default_color"), (0, 255, 0)),
            "primary_locked": self._parse_bgr_color(overlay_cfg.get("primary_locked_color"), (0, 165, 255)),
            "primary_switch_pending": self._parse_bgr_color(
                overlay_cfg.get("primary_switch_pending_color"),
                (0, 255, 255),
            ),
            "primary_lost": self._parse_bgr_color(overlay_cfg.get("primary_lost_color"), (0, 0, 255)),
        }

        self._last_primary_track_id: int | None = None
        self._last_no_frame_warn_ts = 0.0
        self._last_px4_ready_warn_ts = 0.0
        self._px4_waiting_started_ts = 0.0
        self._px4_waiting_ready = False
        self._px4_wait_frames = 0
        self._px4_wait_last_meta: dict[str, Any] = {}
        self._last_platform_action_ts = 0.0

    def _execute_platform_action(self, ts: float, follow_cmd: dict[str, Any] | None) -> None:
        if self.platform_type != "px4":
            return
        if not isinstance(follow_cmd, dict):
            return
        action = str(follow_cmd.get("platform_action", "none")).strip().lower()
        if not action or action == "none":
            return

        payload = dict(follow_cmd.get("platform_action_payload", {}) or {})
        reason = str(payload.get("reason", ""))

        try:
            ok: bool | None = None
            if action == "arm" and hasattr(self.platform, "arm"):
                ok = bool(getattr(self.platform, "arm")())
            elif action == "takeoff" and hasattr(self.platform, "takeoff"):
                ok = bool(getattr(self.platform, "takeoff")())
            elif action == "ensure_offboard" and hasattr(self.platform, "ensure_offboard_started"):
                ok = bool(getattr(self.platform, "ensure_offboard_started")())
            elif action == "land" and hasattr(self.platform, "land"):
                ok = bool(getattr(self.platform, "land")())
            elif action == "disarm" and hasattr(self.platform, "disarm"):
                ok = bool(getattr(self.platform, "disarm")())
            elif action == "hold":
                if hasattr(self.platform, "stop_motion"):
                    getattr(self.platform, "stop_motion")()
                    ok = True
                else:
                    ok = False
            else:
                self.event_logger.log(ts, "platform_action_unsupported", action=action, reason=reason)
                return

            self._last_platform_action_ts = float(ts)
            self.event_logger.log(
                ts,
                "platform_action",
                action=action,
                ok=bool(ok) if ok is not None else True,
                reason=reason,
            )
        except Exception as exc:
            self.event_logger.log(
                ts,
                "platform_action_failed",
                action=action,
                reason=reason,
                error=str(exc),
            )

    @staticmethod
    def _compute_normalized_error(primary_track: dict[str, Any]) -> tuple[float, float]:
        cx, cy = primary_track["center"]
        frame_w = max(1, int(primary_track["frame_w"]))
        frame_h = max(1, int(primary_track["frame_h"]))
        frame_cx = frame_w * 0.5
        frame_cy = frame_h * 0.5
        err_x = (float(cx) - frame_cx) / frame_cx
        err_y = (float(cy) - frame_cy) / frame_cy
        return err_x, err_y

    def _is_px4_runtime_ready(self) -> tuple[bool, dict[str, Any]]:
        if self.platform_type != "px4":
            return True, {}

        try:
            meta = dict(self.platform.metadata() or {})
        except Exception:
            return True, {}

        auto_takeoff_enabled = bool(self._px4_auto_takeoff_for_ready_gate)
        if not self.px4_skip_inference_until_offboard or not auto_takeoff_enabled:
            return True, meta

        offboard_started = bool(meta.get("offboard_started", False)) or bool(meta.get("offboard_mode_active", False))
        rel_alt = float(meta.get("relative_altitude_m", 0.0) or 0.0)
        min_alt = float(self._px4_ready_min_alt_m)
        in_air = bool(meta.get("in_air", False))
        if self._px4_auto_offboard_for_ready_gate:
            ready = in_air and rel_alt >= min_alt and offboard_started
        else:
            ready = in_air and rel_alt >= min_alt
        return ready, meta

    def _log_state_events(self, ts: float, events: list[dict[str, Any]], search_cmd: dict[str, Any]) -> None:
        snapshot = self.memory.snapshot()

        for evt in events:
            name = str(evt.get("event", "unknown"))
            payload = dict(evt)
            payload.pop("event", None)

            if name == "target_lost":
                payload.setdefault("last_seen_side", snapshot.last_seen_side_horizontal)
            if name == "search_started":
                payload.setdefault("mode", search_cmd.get("mode", "fallback_sweep"))

            self.event_logger.log(ts, name, **payload)
            self.metrics.record_event(name)

    def _log_primary_selected(self, ts: float, selected_track: dict[str, Any], reason: str) -> None:
        track_id = int(selected_track.get("track_id", -1))
        if self._last_primary_track_id != track_id:
            self.event_logger.log(ts, "primary_selected", track_id=track_id, reason=reason)
            self._last_primary_track_id = track_id

    def _draw_viz_overlay(
        self,
        frame,
        tracks: list[dict[str, Any]],
        primary_track: dict[str, Any] | None,
        state: str,
        selection_state: str,
        err_x: float | None,
        err_y: float | None,
        yaw_cmd: float,
        pitch_cmd: float,
        follow_cmd: dict[str, Any] | None = None,
    ) -> None:
        h, w = frame.shape[:2]
        frame_cx, frame_cy = w // 2, h // 2

        cv2.circle(frame, (frame_cx, frame_cy), 4, (0, 255, 255), -1)
        cv2.line(frame, (frame_cx, 0), (frame_cx, h), (0, 255, 255), 1)
        cv2.line(frame, (0, frame_cy), (w, frame_cy), (0, 255, 255), 1)

        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr["bbox_xyxy"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.viz_colors["track_default"], 1)

        if primary_track is not None:
            x1, y1, x2, y2 = [int(v) for v in primary_track["bbox_xyxy"]]
            cx, cy = [int(v) for v in primary_track["center"]]
            primary_color = self._resolve_primary_color(selection_state, self.viz_colors)
            cv2.rectangle(frame, (x1, y1), (x2, y2), primary_color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.line(frame, (frame_cx, frame_cy), (cx, cy), (255, 0, 255), 2)
            self._draw_zone_track_overlay(frame, primary_track=primary_track, follow_cmd=follow_cmd)

        follow_mode = "-"
        follow_reason = "-"
        follow_vx = 0.0
        follow_vy = 0.0
        follow_yaw = 0.0
        follow_lock_frames = 0
        follow_action = "none"
        if follow_cmd:
            follow_mode = str(follow_cmd.get("mode", "-"))
            follow_reason = str(follow_cmd.get("reason", "-"))
            follow_vx = float(follow_cmd.get("vx_body", 0.0))
            follow_vy = float(follow_cmd.get("vy_body", 0.0))
            follow_yaw = float(follow_cmd.get("yaw_rate", 0.0))
            follow_lock_frames = int(follow_cmd.get("center_lock_frames", 0))
            follow_action = str(follow_cmd.get("platform_action", "none"))

        status = (
            f"state={state} | sel={selection_state} | tracks={len(tracks)} "
            f"| primary={'yes' if primary_track is not None else 'no'} "
            f"| ex={0.0 if err_x is None else err_x:+.3f} ey={0.0 if err_y is None else err_y:+.3f} "
            f"| yaw={yaw_cmd:+.3f} pitch={pitch_cmd:+.3f} "
            f"| fmode={follow_mode} fprof={self.follow_profile or '-'} freason={follow_reason} "
            f"fvx={follow_vx:+.2f} fvy={follow_vy:+.2f} fyr={follow_yaw:+.2f} "
            f"flock={follow_lock_frames} fact={follow_action}"
        )
        cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_zone_track_overlay(
        self,
        frame,
        *,
        primary_track: dict[str, Any],
        follow_cmd: dict[str, Any] | None,
    ) -> None:
        if not isinstance(follow_cmd, dict):
            return
        if str(follow_cmd.get("xy_strategy", "")).strip().lower() != "zone_track":
            return

        bbox = primary_track.get("bbox_xyxy")
        center = primary_track.get("center")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return
        if not (isinstance(center, (list, tuple)) and len(center) == 2):
            return

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx, cy = [float(v) for v in center]
        except (TypeError, ValueError):
            return

        bw = max(1.0, abs(x2 - x1))
        bh = max(1.0, abs(y2 - y1))
        gw = bw * self._zone_track_box_scale_x
        gh = bh * self._zone_track_box_scale_y
        gx1 = int(round(cx - (gw * 0.5)))
        gy1 = int(round(cy - (gh * 0.5)))
        gx2 = int(round(cx + (gw * 0.5)))
        gy2 = int(round(cy + (gh * 0.5)))

        guide_color = (120, 255, 120)
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), guide_color, 1)
        cv2.line(frame, (int(round(cx)), gy1), (int(round(cx)), gy2), guide_color, 1)
        cv2.line(frame, (gx1, int(round(cy))), (gx2, int(round(cy))), guide_color, 1)

        dz_w = max(2.0, gw * self._zone_track_center_deadzone_x)
        dz_h = max(2.0, gh * self._zone_track_center_deadzone_y)
        dz_x1 = int(round(cx - (dz_w * 0.5)))
        dz_y1 = int(round(cy - (dz_h * 0.5)))
        dz_x2 = int(round(cx + (dz_w * 0.5)))
        dz_y2 = int(round(cy + (dz_h * 0.5)))
        cv2.rectangle(frame, (dz_x1, dz_y1), (dz_x2, dz_y2), (0, 220, 255), 1)

        reason = str(follow_cmd.get("reason", "")).strip().lower()
        zone = ""
        if reason.startswith("zone_track_terminal_"):
            zone = reason.replace("zone_track_terminal_", "terminal:")
        elif reason.startswith("zone_track_"):
            zone = reason.replace("zone_track_", "")
        if zone:
            label = f"zone={zone}"
            tx = max(6, gx1)
            ty = max(18, gy1 - 6)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, guide_color, 1, cv2.LINE_AA)

    @staticmethod
    def _show_viz(frame) -> bool:
        cv2.imshow("RAPTOR RuntimeV2", frame)
        key = cv2.waitKey(1) & 0xFF
        return key in (27, ord("q"))

    @staticmethod
    def _create_video_writer(path: Path, frame_w: int, frame_h: int, fps: float) -> cv2.VideoWriter | None:
        codecs = ["mp4v", "XVID", "MJPG"]
        for codec in codecs:
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*codec),
                float(max(1.0, fps)),
                (int(frame_w), int(frame_h)),
            )
            if writer.isOpened():
                return writer
            writer.release()
        return None

    def run(
        self,
        duration_s: float | None = None,
        viz: bool = False,
        record: bool = False,
        record_path: str | None = None,
        record_fps: float | None = None,
    ) -> dict[str, Any]:
        start_ts = time.time()
        stop_reason = "duration"
        self.event_logger.log(start_ts, "runtime_started", config_path=self.config_path)
        video_path: Path | None = None
        video_writer: cv2.VideoWriter | None = None
        recording_enabled = bool(record)
        recording_fps = float(record_fps) if record_fps is not None else self.control_hz
        if recording_enabled:
            video_path = Path(record_path) if record_path else (self.output_dir / "runtime_viz.mp4")
            video_path.parent.mkdir(parents=True, exist_ok=True)
            self.event_logger.log(
                start_ts,
                "recording_requested",
                path=str(video_path),
                fps=recording_fps,
            )

        if self.verbosity:
            print(f"[RUNTIME] output_dir={self.output_dir}")
            print(f"[RUNTIME] control_hz={self.control_hz:.2f} viz={viz} record={recording_enabled}")
            print(f"[RUNTIME] platform={self.platform_type} camera_topic={self.camera_topic}")
            print(
                f"[RUNTIME] detector={self.detector_type} "
                f"backend={self.detector_backend} "
                f"model={self.detector_model_path if self.detector_model_path else '-'}"
            )
            print(f"[RUNTIME] guidance_backend={self.guidance_backend}")
            if recording_enabled and video_path is not None:
                print(f"[RUNTIME] recording_path={video_path}")

        try:
            while True:
                loop_t0 = time.time()

                if duration_s is not None and (loop_t0 - start_ts) >= float(duration_s):
                    stop_reason = "duration"
                    break

                frame, frame_ts = self.camera.get_latest_frame()
                if frame is None:
                    now = time.time()
                    if now - self._last_no_frame_warn_ts >= self.no_frame_warn_interval_s:
                        if self.verbosity:
                            print("[RUNTIME] waiting for camera frames...")
                        self._last_no_frame_warn_ts = now
                    time.sleep(self.no_frame_sleep_s)
                    continue

                ts = float(frame_ts if frame_ts is not None else time.time())

                px4_ready, px4_meta = self._is_px4_runtime_ready()
                self._px4_wait_last_meta = dict(px4_meta)
                if not px4_ready:
                    if not self._px4_waiting_ready:
                        self._px4_waiting_started_ts = ts
                        self._px4_waiting_ready = True
                        self.event_logger.log(
                            ts,
                            "px4_ready_wait_started",
                            auto_takeoff=bool(self._px4_auto_takeoff_for_ready_gate),
                            auto_offboard=bool(self._px4_auto_offboard_for_ready_gate),
                            offboard_started=bool(px4_meta.get("offboard_started", False)),
                            in_air=bool(px4_meta.get("in_air", False)),
                            rel_alt=float(px4_meta.get("relative_altitude_m", 0.0) or 0.0),
                            min_alt=float(self._px4_ready_min_alt_m),
                        )
                    self._px4_wait_frames += 1
                    # Keep platform command loop alive while PX4 lifecycle is being managed.
                    follow_cmd = self.guidance.compute(
                        state=self.state_machine.state,
                        primary_track=None,
                        err_x=None,
                        err_y=None,
                        dt=self.dt,
                        ts=ts,
                        platform_meta=px4_meta,
                    )
                    wait_action = str(follow_cmd.get("platform_action", "none")) if isinstance(follow_cmd, dict) else "none"
                    wait_reason = ""
                    if isinstance(follow_cmd, dict):
                        wait_reason = str((follow_cmd.get("platform_action_payload", {}) or {}).get("reason", ""))

                    now = time.time()
                    if now - self._last_px4_ready_warn_ts >= self.px4_ready_warn_interval_s:
                        if self.verbosity:
                            print(
                                "[RUNTIME] waiting PX4 ready (gating inference): "
                                f"armed={bool(px4_meta.get('armed', False))} "
                                f"in_air={bool(px4_meta.get('in_air', False))} "
                                f"mode={str(px4_meta.get('flight_mode', '-'))} "
                                f"rel_alt={float(px4_meta.get('relative_altitude_m', 0.0) or 0.0):.2f} "
                                f"offboard={bool(px4_meta.get('offboard_started', False))} "
                                f"err={str(px4_meta.get('last_error', None))} "
                                f"next_action={wait_action} reason={wait_reason}"
                            )
                        self._last_px4_ready_warn_ts = now

                    self._execute_platform_action(ts, follow_cmd)
                    self.platform.publish_commands(
                        0.0,
                        0.0,
                        state=self.state_machine.state,
                        follow_cmd=follow_cmd,
                    )

                    if viz or recording_enabled:
                        viz_frame = frame.copy()
                        wait_status = (
                            "PX4_WAIT_READY "
                            f"mode={str(px4_meta.get('flight_mode', '-'))} "
                            f"alt={float(px4_meta.get('relative_altitude_m', 0.0) or 0.0):.2f} "
                            f"offboard={bool(px4_meta.get('offboard_started', False))}"
                        )
                        cv2.putText(
                            viz_frame,
                            wait_status,
                            (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        if recording_enabled:
                            if video_writer is None and video_path is not None:
                                h, w = viz_frame.shape[:2]
                                video_writer = self._create_video_writer(video_path, w, h, recording_fps)
                                if video_writer is None:
                                    recording_enabled = False
                                    self.event_logger.log(
                                        ts,
                                        "recording_failed",
                                        path=str(video_path),
                                        reason="video_writer_init_failed",
                                    )
                                    if self.verbosity:
                                        print(f"[RUNTIME] recording init failed: {video_path}")
                                else:
                                    self.event_logger.log(ts, "recording_started", path=str(video_path), fps=recording_fps)
                            if recording_enabled and video_writer is not None:
                                video_writer.write(viz_frame)

                        if viz:
                            should_quit = self._show_viz(viz_frame)
                            if should_quit:
                                stop_reason = "user_quit"
                                break

                    spent = time.time() - loop_t0
                    time.sleep(max(0.0, self.dt - spent))
                    continue
                elif self._px4_waiting_ready:
                    wait_dt = max(0.0, ts - self._px4_waiting_started_ts)
                    self.event_logger.log(
                        ts,
                        "px4_ready_wait_finished",
                        wait_s=wait_dt,
                        wait_frames=int(self._px4_wait_frames),
                        rel_alt=float(px4_meta.get("relative_altitude_m", 0.0) or 0.0),
                        offboard_started=bool(px4_meta.get("offboard_started", False)),
                    )
                    self._px4_waiting_ready = False

                detections = self.detector.detect(frame)
                tracks = self.track_adapter.to_tracks(detections, frame.shape, ts)

                selection = self.selector.select_primary(
                    tracks,
                    context={
                        "timestamp": ts,
                        "state": self.state_machine.state,
                        "frame_index": self.metrics.frames_total,
                        "frame_size": (int(frame.shape[1]), int(frame.shape[0])),
                    },
                )
                primary_track = selection.get("selected_primary_target")
                selection_state = str(selection.get("selection_state", "none"))
                has_primary = primary_track is not None

                state_update = self.state_machine.update(has_primary)
                state = state_update.state

                err_x: float | None = None
                err_y: float | None = None

                if primary_track is not None:
                    err_x, err_y = self._compute_normalized_error(primary_track)
                    self.memory.update(
                        selected_track=primary_track,
                        control_state=state,
                        timestamp=ts,
                        err_x=err_x,
                        err_y=err_y,
                        yaw=self.controller.yaw_cmd,
                        pitch=self.controller.pitch_cmd,
                    )
                    self._log_primary_selected(ts, primary_track, str(selection.get("selection_reason", "unknown")))
                else:
                    self._last_primary_track_id = None

                search_cmd = (
                    self.search_policy.compute(self.memory.snapshot(), ts)
                    if state == "SEARCHING"
                    else {"yaw_rate": 0.0, "pitch_rate": 0.0, "mode": "idle"}
                )

                yaw_cmd, pitch_cmd, ctrl_dbg = self.controller.compute(
                    state=state,
                    primary_track=primary_track,
                    err_x=err_x,
                    err_y=err_y,
                    search_cmd=search_cmd,
                    dt=self.dt,
                )
                follow_cmd = self.guidance.compute(
                    state=state,
                    primary_track=primary_track,
                    err_x=err_x,
                    err_y=err_y,
                    dt=self.dt,
                    ts=ts,
                    platform_meta=px4_meta,
                )
                self._execute_platform_action(ts, follow_cmd)
                self.platform.publish_commands(yaw_cmd, pitch_cmd, state=state, follow_cmd=follow_cmd)

                self._log_state_events(ts, state_update.events, search_cmd)

                self.metrics.record_frame(
                    has_detection=bool(detections),
                    has_primary=has_primary,
                    state=state,
                    err_x=err_x,
                    err_y=err_y,
                    yaw_cmd=yaw_cmd,
                    pitch_cmd=pitch_cmd,
                    yaw_saturated=bool(ctrl_dbg.get("yaw_saturated", False)),
                    pitch_saturated=bool(ctrl_dbg.get("pitch_saturated", False)),
                )

                if viz or recording_enabled:
                    viz_frame = frame.copy()
                    self._draw_viz_overlay(
                        frame=viz_frame,
                        tracks=tracks,
                        primary_track=primary_track,
                        state=state,
                        selection_state=selection_state,
                        err_x=err_x,
                        err_y=err_y,
                        yaw_cmd=yaw_cmd,
                        pitch_cmd=pitch_cmd,
                        follow_cmd=follow_cmd,
                    )

                    if recording_enabled:
                        if video_writer is None and video_path is not None:
                            h, w = viz_frame.shape[:2]
                            video_writer = self._create_video_writer(video_path, w, h, recording_fps)
                            if video_writer is None:
                                recording_enabled = False
                                self.event_logger.log(
                                    ts,
                                    "recording_failed",
                                    path=str(video_path),
                                    reason="video_writer_init_failed",
                                )
                                if self.verbosity:
                                    print(f"[RUNTIME] recording init failed: {video_path}")
                            else:
                                self.event_logger.log(ts, "recording_started", path=str(video_path), fps=recording_fps)

                        if recording_enabled and video_writer is not None:
                            video_writer.write(viz_frame)

                    if viz:
                        should_quit = self._show_viz(viz_frame)
                        if should_quit:
                            stop_reason = "user_quit"
                            break

                spent = time.time() - loop_t0
                sleep_time = max(0.0, self.dt - spent)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"

        finally:
            close_fn = getattr(self.platform, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as exc:
                    if self.verbosity:
                        print(f"[RUNTIME] WARN: platform close failed: {exc}")
            if video_writer is not None:
                video_writer.release()
            if viz:
                cv2.destroyAllWindows()

        end_ts = time.time()
        if record and video_path is not None:
            self.event_logger.log(
                end_ts,
                "recording_stopped",
                path=str(video_path),
                enabled=recording_enabled,
            )
        self.event_logger.log(end_ts, "runtime_stopped", reason=stop_reason)
        self.event_logger.close()

        summary = self.metrics.write_summary(
            self.metrics_path,
            extra={
                "stop_reason": stop_reason,
                "duration_s": end_ts - start_ts,
                "output_dir": str(self.output_dir),
                "record_enabled": bool(record),
                "record_path": str(video_path) if video_path is not None else "",
            },
        )

        self.platform_meta = self.platform.metadata()
        run_meta = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_s": end_ts - start_ts,
            "stop_reason": stop_reason,
            "config_path": self.config_path,
            "events_path": str(self.events_path),
            "metrics_path": str(self.metrics_path),
            "camera_topic": self.camera_topic,
            "detector_type": self.detector_type,
            "detector_backend": self.detector_backend,
            "detector_model_path": (self.detector_model_path if self.detector_model_path else None),
            "detector_providers": getattr(self.detector, "providers", None),
            "selector_backend": self.selector_backend,
            "selector_external_callable": str(self.cfg.get("selector", {}).get("external_callable", "")),
            "tracker_type": "iou",
            "platform_type": self.platform_type,
            "follow_mode": str(getattr(self.state_machine, "follow_mode", "off")),
            "guidance_backend": self.guidance_backend,
            "guidance_external_callable": str(self.cfg.get("guidance", {}).get("external_callable", "")),
            "follow_enabled": bool(getattr(self.guidance, "enabled", False)),
            "follow_controller_mode": str(getattr(self.guidance, "mode", "xy")),
            "follow_xy_strategy": str(getattr(self.guidance, "xy_strategy", "zone_track")),
            "follow_profile": str(self.cfg.get("follow", {}).get("profile", "")),
            "platform_meta": self.platform_meta,
            "record_enabled": bool(record),
            "record_path": (str(video_path) if video_path is not None else None),
            "record_fps": recording_fps if record else None,
            "px4_inference_gated": bool(self.px4_skip_inference_until_offboard and self.platform_type == "px4"),
            "px4_ready_gate_auto_takeoff": bool(self._px4_auto_takeoff_for_ready_gate),
            "px4_ready_gate_auto_offboard": bool(self._px4_auto_offboard_for_ready_gate),
            "px4_ready_gate_min_alt_m": float(self._px4_ready_min_alt_m),
            "px4_wait_ready_frames": int(self._px4_wait_frames),
            "px4_wait_ready_last_meta": self._px4_wait_last_meta,
        }
        if "yaw_topic" in self.platform_meta:
            run_meta["yaw_topic"] = self.platform_meta.get("yaw_topic")
        if "pitch_topic" in self.platform_meta:
            run_meta["pitch_topic"] = self.platform_meta.get("pitch_topic")
        with Path(self.run_meta_path).open("w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2, ensure_ascii=True)

        if self.verbosity:
            print(f"[RUNTIME] done | reason={stop_reason} | frames={summary.get('frames_total', 0)}")
            print(f"[RUNTIME] metrics={self.metrics_path}")
            print(f"[RUNTIME] events={self.events_path}")

        return summary

#!/usr/bin/env /usr/bin/python3
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raptor_ai.config.loader import load_config, resolve_camera_topic, resolve_platform_type
from raptor_ai.camera.topic_discovery import discover_image_topics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Gazebo camera viewer")
    parser.add_argument(
        "--config",
        type=str,
        default="raptor_ai/config/default_config.yaml",
        help="Path to runtime YAML config",
    )
    parser.add_argument(
        "--platform-type",
        type=str,
        default=None,
        help="Platform type override for camera topic resolve (gimbal/iris/px4)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Direct Gazebo image topic override",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="RAPTOR camera",
        help="OpenCV window name",
    )
    parser.add_argument(
        "--stats-hz",
        type=float,
        default=1.0,
        help="Console stats print frequency",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=4.0,
        help="Wait timeout for first frame on initial topic before auto-discovery",
    )
    parser.add_argument(
        "--auto-discover-topic",
        action="store_true",
        default=True,
        help="Auto-discover a working image topic if initial topic has no frames (default: true)",
    )
    parser.add_argument(
        "--no-auto-discover-topic",
        dest="auto_discover_topic",
        action="store_false",
    )
    parser.add_argument(
        "--discover-timeout-per-topic",
        type=float,
        default=1.2,
        help="Time budget per discovered topic to check first frame",
    )
    parser.add_argument(
        "--quiet-gz-import",
        action="store_true",
        default=True,
        help="Suppress noisy protobuf stderr lines during gz import/init (default: true)",
    )
    parser.add_argument(
        "--no-quiet-gz-import",
        dest="quiet_gz_import",
        action="store_false",
    )
    return parser.parse_args()


@contextlib.contextmanager
def _suppress_stderr(enabled: bool):
    if not enabled:
        yield
        return
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return

    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


def resolve_topic(args: argparse.Namespace) -> tuple[str, str]:
    if args.topic:
        return str(args.topic), "manual"

    cfg = load_config(args.config)
    ptype = str(args.platform_type).strip().lower() if args.platform_type else resolve_platform_type(cfg)
    topic = resolve_camera_topic(cfg, platform_type=ptype)
    return topic, ptype


def main() -> None:
    args = parse_args()
    with _suppress_stderr(bool(args.quiet_gz_import)):
        from raptor_ai.camera.gazebo_camera_source import GazeboCameraSource

    initial_topic, topic_source = resolve_topic(args)
    with _suppress_stderr(bool(args.quiet_gz_import)):
        src = GazeboCameraSource(topic=initial_topic)

    active_topic = initial_topic
    print(f"[CAM_VIEW] topic={active_topic} source={topic_source}")
    print(f"[CAM_VIEW] GZ_PARTITION={os.environ.get('GZ_PARTITION', '<unset>')}")
    print("[CAM_VIEW] q/ESC: exit | s: save frame")

    stats_period_s = 1.0 / max(0.2, float(args.stats_hz))
    last_stats_ts = 0.0
    frame_counter = 0
    t0 = time.time()
    first_frame_received = False

    def _make_waiting_frame(topic_text: str, stage_text: str) -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Waiting for camera frames...",
            (40, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            stage_text,
            (40, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"topic={topic_text}",
            (40, 285),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _render_wait(topic_text: str, stage_text: str = "startup wait") -> bool:
        wait_frame = _make_waiting_frame(topic_text, stage_text)
        cv2.imshow(args.window_name, wait_frame)
        key = cv2.waitKey(1) & 0xFF
        return key in (27, ord("q"))

    if _render_wait(active_topic, "startup wait"):
        cv2.destroyAllWindows()
        return

    # Initial wait on configured topic.
    startup_t0 = time.time()
    while (time.time() - startup_t0) < max(0.2, float(args.startup_timeout)):
        frame, _ = src.get_latest_frame()
        if frame is not None:
            first_frame_received = True
            break
        if _render_wait(active_topic, "startup wait"):
            cv2.destroyAllWindows()
            return
        time.sleep(0.01)

    if not first_frame_received and bool(args.auto_discover_topic):
        print(f"[CAM_VIEW] no frames on {initial_topic}, trying topic auto-discovery...")
        discovered = [t for t in discover_image_topics() if t != initial_topic]
        if discovered:
            print(f"[CAM_VIEW] discovered image topics: {discovered}")
        else:
            print("[CAM_VIEW] no image topics discovered (check Gazebo is running and GZ_PARTITION matches)")
        for cand in discovered:
            with _suppress_stderr(bool(args.quiet_gz_import)):
                cand_src = GazeboCameraSource(topic=cand)
            probe_t0 = time.time()
            ok = False
            while (time.time() - probe_t0) < max(0.2, float(args.discover_timeout_per_topic)):
                f, _ = cand_src.get_latest_frame()
                if f is not None:
                    ok = True
                    break
                if _render_wait(cand, "probing discovered topic"):
                    cv2.destroyAllWindows()
                    return
                time.sleep(0.01)
            if ok:
                src = cand_src
                active_topic = cand
                first_frame_received = True
                print(f"[CAM_VIEW] switched to discovered topic: {active_topic}")
                break

    waiting_frame = _make_waiting_frame(active_topic, "streaming wait")

    first_warn_printed = False
    while True:
        frame, ts = src.get_latest_frame()
        if frame is not None:
            frame_counter += 1
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            cv2.putText(
                frame,
                f"topic={active_topic}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(args.window_name, frame)

            now = time.time()
            if (now - last_stats_ts) >= stats_period_s:
                elapsed = max(1e-6, now - t0)
                print(
                    f"[CAM_VIEW] frames={frame_counter} avg_fps={frame_counter / elapsed:.2f} "
                    f"size={w}x{h} ts={ts:.3f}"
                )
                last_stats_ts = now
        else:
            cv2.imshow(args.window_name, waiting_frame)
            if not first_warn_printed:
                print(f"[CAM_VIEW] waiting frames on topic={active_topic}")
                first_warn_printed = True
            time.sleep(0.005)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s") and frame is not None:
            out_path = f"/tmp/raptor_frame_{int(time.time())}.png"
            cv2.imwrite(out_path, frame)
            print(f"[CAM_VIEW] saved: {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

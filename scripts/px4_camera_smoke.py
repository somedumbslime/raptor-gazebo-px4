#!/usr/bin/env /usr/bin/python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raptor_ai.camera.topic_discovery import discover_image_topics
from raptor_ai.config.loader import load_config, resolve_camera_topic


def _clamp(v: float, low: float, high: float) -> float:
    return max(low, min(high, v))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PX4 camera smoke-check (topic/fps/frame-size)")
    parser.add_argument("--config", type=str, default="raptor_ai/config/default_config.yaml", help="Config path")
    parser.add_argument("--platform-type", type=str, default="px4", help="Contract profile name in camera.contract")
    parser.add_argument("--topic", type=str, default=None, help="Topic override")
    parser.add_argument("--duration", type=float, default=12.0, help="Measurement duration after first frame")
    parser.add_argument("--startup-timeout", type=float, default=8.0, help="Wait timeout for first frame")
    parser.add_argument("--viz", action="store_true", help="Show camera window while measuring")
    parser.add_argument("--window-name", type=str, default="PX4 Camera Smoke", help="OpenCV window name")
    parser.add_argument("--out-json", type=str, default=None, help="Optional output report json path")
    parser.add_argument("--expected-width", type=int, default=None, help="Override contract expected_width")
    parser.add_argument("--expected-height", type=int, default=None, help="Override contract expected_height")
    parser.add_argument("--min-fps", type=float, default=None, help="Override contract min_fps")
    parser.add_argument("--max-no-frame-gap-s", type=float, default=None, help="Override contract max_no_frame_gap_s")
    parser.add_argument("--stats-hz", type=float, default=1.0, help="Console stats print frequency")
    parser.add_argument(
        "--auto-discover-topic",
        action="store_true",
        default=True,
        help="Auto-discover working image topic if initial topic has no frames (default: true)",
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


def _load_camera_contract(cfg: dict[str, Any], platform_type: str) -> dict[str, Any]:
    camera_cfg = cfg.get("camera", {})
    if not isinstance(camera_cfg, dict):
        return {}
    contract_cfg = camera_cfg.get("contract", {})
    if not isinstance(contract_cfg, dict):
        return {}
    profile = contract_cfg.get(platform_type, {})
    if not isinstance(profile, dict):
        return {}
    return dict(profile)


def _build_contract(args: argparse.Namespace, cfg: dict[str, Any], platform_type: str) -> dict[str, Any]:
    c = _load_camera_contract(cfg, platform_type)
    contract = {
        "expected_width": c.get("expected_width"),
        "expected_height": c.get("expected_height"),
        "min_fps": c.get("min_fps"),
        "max_no_frame_gap_s": c.get("max_no_frame_gap_s", 1.0),
    }
    if args.expected_width is not None:
        contract["expected_width"] = int(args.expected_width)
    if args.expected_height is not None:
        contract["expected_height"] = int(args.expected_height)
    if args.min_fps is not None:
        contract["min_fps"] = float(args.min_fps)
    if args.max_no_frame_gap_s is not None:
        contract["max_no_frame_gap_s"] = float(args.max_no_frame_gap_s)
    return contract


def _wait_first_frame(
    src: Any,
    timeout_s: float,
) -> tuple[Any | None, float | None]:
    t0 = time.time()
    while (time.time() - t0) < max(0.2, float(timeout_s)):
        frame, ts = src.get_latest_frame()
        if frame is not None and ts is not None:
            return frame, float(ts)
        time.sleep(0.01)
    return None, None


def main() -> None:
    args = parse_args()
    with _suppress_stderr(bool(args.quiet_gz_import)):
        from raptor_ai.camera.gazebo_camera_source import GazeboCameraSource

    cfg = load_config(args.config)
    platform_type = str(args.platform_type).strip().lower()
    topic = str(args.topic).strip() if args.topic else resolve_camera_topic(cfg, platform_type=platform_type)
    contract = _build_contract(args, cfg, platform_type)

    print(
        f"[PX4_CAM_SMOKE] topic={topic} platform_type={platform_type} "
        f"duration={float(args.duration):.1f}s startup_timeout={float(args.startup_timeout):.1f}s"
    )
    print(f"[PX4_CAM_SMOKE] GZ_PARTITION={os.environ.get('GZ_PARTITION', '<unset>')}")
    print(f"[PX4_CAM_SMOKE] contract={contract}")

    with _suppress_stderr(bool(args.quiet_gz_import)):
        src = GazeboCameraSource(topic=topic)
    active_topic = topic

    first_frame, first_ts = _wait_first_frame(src, float(args.startup_timeout))
    discovered_candidates: list[str] = []
    used_discovery_fallback = False

    if first_frame is None and bool(args.auto_discover_topic):
        print(f"[PX4_CAM_SMOKE] no frames on {topic}, trying topic auto-discovery...")
        discovered_candidates = [t for t in discover_image_topics() if t != topic]
        if discovered_candidates:
            print(f"[PX4_CAM_SMOKE] discovered image topics: {discovered_candidates}")
        else:
            print("[PX4_CAM_SMOKE] no image topics discovered (check Gazebo is running and GZ_PARTITION matches)")

        for cand in discovered_candidates:
            with _suppress_stderr(bool(args.quiet_gz_import)):
                cand_src = GazeboCameraSource(topic=cand)
            cand_frame, cand_ts = _wait_first_frame(cand_src, float(args.discover_timeout_per_topic))
            if cand_frame is not None and cand_ts is not None:
                src = cand_src
                active_topic = cand
                first_frame = cand_frame
                first_ts = cand_ts
                used_discovery_fallback = True
                print(f"[PX4_CAM_SMOKE] switched to discovered topic: {active_topic}")
                break

    if first_frame is None:
        report = {
            "passed": False,
            "platform_type": platform_type,
            "topic": active_topic,
            "configured_topic": topic,
            "used_discovery_fallback": used_discovery_fallback,
            "discovered_candidates": discovered_candidates,
            "duration_s": 0.0,
            "frames_unique": 0,
            "avg_fps": 0.0,
            "max_no_frame_gap_s": 0.0,
            "last_frame_size": {"w": 0, "h": 0},
            "size_mismatch_count": 0,
            "contract": contract,
            "checks": {
                "has_frames": False,
                "size_ok": False,
                "fps_ok": False,
                "gap_ok": False,
            },
            "failure_reason": "no_frames_startup_timeout",
            "timestamp": time.time(),
        }
        if args.out_json:
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
            print(f"[PX4_CAM_SMOKE] report={out_path}")
        print("[PX4_CAM_SMOKE] FAIL: no frames in startup timeout")
        if discovered_candidates:
            print(f"[PX4_CAM_SMOKE] discovered topics (none produced frames): {discovered_candidates}")
        raise SystemExit(2)

    last_unique_ts = first_ts
    measure_t0 = time.time()
    last_frame_wall_ts = measure_t0
    unique_frames = 1
    max_no_frame_gap_s = 0.0
    size_mismatch_count = 0
    last_size = (int(first_frame.shape[1]), int(first_frame.shape[0]))
    expected_w = contract.get("expected_width")
    expected_h = contract.get("expected_height")

    if isinstance(expected_w, int) and isinstance(expected_h, int):
        if last_size != (expected_w, expected_h):
            size_mismatch_count += 1

    stats_period_s = 1.0 / max(0.2, float(args.stats_hz))
    last_stats_ts = 0.0

    while (time.time() - measure_t0) < max(0.5, float(args.duration)):
        frame, ts = src.get_latest_frame()
        now = time.time()
        if frame is not None and ts is not None:
            tsf = float(ts)
            if tsf > float(last_unique_ts):
                no_frame_gap = now - last_frame_wall_ts
                max_no_frame_gap_s = max(max_no_frame_gap_s, no_frame_gap)
                last_frame_wall_ts = now
                last_unique_ts = tsf
                unique_frames += 1

                w, h = int(frame.shape[1]), int(frame.shape[0])
                last_size = (w, h)
                if isinstance(expected_w, int) and isinstance(expected_h, int) and (w, h) != (expected_w, expected_h):
                    size_mismatch_count += 1

            if args.viz:
                view = frame.copy()
                cv2.putText(
                    view,
                    f"topic={active_topic}",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    view,
                    f"frames={unique_frames} size={last_size[0]}x{last_size[1]}",
                    (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(args.window_name, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

        if (now - last_stats_ts) >= stats_period_s:
            elapsed = max(1e-6, now - measure_t0)
            fps_live = unique_frames / elapsed
            print(
                f"[PX4_CAM_SMOKE] frames={unique_frames} avg_fps={fps_live:.2f} "
                f"size={last_size[0]}x{last_size[1]} max_gap={max_no_frame_gap_s:.3f}s"
            )
            last_stats_ts = now

        time.sleep(0.002)

    if args.viz:
        cv2.destroyAllWindows()

    elapsed = max(1e-6, time.time() - measure_t0)
    avg_fps = unique_frames / elapsed
    limit_min_fps = contract.get("min_fps")
    limit_max_gap = contract.get("max_no_frame_gap_s", 1.0)
    if limit_max_gap is None:
        limit_max_gap = 1.0
    limit_max_gap = float(_clamp(float(limit_max_gap), 0.05, 60.0))

    checks = {
        "has_frames": bool(unique_frames > 0),
        "size_ok": bool(size_mismatch_count == 0),
        "fps_ok": True if limit_min_fps is None else bool(avg_fps >= float(limit_min_fps)),
        "gap_ok": bool(max_no_frame_gap_s <= limit_max_gap),
    }
    passed = all(bool(v) for v in checks.values())

    report = {
        "passed": passed,
        "platform_type": platform_type,
        "topic": active_topic,
        "configured_topic": topic,
        "used_discovery_fallback": used_discovery_fallback,
        "discovered_candidates": discovered_candidates,
        "duration_s": elapsed,
        "frames_unique": unique_frames,
        "avg_fps": avg_fps,
        "max_no_frame_gap_s": max_no_frame_gap_s,
        "last_frame_size": {"w": last_size[0], "h": last_size[1]},
        "size_mismatch_count": size_mismatch_count,
        "contract": contract,
        "checks": checks,
        "timestamp": time.time(),
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"[PX4_CAM_SMOKE] report={out_path}")

    print(
        f"[PX4_CAM_SMOKE] {'PASS' if passed else 'FAIL'} "
        f"| fps={avg_fps:.2f} size={last_size[0]}x{last_size[1]} "
        f"| max_gap={max_no_frame_gap_s:.3f}s"
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

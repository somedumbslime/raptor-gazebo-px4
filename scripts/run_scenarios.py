#!/usr/bin/env /usr/bin/python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAPTOR-AI scenario suite")
    parser.add_argument(
        "--config",
        type=str,
        default="raptor_ai/config/default_config.yaml",
        help="Path to runtime YAML config",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="all",
        help="Comma separated scenario names, or 'all'",
    )
    parser.add_argument("--duration", type=float, default=None, help="Override duration for every scenario")
    parser.add_argument("--viz", action="store_true", help="Enable runtime visualization")
    parser.add_argument("--record", action="store_true", help="Record runtime visualization to video file")
    parser.add_argument("--record-fps", type=float, default=None, help="Video FPS for --record (default: control_hz)")
    parser.add_argument(
        "--target-mode",
        type=str,
        default=None,
        help="Target mode from config scenarios.target_modes (e.g. synthetic, actor)",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs/scenarios",
        help="Directory where scenario run artifacts are stored",
    )
    parser.add_argument(
        "--platform-type",
        type=str,
        default=None,
        choices=("gimbal", "iris", "px4"),
        help="Override platform.type in runtime config",
    )
    parser.add_argument(
        "--detector-type",
        type=str,
        default=None,
        choices=("red", "yolo_onnx"),
        help="Override detector.type in runtime config",
    )
    parser.add_argument(
        "--selector-backend",
        type=str,
        default=None,
        choices=("stub", "external"),
        help="Override selector.backend in runtime config",
    )
    parser.add_argument(
        "--guidance-backend",
        type=str,
        default=None,
        choices=("legacy_internal", "target_guidance"),
        help="Override guidance.backend in runtime config",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=None,
        help="Override detector.yolo_onnx.model_path",
    )
    parser.add_argument(
        "--yolo-target-classes",
        type=str,
        default=None,
        help="Comma-separated classes for detector.yolo_onnx.target_classes",
    )
    parser.add_argument(
        "--follow-profile",
        type=str,
        default=None,
        choices=("safe", "balanced", "aggressive"),
        help="Apply follow tuning preset (safe|balanced|aggressive)",
    )
    parser.add_argument(
        "--follow-mode",
        type=str,
        default=None,
        choices=("off", "xy"),
        help="Override state_machine.follow_mode",
    )
    parser.add_argument(
        "--state-lost-frame-threshold",
        type=int,
        default=None,
        help="Override state_machine.lost_frame_threshold",
    )
    parser.add_argument(
        "--state-reacquire-threshold",
        type=int,
        default=None,
        help="Override state_machine.reacquire_threshold",
    )
    parser.add_argument(
        "--follow-enabled",
        action="store_true",
        default=None,
        help="Set follow.enabled=true",
    )
    parser.add_argument(
        "--no-follow-enabled",
        dest="follow_enabled",
        action="store_false",
    )
    parser.add_argument(
        "--follow-xy-strategy",
        type=str,
        default=None,
        choices=("zone_track",),
        help="Override follow.xy_strategy",
    )
    parser.add_argument(
        "--px4-cv-only",
        action="store_true",
        default=None,
        help="Set platform.px4.cv_only=true (no movement commands, CV-only mode)",
    )
    parser.add_argument(
        "--no-px4-cv-only",
        dest="px4_cv_only",
        action="store_false",
    )
    parser.add_argument(
        "--px4-auto-arm",
        action="store_true",
        default=None,
        help="Set platform.px4.auto_arm=true",
    )
    parser.add_argument(
        "--no-px4-auto-arm",
        dest="px4_auto_arm",
        action="store_false",
    )
    parser.add_argument(
        "--px4-auto-takeoff",
        action="store_true",
        default=None,
        help="Set platform.px4.auto_takeoff=true",
    )
    parser.add_argument(
        "--no-px4-auto-takeoff",
        dest="px4_auto_takeoff",
        action="store_false",
    )
    parser.add_argument(
        "--px4-auto-arm-require-armable",
        action="store_true",
        default=None,
        help="Set platform.px4.auto_arm_require_armable=true",
    )
    parser.add_argument(
        "--no-px4-auto-arm-require-armable",
        dest="px4_auto_arm_require_armable",
        action="store_false",
    )
    parser.add_argument(
        "--px4-auto-arm-require-local-position",
        action="store_true",
        default=None,
        help="Set platform.px4.auto_arm_require_local_position=true",
    )
    parser.add_argument(
        "--no-px4-auto-arm-require-local-position",
        dest="px4_auto_arm_require_local_position",
        action="store_false",
    )
    parser.add_argument(
        "--px4-takeoff-altitude",
        type=float,
        default=None,
        help="Override platform.px4.takeoff_altitude_m",
    )
    parser.add_argument(
        "--px4-takeoff-confirm-alt",
        type=float,
        default=None,
        help="Override platform.px4.takeoff_confirm_alt_m",
    )
    parser.add_argument(
        "--px4-offboard-min-alt",
        type=float,
        default=None,
        help="Override platform.px4.offboard_min_relative_alt_m",
    )
    parser.add_argument(
        "--px4-offboard-delay-after-liftoff",
        type=float,
        default=None,
        help="Override platform.px4.offboard_start_delay_after_liftoff_s",
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


def main() -> None:
    args = parse_args()

    from raptor_ai.config.loader import load_config, resolve_camera_topic, resolve_platform_type
    from raptor_ai.config.overrides import apply_runtime_overrides
    from raptor_ai.scenarios.configuration import resolve_scenarios_config
    try:
        with _suppress_stderr(bool(args.quiet_gz_import)):
            from raptor_ai.runtime.runtime_v2 import RuntimeV2
            from raptor_ai.scenarios.target_motion import TargetMotionThread
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("gz"):
            print("Missing Gazebo Python bindings in this interpreter. Try: /usr/bin/python3 scripts/run_scenarios.py ...")
            raise SystemExit(1) from exc
        raise
    except ImportError as exc:
        msg = str(exc)
        if "numpy.core.multiarray failed to import" in msg or "compiled using NumPy 1.x" in msg:
            print(
                "OpenCV/NumPy ABI mismatch in this interpreter. "
                "Try: PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/run_scenarios.py ..."
            )
            raise SystemExit(1) from exc
        raise

    cfg = load_config(args.config)
    cfg = apply_runtime_overrides(
        cfg,
        platform_type=args.platform_type,
        detector_type=args.detector_type,
        selector_backend=args.selector_backend,
        guidance_backend=args.guidance_backend,
        yolo_model_path=args.yolo_model,
        yolo_target_classes_csv=args.yolo_target_classes,
        follow_profile=args.follow_profile,
        follow_mode=args.follow_mode,
        state_lost_frame_threshold=args.state_lost_frame_threshold,
        state_reacquire_threshold=args.state_reacquire_threshold,
        follow_enabled=args.follow_enabled,
        follow_xy_strategy=args.follow_xy_strategy,
        px4_cv_only=args.px4_cv_only,
        px4_auto_arm=args.px4_auto_arm,
        px4_auto_takeoff=args.px4_auto_takeoff,
        px4_auto_arm_require_armable=args.px4_auto_arm_require_armable,
        px4_auto_arm_require_local_position=args.px4_auto_arm_require_local_position,
        px4_takeoff_altitude_m=args.px4_takeoff_altitude,
        px4_takeoff_confirm_alt_m=args.px4_takeoff_confirm_alt,
        px4_offboard_min_relative_alt_m=args.px4_offboard_min_alt,
        px4_offboard_start_delay_after_liftoff_s=args.px4_offboard_delay_after_liftoff,
    )

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    effective_cfg_path = runs_root / f"effective_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    effective_cfg_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    platform_type = resolve_platform_type(cfg)
    camera_topic = resolve_camera_topic(cfg, platform_type=platform_type)

    scenarios_cfg = cfg.get("scenarios", {})
    target_mode, active_scenarios_cfg, profiles = resolve_scenarios_config(
        scenarios_cfg=scenarios_cfg,
        target_mode_override=args.target_mode,
        platform_type=platform_type,
    )

    if args.scenarios.strip().lower() == "all":
        scenario_names = list(profiles.keys())
    else:
        scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    print(f"[SCENARIOS] target_mode={target_mode}")
    print(f"[SCENARIOS] platform_type={platform_type}")
    print(f"[SCENARIOS] camera_topic={camera_topic}")
    print(f"[SCENARIOS] detector_type={cfg.get('detector', {}).get('type', 'red')}")
    print(f"[SCENARIOS] selector_backend={cfg.get('selector', {}).get('backend', 'stub')}")
    print(f"[SCENARIOS] guidance_backend={cfg.get('guidance', {}).get('backend', 'legacy_internal')}")
    print(f"[SCENARIOS] follow_mode={cfg.get('state_machine', {}).get('follow_mode', 'off')}")
    print(f"[SCENARIOS] follow_enabled={bool(cfg.get('follow', {}).get('enabled', False))}")
    print(f"[SCENARIOS] follow_xy_strategy={cfg.get('follow', {}).get('xy_strategy', 'zone_track')}")
    print(f"[SCENARIOS] effective_config={effective_cfg_path}")
    print(f"[SCENARIOS] selected={scenario_names}")

    suite_results: dict[str, dict] = {}

    for scenario_name in scenario_names:
        if scenario_name not in profiles:
            print(f"[SCENARIOS] skip unknown scenario: {scenario_name}")
            continue

        profile_cfg = dict(profiles[scenario_name])
        duration_s = float(args.duration) if args.duration is not None else float(profile_cfg.get("duration_s", 20.0))

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = runs_root / f"{run_stamp}_{target_mode}_{scenario_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[SCENARIOS] running {scenario_name} for {duration_s:.1f}s -> {out_dir}")

        motion = TargetMotionThread(
            scenarios_cfg=active_scenarios_cfg,
            profile_name=scenario_name,
            profile_cfg=profile_cfg,
            motion_trace_path=str(out_dir / "motion_trace.jsonl"),
        )
        try:
            runtime = RuntimeV2(config_path=str(effective_cfg_path), output_dir=str(out_dir))
        except RuntimeError as exc:
            msg = str(exc)
            if "No module named 'onnxruntime'" in msg and "yolo_onnx" in msg:
                print(
                    "ONNX Runtime is missing in this Python interpreter. "
                    "Use a system-site-packages venv for Gazebo + onnxruntime, for example:\n"
                    "  /usr/bin/python3 -m venv --system-site-packages .venv_gz\n"
                    "  .venv_gz/bin/python -m pip install -U pip onnxruntime\n"
                    "  GZ_PARTITION=raptor_px4 PYTHONNOUSERSITE=1 .venv_gz/bin/python scripts/run_scenarios.py ...\n"
                )
                raise SystemExit(2) from exc
            raise

        try:
            motion.start()
            if not motion.wait_ready(timeout=6.0):
                print("[SCENARIOS] warn: motion thread not ready within 6s, continuing")
            record_path = str(out_dir / "runtime_viz.mp4") if args.record else None
            summary = runtime.run(
                duration_s=duration_s,
                viz=args.viz,
                record=args.record,
                record_path=record_path,
                record_fps=args.record_fps,
            )
            suite_results[scenario_name] = summary
        finally:
            motion.stop()
            motion.join(timeout=2.0)

        with (out_dir / "scenario_meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "scenario": scenario_name,
                    "duration_s": duration_s,
                    "target_mode": target_mode,
                    "world_name": active_scenarios_cfg.get("world_name"),
                    "model_name": active_scenarios_cfg.get("model_name"),
                    "platform_type": platform_type,
                    "camera_topic": camera_topic,
                    "detector_type": cfg.get("detector", {}).get("type", "red"),
                    "selector_backend": cfg.get("selector", {}).get("backend", "stub"),
                    "guidance_backend": cfg.get("guidance", {}).get("backend", "legacy_internal"),
                    "follow_mode": cfg.get("state_machine", {}).get("follow_mode", "off"),
                    "follow_enabled": bool(cfg.get("follow", {}).get("enabled", False)),
                    "follow_xy_strategy": cfg.get("follow", {}).get("xy_strategy", "zone_track"),
                    "effective_config_path": str(effective_cfg_path),
                    "viz": bool(args.viz),
                    "record": bool(args.record),
                    "record_path": (str(out_dir / "runtime_viz.mp4") if args.record else None),
                    "profile": profile_cfg,
                },
                f,
                indent=2,
                ensure_ascii=True,
            )

    suite_path = runs_root / f"suite_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with suite_path.open("w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2, ensure_ascii=True)

    print(f"[SCENARIOS] suite summary: {suite_path}")


if __name__ == "__main__":
    main()

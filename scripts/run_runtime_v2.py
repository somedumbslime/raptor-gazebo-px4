#!/usr/bin/env /usr/bin/python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAPTOR-AI RuntimeV2")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("raptor_ai/config/default_config.yaml")),
        help="Path to runtime YAML config",
    )
    parser.add_argument("--duration", type=float, default=None, help="Optional fixed run duration in seconds")
    parser.add_argument("--viz", action="store_true", help="Enable OpenCV visualization")
    parser.add_argument("--record", action="store_true", help="Record runtime visualization to video file")
    parser.add_argument("--record-path", type=str, default=None, help="Output video path for --record")
    parser.add_argument("--record-fps", type=float, default=None, help="Video FPS for --record (default: control_hz)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
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
    args = parser.parse_args()

    try:
        from raptor_ai.runtime.runtime_v2 import RuntimeV2
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("gz"):
            print("Missing Gazebo Python bindings in this interpreter. Try: /usr/bin/python3 scripts/run_runtime_v2.py ...")
            raise SystemExit(1) from exc
        raise
    except ImportError as exc:
        msg = str(exc)
        if "numpy.core.multiarray failed to import" in msg or "compiled using NumPy 1.x" in msg:
            print(
                "OpenCV/NumPy ABI mismatch in this interpreter. "
                "Try: PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/run_runtime_v2.py ..."
            )
            raise SystemExit(1) from exc
        raise

    from raptor_ai.config.loader import load_config
    from raptor_ai.config.overrides import apply_runtime_overrides

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

    effective_cfg_path = Path(args.output_dir or "runs/latest") / f"effective_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    effective_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    effective_cfg_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(f"[RUNTIME] effective_config={effective_cfg_path}")

    runtime = RuntimeV2(config_path=str(effective_cfg_path), output_dir=args.output_dir)
    runtime.run(
        duration_s=args.duration,
        viz=args.viz,
        record=args.record,
        record_path=args.record_path,
        record_fps=args.record_fps,
    )


if __name__ == "__main__":
    main()

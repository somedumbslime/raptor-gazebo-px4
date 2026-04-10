#!/usr/bin/env /usr/bin/python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raptor_ai.config.loader import load_config
from raptor_ai.scenarios.configuration import resolve_scenarios_config
from raptor_ai.scenarios.gazebo_world import (
    compute_pose_ahead,
    create_model_from_uri,
    set_model_pose,
    wait_for_model_pose,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spawn/reposition actor in PX4 world ahead of x500")
    parser.add_argument("--config", type=str, default="raptor_ai/config/default_config.yaml")
    parser.add_argument("--world", type=str, default=None, help="Gazebo world name (default from actor+px4 config)")
    parser.add_argument("--actor-name", type=str, default=None, help="Actor model name")
    parser.add_argument("--actor-uri", type=str, default=None, help="Model URI for /world/<name>/create")
    parser.add_argument("--reference-model", type=str, default=None, help="Reference model for relative placement")
    parser.add_argument("--ahead-m", type=float, default=None, help="Meters ahead of reference model")
    parser.add_argument("--right-m", type=float, default=None, help="Meters right of reference model")
    parser.add_argument("--z", type=float, default=None, help="Absolute Z for actor model")
    parser.add_argument(
        "--yaw-mode",
        type=str,
        default=None,
        choices=("face_reference", "align_reference", "fixed"),
        help="Actor yaw policy relative to reference",
    )
    parser.add_argument("--fixed-yaw", type=float, default=None, help="Used when --yaw-mode fixed")
    parser.add_argument("--reference-timeout-s", type=float, default=None, help="Pose wait timeout")
    parser.add_argument("--set-pose-timeout-ms", type=int, default=None, help="set_pose service timeout")
    parser.add_argument("--out-json", type=str, default=None, help="Optional report path")
    parser.add_argument("--dry-run", action="store_true", help="Only compute pose, do not call create/set_pose")
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _, actor_cfg, _ = resolve_scenarios_config(
        scenarios_cfg=cfg.get("scenarios", {}),
        target_mode_override="actor",
        platform_type="px4",
    )

    world_name = str(args.world or actor_cfg.get("world_name", "default"))
    actor_name = str(args.actor_name or actor_cfg.get("model_name", "target_actor"))
    actor_uri = str(args.actor_uri or actor_cfg.get("spawn_uri", "")).strip()
    reference_model = str(args.reference_model or actor_cfg.get("reference_model_name", "x500_0"))
    ahead_m = float(args.ahead_m if args.ahead_m is not None else actor_cfg.get("reference_ahead_m", 8.0))
    right_m = float(args.right_m if args.right_m is not None else actor_cfg.get("reference_right_m", 0.0))
    z = float(args.z if args.z is not None else actor_cfg.get("reference_z", actor_cfg.get("z", 1.0)))
    yaw_mode = str(args.yaw_mode or actor_cfg.get("reference_yaw_mode", "face_reference"))
    fixed_yaw = float(args.fixed_yaw if args.fixed_yaw is not None else actor_cfg.get("reference_fixed_yaw", 0.0))
    ref_timeout_s = float(
        args.reference_timeout_s if args.reference_timeout_s is not None else actor_cfg.get("reference_timeout_s", 3.0)
    )
    timeout_ms = int(args.set_pose_timeout_ms if args.set_pose_timeout_ms is not None else actor_cfg.get("set_pose_timeout_ms", 1000))

    print(
        f"[PX4_ACTOR_SETUP] world={world_name} actor={actor_name} reference={reference_model} "
        f"ahead={ahead_m:.2f}m right={right_m:.2f}m z={z:.2f} yaw_mode={yaw_mode}"
    )
    print(f"[PX4_ACTOR_SETUP] GZ_PARTITION={os.environ.get('GZ_PARTITION', '<unset>')}")

    with _suppress_stderr(bool(args.quiet_gz_import)):
        from gz.transport import Node

    node = Node()
    report: dict[str, object] = {
        "world_name": world_name,
        "actor_name": actor_name,
        "actor_uri": actor_uri,
        "reference_model": reference_model,
        "ahead_m": ahead_m,
        "right_m": right_m,
        "z": z,
        "yaw_mode": yaw_mode,
        "fixed_yaw": fixed_yaw,
        "gz_partition": os.environ.get("GZ_PARTITION"),
        "timestamp": time.time(),
    }

    ref = wait_for_model_pose(node=node, world_name=world_name, model_name=reference_model, timeout_s=ref_timeout_s)
    if ref is None:
        print(f"[PX4_ACTOR_SETUP] FAIL: reference model not found: {reference_model}")
        report["ok"] = False
        report["error"] = "reference_model_not_found"
        if args.out_json:
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        raise SystemExit(2)

    tx, ty, tz, tyaw = compute_pose_ahead(
        reference_pose=ref,
        ahead_m=ahead_m,
        right_m=right_m,
        z=z,
        yaw_mode=yaw_mode,
        fixed_yaw=fixed_yaw,
    )
    report["reference_pose"] = {"name": ref.name, "x": ref.x, "y": ref.y, "z": ref.z, "yaw": ref.yaw}
    report["target_pose"] = {"x": tx, "y": ty, "z": tz, "yaw": tyaw}
    print(f"[PX4_ACTOR_SETUP] target pose -> x={tx:+.2f} y={ty:+.2f} z={tz:.2f} yaw={tyaw:+.2f}")

    if args.dry_run:
        report["ok"] = True
        report["mode"] = "dry_run"
    else:
        existing = wait_for_model_pose(
            node=node,
            world_name=world_name,
            model_name=actor_name,
            timeout_s=0.5,
        )
        if existing is not None:
            set_ok, set_rep = set_model_pose(
                node=node,
                world_name=world_name,
                model_name=actor_name,
                x=tx,
                y=ty,
                z=tz,
                yaw=tyaw,
                timeout_ms=timeout_ms,
            )
            if set_ok and set_rep:
                print(f"[PX4_ACTOR_SETUP] actor exists -> repositioned: {actor_name}")
                report["ok"] = True
                report["mode"] = "reposition"
            else:
                # In some Gazebo worlds actor entities cannot be moved via /set_pose.
                # Presence is enough for our runtime, so treat this as success.
                print(f"[PX4_ACTOR_SETUP] actor exists but set_pose failed -> keeping current pose: {actor_name}")
                report["ok"] = True
                report["mode"] = "exists_unmoved"
                report["set_pose_ok"] = set_ok
                report["set_pose_rep"] = set_rep
        else:
            if not actor_uri:
                print("[PX4_ACTOR_SETUP] FAIL: actor is missing and actor_uri is empty")
                report["ok"] = False
                report["error"] = "missing_actor_uri"
                if args.out_json:
                    out_path = Path(args.out_json)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
                raise SystemExit(2)

            create_ok, create_rep = create_model_from_uri(
                node=node,
                world_name=world_name,
                model_name=actor_name,
                model_uri=actor_uri,
                x=tx,
                y=ty,
                z=tz,
                yaw=tyaw,
                timeout_ms=max(timeout_ms, 2000),
            )
            if not (create_ok and create_rep):
                existing_after_fail = wait_for_model_pose(
                    node=node,
                    world_name=world_name,
                    model_name=actor_name,
                    timeout_s=0.5,
                )
                if existing_after_fail is not None:
                    print(f"[PX4_ACTOR_SETUP] actor detected after create fail -> keeping existing: {actor_name}")
                    report["ok"] = True
                    report["mode"] = "exists_after_create_fail"
                    report["create_ok"] = create_ok
                    report["create_rep"] = create_rep
                    if args.out_json:
                        out_path = Path(args.out_json)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
                    raise SystemExit(0)
                print(f"[PX4_ACTOR_SETUP] FAIL: create failed | ok={create_ok} rep={create_rep}")
                report["ok"] = False
                report["error"] = "create_failed"
                report["create_ok"] = create_ok
                report["create_rep"] = create_rep
                if args.out_json:
                    out_path = Path(args.out_json)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
                raise SystemExit(2)

            time.sleep(0.15)
            set_model_pose(
                node=node,
                world_name=world_name,
                model_name=actor_name,
                x=tx,
                y=ty,
                z=tz,
                yaw=tyaw,
                timeout_ms=timeout_ms,
            )
            print(f"[PX4_ACTOR_SETUP] actor spawned: {actor_name}")
            report["ok"] = True
            report["mode"] = "spawn"

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"[PX4_ACTOR_SETUP] report={out_path}")


if __name__ == "__main__":
    main()

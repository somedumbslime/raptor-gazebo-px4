#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_cfg(config_path: str) -> dict[str, Any]:
    from raptor_ai.config.loader import load_config

    cfg = load_config(config_path)
    platform_cfg = cfg.get("platform", {})
    px4_top_cfg = cfg.get("px4", {})
    if not isinstance(platform_cfg, dict):
        platform_cfg = {}
    if not isinstance(px4_top_cfg, dict):
        px4_top_cfg = {}

    px4_platform_cfg = platform_cfg.get("px4", {})
    if not isinstance(px4_platform_cfg, dict):
        px4_platform_cfg = {}

    merged = dict(px4_top_cfg)
    merged.update(px4_platform_cfg)
    return merged


def _wait_for(
    pred,
    *,
    timeout_s: float,
    step_s: float = 0.1,
    status_fn=None,
    status_interval_s: float = 1.0,
) -> bool:
    t0 = time.time()
    last_status = 0.0
    while (time.time() - t0) < timeout_s:
        if pred():
            return True
        if status_fn is not None and (time.time() - last_status) >= status_interval_s:
            status_fn()
            last_status = time.time()
        time.sleep(step_s)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PX4 bridge smoke: connect -> arm -> takeoff -> velocity -> land")
    parser.add_argument("--config", default="raptor_ai/config/default_config.yaml", help="Path to runtime config")
    parser.add_argument("--connect-timeout", type=float, default=20.0, help="Wait for connection timeout")
    parser.add_argument("--health-timeout", type=float, default=25.0, help="Wait for armable/local position timeout")
    parser.add_argument("--require-local-position", action="store_true", default=True, help="Require local position health before arm")
    parser.add_argument("--no-require-local-position", dest="require_local_position", action="store_false")
    parser.add_argument("--arm-retries", type=int, default=3, help="Retries for arm command")
    parser.add_argument("--takeoff-retries", type=int, default=3, help="Retries for takeoff command")
    parser.add_argument("--takeoff-timeout", type=float, default=20.0, help="Wait for in_air timeout")
    parser.add_argument(
        "--min-liftoff-alt",
        type=float,
        default=0.8,
        help="Minimum relative altitude gain (meters) to confirm real liftoff",
    )
    parser.add_argument("--land-timeout", type=float, default=30.0, help="Wait for landing timeout")
    parser.add_argument("--hold-after-takeoff", type=float, default=2.0, help="Hover hold duration after takeoff")
    parser.add_argument("--forward-speed", type=float, default=0.25, help="Body forward speed for velocity block")
    parser.add_argument("--right-speed", type=float, default=0.0, help="Body right speed for velocity block")
    parser.add_argument("--down-speed", type=float, default=0.0, help="Body down speed for velocity block")
    parser.add_argument("--yaw-rate-deg-s", type=float, default=0.0, help="Body yaw rate for velocity block")
    parser.add_argument("--velocity-duration", type=float, default=6.0, help="Velocity block duration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    px4_cfg = _load_cfg(args.config)

    from raptor_ai.platform.px4_bridge import Px4Bridge

    bridge = Px4Bridge(px4_cfg)
    offboard_cfg_enabled = bool(getattr(bridge, "offboard_enabled", False))
    try:
        # Important for deterministic takeoff smoke:
        # do not let automatic OFFBOARD streaming preempt PX4 TAKEOFF mode
        # before we confirm real altitude gain.
        if offboard_cfg_enabled:
            bridge.offboard_enabled = False
            print("[PX4_BRIDGE_SMOKE] offboard disabled during takeoff phase")

        print("[PX4_BRIDGE_SMOKE] waiting connection...")
        ok = _wait_for(
            lambda: bool(bridge.metadata().get("connected", False)),
            timeout_s=args.connect_timeout,
            status_fn=lambda: print(
                f"[PX4_BRIDGE_SMOKE] status connected={bridge.metadata().get('connected')} "
                f"armable={bridge.metadata().get('is_armable')} "
                f"local_pos={bridge.metadata().get('is_local_position_ok')} "
                f"err={bridge.metadata().get('last_error')}"
            ),
        )
        if not ok:
            raise RuntimeError(f"PX4 connection timeout ({args.connect_timeout:.1f}s)")
        print("[PX4_BRIDGE_SMOKE] connected")

        print("[PX4_BRIDGE_SMOKE] waiting health for arm...")
        ok = _wait_for(
            lambda: bool(bridge.metadata().get("is_armable", False))
            and (bool(bridge.metadata().get("is_local_position_ok", False)) if args.require_local_position else True),
            timeout_s=args.health_timeout,
            status_fn=lambda: print(
                f"[PX4_BRIDGE_SMOKE] health armable={bridge.metadata().get('is_armable')} "
                f"local_pos={bridge.metadata().get('is_local_position_ok')} "
                f"connected={bridge.metadata().get('connected')} "
                f"err={bridge.metadata().get('last_error')}"
            ),
        )
        if not ok:
            raise RuntimeError(f"Health timeout before arm ({args.health_timeout:.1f}s)")

        print("[PX4_BRIDGE_SMOKE] arm")
        arm_ok = False
        last_arm_exc: Exception | None = None
        for i in range(max(1, args.arm_retries)):
            try:
                bridge.arm()
                arm_ok = True
                break
            except Exception as exc:
                last_arm_exc = exc
                print(f"[PX4_BRIDGE_SMOKE] arm retry {i + 1}/{args.arm_retries} failed: {exc}")
                time.sleep(0.8)
        if not arm_ok:
            raise RuntimeError(f"arm failed after retries: {last_arm_exc}")

        print("[PX4_BRIDGE_SMOKE] takeoff")
        alt0 = float(bridge.metadata().get("relative_altitude_m") or 0.0)
        takeoff_ok = False
        last_takeoff_exc: Exception | None = None
        for i in range(max(1, args.takeoff_retries)):
            try:
                bridge.takeoff()
                takeoff_ok = True
                break
            except Exception as exc:
                last_takeoff_exc = exc
                print(f"[PX4_BRIDGE_SMOKE] takeoff retry {i + 1}/{args.takeoff_retries} failed: {exc}")
                time.sleep(1.0)
        if not takeoff_ok:
            raise RuntimeError(f"takeoff command failed after retries: {last_takeoff_exc}")

        ok = _wait_for(
            lambda: bool(bridge.metadata().get("in_air", False))
            and (
                float(bridge.metadata().get("relative_altitude_m") or 0.0) - alt0
                >= max(0.05, float(args.min_liftoff_alt))
            ),
            timeout_s=args.takeoff_timeout,
            status_fn=lambda: print(
                f"[PX4_BRIDGE_SMOKE] waiting in_air... "
                f"armed={bridge.metadata().get('armed')} "
                f"in_air={bridge.metadata().get('in_air')} "
                f"mode={bridge.metadata().get('flight_mode')} "
                f"alt={float(bridge.metadata().get('relative_altitude_m') or 0.0):.2f} "
                f"alt_gain={float(bridge.metadata().get('relative_altitude_m') or 0.0) - alt0:+.2f} "
                f"offboard={bridge.metadata().get('offboard_started')} "
                f"err={bridge.metadata().get('last_error')}"
            ),
        )
        if not ok:
            m = bridge.metadata()
            alt = float(m.get("relative_altitude_m") or 0.0)
            raise RuntimeError(
                "Takeoff timeout "
                f"({args.takeoff_timeout:.1f}s): in_air={m.get('in_air')} "
                f"mode={m.get('flight_mode')} alt={alt:.2f} "
                f"alt_gain={alt - alt0:+.2f}m (need >= {args.min_liftoff_alt:.2f}m)"
            )
        print(
            "[PX4_BRIDGE_SMOKE] in air "
            f"(alt={float(bridge.metadata().get('relative_altitude_m') or 0.0):.2f}m, "
            f"gain={float(bridge.metadata().get('relative_altitude_m') or 0.0) - alt0:+.2f}m)"
        )

        if args.hold_after_takeoff > 0.0:
            print(f"[PX4_BRIDGE_SMOKE] hold {args.hold_after_takeoff:.1f}s")
            time.sleep(args.hold_after_takeoff)

        if args.velocity_duration > 0.0:
            if offboard_cfg_enabled:
                bridge.offboard_enabled = True
                print("[PX4_BRIDGE_SMOKE] offboard enabled for velocity block")
                _wait_for(
                    lambda: bool(bridge.metadata().get("offboard_mode_active", False)),
                    timeout_s=4.0,
                    status_fn=lambda: print(
                        f"[PX4_BRIDGE_SMOKE] waiting offboard mode... "
                        f"mode={bridge.metadata().get('flight_mode')} "
                        f"offboard={bridge.metadata().get('offboard_started')} "
                        f"err={bridge.metadata().get('last_error')}"
                    ),
                )

            print(
                "[PX4_BRIDGE_SMOKE] velocity block "
                f"{args.velocity_duration:.1f}s "
                f"(fwd={args.forward_speed:.2f}, right={args.right_speed:.2f}, "
                f"down={args.down_speed:.2f}, yaw_rate={args.yaw_rate_deg_s:.2f})"
            )
            t0 = time.time()
            while (time.time() - t0) < args.velocity_duration:
                bridge.set_velocity_xyz(
                    forward_m_s=args.forward_speed,
                    right_m_s=args.right_speed,
                    down_m_s=args.down_speed,
                    yaw_rate_deg_s=args.yaw_rate_deg_s,
                )
                time.sleep(1.0 / 20.0)

        bridge.stop_motion()
        print("[PX4_BRIDGE_SMOKE] land")
        bridge.land()
        ok = _wait_for(
            lambda: not bool(bridge.metadata().get("in_air", True)),
            timeout_s=args.land_timeout,
            status_fn=lambda: print(
                f"[PX4_BRIDGE_SMOKE] waiting landed... "
                f"armed={bridge.metadata().get('armed')} "
                f"in_air={bridge.metadata().get('in_air')} "
                f"err={bridge.metadata().get('last_error')}"
            ),
        )
        if not ok:
            raise RuntimeError(f"Landing timeout ({args.land_timeout:.1f}s)")
        print("[PX4_BRIDGE_SMOKE] landed")
    finally:
        bridge.offboard_enabled = offboard_cfg_enabled
        bridge.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _load_px4_cfg(config_path: str) -> dict[str, Any]:
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


def _safe_call(label: str, fn) -> None:
    try:
        fn()
        print(f"[PX4_MANUAL] {label}: OK")
    except Exception as exc:
        print(f"[PX4_MANUAL] {label}: FAIL ({exc})")


def _print_help() -> None:
    print(
        """
[PX4_MANUAL] controls:
  w/s : forward +/-
  a/d : right -/+
  r/f : up/down
  q/e : yaw left/right
  space: stop motion
  z/x : speed scale -/+
  1   : arm
  2   : takeoff
  3   : land
  4   : disarm
  m   : print bridge metadata
  h   : show help
  c   : clear current command (same as space)
  ESC or Ctrl+C: exit
"""
    )


def _status_text(
    *,
    cmd_f: float,
    cmd_r: float,
    cmd_d: float,
    cmd_y: float,
    speed_scale: float,
    meta: dict[str, Any],
) -> str:
    return (
        "[PX4_MANUAL] "
        f"conn={meta.get('connected')} arm={meta.get('armed')} in_air={meta.get('in_air')} "
        f"offboard={meta.get('offboard_started')} armable={meta.get('is_armable')} "
        f"lpos={meta.get('is_local_position_ok')} "
        f"cmd[f={cmd_f:+.2f}, r={cmd_r:+.2f}, d={cmd_d:+.2f}, yaw={cmd_y:+.1f}] "
        f"scale={speed_scale:.2f} "
        f"err={meta.get('last_error')}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive manual PX4 control via Px4Bridge")
    parser.add_argument("--config", default="raptor_ai/config/default_config.yaml", help="Path to runtime config")
    parser.add_argument("--hz", type=float, default=20.0, help="Command publish frequency")
    parser.add_argument("--step-forward", type=float, default=0.15, help="Forward velocity step (m/s)")
    parser.add_argument("--step-right", type=float, default=0.15, help="Right velocity step (m/s)")
    parser.add_argument("--step-updown", type=float, default=0.10, help="Vertical velocity step (m/s)")
    parser.add_argument("--step-yaw", type=float, default=8.0, help="Yaw-rate step (deg/s)")
    parser.add_argument("--status-hz", type=float, default=2.0, help="Status print frequency")
    parser.add_argument("--speed-scale-min", type=float, default=0.2, help="Minimum speed scale")
    parser.add_argument("--speed-scale-max", type=float, default=2.0, help="Maximum speed scale")
    parser.add_argument("--speed-scale-step", type=float, default=0.1, help="Speed scale step")
    parser.add_argument(
        "--auto-land-on-exit",
        action="store_true",
        default=True,
        help="Call land() when exiting if vehicle is in air (default: true)",
    )
    parser.add_argument("--no-auto-land-on-exit", dest="auto_land_on_exit", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    px4_cfg = _load_px4_cfg(args.config)

    from raptor_ai.platform.px4_bridge import Px4Bridge

    bridge = Px4Bridge(px4_cfg)
    print(f"[PX4_MANUAL] connecting to {px4_cfg.get('system_address', 'udpout://127.0.0.1:14540')}")
    _print_help()

    cmd_forward = 0.0
    cmd_right = 0.0
    cmd_down = 0.0
    cmd_yaw = 0.0
    speed_scale = 1.0

    max_forward = float(px4_cfg.get("max_forward_m_s", 1.0))
    max_right = float(px4_cfg.get("max_right_m_s", 0.8))
    max_down = float(px4_cfg.get("max_down_m_s", 0.5))
    max_yaw = float(px4_cfg.get("max_yaw_rate_deg_s", 30.0))

    period_s = 1.0 / max(1.0, float(args.hz))
    status_period_s = 1.0 / max(0.2, float(args.status_hz))
    last_status_ts = 0.0

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], period_s)
            if ready:
                ch = sys.stdin.read(1)
                if ch == "\x03":  # Ctrl+C
                    break
                if ch == "\x1b":  # ESC or arrows
                    seq_ready, _, _ = select.select([sys.stdin], [], [], 0.01)
                    if seq_ready:
                        _ = sys.stdin.read(1)
                        seq_ready2, _, _ = select.select([sys.stdin], [], [], 0.01)
                        if seq_ready2:
                            _ = sys.stdin.read(1)
                    else:
                        break
                    continue

                step_f = float(args.step_forward) * speed_scale
                step_r = float(args.step_right) * speed_scale
                step_u = float(args.step_updown) * speed_scale
                step_y = float(args.step_yaw) * speed_scale

                if ch == "w":
                    cmd_forward = _clamp(cmd_forward + step_f, -max_forward, max_forward)
                elif ch == "s":
                    cmd_forward = _clamp(cmd_forward - step_f, -max_forward, max_forward)
                elif ch == "d":
                    cmd_right = _clamp(cmd_right + step_r, -max_right, max_right)
                elif ch == "a":
                    cmd_right = _clamp(cmd_right - step_r, -max_right, max_right)
                elif ch == "r":
                    cmd_down = _clamp(cmd_down - step_u, -max_down, max_down)
                elif ch == "f":
                    cmd_down = _clamp(cmd_down + step_u, -max_down, max_down)
                elif ch == "e":
                    cmd_yaw = _clamp(cmd_yaw + step_y, -max_yaw, max_yaw)
                elif ch == "q":
                    cmd_yaw = _clamp(cmd_yaw - step_y, -max_yaw, max_yaw)
                elif ch in (" ", "c"):
                    cmd_forward = 0.0
                    cmd_right = 0.0
                    cmd_down = 0.0
                    cmd_yaw = 0.0
                elif ch == "z":
                    speed_scale = _clamp(
                        speed_scale - float(args.speed_scale_step),
                        float(args.speed_scale_min),
                        float(args.speed_scale_max),
                    )
                    print(f"[PX4_MANUAL] speed_scale={speed_scale:.2f}")
                elif ch == "x":
                    speed_scale = _clamp(
                        speed_scale + float(args.speed_scale_step),
                        float(args.speed_scale_min),
                        float(args.speed_scale_max),
                    )
                    print(f"[PX4_MANUAL] speed_scale={speed_scale:.2f}")
                elif ch == "1":
                    _safe_call("arm", bridge.arm)
                elif ch == "2":
                    _safe_call("takeoff", bridge.takeoff)
                elif ch == "3":
                    _safe_call("land", bridge.land)
                elif ch == "4":
                    _safe_call("disarm", bridge.disarm)
                elif ch == "m":
                    print(f"[PX4_MANUAL] metadata: {bridge.metadata()}")
                elif ch == "h":
                    _print_help()

            bridge.set_velocity_xyz(
                forward_m_s=cmd_forward,
                right_m_s=cmd_right,
                down_m_s=cmd_down,
                yaw_rate_deg_s=cmd_yaw,
            )

            now = time.time()
            if now - last_status_ts >= status_period_s:
                meta = bridge.metadata()
                print(
                    _status_text(
                        cmd_f=cmd_forward,
                        cmd_r=cmd_right,
                        cmd_d=cmd_down,
                        cmd_y=cmd_yaw,
                        speed_scale=speed_scale,
                        meta=meta,
                    )
                )
                last_status_ts = now
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        bridge.stop_motion()
        if args.auto_land_on_exit:
            meta = bridge.metadata()
            if bool(meta.get("in_air", False)):
                _safe_call("auto_land_on_exit", bridge.land)
        bridge.close()
        print("[PX4_MANUAL] closed")


if __name__ == "__main__":
    main()

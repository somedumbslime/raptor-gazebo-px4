#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OffboardSmokeCfg:
    enabled: bool = True
    duration_s: float = 5.0
    send_hz: float = 10.0
    forward_m_s: float = 0.0
    right_m_s: float = 0.0
    down_m_s: float = 0.0
    yaw_rate_deg_s: float = 0.0


@dataclass
class Px4SmokeCfg:
    system_address: str = "udpin://0.0.0.0:14540"
    connect_timeout_s: float = 20.0
    health_timeout_s: float = 20.0
    required_health_flags: tuple[str, ...] = ("is_armable",)
    takeoff_altitude_m: float = 3.0
    takeoff_wait_s: float = 8.0
    takeoff_min_relative_alt_m: float = 1.0
    post_takeoff_hold_s: float = 2.0
    land_wait_s: float = 20.0
    auto_disarm_wait_s: float = 15.0
    offboard: OffboardSmokeCfg = field(default_factory=OffboardSmokeCfg)


def _sanitize_import_path_for_mavsdk() -> None:
    """Avoid system protobuf shadowing in conda envs.

    Some local setups inject `/usr/lib/python3/dist-packages` via PYTHONPATH,
    which makes mavsdk import system protobuf instead of env protobuf.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if not conda_prefix:
        return

    def _is_bad_path(path: str) -> bool:
        p = (path or "").strip()
        if not p:
            return False
        if p.startswith(conda_prefix):
            return False
        if p.startswith("/usr/lib/python3/dist-packages"):
            return True
        if p.startswith("/usr/local/lib/python"):
            return True
        home_local = str(Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
        if p.startswith(home_local):
            return True
        return False

    cleaned = [p for p in sys.path if not _is_bad_path(p)]
    sys.path[:] = cleaned

    py_path = os.environ.get("PYTHONPATH", "")
    if "dist-packages" in py_path:
        os.environ["PYTHONPATH"] = ""


def _load_cfg(config_path: str) -> Px4SmokeCfg:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        root = yaml.safe_load(f) or {}
    if not isinstance(root, dict):
        raise ValueError("Config root must be a mapping")

    px4 = root.get("px4", {}) if isinstance(root, dict) else {}
    if not isinstance(px4, dict):
        px4 = {}

    off = px4.get("offboard_smoke", {})
    if not isinstance(off, dict):
        off = {}

    offboard_cfg = OffboardSmokeCfg(
        enabled=bool(off.get("enabled", True)),
        duration_s=float(off.get("duration_s", 5.0)),
        send_hz=float(off.get("send_hz", 10.0)),
        forward_m_s=float(off.get("forward_m_s", 0.0)),
        right_m_s=float(off.get("right_m_s", 0.0)),
        down_m_s=float(off.get("down_m_s", 0.0)),
        yaw_rate_deg_s=float(off.get("yaw_rate_deg_s", 0.0)),
    )

    health_flags = px4.get("required_health_flags", ["is_armable"])
    if not isinstance(health_flags, list) or not health_flags:
        health_flags = ["is_armable"]

    return Px4SmokeCfg(
        system_address=str(px4.get("system_address", "udpin://0.0.0.0:14540")),
        connect_timeout_s=float(px4.get("connect_timeout_s", 20.0)),
        health_timeout_s=float(px4.get("health_timeout_s", 20.0)),
        required_health_flags=tuple(str(x) for x in health_flags),
        takeoff_altitude_m=float(px4.get("takeoff_altitude_m", 3.0)),
        takeoff_wait_s=float(px4.get("takeoff_wait_s", 8.0)),
        takeoff_min_relative_alt_m=float(px4.get("takeoff_min_relative_alt_m", 1.0)),
        post_takeoff_hold_s=float(px4.get("post_takeoff_hold_s", 2.0)),
        land_wait_s=float(px4.get("land_wait_s", 20.0)),
        auto_disarm_wait_s=float(px4.get("auto_disarm_wait_s", 15.0)),
        offboard=offboard_cfg,
    )


async def _wait_connected(drone, timeout_s: float) -> None:
    async def _inner() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return

    await asyncio.wait_for(_inner(), timeout=timeout_s)


async def _wait_health(drone, required_flags: tuple[str, ...], timeout_s: float) -> None:
    async def _inner() -> None:
        last_log_ts = 0.0
        async for health in drone.telemetry.health():
            missing = [flag for flag in required_flags if not bool(getattr(health, flag, False))]
            if not missing:
                return
            now = time.monotonic()
            if now - last_log_ts >= 1.0:
                print(f"[PX4_SMOKE] waiting health... missing={missing}")
                last_log_ts = now

    await asyncio.wait_for(_inner(), timeout=timeout_s)


async def _wait_in_air(drone, expected: bool, timeout_s: float) -> None:
    async def _inner() -> None:
        async for in_air in drone.telemetry.in_air():
            if bool(in_air) == expected:
                return

    await asyncio.wait_for(_inner(), timeout=timeout_s)


async def _wait_armed(drone, expected: bool, timeout_s: float) -> None:
    async def _inner() -> None:
        async for armed in drone.telemetry.armed():
            if bool(armed) == expected:
                return

    await asyncio.wait_for(_inner(), timeout=timeout_s)


async def _wait_relative_altitude(drone, min_relative_alt_m: float, timeout_s: float) -> None:
    async def _inner() -> None:
        last_log_ts = 0.0
        async for pos in drone.telemetry.position():
            rel_alt = float(getattr(pos, "relative_altitude_m", 0.0))
            if rel_alt >= float(min_relative_alt_m):
                return
            now = time.monotonic()
            if now - last_log_ts >= 1.0:
                print(f"[PX4_SMOKE] climb... relative_alt={rel_alt:.2f}m target>={min_relative_alt_m:.2f}m")
                last_log_ts = now

    await asyncio.wait_for(_inner(), timeout=timeout_s)


async def _run_offboard_smoke(drone, cfg: OffboardSmokeCfg) -> None:
    from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as exc:
        raise RuntimeError(f"Failed to start offboard mode: {exc}") from exc

    dt = 1.0 / max(cfg.send_hz, 1.0)
    steps = max(1, int(cfg.duration_s / dt))
    cmd = VelocityBodyYawspeed(
        float(cfg.forward_m_s),
        float(cfg.right_m_s),
        float(cfg.down_m_s),
        float(cfg.yaw_rate_deg_s),
    )
    for _ in range(steps):
        await drone.offboard.set_velocity_body(cmd)
        await asyncio.sleep(dt)

    await drone.offboard.stop()


async def run_smoke(cfg: Px4SmokeCfg, skip_offboard: bool) -> None:
    _sanitize_import_path_for_mavsdk()
    try:
        from mavsdk import System
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mavsdk is not installed. Install with: "
            "conda run -n raptor312 python -m pip install mavsdk"
        ) from exc

    drone = System()
    print(f"[PX4_SMOKE] connecting to {cfg.system_address}")
    await drone.connect(system_address=cfg.system_address)
    await _wait_connected(drone, cfg.connect_timeout_s)
    print("[PX4_SMOKE] connected")

    await _wait_health(drone, cfg.required_health_flags, cfg.health_timeout_s)
    print(f"[PX4_SMOKE] health OK ({', '.join(cfg.required_health_flags)})")

    armed = False
    in_air = False
    try:
        print("[PX4_SMOKE] arm")
        await drone.action.arm()
        await _wait_armed(drone, True, timeout_s=5.0)
        armed = True

        print(f"[PX4_SMOKE] set_takeoff_altitude={cfg.takeoff_altitude_m:.2f}m")
        await drone.action.set_takeoff_altitude(cfg.takeoff_altitude_m)
        print("[PX4_SMOKE] takeoff")
        await drone.action.takeoff()
        try:
            await _wait_in_air(drone, True, timeout_s=cfg.takeoff_wait_s)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Takeoff command sent, but vehicle didn't reach in_air state in time. "
                "Check PX4 console for takeoff denial reason (position/health/mode)."
            ) from exc
        in_air = True
        print("[PX4_SMOKE] in air")
        try:
            await _wait_relative_altitude(
                drone,
                min_relative_alt_m=cfg.takeoff_min_relative_alt_m,
                timeout_s=cfg.takeoff_wait_s,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Vehicle armed/in_air, but did not climb above {cfg.takeoff_min_relative_alt_m:.2f}m "
                f"within {cfg.takeoff_wait_s:.1f}s."
            ) from exc
        print(f"[PX4_SMOKE] climb OK (>={cfg.takeoff_min_relative_alt_m:.2f}m)")
        if cfg.post_takeoff_hold_s > 0.0:
            print(f"[PX4_SMOKE] hold after takeoff: {cfg.post_takeoff_hold_s:.1f}s")
            await asyncio.sleep(cfg.post_takeoff_hold_s)

        if cfg.offboard.enabled and not skip_offboard:
            print("[PX4_SMOKE] offboard smoke start")
            await _run_offboard_smoke(drone, cfg.offboard)
            print("[PX4_SMOKE] offboard smoke done")
    finally:
        if in_air:
            print("[PX4_SMOKE] land")
            try:
                await drone.action.land()
                await _wait_in_air(drone, False, timeout_s=cfg.land_wait_s)
                await _wait_armed(drone, False, timeout_s=cfg.auto_disarm_wait_s)
                print("[PX4_SMOKE] landed and disarmed")
            except Exception as exc:
                print(f"[PX4_SMOKE] WARN: landing/disarm cleanup failed: {exc}")
        elif armed:
            print("[PX4_SMOKE] disarm (takeoff was not completed)")
            try:
                await drone.action.disarm()
                await _wait_armed(drone, False, timeout_s=5.0)
                print("[PX4_SMOKE] disarmed")
            except Exception as exc:
                print(f"[PX4_SMOKE] WARN: disarm cleanup failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PX4 SITL smoke test: connect -> arm -> takeoff -> offboard -> land")
    parser.add_argument(
        "--config",
        default="raptor_ai/config/default_config.yaml",
        help="Path to runtime YAML config with `px4` section",
    )
    parser.add_argument(
        "--skip-offboard",
        action="store_true",
        help="Skip offboard smoke block and run only arm/takeoff/land",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)
    asyncio.run(run_smoke(cfg, skip_offboard=bool(args.skip_offboard)))


if __name__ == "__main__":
    main()

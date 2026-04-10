from __future__ import annotations

import asyncio
import math
import os
import threading
import time
from pathlib import Path
import sys
from typing import Any


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sanitize_import_path_for_mavsdk() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if not conda_prefix:
        return

    home_local = str(Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")

    def _is_bad(path: str) -> bool:
        p = (path or "").strip()
        if not p:
            return False
        if p.startswith(conda_prefix):
            return False
        if p.startswith("/usr/lib/python3/dist-packages"):
            return True
        if p.startswith("/usr/local/lib/python"):
            return True
        if p.startswith(home_local):
            return True
        return False

    sys.path[:] = [p for p in sys.path if not _is_bad(p)]
    if "dist-packages" in str(os.environ.get("PYTHONPATH", "")):
        os.environ["PYTHONPATH"] = ""


def _is_offboard_mode(mode_value: str | None) -> bool:
    mode = (mode_value or "").upper()
    return "OFFBOARD" in mode


class Px4Bridge:
    """
    MAVSDK-based PX4 platform bridge.

    Design goals:
    - Keep RuntimeV2 synchronous API (`publish_commands`) unchanged.
    - Run MAVSDK I/O in a dedicated background asyncio loop.
    - Expose explicit action API for phase-5 orchestration:
      arm(), takeoff(), land(), disarm(), set_velocity_xy/xyz(), set_yaw_rate().
    """

    platform_type = "px4"
    DEFAULT_ACTIVE_STATES = {"LOCKED", "LOST", "SEARCHING", "TRACKING_XY", "TRACKING_XYZ"}
    FOLLOW_STATES = {"TRACKING_XY", "TRACKING_XYZ"}

    def __init__(self, cfg: dict[str, Any]):
        self._cfg = dict(cfg)

        self.camera_topic = str(self._cfg.get("camera_topic", "/raptor/iris/camera"))
        self.system_address = str(self._cfg.get("system_address", "udpin://0.0.0.0:14540"))
        self.connect_timeout_s = float(self._cfg.get("connect_timeout_s", 20.0))
        self.action_timeout_s = float(self._cfg.get("action_timeout_s", 15.0))
        self.command_hz = float(self._cfg.get("command_hz", 20.0))
        self.watchdog_timeout_s = float(self._cfg.get("watchdog_timeout_s", 0.5))
        self.offboard_enabled = bool(self._cfg.get("offboard_enabled", True))
        self.offboard_require_in_air = bool(self._cfg.get("offboard_require_in_air", True))
        self.cv_only = bool(self._cfg.get("cv_only", False))
        self.takeoff_altitude_m = float(self._cfg.get("takeoff_altitude_m", 3.0))
        self.offboard_start_confirm_timeout_s = float(self._cfg.get("offboard_start_confirm_timeout_s", 3.0))
        self.legacy_gimbal_mapping = bool(self._cfg.get("legacy_gimbal_mapping", True))
        self.cmd_smoothing_alpha = float(self._cfg.get("cmd_smoothing_alpha", 0.25))
        self.cmd_smoothing_alpha = _clamp(self.cmd_smoothing_alpha, 0.0, 1.0)

        self.max_forward_m_s = float(self._cfg.get("max_forward_m_s", 1.5))
        self.max_right_m_s = float(self._cfg.get("max_right_m_s", 1.5))
        self.max_down_m_s = float(self._cfg.get("max_down_m_s", 0.8))
        self.max_yaw_rate_deg_s = float(self._cfg.get("max_yaw_rate_deg_s", 45.0))
        self.pitch_to_vertical_gain = float(self._cfg.get("pitch_to_vertical_gain", 0.4))
        self.yaw_to_yaw_rate_gain = float(self._cfg.get("yaw_to_yaw_rate_gain", 30.0))

        raw_states = self._cfg.get("active_states", sorted(self.DEFAULT_ACTIVE_STATES))
        if isinstance(raw_states, list):
            self.active_states = {str(s) for s in raw_states}
        else:
            self.active_states = set(self.DEFAULT_ACTIVE_STATES)

        self._lock = threading.Lock()
        self._desired_cmd = {
            "forward_m_s": 0.0,
            "right_m_s": 0.0,
            "down_m_s": 0.0,
            "yaw_rate_deg_s": 0.0,
            "ts": 0.0,
        }
        self._smoothed_cmd = {
            "forward_m_s": 0.0,
            "right_m_s": 0.0,
            "down_m_s": 0.0,
            "yaw_rate_deg_s": 0.0,
        }
        self._last_applied_cmd = {
            "forward_m_s": 0.0,
            "right_m_s": 0.0,
            "down_m_s": 0.0,
            "yaw_rate_deg_s": 0.0,
        }
        self._last_applied_ts = 0.0

        self._connected = False
        self._armed = False
        self._in_air = False
        self._is_armable = False
        self._is_local_position_ok = False
        self._last_heartbeat_ts = 0.0
        self._last_error: str | None = None
        self._offboard_started = False
        self._flight_mode: str | None = None
        self._relative_altitude_m = 0.0
        self._relative_altitude_ned_m = 0.0
        self._home_down_m: float | None = None
        self._vel_north_m_s = 0.0
        self._vel_east_m_s = 0.0
        self._vel_down_m_s = 0.0

        self._stop_evt = threading.Event()
        self._loop_ready_evt = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._drone = None
        self._VelocityBodyYawspeed = None

        self._thread = threading.Thread(target=self._thread_main, name="px4-bridge", daemon=True)
        self._thread.start()
        self._loop_ready_evt.wait(timeout=2.0)

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = str(message)

    def _clear_error(self) -> None:
        with self._lock:
            self._last_error = None

    async def _wait_for_offboard_mode(self, timeout_s: float = 1.2) -> bool:
        t0 = time.time()
        while (time.time() - t0) < max(0.1, float(timeout_s)):
            with self._lock:
                mode = self._flight_mode
            if _is_offboard_mode(mode):
                return True
            await asyncio.sleep(0.05)
        return False

    async def _start_offboard_with_confirmation(self) -> None:
        if self._drone is None:
            raise RuntimeError("PX4 bridge not connected")
        if self._VelocityBodyYawspeed is None:
            raise RuntimeError("VelocityBodyYawspeed is unavailable")

        # Prime PX4 with neutral setpoints before OFFBOARD start.
        for _ in range(5):
            await self._drone.offboard.set_velocity_body(self._VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.05)

        await self._drone.offboard.start()
        # Keep a neutral stream while waiting mode transition.
        # PX4 can reject/fall back to HOLD if offboard setpoints are not continuous.
        deadline = time.time() + max(0.3, self.offboard_start_confirm_timeout_s)
        while time.time() < deadline:
            if await self._wait_for_offboard_mode(timeout_s=0.15):
                self._offboard_started = True
                self._clear_error()
                return
            await self._drone.offboard.set_velocity_body(self._VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.05)
        # Final race-safe check: telemetry update may arrive right at timeout boundary.
        with self._lock:
            mode = self._flight_mode
        if _is_offboard_mode(mode):
            self._offboard_started = True
            self._clear_error()
            return
        raise RuntimeError(f"OFFBOARD start sent, but flight_mode is {self._flight_mode}")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._loop_ready_evt.set()
        try:
            loop.run_until_complete(self._async_main())
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._set_error(f"px4 bridge loop crashed: {exc}")
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()

    async def _async_main(self) -> None:
        _sanitize_import_path_for_mavsdk()
        try:
            from mavsdk import System
            from mavsdk.offboard import VelocityBodyYawspeed
        except Exception as exc:
            self._set_error(f"mavsdk import failed: {exc}")
            return

        self._VelocityBodyYawspeed = VelocityBodyYawspeed
        self._drone = System()

        try:
            await self._drone.connect(system_address=self.system_address)
            await self._wait_connected(self.connect_timeout_s)
        except Exception as exc:
            self._set_error(f"connection failed: {exc}")
            return

        tasks = [
            asyncio.create_task(self._connection_worker()),
            asyncio.create_task(self._armed_worker()),
            asyncio.create_task(self._in_air_worker()),
            asyncio.create_task(self._health_worker()),
            asyncio.create_task(self._flight_mode_worker()),
            asyncio.create_task(self._position_worker()),
            asyncio.create_task(self._position_velocity_ned_worker()),
            asyncio.create_task(self._command_worker()),
        ]
        try:
            while not self._stop_evt.is_set():
                await asyncio.sleep(0.05)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._stop_offboard_if_started()

    async def _wait_connected(self, timeout_s: float) -> None:
        assert self._drone is not None

        async def _inner() -> None:
            async for state in self._drone.core.connection_state():
                if bool(state.is_connected):
                    return

        await asyncio.wait_for(_inner(), timeout=max(1e-3, float(timeout_s)))

    async def _connection_worker(self) -> None:
        assert self._drone is not None
        async for state in self._drone.core.connection_state():
            with self._lock:
                self._connected = bool(state.is_connected)
                if self._connected:
                    self._last_heartbeat_ts = time.time()
            if self._stop_evt.is_set():
                return

    async def _armed_worker(self) -> None:
        assert self._drone is not None
        async for armed in self._drone.telemetry.armed():
            with self._lock:
                self._armed = bool(armed)
            if self._stop_evt.is_set():
                return

    async def _in_air_worker(self) -> None:
        assert self._drone is not None
        async for in_air in self._drone.telemetry.in_air():
            with self._lock:
                self._in_air = bool(in_air)
            if self._stop_evt.is_set():
                return

    async def _health_worker(self) -> None:
        assert self._drone is not None
        async for health in self._drone.telemetry.health():
            with self._lock:
                self._is_armable = bool(getattr(health, "is_armable", False))
                self._is_local_position_ok = bool(getattr(health, "is_local_position_ok", False))
            if self._stop_evt.is_set():
                return

    async def _flight_mode_worker(self) -> None:
        assert self._drone is not None
        async for mode in self._drone.telemetry.flight_mode():
            with self._lock:
                self._flight_mode = str(mode)
            if self._stop_evt.is_set():
                return

    async def _position_worker(self) -> None:
        assert self._drone is not None
        async for pos in self._drone.telemetry.position():
            with self._lock:
                self._relative_altitude_m = float(getattr(pos, "relative_altitude_m", 0.0))
            if self._stop_evt.is_set():
                return

    async def _position_velocity_ned_worker(self) -> None:
        assert self._drone is not None
        async for pvn in self._drone.telemetry.position_velocity_ned():
            pos = getattr(pvn, "position", None)
            vel = getattr(pvn, "velocity", None)
            if vel is None:
                continue
            rel_alt_ned = None
            if pos is not None:
                down_m = float(getattr(pos, "down_m", 0.0))
                with self._lock:
                    if self._home_down_m is None:
                        self._home_down_m = down_m
                    home_down_m = float(self._home_down_m)
                rel_alt_ned = max(0.0, home_down_m - down_m)
            with self._lock:
                self._vel_north_m_s = float(getattr(vel, "north_m_s", 0.0))
                self._vel_east_m_s = float(getattr(vel, "east_m_s", 0.0))
                self._vel_down_m_s = float(getattr(vel, "down_m_s", 0.0))
                if rel_alt_ned is not None:
                    self._relative_altitude_ned_m = rel_alt_ned
            if self._stop_evt.is_set():
                return

    def _read_desired_cmd(self) -> tuple[float, float, float, float, bool]:
        with self._lock:
            forward_m_s = float(self._desired_cmd.get("forward_m_s", 0.0))
            right_m_s = float(self._desired_cmd.get("right_m_s", 0.0))
            down_m_s = float(self._desired_cmd.get("down_m_s", 0.0))
            yaw_rate_deg_s = float(self._desired_cmd.get("yaw_rate_deg_s", 0.0))
            ts = float(self._desired_cmd.get("ts", 0.0))

        if (time.time() - ts) > self.watchdog_timeout_s:
            return 0.0, 0.0, 0.0, 0.0, True

        return forward_m_s, right_m_s, down_m_s, yaw_rate_deg_s, False

    def _reset_smoothed_cmd(self) -> None:
        with self._lock:
            self._smoothed_cmd["forward_m_s"] = 0.0
            self._smoothed_cmd["right_m_s"] = 0.0
            self._smoothed_cmd["down_m_s"] = 0.0
            self._smoothed_cmd["yaw_rate_deg_s"] = 0.0

    def _smooth_cmd(self, forward_m_s: float, right_m_s: float, down_m_s: float, yaw_rate_deg_s: float) -> tuple[float, float, float, float]:
        a = self.cmd_smoothing_alpha
        if a >= 0.999:
            return forward_m_s, right_m_s, down_m_s, yaw_rate_deg_s

        with self._lock:
            prev_f = float(self._smoothed_cmd["forward_m_s"])
            prev_r = float(self._smoothed_cmd["right_m_s"])
            prev_d = float(self._smoothed_cmd["down_m_s"])
            prev_y = float(self._smoothed_cmd["yaw_rate_deg_s"])

        out_f = a * forward_m_s + (1.0 - a) * prev_f
        out_r = a * right_m_s + (1.0 - a) * prev_r
        out_d = a * down_m_s + (1.0 - a) * prev_d
        out_y = a * yaw_rate_deg_s + (1.0 - a) * prev_y

        with self._lock:
            self._smoothed_cmd["forward_m_s"] = out_f
            self._smoothed_cmd["right_m_s"] = out_r
            self._smoothed_cmd["down_m_s"] = out_d
            self._smoothed_cmd["yaw_rate_deg_s"] = out_y

        return out_f, out_r, out_d, out_y

    async def _stop_offboard_if_started(self) -> None:
        if not self._offboard_started or self._drone is None:
            return
        try:
            await self._drone.offboard.stop()
        except Exception:
            pass
        finally:
            self._offboard_started = False

    async def _command_worker(self) -> None:
        period_s = 1.0 / max(1.0, self.command_hz)
        while not self._stop_evt.is_set():
            try:
                await asyncio.sleep(period_s)
                if self._drone is None:
                    continue

                with self._lock:
                    connected = self._connected
                    armed = self._armed
                    flight_mode = self._flight_mode

                if not connected:
                    continue

                if not self.offboard_enabled:
                    if self._offboard_started:
                        await self._stop_offboard_if_started()
                    self._reset_smoothed_cmd()
                    continue

                if not armed:
                    if self._offboard_started:
                        await self._stop_offboard_if_started()
                    self._reset_smoothed_cmd()
                    continue

                # If PX4 left OFFBOARD (e.g. HOLD), force restart path.
                if self._offboard_started and not _is_offboard_mode(flight_mode):
                    self._offboard_started = False
                    self._reset_smoothed_cmd()
                    self._set_error(f"offboard lost: flight_mode={flight_mode}")

                if not self._offboard_started:
                    self._reset_smoothed_cmd()
                    continue

                forward_m_s, right_m_s, down_m_s, yaw_rate_deg_s, stale = self._read_desired_cmd()
                if stale:
                    self._reset_smoothed_cmd()
                forward_m_s, right_m_s, down_m_s, yaw_rate_deg_s = self._smooth_cmd(
                    forward_m_s,
                    right_m_s,
                    down_m_s,
                    yaw_rate_deg_s,
                )
                try:
                    await self._drone.offboard.set_velocity_body(
                        self._VelocityBodyYawspeed(forward_m_s, right_m_s, down_m_s, yaw_rate_deg_s)
                    )
                    with self._lock:
                        self._last_applied_cmd["forward_m_s"] = float(forward_m_s)
                        self._last_applied_cmd["right_m_s"] = float(right_m_s)
                        self._last_applied_cmd["down_m_s"] = float(down_m_s)
                        self._last_applied_cmd["yaw_rate_deg_s"] = float(yaw_rate_deg_s)
                        self._last_applied_ts = time.time()
                    self._clear_error()
                except Exception as exc:
                    self._set_error(f"offboard_set_velocity failed: {exc}")
                    self._offboard_started = False
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Keep command worker alive even on unexpected errors.
                self._set_error(f"command_worker exception: {exc}")
                await asyncio.sleep(min(0.5, period_s))

    def _submit(self, coro, timeout_s: float | None = None):
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("PX4 bridge loop is not running")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout_s or self.action_timeout_s)

    async def _arm_async(self) -> bool:
        if self._drone is None:
            raise RuntimeError("PX4 bridge not connected")
        await self._drone.action.arm()
        return True

    async def _takeoff_async(self) -> bool:
        if self._drone is None:
            raise RuntimeError("PX4 bridge not connected")
        await self._drone.action.set_takeoff_altitude(self.takeoff_altitude_m)
        await self._drone.action.takeoff()
        return True

    async def _land_async(self) -> bool:
        if self._drone is None:
            raise RuntimeError("PX4 bridge not connected")
        await self._drone.action.land()
        return True

    async def _disarm_async(self) -> bool:
        if self._drone is None:
            raise RuntimeError("PX4 bridge not connected")
        await self._drone.action.disarm()
        return True

    async def _ensure_offboard_started_async(self) -> bool:
        if self._drone is None:
            raise RuntimeError("PX4 bridge not connected")
        if not self.offboard_enabled:
            raise RuntimeError("offboard is disabled in config")

        with self._lock:
            connected = bool(self._connected)
            armed = bool(self._armed)
            in_air = bool(self._in_air)
            mode = self._flight_mode

        if not connected:
            raise RuntimeError("vehicle is not connected")
        if not armed:
            raise RuntimeError("vehicle is not armed")
        if self.offboard_require_in_air and not in_air:
            raise RuntimeError("vehicle is not in_air yet")
        if self._offboard_started and _is_offboard_mode(mode):
            return True

        try:
            await self._start_offboard_with_confirmation()
            return True
        except Exception as exc:
            self._set_error(f"ensure_offboard_started failed: {exc}")
            self._offboard_started = False
            raise

    def arm(self, timeout_s: float | None = None) -> bool:
        return bool(self._submit(self._arm_async(), timeout_s))

    def takeoff(self, timeout_s: float | None = None) -> bool:
        return bool(self._submit(self._takeoff_async(), timeout_s))

    def land(self, timeout_s: float | None = None) -> bool:
        return bool(self._submit(self._land_async(), timeout_s))

    def disarm(self, timeout_s: float | None = None) -> bool:
        return bool(self._submit(self._disarm_async(), timeout_s))

    def ensure_offboard_started(self, timeout_s: float | None = None) -> bool:
        return bool(self._submit(self._ensure_offboard_started_async(), timeout_s))

    def set_velocity_xyz(
        self,
        *,
        forward_m_s: float,
        right_m_s: float,
        down_m_s: float,
        yaw_rate_deg_s: float,
    ) -> None:
        with self._lock:
            self._desired_cmd["forward_m_s"] = _clamp(float(forward_m_s), -self.max_forward_m_s, self.max_forward_m_s)
            self._desired_cmd["right_m_s"] = _clamp(float(right_m_s), -self.max_right_m_s, self.max_right_m_s)
            self._desired_cmd["down_m_s"] = _clamp(float(down_m_s), -self.max_down_m_s, self.max_down_m_s)
            self._desired_cmd["yaw_rate_deg_s"] = _clamp(
                float(yaw_rate_deg_s),
                -self.max_yaw_rate_deg_s,
                self.max_yaw_rate_deg_s,
            )
            self._desired_cmd["ts"] = time.time()

    def set_velocity_xy(self, *, forward_m_s: float, right_m_s: float) -> None:
        with self._lock:
            current_down = float(self._desired_cmd.get("down_m_s", 0.0))
            current_yaw = float(self._desired_cmd.get("yaw_rate_deg_s", 0.0))
        self.set_velocity_xyz(
            forward_m_s=forward_m_s,
            right_m_s=right_m_s,
            down_m_s=current_down,
            yaw_rate_deg_s=current_yaw,
        )

    def set_yaw_rate(self, yaw_rate_deg_s: float) -> None:
        with self._lock:
            current_forward = float(self._desired_cmd.get("forward_m_s", 0.0))
            current_right = float(self._desired_cmd.get("right_m_s", 0.0))
            current_down = float(self._desired_cmd.get("down_m_s", 0.0))
        self.set_velocity_xyz(
            forward_m_s=current_forward,
            right_m_s=current_right,
            down_m_s=current_down,
            yaw_rate_deg_s=yaw_rate_deg_s,
        )

    def stop_motion(self) -> None:
        self.set_velocity_xyz(forward_m_s=0.0, right_m_s=0.0, down_m_s=0.0, yaw_rate_deg_s=0.0)

    def publish_commands(
        self,
        yaw_cmd: float,
        pitch_cmd: float,
        *,
        state: str | None = None,
        follow_cmd: dict[str, Any] | None = None,
    ) -> None:
        if self.cv_only:
            self.stop_motion()
            return

        # If runtime provides a follow command, treat it as authoritative guidance contract.
        # Do not fall back to legacy gimbal mapping in LOST/SEARCHING states because that can
        # map pitch search motion into vertical velocity and drag PX4 down.
        if follow_cmd is not None:
            if bool(follow_cmd.get("active", False)) and state in self.FOLLOW_STATES:
                # follow_cmd.vz follows previous platform convention: positive means "up".
                # PX4 body velocity command uses "down" positive.
                self.set_velocity_xyz(
                    forward_m_s=float(follow_cmd.get("vx_body", 0.0)),
                    right_m_s=float(follow_cmd.get("vy_body", 0.0)),
                    down_m_s=-float(follow_cmd.get("vz", 0.0)),
                    yaw_rate_deg_s=math.degrees(float(follow_cmd.get("yaw_rate", 0.0))),
                )
            else:
                self.stop_motion()
            return

        if state in self.active_states and self.legacy_gimbal_mapping:
            yaw_rate_deg_s = float(yaw_cmd) * self.yaw_to_yaw_rate_gain
            down_m_s = float(pitch_cmd) * self.pitch_to_vertical_gain
            self.set_velocity_xyz(
                forward_m_s=0.0,
                right_m_s=0.0,
                down_m_s=down_m_s,
                yaw_rate_deg_s=yaw_rate_deg_s,
            )
            return

        self.stop_motion()

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return {
                "platform_type": self.platform_type,
                "camera_topic": self.camera_topic,
                "system_address": self.system_address,
                "connect_timeout_s": self.connect_timeout_s,
                "command_hz": self.command_hz,
                "watchdog_timeout_s": self.watchdog_timeout_s,
                "offboard_enabled": self.offboard_enabled,
                "offboard_require_in_air": self.offboard_require_in_air,
                "cv_only": self.cv_only,
                "cmd_smoothing_alpha": self.cmd_smoothing_alpha,
                "takeoff_altitude_m": self.takeoff_altitude_m,
                "offboard_start_confirm_timeout_s": self.offboard_start_confirm_timeout_s,
                "connected": self._connected,
                "armed": self._armed,
                "in_air": self._in_air,
                "is_armable": self._is_armable,
                "is_local_position_ok": self._is_local_position_ok,
                "last_heartbeat_ts": self._last_heartbeat_ts,
                "last_error": self._last_error,
                "offboard_started": self._offboard_started,
                "offboard_mode_active": _is_offboard_mode(self._flight_mode),
                "flight_mode": self._flight_mode,
                "relative_altitude_m": max(self._relative_altitude_m, self._relative_altitude_ned_m),
                "relative_altitude_global_m": self._relative_altitude_m,
                "relative_altitude_ned_m": self._relative_altitude_ned_m,
                "vel_north_m_s": self._vel_north_m_s,
                "vel_east_m_s": self._vel_east_m_s,
                "vel_down_m_s": self._vel_down_m_s,
                "desired_cmd": dict(self._desired_cmd),
                "smoothed_cmd": dict(self._smoothed_cmd),
                "last_applied_cmd": dict(self._last_applied_cmd),
                "last_applied_ts": self._last_applied_ts,
                "max_forward_m_s": self.max_forward_m_s,
                "max_right_m_s": self.max_right_m_s,
                "max_down_m_s": self.max_down_m_s,
                "max_yaw_rate_deg_s": self.max_yaw_rate_deg_s,
            }

    def close(self) -> None:
        if self._stop_evt.is_set():
            return
        self.stop_motion()
        self._stop_evt.set()
        if self._loop is not None and self._loop.is_running():
            try:
                self._submit(self._stop_offboard_if_started(), timeout_s=2.0)
            except Exception:
                pass
        self._thread.join(timeout=3.0)

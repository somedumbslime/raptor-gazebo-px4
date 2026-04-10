from __future__ import annotations

import math
import time
from typing import Any

from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.pose_pb2 import Pose
from gz.msgs.twist_pb2 import Twist
from gz.transport import Node


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class IrisPlatform:
    platform_type = "iris"
    ACTIVE_STATES = {"LOCKED", "LOST", "SEARCHING", "TRACKING_XY", "TRACKING_XYZ"}
    FOLLOW_STATES = {"TRACKING_XY", "TRACKING_XYZ"}

    def __init__(self, cfg: dict[str, Any]):
        self._cfg = dict(cfg)

        self.cmd_twist_topic = str(self._cfg.get("cmd_twist_topic", "/raptor/iris/cmd_vel"))
        self.enable_topic = str(self._cfg.get("enable_topic", "/raptor/iris/enable"))
        self.publish_enable = bool(self._cfg.get("publish_enable", False))

        self.control_mode = str(self._cfg.get("control_mode", "hold")).strip().lower()
        if self.control_mode not in ("hold", "limited_twist", "kinematic_xy", "kinematic_xyz"):
            raise ValueError(f"Unsupported iris control_mode: {self.control_mode}")

        self.max_linear_x = float(self._cfg.get("max_linear_x", 1.2))
        self.max_linear_y = float(self._cfg.get("max_linear_y", 1.0))
        self.max_linear_z = float(self._cfg.get("max_linear_z", 0.2))
        self.max_yaw_rate = float(self._cfg.get("max_yaw_rate", 0.4))
        self.pitch_to_vertical_gain = float(self._cfg.get("pitch_to_vertical_gain", 0.3))
        self.yaw_to_yaw_rate_gain = float(self._cfg.get("yaw_to_yaw_rate_gain", 0.5))

        self.world_name = str(self._cfg.get("world_name", "raptor_mvp_iris_actor"))
        self.model_name = str(self._cfg.get("model_name", "iris_uav"))
        self.set_pose_timeout_ms = int(self._cfg.get("set_pose_timeout_ms", 1000))
        self.min_altitude = float(self._cfg.get("min_altitude", 0.8))
        self.max_altitude = float(self._cfg.get("max_altitude", 6.0))
        initial_pose = list(self._cfg.get("initial_pose", [8.0, 0.0, 2.2, 3.14159]))
        if len(initial_pose) < 4:
            initial_pose = [8.0, 0.0, 2.2, 3.14159]
        self._pose_x = float(initial_pose[0])
        self._pose_y = float(initial_pose[1])
        self._pose_z = float(initial_pose[2])
        self._pose_yaw = float(initial_pose[3])
        self._last_cmd_ts: float | None = None
        self._pose_initialized = False
        self._set_pose_service = f"/world/{self.world_name}/set_pose"

        self._node = Node()
        self._twist_pub = self._node.advertise(self.cmd_twist_topic, Twist)
        self._enable_pub = self._node.advertise(self.enable_topic, Boolean) if self.enable_topic else None
        self._enable_sent = False

    def _publish_enable_if_needed(self) -> None:
        if not self.publish_enable or self._enable_sent or self._enable_pub is None:
            return
        msg = Boolean()
        msg.data = True
        self._enable_pub.publish(msg)
        self._enable_sent = True

    @staticmethod
    def _state_allows_commands(state: str | None, allow_states: set[str]) -> bool:
        if state is None:
            return False
        return state in allow_states

    def _publish_twist(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        msg.angular.z = float(yaw_rate)
        self._twist_pub.publish(msg)

    def _send_pose(self, x: float, y: float, z: float, yaw: float) -> None:
        req = Pose()
        req.name = self.model_name
        req.position.x = float(x)
        req.position.y = float(y)
        req.position.z = float(z)
        req.orientation.x = 0.0
        req.orientation.y = 0.0
        req.orientation.z = math.sin(yaw * 0.5)
        req.orientation.w = math.cos(yaw * 0.5)
        self._node.request(self._set_pose_service, req, Pose, Boolean, self.set_pose_timeout_ms)

    def _integrate_kinematic_pose(self, vx_body: float, vy_body: float, vz: float, yaw_rate: float, dt: float) -> None:
        # Convert body-frame x/y velocities to world-frame.
        c = math.cos(self._pose_yaw)
        s = math.sin(self._pose_yaw)
        vx_world = c * vx_body - s * vy_body
        vy_world = s * vx_body + c * vy_body

        self._pose_x += vx_world * dt
        self._pose_y += vy_world * dt
        self._pose_z = _clamp(self._pose_z + vz * dt, self.min_altitude, self.max_altitude)
        self._pose_yaw += yaw_rate * dt

    def publish_commands(
        self,
        yaw_cmd: float,
        pitch_cmd: float,
        *,
        state: str | None = None,
        follow_cmd: dict[str, Any] | None = None,
    ) -> None:
        self._publish_enable_if_needed()

        if self.control_mode == "hold":
            self._publish_twist(0.0, 0.0, 0.0, 0.0)
            return

        if self.control_mode == "limited_twist":
            yaw_rate = 0.0
            vz = 0.0
            if self._state_allows_commands(state, self.ACTIVE_STATES):
                yaw_rate = _clamp(float(yaw_cmd) * self.yaw_to_yaw_rate_gain, -self.max_yaw_rate, self.max_yaw_rate)
                vz = _clamp(float(pitch_cmd) * self.pitch_to_vertical_gain, -self.max_linear_z, self.max_linear_z)
            self._publish_twist(0.0, 0.0, vz, yaw_rate)
            return

        # Phase 5: kinematic follow for static iris model.
        now = time.time()
        if self._last_cmd_ts is None:
            self._last_cmd_ts = now
            if not self._pose_initialized:
                self._send_pose(self._pose_x, self._pose_y, self._pose_z, self._pose_yaw)
                self._pose_initialized = True
            self._publish_twist(0.0, 0.0, 0.0, 0.0)
            return
        dt = max(1e-3, now - self._last_cmd_ts)
        self._last_cmd_ts = now

        vx_body = 0.0
        vy_body = 0.0
        vz = 0.0
        yaw_rate = 0.0
        if follow_cmd is not None and bool(follow_cmd.get("active", False)) and self._state_allows_commands(state, self.FOLLOW_STATES):
            vx_body = _clamp(float(follow_cmd.get("vx_body", 0.0)), -self.max_linear_x, self.max_linear_x)
            vy_body = _clamp(float(follow_cmd.get("vy_body", 0.0)), -self.max_linear_y, self.max_linear_y)
            yaw_rate = _clamp(float(follow_cmd.get("yaw_rate", 0.0)), -self.max_yaw_rate, self.max_yaw_rate)
            if self.control_mode == "kinematic_xyz":
                vz = _clamp(float(follow_cmd.get("vz", 0.0)), -self.max_linear_z, self.max_linear_z)

        self._integrate_kinematic_pose(vx_body, vy_body, vz, yaw_rate, dt)
        self._send_pose(self._pose_x, self._pose_y, self._pose_z, self._pose_yaw)
        self._pose_initialized = True
        self._publish_twist(vx_body, vy_body, vz, yaw_rate)

    def metadata(self) -> dict[str, Any]:
        return {
            "platform_type": self.platform_type,
            "cmd_twist_topic": self.cmd_twist_topic,
            "enable_topic": self.enable_topic,
            "publish_enable": self.publish_enable,
            "control_mode": self.control_mode,
            "world_name": self.world_name,
            "model_name": self.model_name,
            "set_pose_service": self._set_pose_service,
            "max_linear_x": self.max_linear_x,
            "max_linear_y": self.max_linear_y,
            "max_linear_z": self.max_linear_z,
            "max_yaw_rate": self.max_yaw_rate,
            "min_altitude": self.min_altitude,
            "max_altitude": self.max_altitude,
        }

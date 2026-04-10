from __future__ import annotations

from typing import Any

from gz.msgs.double_pb2 import Double
from gz.transport import Node


class GimbalPlatform:
    platform_type = "gimbal"

    def __init__(self, cfg: dict[str, Any]):
        self._cfg = dict(cfg)
        self.yaw_topic = str(self._cfg.get("yaw_topic", "/raptor/gimbal/yaw_cmd"))
        self.pitch_topic = str(self._cfg.get("pitch_topic", "/raptor/gimbal/pitch_cmd"))

        self._node = Node()
        self._yaw_pub = self._node.advertise(self.yaw_topic, Double)
        self._pitch_pub = self._node.advertise(self.pitch_topic, Double)

    def publish_commands(
        self,
        yaw_cmd: float,
        pitch_cmd: float,
        *,
        state: str | None = None,
        follow_cmd: dict[str, Any] | None = None,
    ) -> None:
        del state
        del follow_cmd
        yaw_msg = Double()
        yaw_msg.data = float(yaw_cmd)
        self._yaw_pub.publish(yaw_msg)

        pitch_msg = Double()
        pitch_msg.data = float(pitch_cmd)
        self._pitch_pub.publish(pitch_msg)

    def metadata(self) -> dict[str, Any]:
        return {
            "platform_type": self.platform_type,
            "yaw_topic": self.yaw_topic,
            "pitch_topic": self.pitch_topic,
        }

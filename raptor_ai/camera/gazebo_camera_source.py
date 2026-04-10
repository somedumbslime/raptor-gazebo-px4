from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from gz.msgs.image_pb2 import Image
from gz.transport import Node


class GazeboCameraSource:
    def __init__(self, topic: str):
        self.topic = topic
        self._node = Node()
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_ts: float | None = None
        self.frames_received = 0

        ok = self._node.subscribe(Image, self.topic, self._image_cb)
        if not ok:
            raise RuntimeError(f"Failed to subscribe to {self.topic}")

    @staticmethod
    def decode_image(msg: Image) -> np.ndarray:
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if msg.pixel_format_type == 3:  # RGB_INT8
            frame = data.reshape((msg.height, msg.width, 3))
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if msg.pixel_format_type == 8:  # BGR_INT8
            return data.reshape((msg.height, msg.width, 3)).copy()

        if msg.pixel_format_type == 1:  # L_INT8
            gray = data.reshape((msg.height, msg.width))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        raise ValueError(f"Unsupported pixel_format_type={msg.pixel_format_type}")

    def _image_cb(self, msg: Image) -> None:
        try:
            frame = self.decode_image(msg)
        except Exception:
            return

        with self._lock:
            self._latest_frame = frame
            self._latest_ts = time.time()

        self.frames_received += 1

    def get_latest_frame(self) -> tuple[np.ndarray | None, float | None]:
        with self._lock:
            if self._latest_frame is None:
                return None, None
            return self._latest_frame.copy(), self._latest_ts

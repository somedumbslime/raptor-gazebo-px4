from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from raptor_ai.tracking.track_types import Detection


class RedDetector:
    def __init__(self, cfg: dict[str, Any]):
        self.hsv_ranges = cfg.get("hsv_ranges", [])
        self.min_area = float(cfg.get("min_area", 150.0))
        self.kernel_size = int(cfg.get("morphology_kernel", 5))
        self.cls_id = int(cfg.get("cls_id", 0))
        self.cls_name = str(cfg.get("cls_name", "synthetic_target"))
        self.source = str(cfg.get("source", "red_detector"))

    def _build_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

        for r in self.hsv_ranges:
            lower = np.array(r.get("lower", [0, 100, 80]), dtype=np.uint8)
            upper = np.array(r.get("upper", [10, 255, 255]), dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        k = max(1, self.kernel_size)
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        mask = self._build_mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w * 0.5
            cy = y + h * 0.5
            detections.append(
                {
                    "bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)],
                    "center": [float(cx), float(cy)],
                    "area": area,
                    "conf": 1.0,
                    "cls_id": self.cls_id,
                    "cls_name": self.cls_name,
                    "source": self.source,
                }
            )

        detections.sort(key=lambda d: float(d["area"]), reverse=True)
        return detections

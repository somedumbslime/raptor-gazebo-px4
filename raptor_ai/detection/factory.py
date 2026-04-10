from __future__ import annotations

from typing import Any

from raptor_ai.detection.onnx_yolo_detector import OnnxYoloDetector
from raptor_ai.detection.red_detector import RedDetector


def build_detector(detector_cfg: dict[str, Any]):
    detector_type = str(detector_cfg.get("type", "")).strip().lower()

    if not detector_type:
        if "hsv_ranges" in detector_cfg:
            detector_type = "red"
        elif "model_path" in detector_cfg:
            detector_type = "yolo_onnx"
        else:
            detector_type = "red"

    if detector_type == "red":
        red_cfg = detector_cfg.get("red", detector_cfg)
        if not isinstance(red_cfg, dict):
            raise ValueError("detector.red must be a mapping")
        return RedDetector(red_cfg)

    if detector_type in ("yolo_onnx", "onnx_yolo"):
        yolo_cfg = detector_cfg.get("yolo_onnx", detector_cfg)
        if not isinstance(yolo_cfg, dict):
            raise ValueError("detector.yolo_onnx must be a mapping")
        return OnnxYoloDetector(yolo_cfg)

    raise ValueError(f"Unsupported detector.type: {detector_type}")

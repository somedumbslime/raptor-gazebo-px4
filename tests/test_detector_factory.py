from __future__ import annotations

from raptor_ai.detection.factory import build_detector
from raptor_ai.detection.onnx_yolo_detector import OnnxYoloDetector
from raptor_ai.detection.red_detector import RedDetector


def test_build_detector_red_legacy_cfg() -> None:
    det = build_detector(
        {
            "hsv_ranges": [],
            "min_area": 100,
        }
    )
    assert isinstance(det, RedDetector)


def test_build_detector_red_typed_cfg() -> None:
    det = build_detector(
        {
            "type": "red",
            "red": {
                "hsv_ranges": [],
                "min_area": 100,
            },
        }
    )
    assert isinstance(det, RedDetector)


def test_build_detector_yolo_typed_cfg_lazy() -> None:
    det = build_detector(
        {
            "type": "yolo_onnx",
            "yolo_onnx": {
                "model_path": "models/yolo26n.onnx",
                "backend": "onnxruntime",
                "lazy_init": True,
                "target_classes": ["person"],
            },
        }
    )
    assert isinstance(det, OnnxYoloDetector)

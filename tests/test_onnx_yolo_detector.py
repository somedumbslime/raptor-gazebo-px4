from __future__ import annotations

import numpy as np

from raptor_ai.detection.onnx_yolo_detector import OnnxYoloDetector


def test_postprocess_filters_by_target_class_name() -> None:
    det = OnnxYoloDetector(
        {
            "model_path": "models/yolo26n.onnx",
            "backend": "onnxruntime",
            "lazy_init": True,
            "input_size": [640, 640],
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "target_classes": ["person"],
        }
    )

    # YOLOv8-style rows: [cx, cy, w, h, class0..class79]
    pred = np.zeros((2, 84), dtype=np.float32)
    pred[0, 0:4] = [320.0, 240.0, 100.0, 140.0]
    pred[0, 4 + 0] = 0.90  # person

    pred[1, 0:4] = [200.0, 200.0, 80.0, 120.0]
    pred[1, 4 + 2] = 0.95  # car, should be filtered out

    detections = det._postprocess_predictions(pred, frame_w=640, frame_h=480)
    assert len(detections) == 1
    assert detections[0]["cls_name"] == "person"
    assert detections[0]["cls_id"] == 0


def test_postprocess_with_objectness_layout() -> None:
    det = OnnxYoloDetector(
        {
            "model_path": "models/yolo26n.onnx",
            "backend": "onnxruntime",
            "lazy_init": True,
            "input_size": [640, 640],
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "target_class_ids": [0],
            "has_objectness": True,
        }
    )

    # YOLOv5-style rows: [cx, cy, w, h, obj, class0..class79]
    pred = np.zeros((2, 85), dtype=np.float32)
    pred[0, 0:4] = [300.0, 220.0, 120.0, 160.0]
    pred[0, 4] = 0.90
    pred[0, 5 + 0] = 0.80  # 0.72 final conf

    pred[1, 0:4] = [100.0, 100.0, 80.0, 80.0]
    pred[1, 4] = 0.90
    pred[1, 5 + 0] = 0.10  # 0.09 final conf -> below threshold

    detections = det._postprocess_predictions(pred, frame_w=640, frame_h=480)
    assert len(detections) == 1
    assert detections[0]["conf"] > 0.7


def test_postprocess_end2end_xyxy_layout() -> None:
    det = OnnxYoloDetector(
        {
            "model_path": "models/yolo26n.onnx",
            "backend": "onnxruntime",
            "lazy_init": True,
            "input_size": [640, 640],
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "target_classes": ["person"],
        }
    )

    # End-to-end ONNX rows: [x1, y1, x2, y2, score, class_id]
    pred = np.array(
        [
            [280.0, 180.0, 360.0, 360.0, 0.88, 0.0],  # person
            [250.0, 170.0, 340.0, 350.0, 0.90, 2.0],  # car, filtered out
        ],
        dtype=np.float32,
    )

    detections = det._postprocess_predictions(pred, frame_w=640, frame_h=480)
    assert len(detections) == 1
    d0 = detections[0]
    assert d0["cls_id"] == 0
    assert d0["cls_name"] == "person"
    # Y coordinates are scaled by frame_h / input_h = 480 / 640 = 0.75.
    assert abs(d0["bbox_xyxy"][1] - 135.0) < 1e-4
    assert abs(d0["bbox_xyxy"][3] - 270.0) < 1e-4

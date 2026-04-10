from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from raptor_ai.detection.coco_classes import COCO80_CLASSES
from raptor_ai.tracking.track_types import Detection


class OnnxYoloDetector:
    def __init__(self, cfg: dict[str, Any]):
        model_path = Path(str(cfg.get("model_path", "models/yolo26n.onnx"))).expanduser()
        if not model_path.is_absolute():
            model_path = (Path.cwd() / model_path).resolve()
        self.model_path = model_path

        self.backend = str(cfg.get("backend", "onnxruntime")).strip().lower()
        self.providers = [str(p) for p in cfg.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])]

        input_size = cfg.get("input_size", [640, 640])
        self.input_w = int(input_size[0]) if isinstance(input_size, (list, tuple)) and len(input_size) > 0 else 640
        self.input_h = int(input_size[1]) if isinstance(input_size, (list, tuple)) and len(input_size) > 1 else 640

        self.conf_threshold = float(cfg.get("conf_threshold", 0.25))
        self.iou_threshold = float(cfg.get("iou_threshold", 0.45))
        self.max_detections = int(cfg.get("max_detections", 100))
        self.has_objectness_cfg = cfg.get("has_objectness", "auto")
        self.fallback_to_cpu_on_error = bool(cfg.get("fallback_to_cpu_on_error", True))
        self.source = str(cfg.get("source", "yolo_onnx"))

        raw_class_names = cfg.get("class_names")
        if isinstance(raw_class_names, list) and raw_class_names:
            self.class_names = [str(x) for x in raw_class_names]
        else:
            self.class_names = list(COCO80_CLASSES)

        target_class_ids = cfg.get("target_class_ids", [])
        self._target_class_ids: set[int] = {int(x) for x in target_class_ids}
        name_to_id = {name.lower(): idx for idx, name in enumerate(self.class_names)}
        target_classes = [str(x).strip().lower() for x in cfg.get("target_classes", []) if str(x).strip()]
        for cls_name in target_classes:
            if cls_name not in name_to_id:
                raise ValueError(f"Unknown target class '{cls_name}' for model classes")
            self._target_class_ids.add(int(name_to_id[cls_name]))

        self._lazy_init = bool(cfg.get("lazy_init", False))
        self._ort_session = None
        self._ort_input_name = ""
        self._ort_output_names: list[str] = []
        self._cv2_net = None

        if not self._lazy_init:
            self._ensure_backend_ready()

    def _init_ort_session(self, providers: list[str]) -> None:
        import onnxruntime as ort  # type: ignore

        self.providers = list(providers)
        self._ort_session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        self._ort_input_name = str(self._ort_session.get_inputs()[0].name)
        self._ort_output_names = [str(o.name) for o in self._ort_session.get_outputs()]
        self.backend = "onnxruntime"

    def _ensure_backend_ready(self) -> None:
        if self._ort_session is not None or self._cv2_net is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO ONNX model not found: {self.model_path}")

        backend = self.backend
        errors: list[str] = []

        if backend in ("onnxruntime", "auto"):
            try:
                import onnxruntime as ort  # type: ignore

                available = [str(p) for p in ort.get_available_providers()]
                if self.providers:
                    selected = [p for p in self.providers if p in available]
                else:
                    preferred = [
                        "TensorrtExecutionProvider",
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ]
                    selected = [p for p in preferred if p in available]
                if not selected:
                    raise RuntimeError(
                        f"No compatible ONNX Runtime providers found. "
                        f"requested={self.providers} available={available}"
                    )

                self._init_ort_session(selected)
                return
            except Exception as exc:  # noqa: BLE001 - keep detailed backend diagnostics
                errors.append(f"onnxruntime backend failed: {exc}")

        if backend in ("opencv_dnn", "auto"):
            try:
                self._cv2_net = cv2.dnn.readNetFromONNX(str(self.model_path))
                self.backend = "opencv_dnn"
                return
            except Exception as exc:  # noqa: BLE001 - keep detailed backend diagnostics
                errors.append(f"opencv_dnn backend failed: {exc}")

        if backend not in ("onnxruntime", "opencv_dnn", "auto"):
            raise ValueError(f"Unsupported detector backend: {backend}")

        raise RuntimeError("Unable to initialize YOLO ONNX detector. " + " | ".join(errors))

    def _infer_raw(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        self._ensure_backend_ready()

        if self._ort_session is not None:
            resized = cv2.resize(frame_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            x = rgb.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))[None, ...]
            try:
                outputs = self._ort_session.run(self._ort_output_names or None, {self._ort_input_name: x})
            except Exception as exc:  # noqa: BLE001 - backend runtime failures
                can_fallback = (
                    self.fallback_to_cpu_on_error
                    and "CUDAExecutionProvider" in self.providers
                    and "CPUExecutionProvider" not in self.providers
                ) or (
                    self.fallback_to_cpu_on_error
                    and "CUDAExecutionProvider" in self.providers
                    and len(self.providers) > 1
                )
                if not can_fallback:
                    raise

                print(
                    "[YOLO_ONNX] CUDA runtime failed, switching to CPUExecutionProvider. "
                    f"reason={exc}"
                )
                self._init_ort_session(["CPUExecutionProvider"])
                outputs = self._ort_session.run(self._ort_output_names or None, {self._ort_input_name: x})
            return [np.asarray(o) for o in outputs]

        if self._cv2_net is not None:
            blob = cv2.dnn.blobFromImage(
                frame_bgr,
                scalefactor=1.0 / 255.0,
                size=(self.input_w, self.input_h),
                swapRB=True,
                crop=False,
            )
            self._cv2_net.setInput(blob)
            names = self._cv2_net.getUnconnectedOutLayersNames()
            outputs = self._cv2_net.forward(names)
            if isinstance(outputs, np.ndarray):
                return [outputs]
            return [np.asarray(o) for o in outputs]

        return []

    @staticmethod
    def _extract_predictions(raw_outputs: list[np.ndarray]) -> np.ndarray:
        if not raw_outputs:
            return np.zeros((0, 0), dtype=np.float32)

        arr = np.asarray(raw_outputs[0])
        arr = np.squeeze(arr)

        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])

        if arr.ndim != 2:
            raise ValueError(f"Unsupported prediction tensor shape: {arr.shape}")

        if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 128:
            arr = arr.T
        if arr.shape[1] < 6 and arr.shape[0] >= 6:
            arr = arr.T

        return arr.astype(np.float32, copy=False)

    def _has_objectness(self, num_cols: int) -> bool:
        cfg = self.has_objectness_cfg
        if isinstance(cfg, bool):
            return cfg
        if isinstance(cfg, str):
            lowered = cfg.strip().lower()
            if lowered in ("true", "yes", "1"):
                return True
            if lowered in ("false", "no", "0"):
                return False

        n_cls = len(self.class_names)
        if num_cols == 5 + n_cls:
            return True
        if num_cols == 4 + n_cls:
            return False
        if num_cols == 85:
            return True
        if num_cols == 84:
            return False
        return False

    def _postprocess_predictions(
        self,
        pred: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> list[Detection]:
        if pred.size == 0 or pred.ndim != 2 or pred.shape[1] < 6:
            return []

        # End-to-end ONNX exports (common in Ultralytics with built-in NMS)
        # often return [x1, y1, x2, y2, score, class_id].
        if pred.shape[1] == 6:
            return self._postprocess_end2end_xyxy(pred, frame_w=frame_w, frame_h=frame_h)

        has_obj = self._has_objectness(pred.shape[1])
        cls_start = 5 if has_obj else 4
        if pred.shape[1] <= cls_start:
            return []

        sx = float(frame_w) / float(max(1, self.input_w))
        sy = float(frame_h) / float(max(1, self.input_h))

        boxes_xywh: list[list[float]] = []
        boxes_xyxy: list[list[float]] = []
        scores: list[float] = []
        cls_ids: list[int] = []

        for row in pred:
            cx, cy, bw, bh = [float(v) for v in row[:4]]
            if bw <= 0.0 or bh <= 0.0:
                continue

            cls_scores = row[cls_start:]
            if cls_scores.size <= 0:
                continue

            cls_id = int(np.argmax(cls_scores))
            cls_conf = float(cls_scores[cls_id])
            obj_conf = float(row[4]) if has_obj else 1.0
            conf = obj_conf * cls_conf
            if conf < self.conf_threshold:
                continue

            if self._target_class_ids and cls_id not in self._target_class_ids:
                continue

            x1 = (cx - bw * 0.5) * sx
            y1 = (cy - bh * 0.5) * sy
            x2 = (cx + bw * 0.5) * sx
            y2 = (cy + bh * 0.5) * sy

            x1 = float(np.clip(x1, 0.0, max(0.0, frame_w - 1.0)))
            y1 = float(np.clip(y1, 0.0, max(0.0, frame_h - 1.0)))
            x2 = float(np.clip(x2, 0.0, max(0.0, frame_w - 1.0)))
            y2 = float(np.clip(y2, 0.0, max(0.0, frame_h - 1.0)))

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < 1.0 or h < 1.0:
                continue

            boxes_xywh.append([x1, y1, w, h])
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(conf)
            cls_ids.append(cls_id)

        if not boxes_xywh:
            return []

        nms = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.conf_threshold, self.iou_threshold)
        if nms is None or len(nms) == 0:
            return []

        indices = np.array(nms).reshape(-1).tolist()
        detections: list[Detection] = []
        for idx in indices:
            cls_id = int(cls_ids[int(idx)])
            cls_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else f"class_{cls_id}"
            x1, y1, x2, y2 = boxes_xyxy[int(idx)]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            detections.append(
                {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center": [x1 + w * 0.5, y1 + h * 0.5],
                    "area": w * h,
                    "conf": float(scores[int(idx)]),
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "source": self.source,
                }
            )

        detections.sort(key=lambda d: float(d.get("conf", 0.0)), reverse=True)
        if self.max_detections > 0:
            detections = detections[: self.max_detections]
        return detections

    def _postprocess_end2end_xyxy(
        self,
        pred: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> list[Detection]:
        sx = float(frame_w) / float(max(1, self.input_w))
        sy = float(frame_h) / float(max(1, self.input_h))

        boxes_xywh: list[list[float]] = []
        boxes_xyxy: list[list[float]] = []
        scores: list[float] = []
        cls_ids: list[int] = []

        for row in pred:
            x1, y1, x2, y2, conf, cls_id_raw = [float(v) for v in row[:6]]
            if conf < self.conf_threshold:
                continue

            cls_id = int(round(cls_id_raw))
            if cls_id < 0:
                continue
            if self._target_class_ids and cls_id not in self._target_class_ids:
                continue

            # Some exports output normalized [0..1] boxes, others output pixels
            # in model-input space.
            max_abs = max(abs(x1), abs(y1), abs(x2), abs(y2))
            if max_abs <= 2.0:
                x1 *= float(frame_w)
                x2 *= float(frame_w)
                y1 *= float(frame_h)
                y2 *= float(frame_h)
            else:
                x1 *= sx
                x2 *= sx
                y1 *= sy
                y2 *= sy

            x1 = float(np.clip(min(x1, x2), 0.0, max(0.0, frame_w - 1.0)))
            y1 = float(np.clip(min(y1, y2), 0.0, max(0.0, frame_h - 1.0)))
            x2 = float(np.clip(max(x1, x2), 0.0, max(0.0, frame_w - 1.0)))
            y2 = float(np.clip(max(y1, y2), 0.0, max(0.0, frame_h - 1.0)))

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < 1.0 or h < 1.0:
                continue

            boxes_xywh.append([x1, y1, w, h])
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(float(conf))
            cls_ids.append(cls_id)

        if not boxes_xywh:
            return []

        nms = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.conf_threshold, self.iou_threshold)
        if nms is None or len(nms) == 0:
            return []

        indices = np.array(nms).reshape(-1).tolist()
        detections: list[Detection] = []
        for idx in indices:
            cls_id = int(cls_ids[int(idx)])
            cls_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else f"class_{cls_id}"
            x1, y1, x2, y2 = boxes_xyxy[int(idx)]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            detections.append(
                {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center": [x1 + w * 0.5, y1 + h * 0.5],
                    "area": w * h,
                    "conf": float(scores[int(idx)]),
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "source": self.source,
                }
            )

        detections.sort(key=lambda d: float(d.get("conf", 0.0)), reverse=True)
        if self.max_detections > 0:
            detections = detections[: self.max_detections]
        return detections

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        raw_outputs = self._infer_raw(frame_bgr)
        pred = self._extract_predictions(raw_outputs)
        frame_h, frame_w = frame_bgr.shape[:2]
        return self._postprocess_predictions(pred, frame_w=frame_w, frame_h=frame_h)

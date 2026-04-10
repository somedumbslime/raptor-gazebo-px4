from __future__ import annotations

from typing import Any

from raptor_ai.tracking.track_types import Detection, TrackInfo


class IouTracker:
    def __init__(self, cfg: dict[str, Any]):
        self.iou_match_threshold = float(cfg.get("iou_match_threshold", 0.3))
        self.max_missing_frames = int(cfg.get("max_missing_frames", 10))

        self._next_track_id = 1
        self._active_tracks: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _iou(box_a: list[float], box_b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0

        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0

        return inter_area / union

    @staticmethod
    def _xyxy_to_xywh(box: list[float]) -> list[float]:
        x1, y1, x2, y2 = box
        return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

    def _assign_track_ids(self, detections: list[Detection]) -> list[int]:
        assigned_ids = [-1] * len(detections)
        used_tracks: set[int] = set()

        for det_idx, det in enumerate(detections):
            det_box = list(det["bbox_xyxy"])
            best_track_id = -1
            best_iou = 0.0

            for track_id, state in self._active_tracks.items():
                if track_id in used_tracks:
                    continue
                iou = self._iou(det_box, state["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id != -1 and best_iou >= self.iou_match_threshold:
                assigned_ids[det_idx] = best_track_id
                used_tracks.add(best_track_id)
            else:
                new_track_id = self._next_track_id
                self._next_track_id += 1
                assigned_ids[det_idx] = new_track_id
                used_tracks.add(new_track_id)

        return assigned_ids

    def to_tracks(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int, int],
        timestamp: float,
    ) -> list[TrackInfo]:
        frame_h, frame_w = frame_shape[:2]

        for state in self._active_tracks.values():
            state["missed"] += 1

        if not detections:
            self._active_tracks = {
                track_id: st
                for track_id, st in self._active_tracks.items()
                if st["missed"] <= self.max_missing_frames
            }
            return []

        assigned_ids = self._assign_track_ids(detections)
        tracks: list[TrackInfo] = []

        for det, track_id in zip(detections, assigned_ids):
            box_xyxy = [float(v) for v in det["bbox_xyxy"]]
            center = [float(v) for v in det["center"]]

            track: TrackInfo = {
                "track_id": int(track_id),
                "bbox_xyxy": box_xyxy,
                "bbox_xywh": self._xyxy_to_xywh(box_xyxy),
                "center": center,
                "conf": float(det.get("conf", 1.0)),
                "cls_id": int(det.get("cls_id", 0)),
                "cls_name": str(det.get("cls_name", "target")),
                "area": float(det.get("area", 0.0)),
                "frame_w": int(frame_w),
                "frame_h": int(frame_h),
                "timestamp": float(timestamp),
                "source": str(det.get("source", "iou_tracker")),
                "visible": True,
            }
            tracks.append(track)

            self._active_tracks[int(track_id)] = {
                "bbox_xyxy": box_xyxy,
                "center": center,
                "missed": 0,
                "last_ts": float(timestamp),
            }

        self._active_tracks = {
            track_id: st
            for track_id, st in self._active_tracks.items()
            if st["missed"] <= self.max_missing_frames
        }

        tracks.sort(key=lambda t: float(t["area"]), reverse=True)
        return tracks

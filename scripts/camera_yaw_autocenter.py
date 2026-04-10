#!/usr/bin/env /usr/bin/python3
import math
import threading
import time

import cv2
import numpy as np

from gz.transport import Node
from gz.msgs.image_pb2 import Image
from gz.msgs.pose_pb2 import Pose
from gz.msgs.boolean_pb2 import Boolean

TOPIC = "/raptor/camera"
WORLD_NAME = "raptor_mvp"
SET_POSE_SERVICE = f"/world/{WORLD_NAME}/set_pose"

CAMERA_MODEL_NAME = "raptor_camera"

# Фиксированная позиция камеры
CAM_X = 8.0
CAM_Y = 0.0
CAM_Z = 2.2

# Базовый yaw камеры: смотрим примерно в сторону origin
BASE_YAW = math.pi

# Параметры контроллера
CONTROL_HZ = 15.0
DT = 1.0 / CONTROL_HZ

KP = 1.2                # пропорциональный коэффициент
MAX_YAW_RATE = 0.8      # рад/с
DEADZONE = 0.03         # мертвая зона по нормированной ошибке
EMA_ALPHA = 0.25        # сглаживание err_x

latest_frame = None
lock = threading.Lock()

frames_rx = 0
start_ts = time.time()
last_log_ts = 0.0


def decode_image(msg: Image):
    data = np.frombuffer(msg.data, dtype=np.uint8)

    # 3 = RGB_INT8
    # 8 = BGR_INT8
    # 1 = L_INT8
    if msg.pixel_format_type == 3:
        frame = data.reshape((msg.height, msg.width, 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    if msg.pixel_format_type == 8:
        frame = data.reshape((msg.height, msg.width, 3)).copy()
        return frame

    if msg.pixel_format_type == 1:
        gray = data.reshape((msg.height, msg.width))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported pixel_format_type={msg.pixel_format_type}")


def image_cb(msg: Image):
    global latest_frame, frames_rx
    try:
        frame = decode_image(msg)
    except Exception as e:
        print(f"[WARN] decode failed: {e}")
        return

    with lock:
        latest_frame = frame.copy()

    frames_rx += 1


def detect_red_target(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 100, 80], dtype=np.uint8)
    upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red_2 = np.array([170, 100, 80], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)

    if area < 150:
        return None, mask

    x, y, w, h = cv2.boundingRect(best)
    cx = x + w // 2
    cy = y + h // 2

    return {
        "bbox": (x, y, w, h),
        "center": (cx, cy),
        "area": float(area),
    }, mask


def set_camera_pose(node: Node, yaw: float):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    req = Pose()
    req.name = CAMERA_MODEL_NAME
    req.position.x = CAM_X
    req.position.y = CAM_Y
    req.position.z = CAM_Z
    req.orientation.x = 0.0
    req.orientation.y = 0.0
    req.orientation.z = qz
    req.orientation.w = qw

    ok, rep = node.request(SET_POSE_SERVICE, req, Pose, Boolean, 1000)
    return ok, rep.data if ok else False


def main():
    global last_log_ts

    node = Node()

    ok = node.subscribe(Image, TOPIC, image_cb)
    if not ok:
        raise RuntimeError(f"Failed to subscribe to {TOPIC}")

    yaw = BASE_YAW
    err_x_ema = 0.0

    print(f"[INFO] subscribed to {TOPIC}")
    print(f"[INFO] control service: {SET_POSE_SERVICE}")
    print("[INFO] q / ESC - exit")

    while True:
        loop_t0 = time.time()

        with lock:
            frame = None if latest_frame is None else latest_frame.copy()

        mask = None
        status = "NO FRAME"
        yaw_rate_cmd = 0.0

        if frame is not None:
            h, w = frame.shape[:2]
            frame_cx = w // 2
            frame_cy = h // 2

            target, mask = detect_red_target(frame)

            cv2.circle(frame, (frame_cx, frame_cy), 5, (0, 255, 255), -1)
            cv2.line(frame, (frame_cx, 0), (frame_cx, h), (0, 255, 255), 1)
            cv2.line(frame, (0, frame_cy), (w, frame_cy), (0, 255, 255), 1)

            if target is not None:
                x, y, bw, bh = target["bbox"]
                cx, cy = target["center"]

                err_x_px = cx - frame_cx
                err_x = err_x_px / (w / 2.0)

                err_x_ema = EMA_ALPHA * err_x + (1.0 - EMA_ALPHA) * err_x_ema

                if abs(err_x_ema) < DEADZONE:
                    yaw_rate_cmd = 0.0
                else:
                    # Если камера будет уходить "не туда" — просто поменяй знак на +
                    yaw_rate_cmd = -KP * err_x_ema
                    yaw_rate_cmd = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, yaw_rate_cmd))

                yaw += yaw_rate_cmd * DT

                pose_ok, pose_res = set_camera_pose(node, yaw)

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.line(frame, (frame_cx, frame_cy), (cx, cy), (255, 0, 255), 2)

                status = (
                    f"TARGET | err_x={err_x:+.3f} | err_x_ema={err_x_ema:+.3f} | "
                    f"yaw_rate={yaw_rate_cmd:+.3f} | yaw={yaw:+.3f} | pose_ok={pose_ok and pose_res}"
                )

            else:
                yaw_rate_cmd = 0.0
                status = f"NO TARGET | yaw={yaw:+.3f}"

            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0) if target is not None else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("RAPTOR-AI autocenter", frame)
            if mask is not None:
                cv2.imshow("RAPTOR-AI mask", mask)

            now = time.time()
            if now - last_log_ts >= 1.0:
                elapsed = max(now - start_ts, 1e-6)
                print(
                    f"[AUTO] fps={frames_rx / elapsed:.1f} | "
                    f"yaw={yaw:+.3f} | yaw_rate={yaw_rate_cmd:+.3f} | status={status}"
                )
                last_log_ts = now

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        spent = time.time() - loop_t0
        sleep_t = max(0.0, DT - spent)
        time.sleep(sleep_t)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

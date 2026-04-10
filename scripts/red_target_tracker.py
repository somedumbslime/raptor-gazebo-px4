#!/usr/bin/env /usr/bin/python3
import threading
import time

import cv2
import numpy as np

from gz.transport import Node
from gz.msgs.image_pb2 import Image

TOPIC = "/raptor/camera"

latest_frame = None
lock = threading.Lock()

frames_rx = 0
last_log_ts = 0.0
start_ts = time.time()


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
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return frame

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

    # Красный цвет в HSV обычно ловится двумя диапазонами
    lower_red_1 = np.array([0, 100, 80], dtype=np.uint8)
    upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red_2 = np.array([170, 100, 80], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Чуть чистим шум
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask

    # Берем самый крупный красный объект
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


def main():
    global last_log_ts

    node = Node()
    ok = node.subscribe(Image, TOPIC, image_cb)
    if not ok:
        raise RuntimeError(f"Failed to subscribe to {TOPIC}")

    print(f"[INFO] subscribed to {TOPIC}")
    print("[INFO] q / ESC - exit")

    while True:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is not None:
            h, w = frame.shape[:2]
            frame_cx = w // 2
            frame_cy = h // 2

            target, mask = detect_red_target(frame)

            # Рисуем центр кадра
            cv2.circle(frame, (frame_cx, frame_cy), 5, (0, 255, 255), -1)
            cv2.line(frame, (frame_cx, 0), (frame_cx, h), (0, 255, 255), 1)
            cv2.line(frame, (0, frame_cy), (w, frame_cy), (0, 255, 255), 1)

            status_text = "NO TARGET"

            if target is not None:
                x, y, bw, bh = target["bbox"]
                cx, cy = target["center"]
                area = target["area"]

                # Ошибка в пикселях
                err_x_px = cx - frame_cx
                err_y_px = cy - frame_cy

                # Нормированная ошибка [-1, 1] примерно
                err_x = err_x_px / (w / 2.0)
                err_y = err_y_px / (h / 2.0)

                area_ratio = area / float(w * h)

                # Рисуем bbox и центр цели
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.line(frame, (frame_cx, frame_cy), (cx, cy), (255, 0, 255), 2)

                status_text = (
                    f"TARGET | err_x={err_x:+.3f} | err_y={err_y:+.3f} | "
                    f"area_ratio={area_ratio:.4f}"
                )

                now = time.time()
                if now - last_log_ts >= 1.0:
                    elapsed = max(now - start_ts, 1e-6)
                    print(
                        f"[TRACK] fps={frames_rx / elapsed:.1f} | "
                        f"bbox=({x},{y},{bw},{bh}) | center=({cx},{cy}) | "
                        f"err_x={err_x:+.3f} | err_y={err_y:+.3f} | area_ratio={area_ratio:.4f}"
                    )
                    last_log_ts = now
            else:
                now = time.time()
                if now - last_log_ts >= 1.0:
                    elapsed = max(now - start_ts, 1e-6)
                    print(f"[TRACK] fps={frames_rx / elapsed:.1f} | no target")
                    last_log_ts = now

            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if target is not None else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("RAPTOR-AI tracker", frame)
            cv2.imshow("RAPTOR-AI mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        time.sleep(0.001)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

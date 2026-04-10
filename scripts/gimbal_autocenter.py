#!/usr/bin/env /usr/bin/python3
import threading
import time

import cv2
import numpy as np

from gz.transport import Node
from gz.msgs.image_pb2 import Image
from gz.msgs.double_pb2 import Double

CAM_TOPIC = "/raptor/camera"
YAW_TOPIC = "/raptor/gimbal/yaw_cmd"
PITCH_TOPIC = "/raptor/gimbal/pitch_cmd"

# Ограничения должны соответствовать SDF joint limits
YAW_MIN = -1.4
YAW_MAX = 1.4
PITCH_MIN = -0.7
PITCH_MAX = 0.7

CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

# Yaw controller
KP_YAW = 1.1
DEADZONE_X = 0.03
MAX_YAW_RATE = 0.9
EMA_ALPHA_X = 0.25

# Pitch controller
KP_PITCH = 0.9
DEADZONE_Y = 0.04
MAX_PITCH_RATE = 0.7
EMA_ALPHA_Y = 0.25

latest_frame = None
lock = threading.Lock()

frames_rx = 0
start_ts = time.time()
last_log_ts = 0.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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


def main():
    global last_log_ts

    node = Node()

    ok = node.subscribe(Image, CAM_TOPIC, image_cb)
    if not ok:
        raise RuntimeError(f"Failed to subscribe to {CAM_TOPIC}")

    yaw_pub = node.advertise(YAW_TOPIC, Double)
    pitch_pub = node.advertise(PITCH_TOPIC, Double)

    yaw_cmd = 0.0
    pitch_cmd = 0.0

    err_x_ema = 0.0
    err_y_ema = 0.0

    print(f"[INFO] subscribed to {CAM_TOPIC}")
    print(f"[INFO] publishing to {YAW_TOPIC} and {PITCH_TOPIC}")
    print("[INFO] q / ESC - exit")

    while True:
        t0 = time.time()

        with lock:
            frame = None if latest_frame is None else latest_frame.copy()

        mask = None
        target = None
        yaw_rate = 0.0
        pitch_rate = 0.0

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

                err_x = (cx - frame_cx) / (w / 2.0)
                err_y = (cy - frame_cy) / (h / 2.0)

                err_x_ema = EMA_ALPHA_X * err_x + (1.0 - EMA_ALPHA_X) * err_x_ema
                err_y_ema = EMA_ALPHA_Y * err_y + (1.0 - EMA_ALPHA_Y) * err_y_ema

                # YAW
                if abs(err_x_ema) < DEADZONE_X:
                    yaw_rate = 0.0
                else:
                    # если будет крутить не в ту сторону — поменяй знак
                    yaw_rate = -KP_YAW * err_x_ema
                    yaw_rate = clamp(yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE)

                # PITCH
                if abs(err_y_ema) < DEADZONE_Y:
                    pitch_rate = 0.0
                else:
                    # если будет поднимать/опускать наоборот — поменяй знак
                    pitch_rate = +KP_PITCH * err_y_ema
                    pitch_rate = clamp(pitch_rate, -MAX_PITCH_RATE, MAX_PITCH_RATE)

                yaw_cmd = clamp(yaw_cmd + yaw_rate * DT, YAW_MIN, YAW_MAX)
                pitch_cmd = clamp(pitch_cmd + pitch_rate * DT, PITCH_MIN, PITCH_MAX)

                yaw_msg = Double()
                yaw_msg.data = float(yaw_cmd)
                yaw_pub.publish(yaw_msg)

                pitch_msg = Double()
                pitch_msg.data = float(pitch_cmd)
                pitch_pub.publish(pitch_msg)

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.line(frame, (frame_cx, frame_cy), (cx, cy), (255, 0, 255), 2)

                status = (
                    f"TARGET | ex={err_x:+.3f} ey={err_y:+.3f} | "
                    f"yaw={yaw_cmd:+.3f} pitch={pitch_cmd:+.3f}"
                )
                color = (0, 255, 0)
            else:
                status = f"NO TARGET | yaw={yaw_cmd:+.3f} pitch={pitch_cmd:+.3f}"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("RAPTOR-AI gimbal autocenter", frame)
            if mask is not None:
                cv2.imshow("RAPTOR-AI mask", mask)

            now = time.time()
            if now - last_log_ts >= 1.0:
                elapsed = max(now - start_ts, 1e-6)
                print(
                    f"[GIMBAL] fps={frames_rx / elapsed:.1f} | "
                    f"yaw={yaw_cmd:+.3f} pitch={pitch_cmd:+.3f} | "
                    f"target={'yes' if target is not None else 'no'}"
                )
                last_log_ts = now

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        spent = time.time() - t0
        time.sleep(max(0.0, DT - spent))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
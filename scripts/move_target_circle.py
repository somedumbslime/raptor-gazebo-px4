#!/usr/bin/env /usr/bin/python3
import math
import time

from gz.transport import Node
from gz.msgs.pose_pb2 import Pose
from gz.msgs.boolean_pb2 import Boolean

WORLD_NAME = "raptor_mvp"
MODEL_NAME = "target_stub"

SERVICE = f"/world/{WORLD_NAME}/set_pose"

CENTER_X = 0.0
CENTER_Y = 0.0
Z = 0.85

RADIUS = 1.8
SPEED = 0.7          # м/с по дуге
UPDATE_HZ = 10.0     # частота команд

# omega = v / r
OMEGA = SPEED / RADIUS


def main():
    node = Node()

    print(f"[INFO] moving model '{MODEL_NAME}' via {SERVICE}")
    print(f"[INFO] radius={RADIUS} m | speed={SPEED} m/s | update_hz={UPDATE_HZ}")
    print("[INFO] Ctrl+C to stop")

    t0 = time.time()
    dt = 1.0 / UPDATE_HZ

    try:
        while True:
            t = time.time() - t0

            # Круговая траектория
            x = CENTER_X + RADIUS * math.cos(OMEGA * t)
            y = CENTER_Y + RADIUS * math.sin(OMEGA * t)

            # yaw по касательной
            yaw = OMEGA * t + math.pi / 2.0
            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)

            req = Pose()
            req.name = MODEL_NAME
            req.position.x = float(x)
            req.position.y = float(y)
            req.position.z = float(Z)
            req.orientation.x = 0.0
            req.orientation.y = 0.0
            req.orientation.z = float(qz)
            req.orientation.w = float(qw)

            ok, rep = node.request(SERVICE, req, Pose, Boolean, 1000)

            if not ok:
                print("[WARN] request transport failed")
            elif not rep.data:
                print("[WARN] service returned false")

            print(
                f"\r[POSE] x={x:+.2f} y={y:+.2f} z={Z:.2f} yaw={yaw:+.2f}",
                end="",
                flush=True,
            )

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[INFO] stopped")


if __name__ == "__main__":
    main()

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.entity_factory_pb2 import EntityFactory
from gz.msgs.pose_pb2 import Pose
from gz.msgs.pose_v_pb2 import Pose_V
from gz.transport import Node


@dataclass(frozen=True)
class ModelPose:
    name: str
    x: float
    y: float
    z: float
    yaw: float


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return 0.0, 0.0, math.sin(half), math.cos(half)


def wait_for_model_pose(
    node: Node,
    world_name: str,
    model_name: str,
    timeout_s: float = 3.0,
) -> ModelPose | None:
    world = str(world_name).strip()
    target = str(model_name).strip()
    if not world or not target:
        return None

    topic = f"/world/{world}/pose/info"
    poses: dict[str, ModelPose] = {}
    lock = threading.Lock()

    def _cb(msg: Pose_V) -> None:
        local: dict[str, ModelPose] = {}
        for p in msg.pose:
            q = p.orientation
            local[p.name] = ModelPose(
                name=p.name,
                x=float(p.position.x),
                y=float(p.position.y),
                z=float(p.position.z),
                yaw=yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w)),
            )
        with lock:
            poses.update(local)

    subscribed = node.subscribe(Pose_V, topic, _cb)
    if not subscribed:
        return None

    deadline = time.time() + max(0.1, float(timeout_s))
    pref = f"{target}::"
    while time.time() < deadline:
        with lock:
            exact = poses.get(target)
            if exact is not None:
                return exact

            base = poses.get(f"{target}::base_link")
            if base is not None:
                return base

            pref_hits = [v for k, v in poses.items() if k.startswith(pref)]
            if pref_hits:
                pref_hits.sort(key=lambda p: len(p.name))
                return pref_hits[0]
        time.sleep(0.02)
    return None


def set_model_pose(
    node: Node,
    world_name: str,
    model_name: str,
    x: float,
    y: float,
    z: float,
    yaw: float,
    timeout_ms: int = 1000,
) -> tuple[bool, bool]:
    req = Pose()
    req.name = str(model_name)
    req.position.x = float(x)
    req.position.y = float(y)
    req.position.z = float(z)
    qx, qy, qz, qw = quat_from_yaw(float(yaw))
    req.orientation.x = qx
    req.orientation.y = qy
    req.orientation.z = qz
    req.orientation.w = qw

    ok, rep = node.request(
        f"/world/{str(world_name).strip()}/set_pose",
        req,
        Pose,
        Boolean,
        int(timeout_ms),
    )
    return bool(ok), bool(getattr(rep, "data", False))


def create_model_from_uri(
    node: Node,
    world_name: str,
    model_name: str,
    model_uri: str,
    x: float,
    y: float,
    z: float,
    yaw: float,
    timeout_ms: int = 2000,
) -> tuple[bool, bool]:
    req = EntityFactory()
    req.sdf_filename = str(model_uri)
    req.name = str(model_name)
    req.allow_renaming = False
    req.pose.position.x = float(x)
    req.pose.position.y = float(y)
    req.pose.position.z = float(z)
    qx, qy, qz, qw = quat_from_yaw(float(yaw))
    req.pose.orientation.x = qx
    req.pose.orientation.y = qy
    req.pose.orientation.z = qz
    req.pose.orientation.w = qw

    ok, rep = node.request(
        f"/world/{str(world_name).strip()}/create",
        req,
        EntityFactory,
        Boolean,
        int(timeout_ms),
    )
    return bool(ok), bool(getattr(rep, "data", False))


def compute_pose_ahead(
    reference_pose: ModelPose,
    ahead_m: float,
    right_m: float,
    z: float,
    yaw_mode: str = "face_reference",
    fixed_yaw: float = 0.0,
) -> tuple[float, float, float, float]:
    yaw_ref = float(reference_pose.yaw)
    c = math.cos(yaw_ref)
    s = math.sin(yaw_ref)

    x = float(reference_pose.x) + float(ahead_m) * c - float(right_m) * s
    y = float(reference_pose.y) + float(ahead_m) * s + float(right_m) * c
    out_z = float(z)

    mode = str(yaw_mode).strip().lower()
    if mode == "align_reference":
        yaw = yaw_ref
    elif mode == "fixed":
        yaw = float(fixed_yaw)
    else:
        # default: actor faces the drone/camera direction for better visibility.
        yaw = yaw_ref + math.pi

    return x, y, out_z, yaw

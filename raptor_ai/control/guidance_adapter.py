from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

from raptor_ai.control.follow_controller import FollowController

_REQUIRED_OUTPUT_KEYS = {
    "active",
    "mode",
    "xy_strategy",
    "state",
    "reason",
    "vx_body",
    "vy_body",
    "vz",
    "yaw_rate",
    "area_ratio",
    "area_error",
    "center_lock_frames",
    "platform_action",
    "platform_action_payload",
}


def _extend_pythonpath(paths: list[str], *, base_dir: Path) -> None:
    for raw in paths:
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _load_callable(path: str, *, extra_pythonpath: list[str], base_dir: Path) -> Callable[..., Any]:
    if not path:
        raise ValueError("guidance.external_callable is empty")

    _extend_pythonpath(extra_pythonpath, base_dir=base_dir)

    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, attr_name = path.rsplit(".", 1)

    module = importlib.import_module(module_name)
    fn = getattr(module, attr_name)
    if not callable(fn):
        raise TypeError(f"Guidance callable is not callable: {path}")
    return fn


class GuidanceAdapter:
    """
    Runtime bridge for guidance backends.

    Supported backends:
    - legacy_internal: existing FollowController in raptor_ai/control.
    - target_guidance: external root module (target-guidance/target_guidance).
    """

    def __init__(self, *, follow_cfg: dict[str, Any], guidance_cfg: dict[str, Any] | None = None):
        self.follow_cfg = dict(follow_cfg or {})
        self.guidance_cfg = dict(guidance_cfg or {})
        self.backend = str(self.guidance_cfg.get("backend", "legacy_internal")).strip().lower()

        self.enabled = bool(self.follow_cfg.get("enabled", False))
        self.mode = str(self.follow_cfg.get("mode", "xy")).strip().lower()
        self.xy_strategy = str(self.follow_cfg.get("xy_strategy", "zone_track")).strip().lower()

        self._legacy: FollowController | None = None
        self._external_engine: Any | None = None
        self._guidance_input_type: type | None = None

        if self.backend == "legacy_internal":
            self._legacy = FollowController(self.follow_cfg)
            return

        if self.backend != "target_guidance":
            raise ValueError(f"Unsupported guidance.backend: {self.backend}")

        project_root = Path(__file__).resolve().parents[2]
        external_callable = str(
            self.guidance_cfg.get(
                "external_callable",
                "target_guidance.entrypoint:create_policy",
            )
        )
        external_pythonpath = [str(p) for p in self.guidance_cfg.get("external_pythonpath", ["target-guidance"])]
        external_context = dict(self.guidance_cfg.get("external_context", {}))

        factory = _load_callable(
            external_callable,
            extra_pythonpath=external_pythonpath,
            base_dir=project_root,
        )

        try:
            engine = factory(dict(self.follow_cfg), dict(external_context))
        except TypeError:
            engine = factory(dict(self.follow_cfg))

        if engine is None:
            raise RuntimeError(f"guidance callable returned None: {external_callable}")
        self._external_engine = engine

        contracts_mod = importlib.import_module("target_guidance.contracts")
        self._guidance_input_type = getattr(contracts_mod, "GuidanceInput")

    def _normalize_external_output(self, raw: Any, *, state: str) -> dict[str, Any]:
        if isinstance(raw, dict):
            out: dict[str, Any] = dict(raw)
        elif hasattr(raw, "to_dict") and callable(getattr(raw, "to_dict")):
            out = dict(raw.to_dict())
        else:
            out = {}
            for key in _REQUIRED_OUTPUT_KEYS:
                if hasattr(raw, key):
                    out[key] = getattr(raw, key)

        if not out:
            raise TypeError("Guidance backend returned unsupported output type")

        out.setdefault("active", False)
        out.setdefault("mode", self.mode)
        out.setdefault("xy_strategy", self.xy_strategy)
        out.setdefault("state", state)
        out.setdefault("reason", "external")
        out.setdefault("vx_body", 0.0)
        out.setdefault("vy_body", 0.0)
        out.setdefault("vz", 0.0)
        out.setdefault("yaw_rate", 0.0)
        out.setdefault("area_ratio", 0.0)
        out.setdefault("area_error", 0.0)
        out.setdefault("center_lock_frames", 0)
        out.setdefault("platform_action", "none")
        out.setdefault("platform_action_payload", {})

        return {
            "active": bool(out["active"]),
            "mode": str(out["mode"]),
            "xy_strategy": str(out["xy_strategy"]),
            "state": str(out["state"]),
            "reason": str(out["reason"]),
            "vx_body": float(out["vx_body"]),
            "vy_body": float(out["vy_body"]),
            "vz": float(out["vz"]),
            "yaw_rate": float(out["yaw_rate"]),
            "area_ratio": float(out["area_ratio"]),
            "area_error": float(out["area_error"]),
            "center_lock_frames": int(out["center_lock_frames"]),
            "platform_action": str(out["platform_action"]),
            "platform_action_payload": dict(out["platform_action_payload"] or {}),
        }

    def _compute_external(
        self,
        *,
        state: str,
        primary_track: dict[str, Any] | None,
        err_x: float | None,
        err_y: float | None,
        dt: float,
        ts: float,
        platform_meta: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self._external_engine is None or self._guidance_input_type is None:
            raise RuntimeError("guidance external backend is not initialized")

        command_input = self._guidance_input_type(
            state=str(state),
            primary_track=primary_track,
            err_x=err_x,
            err_y=err_y,
            dt=float(dt),
            ts=float(ts),
            platform_meta=(dict(platform_meta) if isinstance(platform_meta, dict) else None),
        )

        if hasattr(self._external_engine, "compute") and callable(getattr(self._external_engine, "compute")):
            raw_output = self._external_engine.compute(command_input)
        elif callable(self._external_engine):
            raw_output = self._external_engine(command_input)
        else:
            raise TypeError("guidance external engine is neither callable nor has .compute")

        return self._normalize_external_output(raw_output, state=str(state))

    def compute(
        self,
        *,
        state: str,
        primary_track: dict[str, Any] | None,
        err_x: float | None,
        err_y: float | None,
        dt: float,
        ts: float = 0.0,
        platform_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.backend == "legacy_internal":
            assert self._legacy is not None
            return self._legacy.compute(
                state=state,
                primary_track=primary_track,
                err_x=err_x,
                err_y=err_y,
                dt=dt,
            )
        return self._compute_external(
            state=state,
            primary_track=primary_track,
            err_x=err_x,
            err_y=err_y,
            dt=dt,
            ts=ts,
            platform_meta=platform_meta,
        )

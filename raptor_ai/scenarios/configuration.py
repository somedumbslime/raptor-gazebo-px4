from __future__ import annotations

from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def resolve_scenarios_config(
    scenarios_cfg: dict[str, Any],
    target_mode_override: str | None = None,
    platform_type: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """
    Resolve active scenario configuration for synthetic / actor target modes.

    Returns:
      (target_mode, active_cfg, profiles)
    where `active_cfg` is a flattened config ready for TargetMotionThread.

    Supports both:
    - new style with scenarios.target_modes.<mode>
    - legacy style with scenarios.world_name / model_name / profiles
    Also supports optional platform-aware overrides:
    - scenarios.target_modes.<mode>.platform_overrides.<platform_type>
    """
    mode = str(target_mode_override or scenarios_cfg.get("target_mode", "synthetic"))
    platform = str(platform_type or scenarios_cfg.get("platform_type", "")).strip().lower()

    target_modes = scenarios_cfg.get("target_modes")
    if isinstance(target_modes, dict) and target_modes:
        if mode not in target_modes:
            available = sorted(target_modes.keys())
            raise ValueError(f"Unknown target_mode '{mode}'. Available modes: {available}")

        mode_cfg = dict(target_modes.get(mode) or {})
        if platform:
            platform_overrides = mode_cfg.get("platform_overrides")
            if isinstance(platform_overrides, dict):
                platform_cfg = platform_overrides.get(platform)
                if isinstance(platform_cfg, dict):
                    mode_cfg = _deep_merge(mode_cfg, platform_cfg)
        mode_cfg.pop("platform_overrides", None)

        profiles = mode_cfg.get("profiles") or {}
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError(f"No profiles defined for target mode: {mode}")

        active_cfg = dict(scenarios_cfg)
        active_cfg.update(mode_cfg)
        active_cfg.pop("target_modes", None)
        active_cfg["target_mode"] = mode
        if platform:
            active_cfg["platform_type"] = platform

        if "model_name" not in active_cfg and "target_name" in active_cfg:
            active_cfg["model_name"] = active_cfg["target_name"]

        return mode, active_cfg, dict(profiles)

    profiles = scenarios_cfg.get("profiles") or {}
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("No scenarios profiles found. Expected scenarios.profiles or scenarios.target_modes")

    active_cfg = dict(scenarios_cfg)
    active_cfg["target_mode"] = mode
    return mode, active_cfg, dict(profiles)

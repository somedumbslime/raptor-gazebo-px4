from __future__ import annotations

from typing import Any

from .policy_v1 import TargetGuidancePolicyV1


def create_policy(
    follow_cfg: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> TargetGuidancePolicyV1:
    del context
    return TargetGuidancePolicyV1(dict(follow_cfg or {}))


def build_policy(
    follow_cfg: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> TargetGuidancePolicyV1:
    return create_policy(follow_cfg=follow_cfg, context=context)

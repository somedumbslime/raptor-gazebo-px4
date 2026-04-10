from __future__ import annotations

from pathlib import Path
from typing import Any

_SELECTOR_INSTANCE = None
_SELECTOR_SIGNATURE: tuple[str | None, bool, str | None] | None = None


def _resolve_path(raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def _get_selector(context: dict[str, Any]):
    global _SELECTOR_INSTANCE, _SELECTOR_SIGNATURE

    config_path = _resolve_path(context.get("pts_config_path"))
    save_events = bool(context.get("pts_save_events_jsonl", False))
    events_output_path = _resolve_path(context.get("pts_events_output_path"))
    signature = (config_path, save_events, events_output_path)

    if _SELECTOR_INSTANCE is not None and _SELECTOR_SIGNATURE == signature:
        return _SELECTOR_INSTANCE

    try:
        from pts import PrimaryTargetSelection
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import 'pts'. Check selector.external_pythonpath "
            "or install primary-target-selection package."
        ) from exc

    kwargs: dict[str, Any] = {
        "save_events_jsonl": save_events,
    }
    if config_path:
        kwargs["config_path"] = config_path
    if events_output_path:
        kwargs["events_output_path"] = events_output_path

    _SELECTOR_INSTANCE = PrimaryTargetSelection(**kwargs)
    _SELECTOR_SIGNATURE = signature
    return _SELECTOR_INSTANCE


def reset_selector() -> None:
    global _SELECTOR_INSTANCE, _SELECTOR_SIGNATURE
    _SELECTOR_INSTANCE = None
    _SELECTOR_SIGNATURE = None


def select_primary(tracks: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    selector = _get_selector(context)

    frame_size = context.get("frame_size")
    if (
        not isinstance(frame_size, (list, tuple))
        or len(frame_size) != 2
        or int(frame_size[0]) <= 0
        or int(frame_size[1]) <= 0
    ):
        raise ValueError("PTS external selector expects context['frame_size'] = (width, height)")

    frame_idx = int(context.get("frame_index", 0))
    timestamp = float(context.get("timestamp", float(frame_idx)))
    policy_name = context.get("pts_policy_name")
    external_signals = context.get("pts_external_signals")

    output = selector.update(
        tracks=tracks,
        frame_size=(int(frame_size[0]), int(frame_size[1])),
        frame_idx=frame_idx,
        timestamp_s=timestamp,
        policy_name=str(policy_name) if policy_name is not None else None,
        external_signals=dict(external_signals) if isinstance(external_signals, dict) else None,
    )

    return {
        "primary_track_id": output.primary_track_id,
        "selection_state": output.selection_state,
        "selection_reason": str(output.selection_reason),
        "primary_score": output.primary_score,
        "switch_candidate_id": output.switch_candidate_id,
        "events": [
            {
                "event_type": evt.event_type,
                "track_id": evt.track_id,
                "previous_track_id": evt.previous_track_id,
                "selection_reason": str(evt.selection_reason),
            }
            for evt in output.events
        ],
    }

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

from raptor_ai.tracking.track_types import SelectionResult, TrackInfo


def _extend_pythonpath(paths: list[str]) -> None:
    for raw in paths:
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _load_callable(path: str, extra_pythonpath: list[str] | None = None) -> Callable[..., Any]:
    if not path:
        raise ValueError("selector.external_callable is empty")

    if extra_pythonpath:
        _extend_pythonpath(extra_pythonpath)

    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, attr_name = path.rsplit(".", 1)

    module = importlib.import_module(module_name)
    fn = getattr(module, attr_name)
    if not callable(fn):
        raise TypeError(f"External selector is not callable: {path}")
    return fn


class PrimaryTargetSelectorAdapter:
    def __init__(self, cfg: dict[str, Any]):
        self.backend = str(cfg.get("backend", "stub"))
        self.stub_policy = str(cfg.get("stub_policy", "max_area"))
        self.required_fields = list(cfg.get("required_fields", []))
        self.field_map = dict(cfg.get("field_map", {}))
        self._external_callable_path = str(cfg.get("external_callable", ""))
        self.external_pythonpath = [str(x) for x in cfg.get("external_pythonpath", [])]
        self.external_context = dict(cfg.get("external_context", {}))
        self._external_callable: Callable[..., Any] | None = None

        if self.backend == "external":
            self._external_callable = _load_callable(
                self._external_callable_path,
                extra_pythonpath=self.external_pythonpath,
            )

    def _build_payload(self, track: TrackInfo) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field in self.required_fields:
            src_key = self.field_map.get(field, field)
            payload[field] = track.get(src_key)
        return payload

    def _augment_tracks(self, tracks: list[TrackInfo]) -> list[TrackInfo]:
        out: list[TrackInfo] = []
        for tr in tracks:
            item = dict(tr)
            item["selector_payload"] = self._build_payload(item)
            out.append(item)
        return out

    def _stub_select(self, tracks: list[TrackInfo]) -> tuple[TrackInfo | None, str]:
        if not tracks:
            return None, "no_tracks"

        if self.stub_policy == "max_area":
            best = max(tracks, key=lambda t: float(t.get("area", 0.0)))
            return best, "stub:max_area"

        best = tracks[0]
        return best, f"stub:{self.stub_policy}_fallback_first"

    @staticmethod
    def _find_track_by_id(tracks: list[TrackInfo], track_id: int) -> TrackInfo | None:
        for tr in tracks:
            if int(tr.get("track_id", -1)) == int(track_id):
                return tr
        return None

    @staticmethod
    def _normalize_external_events(raw_events: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not raw_events:
            return out
        for evt in raw_events:
            if isinstance(evt, dict):
                out.append(dict(evt))
                continue
            out.append(
                {
                    "event_type": getattr(evt, "event_type", "unknown"),
                    "track_id": getattr(evt, "track_id", None),
                    "previous_track_id": getattr(evt, "previous_track_id", None),
                    "selection_reason": getattr(evt, "selection_reason", None),
                }
            )
        return out

    def _resolve_external_result(
        self,
        external_result: Any,
        tracks: list[TrackInfo],
    ) -> tuple[TrackInfo | None, dict[str, Any]]:
        if external_result is None:
            return None, {"reason": "external:none", "selection_state": "none", "events": []}

        if isinstance(external_result, int):
            selected = self._find_track_by_id(tracks, external_result)
            if selected is not None:
                return selected, {"reason": "external:track_id", "selection_state": "selected", "events": []}
            return None, {"reason": "external:track_id_not_found", "selection_state": "none", "events": []}

        if isinstance(external_result, dict):
            reason = str(external_result.get("selection_reason", "external:dict"))
            state = str(external_result.get("selection_state", "unknown"))
            events = self._normalize_external_events(external_result.get("events", []))

            ext_id = None
            for key in ("primary_track_id", "selected_track_id", "track_id"):
                if external_result.get(key) is not None:
                    ext_id = int(external_result[key])
                    break

            if ext_id is not None:
                selected = self._find_track_by_id(tracks, ext_id)
                if selected is not None:
                    return selected, {"reason": reason, "selection_state": state, "events": events}
                return None, {"reason": "external:dict_track_id_not_found", "selection_state": state, "events": events}

            if "track_id" in external_result:
                ext_id = int(external_result["track_id"])
                selected = self._find_track_by_id(tracks, ext_id)
                if selected is not None:
                    return selected, {"reason": reason, "selection_state": state, "events": events}
                return None, {"reason": "external:dict_track_id_not_found", "selection_state": state, "events": events}

            if all(k in external_result for k in ("bbox_xyxy", "center")):
                return external_result, {"reason": reason, "selection_state": state, "events": events}

        if hasattr(external_result, "primary_track_id"):
            ext_id = getattr(external_result, "primary_track_id", None)
            state = str(getattr(external_result, "selection_state", "unknown"))
            reason = str(getattr(external_result, "selection_reason", "external:selection_output"))
            events = self._normalize_external_events(getattr(external_result, "events", []))
            if ext_id is None:
                return None, {"reason": reason, "selection_state": state, "events": events}
            selected = self._find_track_by_id(tracks, int(ext_id))
            if selected is not None:
                return selected, {"reason": reason, "selection_state": state, "events": events}
            return None, {"reason": "external:primary_track_id_not_found", "selection_state": state, "events": events}

        return None, {"reason": "external:unsupported_result", "selection_state": "none", "events": []}

    def select_primary(self, tracks: list[TrackInfo], context: dict[str, Any]) -> SelectionResult:
        augmented = self._augment_tracks(tracks)

        selected: TrackInfo | None
        reason = "unknown"
        selection_state = "none"
        selector_events: list[dict[str, Any]] = []

        if self.backend == "stub":
            selected, reason = self._stub_select(augmented)
            selection_state = "selected" if selected is not None else "none"
        elif self.backend == "external":
            assert self._external_callable is not None
            merged_context = dict(self.external_context)
            merged_context.update(context)
            payload_tracks = [t.get("selector_payload", {}) for t in augmented]
            result = self._external_callable(payload_tracks, merged_context)
            selected, resolved = self._resolve_external_result(result, augmented)
            reason = str(resolved.get("reason", "external:unknown"))
            selection_state = str(resolved.get("selection_state", "unknown"))
            selector_events = list(resolved.get("events", []))
        else:
            selected, reason = None, f"unknown_backend:{self.backend}"
            selection_state = "none"

        output: SelectionResult = {
            "selected_primary_target": selected,
            "selection_state": selection_state,
            "selection_reason": reason,
        }
        if selector_events:
            output["selector_events"] = selector_events
        return output

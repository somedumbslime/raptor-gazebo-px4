#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raptor_ai.config.loader import load_config, resolve_camera_topic
from raptor_ai.scenarios.configuration import resolve_scenarios_config

DEFAULT_REQUIRED_SCENARIOS = ["slow_circle", "temporary_disappearance"]
REQUIRED_ARTIFACTS = [
    "metrics_summary.json",
    "events.jsonl",
    "run_meta.json",
    "scenario_meta.json",
    "motion_trace.jsonl",
]
REQUIRED_RUNTIME_EVENTS = {"runtime_started", "runtime_stopped"}


@dataclass
class MotionStats:
    samples: int = 0
    span_x: float = 0.0
    span_y: float = 0.0
    span_xy: float = 0.0
    set_pose_success_ratio: float = 0.0
    issues: list[str] | None = None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            txt = line.strip()
            if not txt:
                continue
            try:
                obj = json.loads(txt)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{idx}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_and_merge_summaries(paths: list[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = read_json(path)
        for scenario, metrics in data.items():
            if isinstance(metrics, dict):
                merged[str(scenario)] = metrics
    return merged


def load_expected_contract(config_path: Path, platform_type: str) -> dict[str, Any]:
    cfg = load_config(str(config_path))
    scenarios_cfg = cfg.get("scenarios", {})
    _, active_cfg, _ = resolve_scenarios_config(
        scenarios_cfg=scenarios_cfg,
        target_mode_override="actor",
        platform_type=platform_type,
    )

    world_name = str(active_cfg.get("world_name", "raptor_mvp_iris_actor"))
    model_name = str(active_cfg.get("model_name", active_cfg.get("target_name", "target_actor")))
    camera_topic = str(resolve_camera_topic(cfg, platform_type=platform_type))

    platform_cfg = cfg.get("platform", {}).get(platform_type, {})
    cmd_twist_topic = ""
    if isinstance(platform_cfg, dict):
        cmd_twist_topic = str(platform_cfg.get("cmd_twist_topic", "")).strip()

    return {
        "platform_type": platform_type,
        "target_mode": "actor",
        "world_name": world_name,
        "model_name": model_name,
        "camera_topic": camera_topic,
        "cmd_twist_topic": cmd_twist_topic,
        "config_snapshot": {
            "platform": cfg.get("platform", {}),
            "scenarios": {
                "target_mode": scenarios_cfg.get("target_mode"),
                "target_modes": scenarios_cfg.get("target_modes", {}),
            },
        },
    }


def analyze_motion_trace(path: Path, min_samples: int, min_span: float) -> MotionStats:
    issues: list[str] = []
    if not path.exists():
        return MotionStats(issues=[f"missing artifact: {path.name}"])

    rows = read_jsonl(path)
    samples = len(rows)
    if samples < min_samples:
        issues.append(f"motion_trace has too few samples: {samples} < {min_samples}")

    xs = [float(r.get("x", 0.0)) for r in rows]
    ys = [float(r.get("y", 0.0)) for r in rows]
    span_x = (max(xs) - min(xs)) if xs else 0.0
    span_y = (max(ys) - min(ys)) if ys else 0.0
    span_xy = max(span_x, span_y)
    if span_xy < min_span:
        issues.append(f"movement span too small: {span_xy:.4f} < {min_span:.4f}")

    success = 0
    for row in rows:
        req_ok = bool(row.get("set_pose_ok", False))
        rep_ok = bool(row.get("set_pose_response", False))
        if req_ok and rep_ok:
            success += 1

    success_ratio = (float(success) / float(samples)) if samples > 0 else 0.0
    if success_ratio <= 0.0:
        issues.append("set_pose success ratio is 0.0")

    return MotionStats(
        samples=samples,
        span_x=span_x,
        span_y=span_y,
        span_xy=span_xy,
        set_pose_success_ratio=success_ratio,
        issues=issues,
    )


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "-"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Phase 4 (Iris platform layer) with closure report")
    parser.add_argument(
        "--iris-suites",
        nargs="+",
        required=True,
        help="One or more suite_summary JSON files produced in platform=iris mode",
    )
    parser.add_argument(
        "--config",
        default="raptor_ai/config/default_config.yaml",
        help="Runtime config path",
    )
    parser.add_argument(
        "--required-scenarios",
        default="slow_circle,temporary_disappearance",
        help="Comma separated scenario names required for closure",
    )
    parser.add_argument(
        "--platform-type",
        default="iris",
        help="Expected platform type in run/scenario metadata",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/phase4_closure",
        help="Output directory for closure artifacts",
    )
    parser.add_argument(
        "--min-motion-samples",
        type=int,
        default=20,
        help="Minimum motion trace sample count per scenario",
    )
    parser.add_argument(
        "--min-motion-span",
        type=float,
        default=0.20,
        help="Minimum movement span in x/y (meters) per scenario",
    )
    args = parser.parse_args()

    suite_paths = [Path(p).resolve() for p in args.iris_suites]
    for path in suite_paths:
        if not path.exists():
            raise FileNotFoundError(f"Suite summary not found: {path}")

    required_scenarios = [s.strip() for s in str(args.required_scenarios).split(",") if s.strip()]
    if not required_scenarios:
        raise ValueError("required_scenarios is empty")

    platform_type = str(args.platform_type).strip().lower() or "iris"
    config_path = Path(args.config).resolve()
    expected = load_expected_contract(config_path=config_path, platform_type=platform_type)

    merged = load_and_merge_summaries(suite_paths)
    missing_scenarios = [s for s in required_scenarios if s not in merged]

    artifact_issues: list[str] = []
    runtime_issues: list[str] = []
    contract_issues: list[str] = []
    motion_issues: list[str] = []

    scenario_dirs: dict[str, Path] = {}
    scenario_motion: dict[str, MotionStats] = {}
    scenario_run_meta: dict[str, dict[str, Any]] = {}
    scenario_scenario_meta: dict[str, dict[str, Any]] = {}

    for scenario in required_scenarios:
        metrics = merged.get(scenario)
        if metrics is None:
            continue

        out_dir_raw = metrics.get("output_dir")
        if not isinstance(out_dir_raw, str) or not out_dir_raw.strip():
            artifact_issues.append(f"{scenario}: metrics.output_dir missing")
            continue

        out_dir = Path(out_dir_raw)
        if not out_dir.is_absolute():
            out_dir = (Path.cwd() / out_dir).resolve()
        scenario_dirs[scenario] = out_dir

        if not out_dir.exists():
            artifact_issues.append(f"{scenario}: output_dir does not exist: {out_dir}")
            continue

        for rel in REQUIRED_ARTIFACTS:
            p = out_dir / rel
            if not p.exists():
                artifact_issues.append(f"{scenario}: missing artifact {rel}")

        frames_total = int(metrics.get("frames_total", 0))
        duration_s = float(metrics.get("duration_s", 0.0))
        if frames_total <= 0:
            runtime_issues.append(f"{scenario}: frames_total <= 0")
        if duration_s <= 0.0:
            runtime_issues.append(f"{scenario}: duration_s <= 0")

        events_path = out_dir / "events.jsonl"
        if events_path.exists():
            events = read_jsonl(events_path)
            names = {str(evt.get("event")) for evt in events}
            missing_evt = sorted(REQUIRED_RUNTIME_EVENTS - names)
            if missing_evt:
                runtime_issues.append(f"{scenario}: missing runtime events {missing_evt}")

        run_meta_path = out_dir / "run_meta.json"
        if run_meta_path.exists():
            run_meta = read_json(run_meta_path)
            scenario_run_meta[scenario] = run_meta
            if run_meta.get("platform_type") != expected["platform_type"]:
                contract_issues.append(
                    f"{scenario}: run_meta.platform_type mismatch "
                    f"(expected {expected['platform_type']}, got {run_meta.get('platform_type')})"
                )
            if run_meta.get("camera_topic") != expected["camera_topic"]:
                contract_issues.append(
                    f"{scenario}: run_meta.camera_topic mismatch "
                    f"(expected {expected['camera_topic']}, got {run_meta.get('camera_topic')})"
                )
            if expected["cmd_twist_topic"]:
                cmd_twist = (run_meta.get("platform_meta") or {}).get("cmd_twist_topic")
                if cmd_twist != expected["cmd_twist_topic"]:
                    contract_issues.append(
                        f"{scenario}: run_meta.platform_meta.cmd_twist_topic mismatch "
                        f"(expected {expected['cmd_twist_topic']}, got {cmd_twist})"
                    )

        scenario_meta_path = out_dir / "scenario_meta.json"
        if scenario_meta_path.exists():
            scenario_meta = read_json(scenario_meta_path)
            scenario_scenario_meta[scenario] = scenario_meta
            if scenario_meta.get("target_mode") != expected["target_mode"]:
                contract_issues.append(
                    f"{scenario}: scenario_meta.target_mode mismatch "
                    f"(expected {expected['target_mode']}, got {scenario_meta.get('target_mode')})"
                )
            if scenario_meta.get("platform_type") != expected["platform_type"]:
                contract_issues.append(
                    f"{scenario}: scenario_meta.platform_type mismatch "
                    f"(expected {expected['platform_type']}, got {scenario_meta.get('platform_type')})"
                )
            if scenario_meta.get("world_name") != expected["world_name"]:
                contract_issues.append(
                    f"{scenario}: scenario_meta.world_name mismatch "
                    f"(expected {expected['world_name']}, got {scenario_meta.get('world_name')})"
                )
            if scenario_meta.get("model_name") != expected["model_name"]:
                contract_issues.append(
                    f"{scenario}: scenario_meta.model_name mismatch "
                    f"(expected {expected['model_name']}, got {scenario_meta.get('model_name')})"
                )
            if scenario_meta.get("camera_topic") != expected["camera_topic"]:
                contract_issues.append(
                    f"{scenario}: scenario_meta.camera_topic mismatch "
                    f"(expected {expected['camera_topic']}, got {scenario_meta.get('camera_topic')})"
                )

        motion_path = out_dir / "motion_trace.jsonl"
        stats = analyze_motion_trace(
            path=motion_path,
            min_samples=max(1, int(args.min_motion_samples)),
            min_span=max(0.0, float(args.min_motion_span)),
        )
        scenario_motion[scenario] = stats
        for issue in (stats.issues or []):
            motion_issues.append(f"{scenario}: {issue}")

    scenario_coverage_ok = len(missing_scenarios) == 0
    contract_ok = len(contract_issues) == 0
    motion_ok = len(motion_issues) == 0
    runtime_ok = len(runtime_issues) == 0
    artifact_ok = len(artifact_issues) == 0

    dod_1 = scenario_coverage_ok and contract_ok and motion_ok
    dod_2 = runtime_ok and artifact_ok
    phase4_complete = dod_1 and dod_2

    known_limits: list[str] = []
    if any(float((merged.get(s) or {}).get("frames_with_detection", 0.0)) <= 0.0 for s in required_scenarios if s in merged):
        known_limits.append(
            "frames_with_detection is zero in one or more iris smoke scenarios "
            "(expected if detector does not cover actor target in current setup)."
        )
    control_modes = {
        str((scenario_run_meta.get(s, {}).get("platform_meta") or {}).get("control_mode"))
        for s in required_scenarios
        if s in scenario_run_meta
    }
    if "hold" in control_modes:
        known_limits.append("Iris platform is in safe `hold` control_mode; full follow behavior is deferred to Phase 5.")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "phase4_closure_report.md"
    summary_path = out_dir / "phase4_closure_summary.json"

    lines: list[str] = []
    lines.append("# Phase 4 Closure Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Config: `{config_path}`")
    lines.append(
        "Expected contract: "
        f"platform=`{expected['platform_type']}`, "
        f"target_mode=`{expected['target_mode']}`, "
        f"world=`{expected['world_name']}`, "
        f"model=`{expected['model_name']}`, "
        f"camera_topic=`{expected['camera_topic']}`"
    )
    if expected["cmd_twist_topic"]:
        lines.append(f"Expected cmd_twist_topic: `{expected['cmd_twist_topic']}`")
    lines.append("Suite sources:")
    for p in suite_paths:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## DoD Check")
    lines.append("")
    lines.append(f"- DoD 1 (iris smoke scenarios + platform contract + motion trace): **{'PASS' if dod_1 else 'FAIL'}**")
    lines.append(f"- DoD 2 (runtime + logging artifacts valid on iris runs): **{'PASS' if dod_2 else 'FAIL'}**")
    lines.append(f"- Phase 4 final status: **{'COMPLETED' if phase4_complete else 'NOT COMPLETED'}**")
    lines.append("")

    if missing_scenarios:
        lines.append("Missing scenarios:")
        for item in missing_scenarios:
            lines.append(f"- {item}")
        lines.append("")

    if artifact_issues:
        lines.append("Artifact issues:")
        for item in artifact_issues:
            lines.append(f"- {item}")
        lines.append("")

    if runtime_issues:
        lines.append("Runtime/logging issues:")
        for item in runtime_issues:
            lines.append(f"- {item}")
        lines.append("")

    if contract_issues:
        lines.append("Platform/scenario contract issues:")
        for item in contract_issues:
            lines.append(f"- {item}")
        lines.append("")

    if motion_issues:
        lines.append("Motion trace issues:")
        for item in motion_issues:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Iris Smoke Metrics Table")
    lines.append("")
    cols = [
        "frames_total",
        "frames_with_detection",
        "frames_with_primary",
        "presence_ratio",
        "search_frames",
        "locked_frames",
        "motion_samples",
        "motion_span_xy",
        "set_pose_success_ratio",
    ]
    lines.append("| scenario | " + " | ".join(cols) + " |")
    lines.append("|---|" + "|".join(["---"] * len(cols)) + "|")
    for scenario in required_scenarios:
        row = merged.get(scenario, {})
        motion = scenario_motion.get(scenario, MotionStats())
        values: list[str] = []
        for col in cols:
            if col == "motion_samples":
                values.append(str(motion.samples))
            elif col == "motion_span_xy":
                values.append(f"{motion.span_xy:.4f}")
            elif col == "set_pose_success_ratio":
                values.append(f"{motion.set_pose_success_ratio:.6f}")
            else:
                values.append(fmt_value(row.get(col, "-")))
        lines.append("| " + scenario + " | " + " | ".join(values) + " |")

    lines.append("")
    lines.append("## Known Limits")
    lines.append("")
    if known_limits:
        for item in known_limits:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Config Snapshot")
    lines.append("")
    lines.append("```yaml")
    lines.append(yaml.safe_dump(expected["config_snapshot"], sort_keys=False).rstrip())
    lines.append("```")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    closure_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase4_complete": phase4_complete,
        "dod_1_iris_smoke_contract_motion": dod_1,
        "dod_2_runtime_logging_artifacts": dod_2,
        "required_scenarios": required_scenarios,
        "expected_contract": {
            "platform_type": expected["platform_type"],
            "target_mode": expected["target_mode"],
            "world_name": expected["world_name"],
            "model_name": expected["model_name"],
            "camera_topic": expected["camera_topic"],
            "cmd_twist_topic": expected["cmd_twist_topic"],
        },
        "missing_scenarios": missing_scenarios,
        "known_limits": known_limits,
        "issues": {
            "artifacts": artifact_issues,
            "runtime": runtime_issues,
            "contract": contract_issues,
            "motion": motion_issues,
        },
        "scenarios": {
            scenario: {
                "output_dir": str(scenario_dirs.get(scenario, "")),
                "metrics": merged.get(scenario, {}),
                "run_meta": scenario_run_meta.get(scenario, {}),
                "scenario_meta": scenario_scenario_meta.get(scenario, {}),
                "motion": {
                    "samples": scenario_motion.get(scenario, MotionStats()).samples,
                    "span_xy": scenario_motion.get(scenario, MotionStats()).span_xy,
                    "set_pose_success_ratio": scenario_motion.get(scenario, MotionStats()).set_pose_success_ratio,
                    "issues": scenario_motion.get(scenario, MotionStats()).issues or [],
                },
            }
            for scenario in required_scenarios
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(closure_summary, f, indent=2, ensure_ascii=True)

    print(report_path)
    print(summary_path)
    if not phase4_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

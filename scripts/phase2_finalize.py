#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SCENARIOS = [
    "slow_circle",
    "fast_circle",
    "zigzag",
    "temporary_disappearance",
    "long_absence",
]

REQUIRED_ARTIFACTS = [
    "metrics_summary.json",
    "events.jsonl",
    "run_meta.json",
    "scenario_meta.json",
    "motion_trace.jsonl",
]

REQUIRED_RUNTIME_EVENTS = {
    "runtime_started",
    "runtime_stopped",
}


@dataclass
class MotionStats:
    samples: int = 0
    span_x: float = 0.0
    span_y: float = 0.0
    span_xy: float = 0.0
    set_pose_success_ratio: float = 0.0
    issues: list[str] | None = None


def load_and_merge_summaries(paths: list[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Suite summary must be JSON object: {path}")
        for scenario, metrics in data.items():
            if isinstance(metrics, dict):
                merged[str(scenario)] = metrics
    return merged


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


def load_expected_actor_contract(config_path: Path) -> tuple[str, str, dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    scenarios = cfg.get("scenarios", {}) if isinstance(cfg, dict) else {}
    target_modes = scenarios.get("target_modes", {}) if isinstance(scenarios, dict) else {}
    actor_cfg = target_modes.get("actor", {}) if isinstance(target_modes, dict) else {}
    if not isinstance(actor_cfg, dict):
        actor_cfg = {}

    expected_world = str(actor_cfg.get("world_name", "raptor_mvp_actor"))
    expected_model = str(actor_cfg.get("model_name", actor_cfg.get("target_name", "target_actor")))
    snapshot = {
        "scenarios_target_mode_default": scenarios.get("target_mode"),
        "actor_mode": actor_cfg,
    }
    return expected_world, expected_model, snapshot


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Phase 2 (actor target scenes) with closure report")
    parser.add_argument(
        "--actor-suites",
        nargs="+",
        required=True,
        help="One or more suite_summary JSON files produced with --target-mode actor",
    )
    parser.add_argument(
        "--config",
        default="raptor_ai/config/default_config.yaml",
        help="Runtime config path",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for closure artifacts (default: alongside first suite)",
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

    suite_paths = [Path(p).resolve() for p in args.actor_suites]
    for path in suite_paths:
        if not path.exists():
            raise FileNotFoundError(f"Suite summary not found: {path}")

    config_path = Path(args.config).resolve()
    expected_world, expected_model, actor_cfg_snapshot = load_expected_actor_contract(config_path)

    merged = load_and_merge_summaries(suite_paths)
    missing_scenarios = [s for s in REQUIRED_SCENARIOS if s not in merged]

    artifact_issues: list[str] = []
    runtime_issues: list[str] = []
    actor_contract_issues: list[str] = []
    motion_issues: list[str] = []

    scenario_motion: dict[str, MotionStats] = {}
    scenario_dirs: dict[str, Path] = {}

    for scenario in REQUIRED_SCENARIOS:
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
            if len(events) < 2:
                runtime_issues.append(f"{scenario}: events.jsonl has < 2 rows")
            names = {str(evt.get("event")) for evt in events}
            missing_evt = sorted(REQUIRED_RUNTIME_EVENTS - names)
            if missing_evt:
                runtime_issues.append(f"{scenario}: missing required runtime events: {missing_evt}")

        scenario_meta_path = out_dir / "scenario_meta.json"
        if scenario_meta_path.exists():
            with scenario_meta_path.open("r", encoding="utf-8") as f:
                scenario_meta = json.load(f)
            mode = scenario_meta.get("target_mode")
            world = scenario_meta.get("world_name")
            model = scenario_meta.get("model_name")
            if mode != "actor":
                actor_contract_issues.append(f"{scenario}: target_mode is not actor (got: {mode})")
            if world != expected_world:
                actor_contract_issues.append(f"{scenario}: world_name mismatch (expected {expected_world}, got {world})")
            if model != expected_model:
                actor_contract_issues.append(f"{scenario}: model_name mismatch (expected {expected_model}, got {model})")

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
    actor_contract_ok = len(actor_contract_issues) == 0
    motion_ok = len(motion_issues) == 0
    runtime_ok = len(runtime_issues) == 0
    artifact_ok = len(artifact_issues) == 0

    dod_1 = scenario_coverage_ok and actor_contract_ok and motion_ok
    dod_2 = runtime_ok and artifact_ok
    phase2_complete = dod_1 and dod_2

    out_dir = Path(args.out_dir).resolve() if args.out_dir else suite_paths[0].parent.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "phase2_closure_report.md"
    summary_path = out_dir / "phase2_closure_summary.json"

    lines: list[str] = []
    lines.append("# Phase 2 Closure Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Config: `{config_path}`")
    lines.append(f"Expected actor contract: world=`{expected_world}`, model=`{expected_model}`")
    lines.append("Suite sources:")
    for path in suite_paths:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## DoD Check")
    lines.append("")
    lines.append(f"- DoD 1 (actor scenes A-E + actor contract + movement trace): **{'PASS' if dod_1 else 'FAIL'}**")
    lines.append(f"- DoD 2 (runtime + logging artifacts valid on actor scenes): **{'PASS' if dod_2 else 'FAIL'}**")
    lines.append(f"- Phase 2 final status: **{'COMPLETED' if phase2_complete else 'NOT COMPLETED'}**")
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

    if actor_contract_issues:
        lines.append("Actor contract issues:")
        for item in actor_contract_issues:
            lines.append(f"- {item}")
        lines.append("")

    if motion_issues:
        lines.append("Motion trace issues:")
        for item in motion_issues:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Actor Metrics + Motion Table")
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

    for scenario in REQUIRED_SCENARIOS:
        row = merged.get(scenario, {})
        stats = scenario_motion.get(scenario, MotionStats())
        values: list[str] = []
        for col in cols:
            if col == "motion_samples":
                values.append(str(stats.samples))
            elif col == "motion_span_xy":
                values.append(f"{stats.span_xy:.4f}")
            elif col == "set_pose_success_ratio":
                values.append(f"{stats.set_pose_success_ratio:.6f}")
            else:
                value = row.get(col, "-")
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append(str(value))
        lines.append("| " + scenario + " | " + " | ".join(values) + " |")

    lines.append("")
    lines.append("## Actor Config Snapshot")
    lines.append("")
    lines.append("```yaml")
    lines.append(yaml.safe_dump({"scenarios": actor_cfg_snapshot}, sort_keys=False).rstrip())
    lines.append("```")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    closure_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase2_complete": phase2_complete,
        "dod_1_actor_scenes_and_motion": dod_1,
        "dod_2_runtime_and_logging": dod_2,
        "expected_world": expected_world,
        "expected_model": expected_model,
        "missing_scenarios": missing_scenarios,
        "issues": {
            "artifacts": artifact_issues,
            "runtime": runtime_issues,
            "actor_contract": actor_contract_issues,
            "motion": motion_issues,
        },
        "scenarios": {
            scenario: {
                "output_dir": str(scenario_dirs.get(scenario, "")),
                "metrics": merged.get(scenario, {}),
                "motion": {
                    "samples": scenario_motion.get(scenario, MotionStats()).samples,
                    "span_xy": scenario_motion.get(scenario, MotionStats()).span_xy,
                    "set_pose_success_ratio": scenario_motion.get(scenario, MotionStats()).set_pose_success_ratio,
                    "issues": scenario_motion.get(scenario, MotionStats()).issues or [],
                },
            }
            for scenario in REQUIRED_SCENARIOS
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(closure_summary, f, indent=2, ensure_ascii=True)

    print(report_path)
    print(summary_path)

    if not phase2_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

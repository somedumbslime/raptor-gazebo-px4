#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

REQUIRED_PTS_FIELDS = {
    "track_id",
    "bbox_xyxy",
    "confidence",
    "class_id",
    "class_name",
    "visible",
}


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


def check_selector_contract(config_path: Path) -> tuple[bool, list[str], dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    selector = cfg.get("selector", {})
    issues: list[str] = []

    if selector.get("backend") != "external":
        issues.append("selector.backend is not 'external'")

    required_fields = set(selector.get("required_fields", []))
    if not REQUIRED_PTS_FIELDS.issubset(required_fields):
        missing = sorted(REQUIRED_PTS_FIELDS - required_fields)
        issues.append(f"selector.required_fields missing: {missing}")

    field_map = selector.get("field_map", {})
    expected_map = {
        "confidence": "conf",
        "class_id": "cls_id",
        "class_name": "cls_name",
    }
    for key, expected in expected_map.items():
        if field_map.get(key) != expected:
            issues.append(f"selector.field_map['{key}'] expected '{expected}', got '{field_map.get(key)}'")

    if not selector.get("external_callable"):
        issues.append("selector.external_callable is empty")

    if not selector.get("external_pythonpath"):
        issues.append("selector.external_pythonpath is empty")

    if "stub_policy" not in selector:
        issues.append("selector.stub_policy missing (fallback not explicit)")

    return (len(issues) == 0), issues, selector


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Phase 1 (PTS integration) with a closure report")
    parser.add_argument(
        "--external-suites",
        nargs="+",
        required=True,
        help="One or more suite_summary JSON files produced with selector.backend=external",
    )
    parser.add_argument(
        "--config",
        default="raptor_ai/config/default_config.yaml",
        help="Runtime config path",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for closure report (default: alongside first suite)",
    )
    args = parser.parse_args()

    suite_paths = [Path(p).resolve() for p in args.external_suites]
    for p in suite_paths:
        if not p.exists():
            raise FileNotFoundError(f"Suite summary not found: {p}")

    merged = load_and_merge_summaries(suite_paths)
    cfg_ok, cfg_issues, selector_cfg = check_selector_contract(Path(args.config).resolve())

    missing_scenarios = [s for s in REQUIRED_SCENARIOS if s not in merged]
    scenario_ok = len(missing_scenarios) == 0

    runtime_issues: list[str] = []
    for s in REQUIRED_SCENARIOS:
        m = merged.get(s)
        if m is None:
            continue
        if int(m.get("frames_total", 0)) <= 0:
            runtime_issues.append(f"{s}: frames_total <= 0")
        if int(m.get("frames_with_detection", 0)) <= 0:
            runtime_issues.append(f"{s}: frames_with_detection <= 0")

    runtime_ok = len(runtime_issues) == 0

    dod_1 = scenario_ok and runtime_ok
    dod_2 = cfg_ok
    phase1_complete = dod_1 and dod_2

    out_dir = Path(args.out_dir).resolve() if args.out_dir else suite_paths[0].parent.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "phase1_closure_report.md"

    lines: list[str] = []
    lines.append("# Phase 1 Closure Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Config: `{Path(args.config).resolve()}`")
    lines.append("Suite sources:")
    for p in suite_paths:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## DoD Check")
    lines.append("")
    lines.append(f"- DoD 1 (runtime with real PTS on A-E): **{'PASS' if dod_1 else 'FAIL'}**")
    lines.append(f"- DoD 2 (PTS input only from config contract): **{'PASS' if dod_2 else 'FAIL'}**")
    lines.append(f"- Phase 1 final status: **{'COMPLETED' if phase1_complete else 'NOT COMPLETED'}**")
    lines.append("")

    if missing_scenarios:
        lines.append("Missing scenarios:")
        for s in missing_scenarios:
            lines.append(f"- {s}")
        lines.append("")

    if runtime_issues:
        lines.append("Runtime issues:")
        for issue in runtime_issues:
            lines.append(f"- {issue}")
        lines.append("")

    if cfg_issues:
        lines.append("Config/contract issues:")
        for issue in cfg_issues:
            lines.append(f"- {issue}")
        lines.append("")

    lines.append("## External Metrics Table")
    lines.append("")
    cols = [
        "frames_total",
        "frames_with_detection",
        "frames_with_primary",
        "presence_ratio",
        "lost_target_count",
        "reacquire_count",
        "search_frames",
        "locked_frames",
    ]
    lines.append("| scenario | " + " | ".join(cols) + " |")
    lines.append("|---|" + "|".join(["---"] * len(cols)) + "|")

    for s in REQUIRED_SCENARIOS:
        row = merged.get(s, {})
        vals: list[str] = []
        for c in cols:
            v = row.get(c, "-")
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + s + " | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Selector Config Snapshot")
    lines.append("")
    lines.append("```yaml")
    lines.append(yaml.safe_dump({"selector": selector_cfg}, sort_keys=False).rstrip())
    lines.append("```")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)

    if not phase1_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

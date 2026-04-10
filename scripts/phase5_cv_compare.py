#!/usr/bin/env /usr/bin/python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

SCENARIOS_DEFAULT = [
    "slow_circle",
    "fast_circle",
    "zigzag",
    "temporary_disappearance",
    "long_absence",
]

METRICS = [
    "frames_total",
    "frames_with_detection",
    "frames_with_primary",
    "presence_ratio",
    "lost_target_count",
    "reacquire_count",
    "search_frames",
    "locked_frames",
]


def _parse_list(raw: str | None, default: list[str]) -> list[str]:
    if raw is None or str(raw).strip().lower() in ("", "all"):
        return list(default)
    out = [s.strip() for s in str(raw).split(",")]
    return [s for s in out if s]


def _load_summary(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid suite summary: {path}")
    out: dict[str, dict[str, Any]] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[str(k)] = dict(v)
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, int):
        return str(v)
    if v is None:
        return "-"
    return str(v)


def _load_run_meta(summary_row: dict[str, Any]) -> dict[str, Any]:
    out_dir = str(summary_row.get("output_dir", "")).strip()
    if not out_dir:
        return {}
    run_meta_path = Path(out_dir) / "run_meta.json"
    if not run_meta_path.exists():
        return {}
    try:
        data = json.loads(run_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase-5 CV suites (e.g. red+stub vs yolo+pts)")
    parser.add_argument("--baseline-suite", required=True, help="suite_summary JSON for baseline run")
    parser.add_argument("--candidate-suite", required=True, help="suite_summary JSON for candidate run")
    parser.add_argument("--baseline-label", default="red_stub", help="Label for baseline columns")
    parser.add_argument("--candidate-label", default="yolo_pts", help="Label for candidate columns")
    parser.add_argument("--scenarios", default="all", help="Comma-separated scenarios or 'all'")
    parser.add_argument("--out-dir", default="runs/phase5_cv_compare", help="Output folder")
    parser.add_argument("--min-detections", type=int, default=1, help="DoD check: min frames_with_detection")
    parser.add_argument("--min-primary", type=int, default=1, help="DoD check: min frames_with_primary")
    parser.add_argument("--allow-fail", action="store_true", help="Do not exit non-zero on failed DoD checks")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_suite).resolve()
    candidate_path = Path(args.candidate_suite).resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Not found: {baseline_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"Not found: {candidate_path}")

    baseline = _load_summary(baseline_path)
    candidate = _load_summary(candidate_path)
    scenarios = _parse_list(args.scenarios, SCENARIOS_DEFAULT)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"phase5_cv_compare_{stamp}.md"
    json_path = out_dir / f"phase5_cv_compare_{stamp}.json"

    lines: list[str] = []
    lines.append("# Phase 5 CV Compare")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Baseline: `{baseline_path}` ({args.baseline_label})")
    lines.append(f"Candidate: `{candidate_path}` ({args.candidate_label})")
    lines.append("")

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_suite": str(baseline_path),
        "candidate_suite": str(candidate_path),
        "baseline_label": str(args.baseline_label),
        "candidate_label": str(args.candidate_label),
        "scenarios": {},
        "checks": {},
    }

    # metadata snapshots (from first available scenario row)
    base_meta: dict[str, Any] = {}
    cand_meta: dict[str, Any] = {}
    for s in scenarios:
        if not base_meta and isinstance(baseline.get(s), dict):
            base_meta = _load_run_meta(baseline[s])
        if not cand_meta and isinstance(candidate.get(s), dict):
            cand_meta = _load_run_meta(candidate[s])
    lines.append("## Runtime Meta")
    lines.append("")
    lines.append(f"- baseline detector: `{base_meta.get('detector_type', '-')}` / selector: `{base_meta.get('selector_backend', '-')}`")
    lines.append(f"- candidate detector: `{cand_meta.get('detector_type', '-')}` / selector: `{cand_meta.get('selector_backend', '-')}`")
    lines.append(f"- candidate platform: `{cand_meta.get('platform_type', '-')}` / camera: `{cand_meta.get('camera_topic', '-')}`")
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    header = ["scenario"]
    for m in METRICS:
        header.extend([f"{args.baseline_label}:{m}", f"{args.candidate_label}:{m}", f"delta:{m}"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    failed_detection: list[str] = []
    failed_primary: list[str] = []

    for scenario in scenarios:
        b = baseline.get(scenario, {})
        c = candidate.get(scenario, {})
        row = [scenario]
        row_json: dict[str, Any] = {}
        for m in METRICS:
            b_val = b.get(m)
            c_val = c.get(m)
            delta = None
            if isinstance(b_val, (int, float)) and isinstance(c_val, (int, float)):
                delta = float(c_val) - float(b_val)
            row.extend([_fmt(b_val), _fmt(c_val), _fmt(delta)])
            row_json[m] = {"baseline": b_val, "candidate": c_val, "delta": delta}

        c_det = int(c.get("frames_with_detection", 0) or 0)
        c_primary = int(c.get("frames_with_primary", 0) or 0)
        if c_det < int(args.min_detections):
            failed_detection.append(scenario)
        if c_primary < int(args.min_primary):
            failed_primary.append(scenario)

        lines.append("| " + " | ".join(row) + " |")
        report["scenarios"][scenario] = row_json

    checks = {
        "candidate_detection_positive": {
            "min_detections": int(args.min_detections),
            "failed_scenarios": failed_detection,
            "passed": len(failed_detection) == 0,
        },
        "candidate_primary_positive": {
            "min_primary": int(args.min_primary),
            "failed_scenarios": failed_primary,
            "passed": len(failed_primary) == 0,
        },
    }
    overall_pass = all(bool(v.get("passed", False)) for v in checks.values())
    checks["overall_pass"] = overall_pass
    report["checks"] = checks

    lines.append("")
    lines.append("## DoD Checks")
    lines.append("")
    lines.append(
        f"- candidate detection >={int(args.min_detections)}: "
        f"{'PASS' if checks['candidate_detection_positive']['passed'] else 'FAIL'}"
    )
    lines.append(
        f"- candidate primary >={int(args.min_primary)}: "
        f"{'PASS' if checks['candidate_primary_positive']['passed'] else 'FAIL'}"
    )
    lines.append(f"- overall: **{'PASS' if overall_pass else 'FAIL'}**")
    if failed_detection:
        lines.append(f"- detection failed scenarios: `{', '.join(failed_detection)}`")
    if failed_primary:
        lines.append(f"- primary failed scenarios: `{', '.join(failed_primary)}`")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(md_path)
    print(json_path)
    if not overall_pass and not args.allow_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

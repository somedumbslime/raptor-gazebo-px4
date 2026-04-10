#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

SCENARIOS = [
    "slow_circle",
    "fast_circle",
    "zigzag",
    "temporary_disappearance",
    "long_absence",
]

METRICS = [
    "frames_with_detection",
    "frames_with_primary",
    "presence_ratio",
    "lost_target_count",
    "reacquire_count",
    "search_frames",
    "locked_frames",
]


def load_summary(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid suite summary: {path}")
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare synthetic vs YOLO suite summaries")
    parser.add_argument("--synthetic-suite", required=True, help="suite_summary.json from red detector run")
    parser.add_argument("--yolo-suite", required=True, help="suite_summary.json from yolo_onnx detector run")
    parser.add_argument("--out-dir", default="runs/phase3_compare", help="Output folder for compare artifacts")
    args = parser.parse_args()

    synthetic_path = Path(args.synthetic_suite).resolve()
    yolo_path = Path(args.yolo_suite).resolve()
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Not found: {synthetic_path}")
    if not yolo_path.exists():
        raise FileNotFoundError(f"Not found: {yolo_path}")

    synthetic = load_summary(synthetic_path)
    yolo = load_summary(yolo_path)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"phase3_compare_{stamp}.md"
    json_path = out_dir / f"phase3_compare_{stamp}.json"

    lines: list[str] = []
    lines.append("# Phase 3 Synthetic vs YOLO Comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Synthetic suite: `{synthetic_path}`")
    lines.append(f"YOLO suite: `{yolo_path}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    header_cols = ["scenario"]
    for m in METRICS:
        header_cols.extend([f"synthetic:{m}", f"yolo:{m}", f"delta:{m}"])
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

    report_json: dict[str, Any] = {
        "synthetic_suite": str(synthetic_path),
        "yolo_suite": str(yolo_path),
        "scenarios": {},
    }

    for scenario in SCENARIOS:
        s_row = synthetic.get(scenario, {})
        y_row = yolo.get(scenario, {})
        row_vals = [scenario]
        scenario_json: dict[str, Any] = {}

        for metric in METRICS:
            s_val = s_row.get(metric)
            y_val = y_row.get(metric)
            delta = None
            if isinstance(s_val, (int, float)) and isinstance(y_val, (int, float)):
                delta = float(y_val) - float(s_val)

            def fmt(v: Any) -> str:
                if isinstance(v, float):
                    return f"{v:.6f}"
                if isinstance(v, int):
                    return str(v)
                if v is None:
                    return "-"
                return str(v)

            row_vals.extend([fmt(s_val), fmt(y_val), fmt(delta)])
            scenario_json[metric] = {
                "synthetic": s_val,
                "yolo": y_val,
                "delta": delta,
            }

        lines.append("| " + " | ".join(row_vals) + " |")
        report_json["scenarios"][scenario] = scenario_json

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=True)

    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Aggregate startup benchmark results across multiple runs.

Uses the **representative-run** approach: the run whose total startup time
is the median is selected, and all per-component metrics are reported from
that single run. This ensures components sum consistently to the total.
Min/max across all runs are still reported for range context.

Usage:
    python aggregate_startup_results.py <run_dir> [--output <path>]

Example:
    python aggregate_startup_results.py /tmp/trtllm-startup-bench/Qwen2.5-72B-Instruct_s1_tp8_bs4_nt1024_sl4096/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

METRICS_TO_EXTRACT = [
    ("total_startup_s", "Total startup (server)"),
    ("llm.hf.cache_probe", "HF cache probe"),
    ("llm.hf.remote_download", "HF remote download"),
    ("llm.cached_model_loader", "Cached model loader (server)"),
    ("llm.load_tokenizer", "Load tokenizer (server)"),
    ("llm.create_executor", "Create executor (server)"),
    ("executor_worker.initialize", "Executor worker init (rank 0)"),
    ("executor.load_model_weights", "Weight loading total"),
    ("executor.checkpoint_prefetch", "Checkpoint prefetch"),
    ("executor.checkpoint_parallel_load", "Checkpoint parallel load"),
    ("executor.apply_model_weights.main_weights", "Apply weights"),
    ("executor.warmup.main_model", "Warmup total (1st pass)"),
    ("executor.warmup.torch_compile", "Warmup: torch compile"),
    ("executor.warmup.autotuner", "Warmup: autotuner"),
    ("executor.warmup.autotuner.forward", "Warmup: autotuner forward"),
    ("executor.warmup.cuda_graphs", "Warmup: CUDA graphs"),
    ("executor.warmup.memory_pool", "Warmup: memory pool"),
    ("executor.recreate_py_executor_instance", "Warmup (2nd pass)"),
]


def find_record_duration(records: list[dict], name: str) -> Optional[float]:
    """DFS through hierarchical records to find a timer by name."""
    for rec in records:
        if rec.get("name") == name:
            return rec.get("duration_s")
        children = rec.get("children", [])
        if children:
            result = find_record_duration(children, name)
            if result is not None:
                return result
    return None


def extract_metrics(profile: dict) -> dict[str, Optional[float]]:
    """Extract all key metrics from a single startup profile JSON."""
    metrics: dict[str, Optional[float]] = {}

    metrics["total_startup_s"] = profile.get("total_duration_s")

    server_records = profile.get("records", [])
    for key, _ in METRICS_TO_EXTRACT:
        if key == "total_startup_s":
            continue
        if (
            key.startswith("executor.")
            or key.startswith("executor_worker.")
            or key.startswith("llm.hf.")
        ):
            continue
        metrics[key] = find_record_duration(server_records, key)

    ranks = profile.get("attached_profiles", {}).get("executor_workers", {}).get("ranks", [])
    if ranks:
        rank0 = ranks[0]
        rank0_records = rank0.get("records", [])
        for key, _ in METRICS_TO_EXTRACT:
            if key.startswith("executor.") or key.startswith("executor_worker."):
                metrics[key] = find_record_duration(rank0_records, key)
            elif key.startswith("llm.hf."):
                val = find_record_duration(rank0_records, key)
                if val is None:
                    val = find_record_duration(server_records, key)
                metrics[key] = val

    return metrics


def _select_median_run(
    per_run_metrics: list[dict[str, Optional[float]]],
) -> int:
    """Return the index of the run whose total_startup_s is the median.

    For an odd number of runs, this is the middle value. For an even number,
    we pick the lower-median run (the one just below the midpoint) so that
    the representative run is a real observation rather than an interpolation.
    """
    totals = [(i, m.get("total_startup_s")) for i, m in enumerate(per_run_metrics)]
    totals = [(i, t) for i, t in totals if t is not None]
    if not totals:
        return 0
    totals.sort(key=lambda x: x[1])
    median_idx = (len(totals) - 1) // 2
    return totals[median_idx][0]


def aggregate(run_dir: Path) -> dict[str, Any]:
    """Collect metrics from all run_N subdirectories.

    Selects the **representative run** — the run whose total startup time
    is the median — and reports all metrics from that single run. Min/max
    are computed across all runs for range context.
    """
    run_dirs = sorted(run_dir.glob("run_*"))
    if not run_dirs:
        print(f"ERROR: No run_* subdirectories found in {run_dir}")
        sys.exit(1)

    per_run_metrics: list[dict[str, Optional[float]]] = []
    successful_dirs: list[Path] = []

    for rd in run_dirs:
        profile_path = rd / "startup_profile_server.json"
        if not profile_path.exists():
            print(f"  WARNING: {profile_path} not found, skipping")
            continue

        with open(profile_path) as f:
            profile = json.load(f)

        per_run_metrics.append(extract_metrics(profile))
        successful_dirs.append(rd)

    if not per_run_metrics:
        results = {}
        for key, label in METRICS_TO_EXTRACT:
            results[key] = {"label": label, "median": None, "min": None, "max": None, "n": 0}
        return {
            "run_dir": str(run_dir),
            "num_runs": len(run_dirs),
            "num_successful": 0,
            "representative_run": None,
            "metrics": results,
        }

    rep_idx = _select_median_run(per_run_metrics)
    rep_metrics = per_run_metrics[rep_idx]

    results = {}
    for key, label in METRICS_TO_EXTRACT:
        all_vals = [m.get(key) for m in per_run_metrics if m.get(key) is not None]
        rep_val = rep_metrics.get(key)
        if not all_vals:
            results[key] = {"label": label, "median": None, "min": None, "max": None, "n": 0}
        else:
            results[key] = {
                "label": label,
                "median": rep_val,
                "min": min(all_vals),
                "max": max(all_vals),
                "n": len(all_vals),
            }

    return {
        "run_dir": str(run_dir),
        "num_runs": len(run_dirs),
        "num_successful": len(per_run_metrics),
        "representative_run": str(successful_dirs[rep_idx]),
        "metrics": results,
    }


def format_val(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    if val < 0.01:
        return f"{val * 1000:.2f}ms"
    return f"{val:.1f}s"


def print_summary(agg: dict[str, Any]) -> None:
    print(f"\n{'=' * 72}")
    print(f"Startup Benchmark Aggregate: {agg['run_dir']}")
    print(f"Runs: {agg['num_successful']}/{agg['num_runs']} successful")
    rep = agg.get("representative_run")
    if rep:
        print(f"Representative run (median total): {rep}")
    print(f"{'=' * 72}")
    print(f"{'Metric':<40} {'Rep.Run':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}")

    for key, _ in METRICS_TO_EXTRACT:
        m = agg["metrics"][key]
        label = m["label"]
        if m["n"] == 0:
            print(f"{label:<40} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        else:
            print(
                f"{label:<40} {format_val(m['median']):>10} {format_val(m['min']):>10} {format_val(m['max']):>10}"
            )

    print(f"{'=' * 72}\n")


def write_markdown(agg: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Startup Benchmark Aggregate",
        "",
        f"- Run directory: `{agg['run_dir']}`",
        f"- Successful runs: {agg['num_successful']}/{agg['num_runs']}",
    ]
    rep = agg.get("representative_run")
    if rep:
        lines.append(f"- Representative run (median total): `{rep}`")
    lines += [
        "",
        "| Metric | Rep.Run | Min | Max |",
        "|:-------|-------:|----:|----:|",
    ]
    for key, _ in METRICS_TO_EXTRACT:
        m = agg["metrics"][key]
        label = m["label"]
        if m["n"] == 0:
            lines.append(f"| {label} | N/A | N/A | N/A |")
        else:
            lines.append(
                f"| {label} | {format_val(m['median'])} | {format_val(m['min'])} | {format_val(m['max'])} |"
            )
    lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"Markdown summary written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate startup benchmark results")
    parser.add_argument("run_dir", type=Path, help="Directory containing run_1/, run_2/, etc.")
    parser.add_argument("--output", type=Path, default=None, help="Output markdown file path")
    parser.add_argument("--json", type=Path, default=None, help="Output JSON file path")
    args = parser.parse_args()

    agg = aggregate(args.run_dir)
    print_summary(agg)

    md_path = args.output or (args.run_dir / "aggregate_summary.md")
    write_markdown(agg, md_path)

    json_path = args.json or (args.run_dir / "aggregate_summary.json")
    with open(json_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"JSON summary written to: {json_path}")


if __name__ == "__main__":
    main()

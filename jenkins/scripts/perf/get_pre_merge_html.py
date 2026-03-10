#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Generate a pre-merge HTML report with inline SVG performance charts.

Reads perf_data.yaml files produced by test stages, queries OpenSearch for
historical data and baselines, then generates an HTML report visualizing
key throughput metrics with history, new data, baseline, and threshold lines
for regression comparison.
"""

import argparse
import os
from html import escape as escape_html

import yaml

# Set OPEN_SEARCH_DB_BASE_URL before importing perf_utils, because
# open_search_db captures the env var at module-import time.
if not os.environ.get("OPEN_SEARCH_DB_BASE_URL"):
    os.environ["OPEN_SEARCH_DB_BASE_URL"] = "http://gpuwa.nvidia.com"

from perf_utils import (
    CHART_METRICS,
    METRIC_LABELS,
    _extract_points,
    _generate_svg_chart,
    _get_threshold_for_metric,
    _ts_to_date,
    get_history_data,
)

# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------


def load_perf_data(input_files):
    """Read comma-separated perf_data.yaml paths and return a flat list of new_data dicts."""
    yaml_files = [f.strip() for f in input_files.split(",") if f.strip()]
    all_new_data = []
    load_failures = 0
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
            if content is None or not isinstance(content, list):
                continue
            for e in content:
                if not isinstance(e, dict):
                    continue
                nd = e.get("new_data")
                if isinstance(nd, dict) and "s_test_case_name" in nd:
                    all_new_data.append(nd)
        except (OSError, yaml.YAMLError, UnicodeDecodeError) as exc:
            load_failures += 1
            print(f"Warning: Failed to load {yaml_file}: {exc}")
    if yaml_files and not all_new_data and load_failures == len(yaml_files):
        raise RuntimeError("Failed to load any perf data YAML inputs; cannot generate report.")
    return all_new_data


# ---------------------------------------------------------------------------
# History data query
# ---------------------------------------------------------------------------


def get_pre_merge_history_data(new_data_list):
    """Query OpenSearch for history data matching test cases in *new_data_list*.

    Uses :func:`perf_utils.get_history_data` to fetch post-merge history
    (both baseline and non-baseline), then filters to only the
    (s_test_case_name, s_gpu_type) pairs present in *new_data_list*.

    Returns:
        dict mapping (test_case, gpu_type) -> {
            "history_data": [...],
            "baseline_data": [...],
        }
        or empty dict on failure / no matches.
    """
    if not new_data_list:
        return {}

    # Determine which test case keys are present in new data
    needed_keys = set()
    for nd in new_data_list:
        key = (nd.get("s_test_case_name", ""), nd.get("s_gpu_type", ""))
        needed_keys.add(key)

    grouped = get_history_data(
        extra_must_clauses=[
            {"term": {"b_is_post_merge": True}},
            {"term": {"s_branch": "main"}},
        ]
    )

    if grouped is None:
        print("Warning: Failed to query history data from OpenSearch")
        return {}

    # Filter to only the test cases we have new data for
    filtered = {}
    for key, bucket in grouped.items():
        if key in needed_keys:
            filtered[key] = bucket

    return filtered


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


def _extract_simple_points(data_list, metric):
    """Extract (datetime, float_value) pairs from a list of data dicts."""
    points = []
    for d in data_list:
        ts = d.get("ts_created") or d.get("@timestamp")
        val = d.get(metric)
        if ts is not None and val is not None:
            try:
                points.append((_ts_to_date(ts), float(val)))
            except (ValueError, TypeError):
                pass
    points.sort(key=lambda p: p[0])
    return points


def generate_pre_merge_html(new_data_list, history_grouped, output_file):
    """Generate HTML report visualizing new data against history + baseline.

    For each (test_case, gpu_type) present in *new_data_list*, renders 4
    charts (one per key metric) showing history line, new data points,
    baseline line, and threshold line for regression comparison.
    """
    # Group new data by (test_case, gpu_type)
    new_groups = {}
    for nd in new_data_list:
        key = (nd.get("s_test_case_name", ""), nd.get("s_gpu_type", ""))
        new_groups.setdefault(key, []).append(nd)

    sections_html = []
    for (test_case, gpu_type), new_data_entries in sorted(new_groups.items()):
        bucket = history_grouped.get((test_case, gpu_type), {})
        history_data = bucket.get("history_data", [])
        baseline_data_list = bucket.get("baseline_data", [])

        charts = []
        for metric in CHART_METRICS:
            label = METRIC_LABELS.get(metric, metric)

            # History points (blue line) — use 3-tuple version from perf_utils
            hist_pts = _extract_points(history_data, metric)

            # New data points (red dots)
            new_pts = _extract_simple_points(new_data_entries, metric)

            # Baseline value from the latest baseline entry
            baseline_value = None
            if baseline_data_list:
                latest_bl = baseline_data_list[-1]
                bl_val = latest_bl.get(metric)
                if bl_val is not None:
                    baseline_value = float(bl_val)

            # Threshold line value
            threshold_line_value = None
            if baseline_value is not None:
                threshold = _get_threshold_for_metric(baseline_data_list, metric)
                threshold_line_value = baseline_value * (1 - threshold)

            charts.append(
                _generate_svg_chart(
                    hist_pts,
                    metric,
                    label,
                    new_points=new_pts,
                    baseline_value=baseline_value,
                    threshold_line_value=threshold_line_value,
                )
            )

        header = escape_html(f"{test_case}  [{gpu_type}]")
        section = f"""
        <details class="test-section" open>
            <summary><strong>{header}</strong></summary>
            <div class="charts-grid">
                {"".join(charts)}
            </div>
        </details>
        """
        sections_html.append(section)

    total_new = len(new_data_list)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Perf Sanity Pre-Merge Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }}
        h2 {{ color: #333; }}
        .test-section {{
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: #fff;
            padding: 10px 15px;
        }}
        .test-section summary {{
            cursor: pointer;
            font-size: 14px;
            padding: 6px 0;
        }}
        .charts-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}
        .summary-info {{ color: #666; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <h2>Perf Sanity Pre-Merge Results</h2>
    <p class="summary-info">{len(new_groups)} test case(s) &middot; {total_new} new data point(s)</p>
    {"".join(sections_html)}
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated pre-merge perf report with {len(new_groups)} test cases: {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate a pre-merge HTML report with historical "
        "performance charts, baseline, and threshold lines."
    )
    parser.add_argument(
        "--input-files",
        type=str,
        required=True,
        help="Comma-separated list of perf_data.yaml paths",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output HTML file path")
    args = parser.parse_args()

    new_data_list = load_perf_data(args.input_files)
    history_grouped = get_pre_merge_history_data(new_data_list)
    generate_pre_merge_html(new_data_list, history_grouped, args.output_file)


if __name__ == "__main__":
    main()

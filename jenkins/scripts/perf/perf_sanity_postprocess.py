#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Post-process perf sanity data and generate an HTML report with inline SVG
performance charts.

Each perf_data.yaml entry is a dict with keys:
  - "new_data": the new perf data dict
  - "history_baseline": the latest baseline data dict (or None)
  - "history_data": list of historical (non-baseline) data dicts

History data is already bundled in the YAML by generate_perf_yaml(), so no
OpenSearch query is needed here.
"""

import argparse
from datetime import datetime
from html import escape as escape_html

import yaml

# Metrics to chart (the four key throughput metrics).
CHART_METRICS = [
    "d_total_token_throughput",
    "d_token_throughput",
    "d_seq_throughput",
    "d_user_throughput",
]

METRIC_LABELS = {
    "d_total_token_throughput": "Total Token Throughput (tok/s)",
    "d_token_throughput": "Output Token Throughput (tok/s)",
    "d_seq_throughput": "Request Throughput (req/s)",
    "d_user_throughput": "User Throughput (tok/s)",
}

# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------


def load_perf_data(input_files):
    """Read comma-separated perf_data.yaml paths and return a flat list of
    entry dicts (each with new_data, history_baseline, history_data)."""
    yaml_files = [f.strip() for f in input_files.split(",") if f.strip()]
    all_entries = []
    load_failures = 0
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
            if content is None or not isinstance(content, list):
                continue
            filtered = [
                e for e in content
                if isinstance(e, dict) and isinstance(
                    e.get("new_data"), dict) and "s_test_case_name" in e[
                        "new_data"]
            ]
            all_entries.extend(filtered)
        except (OSError, yaml.YAMLError, UnicodeDecodeError) as e:
            load_failures += 1
            print(f"Warning: Failed to load {yaml_file}: {e}")
    if yaml_files and not all_entries and load_failures == len(yaml_files):
        raise RuntimeError(
            "Failed to load any perf data YAML inputs; cannot generate report."
        )
    return all_entries


def _group_by_test_gpu(entries):
    """Group entries by (s_test_case_name, s_gpu_type) from new_data."""
    groups = {}
    for entry in entries:
        nd = entry["new_data"]
        key = (nd.get("s_test_case_name", ""), nd.get("s_gpu_type", ""))
        groups.setdefault(key, []).append(entry)
    return groups


# ---------------------------------------------------------------------------
# SVG chart generation (pure Python, no external dependencies)
# ---------------------------------------------------------------------------

_SVG_WIDTH = 600
_SVG_HEIGHT = 250
_MARGIN = {"top": 20, "right": 20, "bottom": 50, "left": 70}
_PLOT_W = _SVG_WIDTH - _MARGIN["left"] - _MARGIN["right"]
_PLOT_H = _SVG_HEIGHT - _MARGIN["top"] - _MARGIN["bottom"]


def _ts_to_date(ts):
    """Convert a millisecond timestamp to a datetime."""
    try:
        return datetime.fromtimestamp(int(ts) / 1000)
    except (ValueError, TypeError, OSError):
        return datetime.fromtimestamp(0)


def _format_date(dt):
    return dt.strftime("%m/%d")


def _generate_svg_chart(history_points, new_points, metric, label):
    """Return an SVG string for a single metric chart.

    *history_points* is a list of (datetime, value) already sorted by date.
    *new_points* is a list of (datetime, value) for the current run.
    """
    all_points = history_points + new_points
    if not all_points:
        return (f'<div style="color:#888;padding:10px;">No data for '
                f'{escape_html(label)}</div>')

    values = [v for _, v in all_points if v is not None]
    dates = [d for d, _ in all_points]
    if not values:
        return (f'<div style="color:#888;padding:10px;">No numeric data for '
                f'{escape_html(label)}</div>')

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val != min_val else 1.0
    # Add 5% padding
    min_val -= val_range * 0.05
    max_val += val_range * 0.05
    val_range = max_val - min_val

    min_ts = min(dates).timestamp()
    max_ts = max(dates).timestamp()
    ts_range = max_ts - min_ts if max_ts != min_ts else 1.0

    def _x(dt):
        return _MARGIN["left"] + (dt.timestamp() -
                                   min_ts) / ts_range * _PLOT_W

    def _y(v):
        return (_MARGIN["top"] + _PLOT_H -
                (v - min_val) / val_range * _PLOT_H)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_WIDTH}" '
        f'height="{_SVG_HEIGHT}" style="background:#fff;border:1px solid #ddd;'
        f'border-radius:4px;margin:5px 0;">'
    ]

    # Axes
    svg_parts.append(
        f'<line x1="{_MARGIN["left"]}" y1="{_MARGIN["top"]}" '
        f'x2="{_MARGIN["left"]}" y2="{_MARGIN["top"] + _PLOT_H}" '
        f'stroke="#ccc" stroke-width="1"/>')
    svg_parts.append(
        f'<line x1="{_MARGIN["left"]}" y1="{_MARGIN["top"] + _PLOT_H}" '
        f'x2="{_MARGIN["left"] + _PLOT_W}" y2="{_MARGIN["top"] + _PLOT_H}" '
        f'stroke="#ccc" stroke-width="1"/>')

    # Y-axis ticks (5 ticks)
    for i in range(6):
        v = min_val + val_range * i / 5
        y = _y(v)
        svg_parts.append(
            f'<line x1="{_MARGIN["left"] - 4}" y1="{y:.1f}" '
            f'x2="{_MARGIN["left"] + _PLOT_W}" y2="{y:.1f}" '
            f'stroke="#eee" stroke-width="1"/>')
        svg_parts.append(
            f'<text x="{_MARGIN["left"] - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end" font-size="10" fill="#666">'
            f'{v:.1f}</text>')

    # X-axis date labels (at most 6)
    unique_dates = sorted(set(dates))
    n_labels = min(6, len(unique_dates))
    if len(unique_dates) >= n_labels:
        label_dates = unique_dates[::max(1,
                                         len(unique_dates) //
                                         n_labels)][:n_labels]
    else:
        label_dates = unique_dates
    for dt in label_dates:
        x = _x(dt)
        y_base = _MARGIN["top"] + _PLOT_H
        svg_parts.append(
            f'<text x="{x:.1f}" y="{y_base + 18}" text-anchor="middle" '
            f'font-size="10" fill="#666">{_format_date(dt)}</text>')

    # Title
    svg_parts.append(
        f'<text x="{_SVG_WIDTH / 2}" y="{_MARGIN["top"] - 4}" '
        f'text-anchor="middle" font-size="12" font-weight="bold" '
        f'fill="#333">{escape_html(label)}</text>')

    # History line + dots
    if history_points:
        sorted_hist = sorted([(d, v) for d, v in history_points
                              if v is not None],
                             key=lambda p: p[0])
        if len(sorted_hist) > 1:
            path_d = " ".join(
                f'{"M" if i == 0 else "L"}{_x(d):.1f},{_y(v):.1f}'
                for i, (d, v) in enumerate(sorted_hist))
            svg_parts.append(
                f'<path d="{path_d}" fill="none" stroke="#4285f4" '
                f'stroke-width="2"/>')
        for d, v in sorted_hist:
            svg_parts.append(
                f'<circle cx="{_x(d):.1f}" cy="{_y(v):.1f}" r="3" '
                f'fill="#4285f4"/>')

    # New data points (red)
    for d, v in new_points:
        if v is None:
            continue
        svg_parts.append(
            f'<circle cx="{_x(d):.1f}" cy="{_y(v):.1f}" r="5" '
            f'fill="#d93025" stroke="#fff" stroke-width="1.5"/>')

    # Legend
    legend_y = _MARGIN["top"] + _PLOT_H + 30
    svg_parts.append(
        f'<circle cx="{_MARGIN["left"] + 10}" cy="{legend_y}" r="4" '
        f'fill="#4285f4"/>')
    svg_parts.append(
        f'<text x="{_MARGIN["left"] + 18}" y="{legend_y + 4}" '
        f'font-size="10" fill="#666">History</text>')
    svg_parts.append(
        f'<circle cx="{_MARGIN["left"] + 80}" cy="{legend_y}" r="4" '
        f'fill="#d93025"/>')
    svg_parts.append(
        f'<text x="{_MARGIN["left"] + 88}" y="{legend_y + 4}" '
        f'font-size="10" fill="#666">New</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


# ---------------------------------------------------------------------------
# Helper: extract (datetime, value) points from data dicts
# ---------------------------------------------------------------------------


def _extract_points(data_list, metric):
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


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


def generate_html(all_entries, output_file):
    """Generate HTML report from bundled perf data entries."""
    groups = _group_by_test_gpu(all_entries)

    sections_html = []
    for (test_case, gpu_type), entries in sorted(groups.items()):
        # Merge history_data across entries for the same (test_case, gpu_type)
        merged_history = []
        for entry in entries:
            merged_history.extend(entry.get("history_data") or [])

        # Collect new_data dicts
        new_data_list = [entry["new_data"] for entry in entries]

        # Build charts for each metric
        charts = []
        for metric in CHART_METRICS:
            label = METRIC_LABELS.get(metric, metric)
            hist_pts = _extract_points(merged_history, metric)
            new_pts = _extract_points(new_data_list, metric)
            charts.append(
                _generate_svg_chart(hist_pts, new_pts, metric, label))

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

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Perf Sanity Test Results</title>
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
    <h2>Perf Sanity Test Results</h2>
    <p class="summary-info">{len(groups)} test case(s) &middot; {len(all_entries)} data point(s)</p>
    {"".join(sections_html)}
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated perf sanity report with {len(groups)} test cases: "
          f"{output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Post-process perf sanity data into an HTML report with "
        "historical performance charts.")
    parser.add_argument("--input-files",
                        type=str,
                        required=True,
                        help="Comma-separated list of perf_data.yaml paths")
    parser.add_argument("--output-file",
                        type=str,
                        required=True,
                        help="Output HTML file path")
    args = parser.parse_args()

    all_entries = load_perf_data(args.input_files)
    generate_html(all_entries, args.output_file)


if __name__ == "__main__":
    main()

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
"""Shared utilities for perf sanity scripts.

Contains constants, regression detection algorithms, OpenSearch query helpers,
and HTML/SVG report generation functions used by test.py, get_pre_merge_html.py,
and perf_sanity_triage.py.
"""

import json as _json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from html import escape as escape_html

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from open_search_db import OpenSearchDB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERF_SANITY_PROJECT_NAME = "swdl-trtllm-infra-ci-prod-perf_sanity_info"
QUERY_LOOKBACK_DAYS = 90
MAX_QUERY_SIZE = 9999
DEFAULT_THRESHOLD = 0.05

CHART_METRICS = [
    "d_seq_throughput",
    "d_token_throughput",
    "d_total_token_throughput",
    "d_user_throughput",
]

# Only these 2 metrics determine the overall test-case classification.
CLASSIFICATION_METRICS = [
    "d_token_throughput",
    "d_total_token_throughput",
]

METRIC_LABELS = {
    "d_seq_throughput": "Request Throughput (req/s)",
    "d_token_throughput": "Output Token Throughput (tok/s)",
    "d_total_token_throughput": "Total Token Throughput (tok/s)",
    "d_user_throughput": "User Throughput (tok/s)",
}

# Algorithm parameters
_STABILITY_CV_THRESHOLD = 0.03  # 3%
_REGRESSION_THRESHOLD = 0.05  # 5%
_ROLLING_WINDOW = 7
_MIN_STABLE_SEGMENT = 7
_MIN_CONFIRMATION_DAYS = 3
_DIRECTION_CHANGE_THRESHOLD = 6  # per 30 days
_OUTLIER_ZSCORE = 2.0

# Curve type display
_CURVE_TYPE_COLORS = {
    "no_regression": "#0d904f",
    "sudden_drop": "#d93025",
    "gradual_decline": "#e8710a",
    "significant_fluctuation": "#7b1fa2",
    "occasional_spike": "#c5a600",
    "other_reasons": "#607d8b",
}

_CURVE_TYPE_LABELS = {
    "no_regression": "No Regression",
    "sudden_drop": "Sudden Drop",
    "gradual_decline": "Gradual Decline",
    "significant_fluctuation": "Significant Fluctuation",
    "occasional_spike": "Occasional Spike",
    "other_reasons": "Other Reasons",
}

# ---------------------------------------------------------------------------
# Timestamp / data utilities
# ---------------------------------------------------------------------------

_TIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%b %d, %Y @ %H:%M:%S.%f",
]


def _parse_timestamp(timestamp):
    """Parse a timestamp value into a datetime object."""
    if isinstance(timestamp, (int, float)):
        if timestamp > 1e12:
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp)
    if isinstance(timestamp, datetime):
        return timestamp
    timestamp_str = str(timestamp)
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    return datetime.fromtimestamp(0)


def _ts_to_date(ts):
    """Convert a millisecond timestamp to a datetime."""
    try:
        return datetime.fromtimestamp(int(ts) / 1000)
    except (ValueError, TypeError, OSError):
        return datetime.fromtimestamp(0)


def _extract_points(data_list, metric):
    """Extract (datetime, float_value, data_dict) triples from data dicts."""
    points = []
    for d in data_list:
        ts = d.get("ts_created") or d.get("@timestamp")
        val = d.get(metric)
        if ts is not None and val is not None:
            try:
                points.append((_ts_to_date(ts), float(val), d))
            except (ValueError, TypeError):
                pass
    points.sort(key=lambda p: p[0])
    return points


def _data_dict_to_json_attr(data_dict):
    """Serialize a data dict to an HTML-safe JSON string for embedding in attributes."""
    return escape_html(_json.dumps(data_dict, default=str, ensure_ascii=True))


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------


def _daily_aggregate(points):
    """Aggregate multiple data points on the same day to a single mean value.

    Args:
        points: list of (datetime, float) or (datetime, float, data_dict)
                tuples.

    Returns:
        list of (date_str, float, [data_dicts]) triples sorted by date.
        The third element is a list of original data dicts for that day
        (empty list when input items have no third element).
    """
    by_day = defaultdict(list)
    entries = defaultdict(list)
    for item in points:
        dt, val = item[0], item[1]
        day_key = dt.strftime("%Y-%m-%d")
        by_day[day_key].append(val)
        if len(item) > 2 and item[2] is not None:
            entries[day_key].append(item[2])
    result = []
    for day in sorted(by_day):
        vals = by_day[day]
        result.append((day, sum(vals) / len(vals), entries[day]))
    return result


def _rolling_smooth(values, window=3):
    """Trailing rolling mean with same-length output.

    Early elements use fewer samples (i.e. the first element is itself,
    the second is the mean of the first two, etc.).
    """
    if not values:
        return []
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        w = values[start : i + 1]
        smoothed.append(sum(w) / len(w))
    return smoothed


def _percentile(values, p):
    """Compute the p-th percentile with linear interpolation.

    Args:
        values: non-empty list of floats.
        p: percentile in [0, 100].
    """
    if not values:
        return 0.0
    s = sorted(values)
    k = (p / 100.0) * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


def get_baseline(grouped_data):
    """Compute rolling-smooth + P95 baselines and daily data for all entries.

    For each (test_case, gpu_type) key and each metric, this function:
    1. Extracts data points as 3-tuples (datetime, float, data_dict).
    2. Aggregates to daily values preserving original data entries.
    3. Applies rolling smooth (window=3) to daily values.
    4. Computes P95 of the smoothed values as the baseline.

    Mutates ``grouped_data[key]`` to add:
        "daily_data": {metric: {"dates": [...], "values": [...],
                                "entries": [[data_dicts], ...]}},
        "baselines": {metric: float},
    """
    for key, bucket in grouped_data.items():
        history_data = bucket["history_data"]
        daily_data = {}
        baselines = {}
        for metric in CHART_METRICS:
            points = _extract_points(history_data, metric)
            daily = _daily_aggregate(points)
            daily_dates = [d for d, _, _ in daily]
            daily_vals = [v for _, v, _ in daily]
            daily_entries = [e for _, _, e in daily]

            smoothed = _rolling_smooth(daily_vals, window=3)
            baseline = _percentile(smoothed, 95) if smoothed else 0.0

            daily_data[metric] = {
                "dates": daily_dates,
                "values": daily_vals,
                "entries": daily_entries,
            }
            baselines[metric] = baseline
        bucket["daily_data"] = daily_data
        bucket["baselines"] = baselines


# ---------------------------------------------------------------------------
# Regression classification
# ---------------------------------------------------------------------------


def _extract_jump_commits(daily_entries, daily_dates, js_idx, je_idx):
    """Extract commit and timestamp info at jump interval endpoints.

    Args:
        daily_entries: list of lists of data_dicts (one list per day).
        daily_dates: list of date strings corresponding to daily_entries.
        js_idx: jump-start day index (left endpoint).
        je_idx: jump-end day index (right endpoint).

    Returns:
        {"left": {"s_commit": str, "timestamp": str},
         "right": {"s_commit": str, "timestamp": str}}
        or None if data is unavailable.
    """
    if not daily_entries or not daily_dates:
        return None
    js_idx = max(0, min(js_idx, len(daily_entries) - 1))
    je_idx = max(0, min(je_idx, len(daily_entries) - 1))

    def _pick_last(entries_list):
        """Pick the last chronological entry from a day's entries."""
        if not entries_list:
            return None
        best = entries_list[-1]
        for e in entries_list:
            ts_e = e.get("ts_created") or e.get("@timestamp", 0)
            ts_b = best.get("ts_created") or best.get("@timestamp", 0)
            if ts_e is not None and ts_b is not None and ts_e > ts_b:
                best = e
        commit = best.get("s_commit", "")
        ts_raw = best.get("ts_created") or best.get("@timestamp", "")
        if isinstance(ts_raw, (int, float)):
            if ts_raw > 1e12:
                ts_raw = ts_raw / 1000
            ts_str = datetime.fromtimestamp(ts_raw).strftime("%Y-%m-%d %H:%M")
        else:
            ts_str = str(ts_raw)
        return {"s_commit": str(commit), "timestamp": ts_str}

    left = _pick_last(daily_entries[js_idx])
    right = _pick_last(daily_entries[je_idx])
    if left is None and right is None:
        return None
    return {"left": left, "right": right}


def _cv(values):
    """Coefficient of variation (std / mean). Returns 0 if mean is 0."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / abs(mean)


def _is_stable(values, threshold=_STABILITY_CV_THRESHOLD):
    """Check if CV < threshold."""
    return _cv(values) < threshold


def _rolling_stats(values, window=_ROLLING_WINDOW):
    """Compute rolling means, rolling CVs, and direction change count.

    Returns:
        (rolling_means, rolling_cvs, direction_changes)
    """
    if len(values) < window:
        return [], [], 0

    rolling_means = []
    rolling_cvs = []
    for i in range(len(values) - window + 1):
        w = values[i : i + window]
        m = sum(w) / len(w)
        rolling_means.append(m)
        rolling_cvs.append(_cv(w))

    direction_changes = 0
    for i in range(2, len(rolling_means)):
        d_prev = rolling_means[i - 1] - rolling_means[i - 2]
        d_curr = rolling_means[i] - rolling_means[i - 1]
        if d_prev * d_curr < 0:
            direction_changes += 1

    return rolling_means, rolling_cvs, direction_changes


def _find_change_point(values, window=_ROLLING_WINDOW):
    """Find the optimal split point using segmented approach (Phase 4).

    Returns:
        (split_index, jump_start_index, jump_end_index) or None.
    """
    n = len(values)
    if n < 2 * window:
        return None

    best_score = -1
    best_idx = -1
    eps = 1e-12

    for i in range(window, n - window + 1):
        left = values[:i]
        right = values[i:]
        left_mean = sum(left) / len(left)
        right_mean = sum(right) / len(right)
        left_var = sum((v - left_mean) ** 2 for v in left) / len(left)
        right_var = sum((v - right_mean) ** 2 for v in right) / len(right)
        score = (left_mean - right_mean) ** 2 / (left_var + right_var + eps)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        return None

    pre_level = sum(values[:best_idx]) / best_idx
    post_level = sum(values[best_idx:]) / (n - best_idx)

    if pre_level == post_level:
        return best_idx, best_idx, best_idx

    threshold_start = pre_level - 0.2 * (pre_level - post_level)
    threshold_end = pre_level - 0.8 * (pre_level - post_level)

    jump_start = best_idx
    jump_end = best_idx

    if pre_level > post_level:
        for j in range(n):
            if values[j] < threshold_start:
                jump_start = j
                break
        for j in range(n):
            if values[j] < threshold_end:
                jump_end = j
                break
    else:
        for j in range(n):
            if values[j] > threshold_start:
                jump_start = j
                break
        for j in range(n):
            if values[j] > threshold_end:
                jump_end = j
                break

    return best_idx, jump_start, jump_end


def _is_regression(daily_values, baseline, threshold=_REGRESSION_THRESHOLD):
    """Step 1: Determine whether the metric shows a regression.

    A regression exists when the recent average drops more than
    ``threshold`` compared to the baseline.

    Returns True if regression is detected, False otherwise.
    """
    if not daily_values or baseline == 0:
        return False
    recent_count = min(5, max(3, len(daily_values)))
    recent_avg = sum(daily_values[-recent_count:]) / recent_count
    drop_ratio = (baseline - recent_avg) / baseline
    return drop_ratio > threshold


def _classify_regression_type(daily_values):
    """Step 2: Given that a regression exists, determine its subtype.

    Checks in priority order:
        1. Significant Fluctuation
        2. Occasional Spike
        3. Sudden Drop
        4. Gradual Decline

    If none of the four patterns match, falls back to ``"other_reasons"``.

    Returns (regression_type, jump_interval) where regression_type is one of
    ``"significant_fluctuation"``, ``"occasional_spike"``,
    ``"sudden_drop"``, ``"gradual_decline"``, ``"other_reasons"``.
    """
    n_days = len(daily_values)
    rolling_means, rolling_cvs, direction_changes = _rolling_stats(daily_values)

    # --- Significant Fluctuation ---
    normalized_dir_changes = direction_changes * 30 / n_days if n_days > 0 else 0
    oscillation_windows = 0
    if rolling_means:
        for i in range(len(rolling_means)):
            w = daily_values[i : i + _ROLLING_WINDOW]
            if w and max(w) > 0:
                amp = (max(w) - min(w)) / max(w)
                if amp > _REGRESSION_THRESHOLD:
                    oscillation_windows += 1
        has_long_stable = False
        stable_run = 0
        for cv_val in rolling_cvs:
            if cv_val < _STABILITY_CV_THRESHOLD:
                stable_run += 1
                if stable_run >= 2 * _ROLLING_WINDOW:
                    has_long_stable = True
                    break
            else:
                stable_run = 0

        if (
            normalized_dir_changes > _DIRECTION_CHANGE_THRESHOLD
            and oscillation_windows > len(rolling_means) * 0.3
            and not has_long_stable
        ):
            return "significant_fluctuation", None

    # --- Occasional Spike ---
    if n_days >= 3:
        mean_val = sum(daily_values) / n_days
        std_val = math.sqrt(sum((v - mean_val) ** 2 for v in daily_values) / n_days)
        if std_val > 0:
            outlier_indices = [
                i
                for i, v in enumerate(daily_values)
                if abs(v - mean_val) / std_val > _OUTLIER_ZSCORE
            ]
        else:
            outlier_indices = []
        non_outlier_vals = [v for i, v in enumerate(daily_values) if i not in outlier_indices]
        if len(outlier_indices) < 3 and non_outlier_vals and _is_stable(non_outlier_vals):
            max_consecutive_low = 0
            consecutive = 0
            low_threshold = mean_val - _REGRESSION_THRESHOLD * mean_val
            for v in daily_values:
                if v < low_threshold:
                    consecutive += 1
                    max_consecutive_low = max(max_consecutive_low, consecutive)
                else:
                    consecutive = 0
            if max_consecutive_low < _MIN_CONFIRMATION_DAYS:
                return "occasional_spike", None

    # --- Sudden Drop / Gradual Decline (via change-point analysis) ---
    cp = _find_change_point(daily_values)
    if cp is not None:
        split_idx, jump_start, jump_end = cp
        pre_segment = daily_values[:split_idx]
        post_segment = daily_values[split_idx:]

        adj_left = max(0, jump_start - 1)
        adj_right = jump_end
        if adj_left >= adj_right:
            adj_left = max(0, adj_right - 1)
        if adj_left == adj_right:
            adj_right = min(n_days - 1, adj_right + 1)

        if len(pre_segment) >= _MIN_STABLE_SEGMENT and len(post_segment) >= _MIN_CONFIRMATION_DAYS:
            pre_stable = _is_stable(pre_segment)
            post_stable = _is_stable(post_segment)
            pre_mean = sum(pre_segment) / len(pre_segment)
            post_mean = sum(post_segment) / len(post_segment)

            transition_width = abs(jump_end - jump_start) + 1
            shift = (pre_mean - post_mean) / pre_mean if pre_mean > 0 else 0

            if (
                pre_stable
                and post_stable
                and shift > _REGRESSION_THRESHOLD
                and transition_width <= 2
            ):
                return "sudden_drop", (adj_left, adj_right)

            if (
                pre_stable
                and post_stable
                and shift > _REGRESSION_THRESHOLD
                and transition_width > 2
            ):
                decline_vals = daily_values[jump_start : jump_end + 1]
                if len(decline_vals) >= 3:
                    x_vals = list(range(len(decline_vals)))
                    x_mean = sum(x_vals) / len(x_vals)
                    y_mean = sum(decline_vals) / len(decline_vals)
                    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, decline_vals))
                    ss_xx = sum((x - x_mean) ** 2 for x in x_vals)
                    ss_yy = sum((y - y_mean) ** 2 for y in decline_vals)
                    if ss_xx > 0 and ss_yy > 0:
                        slope = ss_xy / ss_xx
                        r_squared = (ss_xy**2) / (ss_xx * ss_yy)
                        if slope < 0 and r_squared > 0.7:
                            return "gradual_decline", (adj_left, adj_right)

        return "other_reasons", (jump_start, jump_end)

    return "other_reasons", None


def classify_single_metric(daily_values, baseline, threshold=_REGRESSION_THRESHOLD):
    """Two-step classification for one metric's time series.

    Step 1 -- Regression check:
        Is the recent average more than ``threshold`` below the baseline?
        If **no** -> ``"no_regression"``.

    Step 2 -- Regression subtype (only when Step 1 says *yes*):
        Classify into one of ``"significant_fluctuation"``,
        ``"occasional_spike"``, ``"sudden_drop"``,
        ``"gradual_decline"``, or ``"other_reasons"``.

    Returns:
        (curve_type, jump_interval) where jump_interval is
        (start_index, end_index) or None.
    """
    if not daily_values:
        return "no_regression", None

    if not _is_regression(daily_values, baseline, threshold):
        return "no_regression", None

    regression_type, jump_interval = _classify_regression_type(daily_values)
    return regression_type, jump_interval


def _get_threshold_for_metric(baseline_data_list, metric):
    """Get the pre-merge threshold for a metric from the latest baseline data.

    Looks for d_threshold_pre_merge_{metric_suffix} in the latest baseline
    entry.  Returns DEFAULT_THRESHOLD (5%) if not found.
    """
    if not baseline_data_list:
        return DEFAULT_THRESHOLD
    latest_baseline = baseline_data_list[-1]
    metric_suffix = metric[2:]  # Remove "d_" prefix
    threshold_key = f"d_threshold_pre_merge_{metric_suffix}"
    if threshold_key in latest_baseline:
        return latest_baseline[threshold_key]
    return DEFAULT_THRESHOLD


def classify_test_case(grouped_data):
    """Run classification on all metrics and aggregate results.

    Uses threshold from baseline data for each metric.  Reads pre-computed
    ``daily_data`` and ``baselines`` from each entry (populated by
    :func:`get_baseline`) and stores classification results back into
    ``grouped_data[key]``:
        "curve_type": str  (overall)
        "per_metric_info": {metric: {"curve_type": str,
                                      "jump_interval": (date_str, date_str) or None,
                                      "jump_commits": {...} or None}}
    """
    for key, bucket in grouped_data.items():
        daily_data = bucket.get("daily_data", {})
        baselines = bucket.get("baselines", {})
        baseline_data_list = bucket.get("baseline_data", [])
        per_metric_results = {}
        per_metric_info = {}

        for metric in CHART_METRICS:
            md = daily_data.get(metric, {})
            daily_vals = md.get("values", [])
            daily_dates = md.get("dates", [])
            daily_entries = md.get("entries", [])
            baseline = baselines.get(metric, 0.0)

            threshold = _get_threshold_for_metric(baseline_data_list, metric)
            curve_type, jump = classify_single_metric(daily_vals, baseline, threshold)
            per_metric_results[metric] = curve_type

            jump_dates = None
            jump_commits = None
            if jump is not None and daily_dates:
                js, je = jump
                js = max(0, min(js, len(daily_dates) - 1))
                je = max(0, min(je, len(daily_dates) - 1))
                jump_dates = (daily_dates[js], daily_dates[je])
                if curve_type in ("sudden_drop", "gradual_decline", "other_reasons"):
                    jump_commits = _extract_jump_commits(daily_entries, daily_dates, js, je)

            per_metric_info[metric] = {
                "curve_type": curve_type,
                "jump_interval": jump_dates,
                "jump_commits": jump_commits,
            }

        # Aggregate overall type using only CLASSIFICATION_METRICS.
        # Both NR and OS are "transparent" (defer to the other metric).
        # Priority: SF > OR > GD > SD > OS > NR
        classification_types = [
            per_metric_results[m] for m in CLASSIFICATION_METRICS if m in per_metric_results
        ]

        if not classification_types:
            overall = "no_regression"
        elif len(classification_types) == 1:
            overall = classification_types[0]
        else:
            # 6x6 aggregation: merge two types via priority, where NR and
            # OS are transparent (defer to the other curve's type).
            _PRIORITY = {
                "significant_fluctuation": 5,
                "other_reasons": 4,
                "gradual_decline": 3,
                "sudden_drop": 2,
                "occasional_spike": 1,
                "no_regression": 0,
            }
            a, b = classification_types[0], classification_types[1]
            pa, pb = _PRIORITY.get(a, 0), _PRIORITY.get(b, 0)
            overall = a if pa >= pb else b

        bucket["curve_type"] = overall
        bucket["per_metric_info"] = per_metric_info


# ---------------------------------------------------------------------------
# OpenSearch query + grouping
# ---------------------------------------------------------------------------


def get_history_data(extra_must_clauses=None):
    """Query perf data from OpenSearch and group by (s_test_case_name, s_gpu_type).

    Queries both baseline and non-baseline data from the last
    QUERY_LOOKBACK_DAYS days.  Additional filters can be passed via
    *extra_must_clauses*.

    Returns:
        dict mapping (test_case, gpu_type) -> {
            "history_data": [non-baseline entries sorted by time],
            "baseline_data": [baseline entries sorted by time],
        }
        or None on query failure.
    """
    must_clauses = [
        {"term": {"b_is_valid": True}},
        {
            "range": {
                "ts_created": {
                    "gte": int(time.time() - 24 * 3600 * QUERY_LOOKBACK_DAYS)
                    // (24 * 3600)
                    * 24
                    * 3600
                    * 1000,
                }
            }
        },
    ]
    if extra_must_clauses:
        must_clauses.extend(extra_must_clauses)

    data_list = OpenSearchDB.queryPerfDataFromOpenSearchDB(
        PERF_SANITY_PROJECT_NAME, must_clauses, size=MAX_QUERY_SIZE
    )

    if data_list is None:
        return None

    groups = {}
    for data in data_list:
        key = (
            data.get("s_test_case_name", ""),
            data.get("s_gpu_type", ""),
        )
        groups.setdefault(key, {"history_data": [], "baseline_data": []})
        if data.get("b_is_baseline"):
            groups[key]["baseline_data"].append(data)
        else:
            groups[key]["history_data"].append(data)

    for key, bucket in groups.items():
        bucket["history_data"] = sorted(
            bucket["history_data"],
            key=lambda d: _parse_timestamp(d.get("ts_created") or d.get("@timestamp", 0)),
        )
        bucket["baseline_data"] = sorted(
            bucket["baseline_data"],
            key=lambda d: _parse_timestamp(d.get("ts_created") or d.get("@timestamp", 0)),
        )

    return groups


# ---------------------------------------------------------------------------
# SVG chart generation
# ---------------------------------------------------------------------------

_SVG_WIDTH = 620
_SVG_HEIGHT = 280
_MARGIN = {"top": 30, "right": 20, "bottom": 55, "left": 75}
_PLOT_W = _SVG_WIDTH - _MARGIN["left"] - _MARGIN["right"]
_PLOT_H = _SVG_HEIGHT - _MARGIN["top"] - _MARGIN["bottom"]


def _generate_svg_chart(
    history_points,
    metric,
    label,
    new_points=None,
    baseline_value=None,
    threshold_line_value=None,
    curve_type=None,
    jump_interval=None,
):
    """Return an SVG string for a single metric chart.

    Args:
        history_points: list of (datetime, value) or (datetime, value, data_dict)
                        sorted by date.
        metric: metric key string.
        label: display label for the chart title.
        new_points: optional list of (datetime, value) for new data (red dots).
        baseline_value: optional float drawn as a horizontal dashed red line.
        threshold_line_value: optional float drawn as a horizontal dashed
                              orange line (regression threshold).
        curve_type: optional str -- the regression classification for this
                    metric (used for badge display).
        jump_interval: optional (start_date_str, end_date_str) -- regression
                       window shading.
    """
    all_values = [v for _, v, *_ in history_points if v is not None]
    if new_points:
        all_values.extend(v for _, v in new_points if v is not None)
    if baseline_value is not None:
        all_values.append(baseline_value)
    if threshold_line_value is not None:
        all_values.append(threshold_line_value)

    if not history_points and not new_points and baseline_value is None:
        return f'<div style="color:#888;padding:10px;">No data for {escape_html(label)}</div>'
    if not all_values:
        return (
            f'<div style="color:#888;padding:10px;">No numeric data for {escape_html(label)}</div>'
        )

    min_val = min(all_values)
    max_val = max(all_values)
    val_range = max_val - min_val if max_val != min_val else 1.0
    min_val -= val_range * 0.05
    max_val += val_range * 0.05
    val_range = max_val - min_val

    dates = [d for d, *_ in history_points]
    if new_points:
        dates.extend(d for d, _ in new_points)
    if not dates:
        return (
            f'<div style="color:#888;padding:10px;">No data points for {escape_html(label)}</div>'
        )

    min_ts = min(dates).timestamp()
    max_ts = max(dates).timestamp()
    ts_range = max_ts - min_ts if max_ts != min_ts else 1.0

    def _x(dt):
        return _MARGIN["left"] + (dt.timestamp() - min_ts) / ts_range * _PLOT_W

    def _x_date_str(date_str):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        ts = dt.timestamp()
        ts = max(min_ts, min(ts, max_ts))
        return _MARGIN["left"] + (ts - min_ts) / ts_range * _PLOT_W

    def _y(v):
        return _MARGIN["top"] + _PLOT_H - (v - min_val) / val_range * _PLOT_H

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_WIDTH}" '
        f'height="{_SVG_HEIGHT}" style="background:#fff;border:1px solid #ddd;'
        f'border-radius:4px;margin:5px 0;">'
    ]

    # Grid lines (Y axis, 5 ticks)
    for i in range(6):
        v = min_val + val_range * i / 5
        y = _y(v)
        svg.append(
            f'<line x1="{_MARGIN["left"] - 4}" y1="{y:.1f}" '
            f'x2="{_MARGIN["left"] + _PLOT_W}" y2="{y:.1f}" '
            f'stroke="#eee" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{_MARGIN["left"] - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end" font-size="10" fill="#666">{v:.1f}</text>'
        )

    # Jump interval shaded region
    if jump_interval is not None:
        j_start, j_end = jump_interval
        jx1 = _x_date_str(j_start)
        jx2 = _x_date_str(j_end)
        if jx2 - jx1 < 4:
            jx2 = jx1 + 4
        svg.append(
            f'<rect x="{jx1:.1f}" y="{_MARGIN["top"]}" '
            f'width="{jx2 - jx1:.1f}" height="{_PLOT_H}" '
            f'fill="#d93025" opacity="0.10"/>'
        )
        svg.append(
            f'<line x1="{jx1:.1f}" y1="{_MARGIN["top"]}" '
            f'x2="{jx1:.1f}" y2="{_MARGIN["top"] + _PLOT_H}" '
            f'stroke="#d93025" stroke-width="1" stroke-dasharray="4,2" '
            f'opacity="0.5"/>'
        )
        svg.append(
            f'<line x1="{jx2:.1f}" y1="{_MARGIN["top"]}" '
            f'x2="{jx2:.1f}" y2="{_MARGIN["top"] + _PLOT_H}" '
            f'stroke="#d93025" stroke-width="1" stroke-dasharray="4,2" '
            f'opacity="0.5"/>'
        )

    # Axes
    svg.append(
        f'<line x1="{_MARGIN["left"]}" y1="{_MARGIN["top"]}" '
        f'x2="{_MARGIN["left"]}" y2="{_MARGIN["top"] + _PLOT_H}" '
        f'stroke="#ccc" stroke-width="1"/>'
    )
    svg.append(
        f'<line x1="{_MARGIN["left"]}" y1="{_MARGIN["top"] + _PLOT_H}" '
        f'x2="{_MARGIN["left"] + _PLOT_W}" y2="{_MARGIN["top"] + _PLOT_H}" '
        f'stroke="#ccc" stroke-width="1"/>'
    )

    # X-axis date labels
    unique_dates = sorted(set(dates))
    n_labels = min(6, len(unique_dates))
    if len(unique_dates) >= n_labels:
        label_dates = unique_dates[:: max(1, len(unique_dates) // n_labels)][:n_labels]
    else:
        label_dates = unique_dates
    for dt in label_dates:
        x = _x(dt)
        y_base = _MARGIN["top"] + _PLOT_H
        svg.append(
            f'<text x="{x:.1f}" y="{y_base + 18}" text-anchor="middle" '
            f'font-size="10" fill="#666">{dt.strftime("%m/%d")}</text>'
        )

    # Title with curve type badge
    title_text = escape_html(label)
    svg.append(
        f'<text x="{_SVG_WIDTH / 2}" y="{_MARGIN["top"] - 8}" '
        f'text-anchor="middle" font-size="12" font-weight="bold" '
        f'fill="#333">{title_text}</text>'
    )
    if curve_type and curve_type != "no_regression":
        ct_color = _CURVE_TYPE_COLORS.get(curve_type, "#888")
        ct_short = _CURVE_TYPE_LABELS.get(curve_type, curve_type)
        badge_x = _SVG_WIDTH - _MARGIN["right"] - 4
        badge_y = _MARGIN["top"] - 16
        badge_text = ct_short
        if jump_interval:
            badge_text += f"  [{jump_interval[0]} ~ {jump_interval[1]}]"
        text_w = len(badge_text) * 5.5 + 10
        rx = badge_x - text_w
        svg.append(
            f'<rect x="{rx:.0f}" y="{badge_y}" width="{text_w:.0f}" '
            f'height="14" rx="7" fill="{ct_color}" opacity="0.9"/>'
        )
        svg.append(
            f'<text x="{rx + text_w / 2:.0f}" y="{badge_y + 10}" '
            f'text-anchor="middle" font-size="8" fill="#fff" '
            f'font-weight="bold">{escape_html(badge_text)}</text>'
        )

    # Baseline horizontal line (dashed red)
    if baseline_value is not None:
        by = _y(baseline_value)
        svg.append(
            f'<line x1="{_MARGIN["left"]}" y1="{by:.1f}" '
            f'x2="{_MARGIN["left"] + _PLOT_W}" y2="{by:.1f}" '
            f'stroke="#d93025" stroke-width="1.5" stroke-dasharray="6,3"/>'
        )

    # Threshold horizontal line (dashed orange)
    if threshold_line_value is not None:
        ty = _y(threshold_line_value)
        svg.append(
            f'<line x1="{_MARGIN["left"]}" y1="{ty:.1f}" '
            f'x2="{_MARGIN["left"] + _PLOT_W}" y2="{ty:.1f}" '
            f'stroke="#e8710a" stroke-width="1.5" stroke-dasharray="4,4"/>'
        )

    # History line + dots (blue)
    sorted_hist = sorted(
        [(d, v, *rest) for d, v, *rest in history_points if v is not None],
        key=lambda p: p[0],
    )
    if len(sorted_hist) > 1:
        path_d = " ".join(
            f"{'M' if i == 0 else 'L'}{_x(d):.1f},{_y(v):.1f}"
            for i, (d, v, *_) in enumerate(sorted_hist)
        )
        svg.append(f'<path d="{path_d}" fill="none" stroke="#4285f4" stroke-width="2"/>')
    for item in sorted_hist:
        d, v = item[0], item[1]
        dd = item[2] if len(item) > 2 else None
        if dd is not None:
            json_attr = _data_dict_to_json_attr(dd)
            svg.append(
                f'<circle class="data-point" cx="{_x(d):.1f}" cy="{_y(v):.1f}" '
                f'r="4" fill="#4285f4" style="cursor:pointer" '
                f'data-info="{json_attr}">'
                f"<title>{d.strftime('%Y-%m-%d %H:%M')}  {v:.2f}</title></circle>"
            )
        else:
            svg.append(f'<circle cx="{_x(d):.1f}" cy="{_y(v):.1f}" r="3" fill="#4285f4"/>')

    # New data points (red)
    if new_points:
        for d, v in new_points:
            if v is None:
                continue
            svg.append(
                f'<circle cx="{_x(d):.1f}" cy="{_y(v):.1f}" r="5" '
                f'fill="#d93025" stroke="#fff" stroke-width="1.5"/>'
            )

    # Legend
    legend_y = _MARGIN["top"] + _PLOT_H + 35
    legend_x = _MARGIN["left"] + 10
    svg.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="4" fill="#4285f4"/>')
    svg.append(
        f'<text x="{legend_x + 8}" y="{legend_y + 4}" font-size="10" fill="#666">History</text>'
    )
    legend_x += 70
    if new_points:
        svg.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="4" fill="#d93025"/>')
        svg.append(
            f'<text x="{legend_x + 8}" y="{legend_y + 4}" font-size="10" fill="#666">New</text>'
        )
        legend_x += 50
    if baseline_value is not None:
        svg.append(
            f'<line x1="{legend_x}" y1="{legend_y}" '
            f'x2="{legend_x + 20}" y2="{legend_y}" '
            f'stroke="#d93025" stroke-width="1.5" stroke-dasharray="6,3"/>'
        )
        svg.append(
            f'<text x="{legend_x + 25}" y="{legend_y + 4}" '
            f'font-size="10" fill="#666">Baseline ({baseline_value:.2f})</text>'
        )
        legend_x += 150
    if threshold_line_value is not None:
        svg.append(
            f'<line x1="{legend_x}" y1="{legend_y}" '
            f'x2="{legend_x + 20}" y2="{legend_y}" '
            f'stroke="#e8710a" stroke-width="1.5" stroke-dasharray="4,4"/>'
        )
        svg.append(
            f'<text x="{legend_x + 25}" y="{legend_y + 4}" '
            f'font-size="10" fill="#666">Threshold ({threshold_line_value:.2f})</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


# ---------------------------------------------------------------------------
# HTML report generation (post-merge dashboard)
# ---------------------------------------------------------------------------


def generate_post_merge_html(grouped_data, output_file):
    """Generate a post-merge HTML dashboard from grouped perf data.

    This produces a full interactive report with three-way cascading filters
    (GPU Type, Test Case, Curve Type), summary tables, and click-to-inspect
    data-point popups.
    """
    all_gpu_types = sorted(set(gpu for _, gpu in grouped_data.keys()))
    all_test_cases = sorted(set(tc for tc, _ in grouped_data.keys()))
    all_curve_types_set = set()

    sections = []
    section_tuples = []

    for (test_case, gpu_type), bucket in sorted(grouped_data.items()):
        history_data = bucket["history_data"]
        curve_type = bucket.get("curve_type", "no_regression")
        baselines = bucket.get("baselines", {})
        per_metric_info = bucket.get("per_metric_info", {})

        all_curve_types_set.add(curve_type)
        section_tuples.append((gpu_type, test_case, curve_type))

        charts = []
        for metric in CHART_METRICS:
            label = METRIC_LABELS.get(metric, metric)
            hist_pts = _extract_points(history_data, metric)
            baseline_val = baselines.get(metric)
            m_info = per_metric_info.get(metric, {})
            charts.append(
                _generate_svg_chart(
                    hist_pts,
                    metric,
                    label,
                    baseline_value=baseline_val,
                    curve_type=m_info.get("curve_type"),
                    jump_interval=m_info.get("jump_interval"),
                )
            )

        # Summary table
        summary_rows = ""
        if history_data:
            latest = history_data[-1]
            for metric in CHART_METRICS:
                val = latest.get(metric)
                bl_val = baselines.get(metric)
                diff_str = ""
                if val is not None and bl_val is not None and bl_val != 0:
                    diff_pct = (val - bl_val) / bl_val * 100
                    color = "#0d904f" if diff_pct >= 0 else "#d93025"
                    diff_str = f' <span style="color:{color}">({diff_pct:+.2f}%)</span>'
                val_str = f"{val:.2f}" if val is not None else "N/A"
                bl_str = f"{bl_val:.2f}" if bl_val is not None else "N/A"
                m_info = per_metric_info.get(metric, {})
                m_ct = m_info.get("curve_type", "no_regression")
                m_ct_color = _CURVE_TYPE_COLORS.get(m_ct, "#888")
                m_ct_label = _CURVE_TYPE_LABELS.get(m_ct, m_ct)
                m_jump = m_info.get("jump_interval")
                jump_str = ""
                if m_jump:
                    jump_str = (
                        f' <span style="color:#666;font-size:11px;">'
                        f"[{m_jump[0]} ~ {m_jump[1]}]</span>"
                    )
                ct_cell = (
                    f'<span style="display:inline-block;padding:1px 7px;'
                    f"border-radius:8px;background:{m_ct_color};color:#fff;"
                    f'font-size:10px;font-weight:bold;">{m_ct_label}</span>'
                    f"{jump_str}"
                )
                jc = m_info.get("jump_commits")
                jl_cell = ""
                jr_cell = ""
                if jc:
                    left = jc.get("left")
                    right = jc.get("right")
                    if left and left.get("s_commit"):
                        short = left["s_commit"][:8]
                        ts = left.get("timestamp", "")
                        jl_cell = (
                            f"<code>{escape_html(short)}</code>"
                            f'<br><span style="color:#888;font-size:10px;">'
                            f"{escape_html(ts)}</span>"
                        )
                    if right and right.get("s_commit"):
                        short = right["s_commit"][:8]
                        ts = right.get("timestamp", "")
                        jr_cell = (
                            f"<code>{escape_html(short)}</code>"
                            f'<br><span style="color:#888;font-size:10px;">'
                            f"{escape_html(ts)}</span>"
                        )
                summary_rows += (
                    f"<tr><td>{METRIC_LABELS.get(metric, metric)}</td>"
                    f"<td>{val_str}{diff_str}</td>"
                    f"<td>{bl_str}</td>"
                    f"<td>{ct_cell}</td>"
                    f"<td>{jl_cell}</td>"
                    f"<td>{jr_cell}</td></tr>"
                )

        n_points = len(history_data)
        ct_color = _CURVE_TYPE_COLORS.get(curve_type, "#888")
        ct_label = _CURVE_TYPE_LABELS.get(curve_type, curve_type)

        header = escape_html(f"{test_case}  [{gpu_type}]")
        data_gpu = escape_html(gpu_type)
        data_test = escape_html(test_case)
        data_curve = escape_html(curve_type)
        table_header = (
            "<thead><tr><th>Metric</th><th>Latest Value</th>"
            "<th>Baseline (P95)</th><th>Curve Type</th>"
            "<th>Jump Left</th><th>Jump Right</th></tr></thead>"
        )
        section = f"""
        <details class="test-section" data-gpu="{data_gpu}" data-test="{
            data_test
        }" data-curve-type="{data_curve}" open>
            <summary><strong>{header}</strong>
                <span class="badge">{n_points} runs</span>
                <span class="curve-badge" style="background:{ct_color};">{ct_label}</span>
            </summary>
            <div class="charts-grid">
                {"".join(charts)}
            </div>
            {
            ""
            if not summary_rows
            else f'''
            <table class="summary-table">
                {table_header}
                <tbody>{summary_rows}</tbody>
            </table>
            '''
        }
        </details>
        """
        sections.append(section)

    all_curve_types = sorted(all_curve_types_set)

    gpu_to_tests = {}
    test_to_gpus = {}
    for tc, gpu in grouped_data.keys():
        gpu_to_tests.setdefault(gpu, [])
        if tc not in gpu_to_tests[gpu]:
            gpu_to_tests[gpu].append(tc)
        test_to_gpus.setdefault(tc, [])
        if gpu not in test_to_gpus[tc]:
            test_to_gpus[tc].append(gpu)
    for k in gpu_to_tests:
        gpu_to_tests[k].sort()
    for k in test_to_gpus:
        test_to_gpus[k].sort()

    triples_json = _json.dumps(section_tuples)

    gpu_chips = [
        '<button class="chip active" data-filter-gpu="__all__" onclick="selectGpu(this)">All</button>'
    ]
    for gpu in all_gpu_types:
        gpu_chips.append(
            f'<button class="chip" data-filter-gpu="{escape_html(gpu)}" '
            f'onclick="selectGpu(this)">{escape_html(gpu)}</button>'
        )

    test_chips = [
        '<button class="chip active" data-filter-test="__all__" onclick="selectTest(this)">All</button>'
    ]
    for tc in all_test_cases:
        test_chips.append(
            f'<button class="chip" data-filter-test="{escape_html(tc)}" '
            f'onclick="selectTest(this)">{escape_html(tc)}</button>'
        )

    curve_chips = [
        '<button class="chip active" data-filter-curve="__all__" onclick="selectCurve(this)">All</button>'
    ]
    for ct in all_curve_types:
        ct_color = _CURVE_TYPE_COLORS.get(ct, "#888")
        ct_label = _CURVE_TYPE_LABELS.get(ct, ct)
        curve_chips.append(
            f'<button class="chip" data-filter-curve="{escape_html(ct)}" '
            f'onclick="selectCurve(this)">'
            f'<span style="display:inline-block;width:8px;height:8px;'
            f'border-radius:50%;background:{ct_color};margin-right:4px;"></span>'
            f"{escape_html(ct_label)}</button>"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Perf Sanity History Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }}
        h2 {{ color: #333; }}
        .filter-panel {{
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 12px 15px;
            margin-bottom: 15px;
        }}
        .filter-panel h3 {{
            margin: 0 0 8px 0;
            font-size: 13px;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .chip-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 10px;
            max-height: 200px;
            overflow-y: auto;
        }}
        .chip {{
            display: inline-block;
            padding: 4px 12px;
            font-size: 12px;
            border: 1px solid #ccc;
            border-radius: 16px;
            background: #fff;
            color: #555;
            cursor: pointer;
            transition: all 0.15s;
            white-space: nowrap;
        }}
        .chip:hover {{
            border-color: #4285f4;
            color: #4285f4;
        }}
        .chip.active {{
            background: #4285f4;
            color: #fff;
            border-color: #4285f4;
        }}
        .chip.disabled {{
            opacity: 0.35;
            cursor: default;
            pointer-events: none;
        }}
        .match-count {{
            font-size: 12px;
            color: #888;
            margin-top: 4px;
        }}
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
        .badge {{
            font-size: 11px;
            color: #888;
            margin-left: 10px;
        }}
        .curve-badge {{
            font-size: 10px;
            color: #fff;
            padding: 2px 8px;
            border-radius: 10px;
            margin-left: 6px;
            font-weight: bold;
        }}
        .charts-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}
        .summary-table {{
            margin-top: 10px;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .summary-table th, .summary-table td {{
            border: 1px solid #ddd;
            padding: 4px 10px;
            text-align: left;
        }}
        .summary-table th {{
            background: #f5f5f5;
        }}
        .meta {{ color: #666; margin-bottom: 15px; font-size: 13px; }}
        #point-popup {{
            display: none;
            position: fixed;
            background: #fff;
            border: 1px solid #aaa;
            border-radius: 6px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.18);
            padding: 0;
            z-index: 1000;
            max-width: 520px;
            max-height: 70vh;
            overflow: auto;
            font-size: 12px;
        }}
        #point-popup .popup-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 10px;
            background: #f5f5f5;
            border-bottom: 1px solid #ddd;
            border-radius: 6px 6px 0 0;
            position: sticky;
            top: 0;
        }}
        #point-popup .popup-header span {{
            font-weight: bold;
            font-size: 12px;
            color: #333;
        }}
        #point-popup .popup-close {{
            cursor: pointer;
            font-size: 16px;
            color: #888;
            border: none;
            background: none;
            padding: 0 4px;
        }}
        #point-popup .popup-close:hover {{ color: #d93025; }}
        #point-popup table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            font-family: monospace;
        }}
        #point-popup td {{
            padding: 2px 8px;
            border-bottom: 1px solid #eee;
            word-break: break-all;
        }}
        #point-popup td:first-child {{
            color: #555;
            white-space: nowrap;
            font-weight: bold;
            width: 1%;
        }}
        circle.data-point:hover {{ r: 6; stroke: #fff; stroke-width: 2; }}
    </style>
</head>
<body>
    <h2>Perf Sanity History Dashboard</h2>
    <p class="meta">
        {len(grouped_data)} test case(s) &middot;
        Lookback: {QUERY_LOOKBACK_DAYS} days &middot;
        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </p>

    <div class="filter-panel">
        <h3>GPU Type</h3>
        <div class="chip-group" id="gpu-chips">
            {"".join(gpu_chips)}
        </div>
        <h3>Test Case</h3>
        <div class="chip-group" id="test-chips">
            {"".join(test_chips)}
        </div>
        <h3>Curve Type</h3>
        <div class="chip-group" id="curve-chips">
            {"".join(curve_chips)}
        </div>
        <div class="match-count" id="match-count"></div>
    </div>

    <div id="point-popup">
        <div class="popup-header">
            <span>Data Point Details</span>
            <button class="popup-close" onclick="closePopup()">&times;</button>
        </div>
        <div id="popup-body"></div>
    </div>

    <div id="sections-container">
        {"".join(sections)}
    </div>

    <script>
    (function() {{
        var triples = {triples_json};
        var selectedGpu = "__all__";
        var selectedTest = "__all__";
        var selectedCurve = "__all__";

        function getAvailable(filterGpu, filterTest, filterCurve) {{
            var gpus = {{}}, tests = {{}}, curves = {{}};
            for (var i = 0; i < triples.length; i++) {{
                var g = triples[i][0], t = triples[i][1], c = triples[i][2];
                var gOk = (filterGpu === null || filterGpu === g);
                var tOk = (filterTest === null || filterTest === t);
                var cOk = (filterCurve === null || filterCurve === c);
                if (gOk && tOk && cOk) {{
                    gpus[g] = true;
                    tests[t] = true;
                    curves[c] = true;
                }}
            }}
            return {{gpus: gpus, tests: tests, curves: curves}};
        }}

        function updateChipAvailability() {{
            var gFilter = selectedGpu === "__all__" ? null : selectedGpu;
            var tFilter = selectedTest === "__all__" ? null : selectedTest;
            var cFilter = selectedCurve === "__all__" ? null : selectedCurve;

            var forGpus = getAvailable(null, tFilter, cFilter);
            var forTests = getAvailable(gFilter, null, cFilter);
            var forCurves = getAvailable(gFilter, tFilter, null);

            var gpuChips = document.querySelectorAll("#gpu-chips .chip");
            for (var i = 0; i < gpuChips.length; i++) {{
                var val = gpuChips[i].getAttribute("data-filter-gpu");
                if (val === "__all__") {{
                    gpuChips[i].classList.remove("disabled");
                }} else if (forGpus.gpus[val]) {{
                    gpuChips[i].classList.remove("disabled");
                }} else {{
                    gpuChips[i].classList.add("disabled");
                }}
            }}

            var testChips = document.querySelectorAll("#test-chips .chip");
            for (var i = 0; i < testChips.length; i++) {{
                var val = testChips[i].getAttribute("data-filter-test");
                if (val === "__all__") {{
                    testChips[i].classList.remove("disabled");
                }} else if (forTests.tests[val]) {{
                    testChips[i].classList.remove("disabled");
                }} else {{
                    testChips[i].classList.add("disabled");
                }}
            }}

            var curveChips = document.querySelectorAll("#curve-chips .chip");
            for (var i = 0; i < curveChips.length; i++) {{
                var val = curveChips[i].getAttribute("data-filter-curve");
                if (val === "__all__") {{
                    curveChips[i].classList.remove("disabled");
                }} else if (forCurves.curves[val]) {{
                    curveChips[i].classList.remove("disabled");
                }} else {{
                    curveChips[i].classList.add("disabled");
                }}
            }}
        }}

        function applyFilters() {{
            var sections = document.querySelectorAll(".test-section");
            var visible = 0;
            for (var i = 0; i < sections.length; i++) {{
                var s = sections[i];
                var gpuMatch = (selectedGpu === "__all__" || s.getAttribute("data-gpu") === selectedGpu);
                var testMatch = (selectedTest === "__all__" || s.getAttribute("data-test") === selectedTest);
                var curveMatch = (selectedCurve === "__all__" || s.getAttribute("data-curve-type") === selectedCurve);
                if (gpuMatch && testMatch && curveMatch) {{
                    s.style.display = "";
                    visible++;
                }} else {{
                    s.style.display = "none";
                }}
            }}
            document.getElementById("match-count").textContent =
                "Showing " + visible + " of " + sections.length + " test case(s)";
            updateChipAvailability();
        }}

        function resetChipGroup(selector, attrName, newVal) {{
            var chips = document.querySelectorAll(selector);
            for (var i = 0; i < chips.length; i++) {{
                chips[i].classList.remove("active");
                if (chips[i].getAttribute(attrName) === newVal) chips[i].classList.add("active");
            }}
        }}

        function autoResetInvalid() {{
            var gFilter = selectedGpu === "__all__" ? null : selectedGpu;
            var tFilter = selectedTest === "__all__" ? null : selectedTest;
            var cFilter = selectedCurve === "__all__" ? null : selectedCurve;
            var avail = getAvailable(gFilter, tFilter, cFilter);
            var hasResults = Object.keys(avail.gpus).length > 0;
            if (!hasResults) {{
                if (selectedCurve !== "__all__") {{
                    selectedCurve = "__all__";
                    resetChipGroup("#curve-chips .chip", "data-filter-curve", "__all__");
                }} else if (selectedTest !== "__all__") {{
                    selectedTest = "__all__";
                    resetChipGroup("#test-chips .chip", "data-filter-test", "__all__");
                }} else {{
                    selectedGpu = "__all__";
                    resetChipGroup("#gpu-chips .chip", "data-filter-gpu", "__all__");
                }}
            }}
        }}

        window.selectGpu = function(btn) {{
            if (btn.classList.contains("disabled")) return;
            resetChipGroup("#gpu-chips .chip", "data-filter-gpu", btn.getAttribute("data-filter-gpu"));
            btn.classList.add("active");
            selectedGpu = btn.getAttribute("data-filter-gpu");
            autoResetInvalid();
            applyFilters();
        }};

        window.selectTest = function(btn) {{
            if (btn.classList.contains("disabled")) return;
            resetChipGroup("#test-chips .chip", "data-filter-test", btn.getAttribute("data-filter-test"));
            btn.classList.add("active");
            selectedTest = btn.getAttribute("data-filter-test");
            autoResetInvalid();
            applyFilters();
        }};

        window.selectCurve = function(btn) {{
            if (btn.classList.contains("disabled")) return;
            resetChipGroup("#curve-chips .chip", "data-filter-curve", btn.getAttribute("data-filter-curve"));
            btn.classList.add("active");
            selectedCurve = btn.getAttribute("data-filter-curve");
            autoResetInvalid();
            applyFilters();
        }};

        // --- Data-point click popup ---
        var popup = document.getElementById("point-popup");
        var popupBody = document.getElementById("popup-body");
        var activeCircle = null;

        window.closePopup = function() {{
            popup.style.display = "none";
            if (activeCircle) {{
                activeCircle.setAttribute("stroke", "none");
                activeCircle.setAttribute("stroke-width", "0");
                activeCircle = null;
            }}
        }};

        document.addEventListener("click", function(e) {{
            var circle = e.target.closest ? e.target.closest("circle.data-point") : null;
            if (!circle && e.target.classList && e.target.classList.contains("data-point")) {{
                circle = e.target;
            }}
            if (circle) {{
                e.stopPropagation();
                if (activeCircle && activeCircle !== circle) {{
                    activeCircle.setAttribute("stroke", "none");
                    activeCircle.setAttribute("stroke-width", "0");
                }}
                activeCircle = circle;
                circle.setAttribute("stroke", "#d93025");
                circle.setAttribute("stroke-width", "2.5");

                var raw = circle.getAttribute("data-info");
                if (!raw) return;
                var info;
                try {{ info = JSON.parse(raw); }} catch(_) {{ return; }}

                var keys = Object.keys(info).sort();
                var rows = "";
                for (var k = 0; k < keys.length; k++) {{
                    var val = info[keys[k]];
                    if (val === null || val === undefined) val = "";
                    rows += "<tr><td>" + keys[k] + "</td><td>" + String(val) + "</td></tr>";
                }}
                popupBody.innerHTML = "<table>" + rows + "</table>";

                var px = e.clientX + 15;
                var py = e.clientY - 30;
                var pw = 520;
                if (px + pw > window.innerWidth) px = e.clientX - pw - 15;
                if (py < 0) py = 10;
                if (py + 400 > window.innerHeight) py = Math.max(10, window.innerHeight - 420);
                popup.style.left = px + "px";
                popup.style.top = py + "px";
                popup.style.display = "block";
            }} else if (!e.target.closest("#point-popup")) {{
                closePopup();
            }}
        }});

        applyFilters();
    }})();
    </script>
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated perf history report with {len(grouped_data)} test cases: {output_file}")

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
historical post-merge data, computes baselines locally (rolling smooth +
P95), then generates an interactive HTML report visualizing key throughput
metrics with history, new data, baseline, and threshold lines for regression
comparison.

All utility functions are self-contained — no external perf_utils dependency.
Grouping uses match_keys (from test_perf_sanity.py) instead of simple
(s_test_case_name, s_gpu_type) tuples.
"""

import argparse
import json as _json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from html import escape as escape_html

import yaml

# Set OPEN_SEARCH_DB_BASE_URL before importing open_search_db, because
# open_search_db captures the env var at module-import time.
if not os.environ.get("OPEN_SEARCH_DB_BASE_URL"):
    os.environ["OPEN_SEARCH_DB_BASE_URL"] = "http://gpuwa.nvidia.com"

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
# Match keys (from test_perf_sanity.py / perf-regression-detector)
# ---------------------------------------------------------------------------

SERVER_MATCH_KEYS = [
    "s_model_name",
    "l_tp",
    "l_ep",
    "l_pp",
    "l_cp",
    "l_gpus_per_node",
    "l_max_batch_size",
    "b_disable_overlap_scheduler",
    "b_enable_chunked_prefill",
    "b_enable_attention_dp",
    "b_enable_lm_head_tp_in_adp",
    "b_attention_dp_balance",
    "b_enable_cuda_graph",
    "s_kv_cache_dtype",
    "s_cache_transceiver_backend",
    "s_spec_decoding_type",
    "l_num_nextn_predict_layers",
]

CLIENT_MATCH_KEYS = [
    "l_concurrency",
    "l_iterations",
    "l_isl",
    "l_osl",
    "d_random_range_ratio",
    "s_backend",
    "b_use_chat_template",
    "b_streaming",
]

DISAGG_BASE_KEYS = [
    "s_gpu_type",
    "s_runtime",
    "s_benchmark_mode",
    "l_num_ctx_servers",
    "l_num_gen_servers",
]

# ---------------------------------------------------------------------------
# Match keys grouping logic
# ---------------------------------------------------------------------------


def _add_prefix(key, prefix):
    """Add prefix after type marker: s_foo -> s_ctx_foo."""
    return f"{key[:2]}{prefix}_{key[2:]}"


def _get_match_keys_for_data(data):
    """Determine which match_keys apply to a data point based on its runtime."""
    runtime = data.get("s_runtime", "")
    if runtime in ("aggr_server", "multi_node_aggr_server"):
        return ["s_branch", "s_gpu_type", "s_runtime"] + SERVER_MATCH_KEYS + CLIENT_MATCH_KEYS
    elif runtime == "multi_node_disagg_server":
        keys = ["s_branch"] + list(DISAGG_BASE_KEYS)
        if data.get("l_num_ctx_servers", 0) > 0:
            keys.extend(_add_prefix(k, "ctx") for k in SERVER_MATCH_KEYS)
        if data.get("l_num_gen_servers", 0) > 0:
            keys.extend(_add_prefix(k, "gen") for k in SERVER_MATCH_KEYS)
        keys.extend(CLIENT_MATCH_KEYS)
        return keys
    else:
        # Fallback
        return ["s_test_case_name", "s_branch", "s_gpu_type"]


def _compute_group_key(data):
    """Compute a hashable group key from match_keys values."""
    keys = _get_match_keys_for_data(data)
    return tuple((k, data.get(k)) for k in keys)


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
                points.append((_parse_timestamp(ts), float(val), d))
            except (ValueError, TypeError):
                pass
    points.sort(key=lambda p: p[0])
    return points


def _data_dict_to_json_attr(data_dict):
    """Serialize a data dict to an HTML-safe JSON string for embedding in attributes."""
    return escape_html(_json.dumps(data_dict, default=str, ensure_ascii=True))


# ---------------------------------------------------------------------------
# Baseline computation (per data point, no daily aggregation)
# ---------------------------------------------------------------------------


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
    """Compute rolling-smooth + P95 baselines and per-point data for all entries.

    Each data point is kept as an independent observation (no daily
    aggregation).  Rolling smooth uses a trailing window of 3 data points.
    """
    for key, bucket in grouped_data.items():
        history_data = bucket["history_data"]
        point_data = {}
        baselines = {}
        for metric in CHART_METRICS:
            points = _extract_points(history_data, metric)
            # Keep every data point independent — no daily aggregation.
            point_dates = [d.strftime("%Y-%m-%d %H:%M") for d, _, _ in points]
            point_vals = [v for _, v, _ in points]
            # Wrap each data_dict in a list for _extract_jump_commits compat.
            point_entries = [[dd] for _, _, dd in points]

            smoothed = _rolling_smooth(point_vals, window=3)
            baseline = _percentile(smoothed, 95) if smoothed else 0.0

            point_data[metric] = {
                "dates": point_dates,
                "values": point_vals,
                "entries": point_entries,
            }
            baselines[metric] = baseline
        bucket["point_data"] = point_data
        bucket["baselines"] = baselines


# ---------------------------------------------------------------------------
# Regression classification
# ---------------------------------------------------------------------------


def _extract_jump_commits(point_entries, point_dates, js_idx, je_idx):
    """Extract commit and timestamp info at jump interval endpoints."""
    if not point_entries or not point_dates:
        return None
    js_idx = max(0, min(js_idx, len(point_entries) - 1))
    je_idx = max(0, min(je_idx, len(point_entries) - 1))

    def _pick_last(entries_list):
        """Pick the last chronological entry from a point's entries."""
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

    left = _pick_last(point_entries[js_idx])
    right = _pick_last(point_entries[je_idx])
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
    """Find the optimal split point using segmented approach.

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


def _is_regression(values, baseline, threshold=_REGRESSION_THRESHOLD):
    """Determine whether the metric shows a regression.

    Checks the latest single data point against the baseline.
    """
    if not values or baseline == 0:
        return False
    latest = values[-1]
    drop_ratio = (baseline - latest) / baseline
    return drop_ratio > threshold


def _classify_regression_type(values):
    """Given that a regression exists, determine its subtype.

    Checks in priority order:
        1. Significant Fluctuation
        2. Occasional Spike
        3. Sudden Drop
        4. Gradual Decline

    If none of the four patterns match, falls back to ``"other_reasons"``.
    """
    n_pts = len(values)
    rolling_means, rolling_cvs, direction_changes = _rolling_stats(values)

    # --- Significant Fluctuation ---
    normalized_dir_changes = direction_changes * 30 / n_pts if n_pts > 0 else 0
    oscillation_windows = 0
    if rolling_means:
        for i in range(len(rolling_means)):
            w = values[i : i + _ROLLING_WINDOW]
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
    if n_pts >= 3:
        mean_val = sum(values) / n_pts
        std_val = math.sqrt(sum((v - mean_val) ** 2 for v in values) / n_pts)
        if std_val > 0:
            outlier_indices = [
                i for i, v in enumerate(values) if abs(v - mean_val) / std_val > _OUTLIER_ZSCORE
            ]
        else:
            outlier_indices = []
        non_outlier_vals = [v for i, v in enumerate(values) if i not in outlier_indices]
        if len(outlier_indices) < 3 and non_outlier_vals and _is_stable(non_outlier_vals):
            max_consecutive_low = 0
            consecutive = 0
            low_threshold = mean_val - _REGRESSION_THRESHOLD * mean_val
            for v in values:
                if v < low_threshold:
                    consecutive += 1
                    max_consecutive_low = max(max_consecutive_low, consecutive)
                else:
                    consecutive = 0
            if max_consecutive_low < _MIN_CONFIRMATION_DAYS:
                return "occasional_spike", None

    # --- Sudden Drop / Gradual Decline (via change-point analysis) ---
    cp = _find_change_point(values)
    if cp is not None:
        split_idx, jump_start, jump_end = cp
        pre_segment = values[:split_idx]
        post_segment = values[split_idx:]

        adj_left = max(0, jump_start - 1)
        adj_right = jump_end
        if adj_left >= adj_right:
            adj_left = max(0, adj_right - 1)
        if adj_left == adj_right:
            adj_right = min(n_pts - 1, adj_right + 1)

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
                decline_vals = values[jump_start : jump_end + 1]
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


def classify_single_metric(values, baseline, threshold=_REGRESSION_THRESHOLD):
    """Two-step classification for one metric's time series."""
    if not values:
        return "no_regression", None

    if not _is_regression(values, baseline, threshold):
        return "no_regression", None

    regression_type, jump_interval = _classify_regression_type(values)
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
    """Run classification on all metrics and aggregate results."""
    for key, bucket in grouped_data.items():
        point_data = bucket.get("point_data", {})
        baselines = bucket.get("baselines", {})
        baseline_data_list = bucket.get("baseline_data", [])
        per_metric_results = {}
        per_metric_info = {}

        for metric in CHART_METRICS:
            md = point_data.get(metric, {})
            pt_vals = md.get("values", [])
            pt_dates = md.get("dates", [])
            pt_entries = md.get("entries", [])
            baseline = baselines.get(metric, 0.0)

            threshold = _get_threshold_for_metric(baseline_data_list, metric)
            curve_type, jump = classify_single_metric(pt_vals, baseline, threshold)
            per_metric_results[metric] = curve_type

            jump_dates = None
            jump_commits = None
            if jump is not None and pt_dates:
                js, je = jump
                js = max(0, min(js, len(pt_dates) - 1))
                je = max(0, min(je, len(pt_dates) - 1))
                jump_dates = (pt_dates[js], pt_dates[je])
                if curve_type in ("sudden_drop", "gradual_decline", "other_reasons"):
                    jump_commits = _extract_jump_commits(pt_entries, pt_dates, js, je)

            per_metric_info[metric] = {
                "curve_type": curve_type,
                "jump_interval": jump_dates,
                "jump_commits": jump_commits,
            }

        # Aggregate overall type using only CLASSIFICATION_METRICS.
        classification_types = [
            per_metric_results[m] for m in CLASSIFICATION_METRICS if m in per_metric_results
        ]

        if not classification_types:
            overall = "no_regression"
        elif len(classification_types) == 1:
            overall = classification_types[0]
        else:
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
# OpenSearch query + grouping (match_keys based)
# ---------------------------------------------------------------------------


def get_history_data(extra_must_clauses=None):
    """Query perf data from OpenSearch and group by match_keys.

    Uses match_keys from test_perf_sanity.py for authoritative grouping
    instead of simple (s_test_case_name, s_gpu_type) grouping.

    Returns:
        dict mapping group_key -> {
            "history_data": [...],
            "baseline_data": [...],
            "display_label": (test_case_name, branch, gpu_type),
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
        key = _compute_group_key(data)
        if key not in groups:
            groups[key] = {
                "history_data": [],
                "baseline_data": [],
                "display_label": (
                    data.get("s_test_case_name", ""),
                    data.get("s_branch", ""),
                    data.get("s_gpu_type", ""),
                ),
            }
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
    new_data_value=None,
    curve_type=None,
    jump_interval=None,
):
    """Return an SVG string for a single metric chart.

    Args:
        history_points: list of (datetime, value) or (datetime, value, data_dict)
                        sorted by date.
        metric: metric key string.
        label: display label for the chart title.
        new_points: optional list of (datetime, value, data_dict) for new data (red dots).
        baseline_value: optional float drawn as a horizontal dashed red line.
        threshold_line_value: optional float drawn as a horizontal dashed
                              orange line (regression threshold).
        new_data_value: optional float drawn as a horizontal dashed green line
                        (the pre-merge data value from input YAML).
        curve_type: optional str -- the regression classification for this
                    metric (used for badge display).
        jump_interval: optional (start_date_str, end_date_str) -- regression
                       window shading.
    """
    all_values = [v for _, v, *_ in history_points if v is not None]
    if new_points:
        all_values.extend(v for _, v, *_ in new_points if v is not None)
    if baseline_value is not None:
        all_values.append(baseline_value)
    if threshold_line_value is not None:
        all_values.append(threshold_line_value)
    if new_data_value is not None:
        all_values.append(new_data_value)

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
        dates.extend(d for d, *_ in new_points)
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
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        except ValueError:
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

    # New data value horizontal line (dashed green)
    if new_data_value is not None:
        ny = _y(new_data_value)
        svg.append(
            f'<line x1="{_MARGIN["left"]}" y1="{ny:.1f}" '
            f'x2="{_MARGIN["left"] + _PLOT_W}" y2="{ny:.1f}" '
            f'stroke="#0d904f" stroke-width="1.5" stroke-dasharray="5,3"/>'
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

    # New data points (red) — clickable with data-info
    if new_points:
        for item in new_points:
            d, v = item[0], item[1]
            dd = item[2] if len(item) > 2 else None
            if v is None:
                continue
            if dd is not None:
                json_attr = _data_dict_to_json_attr(dd)
                svg.append(
                    f'<circle class="data-point" cx="{_x(d):.1f}" cy="{_y(v):.1f}" r="5" '
                    f'fill="#d93025" stroke="#fff" stroke-width="1.5" style="cursor:pointer" '
                    f'data-info="{json_attr}">'
                    f"<title>{d.strftime('%Y-%m-%d %H:%M')}  {v:.2f}</title></circle>"
                )
            else:
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
        legend_x += 160
    if new_data_value is not None:
        svg.append(
            f'<line x1="{legend_x}" y1="{legend_y}" '
            f'x2="{legend_x + 20}" y2="{legend_y}" '
            f'stroke="#0d904f" stroke-width="1.5" stroke-dasharray="5,3"/>'
        )
        svg.append(
            f'<text x="{legend_x + 25}" y="{legend_y + 4}" '
            f'font-size="10" fill="#666">Latest ({new_data_value:.2f})</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


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

    Uses :func:`get_history_data` to fetch post-merge history (both baseline
    and non-baseline), grouped by match_keys (which includes ``s_branch``).
    Only keeps groups whose match_keys overlap with the input test cases.

    Returns:
        dict mapping group_key -> {
            "history_data": [...],
            "baseline_data": [...],
            "display_label": (test_case_name, branch, gpu_type),
        }
        or empty dict on failure / no matches.
    """
    if not new_data_list:
        return {}

    # Determine which group keys are present in new data
    needed_keys = set()
    for nd in new_data_list:
        key = _compute_group_key(nd)
        needed_keys.add(key)

    grouped = get_history_data(
        extra_must_clauses=[
            {"term": {"b_is_post_merge": True}},
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


def _build_data_table_html(data_dict):
    """Build an HTML key-value table for a data dict."""
    rows = []
    for k in sorted(data_dict.keys()):
        val = data_dict[k]
        if val is None:
            val = ""
        rows.append(f"<tr><td>{escape_html(str(k))}</td><td>{escape_html(str(val))}</td></tr>")
    return f'<table class="kv-table">{"".join(rows)}</table>'


def generate_pre_merge_html(new_data_list, history_grouped, output_file):
    """Generate HTML report visualizing new data against history + baseline.

    For each test case group present in *new_data_list*, renders 4 charts
    (one per key metric) showing history line, new data points, baseline line,
    threshold line, and new data value line for regression comparison.
    """
    # Compute baselines and run classification on history data
    if history_grouped:
        get_baseline(history_grouped)
        classify_test_case(history_grouped)

    # Group new data by match_keys group key
    new_groups = defaultdict(list)
    for nd in new_data_list:
        key = _compute_group_key(nd)
        new_groups[key].append(nd)

    sections_html = []
    section_idx = 0
    for key, new_data_entries in sorted(new_groups.items(), key=lambda kv: str(kv[0])):
        bucket = history_grouped.get(key, {})
        history_data = bucket.get("history_data", [])
        baseline_data_list = bucket.get("baseline_data", [])
        baselines = bucket.get("baselines", {})
        per_metric_info = bucket.get("per_metric_info", {})

        # Determine display label
        display_label = bucket.get("display_label")
        if display_label:
            test_case, branch, gpu_type = display_label
        else:
            # Fallback from new data
            nd0 = new_data_entries[0]
            test_case = nd0.get("s_test_case_name", "")
            branch = nd0.get("s_branch", "")
            gpu_type = nd0.get("s_gpu_type", "")

        charts = []
        for metric in CHART_METRICS:
            label = METRIC_LABELS.get(metric, metric)

            # History points (blue line) — 3-tuple
            hist_pts = _extract_points(history_data, metric)

            # New data points (red dots) — 3-tuple with data dict
            new_pts = _extract_points(new_data_entries, metric)

            # Baseline value (computed locally via rolling smooth + P95)
            baseline_value = baselines.get(metric)

            # Threshold line value
            threshold_line_value = None
            if baseline_value is not None:
                threshold = _get_threshold_for_metric(baseline_data_list, metric)
                threshold_line_value = baseline_value * (1 - threshold)

            # New data value (latest pre-merge result) as horizontal line
            new_data_value = None
            if new_pts:
                new_data_value = new_pts[-1][1]

            m_info = per_metric_info.get(metric, {})

            charts.append(
                _generate_svg_chart(
                    hist_pts,
                    metric,
                    label,
                    new_points=new_pts,
                    baseline_value=baseline_value,
                    threshold_line_value=threshold_line_value,
                    new_data_value=new_data_value,
                    curve_type=m_info.get("curve_type"),
                    jump_interval=m_info.get("jump_interval"),
                )
            )

        # Build header with branch
        header_parts = [test_case]
        if branch:
            header_parts.append(f"[{branch}]")
        if gpu_type:
            header_parts.append(f"[{gpu_type}]")
        header = escape_html("  ".join(header_parts))

        # Latest data table (for "Show Latest Data" button)
        latest_nd = (
            max(
                new_data_entries,
                key=lambda d: _parse_timestamp(d.get("ts_created") or d.get("@timestamp", 0)),
            )
            if new_data_entries
            else {}
        )
        latest_table = _build_data_table_html(latest_nd) if latest_nd else ""
        detail_id = f"latest-detail-{section_idx}"

        section = f"""
        <details class="test-section" open>
            <summary><strong>{header}</strong></summary>
            <div class="charts-grid">
                {"".join(charts)}
            </div>
            <div class="latest-data-section">
                <button class="show-data-btn" onclick="toggleLatestData('{detail_id}')">
                    Show Latest Data
                </button>
                <div id="{detail_id}" class="latest-data-content" style="display:none;">
                    {latest_table}
                </div>
            </div>
        </details>
        """
        sections_html.append(section)
        section_idx += 1

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
        .baseline-info {{
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 10px 15px;
            margin-bottom: 15px;
        }}
        .baseline-info summary {{
            cursor: pointer;
            font-size: 13px;
            font-weight: bold;
            color: #555;
        }}
        .baseline-info ul {{
            margin: 8px 0 0 0;
            padding-left: 20px;
            font-size: 13px;
            color: #555;
            line-height: 1.6;
        }}
        .latest-data-section {{
            margin-top: 10px;
        }}
        .show-data-btn {{
            display: inline-block;
            padding: 4px 12px;
            font-size: 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background: #fff;
            color: #555;
            cursor: pointer;
        }}
        .show-data-btn:hover {{
            border-color: #4285f4;
            color: #4285f4;
        }}
        .latest-data-content {{
            margin-top: 8px;
        }}
        .kv-table {{
            border-collapse: collapse;
            font-size: 12px;
            font-family: monospace;
            max-width: 100%;
        }}
        .kv-table td {{
            padding: 2px 8px;
            border-bottom: 1px solid #eee;
            word-break: break-all;
        }}
        .kv-table td:first-child {{
            color: #555;
            white-space: nowrap;
            font-weight: bold;
            width: 1%;
        }}
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
    <h2>Perf Sanity Pre-Merge Results</h2>
    <p class="summary-info">{len(new_groups)} test case(s) &middot; {total_new} new data point(s)</p>

    <details class="baseline-info">
        <summary>Baseline Calculation</summary>
        <ul>
            <li>Post-merge data is queried from OpenSearch for the last {QUERY_LOOKBACK_DAYS} days.</li>
            <li>Rolling smooth (trailing window = 3 data points) applied to raw values (no daily aggregation).</li>
            <li>Baseline = P95 (95th percentile) of smoothed values.</li>
            <li>Pre-merge threshold is read from baseline data or defaults to {int(DEFAULT_THRESHOLD * 100)}%.</li>
        </ul>
    </details>

    <div id="point-popup">
        <div class="popup-header">
            <span>Data Point Details</span>
            <button class="popup-close" onclick="closePopup()">&times;</button>
        </div>
        <div id="popup-body"></div>
    </div>

    {"".join(sections_html)}

    <script>
    (function() {{
        // --- Show Latest Data toggle ---
        window.toggleLatestData = function(id) {{
            var el = document.getElementById(id);
            if (!el) return;
            if (el.style.display === "none") {{
                el.style.display = "block";
            }} else {{
                el.style.display = "none";
            }}
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
                    var ek = String(keys[k]).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
                    var ev = String(val).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
                    rows += "<tr><td>" + ek + "</td><td>" + ev + "</td></tr>";
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
    }})();
    </script>
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

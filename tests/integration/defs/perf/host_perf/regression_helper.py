# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Soft regression detection and OpenSearch DB collection for module-level host perf tests.

Loads baseline medians from baselines_b200.json and compares against
current measurements. Prints a REGRESSION marker when the current median
exceeds baseline * regression_factor, but does NOT fail the test.

CI pipelines can grep for the REGRESSION marker to flag regressions
without blocking the merge.

Also provides ``collect_module_result()`` for accumulating per-scenario
latency stats that are uploaded to OpenSearch at session end via
``post_module_perf_to_db()``.
"""

import json
import os
import statistics
import subprocess

_BASELINES_FILE = os.path.join(os.path.dirname(__file__), "baselines_b200.json")
_baselines_cache = None

# Whether to upload results to OpenSearch DB.
# Auto-detected from OPEN_SEARCH_DB_BASE_URL: set on CI, absent locally.
UPLOAD_TO_DB = bool(os.environ.get("OPEN_SEARCH_DB_BASE_URL", ""))

SUPPORTED_GPU_MAPPING = {
    "GB200": "gb200",
    "GB300": "gb300",
    "B200": "b200",
    "B300": "b300",
    "H200": "h200",
}

# Model PATH of local dir synced from internal LLM models repo.
MODEL_PATH_DICT = {
    "deepseek_v3_lite_fp8": "DeepSeek-V3-Lite/fp8",
    "llama_v3.1_8b_instruct": "llama-3.1-model/Llama-3.1-8B-Instruct",
}

# Match keys for OpenSearch historical data lookup.
# Module perf tests are identified by GPU type, runtime, component, and scenario.
MODULE_PERF_MATCH_KEYS = [
    "s_gpu_type",
    "s_runtime",
    "s_component",
    "s_scenario_name",
]

# Module-level latency metrics (all minimize — lower is better).
# These are custom metrics not in the standard MAXIMIZE/MINIMIZE sets in
# open_search_db_utils, so regression detection is handled locally.
MODULE_MINIMIZE_METRICS = [
    "d_median_us",
    "d_mean_us",
    "d_p99_us",
]

# Module-level regression is checked against d_median_us only.
MODULE_REGRESSION_METRICS = [
    "d_median_us",
]

# Default thresholds for module perf regression (tighter than serve tests
# because module tests have lower variance).
MODULE_POST_MERGE_THRESHOLD = 0.10  # 10%
MODULE_PRE_MERGE_THRESHOLD = 0.20  # 20%

# ---------------------------------------------------------------------------
# Module-level result collector (populated by collect_module_result)
# ---------------------------------------------------------------------------

# Dict of "component/scenario" -> {component, scenario_name, d_median_us, ...}
_module_results = {}


def _load_baselines():
    global _baselines_cache
    if _baselines_cache is None:
        try:
            with open(_BASELINES_FILE) as f:
                _baselines_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _baselines_cache = {}
    return _baselines_cache


def check_regression(component: str, scenario_name: str, current_median_us: float):
    """Check if the current median exceeds the baseline by more than the regression factor.

    Args:
        component: One of "scheduler", "sampler", "kv_cache".
        scenario_name: The scenario ID matching a key in baselines_b200.json.
        current_median_us: The measured median latency in microseconds.

    Returns:
        True if a regression was detected, False otherwise.
    """
    baselines = _load_baselines()
    factor = baselines.get("_regression_factor", 2.0)
    component_baselines = baselines.get(component, {})
    baseline_us = component_baselines.get(scenario_name)

    if baseline_us is None:
        print(f"  WARNING: No baseline for {component}/{scenario_name}, skipping regression check")
        return False

    threshold_us = baseline_us * factor
    if current_median_us > threshold_us:
        print(
            f"  REGRESSION DETECTED: {component}/{scenario_name} "
            f"median={current_median_us:.1f} µs > "
            f"threshold={threshold_us:.1f} µs "
            f"(baseline={baseline_us:.1f} µs × {factor}x)"
        )
        return True
    else:
        print(
            f"  OK: median={current_median_us:.1f} µs "
            f"<= threshold={threshold_us:.1f} µs "
            f"(baseline={baseline_us:.1f} µs × {factor}x)"
        )
        return False


def collect_module_result(component: str, scenario_name: str, latencies_us):
    """Store full latency stats for later OpenSearch DB upload.

    Call this alongside ``check_regression()`` in each module test to
    accumulate results.  At session end, ``post_module_perf_to_db()``
    uploads everything.

    Args:
        component: One of "scheduler", "sampler", "kv_cache".
        scenario_name: The scenario ID (e.g. "production_gen_only_bs32").
        latencies_us: Raw per-call latencies in microseconds.
    """
    sorted_lat = sorted(latencies_us)
    key = f"{component}/{scenario_name}"
    _module_results[key] = {
        "component": component,
        "scenario_name": scenario_name,
        "d_median_us": statistics.median(sorted_lat),
        "d_mean_us": statistics.mean(sorted_lat),
        "d_p99_us": sorted_lat[int(len(sorted_lat) * 0.99)],
        "d_p999_us": sorted_lat[int(len(sorted_lat) * 0.999)],
        "d_min_us": sorted_lat[0],
        "d_max_us": sorted_lat[-1],
        "l_num_samples": len(sorted_lat),
    }


def get_collected_results():
    """Return the accumulated module results dict."""
    return _module_results


def get_gpu_type() -> str:
    """Detect the GPU type from nvidia-smi."""
    try:
        output = subprocess.check_output(
            "nvidia-smi -q | grep 'Product Name' | head -1",
            shell=True,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        model = output.split()[-1]
        return SUPPORTED_GPU_MAPPING.get(model, "unsupported")
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return "unknown"


def get_model_dir(model_name: str) -> str:
    """Get model directory path from model name."""
    from ...conftest import llm_models_root

    if model_name in MODEL_PATH_DICT:
        return os.path.join(llm_models_root(), MODEL_PATH_DICT[model_name])
    return ""


def report_latencies(label: str, name: str, latencies_us) -> float:
    """Print latency statistics and return median.

    Args:
        label: Header label (e.g. "SCHEDULER_PERF", "KV_CACHE_PERF").
        name: Scenario name.
        latencies_us: List of per-call latencies in microseconds.
    """
    sorted_lat = sorted(latencies_us)
    mean = statistics.mean(sorted_lat)
    median = statistics.median(sorted_lat)
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
    p999 = sorted_lat[int(len(sorted_lat) * 0.999)]
    min_lat = sorted_lat[0]
    max_lat = sorted_lat[-1]

    print(f"\n{label}: {name}")
    print(f"  calls:  {len(latencies_us)}")
    print(f"  mean:   {mean:.1f} µs")
    print(f"  median: {median:.1f} µs")
    print(f"  P99:    {p99:.1f} µs")
    print(f"  P99.9:  {p999:.1f} µs")
    print(f"  min:    {min_lat:.1f} µs")
    print(f"  max:    {max_lat:.1f} µs")
    return median


def _check_module_regression(new_data, history_baseline, is_post_merge):
    """Check module-level regression against historical baseline.

    Unlike the standard pipeline which checks throughput metrics,
    module tests check latency metrics (lower is better).
    """
    if history_baseline is None:
        return False, ""

    info_parts = [
        f"baseline_id: {history_baseline.get('_id', '')}",
        f"baseline_commit: {history_baseline.get('s_commit', '')}",
    ]
    regressive = False

    threshold = MODULE_POST_MERGE_THRESHOLD if is_post_merge else MODULE_PRE_MERGE_THRESHOLD

    for metric in MODULE_REGRESSION_METRICS:
        if metric not in new_data or metric not in history_baseline:
            continue
        baseline_val = history_baseline[metric]
        new_val = new_data[metric]
        if baseline_val == 0:
            continue

        diff_pct = (new_val - baseline_val) / baseline_val * 100
        metric_info = (
            f"{metric}: {new_val:.1f} vs baseline {baseline_val:.1f} "
            f"({diff_pct:+.1f}%, threshold: {threshold * 100:.0f}%)"
        )
        info_parts.append(metric_info)

        # Regressive if new_val > baseline * (1 + threshold)
        if new_val > baseline_val * (1 + threshold):
            regressive = True

    return regressive, ", ".join(info_parts)


def post_module_perf_to_db(test_output_dir: str):
    """Upload accumulated module perf results to OpenSearch DB.

    Follows the same overall pattern as test_host_perf.py but with
    custom regression detection for latency-in-microseconds metrics.

    Args:
        test_output_dir: Directory for output files (perf_data.yaml).
    """
    from defs.trt_test_alternative import print_info

    from ..open_search_db_utils import (
        add_id,
        generate_perf_yaml,
        get_common_values,
        get_history_data,
        get_job_info,
        post_new_perf_data,
        prepare_baseline_data,
    )

    results = get_collected_results()
    if not results:
        print_info("[module_perf_db] No results to upload.")
        return

    gpu_type = get_gpu_type()
    job_config = get_job_info()
    is_post_merge = job_config["b_is_post_merge"]

    new_data_dict = {}
    cmd_idx = 0

    for key, stats in results.items():
        new_data = {
            "s_gpu_type": gpu_type,
            "s_runtime": "module_perf",
            "s_component": stats["component"],
            "s_scenario_name": stats["scenario_name"],
        }
        new_data.update(job_config)

        # Add latency metrics (already have d_ prefix)
        for field, value in stats.items():
            if field.startswith("d_") or field.startswith("l_"):
                new_data[field] = value

        add_id(new_data)
        new_data_dict[cmd_idx] = new_data
        cmd_idx += 1

    if not new_data_dict:
        print_info("[module_perf_db] No valid data to upload.")
        return

    match_keys = MODULE_PERF_MATCH_KEYS

    common_values_dict = get_common_values(new_data_dict, match_keys)

    history_baseline_dict, history_data_dict = get_history_data(
        new_data_dict, match_keys, common_values_dict
    )

    # Custom regression detection for module-level latency metrics
    for idx, new_data in new_data_dict.items():
        history_baseline = history_baseline_dict.get(idx) if history_baseline_dict else None
        is_regressive, info = _check_module_regression(new_data, history_baseline, is_post_merge)
        new_data["b_is_regression"] = is_regressive
        new_data["s_regression_info"] = info

        scenario = new_data.get("s_scenario_name", f"cmd_{idx}")
        component = new_data.get("s_component", "")
        if is_regressive:
            print_info(f"[module_perf_db] REGRESSION {component}/{scenario}: {info}")
        else:
            print_info(f"[module_perf_db] OK {component}/{scenario}")

    # Baseline calculation — prepare_baseline_data only smooths standard
    # MAXIMIZE+MINIMIZE metrics.  We patch in smoothed module-specific latency
    # metrics afterwards so baselines reflect rolling history, not just the
    # last single run.
    if is_post_merge and history_baseline_dict is not None:
        new_baseline_data_dict = prepare_baseline_data(
            history_baseline_dict, history_data_dict, new_data_dict
        )
        # Patch module-specific latency metrics into baselines with
        # rolling-smooth + P5 (lower is better for all module metrics).
        if new_baseline_data_dict:
            from ..open_search_db_utils import _daily_aggregate_values, _percentile, _rolling_smooth

            for idx, baseline in new_baseline_data_dict.items():
                history_list = history_data_dict.get(idx, [])
                new_data = new_data_dict[idx]
                all_data = list(history_list) + [new_data]
                for metric in MODULE_MINIMIZE_METRICS:
                    daily_vals = _daily_aggregate_values(all_data, metric)
                    if not daily_vals:
                        continue
                    smoothed = _rolling_smooth(daily_vals, window=3)
                    baseline[metric] = _percentile(smoothed, 5)
                # Add module-specific thresholds to baseline
                for metric in MODULE_MINIMIZE_METRICS:
                    suffix = metric[2:]  # strip "d_"
                    baseline.setdefault(
                        f"d_threshold_post_merge_{suffix}",
                        MODULE_POST_MERGE_THRESHOLD,
                    )
                    baseline.setdefault(
                        f"d_threshold_pre_merge_{suffix}",
                        MODULE_PRE_MERGE_THRESHOLD,
                    )
    else:
        new_baseline_data_dict = None

    if UPLOAD_TO_DB:
        post_new_perf_data(new_baseline_data_dict, new_data_dict)

    os.makedirs(test_output_dir, exist_ok=True)
    generate_perf_yaml(new_data_dict, output_dir=test_output_dir)

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

from ..test_perf_sanity import SUPPORTED_GPU_MAPPING

_BASELINES_FILE = os.path.join(os.path.dirname(__file__), "baselines_b200.json")
_baselines_cache = None

# Whether to upload results to OpenSearch DB.
# Auto-detected from OPEN_SEARCH_DB_BASE_URL: set on CI, absent locally.
UPLOAD_TO_DB = bool(os.environ.get("OPEN_SEARCH_DB_BASE_URL", ""))

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


def _compute_latency_stats(latencies_us):
    """Compute latency statistics from raw measurements.

    Returns:
        Tuple of (sorted_latencies, stats_dict) where stats_dict contains
        median, mean, p99, p999, min, max, and count.
    """
    sorted_lat = sorted(latencies_us)
    return sorted_lat, {
        "d_median_us": statistics.median(sorted_lat),
        "d_mean_us": statistics.mean(sorted_lat),
        "d_p99_us": sorted_lat[int(len(sorted_lat) * 0.99)],
        "d_p999_us": sorted_lat[int(len(sorted_lat) * 0.999)],
        "d_min_us": sorted_lat[0],
        "d_max_us": sorted_lat[-1],
        "l_num_samples": len(sorted_lat),
    }


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
    _, stats = _compute_latency_stats(latencies_us)
    key = f"{component}/{scenario_name}"
    _module_results[key] = {
        "component": component,
        "scenario_name": scenario_name,
        **stats,
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
    _, stats = _compute_latency_stats(latencies_us)

    print(f"\n{label}: {name}")
    print(f"  calls:  {len(latencies_us)}")
    print(f"  mean:   {stats['d_mean_us']:.1f} µs")
    print(f"  median: {stats['d_median_us']:.1f} µs")
    print(f"  P99:    {stats['d_p99_us']:.1f} µs")
    print(f"  P99.9:  {stats['d_p999_us']:.1f} µs")
    print(f"  min:    {stats['d_min_us']:.1f} µs")
    print(f"  max:    {stats['d_max_us']:.1f} µs")
    return stats["d_median_us"]


def post_module_perf_to_db():
    """Upload accumulated module perf results to OpenSearch DB.

    Uses the generic process_and_upload_test_results pipeline from
    perf_regression_utils for history lookup, regression detection,
    baseline calculation, and upload.
    """
    from defs.trt_test_alternative import print_info

    from ..perf_regression_utils import process_and_upload_test_results

    results = get_collected_results()
    if not results:
        print_info("[module_perf_db] No results to upload.")
        return

    gpu_type = get_gpu_type()

    new_data_dict = {}
    cmd_idx = 0

    for key, stats in results.items():
        new_data = {
            "s_gpu_type": gpu_type,
            "s_runtime": "module_perf",
            "s_component": stats["component"],
            "s_scenario_name": stats["scenario_name"],
        }

        # Add latency metrics (already have d_ prefix)
        for field, value in stats.items():
            if field.startswith("d_") or field.startswith("l_"):
                new_data[field] = value

        new_data_dict[cmd_idx] = new_data
        cmd_idx += 1

    if not new_data_dict:
        print_info("[module_perf_db] No valid data to upload.")
        return

    process_and_upload_test_results(
        new_data_dict=new_data_dict,
        match_keys=MODULE_PERF_MATCH_KEYS,
        maximize_metrics=[],
        minimize_metrics=MODULE_MINIMIZE_METRICS,
        regression_metrics=MODULE_REGRESSION_METRICS,
        upload_to_db=UPLOAD_TO_DB,
        fail_on_regression=False,
    )

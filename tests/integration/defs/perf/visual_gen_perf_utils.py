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
"""Utilities for VisualGen perf sanity tests."""

from __future__ import annotations

import json
from typing import Any

from tensorrt_llm.commands.utils import get_visual_gen_num_gpus as resolve_visual_gen_num_gpus

MAXIMIZE_METRICS = [
    "d_request_throughput",
    "d_per_gpu_throughput",
]

MINIMIZE_METRICS = [
    "d_mean_e2e_latency",
    "d_median_e2e_latency",
    "d_p90_e2e_latency",
    "d_p99_e2e_latency",
]

REGRESSION_METRICS = [
    "d_mean_e2e_latency",
]

MATCH_KEYS = [
    "s_gpu_type",
    "s_runtime",
    "s_model_name",
    "l_gpus",
    "s_attn_backend",
    "s_quant_algo",
    "b_enable_teacache",
    "b_enable_cuda_graph",
    "b_enable_torch_compile",
    "b_enable_two_stage",
    "b_enable_parallel_vae",
    "l_cfg_size",
    "l_ulysses_size",
    "s_generation_mode",
    "s_backend",
    "s_size",
    "l_num_frames",
    "l_fps",
    "l_num_inference_steps",
    "l_max_concurrency",
]

RESULT_METRIC_PATHS = {
    "d_request_throughput": "request_throughput",
    "d_per_gpu_throughput": "per_gpu_throughput",
    "d_mean_e2e_latency": "mean_e2e_latency_ms",
    "d_median_e2e_latency": "median_e2e_latency_ms",
    "d_p90_e2e_latency": "percentiles_e2e_latency_ms.p90",
    "d_p99_e2e_latency": "percentiles_e2e_latency_ms.p99",
}


def _get_nested_value(data: dict[str, Any], path: str, default: Any = None) -> Any:
    """Return a nested value from a dict using dot-separated keys."""
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _infer_generation_mode(client_config: dict[str, Any]) -> str:
    """Infer the request mode for baseline bucketing."""
    explicit_mode = client_config.get("generation_mode")
    if explicit_mode:
        return str(explicit_mode)

    backend = str(client_config.get("backend", ""))
    extra_body = client_config.get("extra_body")
    if backend == "openai-images":
        return "t2i"

    if isinstance(extra_body, str):
        try:
            extra_body = json.loads(extra_body)
        except json.JSONDecodeError:
            extra_body = None

    if isinstance(extra_body, dict) and "input_reference" in extra_body:
        return "i2v"

    if backend == "openai-videos":
        return "t2v"

    return backend or "unknown"


def get_visual_gen_num_gpus_from_server_config(server_config: dict[str, Any]) -> int:
    """Compute the expected GPU count from the VisualGen server config."""
    return int(resolve_visual_gen_num_gpus(server_config))


def extract_visual_gen_metrics(result_data: dict[str, Any]) -> dict[str, float]:
    """Extract OpenSearch metric fields from a benchmark result JSON."""
    metrics: dict[str, float] = {}
    missing_paths: list[str] = []

    for metric_name, path in RESULT_METRIC_PATHS.items():
        value = _get_nested_value(result_data, path)
        if value is None:
            missing_paths.append(path)
            continue
        metrics[metric_name] = float(value)

    if missing_paths:
        missing = ", ".join(sorted(missing_paths))
        raise ValueError(f"Missing VisualGen benchmark metrics in result JSON: {missing}")

    return metrics


def get_visual_gen_match_keys() -> list[str]:
    """Return the match keys used for baseline/regression lookup."""
    return MATCH_KEYS.copy()


def build_visual_gen_db_entry(
    *,
    gpu_type: str,
    model_name: str,
    server_name: str,
    server_config: dict[str, Any],
    client_config: dict[str, Any],
    result_data: dict[str, Any],
    extra_visual_gen_options_path: str = "",
) -> dict[str, Any]:
    """Build one OpenSearch document from VisualGen config and result JSON."""
    expected_num_gpus = get_visual_gen_num_gpus_from_server_config(server_config)
    result_num_gpus = int(result_data.get("num_gpus", expected_num_gpus))
    if result_num_gpus != expected_num_gpus:
        raise ValueError(
            "Benchmark result GPU count mismatch: "
            f"result={result_num_gpus}, expected={expected_num_gpus}"
        )

    client_name = str(client_config.get("name", "default"))
    entry = {
        "s_runtime": "visual_gen",
        "s_gpu_type": gpu_type,
        "s_model_name": str(model_name).lower(),
        "s_server_name": server_name,
        "l_gpus": expected_num_gpus,
        "s_extra_visual_gen_options_path": str(extra_visual_gen_options_path),
        "s_attn_backend": str(_get_nested_value(server_config, "attention.backend", "")),
        "s_quant_algo": str(_get_nested_value(server_config, "quant_config.quant_algo", "")),
        "b_enable_teacache": _get_nested_value(server_config, "cache.cache_backend", "")
        == "teacache",
        "b_enable_cuda_graph": bool(
            _get_nested_value(server_config, "cuda_graph.enable_cuda_graph", False)
        ),
        "b_enable_torch_compile": bool(
            _get_nested_value(server_config, "torch_compile.enable_torch_compile", False)
        ),
        "b_enable_two_stage": bool(
            server_config.get("spatial_upsampler_path") or server_config.get("distilled_lora_path")
        ),
        "b_enable_parallel_vae": bool(
            _get_nested_value(server_config, "parallel.enable_parallel_vae", False)
        ),
        "l_cfg_size": int(_get_nested_value(server_config, "parallel.dit_cfg_size", 1)),
        "l_ulysses_size": int(_get_nested_value(server_config, "parallel.dit_ulysses_size", 1)),
        "s_generation_mode": _infer_generation_mode(client_config),
        "s_backend": str(client_config.get("backend")),
        "s_size": str(client_config.get("size")),
        "l_num_frames": int(client_config.get("num_frames")),
        "l_fps": int(client_config.get("fps")),
        "l_num_inference_steps": int(client_config.get("num_inference_steps")),
        "l_max_concurrency": int(client_config.get("max_concurrency")),
        "s_test_case_name": f"{server_name}-{client_name}",
    }
    entry.update(extract_visual_gen_metrics(result_data))
    return entry

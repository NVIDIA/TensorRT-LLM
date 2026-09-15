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
    "d_mean_latency",
    "d_median_latency",
    "d_p90_latency",
    "d_p99_latency",
    "d_mean_generation",
    "d_median_generation",
    "d_p90_generation",
    "d_p99_generation",
]

REGRESSION_METRICS = [
    "d_median_generation",
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
    "l_cfg_size",
    "l_ulysses_size",
    "l_parallel_vae_size",
    "s_generation_mode",
    "s_backend",
    "s_size",
    "l_num_frames",
    "l_fps",
    "l_num_inference_steps",
    "l_max_concurrency",
]


def result_metric_paths(backend: str) -> dict[str, str]:
    """Where each gated metric lives in the result JSON, per route.

    Both series are client-side wall clock, so what they measure is what a
    caller of the server waits for. The latency series is the whole request.
    The generation series ends once generation is done: the video route
    reports that separately as ``gen_latency``, and the image route, being a
    single leg, cannot tell it apart from the whole request.
    """
    generation = "gen_latency" if backend == "openai-videos" else "e2e_latency"
    return {
        "d_request_throughput": "request_throughput",
        "d_mean_latency": "e2e_latency.mean",
        "d_median_latency": "e2e_latency.median",
        "d_p90_latency": "e2e_latency.percentiles.p90",
        "d_p99_latency": "e2e_latency.percentiles.p99",
        "d_mean_generation": f"{generation}.mean",
        "d_median_generation": f"{generation}.median",
        "d_p90_generation": f"{generation}.percentiles.p90",
        "d_p99_generation": f"{generation}.percentiles.p99",
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

    if isinstance(extra_body, dict) and (
        "image_reference" in extra_body or "video_reference" in extra_body
    ):
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

    for metric_name, path in result_metric_paths(str(result_data.get("backend", ""))).items():
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
    visual_gen_args_path: str = "",
) -> dict[str, Any]:
    """Build one OpenSearch document from VisualGen config and result JSON."""
    expected_num_gpus = get_visual_gen_num_gpus_from_server_config(server_config)
    client_name = str(client_config.get("name", "default"))
    entry = {
        "s_runtime": "visual_gen",
        "s_gpu_type": gpu_type,
        "s_model_name": str(model_name).lower(),
        "s_server_name": server_name,
        "l_gpus": expected_num_gpus,
        "s_visual_gen_args_path": str(visual_gen_args_path),
        "s_attn_backend": str(_get_nested_value(server_config, "attention_config.backend", "")),
        "s_quant_algo": str(_get_nested_value(server_config, "quant_config.quant_algo", "")),
        "b_enable_teacache": _get_nested_value(server_config, "cache_config.cache_backend", "")
        == "teacache",
        "b_enable_cuda_graph": bool(
            _get_nested_value(server_config, "cuda_graph_config.enable", False)
        ),
        "b_enable_torch_compile": bool(
            _get_nested_value(server_config, "torch_compile_config.enable", False)
        ),
        "b_enable_two_stage": bool(
            _get_nested_value(server_config, "pipeline_config.spatial_upsampler_path", None)
            or _get_nested_value(server_config, "pipeline_config.distilled_lora_path", None)
        ),
        "l_cfg_size": int(_get_nested_value(server_config, "parallel_config.cfg_size", 1)),
        "l_ulysses_size": int(_get_nested_value(server_config, "parallel_config.ulysses_size", 1)),
        "l_parallel_vae_size": int(
            _get_nested_value(server_config, "parallel_config.parallel_vae_size", 1)
        ),
        "s_generation_mode": _infer_generation_mode(client_config),
        "s_backend": str(client_config.get("backend")),
        "s_size": str(client_config.get("size")),
        # An image config states neither: the document rejects num_frames on an
        # image route, so the config cannot carry it. Both stay match keys, and
        # 1 keeps an image case in the bucket its baselines were recorded under.
        "l_num_frames": int(client_config.get("num_frames") or 1),
        "l_fps": int(client_config.get("fps") or 1),
        "l_num_inference_steps": int(client_config.get("num_inference_steps")),
        "l_max_concurrency": int(client_config.get("max_concurrency")),
        "s_test_case_name": f"{server_name}-{client_name}",
    }
    entry.update(extract_visual_gen_metrics(result_data))
    # The client does not know the topology, so per-GPU throughput is derived here.
    entry["d_per_gpu_throughput"] = entry["d_request_throughput"] / expected_num_gpus
    return entry

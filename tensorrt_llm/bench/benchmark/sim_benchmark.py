# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Simulation benchmark runner for trtllm-bench --sim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from tensorrt_llm.bench.dataclasses.general import InferenceRequest
from tensorrt_llm.llmapi import LLM, SamplingParams
from tensorrt_llm.llmapi.sim_config import PredictorConfig, SimConfig
from tensorrt_llm.logger import logger


def load_sim_config(sim_config_path: Optional[Path] = None) -> SimConfig:
    """Load SimConfig from YAML file or return default constant predictor.

    Args:
        sim_config_path: Path to YAML file with predictor config.
            If None, returns constant predictor defaults.

    Returns:
        SimConfig instance.
    """
    if sim_config_path is None:
        logger.info("[SimBench] Using default constant predictor "
                    "(10ms prefill, 5ms decode)")
        return SimConfig()

    logger.info(f"[SimBench] Loading sim config from {sim_config_path}")
    with open(sim_config_path) as f:
        data = yaml.safe_load(f)
    predictor_data = data.get("predictor", data)
    return SimConfig(predictor=PredictorConfig(**predictor_data))


def run_sim_benchmark(
    model: str,
    requests: List[InferenceRequest],
    sim_config: SimConfig,
    sampling_params: SamplingParams,
    tp_size: int = 1,
    pp_size: int = 1,
    max_batch_size: Optional[int] = None,
    max_num_tokens: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    kv_cache_free_gpu_mem_fraction: float = 0.90,
) -> dict:
    """Run simulation benchmark and return metrics.

    Args:
        model: HuggingFace model path.
        requests: List of InferenceRequest from dataset.
        sim_config: Simulation configuration.
        sampling_params: Sampling parameters (max_tokens from dataset).
        tp_size: Tensor parallel size.
        pp_size: Pipeline parallel size.
        max_batch_size: Max batch size override.
        max_num_tokens: Max num tokens override.
        max_seq_len: Max sequence length override.
        kv_cache_free_gpu_mem_fraction: KV cache memory fraction.

    Returns:
        Metrics dict from SimClock.
    """
    llm_kwargs = {
        "model": model,
        "sim_config": sim_config,
        "tensor_parallel_size": tp_size,
        "pipeline_parallel_size": pp_size,
        "kv_cache_config": {
            "free_gpu_memory_fraction": kv_cache_free_gpu_mem_fraction,
        },
        "skip_tokenizer_init": True,
    }
    if max_batch_size is not None:
        llm_kwargs["max_batch_size"] = max_batch_size
    if max_num_tokens is not None:
        llm_kwargs["max_num_tokens"] = max_num_tokens
    if max_seq_len is not None:
        llm_kwargs["max_seq_len"] = max_seq_len

    logger.info(f"[SimBench] Creating LLM (tp={tp_size}, pp={pp_size})")
    llm = LLM(**llm_kwargs)

    try:
        # Build prompts: use input_ids (tokenized) from dataset
        prompts = []
        output_lens = []
        for req in requests:
            if req.input_ids is not None:
                prompts.append(req.input_ids)
            else:
                prompts.append(req.prompt)
            output_lens.append(req.output_tokens)

        logger.info(f"[SimBench] Running {len(prompts)} requests")

        # Generate all at once — batch mode
        all_params = []
        for olen in output_lens:
            sp = SamplingParams(
                max_tokens=olen,
                end_id=sampling_params.end_id,
                pad_id=sampling_params.pad_id,
            )
            all_params.append(sp)

        outputs = llm.generate(prompts, sampling_params=all_params)

        logger.info(f"[SimBench] Generation complete, "
                    f"{len(outputs)} outputs")

        clock = sim_config._clock
        if clock is None:
            raise RuntimeError("SimClock not available — "
                               "sim mode may not have initialized correctly")

        metrics = clock.metrics
        # Add request count info
        metrics["num_requests"] = len(requests)
        return metrics

    finally:
        llm.shutdown()


def print_sim_report(metrics: dict, model: str, tp_size: int = 1,
                     predictor_desc: str = "constant") -> None:
    """Print formatted simulation benchmark report.

    Matches trtllm-bench throughput output format with [SIM] banner.
    """
    print("===========================================================")
    print("= SIM THROUGHPUT BENCHMARK RESULTS")
    print("===========================================================")
    print("[METADATA]")
    print(f"  Model:                  {model}")
    print(f"  TP:                     {tp_size}")
    print(f"  Predictor:              {predictor_desc}")
    print(f"  Num Requests:           {metrics.get('num_requests', metrics.get('completed', 0))}")
    print()
    print("[RESULTS]")
    print(f"  Completed Requests:     {metrics.get('completed', 0)}")
    print(f"  Total Input Tokens:     {metrics.get('total_input', 0)}")
    print(f"  Total Output Tokens:    {metrics.get('total_output', 0)}")
    print(f"  Simulated Duration (s): {metrics.get('duration', 0):.4f}")
    print(f"  Request Throughput (req/s): {metrics.get('request_throughput', 0):.2f}")
    print(f"  Output Throughput (tok/s):  {metrics.get('output_throughput', 0):.2f}")
    print()
    print(f"  Mean TTFT (ms):         {metrics.get('mean_ttft_ms', 0):.2f}")
    print(f"  P50 TTFT (ms):          {metrics.get('p50_ttft_ms', 0):.2f}")
    print(f"  P95 TTFT (ms):          {metrics.get('p95_ttft_ms', 0):.2f}")
    print(f"  P99 TTFT (ms):          {metrics.get('p99_ttft_ms', 0):.2f}")
    print()
    print(f"  Mean TPOT (ms):         {metrics.get('mean_tpot_ms', 0):.2f}")
    print(f"  P50 TPOT (ms):          {metrics.get('p50_tpot_ms', 0):.2f}")
    print(f"  P95 TPOT (ms):          {metrics.get('p95_tpot_ms', 0):.2f}")
    print(f"  P99 TPOT (ms):          {metrics.get('p99_tpot_ms', 0):.2f}")
    print()
    print(f"  Mean ITL (ms):          {metrics.get('mean_itl_ms', 0):.2f}")
    print(f"  P95 ITL (ms):           {metrics.get('p95_itl_ms', 0):.2f}")
    print(f"  P99 ITL (ms):           {metrics.get('p99_itl_ms', 0):.2f}")
    print()
    print(f"  Mean E2E Latency (ms):  {metrics.get('mean_e2e_latency_ms', 0):.2f}")
    print(f"  P95 E2E Latency (ms):   {metrics.get('p95_e2e_latency_ms', 0):.2f}")
    print(f"  P99 E2E Latency (ms):   {metrics.get('p99_e2e_latency_ms', 0):.2f}")
    print("===========================================================")


def write_sim_report(metrics: dict, report_path: Path) -> None:
    """Write metrics to JSON file."""
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"[SimBench] Report written to {report_path}")


def write_sim_request_report(sim_config: SimConfig,
                              request_path: Path) -> None:
    """Write per-request data to JSON file."""
    clock = sim_config._clock
    if clock is None:
        return
    with open(request_path, "w") as f:
        for s in clock.request_stats.values():
            f.write(json.dumps({
                "request_id": s.request_id,
                "input_length": s.input_length,
                "output_length": s.output_length,
                "ttft_ms": s.ttft_s * 1000,
                "tpot_ms": s.tpot_s * 1000,
                "e2e_ms": s.e2e_s * 1000,
            }) + "\n")
    logger.info(f"[SimBench] Request report written to {request_path}")

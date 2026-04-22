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
"""Shared utilities for VisualGen benchmarking (online and offline)."""

import json
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

SECONDS_TO_MILLISECONDS = 1000


@dataclass
class VisualGenSampleRequest:
    """A single prompt for visual generation benchmarking."""

    prompt: str


@dataclass
class VisualGenRequestOutput:
    """Timing and status result for a single visual generation request."""

    success: bool = False
    e2e_latency: float = 0.0
    ttff: float = -1.0
    gen_fps: float = -1.0
    error: str = ""
    exception_type: Optional[str] = None


@dataclass
class VisualGenBenchmarkMetrics:
    """Aggregated benchmark metrics across all requests."""

    completed: int
    total_requests: int
    request_throughput: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    min_e2e_latency_ms: float
    max_e2e_latency_ms: float
    percentiles_e2e_latency_ms: list[tuple[float, float]]
    num_gpus: int = 1
    per_gpu_throughput: float = 0.0
    mean_ttff_ms: float = -1.0
    mean_gen_fps: float = -1.0


def calculate_metrics(
    outputs: list[VisualGenRequestOutput],
    dur_s: float,
    selected_percentiles: list[float],
    num_gpus: int = 1,
) -> VisualGenBenchmarkMetrics:
    """Compute aggregate metrics from per-request outputs."""
    e2e_latencies: list[float] = []
    error_counts: dict[str, int] = {}
    completed = 0

    for out in outputs:
        if out.exception_type:
            error_counts[out.exception_type] = error_counts.get(out.exception_type, 0) + 1
        if out.success:
            e2e_latencies.append(out.e2e_latency)
            completed += 1

    total_error_count = sum(error_counts.values())
    for exception_type, count in error_counts.items():
        print(f"Error type: {exception_type}, Count: {count} requests")
    if total_error_count:
        print(f"Total failed requests: {total_error_count}")

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )

    e2e_ms = [v * SECONDS_TO_MILLISECONDS for v in e2e_latencies]

    request_throughput = completed / dur_s if dur_s > 0 else 0
    return VisualGenBenchmarkMetrics(
        completed=completed,
        total_requests=len(outputs),
        request_throughput=request_throughput,
        mean_e2e_latency_ms=float(np.mean(e2e_ms)) if e2e_ms else 0,
        median_e2e_latency_ms=float(np.median(e2e_ms)) if e2e_ms else 0,
        std_e2e_latency_ms=float(np.std(e2e_ms)) if e2e_ms else 0,
        min_e2e_latency_ms=float(np.min(e2e_ms)) if e2e_ms else 0,
        max_e2e_latency_ms=float(np.max(e2e_ms)) if e2e_ms else 0,
        percentiles_e2e_latency_ms=(
            [(p, float(np.percentile(e2e_ms, p))) for p in selected_percentiles]
            if e2e_ms
            else [(p, 0.0) for p in selected_percentiles]
        ),
        num_gpus=num_gpus,
        per_gpu_throughput=request_throughput / num_gpus,
    )


def print_visual_gen_results(
    backend: str,
    model_id: str,
    benchmark_duration: float,
    metrics: VisualGenBenchmarkMetrics,
) -> None:
    """Print benchmark results to stdout."""
    print("{s:{c}^{n}}".format(s=" Benchmark Result (VisualGen) ", n=60, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Model:", model_id))
    print("{:<40} {:<10}".format("Total requests:", metrics.total_requests))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.total_requests - metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10.4f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10}".format("Number of GPUs:", metrics.num_gpus))
    print("{:<40} {:<10.4f}".format("Per-GPU throughput (req/s/GPU):", metrics.per_gpu_throughput))

    if metrics.total_requests - metrics.completed > 0:
        print("=" * 60)
        print(
            f"  !!! {metrics.total_requests - metrics.completed} "
            "FAILED REQUESTS - CHECK LOG FOR ERRORS !!!"
        )
        print("=" * 60)

    print("{s:{c}^{n}}".format(s=" E2E Latency ", n=60, c="-"))
    print("{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("Median E2E Latency (ms):", metrics.median_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("Std Dev E2E Latency (ms):", metrics.std_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("Min E2E Latency (ms):", metrics.min_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("Max E2E Latency (ms):", metrics.max_e2e_latency_ms))
    for p, v in metrics.percentiles_e2e_latency_ms:
        p_word = str(int(p)) if int(p) == p else str(p)
        print("{:<40} {:<10.2f}".format(f"P{p_word} E2E Latency (ms):", v))

    print("{s:{c}^{n}}".format(s=" Placeholder Metrics ", n=60, c="-"))
    print("{:<40} {:<10}".format("TTFF (ms):", "N/A (placeholder)"))
    print("{:<40} {:<10}".format("GenFPS:", "N/A (placeholder)"))
    print("=" * 60)


def load_visual_gen_prompts(
    prompt: Optional[str],
    prompt_file: Optional[str],
    num_prompts: int,
) -> list[VisualGenSampleRequest]:
    """Load prompts from a single string or a file.

    Args:
        prompt: Single text prompt (repeated to fill num_prompts).
        prompt_file: Path to prompt file. Supports plain text (one per line)
            or JSONL with ``text`` / ``prompt`` field.
        num_prompts: Number of prompts to return.

    Returns:
        List of ``VisualGenSampleRequest`` of length *num_prompts*.
    """
    prompts: list[str] = []

    if prompt_file:
        with open(prompt_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    prompts.append(data.get("text", data.get("prompt", line)))
                except json.JSONDecodeError:
                    prompts.append(line)
    elif prompt:
        prompts.append(prompt)
    else:
        raise ValueError("Either prompt or prompt_file must be specified.")

    if len(prompts) < num_prompts:
        repeats = (num_prompts // len(prompts)) + 1
        prompts = (prompts * repeats)[:num_prompts]
    else:
        prompts = prompts[:num_prompts]

    return [VisualGenSampleRequest(prompt=p) for p in prompts]


def build_visual_gen_result_dict(
    backend: str,
    model_id: str,
    benchmark_duration: float,
    metrics: VisualGenBenchmarkMetrics,
    outputs: list[VisualGenRequestOutput],
    gen_params: dict,
) -> dict:
    """Build the result dictionary for JSON serialization."""
    return {
        "backend": backend,
        "model": model_id,
        "duration": benchmark_duration,
        "num_gpus": metrics.num_gpus,
        "total_requests": metrics.total_requests,
        "completed": metrics.completed,
        "request_throughput": metrics.request_throughput,
        "per_gpu_throughput": metrics.per_gpu_throughput,
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
        "std_e2e_latency_ms": metrics.std_e2e_latency_ms,
        "min_e2e_latency_ms": metrics.min_e2e_latency_ms,
        "max_e2e_latency_ms": metrics.max_e2e_latency_ms,
        "percentiles_e2e_latency_ms": {
            f"p{int(p) if int(p) == p else p}": v for p, v in metrics.percentiles_e2e_latency_ms
        },
        "e2e_latencies": [out.e2e_latency for out in outputs],
        "errors": [out.error for out in outputs],
        "gen_params": gen_params,
    }

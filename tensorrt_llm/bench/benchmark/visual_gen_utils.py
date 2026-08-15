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


@dataclass
class VisualGenSampleRequest:
    """A single prompt for visual generation benchmarking."""

    prompt: str


@dataclass
class VisualGenRequestOutput:
    """Timing and status result for a single visual generation request.

    All timings are wall-clock seconds. ``latency`` is the externally
    observed total measured around ``generate()`` (and the save step, if
    persisted by the caller). ``generation`` and ``denoise`` come from the
    engine-side ``VisualGenOutput.metrics`` and are populated when the
    request succeeds; ``generation`` is the engine's wall-clock around its
    inference call (no encode / persist / IPC).
    """

    success: bool = False
    latency: float = 0.0
    generation: float = 0.0
    denoise: float = 0.0
    ttff: float = -1.0
    gen_fps: float = -1.0
    error: str = ""
    exception_type: Optional[str] = None


@dataclass
class VisualGenBenchmarkMetrics:
    """Aggregated benchmark metrics across all requests.

    All ``*_latency`` and ``*_generation`` fields are wall-clock seconds.
    The relationship ``latency >= generation`` should hold per request;
    the gap is the encode + persist + IPC overhead the bench measures
    around the engine, and is the headroom available for overlap-style
    optimizations.
    """

    completed: int
    total_requests: int
    request_throughput: float
    mean_latency: float
    median_latency: float
    std_latency: float
    min_latency: float
    max_latency: float
    percentiles_latency: list[tuple[float, float]]
    mean_generation: float
    median_generation: float
    std_generation: float
    min_generation: float
    max_generation: float
    percentiles_generation: list[tuple[float, float]]
    num_gpus: int = 1
    per_gpu_throughput: float = 0.0
    mean_ttff: float = -1.0
    mean_gen_fps: float = -1.0


def calculate_metrics(
    outputs: list[VisualGenRequestOutput],
    dur_s: float,
    selected_percentiles: list[float],
    num_gpus: int = 1,
) -> VisualGenBenchmarkMetrics:
    """Compute aggregate metrics from per-request outputs."""
    latencies: list[float] = []
    # ``generation`` defaults to 0.0 and is only populated when the engine
    # supplied metrics (``result.metrics is not None`` in the bench loop).
    # Filter zeros so the aggregate reports only what was actually measured;
    # otherwise a backend that doesn't report metrics would push the mean
    # toward zero and obscure the latency comparison.
    generations: list[float] = []
    error_counts: dict[str, int] = {}
    completed = 0

    for out in outputs:
        if out.exception_type:
            error_counts[out.exception_type] = error_counts.get(out.exception_type, 0) + 1
        if out.success:
            latencies.append(out.latency)
            if out.generation > 0:
                generations.append(out.generation)
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

    def _pcts(samples: list[float]) -> list[tuple[float, float]]:
        if samples:
            return [(p, float(np.percentile(samples, p))) for p in selected_percentiles]
        return [(p, 0.0) for p in selected_percentiles]

    request_throughput = completed / dur_s if dur_s > 0 else 0
    return VisualGenBenchmarkMetrics(
        completed=completed,
        total_requests=len(outputs),
        request_throughput=request_throughput,
        mean_latency=float(np.mean(latencies)) if latencies else 0,
        median_latency=float(np.median(latencies)) if latencies else 0,
        std_latency=float(np.std(latencies)) if latencies else 0,
        min_latency=float(np.min(latencies)) if latencies else 0,
        max_latency=float(np.max(latencies)) if latencies else 0,
        percentiles_latency=_pcts(latencies),
        mean_generation=float(np.mean(generations)) if generations else 0,
        median_generation=float(np.median(generations)) if generations else 0,
        std_generation=float(np.std(generations)) if generations else 0,
        min_generation=float(np.min(generations)) if generations else 0,
        max_generation=float(np.max(generations)) if generations else 0,
        percentiles_generation=_pcts(generations),
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

    print("{s:{c}^{n}}".format(s=" Latency ", n=60, c="-"))
    print("{:<40} {:<10.4f}".format("Mean Latency (s):", metrics.mean_latency))
    print("{:<40} {:<10.4f}".format("Median Latency (s):", metrics.median_latency))
    print("{:<40} {:<10.4f}".format("Std Dev Latency (s):", metrics.std_latency))
    print("{:<40} {:<10.4f}".format("Min Latency (s):", metrics.min_latency))
    print("{:<40} {:<10.4f}".format("Max Latency (s):", metrics.max_latency))
    for p, v in metrics.percentiles_latency:
        p_word = str(int(p)) if int(p) == p else str(p)
        print("{:<40} {:<10.4f}".format(f"P{p_word} Latency (s):", v))

    print("{s:{c}^{n}}".format(s=" Generation ", n=60, c="-"))
    print("{:<40} {:<10.4f}".format("Mean Generation (s):", metrics.mean_generation))
    print("{:<40} {:<10.4f}".format("Median Generation (s):", metrics.median_generation))
    print("{:<40} {:<10.4f}".format("Std Dev Generation (s):", metrics.std_generation))
    print("{:<40} {:<10.4f}".format("Min Generation (s):", metrics.min_generation))
    print("{:<40} {:<10.4f}".format("Max Generation (s):", metrics.max_generation))
    for p, v in metrics.percentiles_generation:
        p_word = str(int(p)) if int(p) == p else str(p)
        print("{:<40} {:<10.4f}".format(f"P{p_word} Generation (s):", v))

    print("{s:{c}^{n}}".format(s=" Placeholder Metrics ", n=60, c="-"))
    print("{:<40} {:<10}".format("TTFF (s):", "N/A (placeholder)"))
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
        "mean_latency": metrics.mean_latency,
        "median_latency": metrics.median_latency,
        "std_latency": metrics.std_latency,
        "min_latency": metrics.min_latency,
        "max_latency": metrics.max_latency,
        "percentiles_latency": {
            f"p{int(p) if int(p) == p else p}": v for p, v in metrics.percentiles_latency
        },
        "mean_generation": metrics.mean_generation,
        "median_generation": metrics.median_generation,
        "std_generation": metrics.std_generation,
        "min_generation": metrics.min_generation,
        "max_generation": metrics.max_generation,
        "percentiles_generation": {
            f"p{int(p) if int(p) == p else p}": v for p, v in metrics.percentiles_generation
        },
        "latencies": [out.latency for out in outputs],
        "generations": [out.generation for out in outputs],
        "errors": [out.error for out in outputs],
        "gen_params": gen_params,
    }

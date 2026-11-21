# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared tools of the module perf guard: read and write golden files, and time a module."""

import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from tensorrt_llm._torch.autotuner import AutoTuner, autotune


def load_golden(golden_path: Path) -> dict[str, Any]:
    """Read the golden file of one target folder.

    For example, a golden file holds:

        {
          "gpu_name": "NVIDIA GB200",
          "modules": {
            "input_layernorm": {
              "noise_threshold": 0.05,
              "concurrency_to_latency_us": {"1": 3.1, "2": null}
            }
          }
        }

    ``noise_threshold`` is the slowdown that a run may have before it counts as a regression.
    A latency is ``null`` until ``run_perf_guard.py --update-golden`` records it.
    """
    with open(golden_path) as golden_file:
        return json.load(golden_file)


def save_golden(golden_path: Path, golden: dict[str, Any]) -> None:
    """Write `golden` to the golden file at `golden_path`."""
    with open(golden_path, "w") as golden_file:
        json.dump(golden, golden_file, indent=2)
        golden_file.write("\n")


def measure_latency_us(
    run_module: Callable[[], object], num_iters: int = 100, num_repeats: int = 5
) -> float:
    """Return the median GPU time of one ``run_module()`` call in microseconds.

    As in serving, the autotuner tunes the module's ops in a warmup call first, and the timed
    calls replay a CUDA graph, so host overhead does not count. The warmup starts from an empty
    autotuner cache and passes no cache path, so a saved cache, such as the one that
    ``TLLM_AUTOTUNER_CACHE_PATH`` names for serving, cannot change the kernel choice.
    """
    AutoTuner.get().clear_cache()
    with torch.inference_mode():
        # tune the module's ops, as the runtime warmup does
        with autotune():
            run_module()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_module()

    # The first replay is slower, so keep it out of the timing.
    graph.replay()

    # time several batches of replays and take the median batch
    latencies_us = []
    for _ in range(num_repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            graph.replay()
        end.record()
        end.synchronize()
        latencies_us.append(start.elapsed_time(end) * 1000 / num_iters)
    return statistics.median(latencies_us)

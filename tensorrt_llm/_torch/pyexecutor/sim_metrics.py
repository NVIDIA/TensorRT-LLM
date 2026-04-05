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
"""Metrics dataclasses and computation for simulation mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SimRequestStats:
    """Per-request timing data collected during simulation."""

    request_id: int
    input_length: int
    output_length: int = 0
    created_time: float = 0.0
    gen_token_times: List[float] = field(default_factory=list)

    @property
    def ttft_s(self) -> float:
        if not self.gen_token_times:
            return 0.0
        return self.gen_token_times[0] - self.created_time

    @property
    def itl_s(self) -> List[float]:
        t = self.gen_token_times
        return [t[i] - t[i - 1] for i in range(1, len(t))]

    @property
    def tpot_s(self) -> float:
        itl = self.itl_s
        return sum(itl) / len(itl) if itl else 0.0

    @property
    def e2e_s(self) -> float:
        if not self.gen_token_times:
            return 0.0
        return self.gen_token_times[-1] - self.created_time


@dataclass
class SimIterationRecord:
    """Per-iteration data recorded during simulation."""

    iteration: int
    sim_time_s: float
    predicted_duration_s: float
    num_context_requests: int
    num_context_tokens: int
    num_generation_requests: int


def percentile(data: List[float], pct: float) -> float:
    """Compute percentile without numpy. Returns 0.0 for empty data."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def calc_sim_metrics(request_stats: dict, iterations: list) -> dict:
    """Compute HiSim-compatible metrics from per-request timing.

    Args:
        request_stats: dict of request_id -> SimRequestStats
        iterations: list of SimIterationRecord (unused for now, reserved)

    Returns:
        Dict with TTFT, TPOT, ITL, e2e, throughput metrics in milliseconds.
    """
    if not request_stats:
        return {
            "completed": 0,
            "total_input": 0,
            "total_output": 0,
            "duration": 0.0,
            "request_throughput": 0.0,
            "input_throughput": 0.0,
            "output_throughput": 0.0,
            "mean_ttft_ms": 0.0,
            "p50_ttft_ms": 0.0,
            "p95_ttft_ms": 0.0,
            "p99_ttft_ms": 0.0,
            "mean_tpot_ms": 0.0,
            "p50_tpot_ms": 0.0,
            "p95_tpot_ms": 0.0,
            "p99_tpot_ms": 0.0,
            "mean_itl_ms": 0.0,
            "p50_itl_ms": 0.0,
            "p95_itl_ms": 0.0,
            "p99_itl_ms": 0.0,
            "mean_e2e_latency_ms": 0.0,
            "p95_e2e_latency_ms": 0.0,
            "p99_e2e_latency_ms": 0.0,
        }

    stats = list(request_stats.values())
    ttfts = [s.ttft_s for s in stats if s.gen_token_times]
    tpots = [s.tpot_s for s in stats if s.itl_s]
    itls = [lat for s in stats for lat in s.itl_s]
    e2es = [s.e2e_s for s in stats if s.gen_token_times]

    total_input = sum(s.input_length for s in stats)
    total_output = sum(s.output_length for s in stats)
    duration_s = max(
        (s.gen_token_times[-1] for s in stats if s.gen_token_times),
        default=1e-9)

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "completed": len(stats),
        "total_input": total_input,
        "total_output": total_output,
        "duration": duration_s,
        "request_throughput": len(stats) / duration_s,
        "input_throughput": total_input / duration_s,
        "output_throughput": total_output / duration_s,
        "mean_ttft_ms": _mean(ttfts) * 1000,
        "p50_ttft_ms": percentile(ttfts, 50) * 1000,
        "p95_ttft_ms": percentile(ttfts, 95) * 1000,
        "p99_ttft_ms": percentile(ttfts, 99) * 1000,
        "mean_tpot_ms": _mean(tpots) * 1000,
        "p50_tpot_ms": percentile(tpots, 50) * 1000,
        "p95_tpot_ms": percentile(tpots, 95) * 1000,
        "p99_tpot_ms": percentile(tpots, 99) * 1000,
        "mean_itl_ms": _mean(itls) * 1000,
        "p50_itl_ms": percentile(itls, 50) * 1000,
        "p95_itl_ms": percentile(itls, 95) * 1000,
        "p99_itl_ms": percentile(itls, 99) * 1000,
        "mean_e2e_latency_ms": _mean(e2es) * 1000,
        "p95_e2e_latency_ms": percentile(e2es, 95) * 1000,
        "p99_e2e_latency_ms": percentile(e2es, 99) * 1000,
    }

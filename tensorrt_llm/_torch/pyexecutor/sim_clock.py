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
"""Simulated clock for simulation mode."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sim_metrics import SimIterationRecord, SimRequestStats


class SimClock:
    """Accumulates predicted iteration times and per-request/per-iteration data.

    Not a singleton — owned by SimModelEngine as an instance attribute.
    """

    def __init__(self):
        self._total_time_s: float = 0.0
        self._num_iterations: int = 0
        self._iterations: list[SimIterationRecord] = []
        self._request_stats: dict[int, SimRequestStats] = {}

    def step(self, duration_s: float) -> None:
        """Advance clock by one iteration's predicted duration."""
        self._total_time_s += duration_s
        self._num_iterations += 1

    def record_iteration(self, predicted_duration_s: float, num_ctx_req: int,
                         num_ctx_tokens: int, num_gen_req: int) -> None:
        """Record per-iteration breakdown after a step."""
        from .sim_metrics import SimIterationRecord
        self._iterations.append(
            SimIterationRecord(iteration=self._num_iterations,
                               sim_time_s=self._total_time_s,
                               predicted_duration_s=predicted_duration_s,
                               num_context_requests=num_ctx_req,
                               num_context_tokens=num_ctx_tokens,
                               num_generation_requests=num_gen_req))

    def register_request(self, request_id: int, input_length: int,
                         created_time: float = 0.0) -> None:
        """Register a new request for tracking. No-op if already registered."""
        from .sim_metrics import SimRequestStats
        if request_id not in self._request_stats:
            self._request_stats[request_id] = SimRequestStats(
                request_id=request_id,
                input_length=input_length,
                created_time=created_time)

    def record_token(self, request_id: int) -> None:
        """Record a generated token timestamp for the given request."""
        stats = self._request_stats[request_id]
        stats.gen_token_times.append(self._total_time_s)
        stats.output_length += 1

    @property
    def total_time_s(self) -> float:
        return self._total_time_s

    @property
    def num_iterations(self) -> int:
        return self._num_iterations

    @property
    def iterations(self) -> list:
        """List of SimIterationRecord objects."""
        return self._iterations

    @property
    def request_stats(self) -> dict:
        """Dict of request_id -> SimRequestStats."""
        return self._request_stats

    @property
    def metrics(self) -> dict:
        """Compute HiSim-compatible metrics from recorded data."""
        from .sim_metrics import calc_sim_metrics
        return calc_sim_metrics(self._request_stats, self._iterations)

    def write_metrics(self, output_dir: str) -> None:
        """Write metrics.json, request.jsonl, iteration.jsonl to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)
        with open(os.path.join(output_dir, "request.jsonl"), "w") as f:
            for s in self._request_stats.values():
                f.write(
                    json.dumps({
                        "request_id": s.request_id,
                        "input_length": s.input_length,
                        "output_length": s.output_length,
                        "created_time": s.created_time,
                        "gen_token_times": s.gen_token_times,
                        "ttft_ms": s.ttft_s * 1000,
                        "tpot_ms": s.tpot_s * 1000,
                        "e2e_ms": s.e2e_s * 1000,
                    }) + "\n")
        with open(os.path.join(output_dir, "iteration.jsonl"), "w") as f:
            for r in self._iterations:
                f.write(
                    json.dumps({
                        "iteration": r.iteration,
                        "sim_time_s": r.sim_time_s,
                        "predicted_duration_s": r.predicted_duration_s,
                        "num_context_requests": r.num_context_requests,
                        "num_context_tokens": r.num_context_tokens,
                        "num_generation_requests":
                        r.num_generation_requests,
                    }) + "\n")

    def reset(self) -> None:
        self._total_time_s = 0.0
        self._num_iterations = 0
        self._iterations = []
        self._request_stats = {}

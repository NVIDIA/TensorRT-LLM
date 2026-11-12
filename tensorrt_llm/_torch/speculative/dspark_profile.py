# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime side of ``tensorrt_llm.tools.dspark_profile_costs``.

When ``TLLM_DSPARK_PROFILE_COSTS`` names an output file, the executor replaces
the confidence policy's choice in full decode-only steps with a fixed
round-robin over the captured verifier budgets ``V`` (native first) and records
the wall-clock period between consecutive scheduling decisions. Each budget is
held for several consecutive steps so the overlap scheduler reaches steady
state, and the first steps of every hold are discarded. Every rank runs the
same schedule because it depends only on globally agreed state; rank 0 writes
the samples.
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

PROFILE_OUTPUT_ENV = "TLLM_DSPARK_PROFILE_COSTS"
PROFILE_HOLD_ENV = "TLLM_DSPARK_PROFILE_HOLD_STEPS"
PROFILE_WARMUP_ENV = "TLLM_DSPARK_PROFILE_WARMUP_STEPS"


class DSparkCostProfiler:
    """Force verifier budgets round-robin and record whole-step periods."""

    def __init__(
        self,
        output_path: str,
        *,
        write_samples: bool,
        hold_steps: int = 16,
        warmup_steps: int = 3,
    ) -> None:
        if hold_steps <= warmup_steps:
            raise ValueError("hold_steps must exceed warmup_steps")
        self.output_path = Path(output_path)
        self.write_samples = write_samples
        self.hold_steps = hold_steps
        self.warmup_steps = warmup_steps
        self._decisions_per_g: dict[int, int] = defaultdict(int)
        self._samples: dict[tuple[int, int], list[float]] = defaultdict(list)
        self._previous: Optional[tuple[int, int, int, float, int]] = None
        self._unflushed = 0

    @classmethod
    def from_env(cls, *, rank: int) -> Optional["DSparkCostProfiler"]:
        output_path = os.environ.get(PROFILE_OUTPUT_ENV)
        if not output_path:
            return None
        return cls(
            output_path,
            write_samples=rank == 0,
            hold_steps=int(os.environ.get(PROFILE_HOLD_ENV, "16")),
            warmup_steps=int(os.environ.get(PROFILE_WARMUP_ENV, "3")),
        )

    def choose(
        self,
        *,
        iteration: int,
        graph_batch_size: int,
        budgets: Sequence[int],
        all_ranks_full: bool,
    ) -> int:
        """Return the forced V (0 = native) for this decode step.

        ``budgets`` lists native 0 first, then the captured compact budgets of
        ``graph_batch_size``. Only steps in which every rank has exactly
        ``graph_batch_size`` real requests are timed.
        """
        now = time.perf_counter()
        previous = self._previous
        if previous is not None and iteration == previous[0] + 1:
            _, prev_g, prev_v, prev_time, prev_position = previous
            if prev_position >= self.warmup_steps:
                self._samples[(prev_g, prev_v)].append((now - prev_time) * 1000.0)
                self._unflushed += 1
        if not all_ranks_full:
            # The wave is filling or draining: persist what the full steps
            # measured so far, since the run may end soon.
            self._previous = None
            self._maybe_flush(force=True)
            return 0
        count = self._decisions_per_g[graph_batch_size]
        self._decisions_per_g[graph_batch_size] = count + 1
        cell = (count // self.hold_steps) % len(budgets)
        budget = int(budgets[cell])
        self._previous = (iteration, graph_batch_size, budget, now, count % self.hold_steps)
        self._maybe_flush(force=count % self.hold_steps == 0)
        return budget

    def _maybe_flush(self, *, force: bool) -> None:
        if not self.write_samples or self._unflushed == 0:
            return
        if not force and self._unflushed < self.hold_steps:
            return
        payload = {
            "samples": [
                {"graph_batch_size": g, "verifier_budget": v, "step_time_ms": times}
                for (g, v), times in sorted(self._samples.items())
            ]
        }
        tmp_path = self.output_path.with_name(self.output_path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload))
        os.replace(tmp_path, self.output_path)
        self._unflushed = 0

    def flush(self) -> None:
        self._maybe_flush(force=True)

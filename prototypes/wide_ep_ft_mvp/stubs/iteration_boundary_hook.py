# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""1c.4 (stub): Model-engine iteration-boundary health-check hook.

The single line that ties together the watchdog (1a.4), the EPLB stub
(1b.1-3), and the model engine's main loop:

    if health.generation != self._cached_generation:
        result = eplb.reconfigure_mask_only(health.get_failed_ranks())
        self._cached_generation = health.generation

This hook is what the model engine's ``forward`` would call at the top of
each iteration. The cache-on-generation pattern is critical: it keeps the
hot-path overhead at a single integer comparison when no failure has
occurred. Production PR 1c.4 also wires backpressure, drain semantics, and
``check_health()`` integration; the stub keeps just the generation check.

Stub contract — exercises these seams:

  * **EPGroupHealth.generation as the change-detection trigger.** This is
    Open Question 2 from the prototype plan: is the generation counter
    enough, or do we need an explicit "mask version" channel?
  * **Where in the iteration the check fires.** Top-of-iter (this stub) vs
    after fwd setup vs other points have different "one more iter with stale
    mask" risk — observable via the timeline.
  * **Per-iteration overhead in the no-failure case.** Single int comparison;
    measurable as the steady-state baseline.

Stub deliberately omits, vs. production:

  * Backpressure (pause new requests during reconfigure).
  * Drain semantics for in-flight tokens.
  * ``check_health()`` REST-endpoint integration.
  * Telemetry counters.
  * Multi-failure batching (waiting one tick to coalesce two near-simultaneous
    failures into one reconfigure).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from tensorrt_llm._torch.modules.fused_moe.ep_group_health import EPGroupHealth

from .eplb_slot_remap import EplbSlotRemapStub, ReconfigureResult


@dataclass(frozen=True)
class IterationHookEvent:
    """One iteration-boundary check + its outcome (for the timeline JSON)."""

    iteration: int
    wall_time_sec: float
    cached_generation: int
    observed_generation: int
    triggered_reconfigure: bool
    reconfigure_result: Optional[ReconfigureResult]


class IterationBoundaryHook:
    """Top-of-iteration health-check hook.

    Args:
        health: Shared :class:`EPGroupHealth`.
        eplb: The EPLB slot-remap stub instance to call when generation
            advances.
        on_event: Optional callback invoked for every iteration-boundary check
            (used by the prototype to log to the timeline JSON).
    """

    def __init__(
        self,
        health: EPGroupHealth,
        eplb: EplbSlotRemapStub,
        on_event: Optional[Callable[[IterationHookEvent], None]] = None,
    ) -> None:
        self._health = health
        self._eplb = eplb
        self._on_event = on_event
        self._cached_generation = health.generation
        self._iteration = 0

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def cached_generation(self) -> int:
        return self._cached_generation

    def at_iteration_boundary(self) -> Optional[ReconfigureResult]:
        """Call at the top of every iteration. Returns ``None`` if no reconfigure ran."""
        observed = self._health.generation
        triggered = observed != self._cached_generation
        result: Optional[ReconfigureResult] = None
        if triggered:
            result = self._eplb.reconfigure_mask_only(self._health.get_failed_ranks())
            self._cached_generation = observed
        if self._on_event is not None:
            self._on_event(
                IterationHookEvent(
                    iteration=self._iteration,
                    wall_time_sec=time.monotonic(),
                    cached_generation=self._cached_generation if not triggered else observed,
                    observed_generation=observed,
                    triggered_reconfigure=triggered,
                    reconfigure_result=result,
                )
            )
        self._iteration += 1
        return result

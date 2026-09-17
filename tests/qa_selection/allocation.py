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
"""Place a collected test in one allocation of a caller-supplied GPU ladder.

    GpuDemand.of(test) -> demand.assign_rung(ladder) -> Assignment.of(...)

This sits beside `selector.py` rather than inside it: `Selector` answers whether
a test can run on one machine, while a ladder is scheduling policy the pipeline
owns, so a per-machine object must not hold one.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .selector import CollectedTest, Decision


@dataclass(frozen=True)
class GpuDemand:
    """How many GPUs one test wants, and the marks that say so."""

    # Assumed when a test states no lower bound at all: nothing in its source
    # says one, so `required_gpus_from` stays empty to keep the guess visible.
    ASSUMED_GPUS = 1

    # The markers stating a GPU lower bound, in the order
    # `Selector.resource_blockers` applies them. `skip_less_mpi_world_size` is
    # measured in GPUs because the cluster job gives each rank one.
    MARKERS = ("skip_less_device", "skip_less_mpi_world_size")

    required_gpus: int
    required_gpus_from: Tuple[str, ...]

    @classmethod
    def of(cls, test: CollectedTest) -> "GpuDemand":
        """Read `test`'s lower bounds; the largest is what it demands.

        The evidence names only the maximal bounds, since those are what
        produced the number.
        """
        bounds = cls.bounds_of(test)
        if not bounds:
            return cls(required_gpus=cls.ASSUMED_GPUS, required_gpus_from=())
        demand = max(gpus for _, gpus in bounds)
        return cls(
            required_gpus=demand,
            required_gpus_from=tuple(
                f"{marker}({gpus})" for marker, gpus in bounds if gpus == demand
            ),
        )

    @classmethod
    def bounds_of(cls, test: CollectedTest) -> Tuple[Tuple[str, int], ...]:
        """Every GPU lower bound `test` states, as (marker, value) pairs.

        Read through `closest_requirement`, the accessor `Selector.shortfall`
        uses: demand and feasibility must not disagree about which marker is
        nearest, and they now sit in separate modules.
        """
        bounds = []
        for marker in cls.MARKERS:
            required = test.closest_requirement(marker)
            if required is not None:
                bounds.append((marker, int(required)))
        return tuple(bounds)

    @property
    def assumed(self) -> bool:
        """True when no marker stated a bound, so the demand is our assumption.

        This is the population a wrong guess sends to a 1-GPU allocation, where
        a multi-GPU test tends to hang rather than fail fast.
        """
        return not self.required_gpus_from

    def assign_rung(self, ladder: Sequence[int]) -> Optional[int]:
        """The smallest rung of `ladder` that fits, or None when none does.

        None is never rounded up to the largest rung: an over-sized test belongs
        in a report, not in an allocation that cannot run it.
        """
        fitting = [rung for rung in ladder if rung >= self.required_gpus]
        return min(fitting) if fitting else None


@dataclass(frozen=True)
class Assignment:
    """One test's feasibility decision, paired with its demand and allocation.

    A pairing rather than a wider `Decision`: `Decision.selected` keeps meaning
    feasibility alone, which is what lets a caller apply feasibility first and
    assignment second.
    """

    decision: Decision
    demand: GpuDemand
    rung: Optional[int]
    unassignable: bool

    @classmethod
    def of(
        cls,
        decision: Decision,
        test: CollectedTest,
        ladder: Optional[Sequence[int]] = None,
    ) -> "Assignment":
        """Pair `decision` with the demand read from `test`, placed on `ladder`.

        With no ladder there are no allocations to choose between, so `rung` is
        None and `unassignable` is False -- the distinction a report needs
        between "no ladder was given" and "no rung fits this test".
        """
        demand = GpuDemand.of(test)
        rung = None if ladder is None else demand.assign_rung(ladder)
        return cls(
            decision=decision,
            demand=demand,
            rung=rung,
            unassignable=ladder is not None and rung is None,
        )

    @property
    def nodeid(self) -> str:
        """The node id this assignment is about."""
        return self.decision.nodeid

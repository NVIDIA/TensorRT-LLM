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

Reads marks only, never a `MachineProfile`: a test's demand is the same on every
machine and at every rung. Whether it can run there is `selector.py`'s question.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .selector import CollectedTest, Decision


@dataclass(frozen=True)
class GpuDemand:
    """How many GPUs one test wants, and the marks that say so."""

    # Used when a test states no lower bound; `required_gpus_from` is then empty.
    ASSUMED_GPUS = 1

    # Markers stating a GPU lower bound, in `Selector.resource_blockers` order.
    # `skip_less_mpi_world_size` is measured in GPUs: one rank per GPU.
    MARKERS = ("skip_less_device", "skip_less_mpi_world_size")

    required_gpus: int
    required_gpus_from: Tuple[str, ...]

    @classmethod
    def of(cls, test: CollectedTest) -> "GpuDemand":
        """The largest lower bound `test` states, with the marks that stated it.

        `required_gpus_from` names only the maximal bounds.
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

        Takes the nearest marker of each kind, through the same accessor
        `Selector.shortfall` uses.
        """
        bounds = []
        for marker in cls.MARKERS:
            required = test.closest_requirement(marker)
            if required is not None:
                bounds.append((marker, int(required)))
        return tuple(bounds)

    @property
    def assumed(self) -> bool:
        """True when no marker stated a bound, so `required_gpus` is `ASSUMED_GPUS`."""
        return not self.required_gpus_from

    def assign_rung(self, ladder: Sequence[int]) -> Optional[int]:
        """The smallest rung of `ladder` that fits, or None when none does.

        Rung order is not assumed. A demand larger than every rung gives None,
        never the largest rung.
        """
        fitting = [rung for rung in ladder if rung >= self.required_gpus]
        return min(fitting) if fitting else None


@dataclass(frozen=True)
class Assignment:
    """One test's feasibility decision, paired with its demand and allocation."""

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

        With no ladder, `rung` is None and `unassignable` is False.
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

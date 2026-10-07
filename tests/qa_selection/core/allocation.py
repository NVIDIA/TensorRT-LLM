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
"""Decide how many GPUs a collected test needs, and which rung of the run's ladder holds it.

    GpuDemand.of(test) -> demand.assign_rung(ladder) -> Assignment.of(...)

The GPU count is decided here and nowhere else: the markers `markers.json`
declares as bounding `gpus` are read once, each as it declares, and a demand
above the largest rung is a blocker. Demand reads marks, never a
`MachineProfile`, so it is the same on every machine. Whether the rules allow
the test on the machine's card is `selector.py`'s question.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Tuple

from .markers import ResourceMarker, default_markers
from .selector import CollectedTest, Decision


@dataclass(frozen=True)
class GpuDemand:
    """How many GPUs one test wants, and the marks that say so."""

    # Used when a test states no lower bound; `required_gpus_from` is then empty.
    ASSUMED_GPUS = 1

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
        """Every GPU lower bound `test` states, as (marker name, value) pairs."""
        return tuple(
            (marker.name, int(required))
            for marker in default_markers().bounding(ResourceMarker.GPUS)
            for required in test.requirements(marker)
        )

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

    def blocker_against(self, ladder: Sequence[int]) -> Optional[str]:
        """Why no rung of `ladder` holds this test, or None when one does.

        Names the markers that stated the demand and the largest rung, never the
        machine, so a short ladder and a small node give the same reason.
        """
        if self.assign_rung(ladder) is not None:
            return None
        return (
            f"{', '.join(self.required_gpus_from)}: needs {self.required_gpus} GPUs, "
            f"largest rung is {max(ladder)}"
        )


@dataclass(frozen=True)
class Assignment:
    """One test's rule decision, its GPU demand, and the rung that holds it.

    `blockers` is the whole answer: the rules' blockers first, then the GPU
    count's, so `selected` is the one flag selection and the report read.
    """

    decision: Decision
    demand: GpuDemand
    rung: Optional[int]
    blockers: Tuple[str, ...]

    @classmethod
    def of(
        cls,
        decision: Decision,
        test: CollectedTest,
        ladder: Sequence[int],
    ) -> "Assignment":
        """Pair `decision` with the demand read from `test`, placed on `ladder`."""
        demand = GpuDemand.of(test)
        count_blocker = demand.blocker_against(ladder)
        return cls(
            decision=decision,
            demand=demand,
            rung=demand.assign_rung(ladder),
            blockers=decision.blockers + ((count_blocker,) if count_blocker else ()),
        )

    @property
    def selected(self) -> bool:
        """True when neither the rules nor the GPU count drop this test."""
        return not self.blockers

    @property
    def nodeid(self) -> str:
        """The node id this assignment is about."""
        return self.decision.nodeid

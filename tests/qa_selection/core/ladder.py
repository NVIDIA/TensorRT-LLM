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
"""`--ladder`: the allocation sizes, and which one holds a test.

    GpuDemand.of(test).assign_rung(ladder)       -> smallest rung >= demand, or None
    GpuDemand.of(test).blocker_against(ladder)   -> why no rung holds it, or None
    Ladder.parse("1,4,8"), Ladder.of([8])        -> a validated ladder

GPU count is decided here only. A test's demand is the largest bound among the
markers `markers.json` declares as bounding `gpus`, each read as it declares; a
test stating none needs one GPU. Demand reads marks, never a machine profile.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from .marks import CollectedTest, ResourceMarker, default_markers


@dataclass(frozen=True)
class GpuDemand:
    """How many GPUs one test wants, and the marks that say so."""

    # Used when a test states no lower bound; `required_gpus_from` is then empty.
    ASSUMED_GPUS = 1

    required_gpus: int
    required_gpus_from: Tuple[str, ...]

    @classmethod
    def of(cls, test: CollectedTest) -> GpuDemand:
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


class Ladder(Sequence):
    """The allocation sizes a run may be scheduled at."""

    SEPARATOR = ","

    def __init__(self, rungs: Iterable[int]) -> None:
        self._rungs = tuple(rungs)

    @classmethod
    def of(cls, rungs: Iterable[int]) -> Ladder:
        """Build a ladder from rungs already in integer form.

        Raises ValueError unless there is at least one rung, every rung is a
        positive integer, and the rungs ascend without repeating. A JSON `true`
        is rejected, not counted as 1.
        """
        rungs = list(rungs)
        if not rungs:
            raise ValueError("a ladder needs at least one rung")
        for rung in rungs:
            if not isinstance(rung, int) or isinstance(rung, bool) or rung < 1:
                raise ValueError(f"{rung!r} is not a positive integer")
        if rungs != sorted(set(rungs)):
            raise ValueError(f"rungs must ascend without repeating, got {rungs}")
        return cls(rungs)

    @classmethod
    def parse(cls, text: str) -> Ladder:
        """Read a ladder written as `1,4,8`."""
        rungs = []
        for field in text.split(cls.SEPARATOR):
            rung = field.strip()
            if not rung.isdigit() or int(rung) < 1:
                raise ValueError(f"{rung!r} is not a positive integer")
            rungs.append(int(rung))
        return cls.of(rungs)

    def __getitem__(self, index):
        return self._rungs[index]

    def __len__(self) -> int:
        return len(self._rungs)

    def __str__(self) -> str:
        """The spelling `--ladder` accepts, so a printed ladder can be passed back."""
        return self.SEPARATOR.join(str(rung) for rung in self._rungs)

    def __repr__(self) -> str:
        return f"Ladder({self})"

    @property
    def smallest(self) -> int:
        """The cheapest allocation on this ladder."""
        return min(self._rungs)

    @property
    def largest(self) -> int:
        """The largest allocation; a demand above it fits no rung."""
        return max(self._rungs)

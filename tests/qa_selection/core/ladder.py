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
"""What a legal ladder is, in one place.

    Ladder.parse("1,4,8")   the `--ladder` option
    Ladder.of([8])          the default: one rung of the machine's GPUs per node

Both spellings go through `of`, so the default cannot be a ladder the option
would reject.
"""

from collections.abc import Sequence
from typing import Iterable


class Ladder(Sequence):
    """The allocation sizes a run may be scheduled at."""

    SEPARATOR = ","

    def __init__(self, rungs: Iterable[int]) -> None:
        self._rungs = tuple(rungs)

    @classmethod
    def of(cls, rungs: Iterable[int]) -> "Ladder":
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
    def parse(cls, text: str) -> "Ladder":
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

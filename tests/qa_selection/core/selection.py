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
"""What one collection run decided about every test it was given.

    Selection.of(request, tests) -> selection.partition(anything_parallel)

`tests` are `CollectedTest` views, so nothing here knows what a pytest item is;
`partition` splits any sequence that came from the same order.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple, TypeVar

from .allocation import Assignment
from .request import SelectionRequest
from .selector import CollectedTest, Selector

T = TypeVar("T")


@dataclass(frozen=True)
class Selection:
    """One run's decisions, in the order the tests were given."""

    request: SelectionRequest
    assignments: Tuple[Assignment, ...]

    @classmethod
    def of(cls, request: SelectionRequest, tests: Sequence[CollectedTest]) -> "Selection":
        """Decide every test, then place it on the ladder.

        Feasibility is decided first and independently: a `Decision` says the
        test can run on the machine, and the rung says which allocation it
        belongs to.
        """
        selector = Selector(request.profile)
        return cls(
            request=request,
            assignments=tuple(
                Assignment.of(selector.decide(test), test, request.ladder) for test in tests
            ),
        )

    def is_live(self, assignment: Assignment) -> bool:
        """True when `assignment` runs in this invocation.

        An infeasible test never runs. With a target rung, only the tests placed
        on that rung run; without one, every feasible test does.
        """
        if not assignment.decision.selected:
            return False
        return self.request.target_rung is None or assignment.rung == self.request.target_rung

    def partition(self, items: Sequence[T]) -> Tuple[List[T], List[T]]:
        """`items` split into those to keep and those to drop, in order.

        Paired by position: `assignments` was built from the same sequence in
        one pass.
        """
        kept: List[T] = []
        dropped: List[T] = []
        for item, assignment in zip(items, self.assignments):
            (kept if self.is_live(assignment) else dropped).append(item)
        return kept, dropped

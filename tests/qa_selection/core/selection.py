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

        A `Decision` says whether the rules allow the test on the machine's
        card; the `Assignment` adds whether a rung holds it, and which.
        """
        selector = Selector(request.profile)
        return cls(
            request=request,
            assignments=tuple(
                Assignment.of(selector.decide(test), test, request.ladder) for test in tests
            ),
        )

    def partition(self, items: Sequence[T]) -> Tuple[List[T], List[T]]:
        """`items` split into the selected and the rest, in order.

        Paired by position: `assignments` was built from the same sequence in
        one pass. The kept items are the ones the report publishes, rung by rung.
        """
        kept: List[T] = []
        dropped: List[T] = []
        for item, assignment in zip(items, self.assignments):
            (kept if assignment.selected else dropped).append(item)
        return kept, dropped

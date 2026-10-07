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
"""Decide whether collected tests can run on one target machine.

    Selector(profile).decide(test) -> Decision(selected, blockers)

`CollectedTest` is a framework-neutral view of a pytest `Item`: a node id and
its marks, closest level first.

Marker precedence, reproducing the fixtures that consume each one:

    skipif                     every level; any match drops the test
    skip_device_not_contain    closest marker only
    skip_less_device_memory    every level
    skip_less_host_memory      not evaluated

The two GPU-count markers, `skip_less_device` and `skip_less_mpi_world_size`,
are not read here: `allocation.py` decides GPU count once, against the run's
ladder.

A reason string with no rule keeps the test. The table is curated, so its
silence is a decision, not a gap: nothing is recorded and nothing is reported.
"""

from dataclasses import dataclass, field
from typing import Any, Iterator, List, Mapping, Optional, Sequence, Tuple

from .machines import MachineProfile
from .rules import SkipRuleTable, default_rule_table


@dataclass(frozen=True)
class Mark:
    """The part of a pytest mark this layer reads."""

    name: str
    args: Tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def skipif_reason(self) -> Optional[str]:
        """This skipif's reason string, or None. Keyword form only."""
        reason = self.kwargs.get("reason")
        return reason if isinstance(reason, str) and reason else None

    @property
    def requirement(self) -> Optional[Any]:
        """What a resource marker asks for, as `skip_less_device(8)` asks for 8."""
        return self.args[0] if self.args else None


@dataclass(frozen=True)
class CollectedTest:
    """One collected item: its node id and its marks, closest level first."""

    nodeid: str
    marks: Sequence[Mark] = ()

    def iter_markers(self, name: str) -> Iterator[Mark]:
        """Every mark of this name, closest level first."""
        return (mark for mark in self.marks if mark.name == name)

    def closest_marker(self, name: str) -> Optional[Mark]:
        """The nearest mark of this name, or None."""
        return next(self.iter_markers(name), None)

    def closest_requirement(self, name: str) -> Optional[Any]:
        """What the nearest `name` marker asks for, as get_closest_marker does."""
        mark = self.closest_marker(name)
        return mark.requirement if mark is not None else None


@dataclass(frozen=True)
class Decision:
    """Why one test was kept or dropped for one machine."""

    nodeid: str
    selected: bool
    blockers: Tuple[str, ...]


class Selector:
    """Decides collected tests for one target machine.

    Built once per machine, then asked about many tests.
    """

    def __init__(self, profile: MachineProfile, rules: Optional[SkipRuleTable] = None) -> None:
        self.profile = profile
        self.rules = rules if rules is not None else default_rule_table()

    def decide(self, test: CollectedTest) -> Decision:
        """Return the selection decision for one test, listing every blocker."""
        blockers = self.skipif_blockers(test) + self.resource_blockers(test)
        return Decision(
            nodeid=test.nodeid,
            selected=not blockers,
            blockers=tuple(blockers),
        )

    def skipif_blockers(self, test: CollectedTest) -> List[str]:
        """Evaluate every skipif at every level; one match drops the test.

        A mark the table has no rule for is passed over in silence, whether it
        carries no `reason=` at all or a reason the table declines to encode.
        Both are skips this layer has no opinion about, so both keep the test.
        """
        blockers: List[str] = []
        for mark in test.iter_markers("skipif"):
            # args[0] is the frozen condition: it describes the collecting host,
            # never the target, and is never read.
            reason = mark.skipif_reason
            rule = self.rules.get(reason)
            if rule is not None and rule.blocks(self.profile):
                blockers.append(reason)
        return blockers

    def resource_blockers(self, test: CollectedTest) -> List[str]:
        """Apply the per-card resource markers, each in its production precedence."""
        return self.device_name_blockers(test) + self.device_memory_blockers(test)

    def device_name_blockers(self, test: CollectedTest) -> List[str]:
        """Report skip_device_not_contain when no keyword matches the device."""
        keywords = test.closest_requirement("skip_device_not_contain")
        if keywords is None or any(keyword in self.profile.device_name for keyword in keywords):
            return []
        return [
            f"skip_device_not_contain: {self.profile.device_name!r} "
            f"contains none of {list(keywords)}"
        ]

    def device_memory_blockers(self, test: CollectedTest) -> List[str]:
        """Report every level's demand; a method's marker does not replace its class's."""
        available = self.profile.device_memory_mib
        return [
            f"skip_less_device_memory: needs {int(mark.requirement)} MiB, "
            f"target has {available} MiB"
            for mark in test.iter_markers("skip_less_device_memory")
            if mark.requirement is not None and available < int(mark.requirement)
        ]

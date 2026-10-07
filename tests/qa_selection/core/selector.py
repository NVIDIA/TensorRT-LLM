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

Every `skipif` is read at every level, and any match drops the test. The
resource markers bounding `device_name` and `device_memory_mib` are read as
`markers.json` declares them. GPU-count markers are not read here:
`allocation.py` decides GPU count once, against the run's ladder.

A reason string with no rule keeps the test. The table is curated, so its
silence is a decision, not a gap: nothing is recorded and nothing is reported.
"""

from dataclasses import dataclass, field
from typing import Any, Iterator, List, Mapping, Optional, Sequence, Tuple

from .machines import MachineProfile
from .markers import ResourceMarker, ResourceMarkers, default_markers
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

    def requirements(self, marker: ResourceMarker) -> Tuple[Any, ...]:
        """What `marker` asks of this test, closest level first.

        The nearest mark's alone, as `get_closest_marker` reads it, or every
        level's, as `iter_markers` does, by the marker's `read`.
        """
        marks = list(self.iter_markers(marker.name))
        if not marker.every_level:
            marks = marks[:1]
        return tuple(mark.requirement for mark in marks if mark.requirement is not None)


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

    def __init__(
        self,
        profile: MachineProfile,
        rules: Optional[SkipRuleTable] = None,
        markers: Optional[ResourceMarkers] = None,
    ) -> None:
        self.profile = profile
        self.rules = rules if rules is not None else default_rule_table()
        self.markers = markers if markers is not None else default_markers()

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
        """Apply the per-card resource markers, each read as `markers.json` declares."""
        return self.device_name_blockers(test) + self.device_memory_blockers(test)

    def device_name_blockers(self, test: CollectedTest) -> List[str]:
        """Report each keyword list that matches nothing in the device name."""
        device_name = self.profile.device_name
        return [
            f"{marker.name}: {device_name!r} contains none of {list(keywords)}"
            for marker in self.markers.bounding(ResourceMarker.DEVICE_NAME)
            for keywords in test.requirements(marker)
            if not any(keyword in device_name for keyword in keywords)
        ]

    def device_memory_blockers(self, test: CollectedTest) -> List[str]:
        """Report each memory demand above the card's."""
        available = self.profile.device_memory_mib
        return [
            f"{marker.name}: needs {int(required)} MiB, target has {available} MiB"
            for marker in self.markers.bounding(ResourceMarker.DEVICE_MEMORY_MIB)
            for required in test.requirements(marker)
            if available < int(required)
        ]

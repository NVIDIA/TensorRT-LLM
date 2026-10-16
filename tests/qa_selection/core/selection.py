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
"""The entry `plugin.py` calls: one run's request, and its outcome per test.

    SelectionRequest.of(machine, ladder, out_dir)  -> SelectionRequest; None without a machine
    Selection.of(request, tests)                   -> one Outcome per test, in order
    selection.partition(items)                     -> (selected, deselected)
    selection.report(rootdir)                      -> SelectionOutput

Raises `SelectionError`, whose message names the option to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TypeVar

from .ladder import GpuDemand, Ladder
from .machine import MachineCheck, MachineProfile, ProfileConfigError, default_catalog
from .marks import CollectedTest
from .report import ArtifactNames, Outcome, SelectionOutput, SelectionReport

T = TypeVar("T")


@dataclass(frozen=True)
class Selection:
    """One run's outcomes, in the order the tests were given."""

    request: SelectionRequest
    outcomes: Tuple[Outcome, ...]

    @classmethod
    def of(cls, request: SelectionRequest, tests: Sequence[CollectedTest]) -> Selection:
        """Each test's outcome on the request's machine and ladder."""
        check = MachineCheck(request.profile)
        return cls(
            request=request,
            outcomes=tuple(cls.outcome_of(test, check, request.ladder) for test in tests),
        )

    @staticmethod
    def outcome_of(test: CollectedTest, check: MachineCheck, ladder: Ladder) -> Outcome:
        """`test`'s blockers, the card's then the GPU count's, and the rung that holds it."""
        demand = GpuDemand.of(test)
        count_blocker = demand.blocker_against(ladder)
        return Outcome(
            nodeid=test.nodeid,
            blockers=check.blockers(test) + ((count_blocker,) if count_blocker else ()),
            required_gpus=demand.required_gpus,
            required_gpus_from=demand.required_gpus_from,
            rung=demand.assign_rung(ladder),
        )

    def partition(self, items: Sequence[T]) -> Tuple[List[T], List[T]]:
        """`items` split into (selected, deselected), paired with `outcomes` by position."""
        kept: List[T] = []
        dropped: List[T] = []
        for item, outcome in zip(items, self.outcomes):
            (kept if outcome.selected else dropped).append(item)
        return kept, dropped

    def report(self, rootdir: Path) -> SelectionOutput:
        """The run's record, written to the request's `out_dir` when it has one."""
        request = self.request
        report = SelectionReport.of(request.profile, request.ladder, self.outcomes, rootdir)
        return SelectionOutput.of(report, request.out_dir)


@dataclass(frozen=True)
class SelectionRequest:
    """One run's target: the machine's profile, its ladder, and where to write."""

    machine: str
    profile: MachineProfile
    ladder: Ladder
    out_dir: Optional[Path]

    @classmethod
    def of(
        cls,
        machine: Optional[str],
        ladder: Optional[str] = None,
        out_dir: Optional[str] = None,
    ) -> Optional[SelectionRequest]:
        """The request, or None when no machine is named; no other option is read then."""
        if not machine:
            return None
        profile = cls.profile_for(machine)
        parsed_ladder = cls.ladder_for(ladder, profile)
        return cls(
            machine=machine,
            profile=profile,
            ladder=parsed_ladder,
            out_dir=cls.out_dir_for(out_dir, machine, parsed_ladder),
        )

    @classmethod
    def out_dir_for(cls, text: Optional[str], machine: str, ladder: Ladder) -> Optional[Path]:
        """The output directory, created, or None when none was named."""
        if text is None:
            return None
        out_dir = Path(text)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SelectionError(
                f"--selection-out-dir: cannot create {str(out_dir)!r}: {error.strerror}"
            ) from error
        cls.check_no_orphans(out_dir, machine, ladder)
        return out_dir

    @staticmethod
    def check_no_orphans(out_dir: Path, machine: str, ladder: Ladder) -> None:
        """Refuse `out_dir` when it holds a file of `machine`'s that this run does not write."""
        orphans = ArtifactNames.orphans_in(out_dir, machine, ladder)
        if not orphans:
            return
        writes = ArtifactNames.written_by(machine, ladder)
        raise SelectionError(
            f"--selection-out-dir: {str(out_dir)!r} already holds "
            f"{', '.join(orphans)} for {machine}, which this run does not overwrite; "
            f"it writes {', '.join(writes)}. "
            f"Remove the leftover file(s), or write to a different directory"
        )

    @staticmethod
    def ladder_for(text: Optional[str], profile: MachineProfile) -> Ladder:
        """The parsed `--ladder`, or `[max_gpu_per_node]` when absent; no rung may exceed it."""
        if text is None:
            return Ladder.of([profile.max_gpu_per_node])
        try:
            ladder = Ladder.parse(text)
        except ValueError as error:
            raise SelectionError(f"--ladder: {error}")
        if ladder.largest > profile.max_gpu_per_node:
            raise SelectionError(
                f"--ladder: rung {ladder.largest} exceeds {profile.name}, which has "
                f"{profile.max_gpu_per_node} GPUs per node"
            )
        return ladder

    @staticmethod
    def machines() -> Optional[List[str]]:
        """Every name `--machine` accepts, or None when the catalogue is unreadable."""
        try:
            return sorted(default_catalog())
        except (ProfileConfigError, OSError):
            return None

    @staticmethod
    def profile_for(machine: str) -> MachineProfile:
        """The named machine's profile."""
        try:
            catalog = default_catalog()
        except (ProfileConfigError, OSError) as error:
            raise SelectionError(f"--machine: cannot read the machine catalogue: {error}")
        try:
            return catalog.profile_for(machine)
        except KeyError as error:
            raise SelectionError(f"--machine: {error.args[0]}")


class SelectionError(ValueError):
    """A usage error; the message names the option to fix."""

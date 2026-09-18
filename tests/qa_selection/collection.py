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
"""Everything selection does while pytest is collecting.

    SelectionOptions.add_to(parser) -> SelectionRequest.of(config)
                                    -> Selection.of(request, items)

The options, the markers, the adapter from a pytest item to `core.selector`'s
`CollectedTest`, and the decisions one run produced. `plugin.py` holds the
hooks; `core/` holds the decisions and imports no pytest.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from .core.allocation import Assignment, Ladder
from .core.machines import MachineProfile, ProfileConfigError, default_catalog
from .core.selector import CollectedTest, Mark, Selector


class ResourceMarkers:
    """The marks selection may read a requirement from.

    Membership is the adapter's allowlist: only these carry a value in `args[0]`.
    """

    # Wording matches tests/integration/defs/pytest.ini.
    DESCRIPTIONS = {
        "skip_less_device": "skip when less device detected than the declared",
        "skip_less_mpi_world_size": "skip when less mpi world size detected than the declared",
        "skip_less_device_memory": "skip when less device memory detected than the requested",
        "skip_less_host_memory": "skip when less host memory detected than the requested",
        "skip_device_not_contain": "skip when the device does not contain the specified keyword",
    }

    @classmethod
    def carries_requirement(cls, name: str) -> bool:
        """True when this mark's first argument is a requirement, not a condition."""
        return name in cls.DESCRIPTIONS

    @classmethod
    def declare(cls, config: pytest.Config) -> None:
        """Declare any of these pytest does not already know.

        The integration suite's pytest.ini declares all five, so inside it this
        does nothing.
        """
        known = {line.split(":")[0].split("(")[0].strip() for line in config.getini("markers")}
        for name, description in cls.DESCRIPTIONS.items():
            if name not in known:
                config.addinivalue_line("markers", f"{name}: {description}")


class SelectionOptions:
    """The plugin's four command-line options.

    `--gpus` narrows what is selected; `--ladder` partitions what is written.
    One machine per invocation.
    """

    GROUP = "qa selection"

    MACHINE = "qa_selection_machine"
    GPUS = "qa_selection_gpus"
    LADDER = "qa_selection_ladder"
    OUT_DIR = "qa_selection_out_dir"

    @classmethod
    def add_to(cls, parser: pytest.Parser) -> None:
        """Register every option this plugin reads."""
        group = parser.getgroup(cls.GROUP, "select tests by target machine")
        group.addoption(
            "--machine",
            dest=cls.MACHINE,
            default=None,
            choices=cls.machine_choices(),
            help="target machine to select for; absent leaves collection untouched",
        )
        group.addoption(
            "--gpus",
            dest=cls.GPUS,
            type=int,
            default=None,
            help="GPUs to select for: a rung of --ladder, or a feasibility "
            "ceiling without one (default: the machine's GPUs per node)",
        )
        group.addoption(
            "--ladder",
            dest=cls.LADDER,
            default=None,
            help="ascending allocation sizes, e.g. 1,4,8; partitions the output",
        )
        group.addoption(
            "--selection-out-dir",
            dest=cls.OUT_DIR,
            default=None,
            help="where to write the record and the identifier lists; "
            "must be passed as --selection-out-dir=PATH",
        )

    @classmethod
    def machine_choices(cls) -> Optional[List[str]]:
        """Valid machine names, or None to leave `--machine` unconstrained.

        Returns None rather than raising when the catalogue cannot be read.
        """
        try:
            return sorted(default_catalog())
        except (ProfileConfigError, OSError):
            return None


@dataclass(frozen=True)
class SelectionRequest:
    """One run's target: the named machine, sized for the allocation under test."""

    # Resolved at configure time and kept on `config.stash`, not module-level.
    STASH_KEY = pytest.StashKey()

    machine: str
    profile: MachineProfile
    ladder: Optional[Ladder]
    target_rung: Optional[int]
    out_dir: Optional[Path]

    @classmethod
    def of(cls, config: pytest.Config) -> Optional["SelectionRequest"]:
        """This run's request, or None when no machine is named."""
        machine = config.getoption(SelectionOptions.MACHINE)
        if not machine:
            return None
        gpus = config.getoption(SelectionOptions.GPUS)
        profile = cls.profile_for(machine, gpus)
        ladder = cls.ladder_for(config.getoption(SelectionOptions.LADDER), profile)
        target_rung = cls.target_rung_for(gpus, ladder)
        return cls(
            machine=machine,
            profile=profile,
            ladder=ladder,
            target_rung=target_rung,
            out_dir=cls.out_dir_for(config.getoption(SelectionOptions.OUT_DIR), target_rung),
        )

    @staticmethod
    def out_dir_for(text: Optional[str], target_rung: Optional[int]) -> Optional[Path]:
        """Where to write, or None when nothing is written.

        Naming a rung and asking for artifacts at once is a usage error: the
        rung's list is one of the files the same command writes without
        `--gpus`.
        """
        if text is None:
            return None
        if target_rung is not None:
            raise pytest.UsageError(
                "--selection-out-dir: cannot be combined with --gpus and --ladder; "
                "drop --gpus and every rung is written at once"
            )
        return Path(text)

    @staticmethod
    def ladder_for(text: Optional[str], profile: MachineProfile) -> Optional[Ladder]:
        """The parsed `--ladder`, or None when it was not given.

        A rung larger than the machine's GPUs per node is a usage error: that
        allocation cannot be requested.
        """
        if text is None:
            return None
        try:
            ladder = Ladder.parse(text)
        except ValueError as error:
            raise pytest.UsageError(f"--ladder: {error}")
        if ladder.largest > profile.max_gpu_per_node:
            raise pytest.UsageError(
                f"--ladder: rung {ladder.largest} exceeds {profile.name}, which has "
                f"{profile.max_gpu_per_node} GPUs per node"
            )
        return ladder

    @staticmethod
    def target_rung_for(gpus: Optional[int], ladder: Optional[Ladder]) -> Optional[int]:
        """The rung `--gpus` names, or None when no rung filter applies.

        Without a ladder, `--gpus` is a feasibility ceiling already applied by
        sizing the profile, so it names no rung. A `--gpus` outside the ladder
        is a usage error rather than a run that selects nothing.
        """
        if ladder is None or gpus is None:
            return None
        if gpus not in ladder:
            raise pytest.UsageError(f"--gpus: {gpus} is not a rung of --ladder={ladder}")
        return gpus

    @staticmethod
    def profile_for(machine: str, gpus: Optional[int]) -> MachineProfile:
        """The named machine sized for `gpus`, or a usage error naming the fault.

        `gpus` of None sizes the profile to the machine's whole node.
        """
        try:
            catalog = default_catalog()
        except (ProfileConfigError, OSError) as error:
            raise pytest.UsageError(f"--machine: cannot read the machine catalogue: {error}")
        try:
            node = catalog.profile_for(machine)
        except KeyError as error:
            raise pytest.UsageError(f"--machine: {error.args[0]}")

        if gpus is None:
            return node
        if gpus > node.max_gpu_per_node:
            # The veto the ladder gets: that allocation cannot be requested.
            raise pytest.UsageError(
                f"--gpus: {gpus} exceeds {node.name}, which has "
                f"{node.max_gpu_per_node} GPUs per node"
            )
        try:
            return node.with_gpu_count(gpus)
        except ProfileConfigError as error:
            raise pytest.UsageError(f"--gpus: {error}")


class ItemView:
    """Reduces a pytest item to the framework-neutral view the selector reads.

    Three fields cross this boundary: a mark's name, a resource marker's
    requirement, and a skipif's `reason=`. A skipif's condition is dropped,
    never evaluated or rendered.
    """

    @classmethod
    def of(cls, item: pytest.Item) -> CollectedTest:
        """One item as a `CollectedTest`, marks closest level first."""
        return CollectedTest(
            nodeid=item.nodeid,
            marks=tuple(cls.mark_of(mark) for mark in item.iter_markers()),
        )

    @staticmethod
    def mark_of(mark: pytest.Mark) -> Mark:
        """One mark, reduced to the fields selection is allowed to read."""
        args: Tuple[Any, ...] = ()
        if ResourceMarkers.carries_requirement(mark.name):
            args = tuple(mark.args[:1])

        # Keyword form only; `selector.py` keys its rules on this string.
        reason = mark.kwargs.get("reason")
        kwargs: Dict[str, Any] = {"reason": reason} if isinstance(reason, str) else {}
        return Mark(name=mark.name, args=args, kwargs=kwargs)


@dataclass(frozen=True)
class Selection:
    """What one collection run decided, kept for later hooks to report."""

    STASH_KEY = pytest.StashKey()

    request: SelectionRequest
    assignments: Tuple[Assignment, ...]
    views: Tuple[CollectedTest, ...]

    @classmethod
    def of(cls, request: SelectionRequest, items: List[pytest.Item]) -> "Selection":
        """Decide every collected item, then place it on the ladder.

        Feasibility is decided first and independently: a `Decision` says the
        test can run on the machine, and the rung says which allocation it
        belongs to.

        The views are kept beside the assignments: the report reads them for
        one fact no `Decision` carries, a `skipif` with no `reason=`.
        """
        selector = Selector(request.profile)
        views = tuple(ItemView.of(item) for item in items)
        return cls(
            request=request,
            assignments=tuple(
                Assignment.of(selector.decide(view), view, request.ladder) for view in views
            ),
            views=views,
        )

    def is_live(self, assignment: Assignment) -> bool:
        """True when `assignment` runs in this invocation.

        An infeasible test never runs. With a target rung, only the tests placed
        on that rung run; without one, every feasible test does.
        """
        if not assignment.decision.selected:
            return False
        return self.request.target_rung is None or assignment.rung == self.request.target_rung

    def partition(self, items: List[pytest.Item]) -> Tuple[List[pytest.Item], List[pytest.Item]]:
        """`items` split into those to keep and those to deselect, in order.

        Paired by position: `assignments` was built from `items` in one pass.
        """
        kept: List[pytest.Item] = []
        dropped: List[pytest.Item] = []
        for item, assignment in zip(items, self.assignments):
            (kept if self.is_live(assignment) else dropped).append(item)
        return kept, dropped

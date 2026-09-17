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

The options, the markers, the adapter from a pytest item to `selector.py`'s
`CollectedTest`, and the decisions one run produced. `plugin.py` holds the
hooks; the modules below this one import no pytest.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest

from .machines import MachineProfile, ProfileConfigError, default_catalog
from .selector import CollectedTest, Decision, Mark, Selector


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

    @classmethod
    def of(cls, config: pytest.Config) -> Optional["SelectionRequest"]:
        """This run's request, or None when no machine is named."""
        machine = config.getoption(SelectionOptions.MACHINE)
        if not machine:
            return None
        gpus = config.getoption(SelectionOptions.GPUS)
        return cls(machine=machine, profile=cls.profile_for(machine, gpus))

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
            return catalog.profile_for(machine, gpus)
        except KeyError as error:
            raise pytest.UsageError(f"--machine: {error.args[0]}")
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
    decisions: Tuple[Decision, ...]

    @classmethod
    def of(cls, request: SelectionRequest, items: List[pytest.Item]) -> "Selection":
        """Decide every collected item against the request's machine."""
        selector = Selector(request.profile)
        return cls(
            request=request,
            decisions=tuple(selector.decide(ItemView.of(item)) for item in items),
        )

    def partition(self, items: List[pytest.Item]) -> Tuple[List[pytest.Item], List[pytest.Item]]:
        """`items` split into those to keep and those to deselect, in order.

        Paired by position: `decisions` was built from `items` in one pass.
        """
        kept: List[pytest.Item] = []
        dropped: List[pytest.Item] = []
        for item, decision in zip(items, self.decisions):
            (kept if decision.selected else dropped).append(item)
        return kept, dropped

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
"""Pytest entry point: deselect the tests a target machine cannot run.

    pytest --collect-only -p qa_selection.plugin --machine=B200 --ladder=1,4,8

Load with `-p` on the command line. No hardware is touched: every decision is
read from marks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from .core.machines import ProfileConfigError, default_catalog
from .core.markers import default_markers
from .core.report import SelectionOutput, TerminalSummary, UnassignableWarning
from .core.request import SelectionError, SelectionRequest
from .core.selection import Selection
from .core.selector import CollectedTest, Mark


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the selection options."""
    SelectionOptions.add_to(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Declare the resource markers, and resolve the target before collection."""
    MarkerDeclaration.add_to(config)
    request = SelectionOptions.request_from(config)
    if request is not None:
        config.stash[SelectionStash.REQUEST] = request


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Deselect what the target machine cannot run.

    A non-wrapper on purpose: `trylast` orders it against the other
    non-wrappers, xdist and pytest-split among them.
    """
    request = config.stash.get(SelectionStash.REQUEST, None)
    if request is None:
        return

    selection = Selection.of(request, [ItemView.of(item) for item in items])
    config.stash[SelectionStash.SELECTION] = selection

    kept, dropped = selection.partition(items)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = kept


def pytest_collection_finish(session: pytest.Session) -> None:
    """Build the record and write the artifacts.

    After `modifyitems`, so every other plugin's deselection is already in,
    and before `--collect-only` prints.
    """
    config = session.config
    selection = config.stash.get(SelectionStash.SELECTION, None)
    if selection is None:
        return
    output = SelectionOutput.of(selection, Path(str(config.rootpath)))
    config.stash[SelectionStash.OUTPUT] = output
    # Warned rather than summarised, so a job log can grep the warnings section.
    UnassignableWarning.issue_for(output.report)


def pytest_terminal_summary(terminalreporter) -> None:
    """Render the human-readable block after pytest's own counts."""
    output = terminalreporter.config.stash.get(SelectionStash.OUTPUT, None)
    if output is not None:
        TerminalSummary.render(terminalreporter, output)


class MarkerDeclaration:
    """Declares `core/markers.json`'s markers to a pytest that does not know them."""

    INI = "markers"

    @classmethod
    def add_to(cls, config: pytest.Config) -> None:
        """Declare any marker pytest has not already been told about.

        A no-op inside the integration suite, whose pytest.ini declares all five.
        """
        known = {line.split(":")[0].split("(")[0].strip() for line in config.getini(cls.INI)}
        for line in default_markers().ini_lines():
            if line.split(":")[0] not in known:
                config.addinivalue_line(cls.INI, line)


class SelectionOptions:
    """The plugin's five command-line options; `--help` states each one.

    Only `--gpus` narrows feasibility; `--ladder` partitions what is written and
    `--rung` selects one of its parts. One machine per invocation.
    """

    GROUP = "qa selection"

    MACHINE = "qa_selection_machine"
    GPUS = "qa_selection_gpus"
    LADDER = "qa_selection_ladder"
    RUNG = "qa_selection_rung"
    OUT_DIR = "qa_selection_out_dir"

    @classmethod
    def add_to(cls, parser: pytest.Parser) -> None:
        """Register every option this plugin reads."""
        group = parser.getgroup(cls.GROUP, "select tests by target machine")
        group.addoption(
            "--machine",
            dest=cls.MACHINE,
            metavar="NAME",
            default=None,
            choices=cls.machine_choices(),
            help="target machine to select for, from the profile catalogue. "
            "Absent leaves collection untouched, so loading this plugin "
            "without it changes nothing",
        )
        group.addoption(
            "--gpus",
            dest=cls.GPUS,
            metavar="N",
            type=int,
            default=None,
            help="GPUs this run may use: a feasibility ceiling, so everything "
            "fitting in N is selected. Must not exceed the machine's GPUs per "
            "node (the default). Cannot be combined with --ladder, which "
            "states GPU counts of its own -- name an allocation with --rung",
        )
        group.addoption(
            "--ladder",
            dest=cls.LADDER,
            metavar="RUNGS",
            default=None,
            help="ascending allocation sizes, comma separated, e.g. 1,4,8. "
            "Machine policy, held constant across a rerun. Partitions the "
            "output into one list per rung and narrows nothing. No rung may "
            "exceed the machine's GPUs per node",
        )
        group.addoption(
            "--rung",
            dest=cls.RUNG,
            metavar="N",
            type=int,
            default=None,
            help="the allocation this run occupies: keep only the tests "
            "--ladder assigned to the N-GPU rung. Requires --ladder and must "
            "name one of its rungs. Narrows what executes, never what the "
            "machine can do, so feasibility is decided against the whole node "
            "either way",
        )
        group.addoption(
            "--selection-out-dir",
            dest=cls.OUT_DIR,
            metavar="DIR",
            default=None,
            help="write <machine>.json and the .ids lists here, creating the "
            "directory if needed. Absent writes nothing. The directory is "
            "written into and never emptied, so a leftover list of this "
            "machine's from a different ladder is a usage error. Cannot be "
            "combined with --rung: the same command without it writes every "
            "rung's list in one pass",
        )

    @classmethod
    def machine_choices(cls) -> Optional[List[str]]:
        """Valid machine names, or None when the catalogue cannot be read.

        None leaves `--machine` unconstrained rather than failing the run.
        """
        try:
            return sorted(default_catalog())
        except (ProfileConfigError, OSError):
            return None

    @classmethod
    def request_from(cls, config: pytest.Config) -> Optional[SelectionRequest]:
        """This run's request, or None when no machine is named.

        The one place `core`'s `SelectionError` becomes a `pytest.UsageError`.
        """
        try:
            return SelectionRequest.of(
                machine=config.getoption(cls.MACHINE),
                gpus=config.getoption(cls.GPUS),
                ladder=config.getoption(cls.LADDER),
                rung=config.getoption(cls.RUNG),
                out_dir=config.getoption(cls.OUT_DIR),
            )
        except SelectionError as error:
            raise pytest.UsageError(str(error)) from error


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
        if default_markers().carries_requirement(mark.name):
            args = tuple(mark.args[:1])

        # Keyword form only; `selector.py` keys its rules on this string.
        reason = mark.kwargs.get("reason")
        kwargs: Dict[str, Any] = {"reason": reason} if isinstance(reason, str) else {}
        return Mark(name=mark.name, args=args, kwargs=kwargs)


class SelectionStash:
    """The three handoffs between the hooks above, keyed once at import.

    A `StashKey` is identity-based, so each key must stay a single shared
    object. They live here rather than on the records they carry, which are
    `core/` types and may not import pytest.
    """

    REQUEST = pytest.StashKey()
    SELECTION = pytest.StashKey()
    OUTPUT = pytest.StashKey()

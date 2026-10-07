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

    pytest --collect-only -p qa_selection.plugin --machine=B200 [--ladder=1,4,8]

Load with `-p` on the command line. No hardware is touched: every decision is
read from marks. The session header names the target, and a block after the
run breaks the selection down by rung and the deselection by reason.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from .core.marks import CollectedTest, Mark, default_markers
from .core.report import ArtifactNames, SelectionOutput, SelectionReport
from .core.selection import Selection, SelectionError, SelectionRequest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the selection options."""
    SelectionOptions.add_to(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Declare the resource markers, and resolve the target before collection."""
    MarkerDeclaration.add_to(config)
    request = SelectionOptions.request_from(config)
    if request is not None:
        config.stash[SelectionStash.REQUEST] = request


def pytest_report_header(config: pytest.Config) -> List[str]:
    """Name the target in the session header, which `-q` hides."""
    request = config.stash.get(SelectionStash.REQUEST, None)
    return [] if request is None else [TerminalSummary.header(request)]


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
    output = selection.report(Path(str(config.rootpath)))
    config.stash[SelectionStash.OUTPUT] = output


def pytest_terminal_summary(terminalreporter) -> None:
    """Render the human-readable block after pytest's own counts."""
    output = terminalreporter.config.stash.get(SelectionStash.OUTPUT, None)
    if output is not None:
        TerminalSummary.render(terminalreporter, output)


class TerminalSummary:
    """What a run prints: a header naming the target, and a block after the run.

    The block states each count, then its parts on indented lines: rungs under
    `selected` when the ladder has several, reasons under `deselected`, most
    frequent first. At `-v`, each reason's node ids follow it.
    """

    TITLE = "qa selection"
    LABEL_WIDTH = 14
    INDENT = "  "

    @classmethod
    def header(cls, request: SelectionRequest) -> str:
        """The one header line: machine, card, node size and ladder."""
        profile = request.profile
        return (
            f"{cls.TITLE}: {request.machine} (sm {profile.sm}, "
            f"{profile.max_gpu_per_node} GPUs/node), ladder {request.ladder}"
        )

    @classmethod
    def render(cls, terminalreporter, output: SelectionOutput) -> None:
        """Write the block through pytest's terminal reporter."""
        terminalreporter.section(cls.TITLE, sep="-")
        for line in cls.lines(output, verbose=terminalreporter.verbosity > 0):
            terminalreporter.line(line)

    @classmethod
    def lines(cls, output: SelectionOutput, verbose: bool) -> List[str]:
        """The block's lines, in order."""
        report = output.report
        lines = [
            cls.line("target", f"{report.machine}, ladder {report.ladder}"),
            cls.line("candidates", len(report.outcomes)),
            cls.line("selected", len(report.selected)),
        ]
        if len(report.ladder) > 1:
            lines += [
                cls.line(f"{cls.INDENT}{rung}{ArtifactNames.RUNG_UNIT}", len(outcomes))
                for rung, outcomes in report.rungs.items()
            ]
        lines += cls.deselected_lines(report, verbose)
        if output.written:
            names = " ".join(path.name for path in output.written)
            lines.append(cls.line("written", f"{output.written[0].parent}/: {names}"))
        return lines

    @classmethod
    def deselected_lines(cls, report: SelectionReport, verbose: bool) -> List[str]:
        """The deselected count, then one line per reason with its count.

        A test with several blockers counts under each, as in the record.
        """
        count = len(report.outcomes) - len(report.selected)
        width = len(str(count))
        lines = [cls.line("deselected", count)]
        reasons = sorted(report.deselected_by_reason.items(), key=lambda item: -len(item[1]))
        for reason, nodeids in reasons:
            lines.append(f"{cls.INDENT}{len(nodeids):>{width}}  {reason}")
            if verbose:
                lines += [f"{cls.INDENT}{' ' * width}  {cls.INDENT}{nodeid}" for nodeid in nodeids]
        return lines

    @classmethod
    def line(cls, label: str, value: object) -> str:
        """One `label   value` line, aligned with every other."""
        return f"{label:<{cls.LABEL_WIDTH}}{value}"


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
    """The plugin's three command-line options; `--help` states each one.

    `--machine` names what the rules are asked about, and `--ladder` the
    allocations its GPUs are divided into. One machine per invocation.
    """

    GROUP = "qa selection"

    MACHINE = "qa_selection_machine"
    LADDER = "qa_selection_ladder"
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
            choices=SelectionRequest.machines(),
            help="target machine to select for, from the profile catalogue. "
            "Absent leaves collection untouched, so loading this plugin "
            "without it changes nothing",
        )
        group.addoption(
            "--ladder",
            dest=cls.LADDER,
            metavar="RUNGS",
            default=None,
            help="ascending allocation sizes, comma separated, e.g. 1,4,8. "
            "Defaults to one rung of the machine's GPUs per node. One rung "
            "asks what a single allocation of that size can run; several "
            "divide the machine by an allocation policy, each test landing "
            "on the smallest rung that holds it. A test needing more GPUs "
            "than the largest rung is deselected. No rung may exceed the "
            "machine's GPUs per node",
        )
        group.addoption(
            "--selection-out-dir",
            dest=cls.OUT_DIR,
            metavar="DIR",
            default=None,
            help="write <machine>.json and one <machine>-<rung>gpu.ids per "
            "rung here, creating the directory if needed. Absent writes "
            "nothing. The directory is written into and never emptied, so a "
            "leftover list of this machine's from a different ladder is a "
            "usage error",
        )

    @classmethod
    def request_from(cls, config: pytest.Config) -> Optional[SelectionRequest]:
        """This run's request, or None when no machine is named.

        The one place `core`'s `SelectionError` becomes a `pytest.UsageError`.
        """
        try:
            return SelectionRequest.of(
                machine=config.getoption(cls.MACHINE),
                ladder=config.getoption(cls.LADDER),
                out_dir=config.getoption(cls.OUT_DIR),
            )
        except SelectionError as error:
            raise pytest.UsageError(str(error)) from error


class ItemView:
    """Reduces a pytest item to the `CollectedTest` that `core/` reads.

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

        # Keyword form only; `MachineCheck` keys its rules on this string.
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

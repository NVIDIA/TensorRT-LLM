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
"""Framework for the automated tests of the QA selection pytest plugin.

    run = selection.run("test_arch_cases.py", "--machine=H100")
    -> pytest --collect-only -p qa_selection.plugin --machine=H100 test_arch_cases.py

    run.selected          the node ids the run kept, in collection order
    run.record            <machine>.json, when the run named --selection-out-dir={out}
    run.outcome(nodeid)   one test's entry in that record
    run.ids(rung)         one rung's published identifier list
    run.written           every file name the run left in its output directory
    run.summary           the plugin's terminal block, line by line

`selection.refuse(...)` is the same call for a run that must fail as a usage
error; it returns the same object, with `run.result` for the message.
`selection.without_plugin(...)` runs the same command without `-p`.

Options pass through as the literal strings a user types. `{out}` is the one
substitution: it expands to a directory belonging to that run.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest


class KeptNodeIds:
    """Records the node ids one run kept.

    Uses pytest's own `pytest_collection_finish`, which fires after the
    plugin's `trylast` deselection hook.
    """

    def __init__(self) -> None:
        self.nodeids: List[str] = []

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Copy the surviving item set out of the run under test."""
        self.nodeids = [item.nodeid for item in session.items]


class SelectionRun:
    """One invocation of the plugin's command line, and what it produced."""

    #: Spelled out, never imported from `collection.ArtifactNames`: these are
    #: the tripwire for a rename of the plugin's own output.
    RECORD = "{machine}.json"
    RUNG_IDS = "{machine}-{rung}gpu.ids"

    #: The plugin's terminal block: a section title, then lines that each start
    #: with a padded lowercase label or with indentation.
    SUMMARY_TITLE = "qa selection"
    SUMMARY_LINE = re.compile(r"[a-z]+ {2,}\S| +\S")

    def __init__(
        self,
        command: str,
        machine: Optional[str],
        path: Path,
        out_dir: Path,
        result: pytest.RunResult,
        selected: List[str],
    ) -> None:
        self.command = command
        self.machine = machine
        self.path = path
        self.out_dir = out_dir
        self.result = result
        self.selected = selected

    @property
    def record(self) -> Dict[str, Any]:
        """`<machine>.json`, parsed."""
        return json.loads(self.artifact(self.RECORD.format(machine=self.machine)).read_text())

    def ids(self, rung: int) -> List[str]:
        """One rung's published identifier list, in the order it was written."""
        name = self.RUNG_IDS.format(machine=self.machine, rung=rung)
        return self.artifact(name).read_text(encoding="utf-8").splitlines()

    @property
    def written(self) -> List[str]:
        """Every file name this run left in its output directory, sorted."""
        if not self.out_dir.is_dir():
            return []
        return sorted(entry.name for entry in self.out_dir.iterdir() if entry.is_file())

    @property
    def summary(self) -> List[str]:
        """The plugin's terminal block, line by line, as printed.

        Empty when no block was printed. The block ends at the first line that
        neither starts with a label nor is indented.
        """
        lines: List[str] = []
        found_title = False
        for line in self.result.stdout.lines:
            if not found_title:
                found_title = f" {self.SUMMARY_TITLE} " in line
            elif self.SUMMARY_LINE.match(line):
                lines.append(line)
            else:
                break
        return lines

    def artifact(self, name: str) -> Path:
        """One file the run was expected to write; fails naming the command if absent."""
        path = self.out_dir / name
        assert path.is_file(), (
            f"{name} was not written by:\n  {self.command}\n"
            f"the output directory holds {self.written}"
        )
        return path

    def outcome(self, nodeid: str) -> Dict[str, Any]:
        """The record's entry for one node id; raises when there is none."""
        for outcome in self.record["tests"]:
            if outcome["nodeid"] == nodeid:
                return outcome
        raise AssertionError(
            f"{nodeid!r} is not in the record, which holds "
            f"{[outcome['nodeid'] for outcome in self.record['tests']]}"
        )


class SelectionHarness:
    """Runs one mock module from `cases/` through the plugin's command line."""

    CASES = Path(__file__).parent / "cases"
    MOCK_CONFTEST = "conftest.py"

    PLUGIN = "qa_selection.plugin"
    MODE = "--collect-only"

    OUT_DIR = "out"
    OUT_PLACEHOLDER = "{out}"
    MACHINE_OPTION = "--machine="

    #: Deselecting every candidate still writes a full record, and pytest
    #: reports it as `NO_TESTS_COLLECTED`. A usage error is `USAGE_ERROR` and
    #: still fails here.
    COMPLETED = (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED)

    def __init__(self, pytester: pytest.Pytester) -> None:
        self.pytester = pytester

    def run(self, case: str, *options: str) -> SelectionRun:
        """Collect `case` under `options`, failing loudly if the run could not."""
        return self.completed(self.invoke(case, *options))

    def without_plugin(self, case: str, *options: str) -> SelectionRun:
        """The same collection with the plugin not loaded: `-p` is not passed."""
        return self.completed(self.invoke(case, *options, plugin=False))

    def refuse(self, case: str, *options: str) -> SelectionRun:
        """Run `case` under `options`, requiring a usage error rather than a selection."""
        run = self.invoke(case, *options)
        assert run.result.ret == pytest.ExitCode.USAGE_ERROR, (
            f"the run under test was not refused:\n  {run.command}\n\n{run.result.stdout.str()}"
        )
        return run

    def completed(self, run: SelectionRun) -> SelectionRun:
        """`run`, once it is known to have finished collecting rather than refused."""
        assert run.result.ret in self.COMPLETED, (
            f"the run under test did not complete:\n  {run.command}\n\n{run.result.stdout.str()}"
        )
        return run

    def invoke(self, case: str, *options: str, plugin: bool = True) -> SelectionRun:
        """Run the command line and collect every observation, asserting nothing."""
        loaded = ["-p", self.PLUGIN] if plugin else []
        argv = [self.MODE, *loaded, *map(self.expand, options), self.stage(case)]
        kept = KeptNodeIds()
        result = self.pytester.runpytest(*argv, plugins=[kept])
        return SelectionRun(
            command=" ".join(["pytest", *argv]),
            machine=self.machine_in(options),
            path=self.pytester.path,
            out_dir=self.out_dir,
            result=result,
            selected=kept.nodeids,
        )

    def stage(self, case: str) -> str:
        """Copy the mock conftest and `case` into this run's directory."""
        for name in (self.MOCK_CONFTEST, case):
            shutil.copy(self.CASES / name, self.pytester.path / name)
        return case

    @property
    def out_dir(self) -> Path:
        """Where `{out}` points: a directory of this run's own."""
        return self.pytester.path / self.OUT_DIR

    def expand(self, option: str) -> str:
        """Substitute `{out}`, leaving every other option byte-identical."""
        return option.replace(self.OUT_PLACEHOLDER, str(self.out_dir))

    @classmethod
    def machine_in(cls, options: Iterable[str]) -> Optional[str]:
        """The machine `--machine=` names, or None when the run named none."""
        for option in options:
            if option.startswith(cls.MACHINE_OPTION):
                return option[len(cls.MACHINE_OPTION) :]
        return None

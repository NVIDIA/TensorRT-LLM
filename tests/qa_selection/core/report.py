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
"""Everything selection writes once collection is over.

    SelectionOutput.of(selection, rootdir) -> Artifacts.write(report, out_dir)
                                           -> TerminalSummary.render(...)

Two formats from one record: `.ids` files the pipeline filters with `awk`, and
a JSON record holding the counts and the per-test outcome. `plugin.py` holds
the hooks and renders the summary through pytest's terminal reporter.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .allocation import Assignment
from .artifacts import ArtifactNames
from .ladder import Ladder
from .machines import MachineProfile
from .selection import Selection


class NodeIds:
    """The identifier form this package emits.

    Node ids are unprefixed, matching the test lists on disk; a consumer
    joining against prefixed artifacts adds the prefix itself.
    """

    FORM = "unprefixed"

    @staticmethod
    def of(outcomes: Tuple["Outcome", ...]) -> List[str]:
        """Bare node ids, in collection order.

        No `TIMEOUT (N)`/`SKIP`/`XFAIL` suffix is added: a consumer filters its
        own list by first field, so the suffixes it carries survive.
        """
        return [outcome.nodeid for outcome in outcomes]


class SourceRevision:
    """The checkout the marks were read from.

    The commit alone: no timestamp, so regenerating produces no diff unless a
    `skip_*` decorator changed.
    """

    COMMAND = ("git", "rev-parse", "HEAD")
    TIMEOUT_S = 5

    @classmethod
    def of(cls, rootdir: Path) -> Optional[str]:
        """The checkout's commit, or None when it cannot be read."""
        try:
            done = subprocess.run(
                cls.COMMAND,
                cwd=str(rootdir),
                capture_output=True,
                text=True,
                timeout=cls.TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() or None if done.returncode == 0 else None


@dataclass(frozen=True)
class Outcome:
    """One collected test, as the record states it.

    `required_gpus` is read from a lower bound, never declared, so it always
    travels with the markers it was read from.
    """

    nodeid: str
    selected: bool
    blockers: Tuple[str, ...]
    required_gpus: int
    required_gpus_from: Tuple[str, ...]
    rung: Optional[int]

    @classmethod
    def of(cls, assignment: Assignment) -> "Outcome":
        """One assignment, flattened into the fields the record writes."""
        return cls(
            nodeid=assignment.nodeid,
            selected=assignment.selected,
            blockers=assignment.blockers,
            required_gpus=assignment.demand.required_gpus,
            required_gpus_from=assignment.demand.required_gpus_from,
            rung=assignment.rung,
        )

    def to_mapping(self) -> Dict[str, object]:
        """This outcome as the record's JSON object."""
        return {
            "nodeid": self.nodeid,
            "selected": self.selected,
            "blockers": list(self.blockers),
            "required_gpus": self.required_gpus,
            "required_gpus_from": list(self.required_gpus_from),
            "rung": self.rung,
        }


@dataclass(frozen=True)
class SelectionReport:
    """What one invocation decided, counted once for every output."""

    machine: str
    profile: MachineProfile
    ladder: Ladder
    source_revision: Optional[str]
    outcomes: Tuple[Outcome, ...]

    @classmethod
    def of(cls, selection: Selection, rootdir: Path) -> "SelectionReport":
        """Build the record from one collection run's decisions."""
        request = selection.request
        return cls(
            machine=request.machine,
            profile=request.profile,
            ladder=request.ladder,
            source_revision=SourceRevision.of(rootdir),
            outcomes=tuple(Outcome.of(assignment) for assignment in selection.assignments),
        )

    @property
    def feasible(self) -> Tuple[Outcome, ...]:
        """The tests this machine can run, whatever allocation they land in."""
        return tuple(outcome for outcome in self.outcomes if outcome.selected)

    @property
    def rungs(self) -> Dict[int, Tuple[Outcome, ...]]:
        """The feasible tests partitioned by allocation, empty rungs included.

        Every rung, not only the one a caller is running: one collection
        carries what all of them need.
        """
        return {
            rung: tuple(outcome for outcome in self.feasible if outcome.rung == rung)
            for rung in self.ladder
        }

    @property
    def unclassified_rung(self) -> int:
        """Where a consumer routes an identifier this run never collected.

        The smallest rung. Always empty here: a collect run cannot see what it
        did not collect.
        """
        return self.ladder.smallest

    @property
    def deselected_by_reason(self) -> Dict[str, List[str]]:
        """The dropped tests, grouped by the blocker that dropped each one.

        Answers the first question anyone asks of the record -- why did my test
        not run -- without the reader writing a grouper. Every reason here is
        one the curated table decided, by construction: a skip the table
        declines to encode produces no blocker and drops nothing.

        A test with several blockers appears under each of them, because
        "which rule dropped this" has as many answers as it has blockers.
        Built from the same outcomes as `tests[]`, in collection order, so it
        cannot disagree with them and the file stays byte-stable across runs.
        """
        grouped: Dict[str, List[str]] = {}
        for outcome in self.outcomes:
            for blocker in outcome.blockers:
                grouped.setdefault(blocker, []).append(outcome.nodeid)
        return grouped

    @property
    def counts(self) -> Dict[str, int]:
        """Every population's size."""
        return {
            "candidates": len(self.outcomes),
            "feasible": len(self.feasible),
            "deselected": len(self.outcomes) - len(self.feasible),
            "unclassified": 0,
        }

    def to_mapping(self) -> Dict[str, object]:
        """The record, as written to `<machine>.json`."""
        record: Dict[str, object] = {
            "machine": self.machine,
            "max_gpu_per_node": self.profile.max_gpu_per_node,
            "ladder": list(self.ladder),
            "source_revision": self.source_revision,
            "nodeid_form": NodeIds.FORM,
            "counts": self.counts,
            "rungs": {str(rung): len(outcomes) for rung, outcomes in self.rungs.items()},
            "unclassified": {"route_to_rung": self.unclassified_rung, "nodeids": []},
        }
        # Omitted entirely when nothing was dropped, as the terminal summary
        # omits a population with no members.
        deselected = self.deselected_by_reason
        if deselected:
            record["deselected_by_reason"] = deselected
        record["tests"] = [outcome.to_mapping() for outcome in self.outcomes]
        return record


class Artifacts:
    """The files `--selection-out-dir` receives.

    The JSON is the record; the `.ids` files need no parser:

        awk 'NR==FNR{keep[$0];next} ($1 in keep)' B200-4gpu.ids list.txt
    """

    INDENT = 2

    @classmethod
    def write(cls, report: SelectionReport, out_dir: Path) -> List[Path]:
        """Write the record and the identifier lists; return what was written."""
        out_dir.mkdir(parents=True, exist_ok=True)
        written = [cls.write_record(report, out_dir)]
        written += [
            cls.write_ids(out_dir / name, nodeids) for name, nodeids in cls.id_lists(report).items()
        ]
        return written

    @classmethod
    def write_record(cls, report: SelectionReport, out_dir: Path) -> Path:
        """Write `<machine>.json`."""
        path = out_dir / ArtifactNames.record(report.machine)
        path.write_text(json.dumps(report.to_mapping(), indent=cls.INDENT) + "\n")
        return path

    @staticmethod
    def write_ids(path: Path, nodeids: List[str]) -> Path:
        """Write one identifier list: bare node ids, one per line."""
        path.write_text("".join(f"{nodeid}\n" for nodeid in nodeids), encoding="utf-8")
        return path

    @staticmethod
    def id_lists(report: SelectionReport) -> Dict[str, List[str]]:
        """Filename -> node ids, for every list this run emits.

        One `<machine>-<rung>gpu.ids` per rung, written even when empty. The
        names come from `ArtifactNames`, the same source the configure-time
        guard checked the output directory against.
        """
        return {
            ArtifactNames.rung_ids(report.machine, rung): NodeIds.of(outcomes)
            for rung, outcomes in report.rungs.items()
        }


@dataclass(frozen=True)
class SelectionOutput:
    """The record one run produced, and the files it went to."""

    report: SelectionReport
    written: Tuple[Path, ...]

    @classmethod
    def of(cls, selection: Selection, rootdir: Path) -> "SelectionOutput":
        """Build the record, and write it when an output directory was named."""
        report = SelectionReport.of(selection, rootdir)
        out_dir = selection.request.out_dir
        return cls(
            report=report,
            written=tuple(Artifacts.write(report, out_dir)) if out_dir is not None else (),
        )


class TerminalSummary:
    """The human-readable block, rendered in pytest's terminal summary."""

    TITLE = "qa selection"

    # Every line is a label then a value, in one column pair.
    LABEL_WIDTH = 14

    @classmethod
    def render(cls, writer, output: SelectionOutput) -> None:
        """Write the summary block through pytest's terminal reporter."""
        writer.section(cls.TITLE, sep="-")
        for line in cls.lines(output):
            writer.line(line)

    @classmethod
    def lines(cls, output: SelectionOutput) -> List[str]:
        """The block's lines, so they can be asserted without a terminal."""
        report = output.report
        counts = report.counts
        lines = [
            cls.line("target", cls.target(report)),
            cls.line("candidates", counts["candidates"]),
            cls.line("feasible", f"{counts['feasible']}  ({counts['deselected']} deselected)"),
            cls.line("rungs", cls.rung_counts(report)),
        ]
        lines += [cls.line("written", path) for path in output.written]
        return lines

    @classmethod
    def line(cls, label: str, value: object) -> str:
        """One `label   value` line, aligned with every other."""
        return f"{label:<{cls.LABEL_WIDTH}}{value}"

    @staticmethod
    def target(report: SelectionReport) -> str:
        """The machine and the ladder decisions were made against."""
        return f"{report.machine}, ladder {report.ladder}"

    @staticmethod
    def rung_counts(report: SelectionReport) -> str:
        """One count per rung, in ladder order."""
        return "  ".join(f"{rung}: {len(outcomes)}" for rung, outcomes in report.rungs.items())

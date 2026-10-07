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
"""What a run concluded: the record, the `.ids` lists, and their file names.

    SelectionOutput.of(report, out_dir)                      -> the report, and the files written
    SelectionReport.of(profile, ladder, outcomes, rootdir)   -> the record
    ArtifactNames.written_by(machine, ladder)                -> every file a run writes
    ArtifactNames.orphans_in(out_dir, machine, ladder)       -> this machine's stale files

`<machine>.json` holds the counts and each test's outcome; `<machine>-<rung>gpu.ids`
holds one rung's selected node ids, one per line.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ladder import Ladder
from .machine import MachineProfile


@dataclass(frozen=True)
class SelectionOutput:
    """One run's report, and the files it was written to."""

    report: SelectionReport
    written: Tuple[Path, ...]

    @classmethod
    def of(cls, report: SelectionReport, out_dir: Optional[Path]) -> SelectionOutput:
        """`report`, written to `out_dir` when one was named."""
        return cls(
            report=report,
            written=tuple(Artifacts.write(report, out_dir)) if out_dir is not None else (),
        )


@dataclass(frozen=True)
class SelectionReport:
    """What one run decided, as every output reads it."""

    machine: str
    profile: MachineProfile
    ladder: Ladder
    source_revision: Optional[str]
    outcomes: Tuple[Outcome, ...]

    @classmethod
    def of(
        cls, profile: MachineProfile, ladder: Ladder, outcomes: Tuple[Outcome, ...], rootdir: Path
    ) -> SelectionReport:
        """The record of `outcomes`, decided for `profile` on `ladder`."""
        return cls(
            machine=profile.name,
            profile=profile,
            ladder=ladder,
            source_revision=SourceRevision.of(rootdir),
            outcomes=outcomes,
        )

    @property
    def selected(self) -> Tuple[Outcome, ...]:
        """The tests this run keeps, whichever rung holds them."""
        return tuple(outcome for outcome in self.outcomes if outcome.selected)

    @property
    def rungs(self) -> Dict[int, Tuple[Outcome, ...]]:
        """The selected outcomes by rung, for every rung of the ladder."""
        return {
            rung: tuple(outcome for outcome in self.selected if outcome.rung == rung)
            for rung in self.ladder
        }

    @property
    def unclassified_rung(self) -> int:
        """Where a consumer routes a node id this run did not collect: the smallest rung."""
        return self.ladder.smallest

    @property
    def deselected_by_reason(self) -> Dict[str, List[str]]:
        """Deselected node ids by blocker, in collection order; a test is under each one."""
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
            "selected": len(self.selected),
            "deselected": len(self.outcomes) - len(self.selected),
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
        # Omitted when nothing was deselected.
        deselected = self.deselected_by_reason
        if deselected:
            record["deselected_by_reason"] = deselected
        record["tests"] = [outcome.to_mapping() for outcome in self.outcomes]
        return record


@dataclass(frozen=True)
class Outcome:
    """One test's outcome: its blockers, its GPU demand and its source, its rung.

    `blockers` lists the card's first, then the GPU count's; a test with none is selected.
    """

    nodeid: str
    blockers: Tuple[str, ...]
    required_gpus: int
    required_gpus_from: Tuple[str, ...]
    rung: Optional[int]

    @property
    def selected(self) -> bool:
        """True when neither the rules nor the GPU count drop this test."""
        return not self.blockers

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


class Artifacts:
    """Writes the record and the `.ids` lists to the output directory."""

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
        """File name -> node ids, one list per rung, empty ones included."""
        return {
            ArtifactNames.rung_ids(report.machine, rung): NodeIds.of(outcomes)
            for rung, outcomes in report.rungs.items()
        }


class ArtifactNames:
    """The filenames a run produces, and which files in a directory are its own."""

    RECORD_SUFFIX = ".json"
    IDS_SUFFIX = ".ids"
    RUNG_UNIT = "gpu"

    @classmethod
    def record(cls, machine: str) -> str:
        """The JSON record, written on every run."""
        return f"{machine}{cls.RECORD_SUFFIX}"

    @classmethod
    def rung_ids(cls, machine: str, rung: int) -> str:
        """One rung's identifier list."""
        return f"{machine}-{rung}{cls.RUNG_UNIT}{cls.IDS_SUFFIX}"

    @classmethod
    def written_by(cls, machine: str, ladder: Ladder) -> Tuple[str, ...]:
        """Every file a run for `machine` on `ladder` writes."""
        return (cls.record(machine),) + tuple(cls.rung_ids(machine, r) for r in ladder)

    @classmethod
    def shapes_of(cls, machine: str) -> Tuple[re.Pattern, ...]:
        """This machine's file-name patterns, anchored so `B200` does not match `B200X.ids`.

        The record, a rung's list, and `<machine>.ids`, which no run writes.
        """
        name = re.escape(machine)
        return (
            re.compile(rf"^{name}{re.escape(cls.RECORD_SUFFIX)}$"),
            re.compile(rf"^{name}{re.escape(cls.IDS_SUFFIX)}$"),
            re.compile(rf"^{name}-[0-9]+{cls.RUNG_UNIT}{re.escape(cls.IDS_SUFFIX)}$"),
        )

    @classmethod
    def orphans_in(cls, out_dir: Path, machine: str, ladder: Ladder) -> List[str]:
        """This machine's files in `out_dir` that a run on `ladder` would not write."""
        written = set(cls.written_by(machine, ladder))
        shapes = cls.shapes_of(machine)
        return sorted(
            entry.name
            for entry in out_dir.iterdir()
            if entry.is_file()
            and entry.name not in written
            and any(shape.match(entry.name) for shape in shapes)
        )


class NodeIds:
    """The node-id form the lists use: unprefixed, as in the test lists."""

    FORM = "unprefixed"

    @staticmethod
    def of(outcomes: Tuple[Outcome, ...]) -> List[str]:
        """Bare node ids in collection order, with no `TIMEOUT`/`SKIP`/`XFAIL` suffix."""
        return [outcome.nodeid for outcome in outcomes]


class SourceRevision:
    """The commit the marks were read from; no timestamp, so a rerun writes the same record."""

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

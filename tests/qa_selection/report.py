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

    SelectionReport.of(selection) -> Artifacts.write(report, out_dir)
                                  -> TerminalSummary.render(...)

Two consumers, two formats. The pipeline reads `.ids` files with `awk` and
needs no parser; a human reads the JSON record or the terminal block. Both are
built from the same `SelectionReport`, so they cannot disagree.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .collection import Selection
from .core.allocation import Assignment, GpuDemand, Ladder
from .core.machines import MachineProfile
from .core.selector import CollectedTest


class NodeIds:
    """The identifier form this package emits, and the contract behind it.

    Test lists on disk are unprefixed; `parse_test_list` prefixes the entries at
    load time to match already-prefixed items. So emitting unprefixed ids is the
    on-disk contract, and a consumer joining against prefixed Jenkins artifacts
    must add the prefix itself. The form is named in the record rather than left
    to be inferred.
    """

    FORM = "unprefixed"

    @staticmethod
    def of(assignments: Tuple[Assignment, ...]) -> List[str]:
        """Bare node ids, in collection order, with no list decorations.

        No `TIMEOUT (N)`/`SKIP`/`XFAIL` suffix is added: a consumer filters its
        own list by first field, so every suffix it already carries survives.
        """
        return [assignment.nodeid for assignment in assignments]


class SourceRevision:
    """The checkout the marks were read from.

    Stamped into the record so a stale artifact is visible in a consumer's log
    rather than inferred from a test count. Nothing else is stamped -- notably
    no timestamp, which would make every regeneration differ and destroy the
    property that the artifact changes only when a `skip_*` decorator does.
    """

    COMMAND = ("git", "rev-parse", "HEAD")
    TIMEOUT_S = 5

    @classmethod
    def of(cls, rootdir: Path) -> Optional[str]:
        """The checkout's commit, or None when it cannot be read.

        Absent is a usable answer: the record says so, and the run continues.
        """
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

    `required_gpus` is an inference -- `skip_less_device(4)` declares a lower
    bound for running, not a demand -- so it never appears without the marks it
    was read from.
    """

    nodeid: str
    selected: bool
    blockers: Tuple[str, ...]
    required_gpus: int
    required_gpus_from: Tuple[str, ...]
    rung: Optional[int]
    unassignable: bool
    unknown_reasons: Tuple[str, ...]
    unkeyable_skipif: int

    @classmethod
    def of(cls, assignment: Assignment, view: CollectedTest) -> "Outcome":
        """One assignment plus the one fact only the collected view carries."""
        return cls(
            nodeid=assignment.nodeid,
            selected=assignment.decision.selected,
            blockers=assignment.decision.blockers,
            required_gpus=assignment.demand.required_gpus,
            required_gpus_from=assignment.demand.required_gpus_from,
            rung=assignment.rung,
            unassignable=assignment.unassignable,
            unknown_reasons=assignment.decision.unknown_skipif,
            unkeyable_skipif=cls.unkeyable_skipif_of(view),
        )

    @staticmethod
    def unkeyable_skipif_of(view: CollectedTest) -> int:
        """How many of this test's `skipif` marks carry no `reason=`.

        A keyless mark can never be given a rule, so it is not an unknown
        reason; counting it here is what keeps D22's two skipif populations
        apart. It never blocks: an unkeyable mark leaves the test selected.
        """
        return sum(1 for mark in view.iter_markers("skipif") if mark.skipif_reason is None)

    def to_mapping(self) -> Dict[str, object]:
        """This outcome as the record's JSON object."""
        return {
            "nodeid": self.nodeid,
            "selected": self.selected,
            "blockers": list(self.blockers),
            "required_gpus": self.required_gpus,
            "required_gpus_from": list(self.required_gpus_from),
            "rung": self.rung,
            "unassignable": self.unassignable,
            "unknown_reasons": list(self.unknown_reasons),
            "unkeyable_skipif": self.unkeyable_skipif,
        }


@dataclass(frozen=True)
class SelectionReport:
    """What one invocation decided, in the shape both outputs are built from.

    Counting happens here once so the `.ids` files, the JSON record and the
    terminal block cannot disagree about a number.
    """

    machine: str
    profile: MachineProfile
    ladder: Optional[Ladder]
    target_rung: Optional[int]
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
            target_rung=request.target_rung,
            source_revision=SourceRevision.of(rootdir),
            outcomes=tuple(
                Outcome.of(assignment, view)
                for assignment, view in zip(selection.assignments, selection.views)
            ),
        )

    @property
    def feasible(self) -> Tuple[Outcome, ...]:
        """The tests this machine can run, whatever allocation they land in."""
        return tuple(outcome for outcome in self.outcomes if outcome.selected)

    @property
    def live(self) -> Tuple[Outcome, ...]:
        """The tests this invocation actually kept.

        Equals `feasible` unless `--gpus` named a rung, matching
        `Selection.is_live` -- the same rule, applied to the same decisions.
        """
        if self.target_rung is None:
            return self.feasible
        return tuple(outcome for outcome in self.feasible if outcome.rung == self.target_rung)

    @property
    def rungs(self) -> Dict[int, Tuple[Outcome, ...]]:
        """The feasible tests partitioned by allocation, empty rungs included.

        One collection serves every rung: the partition is complete here, not
        only for the rung a caller happens to be running (design D14).
        """
        if self.ladder is None:
            return {}
        return {
            rung: tuple(outcome for outcome in self.feasible if outcome.rung == rung)
            for rung in self.ladder
        }

    @property
    def unassignable(self) -> Tuple[Outcome, ...]:
        """Tests this machine can run that no rung fits.

        Feasible on purpose: an infeasible test is not waiting for an
        allocation. These are folded into no rung, ever (task 4.7).
        """
        return tuple(outcome for outcome in self.feasible if outcome.unassignable)

    @property
    def unknown_reasons(self) -> Dict[str, List[str]]:
        """Skipif reasons with no rule, each with the tests that carry it.

        Kept, not deselected: an undecidable reason fails open. Expect three
        states under this key -- not yet curated, deliberately undecidable, and
        deliberately deferred -- which `core/`'s README tells apart.
        """
        found: Dict[str, List[str]] = {}
        for outcome in self.outcomes:
            for reason in outcome.unknown_reasons:
                found.setdefault(reason, []).append(outcome.nodeid)
        return found

    @property
    def unkeyable_skipif(self) -> List[str]:
        """Tests carrying a `skipif` with no `reason=` at all.

        Never merged into `unknown_reasons`: no rule could ever name these, so
        advising one would advise an impossible fix (design D22).
        """
        return [outcome.nodeid for outcome in self.outcomes if outcome.unkeyable_skipif]

    @property
    def unclassified_rung(self) -> int:
        """Where a consumer routes an identifier this run never collected.

        Empty in this record by construction -- a collect run cannot see what it
        did not collect. The key and the routing policy are stated anyway so a
        consumer holding a stale list has somewhere defined to put them, and a
        wrong guess costs the cheapest allocation (design D16).
        """
        return self.ladder.smallest if self.ladder is not None else GpuDemand.ASSUMED_GPUS

    @property
    def counts(self) -> Dict[str, int]:
        """Every population's size, for a caller that logs numbers only."""
        return {
            "candidates": len(self.outcomes),
            "feasible": len(self.feasible),
            "deselected": len(self.outcomes) - len(self.feasible),
            "live": len(self.live),
            "unassignable": len(self.unassignable),
            "unknown_reasons": sum(len(ids) for ids in self.unknown_reasons.values()),
            "unkeyable_skipif": len(self.unkeyable_skipif),
            "unclassified": 0,
        }

    def to_mapping(self) -> Dict[str, object]:
        """The record, as written to `<machine>.json`."""
        return {
            "machine": self.machine,
            "gpu_count": self.profile.gpu_count,
            "max_gpu_per_node": self.profile.max_gpu_per_node,
            "ladder": list(self.ladder) if self.ladder is not None else None,
            "target_rung": self.target_rung,
            "source_revision": self.source_revision,
            "nodeid_form": NodeIds.FORM,
            "counts": self.counts,
            "rungs": {str(rung): len(outcomes) for rung, outcomes in self.rungs.items()},
            "unknown_reasons": self.unknown_reasons,
            "unkeyable_skipif": self.unkeyable_skipif,
            "unclassified": {"route_to_rung": self.unclassified_rung, "nodeids": []},
            "tests": [outcome.to_mapping() for outcome in self.outcomes],
        }


class Artifacts:
    """The files `--selection-out-dir` receives.

    Both formats, side by side: the JSON is the auditable contract, the `.ids`
    files make the pipeline's common path a one-liner with no parser --

        awk 'NR==FNR{keep[$0];next} ($1 in keep)' B200-4gpu.ids list.txt
    """

    RECORD_SUFFIX = ".json"
    IDS_SUFFIX = ".ids"
    INDENT = 2

    @classmethod
    def write(cls, report: SelectionReport, out_dir: Path) -> List[Path]:
        """Write the record and the identifier lists; return what was written.

        The record is written on every run. Which `.ids` files join it is
        decided by `--ladder` alone, and there is no third mode.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        written = [cls.write_record(report, out_dir)]
        written += [
            cls.write_ids(out_dir / name, nodeids) for name, nodeids in cls.id_lists(report).items()
        ]
        return written

    @classmethod
    def write_record(cls, report: SelectionReport, out_dir: Path) -> Path:
        """Write `<machine>.json`, the auditable record of the whole run."""
        path = out_dir / f"{report.machine}{cls.RECORD_SUFFIX}"
        path.write_text(json.dumps(report.to_mapping(), indent=cls.INDENT) + "\n")
        return path

    @staticmethod
    def write_ids(path: Path, nodeids: List[str]) -> Path:
        """Write one identifier list: bare node ids, one per line."""
        path.write_text("".join(f"{nodeid}\n" for nodeid in nodeids))
        return path

    @classmethod
    def id_lists(cls, report: SelectionReport) -> Dict[str, List[str]]:
        """Filename -> node ids, for every list this run emits.

        Without a ladder: one `<machine>.ids` of everything feasible. With one:
        `<machine>-<rung>gpu.ids` per rung, every rung written even when empty,
        so a missing file means a failed run rather than an empty allocation.
        """
        if report.ladder is None:
            return {
                f"{report.machine}{cls.IDS_SUFFIX}": NodeIds.of(report.feasible),
            }
        return {
            f"{report.machine}-{rung}gpu{cls.IDS_SUFFIX}": NodeIds.of(outcomes)
            for rung, outcomes in report.rungs.items()
        }


class TerminalSummary:
    """The human-readable block, rendered in pytest's terminal summary.

    Here rather than in the collection hook so it prints once, after pytest's
    own counts, where a reader is already looking for totals.
    """

    TITLE = "qa selection"

    # Every line is `label` then value, so the block reads as one column pair.
    LABEL_WIDTH = 14

    @classmethod
    def render(cls, writer, report: SelectionReport, written: List[Path]) -> None:
        """Write the summary block through pytest's terminal reporter."""
        writer.section(cls.TITLE, sep="-")
        for line in cls.lines(report, written):
            writer.line(line)

    @classmethod
    def lines(cls, report: SelectionReport, written: List[Path]) -> List[str]:
        """The block's lines, so they can be asserted without a terminal."""
        counts = report.counts
        lines = [
            cls.line("target", cls.target(report)),
            cls.line("candidates", counts["candidates"]),
            cls.line("feasible", f"{counts['feasible']}  ({counts['deselected']} deselected)"),
            cls.line("live", f"{counts['live']}{cls.rung_note(report)}"),
        ]
        if report.ladder is not None:
            lines.append(cls.line("rungs", cls.rung_counts(report)))
        lines += cls.population_lines(report)
        lines += [cls.line("written", path) for path in written]
        return lines

    @classmethod
    def line(cls, label: str, value: object) -> str:
        """One `label   value` line, aligned with every other."""
        return f"{label:<{cls.LABEL_WIDTH}}{value}"

    @staticmethod
    def target(report: SelectionReport) -> str:
        """The machine and the allocation shape decisions were made against."""
        target = f"{report.machine}, {report.profile.gpu_count} GPUs"
        return target if report.ladder is None else f"{target}, ladder {report.ladder}"

    @staticmethod
    def rung_note(report: SelectionReport) -> str:
        """Names the rung when `--gpus` narrowed the live set to one."""
        return "" if report.target_rung is None else f"  (rung {report.target_rung} only)"

    @staticmethod
    def rung_counts(report: SelectionReport) -> str:
        """One count per rung, in ladder order."""
        return "  ".join(f"{rung}: {len(outcomes)}" for rung, outcomes in report.rungs.items())

    @classmethod
    def population_lines(cls, report: SelectionReport) -> List[str]:
        """The three unresolved populations, each with the fix it calls for.

        Printed only when non-empty, and never merged: each has a different
        owner, so a combined count would be one nobody can act on (design D22).
        """
        counts = report.counts
        populations = [
            ("unassignable", counts["unassignable"], "demand exceeds every rung"),
            ("unknown", counts["unknown_reasons"], "skipif reason has no rule in core/rules.json"),
            ("unkeyable", counts["unkeyable_skipif"], "skipif carries no reason="),
        ]
        return [
            cls.line(name, f"{count}  -- {advice}") for name, count, advice in populations if count
        ]

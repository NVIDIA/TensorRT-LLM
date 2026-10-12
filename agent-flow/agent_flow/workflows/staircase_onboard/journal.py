# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run journal — recovery state, written as events happen.

Staircase's rule, kept: **the journal is written as events happen, never
retrospectively.** After a session is cut, the journal is the only
explanation of what is on disk. A journal filled in at the end is a summary,
and a summary cannot tell you which of two half-finished things was in
flight.

It lives in the workspace, not in the repository. An onboarding run's
evidence is real and worth keeping, but it is evidence *about a run*, and a
product tree is the wrong home for it — so the journal, the per-dispatch
reports and the plan artifacts stay beside the checkpoint state, and only a
distilled record travels into the repo at delivery.

Timestamps are UTC, read from the clock. Never estimate one from how long
something felt.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPORT_NAME = re.compile(r"^(\d+)-")
#: Beside ``reports/``, not in it: readers of the reports directory expect
#: nothing but reports there.
_CLAIMS_DIRNAME = ".dispatch-ordinals"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(text: str) -> str:
    """Fold arbitrary text into a safe, readable filename component."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:60] or "unnamed"


@dataclass
class Journal:
    """Append-only run log plus the per-dispatch report directory."""

    path: Path
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def reports_dir(self) -> Path:
        return self.path.parent / "reports"

    def open(self, *, slug: str, header_lines: list[str] | None = None) -> None:
        """Start the journal, or continue an existing one.

        Continuing is the common case: re-running the same invocation is how
        a run resumes, and the second run's events belong in the same file
        as the first's.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.read_text(encoding="utf-8").strip():
            self.event("run resumed", f"slug={slug}")
            return
        # A new journal is a new run: its dispatch numbers start again at 01.
        shutil.rmtree(self.claims_dir, ignore_errors=True)
        lines = [f"# Onboarding journal — {slug}", ""]
        lines.extend(header_lines or [])
        lines.append("")
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.event("run opened", f"slug={slug}")

    def event(self, what: str, detail: str = "") -> None:
        """Append one timestamped line. The unit of recovery state."""
        line = f"- `{_utc_now()}` **{what}**"
        if detail:
            line += f" — {detail}"
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def section(self, title: str, body: str) -> None:
        """Append a titled block — a preflight report, a gate summary, a verdict."""
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n## {_utc_now()} — {title}\n\n{body.rstrip()}\n")

    @property
    def claims_dir(self) -> Path:
        """Where each dispatch number is claimed, one empty file per number."""
        return self.path.parent / _CLAIMS_DIRNAME

    def _highest_ordinal(self) -> int:
        seen = [0]
        if self.reports_dir.is_dir():
            seen.extend(
                int(match.group(1))
                for entry in self.reports_dir.iterdir()
                if (match := _REPORT_NAME.match(entry.name))
            )
        if self.claims_dir.is_dir():
            seen.extend(
                int(entry.name) for entry in self.claims_dir.iterdir() if entry.name.isdigit()
            )
        return max(seen)

    def next_ordinal(self) -> int:
        """Claim and return the next dispatch number, counted across the whole run.

        Sequential across every agent, so ``NN`` names exactly one dispatch
        and the next ordinal is visible without counting.

        **Claiming is the allocation**, and it is atomic across threads and
        processes: the number is taken by creating ``.dispatch-ordinals/<NN>``
        with ``O_CREAT | O_EXCL``, which exactly one caller can win. Reading
        the highest number and then writing the report, as this used to, let
        two concurrent smiths of one batch both read the same maximum and
        both record dispatch ``NN`` (duplicate numbers in ``journal.md``,
        measured on the DeepSeek-V4.1 run). A loser moves to the next number.
        """
        self.claims_dir.mkdir(parents=True, exist_ok=True)
        candidate = self._highest_ordinal() + 1
        while True:
            try:
                fd = os.open(self.claims_dir / str(candidate), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                candidate += 1
                continue
            os.close(fd)
            return candidate

    def record_dispatch(self, agent: str, what: str, prompt: str) -> Path:
        """Write a dispatch prompt down **before it is sent**, under a freshly claimed ordinal.

        An agent's context does not survive a session cut, so a child's
        prompt only survives if its parent wrote it first. This is also why
        the file is created here rather than when the agent returns: a run
        that dies mid-dispatch still explains what it was doing.
        """
        ordinal = self.next_ordinal()
        path = self.reports_dir / f"{ordinal:02d}-{agent}-{_slugify(what)}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# {ordinal:02d} · {agent} · {what}\n\n"
            f"- dispatched: `{_utc_now()}`\n\n"
            f"## Dispatch prompt\n\n```\n{prompt}\n```\n\n"
            f"## Result\n\n_pending_\n",
            encoding="utf-8",
        )
        self.event(f"dispatch {ordinal:02d} · {agent}", what)
        return path

    def latest_report(self, agent: str, what: str) -> Path | None:
        """The newest report :meth:`record_dispatch` wrote for this agent and subject.

        Derived from the file name rather than remembered, so a process that
        never made the dispatch -- a resume, or a retry owed from an earlier
        phase -- can still point the next attempt at what the last one did.
        """
        if not self.reports_dir.is_dir():
            return None
        name = f"{agent}-{_slugify(what)}.md"
        found = [
            (int(match.group(1)), entry)
            for entry in self.reports_dir.iterdir()
            if (match := _REPORT_NAME.match(entry.name)) and entry.name[match.end() :] == name
        ]
        return max(found)[1] if found else None

    def record_result(self, report_path: Path, result: str) -> None:
        """Replace a dispatch report's pending result with the agent's final message."""
        text = report_path.read_text(encoding="utf-8")
        text = text.replace(
            "## Result\n\n_pending_\n",
            f"## Result\n\n- returned: `{_utc_now()}`\n\n{result.rstrip()}\n",
        )
        report_path.write_text(text, encoding="utf-8")

    def close(self, *, summary: str) -> None:
        self.section("run closed", summary)


__all__ = ["Journal"]

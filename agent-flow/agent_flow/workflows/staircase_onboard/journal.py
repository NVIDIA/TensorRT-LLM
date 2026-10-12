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

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPORT_NAME = re.compile(r"^(\d+)-")


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

    def next_ordinal(self) -> int:
        """The next dispatch number, counted across the whole run.

        Sequential across every agent, so ``NN`` names exactly one dispatch
        and the next ordinal is visible without counting.
        """
        if not self.reports_dir.is_dir():
            return 1
        seen = [
            int(match.group(1))
            for entry in self.reports_dir.iterdir()
            if (match := _REPORT_NAME.match(entry.name))
        ]
        return max(seen, default=0) + 1

    def record_dispatch(self, agent: str, what: str, prompt: str) -> Path:
        """Write a dispatch prompt down **before it is sent**, and reserve its ordinal.

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

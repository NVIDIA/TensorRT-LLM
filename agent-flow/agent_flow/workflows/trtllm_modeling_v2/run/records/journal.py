# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run journal — recovery state, written as events happen.

The rule: **the journal is written as events happen, never
retrospectively.** After a session is cut, the journal is the only
explanation of what is on disk.

Timestamps are UTC, read from the clock.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPORT_NAME = re.compile(r"^(\d+)-")
#: An event line as :meth:`Journal.event` writes it: its time and what happened.
_EVENT_LINE = re.compile(r"^- `([^`]+)` \*\*(.+?)\*\*")
#: The event a new journal begins with.
OPENED = "run opened"
#: The first time the run goes past preflight to its agents: when its clock starts.
STARTED = "run started"


def utc_now() -> str:
    """Now, in UTC, to the second: how the run's records stamp a time."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:60] or "unnamed"


@dataclass
class Journal:
    """Append-only run log plus the per-dispatch report directory."""

    path: Path
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    #: The last dispatch number handed out; read off ``reports/`` on first use.
    _ordinal: int | None = field(default=None, repr=False, compare=False)

    @property
    def reports_dir(self) -> Path:
        return self.path.parent / "reports"

    def open(self, *, slug: str, header_lines: list[str] | None = None) -> None:
        """Start the journal, or continue an existing one (a resumed run's)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.read_text(encoding="utf-8").strip():
            self.event("run resumed", f"slug={slug}")
            return
        lines = [f"# Onboarding journal — {slug}", ""]
        lines.extend(header_lines or [])
        lines.append("")
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.event(OPENED, f"slug={slug}")

    def mark_started(self) -> None:
        """Say the run goes past preflight to its agents, the first time it does.

        A ``--dry-run`` or a failed preflight opens the journal too, so its
        ``run opened`` says when the journal began, not when the run's work
        did; the milestones are timed from this.
        """
        if self._first(STARTED) is None:
            self.event(STARTED, "past preflight: the first agent turn follows")

    def opened_at(self) -> datetime | None:
        """When the run started: its first ``run started`` event, else its first ``run opened``."""
        return self._first(STARTED) or self._first(OPENED)

    def _first(self, what: str) -> datetime | None:
        """When the journal's first ``what`` event was; ``None`` without one."""
        if not self.path.is_file():
            return None
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                match = _EVENT_LINE.match(line)
                if match and match.group(2) == what:
                    try:
                        return datetime.fromisoformat(match.group(1))
                    except ValueError:
                        return None
        return None

    def event(self, what: str, detail: str = "") -> None:
        """Append one timestamped line. The unit of recovery state."""
        line = f"- `{utc_now()}` **{what}**"
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
                handle.write(f"\n## {utc_now()} — {title}\n\n{body.rstrip()}\n")

    def next_ordinal(self) -> int:
        """The next dispatch number, counted across the whole run, a resume included.

        One orchestrator holds a workspace at a time (its lock), so the number
        is the highest report's plus one, handed out under :attr:`lock`.
        """
        with self.lock:
            if self._ordinal is None:
                reports = self.reports_dir.iterdir() if self.reports_dir.is_dir() else ()
                self._ordinal = max(
                    [0, *(int(m.group(1)) for e in reports if (m := _REPORT_NAME.match(e.name)))]
                )
            self._ordinal += 1
            return self._ordinal

    def record_dispatch(self, agent: str, what: str, prompt: str) -> Path:
        """Write a dispatch prompt down **before it is sent**, under a freshly claimed ordinal."""
        ordinal = self.next_ordinal()
        path = self.reports_dir / f"{ordinal:02d}-{agent}-{_slugify(what)}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# {ordinal:02d} · {agent} · {what}\n\n"
            f"- dispatched: `{utc_now()}`\n\n"
            f"## Dispatch prompt\n\n```\n{prompt}\n```\n\n"
            f"## Result\n\n_pending_\n",
            encoding="utf-8",
        )
        self.event(f"dispatch {ordinal:02d} · {agent}", what)
        return path

    def _reports(self, agent: str, what: str) -> list[Path]:
        """The reports :meth:`record_dispatch` wrote for this agent and subject, oldest first."""
        if not self.reports_dir.is_dir():
            return []
        name = f"{agent}-{_slugify(what)}.md"
        found = sorted(
            (int(match.group(1)), entry)
            for entry in self.reports_dir.iterdir()
            if (match := _REPORT_NAME.match(entry.name)) and entry.name[match.end() :] == name
        )
        return [entry for _, entry in found]

    def returned_reports(self, agent: str, what: str) -> list[Path]:
        """Every report for this agent and subject whose agent returned, oldest first."""
        return [
            entry
            for entry in self._reports(agent, what)
            if "## Result\n\n_pending_\n" not in entry.read_text(encoding="utf-8")
        ]

    def record_result(self, report_path: Path, result: str) -> None:
        """Replace a dispatch report's pending result with the agent's final message."""
        text = report_path.read_text(encoding="utf-8")
        text = text.replace(
            "## Result\n\n_pending_\n",
            f"## Result\n\n- returned: `{utc_now()}`\n\n{result.rstrip()}\n",
        )
        report_path.write_text(text, encoding="utf-8")

    def close(self, *, summary: str) -> None:
        self.section("run closed", summary)


__all__ = ["Journal", "utc_now"]

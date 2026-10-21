# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run journal — recovery state, written as events happen.

The rule: **the journal is written as events happen, never
retrospectively.** After a session is cut, the journal is the only
explanation of what is on disk.

Timestamps are UTC, read from the clock.
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
#: An event line as :meth:`Journal.event` writes it: its time and what happened.
_EVENT_LINE = re.compile(r"^- `([^`]+)` \*\*(.+?)\*\*")
#: The event a new journal begins with.
OPENED = "run opened"
#: The first time the run goes past preflight to its agents: when its clock starts.
STARTED = "run started"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(text: str) -> str:
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
        """Start the journal, or continue an existing one (a resumed run's)."""
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

        **Claiming is the allocation**, and it is atomic across threads and
        processes: the number is taken by creating ``.dispatch-ordinals/<NN>``
        with ``O_CREAT | O_EXCL``, which exactly one caller can win. A loser
        moves to the next number.
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
        """Write a dispatch prompt down **before it is sent**, under a freshly claimed ordinal."""
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

    def latest_report(self, agent: str, what: str) -> Path | None:
        """The newest report :meth:`record_dispatch` wrote for this agent and subject; None without one."""
        reports = self._reports(agent, what)
        return reports[-1] if reports else None

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
            f"## Result\n\n- returned: `{_utc_now()}`\n\n{result.rstrip()}\n",
        )
        report_path.write_text(text, encoding="utf-8")

    def close(self, *, summary: str) -> None:
        self.section("run closed", summary)


__all__ = ["Journal"]

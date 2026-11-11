# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run journal: ``journal.md``, written as events happen, never in retrospect.

After a process is cut, the journal is what explains the files on disk. Each turn's own
prompt, activity and answer are under ``turns/`` (``agents/turns.py``). Times are UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    """Now, in UTC, to the second: how the run's records stamp a time."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Journal:
    """The append-only log of one workspace."""

    path: Path

    def _append(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def open(self, *, slug: str, header_lines: list[str] | None = None) -> None:
        """Start the journal, or go on with the one an earlier start of the run left."""
        if self.path.exists() and self.path.read_text(encoding="utf-8").strip():
            self.event("run resumed", f"slug={slug}")
            return
        self._append(
            "\n".join([f"# Onboarding journal: {slug}", "", *(header_lines or []), "", ""])
        )
        self.event("run opened", f"slug={slug}")

    def event(self, what: str, detail: str = "") -> None:
        """Append one time-stamped line: the unit of what the journal records."""
        self._append(f"- `{utc_now()}` **{what}**" + (f": {detail}" if detail else "") + "\n")

    def section(self, title: str, body: str) -> None:
        """Append a titled block: a preflight report, the deployments, the run's close."""
        self._append(f"\n## {utc_now()}: {title}\n\n{body.rstrip()}\n")

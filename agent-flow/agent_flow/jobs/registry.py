"""Durable per-node job registry backed by an atomically-written ``jobs.json``.

Each node records the detached commands it launched so a later (re)run can
re-attach to a still-running/completed job instead of re-submitting it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

VALID_STATES = frozenset({"EXPECTED", "SUBMITTED", "RUNNING", "COMPLETED", "FAILED", "GONE"})


@dataclass
class JobEntry:
    """One recorded detached command (one row of ``jobs.json``)."""

    handle_key: str
    kind: str
    handle: dict
    state: str
    result_path: str | None = None
    submitted_at: str | None = None

    def __post_init__(self) -> None:
        if self.state not in VALID_STATES:
            raise ValueError(f"invalid job state {self.state!r}")


@dataclass
class JobRegistry:
    """Read/write a node's ``jobs.json``, keyed by ``handle_key``."""

    path: Path
    _entries: dict[str, JobEntry] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.is_file():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for row in raw:
                self._entries[row["handle_key"]] = JobEntry(**row)

    def get(self, handle_key: str) -> JobEntry | None:
        return self._entries.get(handle_key)

    def all(self) -> list[JobEntry]:
        return list(self._entries.values())

    def upsert(self, entry: JobEntry) -> None:
        self._entries[entry.handle_key] = entry
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps([asdict(e) for e in self._entries.values()], indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)  # atomic

"""Shared JSON reservation table under an advisory file lock.

Both the machine-allocation table (``tray.py``) and the worktree-slot table
(``worktree.py``) are the same object: N named slots, at most one holder each,
an append-only history, and a rendered markdown view for humans. A markdown
file edited by hand cannot do this — two agents editing it in the same minute
silently lose one edit — so the canonical state is JSON written under
``flock`` and the markdown is regenerated from it.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime
from pathlib import Path


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()


class LockedTable:
    """Context manager holding an exclusive lock over one reservation table.

    The lock is a sidecar file, so the table itself is only ever opened for a
    whole-file read or an atomic replace.
    """

    def __init__(self, json_path: Path, md_path: Path, default: dict, renderer):
        self.json_path = Path(json_path)
        self.md_path = Path(md_path)
        self.lock_path = self.json_path.with_suffix(".lock")
        self.default = default
        self.renderer = renderer
        self.data: dict = {}

    def __enter__(self) -> LockedTable:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.lock_path, "a+")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        if self.json_path.exists():
            self.data = json.loads(self.json_path.read_text())
        else:
            self.data = json.loads(json.dumps(self.default))
        self.data.setdefault("slots", {})
        self.data.setdefault("history", [])
        return self

    def __exit__(self, *exc) -> None:
        try:
            tmp = self.json_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=2))
            os.replace(tmp, self.json_path)
            self.md_path.write_text(self.renderer(self.data))
        finally:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()

    def log(self, line: str) -> None:
        self.data["history"].append(f"{now()} {line}")


def reconcile(data: dict, declared: dict[str, dict], aliases: dict[str, str]) -> list[str]:
    """Bring an on-disk table in line with the configured slot set.

    Three things happen, in this order, and each is recorded in the history:

    * a slot stored under an alias is renamed to its canonical key (a rename in
      the config must not orphan a live reservation),
    * slots declared in the config but absent from the table are added free,
    * declared metadata (description, job id when the table has none) is
      refreshed.

    Slots present on disk but no longer declared are KEPT, held or not: dropping
    them would silently discard a reservation whose holder is still running.
    """
    notes: list[str] = []
    slots = data["slots"]
    for alias, canonical_key in aliases.items():
        if alias in slots and canonical_key not in slots:
            slots[canonical_key] = slots.pop(alias)
            notes.append(f"renamed slot {alias} -> {canonical_key}")
    for key, meta in declared.items():
        if key not in slots:
            slots[key] = {"holder": None, "purpose": None, "since": None, **meta}
            notes.append(f"added slot {key}")
            continue
        row = slots[key]
        for field_name, value in meta.items():
            if value and not row.get(field_name):
                row[field_name] = value
        if meta.get("description") and row.get("description") != meta["description"]:
            row["description"] = meta["description"]
    ordered = {k: slots[k] for k in declared if k in slots}
    ordered.update({k: v for k, v in slots.items() if k not in ordered})
    data["slots"] = ordered
    for note in notes:
        data["history"].append(f"{now()} {note}")
    return notes

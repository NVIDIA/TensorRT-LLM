# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a gate measured, on which tree, kept for the next tree to be compared against.

An append-only JSON-lines file, one record per line, never rewritten. ``key``
is what makes two records comparable: two records with the same key measured
the same thing on (possibly) different trees.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

#: The file's name in a run's workspace.
RESULTS_FILENAME = "gate_results.jsonl"


KIND_ACCEPTANCE = "acceptance"


KIND_FINDING = "finding"


#: A checkpoint the stock implementation cannot load, on one GPU architecture:
#: the acceptance gate's ``=off`` leg is skipped from then on.
KIND_STOCK_UNAVAILABLE = "stock_unavailable"


@dataclass(frozen=True)
class GateRecord:
    """One measurement, or one finding."""

    kind: str
    key: Mapping[str, Any]
    #: ``git describe --always`` of the tree measured.
    tree: str
    passed: bool
    values: Mapping[str, Any] = field(default_factory=dict)
    #: The full commit, when git could answer; a commit range starts here.
    commit: str | None = None
    recorded_at: str = ""

    def to_json(self) -> str:
        """One line, keys sorted, so two runs' files compare line by line."""
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GateRecord":
        """The inverse of :meth:`to_json`; unknown keys are ignored."""
        return cls(
            kind=str(data.get("kind", "")),
            key=dict(data.get("key") or {}),
            tree=str(data.get("tree", "")),
            passed=bool(data.get("passed", False)),
            values=dict(data.get("values") or {}),
            commit=data.get("commit"),
            recorded_at=str(data.get("recorded_at", "")),
        )


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _same_key(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return json.dumps(dict(a), sort_keys=True) == json.dumps(dict(b), sort_keys=True)


class GateRecords:
    """An append-only JSON-lines file of :class:`GateRecord`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: GateRecord) -> GateRecord:
        """Write ``record`` as one line, stamping the time when it has none.

        One ``write`` of one line on a file opened for append, so two writers
        never interleave inside a record.
        """
        if not record.recorded_at:
            record = GateRecord(**{**asdict(record), "recorded_at": _now()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = record.to_json() + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        return record

    def all(self, kind: str | None = None) -> list[GateRecord]:
        """Every readable record, oldest first. A torn or foreign line is skipped."""
        if not self.path.is_file():
            return []
        records: list[GateRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if not isinstance(data, dict):
                continue
            record = GateRecord.from_mapping(data)
            if kind is None or record.kind == kind:
                records.append(record)
        return records

    def latest(
        self,
        kind: str,
        key: Mapping[str, Any],
        *,
        passed_only: bool = True,
        exclude_tree: str | None = None,
    ) -> GateRecord | None:
        """The newest record of ``kind`` with exactly ``key``.

        ``passed_only`` (the default) skips failed measurements: a regression
        that failed its gate must not become the next tree's baseline.
        ``exclude_tree`` skips records of the tree being measured now.
        """
        for record in reversed(self.all(kind)):
            if passed_only and not record.passed:
                continue
            if exclude_tree is not None and record.tree == exclude_tree:
                continue
            if _same_key(record.key, key):
                return record
        return None


def open_findings(
    records: GateRecords, acknowledged: Mapping[str, str] | None = None
) -> list[GateRecord]:
    """Findings whose newest record is still unresolved and not acknowledged.

    A later record with the same key and ``passed`` true resolves a finding.
    ``acknowledged`` is the CURRENT task's ``accuracy.floor.acknowledge``
    mapping, read at delivery.
    """
    newest: dict[str, GateRecord] = {}
    for record in records.all(KIND_FINDING):
        newest[json.dumps(dict(record.key), sort_keys=True)] = record
    reasons = acknowledged or {}
    return [
        record
        for record in newest.values()
        if not record.passed and str(record.key.get("unit")) not in reasons
    ]


def records_for(workspace: str | Path) -> GateRecords:
    """The results file of a run's workspace."""
    return GateRecords(Path(workspace) / RESULTS_FILENAME)


__all__ = [
    "GateRecord",
    "GateRecords",
    "KIND_ACCEPTANCE",
    "KIND_FINDING",
    "KIND_STOCK_UNAVAILABLE",
    "RESULTS_FILENAME",
    "open_findings",
    "records_for",
]

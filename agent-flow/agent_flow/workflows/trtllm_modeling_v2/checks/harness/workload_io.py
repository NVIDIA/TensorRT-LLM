# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The record format the harness prints.

Standard library only, because this module runs on both sides: imported by the
orchestrator (a login-node interpreter without numpy) to parse a check's output, and by
the harness scripts under the measured tree's interpreter.

A harness prints one line per result, ``<MARKER> <json>``, the JSON on one line with
ASCII escapes. A check reads ONLY those lines, never the engine's log, so a keyword
cannot match ``CPUs 72-143`` in a log line, and a prefix added by ``srun -l`` or a
logger does not hide a record.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

#: One per request the engine driver ran.
RECORD_MARKER = "MV2FLOW_RECORD"
#: What one rank's executor worker runs, printed by that worker as it sets its
#: engine up, once per rank (see ``rankhook/sitecustomize.py``).
RANK_ENGINE_MARKER = "MV2FLOW_RANK_ENGINE"

# ------------------------------------------------------------------- records


def emit(marker: str, payload: Mapping[str, Any]) -> None:
    """Print ``<marker> <json>`` on one line and flush."""
    line = json.dumps(dict(payload), ensure_ascii=True, sort_keys=True)
    print(f"{marker} {line}", flush=True)


def parse_marked(output: str, marker: str) -> list[dict[str, Any]]:
    """Every ``<marker> <json>`` record in ``output``, in order.

    Anything may precede the marker on its line (a rank tag, a logger
    prefix). A line whose JSON does not parse is skipped rather than raised:
    a torn last line of a killed job must not hide the records before it.
    """
    pattern = re.compile(rf"{re.escape(marker)} (\{{.*\}})\s*$")
    records: list[dict[str, Any]] = []
    for line in output.splitlines():
        found = pattern.search(line)
        if not found:
            continue
        try:
            data = json.loads(found.group(1))
        except ValueError:
            continue
        if isinstance(data, dict):
            records.append(data)
    return records


def prompt_digest(text: str) -> str:
    """A short, stable name for a prompt, for pairing records across runs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------- datasets


def is_lfs_pointer(path: str | Path) -> bool:
    """Whether ``path`` is a git-lfs pointer file rather than the content it names."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(64)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs")


__all__ = [
    "RANK_ENGINE_MARKER",
    "RECORD_MARKER",
    "emit",
    "is_lfs_pointer",
    "parse_marked",
    "prompt_digest",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The operator's inbox: messages for the planner of a running run.

``trtllm-modeling-v2 feedback --workspace W "..."`` leaves one in the
workspace. The run takes what waits once the newest message is
:data:`SETTLE_S` old (or the oldest :data:`SETTLE_MAX_S`), so a burst reaches
the planner as one turn, and hands it, urgent, to that turn ahead of every
other event.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

INBOX_DIRNAME = "inbox"
#: Where a taken message is kept, under the inbox: the record of what was said.
TAKEN_DIRNAME = "taken"
OPERATOR_PREFIX = "message from the operator"
#: How long the newest message waits for another before the run takes them all.
SETTLE_S = 120.0
#: How long the oldest waits at most, however long the burst goes on.
SETTLE_MAX_S = 600.0


def post(workspace: Path, text: str) -> Path:
    """Leave ``text`` for the planner of the run in ``workspace``; returns the message file."""
    if not text.strip():
        raise ValueError("the message is empty")
    inbox = Path(workspace) / INBOX_DIRNAME
    inbox.mkdir(parents=True, exist_ok=True)
    now = time.time_ns()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now / 1e9))
    # Names sort by time and never collide, so no message overwrites another.
    path = inbox / f"{stamp}.{now % 10**9:09d}-{os.getpid()}.md"
    partial = inbox / f".{path.name}.partial"
    with open(partial, "x", encoding="utf-8") as handle:
        handle.write(text.strip() + "\n")
    os.replace(partial, path)
    return path


def take(workspace: Path, now: float | None = None) -> list[str]:
    """Every message waiting, oldest first, as planner events; each is moved to ``taken/``.

    None until the burst has settled (:data:`SETTLE_S`, :data:`SETTLE_MAX_S`).
    Moved before it is returned, so a message is handed over once whatever
    the caller does next.
    """
    inbox = Path(workspace) / INBOX_DIRNAME
    if not inbox.is_dir():
        return []
    waiting = sorted(inbox.glob("*.md"))
    ages = []
    for path in waiting:
        try:
            ages.append((time.time() if now is None else now) - path.stat().st_mtime)
        except OSError:
            continue
    if not ages or (min(ages) < SETTLE_S and max(ages) < SETTLE_MAX_S):
        return []
    taken = inbox / TAKEN_DIRNAME
    events = []
    for path in waiting:
        try:
            text = path.read_text(encoding="utf-8").strip()
            taken.mkdir(exist_ok=True)
            os.replace(path, taken / path.name)
        except OSError:
            # Being written or moved by another reader: the next loop takes it.
            continue
        if text:
            events.append(f"{OPERATOR_PREFIX} ({path.stem}): {text}")
    return events


def is_message(event: str) -> bool:
    """Whether a planner event is a message from the operator."""
    return event.startswith(OPERATOR_PREFIX)


__all__ = [
    "INBOX_DIRNAME",
    "OPERATOR_PREFIX",
    "SETTLE_MAX_S",
    "SETTLE_S",
    "TAKEN_DIRNAME",
    "is_message",
    "post",
    "take",
]

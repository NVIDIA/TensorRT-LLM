# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The operator's inbox: messages for the planner of a running run.

``trtllm-modeling-v2 feedback --workspace W "..."`` leaves one in the
workspace; the run takes it at its next loop and hands it, urgent, to the
planner's next turn, ahead of every other event.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

INBOX_DIRNAME = "inbox"
#: Where a taken message is kept, under the inbox: the record of what was said.
TAKEN_DIRNAME = "taken"
OPERATOR_PREFIX = "message from the operator"


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


def take(workspace: Path) -> list[str]:
    """Every message waiting, oldest first, as planner events; each is moved to ``taken/``.

    Moved before it is returned, so a message is handed over once whatever
    the caller does next.
    """
    inbox = Path(workspace) / INBOX_DIRNAME
    if not inbox.is_dir():
        return []
    taken = inbox / TAKEN_DIRNAME
    events = []
    for path in sorted(inbox.glob("*.md")):
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


__all__ = ["INBOX_DIRNAME", "OPERATOR_PREFIX", "TAKEN_DIRNAME", "is_message", "post", "take"]

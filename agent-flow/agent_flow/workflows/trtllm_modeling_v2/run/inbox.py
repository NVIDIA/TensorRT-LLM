# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator's inbox: messages for the planner of a running run.

``trtllm-modeling-v2 feedback --workspace W "..."`` leaves one in the
workspace. The pipeline takes every waiting message between two tasks and asks
the planner what to do about them (``run/pipeline.py``). A message steers the
work in a few sentences, evidence left in a file it names; one longer than
:data:`MESSAGE_MAX_CHARS` reaches the planner as its head and the path of the
whole.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

INBOX_DIRNAME = "inbox"
#: Where a taken message is kept, under the inbox: the record of what was said.
TAKEN_DIRNAME = "taken"
#: A message is a steer, not a report: past this, the planner reads its head and its path.
MESSAGE_MAX_CHARS = 1200


def post(workspace: Path, text: str) -> Path:
    """Leave ``text`` for the planner of the run in ``workspace``; returns the message file."""
    if not text.strip():
        raise ValueError("the message is empty")
    inbox = Path(workspace) / INBOX_DIRNAME
    inbox.mkdir(parents=True, exist_ok=True)
    clock = time.time_ns()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(clock / 1e9))
    # Names sort by time and never collide, so no message overwrites another.
    path = inbox / f"{stamp}.{clock % 10**9:09d}-{os.getpid()}.md"
    partial = inbox / f".{path.name}.partial"
    with open(partial, "x", encoding="utf-8") as handle:
        handle.write(text.strip() + "\n")
    os.replace(partial, path)
    return path


def take(workspace: Path) -> list[str]:
    """Every message waiting, oldest first; each is moved to ``taken/`` before it is returned."""
    inbox = Path(workspace) / INBOX_DIRNAME
    if not inbox.is_dir():
        return []
    taken = inbox / TAKEN_DIRNAME
    messages = []
    for path in sorted(inbox.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8").strip()
            taken.mkdir(exist_ok=True)
            os.replace(path, taken / path.name)
        except OSError:
            continue  # being written: the next look takes it
        if not text:
            continue
        if len(text) > MESSAGE_MAX_CHARS:
            text = (
                f"{text[:MESSAGE_MAX_CHARS].rstrip()} ... [{len(text)} characters; the whole "
                f"message: {taken / path.name}]"
            )
        messages.append(f"({path.stem}) {text}")
    return messages


__all__ = ["INBOX_DIRNAME", "MESSAGE_MAX_CHARS", "TAKEN_DIRNAME", "post", "take"]

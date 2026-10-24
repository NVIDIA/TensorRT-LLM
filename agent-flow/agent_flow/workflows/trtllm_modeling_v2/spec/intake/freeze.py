# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Confirmation: the user agreed to exactly this ``task.yaml``, and the run checks it still is.

A confirmation is a digest of the request's content, recorded beside it in
``<request>.confirmed`` so the request itself, comments included, is never
rewritten. A ``confirmed`` block inside the request is read too.

A request written by hand runs only once confirmed. A request the fast path
derived (``derive.py``) runs at once: its confirmation is recorded when the
operator gives it (``--confirm``, or ``intake confirm`` later) rather than
awaited, and until then ``<request>.started``, the digest of the content its
run started from, stands in for it. Either way an edit made after the run
started is refused until it is confirmed.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import IntakeError, validate_task

#: The blocks a request may carry about itself, left out of its digest.
STAMP_KEYS: tuple[str, ...] = ("confirmed", "derived")
#: Beside the request: who confirmed which content, and what its run started from.
CONFIRMED_SUFFIX = ".confirmed"
STARTED_SUFFIX = ".started"


def digest(task: Mapping[str, Any]) -> str:
    """sha256 of the task without its stamps (:data:`STAMP_KEYS`), independent of key order."""
    body = {k: v for k, v in task.items() if k not in STAMP_KEYS}
    return hashlib.sha256(yaml.safe_dump(body, sort_keys=True).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def confirmation_path(path: Path) -> Path:
    return Path(path).with_name(Path(path).name + CONFIRMED_SUFFIX)


def started_path(path: Path) -> Path:
    return Path(path).with_name(Path(path).name + STARTED_SUFFIX)


def _read_stamp(path: Path) -> dict[str, Any] | None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return dict(data) if isinstance(data, Mapping) and data.get("sha256") else None


def read_confirmation(path: Path) -> dict[str, Any] | None:
    """The confirmation recorded for the request at ``path``: its sidecar, else its own block."""
    stamp = _read_stamp(confirmation_path(path))
    if stamp is not None:
        return stamp
    try:
        task = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    confirmed = task.get("confirmed") if isinstance(task, Mapping) else None
    return dict(confirmed) if isinstance(confirmed, Mapping) and confirmed.get("sha256") else None


def confirm(path: Path, by: str) -> dict[str, Any]:
    """Record that ``by`` confirmed the task at ``path``; refuse one with open questions.

    The record goes to ``<path>.confirmed``; the request is not rewritten.
    """
    task = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    errors = validate_task(task)
    if isinstance(task, Mapping) and task.get("open_questions"):
        errors.append(f"{len(task['open_questions'])} open question(s) must be answered first")
    if errors:
        raise IntakeError("\n".join(errors))
    stamp = {"by": by, "at": _now(), "sha256": digest(task)}
    confirmation_path(path).write_text(yaml.safe_dump(stamp, sort_keys=False), encoding="utf-8")
    return stamp


def unconfirmed_reason(
    task: Mapping[str, Any], confirmation: Mapping[str, Any] | None
) -> str | None:
    """Why ``task`` may not run, or None when it is confirmed and unchanged since.

    ``confirmation`` is :func:`read_confirmation`'s answer.
    """
    if confirmation is None:
        return "task.yaml is not confirmed: run `trtllm-modeling-v2 intake confirm`"
    if confirmation["sha256"] != digest(task):
        return "task.yaml changed after it was confirmed: review it and confirm again"
    return None


def start_reason(path: Path, *, started: bool) -> str | None:
    """Why the derived request at ``path`` may not start or resume its run, or None."""
    task = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    confirmation = read_confirmation(path)
    if confirmation is not None:
        return unconfirmed_reason(task, confirmation)
    if not started:
        return None
    stamp = _read_stamp(started_path(path))
    if stamp is None or stamp["sha256"] != digest(task):
        return (
            "the request changed after its run started from it: confirm the change "
            f"(`trtllm-modeling-v2 intake confirm {path} --by NAME`, or `--confirm` on this "
            "command), or start over with --clean"
        )
    return None


def stamp_start(path: Path) -> dict[str, Any]:
    """Record the content at ``path`` as what its run starts from, in ``<path>.started``."""
    task = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    stamp = {"at": _now(), "sha256": digest(task)}
    started_path(path).write_text(yaml.safe_dump(stamp, sort_keys=False), encoding="utf-8")
    return stamp


def confirmation_status(task: Mapping[str, Any], confirmation: Mapping[str, Any] | None) -> str:
    """One line on whether ``task`` is confirmed, for the journal and the console."""
    if confirmation is None:
        return "not confirmed; the run does not wait for it"
    if confirmation.get("sha256") != digest(task):
        return "changed after it was confirmed"
    return f"confirmed by {confirmation.get('by', '?')} at {confirmation.get('at', '?')}"


__all__ = [
    "CONFIRMED_SUFFIX",
    "STAMP_KEYS",
    "STARTED_SUFFIX",
    "confirm",
    "confirmation_path",
    "confirmation_status",
    "digest",
    "read_confirmation",
    "stamp_start",
    "start_reason",
    "started_path",
    "unconfirmed_reason",
]

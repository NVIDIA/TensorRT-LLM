# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The receipt: the one file a turn reports with, and the two things it can say.

A turn writes one YAML file at the one path its message names (:func:`decision_path`:
the task and its attempt). Its key ``decision`` says ``ready`` (the work is committed
and the task's checks passed: the branch may land) or ``blocked`` (the attempt ends, and
what the worker wrote is the next attempt's note). The pipeline reads it there and
nowhere else: a turn that wrote no valid receipt has none. The other keys are typed
(:func:`shape_problem`); a wrong type costs one short turn and never the decision.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

DECISIONS_DIRNAME = "decisions"
READY, BLOCKED = DECISIONS = ("ready", "blocked")


def decision_path(workspace: Path, task_id: str, attempt: int, turn: int = 1) -> Path:
    """``decisions/<task>-<attempt>[.<turn>].yaml``: where one turn writes its receipt.

    A task's attempt has one path, so a restarted run finds the answer an earlier
    process's turn left, and no attempt reads another's. ``turn`` numbers the short
    turns after the first within one attempt (a worktree to put right before it lands).
    """
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
    token = f"{attempt}.{turn}" if turn > 1 else str(attempt)
    directory = Path(workspace) / DECISIONS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}-{token}.yaml"


def decision_problem(path: Path) -> str:
    """Why ``path`` holds no receipt, in a sentence; ``""`` when it holds one."""
    allowed = " or ".join(f"`{name}`" for name in DECISIONS)
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"there is no file at `{path}`"
    except (OSError, UnicodeError) as exc:
        return f"`{path}` cannot be read: {exc}"
    except yaml.YAMLError as exc:
        return f"`{path}` is not valid YAML: {clip(str(exc), 300)}"
    if not isinstance(raw, dict):
        return f"`{path}` is not a YAML mapping of keys to values"
    if "decision" not in raw:
        return f"`{path}` has no `decision` key; it must be {allowed}"
    if str(raw["decision"]).strip() not in DECISIONS:
        return f"`decision` is {str(raw['decision']).strip()!r} in `{path}`; it must be {allowed}"
    return ""


def shape_problem(path: Path) -> str:
    """Why a key of the receipt at ``path`` is not of its type, in a sentence; ``""`` when each is.

    ``summary`` is text; ``not_done`` and ``checks`` are lists with a line of text per
    item (``- what: why`` too: YAML reads that item as a mapping). The decision of such
    a receipt stands: its worker gets one turn to write it again, and whatever is found
    there afterwards is printed as text.
    """
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return ""
    for key in ("summary", "not_done", "checks") if isinstance(raw, dict) else ():
        value = raw.get(key)
        if value is None or (key == "summary" and isinstance(value, str)):
            continue
        if key == "summary":
            return f"`summary` in `{path}` must be text; it is {clip(repr(value), 80)}"
        flat = isinstance(value, list) and not any(
            isinstance(item, list)
            or (isinstance(item, dict) and any(isinstance(v, (list, dict)) for v in item.values()))
            for item in value
        )
        if not flat:
            return (
                f"`{key}` in `{path}` must be a list with one line of text per item (`[]` "
                f"when nothing); it is {clip(repr(value), 80)}"
            )
    return ""


def read_decision(path: Path) -> dict[str, Any] | None:
    """The receipt at ``path``; ``None`` when it holds none."""
    if decision_problem(path):
        return None
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {**raw, "decision": str(raw["decision"]).strip()}


def clip(text: str, limit: int) -> str:
    """``text`` up to ``limit`` characters, an ellipsis marking what was cut."""
    text = str(text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def first_paragraph(text: Any, limit: int = 400) -> str:
    """The first paragraph of ``text`` as one line, clipped to ``limit``."""
    for paragraph in re.split(r"\n\s*\n", str(text or "").strip()):
        line = " ".join(paragraph.split())
        if line:
            return clip(line, limit)
    return ""

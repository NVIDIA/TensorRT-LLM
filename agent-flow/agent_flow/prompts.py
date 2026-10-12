# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt-bundle helpers shared by workflows."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any


def dump_prompt_bundle(bundle: Any, directory: str | Path) -> None:
    """Write each role's composed system prompt to ``directory/<role>.md``.

    ``bundle`` is a dataclass instance with one ``str`` field per role — the
    shape of every workflow's ``PromptBundle`` — so the roles are read off
    its fields and any workflow's bundle is dumped by the same code. Pass
    the bundle the workflow builds its agents from, after every task-specific
    extension: that composition otherwise exists only in the launching
    process's memory.

    Verbatim, so a snapshot can be diffed against the prompt modules or
    pasted into a session. The directory is rewritten per call, and ``*.md``
    left by an older one (a role since renamed) is dropped, so it never shows
    two versions' prompts as one bundle's. Other files are left alone.

    Anything but a dataclass instance raises ``TypeError`` before the
    directory is touched.
    """
    if not dataclasses.is_dataclass(bundle) or isinstance(bundle, type):
        raise TypeError(f"expected a dataclass instance, got {type(bundle).__name__}")
    roles = [field.name for field in dataclasses.fields(bundle)]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.md"):
        if stale.stem not in roles:
            stale.unlink()
    for role in roles:
        (directory / f"{role}.md").write_text(getattr(bundle, role), encoding="utf-8")

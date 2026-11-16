# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What counts as a place in the engine in a text a worker reads, and what stands for it.

The engine's layout moves, so nothing a worker is told names a place in it: a prompt, a
skill and a card carry behaviour, symbols and searches to run. :func:`engine_paths` finds
what breaks that in a text; :func:`as_search` turns a place somebody wrote into a row (a
surveyor's lead, say) into the name to search for. The few forms that are kept are listed
here, in one place, each with why.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

#: The top-level directories of the engine's repository, and the abbreviation prompts used
#: for its PyTorch package: a slash path that starts with one is a place in the engine.
ENGINE_ROOTS: tuple[str, ...] = (
    "tensorrt_llm",
    "tests",
    "cpp",
    "examples",
    "docs",
    "triton_backend",
    "triton_kernels",
    "_torch",
)
#: The roots that are also plain words. In prose somebody else wrote, one of them and one
#: more word with a slash between (a rate "per sec", a pair of nouns) is a ratio or a
#: pair, not a place: :func:`as_search` leaves it alone unless more of a path follows.
WORD_ROOTS: frozenset[str] = frozenset({"tests", "examples", "docs"})
#: Suffixes of the engine's source files: one named with its directory is a place too.
SOURCE_SUFFIXES: tuple[str, ...] = (
    "py", "h", "hpp", "cpp", "cc", "cu", "cuh", "cmake", "yml", "yaml", "ini", "png", "so",
)  # fmt: skip

#: Kept, and why. An entry is named by its place **inside** the ModelingV2 package, the
#: workflow's subject: ``catalog/<category>/<name>.py`` for a shared one,
#: ``ops/<category>/<name>.py`` for a target's own. That is the vocabulary's key for an
#: entry and the form a target imports it by, not a place a worker is sent to read.
KEPT_FIRST_COMPONENTS: tuple[str, ...] = ("catalog", "ops")
#: Kept, and why: a run's own files. A workspace's directories (decisions, tasks, rows,
#: the options files a spec names) and placeholders for them are the run's, not the engine's.
RUN_FIRST_COMPONENTS: tuple[str, ...] = (
    "<workspace>", "<ws>", "<spool>", "<scripts>", "<your tree>", "<your worktree>",
    "decisions", "tasks", "rows", "engines", "prompts", "turns",
)  # fmt: skip

#: One component of a path: letters, digits, ``_ . - *``, and placeholders in angle
#: brackets, which may hold a space (``<your tree>``).
_PART = r"(?:<[^<>\n]{1,40}>|[\w.*-])+"
#: What a place written relative to something starts with: ``./``, ``<repo>/``, ``$REPO/``.
_PREFIX = r"(?:\./|<[^<>\n]{1,40}>/|\$\{?\w+\}?/)"
#: Not in the middle of a word or of a longer path (so an absolute path, which a run
#: resolved, is never taken for a template's).
_START = r"(?<![\w./<>$}-])"
_ROOTS = "|".join(ENGINE_ROOTS)
_ROOTED = re.compile(rf"{_START}{_PREFIX}?(?:{_ROOTS})/(?:{_PART}/)*(?:{_PART})?")
_SOURCE = re.compile(rf"{_START}(?:{_PART}/)+{_PART}\.(?:{'|'.join(SOURCE_SUFFIXES)})\b")
#: What follows a place in a sentence and is not part of it.
_TRAILING = ".,;:)"


def _spans(text: str) -> list[tuple[int, int]]:
    """Where ``text`` holds something shaped like a place, as spans that never overlap."""
    found = sorted(m.span() for pattern in (_ROOTED, _SOURCE) for m in pattern.finditer(text))
    merged: list[tuple[int, int]] = []
    for start, end in found:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    out = []
    for start, end in merged:
        # Bold markup around a place (`**path**`) is not part of it; one `*` is a pattern's.
        if text.startswith("**", start):
            start += len(text[start:end]) - len(text[start:end].lstrip("*"))
        while end > start and text[end - 1] in _TRAILING:
            end -= 1
        if text.endswith("**", start, end):
            end -= len(text[start:end]) - len(text[start:end].rstrip("*"))
        while end > start and text[end - 1] in _TRAILING:
            end -= 1
        if end > start:
            out.append((start, end))
    return out


def _first(token: str) -> str:
    """The first component of ``token``: a placeholder whole, else up to the first slash."""
    match = re.match(_PART, token)
    return match.group(0) if match else token.split("/", 1)[0]


def _is_place(token: str, allowed: Sequence[str]) -> bool:
    first = _first(token)
    if first in KEPT_FIRST_COMPONENTS or first in RUN_FIRST_COMPONENTS:
        # A run's placeholder in front of an engine directory is a place in the engine.
        rest = token[len(first) :].lstrip("/")
        return rest.split("/", 1)[0] in ENGINE_ROOTS
    return not any(token == a or token.startswith(f"{a}/") for a in allowed)


def engine_paths(text: str, allowed: Iterable[str] = ()) -> list[str]:
    """Every place in the engine ``text`` names, in order; empty when it names none.

    A slash path under one of the engine's top directories (also behind ``./``, a
    placeholder or a shell variable), or a source file named with its directory. Not
    counted: an entry's name inside the subject package (:data:`KEPT_FIRST_COMPONENTS`),
    a run's own files (:data:`RUN_FIRST_COMPONENTS`), an absolute path (resolved for
    this run, never a template's), and whatever starts with one of ``allowed`` (a card's
    own write scope: what its task creates).
    """
    allowed = tuple(a.rstrip("/") for a in allowed if a)
    tokens = (text[start:end] for start, end in _spans(text))
    return [token for token in tokens if _is_place(token, allowed)]


def _prose(token: str) -> bool:
    """Whether ``token`` is two plain words with a slash between them rather than a place."""
    parts = token.split("/")
    if len(parts) != 2 or not parts[1] or parts[0] not in WORD_ROOTS:
        return False
    return "." not in parts[1]


def as_search(text: str, allowed: Sequence[str] = ()) -> str:
    """``text`` with each place in the engine replaced by the name to search for.

    The name is the last component without its suffix (``.../sparse/indexer.py`` becomes
    ``indexer``): what ``git grep`` or ``git ls-files`` finds wherever the file is now. A
    pattern keeps its last component (``models/*.py`` becomes ``*.py``), and two plain
    words with a slash between them are left as they are.
    """
    out, at = [], 0
    allowed = tuple(a.rstrip("/") for a in allowed if a)
    for start, end in _spans(text):
        token = text[start:end]
        if not _is_place(token, allowed) or _prose(token):
            continue
        leaf = token.rstrip("/").rsplit("/", 1)[-1]
        stem = leaf.rsplit(".", 1)[0] if "." in leaf.strip(".") else leaf
        out.append(text[at:start])
        out.append(leaf if "*" in stem or not stem else stem)
        at = end
    out.append(text[at:])
    return "".join(out)


__all__ = [
    "ENGINE_ROOTS",
    "KEPT_FIRST_COMPONENTS",
    "RUN_FIRST_COMPONENTS",
    "SOURCE_SUFFIXES",
    "WORD_ROOTS",
    "as_search",
    "engine_paths",
]

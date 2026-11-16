# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the workflow reads off the engine, found in a checkout by what it defines.

The engine's layout moves; what a file defines moves with it. Each :class:`Anchor`
names one thing the workflow reads or sends an agent to (the attention backends, the
KV-cache dtypes the loader accepts, the accuracy tests ...) and its **marker**: a
definition a file of it holds (a class, a registration, a message, a row). It holds no
path, no directory and no file name of the engine: :func:`resolve` finds the files that
hold the marker wherever they are in the checkout, and an anchor no file holds is
reported missing, never replaced by a place remembered here.

Code reads its anchors through :func:`resolve`. Preflight writes :func:`render_map`
into ``tree_map.md`` in the workspace at each start: what each definition found, at
which commit, and the search that finds it again. That is evidence about one commit,
not a template: prompts, cards and skills name things by behaviour and symbol only.
ModelingV2's own package is the workflow's subject, not an anchor
(:data:`~.coverage.SUBJECT_PATHS`).
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Sequence

from .coverage import CATALOG_TESTS_DIR, MODELING_V2_PACKAGE

#: Files larger than this are not read while searching: generated kernel tables, not code.
MAX_SEARCH_BYTES = 2 * 1024 * 1024
#: Directories a search never enters.
_SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".venv", "venv"})
#: The file names pytest collects: what makes a Python file a test, wherever it is.
_TEST_NAMES = ("test_*.py", "*_test.py", "conftest.py")
#: This workflow's own package, as a path (``<distribution>/workflows/<workflow>/``): a
#: checkout's copy of it is never searched. The whole path, not the distribution's name
#: alone: a directory of the engine may be called the same.
_OWN_PACKAGE = "/".join(__name__.split(".")[:-2]) + "/"


@dataclass(frozen=True)
class Anchor:
    """One thing the workflow reads or points at, and the definition that finds it."""

    key: str
    #: What it is, as ``tree_map.md`` and the lint say it.
    concept: str
    #: A regex (multi-line) a file of it matches: the definition itself.
    marker: str
    #: Fixed strings, one of which every such file holds: the cheap first pass.
    literals: tuple[str, ...]
    #: Which file names a search reads (``fnmatch``).
    name: str = "*.py"
    #: ``False``: never a file pytest collects; ``True``: only those; ``None``: either.
    tests: bool | None = False
    #: Every file that holds the marker; ``False``: the one that defines it (first by path).
    many: bool = True
    #: How a person or an agent finds it by hand, for ``tree_map.md``.
    search: str = ""

    def matches(self, text: str) -> re.Match[str] | None:
        return re.search(self.marker, text, re.M)

    def reads(self, path: str) -> bool:
        """Whether a search reads the file at ``path``: its name, and test or not."""
        base = path.rsplit("/", 1)[-1]
        if not fnmatch.fnmatch(base, self.name):
            return False
        if self.tests is None:
            return True
        return self.tests == any(fnmatch.fnmatch(base, pattern) for pattern in _TEST_NAMES)


@dataclass(frozen=True)
class Resolution:
    """What an anchor's definition found in one checkout."""

    anchor: Anchor
    #: The files that hold the marker, repository-relative, sorted; empty when missing.
    paths: tuple[str, ...]
    #: ``definition`` or ``missing``.
    via: str
    #: The marker's line in the first path.
    line: int | None = None
    #: The commit the checkout was at, when it is a git checkout.
    commit: str = ""

    @property
    def found(self) -> bool:
        return bool(self.paths)

    @property
    def where(self) -> str:
        """The first file with the marker's line: ``a/b.py:78``."""
        if not self.paths:
            return ""
        return f"{self.paths[0]}:{self.line}" if self.line else self.paths[0]

    @property
    def directories(self) -> tuple[tuple[str, int], ...]:
        """The directories that hold the files, fullest first, each with its count."""
        counts = Counter(_directory(p) for p in self.paths)
        return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    @property
    def homes(self) -> tuple[str, ...]:
        """The directories the definition lives in: where its files are concentrated.

        Those that hold at least half as many of the files as the fullest one does, a
        directory inside another of them folded into it. A file that defines the thing
        somewhere else (one model registering an op of its own) widens nothing. What
        stands beside the definitions in a home (a backend's helpers and kernels) is
        part of the thing, so a reader of the whole thing reads the homes. The top of the
        repository is never a home: a file there is taken by itself, not with the tree.
        """
        counted = self.directories
        if not counted:
            return ()
        fullest = counted[0][1]
        return _outermost(d for d, n in counted if d and 2 * n >= fullest)


def _directory(path: str) -> str:
    """The directory of ``path``, repository-relative; ``""`` for a file at the top."""
    parent = str(PurePosixPath(path).parent)
    return "" if parent == "." else parent


def _outermost(paths: Iterable[str]) -> tuple[str, ...]:
    """The paths that are not inside another of them, sorted."""
    kept: list[str] = []
    for path in sorted(set(paths)):
        if not any(path == k or path.startswith(k + "/") for k in kept):
            kept.append(path)
    return tuple(kept)


def _anchors(*anchors: Anchor) -> dict[str, Anchor]:
    return {a.key: a for a in anchors}


#: Every anchor, by key. A new thing the workflow reads off the tree adds one here,
#: and the drift test resolves it against the checkout CI runs in.
ANCHORS: dict[str, Anchor] = _anchors(
    Anchor(
        "modeling_v2_guide",
        "the ModelingV2 developer guide",
        r"^# Modeling V2\s*$",
        ("# Modeling V2",),
        name="*.md",
        tests=None,
        many=False,
        search="the Markdown file whose title is `# Modeling V2`",
    ),
    Anchor(
        "attention_backends",
        "attention backends",
        r"^class \w+\((?:\w+\.)*AttentionBackend\b",
        ("AttentionBackend",),
        search="classes that subclass `AttentionBackend`",
    ),
    Anchor(
        "custom_ops",
        "custom ops",
        r"torch\.library\.custom_op\(|@torch\.library\.register_fake\b",
        ("torch.library.custom_op(", "torch.library.register_fake"),
        search="files that register ops with `torch.library.custom_op`",
    ),
    Anchor(
        "kernels",
        "Python kernels (Triton, CuTe DSL)",
        r"@triton\.jit\b|@cute\.(?:experimental\.)?(?:kernel|jit)\b",
        ("@triton.jit", "@cute."),
        search="files that define a kernel in Python: a function under `@triton.jit`, "
        "`@cute.kernel` or `@cute.jit`",
    ),
    Anchor(
        "moe_modules",
        "MoE modules",
        r"^class \w+\((?:\w+\.)*(?:MoE|MoEImplBase)\b",
        ("MoE",),
        search="classes that subclass `MoE` or `MoEImplBase`",
    ),
    Anchor(
        "builtin_models",
        "built-in (stock) models",
        r"register_auto_model\(",
        ("register_auto_model(",),
        search="modules that call `register_auto_model`",
    ),
    Anchor(
        "kv_cache_dtypes",
        "the KV-cache dtypes the PyTorch loader accepts",
        r"_VALID_KV_CACHE_DTYPES\s*(?::[^=\n]*)?=",
        ("_VALID_KV_CACHE_DTYPES",),
        many=False,
        search="the assignment of `_VALID_KV_CACHE_DTYPES`",
    ),
    Anchor(
        "fmha_head_dims",
        "the head dimensions the trtllm-gen FMHA has kernels for",
        r"HeadDimV must be ([0-9, ]+?),? or a multiple of (\d+)",
        ("HeadDimV must be",),
        name="*.h",
        tests=None,
        many=False,
        search='the check whose message says "HeadDimV must be"',
    ),
    Anchor(
        "unit_tests",
        "the unit tests' pytest root (one op, cache behaviour, transfer or API path)",
        r"^\s*cpu_only\s*:",
        ("cpu_only",),
        name="pytest.ini",
        tests=None,
        search="the `pytest.ini` that registers the `cpu_only` marker; the unit tests "
        "are the `test_*.py` files under its directory",
    ),
    Anchor(
        "accuracy_tests",
        "accuracy tests (configurations that serve and score end to end)",
        r"^class \w+\((?:\w+\.)*LlmapiAccuracyTestHarness\)",
        ("LlmapiAccuracyTestHarness",),
        tests=True,
        search="test classes that subclass `LlmapiAccuracyTestHarness`",
    ),
    Anchor(
        "accuracy_references",
        "accuracy reference scores",
        r"^\s+(?:-\s+)?accuracy:\s*[0-9.]+\s*(?:#.*)?$",
        ("accuracy:",),
        name="*.yaml",
        tests=None,
        search="YAML files whose rows state `accuracy: <score>` under a model's name",
    ),
    Anchor(
        "test_lists",
        "CI test lists (the model and feature combinations CI runs)",
        r"^- condition:\s*$",
        ("- condition:",),
        name="*.yml",
        tests=None,
        search="YAML files whose items are `- condition:` blocks with a `tests:` list",
    ),
)


def _git(repo: Path, *args: str) -> bytes | None:
    """``git`` in ``repo``, its output; ``None`` when ``repo`` is no checkout or git fails."""
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode in (0, 1) else None


def _commit(repo: Path) -> str:
    """The commit ``repo`` is at, abbreviated; ``""`` for a tree git does not know."""
    out = _git(repo, "rev-parse", "--short=12", "HEAD")
    return out.decode("ascii", "replace").strip() if out else ""


def _is_checkout(repo: Path) -> bool:
    """Whether ``repo`` is the top of a git checkout: the only tree ``git grep`` answers for."""
    out = _git(repo, "rev-parse", "--show-toplevel")
    if not out:
        return False
    try:
        return Path(out.decode("utf-8", "replace").strip()).resolve() == repo.resolve()
    except OSError:
        return False


def _walk(repo: Path) -> Iterator[str]:
    """Every file under ``repo``, repository-relative; hidden and build directories are passed over."""
    for here, dirs, files in os.walk(repo):
        dirs[:] = [
            d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".") and d != "build"
        ]
        base = Path(here).relative_to(repo).as_posix()
        for name in files:
            yield name if base == "." else f"{base}/{name}"


def _candidates(repo: Path, anchors: Sequence[Anchor]) -> list[str]:
    """The files that may hold one of ``anchors``' markers, sorted: a fixed-string pass over the tree.

    A git checkout answers through ``git grep`` (tracked files that are present, in well
    under a second); any other tree, and a checkout where that finds nothing at all, is
    walked. Either way every candidate is read again and held to the marker itself, and
    the order is the same: by path.
    """
    if _is_checkout(repo):
        literals = [arg for a in anchors for lit in a.literals for arg in ("-e", lit)]
        # A pathspec with no wildcard names one file at the top: give it every depth too.
        names = sorted({spec for a in anchors for spec in (a.name, f"*/{a.name}")})
        out = _git(repo, "grep", "-l", "-I", "-z", "-F", *literals, "--", *names)
        found = sorted(p for p in (out or b"").decode("utf-8", "replace").split("\0") if p)
        if found:
            return found
    return sorted(_walk(repo))


def _own_copies(paths: Iterable[str]) -> tuple[str, ...]:
    """The directories of a checkout that hold a copy of this workflow's own distribution.

    Its prompts and tests quote the very definitions the anchors look for, so a search
    passes over them: what is found there says nothing about the engine. A copy is known
    by this package's whole path inside it (:data:`_OWN_PACKAGE`); what is passed over is
    the directory that holds the distribution (its tests stand beside the package), or
    the distribution's own directory when it is at the top of the checkout.
    """
    found = set()
    for path in paths:
        at = path.find(_OWN_PACKAGE)
        while at >= 0:
            if at == 0:
                found.add(_OWN_PACKAGE.split("/", 1)[0] + "/")
            elif path[at - 1] == "/":
                found.add(path[:at])
            at = path.find(_OWN_PACKAGE, at + 1)
    return tuple(sorted(found))


def _read(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_SEARCH_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _line(text: str, match: re.Match[str]) -> int:
    return text[: match.start()].count("\n") + 1


def resolve_all(
    repo: str | Path | None, anchors: Iterable[Anchor | str] | None = None
) -> dict[str, Resolution]:
    """Where each of ``anchors`` (every one by default) is in the checkout at ``repo``.

    One pass: the files that may hold a marker are listed once and each is read once. A
    file is one of an anchor's when its name is one the anchor reads, it is a test or
    not as the anchor says, and its text holds the marker. Nothing found is
    ``via == "missing"``.
    """
    chosen = ANCHORS.values() if anchors is None else anchors
    wanted = [ANCHORS[a] if isinstance(a, str) else a for a in chosen]
    if repo is None:
        return {a.key: Resolution(a, (), "missing") for a in wanted}
    repo = Path(repo)
    commit = _commit(repo)
    candidates = _candidates(repo, wanted)
    own = _own_copies(candidates)
    found: dict[str, list[tuple[str, int]]] = {a.key: [] for a in wanted}
    for path in candidates:
        if any(path.startswith(root) for root in own):
            continue
        readers = [a for a in wanted if a.reads(path) and (a.many or not found[a.key])]
        if not readers:
            continue
        text = _read(repo / path)
        if not text:
            continue
        for anchor in readers:
            if not any(literal in text for literal in anchor.literals):
                continue
            match = anchor.matches(text)
            if match:
                found[anchor.key].append((path, _line(text, match)))
    out = {}
    for anchor in wanted:
        hits = found[anchor.key]
        if not hits:
            out[anchor.key] = Resolution(anchor, (), "missing", commit=commit)
        else:
            out[anchor.key] = Resolution(
                anchor, tuple(p for p, _ in hits), "definition", hits[0][1], commit
            )
    return out


def resolve(repo: str | Path | None, anchor: Anchor | str) -> Resolution:
    """Where ``anchor`` is in the checkout at ``repo``: the files that hold its definition."""
    anchor = ANCHORS[anchor] if isinstance(anchor, str) else anchor
    return resolve_all(repo, [anchor])[anchor.key]


def read(repo: str | Path | None, anchor: Anchor | str) -> tuple[str, re.Match[str], str] | None:
    """An anchor's first file: its text, its marker's match and ``path:line``; ``None`` when missing."""
    found = resolve(repo, anchor)
    if not found.found or repo is None:
        return None
    text = _read(Path(repo) / found.paths[0])
    match = found.anchor.matches(text)
    return (text, match, found.where) if match else None


def _home(paths: Sequence[str]) -> str:
    """The deepest directory that holds nearly all of ``paths`` (nine in ten): strays do not widen it."""
    need = len(paths) - len(paths) // 10
    home = ""
    while True:
        below = Counter(
            p[len(home) :].split("/", 1)[0]
            for p in paths
            if p.startswith(home) and "/" in p[len(home) :]
        )
        child, count = max(below.items(), key=lambda item: (item[1], item[0]), default=("", 0))
        if not child or count < need:
            return home.rstrip("/")
        home = f"{home}{child}/"


def _where(found: Resolution) -> str:
    """What a definition found, for the map: the file, or how many and under what."""
    if len(found.paths) == 1:
        return f"`{found.where}`"
    home = _home(found.paths)
    return f"{len(found.paths)} files, " + (
        f"nearly all under `{home}`" if home else "across the tree"
    )


def render_map(repo: str | Path | None, found: Sequence[Resolution] | None = None) -> str:
    """``tree_map.md``: what each anchor's definition found in the checkout at ``repo``.

    ``found`` is every anchor already resolved there; by default they are resolved here.
    Each line is evidence about one commit: the definition, what it found, and the
    search to run again when the tree has moved since.
    """
    found = list(found) if found is not None else list(resolve_all(repo).values())
    commit = next((one.commit for one in found if one.commit), "")
    at = f"at commit `{commit}`" if commit else "as it was"
    lines = [
        "# What this run found in the checkout, by definition",
        "",
        f"Read from `{repo}` {at} when this run started. Nothing here is a fixed "
        "place: each line is one definition and what it found then. When a file is not "
        "where a line says, the tree has moved: run the line's search again (`git grep`) "
        "and trust the tree. Missing means no file of this checkout holds the definition.",
        "",
        f"The run's subject is the ModelingV2 package `{MODELING_V2_PACKAGE}`, with its "
        f"tests under `{CATALOG_TESTS_DIR}`: where this run writes its target.",
        "",
    ]
    listed = []
    for one in found:
        anchor = one.anchor
        if one.found:
            listed.append(f"- **{anchor.concept}**: {_where(one)}. Found by: {anchor.search}.")
        else:
            listed.append(f"- **{anchor.concept}**: missing. Searched for: {anchor.search}.")
    return "\n".join([*lines, *listed]) + "\n"


__all__ = [
    "ANCHORS",
    "MAX_SEARCH_BYTES",
    "Anchor",
    "Resolution",
    "read",
    "render_map",
    "resolve",
    "resolve_all",
]

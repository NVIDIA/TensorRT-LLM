# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where this checkout keeps what the workflow reads off the engine, found by what it defines.

The engine's layout moves; what a file defines moves with it. Each
:class:`Anchor` names one thing the workflow reads or sends an agent to (the
attention backends, the KV-cache dtypes the loader accepts, the accuracy
tests ...), the path it was last seen at, and what identifies it wherever it
went: a definition a file of it holds, or a directory or file name.
:func:`resolve` looks at the path first and searches the tree only when the
path is gone; an anchor neither finds is reported missing, never replaced by
a value remembered here.

Code reads its anchors through :func:`resolve`. Prompts and the feature skill
name things by their symbols, and preflight writes :func:`render_map` into
``tree_map.md`` in the workspace: where each anchor is in the checkout the run
measures. ModelingV2's own package is the workflow's subject, not an anchor
(:data:`~.coverage.MODELING_V2_PACKAGE`).
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .coverage import CATALOG_TESTS_DIR, MODELING_V2_PACKAGE

DIR = "dir"
FILE = "file"
#: Files larger than this are not read while searching: generated kernel tables, not code.
MAX_SEARCH_BYTES = 2 * 1024 * 1024
#: Directories a search never enters.
_SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".venv", "venv"})


@dataclass(frozen=True)
class Anchor:
    """One thing the workflow reads or points at, and how to find it after it moves."""

    key: str
    #: What it is, as ``tree_map.md`` and the lint say it.
    concept: str
    #: Repository-relative path it was last seen at; looked at first.
    hint: str
    #: ``dir``: the directories holding it; ``file``: the files that define it.
    kind: str
    #: Top-level directories a search walks when the hint is gone.
    roots: tuple[str, ...]
    #: A regex (multi-line) a file of it matches. Empty: the name alone identifies it.
    marker: str = ""
    #: Which file names a search reads (``fnmatch``).
    name: str = "*.py"
    #: For a ``dir`` anchor found by name: the directory's base name.
    dirname: str = ""
    #: For a ``file`` anchor: every match, not only the first.
    many: bool = False
    #: How a person or an agent finds it by hand, for ``tree_map.md``.
    search: str = ""
    #: What a run loses when it is missing, for the lint.
    consequence: str = ""

    def matches(self, text: str) -> re.Match[str] | None:
        return re.search(self.marker, text, re.M) if self.marker else None


@dataclass(frozen=True)
class Resolution:
    """Where an anchor is in one checkout."""

    anchor: Anchor
    #: Repository-relative paths, empty when missing.
    paths: tuple[str, ...]
    #: ``hint``, ``search`` or ``missing``.
    via: str
    #: The marker's line in the first path, for a ``file`` anchor.
    line: int | None = None

    @property
    def found(self) -> bool:
        return bool(self.paths)

    @property
    def where(self) -> str:
        """The first path, with the marker's line when there is one: ``a/b.py:78``."""
        if not self.paths:
            return ""
        return f"{self.paths[0]}:{self.line}" if self.line else self.paths[0]


def _anchors(*anchors: Anchor) -> dict[str, Anchor]:
    return {a.key: a for a in anchors}


#: Every anchor, by key. A new thing the workflow reads off the tree adds one here,
#: and the drift test resolves it against the checkout CI runs in.
ANCHORS: dict[str, Anchor] = _anchors(
    Anchor(
        "attention_backends",
        "attention backends",
        "tensorrt_llm/_torch/attention/backends",
        DIR,
        ("tensorrt_llm",),
        marker=r"^class \w+\((?:\w+\.)*AttentionBackend\b",
        search="classes that subclass `AttentionBackend`",
        consequence="the prior-art scan skips the attention backends",
    ),
    Anchor(
        "custom_ops",
        "custom ops",
        "tensorrt_llm/_torch/custom_ops",
        DIR,
        ("tensorrt_llm",),
        marker=r"torch\.library\.custom_op\(|@torch\.library\.register_fake\b",
        search="files that register ops with `torch.library.custom_op`",
        consequence="the prior-art scan skips the custom ops",
    ),
    Anchor(
        "kernels",
        "Python kernels (Triton, CuTe DSL)",
        "tensorrt_llm/_torch/kernels",
        DIR,
        ("tensorrt_llm",),
        dirname="kernels",
        search="directories named `kernels` under `tensorrt_llm/`",
        consequence="the prior-art scan skips the Python kernels",
    ),
    Anchor(
        "moe_modules",
        "MoE modules",
        "tensorrt_llm/_torch/moe",
        DIR,
        ("tensorrt_llm",),
        marker=r"^class \w+\((?:\w+\.)*(?:MoE|MoEImplBase)\b",
        search="classes that subclass `MoE` or `MoEImplBase`",
        consequence="the prior-art scan skips the MoE modules",
    ),
    Anchor(
        "builtin_models",
        "built-in (stock) models",
        "tensorrt_llm/_torch/models",
        DIR,
        ("tensorrt_llm",),
        marker=r"register_auto_model\(",
        search="modules that call `register_auto_model`",
        consequence="the lint cannot tell whether the stock path boots this architecture",
    ),
    Anchor(
        "builtin_model_index",
        "the lazy index of built-in architectures",
        "tensorrt_llm/_torch/models/_arch_index.py",
        FILE,
        ("tensorrt_llm",),
        marker=r"^MODEL_ARCH_TO_MODULE\s*(?::[^=\n]*)?=",
        search="`MODEL_ARCH_TO_MODULE`",
        consequence="architectures only the lazy index registers are not seen",
    ),
    Anchor(
        "kv_cache_dtypes",
        "the KV-cache dtypes the PyTorch loader accepts",
        "tensorrt_llm/_torch/pyexecutor/model_loader.py",
        FILE,
        ("tensorrt_llm",),
        marker=r"_VALID_KV_CACHE_DTYPES\s*(?::[^=\n]*)?=",
        search="`_VALID_KV_CACHE_DTYPES`",
        consequence="a KV-cache dtype the engine does not load is not caught before launch",
    ),
    Anchor(
        "fmha_head_dims",
        "the head dimensions the trtllm-gen FMHA has kernels for",
        "cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h",
        FILE,
        ("cpp",),
        marker=r"HeadDimV must be ([0-9, ]+?),? or a multiple of (\d+)",
        name="*.h",
        search='the check that says "HeadDimV must be"',
        consequence="a head dimension the FMHA has no kernel for is not named before the survey",
    ),
    Anchor(
        "unit_tests",
        "unit tests (one op, cache behaviour, transfer or API path)",
        "tests/unittest",
        DIR,
        ("tests",),
        dirname="unittest",
        name="test_*.py",
        search="the `unittest` directory under `tests/`",
    ),
    Anchor(
        "modeling_v2_tests",
        "ModelingV2's gates, catalog and target tests",
        CATALOG_TESTS_DIR,
        DIR,
        ("tests",),
        name="test_modeling_v2_*.py",
        search="files named `test_modeling_v2_*.py`",
    ),
    Anchor(
        "accuracy_tests",
        "accuracy tests (configurations that serve and score end to end)",
        "tests/integration/defs/accuracy",
        DIR,
        ("tests",),
        marker=r"^class \w+\((?:\w+\.)*LlmapiAccuracyTestHarness\)",
        search="classes that subclass `LlmapiAccuracyTestHarness`",
    ),
    Anchor(
        "accuracy_references",
        "accuracy reference scores",
        "tests/integration/defs/accuracy/references",
        DIR,
        ("tests",),
        name="gsm8k.yaml",
        search="the directory holding `gsm8k.yaml`",
    ),
    Anchor(
        "disagg_launcher",
        "the accuracy tests' disaggregated-serving launcher",
        "tests/integration/defs/accuracy/test_disaggregated_serving.py",
        FILE,
        ("tests",),
        marker=r"^def launch_disaggregated_llm\(",
        search="`launch_disaggregated_llm`",
    ),
    Anchor(
        "perf_tests",
        "perf tests (how serving is measured)",
        "tests/integration/defs/perf",
        DIR,
        ("tests",),
        marker=r"^class PerfTestConfig\b",
        search="`class PerfTestConfig`",
    ),
    Anchor(
        "test_lists",
        "CI test lists (the model and feature combinations CI runs)",
        "tests/integration/test_lists/test-db",
        DIR,
        ("tests",),
        name="l0_*.yml",
        search="files named `l0_*.yml`",
    ),
)


def _walk(repo: Path, roots: Sequence[str]) -> Iterator[tuple[Path, list[str], list[str]]]:
    for root in roots:
        top = repo / root
        if not top.is_dir():
            continue
        for here, dirs, files in os.walk(top):
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("build"))
            yield Path(here), dirs, sorted(files)


def _read(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_SEARCH_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _outermost(paths: Iterable[str]) -> tuple[str, ...]:
    """The paths that are not inside another of them, sorted."""
    kept: list[str] = []
    for path in sorted(set(paths)):
        if not any(path == k or path.startswith(k + "/") for k in kept):
            kept.append(path)
    return tuple(kept)


def _line(text: str, match: re.Match[str]) -> int:
    return text[: match.start()].count("\n") + 1


def _search_dirs(repo: Path, anchor: Anchor) -> tuple[str, ...]:
    found: list[str] = []
    if anchor.dirname:
        # Found by name: a directory of that name holding such files at any depth.
        for here, dirs, _ in _walk(repo, anchor.roots):
            for d in dirs:
                if d == anchor.dirname and next((here / d).rglob(anchor.name), None):
                    found.append((here / d).relative_to(repo).as_posix())
        return _outermost(found)
    for here, _, files in _walk(repo, anchor.roots):
        named = [f for f in files if fnmatch.fnmatch(f, anchor.name)]
        if named and (not anchor.marker or any(anchor.matches(_read(here / f)) for f in named)):
            found.append(here.relative_to(repo).as_posix())
    return _outermost(found)


def _search_files(repo: Path, anchor: Anchor) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for here, _, files in _walk(repo, anchor.roots):
        for f in files:
            if not fnmatch.fnmatch(f, anchor.name):
                continue
            text = _read(here / f)
            match = anchor.matches(text)
            if match:
                found.append(((here / f).relative_to(repo).as_posix(), _line(text, match)))
                if not anchor.many:
                    return found
    return found


def resolve(repo: str | Path | None, anchor: Anchor | str) -> Resolution:
    """Where ``anchor`` is in the checkout at ``repo``: at its hint, else wherever a search finds it.

    A ``dir`` hint counts while the directory exists. A ``file`` hint counts
    only when the file still holds the marker, since the caller reads that
    definition. A search walks ``roots`` (skipping files over
    :data:`MAX_SEARCH_BYTES`). Nothing found is ``via == "missing"``.
    """
    anchor = ANCHORS[anchor] if isinstance(anchor, str) else anchor
    if repo is None:
        return Resolution(anchor, (), "missing")
    repo = Path(repo)
    if anchor.kind == DIR:
        if (repo / anchor.hint).is_dir():
            return Resolution(anchor, (anchor.hint,), "hint")
        found = _search_dirs(repo, anchor)
        return Resolution(anchor, found, "search" if found else "missing")
    hint = repo / anchor.hint
    text = _read(hint) if hint.is_file() else ""
    match = anchor.matches(text) if text else None
    if match:
        return Resolution(anchor, (anchor.hint,), "hint", _line(text, match))
    found = _search_files(repo, anchor)
    if not found:
        return Resolution(anchor, (), "missing")
    return Resolution(anchor, tuple(p for p, _ in found), "search", found[0][1])


def read(repo: str | Path | None, anchor: Anchor | str) -> tuple[str, re.Match[str], str] | None:
    """A ``file`` anchor's text, its marker's match and ``path:line``; ``None`` when missing."""
    found = resolve(repo, anchor)
    if not found.found or repo is None:
        return None
    text = _read(Path(repo) / found.paths[0])
    match = found.anchor.matches(text)
    return (text, match, found.where) if match else None


def missing(repo: str | Path | None, keys: Iterable[str]) -> list[Resolution]:
    """The anchors among ``keys`` this checkout has nowhere."""
    return [r for r in (resolve(repo, k) for k in keys) if not r.found]


def render_map(repo: str | Path | None, found: Sequence[Resolution] | None = None) -> str:
    """``tree_map.md``: where each anchor is in the checkout at ``repo``, and how to find it.

    ``found`` is every anchor already resolved there; by default they are resolved here.
    """
    lines = [
        "# Where things are in this checkout",
        "",
        f"Resolved from `{repo}` when this run started. ModelingV2 itself is "
        f"`{MODELING_V2_PACKAGE}`, its tests `{CATALOG_TESTS_DIR}`. A path below "
        "is where the thing is now; the search beside it finds it if the tree moves "
        "again. Missing means this checkout has it nowhere the workflow looked.",
        "",
    ]
    for one in found if found is not None else [resolve(repo, key) for key in ANCHORS]:
        anchor = one.anchor
        if one.found:
            where = ", ".join(f"`{p}`" for p in one.paths[:6])
            if len(one.paths) > 6:
                where += f" and {len(one.paths) - 6} more"
            moved = f" (moved from `{anchor.hint}`)" if one.via == "search" else ""
            lines.append(f"- **{anchor.concept}**: {where}{moved}. Find: {anchor.search}.")
        else:
            lines.append(
                f"- **{anchor.concept}**: missing (not at `{anchor.hint}`; searched "
                f"{', '.join(f'`{r}/`' for r in anchor.roots)} for {anchor.search})."
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "ANCHORS",
    "DIR",
    "FILE",
    "MAX_SEARCH_BYTES",
    "Anchor",
    "Resolution",
    "missing",
    "read",
    "render_map",
    "resolve",
]

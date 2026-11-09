# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the gate harness reads of TensorRT-LLM beyond its public API, found in the checkout.

The rank hook (``harness/rankhook``) wraps ``BaseWorker.setup_engine``;
``harness/engine_facts.read_worker`` reads the worker's engine internals; and
``runtime_evidence`` parses lines the engine logs. None of these is a public
API, so a TensorRT-LLM change can move one, and the checks resting on it then
report their evidence UNKNOWN, but only once a GPU run is over.
:data:`DEPENDENCIES` names each one and where it lives; :func:`check` looks for
them by parsing the checkout's files, never importing them, so preflight can
name what is gone before any GPU waits.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .runtime_evidence import REUSE_DISABLED_MARKER

_BASE_WORKER = "tensorrt_llm/executor/base_worker.py"
_PY_EXECUTOR = "tensorrt_llm/_torch/pyexecutor/py_executor.py"
_KV_MANAGERS = (
    "tensorrt_llm/_torch/pyexecutor/resource_manager.py",
    "tensorrt_llm/_torch/pyexecutor/kv_cache",
)
_TEXT_SUFFIXES = (".py", ".cpp", ".cc", ".cu", ".h")


@dataclass(frozen=True)
class Dependency:
    """One thing the harness reads, and where in the checkout it may be defined.

    ``kind`` is ``method`` (``needle`` is ``Class.method``, or ``*.method`` for
    any class), ``attribute`` (``needle`` is assigned on ``self`` or declared in
    a class body) or ``text`` (``needle`` is a substring of a source file).
    ``roots`` are files or directories, relative to the checkout.
    """

    what: str
    kind: str
    needle: str
    roots: tuple[str, ...]
    #: The evidence read through it, as the gates name it.
    feeds: str


DEPENDENCIES: tuple[Dependency, ...] = (
    Dependency(
        "BaseWorker.setup_engine",
        "method",
        "BaseWorker.setup_engine",
        (_BASE_WORKER,),
        "every rank's engine record (the rank hook wraps it)",
    ),
    Dependency(
        "worker.llm_args", "attribute", "llm_args", (_BASE_WORKER,), "every rank's configuration"
    ),
    Dependency(
        "worker.engine", "attribute", "engine", (_BASE_WORKER,), "every rank's engine internals"
    ),
    Dependency("worker.rank", "attribute", "rank", (_BASE_WORKER,), "which rank a record is"),
    Dependency(
        "PyExecutor.model_engine",
        "attribute",
        "model_engine",
        (_PY_EXECUTOR,),
        "kv_cache_quant, mamba_ssm_cache_dtype_resolved, cuda_graph_captured",
    ),
    Dependency(
        "PyExecutor.kv_cache_manager",
        "attribute",
        "kv_cache_manager",
        (_PY_EXECUTOR,),
        "block_reuse, kv_max_tokens",
    ),
    Dependency(
        "PyTorchModelEngine._cuda_graph_batch_sizes",
        "attribute",
        "_cuda_graph_batch_sizes",
        ("tensorrt_llm/_torch/pyexecutor/model_engine.py",),
        "cuda_graph_captured",
    ),
    Dependency(
        "KV cache manager enable_block_reuse",
        "attribute",
        "enable_block_reuse",
        _KV_MANAGERS,
        "block_reuse",
    ),
    Dependency(
        "KV cache manager get_kv_cache_stats",
        "method",
        "*.get_kv_cache_stats",
        _KV_MANAGERS,
        "kv_max_tokens",
    ),
    Dependency(
        "LLM.get_kv_cache_capacity",
        "method",
        "*.get_kv_cache_capacity",
        ("tensorrt_llm/llmapi/llm.py",),
        "rank 0's KV-cache capacity",
    ),
    Dependency(
        "the hybrid Mamba reuse-off log line",
        "text",
        REUSE_DISABLED_MARKER,
        ("tensorrt_llm/_torch/pyexecutor",),
        "block_reuse on a hybrid model",
    ),
    Dependency(
        "the paged KV-cache allocation log line",
        "text",
        "GiB for max tokens in paged KV cache (",
        ("cpp/tensorrt_llm/batch_manager",),
        "kv_log_tokens",
    ),
)


def _files(repo: Path, roots: tuple[str, ...]) -> list[Path] | None:
    """The source files under ``roots``; ``None`` when none of the roots is in the checkout."""
    found: list[Path] = []
    present = False
    for root in roots:
        path = repo / root
        if path.is_file():
            present = True
            found.append(path)
        elif path.is_dir():
            present = True
            found.extend(
                p for p in sorted(path.rglob("*")) if p.suffix in _TEXT_SUFFIXES and p.is_file()
            )
    return found if present else None


def _tree(path: Path, parsed: dict[Path, ast.Module | None]) -> ast.Module | None:
    if path not in parsed:
        try:
            parsed[path] = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            parsed[path] = None
    return parsed[path]


def _has_method(tree: ast.Module, needle: str) -> bool:
    owner, _, method = needle.partition(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and owner in ("*", node.name):
            if any(
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method
                for item in node.body
            ):
                return True
    return False


def _has_attribute(tree: ast.Module, name: str) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and node.attr == name
        ):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                return True
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if item.target.id == name:
                        return True
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                    return True  # a property
    return False


def _found(
    dependency: Dependency, files: list[Path], parsed: dict[Path, ast.Module | None]
) -> bool:
    for path in files:
        if dependency.kind == "text":
            try:
                if dependency.needle in path.read_text(encoding="utf-8", errors="replace"):
                    return True
            except OSError:
                continue
            continue
        if path.suffix != ".py":
            continue
        tree = _tree(path, parsed)
        if tree is None:
            continue
        if dependency.kind == "method" and _has_method(tree, dependency.needle):
            return True
        if dependency.kind == "attribute" and _has_attribute(tree, dependency.needle):
            return True
    return False


def check(repo: Path) -> tuple[list[Dependency], list[Dependency]]:
    """``(missing, unchecked)`` among :data:`DEPENDENCIES` in the checkout at ``repo``.

    A dependency is unchecked when none of its roots is in the checkout (a
    sparse one), and missing when its roots are there but none defines it.
    """
    missing: list[Dependency] = []
    unchecked: list[Dependency] = []
    parsed: dict[Path, ast.Module | None] = {}
    for dependency in DEPENDENCIES:
        files = _files(Path(repo), dependency.roots)
        if files is None:
            unchecked.append(dependency)
        elif not _found(dependency, files, parsed):
            missing.append(dependency)
    return missing, unchecked


def describe(missing: list[Dependency], unchecked: list[Dependency]) -> list[str]:
    """Preflight notes for :func:`check`'s answer; empty when the checkout has everything."""
    notes = []
    if missing:
        gone = "; ".join(
            f"{d.what} (looked for in {', '.join(d.roots)}; feeds {d.feeds})" for d in missing
        )
        notes.append(
            f"harness drift: this checkout no longer has {gone}. The checks resting on it will "
            "report their evidence UNKNOWN after the GPU run; this is the flow's harness falling "
            "behind TensorRT-LLM, not the target. Point the reader in checks/harness at where it "
            "now lives and update checks/harness_contract.py"
        )
    if unchecked:
        notes.append(
            "harness dependencies not checked, their files are not in the checkout: "
            + ", ".join(d.what for d in unchecked)
        )
    return notes

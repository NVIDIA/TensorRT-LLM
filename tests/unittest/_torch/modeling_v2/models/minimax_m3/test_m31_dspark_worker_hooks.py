# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every MiniMax-M3.1-Flash DSpark worker's ``prebuild`` checks its optional worker hooks before it uses them.

The ``enable_*`` drafter hooks are optional on the engine's DSpark worker, so ``prebuild`` must return before its
first hook call when any hook it uses is missing (the in-tree DSpark worker paths then run). Static check of each
target's source (stdlib ast): every ``self.enable_*`` call in ``prebuild`` is listed in the module's
``_WORKER_HOOKS``, and the ``hasattr`` guard over ``_WORKER_HOOKS`` with its early return comes before the first call.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tensorrt_llm").is_dir())
MODEL_ROOT = ROOT / "tensorrt_llm/_torch/modeling_v2/models/minimax_m3"
DSPARK_PATHS = sorted(MODEL_ROOT.glob("minimax_m3_1_flash_nvfp4__*/dspark.py"))
assert DSPARK_PATHS, f"No MiniMax-M3.1-Flash dspark.py files found under {MODEL_ROOT}"


def _worker_hooks(tree: ast.Module) -> list:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_WORKER_HOOKS" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError("module defines no _WORKER_HOOKS")


def _prebuild(tree: ast.Module) -> ast.FunctionDef:
    found = [
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "prebuild"
    ]
    assert len(found) == 1, f"expected one prebuild, found {len(found)}"
    return found[0]


def _hook_calls(fn: ast.FunctionDef) -> list:
    calls = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr.startswith("enable_")
        ):
            calls.append((node.lineno, node.func.attr))
    return sorted(calls)


def _guard_line(fn: ast.FunctionDef) -> int:
    """Line of the top-level ``if missing: ... return`` that follows the hasattr scan over _WORKER_HOOKS."""
    scanned = None
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign) and "_WORKER_HOOKS" in ast.unparse(stmt.value):
            scanned = {t.id for t in stmt.targets if isinstance(t, ast.Name)}
        elif (
            scanned
            and isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id in scanned
            and any(isinstance(s, ast.Return) for s in stmt.body)
        ):
            return stmt.lineno
    raise AssertionError("prebuild has no early return guarded by a _WORKER_HOOKS scan")


@pytest.mark.parametrize("path", DSPARK_PATHS, ids=lambda p: p.parent.name)
def test_prebuild_checks_every_hook_before_use(path: Path) -> None:
    tree = ast.parse(path.read_text())
    hooks = set(_worker_hooks(tree))
    prebuild = _prebuild(tree)
    calls = _hook_calls(prebuild)
    assert calls, "prebuild calls no worker hook"
    unlisted = sorted({name for _, name in calls} - hooks)
    assert not unlisted, f"prebuild calls hooks missing from _WORKER_HOOKS: {unlisted}"
    guard = _guard_line(prebuild)
    early = [name for line, name in calls if line < guard]
    assert not early, f"prebuild calls {early} before its _WORKER_HOOKS guard (line {guard})"

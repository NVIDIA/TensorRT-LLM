# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Path check for the optional kernel imports in DSA metadata warmup.

The warmup paths import their kernels inside ``try: ... except ImportError: return``,
so a stale module path would silently skip warmup instead of failing. This test
resolves every such import statically (no kernel module is executed), so it also
runs on stacks without CuTe DSL.
"""

import ast
from pathlib import Path

import pytest

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.sparse.dsa import metadata as dsa_metadata

pytestmark = pytest.mark.cpu_only

_PACKAGE_PARENT = Path(tensorrt_llm.__file__).resolve().parent.parent


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return False
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(
        isinstance(n, ast.Name) and n.id in ("ImportError", "ModuleNotFoundError") for n in names
    )


def _guarded_imports():
    tree = ast.parse(Path(dsa_metadata.__file__).read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not any(map(_catches_import_error, node.handlers)):
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.ImportFrom) and sub.level == 0:
                    found.extend((sub.module, alias.name, sub.lineno) for alias in sub.names)
    return found


def _module_file(module: str):
    base = _PACKAGE_PARENT.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _module_level_names(path: Path) -> set:
    """Names bound at module level, including inside top-level if/try blocks."""
    names = set()
    pending = list(ast.parse(path.read_text()).body)
    while pending:
        stmt = pending.pop()
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in stmt.names)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(stmt, (ast.If, ast.Try, ast.With)):
            for field in ("body", "orelse", "finalbody", "handlers"):
                for child in getattr(stmt, field, []):
                    pending.extend(child.body if isinstance(child, ast.ExceptHandler) else [child])
    return names


_GUARDED = _guarded_imports()


def test_guarded_imports_found():
    # Guards the parametrized test below against passing vacuously.
    assert _GUARDED, "no try/except ImportError imports found in DSA metadata"


@pytest.mark.parametrize("module,name,lineno", _GUARDED, ids=[f"{m}.{n}" for m, n, _ in _GUARDED])
def test_guarded_import_resolves(module: str, name: str, lineno: int):
    module_file = _module_file(module)
    assert module_file is not None, f"metadata.py:{lineno}: module {module} does not exist"
    if _module_file(f"{module}.{name}") is not None:
        return
    assert name in _module_level_names(module_file), (
        f"metadata.py:{lineno}: {module} has no submodule or attribute {name}"
    )

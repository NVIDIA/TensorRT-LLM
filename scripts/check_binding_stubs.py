#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail when collection binding stubs miss a compiled extension.

``tests/integration/defs/stubify_bindings.py`` is a factory: most symbols are
invented on demand. Collection only needs ``_STUB_ROOTS`` to cover every
compiled Python extension the package ships so ``pytest --co`` can import
without a wheel.

This script is stdlib-only so pre-commit and GitHub Release Checks can run it
without a wheel, GPU, or torch.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent
STUB_PATH = REPO_ROOT / "tests" / "integration" / "defs" / "stubify_bindings.py"
SETUP_PATH = REPO_ROOT / "setup.py"
_SKIP_PACKAGE_DATA_PREFIXES = ("libs/", "include/", "runtime/")
_EXTENSION_SUFFIXES = frozenset({".so", ".pyd", ".dll"})


def _stub_roots() -> set[str]:
    """Load ``_STUB_ROOTS`` by importing the collection stub plugin.

    Import runs ``install_bindings_stub()`` (pytest ``-p`` needs that on
    import). This checker then exits, so the meta-path finder is harmless.
    """
    spec = importlib.util.spec_from_file_location("stubify_bindings", STUB_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {STUB_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module._STUB_ROOTS)


def _const_strings(node: ast.expr) -> list[str]:
    """Extract string literals from a list or a single constant."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if not isinstance(node, ast.List):
        return []
    return [
        elt.value
        for elt in node.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    ]


def _setup_package_data_patterns(tree: ast.AST) -> list[str]:
    """Collect ``package_data`` glob strings assigned in setup.py via AST."""
    patterns: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "package_data" for target in node.targets
        ):
            patterns.extend(_const_strings(node.value))
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "package_data"
        ):
            patterns.extend(_const_strings(node.value))
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "append"
                and isinstance(func.value, ast.Name)
                and func.value.id == "package_data"
            ):
                for arg in node.args:
                    patterns.extend(_const_strings(arg))
    return patterns


def _root_extension_module(pattern: str) -> str | None:
    """Map a setuptools package_data glob to ``tensorrt_llm.<ext>`` or None.

    Top-level compiled extensions: ``bindings.*.so`` → ``tensorrt_llm.bindings``.
    One-level Python packages: ``flash_mla/*.py`` → ``tensorrt_llm.flash_mla``.
    Nested native libs (``libs/*.so``) and mypyc trees (``runtime/...``) are
    skipped.
    """
    posix = pattern.replace("\\", "/")
    if posix.startswith(_SKIP_PACKAGE_DATA_PREFIXES):
        return None
    path = PurePosixPath(posix)
    if len(path.parts) == 1:
        if path.suffix.lower() not in _EXTENSION_SUFFIXES:
            return None
        stem = path.stem.removesuffix(".*").rstrip("*")
        if not stem.isidentifier():
            return None
        return f"tensorrt_llm.{stem}"
    if len(path.parts) == 2 and path.parts[1] == "*.py" and path.parts[0].isidentifier():
        return f"tensorrt_llm.{path.parts[0]}"
    return None


def _compiled_extension_roots(setup_source: str) -> set[str]:
    tree = ast.parse(setup_source, filename=str(SETUP_PATH))
    roots: set[str] = set()
    for pattern in _setup_package_data_patterns(tree):
        module = _root_extension_module(pattern)
        if module is not None:
            roots.add(module)
    return roots


def main() -> int:
    stub_roots = _stub_roots()
    compiled_roots = _compiled_extension_roots(SETUP_PATH.read_text(encoding="utf-8"))

    missing_roots = sorted(compiled_roots - stub_roots)
    if not missing_roots:
        print("OK: collection binding stubs cover compiled extensions.")
        return 0

    stub_rel = STUB_PATH.relative_to(REPO_ROOT)
    print("Collection binding stubs are out of date:\n")
    print(
        f"  - {stub_rel}: _STUB_ROOTS is missing compiled modules "
        f"{missing_roots}. Add them when you introduce a new bindings/.so "
        "package so Check Test List can collect without a wheel."
    )
    print(
        "\nUpdate tests/integration/defs/stubify_bindings.py in this change "
        "so Jenkins Check Test List keeps working."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

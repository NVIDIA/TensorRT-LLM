# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The test tooling ships in the wheel for driver teams; no engine module may import it."""

from __future__ import annotations

import ast
import pathlib

import pytest

import tensorrt_llm._torch.disaggregation.testing as testing

pytestmark = pytest.mark.cpu_only

TESTING = "tensorrt_llm._torch.disaggregation.testing"
TESTING_DIR = pathlib.Path(testing.__file__).resolve().parent
PACKAGE_DIR = TESTING_DIR.parents[2]  # .../tensorrt_llm


def _package_of(path: pathlib.Path) -> str:
    """The package a module at ``path`` resolves relative imports against."""
    relative = path.resolve().relative_to(PACKAGE_DIR.parent).with_suffix("")
    return ".".join(relative.parts[:-1])  # a module's parent; a package's own name


def _testing_imports(source: str, package: str) -> list[str]:
    """Absolute names of everything ``source`` imports from the testing package, deferred and
    relative imports included."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                anchor = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                module = f"{anchor}.{module}" if module else anchor
            found.append(module)
            found.extend(f"{module}.{alias.name}" for alias in node.names)
    return [name for name in found if name == TESTING or name.startswith(TESTING + ".")]


def test_no_engine_module_imports_the_testing_package() -> None:
    offenders = {}
    for path in PACKAGE_DIR.rglob("*.py"):
        if TESTING_DIR in path.resolve().parents:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if "testing" not in source:
            continue  # cannot name the package without the word
        found = _testing_imports(source, _package_of(path))
        if found:
            offenders[str(path.relative_to(PACKAGE_DIR))] = found
    assert offenders == {}


def test_the_import_scan_sees_every_kind_of_import() -> None:
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from ..testing import driver_conformance\n"
        "def later():\n"
        "    import tensorrt_llm._torch.disaggregation.testing.memory_store\n"
        "    from ..testing.host_memory import HostBuffer\n"
        "    from . import keys\n"
    )
    found = _testing_imports(source, "tensorrt_llm._torch.disaggregation.blob")
    assert f"{TESTING}.driver_conformance" in found
    assert f"{TESTING}.memory_store" in found
    assert f"{TESTING}.host_memory.HostBuffer" in found
    assert len(found) == 5  # the three modules, two members; nothing from the blob import
    assert _package_of(PACKAGE_DIR / "_torch" / "disaggregation" / "blob" / "store.py") == (
        "tensorrt_llm._torch.disaggregation.blob"
    )
    assert _package_of(PACKAGE_DIR / "_torch" / "disaggregation" / "__init__.py") == (
        "tensorrt_llm._torch.disaggregation"
    )

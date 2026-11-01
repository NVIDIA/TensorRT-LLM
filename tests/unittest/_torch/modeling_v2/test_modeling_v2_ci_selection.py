# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What per-family CI test selection relies on, read from the source tree.

For a change under ``models/<family>/``, CI runs only that family's tests
(``jenkins/scripts/cbts/cbts/rules/modeling_v2.py``). That is safe only while no
family reaches another family's code, no stock code reaches modeling_v2, and each
family's accuracy tests sit in ``test_<family>.py``.

This file sits directly under ``tests/unittest/_torch/modeling_v2/`` so CI keeps it
for every modeling_v2 change. It reads files only.
"""

from __future__ import annotations

import ast
import re
from importlib.util import resolve_name
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_PY = "tensorrt_llm/_torch/_experimental/modeling_v2"
_CPP = "cpp/tensorrt_llm/modeling_v2"
_REPO = Path(__file__).resolve().parents[4]
_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2"
_MODELS_PACKAGE = f"{_PACKAGE}.models"
_NATIVE_SUFFIXES = (".cu", ".cuh", ".h", ".hpp", ".inl", ".cpp")
_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


def _families() -> list[Path]:
    return sorted(p.parent for p in (_REPO / _PY / "models").glob("*/__init__.py"))


def _imported_modules(node: ast.AST, package: str) -> list[str]:
    """The absolute names an import statement in ``package`` reaches, one per imported name."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level:
            module = resolve_name("." * node.level + module, package)
        return [f"{module}.{alias.name}" for alias in node.names]
    return []


def test_families_do_not_reach_each_other():
    """A family's Python imports and C++ includes stay inside the family."""
    families = {p.name for p in _families()}
    reaches = []
    for family_dir in _families():
        family = family_dir.name
        for path in sorted(family_dir.rglob("*")):
            where = path.relative_to(_REPO)
            if path.suffix == ".py":
                package = ".".join((_PACKAGE, *path.parent.relative_to(_REPO / _PY).parts))
                for node in ast.walk(ast.parse(path.read_text())):
                    for module in _imported_modules(node, package):
                        if module.startswith(_MODELS_PACKAGE + "."):
                            owner = module[len(_MODELS_PACKAGE) + 1 :].split(".")[0]
                            if owner != family:
                                reaches.append(f"{where} imports {module}")
            elif path.suffix in _NATIVE_SUFFIXES:
                for include in _INCLUDE.findall(path.read_text(errors="ignore")):
                    owner = include.split("/", 1)[0]
                    if owner in families and owner != family:
                        reaches.append(f"{where} includes {include}")
    assert not reaches, "\n".join(reaches)


_STOCK_PY_REFERENCE = re.compile(
    r"modeling_v2[./]models\b|torch\.ops\.mv2_|register_fake\(\s*f?[\"']mv2_"
)
_STOCK_CPP_INCLUDE = re.compile(r'#\s*include\s*"tensorrt_llm/modeling_v2/')


def test_stock_code_does_not_reach_modeling_v2():
    """Stock code reaches modeling_v2 only through the resolver.

    No family import, op call or fake registration in Python, and no modeling_v2
    header in C++.
    """
    offenders = [
        str(path.relative_to(_REPO))
        for path in sorted((_REPO / "tensorrt_llm").rglob("*.py"))
        if not path.is_relative_to(_REPO / _PY)
        and _STOCK_PY_REFERENCE.search(path.read_text(errors="ignore"))
    ]
    for root in ("cpp/tensorrt_llm", "cpp/include", "cpp/tests"):
        for path in sorted((_REPO / root).rglob("*")):
            # Generated cubin sources are large and include nothing of interest.
            if (
                path.is_file()
                and path.suffix in _NATIVE_SUFFIXES
                and not path.is_relative_to(_REPO / _CPP)
                and path.stat().st_size < (1 << 20)
                and _STOCK_CPP_INCLUDE.search(path.read_text(errors="ignore"))
            ):
                offenders.append(str(path.relative_to(_REPO)))
    assert not offenders, f"stock code reaches modeling_v2: {offenders}"


def test_accuracy_files_are_named_after_a_family():
    """A family's accuracy tests are selected through the file ``test_<family>.py``."""
    families = {p.name for p in _families()}
    accuracy = _REPO / "tests/integration/defs/accuracy/modeling_v2"
    misnamed = [
        p.name for p in sorted(accuracy.glob("test_*.py")) if p.stem[len("test_") :] not in families
    ]
    assert not misnamed, f"accuracy files named after no modeling_v2 family: {misnamed}"

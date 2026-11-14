# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The check that a GLM-5.3-Flash target module's ``UNCERTIFIED_GENERIC_CALLS`` names the stock code it runs.

A target's generic path runs stock tensorrt_llm modules that no catalog entry certifies. Every module of the target
that imports them lists them in ``UNCERTIFIED_GENERIC_CALLS``, so a dependency on stock code is visible and the list
shrinks as entries replace stock calls. The check reads the source and imports nothing. Each target's test directory
runs it on every module of that target that imports stock code or declares the list.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2"
FAMILY = "glm5_next"
FAMILY_DIR = (
    Path(__file__).resolve().parents[6] / "tensorrt_llm/_torch/_experimental/modeling_v2/models"
) / FAMILY

#: tensorrt_llm imports that compute nothing, so the generic path need not declare them: types and configs it
#: reads, enums it passes, the metadata it is handed and its logger.
_NON_COMPUTE_IMPORTS = frozenset(
    {
        "AllReduceStrategy",
        "AttentionMetadata",
        "AuxStreamType",
        "Mapping",
        "ModelConfig",
        "QuantAlgo",
        "QuantConfig",
        "logger",
    }
)


def is_entry_module(module: str, target: str) -> bool:
    """Whether ``module`` is modeling_v2 code the declaring module may call: catalog entries, the native loader, the
    family's ``ops`` and the target's own ``ops``."""
    family_package = f"{PACKAGE}.models.{FAMILY}"
    roots = (
        f"{PACKAGE}.catalog",
        f"{PACKAGE}._native",
        f"{family_package}.ops",
        f"{family_package}.{target}.ops",
    )
    return any(module == root or module.startswith(root + ".") for root in roots)


def declared(tree: ast.Module) -> tuple[str, ...] | None:
    """The string tuple assigned to ``UNCERTIFIED_GENERIC_CALLS``, or None if the module has none."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "UNCERTIFIED_GENERIC_CALLS" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    return None


def stock_imports(tree: ast.Module, target: str) -> set[str]:
    """Every stock tensorrt_llm name the module imports outside modeling_v2 entries, module level, function local or
    for type checking only, except the names that compute nothing.

    ``from <module> import <name>`` contributes ``<module>.<name>``; ``import <module>`` (aliased or not) contributes
    ``<module>``, since every name reached through it is stock code.
    """
    imported = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            and node.module.split(".")[0] == "tensorrt_llm"
            and not is_entry_module(node.module, target)
        ):
            imported.update(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name not in _NON_COMPUTE_IMPORTS
            )
        elif isinstance(node, ast.Import):
            imported.update(
                alias.name
                for alias in node.names
                if alias.name.split(".")[0] == "tensorrt_llm"
                and not is_entry_module(alias.name, target)
            )
    return imported


def audited_modules(target: str, family_dir: Path = FAMILY_DIR) -> list[Path]:
    """The modules of ``target`` that import stock code or declare ``UNCERTIFIED_GENERIC_CALLS``."""
    audited = []
    for path in sorted((family_dir / target).rglob("*.py")):
        tree = ast.parse(path.read_text())
        if declared(tree) is not None or stock_imports(tree, target):
            audited.append(path)
    return audited


def check_names_every_stock_import(path: Path, target: str) -> None:
    """The list of the module at ``path`` in ``target`` names every stock import, and nothing it does not import, so
    it cannot go stale."""
    tree = ast.parse(path.read_text())
    imported = stock_imports(tree, target)
    declaration = declared(tree)
    assert declaration is not None, (
        f"imports stock code but declares no UNCERTIFIED_GENERIC_CALLS: {sorted(imported)}"
    )
    names = set(declaration)
    assert imported <= names, (
        f"imported but not in UNCERTIFIED_GENERIC_CALLS: {sorted(imported - names)}"
    )
    assert names <= imported, (
        f"UNCERTIFIED_GENERIC_CALLS names what it does not import: {sorted(names - imported)}"
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The check that a Kimi K3 module's ``UNCERTIFIED_GENERIC_CALLS`` names the stock code it runs.

The generic path runs stock tensorrt_llm modules that no catalog entry certifies.
The module that imports them lists them in ``UNCERTIFIED_GENERIC_CALLS``, so a
dependency on stock code is visible and the list shrinks as entries replace
stock calls. The check reads the source and imports nothing. Each target's test
directory runs it on that target's declaring modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = "tensorrt_llm._torch.modeling_v2"
FAMILY = "kimi_k3_vl"
FAMILY_DIR = (
    Path(__file__).resolve().parents[6] / "tensorrt_llm/_torch/modeling_v2/models"
) / FAMILY

#: tensorrt_llm imports that compute nothing, so the generic path need not declare them: types and configs it
#: reads, enums it passes, the metadata it is handed and the views it takes of it, the graph-capture context it
#: asks about and its logger.
_NON_COMPUTE_IMPORTS = frozenset(
    {
        "AllReduceStrategy",
        "AttentionMetadata",
        "AuxStreamType",
        "KimiK3HybridCacheManager",
        "Mapping",
        "ModelConfig",
        "MoESchedulerKind",
        "QuantAlgo",
        "QuantConfig",
        "SiTuActivation",
        "is_in_breakable_cuda_graph",
        "k3_mla_decode_view",
        "logger",
    }
)


def is_entry_module(module: str, family: str, target: str | None) -> bool:
    """Whether ``module`` is modeling_v2 code the declaring module may call: catalog
    entries, the native loader, its family's ``ops`` and, in a target, the target's own
    ``ops``."""
    family_package = f"{PACKAGE}.models.{family}"
    roots = [f"{PACKAGE}.catalog", f"{PACKAGE}._native", f"{family_package}.ops"]
    if target is not None:
        roots.append(f"{family_package}.{target}.ops")
    return any(module == root or module.startswith(root + ".") for root in roots)


def declared(tree: ast.Module) -> tuple[str, ...] | None:
    """The string tuple assigned to ``UNCERTIFIED_GENERIC_CALLS``, or None if the module has none."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "UNCERTIFIED_GENERIC_CALLS" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    return None


def stock_imports(tree: ast.Module, family: str, target: str | None) -> set[str]:
    """Every tensorrt_llm name the module imports outside modeling_v2 entries, module level or
    function local, except the names that compute nothing."""
    imported = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            and node.module.split(".")[0] == "tensorrt_llm"
            and not is_entry_module(node.module, family, target)
        ):
            imported.update(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name not in _NON_COMPUTE_IMPORTS
            )
    return imported


def declaring_modules(target: str) -> list[Path]:
    """The modules of ``target`` that declare ``UNCERTIFIED_GENERIC_CALLS``."""
    return sorted(
        p
        for p in (FAMILY_DIR / target).rglob("*.py")
        if declared(ast.parse(p.read_text())) is not None
    )


def check_names_every_stock_import(path: Path, target: str) -> None:
    """The list of the module at ``path`` in ``target`` names every stock import, and nothing it does not import, so
    it cannot go stale."""
    tree = ast.parse(path.read_text())
    names, imported = set(declared(tree)), stock_imports(tree, FAMILY, target)
    assert imported <= names, (
        f"imported but not in UNCERTIFIED_GENERIC_CALLS: {sorted(imported - names)}"
    )
    assert names <= imported, (
        f"UNCERTIFIED_GENERIC_CALLS names what it does not import: {sorted(names - imported)}"
    )

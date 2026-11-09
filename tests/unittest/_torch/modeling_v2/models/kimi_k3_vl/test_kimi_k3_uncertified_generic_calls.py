# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Kimi K3 target's ``UNCERTIFIED_GENERIC_CALLS`` names the stock code it runs.

The generic path runs stock tensorrt_llm modules that no catalog entry certifies.
Each target lists them in ``UNCERTIFIED_GENERIC_CALLS``, so a dependency on stock
code is visible and the list shrinks as entries replace stock calls. This reads
the source and imports nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2"
_FAMILY_DIR = (
    Path(__file__).resolve().parents[6] / "tensorrt_llm/_torch/_experimental/modeling_v2"
    "/models/kimi_k3_vl"
)

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


def _is_entry_module(module: str, target_dir: Path) -> bool:
    """Whether ``module`` is modeling_v2 code a target may call: catalog entries, the
    native loader, its family's ``ops`` and its own ``ops``."""
    family = f"{_PACKAGE}.models.{target_dir.parent.name}"
    roots = (
        f"{_PACKAGE}.catalog",
        f"{_PACKAGE}._native",
        f"{family}.ops",
        f"{family}.{target_dir.name}.ops",
    )
    return any(module == root or module.startswith(root + ".") for root in roots)


def _declared(tree: ast.Module) -> tuple[str, ...] | None:
    """The string tuple assigned to ``UNCERTIFIED_GENERIC_CALLS``, or None if the module has none."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "UNCERTIFIED_GENERIC_CALLS" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    return None


def _stock_imports(tree: ast.Module, target_dir: Path) -> set[str]:
    """Every tensorrt_llm name the module imports outside modeling_v2 entries, module level or
    function local, except the names that compute nothing."""
    imported = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            and node.module.split(".")[0] == "tensorrt_llm"
            and not _is_entry_module(node.module, target_dir)
        ):
            imported.update(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name not in _NON_COMPUTE_IMPORTS
            )
    return imported


def _declaring_targets() -> list[Path]:
    return sorted(
        d
        for d in _FAMILY_DIR.iterdir()
        if (d / "modeling.py").is_file()
        and _declared(ast.parse((d / "modeling.py").read_text())) is not None
    )


def test_some_target_declares_its_generic_calls():
    assert _declaring_targets(), "no Kimi K3 target declares UNCERTIFIED_GENERIC_CALLS"


@pytest.mark.parametrize("target_dir", _declaring_targets(), ids=lambda d: d.name)
def test_uncertified_generic_calls_name_every_stock_import(target_dir):
    """The list names every stock import, and nothing it does not import, so it cannot go stale."""
    tree = ast.parse((target_dir / "modeling.py").read_text())
    declared, imported = set(_declared(tree)), _stock_imports(tree, target_dir)
    assert imported <= declared, (
        f"imported but not in UNCERTIFIED_GENERIC_CALLS: {sorted(imported - declared)}"
    )
    assert declared <= imported, (
        f"UNCERTIFIED_GENERIC_CALLS names what it does not import: {sorted(declared - imported)}"
    )


@pytest.mark.parametrize(
    "module, accepted",
    [
        ("catalog.attention.entry", True),
        ("models.family.ops.attention.entry", True),
        ("models.family.target.ops.attention.entry", True),
        ("models.family.sibling.ops.attention.entry", False),
        ("models.other.ops.attention.entry", False),
        ("models.family.ops_extra.attention.entry", False),
        ("models.family.helper", False),
        ("models.family.kernels.kernel", False),
    ],
)
def test_generic_call_import_scope(tmp_path, module, accepted):
    target_dir = tmp_path / "models/family/target"
    tree = ast.parse(f"from {_PACKAGE}.{module} import op\n")
    assert (_stock_imports(tree, target_dir) == set()) == accepted

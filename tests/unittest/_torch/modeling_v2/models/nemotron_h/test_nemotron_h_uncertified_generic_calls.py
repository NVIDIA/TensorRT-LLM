# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A nemotron_h target's ``UNCERTIFIED_GENERIC_CALLS`` names the stock code it runs.

A target that runs stock tensorrt_llm code no catalog entry certifies -- the BF16 tep8
target's opt-in routed experts and every routed expert of the BF16 dep16 and dep32 targets run in
Modeling V1's ``create_moe`` layer -- lists it in
``UNCERTIFIED_GENERIC_CALLS``, so the dependency is visible and the list shrinks as entries
replace stock calls. The same contract as the Kimi K3 targets'. This reads the source and
imports nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2"
_FAMILY_DIR = (
    Path(__file__).resolve().parents[6] / "tensorrt_llm/_torch/_experimental/modeling_v2"
    "/models/nemotron_h"
)

#: tensorrt_llm imports that compute nothing, so a target need not declare them: types and configs
#: it reads, enums and activation descriptors it passes, the metadata it is handed, the switches it
#: reads from the environment and its logger.
_NON_COMPUTE_IMPORTS = frozenset(
    {
        "ActivationType",
        "AllReduceFusionOp",
        "AllReduceStrategy",
        "AttentionMetadata",
        "DataType",
        "ModelConfig",
        "MoEWeightLoadingMode",
        "SimpleActivation",
        "TrtllmAttentionMetadata",
        "get_env_enable_pdl",
        "logger",
        "step_contract_enabled",
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


def _imports_moe(target_dir: Path) -> bool:
    """Whether the target imports anything from Modeling V1's fused MoE package."""
    tree = ast.parse((target_dir / "modeling.py").read_text())
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("tensorrt_llm._torch.moe")
        for node in ast.walk(tree)
    )


def test_every_target_on_v1_moe_declares_its_generic_calls():
    """A target that reaches Modeling V1's fused MoE says so; the BF16 tep8, dep16 and dep32
    targets do."""
    undeclared = [
        d.name
        for d in sorted(_FAMILY_DIR.iterdir())
        if (d / "modeling.py").is_file() and _imports_moe(d) and d not in _declaring_targets()
    ]
    assert not undeclared, (
        f"import tensorrt_llm._torch.moe without UNCERTIFIED_GENERIC_CALLS: {undeclared}"
    )
    declaring = {d.name for d in _declaring_targets()}
    assert {
        "nemotron3_ultra_bf16__sm_100__tep8",
        "nemotron3_ultra_bf16__sm_100__dep16",
        "nemotron3_ultra_bf16__sm_100__dep32",
    } <= declaring


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

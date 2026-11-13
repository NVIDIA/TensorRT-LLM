# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU test for REJ_MC on the sm_107 TP4 target: import resolution and wiring, from the sources (no import of
the target, no GPU). The TP4 modeling imports this target's m31_chain_spec_mc, the module carries the entry points the
TP4 install calls, and the install / prebuild are called from the TP4 shell's __init__ / cache_derived_state."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[7]
_TP4 = (
    _ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4"
)


def test_the_module_has_the_entry_points():
    src = (_TP4 / "ops/sampling/m31_chain_spec_mc.py").read_text()
    names = {n.name for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    assert {"chain_spec_mc", "chain_spec_mc_op", "prebuild", "ctas_per_row", "sm_count"} <= names
    consts = {
        t.id
        for n in ast.parse(src).body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    assert "MAX_BATCH" in consts


def test_the_tp4_shell_installs_and_prebuilds_rej_mc():
    src = (_TP4 / "modeling.py").read_text()
    tree = ast.parse(src)
    imp = [
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == "ops.sampling"
    ]
    assert any(
        a.name == "m31_chain_spec_mc" and a.asname == "_chain_spec_mc" for n in imp for a in n.names
    )
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"_install_rej_mc", "_prebuild_rej_mc", "_rej_mc_override"} <= set(funcs)
    shell = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ModelingV2MiniMaxM31FlashNvfp4Sm107Tp4"
    )
    calls = {
        m.name: {
            c.func.id
            for c in ast.walk(m)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }
        for m in shell.body
        if isinstance(m, ast.FunctionDef)
    }
    assert "_install_rej_mc" in calls["__init__"]
    assert "_prebuild_rej_mc" in calls["cache_derived_state"]
    assert 'm31_disabled("REJ_MC")' in src

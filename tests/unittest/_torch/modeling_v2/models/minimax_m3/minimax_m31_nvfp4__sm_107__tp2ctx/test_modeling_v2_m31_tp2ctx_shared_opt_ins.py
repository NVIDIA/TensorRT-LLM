# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP2 context route arms the shared sm_107 paths its role runs with.

Routing arms them process-wide (arm_sm107_shared_opt_ins: the ahead-of-time CuTe DSL objects and the shared opt-ins)
before the model is built; the context target's branch of route() is separate from the generation targets', so it
must call the helper itself. CPU.
"""

import ast
from pathlib import Path

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family


def _tp2ctx_branch() -> ast.If:
    path = Path(_family.__file__).resolve().parent / "routing.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    (route,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "route"]
    branches = [
        n
        for n in ast.walk(route)
        if isinstance(n, ast.If)
        and any(
            isinstance(r, ast.Return)
            and isinstance(r.value, ast.Name)
            and r.value.id == "ModelingV2MiniMaxM31Nvfp4Sm107Tp2Ctx"
            for r in n.body
        )
    ]
    assert len(branches) == 1
    return branches[0]


def test_tp2ctx_route_arms_the_shared_opt_ins():
    called = {
        n.func.id
        for n in ast.walk(_tp2ctx_branch())
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "arm_sm107_shared_opt_ins" in called, "the tp2ctx route does not arm the shared opt-ins"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP2 context target arms the shared sm_107 paths its role runs with.

They are model opt-ins (off in the default RuntimeHooks), so a target that does not arm them silently runs the stock
paths; routing also serves the target's ahead-of-time CuTe DSL objects. CPU.
"""

import ast
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (
    modeling as m,
)
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

CONTEXT_OPT_INS = (
    "draft_kv_reserve_guard",
    "kv_gpu_prefetch",
    "context_token_array",
    "msa_prefill_fork",
    "proxy_ms_reuse",
    "context_onehot_index_fill",
    "dflash_host_ctx_alloc",
)


@pytest.mark.parametrize("field", CONTEXT_OPT_INS)
def test_tp2ctx_arms_the_shared_opt_in(field):
    assert getattr(m.ModelingV2MiniMaxM31Nvfp4Sm107Tp2Ctx.runtime_hooks, field) is True
    assert getattr(RuntimeHooks(), field) is False


def test_routing_serves_the_aot_objects_for_tp2ctx():
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
    calls = [
        n
        for n in ast.walk(branches[0])
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "arm"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "cute_aot"
    ]
    assert calls, "the tp2ctx route does not call cute_aot.arm()"

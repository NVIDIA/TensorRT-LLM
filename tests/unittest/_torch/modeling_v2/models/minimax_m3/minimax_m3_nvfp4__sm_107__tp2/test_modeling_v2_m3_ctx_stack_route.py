# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The prefill target's context steps and the built-in CTX stack (modeling.CTX_STACK).

A step with context rows of at least CTX_STACK_MIN_ROWS rows takes the built-in CTX stack; a step without context rows
never does. Inside the stack the MoE runs the built-in forward only from the CTX norm's rows
(``_CTX_NORM_MXFP8_MIN_ROWS``), where that norm emits the NVFP4 MoE input; the boundary into the final norm is never
the CTX one. CTX_STACK=False routes nothing.

CPU only: nothing here launches a kernel.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (
    modeling,
)
from tensorrt_llm._torch.models import modeling_minimaxm3 as builtin

pytestmark = pytest.mark.cpu_only


def test_the_ctx_stack_defaults() -> None:
    # On by default from 4096 rows; the MoE hand-off and the CTX norm start at the norm's floor, above the stack's.
    assert modeling.CTX_STACK is True
    assert modeling.CTX_STACK_MIN_ROWS == 4096
    assert modeling.CTX_NORM_MIN_ROWS == builtin._CTX_NORM_MXFP8_MIN_ROWS == 8192


@pytest.mark.parametrize(
    "stack, num_contexts, rows, expected",
    [
        (True, 1, 4096, True),
        (True, 1, 4095, False),
        (True, 2, 16384, True),
        (True, 0, 16384, False),
        (False, 1, 16384, False),
    ],
)
def test_ctx_stack_step(monkeypatch, stack, num_contexts, rows, expected) -> None:
    monkeypatch.setattr(modeling, "CTX_STACK", stack)
    assert modeling._ctx_stack_step({"num_contexts": num_contexts}, rows) is expected


@pytest.mark.parametrize(
    "stack, rows, expected", [(True, 8192, True), (True, 8191, False), (False, 8192, False)]
)
def test_ctx_norm_rows(monkeypatch, stack, rows, expected) -> None:
    monkeypatch.setattr(modeling, "CTX_STACK", stack)
    assert modeling._ctx_norm_rows(torch.empty((rows, 8))) is expected


class _TargetPathTaken(Exception):
    """Raised where the target's own MoE path starts, so a test sees which way the step went."""


class _Core:
    def __getattr__(self, name):
        raise _TargetPathTaken(name)


def _moe_route(rows: int, **kwargs) -> str:
    """Which MoE forward ``PrefillTarget._moe`` takes for ``rows``: 'built-in' or 'target'."""
    calls = []
    target = SimpleNamespace(
        core=_Core(), _ctx_moe=lambda i, h, nvfp4: calls.append((i, h, nvfp4)) or "built-in"
    )
    hidden = torch.empty((rows, 8), dtype=torch.bfloat16)
    try:
        out, routed = modeling.PrefillTarget._moe(target, 3, hidden, **kwargs)
    except _TargetPathTaken:
        return "target"
    assert routed is None and len(calls) == 1 and calls[0][1] is hidden
    return out


def test_moe_hands_ctx_steps_to_the_built_in_forward() -> None:
    assert _moe_route(16384, return_parts=False, ctx=True, ctx_nvfp4=None) == "built-in"


@pytest.mark.parametrize(
    "kwargs",
    [dict(return_parts=False, ctx=False), dict(return_parts=True, ctx=True)],
    ids=["not_ctx", "parts"],
)
def test_moe_keeps_the_target_path(kwargs) -> None:
    # A step outside the stack, or one whose boundary takes the shared and routed parts separately.
    assert _moe_route(16384, **kwargs) == "target"


@pytest.mark.parametrize("rows, expected", [(16384, True), (16385, False)])
def test_nvfp4_moe_input_only_for_one_runner_call(rows, expected) -> None:
    # The CTX norm's NVFP4 output stands in for the experts' quantize only in one runner call (moe_max_num_tokens).
    scale = torch.ones(1, dtype=torch.float32)
    core = SimpleNamespace(
        moe_max_num_tokens=16384, moe={3: SimpleNamespace(fc31_input_scale=scale)}
    )
    got = modeling._ctx_fp4_input_scale(core, 3, rows)
    assert (got is scale) if expected else got is None


def test_the_final_boundary_is_never_the_ctx_norm(monkeypatch) -> None:
    # The last layer's boundary feeds the final norm (no next QKV takes an MXFP8 copy): the generic boundary.
    reduced = []
    monkeypatch.setattr(modeling, "_ctx_reduce_norm", lambda *args: reduced.append(args))
    monkeypatch.setattr(modeling, "_next_norm", lambda core, i: None)
    generic = []
    target = SimpleNamespace(
        core=SimpleNamespace(num_layers=4),
        _generic_boundary=lambda x, x2, residual, norm_w: generic.append(x) or ("n", "u"),
    )
    x = torch.empty((16384, 8))
    assert modeling.PrefillTarget._boundary(target, 3, x, None, x) == ("n", "u")
    assert reduced == [] and len(generic) == 1


def test_no_ctx_stack_inside_a_breakable_graph(monkeypatch) -> None:
    # A captured breakable-graph bucket (its 4096-row one included) runs the generic path, eager steps the stack.
    monkeypatch.setattr(modeling, "is_in_breakable_cuda_graph", lambda: True)
    assert modeling._ctx_stack_step({"num_contexts": 1}, 4096) is False
    assert modeling._ctx_norm_rows(torch.empty((8192, 8))) is False
    monkeypatch.setattr(modeling, "is_in_breakable_cuda_graph", lambda: False)
    assert modeling._ctx_stack_step({"num_contexts": 1}, 4096) is True
    assert modeling._ctx_norm_rows(torch.empty((8192, 8))) is True


def test_an_eagle3_capture_layer_keeps_the_generic_boundary(monkeypatch) -> None:
    reduced = []
    monkeypatch.setattr(modeling, "_ctx_reduce_norm", lambda *args: reduced.append(args))
    monkeypatch.setattr(modeling, "_next_norm", lambda core, i: None)
    generic = []
    target = SimpleNamespace(
        core=SimpleNamespace(num_layers=8),
        _generic_boundary=lambda x, x2, residual, norm_w: generic.append(x) or ("n", "u"),
    )
    x = torch.empty((16384, 8))
    assert modeling.PrefillTarget._boundary(target, 3, x, None, x, True) == ("n", "u")
    assert reduced == [] and len(generic) == 1


def test_a_captured_4096_row_bucket_takes_the_breakable_boundary(monkeypatch) -> None:
    # The 4096-row breakable bucket reaches CTX_STACK_MIN_ROWS, but inside a capture the sparse attention is the
    # breakable path (captured projection + eager boundary): the CTX stack's fused QKV and armed core never run.
    monkeypatch.setattr(modeling, "is_in_breakable_cuda_graph", lambda: True)

    def _ctx_called(*args, **kwargs):
        raise AssertionError("the CTX stack ran inside a capture")

    monkeypatch.setattr(modeling, "_ctx_fused_qkv", _ctx_called)
    rows = modeling.CTX_STACK_MIN_ROWS
    packed = torch.empty((rows, 16), dtype=torch.bfloat16)
    linears = []
    monkeypatch.setattr(
        modeling, "_linear", lambda core, x, wk, sk, low_m: linears.append(wk) or (packed, False)
    )
    eager = []
    monkeypatch.setattr(modeling, "_prefill_attention_eager", lambda *args: eager.append(args[-1]))
    core = SimpleNamespace(sparse_layers={5}, fused_projection=True, heads=2, head_dim=4)
    target = SimpleNamespace(core=core, _bcg_key="k", _ctx_core_then_o_proj=_ctx_called)
    step = {"num_contexts": 1}
    hidden = torch.empty((rows, 16), dtype=torch.bfloat16)
    out, pushed = modeling.PrefillTarget._attention(target, 5, None, hidden, step)
    assert out is packed and pushed is False
    assert linears == ["l5_qkv_w", "l5_o_w"] and eager == [5]

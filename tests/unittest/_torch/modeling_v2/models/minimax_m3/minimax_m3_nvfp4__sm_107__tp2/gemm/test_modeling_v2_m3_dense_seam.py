# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the dense MLP's activation seam (``modeling._dense_swiglu``, the ``m3_dense_swiglu`` C++ op): between
the dense MLP's two decode GEMVs, the seam's down-GEMV output is bit for bit the Triton swigluoai's
(``m3_silu_and_mul``) at every decode row count, eager and in a CUDA graph, with PDL; and the seam declines (to the
Triton kernel) what it does not cover."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.gemm.m3_mx_gemv import (  # noqa: E501
    DENSE_DOWN,
    DENSE_GATE_UP,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.mx_gemv import (  # noqa: E501
    op as gemv,
)

assert torch.cuda.is_available(), "the dense seam test requires a CUDA device"

ALPHA, LIMIT, BETA = 1.702, 7.0, 1.0
ROWS = list(range(1, modeling.TINY_MAX_TOKENS + 1))
CORE = SimpleNamespace(swiglu_alpha=ALPHA, swiglu_limit=LIMIT, swiglu_beta=BETA)


def _weights(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    return wq.view(n, k), wsf


GATE_UP = None
DOWN = None


@pytest.fixture(scope="module", autouse=True)
def _mlp_weights():
    global GATE_UP, DOWN
    GATE_UP = _weights(*DENSE_GATE_UP, 11)
    DOWN = _weights(*DENSE_DOWN, 13)


def _gemv(x, wts):
    out = gemv.m3_mx_gemv_bf16(x.contiguous(), *wts, use_pdl=True)
    assert out is not None, (
        "the target's bf16 GEMV serves every decode row of the dense projections"
    )
    return out


def _mlp(x, seam: bool):
    """gate_up GEMV, the activation (the seam, or the Triton kernel), down GEMV."""
    gu = _gemv(x, GATE_UP)
    if seam:
        act = modeling._dense_swiglu(CORE, gu)
    else:
        act = modeling.m3_silu_and_mul(gu, LIMIT, ALPHA, BETA)
    return _gemv(act, DOWN)


def _x(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, DENSE_GATE_UP[1], generator=g, device="cuda") * 2).to(torch.bfloat16)


@pytest.fixture
def op_calls(monkeypatch):
    """Records each call that reaches the C++ op through the seam."""
    calls = []
    real = modeling.m3_dense_swiglu

    class _Recording:
        CELLS = real.CELLS

        def __call__(self, gu, *args):
            calls.append(gu.shape[0])
            return real(gu, *args)

    monkeypatch.setattr(modeling, "m3_dense_swiglu", _Recording())
    return calls


@pytest.mark.parametrize("rows", ROWS)
def test_seam_on_is_bitwise_seam_off(op_calls, rows) -> None:
    for trial in range(3):
        x = _x(rows, 100 * rows + trial)
        on = _mlp(x, True)
        off = _mlp(x, False)
        torch.cuda.synchronize()
        assert torch.equal(on.view(torch.int16), off.view(torch.int16)), f"trial {trial}"
    assert op_calls == [rows] * 3, "the seam runs the C++ op at every decode row count"


@pytest.mark.parametrize("rows", [1, 4, modeling.TINY_MAX_TOKENS])
def test_seam_in_a_graph_is_bitwise_seam_off(rows) -> None:
    """Both paths captured; replays on new inputs (PDL edges inside the graph)."""
    x = _x(rows, 7)
    _mlp(x, True)  # compile outside the capture
    graph_on = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_on):
        on = _mlp(x, True)
    _mlp(x, False)
    graph_off = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_off):
        off = _mlp(x, False)
    for step in range(3):
        x.copy_(_x(rows, 31 + step))
        graph_on.replay()
        graph_off.replay()
        torch.cuda.synchronize()
        assert torch.equal(on.view(torch.int16), off.view(torch.int16)), f"replay {step}"


def test_seam_declines_what_it_does_not_cover(op_calls, monkeypatch) -> None:
    """Rows past the op's certified rows and other activation parameters take the Triton kernel (the same bits)."""
    max_rows = max(cell.spec["rows"] for cell in modeling.m3_dense_swiglu.CELLS)
    # Past the decode rows the GEMV declines too; the seam only reads the gate_up rows, so any BF16 rows will do.
    g = torch.Generator(device="cuda").manual_seed(5)
    gu = torch.randn(max_rows + 1, DENSE_GATE_UP[0], generator=g, device="cuda").to(torch.bfloat16)
    got = modeling._dense_swiglu(CORE, gu)
    assert op_calls == [], "rows past the op's certified rows"
    assert torch.equal(got, modeling.m3_silu_and_mul(gu, LIMIT, ALPHA, BETA))
    other = SimpleNamespace(swiglu_alpha=1.0, swiglu_limit=LIMIT, swiglu_beta=BETA)
    gu = _gemv(_x(4, 5), GATE_UP)
    got = modeling._dense_swiglu(other, gu)
    assert op_calls == [], "other activation parameters"
    assert torch.equal(got, modeling.m3_silu_and_mul(gu, LIMIT, 1.0, BETA))


def test_seam_needs_no_capture_scope(op_calls, monkeypatch) -> None:
    """The op replaces only the activation kernel: the small-batch decode graphs are captured outside the FlashInfer
    MXFP8 capture scope (r4.10.1 never engaged at 1-8 rows)."""
    monkeypatch.setattr(modeling, "flashinfer_mxfp8_decode_graph_capture_active", lambda: False)
    modeling._dense_swiglu(CORE, _gemv(_x(4, 5), GATE_UP))
    assert op_calls == [4], "outside the FlashInfer capture scope"

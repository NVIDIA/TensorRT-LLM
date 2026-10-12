# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the cublas_mm call the DeepSeek-V4.1 target's router gate makes:
bf16 [M, 5120] x bf16 [5120, 384] -> fp32 [M, 384], i.e.
``cublas_mm(x, gate_w.t(), None, torch.float32)`` (tensor cores, fp32 accumulation).

Every product of two bf16 values is exact in fp32, so this call differs from the
checkpoint's fp32 gate GEMM (``linear(x.float(), weight.float())``) only in the
order and rounding of the fp32 summation; it is not bitwise equal to it. The
entry's cells hold the call at M 1 and 8192 with the entry's bf16-scored gate;
this file holds the tighter gate measured on sm_103:

* eager, every M from 1 to 8192 (the target splits larger router inputs before
  this call): within 5e-3 of the float64 product for unit-variance operands
  (the entry's fp32 cells' gate at K = 5120; worst M measured 2.52e-3), and
  bitwise repeatable;
* the expert selection the target makes from the logits (top-6 of
  ``sqrt(softplus(logit)) + bias``, the ``gate_forward`` rule) at router-scale
  logits agrees with the selection from the fp32 x fp32 call on >= 99.9 % of rows;
* CUDA graphs at the router's captured row counts: G and 6G (tep4 / tp4 decode and
  DSpark verify) and 4G and 24G (dep4's gathered rows) over the CUDA-graph batch
  set; every replay is bitwise equal to an eager call on the same input.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm as op

assert torch.cuda.is_available(), "cublas_mm requires a CUDA device"

K, N = 5120, 384
M_MAX = 8192
TOL = 5e-3
TOPK = 6
_GSET = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
_ROUTER_ROWS = tuple(sorted({f * g for g in _GSET for f in (1, 4, 6, 24)}))
assert len(_ROUTER_ROWS) == 122 and _ROUTER_ROWS[-1] == 6144


@pytest.fixture(scope="module")
def operands():
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(M_MAX, K, generator=g, device="cuda").to(torch.bfloat16)
    # [E, H], as the checkpoint holds it.
    w = torch.randn(N, K, generator=g, device="cuda").to(torch.bfloat16)
    ref = x.double() @ w.double().t()
    return x, w, ref


def _gate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return op(x, w.t(), None, torch.float32)


def _selection(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return (torch.sqrt(F.softplus(logits)) + bias).topk(TOPK, dim=-1).indices.sort(-1).values


def test_every_row_count_eager(operands):
    x, w, ref = operands
    worst = 0.0
    for m in range(1, M_MAX + 1):
        out = _gate(x[:m], w)
        assert out.dtype == torch.float32 and tuple(out.shape) == (m, N) and out.is_contiguous()
        err = float((out.double() - ref[:m]).abs().max())
        assert err <= TOL, (m, err)
        worst = max(worst, err)
        if m % 64 == 1 or m in (6, 24, 384, 1536, 6144, M_MAX):
            again = _gate(x[:m], w)
            assert torch.equal(out.view(torch.int32), again.view(torch.int32)), (
                f"not repeatable at M={m}"
            )
    print(
        f"\nbf16 -> fp32 5120 x 384: every M 1..{M_MAX} within {TOL} of float64 (worst {worst:.3g})"
    )


def test_expert_selection_matches_the_fp32_call(operands):
    x, w, _ = operands
    # Router-scale logits (std ~1.1): w / 64 is exact in bf16, so the products stay exact.
    w_r = (w.float() / 64).to(torch.bfloat16)
    bias = (
        torch.randn(N, generator=torch.Generator(device="cuda").manual_seed(1), device="cuda") * 0.1
    )
    tc = _selection(_gate(x, w_r), bias)
    fp32 = _selection(op(x.float(), w_r.float().t()), bias)
    differ = int((tc != fp32).any(-1).sum())
    assert differ <= M_MAX // 1000, differ
    print(f"\ntop-{TOPK} selection differs from the fp32 call's on {differ} of {M_MAX} rows")


def test_cuda_graph_at_the_router_rows(operands):
    x, w, ref = operands
    side = torch.cuda.Stream()
    for m in _ROUTER_ROWS:
        a = x[:m].clone()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _gate(a, w)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _gate(a, w)
        for shift in (1, 2):
            rows = (torch.arange(m, device="cuda") + shift * 131) % M_MAX
            a.copy_(x.index_select(0, rows))
            out.fill_(float("nan"))
            graph.replay()
            eager = _gate(a, w)
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.int32), eager.view(torch.int32)), (m, shift)
        a.copy_(x[:m])
        graph.replay()
        torch.cuda.synchronize()
        assert float((out.double() - ref[:m]).abs().max()) <= TOL, m
        del graph
    print(
        f"\ncaptured at {len(_ROUTER_ROWS)} row counts ({_ROUTER_ROWS[0]}..{_ROUTER_ROWS[-1]}), "
        "2 replays each"
    )

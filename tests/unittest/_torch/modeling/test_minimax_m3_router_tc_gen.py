# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 generation-step router: the TF32 tensor-core router (minimax_m3_router_tc, 16-CTA clusters) at 9..16 rows.

Checks on the device: the 16-CTA cluster query; router_tc against the stock router of these steps (fp32 linear under
cuBLAS TF32) within the documented rule-1b tolerance and against the fp32 product; eager == CUDA graph bitwise; the
NVFP4 activation payload (packed e2m1 and linear scales) bitwise vs fp4_quantize's unswizzled layout;
MiniMaxM3Gate.forward dispatch (9..16 generation rows take router_tc, 4 rows / context steps / the kill switch keep the
stock router).
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.m3_router_tc import (
    tf32_rna,
)
from tensorrt_llm._torch.models import modeling_minimaxm3 as mm
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)

E, K = 128, 6144


def _inputs(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(E, K, device="cuda", generator=g) * 0.02).float().contiguous()
    x = (torch.randn(m, K, device="cuda", generator=g) * 0.5).to(torch.bfloat16).contiguous()
    return x, w


def _stock(x, w):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        return torch.nn.functional.linear(x.float(), w)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _fp32_exact(x: torch.Tensor, w: torch.Tensor, full_fp32: bool = False) -> torch.Tensor:
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return x.float() @ (w if full_fp32 else tf32_rna(w)).t()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def test_cluster16_query():
    assert isinstance(torch.ops.trtllm.minimax_m3_router_tc_cluster16(K), bool)


@pytest.mark.parametrize("m", [9, 12, 16])
def test_router_tc_vs_stock_and_fp32(m):
    if not torch.ops.trtllm.minimax_m3_router_tc_cluster16(K):
        pytest.skip(
            "this device cannot co-schedule 16-CTA clusters (the gate keeps the stock router)"
        )
    x, w = _inputs(m, 1000 + m)
    tc = torch.ops.trtllm.minimax_m3_router_tc(x, w, None, False)[0]
    assert (tc - _fp32_exact(x, w)).abs().max().item() <= 1e-5
    stock = _stock(x, w)
    assert (tc - stock).abs().max().item() <= 2e-4
    assert torch.equal(tc.argmax(-1), stock.argmax(-1))


@pytest.mark.parametrize("m", [9, 16])
@pytest.mark.parametrize("full_fp32", [False, True])
def test_router_tc_eager_equals_graph_and_fp4_payload(m: int, full_fp32: bool) -> None:
    if not torch.ops.trtllm.minimax_m3_router_tc_cluster16(K):
        pytest.skip("this device cannot co-schedule 16-CTA clusters")
    x, w = _inputs(m, 2000 + m)
    sf_scale = torch.tensor([448.0 * 6.0 / 8.0], device="cuda", dtype=torch.float32)
    if full_fp32:
        assert torch.ops.trtllm.minimax_m3_router_tc_cluster16(K, full_fp32=True)
    eager = torch.ops.trtllm.minimax_m3_router_tc(x, w, sf_scale, True, full_fp32=full_fp32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.ops.trtllm.minimax_m3_router_tc(x, w, sf_scale, True, full_fp32=full_fp32)
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(eager, out):
        assert torch.equal(a, b)
    # router_tc emits LINEAR (unswizzled) E4M3 scales [M, K / 16]: compare against fp4_quantize's linear layout, the one
    # the MoE's pre-quantized input slot takes (the swizzled default pads M to 128 rows).
    q, sf = torch.ops.trtllm.fp4_quantize(x, sf_scale, 16, False, False)[:2]
    assert torch.equal(eager[1].contiguous().view(torch.uint8), q.contiguous().view(torch.uint8))
    assert eager[2].numel() == m * K // 16
    assert torch.equal(
        eager[2].contiguous().view(torch.uint8).flatten(),
        sf.contiguous().view(torch.uint8).flatten(),
    )


def test_gate_dispatch(monkeypatch):
    if not torch.ops.trtllm.minimax_m3_router_tc_cluster16(K):
        pytest.skip("this device cannot co-schedule 16-CTA clusters")
    monkeypatch.setattr(
        mm, "_ROUTER_TC_GEN", True
    )  # Exercise dispatch independently of the process environment.
    gate = mm.MiniMaxM3Gate(K, E, 4, 1.0).cuda()
    x16, w = _inputs(16, 3000)
    gate.weight.data.copy_(w)
    assert torch.ops.trtllm.minimax_m3_router_tc_cluster16(K, full_fp32=True)
    tc = torch.ops.trtllm.minimax_m3_router_tc(x16, w, None, False, full_fp32=True)[0]
    assert torch.equal(gate(x16, generation_only=True), tc)  # 9..16 generation rows: router_tc

    def stock_path(
        x,
    ):  # Match the unloaded gate's actual stock operator, including its compute policy.
        return torch.ops.trtllm.cublas_mm(
            x.to(torch.float32).contiguous(), w.t(), bias=None, out_dtype=torch.float32
        )

    x4 = x16[:4].contiguous()
    assert torch.equal(gate(x4, generation_only=True), stock_path(x4))  # 4 rows: the stock router
    assert torch.equal(
        gate(x16, generation_only=False), stock_path(x16)
    )  # context step: never router_tc
    monkeypatch.setattr(mm, "_ROUTER_TC_GEN", False)
    killed = mm.MiniMaxM3Gate(K, E, 4, 1.0).cuda()
    killed.weight.data.copy_(w)
    assert not killed._router_tc_gen
    assert torch.equal(
        killed(x16, generation_only=True), stock_path(x16)
    )  # kill switch: the stock router


@pytest.mark.parametrize("m", [1, 9, 16, 48])
@pytest.mark.parametrize("strided", [False, True])
def test_full_precision_router_retains_weight_residuals(m: int, strided: bool) -> None:
    if not torch.ops.trtllm.minimax_m3_router_tc_cluster16(K):
        pytest.skip("this device cannot co-schedule 16-CTA clusters")
    assert torch.ops.trtllm.minimax_m3_router_tc_cluster16(K, full_fp32=True)
    x, w = _inputs(m, 4000 + m)
    if strided:
        padded = torch.empty((m, K + 8), dtype=x.dtype, device=x.device)
        padded[:, :K].copy_(x)
        x = padded[:, :K]
    result = torch.ops.trtllm.minimax_m3_router_tc(x, w, None, False, full_fp32=True)[0]
    reference = x.double() @ w.double().t()
    assert (result.double() - reference).abs().max().item() <= 1e-5
    assert (result - _fp32_exact(x, w, full_fp32=True)).abs().max().item() <= 1e-5
    # A cancellation pair makes the discarded FP32 weight bits the entire signal.
    x = torch.ones_like(x)
    w = torch.zeros_like(w)
    w[:, 0] = 1.0 + 2.0**-11 + 2.0**-23
    w[:, 1] = -(1.0 + 2.0**-11)
    result = torch.ops.trtllm.minimax_m3_router_tc(x, w, None, False, full_fp32=True)[0]
    torch.testing.assert_close(result, torch.full_like(result, 2.0**-23), rtol=0, atol=0)


@pytest.mark.parametrize("sign", [-1, 1])
def test_full_precision_router_high_finite_weight(sign: int) -> None:
    if not torch.ops.trtllm.minimax_m3_router_tc_cluster16(K):
        pytest.skip("this device cannot co-schedule 16-CTA clusters")
    assert torch.ops.trtllm.minimax_m3_router_tc_cluster16(K, full_fp32=True)
    x = torch.zeros((16, K), device="cuda", dtype=torch.bfloat16)
    w = torch.zeros((E, K), device="cuda", dtype=torch.float32)
    x[:, 0] = 2.0**-64
    w[:, 0] = sign * torch.finfo(torch.float32).max
    result = torch.ops.trtllm.minimax_m3_router_tc(x, w, None, False, full_fp32=True)[0]
    reference = (x.double() @ w.double().t()).float()
    assert torch.isfinite(reference).all()
    torch.testing.assert_close(result, reference, rtol=0, atol=0)

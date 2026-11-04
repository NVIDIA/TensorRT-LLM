# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate a for the batch-invariant TF32 router GEMM at the M3 gate shape ([M, 6144] x [6144, 128]): rows bitwise
identical for every M (the breakable-prefill buckets included), run to run; error vs fp64 no worse than the stock
TF32 F.linear comparison; the actual loaded stock gate and complete routing have a separate test.
The split-K kernel (default) and the
single-pass kernel (kill TRTLLM_CTX_ROUTER_TF32_SPLITK=0) each meet all of it."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.models import modeling_minimaxm3
from tensorrt_llm._torch.moe.fused_moe import router_gemm_tf32 as rgt
from tensorrt_llm._torch.moe.fused_moe.router_gemm_tf32 import (
    prepare_router_weight_tf32,
    router_gemm_tf32,
)

H, E, TOP_K = 6144, 128, 4


@pytest.fixture(scope="module")
def data():
    g = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(16384, H, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(E, H, device="cuda", generator=g) / H**0.5).float().contiguous()
    bias = (0.05 * torch.randn(E, device="cuda", generator=g)).float()
    return x, w, prepare_router_weight_tf32(w), bias


def test_prepare_is_cvt_rna(data) -> None:
    _, w, wr, _ = data
    # TF32 keeps 10 mantissa bits: the low 13 are zero and |wr - w| <= half a TF32 ulp
    assert int((wr.view(torch.int32) & 0x1FFF).abs().max()) == 0
    ulp = torch.ldexp(torch.ones_like(w), torch.frexp(w)[1] - 11)
    assert bool(((wr - w).abs() <= ulp / 2).all())


def test_m_invariant_and_deterministic(data) -> None:
    x, _, wr, _ = data
    full = router_gemm_tf32(x, wr)
    capped = router_gemm_tf32(x[: rgt.SPLITK_MAX_M].contiguous(), wr)
    for _ in range(3):
        assert torch.equal(router_gemm_tf32(x, wr).view(torch.int32), full.view(torch.int32))
    for m in (
        0,
        1,
        2,
        13,
        32,
        48,
        64,
        127,
        128,
        129,
        256,
        512,
        1024,
        1160,
        2048,
        4096,
        7304,
        8192,
        12288,
        14336,
        16383,
    ):
        o = router_gemm_tf32(x[:m].contiguous(), wr)
        assert o.shape == (m, E)
        ref = (
            capped if m <= rgt.SPLITK_MAX_M else full
        )  # M-invariant within each side of the split-K cap
        assert torch.equal(o.view(torch.int32), ref[:m].view(torch.int32)), f"rows differ at M={m}"


def test_error_and_top4_vs_tf32_linear(data) -> None:
    x, w, wr, bias = data
    ref64 = x.double() @ w.double().t()
    prev = torch.backends.cuda.matmul.allow_tf32
    try:
        # Same-precision comparison; the loaded stock gate is tested separately.
        torch.backends.cuda.matmul.allow_tf32 = True
        stock = F.linear(x.float(), w)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    mine = router_gemm_tf32(x, wr)
    err_stock = float((stock.double() - ref64).abs().max())
    err_mine = float((mine.double() - ref64).abs().max())
    assert err_mine <= 1.05 * err_stock, (err_mine, err_stock)

    def top4(lg):
        return torch.topk(torch.sigmoid(lg) + bias, TOP_K, dim=1).indices.sort(dim=1).values

    flips = int((top4(mine) != top4(stock)).any(1).sum())
    assert flips <= x.shape[0] // 1000, flips  # near-ties only


def test_cuda_graph(data) -> None:
    x, _, wr, _ = data
    for m in (1, 512, 1024):  # CTX piecewise graph pieces
        xs = x[:m].clone()
        eager = router_gemm_tf32(xs, wr)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = router_gemm_tf32(xs, wr)
        xs.copy_(x[m : 2 * m])
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            out.view(torch.int32), router_gemm_tf32(x[m : 2 * m].contiguous(), wr).view(torch.int32)
        )
        assert not torch.equal(out, eager)


@pytest.mark.threadleak(enabled=False)  # torch.compile starts a tqdm monitor thread
def test_torch_compile(data) -> None:
    x, _, wr, _ = data
    compiled = torch.compile(
        lambda a, b: torch.ops.trtllm.router_gemm_tf32(a, b), fullgraph=True, dynamic=True
    )
    for m in (1, 7, 1024):
        assert torch.equal(
            compiled(x[:m], wr).view(torch.int32), router_gemm_tf32(x[:m], wr).view(torch.int32)
        )


@pytest.fixture(params=[True, False], ids=["splitk", "single_pass"])
def kernel_mode(request, monkeypatch):
    monkeypatch.setattr(rgt, "SPLITK", request.param)
    return request.param


def test_both_kernels_m_invariant_and_error(data, kernel_mode) -> None:
    x, w, wr, _ = data
    x = x[: rgt.SPLITK_MAX_M].contiguous()  # the split kernel's side of the cap
    full = router_gemm_tf32(x, wr)
    for m in (1, 32, 129, 512, 2048, 4096, rgt.SPLITK_MAX_M - 1):
        o = router_gemm_tf32(x[:m].contiguous(), wr)
        assert torch.equal(o.view(torch.int32), full[:m].view(torch.int32)), (kernel_mode, m)
    ref64 = x.double() @ w.double().t()
    prev = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        stock = F.linear(x.float(), w)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    assert float((full.double() - ref64).abs().max()) <= 1.05 * float(
        (stock.double() - ref64).abs().max()
    )


def test_kill_switch_is_the_single_pass_kernel(data, monkeypatch) -> None:
    x, _, wr, _ = data
    monkeypatch.setattr(rgt, "SPLITK", False)
    single = router_gemm_tf32(x[:2048].contiguous(), wr)
    monkeypatch.setattr(rgt, "SPLITK", True)
    split = router_gemm_tf32(x[:2048].contiguous(), wr)
    assert not torch.equal(
        split.view(torch.int32), single.view(torch.int32)
    )  # split-K engaged by default
    assert float((split - single).abs().max()) < 1e-3


def test_split_k_cap_keeps_the_single_pass_kernel_above(data, monkeypatch) -> None:
    x, _, wr, _ = data
    m = rgt.SPLITK_MAX_M + 1
    monkeypatch.setattr(rgt, "SPLITK", False)
    single = router_gemm_tf32(x[:m].contiguous(), wr)
    monkeypatch.setattr(rgt, "SPLITK", True)
    above = router_gemm_tf32(x[:m].contiguous(), wr)
    assert torch.equal(
        above.view(torch.int32), single.view(torch.int32)
    )  # bitwise to the kill above the cap


#: Rows of the prefill router calls the oracle covers: one-row and small chunks, the breakable-prefill buckets, and the
#: split-K cap.
ORACLE_ROWS = (1, 7, 32, 129, 512, 2048, 4096, 8192)


@pytest.mark.parametrize("m", ORACLE_ROWS)
def test_split_k_matches_the_single_pass_kernel_within_the_stated_bound(
    data, monkeypatch, m
) -> None:
    """The split-K kernel against the single-pass kernel (the kill) at the router shape [m, 6144] x [6144, 128].

    STATED TOLERANCE. Both kernels multiply the same TF32 operands exactly into fp32 and differ only in the order
    of the fp32 additions (K split into SPLIT_K slices whose partials are summed in slice order, vs one serial chain).
    For any summation order, an fp32 sum of K products is within gamma_K * sum_k |x_k w_k| of the exact sum, with
    gamma_K = K u / (1 - K u) and u = 2**-24; so the two kernels differ by at most
        |split - single| <= 2 * gamma_K * sum_k |x_k| |w_k|     (per logit, K = 6144),
    which this test asserts element-wise. It also asserts the routing outcome: the top-4 experts (sigmoid + bias)
    differ on at most m // 1000 rows (near-ties only), the stock router comparison's own allowance."""
    x, _, wr, bias = data
    x = x[:m].contiguous()
    monkeypatch.setattr(rgt, "SPLITK", False)
    single = router_gemm_tf32(x, wr)
    monkeypatch.setattr(rgt, "SPLITK", True)
    split = router_gemm_tf32(x, wr)
    k = x.shape[1]
    u = 2.0**-24
    gamma = k * u / (1.0 - k * u)
    bound = 2.0 * gamma * (x.double().abs() @ wr.double().abs().t())
    diff = (split.double() - single.double()).abs()
    print(
        f"ROUTERTF32 split vs single M={m}: max |d| {float(diff.max()):.3e}, "
        f"max d / bound {float((diff / bound).max()):.3e}"
    )
    assert bool((diff <= bound).all()), (
        f"M={m}: {int((diff > bound).sum())} logits outside the stated bound"
    )

    def top4(lg):
        return torch.topk(torch.sigmoid(lg) + bias, TOP_K, dim=1).indices.sort(dim=1).values

    flips = int((top4(split) != top4(single)).any(1).sum())
    assert flips <= m // 1000, (m, flips)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="SM107 router replay validation",
)
def test_gate_tf32_cuda_graph_replay_after_refit(monkeypatch) -> None:
    monkeypatch.setattr(modeling_minimaxm3, "_CTX_ROUTER_TF32_FIXED", True)
    monkeypatch.setattr(modeling_minimaxm3, "_ROUTER_TC_GEN", False)
    gate = modeling_minimaxm3.MiniMaxM3Gate(H, E, TOP_K, 1.0).cuda()
    weight = torch.randn(E, H, device="cuda") / H**0.5
    gate.load_weights([{"weight": weight}])
    gate.cache_tf32_weight()
    ptr = gate._weight_tf32.data_ptr()
    x = torch.randn(128, H, device="cuda").bfloat16()
    for _ in range(3):
        gate(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gate(x)
    before = captured.clone()
    gate.load_weights([{"weight": -weight}])
    gate.cache_tf32_weight()
    assert gate._weight_tf32.data_ptr() == ptr
    graph.replay()
    torch.testing.assert_close(captured, gate(x), rtol=0, atol=0)
    assert not torch.equal(captured, before)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="SM107 gate validation against the loaded stock router",
)
@pytest.mark.parametrize("rows", [48, 2048, 8192, 8193])
def test_gate_tf32_vs_loaded_split_weight_and_complete_routing(
    monkeypatch, data, rows: int
) -> None:
    """Compare the production gate and full routing against the actual loaded stock path.

    Keep the existing top-k flip budget (at most one per thousand rows), require
    every flip to lie within the measured score-perturbation tie band, and bound
    normalized logit and matching-expert routing-weight errors by 5e-4.
    This is a kernel/gate budget; model-level accuracy qualification is separate.
    """
    x, weight, _, bias = data
    x = x[:rows].contiguous()
    gate = modeling_minimaxm3.MiniMaxM3Gate(H, E, TOP_K, 1.0).cuda()
    gate.load_weights([{"weight": weight, "e_score_correction_bias": bias}])
    assert gate.split_weight is not None
    assert gate._weight_tf32 is None
    stock = gate(x)
    stock_ids, stock_weights = gate.routing_method.apply(stock)

    monkeypatch.setattr(modeling_minimaxm3, "_CTX_ROUTER_TF32_FIXED", True)
    gate.cache_tf32_weight()
    optimized = gate(x)
    optimized_ids, optimized_weights = gate.routing_method.apply(optimized)
    reference = x.double() @ weight.double().t()
    reference_scale = reference.abs().max().clamp_min(1e-30)
    normalized_logit_error = float((optimized.double() - reference).abs().max() / reference_scale)
    stock_logit_error = float((stock.double() - reference).abs().max() / reference_scale)
    assert normalized_logit_error <= 5e-4, (normalized_logit_error, stock_logit_error)

    stock_order = stock_ids.argsort(dim=1)
    optimized_order = optimized_ids.argsort(dim=1)
    matched = (stock_ids.gather(1, stock_order) == optimized_ids.gather(1, optimized_order)).all(1)
    flips = int((~matched).sum())
    assert flips <= rows // 1000, flips
    scores = torch.sigmoid(stock) + bias
    score_delta = (torch.sigmoid(optimized) - torch.sigmoid(stock)).abs().amax(dim=1)
    ranked = scores.topk(TOP_K + 1, dim=1).values
    clear_margin = ranked[:, TOP_K - 1] - ranked[:, TOP_K] > 2 * score_delta
    assert not bool(((~matched) & clear_margin).any()), "routing changed away from a near-tie"
    stock_aligned = stock_weights.gather(1, stock_order)[matched]
    optimized_aligned = optimized_weights.gather(1, optimized_order)[matched]
    weight_error = float((optimized_aligned - stock_aligned).abs().max())
    assert weight_error <= 5e-4, weight_error
    print(
        f"loaded-stock comparison: rows={rows}, stock_logit_error={stock_logit_error:.8g}, "
        f"tf32_logit_error={normalized_logit_error:.8g}, complete_routing_flips={flips}, "
        f"routing_weight_error={weight_error:.8g}"
    )

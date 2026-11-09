# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-tile three-term BF16 router: precision, loaded-stock routing and replay.

Rows are bitwise invariant within each split-K regime. The historical TF32 F.linear error baseline remains,
while routing compatibility is checked against the actual loaded three-term stock gate.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.models import modeling_minimaxm3
from tensorrt_llm._torch.moe.fused_moe import router_gemm as rgt
from tensorrt_llm._torch.moe.fused_moe.router_gemm import router_gemm

H, E, TOP_K = 6144, 128, 4
RouterData = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@pytest.fixture(scope="module")
def data() -> RouterData:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    g = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(16384, H, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(E, H, device="cuda", generator=g) / H**0.5).float().contiguous()
    bias = (0.05 * torch.randn(E, device="cuda", generator=g)).float()
    return x, w, modeling_minimaxm3.split_router_weight_bf16(w), bias


@pytest.mark.parametrize(
    "invalid",
    [
        "activation_dtype",
        "activation_layout",
        "weight_dtype",
        "weight_layout",
        "packed_width",
        "zero_k",
        "unaligned_k",
        "small_n",
        "large_n",
        "non_power_two_n",
    ],
)
def test_reject_invalid_operands(data: RouterData, invalid: str) -> None:
    # Small real CUDA operands reach the public custom-op checks without launching a kernel.
    x = torch.zeros((2, 32), device="cuda", dtype=torch.bfloat16)
    packed = torch.zeros((16, 96), device="cuda", dtype=torch.bfloat16)
    if invalid == "activation_dtype":
        x = x.float()
    elif invalid == "activation_layout":
        x = torch.zeros((32, 2), device="cuda", dtype=torch.bfloat16).t()
    elif invalid == "weight_dtype":
        packed = packed.float()
    elif invalid == "weight_layout":
        packed = torch.zeros((96, 16), device="cuda", dtype=torch.bfloat16).t()
    elif invalid == "packed_width":
        packed = packed[:, :-1].contiguous()
    elif invalid in ("zero_k", "unaligned_k"):
        k = 0 if invalid == "zero_k" else 33
        x = x.new_zeros((2, k))
        packed = packed.new_zeros((16, 3 * k))
    else:
        n = {"small_n": 8, "large_n": 512, "non_power_two_n": 24}[invalid]
        packed = packed.new_zeros((n, 96))
    with pytest.raises(ValueError, match="router"):
        router_gemm(x, packed)


def test_small_k_uses_single_pass_when_split_is_unaligned(data: RouterData, monkeypatch) -> None:
    generator = torch.Generator(device="cuda").manual_seed(732)
    x = torch.randn((129, 32), device="cuda", generator=generator).bfloat16()
    weight = torch.randn((16, 32), device="cuda", generator=generator)
    packed = modeling_minimaxm3.split_router_weight_bf16(weight)
    monkeypatch.setattr(rgt, "SPLITK", True)
    # K=32 cannot be divided into eight aligned K tiles.
    monkeypatch.setattr(rgt, "_split_k", lambda *args: pytest.fail("unaligned split-K launched"))
    actual = router_gemm(x, packed)
    reference = x.double() @ weight.double().t()
    torch.testing.assert_close(actual.double(), reference, rtol=2e-5, atol=2e-5)
    monkeypatch.setattr(rgt, "SPLITK", False)
    torch.testing.assert_close(router_gemm(x, packed), actual, rtol=0, atol=0)


def test_weight_terms_preserve_sampled_fp32_weights(data: RouterData) -> None:
    _, weight, packed, _ = data
    reconstructed = packed.view(E, rgt.WEIGHT_TERMS, H).double().sum(dim=1)
    torch.testing.assert_close(reconstructed, weight.double(), rtol=0, atol=0)


def test_m_invariant_and_deterministic(data) -> None:
    x, _, wr, _ = data
    full = router_gemm(x, wr)
    torch.cuda.synchronize()  # Attribute asynchronous faults to the single-pass launch.
    capped = router_gemm(x[: rgt.SPLITK_MAX_M].contiguous(), wr)
    torch.cuda.synchronize()  # Attribute partial/reduction faults before subsequent launches.
    for _ in range(3):
        assert torch.equal(router_gemm(x, wr).view(torch.int32), full.view(torch.int32))
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
        o = router_gemm(x[:m].contiguous(), wr)
        assert o.shape == (m, E)
        ref = (
            capped if m <= rgt.SPLITK_MAX_M else full
        )  # M-invariant within each side of the split-K cap
        assert torch.equal(o.view(torch.int32), ref[:m].view(torch.int32)), f"rows differ at M={m}"


def test_error_vs_legacy_tf32_and_routing_vs_loaded_stock(data: RouterData) -> None:
    x, weight, packed, bias = data
    reference = x.double() @ weight.double().t()
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        legacy = F.linear(x.float(), weight)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    optimized = router_gemm(x, packed)
    legacy_error = float((legacy.double() - reference).abs().max())
    optimized_error = float((optimized.double() - reference).abs().max())
    assert optimized_error <= 1.05 * legacy_error, (optimized_error, legacy_error)
    gate = _loaded_stock_gate(weight, bias)
    _assert_loaded_stock_routing(gate, x, weight, bias, gate(x), optimized)


def test_cuda_graph(data) -> None:
    x, _, wr, _ = data
    for m in (1, 512, 1024):  # CTX piecewise graph pieces
        xs = x[:m].clone()
        eager = router_gemm(xs, wr)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = router_gemm(xs, wr)
        xs.copy_(x[m : 2 * m])
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            out.view(torch.int32), router_gemm(x[m : 2 * m].contiguous(), wr).view(torch.int32)
        )
        assert not torch.equal(out, eager)


@pytest.mark.threadleak(enabled=False)  # torch.compile starts a tqdm monitor thread
def test_torch_compile(data) -> None:
    x, _, wr, _ = data
    compiled = torch.compile(
        lambda a, b: torch.ops.trtllm.router_gemm(a, b), fullgraph=True, dynamic=True
    )
    for m in (1, 7, 1024):
        assert torch.equal(
            compiled(x[:m], wr).view(torch.int32), router_gemm(x[:m], wr).view(torch.int32)
        )


@pytest.fixture(params=[True, False], ids=["splitk", "single_pass"])
def kernel_mode(request, monkeypatch):
    monkeypatch.setattr(rgt, "SPLITK", request.param)
    return request.param


def test_both_kernels_m_invariant_and_error(data, kernel_mode) -> None:
    x, w, wr, _ = data
    x = x[: rgt.SPLITK_MAX_M].contiguous()  # the split kernel's side of the cap
    full = router_gemm(x, wr)
    for m in (1, 32, 129, 512, 2048, 4096, rgt.SPLITK_MAX_M - 1):
        o = router_gemm(x[:m].contiguous(), wr)
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
    single = router_gemm(x[:2048].contiguous(), wr)
    monkeypatch.setattr(rgt, "SPLITK", True)
    split = router_gemm(x[:2048].contiguous(), wr)
    assert not torch.equal(
        split.view(torch.int32), single.view(torch.int32)
    )  # split-K engaged by default
    assert float((split - single).abs().max()) < 1e-3


def test_split_k_cap_keeps_the_single_pass_kernel_above(data, monkeypatch) -> None:
    x, _, wr, _ = data
    m = rgt.SPLITK_MAX_M + 1
    monkeypatch.setattr(rgt, "SPLITK", False)
    single = router_gemm(x[:m].contiguous(), wr)
    monkeypatch.setattr(rgt, "SPLITK", True)
    above = router_gemm(x[:m].contiguous(), wr)
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

    STATED TOLERANCE. Both kernels multiply the same three BF16 terms exactly into FP32 and differ only in
    FP32 addition order. With L = 3*K + SPLIT_K (including the partial reduction), u = 2**-24 and
    gamma_L = L*u/(1-L*u), the conservative forward-error bound for the two summation orders is
        |split - single| <= 2 * gamma_L * sum_k,term |x_k| |w_term,k|,
    which this test asserts element-wise. It also asserts the routing outcome: the top-4 experts (sigmoid + bias)
    differ on at most m // 1000 rows (near-ties only), the stock router comparison's own allowance."""
    x, _, wr, bias = data
    x = x[:m].contiguous()
    monkeypatch.setattr(rgt, "SPLITK", False)
    single = router_gemm(x, wr)
    monkeypatch.setattr(rgt, "SPLITK", True)
    split = router_gemm(x, wr)
    k = x.shape[1]
    u = 2.0**-24
    additions = rgt.WEIGHT_TERMS * k + rgt.SPLIT_K
    gamma = additions * u / (1.0 - additions * u)
    absolute_terms = wr.view(E, rgt.WEIGHT_TERMS, k).double().abs().sum(dim=1)
    bound = 2.0 * gamma * (x.double().abs() @ absolute_terms.t())
    diff = (split.double() - single.double()).abs()
    print(
        f"ROUTER BF16 split vs single M={m}: max |d| {float(diff.max()):.3e}, "
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
def test_gate_fixed_router_cuda_graph_replay_after_refit(monkeypatch) -> None:
    monkeypatch.setattr(modeling_minimaxm3, "_CTX_ROUTER_TF32_FIXED", True)
    monkeypatch.setattr(modeling_minimaxm3, "_ROUTER_TC_GEN", False)
    gate = modeling_minimaxm3.MiniMaxM3Gate(H, E, TOP_K, 1.0).cuda()
    weight = torch.randn(E, H, device="cuda") / H**0.5
    gate.load_weights([{"weight": weight}])
    gate.cache_fixed_router_weight()
    ptr = gate.split_weight.data_ptr()
    x = torch.randn(128, H, device="cuda").bfloat16()
    for _ in range(3):
        gate(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gate(x)
    before = captured.clone()
    gate.load_weights([{"weight": -weight}])
    assert gate.split_weight.data_ptr() == ptr
    graph.replay()
    torch.testing.assert_close(captured, gate(x), rtol=0, atol=0)
    assert not torch.equal(captured, before)


def _loaded_stock_gate(
    weight: torch.Tensor, bias: torch.Tensor
) -> modeling_minimaxm3.MiniMaxM3Gate:
    gate = modeling_minimaxm3.MiniMaxM3Gate(H, E, TOP_K, 1.0).to(weight.device)
    gate.load_weights([{"weight": weight, "e_score_correction_bias": bias}])
    assert gate.split_weight is not None and not gate._use_fixed_router
    return gate


def _assert_loaded_stock_routing(
    gate: modeling_minimaxm3.MiniMaxM3Gate,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stock: torch.Tensor,
    optimized: torch.Tensor,
) -> None:
    """Keep the loaded-stock logit, complete-routing, near-tie and routing-weight budgets unchanged."""
    rows = x.shape[0]
    stock_ids, stock_weights = gate.routing_method.apply(stock)
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
        f"fixed_logit_error={normalized_logit_error:.8g}, complete_routing_flips={flips}, "
        f"routing_weight_error={weight_error:.8g}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="SM107 gate validation against the loaded stock router",
)
@pytest.mark.usefixtures("kernel_mode")
@pytest.mark.parametrize("rows", [48, 2048, 8192, 8193])
def test_gate_fixed_router_vs_loaded_split_weight_and_complete_routing(
    monkeypatch: pytest.MonkeyPatch, data: RouterData, rows: int
) -> None:
    """Both kernel modes meet the same actual loaded-stock routing budget."""
    x, weight, _, bias = data
    x = x[:rows].contiguous()
    gate = _loaded_stock_gate(weight, bias)
    stock = gate(x)
    monkeypatch.setattr(modeling_minimaxm3, "_CTX_ROUTER_TF32_FIXED", True)
    gate.cache_fixed_router_weight()
    assert gate._use_fixed_router
    _assert_loaded_stock_routing(gate, x, weight, bias, stock, gate(x))


@pytest.mark.usefixtures("kernel_mode")
@pytest.mark.parametrize("rows", [1, 129])
def test_router_retains_the_third_weight_term_after_cancellation(
    data: RouterData, rows: int
) -> None:
    """The first two terms cancel; discarding the third would incorrectly return zero."""
    device = data[0].device
    coefficient = torch.ldexp(
        torch.ones(E, dtype=torch.float32, device=device), torch.arange(E, device=device) % 8 - 4
    )
    base = 1.0 + 2.0**-7 + 2.0**-15
    weight = torch.zeros((E, H), dtype=torch.float32, device=device)
    weight[:, 0] = (base + 2.0**-23) * coefficient
    weight[:, 1] = -base * coefficient
    packed = modeling_minimaxm3.split_router_weight_bf16(weight)
    terms = packed.view(E, rgt.WEIGHT_TERMS, H)
    assert bool((terms[:, :2].double().sum(dim=(1, 2)) == 0).all())
    x = torch.ones((rows, H), dtype=torch.bfloat16, device=device)
    reference = x.double() @ weight.double().t()
    assert bool((reference != 0).all())
    optimized = router_gemm(x, packed)
    assert torch.equal(optimized.view(torch.int32), reference.float().view(torch.int32))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="SM107 target context router validation",
)
@pytest.mark.parametrize("rows", [2048, 8193])
def test_tp2_context_dispatch_matches_loaded_stock_routing(
    monkeypatch: pytest.MonkeyPatch, data: RouterData, rows: int
) -> None:
    """The target's real context dispatcher uses the same precision-preserving kernel."""
    # Importing this optional native target is confined to the matching architecture test.
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (  # noqa: E501
        modeling,
    )

    x, weight, packed, bias = data
    x = x[:rows].contiguous()
    core = modeling.MiniMaxM3Tp2ModelingV2.__new__(modeling.MiniMaxM3Tp2ModelingV2)
    torch.nn.Module.__init__(core)
    core.w = {"l0_router": weight}
    core.router_w_split = {0: packed}
    monkeypatch.setattr(
        modeling, "_router_logits", lambda *args: pytest.fail("context router fell back")
    )
    gate = _loaded_stock_gate(weight, bias)
    optimized = modeling._context_router_logits(core, 0, x)
    _assert_loaded_stock_routing(gate, x, weight, bias, gate(x), optimized)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="SM107 generation TC router validation",
)
def test_generation_tc_router_matches_loaded_stock_complete_routing(
    monkeypatch: pytest.MonkeyPatch, data: RouterData
) -> None:
    """Evaluate native generation tiles against loaded stock, including complete routing."""
    monkeypatch.setattr(modeling_minimaxm3, "_CTX_ROUTER_TF32_FIXED", False)
    monkeypatch.setattr(modeling_minimaxm3, "_ROUTER_TC_GEN", True)
    x, weight, _, bias = data
    x = x[:8192].contiguous()
    gate = _loaded_stock_gate(weight, bias)
    stock = gate(x)
    gate.cache_fixed_router_weight()
    assert not gate._use_fixed_router
    if not gate._router_tc_gen:
        pytest.skip("device cannot co-schedule the optional 16-CTA generation router")
    assert gate.router_tc_applies(x[:16], generation_only=True)
    optimized = torch.cat([gate(tile, generation_only=True) for tile in x.split(16)])
    _assert_loaded_stock_routing(gate, x, weight, bias, stock, optimized)

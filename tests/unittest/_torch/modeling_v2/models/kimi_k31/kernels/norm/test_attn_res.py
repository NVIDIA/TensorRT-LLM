# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's PDL attn_res kernel against the engine's C++ attn_res ops.

The engine ops are what the general target runs today. The kernel's claims against them: the
updated prefix sum is bit-identical, and the normalized output differs by at most one BF16
rounding step in a small fraction of the elements (the statistics are summed in another order).
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the attn_res kernel is a CuTe DSL kernel", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.norm import (  # noqa: E402
    attn_res,
)

assert torch.cuda.is_available(), "the attn_res kernel requires a CUDA device"
if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
    pytest.skip("the attn_res kernel runs on SM100 / SM103 / SM107", allow_module_level=True)

H = attn_res.HIDDEN_SIZE
RMS_EPS, OUT_EPS = 1e-6, 1e-5


def _inputs(m: int, n: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).to(torch.bfloat16)

    layer = r(m, H, scale=4.0)
    addend = r(m, H)
    block = r(n - 1, m, H, scale=4.0)
    res_w, rms_w, out_w = r(H, scale=0.05), (1 + r(H, scale=0.1).float()).bfloat16(), r(H)
    return layer, addend, block, res_w, rms_w, out_w


def _engine_rmsnorm(layer, block, res_w, rms_w, out_w):
    m, k = layer.shape[0], block.shape[0]
    out = torch.ops.trtllm.attn_res_rmsnorm_fwd(
        layer.reshape(m, 1, H), block.reshape(k, m, 1, H), res_w, rms_w, out_w, RMS_EPS, OUT_EPS
    )
    return out.reshape(m, H)


def _engine_add_rmsnorm(prefix, addend, block, res_w, rms_w, out_w):
    m, k = prefix.shape[0], block.shape[0]
    updated, out = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
        prefix.reshape(m, 1, H),
        addend.reshape(m, 1, H),
        block.reshape(k, m, 1, H),
        res_w,
        rms_w,
        out_w,
        RMS_EPS,
        OUT_EPS,
    )
    return updated.reshape(m, H), out.reshape(m, H)


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.floor(torch.log2(x.abs().clamp_min(2.0**-126))) - 7)


def _one_rounding_step(
    got: torch.Tensor, want: torch.Tensor, out_w: torch.Tensor, max_fraction: float
) -> float:
    """``out = bf16(normed * out_w)``, where the mixture and the normalized row are each rounded to
    BF16 and may land one step apart. The mixture is a softmax-weighted sum of candidates of unit
    RMS (after the norm), so near a cancellation its error is a step at the row's scale (1), not
    at the element's: each element stays within ``2 ulp(max(1, |normed|)) * |out_w| + ulp(out)``
    of the engine's, and few elements differ at all."""
    g, w = got.float(), want.float()
    weight = out_w.float().abs()
    normed = (w.abs() / weight.clamp_min(2.0**-20)).clamp_min(1.0)
    tol = 2 * _bf16_ulp(normed) * weight + _bf16_ulp(torch.maximum(g.abs(), w.abs()))
    diff = (g - w).abs()
    assert torch.all(diff <= tol * 1.0001), f"max diff {diff.max().item()} beyond one step"
    fraction = (diff > 0).float().mean().item()
    assert fraction <= max_fraction, f"{fraction:.4%} of the elements differ"
    return fraction


# (rows, candidates): the 8-CTA clusters (<= 16 rows, >= 4 candidates), the 4-CTA ones, the
# generation servers' row limits (192 attention TP, 256 attention DP) and the kernel's own (1024).
SHAPES = ((1, 2), (1, 12), (8, 7), (16, 4), (17, 4), (64, 3), (192, 12), (256, 9), (1024, 5))


@pytest.mark.parametrize("m,n", SHAPES)
def test_rmsnorm_vs_engine(m: int, n: int) -> None:
    layer, _, block, res_w, rms_w, out_w = _inputs(m, n, seed=m * 16 + n)
    got = attn_res.k31_attn_res_rmsnorm(layer, block, res_w, rms_w, out_w, RMS_EPS, OUT_EPS)
    fraction = _one_rounding_step(
        got, _engine_rmsnorm(layer, block, res_w, rms_w, out_w), out_w, 0.005
    )
    print(f"rmsnorm m={m} n={n}: {fraction:.4%} of the outputs one BF16 step from the engine op")


@pytest.mark.parametrize("m,n", SHAPES)
def test_add_rmsnorm_vs_engine(m: int, n: int) -> None:
    prefix, addend, block, res_w, rms_w, out_w = _inputs(m, n, seed=m * 16 + n + 7)
    updated, out = attn_res.k31_attn_res_add_rmsnorm(
        prefix, addend, block, res_w, rms_w, out_w, RMS_EPS, OUT_EPS
    )
    ref_updated, ref_out = _engine_add_rmsnorm(prefix, addend, block, res_w, rms_w, out_w)
    assert torch.equal(updated.view(torch.int16), ref_updated.view(torch.int16)), (
        "the updated prefix sum must be bit-identical to the engine op's"
    )
    fraction = _one_rounding_step(out, ref_out, out_w, 0.005)
    print(
        f"add_rmsnorm m={m} n={n}: {fraction:.4%} of the outputs one BF16 step from the engine op"
    )


@pytest.mark.parametrize("m,n", ((1, 2), (6, 5), (8, 9), (24, 9), (72, 5), (99, 9), (224, 9)))
def test_moe_add_rmsnorm_vs_torch_adds(m: int, n: int) -> None:
    """The two-addend variant replaces ``prefix + (routed + shared)`` (two torch adds) and the
    pre-attention mix that reads the sum: the update is bitwise, the output as the one-add op's."""
    prefix, routed, block, res_w, rms_w, out_w = _inputs(m, n, seed=m * 16 + n + 11)
    shared = (routed.float() * 0.37 + 0.01).to(torch.bfloat16)
    # Values whose FP32 sums are subnormal: a flush-to-zero add would lose them.
    routed[0, :8] = torch.tensor([1e-39, -1e-39, 3e-39, 0.0, 1e-38, -2e-39, 5e-39, 7e-39]).to(
        routed
    )
    shared[0, :8] = torch.tensor([1e-39, 2e-39, -1e-39, 4e-39, 0.0, 1e-39, -5e-39, 1e-39]).to(
        shared
    )
    prefix[0, :8] = 0
    updated, out = attn_res.k31_attn_res_moe_add_rmsnorm(
        prefix, routed, shared, block, res_w, rms_w, out_w, RMS_EPS, OUT_EPS
    )
    want = prefix + (routed + shared)
    assert torch.equal(updated.view(torch.int16), want.view(torch.int16))
    _, ref_out = _engine_add_rmsnorm(prefix, routed + shared, block, res_w, rms_w, out_w)
    _one_rounding_step(out, ref_out, out_w, 0.005)


def _filled_mailbox(moe: torch.Tensor, capacity: int) -> torch.Tensor:
    """A tail mailbox whose first rows hold ``moe`` (the tail stores +0 for -0) and the rest empty."""
    mailbox = torch.full((1, capacity, H), -0.0, dtype=torch.bfloat16, device="cuda")
    rows = moe.clone()
    rows.view(torch.int16)[rows.view(torch.int16) == -0x8000] = 0
    mailbox[0, : moe.shape[0]] = rows
    return mailbox


@pytest.mark.parametrize("m,n", ((1, 2), (6, 5), (8, 9), (24, 9), (72, 5), (192, 9)))
def test_mailbox_add_rmsnorm(m: int, n: int) -> None:
    """The mailbox variant polls the fused MoE tail's output, adds it to the residual (the bits of
    the torch add) and re-arms exactly the rows it read."""
    prefix, moe, block, res_w, rms_w, out_w = _inputs(m, n, seed=m * 16 + n + 13)
    capacity = 192
    mailbox = _filled_mailbox(moe, capacity)
    updated, out = attn_res.k31_attn_res_mailbox_add_rmsnorm(
        prefix, mailbox, block, res_w, rms_w, out_w, RMS_EPS, OUT_EPS
    )
    torch.cuda.synchronize()
    assert torch.equal(updated.view(torch.int16), (prefix + moe).view(torch.int16))
    _, ref_out = _engine_add_rmsnorm(prefix, moe, block, res_w, rms_w, out_w)
    _one_rounding_step(out, ref_out, out_w, 0.005)
    assert bool((mailbox.view(torch.int16) == -0x8000).all()), "every row must be re-armed"


def test_cuda_graph_replay() -> None:
    m, n = 96, 6
    layer, _, block, res_w, rms_w, out_w = _inputs(m, n, seed=1)
    args = (layer, block, res_w, rms_w, out_w, RMS_EPS, OUT_EPS)
    attn_res.k31_attn_res_rmsnorm(*args)  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = attn_res.k31_attn_res_rmsnorm(*args)
    for seed in (2, 3):
        new_layer, _, new_block, *_ = _inputs(m, n, seed=seed)
        layer.copy_(new_layer)
        block.copy_(new_block)
        graph.replay()
        torch.cuda.synchronize()
        want = attn_res.k31_attn_res_rmsnorm(*args)
        assert torch.equal(out.view(torch.int16), want.view(torch.int16))


def _run_all_variants(m: int, n: int, seed: int):
    """rmsnorm, add, MoE add and mailbox outputs of the vectorized kernel for one shape."""
    layer, addend, block, res_w, rms_w, out_w = _inputs(m, n, seed)
    w = (res_w, rms_w, out_w, RMS_EPS, OUT_EPS)
    outs = [attn_res.k31_attn_res_rmsnorm(layer, block, *w)]
    outs += attn_res.k31_attn_res_add_rmsnorm(layer, addend, block, *w)
    outs += attn_res.k31_attn_res_moe_add_rmsnorm(layer, addend, layer, block, *w)
    outs += attn_res.k31_attn_res_mailbox_add_rmsnorm(layer, _filled_mailbox(addend, m), block, *w)
    torch.cuda.synchronize()
    return outs


@pytest.mark.parametrize(
    "m,n", ((65, 5), (72, 9), (100, 6), (120, 7), (140, 9), (185, 8), (192, 12))
)
def test_register_cap_keeps_the_bits(m: int, n: int, monkeypatch) -> None:
    """The register-capped build of large-row calls (``reg_cap``, the attention-TP targets' 65 rows) gives the
    uncapped build's bits."""
    monkeypatch.setattr(attn_res, "REG_CAP_MIN_TOKENS", 65)
    assert attn_res.reg_cap(m, n, attn_res.fast_cluster_size(m, n), False) > 0
    capped = _run_all_variants(m, n, seed=m * 32 + n)
    monkeypatch.setattr(attn_res, "REG_CAP_MIN_TOKENS", 0)
    uncapped = _run_all_variants(m, n, seed=m * 32 + n)
    for got, want in zip(capped, uncapped):
        assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_lamport_helpers_are_the_family_module() -> None:
    """The Lamport variants run the family's all-reduce helpers (kernels/comm/lamport_ar)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.comm import (
        lamport_ar,
    )

    assert attn_res._lamport_ops() is lamport_ar

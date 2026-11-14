# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ``run_triton_fp8_block_scale_moe`` and ``moe_align_block_size``.

Run as a pytest:
    pytest tests/unittest/_torch/moe/fused_moe/test_triton_fp8_block_scale.py -v
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from _torch.helpers import calc_diff, per_block_cast_to_fp8_e8m0

import tensorrt_llm._torch.moe.fused_moe.cutlass.fp8_block_scales_triton as triton_impl
import tensorrt_llm._torch.moe.fused_moe.fused_moe_triton_fp8_block_scale as triton_moe
from tensorrt_llm._torch.moe.fused_moe.fused_moe_triton_fp8_block_scale import (
    _BLOCK_SIZE_M,
    _BLOCK_SIZE_M_LARGE,
    _block_size_m,
    moe_align_block_size,
    run_triton_fp8_block_scale_moe,
)
from tensorrt_llm._torch.moe.fused_moe.interface import ActivationType
from tensorrt_llm._utils import is_sm_120f

skip_unsupported = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_120f(),
    reason="Requires CUDA SM120 for the Triton FP8 block-scale MoE path",
)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def _run_moe_bf16_reference(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w3_w1: torch.Tensor,
    w3_w1_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    activation_type: int,
    output_dtype: torch.dtype | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Dequantize FP8 weights and run the MoE in BF16."""
    num_tokens, hidden = x.shape
    _, gate_up_size, _ = w3_w1.shape
    intermediate = gate_up_size // 2
    top_k = token_selected_experts.shape[1]
    device = x.device

    tok_exp = token_selected_experts.long()
    tok_wt = token_final_scales

    def dequant_weight(w_fp8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        num_experts, out_features, in_features = w_fp8.shape
        num_n_blocks, num_k_blocks = scales.shape[1], scales.shape[2]
        out = torch.zeros(
            num_experts,
            out_features,
            in_features,
            dtype=torch.bfloat16,
            device=device,
        )
        for expert_idx in range(num_experts):
            for n_block in range(num_n_blocks):
                for k_block in range(num_k_blocks):
                    n_slice = slice(n_block * 128, (n_block + 1) * 128)
                    k_slice = slice(k_block * 128, (k_block + 1) * 128)
                    block = w_fp8[expert_idx, n_slice, k_slice].float()
                    out[expert_idx, n_slice, k_slice] = (
                        block * scales[expert_idx, n_block, k_block]
                    ).bfloat16()
        return out

    w31_bf16 = dequant_weight(w3_w1, w3_w1_scales)
    w2_bf16 = dequant_weight(w2, w2_scales)

    output = torch.zeros(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    for t in range(num_tokens):
        acc = torch.zeros(hidden, dtype=torch.float32, device=device)
        for k in range(top_k):
            e = int(tok_exp[t, k])
            gate_up = x[t].float() @ w31_bf16[e].float().T
            u, g = gate_up[:intermediate], gate_up[intermediate:]
            act_int = int(activation_type)
            if swiglu_limit is not None:
                # The CUTLASS ``SwigluBiasAdaptor`` order: cap the SiLU input
                # from above, the linear half symmetrically.
                g = g.clamp(max=swiglu_limit)
                u = u.clamp(-swiglu_limit, swiglu_limit)
            if act_int in (int(ActivationType.Swiglu), int(ActivationType.Silu)):
                ic2 = F.silu(g) * u
            else:
                ic2 = F.gelu(g) * u
            acc += (ic2 @ w2_bf16[e].float().T) * float(tok_wt[t, k])
        output[t] = acc.bfloat16()

    if output_dtype is not None and output.dtype != output_dtype:
        output = output.to(output_dtype)
    return output


@dataclass(frozen=True)
class MoeShape:
    name: str
    num_tokens: int
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int


# Laguna XS.2 routed-MoE dimensions. The shared expert is outside this kernel.
LAGUNA_HIDDEN = 2048
LAGUNA_INTER = 512
LAGUNA_NUM_EXPERTS = 256
LAGUNA_TOP_K = 8

SHAPES = [
    MoeShape("tiny", num_tokens=64, hidden=256, intermediate=512, num_experts=8, top_k=2),
    MoeShape("tiny_topk1", num_tokens=64, hidden=256, intermediate=512, num_experts=8, top_k=1),
    # Enough rows per expert for the prefill block size
    MoeShape("tiny_prefill", num_tokens=320, hidden=256, intermediate=512, num_experts=8, top_k=2),
    # The same, through the unfused second-GEMM store
    MoeShape(
        "tiny_prefill_topk1", num_tokens=640, hidden=256, intermediate=512, num_experts=8, top_k=1
    ),
    # DeepSeek-V3-Lite routes over 72 experts, which is not a power of 2.
    MoeShape("experts_72", num_tokens=64, hidden=256, intermediate=512, num_experts=72, top_k=6),
    MoeShape(
        "laguna_corr",
        num_tokens=64,
        hidden=LAGUNA_HIDDEN,
        intermediate=LAGUNA_INTER,
        num_experts=LAGUNA_NUM_EXPERTS,
        top_k=LAGUNA_TOP_K,
    ),
    MoeShape(
        "laguna_bench",
        num_tokens=4096,
        hidden=LAGUNA_HIDDEN,
        intermediate=LAGUNA_INTER,
        num_experts=LAGUNA_NUM_EXPERTS,
        top_k=LAGUNA_TOP_K,
    ),
]

# The BF16 reference is intentionally simple and slow, so keep the large shape
# available for manual experiments but out of the correctness parametrization.
CORRECTNESS_SHAPES = [s for s in SHAPES if s.name != "laguna_bench"]

# Global shapes split over EP_SIZE ranks: 80 and 16 rows per local expert slot,
# so both tile heights.
EP_SIZE = 2
EP_SHAPES = [
    MoeShape("ep2_prefill", num_tokens=320, hidden=256, intermediate=512, num_experts=16, top_k=2),
    MoeShape("ep2_decode", num_tokens=64, hidden=256, intermediate=512, num_experts=16, top_k=2),
]
SHAPES_BY_NAME = {s.name: s for s in SHAPES + EP_SHAPES}


def _make_inputs(
    shape: MoeShape,
    *,
    device: str = "cuda",
    seed: int = 0,
    activation_type: int = int(ActivationType.Swiglu),
    swiglu_limit: float | None = None,
) -> dict[str, torch.Tensor | int | float | None]:
    g = torch.Generator(device=device).manual_seed(seed)
    num_tokens = shape.num_tokens
    hidden = shape.hidden
    inter = shape.intermediate
    num_experts = shape.num_experts
    top_k = shape.top_k

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device, generator=g) * 0.5

    gate_logits = torch.randn((num_tokens, num_experts), device=device, generator=g)
    topk_vals, topk_ids = gate_logits.topk(top_k, dim=-1)
    token_selected_experts = topk_ids.to(torch.int64)
    token_final_scales = torch.softmax(topk_vals.float(), dim=-1)

    w3_w1_bf16 = (
        torch.randn(
            (num_experts, 2 * inter, hidden), dtype=torch.bfloat16, device=device, generator=g
        )
        * 0.05
    )
    w2_bf16 = (
        torch.randn((num_experts, hidden, inter), dtype=torch.bfloat16, device=device, generator=g)
        * 0.05
    )
    w3_w1_fp8, w3_w1_sf = per_block_cast_to_fp8_e8m0(w3_w1_bf16)
    w2_fp8, w2_sf = per_block_cast_to_fp8_e8m0(w2_bf16)

    return dict(
        x=x,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        w3_w1=w3_w1_fp8,
        w3_w1_scales=w3_w1_sf,
        w2=w2_fp8,
        w2_scales=w2_sf,
        activation_type=activation_type,
        swiglu_limit=swiglu_limit,
    )


def _local_slot_inputs(
    inputs: dict[str, torch.Tensor | int | float | None], ep_size: int, ep_rank: int
) -> dict[str, torch.Tensor | int | float | None]:
    """What one EP rank's ``TrtllmTritonFp8BlockScalesImpl.run_moe`` passes: the
    weights of its experts only, and every pair of another rank's expert
    clamped onto an edge slot with a zero scale (``_local_slot_ids``)."""
    num_local_experts = inputs["w3_w1"].shape[0] // ep_size
    slot_start = ep_rank * num_local_experts
    offsets = inputs["token_selected_experts"] - slot_start
    local_ids = offsets.clamp(0, num_local_experts - 1)
    local = slice(slot_start, slot_start + num_local_experts)
    return {
        **inputs,
        "token_selected_experts": local_ids,
        "token_final_scales": inputs["token_final_scales"] * (local_ids == offsets),
        "w3_w1": inputs["w3_w1"][local],
        "w3_w1_scales": inputs["w3_w1_scales"][local],
        "w2": inputs["w2"][local],
        "w2_scales": inputs["w2_scales"][local],
    }


def _assert_matches_reference(out: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    """Cosine difference and norms of the whole output and of every token, so
    that one wrong, missing or wrongly scaled row cannot hide in the aggregate.
    The cosine difference barely sees scale: 10% too large is 4.5e-3."""
    diff = calc_diff(out, ref).item()
    assert diff < 1e-2, f"shape={name} cosine-diff={diff:.4e} exceeds 1e-2"
    norm_ratio = (out.double().norm() / ref.double().norm()).item()
    assert abs(norm_ratio - 1) < 1e-2, (
        f"shape={name} norm ratio {norm_ratio:.4f} is off by 1% or more"
    )

    x, y = out.double(), ref.double()
    denominator = (x * x + y * y).sum(dim=-1)
    # A row both sides leave at zero (no pair of this rank's experts) matches.
    row_diff = torch.where(
        denominator > 0, 1 - 2 * (x * y).sum(dim=-1) / denominator.clamp_min(1e-300), 0.0
    )
    worst = row_diff.argmax().item()
    worst_diff = row_diff[worst].item()
    assert worst_diff < 1e-2, (
        f"shape={name} token {worst} cosine-diff={worst_diff:.4e} exceeds 1e-2"
    )

    # Rows the reference leaves near zero are held to a bound relative to the
    # largest row instead.
    out_norms, ref_norms = x.norm(dim=-1), y.norm(dim=-1)
    excess = (out_norms - ref_norms).abs() - (5e-2 * ref_norms + 1e-2 * ref_norms.max())
    worst = excess.argmax().item()
    assert excess[worst] <= 0, (
        f"shape={name} token {worst} norm {out_norms[worst]:.4e}, reference {ref_norms[worst]:.4e}"
    )


ACTIVATIONS = [
    pytest.param(int(ActivationType.Swiglu), id="swiglu"),
    pytest.param(int(ActivationType.Geglu), id="geglu"),
]


@skip_unsupported
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=lambda s: s.name)
@pytest.mark.parametrize("activation_type", ACTIVATIONS)
def test_fp8_matches_bf16_reference(shape: MoeShape, activation_type: int) -> None:
    inputs = _make_inputs(shape, activation_type=activation_type)

    out_fp8 = run_triton_fp8_block_scale_moe(**inputs)
    out_ref = _run_moe_bf16_reference(**inputs)

    assert out_fp8.shape == out_ref.shape == (shape.num_tokens, shape.hidden)
    assert out_fp8.dtype == torch.bfloat16

    _assert_matches_reference(out_fp8, out_ref, shape.name)


@skip_unsupported
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=lambda s: s.name)
def test_swiglu_clamp_matches_bf16_reference(shape: MoeShape) -> None:
    """A SwiGLU clamp reaches the gated-activation kernel and bites.

    The limit is chosen below the FC1 output range of ``_make_inputs``, so the
    unclamped output differs -- a dropped clamp would not just be inaccurate,
    it would be invisible without that check.
    """
    inputs = _make_inputs(shape, swiglu_limit=0.1)

    out_fp8 = run_triton_fp8_block_scale_moe(**inputs)
    out_ref = _run_moe_bf16_reference(**inputs)
    out_unclamped = run_triton_fp8_block_scale_moe(**{**inputs, "swiglu_limit": None})

    _assert_matches_reference(out_fp8, out_ref, shape.name)
    assert not torch.allclose(out_fp8, out_unclamped), "the clamp changed nothing"


@skip_unsupported
def test_clamp_is_rejected_for_a_non_swiglu_activation() -> None:
    """Geglu carries no clamp, so the kernel refuses one instead of applying
    the SwiGLU order to a GELU gate."""
    inputs = _make_inputs(
        CORRECTNESS_SHAPES[0], activation_type=int(ActivationType.Geglu), swiglu_limit=0.1
    )

    with pytest.raises(ValueError, match="clamps SwiGLU only"):
        run_triton_fp8_block_scale_moe(**inputs)


@pytest.mark.cpu_only
def test_block_size_m_threshold() -> None:
    """Pins the switch to the large tile, and which test shapes take it, so that
    moving the threshold cannot silently drop the 64-row path from the tests."""
    assert _block_size_m(257, 2, 8) == _BLOCK_SIZE_M_LARGE
    assert _block_size_m(256, 2, 8) == _BLOCK_SIZE_M

    def block_size_m(name: str) -> int:
        s = SHAPES_BY_NAME[name]
        num_local_experts = s.num_experts // (EP_SIZE if s in EP_SHAPES else 1)
        return _block_size_m(s.num_tokens, s.top_k, num_local_experts)

    for name in ("tiny_prefill", "tiny_prefill_topk1", "ep2_prefill"):
        assert block_size_m(name) == _BLOCK_SIZE_M_LARGE, name
    for name in ("tiny", "tiny_topk1", "ep2_decode"):
        assert block_size_m(name) == _BLOCK_SIZE_M, name


@skip_unsupported
@pytest.mark.parametrize(
    ("name", "block_size_m"),
    [
        ("tiny", _BLOCK_SIZE_M),
        ("tiny_prefill", _BLOCK_SIZE_M_LARGE),
        ("tiny_prefill_topk1", _BLOCK_SIZE_M_LARGE),
    ],
)
def test_one_block_size_reaches_the_alignment_and_both_gemms(
    name: str, block_size_m: int, monkeypatch
) -> None:
    """The kernel reads one expert id per BLOCK_SIZE_M rows of the alignment, so
    the alignment and both GEMMs must get the same block size."""
    seen = []
    align = triton_moe.moe_align_block_size
    gemm = triton_moe._invoke_bf16act_fp8w_moe_kernel

    def recording_align(topk_ids, block_size, num_experts):
        seen.append(block_size)
        return align(topk_ids, block_size, num_experts)

    def recording_gemm(*args, block_size_m, **kwargs):
        seen.append(block_size_m)
        return gemm(*args, block_size_m=block_size_m, **kwargs)

    monkeypatch.setattr(triton_moe, "moe_align_block_size", recording_align)
    monkeypatch.setattr(triton_moe, "_invoke_bf16act_fp8w_moe_kernel", recording_gemm)
    run_triton_fp8_block_scale_moe(**_make_inputs(SHAPES_BY_NAME[name]))

    assert seen == [block_size_m] * 3


@skip_unsupported
@pytest.mark.parametrize("shape", EP_SHAPES, ids=lambda s: s.name)
def test_ep_ranks_sum_to_bf16_reference(shape: MoeShape) -> None:
    """Each rank computes the pairs of other ranks' experts on its edge slots with
    a zero scale; the ranks' outputs must still add up to the whole MoE."""
    inputs = _make_inputs(shape)

    out_ref = _run_moe_bf16_reference(**inputs)
    out_fp8 = sum(
        run_triton_fp8_block_scale_moe(**_local_slot_inputs(inputs, EP_SIZE, ep_rank)).float()
        for ep_rank in range(EP_SIZE)
    )

    _assert_matches_reference(out_fp8.bfloat16(), out_ref, shape.name)


@skip_unsupported
def test_each_block_size_is_tuned_separately() -> None:
    """BLOCK_SIZE_M is in the autotune key, so that one tile height never runs
    the config tuned for the other."""
    for name in ("tiny", "tiny_prefill"):
        run_triton_fp8_block_scale_moe(**_make_inputs(SHAPES_BY_NAME[name]))

    tuned = {key[:3] for key in triton_moe._fused_moe_bf16act_fp8w_kernel.cache}
    hidden, intermediate = SHAPES_BY_NAME["tiny"].hidden, SHAPES_BY_NAME["tiny"].intermediate
    for n, k in ((2 * intermediate, hidden), (hidden, intermediate)):
        assert {(n, k, _BLOCK_SIZE_M), (n, k, _BLOCK_SIZE_M_LARGE)} <= tuned, (n, k, tuned)


def _prewarm_stub(intermediate: int, use_dp: bool, top_k: int = 2) -> types.SimpleNamespace:
    """The attributes ``prewarm_decode_kernels`` reads off the impl: 8 local
    experts, hidden 256, top_k 2 by default, and 4 attention-DP ranks when
    ``use_dp``."""
    num_local_experts, hidden = 8, 256
    return types.SimpleNamespace(
        w3_w1_weight=torch.empty(num_local_experts, 2 * intermediate, hidden),
        w2_weight=torch.empty(num_local_experts, hidden, intermediate),
        quant_scales=types.SimpleNamespace(fc_weight_scales=None, proj_weight_scales=None),
        routing_method=types.SimpleNamespace(top_k=top_k),
        use_dp=use_dp,
        parallel_size=4,
        activation_type=int(ActivationType.Swiglu),
        act_clamp=None,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("max_num_tokens", "use_dp", "token_counts"),
    [
        # 8 local experts at top_k 2 take the large tile above 256 tokens.
        (256, False, [1]),
        (257, False, [1, 257]),
        (4096, False, [1, 4096]),
        # Under attention DP a rank computes all 4 ranks' tokens.
        (64, True, [1]),
        (100, True, [1, 400]),
    ],
)
def test_prewarm_tunes_each_tile_height_once(
    max_num_tokens: int, use_dp: bool, token_counts: list[int], monkeypatch
) -> None:
    """Both GEMMs are tuned at every tile height a forward can reach, once per
    shape and tile height across layers."""
    calls = []
    monkeypatch.setattr(
        triton_impl, "run_triton_fp8_block_scale_moe", lambda x, *a, **kw: calls.append(x.shape[0])
    )
    monkeypatch.setattr(triton_impl, "_prewarmed_gemm_keys", set())
    prewarm = triton_impl.TrtllmTritonFp8BlockScalesImpl.prewarm_decode_kernels

    layer = _prewarm_stub(intermediate=512, use_dp=use_dp)
    assert prewarm(layer, max_num_tokens, False) == 2 * len(token_counts)
    assert calls == token_counts
    # Another layer of the same shapes is already tuned.
    assert prewarm(_prewarm_stub(intermediate=512, use_dp=use_dp), max_num_tokens, False) == 0
    # A layer of other shapes is not.
    assert prewarm(_prewarm_stub(intermediate=1024, use_dp=use_dp), max_num_tokens, False) == (
        2 * len(token_counts)
    )
    assert calls == token_counts * 2


@pytest.mark.cpu_only
def test_prewarm_tunes_each_top_k(monkeypatch) -> None:
    """top_k is a constexpr of the GEMM kernels, so a layer of the same shapes
    and another top_k is tuned again, at both tile heights."""
    calls = []
    monkeypatch.setattr(
        triton_impl, "run_triton_fp8_block_scale_moe", lambda x, *a, **kw: calls.append(x.shape[0])
    )
    monkeypatch.setattr(triton_impl, "_prewarmed_gemm_keys", set())
    prewarm = triton_impl.TrtllmTritonFp8BlockScalesImpl.prewarm_decode_kernels

    for top_k in (2, 1):
        layer = _prewarm_stub(intermediate=512, use_dp=False, top_k=top_k)
        assert prewarm(layer, 4096, False) == 4
        assert prewarm(layer, 4096, False) == 0
    # 4,096 tokens take the large tile at either top_k.
    assert calls == [1, 4096, 1, 4096]


@requires_cuda
@pytest.mark.parametrize("num_experts", [8, 60, 72, 256, 288])
def test_moe_align_block_size_matches_reference(num_experts: int) -> None:
    """Every expert gets a block-padded range of its own tokens, whatever the expert count.

    The alignment kernels do not depend on the architecture, so unlike the MoE
    tests above this runs on any CUDA GPU.
    """
    block_size = 16
    g = torch.Generator(device="cuda").manual_seed(0)
    topk_ids = torch.randint(0, num_experts, (64, 8), device="cuda", generator=g)

    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(topk_ids, block_size, num_experts)

    flat_ids = topk_ids.flatten()
    counts = torch.bincount(flat_ids, minlength=num_experts)
    padded = (counts + block_size - 1) // block_size * block_size
    offsets = torch.cumsum(padded, dim=0) - padded
    assert num_post_pad.item() == padded.sum().item()

    expected_expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device="cuda"), padded // block_size
    )
    torch.testing.assert_close(expert_ids[: expected_expert_ids.numel()], expected_expert_ids)

    for expert in range(num_experts):
        start, count = offsets[expert].item(), counts[expert].item()
        end = start + padded[expert].item()
        expected_tokens = torch.nonzero(flat_ids == expert).flatten().to(torch.int32)
        torch.testing.assert_close(sorted_ids[start : start + count].sort().values, expected_tokens)
        # Padding slots hold the one-past-the-end token index.
        assert (sorted_ids[start + count : end] == flat_ids.numel()).all()

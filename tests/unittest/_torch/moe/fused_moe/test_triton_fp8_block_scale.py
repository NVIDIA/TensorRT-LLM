# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ``run_triton_fp8_block_scale_moe``, ``run_triton_fp8_block_scale_moe_fp8_act``
and ``moe_align_block_size``.

Run as a pytest:
    pytest tests/unittest/_torch/moe/fused_moe/test_triton_fp8_block_scale.py -v
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
import triton
from _torch.helpers import calc_diff, per_block_cast_to_fp8, per_block_cast_to_fp8_e8m0

import tensorrt_llm._torch.moe.fused_moe.cutlass.fp8_block_scales_triton as triton_impl
from tensorrt_llm._torch.moe.fused_moe import fused_moe_triton_fp8_block_scale
from tensorrt_llm._torch.moe.fused_moe.fused_moe_triton_fp8_block_scale import (
    _BLOCK_SIZE_M,
    _BLOCK_SIZE_M_LARGE,
    _block_size_m,
    _moe_align_small,
    moe_align_block_size,
    prewarm_triton_fp8_block_scale_moe_fp8_act,
    run_triton_fp8_block_scale_moe,
    run_triton_fp8_block_scale_moe_fp8_act,
)
from tensorrt_llm._torch.moe.fused_moe.interface import ActivationType
from tensorrt_llm._utils import get_sm_version, is_sm_120f

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
    """Cosine difference of the whole output and of every token, so that one
    wrong or missing row cannot hide in the aggregate, and the ratio of the
    norms, which the cosine difference barely sees: 10% too large is 4.5e-3."""
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
    align = fused_moe_triton_fp8_block_scale.moe_align_block_size
    gemm = fused_moe_triton_fp8_block_scale._invoke_bf16act_fp8w_moe_kernel

    def recording_align(topk_ids, block_size, num_experts):
        seen.append(block_size)
        return align(topk_ids, block_size, num_experts)

    def recording_gemm(*args, block_size_m, **kwargs):
        seen.append(block_size_m)
        return gemm(*args, block_size_m=block_size_m, **kwargs)

    monkeypatch.setattr(fused_moe_triton_fp8_block_scale, "moe_align_block_size", recording_align)
    monkeypatch.setattr(
        fused_moe_triton_fp8_block_scale, "_invoke_bf16act_fp8w_moe_kernel", recording_gemm
    )
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

    tuned = {
        key[:3] for key in fused_moe_triton_fp8_block_scale._fused_moe_bf16act_fp8w_kernel.cache
    }
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
    # top_k is a constexpr of the GEMM kernels, so another top_k compiles anew.
    assert prewarm(_prewarm_stub(intermediate=512, use_dp=use_dp, top_k=1), max_num_tokens, False)


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


skip_pre_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 90,
    reason="FP8 Triton dot needs SM90 or newer",
)

_BLOCK = 16  # _BLOCK_SIZE_M of the FP8-activation GEMMs.


def _per_block_cast_to_fp8_fp32(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(E, N, K) -> FP8 weights with FP32 (non-power-of-two) 128 x 128 scales, as
    DeepSeek-V3/R1 and Qwen3 FP8 checkpoints ship them."""
    pairs = [per_block_cast_to_fp8(expert) for expert in w]
    return torch.stack([q for q, _ in pairs]), torch.stack([sf for _, sf in pairs])


def _dequant_blocks(w_fp8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    num_experts, out_features, in_features = w_fp8.shape
    expanded = scales.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    return w_fp8.float() * expanded[:, :out_features, :in_features]


def _fake_quant_1x128(v: torch.Tensor) -> torch.Tensor:
    """Round a 1-D FP32 vector through FP8 with one scale per 128 elements, using the
    CUTLASS MoE arithmetic (multiply by 448 / amax, dequantize by its reciprocal).

    Both divisions are tensor by tensor: torch evaluates ``scalar / tensor`` as
    ``tensor.reciprocal() * scalar``, which is not correctly rounded, and one ulp
    of scale moves values that sit on an FP8 rounding midpoint.
    """
    groups = v.view(-1, 128)
    amax = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    quant_scale = torch.full_like(amax, 448.0) / amax
    dequant_scale = torch.ones_like(quant_scale) / quant_scale
    return ((groups * quant_scale).to(torch.float8_e4m3fn).float() * dequant_scale).view(-1)


def _run_moe_fp8_act_reference(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w3_w1: torch.Tensor,
    w3_w1_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    swiglu_limit: float | None,
    swiglu_clamp_after_silu: bool,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FP32 reference over dequantized weights and 1 x 128 FP8-rounded activations,
    with the CUTLASS SwiGLU clamp and BF16 rounding steps."""
    intermediate = w3_w1.shape[1] // 2
    w31 = _dequant_blocks(w3_w1, w3_w1_scales)
    w2f = _dequant_blocks(w2, w2_scales)
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        x_q = _fake_quant_1x128(x[t].float())
        for k in range(token_selected_experts.shape[1]):
            e = int(token_selected_experts[t, k])
            gate_up = (w31[e] @ x_q).bfloat16().float()
            up, gate = gate_up[:intermediate], gate_up[intermediate:]
            if swiglu_limit is not None:
                up = up.clamp(-swiglu_limit, swiglu_limit)
                if swiglu_clamp_after_silu:
                    act = F.silu(gate).clamp(max=swiglu_limit)
                else:
                    act = F.silu(gate.clamp(max=swiglu_limit))
            else:
                act = F.silu(gate)
            act_q = _fake_quant_1x128((act * up).bfloat16().float())
            out[t] += (w2f[e] @ act_q).bfloat16().float() * float(token_final_scales[t, k])
    return out.to(output_dtype)


def _check_alignment(
    topk_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_post_pad: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> None:
    """Every expert gets a block-padded range holding exactly its own pairs."""
    flat_ids = topk_ids.flatten()
    counts = torch.bincount(flat_ids, minlength=num_experts)
    padded = (counts + block_size - 1) // block_size * block_size
    offsets = torch.cumsum(padded, dim=0) - padded
    assert num_post_pad.item() == padded.sum().item()

    expected_expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=topk_ids.device),
        padded // block_size,
    )
    torch.testing.assert_close(expert_ids[: expected_expert_ids.numel()], expected_expert_ids)
    for expert in range(num_experts):
        start, count = offsets[expert].item(), counts[expert].item()
        end = start + padded[expert].item()
        expected_tokens = torch.nonzero(flat_ids == expert).flatten().to(torch.int32)
        torch.testing.assert_close(sorted_ids[start : start + count].sort().values, expected_tokens)
        # Padding slots hold the one-past-the-end pair index.
        assert (sorted_ids[start + count : end] == flat_ids.numel()).all()


@requires_cuda
@pytest.mark.parametrize("num_experts", [8, 60, 72, 288])
@pytest.mark.parametrize("num_tokens, top_k", [(1, 8), (5, 4), (16, 8)])
def test_moe_align_small_matches_reference(num_experts: int, num_tokens: int, top_k: int) -> None:
    """The single-program alignment, sized as the FP8-activation path sizes it."""
    g = torch.Generator(device="cuda").manual_seed(num_experts * 100 + num_tokens)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda", generator=g)
    num_pairs = topk_ids.numel()
    num_m_blocks = min(num_pairs, num_pairs // _BLOCK + num_experts)

    sorted_ids, expert_ids, num_post_pad = _moe_align_small(
        topk_ids, _BLOCK, num_experts, num_m_blocks
    )

    _check_alignment(topk_ids, sorted_ids, expert_ids, num_post_pad, num_experts, _BLOCK)


@requires_cuda
def test_moe_align_small_concentrated_routing() -> None:
    """All pairs on two experts, so each spans several blocks (the common case
    under expert parallelism, where pairs pile onto the local experts)."""
    topk_ids = torch.tensor([[0, 3]] * 40, device="cuda")
    num_pairs = topk_ids.numel()
    sorted_ids, expert_ids, num_post_pad = _moe_align_small(
        topk_ids, _BLOCK, 4, min(num_pairs, num_pairs // _BLOCK + 4)
    )
    _check_alignment(topk_ids, sorted_ids, expert_ids, num_post_pad, 4, _BLOCK)


@skip_pre_hopper
# 24 tokens x top-8 exceeds the single-program alignment and takes the multi-kernel one;
# hidden 2048 at 1 and 4 tokens runs the narrow-N tile that real hidden sizes use.
@pytest.mark.parametrize(
    "num_tokens, hidden",
    [(1, 512), (4, 512), (16, 512), (24, 512), (1, 2048), (4, 2048)],
)
@pytest.mark.parametrize(
    "swiglu_limit, swiglu_clamp_after_silu",
    [(None, False), (0.5, False), (0.5, True)],
    ids=["no_clamp", "clamp", "clamp_after_silu"],
)
@pytest.mark.parametrize("fp32_scales", [False, True], ids=["e8m0_scales", "fp32_scales"])
def test_fp8_act_matches_reference(
    num_tokens: int,
    hidden: int,
    swiglu_limit: float | None,
    swiglu_clamp_after_silu: bool,
    fp32_scales: bool,
) -> None:
    # A non-power-of-two expert count, as in decode-time routing of large MoEs.
    shape = MoeShape(
        "decode", num_tokens=num_tokens, hidden=hidden, intermediate=256, num_experts=72, top_k=8
    )
    inputs = _make_inputs(shape, seed=num_tokens, swiglu_limit=swiglu_limit)
    del inputs["activation_type"]
    if fp32_scales:
        g = torch.Generator(device="cuda").manual_seed(num_tokens + 1)
        w3_w1 = torch.randn((72, 512, hidden), device="cuda", generator=g) * 0.05
        w2 = torch.randn((72, hidden, 256), device="cuda", generator=g) * 0.05
        inputs["w3_w1"], inputs["w3_w1_scales"] = _per_block_cast_to_fp8_fp32(w3_w1.bfloat16())
        inputs["w2"], inputs["w2_scales"] = _per_block_cast_to_fp8_fp32(w2.bfloat16())

    out = run_triton_fp8_block_scale_moe_fp8_act(
        **inputs, swiglu_clamp_after_silu=swiglu_clamp_after_silu
    )
    ref = _run_moe_fp8_act_reference(**inputs, swiglu_clamp_after_silu=swiglu_clamp_after_silu)

    assert out.shape == ref.shape == (num_tokens, shape.hidden)
    assert out.dtype == torch.bfloat16
    diff = calc_diff(out, ref).item()
    assert diff < 1e-4, f"cosine-diff={diff:.4e}"


@skip_pre_hopper
def test_fp8_act_zero_weights_and_output_dtype() -> None:
    """Zero-weight pairs (the non-local pairs under EP) contribute nothing, and
    the top-k sum writes the requested output dtype."""
    shape = MoeShape("decode", num_tokens=4, hidden=512, intermediate=256, num_experts=8, top_k=4)
    inputs = _make_inputs(shape, seed=11)
    del inputs["activation_type"]
    inputs["token_final_scales"][:, 2:] = 0.0

    out = run_triton_fp8_block_scale_moe_fp8_act(**inputs, output_dtype=torch.float32)
    kept = dict(
        inputs,
        token_selected_experts=inputs["token_selected_experts"][:, :2],
        token_final_scales=inputs["token_final_scales"][:, :2],
    )
    ref = _run_moe_fp8_act_reference(
        **kept, swiglu_clamp_after_silu=False, output_dtype=torch.float32
    )

    assert out.dtype == torch.float32
    diff = calc_diff(out, ref).item()
    assert diff < 1e-4, f"cosine-diff={diff:.4e}"


def _compiled_kernel_count() -> int:
    """Kernels compiled so far, on every device, by the module's Triton functions."""
    count = 0
    for value in vars(fused_moe_triton_fp8_block_scale).values():
        while not isinstance(value, triton.runtime.JITFunction) and hasattr(value, "fn"):
            value = value.fn
        if isinstance(value, triton.runtime.JITFunction):
            count += sum(len(cache[0]) for cache in value.device_caches.values())
    return count


@skip_pre_hopper
# Hidden 2048 runs the narrow-N tile at up to 4 tokens and the wide one above.
@pytest.mark.parametrize("hidden", [512, 2048])
def test_fp8_act_prewarm_leaves_nothing_to_compile(hidden: int) -> None:
    """After the prewarm, no call of 1 to ``max_tokens`` tokens on the same weights compiles a kernel,
    whatever the routing: routing changes device data only, never a compile-time choice."""
    num_experts, top_k, max_tokens, limit = 72, 8, 32, 0.5
    shape = MoeShape(
        "decode", 1, hidden=hidden, intermediate=256, num_experts=num_experts, top_k=top_k
    )
    inputs = _make_inputs(shape, seed=hidden)
    weights = {name: inputs[name] for name in ("w3_w1", "w3_w1_scales", "w2", "w2_scales")}

    prewarm_triton_fp8_block_scale_moe_fp8_act(
        **weights, top_k=top_k, max_tokens=max_tokens, swiglu_limit=limit
    )
    compiled = _compiled_kernel_count()
    assert compiled > 0

    g = torch.Generator(device="cuda").manual_seed(hidden)
    for num_tokens in range(1, max_tokens + 1):
        # As the routing hands them over: int32 expert IDs and FP32 weights.
        experts = (
            torch.rand((num_tokens, num_experts), device="cuda", generator=g).topk(top_k).indices
        )
        run_triton_fp8_block_scale_moe_fp8_act(
            torch.randn((num_tokens, hidden), device="cuda", generator=g).bfloat16(),
            experts.to(torch.int32),
            torch.rand((num_tokens, top_k), device="cuda", generator=g),
            **weights,
            swiglu_limit=limit,
        )
        assert _compiled_kernel_count() == compiled, f"{num_tokens} tokens compiled a kernel"

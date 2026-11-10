# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ``run_triton_fp8_block_scale_moe``, ``run_triton_fp8_block_scale_moe_fp8_act``
and ``moe_align_block_size``.

Run as a pytest:
    pytest tests/unittest/_torch/moe/fused_moe/test_triton_fp8_block_scale.py -v
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from _torch.helpers import calc_diff, per_block_cast_to_fp8, per_block_cast_to_fp8_e8m0

from tensorrt_llm._torch.moe.fused_moe.fused_moe_triton_fp8_block_scale import (
    _moe_align_small,
    moe_align_block_size,
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

    diff = calc_diff(out_fp8, out_ref).item()
    assert diff < 1e-2, f"shape={shape.name} cosine-diff={diff:.4e} exceeds 1e-2"


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

    diff = calc_diff(out_fp8, out_ref).item()
    assert diff < 1e-2, f"shape={shape.name} cosine-diff={diff:.4e} exceeds 1e-2"
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

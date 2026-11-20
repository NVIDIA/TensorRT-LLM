# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct kernel contracts for trtllm_triton_grouped_gemm."""

from __future__ import annotations

import pytest
import torch
from _torch.helpers import calc_diff
from _torch.moe.impls._triton_fp8_test_utils import (
    _BLOCK,
    ACTIVATIONS,
    CORRECTNESS_SHAPES,
    MoeShape,
    _check_alignment,
    _make_inputs,
    _run_moe_bf16_reference,
    requires_cuda,
    skip_unsupported,
)

from tensorrt_llm._torch.moe.fused_moe.fused_moe_triton_fp8_block_scale import (
    _moe_align_small,
    moe_align_block_size,
    run_triton_fp8_block_scale_moe,
)
from tensorrt_llm._torch.moe.fused_moe.interface import ActivationType


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

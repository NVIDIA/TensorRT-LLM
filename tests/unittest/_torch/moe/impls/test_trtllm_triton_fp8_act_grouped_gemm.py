# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct kernel contracts for trtllm_triton_fp8_act_grouped_gemm."""

from __future__ import annotations

import pytest
import torch
from _torch.helpers import calc_diff
from _torch.moe.impls._triton_fp8_test_utils import (
    MoeShape,
    _compiled_kernel_count,
    _make_inputs,
    _per_block_cast_to_fp8_fp32,
    _run_moe_fp8_act_reference,
    skip_pre_hopper,
)

from tensorrt_llm._torch.moe.fused_moe.fused_moe_triton_fp8_block_scale import (
    prewarm_triton_fp8_block_scale_moe_fp8_act,
    run_triton_fp8_block_scale_moe_fp8_act,
)


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

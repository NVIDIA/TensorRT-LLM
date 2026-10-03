# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ``run_triton_fp8_block_scale_moe``.

Run as a pytest:
    pytest tests/unittest/_torch/moe/fused_moe/test_triton_fp8_block_scale.py -v
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from _torch.helpers import calc_diff, per_block_cast_to_fp8_e8m0

from tensorrt_llm._torch.moe.fused_moe.fused_moe_triton_fp8_block_scale import (
    run_triton_fp8_block_scale_moe,
)
from tensorrt_llm._torch.moe.fused_moe.interface import ActivationType
from tensorrt_llm._utils import get_sm_version

skip_unsupported = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="Requires CUDA SM120 or SM121 for the Triton FP8 block-scale MoE path",
)


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
            if act_int == int(ActivationType.Relu2):
                # Non-gated: the whole projection is the up half.
                ic2 = F.relu(gate_up).square()
            elif act_int in (int(ActivationType.Swiglu), int(ActivationType.Silu)):
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

# Nemotron-H Lightning expert dimensions; 1856 is not a multiple of the 128-row scale block.
LIGHTNING_HIDDEN = 2688
LIGHTNING_INTER = 1856
LIGHTNING_PADDED_INTER = 1920
LIGHTNING_CORR = MoeShape(
    "lightning_corr",
    num_tokens=32,
    hidden=LIGHTNING_HIDDEN,
    intermediate=LIGHTNING_INTER,
    num_experts=16,
    top_k=6,
)

# Non-gated (Relu2) experts: one aligned size and two that end in a partial scale block.
RELU2_SHAPES = [
    MoeShape("tiny_aligned", num_tokens=64, hidden=256, intermediate=512, num_experts=8, top_k=2),
    MoeShape("tiny_unaligned", num_tokens=64, hidden=256, intermediate=192, num_experts=8, top_k=2),
    LIGHTNING_CORR,
]


def _make_inputs(
    shape: MoeShape,
    *,
    device: str = "cuda",
    seed: int = 0,
    activation_type: int = int(ActivationType.Swiglu),
) -> dict[str, torch.Tensor | int]:
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

    # Non-gated (Relu2) experts carry only the up projection.
    gate_up_size = inter if activation_type == int(ActivationType.Relu2) else 2 * inter
    w3_w1_bf16 = (
        torch.randn(
            (num_experts, gate_up_size, hidden), dtype=torch.bfloat16, device=device, generator=g
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
@pytest.mark.parametrize("shape", RELU2_SHAPES, ids=lambda s: s.name)
def test_fp8_relu2_matches_bf16_reference(shape: MoeShape) -> None:
    inputs = _make_inputs(shape, activation_type=int(ActivationType.Relu2))

    out_fp8 = run_triton_fp8_block_scale_moe(**inputs)
    out_ref = _run_moe_bf16_reference(**inputs)

    assert out_fp8.shape == out_ref.shape == (shape.num_tokens, shape.hidden)
    assert out_fp8.dtype == torch.bfloat16

    diff = calc_diff(out_fp8, out_ref).item()
    assert diff < 1e-2, f"shape={shape.name} cosine-diff={diff:.4e} exceeds 1e-2"


@skip_unsupported
def test_fp8_relu2_padded_weights_match_unpadded_reference() -> None:
    """Lightning experts as the loader leaves them: 1856 rows padded to 1920 with a zero tail."""
    inputs = _make_inputs(LIGHTNING_CORR, activation_type=int(ActivationType.Relu2))
    pad = LIGHTNING_PADDED_INTER - LIGHTNING_INTER
    # The 15-block scale grids already cover 1920 rows, so only the FP8 weights grow.
    padded_inputs = dict(
        inputs,
        w3_w1=F.pad(inputs["w3_w1"].view(torch.uint8), (0, 0, 0, pad)).view(torch.float8_e4m3fn),
        w2=F.pad(inputs["w2"].view(torch.uint8), (0, pad)).view(torch.float8_e4m3fn),
    )

    out_fp8 = run_triton_fp8_block_scale_moe(**padded_inputs)
    out_ref = _run_moe_bf16_reference(**inputs)

    assert padded_inputs["w3_w1"].shape[1] == padded_inputs["w2"].shape[2] == LIGHTNING_PADDED_INTER
    assert out_fp8.shape == out_ref.shape == (LIGHTNING_CORR.num_tokens, LIGHTNING_CORR.hidden)

    diff = calc_diff(out_fp8, out_ref).item()
    assert diff < 1e-2, f"padded lightning_corr cosine-diff={diff:.4e} exceeds 1e-2"


@skip_unsupported
def test_non_gated_experts_reject_gated_activation() -> None:
    inputs = _make_inputs(RELU2_SHAPES[0], activation_type=int(ActivationType.Relu2))
    inputs["activation_type"] = int(ActivationType.Swiglu)

    with pytest.raises(ValueError, match="Unsupported non-gated activation_type"):
        run_triton_fp8_block_scale_moe(**inputs)

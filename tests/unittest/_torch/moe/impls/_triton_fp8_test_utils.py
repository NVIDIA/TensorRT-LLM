# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""References and inputs shared by the two Triton FP8 MoE implementations."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
import triton
from _torch.helpers import per_block_cast_to_fp8, per_block_cast_to_fp8_e8m0

from tensorrt_llm._torch.moe.fused_moe import fused_moe_triton_fp8_block_scale
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


def _compiled_kernel_count() -> int:
    """Kernels compiled so far, on every device, by the module's Triton functions."""
    count = 0
    for value in vars(fused_moe_triton_fp8_block_scale).values():
        while not isinstance(value, triton.runtime.JITFunction) and hasattr(value, "fn"):
            value = value.fn
        if isinstance(value, triton.runtime.JITFunction):
            count += sum(len(cache[0]) for cache in value.device_caches.values())
    return count

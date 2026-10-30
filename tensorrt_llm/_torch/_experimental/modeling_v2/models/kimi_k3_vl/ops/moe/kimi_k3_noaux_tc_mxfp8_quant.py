# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode MoE input in one launch: noaux_tc routing (896 experts, top-16) + MXFP8 quantization."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class KimiK3NoauxTcMxfp8Quant(OpWrapper):
    """Route each token to 16 of 896 experts and quantize its [3584] hidden row to MXFP8.

    Returns `(topk_ids [M, 16] int32, topk_weights [M, 16] bf16, data [M, 3584] float8_e4m3fn,
    scales [M, 112] uint8)`, all newly allocated and contiguous: ids first (the opposite order
    to noaux_tc_op), linear UE8M0 scales, one per 32 elements.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"M={tokens}, scale={scale}: K3 decode batch and routing renormalization",
                spec=dict(tokens=tokens, routed_scaling_factor=scale, logits="random"),
            )
            for tokens in (1, 2, 5, 8, 16, 33, 63, 64)
            for scale in (1.0, 2.5)
        ),
        *(
            Cell(
                why=f"M={tokens}, {logits}: stable tie order or tanh sigmoid saturation",
                spec=dict(tokens=tokens, routed_scaling_factor=1.0, logits=logits),
            )
            for tokens in (1, 17, 64)
            for logits in ("tied", "saturated")
        ),
        *(
            Cell(
                why=f"hidden scale {scale}: quantization magnitude and all-zero blocks",
                spec=dict(
                    tokens=8,
                    routed_scaling_factor=1.0,
                    logits="random",
                    hidden_scale=scale,
                    zero_blocks=True,
                ),
            )
            for scale in (1e-3, 1.0, 1e4)
        ),
    )
    note = """
    The router GEMM, MXFP8 x MXFP4 experts and combine are the caller's.
    Expert count 896, top-k 16, one group and hidden width 3584 are fixed.
    The first M CTAs route and the next M quantize; they share no data.
    Selection uses the tanh-form sigmoid plus bias, ties favor lower expert ids;
    combine weights use unbiased scores. Renormalization sums in fp32 then
    computes in fp64, rounding once to bf16. noaux_tc_op with fp32 logits rounds
    to fp32 first, so casting its result can differ by one bf16 ulp at midpoints.
    router_logits names the underlying schema's scores argument.

    Quantization matches mxfp8_quantize with linear scales and alignment=512;
    3584 needs no padding, and the scale buffer is viewed as [M,112] rather than
    flat. Non-finite logits/hidden blocks and denormal blocks inherit those
    underlying ops' behavior; blocks below 448*2^-127 take a flush-to-zero path
    not modeled by this reference. These cases are outside the certified cells.

    No metadata, workspace, cache or tuner state. Inputs are not mutated or
    retained; outputs are new. Calls use the ambient stream, capture in CUDA
    graphs and replay rewritten inputs with eager bits. TRTLLM_ENABLE_PDL enables
    programmatic dependent launch by default, affecting scheduling only.
    K3 uses this entry through 64 tokens, then separate noaux_tc_op and
    mxfp8_quantize(alignment=512) above that. Invalid shapes, dtypes, layouts,
    devices, token counts and architectures are rejected by the op.
    """

    def __call__(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        hidden_states: torch.Tensor,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(
            router_logits, bias, hidden_states, routed_scaling_factor
        )

    def reference(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        hidden_states: torch.Tensor,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = 0.5 * torch.tanh(0.5 * router_logits.float()) + 0.5
        ids = torch.sort(scores + bias.float(), dim=-1, descending=True, stable=True).indices[
            :, :16
        ]
        weights = scores.gather(1, ids)
        total = weights.sum(-1, keepdim=True).double()
        weights = weights.double() / (total + 1e-20) * routed_scaling_factor
        blocks = hidden_states.float().reshape(hidden_states.shape[0], 112, 32)
        amax = blocks.abs().amax(dim=-1)
        bits = (amax / 448.0).view(torch.int32)
        byte = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) > 0).to(torch.int32)).clamp(0, 254)
        scale = torch.exp2(byte.float() - 127.0).unsqueeze(-1)
        data = (blocks / scale).reshape(hidden_states.shape).to(torch.float8_e4m3fn)
        return ids.to(torch.int32), weights.to(torch.bfloat16), data, byte.to(torch.uint8)


kimi_k3_noaux_tc_mxfp8_quant = KimiK3NoauxTcMxfp8Quant()

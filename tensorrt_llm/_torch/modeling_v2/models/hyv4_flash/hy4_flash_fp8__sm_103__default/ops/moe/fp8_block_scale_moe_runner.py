# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-style FP8 block-scale mixture-of-experts, one trtllm-gen call.

The target passes router logits and bias with no ids, so the kernel routes in-op over the local
expert window; the pre-routed path is the same op call.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: Block edge on every scale axis: 1x128 activations, 128x128 weights.
BLOCK = 128

#: Largest finite e4m3 magnitude.
E4M3_MAX = 448.0

#: The activation kernel's block-amax floor (`kDsActAmaxEpsilon`).
ACT_AMAX_FLOOR = 1.0e-10

#: `RoutingMethodType::DeepSeekV3`; with `n_group <= 1` the kernel routes with `routingCustom`.
ROUTING_DEEPSEEK_V3 = 2


def dequantize_block_weight(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """One expert's `[M, K]` fp32 weight from e4m3 codes and `[M/128, K/128]` scales."""
    m, k = codes.shape
    blocks = codes.float().reshape(m // BLOCK, BLOCK, k // BLOCK, BLOCK)
    return (blocks * scales.float().reshape(m // BLOCK, 1, k // BLOCK, 1)).reshape(m, k)


def requantize_1x128(x: torch.Tensor) -> torch.Tensor:
    """The kernel's e4m3 round trip of both intermediates: `scale = max(amax, 1e-10) / 448` per 1x128."""
    shape = x.shape
    blocks = x.reshape(*shape[:-1], shape[-1] // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=ACT_AMAX_FLOOR) / E4M3_MAX
    codes = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (codes.float() * scale).reshape(shape)


def in_op_routing(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's own router (`routingCustom`, SigmoidBias + ScaledSumNormalize), in its arithmetic.

        sel[j] = 0.5 * tanh(0.5 * logit[j]) + 0.5 + bias[j]     # fp32
        top-k  = the k largest `sel`, ties to the lower id, over all `num_experts`
        w[j]   = (sel[j] - bias[j]) * scale / sum(sel - bias), stored bf16

    Returns `(topk_ids [T, top_k] int32, topk_weights [T, top_k] bf16)`.
    """
    scores = 0.5 * torch.tanh(0.5 * routing_logits.float()) + 0.5
    choice = scores + routing_bias.float()
    ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :top_k]
    sigmoid = torch.gather(choice, 1, ids) - routing_bias.float()[ids]
    total = sigmoid.sum(dim=-1, keepdim=True)
    weights = (sigmoid * routed_scaling_factor / total).to(torch.bfloat16)
    return ids.to(torch.int32).contiguous(), weights.contiguous()


#: The checkpoint's routed experts; all 512 are local at tp4.
_HY4 = dict(
    hidden=1536,
    inter=1536,
    num_experts=512,
    top_k=16,
    n_group=1,
    topk_group=1,
    local_expert_offset=0,
    local_num_experts=512,
    gemm1_clamp_limit=10.0,
    routed_scaling_factor=3.998,
)


def _window(offset: int, local: int = 128) -> dict:
    """One EP rank's window (128 wide at EP4, 256 at EP2); the router still ranks all 512."""
    return {**_HY4, "local_expert_offset": offset, "local_num_experts": local}


class Fp8BlockScaleMoeRunner(OpWrapper):
    """One FP8 block-scale MoE layer over e4m3 activations.

    Per token `t` and slot `j` with local expert `e = topk_ids[t, j] - local_expert_offset`
    (slots outside the window contribute nothing):

        up, gate = requant(A[t] @ [W_up; W_gate].T).split(I)
        gate, up = min(gate, lim), clamp(up, -lim, lim)    # gemm1_clamp_limit
        out[t]  += topk_weights[t, j] * bf16(requant(silu(gate) * up) @ W_down.T)

    FC1 rows are `[up; gate]` (the loader reorders the checkpoint's gate-first rows). With
    `topk_ids` None the kernel routes in-op (`in_op_routing`). `hidden_states_scale` is
    `[H/128, T]` as `fp8_quantize_1x128` returns it; at `T == H/128` its transpose passes every
    check and is silently wrong.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # tp4: the whole stack is local.
        Cell(
            why="decode: one token through all 512 experts (tp4's local stack)",
            spec=dict(tokens=1, **_HY4),
        ),
        Cell(
            why="12 rows == H/128: the activation scale buffer is square",
            spec=dict(tokens=12, **_HY4),
        ),
        Cell(
            why="64 rows: a short batch through all 512 experts",
            spec=dict(tokens=64, **_HY4),
        ),
        Cell(
            why="prefill chunk: 5120 rows through all 512 experts",
            spec=dict(tokens=5120, **_HY4),
        ),
        # EP4: one 128-expert window per rank.
        Cell(
            why="EP4 decode: one token, rank 0's 128-expert window",
            spec=dict(tokens=1, **_window(0)),
        ),
        Cell(
            why="EP4: 64 rows, rank 1's 128-expert window",
            spec=dict(tokens=64, **_window(128)),
        ),
        Cell(
            why="EP4 prefill chunk: 5120 rows, rank 2's 128-expert window",
            spec=dict(tokens=5120, **_window(256)),
        ),
        Cell(
            why="EP4 decode: one token, the last 128-expert window (384)",
            spec=dict(tokens=1, **_window(384)),
        ),
        # EP2: two 256-expert windows (the disaggregated context engine at tep2).
        Cell(
            why="the ctx engine's decode, one token, through EP2 rank 0's 256-expert window",
            spec=dict(tokens=1, **_window(0, 256)),
        ),
        Cell(
            why="a 64-row batch, rank 0's 256-expert window",
            spec=dict(tokens=64, **_window(0, 256)),
        ),
        Cell(
            why="a 5120-token prefill chunk, rank 1's 256-expert window",
            spec=dict(tokens=5120, **_window(256, 256)),
        ),
        Cell(
            why="the ctx engine's decode, one token, through EP2 rank 1's 256-expert window",
            spec=dict(tokens=1, **_window(256, 256)),
        ),
        # The target's path: router logits and fp32 bias in, routed in-op.
        Cell(
            why="the engine's decode: one token through all 512 experts at tp4, routed in-op "
            "from the router logits and bias",
            spec=dict(
                entry="in-kernel-routing",
                tokens=1,
                routing_method_type=ROUTING_DEEPSEEK_V3,
                topk_ids=None,
                topk_weights=None,
                **_HY4,
            ),
        ),
        Cell(
            why="the engine's 5120-token prefill chunk and one EP4 window (128 experts at "
            "offset 128): the in-op router ranks all 512",
            spec=dict(
                entry="in-kernel-routing",
                tokens=5120,
                routing_method_type=ROUTING_DEEPSEEK_V3,
                topk_ids=None,
                topk_weights=None,
                **_window(128),
            ),
        ),
    )

    note = """
    In-op routing is certified at `routing_method_type=2`, `n_group <= 1` only; `num_experts` is the
    routing space (512) on every window call. The registered fake returns `[T, 2H]` (upstream bug).
    """

    def __call__(
        self,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_fused_shared_experts: Optional[int],
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        act_type: int = 0,
        gemm1_clamp_limit: Optional[float] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> torch.Tensor:
        return torch.ops.trtllm.fp8_block_scale_moe_runner(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            num_experts,
            top_k,
            num_fused_shared_experts,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            routing_method_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            act_type=act_type,
            gemm1_clamp_limit=gemm1_clamp_limit,
            output=output,
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=use_dp,
        )

    def reference(
        self,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_fused_shared_experts: Optional[int],
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        act_type: int = 0,
        gemm1_clamp_limit: Optional[float] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> torch.Tensor:
        """fp32 GEMMs over bit-identical operands, modeling the kernel's e4m3 and bf16 stores."""
        if routing_logits is not None:
            assert topk_ids is None and topk_weights is None, (
                "pass routing_logits or topk_ids, not both"
            )
            topk_ids, topk_weights = in_op_routing(
                routing_logits, routing_bias, top_k, routed_scaling_factor
            )
        else:
            assert topk_ids is not None and topk_weights is not None, (
                "the reference needs a routing"
            )
        num_tokens, hidden = hidden_states.shape
        nb = hidden // BLOCK
        act_in = (
            hidden_states.float().reshape(num_tokens, nb, BLOCK)
            * hidden_states_scale.t().reshape(num_tokens, nb, 1).float()
        ).reshape(num_tokens, hidden)
        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=hidden_states.device)
        local = topk_ids.long() - local_expert_offset
        live = local[(local >= 0) & (local < local_num_experts)]
        with true_fp32_matmul():
            for e in torch.unique(live).tolist():
                tok, slot = (local == e).nonzero(as_tuple=True)
                w1 = dequantize_block_weight(gemm1_weights[e], gemm1_weights_scale[e])
                y1 = requantize_1x128(act_in[tok] @ w1.t())
                up, gate = y1[:, :intermediate_size], y1[:, intermediate_size:]
                if gemm1_clamp_limit is not None:
                    gate = gate.clamp(max=gemm1_clamp_limit)
                    up = up.clamp(-gemm1_clamp_limit, gemm1_clamp_limit)
                act = requantize_1x128(torch.nn.functional.silu(gate) * up)
                w2 = dequantize_block_weight(gemm2_weights[e], gemm2_weights_scale[e])
                # Each expert's FC2 row is stored bf16 before the combine.
                y = (act @ w2.t()).to(torch.bfloat16).float()
                out.index_add_(0, tok, y * topk_weights[tok, slot].float().unsqueeze(1))
        return out.to(torch.bfloat16)

    def is_valid(
        self,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_fused_shared_experts: Optional[int],
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        act_type: int = 0,
        gemm1_clamp_limit: Optional[float] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> None:
        # The kernel takes raw data pointers: a strided view is read as dense.
        for name, tensor in (
            ("routing_logits", routing_logits),
            ("routing_bias", routing_bias),
            ("hidden_states", hidden_states),
            ("hidden_states_scale", hidden_states_scale),
            ("gemm1_weights", gemm1_weights),
            ("gemm1_weights_scale", gemm1_weights_scale),
            ("gemm2_weights", gemm2_weights),
            ("gemm2_weights_scale", gemm2_weights_scale),
            ("topk_weights", topk_weights),
            ("topk_ids", topk_ids),
            ("output", output),
        ):
            if tensor is not None:
                assert tensor.is_contiguous(), f"{name} must be contiguous"
        # The Python runner always computes SwiGLU.
        assert act_type == 0, f"act_type={act_type} is silently ignored"
        # Once topk_ids carry the routing, nothing reads the bias.
        if routing_bias is not None:
            assert topk_ids is None, "routing_bias is ignored when topk_ids are given"
        # Outside the routing space the op answers zeros or reads out of bounds.
        assert 0 <= local_expert_offset < num_experts, (
            f"local_expert_offset={local_expert_offset} is outside [0, num_experts={num_experts})"
        )


fp8_block_scale_moe_runner = Fp8BlockScaleMoeRunner()

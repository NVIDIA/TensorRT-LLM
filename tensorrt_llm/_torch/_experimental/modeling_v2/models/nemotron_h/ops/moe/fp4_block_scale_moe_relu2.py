# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 MoE with the Relu2 activation, over the shared trtllm-gen runner.

The catalog's `fp4_block_scale_moe_runner` certifies SwiGlu only, and its
`__call__` already passes `act_type` through to the op. This family op calls it
with every argument unchanged and certifies what the Nemotron-3 Ultra NVFP4
targets run on top: Relu2 (`act_type` 1), a non-gated FC1, per-expert
`weight_scale_2`, and their routing, at the geometry they call it with.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    dequantize_nvfp4,
    fp4_block_scale_moe_runner,
    quantize_nvfp4,
)

#: `act_type` for Relu2, `relu(x) ** 2`, in the trtllm-gen runner's enum: element-wise, so FC1
#: is the up projection alone, I rows per expert.
ACT_TYPE_RELU2 = 1

#: The trtllm-gen activations whose FC1 is a gated pair: SwiGlu (0) and SiTu (3). Relu2 (1)
#: and Silu (2) are element-wise.
_GATED_ACT_TYPES = frozenset({0, 3})

#: One expert stack as the reference takes it: `(codes[E, M, K], scales[E, M, K/16])`,
#: unshuffled and unswizzled, with each expert's fp32 `weight_scale_2` as a third element.
Operand = Tuple[torch.Tensor, ...]


def _project(x: torch.Tensor, operand: Operand, expert: int) -> torch.Tensor:
    """`x @ W_e^T` for one expert, fp32. A per-expert `weight_scale_2` multiplies the
    product rather than the weight, which is where the kernel applies it too: folded into
    the accumulator's dequantization scalar."""
    y = x @ dequantize_nvfp4(operand[0][expert], operand[1][expert]).t()
    return y * operand[2][expert].float() if len(operand) > 2 else y


class Fp4BlockScaleMoeRelu2(OpWrapper):
    """One NVFP4-weight MoE layer with Relu2 experts over pre-routed NVFP4 tokens.

    Returns what the shared runner returns: `[out]`, `out` a fresh `[T, hidden]` bf16 (with
    `do_finalize`). Ids are global; this rank answers for `[local_expert_offset,
    local_expert_offset + local_num_experts)` and every other slot contributes zero.
    """

    ARCHS = frozenset({Arch.SM_100})

    # Nemotron-3 Ultra's routed experts in the 2048-wide latent space, as its NVFP4 tep4
    # and dep4 targets issue them: 512 experts in four 128-wide windows, top-22, FC1 the
    # up projection alone (5120 rows). The slot weights are the router's sigmoid scores
    # of the top 22, renormalized and scaled to sum to 5.0 (`routing="sigmoid_x5"`). Every
    # expert carries its own `weight_scale_2` while both input scales are shared by the
    # layer. Nothing is padded: H % 256, I % 128, H/16 % 4 and I/16 % 4 are all zero.
    _ULTRA = dict(
        hidden=2048,
        intermediate=5120,
        num_experts=512,
        top_k=22,
        local_num_experts=128,
        act_type=ACT_TYPE_RELU2,
        routing="sigmoid_x5",
    )

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="Ultra tep4 decode, 1 row, rank 0: relu2 FC1 of 5120 rows, top-22 of 512, "
            "a 128-wide window",
            spec=dict(tokens=1, local_expert_offset=0, **_ULTRA),
        ),
        Cell(
            why="Ultra decode, 1 row, rank 3: the last window, ids 384-511 -- the top of "
            "the expert range",
            spec=dict(tokens=1, local_expert_offset=384, **_ULTRA),
        ),
        Cell(
            why="Ultra at 8 rows, rank 1: the pre-routed path's dynamic-block permutation kernel",
            spec=dict(tokens=8, local_expert_offset=128, **_ULTRA),
        ),
        Cell(
            why="Ultra decode ceiling, 32 rows, rank 0: the single-cluster permutation kernel",
            spec=dict(tokens=32, local_expert_offset=0, **_ULTRA),
        ),
        Cell(
            why="Ultra decode ceiling, 32 rows, rank 3",
            spec=dict(tokens=32, local_expert_offset=384, **_ULTRA),
        ),
        Cell(
            why="Ultra prefill chunk, 300 rows, rank 0: a mid-width FC1 token tile",
            spec=dict(tokens=300, local_expert_offset=0, **_ULTRA),
        ),
        Cell(
            why="Ultra prefill chunk, 300 rows, rank 3",
            spec=dict(tokens=300, local_expert_offset=384, **_ULTRA),
        ),
        Cell(
            why="Ultra prefill at 8192 rows, rank 0: the widest FC1 token tile and the top "
            "of the certified column",
            spec=dict(tokens=8192, local_expert_offset=0, **_ULTRA),
        ),
        Cell(
            why="Ultra prefill at 8192 rows, rank 3",
            spec=dict(tokens=8192, local_expert_offset=384, **_ULTRA),
        ),
    )

    note = """
    Certified pre-routed only (`routing_logits` None, `topk_ids` / `topk_weights` given),
    with the routing arguments the targets pass: `routing_method_type` 2 (DeepSeekV3),
    `n_group` 1, `topk_group` 1, `routed_scaling_factor` 5.0. They are inert on a cold
    cache; under `autotune()` the tuner's dummy routing of that family needs all three as
    numbers.

    Scalars, mapped from a ModelOpt NVFP4 checkpoint (`*_input_scale` shared by the layer,
    `*_weight_scale_2` per expert): `output1_scale_gate_scalar = up_input_scale *
    up_weight_scale_2[e]` is applied before the relu2 and `output1_scale_scalar = 1 /
    down_input_scale` after it, so for Relu2 the latter is the FC2-input quantization scale
    alone, unlike SwiGlu's, which also carries the dequantization. Handing Relu2 the
    SwiGlu-form scalar gives an all-zero output. `output2_scale_scalar = down_input_scale *
    down_weight_scale_2[e]`. A rank slices the three per-expert vectors to its window.

    The FC1 epilogue requantizes the relu2 output to NVFP4 under `1 / down_input_scale`
    before FC2, so `reference` does too.

    The weights take the layout of the shared runner's contract: rows block-shuffled in
    32-row blocks (no gate interleave, since FC1 is a single projection) and the block
    scales 128x4-swizzled.
    """

    def __call__(
        self,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool,
        act_type: int = ACT_TYPE_RELU2,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> list[torch.Tensor]:
        return fp4_block_scale_moe_runner(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm1_bias,
            gemm1_alpha,
            gemm1_beta,
            gemm1_clamp_limit,
            gemm2_weights,
            gemm2_weights_scale,
            gemm2_bias,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            routing_method_type,
            do_finalize,
            act_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=use_dp,
        )

    def reference(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        up: Operand,
        down: Operand,
        fc2_global_scale: float,
        local_expert_offset: int = 0,
        local_num_experts: Optional[int] = None,
    ) -> torch.Tensor:
        """The layer, in plain torch over dequantized weights, fp32 throughout.

        The documented exception to mirroring `__call__`, as for the shared runner: this
        takes the per-expert operands the prepared layout was built from, plus the routing
        the call was entered with. `hidden_states` is the activation's exact value.

        The relu2 output is requantized to NVFP4 under `fc2_global_scale` before FC2,
        because the kernel's FC1 epilogue emits it as NVFP4.
        """
        num_tokens, hidden = hidden_states.shape
        local_num_experts = up[0].shape[0] if local_num_experts is None else local_num_experts
        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=hidden_states.device)
        for local_e in range(local_num_experts):
            tok, slot = (topk_ids == local_expert_offset + local_e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            r = torch.relu(_project(hidden_states[tok], up, local_e))
            act = quantize_nvfp4(r * r, fc2_global_scale) / fc2_global_scale
            y = _project(act, down, local_e)
            out.index_add_(0, tok, y * topk_weights[tok, slot].float().unsqueeze(1))
        return out

    def is_valid(
        self,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool,
        act_type: int = ACT_TYPE_RELU2,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> None:
        """The shared runner's guards, plus the FC1 row count against `act_type`.

        Gated or not is read off FC1's row count -- 2 * I rows is a gated [up; gate] pair,
        I rows a single projection -- while the kernel that runs is chosen by `act_type`,
        and the op checks only that the rows are one or the other. Relu2 handed a 2I-row
        SwiGlu stack walks it at I-row strides and returns plausible rows from the wrong
        weights (measured on the source branch); SwiGlu handed an I-row stack would read 2I
        rows per expert, past the end of the stack.
        """
        assert act_type == ACT_TYPE_RELU2, (
            f"act_type {act_type}: this op certifies Relu2; SwiGlu is the shared runner's"
        )
        rows = gemm1_weights.shape[1]
        expected = (2 if act_type in _GATED_ACT_TYPES else 1) * intermediate_size
        assert rows not in (intermediate_size, 2 * intermediate_size) or rows == expected, (
            f"act_type {act_type} needs FC1 with {expected} rows (intermediate_size "
            f"{intermediate_size}), got {rows}: the op accepts either count and answers wrongly"
        )
        fp4_block_scale_moe_runner.is_valid(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm1_bias,
            gemm1_alpha,
            gemm1_beta,
            gemm1_clamp_limit,
            gemm2_weights,
            gemm2_weights_scale,
            gemm2_bias,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            routing_method_type,
            do_finalize,
            act_type,
            topk_weights,
            topk_ids,
            output,
            tune_max_num_tokens,
            use_dp,
        )


fp4_block_scale_moe_relu2 = Fp4BlockScaleMoeRelu2()

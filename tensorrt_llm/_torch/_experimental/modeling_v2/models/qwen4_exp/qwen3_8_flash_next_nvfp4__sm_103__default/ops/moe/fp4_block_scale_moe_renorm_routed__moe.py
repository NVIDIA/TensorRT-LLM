# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 MoE layer over one rank's expert window, entered with router logits: the
trtllm-gen runner's own Renormalize routing (top-k over the raw logits of the whole
routing space, softmax over those k) + grouped FC1 GEMM + SwiGLU + NVFP4
requantization + grouped FC2 GEMM + routing-weighted combine, in one trtllm-gen call.

It runs over an expert window (`local_expert_offset`, `local_num_experts`) at the
intermediate width a rank holds; the tactic is the AutoTuner's, as on the pre-routed
`fp4_block_scale_moe_runner`."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner as _pre_routed,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op as _renorm,
)

#: `RoutingMethodType::Renormalize`: NoOp preprocess, top-k, softmax over the k.
ROUTING_RENORMALIZE = 1
#: `act_type` SwiGlu.
ACT_SWIGLU = 0

#: The (num_experts, top_k) pairs this entry certifies. The Renormalize routing policy
#: dispatches on a compiled tier list, and a pair outside every tier is a logged error
#: with no exception and **no routing launch** -- the GEMMs then read stale routing
#: buffers. So the guard admits what was measured, not what the tier list might cover.
CERTIFIED_PAIRS = frozenset({(512, 10)})


def _layer(tokens: int, off: int, local: int, inter: int = 640) -> dict:
    """qwen4_exp's routed geometry (512 experts, top-10, H 2560, bf16 logits) at one
    rank's window `[off, off + local)` and intermediate width `inter`."""
    return dict(
        tokens=tokens,
        hidden=2560,
        intermediate=inter,
        num_experts=512,
        top_k=10,
        local_expert_offset=off,
        local_num_experts=local,
        logits_dtype=torch.bfloat16,
        dist="randn",
    )


class Fp4BlockScaleMoeRenormRouted(OpWrapper):
    """One NVFP4 MoE layer whose routing happens inside the call, over this rank's
    expert window.

    `routing_logits` is the router GEMM's `[T, num_experts]` output over the whole
    routing space. Per token the kernel picks the `top_k` largest logits (ties to
    the LOWER expert index) and weights them by an fp32 softmax over just those
    `top_k`, rounded to bf16; slots routed outside
    `[local_expert_offset, local_expert_offset + local_num_experts)` add nothing.
    The same function as `renorm_moe_routing_op` followed by the pre-routed
    `fp4_block_scale_moe_runner` over the same window, and bitwise equal to that
    composition at every cell. Returns the combined `[T, H]` bf16 rows (this
    window's partial).

    Fixed, not arguments: no routing bias, Renormalize routing, SwiGlu,
    `do_finalize=True`. The weight layout (row shuffle + 128x4 scale swizzle), the
    window's weight stacks and the three per-expert scalars are exactly the
    pre-routed entry's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The default target's call shapes: tep4 / dep4 a 128-expert window at I 640,
    # dep2 a 256-expert window, tp4 all 512 experts at its 160-wide slice padded to
    # 256. Rows: 1 (decode), 4 (the ll MTP-3 verify step), 5 (the first count past the
    # block routing kernel), 128 (dep4's 4 x 32 gathered rows, tep4's 128 requests),
    # 1024 (MTP-7 verify of 128 requests) and 8192 (a prefill chunk). Offsets cover
    # every rank.
    CELLS: tuple[Cell, ...] = (
        Cell(why="tep4 decode: one token, rank 0's 128-expert window", spec=_layer(1, 0, 128)),
        Cell(why="tep4 MTP-3 verify at con1: 4 rows, rank 1's window", spec=_layer(4, 128, 128)),
        Cell(why="ep4 at 5 rows, rank 2's window: past the block kernel", spec=_layer(5, 256, 128)),
        Cell(
            why="dep4 gathered decode 4 x 32 = 128 rows, rank 3's window",
            spec=_layer(128, 384, 128),
        ),
        Cell(why="tep4 MTP-7 verify of 128 requests: 1024 rows", spec=_layer(1024, 0, 128)),
        Cell(why="ep4 prefill chunk at 8192 rows, rank 3's window", spec=_layer(8192, 384, 128)),
        Cell(why="tp4 MTP-3 verify: 4 rows, 512 experts at I 256", spec=_layer(4, 0, 512, 256)),
        Cell(why="tp4 at 128 rows, 512 experts at I 256", spec=_layer(128, 0, 512, 256)),
        Cell(
            why="tp4 prefill chunk 8192 rows, 512 experts at I 256", spec=_layer(8192, 0, 512, 256)
        ),
        Cell(
            why="dep2 gathered decode: 2 rows, rank 1's 256-expert window", spec=_layer(2, 256, 256)
        ),
        Cell(
            why="dep2 prefill chunk at 8192 rows, rank 0's 256-expert window",
            spec=_layer(8192, 0, 256),
        ),
    )

    note = """
    Routing semantics, against the composition `renorm_moe_routing_op` ->
    pre-routed `fp4_block_scale_moe_runner` over the same window: bitwise-equal
    finalized output at every cell (the test re-drives this). The routing kernel
    ranks the whole `[T, num_experts]` row and keeps only the window's slots for
    the permutation, as the pre-routed path does with caller ids; both rank by a
    packed (twiddled value, 65535 - index) key (ties to the lower expert index),
    take an fp32 `expf` softmax over the k and round the weight to bf16 once.

    Kernel choice inside the call is by row count (the block routing kernel up to
    a small count, single-cluster up to the cluster cap, then the multi-kernel
    histogram path); every one of them shares the top-k and softmax code.
    `routing_bias` is None (the thop defaults the bias dtype to bf16 and never
    reads it on this policy). PDL (`TRTLLM_ENABLE_PDL`) is read by the routing
    launch. The tactic is the AutoTuner's (`torch.ops.trtllm.fp4_block_scale_moe_runner`),
    which profiles the logits-routed call with its own dummy logits and keys its
    cache on the window and intermediate width.

    Silent, refused by `is_valid`: a non-contiguous operand (raw pointers, dense
    layout assumed), a (num_experts, top_k) pair outside `CERTIFIED_PAIRS` (a
    tier-list miss logs and launches nothing; the test holds this refusal), and a
    weight stack that is not the window's (the op compares only the scales' expert
    dimension with `local_num_experts`; the whole stack beside a window's scales is
    read as its first `local_num_experts` experts, ~360 bf16 ulp RMS off).

    Not certified: fp32 logits (the runner's other accepted logits dtype); windows
    of the padded I 256 stack (tp4 does not split experts); rows past 8192 (the
    caller chunks at `MAX_CERTIFIED_TOKENS`).

    Hard limits
      T            certified 1..8192 (cells 1, 2, 4, 5, 128, 1024, 8192)
      num_experts  512, top_k 10 (CERTIFIED_PAIRS)
      window       128 of 512 at I 640 (every offset), 256 of 512 at I 640, 512 at I 256
    """

    def __call__(
        self,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.fp4_block_scale_moe_runner(
            routing_logits,
            None,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            None,
            None,
            None,
            None,
            gemm2_weights,
            gemm2_weights_scale,
            None,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            num_experts,
            top_k,
            None,
            None,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            None,
            ROUTING_RENORMALIZE,
            True,
            ACT_SWIGLU,
        )[0]

    def reference(
        self,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        top_k: int,
        up: tuple,
        gate: tuple,
        down: tuple,
        fc2_global_scale: float,
        local_expert_offset: int = 0,
        local_num_experts: int | None = None,
    ) -> torch.Tensor:
        """The routing, then the window's layer, both by their own entries' references.

        Routing: `renorm_moe_routing_op.reference` (stable descending sort in the
        logits' dtype, fp32 softmax over the k, bf16 weights -- the dtype the runner's
        routing writes) over the whole routing space. The layer: the pre-routed
        runner entry's fp32 reference over the window's dequantized
        `(codes, scales)` per-expert operands (`hidden_states` is the exact fp32 value
        of the NVFP4 activation). Returns fp32 `[T, H]`.
        """
        ids, weights = _renorm.reference(routing_logits, top_k, torch.bfloat16)
        return _pre_routed.reference(
            hidden_states,
            ids,
            weights,
            up,
            gate,
            down,
            fc2_global_scale,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
        )

    def is_valid(
        self,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
    ) -> None:
        # Raw data pointers, dense row-major assumed: a strided view is read as if
        # dense. A router GEMM output written into a wider buffer is the realistic
        # way to get here.
        for name, tensor in (
            ("routing_logits", routing_logits),
            ("hidden_states", hidden_states),
            ("hidden_states_scale", hidden_states_scale),
            ("gemm1_weights", gemm1_weights),
            ("gemm1_weights_scale", gemm1_weights_scale),
            ("gemm2_weights", gemm2_weights),
            ("gemm2_weights_scale", gemm2_weights_scale),
            ("output1_scale_scalar", output1_scale_scalar),
            ("output1_scale_gate_scalar", output1_scale_gate_scalar),
            ("output2_scale_scalar", output2_scale_scalar),
        ):
            assert tensor.is_contiguous(), (
                f"{name} must be contiguous; a strided view is read as if dense "
                "and silently produces wrong results"
            )
        # A tier-list miss in the Renormalize routing policy logs an error and
        # launches nothing; the GEMMs then consume stale routing buffers.
        assert (num_experts, top_k) in CERTIFIED_PAIRS, (
            f"(num_experts, top_k) = ({num_experts}, {top_k}) is not a certified "
            f"Renormalize routing pair {sorted(CERTIFIED_PAIRS)}"
        )
        # Same layout rule as the pre-routed entry: the prepared block-scale layout
        # is silently misread below a hidden multiple of 256.
        assert gemm1_weights.shape[-1] * 2 % 256 == 0, (
            "hidden size (gemm1_weights.shape[-1] * 2) must be a multiple of 256"
        )
        # The op checks the scales' expert dimension, not the weights'.
        for name, tensor in (("gemm1_weights", gemm1_weights), ("gemm2_weights", gemm2_weights)):
            assert tensor.shape[0] == local_num_experts, (
                f"{name} holds {tensor.shape[0]} experts but local_num_experts is "
                f"{local_num_experts}; pass the window's own weights"
            )


fp4_block_scale_moe_renorm_routed = Fp4BlockScaleMoeRenormRouted()

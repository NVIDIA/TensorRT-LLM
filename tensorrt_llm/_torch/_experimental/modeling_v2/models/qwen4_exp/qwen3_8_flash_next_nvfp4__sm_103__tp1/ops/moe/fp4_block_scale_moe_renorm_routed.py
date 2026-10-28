# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 MoE layer entered with router logits: the trtllm-gen runner's own
Renormalize routing (top-k over the raw logits, softmax over those k) + grouped
FC1 GEMM + SwiGLU + NVFP4 requantization + grouped FC2 GEMM + routing-weighted
combine, in one trtllm-gen call."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    fp4_block_scale_moe_runner as _pre_routed,
)

from .renorm_moe_routing_op import renorm_moe_routing_op as _renorm

#: `RoutingMethodType::Renormalize`: NoOp preprocess, top-k, softmax over the k.
ROUTING_RENORMALIZE = 1
#: `act_type` SwiGlu.
ACT_SWIGLU = 0

#: The (num_experts, top_k) pairs this entry certifies. The Renormalize routing
#: policy dispatches on a compiled tier list, and a pair outside every tier is a
#: logged error with no exception and **no routing launch** -- the GEMMs then
#: read stale routing buffers. So the guard admits what was measured, not what
#: the tier list might also cover.
CERTIFIED_PAIRS = frozenset({(512, 10)})


def _ckpt(tokens: int, dist: str = "randn") -> dict:
    """qwen4_exp's routed geometry: 512 experts, top-10, H 2560 / I 640."""
    return dict(
        tokens=tokens,
        hidden=2560,
        intermediate=640,
        num_experts=512,
        top_k=10,
        logits_dtype=torch.bfloat16,
        dist=dist,
    )


class Fp4BlockScaleMoeRenormRouted(OpWrapper):
    """One NVFP4 MoE layer whose routing happens inside the call.

    `routing_logits` is the router GEMM's `[T, num_experts]` output. Per token
    the kernel picks the `top_k` largest logits (ties to the LOWER expert index)
    and weights them by an fp32 softmax over just those `top_k`, rounded to
    bf16 -- the same function as `ops/moe/renorm_moe_routing_op` followed by the
    pre-routed `catalog/moe/fp4_block_scale_moe_runner`, and bitwise equal to
    that composition at every cell. Returns the combined `[T, H]` bf16 rows.

    Fixed, not arguments: no routing bias, Renormalize routing, SwiGlu, no
    expert parallelism (`local_num_experts == num_experts`), `do_finalize=True`.
    The weight layout (row shuffle + 128x4 scale swizzle) and the three
    per-expert scalars are exactly the pre-routed entry's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # One cell per routing kernel the call picks by row count: block, single-cluster
    # and the multi-kernel histogram path.
    CELLS: tuple[Cell, ...] = (
        Cell(why="_moe_block decode: one token; the block routing kernel", spec=_ckpt(1)),
        Cell(why="128 rows: the con=128 decode batch; the cluster routing kernel", spec=_ckpt(128)),
        Cell(
            why="16384 rows: max_num_tokens of the tuned config; past the 8192 tuning bucket",
            spec=_ckpt(16384),
        ),
    )

    note = """
    Routing semantics, measured against the composition it replaces
    (`renorm_moe_routing_op` -> pre-routed runner): bitwise-equal finalized
    output at every cell (the test re-drives this), and at bring-up on randn /
    coarse / large-magnitude / all-equal logits, bf16 and fp32 logits,
    1..16384 rows. Both kernels rank by a packed
    (twiddled value, 65535 - index) key (ties to the lower expert index), take
    an fp32 `expf` softmax over the k with the same 32-lane warp reduction, and
    round the weight to bf16 once.

    Kernel choice inside the call is by row count (block kernel up to a small
    count, single-cluster up to the cluster cap, then the multi-kernel
    histogram path); every one of them shares the top-k and softmax code.
    `routing_bias` is None (the thop defaults the bias dtype to bf16 and never
    reads it on this policy). PDL (`TRTLLM_ENABLE_PDL`) is read by the routing
    launch.

    Silent, refused by `is_valid`: a non-contiguous operand (raw pointers,
    dense layout assumed) and a (num_experts, top_k) pair outside
    `CERTIFIED_PAIRS` (a tier-list miss logs and launches nothing; the test
    holds this refusal).

    Not certified: fp32 logits (the runner's other accepted logits dtype).

    Hard limits
      T            certified at 1, 128 and 16384 (max_num_tokens of the tuned config).
                   Autotuner buckets stop at 8192, the op's
                   `tune_max_num_tokens` default, which this entry does not
                   pass: a call past 8192 rows runs the 8192 bucket's tactic
      num_experts  512, top_k 10 (CERTIFIED_PAIRS)
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
            0,
            num_experts,
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
    ) -> torch.Tensor:
        """The routing, then the layer, both by their own entries' references.

        Routing: `renorm_moe_routing_op.reference` (stable descending sort in the
        logits' dtype, fp32 softmax over the k, bf16 weights -- the dtype the
        runner's routing writes). The layer: the pre-routed runner entry's fp32
        reference over the dequantized `(codes, scales)` per-expert operands, as
        that entry documents (`hidden_states` is the exact fp32 value of the
        NVFP4 activation). Returns fp32 `[T, H]`.
        """
        ids, weights = _renorm.reference(routing_logits, top_k, torch.bfloat16)
        return _pre_routed.reference(hidden_states, ids, weights, up, gate, down, fc2_global_scale)

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
    ) -> None:
        # Raw data pointers, dense row-major assumed: a strided view is read as
        # if dense. A router GEMM output written into a wider buffer is the
        # realistic way to get here.
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
        # Same layout rule as the pre-routed entry: the prepared block-scale
        # layout is silently misread below a hidden multiple of 256.
        assert gemm1_weights.shape[-1] * 2 % 256 == 0, (
            "hidden size (gemm1_weights.shape[-1] * 2) must be a multiple of 256"
        )


fp4_block_scale_moe_renorm_routed = Fp4BlockScaleMoeRenormRouted()

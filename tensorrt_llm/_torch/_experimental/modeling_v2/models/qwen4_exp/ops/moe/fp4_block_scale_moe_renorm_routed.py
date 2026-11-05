# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 MoE layer entered with router logits: the trtllm-gen runner's own
Renormalize routing (top-k over the raw logits, softmax over those k) + grouped
FC1 GEMM + SwiGLU + NVFP4 requantization + grouped FC2 GEMM + routing-weighted
combine, in one trtllm-gen call.

It runs over an expert window (`local_expert_offset`, `local_num_experts`; the whole routing
space by default) at the intermediate width a rank holds: the window of a tep4 / dep4 / dep2
rank, every expert at a tp4 rank's padded slice, every expert at tp1. A decode call over the
whole 512-expert layer at I 640 runs its pinned tactic (`DECODE_TACTICS`), every other call the
AutoTuner's."""

from __future__ import annotations

from functools import cache

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    fp4_block_scale_moe_runner as _pre_routed,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op as _renorm,
)
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
from tensorrt_llm._torch.utils import last_positive_power_of_2
from tensorrt_llm.logger import logger

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

#: The geometry `DECODE_TACTICS` was measured on: (num_experts, top_k, hidden, intermediate).
DECODE_TACTICS_GEOMETRY = (512, 10, 2560, 640)

#: trtllm-gen tactic `(tileN, FC1 x FC2 config index)` per AutoTuner bucket (the row count
#: rounded down to a power of two) for the decode row counts, at `DECODE_TACTICS_GEOMETRY`.
#: Each is the fastest valid tactic of its bucket on GB300 over a 48-layer chain of distinct
#: expert stacks (HBM-cold weights, as a decode step streams them), with router logits whose
#: distinct experts per layer match serving's routing skew, in one CUDA graph with PDL. The
#: tactics compared per bucket (the fastest few, the runner's default, the AutoTuner's picks)
#: all gave bitwise the same output, so the choice moves time only; the entry's test holds
#: each pinned tactic bitwise to the AutoTuner path.
#:
#: A call outside AutoTuner tuning mode whose bucket is listed runs its tactic directly; any
#: other call goes through `torch.ops.trtllm.fp4_block_scale_moe_runner` (the AutoTuner), so
#: the larger buckets are still profiled at warmup. The config index addresses the runner's
#: compiled cubin list: a tactic the runner does not list as valid is not run; its bucket goes
#: through the AutoTuner instead, with one warning.
DECODE_TACTICS: dict[int, tuple[int, int]] = {
    1: (8, 55),
    2: (8, 53),
    4: (8, 44),
    8: (8, 0),
    16: (8, 44),
    32: (8, 44),
    64: (16, 38),
    128: (16, 36),
    256: (32, 32),
    512: (32, 32),
    1024: (64, 5),
}


@cache
def _runner():
    """The SwiGlu trtllm-gen runner object the AutoTuner path also uses (one per act type;
    the other constructor arguments only shape the AutoTuner's tuning config)."""
    num_experts, top_k, _, intermediate = DECODE_TACTICS_GEOMETRY
    return FP4BlockScaleMoERunner(
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=ROUTING_RENORMALIZE,
        do_finalize=True,
        act_type=ACT_SWIGLU,
    ).get_runner()


@cache
def _checked_tactic(bucket: int) -> tuple[int, int] | None:
    """`DECODE_TACTICS[bucket]` if it is a valid config for the bucket's row count, else None
    (the AutoTuner path), warned once per bucket."""
    tactic = DECODE_TACTICS[bucket]
    num_experts, top_k, hidden, intermediate = DECODE_TACTICS_GEOMETRY
    valid = _runner().get_valid_configs(top_k, hidden, intermediate, num_experts, bucket)
    if list(tactic) not in [list(t) for t in valid]:
        logger.warning(
            f"pinned tactic {tactic} for {bucket} rows is not a valid config of this build's "
            "trtllm-gen FP4 MoE runner; using the AutoTuner for this bucket. Re-measure "
            "DECODE_TACTICS."
        )
        return None
    return tactic


def _pinned_tactic(
    num_tokens: int, num_experts: int, top_k: int, hidden: int, intermediate: int
) -> tuple[int, int] | None:
    """The pinned tactic for this call, or None for the AutoTuner path."""
    if (num_experts, top_k, hidden, intermediate) != DECODE_TACTICS_GEOMETRY:
        return None
    bucket = last_positive_power_of_2(num_tokens)
    if bucket not in DECODE_TACTICS or AutoTuner.get().is_tuning_mode:
        return None
    return _checked_tactic(bucket)


def run_renorm_moe(
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
    do_finalize: bool,
    local_expert_offset: int = 0,
    local_num_experts: int | None = None,
) -> list[torch.Tensor]:
    """One logits-routed trtllm-gen FP4 MoE call (Renormalize, SwiGlu) over the expert window
    `[local_expert_offset, local_expert_offset + local_num_experts)` (all experts when
    `local_num_experts` is None): `[out]` with `do_finalize`, else `[rows, topk_weights,
    topk_rows]`. A `DECODE_TACTICS` bucket of the whole layer runs its pinned tactic; any other
    call is the AutoTuner's."""
    if local_num_experts is None:
        local_num_experts = num_experts
    whole = local_expert_offset == 0 and local_num_experts == num_experts
    tactic = (
        _pinned_tactic(
            hidden_states.shape[0],
            num_experts,
            top_k,
            hidden_states.shape[1] * 2,
            intermediate_size,
        )
        if whole
        else None
    )
    if tactic is None:
        return _pre_routed(
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
            do_finalize,
            ACT_SWIGLU,
        )
    return _runner().run_moe(
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
        do_finalize,
        list(tactic),
        None,
        None,
        None,
    )


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


def _layer(tokens: int, off: int, local: int, inter: int = 640) -> dict:
    """qwen4_exp's routed geometry (512 experts, top-10, H 2560, bf16 logits) at one rank's
    window `[off, off + local)` and intermediate width `inter`."""
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
    """One NVFP4 MoE layer whose routing happens inside the call, over an expert window.

    `routing_logits` is the router GEMM's `[T, num_experts]` output over the whole routing
    space. Per token the kernel picks the `top_k` largest logits (ties to the LOWER expert
    index) and weights them by an fp32 softmax over just those `top_k`, rounded to bf16;
    slots routed outside `[local_expert_offset, local_expert_offset + local_num_experts)` add
    nothing. The same function as `ops/moe/renorm_moe_routing_op` followed by the pre-routed
    `catalog/moe/fp4_block_scale_moe_runner` over the same window, and bitwise equal to that
    composition at every cell. Returns the combined `[T, H]` bf16 rows (the window's partial).

    Fixed, not arguments: no routing bias, Renormalize routing, SwiGlu, `do_finalize=True`.
    The weight layout (row shuffle + 128x4 scale swizzle), the window's weight stacks and the
    three per-expert scalars are exactly the pre-routed entry's.
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
        # The windows: tep4 / dep4 a 128-expert window at I 640, dep2 a 256-expert window, tp4
        # all 512 experts at its 160-wide slice padded to 256. Rows: 1 (decode), 4 (an MTP-3
        # verify step), 5 (past the block routing kernel), 128 (dep4's 4 x 32 gathered rows),
        # 1024 (MTP-7 verify of 128 requests) and 8192 (a prefill chunk); offsets cover every
        # rank.
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
    Routing semantics, measured against the two-entry composition
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

    Tactics: a call of 1..2047 rows outside AutoTuner tuning mode runs the
    `DECODE_TACTICS` tactic of its power-of-two bucket (measured HBM-cold with
    serving's routing skew); the AutoTuner, which profiles one layer's experts
    warm in L2 under uniform routing, takes the rest and the warmup's tuning
    pass. The test holds each pinned tactic bitwise to the AutoTuner path.

    Windows: a window's weight stacks are its own experts only (the op compares the scales'
    expert dimension with `local_num_experts`; the whole stack beside a window's scales would be
    read as its first experts, refused by `is_valid`; the windows test holds this refusal).
    Not certified: windows of the padded I 256 stack (tp4 does not split experts); window rows
    past 8192 (the caller chunks).

    Hard limits
      window       the whole 512 at I 640 (tp1), 128 of 512 at I 640 (every offset), 256 of
                   512 at I 640, 512 at I 256
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
        local_expert_offset: int = 0,
        local_num_experts: int | None = None,
    ) -> torch.Tensor:
        return run_renorm_moe(
            routing_logits,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            num_experts,
            top_k,
            intermediate_size,
            True,
            local_expert_offset,
            local_num_experts,
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
        logits' dtype, fp32 softmax over the k, bf16 weights -- the dtype the
        runner's routing writes). The layer: the pre-routed runner entry's fp32
        reference over the dequantized `(codes, scales)` per-expert operands, as
        that entry documents (`hidden_states` is the exact fp32 value of the
        NVFP4 activation). Returns fp32 `[T, H]`.
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
        local_expert_offset: int = 0,
        local_num_experts: int | None = None,
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
        # The op checks the scales' expert dimension, not the weights'.
        local = num_experts if local_num_experts is None else local_num_experts
        for name, tensor in (("gemm1_weights", gemm1_weights), ("gemm2_weights", gemm2_weights)):
            assert tensor.shape[0] == local, (
                f"{name} holds {tensor.shape[0]} experts but the window has {local}; pass the "
                "window's own weights"
            )


fp4_block_scale_moe_renorm_routed = Fp4BlockScaleMoeRenormRouted()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 sharded MoE front (router + latent_down + top-16 + MXFP8, optional shared gate_up + SiTU)
through the family's ``moe_front`` kernel."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.moe import moe_front as _front
else:  # pragma: no cover - the op is SM10x + CuTe DSL only
    _front = None

#: Model hidden size and experts the kernel is built for (Kimi K3.1).
HIDDEN = 7168
NUM_EXPERTS = 896
#: Rows of one front call (the head), and rows up to which it also computes the shared columns.
MAX_ROWS = 32
SHARED_MAX_ROWS = 16
#: Kimi K3.1 shared experts: 2 x 3072 intermediate columns (gate and up each), SiTU soft-caps.
SHARED_INTERMEDIATE = 6144


def _cell(world: int, rows: int, why: str, shared: bool = False) -> Cell:
    return Cell(
        why=why,
        spec=dict(
            world=world,
            rows=rows,
            shared_cols=SHARED_INTERMEDIATE // world if shared else 0,
            scale=2.5,
        ),
    )


class MoeFront(OpWrapper):
    """The MoE input's routing and routed-expert input, computed tensor-parallel on one kernel.

    Every rank of the attention-TP group holds the same ``x [M, 7168]`` BF16. Rank ``r`` of ``W``
    multiplies it by its 1/W slice of the BF16 latent_down weight (3584 / W rows) and of the BF16
    router weight (896 / W rows) -- ``front_weight`` packs both, plus optionally this rank's shared
    gate_up rows, into ``front_weight`` -- multicasts the slices into ``workspace`` (an MNNVL
    multicast buffer of the TP group) and, per token, takes the top-16 of the gathered FP32 logits
    and MXFP8-quantizes the gathered (BF16-rounded) latent.

    Returns ``(topk_ids int32 [M, 16], topk_weights bf16 [M, 16], x_fp8 e4m3 [M, 3584],
    x_sf uint8 [M, 112], shared_act bf16 [M, shared_cols] or None)``: the tuple the TRTLLM-Gen MXFP4 x
    MXFP8 runner takes as a precomputed routing (``kimi_k3_noaux_tc_mxfp8_quant``'s outputs, linear
    scale layout; the ``routed`` tuple the routed experts take), and with ``shared_cols`` the shared
    expert's ``SiTU(gate) * SiTU_lin(up)`` of this rank's TP shard of the shared gate_up. Identical bytes on every rank.

    Not bitwise against the replicated path: the logits and the latent are FP32 sums over K in
    another order (split over ranks and eight K slices), so an MXFP8 byte can be one step apart
    where the BF16 latent rounds the other way, and a top-16 can flip on a near-tie of the biased
    scores. Given the same logits and latent the selection and the weights are the TRTLLM-Gen runner's
    in-kernel routing (what the target runs at attention TP), bit for bit -- not the fused C++ op's,
    whose FP64-rounded weights differ from the runner's in about 1 of 10^5 -- and the quantization is
    ``kimi_k3_noaux_tc_mxfp8_quant``'s.
    """

    ARCHS = frozenset({Arch.SM_103})

    # One GB300 tray is four GPUs, so the driven cells are W = 4; the deployed widths (16, 32) use
    # the same kernel with smaller head slices.
    CELLS: tuple[Cell, ...] = (
        _cell(4, 1, "one decode row"),
        _cell(4, 6, "TEP16 low-latency verify shape: one request x 6 rows"),
        _cell(4, 8, "TEP32 B1 verify (8 rows): the 8-slot launch"),
        _cell(4, 14, "TEP16 mid verify (~2.3 requests x 6 rows): a 16-slot launch"),
        _cell(4, 24, "TEP16 B4 verify (24 rows): two token halves in one launch"),
        _cell(4, 32, "the front's row limit (two halves of 16 slots)"),
        _cell(4, 8, "TEP32 B1 with the shared expert's gate_up + SiTU", shared=True),
        _cell(4, 16, "the shared columns' row limit", shared=True),
    )

    note = """
    Collective: every rank of the TP group calls it with the same x, in the same order, with the
    workspace of that group (``front_workspace``, created once per mapping outside CUDA-graph
    capture: it allocates the MNNVL multicast buffer and synchronizes the ranks). Each call
    alternates between the workspace's two buffers. The kernel compiles once per (row count
    bucket, shared columns, soft-caps) and must have run once per configuration before a CUDA graph
    captures it (``is_compiled``). Launched with PDL unless TRTLLM_ENABLE_PDL=0; its GEMV CTAs
    trigger the dependent grid (the routed experts' FC1) early.

    Rows: 1..32 (one launch per 32 rows, 8 or 16 MMA token slots, up to two token halves), shared
    columns up to 16 rows, TP worlds 4 / 8 / 16 / 32 (``supported``). The TEP target runs it at attention
    TP 8 / 16 / 32, the shared part at TP 32 (``_K31_FRONT_SHARED_TP``).
    Shared columns must be a multiple of 64 (6144 / W: 1536 at W = 4, 384 at 16, 192 at 32).
    """

    def __call__(
        self,
        x: torch.Tensor,
        front_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        workspace: dict,
        shared_cols: int,
        gate_cap: float,
        linear_cap: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        ids, weights, x_fp8, x_sf, shared, _ = _front.run_moe_front(
            x,
            front_weight,
            e_score_correction_bias,
            float(routed_scaling_factor),
            workspace,
            shared_cols=shared_cols,
            gate_cap=float(gate_cap),
            linear_cap=float(linear_cap),
        )
        return ids, weights, x_fp8, x_sf, shared

    def reference(
        self,
        x: torch.Tensor,
        latent_down_weight: torch.Tensor,
        router_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        shared_gate_up_shard: Optional[torch.Tensor],
        gate_cap: float,
        linear_cap: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """The replicated path the target runs today, on every rank.

        The documented collective exception to mirroring ``__call__``: it takes the unsharded
        weights instead of this rank's packed slice and no workspace. FP32 router logits through
        ``dsv3_router_gemm_op``, the BF16 latent through ``dsv3_fused_a_gemm_op`` (the latent_down
        projection), ``kimi_k3_noaux_tc_mxfp8_quant`` on both; with ``shared_gate_up_shard``
        (this rank's ``[2 I, 7168]`` gate rows then up rows) the SiTU of its product, FP32 math
        rounded to BF16.
        """
        trt = torch.ops.trtllm
        logits = trt.dsv3_router_gemm_op(x, router_weight.t(), None, torch.float32)
        latent = trt.dsv3_fused_a_gemm_op(x, latent_down_weight.t(), None, None)
        ids, weights, x_fp8, x_sf = trt.kimi_k3_noaux_tc_mxfp8_quant(
            logits, e_score_correction_bias, latent, float(routed_scaling_factor)
        )
        shared = None
        if shared_gate_up_shard is not None:
            with true_fp32_matmul():
                gu = (x.float() @ shared_gate_up_shard.float().t()).to(torch.bfloat16).float()
            inter = gu.shape[1] // 2
            gate, up = gu[:, :inter], gu[:, inter:]
            g = gate_cap * torch.tanh(gate / gate_cap) * torch.sigmoid(gate)
            u = linear_cap * torch.tanh(up / linear_cap)
            shared = (g * u).to(torch.bfloat16)
        return ids, weights, x_fp8, x_sf.view(torch.uint8).reshape(x.shape[0], -1), shared

    def is_valid(
        self,
        x: torch.Tensor,
        front_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        workspace: dict,
        shared_cols: int,
        gate_cap: float,
        linear_cap: float,
    ) -> None:
        world = workspace["world"]
        assert x.dim() == 2 and x.shape[1] == HIDDEN and x.dtype == torch.bfloat16, (
            f"x must be BF16 [M, {HIDDEN}]"
        )
        assert 1 <= x.shape[0] <= MAX_ROWS, f"1 <= M <= {MAX_ROWS}"
        assert shared_cols == 0 or x.shape[0] <= SHARED_MAX_ROWS, (
            f"the shared columns serve at most {SHARED_MAX_ROWS} rows"
        )
        assert e_score_correction_bias.dtype == torch.float32 and (
            e_score_correction_bias.numel() == NUM_EXPERTS
        ), f"the bias must be FP32 [{NUM_EXPERTS}]"
        head = _front.head_tiles(world) * _front.CTA_M
        assert front_weight.dtype == torch.bfloat16 and front_weight.shape[1] == HIDDEN, (
            "front_weight must be the BF16 weight front_weight() packs"
        )
        assert front_weight.shape[0] >= head + 2 * shared_cols, (
            "front_weight lacks the shared gate_up rows this call reads"
        )
        assert supported(world, shared_cols, x.device), (
            f"world {world} with {shared_cols} shared columns does not fit the kernel's geometry"
        )


def supported(world: int, shared_cols: int, device: torch.device) -> bool:
    """Whether the kernel serves this TP world and shared-column count on ``device``."""
    if _front is None:
        return False
    return _front.weight_supported(world, shared_cols, HIDDEN, device)


def front_workspace(mapping) -> dict:
    """The TP group's head workspace (MNNVL multicast), created on first use per mapping.

    Collective: every rank of ``mapping``'s TP group calls it at the same point, outside CUDA-graph
    capture."""
    return _front.head_workspace(mapping)


def front_weight(
    latent_down_weight: torch.Tensor,
    router_weight: torch.Tensor,
    rank: int,
    world: int,
    shared_gate_up_shard: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """This rank's packed front weight from the unsharded BF16 latent_down ``[3584, 7168]`` and router
    ``[896, 7168]`` weights (and this rank's shared gate_up shard ``[2 I, 7168]``: gate rows, then up
    rows). Build it once after weight loading (~5 MB per layer at W = 16 without the shared rows)."""
    head = _front.head_weight(latent_down_weight, router_weight.to(torch.bfloat16), rank, world)
    return _front.front_weight(head, shared_gate_up_shard)


def is_compiled(
    num_tokens: int,
    world: int,
    shared_cols: int,
    device: torch.device,
    gate_cap: float,
    linear_cap: float,
    runner_sort: bool = False,
) -> bool:
    """Whether every launch of a ``num_tokens``-row call (with ``runner_sort``: of
    :func:`moe_front_runner_sort`) is compiled (a CUDA graph may capture it)."""
    return _front.is_compiled(
        num_tokens,
        HIDDEN,
        world,
        shared_cols,
        device,
        gate_cap=gate_cap,
        linear_cap=linear_cap,
        do_sort=runner_sort,
    )


def runner_sort_fits(num_tokens: int, local_num_experts: int, tile: int) -> bool:
    """Whether :func:`moe_front_runner_sort` serves ``num_tokens`` rows over ``local_num_experts`` local experts
    with token tiles of ``tile`` rows."""
    return _front is not None and _front.runner_sort_fits(num_tokens, local_num_experts, tile)


def moe_front_runner_sort(
    x: torch.Tensor,
    front_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    workspace: dict,
    shared_cols: int,
    gate_cap: float,
    linear_cap: float,
    tile: int,
    local_expert_offset: int,
    local_num_experts: int,
) -> tuple:
    """``moe_front`` that also writes the TRTLLM-Gen MXFP4 runner's routing of this rank's local experts:
    ``(ids, weights, x_fp8, x_sf, shared, prerouted)`` with ``prerouted`` = ``[expanded_idx_to_permuted_idx,
    permuted_idx_to_token_idx, total_num_padded_tokens, tile_idx_to_expert_idx, tile_idx_to_mn_limit,
    num_non_exiting_tiles]`` for token tiles of ``tile`` rows (the layout ``k31_w4a8_moe_cute`` takes as
    ``prerouted``). The first five outputs are ``moe_front``'s, bit for bit."""
    return _front.run_moe_front(
        x,
        front_weight,
        e_score_correction_bias,
        float(routed_scaling_factor),
        workspace,
        shared_cols=shared_cols,
        gate_cap=float(gate_cap),
        linear_cap=float(linear_cap),
        sort=(int(tile), int(local_expert_offset), int(local_num_experts), NUM_EXPERTS),
    )


moe_front = MoeFront()

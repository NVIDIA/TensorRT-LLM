# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE finalize + multicast all-reduce + residual + RMSNorm (+ MXFP8 copy) of the decode MoE boundary, 4 ranks."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import OpWrapper as OpWrapper

from ....ops.comm.reference import McFinalizeAllreduceNormReference
from ....ops.comm.reference import boundary as boundary
from ....ops.comm.reference import finalize_partial as finalize_partial


class M3McFinalizeAllreduceNorm(McFinalizeAllreduceNormReference):
    """The MoE boundary with the routed experts' finalize folded in.

    Per rank the partial is ``bf16(bf16(sum_k scales[t, k] * fc2[perm[t, k]]) + shared[t])`` (``perm`` -1 = dropped,
    ``scales`` fp32 or bf16), then the group's all-reduce, residual add and RMSNorm exactly as
    ``m3_mc_allreduce_norm``: same workspace, flags and call-order contract. Returns ``[normed, updated]`` or, with
    ``quant``, also the MXFP8 copy of ``normed``.
    """

    ARCHS = frozenset({Arch.SM_107})

    # Not taken at tp2 (modeling's MC_FINALIZE is empty: the decode step is at most 8 rows); the cells are the tp4
    # target's, on the SM107 kernel. The tp4 target takes this path from 9 tokens (top-4 of 128 experts, fp32 routing
    # weights, the shared expert always present): the 16- and 48-token buckets, epilogue option 15.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the smallest finalize call (9 tokens, 16-token bucket) with the next QKV's MXFP8 copy",
            spec=dict(m=9, maxt=16, opts=15, topk=4, quant=True, dropped=0),
        ),
        Cell(
            why="a full 16-token bucket with dropped expert slots",
            spec=dict(m=16, maxt=16, opts=15, topk=4, quant=False, dropped=5),
        ),
        Cell(
            why="the 48-token bucket at a partial fill",
            spec=dict(m=24, maxt=48, opts=15, topk=4, quant=False, dropped=0),
        ),
        Cell(
            why="the 48-token bucket full, MXFP8 copy",
            spec=dict(m=48, maxt=48, opts=15, topk=4, quant=True, dropped=3),
        ),
        # Every other row count of the unfinalized MoE (MC_FINALIZE): the bucket and options the target picks.
        *(
            Cell(
                why=f"{m} tokens: a finalize row count between the ones above",
                spec=dict(
                    m=m,
                    maxt=16 if m <= 16 else 48,
                    opts=15,
                    topk=4,
                    quant=m % 2 == 0,
                    dropped=m % 4,
                ),
            )
            for m in range(9, 49)
            if m not in (9, 16, 24, 48)
        ),
    )

    note = """
    BF16 routing weights and the 8-token bucket are accepted by the op (option 7 there) but the target only enters
    this path at 9 tokens and up, with the built-in MoE's fp32 weights; neither is a cell. The rest is as in
    `m3_mc_allreduce_norm`'s note.
    """

    def __call__(
        self,
        fc2: torch.Tensor,
        perm: torch.Tensor,
        scales: torch.Tensor,
        shared: Optional[torch.Tensor],
        residual: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        flags: torch.Tensor,
        rank: int,
        world: int,
        cluster: int,
        pdl: bool,
        maxt: int,
        quant: bool,
        opts: int,
        cap_out: Optional[torch.Tensor] = None,
        cap_w: Optional[torch.Tensor] = None,
        cap_eps: float = 0.0,
        own_op: bool = False,
    ) -> list[torch.Tensor]:
        # The capture operands only with a capture: an op without the capture epilogue (an older image) takes the
        # rest unchanged. own_op: this target's copy (trtllm::minimax_m3_sm107_tp8_mc_allreduce_norm, the same
        # kernels with the capture epilogue; modeling's MC_AR_OWN_OP).
        op = (
            torch.ops.trtllm.minimax_m3_sm107_tp8_mc_allreduce_norm
            if own_op
            else torch.ops.trtllm.minimax_m3_sm107_tp2_mc_allreduce_norm
        )
        return op(
            fc2,
            shared,
            residual,
            norm_w,
            eps,
            ws_uc,
            ws_mc_ptr,
            flags,
            rank,
            world,
            cluster,
            pdl,
            perm,
            scales,
            maxt,
            quant,
            0,
            opts,
            *((cap_out, cap_w, cap_eps) if cap_out is not None else ()),
        )

    def is_valid(
        self,
        fc2: torch.Tensor,
        perm: torch.Tensor,
        scales: torch.Tensor,
        shared: Optional[torch.Tensor],
        residual: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        flags: torch.Tensor,
        rank: int,
        world: int,
        cluster: int,
        pdl: bool,
        maxt: int,
        quant: bool,
        opts: int,
        cap_out: Optional[torch.Tensor] = None,
        cap_w: Optional[torch.Tensor] = None,
        cap_eps: float = 0.0,
        own_op: bool = False,
    ) -> None:
        h = residual.shape[-1]
        assert norm_w.is_contiguous(), (
            "norm_w must be contiguous; the kernel reads it as packed memory"
        )
        need = 4 * maxt * world * (h // 2)
        assert ws_uc.is_contiguous() and ws_uc.numel() >= need, (
            f"ws_uc holds {ws_uc.numel()} words, the {maxt}-token bucket of {world} ranks needs {need}"
        )
        assert flags.numel() == maxt * cluster, (
            f"flags has {flags.numel()} counters; the {maxt}-token bucket's are exactly {maxt * cluster}"
        )
        # The kernel gathers fc2[perm] without a bound: an index past the FC2 rows reads whatever follows them.
        lo, hi = int(perm.min()), int(perm.max())
        assert lo >= -1 and hi < fc2.shape[0], (
            f"perm spans [{lo}, {hi}]; entries must be -1 (dropped) or a row of fc2 (< {fc2.shape[0]})"
        )


m3_mc_finalize_allreduce_norm = M3McFinalizeAllreduceNorm()

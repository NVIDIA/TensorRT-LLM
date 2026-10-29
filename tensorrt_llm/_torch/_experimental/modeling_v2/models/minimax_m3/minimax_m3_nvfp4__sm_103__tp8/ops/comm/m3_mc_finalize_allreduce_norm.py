# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE finalize + multicast all-reduce + residual + RMSNorm (+ MXFP8 copy) of the decode MoE boundary, 8 ranks."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .m3_mc_allreduce_norm import boundary


def finalize_partial(
    fc2: torch.Tensor, perm: torch.Tensor, scales: torch.Tensor, shared: Optional[torch.Tensor]
) -> torch.Tensor:
    """One rank's BF16 partial: ``bf16(bf16(sum_k scale_k * fc2[perm_k]) + shared)``, the routed sum in FP32 in k
    order and rounded to BF16, then the shared expert added as stock adds it (one more BF16 rounding).

    The kernel's per-k step is a fused multiply-add; it is reproduced exactly by the float64 product (exact for two
    FP32 operands) plus the FP32 accumulator, rounded back to FP32. Dropped entries (perm -1) are skipped.
    """
    m, topk = perm.shape
    acc = torch.zeros(m, fc2.shape[1], dtype=torch.float32, device=fc2.device)
    sc = scales.reshape(m, topk).float()
    for k in range(topk):
        rows = perm[:, k].long()
        keep = (rows >= 0).unsqueeze(-1)
        gathered = fc2[rows.clamp_min(0)].double()
        step = (gathered * sc[:, k : k + 1].double() + acc.double()).float()
        acc = torch.where(keep, step, acc)
    if shared is not None:
        acc = acc.to(torch.bfloat16).float() + shared.float()
    return acc.to(torch.bfloat16)


class M3McFinalizeAllreduceNorm(OpWrapper):
    """The MoE boundary with the routed experts' finalize folded in.

    Per rank the partial is ``bf16(bf16(sum_k scales[t, k] * fc2[perm[t, k]]) + shared[t])`` (``perm`` -1 = dropped,
    ``scales`` fp32 or bf16), then the group's all-reduce, residual add and RMSNorm exactly as
    ``m3_mc_allreduce_norm``: same workspace, flags and call-order contract. Returns ``[normed, updated]`` or, with
    ``quant``, also the MXFP8 copy of ``normed``.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The tp8 target takes this path from 9 tokens (top-4 of 128 experts, fp32 routing weights, the shared expert
    # always present): the 16- and 48-token buckets, epilogue option 15.
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
        # Every other row count of the unfinalized MoE (dispatch.MC_FINALIZE): the bucket and options the target picks.
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
    ) -> list[torch.Tensor]:
        return torch.ops.trtllm.minimax_m3_mc_allreduce_norm(
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
        )

    def reference(
        self,
        per_rank_fc2: Sequence[torch.Tensor],
        per_rank_perm: Sequence[torch.Tensor],
        per_rank_scales: Sequence[torch.Tensor],
        per_rank_shared: Optional[Sequence[torch.Tensor]],
        residual: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        quant: bool,
    ) -> tuple[torch.Tensor, ...]:
        """The boundary the group is supposed to compute, from every rank's MoE outputs in rank order (the
        collective exception, as in ``m3_mc_allreduce_norm.reference``)."""
        partials = [
            finalize_partial(
                fc2,
                per_rank_perm[r],
                per_rank_scales[r],
                None if per_rank_shared is None else per_rank_shared[r],
            )
            for r, fc2 in enumerate(per_rank_fc2)
        ]
        return boundary(partials, residual, norm_w, eps, quant)

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

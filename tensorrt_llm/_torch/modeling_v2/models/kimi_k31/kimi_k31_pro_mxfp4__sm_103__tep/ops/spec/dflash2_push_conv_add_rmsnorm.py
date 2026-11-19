# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 DFlash2 drafter: row-parallel projection + TP all-reduce + block convolution + residual add + RMSNorm as
two kernels, the push GEMM and the sublayer epilogue that finishes the all-reduce."""

from __future__ import annotations

from typing import Sequence

import torch

from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.distributed import ops as _dist_ops
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm import lamport_ar as _lam
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.allreduce_module import mnnvl_group

from ...kernels.comm import bf16_ar_push_gemm as _push_gemm
from ...kernels.spec.dflash2_ar_conv_add_rmsnorm import dflash2_ar_conv_add_rmsnorm as _fused
from ..comm.oproj_allreduce_push import (
    DEFAULT_ONE_SHOT_MAX_BYTES,
    lamport_workspace,
    oproj_allreduce_push,
    push_serves,
)
from .dflash2_conv_add_rmsnorm import dflash2_conv_add_rmsnorm

#: Largest row count the epilogue finishes the all-reduce for; above it the push GEMM + finish kernel + epilogue run.
#: GB300, W = 4, K = 768, us per site push + finish + epilogue -> push + finishing epilogue: 10 rows 10.46 -> 9.95,
#: 60: 13.26 -> 11.13, 96: 16.21 -> 13.24, 160: 19.85 -> 16.55, 192 (the push GEMM's last row count at K = 768):
#: 20.39 -> 17.48; K = 384 / 448 at 8 rows: 9.94 -> 9.37 / 10.14 -> 9.59.
FUSED_FINISH_MAX_ROWS = 192
#: Full-message size (``rows * hidden * world * 2``) up to which a call takes the one-shot transport (and the
#: two-kernel path); the engine's threshold, which the drafter's blocks at TP16 / TP32 exceed. Tests set 0 to drive
#: the fused two-shot path on four ranks.
ONE_SHOT_MAX_BYTES = DEFAULT_ONE_SHOT_MAX_BYTES


def _cell(rows: int, block: int, k: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(world=4, rows=rows, block=block, k=k))


class K31DFlash2PushConvAddRmsnorm(OpWrapper):
    """``(out, residual)``: ``c = Conv(all_reduce(input @ weight^T); base, delta)`` within each drafted block, then
    ``residual = bf16(c + residual)`` (in place) and ``out = bf16(rmsnorm(c + residual) * norm_weight)`` (a new tensor):
    the drafter's row-parallel o_proj or MLP down, its TP all-reduce and the sublayer epilogue
    (``k31_dflash2_conv_add_rmsnorm``).

    ``input [M, K]`` / ``weight [7168, K]``: this rank's shard of the row-parallel projection (BF16, contiguous);
    ``all_reduce``: the engine ``AllReduce`` module of the drafter's TP group; the rest as
    ``k31_dflash2_conv_add_rmsnorm``. Where the group runs MNNVL, the push GEMM has a tactic for (M, 7168, K), the
    call takes the two-shot transport and M is at most ``FUSED_FINISH_MAX_ROWS``, two kernels run: the push GEMM
    stores its BF16 partial rows into the Lamport slots of the group's MNNVL workspace, and the PDL epilogue finishes
    the all-reduce (the finish kernel's reduction order and rounding) before it convolves, adds and normalizes.
    Otherwise ``oproj_allreduce_push`` (push GEMM + finish kernel, or cuBLAS + the engine all-reduce) and
    ``k31_dflash2_conv_add_rmsnorm`` run.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        _cell(8, 8, 384, "TEP32 o_proj shard, ll B1 block of 8"),
        _cell(8, 8, 448, "TEP32 MLP down shard (K tail), ll B1 block of 8"),
        _cell(10, 5, 768, "TEP16 o_proj shard, mid B2 blocks of 5"),
        _cell(60, 5, 896, "TEP16 MLP down shard: no push tactic, the two-kernel path"),
        _cell(60, 5, 768, "TEP16 o_proj shard, M2 B12 blocks of 5"),
        _cell(120, 5, 768, "TEP16 o_proj shard, M2 B24 blocks of 5"),
        _cell(
            256, 8, 768, "TEP16 o_proj shard past the push table: cuBLAS + the engine all-reduce"
        ),
    )
    note = """
    Bitwise the two-kernel chain it replaces (push GEMM + finish kernel, then the epilogue): the epilogue sums the
    pushed partials in the finish kernel's order (rank order, chunks as the reproduced engine kernel), sends -0 as +0
    like it, and then runs the epilogue's arithmetic on the finished rows. The push GEMM half is the
    `oproj_allreduce_push` GEMM (not bitwise against cuBLAS by construction; see its note). Workspace and ordering
    rules as `oproj_allreduce_push`: the group's engine MNNVL workspace (grown before CUDA-graph capture), every rank
    the same calls; the epilogue must be the next kernel on the workspace after the push GEMM. Two-shot transport
    only (the drafter's blocks of 5+ rows at TP16 / TP32 are over the one-shot threshold); a one-shot call takes the
    two-kernel path. The cells run four ranks on one tray with MNNVL forced.
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        all_reduce: AllReduce,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        block_size: int,
        group_size: int,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, hidden = input.shape[0], weight.shape[0]
        fused = push_serves(input, weight, all_reduce) and rows <= FUSED_FINISH_MAX_ROWS
        if fused:
            world = mnnvl_group(all_reduce)[0].tp_size
            fused = not _lam.one_shot_for(rows, hidden, world, ONE_SHOT_MAX_BYTES)
        if not fused:
            x = oproj_allreduce_push(input, weight, all_reduce)
            return dflash2_conv_add_rmsnorm(
                x, delta, base_kernel, block_size, group_size, residual, norm_weight, eps
            )
        ws = lamport_workspace(all_reduce, rows, hidden, False)
        cpp_one_shot = _lam.one_shot_for(
            rows, hidden, ws.world, _dist_ops._MNNVL_ONE_SHOT_THRESHOLD_BYTES
        )
        use_pdl = get_env_enable_pdl()
        out = torch.empty_like(residual)
        # The push GEMM only reads the all-reduce output's shape ([rows, 7168], not written).
        _push_gemm.run_bf16_gemm_ar_push(input, weight, ws, False, out, use_pdl=use_pdl)
        _fused(
            ws,
            rows,
            delta,
            base_kernel,
            block_size,
            group_size,
            residual,
            norm_weight,
            eps,
            out,
            _lam.reduce_chunk(ws.world, cpp_one_shot),
            use_pdl,
        )
        return out, residual

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        weights_in_rank_order: Sequence[torch.Tensor],
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        block_size: int,
        group_size: int,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collective: every rank's ``input`` and ``weight`` in rank order. The FP32 sum of the partial products
        rounded once, then the epilogue's reference (on a copy of ``residual``)."""
        x = oproj_allreduce_push.reference(inputs_in_rank_order, weights_in_rank_order, None)
        return dflash2_conv_add_rmsnorm.reference(
            x, delta, base_kernel, block_size, group_size, residual, norm_weight, eps
        )

    def is_valid(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        all_reduce: AllReduce,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        block_size: int,
        group_size: int,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> None:
        assert input.dtype == weight.dtype == residual.dtype == torch.bfloat16, "BF16 only"
        assert input.dim() == 2 and weight.dim() == 2 and input.shape[1] == weight.shape[1]
        assert residual.shape == (input.shape[0], weight.shape[0]) and residual.is_contiguous()
        dflash2_conv_add_rmsnorm.is_valid(
            residual, delta, base_kernel, block_size, group_size, residual.clone(), norm_weight, eps
        )


k31_dflash2_push_conv_add_rmsnorm = K31DFlash2PushConvAddRmsnorm()

__all__ = [
    "FUSED_FINISH_MAX_ROWS",
    "ONE_SHOT_MAX_BYTES",
    "K31DFlash2PushConvAddRmsnorm",
    "k31_dflash2_push_conv_add_rmsnorm",
]

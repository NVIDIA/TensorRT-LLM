# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A decode block's TP sum and the hyper-connection site's injection + grouped norm in one launch.

`native_ops.hc_allreduce_inject_norm` (`csrc/hcAllreduceInjectNorm.cu`) does the IPC one-shot
Lamport all-reduce of `catalog/comm/allreduce` over the same workspace and protocol (so it
interleaves with the plain all-reduce's calls), then `_hc_inject_norm_launch`'s injection and
norm from the summed row, its sum of squares in the Triton program's reduction tree at
`program_warps` warps. Both outputs are bitwise the all-reduce followed by the Triton kernel.

Only the norm weights are read before the grid-dependency wait; the residual, injection weights
and shared-expert operands are read after it.
"""

from __future__ import annotations

from typing import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_inject_norm import (  # noqa: E501
    qwen4_exp_hc_inject_norm,
)

from ... import native_ops

HIDDEN = 2560
HC_COUNT = 4

#: Decode-site row counts certified at 2 and 4 ranks, each with and without the MoE combine,
#: and why each is there.
_ROWS = (
    (33, "the smallest fused site (`_HC_AR_INJECT_MIN_ROWS`)"),
    (96, "a decode batch on 4-warp programs"),
    (168, "a decode batch on 4-warp programs"),
    (192, "a decode batch on 4-warp programs"),
    (216, "a decode batch on 4-warp programs"),
    (288, "a decode batch on 4-warp programs"),
    (384, "a decode batch on 4-warp programs"),
    (480, "a decode batch on 4-warp programs"),
    (511, "the last row count on 4-warp programs"),
    (512, "the first row count on 2-warp programs"),
    (576, "a decode batch on 2-warp programs"),
    (640, "a decode batch on 2-warp programs"),
    (1024, "a decode batch on 2-warp programs"),
)


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0 and t.stride(-1) == 1 and t.stride(0) % 8 == 0


def fits(hc, partial, inject, shared=None, gate=None) -> bool:
    """Whether the native kernel takes these operands (else the caller sums and injects apart)."""
    rows = hc.shape[0]
    if partial.dim() != 2 or partial.shape != (rows, HIDDEN) or not partial.is_contiguous():
        return False
    if hc.shape != (rows, HC_COUNT * HIDDEN) or not _aligned(hc):
        return False
    if inject is None or inject.shape[0] != rows or inject.shape[1] < HC_COUNT:
        return False
    if inject.stride(-1) != 1:
        return False
    if (shared is None) != (gate is None):
        return False
    if shared is not None and (shared.shape != (rows, HIDDEN) or not _aligned(shared)):
        return False
    dtypes = {t.dtype for t in (hc, partial, inject, shared, gate) if t is not None}
    return dtypes == {torch.bfloat16}


class HcAllreduceInjectNorm(OpWrapper):
    """`(hc_out, normed)` of the site after a block whose per-rank output is `partial`: bitwise
    `_hc_inject_norm_launch(hc, tp_sum(partial), inject, norm, eps, 4, shared, gate,
    program_warps)`. `workspace`, `rank` and `nranks` are the stock IPC all-reduce's
    (`catalog/comm/allreduce`); `fits` says which operands the kernel takes."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"tp{world}: {rows} rows, {why}",
            spec=dict(
                world=world, rows=rows, hidden=HIDDEN, hc_count=HC_COUNT, dtype=torch.bfloat16
            ),
        )
        for world in (2, 4)
        for rows, why in _ROWS
    )

    note = """
    Certified bitwise the IPC all-reduce followed by the Triton injection, with and without the
    MoE combine, interleaved with plain all-reduces and in a replayed CUDA graph chain.
    Not certified: hc_count other than 4, hidden other than 2560, groups spanning nodes.
    """

    def __call__(
        self,
        partial: torch.Tensor,
        workspace: torch.Tensor,
        rank: int,
        nranks: int,
        hc: torch.Tensor,
        inject: torch.Tensor,
        norm: torch.Tensor,
        eps: float,
        shared: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        program_warps: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hc_out, normed = native_ops.hc_allreduce_inject_norm(
            partial,
            workspace,
            int(rank),
            int(nranks),
            hc,
            inject,
            shared,
            gate,
            norm,
            float(eps),
            int(program_warps),
            # trigger_completion_at_end=False: release the next launch right after this kernel's
            # wait.
            False,
        )
        return hc_out, normed

    def reference(
        self,
        partials_in_rank_order: Sequence[torch.Tensor],
        hc: torch.Tensor,
        inject: torch.Tensor,
        norm: torch.Tensor,
        eps: float,
        shared: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The fp32 sum of every rank's `partial`, in rank order, rounded once, then
        `qwen4_exp_hc_inject_norm`'s reference of the site. The documented collective exception
        to mirroring `__call__`; `program_warps` only orders the kernel's sum of squares."""
        total = torch.zeros_like(partials_in_rank_order[0], dtype=torch.float32)
        for partial in partials_in_rank_order:
            total += partial.float()
        block = total.to(partials_in_rank_order[0].dtype)
        return qwen4_exp_hc_inject_norm.reference(
            hc, block, inject[:, :HC_COUNT], norm, eps, HC_COUNT, shared, gate
        )

    def is_valid(
        self,
        partial: torch.Tensor,
        workspace: torch.Tensor,
        rank: int,
        nranks: int,
        hc: torch.Tensor,
        inject: torch.Tensor,
        norm: torch.Tensor,
        eps: float,
        shared: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        program_warps: int = 4,
    ) -> None:
        assert fits(hc, partial, inject, shared, gate), "operands the kernel does not take"
        assert nranks in (2, 4) and 0 <= rank < nranks, (rank, nranks)
        assert program_warps in (2, 4), program_warps


hc_allreduce_inject_norm = HcAllreduceInjectNorm()

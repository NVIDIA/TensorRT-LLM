# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4 all-reduce, residual add and RMSNorm, plus the NVFP4 quantize of the norm."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)

from ._lamport import LamportWorkspace, check_rows, tail
from .cute_dsl_allreduce_add_rmsnorm import cute_dsl_allreduce_add_rmsnorm

#: NVFP4: one e4m3 scale per 16 elements.
SF_VEC = 16


class CuteDslAllreduceAddRmsnormQuantNvfp4(OpWrapper):
    """`cute_dsl_allreduce_add_rmsnorm`, plus the bf16 norm quantized to NVFP4.

    Writes `out` (the bf16 norm, returned), updates `residual` in place, and
    writes `fp4_out` uint8 `[rows, hidden/2]` (two e2m1 codes per byte, even
    element in the low nibble) and `scale_out` uint8 `[rows, hidden/16]` (one
    e4m3 scale per 16 elements, row-major: the linear layout the routed
    experts' grouped GEMM reads). The quantize reads the bf16 norm it writes,
    with the scale `e4m3(global_scale * block_amax / 6)`, so the codes are what
    `fp4_quantize` makes of `out` with `sf_vec_size=16` and linear scales.
    `global_scale` is fp32 with one element. Always one-shot.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target calls this instead of the plain add+RMSNorm entry where an MoE
    # layer's MLP opens at a row count the routed experts quantize on their own
    # kernel: 1..64 rows. The bf16 norm still goes to the router and the shared
    # expert.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode row", spec=dict(rows=1)),
        Cell(
            why="two rows: each row is its own CTA cluster, so one row's RMS could leak into the other",
            spec=dict(rows=2),
        ),
        Cell(
            why="the full decode batch, 64 rows, the routed quantizer's ceiling", spec=dict(rows=64)
        ),
    )

    note = """
    The target calls this entry at 1..64 rows, where the routed experts read
    the NVFP4 payload it writes; at any other row count it calls
    `cute_dsl_allreduce_add_rmsnorm` and the routed experts quantize on their
    own. The kernel is one-shot at every row count (there is no two-shot
    variant of this epilogue in the workspace); nothing above 64 rows is
    certified. All ranks must call with the same row count, or the call hangs.
    """

    def __call__(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        global_scale: torch.Tensor,
        out: torch.Tensor,
        fp4_out: torch.Tensor,
        scale_out: torch.Tensor,
        workspace: LamportWorkspace,
    ) -> torch.Tensor:
        workspace.fused_quant(
            partial,
            residual,
            weight,
            global_scale,
            out,
            fp4_out,
            scale_out,
            residual,
            *tail(workspace, fused=True),
        )
        return out

    def reference(
        self,
        partials: Sequence[torch.Tensor],
        residual: torch.Tensor,
        weight: torch.Tensor,
        global_scale: torch.Tensor,
        out: torch.Tensor,
        fp4_out: torch.Tensor,
        scale_out: torch.Tensor,
        workspace: LamportWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(norm, new_residual, fp4, scales)`; nothing passed in is modified.

        `partials` is the group's partials in rank order (the collective
        exception to mirroring `__call__`). `norm` and `new_residual` are
        `cute_dsl_allreduce_add_rmsnorm`'s reference; `fp4` `[rows, hidden/2]`
        and `scales` `[rows, hidden/16]` are `fp4_quantize`'s reference of that
        bf16 norm with linear scales. `out`, `fp4_out` and `scale_out` are
        ignored.
        """
        norm, new_residual = cute_dsl_allreduce_add_rmsnorm.reference(
            partials, residual, weight, out, workspace
        )
        fp4, scales = fp4_quantize.reference(norm, global_scale, SF_VEC, False, False)
        return norm, new_residual, fp4, scales.view(norm.shape[0], -1)

    def is_valid(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        global_scale: torch.Tensor,
        out: torch.Tensor,
        fp4_out: torch.Tensor,
        scale_out: torch.Tensor,
        workspace: LamportWorkspace,
    ) -> None:
        rows = int(partial.shape[0])
        check_rows(rows, workspace)
        # The kernel loads one scalar and ignores the rest: a per-token scale
        # would be applied as global_scale[0] to every row.
        assert global_scale.numel() == 1, (
            "global_scale must hold exactly one element; the kernel applies element 0 to every row"
        )
        # The kernel writes both outputs through raw pointers at compact row
        # strides (`row * hidden/2`, `row * hidden/16`), so a strided view or a
        # shorter buffer is written as if it were the compact `[rows, ...]` one.
        hidden = int(partial.shape[1])
        for name, t, shape in (
            ("fp4_out", fp4_out, (rows, hidden // 2)),
            ("scale_out", scale_out, (rows, hidden // SF_VEC)),
        ):
            assert tuple(t.shape) == shape, f"{name} must be {list(shape)}, got {tuple(t.shape)}"
            assert t.is_contiguous(), f"{name} must be contiguous"


cute_dsl_allreduce_add_rmsnorm_quant_nvfp4 = CuteDslAllreduceAddRmsnormQuantNvfp4()

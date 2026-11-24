# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's norms with an early PDL release: a plain RMSNorm, and the block
convolution's output side fused with the residual add and the RMSNorm (CuTe DSL)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.dflash2_conv_add_rmsnorm import (
    dflash2_conv_add_rmsnorm_into,
    dflash2_rmsnorm_into,
)
from ...kernels.spec.dflash2_grouped_conv import VEC

#: Kimi K3.1 DFlash2 drafter: hidden 7168, q_lora 1536, conv_kernel_size 2, conv_group_size 16.
_HIDDEN = 7168
_Q_LORA = 1536
_Q_KV_A = 2112
_TAPS = 2
_GROUP = 16


class K31DFlash2RmsNorm(OpWrapper):
    """``out = x / sqrt(mean(x^2) + eps) * weight``: FlashInfer's CuTe DSL ``rmsnorm`` kernel
    (configuration, thread-value layout, row reduction and arithmetic) with the dependents
    released right after ``griddepcontrol.wait`` when launched with PDL. Bitwise FlashInfer's
    ``rmsnorm`` on the same input. Returns a new tensor."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix16 drafter layer 0 input norm: 8 rows x 7168, PDL",
            spec=dict(rows=8, hidden=_HIDDEN, pitch=_HIDDEN, pdl=True),
        ),
        Cell(
            why="helix16 drafter q_a norm: 8 rows x 1536, a row-strided view of the [q_a|kv_a] GEMM",
            spec=dict(rows=8, hidden=_Q_LORA, pitch=_Q_KV_A, pdl=True),
        ),
        Cell(
            why="64 requests at width 8 (512 rows), strided, without PDL",
            spec=dict(rows=512, hidden=_Q_LORA, pitch=_Q_KV_A, pdl=False),
        ),
        Cell(
            why="512 rows x 7168, PDL: several rows per CTA tile and many CTAs",
            spec=dict(rows=512, hidden=_HIDDEN, pitch=_HIDDEN, pdl=True),
        ),
    )

    note = """
    The kernel subclasses ``flashinfer.norm.kernels.rmsnorm.RMSNormKernel`` and repeats its
    kernel body with one change: the PDL release moves from after the last store to right after
    the wait. A dependent that waits before reading the output (the drafter's split-K GEMMs)
    starts while the norm runs; one that reads without waiting must not follow it. Without
    ``pdl`` the launch is the plain one. Row-strided inputs (unit last stride) take FlashInfer's
    strided compile. Compiles once per (dtype, hidden, pdl, contiguity) on the first call, which
    must be eager.
    """

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float, pdl: bool = False
    ) -> torch.Tensor:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        return dflash2_rmsnorm_into(x, weight, eps, out, pdl)

    def reference(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float, pdl: bool = False
    ) -> torch.Tensor:
        """FlashInfer's ``rmsnorm`` (the kernel this entry is bitwise to); ``pdl`` is ignored."""
        from flashinfer.norm import rmsnorm

        return rmsnorm(x, weight, eps)

    def is_valid(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float, pdl: bool = False
    ) -> None:
        assert x.dtype in (torch.bfloat16, torch.float16), f"BF16/FP16 input, got {x.dtype}"
        assert weight.dtype == x.dtype and weight.shape == (x.shape[-1],), (
            "weight must be [hidden] in the input dtype"
        )
        assert x.dim() == 2 and x.stride(-1) == 1, (
            "x must be [rows, hidden] with a unit last stride"
        )


class K31DFlash2ConvAddRmsNorm(OpWrapper):
    """``residual += Conv(a)``, ``out = RMSNorm(residual) * weight``: the block convolution's
    output side (``dflash2_grouped_conv``) fused into FlashInfer's ``fused_add_rmsnorm`` kernel,
    one launch. Bitwise the two launches it replaces. ``residual`` is updated in place; returns
    ``(out, residual)``. With PDL it waits for the previous kernel first; ``early`` releases the
    next kernel right after the wait, else after the last store (FlashInfer's placement)."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix16 drafter: 1 request at width 8, attention side 1 into post_norm, early",
            spec=dict(batch=1, width=8, side=1, pdl=True, early=True, fill="randn"),
        ),
        Cell(
            why="the last layer's MLP side 1 into the final norm: the release stays at the end",
            spec=dict(batch=1, width=8, side=1, pdl=True, early=False, fill="randn"),
        ),
        Cell(
            why="64 requests at width 8 (512 rows), without PDL",
            spec=dict(batch=64, width=8, side=1, pdl=False, early=True, fill="randn"),
        ),
        Cell(
            why="odd batch with NaN rows: a padding request's NaN never leaks into the next block",
            spec=dict(batch=7, width=8, side=1, pdl=True, early=True, fill="nan_prev"),
        ),
        Cell(
            why="exact zeros and negative coefficients: signed zeros match bit for bit",
            spec=dict(batch=13, width=5, side=1, pdl=True, early=True, fill="zeros"),
        ),
        Cell(
            why="helix16 drafter, the row over a 7-CTA cluster (as served), early",
            spec=dict(batch=1, width=8, side=1, pdl=True, early=True, fill="randn", ctas=7),
        ),
        Cell(
            why="7-CTA cluster, the final norm's release at the end",
            spec=dict(batch=1, width=8, side=1, pdl=True, early=False, fill="randn", ctas=7),
        ),
        Cell(
            why="7-CTA cluster, odd batch with NaN rows (no NaN leaks), 392 CTAs",
            spec=dict(batch=7, width=8, side=1, pdl=True, early=True, fill="nan_prev", ctas=7),
        ),
    )

    note = """
    The kernel subclasses ``flashinfer.norm.kernels.fused_add_rmsnorm.FusedAddRMSNormKernel``:
    FlashInfer's configuration and row reduction, its kernel body with the input load replaced by
    the convolution (``dflash2_grouped_conv``'s round-to-nearest FP32 arithmetic, rounded to the
    activation dtype). It needs FlashInfer's layout to put one row in one CTA tile (hidden 7168 on
    sm_103 / sm_107: 128 threads x 7 vectors); other layouts are refused when it is made. With
    ``ctas`` (the helix16 target's ``_DRAFT_NORM_CTAS``) each row is split over a cluster of
    that many CTAs, one 8-wide vector per thread, the sum of squares reduced over the cluster
    in DSMEM by FlashInfer's cluster reduction: hidden 7168 at 7 CTAs = 7 x 128 threads. The
    residual is bitwise; the output differs from the reference only through the sum's order. The
    convolution input, the residual and the output must not overlap. Compiles once per
    (dtype, hidden, taps, group, pdl, early) on the first call, which must be eager.
    """

    def __call__(
        self,
        a: torch.Tensor,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        block_size: int,
        group_size: int,
        pdl: bool = False,
        early: bool = True,
        ctas: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(a)
        dflash2_conv_add_rmsnorm_into(
            a,
            delta,
            base_kernel,
            residual,
            weight,
            eps,
            block_size,
            group_size,
            out,
            pdl,
            early,
            ctas,
        )
        return out, residual

    def reference(
        self,
        a: torch.Tensor,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        block_size: int,
        group_size: int,
        pdl: bool = False,
        early: bool = True,
        ctas: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two launches it replaces: ``dflash2_grouped_conv`` and FlashInfer's
        ``fused_add_rmsnorm`` (in place on ``residual``); ``pdl`` / ``early`` / ``ctas`` are
        ignored (with ``ctas`` the residual is bitwise this, the output within the norm's
        summation-order error)."""
        from flashinfer.norm import fused_add_rmsnorm

        from .dflash2_grouped_conv import dflash2_grouped_conv

        out = dflash2_grouped_conv(a, delta, base_kernel, block_size, group_size)
        fused_add_rmsnorm(out, residual, weight, eps)
        return out, residual

    def is_valid(
        self,
        a: torch.Tensor,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        block_size: int,
        group_size: int,
        pdl: bool = False,
        early: bool = True,
        ctas: int = 0,
    ) -> None:
        dtype = a.dtype
        assert dtype in (torch.bfloat16, torch.float16), f"BF16/FP16 activations, got {dtype}"
        assert all(t.dtype == dtype for t in (delta, base_kernel, residual, weight)), (
            "every operand in the activation dtype"
        )
        assert a.dim() == 2 and a.is_contiguous() and residual.is_contiguous(), (
            "a and residual must be contiguous [rows, hidden]"
        )
        assert residual.shape == a.shape, "residual must match a"
        rows, hidden = a.shape
        taps = base_kernel.shape[0]
        assert group_size % VEC == 0 and hidden % group_size == 0, (
            f"group_size {group_size} must be a multiple of {VEC} dividing hidden {hidden}"
        )
        assert base_kernel.shape == (taps, hidden) and base_kernel.is_contiguous(), (
            "base_kernel must be a contiguous [taps, hidden]"
        )
        assert delta.shape == (rows, taps, hidden // group_size) and delta.stride(-1) == 1, (
            "delta must be [rows, taps, hidden // group_size] with a unit last stride"
        )
        span = rows * hidden * a.element_size()
        assert abs(a.data_ptr() - residual.data_ptr()) >= span, "a and residual overlap"
        assert block_size >= 1, f"block_size must be positive, got {block_size}"


dflash2_rmsnorm = K31DFlash2RmsNorm()
dflash2_conv_add_rmsnorm = K31DFlash2ConvAddRmsNorm()

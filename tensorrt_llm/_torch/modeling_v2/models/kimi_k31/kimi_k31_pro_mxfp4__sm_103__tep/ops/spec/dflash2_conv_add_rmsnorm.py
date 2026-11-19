# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's sublayer epilogue (block convolution + residual add + RMSNorm) in one launch."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.spec.dflash2_grouped_conv import (
    dflash2_grouped_conv_into,
)

from ...kernels.spec.dflash2_conv_add_rmsnorm import dflash2_conv_add_rmsnorm as _fused

#: Kimi K3.1 DFlash2 drafter: hidden 7168, conv_kernel_size 2, conv_group_size 16, RMSNorm eps.
_HIDDEN = 7168
_TAPS = 2
_GROUP = 16
_EPS = 1e-6


class K31DFlash2ConvAddRmsnorm(OpWrapper):
    """``(out, residual)``: ``c = Conv(x; base, delta)`` within each drafted block, then the fused
    residual add + RMSNorm of ``c``: ``residual = bf16(c + residual)`` (in place) and
    ``out = bf16(rmsnorm(c + residual) * weight)`` (a new tensor).

    The drafter's attention / MLP ``finish`` followed by ``flashinfer_fused_add_rmsnorm``, in one
    launch; bitwise those two ops (the reference runs them): the convolution's FP32 arithmetic of
    ``dflash2_grouped_conv`` and FlashInfer's CuTe DSL fused add + RMSNorm kernel, whose tiling,
    reduction and normalization this kernel inherits.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{gens} request(s) x width {width}: {why}",
            spec=dict(gens=gens, width=width, hidden=_HIDDEN, taps=_TAPS, group=_GROUP, eps=_EPS),
        )
        for gens, width, why in (
            (1, 8, "ll B1, max_draft_len 7"),
            (2, 6, "mid B2, max_draft_len 5"),
            (12, 6, "M2 B12"),
            (64, 8, "the largest decode step"),
            (3, 5, "an odd batch at width 5"),
        )
    )
    note = """
    `x` is read-only (a row's convolution reads the previous rows of its block) and `out` is a new
    tensor; `residual` is updated in place, as the reference op does. `delta` may be the strided
    finish-side view of the convolution's kernel projection (unit last stride). The kernel
    compiles once per (hidden, taps, group, dtype, PDL) on the first, eager call. It imports
    FlashInfer's CuTe DSL fused add + RMSNorm kernel class: a FlashInfer upgrade that changes that
    kernel changes both sides together (and this entry's test catches any divergence).
    TRTLLM_ENABLE_PDL changes launch scheduling, not results.
    """

    def __call__(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        block_size: int,
        group_size: int,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(x)
        _fused(
            x,
            delta,
            base_kernel,
            block_size,
            group_size,
            residual,
            weight,
            eps,
            out,
            get_env_enable_pdl(),
        )
        return out, residual

    def reference(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        block_size: int,
        group_size: int,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two ops it replaces: the family convolution, then the catalog fused add + RMSNorm
        (on copies: returns `(out, residual_out)` and leaves `residual` as it was)."""
        c = torch.empty_like(x)
        dflash2_grouped_conv_into(x, delta, base_kernel, block_size, group_size, c)
        r = residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(c, r, weight, eps)
        return c, r

    def is_valid(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        block_size: int,
        group_size: int,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> None:
        dtype = x.dtype
        assert dtype in (torch.bfloat16, torch.float16), f"BF16/FP16 activations, got {dtype}"
        for name, t in (("delta", delta), ("base_kernel", base_kernel), ("residual", residual)):
            assert t.dtype == dtype, f"{name} must be {dtype}, got {t.dtype}"
        assert weight.dtype == dtype, f"weight must be {dtype}, got {weight.dtype}"
        assert x.is_contiguous() and residual.is_contiguous() and base_kernel.is_contiguous(), (
            "x, residual and base_kernel must be contiguous"
        )
        rows, hidden = x.shape
        taps = base_kernel.shape[0]
        assert residual.shape == x.shape and weight.shape == (hidden,), "shape mismatch"
        assert delta.shape == (rows, taps, hidden // group_size) and delta.stride(-1) == 1, (
            f"delta must be [{rows}, {taps}, {hidden // group_size}] with a unit last stride"
        )
        assert residual.data_ptr() != x.data_ptr(), "residual must not alias x"
        assert block_size >= 1, f"block_size must be positive, got {block_size}"


dflash2_conv_add_rmsnorm = K31DFlash2ConvAddRmsnorm()

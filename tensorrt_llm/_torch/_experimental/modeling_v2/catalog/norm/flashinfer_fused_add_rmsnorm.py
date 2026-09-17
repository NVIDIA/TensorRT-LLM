# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place fused residual add + RMS normalization via the flashinfer kernel.

Everything a target author needs in order to decide whether this op covers
their case is below, and all of it is checked: `reference` is what CI compares
the kernel against, and `CELLS` is what CI drives.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper


class _FlashinferFusedAddRmsnorm(OpWrapper):
    """`residual += x; x = rmsnorm(residual) * weight`, in place, returns None.

    Both tensors are written. The residual output is the fp32 sum rounded once
    to its own dtype; the normalization reads that sum in fp32, not the rounded
    value, which is what `reference` mirrors.

    Fusion boundary: add, normalization and weight scaling. No gemma `(1 + w)`
    form, no quantization.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The two hidden sizes the shipped targets normalize over, at a decode
    # shape and a prefill shape. Both targets call this op only on the
    # residual stream, so there is no third width to cover and no 3-D case:
    # unlike plain rmsnorm this one never runs on a per-head view.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode-shaped, gpt-oss-120b hidden",
            spec=dict(shape=(1, 2880), dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="decode-shaped, deepseek-r1 hidden",
            spec=dict(shape=(1, 7168), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill-shaped: enough rows to cross the kernel's grid tiling",
            spec=dict(shape=(2048, 7168), dtype=torch.bfloat16, eps=1e-6),
        ),
    )

    note = """
    Registered only when flashinfer is importable (`IS_FLASHINFER_AVAILABLE`).

    PDL is controlled by `TRTLLM_ENABLE_PDL` (default on) inside the trtllm
    custom op; it changes scheduling, not results, and is not a cell axis.

    Both targets pass contiguous tensors here -- the residual stream is a
    freshly allocated buffer -- so the strided-row path this kernel also has
    is not certified by this entry.
    """

    def __call__(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> None:
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(x, residual, weight, eps)

    def reference(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two values the op writes, as (x_out, residual_out).

        Returns rather than mutates so a cell can drive both sides from the
        same inputs; the kernel's in-place write is what `__call__` does.
        """
        h = x.float() + residual.float()
        normed = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * weight.float()).to(x.dtype), h.to(residual.dtype)

    def is_valid(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> None:
        # Only the cases the op takes and answers wrongly. Rank and width
        # disagreements it rejects itself, loudly.
        assert x.dtype is not torch.float8_e4m3fn, (
            "float8_e4m3fn is accepted by the kernel's dtype table and returns "
            "a plausible fp8 result; dequantize before normalizing"
        )
        assert x.stride(-1) == 1 and residual.stride(-1) == 1, (
            f"the normalized dim must be contiguous; got strides "
            f"{x.stride(-1)} and {residual.stride(-1)}"
        )
        # Aliasing is the trap specific to the in-place form: the kernel writes
        # both tensors, so one buffer passed twice is overwritten mid-kernel
        # and the result is neither value.
        assert x.data_ptr() != residual.data_ptr(), (
            "x and residual must be distinct buffers; the op writes both"
        )


flashinfer_fused_add_rmsnorm = _FlashinferFusedAddRmsnorm()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma fused residual add + RMS normalization that releases its PDL dependents as soon as
its loads have landed: the layer-boundary norm at decode row counts.

The kernel is flashinfer's CuTe-DSL ``FusedAddRMSNormKernel`` (the stock gemma add+norm)
with ``griddepcontrol.launch_dependents`` moved from after the stores to right after the
pass-1 loads; it lives in ``_gemma_add_rmsnorm_early_trigger``. The next layer's first
GEMM (dense Marlin W4A16, ``in_proj_qkvz`` or ``qkv_proj``) then launches while the norm
computes and prefetches its weights before its own ``griddepcontrol.wait``.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ._gemma_add_rmsnorm_early_trigger import gemma_fused_add_rmsnorm_early_trigger
from .flashinfer_gemma_fused_add_rmsnorm import flashinfer_gemma_fused_add_rmsnorm

#: The checkpoint's residual stream: hidden 5120, bf16, `rms_norm_eps` 1e-6.
_CKPT = dict(hidden=5120, dtype=torch.bfloat16, eps=1e-6)


class GemmaAddRmsnormEarlyTrigger(OpWrapper):
    """`residual = x + residual; x = residual / sqrt(mean(residual^2) + eps) * (1 + w)`, in
    place; returns `(x, residual)`, the same two tensors.

    Bitwise the stock gemma add+norm (`flashinfer_gemma_fused_add_rmsnorm`): same loads,
    arithmetic and stores. Only the PDL trigger moves, so a dependent kernel may start
    before this one has stored -- which is safe exactly when the dependent reads this
    kernel's outputs only after its own `griddepcontrol.wait`.

    Fusion boundary: as the stock add+norm. The consumer is the next layer's input
    projection.
    """

    ARCHS = frozenset({Arch.SM_120})

    # The next layer's input_layernorm (every layer boundary but the final norm) at the
    # decode row counts, 1..32.
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode 1 row (c1): one CTA, one live row", spec=dict(rows=1, **_CKPT)),
        Cell(why="decode 8 rows (c8): 4 CTAs of 2 rows", spec=dict(rows=8, **_CKPT)),
        Cell(
            why="decode 32 rows: the top of the band, above the 1-row kernel's lower edge",
            spec=dict(rows=32, **_CKPT),
        ),
    )

    note = """
    Metadata consumed: none.

    Kernel: flashinfer's CuTe-DSL `FusedAddRMSNormKernel` with `weight_bias=1.0`,
    compact layout, the parent's geometry (hidden 5120 bf16 on sm_120: 64 threads a
    row, 2 rows a CTA, cluster 1); its own compile cache, one kernel per (dtype,
    hidden, PDL, SM). PDL from `TRTLLM_ENABLE_PDL` (default on): with it off the
    kernel has no griddepcontrol at all and is the stock kernel without PDL.

    The early trigger is safe only for dependents that wait before reading: a kernel
    launched with the PDL attribute must reach `griddepcontrol.wait` before it reads
    `x` or `residual` (one launched without it is ordered by the stream as usual).
    With the stock trigger placement such a kernel only risked reading before the
    flush; here it can run while the stores are still being issued.

    Preconditions, enforced loudly (AssertionError before launch): contiguous 2-D
    `x` and `residual` of one shape, contiguous `weight`, at most 2^31 - 1 elements
    (the compact layout's offsets).

    Numerics: bitwise the stock gemma add+norm on the residual and `x` (identical
    instruction sequence up to the trigger, measured bitwise in CUDA-graph chains);
    against the fp32 reference, the stock norm's band.

    Certified: hidden 5120 bf16 at 1, 8 and 32 rows. Not certified: fp16 / fp32,
    other widths, more than 32 rows (the target runs the 1-row or stock kernel
    there).
    """

    def __call__(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Writes `x` (the normalized output) and `residual` (the sum) in place and
        returns both, in that order."""
        gemma_fused_add_rmsnorm_early_trigger(x, residual, weight, eps, get_env_enable_pdl())
        return x, residual

    def reference(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(x_out, residual_out)`: the stock gemma add+norm's fp32 reference."""
        return flashinfer_gemma_fused_add_rmsnorm.reference(x, residual, weight, eps)

    def is_valid(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> None:
        # The op writes both tensors: one buffer passed twice ends up holding the
        # normalized output, and the returned residual is not the sum.
        assert x.data_ptr() != residual.data_ptr(), (
            "x and residual must be distinct buffers; the op writes both"
        )


gemma_add_rmsnorm_early_trigger = GemmaAddRmsnormEarlyTrigger()

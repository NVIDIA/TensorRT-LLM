# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-style fused residual add + RMS normalization (weight applied as ``1 + w``), in
place, via the flashinfer CuTe-DSL kernel behind ``trtllm::flashinfer_gemma_fused_add_rmsnorm``."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The checkpoint's residual stream: hidden 5120, bf16, `rms_norm_eps` 1e-6.
_CKPT = dict(hidden=5120, dtype=torch.bfloat16, eps=1e-6)


class FlashinferGemmaFusedAddRmsnorm(OpWrapper):
    """`residual = x + residual; x = residual / sqrt(mean(residual^2) + eps) * (1 + w)`, in
    place; returns `(x, residual)` -- the same two tensors, written and handed back so the
    call site shows what it touched.

    The residual is the fp32 sum of the two bf16 inputs rounded once; the normalization
    reads that fp32 sum, not the rounded value, and `(1 + w)` is formed in fp32. Only the
    two stores round.

    Fusion boundary: add, normalization and the biased weight. No quantization (the
    NVFP4-epilogue form is `gemma_add_rmsnorm_nvfp4`), no PDL trigger placement
    (`gemma_add_rmsnorm_early_trigger`), one fixed geometry (the one-row-per-CTA
    form is `gemma_add_rmsnorm_1row`). All four write bitwise the same residual and,
    where they write it, the same `x`.
    """

    ARCHS = frozenset({Arch.SM_120})

    # The rows where the target's three gemma add+norm sites fall back to the stock
    # geometry: post_attention_layernorm below 16 rows, the next layer's input_layernorm
    # from 129 rows, the final norm below 16 and from 129 rows.
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode 1 row: post-attention and final norm at c1", spec=dict(rows=1, **_CKPT)),
        Cell(
            why="decode 15 rows: the top of the stock decode band (2 rows a CTA, odd count)",
            spec=dict(rows=15, **_CKPT),
        ),
        Cell(
            why="129 rows: the next layer's input norm and the final norm above the 1-row band",
            spec=dict(rows=129, **_CKPT),
        ),
        Cell(
            why="prefill 8192 rows (max_num_tokens): the layer-boundary norm of a full chunk",
            spec=dict(rows=8192, **_CKPT),
        ),
    )

    note = """
    Metadata consumed: none. Registered only when flashinfer is importable.

    Path: `trtllm::flashinfer_gemma_fused_add_rmsnorm` -> flashinfer
    `gemma_fused_add_rmsnorm` -> the CuTe-DSL `FusedAddRMSNormKernel` with
    `weight_bias=1.0`, compact (contiguous) layout. At hidden 5120 bf16 on sm_120 its
    geometry is 64 threads a row, 2 rows a CTA, 16-byte vectors, cluster 1. PDL
    (`TRTLLM_ENABLE_PDL`, default on) changes scheduling, not results.

    Preconditions: contiguous 2-D bf16 `x` and `residual` of one shape, `weight` bf16
    `[hidden]`. The CUDA-JIT norm (`FLASHINFER_USE_CUDA_NORM=1`) is a different
    implementation and is not certified; the gemma NVFP4, early-trigger and 1-row
    entries are bitwise equal to the CuTe-DSL kernel, not to that one.

    Numerics: residual bitwise (one rounding of an exact fp32 sum); `x` inside torch's
    default bf16 band of the fp32 reference (rsqrt.approx, warp-tree row sum).

    Certified: hidden 5120 bf16 at 1, 15, 129 and 8192 rows. Not certified: fp16 /
    fp32, other widths, strided rows (the strided-kernel variant), more than 2^31 - 1
    elements (also the strided variant).
    """

    def __call__(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Writes `x` (the normalized output) and `residual` (the sum) in place and
        returns both, in that order."""
        torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, residual, weight, eps)
        return x, residual

    def reference(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(x_out, residual_out)`: fp32 throughout, `(1 + w)` in fp32, each rounded once.

        Returns rather than mutates, so a cell can drive both sides from the same inputs.
        """
        h = x.float() + residual.float()
        normed = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * (1.0 + weight.float())).to(x.dtype), h.to(residual.dtype)

    def is_valid(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> None:
        # The op writes both tensors: one buffer passed twice ends up holding the
        # normalized output, and the returned residual is not the sum.
        assert x.data_ptr() != residual.data_ptr(), (
            "x and residual must be distinct buffers; the op writes both"
        )


flashinfer_gemma_fused_add_rmsnorm = FlashinferGemmaFusedAddRmsnorm()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma fused residual add + RMS normalization whose output is NVFP4-quantized in the same
kernel: the post-attention norm handing the dense MLP's gate_up its FP4 input.

The kernel is flashinfer's CuTe-DSL ``FusedAddRMSNormKernel`` (the stock gemma add+norm)
subclassed with an NVFP4 epilogue that reproduces ``fp4_quantize``'s arithmetic bit for
bit; it lives in ``_gemma_add_rmsnorm_nvfp4``.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ._gemma_add_rmsnorm_nvfp4 import gemma_fused_add_rmsnorm_nvfp4_cute
from .flashinfer_gemma_fused_add_rmsnorm import flashinfer_gemma_fused_add_rmsnorm

#: The checkpoint's residual stream: hidden 5120, `rms_norm_eps` 1e-6.
_CKPT = dict(hidden=5120, eps=1e-6)


class GemmaAddRmsnormNvfp4(OpWrapper):
    """`residual = x + residual` in place, then the NVFP4 quantization of
    `y = bf16(residual / sqrt(mean(residual^2) + eps) * (1 + w))`; returns
    `(fp4 [M, H/2] uint8, sf, residual)`.

    `sf` is the 1-D 128x4-swizzled e4m3 scale buffer of `fp4_quantize(y, global_scale, 16,
    False, True)`, `ceil(M / 128) * 128 * H / 16` bytes, its padding rows `[M, ceil128(M))`
    zeroed. `residual` is the tensor passed in, written and handed back. `x` is read only:
    unlike the stock add+norm, `y` is never stored in BF16.

    Fusion boundary: add, normalization, the biased weight and the NVFP4 quantization of
    the BF16-rounded `y`. The consumer is gate_up's NVFP4 GEMM, which takes `(fp4, sf)`
    as its already-quantized input.
    """

    ARCHS = frozenset({Arch.SM_120})

    # The post-attention norm runs here at 33..112 and from 129 rows (by the step's row
    # count); the final layer's row slice runs it on one row per sequence, routed by the
    # unsliced step. `global_scale` (`448 * 6 / amax`, per layer) walks e4m3 scale
    # saturation (134.4), normal and subnormal scales (0.05).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode 33 rows: the first row count above the BF16 decode band",
            spec=dict(rows=33, global_scale=134.4, **_CKPT),
        ),
        Cell(
            why="decode 112 rows: the top of the 33..112 band, 16 SF padding rows",
            spec=dict(rows=112, global_scale=100.0, **_CKPT),
        ),
        Cell(
            why="129 rows: past the 113..128 BF16 band, 127 SF padding rows",
            spec=dict(rows=129, global_scale=1.0, **_CKPT),
        ),
        Cell(
            why="mixed step 2111 rows (2048 context + 63 decode): a partial last SF tile",
            spec=dict(rows=2111, global_scale=134.4, **_CKPT),
        ),
        Cell(
            why="prefill 8192 rows (max_num_tokens): whole SF tiles, no padding CTAs",
            spec=dict(rows=8192, global_scale=0.05, **_CKPT),
        ),
        Cell(
            why="final-layer row slice: 3 rows (one per sequence) of a >= 2048-row step",
            spec=dict(rows=3, global_scale=100.0, **_CKPT),
        ),
    )

    note = """
    Metadata consumed: none. `global_scale` is gate_up's static NVFP4 input scale
    (the one `fp4_quantize` would be given), one fp32 element.

    Kernel: flashinfer's CuTe-DSL `FusedAddRMSNormKernel` geometry (hidden 5120 bf16:
    64 threads a row, 2 rows a CTA, 16-byte vectors, cluster 1), its pass 1 and the
    `y` expression unchanged; the epilogue rounds `y` to BF16 and quantizes it in
    registers: block max over 16 values (two lanes, one xor-1 shuffle),
    `SF = e4m3_satfinite(gs * (vmax * rcp.approx(6)))`,
    `outScale = rcp.approx(f32(SF) * rcp.approx(gs))` (0 for an all-zero block),
    `codes = cvt.rn.satfinite.e2m1x2(y * outScale)` -- `cvt_warp_fp16_to_fp4`'s
    sequence. Grid `ceil(M / 2)` row CTAs plus 8 CTAs zeroing the SF padding rows of
    the last 128-row tile when `M % 128 != 0`. Compiled once per (dtype, hidden, PDL,
    SM) on first use. PDL from `TRTLLM_ENABLE_PDL` (default on): waits before its
    loads, triggers after its stores.

    Preconditions, enforced loudly (AssertionError / ValueError before launch):
    contiguous 2-D `x` and `residual` of one shape, contiguous `weight`, fp32
    `global_scale`, and a hidden size whose stock geometry is one CTA a row-pair with
    8-wide vectors covering the row exactly (5120 does).

    Numerics: bitwise equal to `gemma fused add+RMSNorm (CuTe-DSL)` followed by
    `fp4_quantize` -- the residual, every FP4 byte and the whole SF buffer including
    its padding (measured: rows 1..32, 48, 64 and 127..129 over four input families
    and four global scales, 444 configurations; every row count 33..128; 160, 2047,
    2048, 4096, 8181 and 8192 rows). Against the fp32 reference: the residual
    bitwise; `y` inside the stock norm's band; the quantization exact wherever the
    BF16 `y` agrees, up to `fp4_quantize`'s reciprocal tie window.

    Certified: hidden 5120 bf16 at 3, 33, 112, 129, 2111 and 8192 rows, global scale
    134.4 / 100 / 1.0 / 0.05. Not certified: fp16, other widths. Correct at every
    row count; at decode steps of 1..32 and 113..128 rows the target runs the BF16
    norm and a separate quantize instead (measured faster there in serving).

    Hard limits
      M  row index and FP4 offset in int64; SF buffer `ceil128(M) * H / 16` bytes
      (largest run 8192 rows)
    """

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        global_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Writes `residual` in place; returns `(fp4, sf, residual)`."""
        fp4, sf = gemma_fused_add_rmsnorm_nvfp4_cute(
            x, residual, weight, global_scale, eps, enable_pdl=get_env_enable_pdl()
        )
        return fp4, sf, residual

    def reference(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        global_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(fp4, sf, residual_out)`: the stock gemma add+norm's fp32 reference rounded once
        to BF16, then `fp4_quantize`'s native-torch NVFP4 with swizzled scales. Neither input
        is mutated."""
        y, h = flashinfer_gemma_fused_add_rmsnorm.reference(x, residual, weight, eps)
        packed, sf = fp4_quantize.reference(y, global_scale, 16, False, True)
        return packed, sf, h

    def is_valid(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        global_scale: torch.Tensor,
    ) -> None:
        # The epilogue loads global_scale[0] and nothing else, so a per-row scale is
        # silently applied as its first element to every row.
        assert global_scale.numel() == 1, (
            "global_scale must hold exactly one element; the kernel reads global_scale[0] "
            "and applies it to every row"
        )


gemma_add_rmsnorm_nvfp4 = GemmaAddRmsnormNvfp4()

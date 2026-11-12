# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-row W4A16 NVFP4 GEMV on CUDA cores (``native_ops.nvfp4_w4a16_gemv``, built from the target's ``csrc/``).

At one to three rows the decode copies of the FP8 per-tensor projections run a
GEMV instead of Marlin: each CTA owns two output rows over the whole K, issues its
weight and scale loads before the PDL wait (they do not depend on the producer of
``x``) and triggers its dependents once its FMAs are done. The weight is the same
NVFP4 values as the Marlin copy in a second layout (``_nvfp4_recipes.gemv_layout``):
packed E2M1 rows and linear E4M3 block scales.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ... import native_ops
from ..quantization._nvfp4_reference import dequantize


def _cell(why: str, n: int, k: int, m: int) -> Cell:
    return Cell(why=why, spec=dict(n=n, k=k, m=m))


class Nvfp4W4a16Gemv(OpWrapper):
    """``bf16((x @ (e2m1 * sf).T) * global_scale)`` for BF16 ``x`` ``[M, K]``, ``M <= 4``.

    ``weight`` uint8 ``[N, K/2]`` (element ``2i`` in the low nibble), ``weight_sf`` uint8
    ``[N, K/16]`` linear E4M3, ``global_scale`` a float. Returns a fresh ``[M, N]`` BF16
    tensor.
    """

    ARCHS = frozenset({Arch.SM_120})

    # The decode copies' GEMV rows (dispatch table): (5120, 6144) out_proj / o_proj and GDN
    # in_proj_qkvz (16384, 5120) at rows 1-3, attention qkv_proj (14336, 5120) at rows 1-2
    # (mixed steps of up to 3 rows included; pure decode qkvz takes the fused conv variant).
    CELLS: tuple[Cell, ...] = (
        _cell("out_proj/o_proj M=1", 5120, 6144, 1),
        _cell("out_proj/o_proj M=3: the (5120, 6144) GEMV cap", 5120, 6144, 3),
        _cell("qkvz M=1 (mixed-step row; decode takes gemv_conv)", 16384, 5120, 1),
        _cell("qkvz M=3: the qkvz GEMV cap", 16384, 5120, 3),
        _cell("qkv M=1", 14336, 5120, 1),
        _cell("qkv M=2: the qkv GEMV cap", 14336, 5120, 2),
    )

    note = """
    Metadata consumed: none. `weight`, `weight_sf`, `global_scale` are
    `_nvfp4_recipes.gemv_layout(...)` (`GemvNvfp4Weight`).

    Preconditions: `x` BF16, contiguous `[M, K]`, `1 <= M <= 4`, its data pointer
    16-byte aligned (the kernel loads `x` with 16-byte vectors and does not check);
    K in {5120, 6144}, N even (the op rejects other shapes and devices loudly).

    Numerics: E2M1 and E4M3 are converted exactly; products and sums in fp32 with a
    fixed order (per 16-block partials, a warp xor-shuffle tree, warps in order), times
    `global_scale`, one rounding to BF16 -- bitwise repeatable, reordered against the
    Marlin copy of the same values. Unlike Marlin, subnormal block scales are kept.

    PDL: launched with the PDL attribute (TRTLLM_ENABLE_PDL); weight and scale loads are
    issued before `griddepcontrol.wait`, `x` is read after it.

    Hard limits
      M   1..4 (the op's limit)     the target uses 1..3
      K   5120 or 6144              both driven
    """

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, weight_sf: torch.Tensor, global_scale: float
    ) -> torch.Tensor:
        return native_ops.nvfp4_w4a16_gemv(x, weight, weight_sf, global_scale)

    def reference(
        self, x: torch.Tensor, weight: torch.Tensor, weight_sf: torch.Tensor, global_scale: float
    ) -> torch.Tensor:
        """Exact dequantization, one fp32 product (TF32 off), the global scale, one rounding."""
        w = dequantize(weight, weight_sf)
        with true_fp32_matmul():
            return ((x.float() @ w.t()) * global_scale).to(torch.bfloat16)

    def is_valid(
        self, x: torch.Tensor, weight: torch.Tensor, weight_sf: torch.Tensor, global_scale: float
    ) -> None:
        # The op checks dtype, contiguity and shapes; it does not check the alignment its
        # 16-byte vector loads of `x` need (a row slice of a wider tensor can miss it).
        assert x.data_ptr() % 16 == 0, "x must be 16-byte aligned (16-byte vector loads)"


nvfp4_w4a16_gemv = Nvfp4W4a16Gemv()

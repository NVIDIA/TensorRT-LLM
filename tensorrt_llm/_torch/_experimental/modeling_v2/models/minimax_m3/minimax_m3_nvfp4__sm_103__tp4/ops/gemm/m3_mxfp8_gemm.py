# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode-graph MXFP8 GEMM on an activation its producer already quantized (rows the GEMV does not serve)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.custom_ops.torch_custom_ops import mxfp8_gemm_autotuned

from .m3_mxfp8_linear import mxfp8_reference


class M3Mxfp8Gemm(OpWrapper):
    """``act @ weight^T`` for an MXFP8 activation (e4m3 [M, K] + 128x4-swizzled UE8M0 scales) and an MXFP8 weight
    [N, K] (swizzled scales), bf16 out. The GEMM backend per token bucket is the AutoTuner's (the same runner, op key
    and tactic cache as the engine-tuned quantize + GEMM); ``tune`` profiles in the engine's tuning pass."""

    ARCHS = frozenset({Arch.SM_103})

    # The decode dispatch hands it the boundary / merge MXFP8 activation on rows the GEMV does not serve: the
    # unsplit shared down projection at 17..32 rows.
    CELLS: tuple[Cell, ...] = (
        Cell(why="shared expert down projection, 17 rows", spec=dict(m=17, n=6144, k=768)),
        Cell(why="shared expert down projection, 32 rows", spec=dict(m=32, n=6144, k=768)),
    )

    note = """
    Exactly the MXFP8 product (fp32 accumulation, one bf16 rounding): `reference` dequantizes both operands and
    multiplies in fp32; the test's band covers the accumulation order.
    """

    def __call__(
        self,
        act: torch.Tensor,
        act_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype,
        tune: bool,
    ) -> torch.Tensor:
        return mxfp8_gemm_autotuned(act, act_scale, weight, weight_scale, output_dtype, tune=tune)

    def reference(
        self,
        act: torch.Tensor,
        act_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype,
        tune: bool,
    ) -> torch.Tensor:
        """The dequantized product; ``tune`` is ignored."""
        return mxfp8_reference(act, act_scale, weight, weight_scale).to(output_dtype)


m3_mxfp8_gemm = M3Mxfp8Gemm()

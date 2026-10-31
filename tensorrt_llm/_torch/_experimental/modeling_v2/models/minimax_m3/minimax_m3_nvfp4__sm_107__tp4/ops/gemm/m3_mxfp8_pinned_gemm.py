# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode-graph MXFP8 GEMM at a fixed FlashInfer tactic (MX-TAC): the same choice on every rank and every start."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.custom_ops.torch_custom_ops import FlashInferMXFP8GemmRunner

from .m3_mxfp8_linear import mxfp8_reference

# One runner per output dtype (a runner holds no per-call state).
_RUNNERS: Dict[Tuple[type, torch.dtype], object] = {}


def _runner(dtype: torch.dtype):
    runner = _RUNNERS.get((FlashInferMXFP8GemmRunner, dtype))
    if runner is None:
        runner = _RUNNERS[(FlashInferMXFP8GemmRunner, dtype)] = FlashInferMXFP8GemmRunner(dtype)
    return runner


class M3Mxfp8PinnedGemm(OpWrapper):
    """``act @ weight^T`` for an MXFP8 activation (e4m3 [M, K] + swizzled UE8M0 scales) and an MXFP8 weight [N, K]
    (swizzled scales) on FlashInfer's MXFP8 GEMM at ``tactic``, no AutoTuner; ``output_dtype`` out."""

    ARCHS = frozenset({Arch.SM_107})

    # The decode graph's MXFP8 GEMMs on rows the GEMVs do not serve: the shared down projection on its boundary
    # activation (17..32 rows) and the bf16 projections past the GEMV rows.
    CELLS: tuple[Cell, ...] = (
        Cell(why="shared expert down projection, 17 rows", spec=dict(m=17, n=6144, k=768)),
        Cell(why="packed QKV + index projection, 48 rows", spec=dict(m=48, n=4992, k=6144)),
    )

    note = """
    The MXFP8 product (fp32 accumulation, one rounding to the output dtype): `reference` dequantizes both operands and
    multiplies in fp32; the test's band covers the accumulation order. The tactic only picks the kernel.
    """

    def __call__(
        self,
        act: torch.Tensor,
        act_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype,
        tactic: int,
    ) -> torch.Tensor:
        return _runner(output_dtype).forward([act, act_scale, weight, weight_scale], tactic=tactic)

    def reference(
        self,
        act: torch.Tensor,
        act_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype,
        tactic: int,
    ) -> torch.Tensor:
        """The dequantized product; ``tactic`` is ignored."""
        return mxfp8_reference(act, act_scale, weight, weight_scale).to(output_dtype)


m3_mxfp8_pinned_gemm = M3Mxfp8PinnedGemm()

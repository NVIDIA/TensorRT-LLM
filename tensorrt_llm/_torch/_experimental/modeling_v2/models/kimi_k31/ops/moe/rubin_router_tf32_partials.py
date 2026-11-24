# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 FP32 router on SM107: the 8 K-slice partials of cuBLAS's TF32 split-K GEMM."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

from ._rubin_moe_common import HIDDEN, NUM_EXPERTS

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.moe.rubin_router_tf32 import SLICE_K, SPLITS
    from ...kernels.moe.rubin_router_tf32 import run_router_tf32_partials as _launch
else:
    SPLITS, SLICE_K = 8, HIDDEN // 8


class RubinRouterTf32Partials(OpWrapper):
    """FP32 ``[8, M, 896]``: K slice ``s`` (``[896 s, 896 s + 896)``) of ``x @ weight.T`` for FP32
    ``x [M, 7168]`` and ``weight [896, 7168]``, on TF32 tensor cores, allocated per call.

    Summed in slice order, ``((p[0] + p[1]) + ...) + p[7]``, the partials have the bits of
    ``F.linear(x, weight)`` under TF32 (cuBLAS's ``nvjet_sm107_sss_tf32`` split-K GEMM and its
    ``splitKreduce``) when both operands hold TF32-exact values, as BF16 values widened to FP32 do:
    each partial is cuBLAS's own slice partial bit for bit. The routing tail
    (``rubin_route_topk_out`` on the partials) does that sum. One launch on the current stream;
    the weight streams in before ``griddepcontrol.wait`` (PDL), the rows after it.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=why, spec=dict(rows=rows))
        for rows, why in (
            (96, "DEP16 generation b12 k7: 96 rows, the router_tf32_splitk lever's rows"),
            (88, "b11 k7: 88 rows, a part row tile"),
            (64, "b8 k7: 64 rows"),
            (48, "b6 k7: 48 rows"),
            (160, "160 rows: two row tiles"),
        )
    )
    note = """
    ``x`` and ``weight`` are FP32, dense and 16-byte aligned on one SM107 device, ``weight`` exactly
    ``[896, 7168]``; their values are TF32-exact (low 13 mantissa bits zero), as K3.1's widened BF16
    router input and BF16-checkpoint router weight are. The op does not check the values: operands
    with low mantissa bits are truncated by the tensor cores and the bits then differ from cuBLAS's.
    No workspace or state; a captured CUDA graph replays the launch as is.
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = torch.empty(SPLITS, x.shape[0], NUM_EXPERTS, dtype=torch.float32, device=x.device)
        return _launch(x, weight, out)

    def reference(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Each slice's FP64 product rounded once to FP32: ``[8, M, 896]``."""
        xd, wd = x.double(), weight.double()
        return torch.stack(
            [
                (
                    xd[:, s * SLICE_K : (s + 1) * SLICE_K]
                    @ wd[:, s * SLICE_K : (s + 1) * SLICE_K].t()
                )
                for s in range(SPLITS)
            ]
        ).float()

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor) -> None:
        """The launcher takes raw layouts: a mistyped, strided, misaligned or misshaped operand
        is read silently wrong."""
        assert x.dim() == 2 and x.shape[0] >= 1 and x.shape[1] == HIDDEN, (
            f"x must be [M >= 1, {HIDDEN}], got {tuple(x.shape)}"
        )
        assert tuple(weight.shape) == (NUM_EXPERTS, HIDDEN), (
            f"weight must be [{NUM_EXPERTS}, {HIDDEN}], got {tuple(weight.shape)}"
        )
        for name, t in (("x", x), ("weight", weight)):
            assert t.dtype == torch.float32, f"{name} must be FP32"
            assert t.is_contiguous() and t.data_ptr() % 16 == 0, (
                f"{name} must be dense and 16-byte aligned"
            )
            assert t.device == x.device, f"{name} is on another device"


rubin_router_tf32_partials = RubinRouterTf32Partials()

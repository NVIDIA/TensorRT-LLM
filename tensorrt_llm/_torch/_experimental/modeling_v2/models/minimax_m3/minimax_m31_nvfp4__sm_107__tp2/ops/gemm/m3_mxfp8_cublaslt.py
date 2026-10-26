# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 GEMM through cuBLASLt for generation steps wider than the decode kernels' rows.

cuBLASLt's block-scaled MXFP8 matmul (``torch._scaled_mm`` with UE8M0 block scales), on an MXFP8 activation or on a
bf16 one it quantizes first.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8QuantizeRunner

from .m3_mx_gemv import DENSE_DOWN, DENSE_GATE_UP, O_PROJ, QKV_SPARSE, SHARED_DOWN, SHARED_GATE_UP
from .m3_mxfp8_linear import mxfp8_reference

# The projections the cuBLASLt GEMM takes, and its rows: past the R1-VR-TP8 decode bound (32) up to the K7 verify rows
# at batch 8 (64). SM107 microbench (us/call on the same MXFP8 activation, 33..64 rows): QKV_SPARSE
# 6.94..6.98 vs FlashInfer's 10.14..10.18, O_PROJ 3.06..3.10 vs 3.81..3.86, SHARED_GATE_UP 6.81..6.82 vs 10.04..10.08,
# SHARED_DOWN 2.49..2.51 vs 2.86..2.87, DENSE_GATE_UP 7.21..7.24 vs 10.46..10.47, DENSE_DOWN 3.62..3.77 vs 4.48..4.64.
SHAPES = (QKV_SPARSE, O_PROJ, SHARED_GATE_UP, SHARED_DOWN, DENSE_GATE_UP, DENSE_DOWN)
MIN_TOKENS = 33  # capacity: fixed: past the R1-VR-TP8 decode bound (32 rows)
MAX_TOKENS = 64  # capacity: fixed: the K7 verify rows at batch 8

_QUANTIZE = {}


def mxfp8_quantize(x: torch.Tensor):
    """The bf16 activation's MXFP8 form by FlashInfer's CuTe DSL quantize.

    E4M3 [M, K] + 128x4-swizzled UE8M0 scales: the bytes of ``torch.ops.trtllm.mxfp8_quantize(x, True)``.
    """
    runner = _QUANTIZE.get(x.dtype)
    if runner is None:
        runner = _QUANTIZE[x.dtype] = MXFP8QuantizeRunner(x.dtype)
    act, act_scale = runner([x], tactic=MXFP8QuantizeRunner.CUTE_DSL)
    return act.view(x.shape[0], x.shape[1]), act_scale


def _e4m3(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float8_e4m3fn else t.view(torch.float8_e4m3fn)


def _ue8m0(scale: torch.Tensor) -> torch.Tensor:
    """The swizzled scale bytes as one flat UE8M0 vector (whatever element type the buffer was allocated with)."""
    return scale.reshape(-1).view(torch.uint8).view(torch.float8_e8m0fnu)


class M3Mxfp8Cublaslt(OpWrapper):
    """``act @ weight^T`` in bf16 for an MXFP8 weight [N, K] (swizzled scales), via cuBLASLt.

    The activation is MXFP8 (``act`` e4m3 [M, K] + ``act_scale`` swizzled) or, with ``act_scale`` None, bf16 and
    quantized first. The GEMM is cuBLASLt's block-scaled MXFP8 matmul (VEC32 UE8M0 scales), reached through
    ``torch._scaled_mm``; the algorithm is cuBLASLt's heuristic for the shape and the process's workspace (no
    timing-based selection).
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"[{n}, {k}], {m} rows, {src}", spec=dict(m=m, n=n, k=k, prequant=src == "MXFP8"))
        for (n, k) in SHAPES
        for m in range(MIN_TOKENS, MAX_TOKENS + 1)
        for src in ("MXFP8", "bf16")
    )

    note = """
    Exactly the MXFP8 product (fp32 accumulation, one bf16 rounding): `reference` dequantizes both operands and
    multiplies in fp32; the test's band covers the accumulation order. Against FlashInfer's MXFP8 GEMM on the same
    operands (SM107) the output was within 0.61 ULP[bf16] at every shape and row, bit-identical on most cells.
    Not covered: shapes outside SHAPES, rows outside 33..64.
    """

    def __call__(
        self,
        act: torch.Tensor,
        act_scale: Optional[torch.Tensor],
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        if act_scale is None:
            act, act_scale = mxfp8_quantize(act)
        return torch._scaled_mm(
            _e4m3(act),
            _e4m3(weight).t(),
            scale_a=_ue8m0(act_scale),
            scale_b=_ue8m0(weight_scale),
            out_dtype=torch.bfloat16,
        )

    def reference(
        self,
        act: torch.Tensor,
        act_scale: Optional[torch.Tensor],
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """The dequantized product of the MXFP8 operands (a bf16 activation quantized first)."""
        if act_scale is None:
            act, act_scale = mxfp8_quantize(act)
        return mxfp8_reference(act, act_scale, weight, weight_scale).to(torch.bfloat16)

    def is_valid(
        self,
        act: torch.Tensor,
        act_scale: Optional[torch.Tensor],
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        m, k = act.shape
        n = weight.shape[0]
        assert (n, k) in SHAPES and MIN_TOKENS <= m <= MAX_TOKENS, (
            f"m3_mxfp8_cublaslt is certified for {SHAPES} at {MIN_TOKENS}..{MAX_TOKENS} rows, got [{n}, {k}] x {m}"
        )
        assert act.is_contiguous() and weight.is_contiguous(), (
            "m3_mxfp8_cublaslt takes contiguous operands"
        )
        assert act_scale is not None or act.dtype == torch.bfloat16, (
            "a bf16 activation, or an MXFP8 one + scales"
        )


m3_mxfp8_cublaslt = M3Mxfp8Cublaslt()

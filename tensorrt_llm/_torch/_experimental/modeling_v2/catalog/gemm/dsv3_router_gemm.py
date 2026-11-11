# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE router logits: BF16 hidden rows times the column-major BF16 router weight, FP32 out, via
the trtllm dsv3_router_gemm_op (a small-M kernel for the router shapes it is built for, cuBLASLt
otherwise)."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: GLM-5.3-Flash's router: hidden 4096 -> 288 routed experts, replicated on every rank.
GLM_ROUTER = dict(hidden=4096, experts=288)


class Dsv3RouterGemm(OpWrapper):
    """`mat_a @ mat_b (+ bias)`, accumulated in fp32, one `dsv3_router_gemm_op` call.

    `mat_a` is `[M, K]` row-major, `mat_b` is `[K, N]` read column-major -- the stored
    `[N, K]` router weight's `.t()`, so no transpose kernel runs. GLM-5.3-Flash calls it with
    BF16 `[M, 4096]` hidden rows, its BF16 `[288, 4096]` router weight `.t()`, no bias and
    `out_dtype=torch.float32`: FP32 `[M, 288]` logits, the input of the in-kernel routing of
    `fp8_block_scale_moe_runner`.

    Two kernels behind one call, both fp32-accumulated over exact bf16 products:

    * `1 <= M <= 16`, BF16 operands, FP32 out, no bias, and `(K, N)` one of the shapes the op
      is built for -- `(4096, 288)` is GLM-5.3-Flash's -- run a small-M kernel
      (`invokeRouterGemm`): one 128-thread CTA per expert column, each thread summing 32
      products in order with IEEE fp32 adds, then a warp butterfly and a 4-warp sum.
    * Everything else falls back to cuBLASLt (`cublas_mm_out`, compute type 32F) on the
      tensor cores, whose fp32 accumulation is coarser than IEEE round-to-nearest -- tens of
      fp32 ulp of the row scale at GLM's shape rather than about one: at GLM's shape, every
      step of more than 16 rows, and any call with a bias or a non-FP32 output.
    """

    ARCHS = frozenset({Arch.SM_100})

    # Only GLM-5.3-Flash calls this entry. Its own cells are the boundary between the two
    # kernels at GLM's router shape; the target's row counts are in the test.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="16 rows: the small-M kernel's last row count",
            spec=dict(rows=16, **GLM_ROUTER),
        ),
        Cell(
            why="17 rows: the first row count on the cuBLASLt fallback",
            spec=dict(rows=17, **GLM_ROUTER),
        ),
    )

    note = """
    Source: compiled-in C++, `torch.ops.trtllm.dsv3_router_gemm_op` (thop/dsv3RouterGemmOp.cpp).
    The small-M kernel is instantiated per row count (1..16) and per built-in `(K, N)`; the
    `(4096, 288)` instantiation is GLM-5.3-Flash's. Above 16 rows, or for any other shape,
    dtype, output dtype or a bias, the op runs cuBLASLt instead -- slower at small M, and
    accumulated on the tensor cores. The two kernels' logits are not bitwise equal; both stay
    within a few times 1e-5 of the row scale (the test's band), far below what a bf16
    rounding of the logits would move.

    Layout handling is the op's own and loud: `mat_a` must have a unit column stride and
    `mat_b` a unit row stride (it raises otherwise); an operand with padded rows or a
    misaligned start is copied into a dense 16-byte-aligned layout before either kernel
    reads it. So no layout guard is needed.

    The output is a new `[M, N]` tensor in `out_dtype` (default `mat_a.dtype`). For the
    routing that consumes it the FP32 output is what matters: rounding the logits to bf16 can
    reorder near-equal experts.

    Not certified: a bias (it goes to the same cuBLASLt call as `cublas_mm`, whose hazards
    `is_valid` repeats), outputs other than FP32, operand dtypes other than BF16, router
    shapes other than GLM-5.3-Flash's, CUDA-graph capture, sm_103.
    """

    def __call__(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return torch.ops.trtllm.dsv3_router_gemm_op(mat_a, mat_b, bias, out_dtype)

    def reference(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Accumulated in float64 and rounded once to `out_dtype`.

        The op's own output is FP32, so an fp32-accumulated reference would be no more
        accurate than the op and the test's band would measure the reference. In float64 the
        bf16 products and their 4096-term sums are exact to well below one fp32 ulp, and no
        TF32 setting applies.
        """
        ref = mat_a.double() @ mat_b.double()
        if bias is not None:
            ref = ref + bias.double()
        return ref.to(out_dtype if out_dtype is not None else mat_a.dtype)

    def is_valid(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Only the bias can be wrong without a loud error.

        Every layout mistake the op either rejects or repairs (see the note). A bias sends the
        call to the cuBLASLt epilogue that `cublas_mm` runs, which takes a bias of the wrong
        dtype or shape and returns a wrong answer, and ignores a bias on fp32 operands; those
        conditions are repeated here from that entry. GLM-5.3-Flash passes no bias.
        """
        if bias is None:
            return
        out_dt = out_dtype if out_dtype is not None else mat_a.dtype
        assert mat_a.dtype != torch.float32, "bias is silently ignored for fp32 inputs"
        assert bias.dtype == out_dt, f"bias dtype {bias.dtype} must equal the output dtype {out_dt}"
        assert bias.shape == (mat_b.shape[1],) and bias.is_contiguous(), (
            "bias must be a contiguous [N] tensor"
        )


dsv3_router_gemm = Dsv3RouterGemm()

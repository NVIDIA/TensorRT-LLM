# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General matmul with optional fused bias via the trtllm cuBLASLt gemm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import ULP, Arch, Cell, OpWrapper, assert_within_ulp, true_fp32_matmul


class _CublasMm(OpWrapper):
    """`mat_a @ mat_b (+ bias)`, fp32-accumulated, one cublas_mm call.

    `mat_b` is read column-major, which is what a stored weight's `.t()` already
    is -- the targets pass `weight.t()` and no transpose kernel runs.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Both targets use this for every dense projection, so the cells are the
    # three argument shapes that differ in kind rather than in size: fused bias
    # (gpt-oss carries one on qkv, o and the router), no bias (deepseek's
    # projections), and an out_dtype that is not the input's (deepseek's router
    # asks for fp32 logits).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="gpt-oss qkv with fused bias: hidden 2880 -> (64 + 8 + 8) * 64",
            spec=dict(m=16, k=2880, n=5120, dtype=torch.bfloat16, bias=True, out_dtype=None),
        ),
        Cell(
            why="deepseek q_a, no bias: hidden 7168 -> q_lora 1536",
            spec=dict(m=16, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek router: bf16 in, fp32 logits out, 256 experts",
            spec=dict(
                m=16, k=7168, n=256, dtype=torch.bfloat16, bias=False, out_dtype=torch.float32
            ),
        ),
        Cell(
            why="decode-shaped: one token, the shape every projection sees per step",
            spec=dict(m=1, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
    )

    note = """
    `output_buffer_kind` and `group` are the op's multi-stream and collective
    arguments. Both shipped targets leave them at their defaults, so neither is
    a cell axis and neither is certified here.
    """

    def __call__(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> torch.Tensor:
        return torch.ops.trtllm.cublas_mm(mat_a, mat_b, bias, out_dtype, output_buffer_kind, group)

    def reference(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> torch.Tensor:
        """fp32-accumulated, TF32 off -- see `true_fp32_matmul`."""
        with true_fp32_matmul():
            ref = mat_a.float() @ mat_b.float()
        if bias is not None:
            ref = ref + bias.float()
        return ref.to(out_dtype if out_dtype is not None else mat_a.dtype)

    def is_valid(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> None:
        # The kernel reads mat_a as dense row-major and mat_b as dense
        # column-major; other layouts produce silently wrong results.
        assert mat_a.is_contiguous(), "mat_a must be dense row-major [M, K]"
        assert mat_b.stride(0) == 1 and mat_b.stride(1) == mat_b.shape[0], (
            "mat_b must be dense column-major [K, N] (e.g. weight.t())"
        )
        if bias is not None:
            out_dt = out_dtype if out_dtype is not None else mat_a.dtype
            # A bias in the wrong dtype or shape is accepted by the op and
            # produces silently wrong results; with fp32 inputs the bias is
            # accepted but silently ignored.
            assert mat_a.dtype != torch.float32, "bias is silently ignored for fp32 inputs"
            assert bias.dtype == out_dt, "bias dtype must equal the output dtype"
            assert bias.shape == (mat_b.shape[1],) and bias.is_contiguous(), (
                "bias must be a contiguous [N] tensor"
            )

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """One GEMM's worth of accumulation-order difference.

        Kernel and reference consume bit-identical operands and differ only in
        the order cuBLAS and torch sum over K. Over random operands a few
        outputs land near zero, where that reordering shows up as a relative
        difference of 10 on an absolute difference of 1.5e-4 -- which is what
        torch's default band rejects, and what no bf16 result could represent
        anyway: one ulp at these rows' scale is about 6e-1.

        So: 4 ulp element-wise against the row scale, 2 ulp of relative RMS.
        Measured worst case across this entry's cells is 1.5e-4 absolute, or
        roughly 3e-4 ulp -- four orders of margin, and a band that a genuinely
        wrong product would miss by orders rather than squeak past.
        """
        # Scored in bf16 ulp even when the output is fp32: every cell's
        # operands are bf16, so the fp32 out_dtype the router asks for is a
        # wider container for a bf16-limited product, not a more precise one.
        # Against fp32's own ulp the same 2.4e-6 difference reads as 5x over.
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=ULP[torch.bfloat16])


cublas_mm = _CublasMm()

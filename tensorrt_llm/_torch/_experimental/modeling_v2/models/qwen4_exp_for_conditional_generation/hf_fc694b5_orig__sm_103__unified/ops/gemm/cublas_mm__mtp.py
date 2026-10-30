# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General matmul with optional fused bias via the trtllm cuBLASLt gemm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class CublasMm(OpWrapper):
    """`mat_a @ mat_b (+ bias)`, fp32-accumulated, one cublas_mm call.

    `mat_b` is read column-major, which is what a stored weight's `.t()` already
    is -- the targets pass `weight.t()` and no transpose kernel runs.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # Both targets use this for every dense projection. This entry's test
    # carries their three argument shapes that differ in kind rather than in size: fused
    # bias (gpt-oss carries one on qkv, o and the router), no bias (deepseek's
    # projections), and an out_dtype that is not the input's (deepseek's router
    # asks for fp32 logits).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode-shaped: one token, the shape every projection sees per step",
            spec=dict(m=1, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
    )

    note = """
    `output_buffer_kind` and `group` are the op's multi-stream and collective
    arguments. Both shipped targets leave them at their defaults, so neither is
    a cell axis and neither is certified here.

    The op is compiled C++ (`cpp/tensorrt_llm/thop/cublasScaledMM.cpp`,
    `TORCH_LIBRARY_FRAGMENT(trtllm)`), so it is in the build or not; nothing is
    JIT-compiled. It passes `lda = ldb = k` and `ldc = n` to cuBLASLt rather
    than the tensors' strides, which is why `is_valid` requires dense operands.
    `find_special_algo` looks the call up by `(max(next_pow2(m), 8), k, n)` in
    `cublasScaledMMLut.h`'s `bf16_algo_list`, which holds only DeepSeek's
    router shapes (k 7168, n 256). Every qwen4 call takes cuBLASLt's own
    heuristic, which picks the kernel (and any split-K) per shape, so the
    qwen4 cells are listed per shape rather than inferred from one.

    The qwen4 unified target's MTP-layer cells (site `mtp`, its test's
    `TARGET_CELLS`) were driven on sm_103 only, the only architecture that
    target runs on: fc_e / fc_h (2560 -> 2560, fc_h on rows x hc_count 4, up
    to 8192 x 4 = 32768 rows), the per-rank gated q (2560 -> 3072) and k / v
    (2560 -> 256) at tep4, the replicated index projection (2560 -> 640) and
    the shared-expert gate (2560 -> 1, a 2-byte `ldc`). The target slices the
    gate's [1, 2560] row out of the fused shared-expert weight; its `.t()` has
    strides (1, 2560), the dense column-major form the cells build, and
    computes bitwise what a standalone copy does.

    Measured on GB300 at those cells: at most 1.95 bf16 ulp of the row scale
    element-wise and 0.12 relative RMS, cosine >= 0.9999999. The n = 1 gate
    is scored over its column: a one-element row's scale is the element
    itself, which cancellation over K puts near zero (4.3 and 8.7 ulp per row
    at 8192 rows on two seeds, at most 1.24 over the column).

    Not certified at this site: the MTP shared expert's gate_up (2560 -> 1280)
    and down (640 -> 2560), its router (2560 -> 512) and o_proj (1536 -> 2560)
    -- the last two have trunk cells in `cublas_mm__r1`; the index projection
    at rows other than 1; q / k / v at dep4's full widths (24 / 2 heads); and
    row counts between the certified ones, since cuBLASLt may pick another
    kernel per m.
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


cublas_mm = CublasMm()

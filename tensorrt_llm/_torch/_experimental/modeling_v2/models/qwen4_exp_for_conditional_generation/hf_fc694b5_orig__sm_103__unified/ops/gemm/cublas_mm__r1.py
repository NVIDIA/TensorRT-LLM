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
    JIT-compiled. `find_special_algo` looks the call up by
    `(max(next_pow2(m), 8), k, n)` in `cublasScaledMMLut.h`'s `bf16_algo_list`,
    which holds only DeepSeek's router shapes (k 7168, n 256). Every other
    call, every qwen4 cell included, takes cuBLASLt's own heuristic, which
    picks the kernel (and any split-K) per shape. That is why the qwen4
    cells are listed per shape rather than inferred from one.

    The qwen4 unified target's cells (its test's `TARGET_CELLS`) were driven
    on sm_103 only, the only architecture that target runs on.

    Not certified for that target, though the tp1 forward calls this op
    there: the vision tower's biased GEMMs (hidden 1152, MLP 4304, merger
    4608 -> 2560, bf16 bias); the MTP layer's `fc_e` / `fc_h` at rows other
    than the trunk's k 2560 n 2560 cell; the MTP layer's separate q / k / v /
    index projections; and its n=1 shared-expert gate.
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

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

    A bias goes to cuBLASLt's `CUBLASLT_EPILOGUE_BIAS` epilogue
    (`CublasMMWrapper::setBiasDescriptor`) with no bias data type set, so it is
    read in the output dtype: hence the bias-dtype guard.

    The qwen4 unified target's vision-tower cells (site `vision`, its test's
    `TARGET_CELLS`) were driven on sm_103 only, the only architecture that
    target runs on. They are tp1's VisionTower projections, each with a bf16
    bias: qkv 1152 -> 3456, proj 1152 -> 1152, fc1 1152 -> 4304,
    fc2 4304 -> 1152 at 1024 and 16384 patches, and the patch merger
    4608 -> 4608 and 4608 -> 2560 at 256 and 4096 merged rows. fc1 also runs
    at 32768 patches and merger fc1 at 8192 rows, the envelope's largest tower
    call. The tower runs eager and replicated per rank, so the shapes do not
    change with the member.

    Measured on GB300 at those six shapes, 256 to 32768 rows, three seeds:
    at most 1.98 bf16 ulp of the row scale element-wise and 0.042 relative
    RMS, cosine >= 0.9999998, 99.9th percentile 0.033 ulp. The epilogue adds
    the bias to the fp32 sum before the one rounding: the output sits 0.02 to
    0.04 ulp RMS from that reference and 0.73 from a product rounded to bf16
    before the bias is added.

    A bias pointer off a 16-byte boundary (a contiguous [N] slice at an odd
    element offset) is accepted and changes the kernel cuBLASLt picks. At 13 of
    the 24 (shape, rows) pairs probed (every shape at 256 rows; proj, fc2 and
    merger fc2 at 1024; both merger shapes at 4096 and 16384), the output then
    reads 0.55 to 0.84 ulp
    RMS and up to 2.55 element-wise from the reference. That is within the band, but it is not the cells'
    precision and it is not certified. A target keeps each bias in a tensor
    of its own, as the tp1 tower's weight table does.

    Not certified at this site: other row counts (cuBLASLt may pick another
    kernel per m); proj, qkv and fc2 at 32768 patches; merger fc2 at 8192 rows;
    biases off 16-byte alignment; fp16 vision weights.
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

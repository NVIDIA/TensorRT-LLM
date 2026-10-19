# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 per-tensor GEMM via the trtllm cuBLASLt scaled-mm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: (call site, K, N, the checkpoint's input_scale, weight_scale) of every FP8
#: linear of the Nemotron-3 Ultra NVFP4 targets. tep4 is per rank: in_proj's N
#: is the rank's z 4096 | xBC 4608 | dt 64 rows, out_proj and the shared-expert
#: down_proj are K-split. dep4 runs the same linears unsharded. The scales sit
#: at the extremes of the checkpoint's ranges -- per-tensor, so the same for a
#: tep4 slice and the dep4 whole -- and only set the output magnitude, which
#: the band is relative to.
_SITES = (
    ("tep4 mamba in_proj", 8192, 8768, 0.0166015625, 0.0006321498076431453),
    ("tep4 mamba out_proj", 4096, 8192, 1.8214285373687744, 0.0038888114504516125),
    ("tep4 shared up_proj", 8192, 2560, 0.1841517835855484, 0.0049874442629516125),
    ("tep4 shared down_proj", 2560, 8192, 6.035714149475098, 0.0008675711578689516),
    # dep4: the (input_scale, weight_scale) of one checkpoint layer each -- the
    # layer holding that linear's largest input_scale (in_proj layer 97, shared
    # down layer 103) or its smallest (out_proj layer 2, shared up layer 1).
    ("dep4 mamba in_proj", 8192, 35072, 0.3549107015132904, 0.0011771066347137094),
    ("dep4 mamba out_proj", 16384, 8192, 0.00788225419819355, 0.0012381417909637094),
    ("dep4 shared up_proj", 8192, 10240, 0.00177001953125, 0.0025809151120483875),
    ("dep4 shared down_proj", 10240, 8192, 6.035714149475098, 0.00177001953125),
)

#: Token counts per step: one decode sequence, a decode batch off the 8-row
#: tile, the largest captured decode batch, and a full prefill chunk.
_ROWS = (
    (1, "one decode sequence"),
    (7, "a decode batch off the 8-row tile"),
    (32, "the largest captured decode batch"),
    (8192, "a full prefill chunk"),
)


class CublasScaledMm(OpWrapper):
    """`(mat_a @ mat_b) * scale_a * scale_b`, fp32-accumulated, one cuBLASLt call.

    `mat_a` is the e4m3 activation [M, K], dense row-major. `mat_b` is the
    e4m3 weight read column-major [K, N] -- a stored [N, K] weight's `.t()`,
    so no transpose kernel runs. `scale_a` is the activation's static
    `input_scale` and `scale_b` the weight's `weight_scale`, each fp32 with one
    element, passed as the checkpoint stores them ([1] and 0-d are both
    accepted): the epilogue multiplies by their product, so neither is
    reciprocated. Pass `out_dtype=torch.bfloat16`; None means `mat_a`'s fp8,
    which cuBLASLt refuses.

    `static_quantize_e4m3_per_tensor` produces `mat_a` from the bf16
    activation with the same `input_scale`. They are two entries rather than
    one linear so that a fused producer of the fp8 activation (a quantizing
    norm or all-reduce) can feed this entry directly.
    """

    ARCHS = frozenset({Arch.SM_107})

    # Every FP8 linear of both targets (tep4 per rank, dep4 unsharded) at the
    # four token counts a step takes. The shape is the cell axis here, not an argument kind: with
    # no tuned table entry for any of these (M, K, N) the op asks cuBLASLt's
    # heuristic per call, so each shape is its own kernel choice.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{site} M={m} ({phase}): K={k}, N={n}",
            spec=dict(m=m, k=k, n=n, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16),
        )
        for site, k, n, sa, sb in _SITES
        for m, phase in _ROWS
    )

    note = """
    Loud preconditions (thop/cublasScaledMM.cpp): both operands
    float8_e4m3fn, 2-D, K and N multiples of 16 (stride checks), scales CUDA
    and contiguous, M >= 1 (M = 0 raises inside cuBLASLt), an fp8 out_dtype
    (the default) raises.

    Weight layout is the one silent trap a target's loader can walk into: the
    op hands cuBLASLt `lda = ldb = K` without checking either operand's row
    stride against K. A K-split weight (out_proj and the shared-expert
    down_proj at tep4) taken as a column slice of the full checkpoint tensor
    is a view with row stride 2K-4K; its `.t()` passes every check the op
    makes and returns a product of the wrong weights (99.9% of outputs wrong,
    measured). Materialize the slice with `.contiguous()` at load; `is_valid`
    refuses the view.

    Determinism: two calls return bitwise-identical results at every cell.
    CUDA graphs: the scales are device pointers read at replay, and the
    cuBLASLt workspace is a persistent per-(device, stream) buffer.

    Not certified: `bias` (a bf16 [N] bias is fused correctly, measured, but
    no call site has one), fp32 `out_dtype`, and `output_buffer_kind` /
    `group` (the NCCL-window output for a following all-reduce).

    Perf, measured, not gated: 10-13 us at M = 1 and 43-121 us at M = 8192
    for these shapes. The CUDA-core `cuda_scaled_mm` (M <= 8) measured 6.5-8.2
    us at M = 1 with the same output, and slower than this op at M = 7; it is
    a separate op and is not wrapped here.
    """

    def __call__(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> torch.Tensor:
        return torch.ops.trtllm.cublas_scaled_mm(
            mat_a, mat_b, scale_a, scale_b, bias, out_dtype, output_buffer_kind, group
        )

    def reference(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> torch.Tensor:
        """fp32 product of the exact e4m3 values, scaled once, rounded once.

        Every e4m3 x e4m3 product is exact in fp32, so the only rounding before
        the output's is the accumulation's. TF32 off -- see `true_fp32_matmul`.
        `output_buffer_kind` and `group` choose where the output is allocated
        and do not enter the arithmetic.
        """
        with true_fp32_matmul():
            acc = mat_a.float() @ mat_b.float()
        ref = acc * (scale_a.float().reshape(()) * scale_b.float().reshape(()))
        if bias is not None:
            ref = ref + bias.float()
        return ref.to(out_dtype if out_dtype is not None else mat_a.dtype)

    def is_valid(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> None:
        # Each guard is a violation measured to return a plausible result
        # rather than raise.
        assert mat_a.is_contiguous(), (
            "mat_a must be dense row-major [M, K]: the op reads it with row stride K, "
            "so a strided view (e.g. a column slice) is silently multiplied from the "
            "wrong elements"
        )
        assert mat_b.stride(0) == 1 and mat_b.stride(1) == mat_b.shape[0], (
            "mat_b must be dense column-major [K, N] -- the .t() of a contiguous [N, K] "
            "weight. A column slice of a wider weight (a K-split linear) has row stride "
            "> K and is silently read as the wrong weights; .contiguous() it at load"
        )
        for name, s in (("scale_a", scale_a), ("scale_b", scale_b)):
            # cuBLASLt reads one fp32 scalar through each pointer.
            assert s.numel() == 1, (
                f"{name} must hold exactly one value; the op accepts a per-row vector "
                f"and silently applies its first element to every row"
            )
            assert s.dtype == torch.float32, (
                f"{name} must be float32; a {s.dtype} scale is accepted and its bytes "
                f"are read as fp32"
            )


cublas_scaled_mm = CublasScaledMm()

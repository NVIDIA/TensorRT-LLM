# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General matmul with optional fused bias via the trtllm cuBLASLt gemm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul


class CublasMm(OpWrapper):
    """`mat_a @ mat_b (+ bias)`, fp32-accumulated, one cublas_mm call.

    `mat_b` is read column-major, which is what a stored weight's `.t()` already
    is -- the targets pass `weight.t()` and no transpose kernel runs.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # The shipped targets use this for their dense bf16 projections. This
    # entry's test carries their shapes per target (`TARGET_CELLS`), among them
    # the argument shapes that differ in kind rather than in size: fused bias
    # (gpt-oss's qkv, o and router; the qwen4 vision tower), no bias (deepseek's
    # and the qwen4 trunk's projections), an out_dtype that is not the input's
    # (deepseek's router and the qwen4 HC first projection ask for fp32), and a
    # one-column output (the qwen4 MTP shared-expert gate).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode-shaped: one token, the shape every projection sees per step",
            spec=dict(m=1, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
    )

    note = """
    `output_buffer_kind` and `group` are the op's multi-stream and collective
    arguments. Every shipped target leaves them at their defaults, so neither
    is a cell axis and neither is certified here.

    The op is compiled C++ (`cpp/tensorrt_llm/thop/cublasScaledMM.cpp`,
    `TORCH_LIBRARY_FRAGMENT(trtllm)`), so it is in the build or not; nothing is
    JIT-compiled. It passes `lda = ldb = k` and `ldc = n` to cuBLASLt rather
    than the tensors' strides, which is why `is_valid` requires dense operands.
    `find_special_algo` looks the call up by `(max(next_pow2(m), 8), k, n)` in
    `cublasScaledMMLut.h`'s `bf16_algo_list`, which holds only DeepSeek's
    router shapes (k 7168, n 256). Every other call, every qwen4 cell
    included, takes cuBLASLt's own heuristic, which picks the kernel (and any
    split-K) per shape; that is why cells are listed per shape rather than
    inferred from one. Accumulation is `CUBLAS_COMPUTE_32F` throughout.

    A bias goes to cuBLASLt's `CUBLASLT_EPILOGUE_BIAS` epilogue
    (`CublasMMWrapper::setBiasDescriptor`) with no bias data type set, so it is
    read in the output dtype: hence the bias-dtype guard. The epilogue adds the
    bias to the fp32 sum before the one rounding: on the qwen4 vision cells the
    output sits 0.02 to 0.04 bf16 ulp RMS from that reference and 0.73 from a
    product rounded to bf16 before the bias is added.

    A bias pointer off a 16-byte boundary (a contiguous [N] slice at an odd
    element offset) is accepted and changes the kernel cuBLASLt picks.
    Measured on GB300 at the qwen4 vision shapes: at 13 of the 24 (shape, rows)
    pairs probed (every shape at 256 rows; proj, fc2 and merger fc2 at 1024;
    both merger shapes at 4096 and 16384) the output then reads 0.55 to 0.84
    ulp RMS and up to 2.55 element-wise from the reference. That is within the
    test's band, but it is not the cells' precision and it is not certified. A
    target keeps each bias in a tensor of its own, as the tp1 tower's weight
    table does.

    Measured on GB300 over every cell in the test, against `reference`, in
    units of the output's row scale:
    - bf16 outputs: at most 1.98 bf16 ulp element-wise (one rounding flip on an
      element at the bottom of the row maximum's binade) and 0.12 relative RMS
      (the qwen4 MTP cells; 0.048 on the trunk cells, 0.042 on the vision
      cells), cosine >= 0.9999998. The deepseek cells read up to 1.47 bf16 ulp
      element-wise (o_proj): every shape's worst element sits about one
      rounding step off, so the band's margin is 2x, not orders.
    - fp32 outputs: at most 192 fp32 ulp element-wise and 97 RMS (the qwen4 HC
      first projection at 8192 rows); the deepseek router reads 22 and 15.
      The operands are bf16, but the fp32 output is not bf16-rounded: the same
      product rounded through bf16 reads at least 24653 and 13836 fp32 ulp.
    - one-column outputs (the qwen4 MTP shared-expert gate, n = 1, a 2-byte
      `ldc`): a one-element row's scale is the element itself, which
      cancellation over K puts near zero, so the test scores the column. Per
      row it read 4.3 and 8.7 bf16 ulp at 8192 rows on two seeds, on an
      absolute difference of at most one bf16 step; over the column at most
      1.24, cosine 0.9999999.

    The qwen4 unified target's cells (its key in the test's `TARGET_CELLS`)
    were driven on sm_103 only, the only architecture that target runs on.
    They are its trunk projections at the tp4 / tep4 and dep4 per-rank widths
    (decode 1 to 128 rows, an MTP-7 verify at 1024, an 8192-row prefill chunk,
    the dep4 router over 4 x 8192 gathered rows, the HC first projection with
    fp32 out, the PLE key and value projections, lm_head); its MTP layer at
    tep4 (fc_e / fc_h 2560 -> 2560, fc_h on rows x hc_count 4 up to 32768
    rows, the gated q 2560 -> 3072, k / v 2560 -> 256, the index projection
    2560 -> 640 and the shared-expert gate 2560 -> 1); and its vision tower
    with a bf16 bias (qkv 1152 -> 3456, proj 1152 -> 1152, fc1 1152 -> 4304,
    fc2 4304 -> 1152 at 1024 and 16384 patches, the patch merger 4608 -> 4608
    and 4608 -> 2560 at 256 and 4096 merged rows, fc1 at 32768 patches and
    merger fc1 at 8192 rows). The tower runs eager and replicated per rank, so
    its shapes do not change with the member. The MTP gate is sliced as a
    [1, 2560] row out of the fused shared-expert weight; its `.t()` has strides
    (1, 2560), the dense column-major form the cells build, and computes
    bitwise what a standalone copy does.

    Not certified:
    - sm_100 for any qwen4 cell.
    - Row counts between the certified ones (for example unpadded capture
      batches 3 to 31, prefill captures 128 to 4096, a dep2 gathered router up
      to 16384, the MTP drafter at 2 to 127, small images at 64 to 1023
      patches): cuBLASLt may pick another kernel per m.
    - At the MTP layer: the shared expert's gate_up (2560 -> 1280) at any row;
      its down (640 -> 2560), router (2560 -> 512) and o_proj (1536 -> 2560)
      at the 1024-row verify (the trunk cells hold those shapes at 1, 128,
      8192 and 32768 rows); the index projection at rows other than 1; q / k /
      v at 128 rows; a fused q | k | v | index row; dep4's full widths (q
      2560 -> 12288, k / v -> 512).
    - At the vision tower: proj, qkv and fc2 at 32768 patches and merger fc2
      at 8192 rows (probed within 1.98 / 0.042, but not cells); biases off
      16-byte alignment; fp16 weights or an fp32 out_dtype.
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

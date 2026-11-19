# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MoE router logits: BF16 hidden states times the BF16 gate weight, FP32 out."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

_HIDDEN = 7168


class Dsv3RouterGemmOp(OpWrapper):
    """Return `mat_a @ mat_b` as a new [M, N] FP32 tensor.

    `mat_a` is row-major [M, K]; `mat_b` is read column-major [K, N], which is what the stored
    [N, K] gate weight's `.t()` already is.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3.1 router, M={rows}: the tensor-core 896 x 7168 kernel (1 <= M <= 16)",
                spec=dict(rows=rows),
            )
            for rows in (1, 2, 3, 4, 5, 8, 10, 13, 16)
        ),
        *(
            Cell(
                why=f"K3.1 router, M={rows}: past 16 rows the op falls back to cuBLAS",
                spec=dict(rows=rows),
            )
            for rows in (17, 40, 256, 2048, 8192)
        ),
        Cell(
            why="K3.1 router, M=8 over a row-padded input: the op packs it before the kernel",
            spec=dict(rows=8, row_width=_HIDDEN + 64),
        ),
        *(
            Cell(why=f"K3.1 sm_107 Helix CP 8 generation, M={rows}: cuBLAS", spec=dict(rows=rows))
            for rows in (64, 96, 144)
        ),
        Cell(why="K3.1 sm_107 context, a 16384-row chunk: cuBLAS", spec=dict(rows=16384)),
    )
    note = """
    Kimi K3.1 calls it with `mat_a` [M, 7168] bf16 (the rank's tokens) and
    `mat_b` the [896, 7168] bf16 gate weight's `.t()`, only under attention TP:
    attention DP routes through an FP32 linear instead (the bf16-input kernel's
    reduction order flips borderline top-16 picks at DEP batch sizes).

    For 1 <= M <= 16 with bf16 operands, fp32 output, no bias and the K3 shape
    (896 x 7168) the op runs its tensor-core kernel, one instantiation per M;
    every other call takes cuBLAS (`cublas_mm_out`, fp32 accumulation). A
    row-padded or 16-byte-misaligned operand is copied into a packed one first,
    so there is no layout it takes and answers wrongly (no is_valid).
    Both kernels accumulate in fp32 and are deterministic for a fixed M. On
    GB300 the tensor-core kernel lands within 5 fp32 ulp (of the row's largest
    logit) of the exact product, the cuBLAS path within ~110 ulp (67 ulp RMS),
    which is tensor-core fp32 accumulation, not a wrong product. No
    workspace, metadata or state. The output is allocated per call; launches
    capture in a CUDA graph after warm-up.

    ARCHS lists only what was driven: sm_103 (GB300) and sm_107 (VR200).
    """

    def __call__(self, mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.dsv3_router_gemm_op(
            mat_a, mat_b, bias=None, out_dtype=torch.float32
        )

    def reference(self, mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
        """True-fp32 product of the bf16 operands (exact products, fp32 sums)."""
        with true_fp32_matmul():
            return mat_a.float() @ mat_b.float()


dsv3_router_gemm_op = Dsv3RouterGemmOp()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 linear: e4m3 activations (1x128 fp32 scales) x e4m3 weight (128x128 fp32 scales), bf16 out, via
`trtllm::cute_dsl_fp8_gemm_blackwell`."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

BLOCK = 128


def _ceil_blocks(n: int) -> int:
    return (n + BLOCK - 1) // BLOCK


def dequantize_1x128(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """`[M, K]` fp32 from e4m3 codes and K-block-major `[ceil(K/128), M]` scales."""
    k = data.shape[1]
    return data.float() * scales.float().t().repeat_interleave(BLOCK, dim=1)[:, :k]


def dequantize_128x128(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """`[N, K]` fp32 from e4m3 codes and `[ceil(N/128), ceil(K/128)]` scales."""
    n, k = codes.shape
    expanded = scales.float().repeat_interleave(BLOCK, dim=0)[:n]
    return codes.float() * expanded.repeat_interleave(BLOCK, dim=1)[:, :k]


#: `(N, K)` of the stored weights: sparse (32-head) and SWA (64-head) MLA layers, and the draft MLP.
SPARSE_Q_B = dict(n=8192, k=1024)
SPARSE_O_PROJ = dict(n=3072, k=8192)
SPARSE_LINEAR_GATE = dict(n=8192, k=3072)
SWA_Q_B = dict(n=16384, k=1024)
SWA_O_PROJ = dict(n=3072, k=16384)
SWA_LINEAR_GATE = dict(n=16384, k=3072)
DRAFT_MLP_UP = dict(n=9216, k=3072)
DRAFT_MLP_DOWN = dict(n=3072, k=9216)


class BlockFp8Gemm(OpWrapper):
    """`(input * input_scale) @ (weight * weight_scale).T`, fp32-accumulated, new `[M, N]` tensor.

    `input_scale` is K-block-major `[ceil(K/128), M]` as `fp8_quantize_1x128` returns it; `weight_scale` is
    `[ceil(N/128), ceil(K/128)]`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="sparse q_b 8192x1024, 1 rows: one decode row",
            spec=dict(m=1, **SPARSE_Q_B),
        ),
        Cell(
            why="sparse q_b 8192x1024, 6 rows: the verify width: five drafts and the anchor",
            spec=dict(m=6, **SPARSE_Q_B),
        ),
        Cell(
            why="sparse q_b 8192x1024, 64 rows: 64 requests: the widest decode batch the engine captures",
            spec=dict(m=64, **SPARSE_Q_B),
        ),
        Cell(
            why="sparse q_b 8192x1024, 384 rows: width 6 at 64 requests",
            spec=dict(m=384, **SPARSE_Q_B),
        ),
        Cell(
            why="sparse q_b 8192x1024, 512 rows: one chunked-prefill chunk",
            spec=dict(m=512, **SPARSE_Q_B),
        ),
        Cell(
            why="sparse q_b 8192x1024, 5120 rows: the prefill envelope's largest row count",
            spec=dict(m=5120, **SPARSE_Q_B),
        ),
        Cell(
            why="sparse o_proj 3072x8192, 1 rows: one decode row",
            spec=dict(m=1, **SPARSE_O_PROJ),
        ),
        Cell(
            why="sparse o_proj 3072x8192, 6 rows: the verify width: five drafts and the anchor",
            spec=dict(m=6, **SPARSE_O_PROJ),
        ),
        Cell(
            why="sparse o_proj 3072x8192, 64 rows: 64 requests: the widest decode batch the engine captures",
            spec=dict(m=64, **SPARSE_O_PROJ),
        ),
        Cell(
            why="sparse o_proj 3072x8192, 384 rows: width 6 at 64 requests",
            spec=dict(m=384, **SPARSE_O_PROJ),
        ),
        Cell(
            why="sparse o_proj 3072x8192, 512 rows: one chunked-prefill chunk",
            spec=dict(m=512, **SPARSE_O_PROJ),
        ),
        Cell(
            why="sparse o_proj 3072x8192, 5120 rows: the prefill envelope's largest row count",
            spec=dict(m=5120, **SPARSE_O_PROJ),
        ),
        Cell(
            why="sparse linear_gate 8192x3072, 1 rows: one decode row",
            spec=dict(m=1, **SPARSE_LINEAR_GATE),
        ),
        Cell(
            why="sparse linear_gate 8192x3072, 6 rows: the verify width: five drafts and the anchor",
            spec=dict(m=6, **SPARSE_LINEAR_GATE),
        ),
        Cell(
            why="sparse linear_gate 8192x3072, 64 rows: 64 requests: the widest decode batch the engine captures",
            spec=dict(m=64, **SPARSE_LINEAR_GATE),
        ),
        Cell(
            why="sparse linear_gate 8192x3072, 384 rows: width 6 at 64 requests",
            spec=dict(m=384, **SPARSE_LINEAR_GATE),
        ),
        Cell(
            why="sparse linear_gate 8192x3072, 512 rows: one chunked-prefill chunk",
            spec=dict(m=512, **SPARSE_LINEAR_GATE),
        ),
        Cell(
            why="sparse linear_gate 8192x3072, 5120 rows: the prefill envelope's largest row count",
            spec=dict(m=5120, **SPARSE_LINEAR_GATE),
        ),
        Cell(
            why="SWA q_b 16384x1024, 1 rows: one decode row",
            spec=dict(m=1, **SWA_Q_B),
        ),
        Cell(
            why="SWA q_b 16384x1024, 6 rows: the verify width: five drafts and the anchor",
            spec=dict(m=6, **SWA_Q_B),
        ),
        Cell(
            why="SWA q_b 16384x1024, 64 rows: 64 requests: the widest decode batch the engine captures",
            spec=dict(m=64, **SWA_Q_B),
        ),
        Cell(
            why="SWA q_b 16384x1024, 384 rows: width 6 at 64 requests",
            spec=dict(m=384, **SWA_Q_B),
        ),
        Cell(
            why="SWA q_b 16384x1024, 512 rows: one chunked-prefill chunk",
            spec=dict(m=512, **SWA_Q_B),
        ),
        Cell(
            why="SWA q_b 16384x1024, 5120 rows: the prefill envelope's largest row count",
            spec=dict(m=5120, **SWA_Q_B),
        ),
        Cell(
            why="SWA o_proj 3072x16384, 1 rows: one decode row",
            spec=dict(m=1, **SWA_O_PROJ),
        ),
        Cell(
            why="SWA o_proj 3072x16384, 6 rows: the verify width: five drafts and the anchor",
            spec=dict(m=6, **SWA_O_PROJ),
        ),
        Cell(
            why="SWA o_proj 3072x16384, 64 rows: 64 requests: the widest decode batch the engine captures",
            spec=dict(m=64, **SWA_O_PROJ),
        ),
        Cell(
            why="SWA o_proj 3072x16384, 384 rows: width 6 at 64 requests",
            spec=dict(m=384, **SWA_O_PROJ),
        ),
        Cell(
            why="SWA o_proj 3072x16384, 512 rows: one chunked-prefill chunk",
            spec=dict(m=512, **SWA_O_PROJ),
        ),
        Cell(
            why="SWA o_proj 3072x16384, 5120 rows: the prefill envelope's largest row count",
            spec=dict(m=5120, **SWA_O_PROJ),
        ),
        Cell(
            why="SWA linear_gate 16384x3072, 1 rows: one decode row",
            spec=dict(m=1, **SWA_LINEAR_GATE),
        ),
        Cell(
            why="SWA linear_gate 16384x3072, 6 rows: the verify width: five drafts and the anchor",
            spec=dict(m=6, **SWA_LINEAR_GATE),
        ),
        Cell(
            why="SWA linear_gate 16384x3072, 64 rows: 64 requests: the widest decode batch the engine captures",
            spec=dict(m=64, **SWA_LINEAR_GATE),
        ),
        Cell(
            why="SWA linear_gate 16384x3072, 384 rows: width 6 at 64 requests",
            spec=dict(m=384, **SWA_LINEAR_GATE),
        ),
        Cell(
            why="SWA linear_gate 16384x3072, 512 rows: one chunked-prefill chunk",
            spec=dict(m=512, **SWA_LINEAR_GATE),
        ),
        Cell(
            why="SWA linear_gate 16384x3072, 5120 rows: the prefill envelope's largest row count",
            spec=dict(m=5120, **SWA_LINEAR_GATE),
        ),
        Cell(
            why="draft MLP gate/up 9216x3072, 6 rows: width 6: five drafts and the anchor",
            spec=dict(m=6, **DRAFT_MLP_UP),
        ),
        Cell(
            why="draft MLP gate/up 9216x3072, 384 rows: width 6 at 64 requests",
            spec=dict(m=384, **DRAFT_MLP_UP),
        ),
        Cell(
            why="draft MLP down 3072x9216, 6 rows: width 6: five drafts and the anchor",
            spec=dict(m=6, **DRAFT_MLP_DOWN),
        ),
        Cell(
            why="draft MLP down 3072x9216, 384 rows: width 6 at 64 requests",
            spec=dict(m=384, **DRAFT_MLP_DOWN),
        ),
    )

    note = """
    Operands are read through raw pointers as dense, 16-byte-aligned layouts; K comes from `weight`. At
    `M == K/128` a token-major activation scale has the K-block-major shape and no guard can tell them apart.
    The checkpoint's bf16 `weight_scale_inv` must be upcast to fp32 at load.
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
    ) -> torch.Tensor:
        return torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell(
            input, weight, input_scale, weight_scale, output_dtype, use_tvm_ffi
        )

    def reference(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
    ) -> torch.Tensor:
        """Both operands dequantized to fp32, one IEEE fp32 matmul, one rounding; `use_tvm_ffi` is ignored."""
        a = dequantize_1x128(input, input_scale)
        b = dequantize_128x128(weight, weight_scale)
        with true_fp32_matmul():
            out = a @ b.t()
        return out.to(output_dtype)

    def is_valid(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
    ) -> None:
        assert input.dim() == 2 and weight.dim() == 2, (
            f"input and weight must be 2-D, got {tuple(input.shape)} and {tuple(weight.shape)}"
        )
        assert input.dtype == torch.float8_e4m3fn and weight.dtype == torch.float8_e4m3fn, (
            f"input and weight must be float8_e4m3fn, got {input.dtype} and {weight.dtype}"
        )
        assert input_scale.dtype == torch.float32 and weight_scale.dtype == torch.float32, (
            f"both scales must be fp32, got {input_scale.dtype} and {weight_scale.dtype}"
        )
        for name, tensor in (
            ("input", input),
            ("weight", weight),
            ("input_scale", input_scale),
            ("weight_scale", weight_scale),
        ):
            assert tensor.is_contiguous(), f"{name} must be contiguous"
            assert tensor.data_ptr() % 16 == 0, f"{name} must start 16-byte aligned"
        m, k = input.shape
        n = weight.shape[0]
        assert weight.shape[1] == k, f"input has K={k} but weight has K={weight.shape[1]}"
        assert input_scale.shape == (_ceil_blocks(k), m), (
            f"input_scale must be K-block-major {(_ceil_blocks(k), m)}, got {tuple(input_scale.shape)}"
        )
        assert weight_scale.shape == (_ceil_blocks(n), _ceil_blocks(k)), (
            f"weight_scale must be {(_ceil_blocks(n), _ceil_blocks(k))}, got {tuple(weight_scale.shape)}"
        )


block_fp8_gemm = BlockFp8Gemm()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 linear: e4m3 activations with 1x128 FP32 scales times an e4m3 weight with 128x128
FP32 block scales, BF16 out, via the CuTe DSL Blackwell blockwise GEMM
(`trtllm::cute_dsl_fp8_gemm_blackwell`).

Adapted from glm5_next's target entry
(`models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/gemm/fp8_blockwise_gemm.py`), which wraps
the same op: its dequantizing reference and its guards are unchanged. The shapes and cells are
re-derived from this checkpoint's index -- the sparse MLA layers' 32-head q_b/o_proj/
linear_gate, the SWA layers' 64-head ones, and the five DFly2 draft layers' q_b/o/gate and
9216-wide MLP.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: Block edge on every scale axis: 1x128 activations, 128x128 weights.
BLOCK = 128


def _ceil_blocks(n: int) -> int:
    return (n + BLOCK - 1) // BLOCK


def dequantize_1x128(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """`[M, K]` fp32 activations from e4m3 codes and K-block-major `[ceil(K/128), M]` scales."""
    k = data.shape[1]
    return data.float() * scales.float().t().repeat_interleave(BLOCK, dim=1)[:, :k]


def dequantize_128x128(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """`[N, K]` fp32 weight from e4m3 codes and `[ceil(N/128), ceil(K/128)]` scales."""
    n, k = codes.shape
    expanded = scales.float().repeat_interleave(BLOCK, dim=0)[:n]
    return codes.float() * expanded.repeat_interleave(BLOCK, dim=1)[:, :k]


#: The 32-head sparse MLA layers (11 of the 41 MoE layers): `(N, K)` of the stored weight.
SPARSE_Q_B = dict(n=8192, k=1024)
SPARSE_O_PROJ = dict(n=3072, k=8192)
SPARSE_LINEAR_GATE = dict(n=8192, k=3072)
#: The 64-head SWA layers (the other 30): twice the head width, so twice N at the same K.
SWA_Q_B = dict(n=16384, k=1024)
SWA_O_PROJ = dict(n=3072, k=16384)
SWA_LINEAR_GATE = dict(n=16384, k=3072)
#: The five DFly2 draft layers: the sparse attention shapes above, plus an MLP at hidden 9216.
DRAFT_MLP_UP = dict(n=9216, k=3072)
DRAFT_MLP_DOWN = dict(n=3072, k=9216)


class BlockFp8Gemm(OpWrapper):
    """`(input * input_scale) @ (weight * weight_scale).T`, fp32-accumulated, bf16 out.

    `input` is `[M, K]` e4m3 and `input_scale` its `[ceil(K/128), M]` fp32 scales -- K-block-
    major, exactly what `fp8_quantize_1x128` returns on the sm_100 family. `weight` is the
    stored `[N, K]` e4m3 weight and `weight_scale` its `[ceil(N/128), ceil(K/128)]` fp32 block
    scales. Returns a new `[M, N]` bf16 tensor.

    The kernel multiplies each 128-wide K block's fp32 MMA partial by its activation and weight
    scales and accumulates the scaled partials in fp32: the block-FP8 `Linear` of this
    checkpoint's 41 MoE layers, quantized with 128x128 weight blocks.

    Fusion boundary: the GEMM and its dequantization only. No bias, no activation, no
    quantization of the input -- the caller quantizes the bf16 rows with `fp8_quantize_1x128`
    and may reuse one quantization for several GEMMs that read the same rows.
    """

    ARCHS = frozenset({Arch.SM_103})

    #: Row counts every attention projection runs at, from the vocabulary's call sites: one decode
    #: row, the 6-row verify width (five drafts and the anchor), 64 requests (the widest decode
    #: batch the engine captures), width 6 at 64 requests, one 512-row chunked-prefill chunk, and
    #: the prefill envelope's largest. The drafter is decode-only: its MLP sees only 6 and 384 rows.
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
    Metadata consumed: none.

    Source: the CuTe DSL kernel `Sm100BlockwiseGemmKernel`
    (`_torch/kernels/blackwell/blockwise_gemm/blockwise_gemm.py`) behind the
    `trtllm::cute_dsl_fp8_gemm_blackwell` custom op (`_torch/custom_ops/cute_dsl_custom_ops.py`),
    JIT-compiled per tactic on first use, cached in the CuTe DSL file cache, and autotuned by
    `AutoTuner`; outside an `autotune()` context a cold cache runs the default tactic (1-CTA,
    128x128 MMA tile, 1x1 cluster). The op raises off the sm_100 family and for any
    `output_dtype` but bf16. `use_tvm_ffi` changes the launch path, not the result.

    The kernel reads all four operands through raw data pointers with fixed dense layouts --
    `input` `[M, K]` and `weight` `[N, K]` K-contiguous, `input_scale` `[ceil(K/128), M]` with
    M contiguous, `weight_scale` `[ceil(N/128), ceil(K/128)]` row-major -- each assumed
    16-byte aligned, and takes K from `weight`. Strides, dtypes and the operands' shapes are
    not checked against each other; `is_valid` states what has to hold.

    The checkpoint stores `weight_scale_inv` as BF16 (`[N/128, K/128]`: `[64, 8]` for a sparse
    layer's q_b, `[128, 8]` for an SWA layer's). Upcast it to fp32 once when the weight is
    loaded -- exact, and the layout is already the one the kernel reads; the kernel reads
    fp32 scale bytes and a BF16 tensor handed to it is misread, which `is_valid` refuses. The
    activation scales are fp32 from `fp8_quantize_1x128` and pass through as returned.

    At `M == K/128` (8 rows at K 1024, 24 at 3072, 64 at 8192, 72 at 9216, 128 at 16384) a
    token-major `[M, K/128]` activation scale has the shape of the K-block-major one, so no
    check can tell them apart. The 64-row o_proj cell (K 8192) is square; it passes only
    because the scales are the ones `fp8_quantize_1x128` returns. A row slice `x_fp8[a:b]`
    needs `x_scale[:, a:b].contiguous()`, not the strided view.

    Certified: every block-FP8 projection of this checkpoint -- the sparse layers'
    8192x1024/3072x8192/8192x3072 q_b, o_proj and linear_gate, the SWA layers'
    16384x1024/3072x16384/16384x3072 ones, and the draft layers' 9216x3072/3072x9216 MLP --
    at 1, 6, 64, 384, 512 and 5120 rows (the draft MLP at 6 and 384). Measured on sm_103:
    every cell inside the 4-ulp element / 2-ulp relative-RMS band, the worst relative RMS
    1.04e-4 (0.013 ulp) at the SWA o_proj 5120-row cell; at least 99.85% of elements
    bit-identical to the reference, and the 1-row sparse cells bit-identical. A transposed
    activation scale measures 4.0e-1 relative RMS and a weight scale one K block off 4.0e-1,
    51 times the RMS band.

    Not certified: N or K not a multiple of 128 (the kernel needs K % 16 == 0 and N % 8 == 0
    for its TMA alignment), rows between the cells, `M == 0`, a BF16 `weight_scale_inv`
    passed without the fp32 upcast (guarded), `output_dtype` other than bf16 (the op raises),
    non-contiguous or misaligned operands (guarded), the autotuner's non-default tactics (the
    test runs the cold-cache default tactic; a served engine tunes in warmup and the model's
    accuracy gates cover that), `use_tvm_ffi=False` beyond one shape (bit-identical to the
    default there), and every arch but sm_103.
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
        """Both operands dequantized to fp32, one IEEE fp32 matmul, one rounding to
        `output_dtype`. `use_tvm_ffi` is ignored."""
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
        """What the kernel assumes and never checks, read from its launch code.

        Every operand reaches the kernel as a bare pointer: a wrong dtype, a strided view, a
        misaligned slice or a scale of the wrong shape is read as if it were the layout above,
        with no error. The `M == K/128` transposed-scale case is the one this cannot see.
        """
        assert input.dim() == 2 and weight.dim() == 2, (
            f"input and weight must be 2-D [M, K] and [N, K], got {tuple(input.shape)} and "
            f"{tuple(weight.shape)}; the kernel reads the first two dims as rows and K"
        )
        assert input.dtype == torch.float8_e4m3fn and weight.dtype == torch.float8_e4m3fn, (
            f"input and weight must be float8_e4m3fn, got {input.dtype} and {weight.dtype}; "
            "other dtypes are read as e4m3 bytes"
        )
        assert input_scale.dtype == torch.float32 and weight_scale.dtype == torch.float32, (
            f"both scales must be fp32, got {input_scale.dtype} and {weight_scale.dtype}; the "
            "checkpoint's weight_scale_inv is bf16 and must be upcast to fp32 at load, and "
            "bf16 bytes read as fp32 are garbage"
        )
        for name, tensor in (
            ("input", input),
            ("weight", weight),
            ("input_scale", input_scale),
            ("weight_scale", weight_scale),
        ):
            assert tensor.is_contiguous(), (
                f"{name} must be contiguous; a strided view is read as if dense"
            )
            assert tensor.data_ptr() % 16 == 0, (
                f"{name} must start 16-byte aligned; the kernel assumes it"
            )
        m, k = input.shape
        n = weight.shape[0]
        assert weight.shape[1] == k, (
            f"input has K={k} but weight has K={weight.shape[1]}; the kernel takes K from "
            "weight and reads input with the wrong row length"
        )
        assert input_scale.shape == (_ceil_blocks(k), m), (
            f"input_scale must be K-block-major [ceil(K/128), M] = {(_ceil_blocks(k), m)}, got "
            f"{tuple(input_scale.shape)}; pass the scales fp8_quantize_1x128 returns"
        )
        assert weight_scale.shape == (_ceil_blocks(n), _ceil_blocks(k)), (
            f"weight_scale must be [ceil(N/128), ceil(K/128)] = "
            f"{(_ceil_blocks(n), _ceil_blocks(k))}, got {tuple(weight_scale.shape)}"
        )


block_fp8_gemm = BlockFp8Gemm()

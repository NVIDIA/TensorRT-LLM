# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 linear: e4m3 activations with 1x128 FP32 scales times an e4m3 weight with 128x128
FP32 scales, BF16 out, via the CuTe DSL Blackwell blockwise GEMM
(`cute_dsl_fp8_gemm_blackwell`)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

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


#: GLM-5.3-Flash's block-FP8 FFN projections at TP4, as `(N, K)` of the stored weight:
#: hidden 4096, dense intermediate 12288 / 4, shared-expert intermediate 2048 / 4.
GLM_DENSE_GATE_UP = dict(n=2 * 3072, k=4096)
GLM_DENSE_DOWN = dict(n=4096, k=3072)
GLM_SHARED_GATE_UP = dict(n=2 * 512, k=4096)
GLM_SHARED_DOWN = dict(n=4096, k=512)


class Fp8BlockwiseGemm(OpWrapper):
    """`(input * input_scale) @ (weight * weight_scale).T`, fp32-accumulated, bf16 out.

    `input` is `[M, K]` e4m3 and `input_scale` its `[ceil(K/128), M]` fp32 scales --
    K-block-major, exactly what `fp8_quantize_1x128` returns on the sm_100 family.
    `weight` is the stored `[N, K]` e4m3 weight and `weight_scale` its `[ceil(N/128),
    ceil(K/128)]` fp32 block scales -- a block-FP8 checkpoint's `weight_scale_inv` as stored,
    not resmoothed to UE8M0 powers of two. Returns a new `[M, N]` bf16 tensor.

    The kernel multiplies each 128-wide K block's fp32 MMA partial by its activation and
    weight scales and accumulates the scaled partials in fp32. That is the block-FP8 `Linear`
    of a checkpoint quantized with 128x128 weight blocks on SM100/SM103 when the DeepGEMM
    path is off (`fp8_quantize_1x128` then this op).

    Fusion boundary: the GEMM and its dequantization only. No bias, no activation, no
    quantization of the input -- the caller quantizes the bf16 rows with
    `fp8_quantize_1x128` and may reuse one quantization for several GEMMs that read the same
    rows.
    """

    ARCHS = frozenset({Arch.SM_100})

    # Only GLM-5.3-Flash calls this entry. Its own cells are the M-tile edges at GLM's widest
    # FFN projection; the projections and row counts the target runs are in the test.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one row: the decode row count, a single partial M tile",
            spec=dict(m=1, **GLM_DENSE_GATE_UP),
        ),
        Cell(
            why="200 rows: one full and one partial 128-row M tile",
            spec=dict(m=200, **GLM_DENSE_GATE_UP),
        ),
    )

    note = """
    Source: the CuTe DSL kernel `Sm100BlockwiseGemmKernel`
    (`_torch/kernels/blackwell/blockwise_gemm/blockwise_gemm.py`) behind the
    `trtllm::cute_dsl_fp8_gemm_blackwell` custom op (`_torch/custom_ops/cute_dsl_custom_ops.py`),
    JIT-compiled per tactic on first use and autotuned by `AutoTuner`; outside an `autotune()`
    context a cold cache runs the default tactic (1-CTA, 128x128 MMA tile, 1x1 cluster). The
    op raises off the sm_100 family and for any `output_dtype` but bf16. `use_tvm_ffi`
    changes the launch path, not the result.

    The kernel reads all four operands through raw data pointers with fixed dense layouts --
    `input` `[M, K]` and `weight` `[N, K]` K-contiguous, `input_scale` `[ceil(K/128), M]` with
    M contiguous, `weight_scale` `[ceil(N/128), ceil(K/128)]` row-major -- each assumed
    16-byte aligned, and takes K from `weight`. Strides, dtypes and the operands' shapes are
    not checked against each other; `is_valid` states what has to hold.

    At `M == K/128` (32 rows at K 4096, 24 at 3072, 12 at 1536, 4 at 512) a token-major
    `[M, K/128]` activation scale has the shape of the K-block-major one, so no check can tell
    them apart: pass the scales `fp8_quantize_1x128` returns, as returned. A row slice
    `x_fp8[a:b]` needs `x_scale[:, a:b].contiguous()`, not the strided view.

    Certified: GLM-5.3-Flash's block-FP8 projections at TP4 -- the dense MLP (6144 x 4096 and
    4096 x 3072), the shared expert (1024 x 4096 and 4096 x 512) and the sequence-parallel
    MLA prefill's q_b and o (16384 x 1536, 4096 x 16384) -- from 1 to 13,312 rows, all with N
    and K multiples of 128.

    Not certified: N or K not a multiple of 128 (the kernel needs K % 16 == 0 and N % 8 == 0
    for its TMA alignment), a warm autotuner cache (a tuned tactic other than the default),
    CUDA-graph capture, sm_103.
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
        misaligned slice or a scale of the wrong shape is read as if it were the layout
        above, with no error. The `M == K/128` transposed-scale case is the one this cannot
        see.
        """
        assert input.dtype == torch.float8_e4m3fn and weight.dtype == torch.float8_e4m3fn, (
            f"input and weight must be float8_e4m3fn, got {input.dtype} and {weight.dtype}; "
            "other dtypes are read as e4m3 bytes"
        )
        assert input_scale.dtype == torch.float32 and weight_scale.dtype == torch.float32, (
            f"both scales must be fp32 (the checkpoint's weight_scale_inv as stored), got "
            f"{input_scale.dtype} and {weight_scale.dtype}; packed UE8M0 scales are misread"
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


fp8_blockwise_gemm = Fp8BlockwiseGemm()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 query projection with BF16 RoPE and speculative unit-scale FP8 output."""

from functools import cache

import torch
import triton
import triton.language as tl


@cache
def _kernel_type() -> type:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    class QueryGemm(Sm100BlockScaledPersistentDenseGemmKernel):
        @cute.jit
        def wrapper_query(
            self,
            m: cutlass.Int64,
            n: cutlass.Int64,
            k: cutlass.Int64,
            sf_m: cutlass.Int64,
            sf_n: cutlass.Int64,
            sf_k: cutlass.Int64,
            cos_sin_rows: cutlass.Int64,
            a: cute.Pointer,
            b: cute.Pointer,
            sa: cute.Pointer,
            sb: cute.Pointer,
            q: cute.Pointer,
            q_fp8: cute.Pointer,
            positions: cute.Pointer,
            cos_sin: cute.Pointer,
            alpha: cute.Tensor,
            max_active_clusters: cutlass.Constexpr,
            stream: cuda.CUstream,
            quantize: cutlass.Constexpr,
        ):
            a_tensor = cute.make_tensor(a, cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)))
            b_tensor = cute.make_tensor(b, cute.make_ordered_layout((n, k, 1), order=(1, 0, 2)))
            sa_tensor = cute.make_tensor(
                sa, cute.make_ordered_layout((32, 4, sf_m, 4, sf_k, 1), order=(2, 1, 4, 0, 3, 5))
            )
            sb_tensor = cute.make_tensor(
                sb, cute.make_ordered_layout((32, 4, sf_n, 4, sf_k, 1), order=(2, 1, 4, 0, 3, 5))
            )
            q_tensor = cute.make_tensor(q, cute.make_ordered_layout((m, n, 1), order=(1, 0, 2)))
            quant_tensor = None
            if cutlass.const_expr(quantize):
                quant_tensor = cute.make_tensor(
                    q_fp8, cute.make_ordered_layout((m, n, 1), order=(1, 0, 2))
                )
            position_tensor = cute.make_tensor(positions, cute.make_layout((m,)))
            rope_tensor = cute.make_tensor(
                cos_sin, cute.make_ordered_layout((cos_sin_rows, 64), order=(1, 0))
            )
            self(
                a_tensor,
                b_tensor,
                sa_tensor,
                sb_tensor,
                q_tensor,
                alpha,
                max_active_clusters,
                stream,
                position_ids_tensor=position_tensor,
                cos_sin_cache_tensor=rope_tensor,
                q_fp8_tensor=quant_tensor,
            )

        @cute.jit
        def wrapper_query_fp8(
            self,
            m: cutlass.Int64,
            sf_m: cutlass.Int64,
            cos_sin_rows: cutlass.Int64,
            n: cutlass.Constexpr,
            k: cutlass.Constexpr,
            a: cute.Pointer,
            b: cute.Pointer,
            sa: cute.Pointer,
            sb: cute.Pointer,
            shape: cute.Pointer,
            q_fp8: cute.Pointer,
            positions: cute.Pointer,
            cos_sin: cute.Pointer,
            amax: cute.Pointer,
            alpha: cute.Tensor,
            max_active_clusters: cutlass.Constexpr,
            stream: cuda.CUstream,
        ):
            # ``shape`` only sizes the never-stored BF16 output (and the tile grid); ``amax``
            # is read only by a kernel built with ``q_fp8_amax``.
            self(
                cute.make_tensor(a, cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))),
                cute.make_tensor(b, cute.make_ordered_layout((n, k, 1), order=(1, 0, 2))),
                cute.make_tensor(
                    sa,
                    cute.make_ordered_layout(
                        (32, 4, sf_m, 4, k // 128, 1), order=(2, 1, 4, 0, 3, 5)
                    ),
                ),
                cute.make_tensor(
                    sb,
                    cute.make_ordered_layout(
                        (32, 4, n // 128, 4, k // 128, 1), order=(2, 1, 4, 0, 3, 5)
                    ),
                ),
                cute.make_tensor(shape, cute.make_ordered_layout((m, n, 1), order=(1, 0, 2))),
                alpha,
                max_active_clusters,
                stream,
                position_ids_tensor=cute.make_tensor(positions, cute.make_layout((m,))),
                cos_sin_cache_tensor=cute.make_tensor(
                    cos_sin, cute.make_ordered_layout((cos_sin_rows, 64), order=(1, 0))
                ),
                q_fp8_tensor=cute.make_tensor(
                    q_fp8, cute.make_ordered_layout((m, n, 1), order=(1, 0, 2))
                ),
                q_amax_tensor=(
                    cute.make_tensor(amax, cute.make_layout((1,))) if self.q_fp8_amax else None
                ),
            )

    return QueryGemm


_COMPILED = {}


def _check_query_inputs(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
) -> None:
    if (
        activation.ndim != 2
        or activation.shape[0] == 0
        or activation.shape[1] == 0
        or activation.shape[1] % 128
        or weight.ndim != 2
        or weight.shape[0] == 0
        or weight.shape[0] % 512
        or weight.shape[1] != activation.shape[1]
        or activation.dtype != torch.float8_e4m3fn
        or weight.dtype != activation.dtype
        or positions.shape != (activation.shape[0],)
        or positions.dtype not in (torch.int32, torch.int64)
        or cos_sin.dtype != torch.float32
        or cos_sin.numel() % 64
        or alpha.dtype != torch.float32
        or alpha.numel() != 1
        or activation_scale.dtype != torch.uint8
        or weight_scale.dtype != torch.uint8
        or any(
            not value.is_cuda or value.device != activation.device or not value.is_contiguous()
            for value in (
                activation,
                activation_scale,
                weight,
                weight_scale,
                alpha,
                positions,
                cos_sin,
            )
        )
    ):
        raise ValueError(
            "CSA2 query fusion requires contiguous CUDA MXFP8 [M, K] / [heads * 512, K] "
            "inputs with 128-aligned K and GPT-J RoPE"
        )
    m, k = activation.shape
    if activation_scale.numel() != triton.cdiv(m, 128) * 128 * (
        k // 32
    ) or weight_scale.numel() != weight.shape[0] * (k // 32):
        raise ValueError("CSA2 query fusion requires CUTLASS-swizzled 128x4 MXFP8 scales")


def query_gemm_rope(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    *,
    quantize: bool,
    quant_q: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    tile: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Project 512-channel MXFP8 query heads and apply their 64-channel GPT-J RoPE tails.

    ``alpha`` is the native linear layer's immutable unit MXFP8 alpha. Optional
    FP8 output uses scale one; ``fixup_fp8_query`` resolves the live context scale.
    ``out`` (contiguous BF16 [M, N]) receives Q when given.
    """
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import make_ptr

    _check_query_inputs(
        activation, activation_scale, weight, weight_scale, alpha, positions, cos_sin
    )
    m, k = activation.shape
    n = weight.shape[0]
    if quant_q is not None and (
        not quantize
        or quant_q.shape != (m, n)
        or quant_q.dtype != torch.float8_e4m3fn
        or quant_q.device != activation.device
        or not quant_q.is_contiguous()
    ):
        raise ValueError("CSA2 query fusion requires a contiguous FP8 output matching [M, N]")
    if out is not None and (
        out.shape != (m, n)
        or out.dtype != torch.bfloat16
        or out.device != activation.device
        or not out.is_contiguous()
    ):
        raise ValueError("CSA2 query fusion requires a contiguous BF16 output matching [M, N]")
    q = torch.empty((m, n), device=activation.device, dtype=torch.bfloat16) if out is None else out
    q_fp8 = quant_q
    if quantize and q_fp8 is None:
        q_fp8 = torch.empty((m, n), device=activation.device, dtype=torch.float8_e4m3fn)
    tile = tile or ((128, 128) if m <= 128 else (256, 256))
    cluster = (tile[0] // 128, 1)
    key = (activation.device.index, tile, positions.dtype, alpha.ndim, quantize)
    shape_args = (
        m,
        n,
        k,
        triton.cdiv(m, 128),
        triton.cdiv(n, 128),
        triton.cdiv(k // 32, 4),
        cos_sin.numel() // 64,
    )
    tensors = (
        activation,
        weight,
        activation_scale,
        weight_scale,
        q,
        q_fp8 if quantize else q,
        positions,
        cos_sin,
    )
    if key not in _COMPILED:
        dtypes = (
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            cutlass.Float8E8M0FNU,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Int32 if positions.dtype == torch.int32 else cutlass.Int64,
            cutlass.Float32,
        )
        pointers = [
            make_ptr(
                dtype,
                tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16 if i != 6 else positions.element_size(),
            )
            for i, (tensor, dtype) in enumerate(zip(tensors, dtypes))
        ]
        kernel = _kernel_type()(32, tile, cluster, q_rope_fusion=True)
        active = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0])
        _COMPILED[key] = cute.compile(
            kernel.wrapper_query,
            *shape_args,
            *pointers,
            cute.runtime.from_dlpack(alpha),
            active,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            quantize,
            options="--opt-level 2 --enable-tvm-ffi",
        )
    _COMPILED[key](*shape_args, *(tensor.data_ptr() for tensor in tensors), alpha)
    return q, q_fp8


@triton.jit(do_not_specialize=["N"])
def _fixup_fp8_query(Q, QUANT_Q, SCALE, N, BLOCK: tl.constexpr):
    scale = tl.load(SCALE)
    if scale != 1.0:
        for block in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
            offset = block * BLOCK + tl.arange(0, BLOCK)
            values = tl.load(Q + offset, offset < N, 0).to(tl.float32)
            quantized = (values * scale).to(tl.float8e4nv)
            tl.store(QUANT_Q + offset, quantized, offset < N)


def fixup_fp8_query(q: torch.Tensor, quant_q: torch.Tensor, quant_scale: torch.Tensor) -> None:
    """Requantize from BF16 only when the live scale differs from the epilogue's unit scale."""
    _fixup_fp8_query[(min(triton.cdiv(q.numel(), 1024), 1024),)](
        q, quant_q, quant_scale, q.numel(), 1024
    )


def fp8_query_gemm(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    quant_q: torch.Tensor,
    amax: torch.Tensor | None = None,
    *,
    tile: tuple[int, int] | None = None,
    cluster: tuple[int, int] | None = None,
) -> bool:
    """``query_gemm_rope`` emitting only its unit-scale E4M3 Q, and optionally Q's amax.

    Writes ``static_quantize_e4m3_per_tensor(q, 1)`` into ``quant_q`` without storing
    ``q``. With ``amax`` (int32), also raises ``amax[0]`` (zero on entry) to the FP32 bits
    of ``amax(|q|)``. Returns False, writing nothing, while a graph capture would have to
    compile it.
    """
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import make_ptr

    _check_query_inputs(
        activation, activation_scale, weight, weight_scale, alpha, positions, cos_sin
    )
    m, k = activation.shape
    n = weight.shape[0]
    if (
        quant_q.shape != (m, n)
        or quant_q.dtype != torch.float8_e4m3fn
        or quant_q.device != activation.device
        or not quant_q.is_contiguous()
        or (
            amax is not None
            and (amax.dtype != torch.int32 or amax.device != activation.device or amax.numel() < 1)
        )
    ):
        raise ValueError("CSA2 FP8 query needs a contiguous E4M3 [M, N] output and an int32 word")
    # Measured on GB300: with only the E4M3 output, 2-CTA 256x192 tiles beat 256x256 ones
    # from 512 rows on and tie at 256.
    tile = tile or ((128, 128) if m <= 128 else (256, 192))
    cluster = cluster or (tile[0] // 128, 1)
    key = ("fp8", amax is not None, activation.device.index, tile, cluster, positions.dtype, n, k)
    if key not in _COMPILED and torch.cuda.is_current_stream_capturing():
        return False
    shape_args = (m, triton.cdiv(m, 128), cos_sin.numel() // 64)
    tensors = (
        activation,
        weight,
        activation_scale,
        weight_scale,
        quant_q,
        quant_q,
        positions,
        cos_sin,
        positions if amax is None else amax,
    )
    if key not in _COMPILED:
        dtypes = (
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            cutlass.Float8E8M0FNU,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Int32 if positions.dtype == torch.int32 else cutlass.Int64,
            cutlass.Float32,
            cutlass.Int32,
        )
        pointers = [
            make_ptr(
                dtype,
                tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=positions.element_size() if i == 6 else 4 if i == 8 else 16,
            )
            for i, (tensor, dtype) in enumerate(zip(tensors, dtypes))
        ]
        kernel = _kernel_type()(
            32,
            tile,
            cluster,
            q_rope_fusion=True,
            q_fp8_amax=amax is not None,
            q_fp8_only=True,
        )
        active = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
        _COMPILED[key] = cute.compile(
            kernel.wrapper_query_fp8,
            *shape_args,
            n,
            k,
            *pointers,
            cute.runtime.from_dlpack(alpha),
            active,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    _COMPILED[key](*shape_args, *(tensor.data_ptr() for tensor in tensors), alpha)
    return True


@triton.jit
def _rescale_fp8_query(QUANT_Q, SCALE, N, BLOCK: tl.constexpr):
    factor = tl.math.div_rn(1.0, tl.load(SCALE))
    if factor != 1.0:
        for block in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
            offset = block * BLOCK + tl.arange(0, BLOCK)
            values = tl.load(QUANT_Q + offset, offset < N).to(tl.float32)
            tl.store(QUANT_Q + offset, (values * factor).to(tl.float8e4nv), offset < N)


def rescale_fp8_query(quant_q: torch.Tensor, scale: torch.Tensor) -> None:
    """Move unit-scale E4M3 Q to the live scale in place; nothing happens at a unit scale.

    The staging scale is a power of two whose exponent Q's amax caps, so the rescaled
    codes stay in range and every normal E4M3 code rescales exactly; only codes that
    were subnormal at the unit scale keep their coarser resolution.
    """
    _rescale_fp8_query[(min(triton.cdiv(quant_q.numel(), 4096), 1024),)](
        quant_q, scale, quant_q.numel(), 4096
    )

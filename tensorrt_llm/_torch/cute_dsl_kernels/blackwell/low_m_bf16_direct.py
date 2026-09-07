# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer (https://github.com/flashinfer-ai/flashinfer) under the Apache-2.0
# license.  Original file: flashinfer/gemm/kernels/dense_bf16_gemm_direct.py
# Original copyright: Copyright contributors to the vLLM project
#                     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# TODO(TRTLLM-15304): This file is a vendored copy of the FlashInfer direct (SIMT) kernel.
# Once FlashInfer releases v0.6.18 (which packages these kernels as importable Python
# modules), remove this file and import directly from flashinfer instead.
# Tracking: https://github.com/flashinfer-ai/flashinfer/pull/4266
"""Register-prefetch BF16 GEMM for low-M, long-K decode shapes on SM10x.

The kernel keeps a complete output dot product inside one CTA and reuses each
prefetched B-row (N row of the weight matrix) across several public-M rows.
It is intentionally a separate autotuner runner from the Blackwell tensor-core
split-K kernel: the two algorithms have different useful shape regions and
tactic spaces.

Unlike the split-K kernel this path uses SIMT (scalar) FP32 accumulators and
a register-file prefetch of the entire K dimension, which avoids TMA and TMEM
overhead and excels for **M=1..4, K=8192, N≤4608** decode shapes where the
tensor-core path has too little N-dim parallelism to fill the SMs.

Public API::

    run_direct_dense(a, b, out, pdl, tactic)  # a=[M,K], b=[K,N] col-major

    prefer_direct_bf16_gemm_sm100(m, n, k)  # heuristic crossover predicate
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from types import MappingProxyType

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math
import torch as _torch
from cutlass import const_expr
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack

#: Vectorised load width (BF16 elements per G2R copy, == 16 bytes / 2 = 8).
_VECTOR_WIDTH = 8

#: CTA thread counts explored during autotuning.
_SUPPORTED_BLOCK_SIZES = (32, 64, 96, 128, 192, 256, 384)

#: Output columns processed together by one CTA tile.
_SUPPORTED_OUTPUTS_PER_BLOCK = (1, 2, 4)

#: Largest public M this low-M direct policy serves.
MAX_M = 32

#: ptxas register cap (keeps occupancy predictable across block sizes).
_COMPILE_OPTIONS = "--ptxas-options -maxrregcount=64"

#: Epilogue modes; "none" preserves the generic GEMM store path.
_EPILOGUE_MODES = ("none", "silu_prefix")


def _sigmoid_f32(value):
    return 1.0 / (cute_math.exp(value * -1.0) + 1.0)


@dataclasses.dataclass(frozen=True)
class DirectTactic:
    """One direct-kernel specialisation.

    Attributes:
        block_size: Threads per CTA.  Must be in ``_SUPPORTED_BLOCK_SIZES``.
        outputs_per_block: N-columns handled by one CTA block.
        rows_per_block: A-rows (public M) handled by one CTA block.
    """

    block_size: int
    outputs_per_block: int
    rows_per_block: int


def _default_rows_per_block(m: int) -> int:
    if m <= 8:
        return m
    return next(rows for rows in (8, 4, 2, 1) if m % rows == 0)


def validate_tactic(tactic: DirectTactic, m: int, n: int, k: int) -> None:
    """Raise ``ValueError`` when *tactic* cannot serve ``(m, n, k)``."""
    if tactic.block_size not in _SUPPORTED_BLOCK_SIZES:
        raise ValueError(f"unsupported block_size={tactic.block_size}")
    if tactic.outputs_per_block not in _SUPPORTED_OUTPUTS_PER_BLOCK:
        raise ValueError(f"unsupported outputs_per_block={tactic.outputs_per_block}")
    if not 1 <= m <= MAX_M:
        raise ValueError(f"direct GEMM requires 1 <= M <= {MAX_M}, got {m}")
    if not 1 <= tactic.rows_per_block <= m or m % tactic.rows_per_block:
        raise ValueError(f"rows_per_block={tactic.rows_per_block} must divide M={m}")
    if n <= 0 or n % tactic.outputs_per_block:
        raise ValueError(f"N={n} must be divisible by outputs_per_block={tactic.outputs_per_block}")
    k_tile = tactic.block_size * _VECTOR_WIDTH
    if k <= 0 or k % k_tile:
        raise ValueError(f"K={k} must be divisible by block_size×{_VECTOR_WIDTH}={k_tile}")


def default_tactic(m: int, n: int, k: int) -> DirectTactic:
    """Return the occupancy-oriented default tactic for ``(m, n, k)``."""
    block_size = next(
        (block for block in (256, 192, 128, 96, 64, 32) if k % (block * _VECTOR_WIDTH) == 0),
        None,
    )
    if block_size is None:
        raise ValueError("direct GEMM: no supported block size divides K evenly")
    outputs_per_block = next(outputs for outputs in (2, 1) if n % outputs == 0)
    tactic = DirectTactic(
        block_size,
        outputs_per_block,
        _default_rows_per_block(m),
    )
    validate_tactic(tactic, m, n, k)
    return tactic


def autotune_tactics(m: int, n: int, k: int) -> list[DirectTactic]:
    """Enumerate valid tactics for ``(m, n, k)`` across the block-size sweep.

    Row tiling is fixed at the default to keep JIT-compilation cost bounded.
    """
    try:
        default = default_tactic(m, n, k)
    except ValueError:
        return []
    tactics = [default]
    for block_size in _SUPPORTED_BLOCK_SIZES:
        for outputs_per_block in _SUPPORTED_OUTPUTS_PER_BLOCK:
            tactic = DirectTactic(block_size, outputs_per_block, default.rows_per_block)
            try:
                validate_tactic(tactic, m, n, k)
            except ValueError:
                continue
            tactics.append(tactic)
    # Preserve insertion order while deduplicating.
    return list(dict.fromkeys(tactics))


def prefer_direct_bf16_gemm_sm100(m: int, n: int, k: int) -> bool:
    """Conservative Blackwell crossover for direct GEMM versus the normal path.

    The single-row bands are a compact fit to cold-L2 CUDA-graph measurements
    against both cuBLAS and Split-K. Narrower batches retain the original
    K=8192 bands. Shapes outside these conservative regions keep the caller's
    normal GEMM or explicit autotuner path.

    * ``M=1, N≤2048`` for any supported K.
    * ``M=1, K≥4096, N≤4608`` for deeper projections.
    * ``M≤4, K=8192, N≤512`` and ``M≤8, K=8192, N≤256``.

    The caller still validates dtype, device, layout, alignment, and kernel
    tactic support before dispatching.
    """
    if m == 1:
        return n <= 2048 or (k >= 4096 and n <= 4608)
    return k == 8192 and ((m <= 4 and n <= 512) or (m <= 8 and n <= 256))


class _DirectDenseGemmKernel:
    """K-specialised direct GEMM with whole-mainloop vector register prefetch."""

    def __init__(
        self,
        *,
        element_type,
        num_rows: int,
        k_extent: int,
        tactic: DirectTactic,
        use_pdl: bool,
        epilogue_mode: str = "none",
        epilogue_scale: float = 1.0,
        epilogue_prefix: int = 0,
    ) -> None:
        validate_tactic(tactic, num_rows, tactic.outputs_per_block, k_extent)
        if epilogue_mode not in _EPILOGUE_MODES:
            raise ValueError(f"unsupported epilogue_mode={epilogue_mode}")
        if epilogue_mode == "silu_prefix" and (
            not isinstance(epilogue_prefix, int) or isinstance(epilogue_prefix, bool)
        ):
            raise ValueError("silu epilogue_prefix must be an integer")
        if epilogue_mode == "silu_prefix" and epilogue_prefix <= 0:
            raise ValueError(f"silu epilogue_prefix must be positive, got {epilogue_prefix}")
        self.element_type = element_type
        self.num_rows = num_rows
        self.rows_per_block = tactic.rows_per_block
        self.k_extent = k_extent
        self.block_size = tactic.block_size
        self.outputs_per_block = tactic.outputs_per_block
        self.vector_width = _VECTOR_WIDTH
        self.use_pdl = use_pdl
        self.epilogue_mode = epilogue_mode
        self.epilogue_scale = epilogue_scale
        self.epilogue_prefix = epilogue_prefix
        self.num_warps = tactic.block_size // cute.arch.WARP_SIZE
        self.num_k_tiles = k_extent // (tactic.block_size * _VECTOR_WIDTH)

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        stream: _cuda.CUstream,
    ) -> None:
        n = cute.size(gB, mode=[0])
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(gA, gB, gC, copy_a, copy_b).launch(
            grid=[
                cute.ceil_div(n, self.outputs_per_block),
                self.num_rows // self.rows_per_block,
                1,
            ],
            block=[self.block_size, 1, 1],
            smem=self.rows_per_block * self.outputs_per_block * self.num_warps * 4,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, block_m, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        num_rows: cutlass.Constexpr = self.rows_per_block
        outputs_per_block: cutlass.Constexpr = self.outputs_per_block
        vector_width: cutlass.Constexpr = self.vector_width
        block_size: cutlass.Constexpr = self.block_size
        num_warps: cutlass.Constexpr = self.num_warps
        num_k_tiles: cutlass.Constexpr = self.num_k_tiles

        # FP32 accumulators in registers — one per (row, output) pair.
        acc = cute.make_rmem_tensor(
            cute.make_layout((num_rows, outputs_per_block), stride=(outputs_per_block, 1)),
            cutlass.Float32,
        )
        acc.fill(0.0)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        n_base = block_idx * outputs_per_block
        m_base = block_m * num_rows

        # Vectorised view of A and B for G2R copy.
        gA_vec = cute.logical_divide(gA, (None, vector_width))
        gB_vec = cute.logical_divide(gB, (None, vector_width))
        tA_all = cute.logical_divide(gA_vec, (None, (None, block_size)))
        tB_all = cute.logical_divide(gB_vec, (None, (None, block_size)))
        tA = tA_all[None, (None, (tidx, None))]

        # Prefetch the entire K dimension of each requested B (weight) row
        # into registers before accumulation begins.
        b_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (outputs_per_block, num_k_tiles, vector_width),
                stride=(num_k_tiles * vector_width, vector_width, 1),
            ),
            self.element_type,
        )
        for ni in cutlass.range_constexpr(outputs_per_block):
            tB = tB_all[n_base + ni, (None, (tidx, None))]
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                cute.copy(copy_b, tB[None, k_tile], b_regs[ni, k_tile, None])

        # Stream each A row from global memory and accumulate against cached B.
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_k_tiles, vector_width), stride=(vector_width, 1)),
            self.element_type,
        )
        for mi in cutlass.range_constexpr(num_rows):
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                cute.copy(copy_a, tA[m_base + mi, None, k_tile], a_regs[k_tile, None])
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                for vi in cutlass.range_constexpr(vector_width):
                    a_value = a_regs[k_tile, vi].to(cutlass.Float32)
                    for ni in cutlass.range_constexpr(outputs_per_block):
                        acc[mi, ni] = acc[mi, ni] + a_value * b_regs[ni, k_tile, vi].to(
                            cutlass.Float32
                        )

        # Warp-level reduction then cross-warp reduction via shared memory.
        for mi in cutlass.range_constexpr(num_rows):
            for ni in cutlass.range_constexpr(outputs_per_block):
                acc[mi, ni] = cute.arch.warp_reduction_sum(acc[mi, ni])

        smem_layout = cute.make_layout(
            (num_rows, outputs_per_block, num_warps),
            stride=(outputs_per_block * num_warps, num_warps, 1),
        )
        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(cutlass.Float32, smem_layout, byte_alignment=16)
        with cute.arch.elect_one():
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    partials[mi, ni, warp_idx] = acc[mi, ni]

        cute.arch.sync_threads()
        if tidx == 0:
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    total = cutlass.Float32(0.0)
                    for warp in cutlass.range_constexpr(num_warps):
                        total = total + partials[mi, ni, warp]
                    value = total.to(self.element_type)
                    if const_expr(self.epilogue_mode == "silu_prefix"):
                        if n_base + ni < self.epilogue_prefix:
                            # Preserve GEMM -> BF16 -> activation rounding when
                            # replacing the separate pointwise kernel.
                            scaled = value.to(cutlass.Float32) * self.epilogue_scale
                            value = (scaled * _sigmoid_f32(scaled)).to(self.element_type)
                    gC[m_base + mi, n_base + ni] = value

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class _DirectDenseGemmInputQuantKernel(_DirectDenseGemmKernel):
    """Experimental direct GEMM that also emits an FP8 copy of its M=1 input."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.num_rows != 1:
            raise ValueError("input-quant side output currently requires M=1")
        if self.k_extent % 128:
            raise ValueError("input-quant side output requires K divisible by 128")
        self.num_quant_blocks = self.k_extent // 128

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        gAQuant: cute.Tensor,
        gAScale: cute.Tensor,
        stream: _cuda.CUstream,
    ) -> None:
        n = cute.size(gB, mode=[0])
        if const_expr(cute.ceil_div(n, self.outputs_per_block) < self.num_quant_blocks):
            raise ValueError("direct GEMM grid is too small to emit every input scale block")
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(gA, gB, gC, gAQuant, gAScale, copy_a, copy_b).launch(
            grid=[
                cute.ceil_div(n, self.outputs_per_block),
                self.num_rows // self.rows_per_block,
                1,
            ],
            block=[self.block_size, 1, 1],
            smem=self.rows_per_block * self.outputs_per_block * self.num_warps * 4,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        gAQuant: cute.Tensor,
        gAScale: cute.Tensor,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, block_m, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        num_rows: cutlass.Constexpr = self.rows_per_block
        outputs_per_block: cutlass.Constexpr = self.outputs_per_block
        vector_width: cutlass.Constexpr = self.vector_width
        block_size: cutlass.Constexpr = self.block_size
        num_warps: cutlass.Constexpr = self.num_warps
        num_k_tiles: cutlass.Constexpr = self.num_k_tiles
        num_quant_blocks: cutlass.Constexpr = self.num_quant_blocks
        quant_vectors: cutlass.Constexpr = 128 // self.vector_width

        acc = cute.make_rmem_tensor(
            cute.make_layout((num_rows, outputs_per_block), stride=(outputs_per_block, 1)),
            cutlass.Float32,
        )
        acc.fill(0.0)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        n_base = block_idx * outputs_per_block
        m_base = block_m * num_rows

        gA_vec = cute.logical_divide(gA, (None, vector_width))
        gB_vec = cute.logical_divide(gB, (None, vector_width))
        tA_all = cute.logical_divide(gA_vec, (None, (None, block_size)))
        tB_all = cute.logical_divide(gB_vec, (None, (None, block_size)))
        tA = tA_all[None, (None, (tidx, None))]

        b_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (outputs_per_block, num_k_tiles, vector_width),
                stride=(num_k_tiles * vector_width, vector_width, 1),
            ),
            self.element_type,
        )
        for ni in cutlass.range_constexpr(outputs_per_block):
            tB = tB_all[n_base + ni, (None, (tidx, None))]
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                cute.copy(copy_b, tB[None, k_tile], b_regs[ni, k_tile, None])

        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_k_tiles, vector_width), stride=(vector_width, 1)),
            self.element_type,
        )
        for mi in cutlass.range_constexpr(num_rows):
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                cute.copy(copy_a, tA[m_base + mi, None, k_tile], a_regs[k_tile, None])
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                for vi in cutlass.range_constexpr(vector_width):
                    a_value = a_regs[k_tile, vi].to(cutlass.Float32)
                    for ni in cutlass.range_constexpr(outputs_per_block):
                        acc[mi, ni] = acc[mi, ni] + a_value * b_regs[ni, k_tile, vi].to(
                            cutlass.Float32
                        )

        # The first K/128 output CTAs each quantize one disjoint input block.
        # All of them already hold the complete M=1 input in registers.
        if block_m == 0:
            if block_idx < num_quant_blocks:
                quant_vector_begin = block_idx * quant_vectors
                quant_vector_end = quant_vector_begin + quant_vectors
                amax = cutlass.Float32(0.0)
                for k_tile in cutlass.range_constexpr(num_k_tiles):
                    vector_idx = k_tile * block_size + tidx
                    for vi in cutlass.range_constexpr(vector_width):
                        value = cutlass.Float32(0.0)
                        if vector_idx >= quant_vector_begin:
                            if vector_idx < quant_vector_end:
                                value = a_regs[k_tile, vi].to(cutlass.Float32)
                        lane_amax = cute.arch.warp_redux_sync(value, "fmax", abs=True)
                        amax = cute.arch.fmax(amax, lane_amax)

                amax = cute.arch.fmax(amax, cutlass.Float32(1e-10))
                quant_scale = cutlass.Float32(448.0) / amax
                for k_tile in cutlass.range_constexpr(num_k_tiles):
                    vector_idx = k_tile * block_size + tidx
                    if vector_idx >= quant_vector_begin:
                        if vector_idx < quant_vector_end:
                            for vi in cutlass.range_constexpr(vector_width):
                                gAQuant[0, vector_idx * vector_width + vi] = (
                                    a_regs[k_tile, vi].to(cutlass.Float32) * quant_scale
                                ).to(cutlass.Float8E4M3FN)
                            if vector_idx == quant_vector_begin:
                                gAScale[block_idx, 0] = cutlass.Float32(1.0) / quant_scale

        for mi in cutlass.range_constexpr(num_rows):
            for ni in cutlass.range_constexpr(outputs_per_block):
                acc[mi, ni] = cute.arch.warp_reduction_sum(acc[mi, ni])

        smem_layout = cute.make_layout(
            (num_rows, outputs_per_block, num_warps),
            stride=(outputs_per_block * num_warps, num_warps, 1),
        )
        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(cutlass.Float32, smem_layout, byte_alignment=16)
        with cute.arch.elect_one():
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    partials[mi, ni, warp_idx] = acc[mi, ni]

        cute.arch.sync_threads()
        if tidx == 0:
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    total = cutlass.Float32(0.0)
                    for warp in cutlass.range_constexpr(num_warps):
                        total = total + partials[mi, ni, warp]
                    gC[m_base + mi, n_base + ni] = total.to(self.element_type)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()


def _from_dlpack_static(tensor: _torch.Tensor):
    """Wrap a PyTorch tensor with a static (shape-specialised) CuTe view."""
    return from_dlpack(tensor, assumed_align=32)


def _make_compile_repr_tensors(dtype: _torch.dtype, m: int, n: int, k: int):
    return tuple(
        _from_dlpack_static(tensor)
        for tensor in (
            _torch.empty((m, k), dtype=dtype, device="cuda"),  # gA [M, K] row-major
            _torch.empty((n, k), dtype=dtype, device="cuda"),  # gB [N, K] row-major (=b.T)
            _torch.empty((m, n), dtype=dtype, device="cuda"),  # gC [M, N] row-major
        )
    )


def _make_compile_repr_quant_tensors(dtype: _torch.dtype, m: int, n: int, k: int):
    return tuple(
        _from_dlpack_static(tensor)
        for tensor in (
            _torch.empty((m, k), dtype=dtype, device="cuda"),
            _torch.empty((n, k), dtype=dtype, device="cuda"),
            _torch.empty((m, n), dtype=dtype, device="cuda"),
            _torch.empty((m, k), dtype=_torch.float8_e4m3fn, device="cuda"),
            _torch.empty((k // 128, m), dtype=_torch.float32, device="cuda"),
        )
    )


@functools.cache
def _get_compiled_direct_kernel(
    dtype: _torch.dtype,
    m: int,
    n: int,
    k: int,
    tactic: DirectTactic,
    use_pdl: bool,
    epilogue_mode: str = "none",
    epilogue_scale: float = 1.0,
    epilogue_prefix: int = 0,
):
    """JIT-compile and cache a specialised ``_DirectDenseGemmKernel``."""
    if dtype != _torch.bfloat16:
        raise ValueError(f"direct GEMM supports BF16; got {dtype}")
    kernel = _DirectDenseGemmKernel(
        element_type=cutlass.BFloat16,
        num_rows=m,
        k_extent=k,
        tactic=tactic,
        use_pdl=use_pdl,
        epilogue_mode=epilogue_mode,
        epilogue_scale=epilogue_scale,
        epilogue_prefix=epilogue_prefix,
    )
    tensors = _make_compile_repr_tensors(dtype, m, n, k)
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(kernel, *tensors, stream, options=_COMPILE_OPTIONS)


@functools.cache
def _get_compiled_direct_quant_kernel(
    dtype: _torch.dtype,
    m: int,
    n: int,
    k: int,
    tactic: DirectTactic,
    use_pdl: bool,
):
    if dtype != _torch.bfloat16:
        raise ValueError(f"direct GEMM supports BF16; got {dtype}")
    kernel = _DirectDenseGemmInputQuantKernel(
        element_type=cutlass.BFloat16,
        num_rows=m,
        k_extent=k,
        tactic=tactic,
        use_pdl=use_pdl,
    )
    tensors = _make_compile_repr_quant_tensors(dtype, m, n, k)
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(kernel, *tensors, stream, options=_COMPILE_OPTIONS)


def _validate_runtime_tensors(
    a: _torch.Tensor,
    b: _torch.Tensor,
    out: _torch.Tensor,
    tactic: DirectTactic,
) -> tuple[int, int, int]:
    """Validate shapes/layouts and return ``(m, n, k)``.

    Expected layouts:
        * ``a``:   ``[M, K]`` row-major.
        * ``b``:   ``[K, N]`` column-major (i.e. ``b.T`` is row-major ``[N, K]``).
        * ``out``: ``[M, N]`` row-major.
    """
    if any(not isinstance(t, _torch.Tensor) for t in (a, b, out)):
        raise ValueError("a, b, and out must be torch tensors")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("direct GEMM accepts only 2D tensors")
    if a.device.type != "cuda" or b.device != a.device or out.device != a.device:
        raise ValueError("a, b, and out must be on the same CUDA device")
    if a.dtype != _torch.bfloat16 or b.dtype != a.dtype or out.dtype != a.dtype:
        raise ValueError("a, b, and out must share BF16 dtype")
    if not a.is_contiguous() or not b.T.is_contiguous() or not out.is_contiguous():
        raise ValueError(
            "direct GEMM requires row-major a and out, and column-major b "
            "(so that b.T is row-major [N, K])"
        )
    if any(t.data_ptr() % 32 for t in (a, b, out)):
        raise ValueError("a, b, and out must be 32-byte aligned")
    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(f"incompatible shapes: a is {tuple(a.shape)}, b is {tuple(b.shape)}")
    n = b.shape[1]
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    validate_tactic(tactic, m, n, k)
    return m, n, k


def run_direct_dense(
    a: _torch.Tensor,
    b: _torch.Tensor,
    out: _torch.Tensor,
    pdl: bool,
    tactic: DirectTactic,
) -> _torch.Tensor:
    """Run ``A[M,K] @ B[K,N]`` via the direct (SIMT) register-prefetch kernel.

    Args:
        a:      Input ``[M, K]`` row-major BF16 tensor.
        b:      Weight ``[K, N]`` column-major BF16 tensor.
                Pass ``weight.t()`` when the original weight is ``[N, K]``
                row-major — the view is zero-copy.
        out:    Pre-allocated ``[M, N]`` row-major output.
        pdl:    Whether to use Programmatic Dependent Launch.
        tactic: A ``DirectTactic`` specifying block and tiling parameters.

    Returns:
        The ``out`` tensor filled with the GEMM result.
    """
    m, n, k = _validate_runtime_tensors(a, b, out, tactic)
    compiled = _get_compiled_direct_kernel(a.dtype, m, n, k, tactic, pdl)
    # Pass b.T (row-major [N, K]) so the kernel sees gB[N, K].
    tensors = tuple(_from_dlpack_static(t) for t in (a, b.T, out))
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    compiled(*tensors, stream)
    return out


def run_direct_dense_silu_prefix(
    a: _torch.Tensor,
    b: _torch.Tensor,
    out: _torch.Tensor,
    pdl: bool,
    tactic: DirectTactic,
    scale: float,
    prefix: int,
) -> _torch.Tensor:
    """Apply ``silu(scale * x)`` to a GEMM output prefix.

    Columns after ``prefix`` retain the ordinary GEMM result.
    """
    m, n, k = _validate_runtime_tensors(a, b, out, tactic)
    if not isinstance(prefix, int) or isinstance(prefix, bool):
        raise ValueError("prefix must be an integer")
    if not 0 < prefix <= n:
        raise ValueError(f"prefix must be in [1, {n}], got {prefix}")
    compiled = _get_compiled_direct_kernel(
        a.dtype,
        m,
        n,
        k,
        tactic,
        pdl,
        "silu_prefix",
        scale,
        prefix,
    )
    # Pass b.T (row-major [N, K]) so the kernel sees gB[N, K].
    tensors = tuple(_from_dlpack_static(t) for t in (a, b.T, out))
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    compiled(*tensors, stream)
    return out


_DIRECT_DENSE_EPILOGUE_RUNNERS = MappingProxyType(
    {
        "none": run_direct_dense,
        "silu_prefix": run_direct_dense_silu_prefix,
    }
)


def get_direct_dense_epilogue_runner(name: str) -> Callable[..., _torch.Tensor]:
    """Return the registered Direct GEMM runner for an epilogue name."""
    if not isinstance(name, str):
        raise ValueError(f"Direct GEMM epilogue name must be a string, got {name!r}")
    try:
        return _DIRECT_DENSE_EPILOGUE_RUNNERS[name]
    except KeyError as error:
        available = ", ".join(_DIRECT_DENSE_EPILOGUE_RUNNERS)
        raise ValueError(
            f"unsupported Direct GEMM epilogue {name!r}; expected one of: {available}"
        ) from error


def run_direct_dense_with_input_quant(
    a: _torch.Tensor,
    b: _torch.Tensor,
    out: _torch.Tensor,
    quantized_a: _torch.Tensor,
    a_scale: _torch.Tensor,
    pdl: bool,
    tactic: DirectTactic,
) -> tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
    """Run M=1 direct GEMM and emit its input in FP8 1x128 block-scale form."""
    m, n, k = _validate_runtime_tensors(a, b, out, tactic)
    if m != 1 or k % 128:
        raise ValueError("input-quant side output requires M=1 and K divisible by 128")
    if (n + tactic.outputs_per_block - 1) // tactic.outputs_per_block < k // 128:
        raise ValueError("direct GEMM grid is too small to emit every input scale block")
    if quantized_a.shape != a.shape or quantized_a.dtype != _torch.float8_e4m3fn:
        raise ValueError("quantized_a must be contiguous FP8 E4M3 with the same shape as a")
    if a_scale.shape != (k // 128, m) or a_scale.dtype != _torch.float32:
        raise ValueError(f"a_scale must be contiguous FP32 with shape {(k // 128, m)}")
    if quantized_a.device != a.device or a_scale.device != a.device:
        raise ValueError("quantized_a and a_scale must be on the same CUDA device as a")
    if not quantized_a.is_contiguous() or not a_scale.is_contiguous():
        raise ValueError("quantized_a and a_scale must be contiguous")

    compiled = _get_compiled_direct_quant_kernel(a.dtype, m, n, k, tactic, pdl)
    tensors = tuple(_from_dlpack_static(t) for t in (a, b.T, out, quantized_a, a_scale))
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    compiled(*tensors, stream)
    return out, quantized_a, a_scale


__all__ = [
    "MAX_M",
    "DirectTactic",
    "autotune_tactics",
    "default_tactic",
    "get_direct_dense_epilogue_runner",
    "prefer_direct_bf16_gemm_sm100",
    "run_direct_dense",
    "run_direct_dense_silu_prefix",
    "run_direct_dense_with_input_quant",
    "validate_tactic",
]

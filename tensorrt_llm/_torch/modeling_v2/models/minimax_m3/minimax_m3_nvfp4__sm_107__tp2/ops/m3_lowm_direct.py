# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Register-prefetch BF16 GEMM for low-M, long-K decode shapes.

This target's copy of the installed FlashInfer's ``flashinfer/gemm/kernels/dense_bf16_gemm_direct.py`` with one
addition, the drafter-boundary push (``ar_push``, modeling's DRAFT_AR_PUSH_DIRECT): after its one store of each output,
the owning thread also stores the CTA's column pairs as the boundary all-reduce's canonical exchange words (kernels/
m3_mc_allreduce) at the call parity, before the kernel's (unchanged, last) release of its dependents. The plain
kernel and its output are FlashInfer's; only the push variant is called here (``run_direct_dense_push``).

The kernel keeps a complete output dot product inside one CTA and reuses each
prefetched B value across several public-M rows.  It is intentionally a
separate autotuner runner from the Blackwell tensor-core split-K kernel: the
two algorithms have different useful shape regions and tactic spaces.
"""

from __future__ import annotations

import dataclasses
import functools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch as _torch
from cutlass import Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

_VECTOR_WIDTH = 8  # capacity: fixed: a vector load width
_SUPPORTED_BLOCK_SIZES = (32, 64, 96, 128, 192, 256, 384)
_SUPPORTED_OUTPUTS_PER_BLOCK = (1, 2, 4)
_MAX_M = 32  # capacity: fixed: the copied FlashInfer kernel's row bound
_COMPILE_OPTIONS = "--ptxas-options -maxrregcount=64"


@dsl_user_op
def _ld_volatile_s32(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.s32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _st_global_b32(addr, value, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _bf16x2_word(lo, hi, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .b32 w, l, h;\n\t.reg .pred pl, ph;\n\tcvt.rn.bf16x2.f32 w, $1, $2;\n\t"
            "and.b32 l, w, 0xFFFF;\n\tsetp.eq.u32 pl, l, 0x8000;\n\t@pl and.b32 w, w, 0xFFFF0000;\n\t"
            "shr.u32 h, w, 16;\n\tsetp.eq.u32 ph, h, 0x8000;\n\t@ph and.b32 w, w, 0x0000FFFF;\n\t"
            "mov.b32 $0, w;\n\t}",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dataclasses.dataclass(frozen=True)
class DirectTactic:
    """One direct-kernel specialization."""

    block_size: int
    outputs_per_block: int
    rows_per_block: int


def _default_rows_per_block(m: int) -> int:
    if m <= 8:
        return m
    return next(rows for rows in (8, 4, 2, 1) if m % rows == 0)


def validate_tactic(tactic: DirectTactic, m: int, n: int, k: int) -> None:
    """Reject a direct tactic that cannot serve ``(m, n, k)``."""
    if tactic.block_size not in _SUPPORTED_BLOCK_SIZES:
        raise ValueError(f"unsupported block_size={tactic.block_size}")
    if tactic.outputs_per_block not in _SUPPORTED_OUTPUTS_PER_BLOCK:
        raise ValueError(f"unsupported outputs_per_block={tactic.outputs_per_block}")
    if not 1 <= m <= _MAX_M:
        raise ValueError(f"direct GEMM requires 1 <= M <= {_MAX_M}, got {m}")
    if not 1 <= tactic.rows_per_block <= m or m % tactic.rows_per_block:
        raise ValueError(f"rows_per_block={tactic.rows_per_block} must divide M={m}")
    if n <= 0 or n % tactic.outputs_per_block:
        raise ValueError(f"N={n} must be divisible by outputs_per_block={tactic.outputs_per_block}")
    k_tile = tactic.block_size * _VECTOR_WIDTH
    if k <= 0 or k % k_tile:
        raise ValueError(f"K={k} must be divisible by {k_tile}")


def default_tactic(m: int, n: int, k: int) -> DirectTactic:
    """Choose the measured register-prefetch fallback tactic."""
    block_size = next(
        (block for block in (256, 192, 128, 96, 64, 32) if k % (block * _VECTOR_WIDTH) == 0),
        None,
    )
    if block_size is None:
        raise ValueError("direct GEMM requires a supported 16-byte K tiling")
    outputs_per_block = next(outputs for outputs in (2, 1) if n % outputs == 0)
    tactic = DirectTactic(
        block_size,
        outputs_per_block,
        _default_rows_per_block(m),
    )
    validate_tactic(tactic, m, n, k)
    return tactic


def autotune_tactics(m: int, n: int, k: int) -> list[DirectTactic]:
    """Enumerate the compact tactic space used by FlashInfer autotuning.

    Block sizes cover every configuration exercised in the H100/B200 sweep;
    output grouping spans the measured 1/2/4-column choices.  Row tiling stays
    at the occupancy-oriented default to keep JIT cost bounded.
    """
    try:
        default = default_tactic(m, n, k)
    except ValueError:
        return []
    tactics = [default]
    for block_size in _SUPPORTED_BLOCK_SIZES:
        for outputs_per_block in _SUPPORTED_OUTPUTS_PER_BLOCK:
            tactic = DirectTactic(
                block_size,
                outputs_per_block,
                default.rows_per_block,
            )
            try:
                validate_tactic(tactic, m, n, k)
            except ValueError:
                continue
            tactics.append(tactic)
    return list(dict.fromkeys(tactics))


def prefer_direct_bf16_gemm_sm100(m: int, n: int, k: int) -> bool:
    """Return the conservative B200 no-autotune crossover heuristic.

    The three bands are a compact fit to a warm/cold sweep over M=1..16,24,32,
    18 N values, and 11 K values.  This is deliberately not a blanket rule for
    K=8192: direct wins only where public M and N leave the tensor-core path
    with too little independent output work.
    """
    return k == 8192 and ((m == 1 and n <= 4608) or (m <= 4 and n <= 512) or (m <= 8 and n <= 256))


class DirectDenseGemmKernel:
    """K-specialized direct GEMM with whole-mainloop vector prefetch."""

    def __init__(
        self,
        *,
        element_type,
        num_rows: int,
        k_extent: int,
        tactic: DirectTactic,
        use_pdl: bool,
        ar_push: bool = False,
    ) -> None:
        validate_tactic(tactic, num_rows, tactic.outputs_per_block, k_extent)
        self.ar_push = ar_push
        self.element_type = element_type
        self.num_rows = num_rows
        self.rows_per_block = tactic.rows_per_block
        self.k_extent = k_extent
        self.block_size = tactic.block_size
        self.outputs_per_block = tactic.outputs_per_block
        self.vector_width = _VECTOR_WIDTH
        self.use_pdl = use_pdl
        self.num_warps = tactic.block_size // cute.arch.WARP_SIZE
        self.num_k_tiles = k_extent // (tactic.block_size * _VECTOR_WIDTH)

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        stream: _cuda.CUstream,
        ar_ws: cutlass.Int64 = cutlass.Int64(0),
        ar_flags: cutlass.Int64 = cutlass.Int64(0),
        ar_rank: Int32 = Int32(0),
        ar_world: Int32 = Int32(1),
        ar_maxt: Int32 = Int32(8),
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
        self.kernel(gA, gB, gC, copy_a, copy_b, ar_ws, ar_flags, ar_rank, ar_world, ar_maxt).launch(
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
        ar_ws: cutlass.Int64,
        ar_flags: cutlass.Int64,
        ar_rank: Int32,
        ar_world: Int32,
        ar_maxt: Int32,
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
                cute.copy(
                    copy_a,
                    tA[m_base + mi, None, k_tile],
                    a_regs[k_tile, None],
                )
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                for vi in cutlass.range_constexpr(vector_width):
                    a_value = a_regs[k_tile, vi].to(cutlass.Float32)
                    for ni in cutlass.range_constexpr(outputs_per_block):
                        acc[mi, ni] = acc[mi, ni] + a_value * b_regs[ni, k_tile, vi].to(
                            cutlass.Float32
                        )

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
                    if const_expr(self.ar_push):
                        acc[mi, ni] = total
            if const_expr(self.ar_push):
                # Push the CTA's column pairs of every row into the boundary all-reduce's exchange (prepushed 2), at the
                # (token, column / (N / 8)) call parity; the release below follows the pushes (the prepushed contract).
                n_cols = cute.size(gC, mode=[1])
                half = cutlass.Int64(ar_maxt) * cutlass.Int64(ar_world) * cutlass.Int64(n_cols // 2)
                for mi in cutlass.range_constexpr(num_rows):
                    t = m_base + mi
                    for pi in cutlass.range_constexpr(outputs_per_block // 2):
                        col = n_base + 2 * pi
                        slot = cutlass.Int64(t * 8 + col // (n_cols // 8))
                        par = _ld_volatile_s32(ar_flags + cutlass.Int64(4) * slot) & Int32(1)
                        row = cutlass.Int64(t * ar_world + ar_rank)
                        off = (
                            cutlass.Int64(par) * half
                            + row * cutlass.Int64(n_cols // 2)
                            + cutlass.Int64(col // 2)
                        )
                        word = _bf16x2_word(acc[mi, 2 * pi], acc[mi, 2 * pi + 1])
                        _st_global_b32(ar_ws + off * cutlass.Int64(4), word)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()


def _from_dlpack_static(tensor: _torch.Tensor):
    # K is specialized and the row stride must retain its 16-byte divisibility
    # for the verifier to accept vectorized G2R copies.
    return from_dlpack(tensor, assumed_align=32)


def _make_compile_repr_tensors(dtype, m: int, n: int, k: int):
    return tuple(
        _from_dlpack_static(tensor)
        for tensor in (
            _torch.empty((m, k), dtype=dtype, device="cuda"),
            _torch.empty((n, k), dtype=dtype, device="cuda"),
            _torch.empty((m, n), dtype=dtype, device="cuda"),
        )
    )


@functools.cache
def _get_compiled_direct_kernel(
    dtype,
    m: int,
    n: int,
    k: int,
    tactic: DirectTactic,
    use_pdl: bool,
):
    if dtype != _torch.bfloat16:
        raise ValueError(f"direct GEMM supports BF16; got {dtype}")
    kernel = DirectDenseGemmKernel(
        element_type=cutlass.BFloat16,
        num_rows=m,
        k_extent=k,
        tactic=tactic,
        use_pdl=use_pdl,
    )
    tensors = _make_compile_repr_tensors(dtype, m, n, k)
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(kernel, *tensors, stream, options=_COMPILE_OPTIONS)


def _validate_runtime_tensors(a, b, out, tactic: DirectTactic):
    if any(not isinstance(tensor, _torch.Tensor) for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be torch tensors")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("direct GEMM accepts only 2D tensors")
    if a.device.type != "cuda" or b.device != a.device or out.device != a.device:
        raise ValueError("a, b, and out must be on the same CUDA device")
    if a.dtype != _torch.bfloat16 or b.dtype != a.dtype or out.dtype != a.dtype:
        raise ValueError("a, b, and out must share BF16 dtype")
    if not a.is_contiguous() or not b.T.is_contiguous() or not out.is_contiguous():
        raise ValueError("direct GEMM requires row-major A/out and column-major B")
    if any(tensor.data_ptr() % 32 for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be 32-byte aligned")

    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(f"incompatible shapes: a is {tuple(a.shape)}, b is {tuple(b.shape)}")
    n = b.shape[1]
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    validate_tactic(tactic, m, n, k)
    return m, n, k


def run_direct_dense(a, b, out, pdl: bool, tactic: DirectTactic):
    """Run direct ``A[M,K] @ B[K,N]`` with the ``mm_bf16`` layouts."""
    m, n, k = _validate_runtime_tensors(a, b, out, tactic)
    compiled = _get_compiled_direct_kernel(a.dtype, m, n, k, tactic, pdl)
    tensors = tuple(_from_dlpack_static(tensor) for tensor in (a, b.T, out))
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    compiled(*tensors, stream)
    return out


__all__ = [
    "DirectTactic",
    "autotune_tactics",
    "default_tactic",
    "prefer_direct_bf16_gemm_sm100",
    "run_direct_dense",
    "run_direct_dense_push",
    "compile_direct_push",
]


@functools.cache
def _get_compiled_direct_push(dtype, m: int, n: int, k: int, tactic: DirectTactic, use_pdl: bool):
    assert tactic.outputs_per_block % 2 == 0, "the push writes column pairs"
    kernel = DirectDenseGemmKernel(
        element_type=cutlass.BFloat16,
        num_rows=m,
        k_extent=k,
        tactic=tactic,
        use_pdl=use_pdl,
        ar_push=True,
    )
    tensors = _make_compile_repr_tensors(dtype, m, n, k)
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(
        kernel,
        *tensors,
        stream,
        cutlass.Int64(0),
        cutlass.Int64(0),
        Int32(0),
        Int32(1),
        Int32(8),
        options=_COMPILE_OPTIONS,
    )


def compile_direct_push(a, b, out, pdl: bool, tactic) -> None:
    """Compile (only) the push variant for these operands' shapes."""
    m, n, k = _validate_runtime_tensors(a, b, out, DirectTactic(*tactic))
    _get_compiled_direct_push(a.dtype, m, n, k, DirectTactic(*tactic), pdl)


def run_direct_dense_push(a, b, out, pdl: bool, tactic: DirectTactic, ar: tuple):
    """``run_direct_dense`` that also pushes ``out``'s rows into a drafter boundary all-reduce's exchange."""
    m, n, k = _validate_runtime_tensors(a, b, out, tactic)
    compiled = _get_compiled_direct_push(a.dtype, m, n, k, tactic, pdl)
    tensors = tuple(_from_dlpack_static(tensor) for tensor in (a, b.T, out))
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    ws, flags, rank, world, maxt = ar
    compiled(
        *tensors,
        stream,
        cutlass.Int64(ws),
        cutlass.Int64(flags),
        Int32(rank),
        Int32(world),
        Int32(maxt),
    )
    return out

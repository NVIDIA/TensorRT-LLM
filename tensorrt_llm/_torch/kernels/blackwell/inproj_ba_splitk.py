# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 projection to a few outputs, ``y[M, N] = x[M, K] @ W[N, K].T``, with the arithmetic of cuBLAS' split-K nvjet
kernel (four FP32 partials over K / 4 each, summed in order, then BF16; bitwise to F.linear where cuBLAS picks that
kernel), in one launch that loads W before the PDL wait: a 4-CTA cluster per 32-row token tile, CTA rank = K split,
in-order tcgen05 kind::f16 MMAs (M 64, the outputs in its first N rows), partials summed in the leader's DSMEM.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

from ...flashinfer_utils import get_env_enable_pdl

SPLITS = 4  # K splits, one CTA each
KBOX = 64  # bf16 per 128-byte swizzle row
TOK = 32  # token rows per tile (MMA N)
MROWS = 64  # MMA M: the N outputs are its first rows; the others only feed accumulator rows that are never read
X_BOX = TOK * KBOX  # elements per x box (4 KB)
XG = 4  # x boxes per barrier
THREADS = 128
# Most rows of a call (three token tiles): cuBLAS keeps its split-K kernel up to here at in_proj_ba shapes.
MAX_ROWS = 96


@dsl_user_op
def _st_bf16(addr: Int64, v: Float32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tst.global.b16 [$0], h;\n\t}",
        "l,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@functools.lru_cache(maxsize=None)
def _smem_limit() -> int:
    """A CTA's opt-in shared memory on the current device."""
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).shared_memory_per_block_optin


def _smem_bytes(k: int, n: int) -> int:
    nbox = k // SPLITS // KBOX
    return 2 * (nbox * n * KBOX + (MROWS - n) * KBOX + nbox * X_BOX) + 4 * SPLITS * TOK * n + 1024


def plan(m: int, k: int, n: int) -> int | None:
    """Token tiles, or None outside 1..MAX_ROWS rows or the kernel's shapes: K in whole barrier groups of x boxes per
    split; N in whole 8-row swizzle atoms within the 16 accumulator rows of TMEM sub-partition 0 (MMA M 64), which
    warp 0 reads; the CTA's shared memory within one SM."""
    if m < 1 or m > MAX_ROWS or k <= 0 or k % (SPLITS * KBOX * XG) or n not in (8, 16):
        return None
    if _smem_bytes(k, n) > _smem_limit():
        return None
    return -(-m // TOK)


def supports(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Whether :func:`ba_splitk` takes x [M, K] and w [N, K]: BF16, contiguous, at the compiled kernel's 128-byte
    alignment, and a shape :func:`plan` covers."""
    return (
        x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.dim() == 2
        and w.dim() == 2
        and x.shape[1] == w.shape[1]
        and x.is_contiguous()
        and w.is_contiguous()
        and x.data_ptr() % 128 == 0
        and w.data_ptr() % 128 == 0
        and plan(x.shape[0], x.shape[1], w.shape[0]) is not None
    )


@cute.kernel
def _kernel(
    tma_x_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_w_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    gOut: cute.Tensor,
    m_rows: cutlass.Constexpr[int],
    k_total: cutlass.Constexpr[int],
    n_out: cutlass.Constexpr[int],
):
    AB: cutlass.Constexpr = tma_w_desc.dtype
    NB: cutlass.Constexpr[int] = n_out  # output features
    SWZ: cutlass.Constexpr = tma_w_desc.swizzle.to(prims.Tcgen05SmemSwizzle)
    NKS: cutlass.Constexpr[int] = KBOX // 16  # MMAs per 64-K box
    NBOX: cutlass.Constexpr[int] = k_total // SPLITS // KBOX  # boxes per split
    W_BOX: cutlass.Constexpr[int] = NB * KBOX  # elements per W box (N rows), the boxes packed
    # past the last box: the rows the MMA reads beyond the N outputs (contents irrelevant)
    W_SLACK: cutlass.Constexpr[int] = (MROWS - NB) * KBOX

    smem_w = cutlass.Array(
        AB, NBOX * W_BOX + W_SLACK, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_x = cutlass.Array(AB, NBOX * X_BOX, space=cutlass.AddressSpace.smem, alignment=1024)
    part = cutlass.Array(
        Float32, SPLITS * TOK * NB, space=cutlass.AddressSpace.smem, alignment=16
    )  # leader's slots
    w_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    x_full = cutlass.Array(cutlass.Int64, NBOX // XG, space=cutlass.AddressSpace.smem, alignment=8)
    acc_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_i32 = cutlass.Array(Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    warp_idx = cute.arch.warp_idx()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % Int32(32)
    split = cute.arch.block_idx_in_cluster()
    tile = bidx // Int32(SPLITS)
    tok0 = tile * Int32(TOK)
    k0 = split * Int32(NBOX)  # first 64-K box of this split
    out_ptr = Int64(gOut.iterator.toint())

    if warp_idx == 0:
        if prims.elect_sync():
            prims.mbarrier_init(w_bar, 1)
            for g in cutlass.range_constexpr(NBOX // XG):
                prims.mbarrier_init(x_full.subview(g), 1)
            prims.mbarrier_init(acc_bar, 1)
    prims.fence_mbarrier_init()
    if warp_idx == 1:
        prims.tcgen05_alloc(tmem_i32, 32)
    prims.barrier_cta_sync(0)
    # Every CTA of the cluster has started before a peer stores into the leader's shared memory (waited below).
    prims.barrier_cluster_arrive_relaxed()
    tmem_raw = tmem_i32.load()
    tmem_ptr = cutlass.inttoptr(tmem_raw, 6, Float32)
    if warp_idx == 1:
        prims.tcgen05_relinquish_alloc_permit()
    x_ptr = tma_x_desc.get_ptr()
    w_ptr = tma_w_desc.get_ptr()
    w_bytes = tma_w_desc.global_tx_bytes()
    x_bytes = tma_x_desc.global_tx_bytes()

    # ---- TMA warp: the W chunk before the wait, x after it (whole-warp polls, elect only around issues) ----
    if warp_idx == 0:
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(w_bar, NBOX * w_bytes)
            for b in cutlass.range_constexpr(NBOX):
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_w.subview(b * W_BOX),
                    w_ptr,
                    (k0 * Int32(KBOX) + Int32(b * KBOX), Int32(0)),
                    w_bar,
                )
        cute.arch.griddepcontrol_wait()
        if prims.elect_sync():
            for g in cutlass.range_constexpr(NBOX // XG):
                prims.mbarrier_arrive_expect_tx(x_full.subview(g), XG * x_bytes)
                for i in cutlass.range_constexpr(XG):
                    b = g * XG + i
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_x.subview(b * X_BOX),
                        x_ptr,
                        (k0 * Int32(KBOX) + Int32(b * KBOX), tok0),
                        x_full.subview(g),
                    )

    # ---- MMA warp: 64 MMAs in K order ----
    if warp_idx == 1:
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=Float32, a_dtype=AB, b_dtype=AB, n_dim=TOK, m_dim=MROWS
        )
        desc_w = prims.Tcgen05SmemDesc.build(
            start_address=smem_w, leading_byte_offset=128, stride_byte_offset=1024, layout=SWZ
        )
        desc_x = prims.Tcgen05SmemDesc.build(
            start_address=smem_x, leading_byte_offset=128, stride_byte_offset=1024, layout=SWZ
        )
        while not prims.mbarrier_try_wait_parity(w_bar, 0, time_limit=10_000_000):
            pass
        for b in cutlass.range_constexpr(NBOX):
            if cutlass.const_expr(b % XG == 0):
                while not prims.mbarrier_try_wait_parity(
                    x_full.subview(b // XG), 0, time_limit=10_000_000
                ):
                    pass
            for ks in cutlass.range_constexpr(NKS):
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16,
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        desc_w.advance_start_address(Int32(b * W_BOX * 2 + ks * 32)),
                        desc_x.advance_start_address(Int32(b * X_BOX * 2 + ks * 32)),
                        idesc,
                        (b > 0) or (ks > 0),
                    )
        if prims.elect_sync():
            prims.tcgen05_commit(acc_bar)

    prims.barrier_cluster_wait()
    # ---- partial: lanes 0..N-1 of sub-partition 0 hold the N features x 32 tokens; push them into the leader ----
    if warp_idx == 0:
        while not prims.mbarrier_try_wait_parity(acc_bar, 0, time_limit=10_000_000):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc = cutlass.Array(Float32, TOK, alignment=32)
        base = prims.TmemAddr(tmem_raw)
        acc.store(
            prims.tcgen05_ld(
                "32x32b",
                prims.TmemAddr.from_row_col(base.row_id, base.col_id).as_ptr(Float32),
                num=TOK,
            ),
            0,
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        remote = prims.mapa(part, Int32(0))
        if lane < Int32(NB):
            for c in cutlass.range_constexpr(TOK):
                remote.store(acc[c], (split * Int32(TOK) + Int32(c)) * Int32(NB) + lane)
    cute.arch.sync_warp()
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cluster_arrive()  # release: the DSMEM stores before it
    prims.barrier_cluster_wait()  # acquire
    cute.arch.griddepcontrol_launch_dependents()
    if split == Int32(0):
        for i in cutlass.range_constexpr(TOK * NB // THREADS):
            idx = tidx + Int32(i * THREADS)
            t = idx // Int32(NB)
            f = idx % Int32(NB)
            tok = tok0 + t
            if tok < Int32(m_rows):
                p = []
                for s in cutlass.range_constexpr(SPLITS):
                    p.append(part.load(Int32(s * TOK * NB) + idx))
                v = ((p[0] + p[1]) + p[2]) + p[3]
                _st_bf16(out_ptr + (Int64(tok) * Int64(NB) + Int64(f)) * Int64(2), v)
    if warp_idx == 1:
        prims.tcgen05_dealloc(tmem_ptr, 32)


@cute.jit
def _launch(
    x: cute.Tensor,
    w: cute.Tensor,
    out: cute.Tensor,
    stream: cuda.CUstream,
    m_rows: cutlass.Constexpr[int],
    k_total: cutlass.Constexpr[int],
    n_out: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
):
    tma_x = cuda_exp.create_tensor_map_tiled(
        global_address=x.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[k_total, m_rows],
        global_strides=[k_total * 2 // 16],
        box_dims=[KBOX, TOK],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    tma_w = cuda_exp.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[k_total, n_out],
        global_strides=[k_total * 2 // 16],
        box_dims=[KBOX, n_out],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    _kernel(tma_x, tma_w, out, m_rows, k_total, n_out).launch(
        grid=(-(-m_rows // TOK) * SPLITS, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(SPLITS, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def _compile(m: int, k: int, n: int, use_pdl: bool):
    mk = cute.runtime.make_fake_compact_tensor
    return cute.compile(
        _launch,
        mk(cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=128),
        mk(cutlass.BFloat16, (n, k), stride_order=(1, 0), assumed_align=128),
        mk(cutlass.BFloat16, (m, n), stride_order=(1, 0), assumed_align=16),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        m_rows=m,
        k_total=k,
        n_out=n,
        use_pdl=use_pdl,
        options="--enable-tvm-ffi",
    )


_COMPILED = {}


def _compiled(*key):
    """``_compile(*key)``, once per key; the first call of a key must precede CUDA graph capture."""
    fn = _COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"compile the split-K in_proj_ba GEMM {key[:3]} before CUDA graph capture"
            )
        fn = _COMPILED[key] = _compile(*key)
    return fn


def ba_splitk(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """y[M, N] BF16 = x[M, K] BF16 @ w[N, K].T with cuBLAS' split-K 4 arithmetic (``supports`` operands)."""
    m, k = x.shape
    n = w.shape[0]
    assert supports(x, w)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _compiled(m, k, n, get_env_enable_pdl())(x, w, out)
    return out

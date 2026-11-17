# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Single-launch CuTe DSL attention_output_tail: W_UV and o_proj in one kernel (SM100/SM103).

One CTA per SM; 8 compute warps plus 1 producer warp.

* Producer warp: streams the CTA's 40-41 contiguous ``w_o`` rows into a
  row-padded shared-memory slab with one 1-D ``cp.async.bulk`` per 4 KiB row,
  each completing on its own mbarrier, keeping at most ``window`` rows in
  flight. Bounding the in-flight weight traffic keeps the memory system's
  queues short, so the latency-critical W_UV loads, ``v`` stores, flag
  updates and ``v`` loads of the other phases are not stuck behind tens of
  megabytes of queued copies; a window of a few rows per SM is still enough
  to keep HBM saturated.
* Phase 1, W_UV: CTAs ``0 .. K/16 - 1`` each own 16 columns of ``v`` (one
  head, 16 of its V rows of ``w_uv``). The compute warps split the 512-wide
  latent dimension, run ``mma.sync.m16n8k16`` (BF16 in, FP32 accumulate), add
  their partials in fixed warp order and store ``v`` rounded to BF16 -- the
  native intermediate rounding -- then publish the slice by incrementing a
  per-layer ``done`` counter (``fence.acq_rel.gpu`` + ``red``).
* Phase 2, o_proj: every CTA waits (acquire) until ``done`` reaches the target
  of its invocation, loads ``v`` from L2, and runs ``mma.sync`` over its slab
  rows as they land, with a fixed-order cross-warp reduction and one BF16
  rounding of the partial. At exit every CTA adds 1 to ``finished``.

With ``pdl_entry`` the kernel is a PDL dependent of the latent's producer:
before ``griddepcontrol.wait`` it only streams immutable weights into its own
shared memory and loads ``w_uv``; the latent, ``v``, the counters and ``out``
are touched only after the wait, which guarantees the producer grid has
completed and its writes are visible (the same contract as stream order).

Invocation ``e`` of a state buffer is ``finished_at_entry // grid``: all CTAs
of earlier invocations have exited (stream order) and fewer than ``grid`` CTAs
of this one, so the value lies in ``[e * grid, (e + 1) * grid)``. Its slices
are complete when ``done >= (e + 1) * K / 16`` (with ``pdl_entry``,
``finished`` is read after the wait). Both counters are monotonic 64-bit
integers and never reset. The grid is one CTA per SM and the kernel never
triggers its dependents early, so a waiting CTA waits only for CTAs that
either run or will run once concurrent, independent work leaves their SM. One
state buffer must not be used by two concurrently running invocations (one per
layer and one stream, as in the MLA adapter).
"""

import functools
from typing import Optional

import torch

from .cute_oproj import _TILE_M, _WARPS, _sm_count, plan

_SLICE = 16  # v columns per W_UV CTA
_COMPUTE_THREADS = _WARPS * 32
_THREADS = _COMPUTE_THREADS + 32  # plus one producer warp
_SPIN_LIMIT = 1 << 26
DEFAULT_WINDOW = 8
"""Rows (4 KiB each) of ``w_o`` a CTA keeps in flight."""


def check_fused_shapes(heads: int, v_dim: int, c_dim: int, n_dim: int, grid: int) -> Optional[str]:
    """Return why the fused kernel cannot take these shapes, or ``None``."""
    k_dim = heads * v_dim
    if v_dim % _SLICE or c_dim % (_WARPS * 16):
        return "W_UV shape not a multiple of the fused tiles"
    if k_dim // _SLICE > grid:
        return "more W_UV slices than CTAs"
    if k_dim % (_WARPS * 16):
        return "K is not a multiple of 128"
    p = plan(n_dim, k_dim, grid)
    if p["smem_bytes"] + _WARPS * 2 * 128 * 4 > 200 * 1024:
        return "weight slab does not fit in shared memory"
    return None


@functools.lru_cache(maxsize=None)
def _build_fused(
    heads: int,
    v_dim: int,
    c_dim: int,
    n_dim: int,
    grid: int,
    nt: int,
    traced: bool,
    window: int,
    pdl: bool,
):
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cute import runtime as cute_runtime
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.utils.smem_allocator import SmemAllocator

    from tensorrt_llm._torch.kernels.blackwell.cute_ptx_utils import mma_sync

    k_dim = heads * v_dim
    geo = plan(n_dim, k_dim, grid)
    rows_max = geo["rows_max"]
    num_tiles = geo["num_tiles"]
    pitch = geo["pitch"]
    half_pitch = pitch // 2
    k_per_warp = geo["k_per_warp"]
    steps = k_per_warp // 16
    row_bytes = k_dim * 2
    red_stride = 128
    num_slices = k_dim // _SLICE
    slices_per_head = v_dim // _SLICE
    c_per_warp = c_dim // _WARPS
    c_steps = c_per_warp // 16
    c_words = c_dim // 2
    k_words = k_dim // 2
    compute_bar = 1

    def _asm(res, operands, text, constraints, loc, ip):
        return llvm.inline_asm(
            res, operands, text, constraints, has_side_effects=True, asm_dialect=0, loc=loc, ip=ip
        )

    @dsl_user_op
    def bulk_g2s(dst_smem, src_gmem, num_bytes, mbar, *, loc=None, ip=None):
        _asm(
            None,
            [
                dst_smem.toint(loc=loc, ip=ip).ir_value(),
                src_gmem.toint(loc=loc, ip=ip).ir_value(),
                Int32(num_bytes).ir_value(loc=loc, ip=ip),
                mbar.toint(loc=loc, ip=ip).ir_value(),
            ],
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [$0], [$1], $2, [$3];",
            "r,l,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def red_add_u64(addr, val, *, loc=None, ip=None):
        _asm(
            None,
            [Int64(addr).ir_value(loc=loc, ip=ip), Int64(val).ir_value(loc=loc, ip=ip)],
            "red.relaxed.gpu.global.add.u64 [$0], $1;",
            "l,l",
            loc,
            ip,
        )

    @dsl_user_op
    def ld_relaxed_u64(addr, *, loc=None, ip=None):
        return Int64(
            _asm(
                T.i64(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.relaxed.gpu.global.u64 $0, [$1];",
                "=l,l",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def global_timer(*, loc=None, ip=None):
        return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc, ip))

    @dsl_user_op
    def ld_acquire_u64(addr, *, loc=None, ip=None):
        return Int64(
            _asm(
                T.i64(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.acquire.gpu.global.u64 $0, [$1];",
                "=l,l",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def ld_cg_b32(addr, *, loc=None, ip=None):
        return Int32(
            _asm(
                T.i32(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.global.cg.b32 $0, [$1];",
                "=r,l",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def grid_dependency_wait(*, loc=None, ip=None):
        _asm(None, [], "griddepcontrol.wait;", "", loc, ip)

    @dsl_user_op
    def fence_gpu(*, loc=None, ip=None):
        _asm(None, [], "fence.acq_rel.gpu;", "", loc, ip)

    @dsl_user_op
    def sleep_ns(*, loc=None, ip=None):
        _asm(None, [], "nanosleep.u32 32;", "", loc, ip)

    @dsl_user_op
    def trap(*, loc=None, ip=None):
        _asm(None, [], "trap;", "", loc, ip)

    @cute.jit
    def compute_sync():
        cute.arch.barrier(barrier_id=compute_bar, number_of_threads=_COMPUTE_THREADS)

    class FusedTail:
        @cute.kernel
        def kernel(
            self,
            latent: cute.Tensor,
            w_uv: cute.Tensor,
            w: cute.Tensor,
            v: cute.Tensor,
            out: cute.Tensor,
            state: cute.Tensor,
            trace: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            warp = tidx // Int32(32)
            lane = tidx % Int32(32)
            grp = lane // Int32(4)
            quad = lane % Int32(4)
            t_entry = Int64(0)
            if cutlass.const_expr(traced):
                t_entry = global_timer()

            num_rows = Int32(latent.shape[0])
            last_row = num_rows - Int32(1)
            out_rows = Int32(out.shape[0])
            n0 = (bidx * Int32(n_dim)) // Int32(grid)
            n1 = ((bidx + Int32(1)) * Int32(n_dim)) // Int32(grid)
            nrows = n1 - n0

            smem = SmemAllocator()
            slab = smem.allocate_tensor(
                BFloat16, cute.make_layout((rows_max * pitch,)), byte_alignment=128
            )
            red1 = smem.allocate_tensor(
                Float32, cute.make_layout((_WARPS * nt * red_stride,)), byte_alignment=16
            )
            row_bar = smem.allocate_tensor(Int64, cute.make_layout((rows_max,)), byte_alignment=8)
            slab32 = cute.recast_tensor(slab, Int32)
            slab_f32 = cute.recast_tensor(slab, Float32)
            lat32 = cute.recast_tensor(latent, Int32)
            wuv32 = cute.recast_tensor(w_uv, Int32)
            finished_addr = state.iterator.toint()
            done_addr = finished_addr + Int64(8)
            v_addr = v.iterator.toint()

            # Thread 0 arms one mbarrier per row (4 KiB of expected bytes)
            # and reads the invocation count.
            finished = Int64(0)
            if tidx == Int32(0):
                if cutlass.const_expr(not pdl):
                    finished = ld_relaxed_u64(finished_addr)
                for i in cutlass.range_constexpr(rows_max):
                    cute.arch.mbarrier_init(row_bar.iterator + i, 1)
                for i in cutlass.range_constexpr(rows_max):
                    if Int32(i) < nrows:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            row_bar.iterator + i, Int32(row_bytes)
                        )
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            t_issued = Int64(0)
            t_published = Int64(0)
            t_ready = Int64(0)
            t_tile0 = Int64(0)
            if warp == Int32(_WARPS):
                # Producer: at most `window` rows in flight. The whole warp
                # runs the loop so every mbarrier wait is warp-uniform; lane 0
                # issues the copies.
                i = Int32(0)
                while i < nrows:
                    if i >= Int32(window):
                        cute.arch.mbarrier_wait(row_bar.iterator + (i - Int32(window)), 0)
                    if lane == Int32(0):
                        bulk_g2s(
                            slab.iterator + i * Int32(pitch),
                            w.iterator + (n0 + i) * Int32(k_dim),
                            row_bytes,
                            row_bar.iterator + i,
                        )
                    i = i + Int32(1)
                if cutlass.const_expr(traced):
                    t_issued = global_timer()
            else:
                # Phase 1: CTA b < num_slices computes 16 columns of v.
                if bidx < Int32(num_slices):
                    head = bidx // Int32(slices_per_head)
                    col0 = (bidx % Int32(slices_per_head)) * Int32(_SLICE)
                    wrow_lo = head * Int32(v_dim) + col0 + grp
                    wrow_hi = wrow_lo + Int32(8)
                    c_word0 = warp * Int32(c_per_warp // 2) + quad
                    lat_word0 = head * Int32(c_words) + c_word0
                    wa = cute.make_rmem_tensor((8, c_steps), BFloat16)
                    wa32 = cute.recast_tensor(wa, Int32)
                    la = cute.make_rmem_tensor((4, nt, c_steps), BFloat16)
                    la32 = cute.recast_tensor(la, Int32)
                    for s in cutlass.range_constexpr(c_steps):
                        wa32[0, s] = wuv32[wrow_lo, c_word0 + Int32(8 * s)]
                        wa32[1, s] = wuv32[wrow_hi, c_word0 + Int32(8 * s)]
                        wa32[2, s] = wuv32[wrow_lo, c_word0 + Int32(8 * s + 4)]
                        wa32[3, s] = wuv32[wrow_hi, c_word0 + Int32(8 * s + 4)]
                    if cutlass.const_expr(pdl):
                        grid_dependency_wait()
                    for s in cutlass.range_constexpr(c_steps):
                        for j in cutlass.range_constexpr(nt):
                            r = Int32(8 * j) + grp
                            rc = r
                            if rc > last_row:
                                rc = last_row
                            lo = lat32[rc, lat_word0 + Int32(8 * s)]
                            hi = lat32[rc, lat_word0 + Int32(8 * s + 4)]
                            if r > last_row:
                                lo = Int32(0)
                                hi = Int32(0)
                            la32[0, j, s] = lo
                            la32[1, j, s] = hi
                    acc1 = cute.make_rmem_tensor((4, nt), Float32)
                    acc1.fill(0.0)
                    for s in cutlass.range_constexpr(c_steps):
                        for j in cutlass.range_constexpr(nt):
                            acc1[None, j] = mma_sync(wa[None, s], la[None, j, s], acc1[None, j])
                    for j in cutlass.range_constexpr(nt):
                        for e in cutlass.range_constexpr(4):
                            red1[
                                (warp * Int32(nt) + Int32(j)) * Int32(red_stride)
                                + lane * Int32(4)
                                + Int32(e)
                            ] = acc1[e, j]
                    compute_sync()
                    idx1 = tidx
                    while idx1 < Int32(nt * red_stride):
                        j_i = idx1 // Int32(red_stride)
                        l_i = (idx1 // Int32(4)) % Int32(32)
                        e_i = idx1 % Int32(4)
                        vsum = Float32(0.0)
                        for wi in cutlass.range_constexpr(_WARPS):
                            vsum = vsum + red1[Int32(wi * nt * red_stride) + idx1]
                        m = l_i // Int32(4) + Int32(8) * (e_i // Int32(2))
                        r = j_i * Int32(8) + Int32(2) * (l_i % Int32(4)) + (e_i % Int32(2))
                        if r < num_rows:
                            v[r, head * Int32(v_dim) + col0 + m] = vsum.to(BFloat16)
                        idx1 = idx1 + Int32(_COMPUTE_THREADS)
                    compute_sync()
                    if tidx == Int32(0):
                        fence_gpu()
                        red_add_u64(done_addr, Int64(1))
                        if cutlass.const_expr(traced):
                            t_published = global_timer()

                else:
                    if cutlass.const_expr(pdl):
                        grid_dependency_wait()

                # Phase 2: wait for all slices of this invocation.
                if tidx == Int32(0):
                    if cutlass.const_expr(pdl):
                        # After griddepcontrol.wait: every earlier invocation
                        # of this state has completed.
                        finished = ld_relaxed_u64(finished_addr)
                    target = (finished // Int64(grid) + Int64(1)) * Int64(num_slices)
                    spins = Int32(0)
                    seen = ld_acquire_u64(done_addr)
                    while seen < target:
                        sleep_ns()
                        spins = spins + Int32(1)
                        if spins > Int32(_SPIN_LIMIT):
                            trap()
                        seen = ld_acquire_u64(done_addr)
                compute_sync()
                if cutlass.const_expr(traced):
                    t_ready = global_timer()

                vb = cute.make_rmem_tensor((4, nt, steps), BFloat16)
                vb32 = cute.recast_tensor(vb, Int32)
                k_word0 = warp * Int32(k_per_warp // 2) + quad
                for s in cutlass.range_constexpr(steps):
                    for j in cutlass.range_constexpr(nt):
                        r = Int32(8 * j) + grp
                        rc = r
                        if rc > last_row:
                            rc = last_row
                        word = rc * Int32(k_words) + k_word0 + Int32(8 * s)
                        lo = ld_cg_b32(v_addr + Int64(word) * Int64(4))
                        hi = ld_cg_b32(v_addr + Int64(word + Int32(4)) * Int64(4))
                        if r > last_row:
                            lo = Int32(0)
                            hi = Int32(0)
                        vb32[0, j, s] = lo
                        vb32[1, j, s] = hi

                acc = cute.make_rmem_tensor((4, nt, num_tiles), Float32)
                acc.fill(0.0)
                a2 = cute.make_rmem_tensor((8,), BFloat16)
                a2_32 = cute.recast_tensor(a2, Int32)
                last = nrows - Int32(1)
                for t in cutlass.range_constexpr(num_tiles):
                    # Warp-uniform waits on every row of the tile.
                    for q in cutlass.range_constexpr(_TILE_M):
                        wrow = Int32(t * _TILE_M + q)
                        if wrow < nrows:
                            cute.arch.mbarrier_wait(row_bar.iterator + wrow, 0)
                    row_lo = Int32(t * _TILE_M) + grp
                    row_hi = row_lo + Int32(8)
                    if row_lo > last:
                        row_lo = last
                    if row_hi > last:
                        row_hi = last
                    if cutlass.const_expr(traced and t == 0):
                        t_tile0 = global_timer()
                    base_lo = row_lo * Int32(half_pitch) + k_word0
                    base_hi = row_hi * Int32(half_pitch) + k_word0
                    for s in cutlass.range_constexpr(steps):
                        a2_32[0] = slab32[base_lo + Int32(8 * s)]
                        a2_32[1] = slab32[base_hi + Int32(8 * s)]
                        a2_32[2] = slab32[base_lo + Int32(8 * s + 4)]
                        a2_32[3] = slab32[base_hi + Int32(8 * s + 4)]
                        for j in cutlass.range_constexpr(nt):
                            acc[None, j, t] = mma_sync(a2, vb[None, j, s], acc[None, j, t])

                # Fixed-order cross-warp reduction through the idle slab.
                compute_sync()
                for t in cutlass.range_constexpr(num_tiles):
                    for j in cutlass.range_constexpr(nt):
                        base = (
                            (warp * Int32(num_tiles) + Int32(t)) * Int32(nt) + Int32(j)
                        ) * Int32(red_stride) + (lane * Int32(4))
                        for e in cutlass.range_constexpr(4):
                            slab_f32[base + Int32(e)] = acc[e, j, t]
                compute_sync()

                total = num_tiles * nt * red_stride
                out_limit = out_rows
                if out_limit > Int32(8 * nt):
                    out_limit = Int32(8 * nt)
                idx = tidx
                while idx < Int32(total):
                    t_i = idx // Int32(nt * red_stride)
                    j_i = (idx // Int32(red_stride)) % Int32(nt)
                    l_i = (idx // Int32(4)) % Int32(32)
                    e_i = idx % Int32(4)
                    acc_sum = Float32(0.0)
                    for wi in cutlass.range_constexpr(_WARPS):
                        acc_sum = acc_sum + slab_f32[Int32(wi * num_tiles * nt * red_stride) + idx]
                    m = t_i * Int32(_TILE_M) + l_i // Int32(4) + Int32(8) * (e_i // Int32(2))
                    r = j_i * Int32(8) + Int32(2) * (l_i % Int32(4)) + (e_i % Int32(2))
                    if m < nrows:
                        if r < out_limit:
                            out[r, n0 + m] = acc_sum.to(BFloat16)
                    idx = idx + Int32(_COMPUTE_THREADS)
                compute_sync()
                if tidx == Int32(0):
                    red_add_u64(finished_addr, Int64(1))
                    if cutlass.const_expr(traced):
                        trace[bidx, 0] = t_entry
                        trace[bidx, 2] = t_published
                        trace[bidx, 3] = t_ready
                        trace[bidx, 4] = t_tile0
                        trace[bidx, 5] = global_timer()
            if cutlass.const_expr(traced):
                if warp == Int32(_WARPS):
                    if lane == Int32(0):
                        trace[bidx, 1] = t_issued

        @cute.jit
        def __call__(
            self,
            latent: cute.Tensor,
            w_uv: cute.Tensor,
            w: cute.Tensor,
            v: cute.Tensor,
            out: cute.Tensor,
            state: cute.Tensor,
            trace: cute.Tensor,
            stream,
        ):
            self.kernel(latent, w_uv, w, v, out, state, trace).launch(
                grid=(grid, 1, 1),
                block=(_THREADS, 1, 1),
                stream=stream,
                use_pdl=pdl,
                min_blocks_per_mp=1,
            )

    r_sym, v_sym, o_sym = cute.sym_int(), cute.sym_int(), cute.sym_int()

    def fake(dtype, shape):
        return cute_runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=tuple(range(len(shape) - 1, -1, -1)), assumed_align=16
        )

    return cute.compile(
        FusedTail(),
        fake(BFloat16, (r_sym, heads * c_dim)),
        fake(BFloat16, (heads * v_dim, c_dim)),
        fake(BFloat16, (n_dim, k_dim)),
        fake(BFloat16, (v_sym, k_dim)),
        fake(BFloat16, (o_sym, n_dim)),
        cute_runtime.make_fake_compact_tensor(Int64, (2,), stride_order=(0,), assumed_align=16),
        cute_runtime.make_fake_compact_tensor(
            Int64, (grid, 8), stride_order=(1, 0), assumed_align=16
        ),
        stream=cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def new_state(device: torch.device) -> torch.Tensor:
    """Allocate the zeroed ``[finished, done]`` counters of one tail instance."""
    return torch.zeros(2, dtype=torch.int64, device=device)


_NO_TRACE = {}


def fused_tail(
    latent: torch.Tensor,
    w_uv: torch.Tensor,
    w_o: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    v_scratch: Optional[torch.Tensor] = None,
    trace: Optional[torch.Tensor] = None,
    window: int = DEFAULT_WINDOW,
    pdl_entry: bool = False,
) -> None:
    """Launch the single-kernel tail on the current stream.

    Args:
        latent: ``[num_rows, H * C]`` contiguous BF16, ``num_rows <= 16``.
        w_uv: ``[H, V, C]`` contiguous BF16.
        w_o: ``[N, H * V]`` contiguous BF16, 16-byte aligned.
        out: ``[out_rows, N]`` contiguous BF16; rows ``[0, min(out_rows, 16))``
            are written (rows past ``num_rows`` as zeros).
        state: ``[2]`` int64 counters from ``new_state``, private to the
            caller and never used by two invocations at once.
        v_scratch: optional ``[num_rows, H * V]`` BF16 buffer for ``v``.
        trace: optional ``[grid, 8]`` int64 buffer; when given, a traced
            variant records ``%globaltimer`` at entry, after the last weight
            copy is issued, after the v slice is published, after the wait,
            when the first weight tile is ready, and at exit (columns 0-5).
        window: ``w_o`` rows a CTA keeps in flight.
        pdl_entry: launch as a PDL dependent of the latent's producer. The
            producer warp then streams ``w_o`` and W_UV CTAs load ``w_uv``
            before ``griddepcontrol.wait``; every latent read, every global
            write and the read of ``finished`` follow the wait.
    """
    heads, v_dim, c_dim = w_uv.shape
    n_dim, k_dim = w_o.shape
    num_rows = latent.shape[0]
    if v_scratch is None:
        v_scratch = torch.empty((num_rows, k_dim), dtype=w_o.dtype, device=w_o.device)
    grid = _sm_count(w_o.device.index or 0)
    nt = 1 if max(num_rows, out.shape[0]) <= 8 else 2
    traced = trace is not None
    if trace is None:
        key = (w_o.device.index or 0, grid)
        if key not in _NO_TRACE:
            _NO_TRACE[key] = torch.empty((grid, 8), dtype=torch.int64, device=w_o.device)
        trace = _NO_TRACE[key]
    compiled = _build_fused(heads, v_dim, c_dim, n_dim, grid, nt, traced, window, pdl_entry)
    compiled(latent, w_uv.view(heads * v_dim, c_dim), w_o, v_scratch, out, state, trace)


__all__ = ["DEFAULT_WINDOW", "check_fused_shapes", "fused_tail", "new_state"]

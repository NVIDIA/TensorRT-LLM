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
"""attention_output_tail streaming o_proj that also publishes into the post_attention_hop transport
(``attention_output_into_hop``).

The kernel of ``cute_oproj.stream_oproj`` (one CTA per SM streams its
``w_o`` rows into a padded shared-memory slab around its PDL wait, 8 K-split
warps run ``mma.sync``, fixed-order cross-warp reduction, one BF16
rounding), with two changes for ``attention_output_into_hop``:

* rows are split at even boundaries (40 or 42 per CTA on 152 SMs), so each
  thread of the epilogue owns two adjacent output columns of one row and
  stores them as one 32-bit word;
* besides the ordinary partial (the attention_output_tail contract stays available), the epilogue
  writes the words of every row of ``out`` (live rows and the zero rows
  past them, as the post_attention_hop sums them from the ordinary partial), with each
  bf16 ``-0.0`` mapped to ``+0.0`` (the Lamport sentinel can never appear),
  into slot ``[row][rank]`` of the current Lamport buffer of the post_attention_hop
  workspace through its multicast view. The hop polls exactly the rows of
  its residual and resets exactly those rows of the buffer later, so the
  caller passes an ``out`` with the hop's row count. The buffer index is read from the protocol state after
  ``griddepcontrol.wait``; the PDL successor is released only after that
  wait (producer side of the contract in
  ``cpp/tensorrt_llm/kernels/glmMoeDsa/postAttentionHopFromPartials.h``).
  When every CTA owns a multiple of 8 columns (e.g. 48 on 128 CTAs) the
  words are first gathered in shared memory and each row chunk of 8 columns
  goes out as one 16-byte multicast store, as the post_attention_hop's own publish does;
  otherwise every word is its own 32-bit multicast store.

The arithmetic of every output element (K split, MMA order, warp order,
rounding) is the same as ``stream_oproj``'s, so the ordinary partial equals
``stream_oproj``'s output bit for bit.
"""

import functools
import math
from typing import Optional

import torch

_THREADS = 256
_HIDDEN = 6144
_NUM_RANKS = 8
_LAMPORT_BUFFER_WORDS = 8 * _NUM_RANKS * _HIDDEN * 2 // 4
_WARPS = _THREADS // 32
_ROW_PAD = 8  # BF16 elements of padding per shared-memory row (16 bytes)
_TILE_M = 16


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def plan(n_dim: int, k_dim: int, grid: int) -> dict:
    """Compile-time geometry of the kernel for one weight shape."""
    rows_max = 2 * math.ceil(n_dim / 2 / grid)
    pitch = k_dim + _ROW_PAD
    return {
        "rows_max": rows_max,
        "num_tiles": math.ceil(rows_max / _TILE_M),
        "pitch": pitch,
        "smem_bytes": rows_max * pitch * 2,
        "k_per_warp": k_dim // _WARPS,
    }


def check_shapes(n_dim: int, k_dim: int, grid: int) -> Optional[str]:
    """Return why the kernel cannot take this ``[n_dim, k_dim]`` weight."""
    if k_dim % (_WARPS * 16):
        return "K is not a multiple of 128"
    p = plan(n_dim, k_dim, grid)
    red_bytes = _WARPS * p["num_tiles"] * 2 * 32 * 4 * 4
    if p["smem_bytes"] < red_bytes:
        return "slab smaller than the reduction buffer"
    # 227 KB of opt-in shared memory per CTA on SM100/SM103, minus the
    # tile mbarriers and alignment.
    if p["smem_bytes"] > 226 * 1024:
        return "weight slab does not fit in shared memory"
    if (k_dim * 2) % 16:
        return "row bytes not a 16-byte multiple"
    return None


def _build(
    n_dim: int,
    k_dim: int,
    grid: int,
    n_tiles_r: int,
    use_pdl: bool,
    early_rows: int,
    trigger_dependents: bool,
    trigger_late: bool,
    publish_rank: int,
    vector_publish: bool,
):
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cute import runtime as cute_runtime
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.utils.smem_allocator import SmemAllocator

    from tensorrt_llm._torch.kernels.blackwell.cute_ptx_utils import mma_sync

    geo = plan(n_dim, k_dim, grid)
    rows_max = geo["rows_max"]
    num_tiles = geo["num_tiles"]
    pitch = geo["pitch"]
    half_pitch = pitch // 2
    k_per_warp = geo["k_per_warp"]
    steps = k_per_warp // 16
    row_bytes = k_dim * 2
    nt = n_tiles_r
    red_stride = 128  # 32 lanes x 4 accumulators
    # 16-byte publish: every CTA owns the same whole number of 8-column
    # chunks; the words are staged right after the reduction buffer.
    vec_publish = vector_publish and n_dim % (8 * grid) == 0
    cols = n_dim // grid
    chunks = cols // 8
    stage_pitch = cols // 2
    stage0 = _WARPS * num_tiles * nt * red_stride
    if vec_publish:
        assert stage0 + 8 * stage_pitch <= rows_max * pitch // 2

    @dsl_user_op
    def bulk_g2s(dst_smem, src_gmem, num_bytes, mbar, *, loc=None, ip=None):
        llvm.inline_asm(
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
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def grid_launch_dependents(*, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [],
            "griddepcontrol.launch_dependents;",
            "",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def ld_volatile_u32(addr, *, loc=None, ip=None):
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.volatile.global.u32 $0, [$1];",
                "=r,l",
                has_side_effects=True,
                asm_dialect=0,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def st_global_b32(addr, val, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
            "st.global.b32 [$0], $1;",
            "l,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def st_global_v4_b32(addr, w0, w1, w2, w3, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(w0).ir_value(loc=loc, ip=ip),
                Int32(w1).ir_value(loc=loc, ip=ip),
                Int32(w2).ir_value(loc=loc, ip=ip),
                Int32(w3).ir_value(loc=loc, ip=ip),
            ],
            "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
            "l,r,r,r,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def grid_dependency_wait(*, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [],
            "griddepcontrol.wait;",
            "",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    class StreamOprojIntoHop:
        @cute.kernel
        def kernel(
            self,
            v: cute.Tensor,
            w: cute.Tensor,
            out: cute.Tensor,
            mc: cute.Tensor,
            flags: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            warp = tidx // Int32(32)
            lane = tidx % Int32(32)
            grp = lane // Int32(4)
            quad = lane % Int32(4)

            num_rows = Int32(v.shape[0])
            out_rows = Int32(out.shape[0])
            # Even row boundaries: each epilogue thread stores a 32-bit pair.
            n0 = Int32(2) * ((bidx * Int32(n_dim // 2)) // Int32(grid))
            n1 = Int32(2) * (((bidx + Int32(1)) * Int32(n_dim // 2)) // Int32(grid))
            nrows = n1 - n0

            smem = SmemAllocator()
            slab = smem.allocate_tensor(
                BFloat16, cute.make_layout((rows_max * pitch,)), byte_alignment=128
            )
            mbar = smem.allocate_tensor(
                cutlass.Int64, cute.make_layout((num_tiles,)), byte_alignment=8
            )
            slab32 = cute.recast_tensor(slab, Int32)
            slab_f32 = cute.recast_tensor(slab, Float32)
            v32 = cute.recast_tensor(v, Int32)

            if tidx == Int32(0):
                for t in cutlass.range_constexpr(num_tiles):
                    cute.arch.mbarrier_init(mbar.iterator + t, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            # Stream this CTA's rows of w into shared memory before waiting
            # for the producer of v: lane 0 arms the tile mbarriers, then
            # warp 0 issues one 1-D bulk copy per row.
            if tidx == Int32(0):
                for t in cutlass.range_constexpr(num_tiles):
                    tile_rows = nrows - Int32(t * _TILE_M)
                    if tile_rows > Int32(_TILE_M):
                        tile_rows = Int32(_TILE_M)
                    if tile_rows < Int32(0):
                        tile_rows = Int32(0)
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar.iterator + t, tile_rows * Int32(row_bytes)
                    )
            cute.arch.sync_warp()
            # Rows [0, early_rows) go out before the wait on v's producer and
            # the rest after it: a bounded early stream overlaps the producer
            # without queueing ~24 MiB of copies in front of its traffic.
            if warp == Int32(0):
                for q in cutlass.range_constexpr((rows_max + 31) // 32):
                    row = lane + Int32(32 * q)
                    if row < nrows:
                        if row < Int32(early_rows):
                            bulk_g2s(
                                slab.iterator + row * Int32(pitch),
                                w.iterator + (n0 + row) * Int32(k_dim),
                                row_bytes,
                                mbar.iterator + row // Int32(_TILE_M),
                            )

            if cutlass.const_expr(use_pdl):
                grid_dependency_wait()

            if warp == Int32(0):
                for q in cutlass.range_constexpr((rows_max + 31) // 32):
                    row = lane + Int32(32 * q)
                    if row < nrows:
                        if row >= Int32(early_rows):
                            bulk_g2s(
                                slab.iterator + row * Int32(pitch),
                                w.iterator + (n0 + row) * Int32(k_dim),
                                row_bytes,
                                mbar.iterator + row // Int32(_TILE_M),
                            )
            if cutlass.const_expr(trigger_dependents and not trigger_late):
                # Dependents still wait for this whole grid (and its memory)
                # before they read anything it produces.
                grid_launch_dependents()

            # v fragments of this warp's K range (B operand, [n][k] storage).
            vb = cute.make_rmem_tensor((4, nt, steps), BFloat16)
            vb32 = cute.recast_tensor(vb, Int32)
            k_word0 = warp * Int32(k_per_warp // 2) + quad
            for s in cutlass.range_constexpr(steps):
                for j in cutlass.range_constexpr(nt):
                    r = Int32(8 * j) + grp
                    lo = Int32(0)
                    hi = Int32(0)
                    if r < num_rows:
                        lo = v32[r, k_word0 + Int32(8 * s)]
                        hi = v32[r, k_word0 + Int32(8 * s + 4)]
                    vb32[0, j, s] = lo
                    vb32[1, j, s] = hi

            acc = cute.make_rmem_tensor((4, nt, num_tiles), Float32)
            acc.fill(0.0)
            a = cute.make_rmem_tensor((8,), BFloat16)
            a32 = cute.recast_tensor(a, Int32)
            last = nrows - Int32(1)
            for t in cutlass.range_constexpr(num_tiles):
                cute.arch.mbarrier_wait(mbar.iterator + t, 0)
                row_lo = Int32(t * _TILE_M) + grp
                row_hi = row_lo + Int32(8)
                if row_lo > last:
                    row_lo = last
                if row_hi > last:
                    row_hi = last
                base_lo = row_lo * Int32(half_pitch) + k_word0
                base_hi = row_hi * Int32(half_pitch) + k_word0
                for s in cutlass.range_constexpr(steps):
                    a32[0] = slab32[base_lo + Int32(8 * s)]
                    a32[1] = slab32[base_hi + Int32(8 * s)]
                    a32[2] = slab32[base_lo + Int32(8 * s + 4)]
                    a32[3] = slab32[base_hi + Int32(8 * s + 4)]
                    for j in cutlass.range_constexpr(nt):
                        acc[None, j, t] = mma_sync(a, vb[None, j, s], acc[None, j, t])

            if cutlass.const_expr(trigger_dependents and trigger_late):
                # Late trigger: only once this CTA's MMAs are done.
                grid_launch_dependents()
            # Fixed-order cross-warp reduction through the (now idle) slab.
            cute.arch.sync_threads()
            for t in cutlass.range_constexpr(num_tiles):
                for j in cutlass.range_constexpr(nt):
                    base = ((warp * Int32(num_tiles) + Int32(t)) * Int32(nt) + Int32(j)) * Int32(
                        red_stride
                    ) + lane * Int32(4)
                    for e in cutlass.range_constexpr(4):
                        slab_f32[base + Int32(e)] = acc[e, j, t]
            cute.arch.sync_threads()

            out_limit = out_rows
            if out_limit > Int32(8 * nt):
                out_limit = Int32(8 * nt)
            # After the wait: the protocol state written by the previous hop
            # is final.
            buf_word0 = ld_volatile_u32(flags.iterator.toint()) * Int32(_LAMPORT_BUFFER_WORDS)
            mc_base = mc.iterator.toint()
            pk = cute.make_rmem_tensor((2,), BFloat16)
            pk32 = cute.recast_tensor(pk, Int32)
            total_pairs = num_tiles * nt * 64
            p = tidx
            while p < Int32(total_pairs):
                t_i = p // Int32(nt * 64)
                j_i = (p // Int32(64)) % Int32(nt)
                q = p % Int32(64)
                gh = q // Int32(16)
                quad_i = (q // Int32(4)) % Int32(4)
                e_i = q % Int32(4)
                idx0 = (
                    (t_i * Int32(nt) + j_i) * Int32(red_stride)
                    + (gh * Int32(8) + quad_i) * Int32(4)
                    + e_i
                )
                idx1 = idx0 + Int32(16)
                s0 = Float32(0.0)
                s1 = Float32(0.0)
                for wi in cutlass.range_constexpr(_WARPS):
                    s0 = s0 + slab_f32[Int32(wi * num_tiles * nt * red_stride) + idx0]
                    s1 = s1 + slab_f32[Int32(wi * num_tiles * nt * red_stride) + idx1]
                m0 = t_i * Int32(_TILE_M) + gh * Int32(2) + Int32(8) * (e_i // Int32(2))
                r = j_i * Int32(8) + Int32(2) * quad_i + e_i % Int32(2)
                if m0 < nrows:
                    if r < out_limit:
                        pk[0] = s0.to(BFloat16)
                        pk[1] = s1.to(BFloat16)
                        out[r, n0 + m0] = pk[0]
                        out[r, n0 + m0 + Int32(1)] = pk[1]
                        # Every row of out is published, the zero rows past
                        # the live ones included: the hop polls all rows of
                        # its residual (out_rows of them).
                        word = pk32[0]
                        lo = word & Int32(0xFFFF)
                        hi = (word >> Int32(16)) & Int32(0xFFFF)
                        if lo == Int32(0x8000):
                            lo = Int32(0)
                        if hi == Int32(0x8000):
                            hi = Int32(0)
                        word = lo | (hi << Int32(16))
                        if cutlass.const_expr(vec_publish):
                            slab32[Int32(stage0) + r * Int32(stage_pitch) + m0 // Int32(2)] = word
                        else:
                            widx = buf_word0 + (
                                (r * Int32(_NUM_RANKS) + Int32(publish_rank)) * Int32(_HIDDEN)
                                + n0
                                + m0
                            ) // Int32(2)
                            st_global_b32(mc_base + Int64(widx) * Int64(4), word)
                p = p + Int32(_THREADS)
            if cutlass.const_expr(vec_publish):
                cute.arch.sync_threads()
                if tidx < out_limit * Int32(chunks):
                    pr = tidx // Int32(chunks)
                    pc = tidx % Int32(chunks)
                    sb = Int32(stage0) + pr * Int32(stage_pitch) + pc * Int32(4)
                    vidx = buf_word0 + (
                        (pr * Int32(_NUM_RANKS) + Int32(publish_rank)) * Int32(_HIDDEN)
                        + n0
                        + pc * Int32(8)
                    ) // Int32(2)
                    st_global_v4_b32(
                        mc_base + Int64(vidx) * Int64(4),
                        slab32[sb],
                        slab32[sb + Int32(1)],
                        slab32[sb + Int32(2)],
                        slab32[sb + Int32(3)],
                    )

        @cute.jit
        def __call__(
            self,
            v: cute.Tensor,
            w: cute.Tensor,
            out: cute.Tensor,
            mc: cute.Tensor,
            flags: cute.Tensor,
            stream,
        ):
            self.kernel(v, w, out, mc, flags).launch(
                grid=(grid, 1, 1),
                block=(_THREADS, 1, 1),
                stream=stream,
                use_pdl=use_pdl,
                min_blocks_per_mp=1,
            )

    rows_sym, out_sym = cute.sym_int(), cute.sym_int()
    v_fake = cute_runtime.make_fake_compact_tensor(
        BFloat16, (rows_sym, k_dim), stride_order=(1, 0), assumed_align=16
    )
    w_fake = cute_runtime.make_fake_compact_tensor(
        BFloat16, (n_dim, k_dim), stride_order=(1, 0), assumed_align=16
    )
    out_fake = cute_runtime.make_fake_compact_tensor(
        BFloat16, (out_sym, n_dim), stride_order=(1, 0), assumed_align=16
    )
    mc_fake = cute_runtime.make_fake_compact_tensor(
        Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=16
    )
    flags_fake = cute_runtime.make_fake_compact_tensor(
        Int32, (4,), stride_order=(0,), assumed_align=16
    )
    stream = cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        StreamOprojIntoHop(),
        v_fake,
        w_fake,
        out_fake,
        mc_fake,
        flags_fake,
        stream=stream,
        options="--enable-tvm-ffi",
    )


_COMPILED = {}


DEFAULT_EARLY_ROWS = 16


def stream_oproj_into_hop(
    v: torch.Tensor,
    w_o: torch.Tensor,
    out: torch.Tensor,
    use_pdl: bool,
    mc_words: torch.Tensor,
    flags: torch.Tensor,
    rank: int,
    grid: Optional[int] = None,
    early_rows: int = DEFAULT_EARLY_ROWS,
    trigger_dependents: bool = False,
    vector_publish: bool = True,
) -> None:
    """Launch the publishing streaming o_proj on the current stream.

    Arguments as ``cute_oproj.stream_oproj``, plus the post_attention_hop workspace's
    multicast int32 view ``mc_words``, its protocol state ``flags`` and this
    rank. Every row of ``out`` is published (rows past ``v``'s live rows as
    zeros), so ``out`` must have exactly the row count of the hop that
    consumes the transport, at most 8 (the transport's row capacity).
    ``vector_publish`` allows 16-byte publish stores where the column split
    permits them.
    """
    n_dim, k_dim = w_o.shape
    if grid is None:
        grid = _sm_count(w_o.device.index or 0)
    if not 1 <= v.shape[0] <= out.shape[0] <= 8 or n_dim != _HIDDEN:
        raise ValueError(
            "the post_attention_hop transport holds 1..8 rows of 6144 and out must hold "
            f"every live row; got v rows {v.shape[0]}, out rows "
            f"{out.shape[0]}, N {n_dim}"
        )
    n_tiles_r = 1 if out.shape[0] <= 8 else 2
    if not use_pdl:
        early_rows = 1 << 30
    key = (
        n_dim,
        k_dim,
        grid,
        n_tiles_r,
        use_pdl,
        early_rows,
        trigger_dependents,
        rank,
        vector_publish,
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = _build(
            n_dim,
            k_dim,
            grid,
            n_tiles_r,
            use_pdl,
            early_rows,
            trigger_dependents,
            False,
            rank,
            vector_publish,
        )
        _COMPILED[key] = compiled
    compiled(v, w_o, out, mc_words, flags)


__all__ = ["check_shapes", "plan", "stream_oproj_into_hop"]

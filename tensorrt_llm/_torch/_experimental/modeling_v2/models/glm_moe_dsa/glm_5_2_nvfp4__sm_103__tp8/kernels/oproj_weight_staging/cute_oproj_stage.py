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
"""oproj_weight_staging ``oproj_weight_staging``: the served
attention_output_tail/attention_output_into_hop streaming o_proj with after-wait row staging.

A fork of ``attention_output_tail/into_hop.py`` (the o_proj of ``INTO_HOP_SPARSE_DECODE_SCHEDULE``:
one CTA per row block streams its ``w_o`` rows into a padded shared-memory slab
around its PDL wait, 8 K-split warps run ``mma.sync``, fixed-order
cross-warp reduction, one BF16 rounding; optionally the attention_output_into_hop publish into
the post_attention_hop transport). The arithmetic, the row split (even boundaries), the
epilogue and the publish are unchanged, so ``out`` and the published words
equal ``cute_oproj_into_hop.stream_oproj_into_hop``'s bit for bit, and with
``publish=False`` ``out`` equals ``cute_oproj.stream_oproj``'s.

What changes is when the rows that the served schedule copies only after
``griddepcontrol.wait`` (rows ``[early_rows, nrows)`` of each CTA) leave
DRAM:

* ``stage="l2"``: rows ``[early_rows, early_rows + stage_rows)`` are
  prefetched into L2 (``cp.async.bulk.prefetch.L2``, policy
  ``prefetch_policy``) before the wait. They are still copied into shared
  memory after the wait, exactly as served, so correctness never depends on
  the prefetch.
* ``stage="smem"``: those rows are copied into the slab before the wait
  (the same ``cp.async.bulk`` + tile mbarrier path as the early rows).

``when`` places the staging: ``"entry"`` right after the early copies (by
the copy warp), ``"landed"`` by a second warp once every tile that holds
only early rows has landed, so the staging traffic never queues behind (or
in front of) the early stream; ``"chain"`` (``stage="smem"`` only) by the
second warp one tile at a time, each tile's staged rows once the previous
tile has landed. ``copy_policy="evict_first"`` adds an L2
``evict_first`` cache hint to every ``w_o`` row copy (the consumer read of a
single-use stream). Rows below ``early_rows`` are never staged again.

Memory order: before the wait only immutable ``w_o``
rows are read (copies into the CTA's own shared memory, L2 prefetches); every
read of ``v`` and of the into-hop protocol state and every global write follows
the wait; the only new wait is a warp waiting for copies its own CTA issued.

``trace=True`` (tests and diagnostics only) records ``%globaltimer`` per CTA
at entry, after the wait, after warp 0's MMA loop and at exit.
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
_STAGE_WARP = 1
TRACE_POINTS = 4

STAGES = ("none", "l2", "smem")
WHENS = ("entry", "landed", "chain")
PREFETCH_POLICIES = ("normal", "evict_last")
COPY_POLICIES = ("none", "evict_first")


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def plan(n_dim: int, k_dim: int, grid: int) -> dict:
    """Compile-time geometry of the kernel for one weight shape (as
    ``cute_oproj_into_hop.plan``)."""
    rows_max = 2 * math.ceil(n_dim / 2 / grid)
    pitch = k_dim + _ROW_PAD
    return {
        "rows_max": rows_max,
        "rows_min": 2 * ((n_dim // 2) // grid),
        "num_tiles": math.ceil(rows_max / _TILE_M),
        "pitch": pitch,
        "smem_bytes": rows_max * pitch * 2,
        "k_per_warp": k_dim // _WARPS,
    }


def check_shapes(n_dim: int, k_dim: int, grid: int) -> Optional[str]:
    """Return why the kernel cannot take this ``[n_dim, k_dim]`` weight."""
    if k_dim % (_WARPS * 16):
        return "K is not a multiple of 128"
    if n_dim % 2:
        return "N is odd"
    p = plan(n_dim, k_dim, grid)
    red_bytes = _WARPS * p["num_tiles"] * 2 * 32 * 4 * 4
    if p["smem_bytes"] < red_bytes:
        return "slab smaller than the reduction buffer"
    if p["smem_bytes"] > 226 * 1024:
        return "weight slab does not fit in shared memory"
    if (k_dim * 2) % 16:
        return "row bytes not a 16-byte multiple"
    return None


def check_stage(stage: str, when: str, prefetch_policy: str, copy_policy: str) -> Optional[str]:
    """Return why a staging configuration is invalid, or ``None``."""
    if stage not in STAGES:
        return f"stage {stage!r} not in {STAGES}"
    if when not in WHENS:
        return f"when {when!r} not in {WHENS}"
    if prefetch_policy not in PREFETCH_POLICIES:
        return f"prefetch_policy {prefetch_policy!r} not in {PREFETCH_POLICIES}"
    if copy_policy not in COPY_POLICIES:
        return f"copy_policy {copy_policy!r} not in {COPY_POLICIES}"
    if when == "chain" and stage == "l2":
        return "chain pacing waits for staged rows to land: smem staging only"
    return None


def _build(
    n_dim: int,
    k_dim: int,
    grid: int,
    n_tiles_r: int,
    use_pdl: bool,
    early_rows: int,
    trigger_dependents: bool,
    publish: bool,
    publish_rank: int,
    vector_publish: bool,
    stage: str,
    when: str,
    stage_rows: int,
    prefetch_policy: str,
    copy_policy: str,
    trace: bool,
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
    vec_publish = publish and vector_publish and n_dim % (8 * grid) == 0
    cols = n_dim // grid
    chunks = cols // 8
    stage_pitch = cols // 2
    stage0 = _WARPS * num_tiles * nt * red_stride
    if vec_publish:
        assert stage0 + 8 * stage_pitch <= rows_max * pitch // 2
    # Rows [early_rows, stage_end) of every CTA are staged before the wait.
    staging = use_pdl and stage != "none" and stage_rows > 0
    stage_end = early_rows + stage_rows if staging else early_rows
    # Tiles that hold only early rows in every CTA: their mbarriers complete
    # without any copy issued after the wait.
    early_tiles = min(early_rows, geo["rows_min"]) // _TILE_M
    copy_hint = copy_policy == "evict_first"
    pf_hint = prefetch_policy == "evict_last"
    landed = staging and when == "landed"
    chained = staging and when == "chain"
    # chain: tiles holding staged rows, each issued once the previous landed.
    first_tile = min(early_rows, rows_max) // _TILE_M
    last_tile = min(stage_end - 1, rows_max - 1) // _TILE_M

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
    def bulk_g2s_evict_first(dst_smem, src_gmem, num_bytes, mbar, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [
                dst_smem.toint(loc=loc, ip=ip).ir_value(),
                src_gmem.toint(loc=loc, ip=ip).ir_value(),
                Int32(num_bytes).ir_value(loc=loc, ip=ip),
                mbar.toint(loc=loc, ip=ip).ir_value(),
            ],
            "{\n\t.reg .b64 pol;\n\t"
            "createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n\t"
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
            ".L2::cache_hint [$0], [$1], $2, [$3], pol;\n\t}",
            "r,l,r,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def prefetch_l2(src_gmem, num_bytes, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [
                src_gmem.toint(loc=loc, ip=ip).ir_value(),
                Int32(num_bytes).ir_value(loc=loc, ip=ip),
            ],
            "cp.async.bulk.prefetch.L2.global [$0], $1;",
            "l,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def prefetch_l2_evict_last(src_gmem, num_bytes, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [
                src_gmem.toint(loc=loc, ip=ip).ir_value(),
                Int32(num_bytes).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .b64 pol;\n\t"
            "createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n\t"
            "cp.async.bulk.prefetch.L2.global.L2::cache_hint [$0], $1, pol;"
            "\n\t}",
            "l,r",
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
    def st_global_b64(addr, val, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [Int64(addr).ir_value(loc=loc, ip=ip), Int64(val).ir_value(loc=loc, ip=ip)],
            "st.global.b64 [$0], $1;",
            "l,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def globaltimer(*, loc=None, ip=None):
        return Int64(
            llvm.inline_asm(
                T.i64(),
                [],
                "mov.u64 $0, %globaltimer;",
                "=l",
                has_side_effects=True,
                asm_dialect=0,
                loc=loc,
                ip=ip,
            )
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

    def copy_row(slab, w, mbar, n0, row):
        """``cp.async.bulk`` of ``w_o`` row ``n0 + row`` into the slab
        (traced inline; the branch is resolved at compile time)."""
        if copy_hint:
            bulk_g2s_evict_first(
                slab.iterator + row * Int32(pitch),
                w.iterator + (n0 + row) * Int32(k_dim),
                row_bytes,
                mbar.iterator + row // Int32(_TILE_M),
            )
        else:
            bulk_g2s(
                slab.iterator + row * Int32(pitch),
                w.iterator + (n0 + row) * Int32(k_dim),
                row_bytes,
                mbar.iterator + row // Int32(_TILE_M),
            )

    def stage_row(slab, w, mbar, n0, row):
        """Stage after-wait row ``row`` before the wait (L2 or smem)."""
        if stage == "smem":
            copy_row(slab, w, mbar, n0, row)
        elif pf_hint:
            prefetch_l2_evict_last(w.iterator + (n0 + row) * Int32(k_dim), row_bytes)
        else:
            prefetch_l2(w.iterator + (n0 + row) * Int32(k_dim), row_bytes)

    def stamp_addr(tr, bidx, k):
        return tr.iterator.toint() + Int64(bidx * Int32(TRACE_POINTS) + Int32(k)) * Int64(8)

    class StreamOprojStage:
        @cute.kernel
        def kernel(
            self,
            v: cute.Tensor,
            w: cute.Tensor,
            out: cute.Tensor,
            mc: cute.Tensor,
            flags: cute.Tensor,
            tr: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            warp = tidx // Int32(32)
            lane = tidx % Int32(32)
            grp = lane // Int32(4)
            quad = lane % Int32(4)
            if cutlass.const_expr(trace):
                if tidx == Int32(0):
                    st_global_b64(stamp_addr(tr, bidx, 0), globaltimer())

            num_rows = Int32(v.shape[0])
            out_rows = Int32(out.shape[0])
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

            # Lane 0 arms every tile mbarrier with all of its rows' bytes,
            # whichever warp and phase issues the copies.
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
            # Served: rows [0, early_rows) before the wait on v's producer.
            if warp == Int32(0):
                for q in cutlass.range_constexpr((rows_max + 31) // 32):
                    row = lane + Int32(32 * q)
                    if row < nrows:
                        if row < Int32(early_rows):
                            copy_row(slab, w, mbar, n0, row)
                if cutlass.const_expr(staging and when == "entry"):
                    # oproj_weight_staging, entry: stage the after-wait rows right after the
                    # early copies.
                    for q in cutlass.range_constexpr((rows_max + 31) // 32):
                        row = lane + Int32(32 * q)
                        if row < nrows:
                            if row >= Int32(early_rows):
                                if row < Int32(stage_end):
                                    stage_row(slab, w, mbar, n0, row)
            if cutlass.const_expr(landed):
                # oproj_weight_staging, landed: a second warp stages the after-wait rows once
                # the early-only tiles are in shared memory (copies this CTA
                # issued above; no other CTA or kernel is waited on).
                if warp == Int32(_STAGE_WARP):
                    for t in cutlass.range_constexpr(early_tiles):
                        cute.arch.mbarrier_wait(mbar.iterator + t, 0)
                    for q in cutlass.range_constexpr((rows_max + 31) // 32):
                        row = lane + Int32(32 * q)
                        if row < nrows:
                            if row >= Int32(early_rows):
                                if row < Int32(stage_end):
                                    stage_row(slab, w, mbar, n0, row)
            if cutlass.const_expr(chained):
                # oproj_weight_staging, chain: the second warp issues the staged rows of tile t
                # once tile t - 1 has landed (copies its own CTA issued).
                if warp == Int32(_STAGE_WARP):
                    for i in cutlass.range_constexpr(last_tile + 1 - first_tile):
                        if cutlass.const_expr(first_tile + i >= 1):
                            cute.arch.mbarrier_wait(mbar.iterator + (first_tile + i - 1), 0)
                        row = Int32((first_tile + i) * _TILE_M) + lane
                        if lane < Int32(_TILE_M):
                            if row < nrows:
                                if row >= Int32(early_rows):
                                    if row < Int32(stage_end):
                                        stage_row(slab, w, mbar, n0, row)

            if cutlass.const_expr(use_pdl):
                grid_dependency_wait()
            if cutlass.const_expr(trace):
                if tidx == Int32(0):
                    st_global_b64(stamp_addr(tr, bidx, 1), globaltimer())

            # Rows not yet copied into shared memory go out after the wait
            # (with stage="l2" these copies read the prefetched lines).
            if warp == Int32(0):
                for q in cutlass.range_constexpr((rows_max + 31) // 32):
                    row = lane + Int32(32 * q)
                    if row < nrows:
                        if cutlass.const_expr(staging and stage == "smem"):
                            if row >= Int32(stage_end):
                                copy_row(slab, w, mbar, n0, row)
                        else:
                            if row >= Int32(early_rows):
                                copy_row(slab, w, mbar, n0, row)
            if cutlass.const_expr(trigger_dependents):
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
            if cutlass.const_expr(trace):
                if tidx == Int32(0):
                    st_global_b64(stamp_addr(tr, bidx, 2), globaltimer())

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
            buf_word0 = Int32(0)
            mc_base = Int64(0)
            if cutlass.const_expr(publish):
                # After the wait: the protocol state written by the previous
                # hop is final.
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
                        if cutlass.const_expr(publish):
                            word = pk32[0]
                            lo = word & Int32(0xFFFF)
                            hi = (word >> Int32(16)) & Int32(0xFFFF)
                            if lo == Int32(0x8000):
                                lo = Int32(0)
                            if hi == Int32(0x8000):
                                hi = Int32(0)
                            word = lo | (hi << Int32(16))
                            if cutlass.const_expr(vec_publish):
                                slab32[Int32(stage0) + r * Int32(stage_pitch) + m0 // Int32(2)] = (
                                    word
                                )
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
            if cutlass.const_expr(trace):
                if tidx == Int32(0):
                    st_global_b64(stamp_addr(tr, bidx, 3), globaltimer())

        @cute.jit
        def __call__(
            self,
            v: cute.Tensor,
            w: cute.Tensor,
            out: cute.Tensor,
            mc: cute.Tensor,
            flags: cute.Tensor,
            tr: cute.Tensor,
            stream,
        ):
            self.kernel(v, w, out, mc, flags, tr).launch(
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
    tr_fake = cute_runtime.make_fake_compact_tensor(
        Int64, (cute.sym_int(),), stride_order=(0,), assumed_align=16
    )
    stream = cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        StreamOprojStage(),
        v_fake,
        w_fake,
        out_fake,
        mc_fake,
        flags_fake,
        tr_fake,
        stream=stream,
        options="--enable-tvm-ffi",
    )


_COMPILED = {}
_DUMMIES = {}


def _dummies(device: torch.device):
    """Placeholder publish / trace tensors for variants that do not use them
    (never read or written by those variants)."""
    key = device.index or 0
    if key not in _DUMMIES:
        _DUMMIES[key] = (
            torch.zeros(4, dtype=torch.int32, device=device),
            torch.zeros(4, dtype=torch.int32, device=device),
            torch.zeros(TRACE_POINTS, dtype=torch.int64, device=device),
        )
    return _DUMMIES[key]


def stream_oproj_stage(
    v: torch.Tensor,
    w_o: torch.Tensor,
    out: torch.Tensor,
    use_pdl: bool,
    early_rows: int,
    grid: Optional[int] = None,
    trigger_dependents: bool = False,
    stage: str = "l2",
    when: str = "landed",
    stage_rows: int = -1,
    prefetch_policy: str = "normal",
    copy_policy: str = "none",
    hop_transport=None,
    vector_publish: bool = True,
    trace: Optional[torch.Tensor] = None,
) -> None:
    """Launch the staging o_proj on the current stream.

    Args:
        v: ``[num_rows, K]`` contiguous BF16 (``num_rows <= 16``).
        w_o: ``[N, K]`` contiguous BF16, 16-byte aligned.
        out: ``[out_rows, N]`` contiguous BF16; rows past ``num_rows`` are
            written as zeros (at most 16 rows written).
        use_pdl: launch as a PDL dependent of the previous kernel.
        early_rows: rows per CTA copied before ``griddepcontrol.wait``
            (served: ``TailConfig.early_rows``).
        grid: CTA count (default: the device's SM count).
        trigger_dependents: execute ``griddepcontrol.launch_dependents`` once
            every row copy is issued.
        stage, when, stage_rows, prefetch_policy, copy_policy: oproj_weight_staging
            (module docstring); ``stage_rows=-1`` stages every after-wait
            row.
        hop_transport: the ``IntoHopTransport`` of ``attention_output_into_hop``; ``out``
            (1..8 rows, the hop's rows) is then also published into it.
        vector_publish: 16-byte publish stores where the split permits.
        trace: optional int64 ``[grid * TRACE_POINTS]`` buffer for
            ``%globaltimer`` stamps (diagnostics only).
    """
    n_dim, k_dim = w_o.shape
    if grid is None:
        grid = _sm_count(w_o.device.index or 0)
    reason = check_stage(stage, when, prefetch_policy, copy_policy)
    if reason is not None:
        raise ValueError(reason)
    publish = hop_transport is not None
    if publish:
        if not 1 <= v.shape[0] <= out.shape[0] <= 8 or n_dim != _HIDDEN:
            raise ValueError(
                "the post_attention_hop transport holds 1..8 rows of 6144 and out must hold "
                f"every live row; got v rows {v.shape[0]}, out rows "
                f"{out.shape[0]}, N {n_dim}"
            )
    n_tiles_r = 1 if out.shape[0] <= 8 else 2
    if not use_pdl:
        early_rows = 1 << 30
    rows_max = plan(n_dim, k_dim, grid)["rows_max"]
    late = max(0, rows_max - min(early_rows, rows_max))
    if stage_rows < 0 or stage_rows > late:
        stage_rows = late
    if stage == "none":
        stage_rows = 0
    rank = hop_transport.rank if publish else 0
    key = (
        n_dim,
        k_dim,
        grid,
        n_tiles_r,
        use_pdl,
        early_rows,
        trigger_dependents,
        publish,
        rank,
        vector_publish and publish,
        stage,
        when,
        stage_rows,
        prefetch_policy,
        copy_policy,
        trace is not None,
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
            publish,
            rank,
            vector_publish and publish,
            stage,
            when,
            stage_rows,
            prefetch_policy,
            copy_policy,
            trace is not None,
        )
        _COMPILED[key] = compiled
    mc_dummy, flags_dummy, tr_dummy = _dummies(w_o.device)
    mc = hop_transport.mc_words if publish else mc_dummy
    flags = hop_transport.flags if publish else flags_dummy
    compiled(v, w_o, out, mc, flags, trace if trace is not None else tr_dummy)


__all__ = [
    "COPY_POLICIES",
    "PREFETCH_POLICIES",
    "STAGES",
    "TRACE_POINTS",
    "WHENS",
    "check_shapes",
    "check_stage",
    "plan",
    "stream_oproj_stage",
]

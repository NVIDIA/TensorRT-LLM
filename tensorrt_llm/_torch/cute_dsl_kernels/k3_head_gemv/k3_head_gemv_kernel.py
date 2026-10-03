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
# =============================================================================
# Kimi K3 vocab-shard head GEMV -- CTM (prims/cute) kernel, M <= 8 tokens, bf16 in/out, persistent
# =============================================================================
#
#   y[t, n] = sum_k x[t, k] * W[n, k]          (fp32 accumulation in TMEM, one bf16 rounding)
#     A = W  (N, K) K-major bf16                (the streamed weight: an lm_head shard, N ~ 10^4)
#     B = x  (M <= 8, K) K-major bf16           (8 token columns; rows past M arrive as zeros)
#
# Work units: the weight is cut into 128-row tiles and every tile's K into CHUNKS chunks of CH k-tiles (128 each);
# unit u is (tile u // CHUNKS, chunk u % CHUNKS). One CTA per SM (GRID CTAs), each running units until the pool is
# empty: unit bx is static (its first stages are loaded before griddepcontrol.wait), the rest are claimed from a
# global counter (every CTA makes exactly one failing claim; the last ticket rolls the counter back to 0, so every
# launch and graph replay starts from 0). Units are claimed in tile-major order, so a tile's chunks finish together.
#
# Split-K combine (CHUNKS > 1): a unit's epilogue stores its fp32 partial [128 rows][8 tokens] to ws[u], then
# thread 0 of the epilogue warps counts the unit on cnt[tile] with an acq_rel atomic after a barrier of those warps
# (the barrier orders their stores before the release). The unit that brings the count to CHUNKS finalizes the
# tile: it adds the CHUNKS partials in chunk order (its own from registers), rounds once to bf16 and stores y's
# [M][128] block, and resets cnt[tile] to 0. A unit's partial does not depend on which CTA computes it and the
# combine order is fixed, so y is bit-identical from run to run whatever the claim order.
#
# Stage = one k-tile: A 32 KB (both 64-element halves of the 128-byte swizzle, one 5-D TMA call) + B 2 KB (two TMA
# calls from x). RING stages; the phases run straight through unit boundaries. TMEM: two 8-column fp32
# accumulators, so the epilogue of one unit overlaps the MMAs of the next.
#
# L2: the weight loads are EVICT_FIRST, except for tiles < keep_tiles (normal priority), so that a later reader of
# the same shard (the drafter head after the target head) can find them in L2.
#
# Warps: 0 claimer + A/B TMA (one elected lane), 2 TMEM allocation + tcgen05 MMA (M 128, N 8, K 16), 4-7 epilogue
# (TMEM -> registers -> partial / fixup -> bf16 staging -> 16-byte stores), 1 and 3 idle.
#
# PDL: before the grid-dependency wait only barrier init, TMEM allocation, the tensormap prefetches and the static
# unit's A loads (a weight: nothing in the graph writes it). After it: B loads, claims, every global write. The
# dependents are released after the CTA's failing claim, so they launch only once every CTA of this grid has
# started (no dependent CTA can hold an SM that a static unit still needs).
# =============================================================================
"""Persistent CTM head GEMV for Kimi K3 (``y = x @ W^T``, M <= 8, a large vocab shard): dynamic (tile, k-chunk)
units with a deterministic split-K combine, the weight streamed before the grid-dependency wait."""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

CTA_M = 128  # weight rows per tile = the MMA's M
MMA_N = 8  # token columns
CTA_K = 128  # one k-tile: two 64-element halves of the 128-byte swizzle
MMA_K = 16
TMA_K_BOX = 64
TMA_COPY_ITERS = CTA_K // TMA_K_BOX
K_BLOCKS_PER_HALF = TMA_K_BOX // MMA_K
THREADS = 256
EPI_THREADS = 128
ELEM_BYTES = 2
VEC = 8  # bf16 per 16-byte vector
TMEM_COLS = 32
UNIT_RING = (
    16  # claimed-unit broadcast slots (the claimer runs at most RING k-tiles + 2 units ahead)
)
EVICT_FIRST = 0x12F0000000000000  # createpolicy.fractional.L2::evict_first, fraction 1.0 (sm_100)
SMEM_BYTES = 227 * 1024
MAX_RING = 6
FLAG_BACKOFF_NS = (
    256  # between polls of a split tile's piece flags (the pieces finish long before the finalizer)
)

# Shared-memory descriptor strides for the 128-byte swizzle, in 16-byte units.
LEADING = 16
STRIDE = 8 * TMA_K_BOX * ELEM_BYTES
A_HALF_ELEMS = CTA_M * TMA_K_BOX
B_HALF_ELEMS = MMA_N * TMA_K_BOX
STEP = (MMA_K * ELEM_BYTES) >> 4
A_BOX = A_HALF_ELEMS >> 3
B_BOX = B_HALF_ELEMS >> 3
STAGE_A = (CTA_M * CTA_K * ELEM_BYTES) >> 4
STAGE_B = (MMA_N * CTA_K * ELEM_BYTES) >> 4
A_BYTES = CTA_M * CTA_K * ELEM_BYTES
B_BYTES = MMA_N * CTA_K * ELEM_BYTES

io_dtype = cutlass.BFloat16


def num_k_tiles(k_in: int) -> int:
    return k_in // CTA_K


def num_units(n_out: int, k_in: int, chunk_tiles: int) -> int:
    return (n_out // CTA_M) * (num_k_tiles(k_in) // chunk_tiles)


def smem_bytes(ring: int) -> int:
    """Shared memory of a CTA: the A and B rings, the bf16 output staging tile, barriers and slots."""
    return ring * (A_BYTES + B_BYTES) + MMA_N * CTA_M * ELEM_BYTES + 1024


def supports(n_out: int, k_in: int, chunk_tiles: int, ring: int) -> bool:
    """Shapes the kernel runs: whole 128-row tiles, whole k-tiles, chunks that tile K, a ring that fits."""
    k_tiles = num_k_tiles(k_in)
    return (
        n_out > 0
        and n_out % CTA_M == 0
        and k_in > 0
        and k_in % CTA_K == 0
        and 0 < chunk_tiles <= k_tiles
        and k_tiles % chunk_tiles == 0
        and 1 <= ring <= MAX_RING
        and smem_bytes(ring) <= SMEM_BYTES
    )


@dsl_user_op
def _atomic_add_acq_rel(addr_i64, val, *, loc=None, ip=None):
    """atom.acq_rel.gpu.global.add.u32, returning the old value."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;", "=r,l,r", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _atomic_add_relaxed(addr_i64, val, *, loc=None, ip=None):
    """atom.relaxed.gpu.global.add.u32, returning the old value (the unit claim)."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.u32 $0, [$1], $2;", "=r,l,r", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _ld_acquire(addr_i64, *, loc=None, ip=None):
    """ld.acquire.gpu.global.u32 (a flag written by another SM)."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u32 $0, [$1];", "=r,l", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _flag_release(addr_i64, *, loc=None, ip=None):
    """fence.acq_rel.gpu, then st.relaxed.gpu 1: the release of everything the thread (and, through a preceding
    barrier, its CTA) stored before."""
    _llvm.inline_asm(
        None, [addr_i64.ir_value(loc=loc, ip=ip)],
        "fence.acq_rel.gpu;\n\tst.relaxed.gpu.global.u32 [$0], 1;", "l", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    """add.rn.f32: an fp32 add the compiler cannot reassociate or contract."""
    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(), [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _store_tile(stage_out, y, vals, row, tid, tile, n_out):
    """The finalized [8 tokens] sums of tile row ``row`` -> bf16 -> y[token, tile rows] for all 8 token rows (y has 8
    rows; rows past M hold the zero-filled x rows' zeros): staged per token in shared memory, then 16-byte stores
    (rows contiguous per token). The epilogue warps' barriers bracket the staging tile's reuse."""
    for t in range(MMA_N):
        stage_out.store(vals[t].to(io_dtype), idx=cutlass.Int32(t * CTA_M) + row)
    prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
    ct = tid // cutlass.Int32(CTA_M // VEC)
    cc = tid % cutlass.Int32(CTA_M // VEC)
    v = stage_out.load(
        idx=ct * cutlass.Int32(CTA_M) + cc * cutlass.Int32(VEC), vector_size=VEC, alignment=16
    )
    y.store(v, idx=ct * cutlass.Int32(n_out) + tile * cutlass.Int32(CTA_M) + cc * cutlass.Int32(VEC),
            vector_size=VEC, alignment=16)  # fmt: skip
    prims.barrier_cta_sync(1, thread_count=EPI_THREADS)


@cute.kernel
def k3_head_gemv_kernel(
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],  # W [N, K] bf16, 5-D, one call per k-tile
    tma_desc_x: cutlass.GridConstant[cuda.TensorMap],  # x [M, K] bf16, box 64 x 8
    y: cutlass.Array,  # [8 * N] bf16, token-major (all 8 token rows are written)
    ws: cutlass.Array,  # fp32 [units * 128 * 8]: the units' partials (CHUNKS > 1)
    cnt: cutlass.Array,  # int32 [tiles]: units of the tile done (0 between launches)
    claim: cutlass.Array,  # int32 [1]: the unit pool's ticket counter (0 between launches)
    num_tokens: cutlass.Int32,
    keep_tiles: cutlass.Int32,  # tiles [0, keep_tiles) load at normal L2 priority, the rest EVICT_FIRST
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    chunk_tiles: cutlass.Constexpr[int],  # CH: k-tiles per unit
    ring: cutlass.Constexpr[int],
    grid: cutlass.Constexpr[int],
):
    """Persistent: unit bx, then claimed units until the pool is empty; the last unit of a tile finalizes it."""
    k_tiles = num_k_tiles(k_in)
    chunks = k_tiles // chunk_tiles
    units = (n_out // CTA_M) * chunks
    tickets = max(
        units, grid
    )  # claims made in one launch: units - grid succeed (if positive), grid fail
    pre = min(ring, chunk_tiles)  # stages of the static unit loaded before the grid-dependency wait
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp_id = cute.arch.warp_idx()
    tma_ptr_w = tma_desc_w.get_ptr()
    tma_ptr_x = tma_desc_x.get_ptr()

    smem_a = cutlass.Array(
        io_dtype, ring * CTA_M * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_b = cutlass.Array(
        io_dtype, ring * MMA_N * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    stage_out = cutlass.Array(
        io_dtype, MMA_N * CTA_M, space=cutlass.AddressSpace.smem, alignment=16
    )
    full = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    empty = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    acc_full = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    acc_empty = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    unit_ready = cutlass.Array(
        cutlass.Int64, UNIT_RING, space=cutlass.AddressSpace.smem, alignment=8
    )
    unit_slot = cutlass.Array(
        cutlass.Int32, UNIT_RING, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_last = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    if warp_id == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        prims.prefetch_tensormap(tma_ptr_x)
        if prims.elect_sync():
            for s in cutlass.range_constexpr(ring):
                prims.mbarrier_init(full.subview(s), 1)
                prims.mbarrier_init(empty.subview(s), 1)
            for b in cutlass.range_constexpr(2):
                prims.mbarrier_init(acc_full.subview(b), 1)
                prims.mbarrier_init(acc_empty.subview(b), 4)  # one elected lane per epilogue warp
            for s in cutlass.range_constexpr(UNIT_RING):
                prims.mbarrier_init(unit_ready.subview(s), 1)
    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    if warp_id == 0:
        # =====================================================================
        # Claimer + TMA: the static unit's first stages before the wait, then
        # every stage of every unit this CTA gets, A and B on one barrier.
        # =====================================================================
        if prims.elect_sync():
            unit = cutlass.Int32(cutlass.select_(bx < cutlass.Int32(units), bx, cutlass.Int32(-1)))
            unit_slot.store(unit, idx=0)
            prims.mbarrier_arrive(unit_ready.subview(0))
            static_tile = unit // cutlass.Int32(chunks)
            static_k0 = (unit % cutlass.Int32(chunks)) * cutlass.Int32(chunk_tiles)
            if unit >= cutlass.Int32(0):
                for jj in cutlass.range_constexpr(pre):
                    prims.mbarrier_arrive_expect_tx(full.subview(jj), A_BYTES + B_BYTES)
                    if static_tile < keep_tiles:
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_a.subview(jj * CTA_M * CTA_K),
                            tma_ptr_w,
                            (
                                cutlass.Int32(0),
                                static_tile * cutlass.Int32(CTA_M),
                                (static_k0 + cutlass.Int32(jj)) * cutlass.Int32(TMA_COPY_ITERS),
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                            ),
                            full.subview(jj),
                        )
                    else:
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_a.subview(jj * CTA_M * CTA_K),
                            tma_ptr_w,
                            (
                                cutlass.Int32(0),
                                static_tile * cutlass.Int32(CTA_M),
                                (static_k0 + cutlass.Int32(jj)) * cutlass.Int32(TMA_COPY_ITERS),
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                            ),
                            full.subview(jj),
                            l2_cache_hint=EVICT_FIRST,
                        )
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            issued = cutlass.Int32(0)  # k-tiles issued by this CTA (the ring position)
            next_j = cutlass.Int32(0)  # first k-tile of the current unit still to issue
            if unit >= cutlass.Int32(0):
                for jj in cutlass.range_constexpr(pre):
                    for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_b.subview(jj * MMA_N * CTA_K + half * B_HALF_ELEMS),
                            tma_ptr_x,
                            ((static_k0 + cutlass.Int32(jj)) * cutlass.Int32(CTA_K) + cutlass.Int32(half * TMA_K_BOX),
                             cutlass.Int32(0)),
                            full.subview(jj),
                        )  # fmt: skip
                issued = cutlass.Int32(pre)
                next_j = cutlass.Int32(pre)
            n_units = cutlass.Int32(0)  # units published so far - 1
            pending = cutlass.Boolean(True)
            while pending:
                if unit >= cutlass.Int32(0):
                    tile = unit // cutlass.Int32(chunks)
                    k0 = (unit % cutlass.Int32(chunks)) * cutlass.Int32(chunk_tiles)
                    j = next_j
                    while j < cutlass.Int32(chunk_tiles):
                        stage = issued % cutlass.Int32(ring)
                        # The stage's previous MMAs committed (a fresh barrier passes parity 1 at once).
                        parity = (
                            (issued // cutlass.Int32(ring)) & cutlass.Int32(1)
                        ) ^ cutlass.Int32(1)
                        while not cute.arch.mbarrier_try_wait(
                            empty.subview(stage).data_ptr(), parity
                        ):
                            pass
                        k = k0 + j
                        prims.mbarrier_arrive_expect_tx(full.subview(stage), A_BYTES + B_BYTES)
                        if tile < keep_tiles:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                smem_a.subview(stage * cutlass.Int32(CTA_M * CTA_K)), tma_ptr_w,
                                (cutlass.Int32(0), tile * cutlass.Int32(CTA_M), k * cutlass.Int32(TMA_COPY_ITERS),
                                 cutlass.Int32(0), cutlass.Int32(0)),
                                full.subview(stage),
                            )  # fmt: skip
                        else:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                smem_a.subview(stage * cutlass.Int32(CTA_M * CTA_K)), tma_ptr_w,
                                (cutlass.Int32(0), tile * cutlass.Int32(CTA_M), k * cutlass.Int32(TMA_COPY_ITERS),
                                 cutlass.Int32(0), cutlass.Int32(0)),
                                full.subview(stage), l2_cache_hint=EVICT_FIRST,
                            )  # fmt: skip
                        for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                smem_b.subview(
                                    stage * cutlass.Int32(MMA_N * CTA_K)
                                    + cutlass.Int32(half * B_HALF_ELEMS)
                                ),
                                tma_ptr_x,
                                (
                                    k * cutlass.Int32(CTA_K) + cutlass.Int32(half * TMA_K_BOX),
                                    cutlass.Int32(0),
                                ),
                                full.subview(stage),
                            )
                        issued = issued + cutlass.Int32(1)
                        j = j + cutlass.Int32(1)
                # Claim the next unit; the holder of the last ticket rolls the counter back to 0.
                ticket = _atomic_add_relaxed(claim.data_ptr().toint(), cutlass.Int32(1))
                if ticket == cutlass.Int32(tickets - 1):
                    _atomic_add_relaxed(claim.data_ptr().toint(), cutlass.Int32(-tickets))
                claimed = ticket + cutlass.Int32(grid)
                unit = cutlass.Int32(
                    cutlass.select_(claimed < cutlass.Int32(units), claimed, cutlass.Int32(-1))
                )
                n_units = n_units + cutlass.Int32(1)
                slot = n_units % cutlass.Int32(UNIT_RING)
                unit_slot.store(unit, idx=slot)
                prims.mbarrier_arrive(unit_ready.subview(slot))
                next_j = cutlass.Int32(0)
                pending = unit >= cutlass.Int32(0)
            # Every CTA of this grid has started and made its last claim: the dependents may launch.
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    elif warp_id == 2:
        # =====================================================================
        # MMA: each unit's CH k-tiles into accumulator (unit count) % 2.
        # =====================================================================
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=io_dtype, b_dtype=io_dtype, n_dim=MMA_N, m_dim=CTA_M
        )
        desc_a_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_a, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        desc_b_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_b, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        tmem_base = tmem_ptr_i32.load()
        used = cutlass.Int32(0)  # k-tiles consumed (the ring position)
        count = cutlass.Int32(0)  # units done by this CTA
        while not cute.arch.mbarrier_try_wait(unit_ready.subview(0).data_ptr(), 0):
            pass
        unit = unit_slot.load(idx=0)
        while unit >= cutlass.Int32(0):
            buf = count & cutlass.Int32(1)
            if count >= cutlass.Int32(2):
                # The epilogue has read this accumulator's previous unit.
                while not cute.arch.mbarrier_try_wait(
                    acc_empty.subview(buf).data_ptr(),
                    ((count >> cutlass.Int32(1)) - cutlass.Int32(1)) & cutlass.Int32(1),
                ):
                    pass
            tmem_acc = cutlass.inttoptr(tmem_base + buf * cutlass.Int32(MMA_N), 6, cutlass.Int32)
            for j in cutlass.range(chunk_tiles, unroll=1):
                stage = used % cutlass.Int32(ring)
                while not cute.arch.mbarrier_try_wait(
                    full.subview(stage).data_ptr(), (used // cutlass.Int32(ring)) & cutlass.Int32(1)
                ):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                    box = kb // K_BLOCKS_PER_HALF
                    within = kb % K_BLOCKS_PER_HALF
                    desc_a = desc_a_base + (
                        stage * cutlass.Int32(STAGE_A) + cutlass.Int32(box * A_BOX + within * STEP)
                    )
                    desc_b = desc_b_base + (
                        stage * cutlass.Int32(STAGE_B) + cutlass.Int32(box * B_BOX + within * STEP)
                    )
                    accumulate = cutlass.Boolean(True)
                    if cutlass.const_expr(kb == 0):
                        accumulate = j > cutlass.Int32(0)
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            tmem_acc,
                            desc_a,
                            desc_b,
                            idesc,
                            accumulate,
                        )
                if prims.elect_sync():
                    prims.tcgen05_commit(empty.subview(stage))
                used = used + cutlass.Int32(1)
            if prims.elect_sync():
                prims.tcgen05_commit(acc_full.subview(buf))
            count = count + cutlass.Int32(1)
            slot = count % cutlass.Int32(UNIT_RING)
            while not cute.arch.mbarrier_try_wait(
                unit_ready.subview(slot).data_ptr(),
                (count // cutlass.Int32(UNIT_RING)) & cutlass.Int32(1),
            ):
                pass
            unit = unit_slot.load(idx=slot)
    elif warp_id >= 4:
        # =====================================================================
        # Epilogue: TMEM -> registers -> partial + count (or the whole sum when
        # K is one chunk) -> the finalizer's ordered sum -> bf16 -> y.
        #   warp 4 + w reads TMEM lanes 32 w .. 32 w + 31 = tile rows 32 w + lane.
        # =====================================================================
        tid = tx - cutlass.Int32(EPI_THREADS)
        lane = tx % 32
        w = warp_id - 4
        row = w * cutlass.Int32(32) + lane
        tmem_base = tmem_ptr_i32.load()
        prims.griddepcontrol(prims.GridDepAction.WAIT)
        count = cutlass.Int32(0)
        while not cute.arch.mbarrier_try_wait(unit_ready.subview(0).data_ptr(), 0):
            pass
        unit = unit_slot.load(idx=0)
        while unit >= cutlass.Int32(0):
            buf = count & cutlass.Int32(1)
            while not cute.arch.mbarrier_try_wait(
                acc_full.subview(buf).data_ptr(), (count >> cutlass.Int32(1)) & cutlass.Int32(1)
            ):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            acc = prims.tcgen05_ld(
                "32x32b",
                cutlass.inttoptr(
                    tmem_base
                    + ((w * cutlass.Int32(32)) << cutlass.Int32(16))
                    + buf * cutlass.Int32(MMA_N),
                    6,
                    cutlass.Float32,
                ),
                num=MMA_N,
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if prims.elect_sync():
                prims.mbarrier_arrive(acc_empty.subview(buf))
            vals = [cutlass.Float32(acc[t]) for t in range(MMA_N)]
            tile = unit // cutlass.Int32(chunks)
            if cutlass.const_expr(chunks == 1):
                _store_tile(stage_out, y, vals, row, tid, tile, n_out)
            else:
                chunk = unit % cutlass.Int32(chunks)
                part = (unit * cutlass.Int32(CTA_M) + row) * cutlass.Int32(MMA_N)
                ws.store(cutlass.Vector.from_elements((vals[0], vals[1], vals[2], vals[3]), cutlass.Float32), idx=part,
                         vector_size=4, alignment=16)  # fmt: skip
                ws.store(cutlass.Vector.from_elements((vals[4], vals[5], vals[6], vals[7]), cutlass.Float32),
                         idx=part + cutlass.Int32(4), vector_size=4, alignment=16)  # fmt: skip
                # Every epilogue thread's partial is stored; thread 0's release covers them.
                prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
                if tid == cutlass.Int32(0):
                    done = _atomic_add_acq_rel(
                        cnt.subview(tile).data_ptr().toint(), cutlass.Int32(1)
                    )
                    last = cutlass.Int32(
                        cutlass.select_(
                            done == cutlass.Int32(chunks - 1), cutlass.Int32(1), cutlass.Int32(0)
                        )
                    )
                    s_last.store(last, idx=0)
                prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
                if s_last.load(idx=0) != cutlass.Int32(0):
                    # The tile's partials in chunk order (this unit's own from registers). Volatile loads: other
                    # SMs wrote them; thread 0's acquire and the barrier above order these reads after the writes.
                    total = []
                    for c in cutlass.range_constexpr(chunks):
                        src = (
                            (tile * cutlass.Int32(chunks) + cutlass.Int32(c)) * cutlass.Int32(CTA_M)
                            + row
                        ) * cutlass.Int32(MMA_N)
                        lo = ws.load(idx=src, vector_size=4, alignment=16, is_volatile=True)
                        hi = ws.load(
                            idx=src + cutlass.Int32(4),
                            vector_size=4,
                            alignment=16,
                            is_volatile=True,
                        )
                        mine = chunk == cutlass.Int32(c)
                        for t in cutlass.range_constexpr(MMA_N):
                            got = cutlass.Float32(lo[t]) if t < 4 else cutlass.Float32(hi[t - 4])
                            got = cutlass.Float32(cutlass.select_(mine, vals[t], got))
                            if cutlass.const_expr(c == 0):
                                total.append(got)
                            else:
                                total[t] = _add_rn(total[t], got)
                    if tid == cutlass.Int32(0):
                        cnt.store(cutlass.Int32(0), idx=tile)
                    _store_tile(stage_out, y, total, row, tid, tile, n_out)
            count = count + cutlass.Int32(1)
            slot = count % cutlass.Int32(UNIT_RING)
            while not cute.arch.mbarrier_try_wait(
                unit_ready.subview(slot).data_ptr(),
                (count // cutlass.Int32(UNIT_RING)) & cutlass.Int32(1),
            ):
                pass
            unit = unit_slot.load(idx=slot)
        prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
        if warp_id == 4:
            prims.tcgen05_dealloc(cutlass.inttoptr(tmem_base, 6, cutlass.Int32), TMEM_COLS)


# =============================================================================
# Stream-K variant: the (tile, k-tile) space flattened tile-major and cut into GRID equal ranges, CTA c owning
# [c T / G, (c + 1) T / G). A CTA's range is a sequence of segments (one tile's contiguous k-tiles each); a tile
# split over several CTAs ("pieces", in k order = CTA order) is combined by the CTA of its piece 0 (the one holding
# k-tile 0), which is the end of that CTA's range: every other piece starts its CTA's range (or is all of it), so
# it is done early. A piece p > 0 stores its fp32 partial to ws[tile][p] and raises flag[tile][p] (fence.acq_rel +
# a relaxed store after a barrier of the epilogue warps); the finalizer acquires those flags and loads the partials
# while its own MMAs still run, then adds them after its accumulator in k order, rounds once, stores y and lowers
# the flags. The partition is fixed, so the sums are too. No claims: every warp walks the same static sequence, and
# the whole ring is loaded before the grid-dependency wait (optionally the next k-tiles are prefetched into L2).
# =============================================================================
def streamk_cta_of(q: int, total: int, grid: int) -> int:
    """The CTA whose range holds flat k-tile q (ranges [c T / G, (c + 1) T / G), some empty when G > T)."""
    return ((q + 1) * grid - 1) // total


def streamk_max_pieces(n_out: int, k_in: int, grid: int) -> int:
    """The most CTAs any tile is split over."""
    k_tiles = num_k_tiles(k_in)
    tiles = n_out // CTA_M
    total = tiles * k_tiles
    return max(
        streamk_cta_of((t + 1) * k_tiles - 1, total, grid)
        - streamk_cta_of(t * k_tiles, total, grid)
        + 1
        for t in range(tiles)
    )


def streamk_supports(n_out: int, k_in: int, ring: int) -> bool:
    return (
        n_out > 0
        and n_out % CTA_M == 0
        and k_in > 0
        and k_in % CTA_K == 0
        and 1 <= ring <= MAX_RING
        and (smem_bytes(ring) <= SMEM_BYTES)
    )


@cute.kernel
def k3_head_gemv_sk_kernel(
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],  # W [N, K] bf16, 5-D, one call per k-tile
    tma_desc_x: cutlass.GridConstant[cuda.TensorMap],  # x [M, K] bf16, box 64 x 8
    y: cutlass.Array,  # [8 * N] bf16, token-major (all 8 token rows are written)
    ws: cutlass.Array,  # fp32 [tiles * max_pieces * 128 * 8]: the pieces' partials
    flags: cutlass.Array,  # int32 [tiles * max_pieces]: piece p > 0 of the tile stored its partial (0 between launches)
    num_tokens: cutlass.Int32,
    keep_tiles: cutlass.Int32,  # tiles [0, keep_tiles) load at normal L2 priority, the rest EVICT_FIRST
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    grid: cutlass.Constexpr[int],
    max_pieces: cutlass.Constexpr[int],
    prefetch: cutlass.Constexpr[
        int
    ],  # k-tiles after the ring prefetched into L2 before the grid-dependency wait
):
    """Stream-K: this CTA's equal share of the flat (tile, k-tile) space; split tiles combined by their last piece."""
    k_tiles = num_k_tiles(k_in)
    total = (n_out // CTA_M) * k_tiles
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp_id = cute.arch.warp_idx()
    tma_ptr_w = tma_desc_w.get_ptr()
    tma_ptr_x = tma_desc_x.get_ptr()
    lo = (bx * cutlass.Int32(total)) // cutlass.Int32(grid)
    hi = ((bx + cutlass.Int32(1)) * cutlass.Int32(total)) // cutlass.Int32(grid)
    count = hi - lo

    smem_a = cutlass.Array(
        io_dtype, ring * CTA_M * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_b = cutlass.Array(
        io_dtype, ring * MMA_N * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    stage_out = cutlass.Array(
        io_dtype, MMA_N * CTA_M, space=cutlass.AddressSpace.smem, alignment=16
    )
    full = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    empty = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    acc_full = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    acc_empty = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    if warp_id == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        prims.prefetch_tensormap(tma_ptr_x)
        if prims.elect_sync():
            for s in cutlass.range_constexpr(ring):
                prims.mbarrier_init(full.subview(s), 1)
                prims.mbarrier_init(empty.subview(s), 1)
            for b in cutlass.range_constexpr(2):
                prims.mbarrier_init(acc_full.subview(b), 1)
                prims.mbarrier_init(acc_empty.subview(b), 4)  # one elected lane per epilogue warp
    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    if warp_id == 0:
        # =====================================================================
        # TMA: the range's first RING k-tiles' weight before the wait, their
        # activation after it, then every later k-tile when its stage frees.
        # =====================================================================
        if prims.elect_sync():
            for jj in cutlass.range_constexpr(ring):
                if cutlass.Int32(jj) < count:
                    q0 = lo + cutlass.Int32(jj)
                    t0 = q0 // cutlass.Int32(k_tiles)
                    prims.mbarrier_arrive_expect_tx(full.subview(jj), A_BYTES + B_BYTES)
                    if t0 < keep_tiles:
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_a.subview(jj * CTA_M * CTA_K),
                            tma_ptr_w,
                            (
                                cutlass.Int32(0),
                                t0 * cutlass.Int32(CTA_M),
                                (q0 % cutlass.Int32(k_tiles)) * cutlass.Int32(TMA_COPY_ITERS),
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                            ),
                            full.subview(jj),
                        )
                    else:
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_a.subview(jj * CTA_M * CTA_K),
                            tma_ptr_w,
                            (
                                cutlass.Int32(0),
                                t0 * cutlass.Int32(CTA_M),
                                (q0 % cutlass.Int32(k_tiles)) * cutlass.Int32(TMA_COPY_ITERS),
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                            ),
                            full.subview(jj),
                            l2_cache_hint=EVICT_FIRST,
                        )
            # The next k-tiles of the range into L2 while the predecessor runs (a weight; the loads below hit L2).
            for jp in cutlass.range_constexpr(ring, ring + prefetch):
                if cutlass.Int32(jp) < count:
                    qp = lo + cutlass.Int32(jp)
                    prims.cp_async_bulk_tensor_prefetch(
                        tma_ptr_w,
                        [
                            cutlass.Int32(0),
                            (qp // cutlass.Int32(k_tiles)) * cutlass.Int32(CTA_M),
                            (qp % cutlass.Int32(k_tiles)) * cutlass.Int32(TMA_COPY_ITERS),
                            cutlass.Int32(0),
                            cutlass.Int32(0),
                        ],
                        [],  # tile mode: no im2col offsets
                    )
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            for jj in cutlass.range_constexpr(ring):
                if cutlass.Int32(jj) < count:
                    kb0 = (lo + cutlass.Int32(jj)) % cutlass.Int32(k_tiles)
                    for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_b.subview(jj * MMA_N * CTA_K + half * B_HALF_ELEMS),
                            tma_ptr_x,
                            (kb0 * cutlass.Int32(CTA_K) + cutlass.Int32(half * TMA_K_BOX), cutlass.Int32(0)),
                            full.subview(jj),
                        )  # fmt: skip
            n = cutlass.Int32(ring)
            while n < count:
                stage = n % cutlass.Int32(ring)
                # The stage's previous MMAs committed.
                while not cute.arch.mbarrier_try_wait(
                    empty.subview(stage).data_ptr(),
                    ((n // cutlass.Int32(ring)) & cutlass.Int32(1)) ^ cutlass.Int32(1),
                ):
                    pass
                q = lo + n
                tile = q // cutlass.Int32(k_tiles)
                k = q % cutlass.Int32(k_tiles)
                prims.mbarrier_arrive_expect_tx(full.subview(stage), A_BYTES + B_BYTES)
                if tile < keep_tiles:
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_a.subview(stage * cutlass.Int32(CTA_M * CTA_K)), tma_ptr_w,
                        (cutlass.Int32(0), tile * cutlass.Int32(CTA_M), k * cutlass.Int32(TMA_COPY_ITERS),
                         cutlass.Int32(0), cutlass.Int32(0)),
                        full.subview(stage),
                    )  # fmt: skip
                else:
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_a.subview(stage * cutlass.Int32(CTA_M * CTA_K)), tma_ptr_w,
                        (cutlass.Int32(0), tile * cutlass.Int32(CTA_M), k * cutlass.Int32(TMA_COPY_ITERS),
                         cutlass.Int32(0), cutlass.Int32(0)),
                        full.subview(stage), l2_cache_hint=EVICT_FIRST,
                    )  # fmt: skip
                for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_b.subview(stage * cutlass.Int32(MMA_N * CTA_K) + cutlass.Int32(half * B_HALF_ELEMS)),
                        tma_ptr_x,
                        (k * cutlass.Int32(CTA_K) + cutlass.Int32(half * TMA_K_BOX), cutlass.Int32(0)),
                        full.subview(stage),
                    )  # fmt: skip
                n = n + cutlass.Int32(1)
            # All of this CTA's loads are issued: the dependents may launch once every CTA gets here.
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    elif warp_id == 2:
        # =====================================================================
        # MMA: each segment's k-tiles into accumulator (segment count) % 2.
        # =====================================================================
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=io_dtype, b_dtype=io_dtype, n_dim=MMA_N, m_dim=CTA_M
        )
        desc_a_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_a, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        desc_b_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_b, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        tmem_base = tmem_ptr_i32.load()
        used = cutlass.Int32(0)
        seg = cutlass.Int32(0)
        q = lo
        while q < hi:
            seg_tile = q // cutlass.Int32(k_tiles)
            seg_end = cutlass.Int32(
                cutlass.select_(
                    (seg_tile + cutlass.Int32(1)) * cutlass.Int32(k_tiles) < hi,
                    (seg_tile + cutlass.Int32(1)) * cutlass.Int32(k_tiles),
                    hi,
                )
            )
            buf = seg & cutlass.Int32(1)
            if seg >= cutlass.Int32(2):
                # The epilogue has read this accumulator's previous segment.
                while not cute.arch.mbarrier_try_wait(
                    acc_empty.subview(buf).data_ptr(),
                    ((seg >> cutlass.Int32(1)) - cutlass.Int32(1)) & cutlass.Int32(1),
                ):
                    pass
            tmem_acc = cutlass.inttoptr(tmem_base + buf * cutlass.Int32(MMA_N), 6, cutlass.Int32)
            j = q
            while j < seg_end:
                stage = used % cutlass.Int32(ring)
                while not cute.arch.mbarrier_try_wait(
                    full.subview(stage).data_ptr(), (used // cutlass.Int32(ring)) & cutlass.Int32(1)
                ):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                    box = kb // K_BLOCKS_PER_HALF
                    within = kb % K_BLOCKS_PER_HALF
                    desc_a = desc_a_base + (
                        stage * cutlass.Int32(STAGE_A) + cutlass.Int32(box * A_BOX + within * STEP)
                    )
                    desc_b = desc_b_base + (
                        stage * cutlass.Int32(STAGE_B) + cutlass.Int32(box * B_BOX + within * STEP)
                    )
                    accumulate = cutlass.Boolean(True)
                    if cutlass.const_expr(kb == 0):
                        accumulate = j > q
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            tmem_acc,
                            desc_a,
                            desc_b,
                            idesc,
                            accumulate,
                        )
                if prims.elect_sync():
                    prims.tcgen05_commit(empty.subview(stage))
                used = used + cutlass.Int32(1)
                j = j + cutlass.Int32(1)
            if prims.elect_sync():
                prims.tcgen05_commit(acc_full.subview(buf))
            seg = seg + cutlass.Int32(1)
            q = seg_end
    elif warp_id >= 4:
        # =====================================================================
        # Epilogue: per segment, TMEM -> registers -> y (a whole tile), or piece
        # p > 0's partial + flag, or piece 0's ordered sum of all pieces -> y.
        # =====================================================================
        tid = tx - cutlass.Int32(EPI_THREADS)
        lane = tx % 32
        w = warp_id - 4
        row = w * cutlass.Int32(32) + lane
        tmem_base = tmem_ptr_i32.load()
        prims.griddepcontrol(prims.GridDepAction.WAIT)
        seg = cutlass.Int32(0)
        q = lo
        while q < hi:
            seg_tile = q // cutlass.Int32(k_tiles)
            seg_end = cutlass.Int32(
                cutlass.select_(
                    (seg_tile + cutlass.Int32(1)) * cutlass.Int32(k_tiles) < hi,
                    (seg_tile + cutlass.Int32(1)) * cutlass.Int32(k_tiles),
                    hi,
                )
            )
            buf = seg & cutlass.Int32(1)
            acc_addr = (
                tmem_base
                + ((w * cutlass.Int32(32)) << cutlass.Int32(16))
                + buf * cutlass.Int32(MMA_N)
            )
            acc_parity = (seg >> cutlass.Int32(1)) & cutlass.Int32(1)
            # The tile's pieces: the CTAs whose ranges hold its first and last k-tiles, and everything between.
            first_cta = (
                (seg_tile * cutlass.Int32(k_tiles) + cutlass.Int32(1)) * cutlass.Int32(grid)
                - cutlass.Int32(1)
            ) // cutlass.Int32(total)
            last_cta = (
                (seg_tile + cutlass.Int32(1)) * cutlass.Int32(k_tiles) * cutlass.Int32(grid)
                - cutlass.Int32(1)
            ) // cutlass.Int32(total)
            pieces = last_cta - first_cta + cutlass.Int32(1)
            piece = bx - first_cta
            slot0 = seg_tile * cutlass.Int32(max_pieces)
            if pieces == cutlass.Int32(1):
                while not cute.arch.mbarrier_try_wait(acc_full.subview(buf).data_ptr(), acc_parity):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                acc = prims.tcgen05_ld(
                    "32x32b", cutlass.inttoptr(acc_addr, 6, cutlass.Float32), num=MMA_N
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if prims.elect_sync():
                    prims.mbarrier_arrive(acc_empty.subview(buf))
                _store_tile(
                    stage_out,
                    y,
                    [cutlass.Float32(acc[t]) for t in range(MMA_N)],
                    row,
                    tid,
                    seg_tile,
                    n_out,
                )
            elif piece == cutlass.Int32(0):
                # The finalizer. The other pieces' partials first, while this segment's MMAs run: thread 0
                # acquires the flags (backing off between polls, so the epilogue warps do not load the memory
                # system while the weight streams), the barrier orders every thread's loads after its acquires,
                # then each thread loads its row (a slot past the piece count reloads piece 1's and is not added).
                if tid == cutlass.Int32(0):
                    for p in cutlass.range_constexpr(1, max_pieces):
                        if cutlass.Int32(p) < pieces:
                            while _ld_acquire(
                                flags.subview(slot0 + cutlass.Int32(p)).data_ptr().toint()
                            ) == cutlass.Int32(0):
                                prims.nanosleep(FLAG_BACKOFF_NS)
                prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
                pre = []
                for p in cutlass.range_constexpr(1, max_pieces):
                    live = cutlass.Int32(p) < pieces
                    src = (
                        (
                            slot0
                            + cutlass.Int32(
                                cutlass.select_(live, cutlass.Int32(p), cutlass.Int32(1))
                            )
                        )
                        * cutlass.Int32(CTA_M)
                        + row
                    ) * cutlass.Int32(MMA_N)
                    pre.append(
                        (
                            ws.load(idx=src, vector_size=4, alignment=16, is_volatile=True),
                            ws.load(
                                idx=src + cutlass.Int32(4),
                                vector_size=4,
                                alignment=16,
                                is_volatile=True,
                            ),
                        )
                    )
                while not cute.arch.mbarrier_try_wait(acc_full.subview(buf).data_ptr(), acc_parity):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                acc = prims.tcgen05_ld(
                    "32x32b", cutlass.inttoptr(acc_addr, 6, cutlass.Float32), num=MMA_N
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if prims.elect_sync():
                    prims.mbarrier_arrive(acc_empty.subview(buf))
                # Piece 0 (k from 0), then pieces 1, 2, ... in k order.
                total_v = [cutlass.Float32(acc[t]) for t in range(MMA_N)]
                for p in cutlass.range_constexpr(1, max_pieces):
                    live = cutlass.Int32(p) < pieces
                    lo_v, hi_v = pre[p - 1]
                    for t in cutlass.range_constexpr(MMA_N):
                        got = cutlass.Float32(lo_v[t]) if t < 4 else cutlass.Float32(hi_v[t - 4])
                        total_v[t] = cutlass.Float32(
                            cutlass.select_(live, _add_rn(total_v[t], got), total_v[t])
                        )
                _store_tile(stage_out, y, total_v, row, tid, seg_tile, n_out)
                # Every epilogue thread's acquires are behind the store helper's barriers: lower the flags.
                if tid == cutlass.Int32(0):
                    for p in cutlass.range_constexpr(1, max_pieces):
                        if cutlass.Int32(p) < pieces:
                            flags.store(cutlass.Int32(0), idx=slot0 + cutlass.Int32(p))
            else:
                while not cute.arch.mbarrier_try_wait(acc_full.subview(buf).data_ptr(), acc_parity):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                acc = prims.tcgen05_ld(
                    "32x32b", cutlass.inttoptr(acc_addr, 6, cutlass.Float32), num=MMA_N
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if prims.elect_sync():
                    prims.mbarrier_arrive(acc_empty.subview(buf))
                part = ((slot0 + piece) * cutlass.Int32(CTA_M) + row) * cutlass.Int32(MMA_N)
                ws.store(
                    cutlass.Vector.from_elements(
                        (
                            cutlass.Float32(acc[0]),
                            cutlass.Float32(acc[1]),
                            cutlass.Float32(acc[2]),
                            cutlass.Float32(acc[3]),
                        ),
                        cutlass.Float32,
                    ),
                    idx=part,
                    vector_size=4,
                    alignment=16,
                )
                ws.store(
                    cutlass.Vector.from_elements(
                        (
                            cutlass.Float32(acc[4]),
                            cutlass.Float32(acc[5]),
                            cutlass.Float32(acc[6]),
                            cutlass.Float32(acc[7]),
                        ),
                        cutlass.Float32,
                    ),
                    idx=part + cutlass.Int32(4),
                    vector_size=4,
                    alignment=16,
                )
                # Every epilogue thread's partial is stored; thread 0's release covers them.
                prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
                if tid == cutlass.Int32(0):
                    _flag_release(flags.subview(slot0 + piece).data_ptr().toint())
            seg = seg + cutlass.Int32(1)
            q = seg_end
        prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
        if warp_id == 4:
            prims.tcgen05_dealloc(cutlass.inttoptr(tmem_base, 6, cutlass.Int32), TMEM_COLS)


def _weight_tensor_map(w, n_out, k_in):
    """W as five TMA dimensions (64-element column chunk, row, 64-element chunk index, 1, 1) so one call
    per k-tile lands both 128-byte-swizzled halves; strides in 16-byte units."""
    return cuda.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[TMA_K_BOX, n_out, k_in // TMA_K_BOX, 1, 1],
        global_strides=[
            (k_in * ELEM_BYTES) // 16,
            (TMA_K_BOX * ELEM_BYTES) // 16,
            (n_out * k_in * ELEM_BYTES) // 16,
            (n_out * k_in * ELEM_BYTES) // 16,
        ],
        box_dims=[TMA_K_BOX, CTA_M, TMA_COPY_ITERS, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


def _activation_tensor_map(x, cols, num_tokens):
    """x [M, cols] (rows dense) as (cols, M) with an 8-row box: rows past num_tokens arrive as zeros."""
    return cuda.create_tensor_map_tiled(
        global_address=x.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[cols, num_tokens],
        global_strides=[(cols * ELEM_BYTES) // 16],
        box_dims=[TMA_K_BOX, MMA_N],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


@cute.jit
def k3_head_gemv(
    w: cute.Tensor,  # [N, K] bf16, K contiguous
    x: cute.Tensor,  # [M, K] bf16, K contiguous, M <= 8
    y: cute.Tensor,  # [8 * N] bf16
    ws: cute.Tensor,  # fp32 [units * 128 * 8]
    cnt: cute.Tensor,  # int32 [tiles], zero
    claim: cute.Tensor,  # int32 [1], zero
    num_tokens: cutlass.Int32,
    keep_tiles: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    chunk_tiles: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    grid: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """``y = x @ w^T`` over ``grid`` persistent CTAs."""
    tma_desc_w = _weight_tensor_map(w, n_out, k_in)
    tma_desc_x = _activation_tensor_map(x, k_in, num_tokens)
    k3_head_gemv_kernel(
        tma_desc_w,
        tma_desc_x,
        y,
        ws,
        cnt,
        claim,
        num_tokens,
        keep_tiles,
        n_out,
        k_in,
        chunk_tiles,
        ring,
        grid,
    ).launch(
        grid=[grid, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
        use_pdl=use_pdl,
    )


@cute.jit
def k3_head_gemv_sk(
    w: cute.Tensor,  # [N, K] bf16, K contiguous
    x: cute.Tensor,  # [M, K] bf16, K contiguous, M <= 8
    y: cute.Tensor,  # [8 * N] bf16
    ws: cute.Tensor,  # fp32 [tiles * max_pieces * 128 * 8]
    flags: cute.Tensor,  # int32 [tiles * max_pieces], zero
    num_tokens: cutlass.Int32,
    keep_tiles: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    grid: cutlass.Constexpr[int],
    max_pieces: cutlass.Constexpr[int],
    prefetch: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """``y = x @ w^T`` over ``grid`` stream-K CTAs."""
    tma_desc_w = _weight_tensor_map(w, n_out, k_in)
    tma_desc_x = _activation_tensor_map(x, k_in, num_tokens)
    k3_head_gemv_sk_kernel(
        tma_desc_w,
        tma_desc_x,
        y,
        ws,
        flags,
        num_tokens,
        keep_tiles,
        n_out,
        k_in,
        ring,
        grid,
        max_pieces,
        prefetch,
    ).launch(
        grid=[grid, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
        use_pdl=use_pdl,
    )

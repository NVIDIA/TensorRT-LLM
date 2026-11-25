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
# Kimi K3 decode GEMV -- CTM (prims/cute) kernel, M <= 8 tokens, bf16 in/out
# =============================================================================
#
#   y[t, n] = sum_k B[t, k] * W[n, k]          (A @ B^T, fp32 accumulation in TMEM)
#     A = W  (N, K <= 768) K-major bf16         (the streamed weight, 128 rows per cluster)
#     B      (M <= 8, K) K-major bf16           (8 token columns; rows past M are zero)
#
# Variants (compile-time):
#   PLAIN  B = x                                               (KDA / MLA o_proj)
#   TAIL   y = rsqrt(mean(latent^2) + eps) * (latent_slice @ W_lat^T) + act @ W_act^T
#          (two TMA maps, two TMEM accumulators; the row-parallel MoE tail)
#   GATED  B = bf16(a * bf16(sigmoid(g)))                      (MLA o_proj with its output gate)
#   GATED_S B = bf16(a * s) with s = bf16(sigmoid(g)) precomputed (the long-K GEMV's sigmoid rows)
#   SWIGLU B = bf16(g * sigmoid(g) * a) in fp32, g and a the two halves of a gate_up output (silu_and_mul folded
#          into the down projection)
#          a and g land by TMA in two identically swizzled tiles; the epilogue warps rewrite B in
#          place (same offsets in both tiles), fence the generic writes to the async proxy and
#          release the MMA through a 128-arrival barrier (the resident-B prolog of
#          fuse_rmsnorm_qkv_rope).
#
# Geometry: one 128-row weight tile per cluster of SPLIT CTAs (1, 2 or 4). Rank r takes the interleaved
# k-tiles r, r + SPLIT, ... (when SPLIT does not divide the k-tiles, the first k-tiles % SPLIT ranks take one
# more) and owns output rows [R r, R r + R) of the tile, R = 128 / SPLIT (the TMEM
# lanes of epilogue warps r * 4 / SPLIT ...); every other rank's warps holding those rows send their
# fp32 partials to slot [source rank] of the owner's mailbox; the owner adds the SPLIT partials in rank
# order, then rounds once. Two ways to send (compile-time PUSH, the same sums): DSMEM stores and a release
# arrive at cluster scope on the owner's barrier, acquired by a try_wait at cluster scope; or 16-byte
# st.async stores that complete the owner's barrier by bytes, which the owner spins on with test_wait (acquire at
# cluster scope).
#
# Every k-tile of the CTA has its own shared-memory stage, and the whole weight slice is TMA'd
# (EVICT_FIRST) before griddepcontrol.wait: launched early under PDL the kernel streams its weight
# while the predecessor runs. Only the activation loads follow the wait.
#
# Warps: 0 weight TMA (+ early dependent trigger), 1 activation TMA after the grid dependency,
# 2 TMEM allocation + tcgen05 MMA (M 128, N 8, K 16), 3 idle, 4-7 epilogue (gate prologue, TMEM ->
# registers, split-K reduce, bf16 staging, 16-byte coalesced stores).
# =============================================================================
"""CTM decode GEMV for Kimi K3 (``y = B @ W^T``, M <= 8): plain, MoE-tail and output-gated variants."""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

CTA_M = 128  # weight rows (output features) per cluster = the MMA's M
MMA_N = 8  # token columns
CTA_K = 128  # one k-tile: two 64-element halves of the 128-byte swizzle
MMA_K = 16
TMA_K_BOX = 64
TMA_COPY_ITERS = CTA_K // TMA_K_BOX
K_BLOCKS_PER_HALF = TMA_K_BOX // MMA_K
MAX_K_TILES = 6  # 6 x (32 KB weight + 2 KB B [+ 2 KB gate]) of shared memory per CTA
THREADS = 256
EPI_THREADS = 128
TMEM_COLS = 32
ELEM_BYTES = 2
VEC = 8  # bf16 per 16-byte vector
EVICT_FIRST = 0x12F0000000000000  # createpolicy.fractional.L2::evict_first, fraction 1.0 (sm_100)

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

PLAIN = 0
TAIL = 1
GATED = 2  # B = bf16(a * bf16(sigmoid(g))), g given
GATED_S = 3  # B = bf16(a * s), s = bf16(sigmoid(g)) given (e.g. by k3_ctm_gemv_long's sigmoid rows)
SWIGLU = 4  # B = bf16(g * sigmoid(g) * a) in fp32: silu_and_mul of a gate_up output (g the first half, a the second)

io_dtype = cutlass.BFloat16


def num_k_tiles(k_in: int) -> int:
    return (k_in + CTA_K - 1) // CTA_K


def supports(n_out: int, k_in: int, split: int = 1) -> bool:
    """Shapes the kernel runs: whole 128-row tiles, whole k-tiles, at least one and at most MAX_K_TILES k-tiles per
    CTA (the first k-tiles % split ranks take one more)."""
    k_tiles = num_k_tiles(k_in)
    return (
        n_out % CTA_M == 0
        and k_in % CTA_K == 0
        and split in (1, 2, 4)
        and split <= k_tiles
        and (k_tiles + split - 1) // split <= MAX_K_TILES
    )


@dsl_user_op
def _div_rn(a, b, *, loc=None, ip=None):
    """IEEE fp32 division (div.rn.f32), as torch's ``one / (one + exp(-x))``."""
    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(), [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _mapa_u32(smem_ptr, peer, *, loc=None, ip=None):
    """The shared::cluster address of this CTA's shared-memory location in cluster CTA ``peer``."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [smem_ptr.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(peer).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _st_async_v4(dst, a, b, c, d, mbar, *, loc=None, ip=None):
    """st.async of four fp32 (16 bytes) to a shared::cluster address, completing ``mbar`` (a shared::cluster address)
    by 16 bytes."""
    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip), cutlass.Float32(c).ir_value(loc=loc, ip=ip),
         cutlass.Float32(d).ir_value(loc=loc, ip=ip), cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [$0], {$1, $2, $3, $4}, [$5];", "r,f,f,f,f,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _test_wait_cluster(mbar, parity, *, loc=None, ip=None):
    """mbarrier.test_wait.parity.acquire.cluster on a barrier of this CTA (``mbar``, its shared-memory pointer): whether
    phase ``parity`` has completed, acquiring at cluster scope. The barriers it is used on are completed by other CTAs
    (st.async complete_tx, remote arrives), which release at cluster scope."""
    done = cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [mbar.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(parity).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}", "=r,r,r,~{memory}", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip
    return done != cutlass.Int32(0)


def _bf16_rn(v):
    """fp32 -> bf16 precision (round to nearest even) in the integer domain, kept as fp32.

    Bit operations, not a float cast, so no fptrunc/fpext pair exists that the compiler could
    fold into the following multiply and skip this rounding.
    """
    u = v.bitcast(cutlass.Int32)
    u = u + (((u >> 16) & 1) + 0x7FFF)
    u = (u >> 16) << 16
    return u.bitcast(cutlass.Float32)


def _sigmoid_bf16(g):
    """bf16(sigmoid(g)) for a bf16 value g held in fp32, as torch computes it for bf16 tensors."""
    one = cutlass.Float32(1.0)
    return _bf16_rn(_div_rn(one, one + cute.math.exp(-g, fastmath=False)))


@cute.kernel
def k3_ctm_gemv_kernel(
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],  # W [N, K] bf16, 5-D, one call per k-tile
    tma_desc_x: cutlass.GridConstant[
        cuda.TensorMap
    ],  # plain: x; tail: latent; gated: a. Box 64 x 8
    tma_desc_x2: cutlass.GridConstant[
        cuda.TensorMap
    ],  # tail: act; gated: the tensor holding g. Box 64 x 8
    y: cutlass.Array,  # [M * N] bf16, token-major
    rms_src: cutlass.Array,  # tail: int32 words of the bf16 [M, rms_cols] latent rows
    num_tokens: cutlass.Int32,
    x_col0: cutlass.Int32,  # tail: column of the latent where k-tile 0 starts
    x2_col0: cutlass.Int32,  # gated: column of g in its tensor
    eps: cutlass.Float32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    variant: cutlass.Constexpr[int],
    x_tiles: cutlass.Constexpr[
        int
    ],  # tail: latent k-tiles (the rest accumulate from x2 into accumulator 1)
    rms_cols: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    push: cutlass.Constexpr[
        bool
    ],  # split-K partials by st.async (test_wait owner) instead of stores + arrive
):
    """One weight tile of 128 rows per cluster; the CTA's k-tiles all resident; B by TMA (gated: rewritten)."""
    k_tiles = num_k_tiles(k_in)
    extra = k_tiles % split
    my_tiles = (k_tiles + split - 1) // split  # stages (the last one unused on ranks >= extra)
    rows_owned = CTA_M // split  # output rows reduced and stored by this CTA
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp_id = cute.arch.warp_idx()
    tma_ptr_w = tma_desc_w.get_ptr()
    tma_ptr_x = tma_desc_x.get_ptr()
    tma_ptr_x2 = tma_desc_x2.get_ptr()
    rank = cutlass.Int32(0)
    if cutlass.const_expr(split > 1):
        rank = cute.arch.block_idx_in_cluster()
    m_offset = (bx // cutlass.Int32(split)) * cutlass.Int32(CTA_M)
    my_count = cutlass.Int32(my_tiles)
    if cutlass.const_expr(extra > 0):
        my_count = cutlass.Int32(k_tiles // split) + cutlass.Int32(
            cutlass.select_(rank < cutlass.Int32(extra), cutlass.Int32(1), cutlass.Int32(0))
        )

    # Allocation order is the same in every CTA, so mapa() addresses the peer's mailbox and barrier.
    smem_a = cutlass.Array(
        io_dtype, my_tiles * CTA_M * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_b = cutlass.Array(
        io_dtype, my_tiles * MMA_N * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_g = smem_b
    if cutlass.const_expr(variant >= GATED):
        smem_g = cutlass.Array(
            io_dtype, my_tiles * MMA_N * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
        )
    stage = cutlass.Array(
        io_dtype, MMA_N * rows_owned, space=cutlass.AddressSpace.smem, alignment=16
    )
    weight_full = cutlass.Array(
        cutlass.Int64, my_tiles, space=cutlass.AddressSpace.smem, alignment=8
    )
    act_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    mail_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    if cutlass.const_expr(variant == TAIL):
        row_scale = cutlass.Array(
            cutlass.Float32, MMA_N, space=cutlass.AddressSpace.smem, alignment=16
        )
    if cutlass.const_expr(split > 1):
        # [source rank][row within the owned block][token] fp32 partials (the own rank's slot stays unused).
        mailbox = cutlass.Array(
            cutlass.Float32,
            split * rows_owned * MMA_N,
            space=cutlass.AddressSpace.smem,
            alignment=16,
        )

    if warp_id == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        prims.prefetch_tensormap(tma_ptr_x)
        if cutlass.const_expr(variant != PLAIN):
            prims.prefetch_tensormap(tma_ptr_x2)
        if prims.elect_sync():
            for i in cutlass.range_constexpr(my_tiles):
                prims.mbarrier_init(weight_full.subview(i), 1)
            prims.mbarrier_init(act_full, 1)
            prims.mbarrier_init(acc_done, 1)
            if cutlass.const_expr(variant >= GATED):
                prims.mbarrier_init(b_ready, EPI_THREADS)
            if cutlass.const_expr(split > 1):
                if cutlass.const_expr(push):
                    # The other ranks' partials of this CTA's rows arrive by st.async (16-byte stores completing
                    # this barrier's transaction count); expected here, before cluster formation.
                    prims.mbarrier_init(mail_full, 1)
                    prims.mbarrier_arrive_expect_tx(mail_full, (split - 1) * rows_owned * MMA_N * 4)
                else:
                    # Every lane of the other ranks' epilogue warps holding this CTA's rows arrives once.
                    prims.mbarrier_init(mail_full, (split - 1) * rows_owned)
    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    if cutlass.const_expr(split > 1):
        # Cluster formation: the peer's shared memory and barriers are addressable from here on.
        prims.barrier_cluster_arrive_relaxed()
        prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    tmem_ptr = cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32)

    if warp_id == 0:
        # =====================================================================
        # Weight TMA: every k-tile of this CTA, ahead of the grid dependency.
        # =====================================================================
        if prims.elect_sync():
            for i in cutlass.range_constexpr(my_tiles):
                k = rank + cutlass.Int32(i * split)
                if cutlass.Int32(i) < my_count:
                    prims.mbarrier_arrive_expect_tx(
                        weight_full.subview(i), CTA_M * CTA_K * ELEM_BYTES
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_a.subview(i * CTA_M * CTA_K),
                        tma_ptr_w,
                        (
                            cutlass.Int32(0),
                            m_offset,
                            k * cutlass.Int32(TMA_COPY_ITERS),
                            cutlass.Int32(0),
                            cutlass.Int32(0),
                        ),
                        weight_full.subview(i),
                        l2_cache_hint=EVICT_FIRST,
                    )
        if cutlass.const_expr(trigger_early):
            # Dependents may launch now; they wait for this whole grid before reading y.
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    elif warp_id == 1:
        # =====================================================================
        # Activation TMA: B (and g), written by predecessors -> after the wait.
        # =====================================================================
        prims.griddepcontrol(prims.GridDepAction.WAIT)
        if prims.elect_sync():
            tile_bytes = MMA_N * CTA_K * ELEM_BYTES
            if cutlass.const_expr(variant >= GATED):
                tile_bytes = 2 * tile_bytes
            prims.mbarrier_arrive_expect_tx(act_full, my_count * cutlass.Int32(tile_bytes))
            for i in cutlass.range_constexpr(my_tiles):
                k = rank + cutlass.Int32(i * split)
                k_c = cutlass.Int32(cutlass.select_(cutlass.Int32(i) < my_count, k, rank))
                for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                    b_off = i * MMA_N * CTA_K + half * B_HALF_ELEMS
                    col = k_c * cutlass.Int32(CTA_K) + cutlass.Int32(half * TMA_K_BOX)
                    if cutlass.const_expr(variant == TAIL):
                        if cutlass.const_expr(i < x_tiles):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                smem_b.subview(b_off),
                                tma_ptr_x,
                                (x_col0 + col, cutlass.Int32(0)),
                                act_full,
                            )
                        else:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                smem_b.subview(b_off),
                                tma_ptr_x2,
                                (
                                    cutlass.Int32((i - x_tiles) * CTA_K + half * TMA_K_BOX),
                                    cutlass.Int32(0),
                                ),
                                act_full,
                            )
                    else:
                        if cutlass.Int32(i) < my_count:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                smem_b.subview(b_off),
                                tma_ptr_x,
                                (x_col0 + col, cutlass.Int32(0)),
                                act_full,
                            )
                            if cutlass.const_expr(variant >= GATED):
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    smem_g.subview(b_off),
                                    tma_ptr_x2,
                                    (x2_col0 + col, cutlass.Int32(0)),
                                    act_full,
                                )
    elif warp_id == 2:
        # =====================================================================
        # MMA: this CTA's k-tiles in ascending order into one TMEM accumulator
        # (tail: the act k-tiles into a second one).
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
        split_acc = variant == TAIL
        tmem_acc1 = tmem_ptr
        if cutlass.const_expr(split_acc):
            tmem_acc1 = cutlass.inttoptr(
                tmem_ptr_i32.load() + cutlass.Int32(MMA_N), 6, cutlass.Int32
            )
        # B is ready when its TMA lands (gated: when the epilogue warps have rewritten it).
        if cutlass.const_expr(variant >= GATED):
            while not cute.arch.mbarrier_try_wait(b_ready.data_ptr(), 0):
                pass
        else:
            while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
                pass
        for i in cutlass.range_constexpr(my_tiles):
            second = split_acc and i >= x_tiles
            first_tile = i == 0 or (split_acc and i == x_tiles)
            if cutlass.Int32(i) < my_count:
                while not cute.arch.mbarrier_try_wait(weight_full.subview(i).data_ptr(), 0):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                    box = kb // K_BLOCKS_PER_HALF
                    within = kb % K_BLOCKS_PER_HALF
                    desc_a = desc_a_base + (i * STAGE_A + box * A_BOX + within * STEP)
                    desc_b = desc_b_base + (i * STAGE_B + box * B_BOX + within * STEP)
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, tmem_acc1 if second else tmem_ptr,
                            desc_a, desc_b, idesc, not (first_tile and kb == 0),
                        )  # fmt: skip
        if prims.elect_sync():
            prims.tcgen05_commit(acc_done)
    elif warp_id >= 4:
        # =====================================================================
        # Epilogue warps: [gate prologue] -> TMEM -> registers -> [split-K
        # reduce] -> bf16 staging -> 16-byte stores.
        #   warp 4 + w reads TMEM lanes 32 w .. 32 w + 31 = tile rows 32 w + lane.
        # =====================================================================
        tid = tx - cutlass.Int32(EPI_THREADS)
        lane = tx % 32
        w = warp_id - 4
        if cutlass.const_expr(variant >= GATED):
            # B = bf16(a * bf16(sigmoid(g))): a and g share one swizzled layout, so chunk c of B is
            # chunk c of both inputs. 16-byte chunks, consecutive threads -> no bank conflicts.
            while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
                pass
            for j in cutlass.range_constexpr(my_tiles * MMA_N * CTA_K // (VEC * EPI_THREADS)):
                c = (tid + cutlass.Int32(j * EPI_THREADS)) * cutlass.Int32(VEC)
                av = smem_b.load(idx=c, vector_size=VEC, alignment=16)
                gv = smem_g.load(idx=c, vector_size=VEC, alignment=16)
                outs = []
                for e in cutlass.range_constexpr(VEC):
                    if cutlass.const_expr(variant == SWIGLU):
                        # silu_and_mul's order in fp32: g * sigmoid(g), then times the up value, one rounding.
                        g = cutlass.Float32(gv[e])
                        one = cutlass.Float32(1.0)
                        sig = _div_rn(one, one + cute.math.exp(-g, fastmath=False))
                        outs.append(((g * sig) * cutlass.Float32(av[e])).to(io_dtype))
                    else:
                        if cutlass.const_expr(variant == GATED):
                            gate = _sigmoid_bf16(cutlass.Float32(gv[e]))
                        else:
                            gate = cutlass.Float32(gv[e])
                        outs.append((cutlass.Float32(av[e]) * gate).to(io_dtype))
                smem_b.store(
                    cutlass.Vector.from_elements(tuple(outs), io_dtype),
                    idx=c,
                    vector_size=VEC,
                    alignment=16,
                )
            # Generic-proxy writes of B -> the tensor core's async-proxy reads, then release the MMA.
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            prims.mbarrier_arrive(b_ready)
        if cutlass.const_expr(variant == TAIL):
            # Per-token RMS of the latent rows while the MMA runs: epilogue warp w owns tokens 2w, 2w+1.
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            for j in cutlass.range_constexpr(MMA_N // 4):
                t = w * cutlass.Int32(MMA_N // 4) + cutlass.Int32(j)
                sum_sq = cutlass.Float32(0.0)
                if t < num_tokens:
                    for v in cutlass.range_constexpr(rms_cols // (8 * 32)):
                        words = rms_src.load(
                            idx=t * cutlass.Int32(rms_cols // 2)
                            + cutlass.Int32(v * 32 * 4)
                            + lane * cutlass.Int32(4),
                            vector_size=4,
                            alignment=16,
                        )
                        for q in cutlass.range_constexpr(4):
                            lo = (words[q] << cutlass.Int32(16)).bitcast(cutlass.Float32)
                            hi = (words[q] & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                            sum_sq = sum_sq + lo * lo + hi * hi
                for offset in (16, 8, 4, 2, 1):
                    sum_sq = sum_sq + cute.arch.shuffle_sync_bfly(sum_sq, offset=offset)
                if lane == 0:
                    row_scale.store(
                        cute.math.rsqrt(sum_sq * cutlass.Float32(1.0 / rms_cols) + eps), idx=t
                    )
            prims.barrier_cta_sync(1, thread_count=EPI_THREADS)

        while not cute.arch.mbarrier_try_wait(acc_done.data_ptr(), 0):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc0 = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Float32), num=MMA_N
        )
        acc1 = acc0
        if cutlass.const_expr(variant == TAIL):
            acc1 = prims.tcgen05_ld(
                "32x32b",
                cutlass.inttoptr(tmem_ptr_i32.load() + cutlass.Int32(MMA_N), 6, cutlass.Float32),
                num=MMA_N,
            )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        vals = []
        for t in cutlass.range_constexpr(MMA_N):
            value = cutlass.Float32(acc0[t])
            if cutlass.const_expr(variant == TAIL):
                value = value * row_scale.load(idx=t) + cutlass.Float32(acc1[t])
            vals.append(value)

        row = w * cutlass.Int32(32) + lane  # tile row of this thread
        local_row = row
        if cutlass.const_expr(split > 1):
            # Rows [R r, R r + R) belong to rank r = w // (4 / SPLIT); the other ranks push, the owner adds.
            owner = w // cutlass.Int32(rows_owned // 32)
            local_row = row - owner * cutlass.Int32(rows_owned)
            if owner != rank:
                slot = (rank * cutlass.Int32(rows_owned) + local_row) * cutlass.Int32(MMA_N)
                if cutlass.const_expr(push):
                    peer_slot = _mapa_u32(mailbox.subview(slot).data_ptr(), owner)
                    mbar_peer = _mapa_u32(mail_full.data_ptr(), owner)
                    _st_async_v4(peer_slot, vals[0], vals[1], vals[2], vals[3], mbar_peer)
                    _st_async_v4(
                        peer_slot + cutlass.Int32(16), vals[4], vals[5], vals[6], vals[7], mbar_peer
                    )
                else:
                    for t in cutlass.range_constexpr(MMA_N):
                        prims.mapa(mailbox.subview(slot + cutlass.Int32(t)), owner).store(vals[t])
                    # Release at cluster scope: the partials above are visible to the owner's acquire.
                    prims.mbarrier_arrive(
                        prims.mapa(mail_full, owner), scope=prims.MemScope.CLUSTER
                    )
            else:
                if cutlass.const_expr(push):
                    # test_wait spin: a warp suspended in try_wait on a barrier completed by peers wakes late.
                    # Acquire at cluster scope: the bytes come from other CTAs' st.async (complete_tx releases at
                    # cluster scope).
                    while not _test_wait_cluster(mail_full.data_ptr(), 0):
                        pass
                else:
                    while not prims.mbarrier_try_wait_parity(
                        mail_full, 0, scope=prims.MBarrierScope.CLUSTER
                    ):
                        pass
                # The SPLIT partials in rank order (this rank's own from registers).
                summed = [cutlass.Float32(0.0)] * MMA_N
                for q in cutlass.range_constexpr(split):
                    slot = (cutlass.Int32(q * rows_owned) + local_row) * cutlass.Int32(MMA_N)
                    for t in cutlass.range_constexpr(MMA_N):
                        part = mailbox.load(idx=slot + cutlass.Int32(t))
                        summed[t] = summed[t] + cutlass.Float32(
                            cutlass.select_(rank == cutlass.Int32(q), vals[t], part)
                        )
                for t in cutlass.range_constexpr(MMA_N):
                    stage.store(
                        summed[t].to(io_dtype), idx=cutlass.Int32(t * rows_owned) + local_row
                    )
        else:
            for t in cutlass.range_constexpr(MMA_N):
                stage.store(vals[t].to(io_dtype), idx=cutlass.Int32(t * rows_owned) + local_row)
        # Every TMEM reader has waited for its load; the staging tile is complete.
        prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
        if warp_id == 4:
            prims.tcgen05_dealloc(tmem_ptr, TMEM_COLS)
        # [token][rows_owned] bf16 -> y[token, rows]: 16 bytes per thread, rows contiguous per token.
        chunks_per_token = rows_owned // VEC
        ct = tid // cutlass.Int32(chunks_per_token)
        cc = tid % cutlass.Int32(chunks_per_token)
        if ct < num_tokens:
            if ct < cutlass.Int32(MMA_N):
                v = stage.load(
                    idx=ct * cutlass.Int32(rows_owned) + cc * cutlass.Int32(VEC),
                    vector_size=VEC,
                    alignment=16,
                )
                y.store(
                    v,
                    idx=ct * cutlass.Int32(n_out)
                    + m_offset
                    + rank * cutlass.Int32(rows_owned)
                    + cc * cutlass.Int32(VEC),
                    vector_size=VEC,
                    alignment=16,
                )


# =============================================================================
# Long-K variant (K > 6 k-tiles, e.g. the MLA [W_a; W_g] projection 2880 x 7168): one 128-row tile per cluster
# of SPLIT >= 4 CTAs, rank r streaming the interleaved k-tiles r, r + SPLIT, ... (the first `extra` ranks one
# more when SPLIT does not divide the k-tiles) through a RING-stage ring filled before griddepcontrol.wait,
# the rest of its k-tiles prefetched into L2 at the same time, so a latency-bound predecessor (an all-reduce)
# hides the whole weight read. B (all of the rank's k-tiles of x) is resident. Rows [32 w, 32 w + 32) of the
# tile belong to rank w < 4 (epilogue warp w's TMEM lanes; with SPLIT 2, rank w // 2 owns two such blocks); every
# other rank sends its fp32 partials of them to slot [rank] of the owner's mailbox (PUSH as in the short kernel:
# DSMEM stores + a release arrive at cluster scope, or st.async completing the barrier by bytes); the owner adds
# the SPLIT partials in rank order and rounds once. Output rows >= sig_row0 store bf16(sigmoid(bf16(acc))) instead
# of bf16(acc): the MLA output gate, ready for GATED_S.
# =============================================================================
LONG_ROWS_PER_WARP = 32


def long_supports(n_out: int, k_in: int, split: int, ring: int) -> bool:
    """K in whole 64-column halves: a last k-tile of 64 columns has its second half past K, which the TMA fills with
    zeros in both W and x (e.g. the dense MLP's down projection, K = 2112 at TP16)."""
    k_tiles = num_k_tiles(k_in)
    return (
        k_in % TMA_K_BOX == 0
        and (split == 2 or 4 <= split <= 8)
        and 1 <= ring <= k_tiles // split
        and ring * CTA_M * CTA_K * ELEM_BYTES + (k_tiles // split + 1) * MMA_N * CTA_K * ELEM_BYTES
        <= 216 * 1024
        and n_out > 0
    )


@cute.kernel
def k3_ctm_gemv_long_kernel(
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],  # W [N, K] bf16, 5-D, one call per k-tile
    tma_desc_x: cutlass.GridConstant[cuda.TensorMap],  # x [M, K] bf16, box 64 x 8
    y: cutlass.Array,  # [M * N] bf16, token-major
    num_tokens: cutlass.Int32,
    sig_row0: cutlass.Int32,  # rows >= sig_row0 store bf16(sigmoid(bf16(acc))); n_out: none
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    push: cutlass.Constexpr[
        bool
    ],  # split-K partials by st.async (test_wait owner) instead of stores + arrive
):
    """One 128-row tile per cluster of `split` CTAs; weight ring + L2 prefetch before the grid wait."""
    k_tiles = num_k_tiles(k_in)
    extra = k_tiles % split
    max_tiles = k_tiles // split + (1 if extra > 0 else 0)
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp_id = cute.arch.warp_idx()
    rank = cute.arch.block_idx_in_cluster()
    tma_ptr_w = tma_desc_w.get_ptr()
    tma_ptr_x = tma_desc_x.get_ptr()
    m_offset = (bx // cutlass.Int32(split)) * cutlass.Int32(CTA_M)
    my_tiles = cutlass.Int32(k_tiles // split)
    if cutlass.const_expr(extra > 0):
        if rank < cutlass.Int32(extra):
            my_tiles = my_tiles + cutlass.Int32(1)

    smem_a = cutlass.Array(
        io_dtype, ring * CTA_M * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_b = cutlass.Array(
        io_dtype, max_tiles * MMA_N * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    tma_full = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    mma_done = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    act_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    mail_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # [source rank][owned block][row in the block][token] fp32 partials (the own rank's slots stay unused). A rank owns
    # one 32-row block with SPLIT >= 4, two with SPLIT 2.
    blocks = 4 // split if split < 4 else 1
    mailbox = cutlass.Array(
        cutlass.Float32,
        split * blocks * LONG_ROWS_PER_WARP * MMA_N,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )

    if warp_id == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        prims.prefetch_tensormap(tma_ptr_x)
        if prims.elect_sync():
            for s in cutlass.range_constexpr(ring):
                prims.mbarrier_init(tma_full.subview(s), 1)
                prims.mbarrier_init(mma_done.subview(s), 1)
            prims.mbarrier_init(act_full, 1)
            prims.mbarrier_init(acc_done, 1)
            if cutlass.const_expr(push):
                # Owner ranks (< 4): the other ranks' partials of the owned rows arrive by st.async (completing
                # the transaction count expected here, before cluster formation); other ranks never wait on it.
                prims.mbarrier_init(mail_full, 1)
                prims.mbarrier_arrive_expect_tx(
                    mail_full, (split - 1) * blocks * LONG_ROWS_PER_WARP * MMA_N * 4
                )
            else:
                # Owner ranks: every lane of the other ranks' epilogue warps of the owned blocks arrives once.
                prims.mbarrier_init(mail_full, (split - 1) * blocks * LONG_ROWS_PER_WARP)
    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    # Cluster formation: the peers' shared memory and barriers are addressable from here on.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    tmem_ptr = cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32)

    if warp_id == 0:
        # =====================================================================
        # Weight TMA: fill the ring and prefetch the rest into L2 before the
        # grid dependency; then refill each stage when its MMAs are done.
        # =====================================================================
        if prims.elect_sync():
            for i in cutlass.range_constexpr(ring):
                k = rank + cutlass.Int32(i * split)
                prims.mbarrier_arrive_expect_tx(tma_full.subview(i), CTA_M * CTA_K * ELEM_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a.subview(i * CTA_M * CTA_K),
                    tma_ptr_w,
                    (
                        cutlass.Int32(0),
                        m_offset,
                        k * cutlass.Int32(TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ),
                    tma_full.subview(i),
                    l2_cache_hint=EVICT_FIRST,
                )
            for i in range(ring, my_tiles):
                k = rank + i * cutlass.Int32(split)
                prims.cp_async_bulk_tensor_prefetch(
                    tma_ptr_w,
                    [
                        cutlass.Int32(0),
                        m_offset,
                        k * cutlass.Int32(TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ],
                    [],  # tile mode: no im2col offsets
                )
        if cutlass.const_expr(trigger_early):
            # Dependents may launch now; they wait for this whole grid before reading y.
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        if prims.elect_sync():
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)
            for i in range(ring, my_tiles):
                while not cute.arch.mbarrier_try_wait(mma_done.subview(stage).data_ptr(), phase):
                    pass
                k = rank + i * cutlass.Int32(split)
                prims.mbarrier_arrive_expect_tx(tma_full.subview(stage), CTA_M * CTA_K * ELEM_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a.subview(stage * cutlass.Int32(CTA_M * CTA_K)),
                    tma_ptr_w,
                    (
                        cutlass.Int32(0),
                        m_offset,
                        k * cutlass.Int32(TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ),
                    tma_full.subview(stage),
                    l2_cache_hint=EVICT_FIRST,
                )
                stage = stage + cutlass.Int32(1)
                if stage == cutlass.Int32(ring):
                    stage = cutlass.Int32(0)
                    phase = phase ^ cutlass.Int32(1)
    elif warp_id == 1:
        # =====================================================================
        # Activation TMA: the rank's k-tiles of x, resident, after the wait.
        # =====================================================================
        prims.griddepcontrol(prims.GridDepAction.WAIT)
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(
                act_full, my_tiles * cutlass.Int32(MMA_N * CTA_K * ELEM_BYTES)
            )
            for i in range(my_tiles):
                k = rank + i * cutlass.Int32(split)
                for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_b.subview(
                            i * cutlass.Int32(MMA_N * CTA_K) + cutlass.Int32(half * B_HALF_ELEMS)
                        ),
                        tma_ptr_x,
                        (
                            k * cutlass.Int32(CTA_K) + cutlass.Int32(half * TMA_K_BOX),
                            cutlass.Int32(0),
                        ),
                        act_full,
                    )
    elif warp_id == 2:
        # =====================================================================
        # MMA: the rank's k-tiles in order, one TMEM accumulator (partial sum).
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
        while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
            pass
        stage = cutlass.Int32(0)
        phase = cutlass.Int32(0)
        for i in range(my_tiles):
            while not cute.arch.mbarrier_try_wait(tma_full.subview(stage).data_ptr(), phase):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                box = kb // K_BLOCKS_PER_HALF
                within = kb % K_BLOCKS_PER_HALF
                desc_a = desc_a_base + (
                    stage * cutlass.Int32(STAGE_A) + cutlass.Int32(box * A_BOX + within * STEP)
                )
                desc_b = desc_b_base + (
                    i * cutlass.Int32(STAGE_B) + cutlass.Int32(box * B_BOX + within * STEP)
                )
                accumulate = cutlass.Boolean(True)
                if cutlass.const_expr(kb == 0):
                    accumulate = i > cutlass.Int32(0)
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16,
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        desc_a,
                        desc_b,
                        idesc,
                        accumulate,
                    )
            if prims.elect_sync():
                prims.tcgen05_commit(mma_done.subview(stage))
            stage = stage + cutlass.Int32(1)
            if stage == cutlass.Int32(ring):
                stage = cutlass.Int32(0)
                phase = phase ^ cutlass.Int32(1)
        if prims.elect_sync():
            prims.tcgen05_commit(acc_done)
    elif warp_id >= 4:
        # =====================================================================
        # Epilogue: TMEM -> registers, push to / reduce at the row owner, store.
        # =====================================================================
        lane = tx % 32
        w = warp_id - 4  # TMEM lanes 32 w .. 32 w + 31: tile rows owned by rank w
        while not cute.arch.mbarrier_try_wait(acc_done.data_ptr(), 0):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Float32), num=MMA_N
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        slot_row = lane * cutlass.Int32(MMA_N)
        owner = w // cutlass.Int32(blocks)
        block_slot = (w % cutlass.Int32(blocks)) * cutlass.Int32(
            LONG_ROWS_PER_WARP * MMA_N
        ) + slot_row
        if owner != rank:
            base = rank * cutlass.Int32(blocks * LONG_ROWS_PER_WARP * MMA_N) + block_slot
            if cutlass.const_expr(push):
                peer_slot = _mapa_u32(mailbox.subview(base).data_ptr(), owner)
                mbar_peer = _mapa_u32(mail_full.data_ptr(), owner)
                _st_async_v4(peer_slot, acc[0], acc[1], acc[2], acc[3], mbar_peer)
                _st_async_v4(
                    peer_slot + cutlass.Int32(16), acc[4], acc[5], acc[6], acc[7], mbar_peer
                )
            else:
                for t in cutlass.range_constexpr(MMA_N):
                    prims.mapa(mailbox.subview(base + cutlass.Int32(t)), owner).store(
                        cutlass.Float32(acc[t])
                    )
                # Release at cluster scope: this lane's partials are visible to the owner's acquire.
                prims.mbarrier_arrive(prims.mapa(mail_full, owner), scope=prims.MemScope.CLUSTER)
        else:
            if cutlass.const_expr(push):
                # test_wait spin: a warp suspended in try_wait on a barrier completed by peers wakes late.
                # Acquire at cluster scope: the bytes come from other CTAs' st.async (complete_tx releases at
                # cluster scope).
                while not _test_wait_cluster(mail_full.data_ptr(), 0):
                    pass
            else:
                while not prims.mbarrier_try_wait_parity(
                    mail_full, 0, scope=prims.MBarrierScope.CLUSTER
                ):
                    pass
            n = m_offset + w * cutlass.Int32(LONG_ROWS_PER_WARP) + lane
            for t in cutlass.range_constexpr(MMA_N):
                total = cutlass.Float32(0.0)
                for q in cutlass.range_constexpr(split):
                    part = mailbox.load(
                        idx=cutlass.Int32(q * blocks * LONG_ROWS_PER_WARP * MMA_N)
                        + block_slot
                        + cutlass.Int32(t)
                    )
                    total = total + cutlass.Float32(
                        cutlass.select_(rank == cutlass.Int32(q), cutlass.Float32(acc[t]), part)
                    )
                out = total.to(io_dtype)
                if n >= sig_row0:
                    out = _sigmoid_bf16(_bf16_rn(total)).to(io_dtype)
                if cutlass.Int32(t) < num_tokens:
                    if n < cutlass.Int32(n_out):
                        y.store(out, idx=cutlass.Int32(t * n_out) + n)
        prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
        if warp_id == 4:
            prims.tcgen05_dealloc(tmem_ptr, TMEM_COLS)


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


def _launch(kernel, n_out, split, use_pdl, stream):
    kernel.launch(
        grid=((n_out // CTA_M) * split, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(split, 1, 1) if split > 1 else None,
        stream=stream,
        use_pdl=use_pdl,
    )


@cute.jit
def k3_ctm_gemv(
    w: cute.Tensor,  # [N, K] bf16, K contiguous
    x: cute.Tensor,  # [M, K] bf16, K contiguous, M <= 8
    y: cute.Tensor,  # [M * N] bf16
    num_tokens: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    push: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """``y = x @ w^T``."""
    tma_desc_w = _weight_tensor_map(w, n_out, k_in)
    tma_desc_x = _activation_tensor_map(x, k_in, num_tokens)
    kernel = k3_ctm_gemv_kernel(
        tma_desc_w, tma_desc_x, tma_desc_x, y, y, num_tokens, cutlass.Int32(0), cutlass.Int32(0),
        cutlass.Float32(0.0), n_out, k_in, PLAIN, num_k_tiles(k_in), 0, split, trigger_early, push,
    )  # fmt: skip
    _launch(kernel, n_out, split, use_pdl, stream)


@cute.jit
def k3_ctm_gemv_tail(
    w: cute.Tensor,  # [N, K_lat + K_act] bf16: latent-up columns of this rank's slice (zero-padded) | shared down
    latent: cute.Tensor,  # [M, rms_cols] bf16, the whole reduced latent row
    latent_words: cute.Tensor,  # the same memory as int32 words, for the RMS
    act: cute.Tensor,  # [M, K_act] bf16, the shared-expert activation
    y: cute.Tensor,  # [M * N] bf16
    num_tokens: cutlass.Int32,
    lat_col0: cutlass.Int32,  # first latent column of this rank's slice
    eps: cutlass.Float32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    lat_tiles: cutlass.Constexpr[int],
    rms_cols: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """``y = rmsnorm(latent)[:, slice] @ W_lat^T + act @ W_act^T`` (the RMS applied to the latent accumulator)."""
    tma_desc_w = _weight_tensor_map(w, n_out, k_in)
    tma_desc_lat = _activation_tensor_map(latent, rms_cols, num_tokens)
    tma_desc_act = _activation_tensor_map(act, k_in - lat_tiles * CTA_K, num_tokens)
    kernel = k3_ctm_gemv_kernel(
        tma_desc_w, tma_desc_lat, tma_desc_act, y, latent_words, num_tokens, lat_col0, cutlass.Int32(0), eps,
        n_out, k_in, TAIL, lat_tiles, rms_cols, 1, trigger_early, False,
    )  # fmt: skip
    _launch(kernel, n_out, 1, use_pdl, stream)


@cute.jit
def k3_ctm_gemv_gated(
    w: cute.Tensor,  # [N, K] bf16 (the o_proj weight), K contiguous
    a: cute.Tensor,  # [M, K] bf16, the attention output (o_proj's input before the gate)
    gsrc: cute.Tensor,  # [M, gsrc_cols] bf16, dense rows; g = gsrc[:, g_col0 : g_col0 + K]
    y: cute.Tensor,  # [M * N] bf16
    num_tokens: cutlass.Int32,
    g_col0: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    gsrc_cols: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    gate_sigmoid: cutlass.Constexpr[bool],  # True: gsrc holds g; False: it holds bf16(sigmoid(g))
    trigger_early: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """``y = (a * sigmoid(g)) @ w^T`` with torch's bf16 roundings of the sigmoid and of the product."""
    tma_desc_w = _weight_tensor_map(w, n_out, k_in)
    tma_desc_a = _activation_tensor_map(a, k_in, num_tokens)
    tma_desc_g = _activation_tensor_map(gsrc, gsrc_cols, num_tokens)
    kernel = k3_ctm_gemv_kernel(
        tma_desc_w, tma_desc_a, tma_desc_g, y, y, num_tokens, cutlass.Int32(0), g_col0, cutlass.Float32(0.0),
        n_out, k_in, GATED if gate_sigmoid else GATED_S, num_k_tiles(k_in), 0, split, trigger_early, False,
    )  # fmt: skip
    _launch(kernel, n_out, split, use_pdl, stream)


@cute.jit
def k3_ctm_gemv_swiglu(
    w: cute.Tensor,  # [N, K] bf16 (the down projection), K contiguous
    gu: cute.Tensor,  # [M, 2 K] bf16, dense rows: the gate_up output (gate columns first)
    y: cute.Tensor,  # [M * N] bf16
    num_tokens: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    push: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """``y = (silu(gu[:, :K]) * gu[:, K:]) @ w^T`` with silu_and_mul's fp32 arithmetic and bf16 rounding."""
    tma_desc_w = _weight_tensor_map(w, n_out, k_in)
    tma_desc_gu = _activation_tensor_map(gu, 2 * k_in, num_tokens)
    kernel = k3_ctm_gemv_kernel(
        tma_desc_w, tma_desc_gu, tma_desc_gu, y, y, num_tokens, cutlass.Int32(k_in), cutlass.Int32(0),
        cutlass.Float32(0.0), n_out, k_in, SWIGLU, num_k_tiles(k_in), 0, split, trigger_early, push,
    )  # fmt: skip
    _launch(kernel, n_out, split, use_pdl, stream)


@cute.jit
def k3_ctm_gemv_long(
    w: cute.Tensor,  # [N, K] bf16, K contiguous
    x: cute.Tensor,  # [M, K] bf16, K contiguous, M <= 8
    y: cute.Tensor,  # [M * N] bf16
    num_tokens: cutlass.Int32,
    sig_row0: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    push: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """``y = x @ w^T``; output rows >= sig_row0 hold bf16(sigmoid(bf16(.))) instead."""
    tma_desc_w = _weight_tensor_map(w, n_out, k_in)
    tma_desc_x = _activation_tensor_map(x, k_in, num_tokens)
    k3_ctm_gemv_long_kernel(
        tma_desc_w,
        tma_desc_x,
        y,
        num_tokens,
        sig_row0,
        n_out,
        k_in,
        split,
        ring,
        trigger_early,
        push,
    ).launch(
        grid=(((n_out + CTA_M - 1) // CTA_M) * split, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(split, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


# =============================================================================
# Wide variant (up to 64 tokens: DSpark verify of several requests): the long kernel's geometry, weight ring and L2
# prefetch, with all the call's token columns in one MMA of N = N_TILE (16, 32 or 64). x no longer fits resident
# beside the ring, so each ring stage holds a weight k-tile and the matching x k-tile: the weight half is loaded
# before griddepcontrol.wait (the first RING stages; the rest prefetched into L2), the x half after it, each refilled
# once the stage's MMAs are done.
# Split-K reduction by token: rank r owns tokens [r T, r T + T) of the tile (T = N_TILE / SPLIT rounded up to 4) for
# all 128 rows. Every epilogue thread (one row) sends each other owner its partials of that owner's tokens by st.async
# (16-byte chunks of 4 tokens, chunk j at j ^ f(lane) so a quarter-warp's 8 rows fall in 8 bank groups), writes its
# own into its own slot, then sums its row's T tokens over the SPLIT slots in rank order: every token's sums are the
# long kernel's (same k-tile order per rank, same rank order). Each warp stages its 32 rows as [token][row] and stores
# them with one TMA store (tokens >= M clipped; the block holding row N - 1 when N is not a multiple of 32 has its own
# box): bf16 in the x ring (free once the MMAs are done; rows >= sig_row0 as bf16(sigmoid(bf16(acc)))), or fp32
# (OUT_FP32: the MoE head's router logits) in place of its own slot.
# =============================================================================
WIDE_TILES = (16, 32, 64)
WIDE_SMEM_BYTES = 220 * 1024  # the ring, the x stages and the mailbox; the barriers fit in the rest


def wide_tile(num_tokens: int) -> int:
    """The token tile (MMA N) that holds ``num_tokens`` columns."""
    for n_tile in WIDE_TILES:
        if num_tokens <= n_tile:
            return n_tile
    raise ValueError(f"k3_ctm_gemv_wide: {num_tokens} tokens exceed {WIDE_TILES[-1]}")


def wide_owned_tokens(n_tile: int, split: int) -> int:
    """Tokens of the tile a rank reduces and stores (the last owner may have fewer, later ranks none)."""
    return (n_tile + split - 1) // split + 3 & ~3


def wide_smem_bytes(split: int, ring: int, n_tile: int, x_ring: int) -> int:
    return (
        ring * CTA_M + x_ring * n_tile
    ) * CTA_K * ELEM_BYTES + split * CTA_M * wide_owned_tokens(n_tile, split) * 4


def wide_supports(n_out: int, k_in: int, split: int, ring: int, n_tile: int, x_ring: int) -> bool:
    """K in whole 64-column halves (a last half k-tile reads zeros past K, as in the long kernel); N a multiple of 8
    (the output rows of a token are a whole number of 16-byte units for the TMA store)."""
    return (
        k_in % TMA_K_BOX == 0
        and n_out % 8 == 0
        and n_tile in WIDE_TILES
        and (split == 2 or 4 <= split <= 8)
        and 1 <= ring <= num_k_tiles(k_in) // split
        and 1 <= x_ring <= ring
        and wide_smem_bytes(split, ring, n_tile, x_ring) <= WIDE_SMEM_BYTES
        and n_out > 0
    )


def _chunk_swizzle(lane, chunks: int):
    """XOR for the 16-byte chunk index of a row of ``chunks`` chunks: the 8 rows of a quarter-warp in 8 bank groups."""
    if chunks == 2:
        return (lane >> cutlass.Int32(2)) & cutlass.Int32(1)
    if chunks == 4:
        return (lane >> cutlass.Int32(1)) & cutlass.Int32(3)
    if chunks >= 8:
        return lane & cutlass.Int32(7)
    return cutlass.Int32(0)  # 1 or 3 chunks: rows already in distinct bank groups


@cute.kernel
def k3_ctm_gemv_wide_kernel(
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],  # W [N, K] bf16, 5-D, one call per k-tile
    tma_desc_x: cutlass.GridConstant[cuda.TensorMap],  # x [M, K] bf16, box 64 x n_tile
    tma_desc_y: cutlass.GridConstant[
        cuda.TensorMap
    ],  # y [M, N] bf16 (fp32 with out_fp32), box 32 x T
    tma_desc_y_tail: cutlass.GridConstant[cuda.TensorMap],  # the same, box (N % 32 or 32) x T
    sig_row0: cutlass.Int32,  # bf16 output: rows >= sig_row0 store bf16(sigmoid(bf16(acc))); n_out: none
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    x_ring: cutlass.Constexpr[int],
    n_tile: cutlass.Constexpr[int],
    out_fp32: cutlass.Constexpr[bool],
    trigger_early: cutlass.Constexpr[bool],
):
    """One 128-row tile per cluster of `split` CTAs; a weight ring filled before the grid wait, an x ring after it."""
    k_tiles = num_k_tiles(k_in)
    extra = k_tiles % split
    tmem_cols = max(TMEM_COLS, n_tile)
    b_stage = n_tile * CTA_K  # elements of one x k-tile
    b_half = n_tile * TMA_K_BOX
    tail_rows = n_out % LONG_ROWS_PER_WARP or LONG_ROWS_PER_WARP
    owned = wide_owned_tokens(n_tile, split)  # T
    slot_elems = CTA_M * owned  # one source rank's partials of the owner's tokens: [row][T]
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp_id = cute.arch.warp_idx()
    rank = cute.arch.block_idx_in_cluster()
    tma_ptr_w = tma_desc_w.get_ptr()
    tma_ptr_x = tma_desc_x.get_ptr()
    tma_ptr_y = tma_desc_y.get_ptr()
    tma_ptr_y_tail = tma_desc_y_tail.get_ptr()
    m_offset = (bx // cutlass.Int32(split)) * cutlass.Int32(CTA_M)
    my_tiles = cutlass.Int32(k_tiles // split)
    if cutlass.const_expr(extra > 0):
        if rank < cutlass.Int32(extra):
            my_tiles = my_tiles + cutlass.Int32(1)
    # The tokens this rank owns: [rank T, rank T + my_owned).
    my_owned = cutlass.Int32(n_tile) - rank * cutlass.Int32(owned)
    if my_owned > cutlass.Int32(owned):
        my_owned = cutlass.Int32(owned)

    smem_a = cutlass.Array(
        io_dtype, ring * CTA_M * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_b = cutlass.Array(
        io_dtype, x_ring * b_stage, space=cutlass.AddressSpace.smem, alignment=1024
    )
    a_full = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    b_full = cutlass.Array(cutlass.Int64, x_ring, space=cutlass.AddressSpace.smem, alignment=8)
    mma_done = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    x_done = cutlass.Array(cutlass.Int64, x_ring, space=cutlass.AddressSpace.smem, alignment=8)
    acc_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    mail_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # [source rank][row][T tokens, chunks swizzled] fp32 partials of this rank's tokens (its own slot written locally).
    mailbox = cutlass.Array(
        cutlass.Float32, split * slot_elems, space=cutlass.AddressSpace.smem, alignment=128
    )

    if warp_id == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        prims.prefetch_tensormap(tma_ptr_x)
        prims.prefetch_tensormap(tma_ptr_y)
        prims.prefetch_tensormap(tma_ptr_y_tail)
        if prims.elect_sync():
            for s in cutlass.range_constexpr(ring):
                prims.mbarrier_init(a_full.subview(s), 1)
                prims.mbarrier_init(mma_done.subview(s), 1)
            for s in cutlass.range_constexpr(x_ring):
                prims.mbarrier_init(b_full.subview(s), 1)
                prims.mbarrier_init(x_done.subview(s), 1)
            prims.mbarrier_init(acc_done, 1)
            # The other ranks' partials of this rank's tokens arrive by st.async, completing the transaction count
            # expected here, before cluster formation (a rank owning no tokens never waits on it).
            prims.mbarrier_init(mail_full, 1)
            if my_owned > cutlass.Int32(0):
                prims.mbarrier_arrive_expect_tx(
                    mail_full, cutlass.Int32((split - 1) * CTA_M * 4) * my_owned
                )
    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, tmem_cols)
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    # Cluster formation: the peers' shared memory and barriers are addressable from here on.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    tmem_ptr = cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32)

    if warp_id == 0:
        # =====================================================================
        # Weight TMA: fill the ring and prefetch the rest into L2 before the
        # grid dependency; then refill each stage when its MMAs are done.
        # =====================================================================
        if prims.elect_sync():
            for i in cutlass.range_constexpr(ring):
                k = rank + cutlass.Int32(i * split)
                prims.mbarrier_arrive_expect_tx(a_full.subview(i), CTA_M * CTA_K * ELEM_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a.subview(i * CTA_M * CTA_K),
                    tma_ptr_w,
                    (
                        cutlass.Int32(0),
                        m_offset,
                        k * cutlass.Int32(TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ),
                    a_full.subview(i),
                    l2_cache_hint=EVICT_FIRST,
                )
            for i in range(ring, my_tiles):
                k = rank + i * cutlass.Int32(split)
                prims.cp_async_bulk_tensor_prefetch(
                    tma_ptr_w,
                    [
                        cutlass.Int32(0),
                        m_offset,
                        k * cutlass.Int32(TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ],
                    [],  # tile mode: no im2col offsets
                )
        if cutlass.const_expr(trigger_early):
            # Dependents may launch now; they wait for this whole grid before reading y.
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        if prims.elect_sync():
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)
            for i in range(ring, my_tiles):
                while not cute.arch.mbarrier_try_wait(mma_done.subview(stage).data_ptr(), phase):
                    pass
                k = rank + i * cutlass.Int32(split)
                prims.mbarrier_arrive_expect_tx(a_full.subview(stage), CTA_M * CTA_K * ELEM_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a.subview(stage * cutlass.Int32(CTA_M * CTA_K)),
                    tma_ptr_w,
                    (
                        cutlass.Int32(0),
                        m_offset,
                        k * cutlass.Int32(TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ),
                    a_full.subview(stage),
                    l2_cache_hint=EVICT_FIRST,
                )
                stage = stage + cutlass.Int32(1)
                if stage == cutlass.Int32(ring):
                    stage = cutlass.Int32(0)
                    phase = phase ^ cutlass.Int32(1)
    elif warp_id == 1:
        # =====================================================================
        # Activation TMA after the wait: the x k-tile of each weight k-tile
        # into the x ring, refilled once the stage's MMAs are done.
        # =====================================================================
        prims.griddepcontrol(prims.GridDepAction.WAIT)
        if prims.elect_sync():
            for i in cutlass.range_constexpr(x_ring):
                k = rank + cutlass.Int32(i * split)
                prims.mbarrier_arrive_expect_tx(b_full.subview(i), b_stage * ELEM_BYTES)
                for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_b.subview(i * b_stage + half * b_half),
                        tma_ptr_x,
                        (
                            k * cutlass.Int32(CTA_K) + cutlass.Int32(half * TMA_K_BOX),
                            cutlass.Int32(0),
                        ),
                        b_full.subview(i),
                    )
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)
            for i in range(x_ring, my_tiles):
                while not cute.arch.mbarrier_try_wait(x_done.subview(stage).data_ptr(), phase):
                    pass
                k = rank + i * cutlass.Int32(split)
                prims.mbarrier_arrive_expect_tx(b_full.subview(stage), b_stage * ELEM_BYTES)
                for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_b.subview(
                            stage * cutlass.Int32(b_stage) + cutlass.Int32(half * b_half)
                        ),
                        tma_ptr_x,
                        (
                            k * cutlass.Int32(CTA_K) + cutlass.Int32(half * TMA_K_BOX),
                            cutlass.Int32(0),
                        ),
                        b_full.subview(stage),
                    )
                stage = stage + cutlass.Int32(1)
                if stage == cutlass.Int32(x_ring):
                    stage = cutlass.Int32(0)
                    phase = phase ^ cutlass.Int32(1)
    elif warp_id == 2:
        # =====================================================================
        # MMA: the rank's k-tiles in order, one TMEM accumulator (partial sum)
        # of N = n_tile token columns.
        # =====================================================================
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=io_dtype, b_dtype=io_dtype, n_dim=n_tile, m_dim=CTA_M
        )
        desc_a_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_a, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        desc_b_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_b, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        stage = cutlass.Int32(0)
        phase = cutlass.Int32(0)
        x_stage = cutlass.Int32(0)
        x_phase = cutlass.Int32(0)
        for i in range(my_tiles):
            while not cute.arch.mbarrier_try_wait(a_full.subview(stage).data_ptr(), phase):
                pass
            while not cute.arch.mbarrier_try_wait(b_full.subview(x_stage).data_ptr(), x_phase):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                box = kb // K_BLOCKS_PER_HALF
                within = kb % K_BLOCKS_PER_HALF
                desc_a = desc_a_base + (
                    stage * cutlass.Int32(STAGE_A) + cutlass.Int32(box * A_BOX + within * STEP)
                )
                desc_b = desc_b_base + (
                    x_stage * cutlass.Int32(b_stage >> 3)
                    + cutlass.Int32(box * (b_half >> 3) + within * STEP)
                )
                accumulate = cutlass.Boolean(True)
                if cutlass.const_expr(kb == 0):
                    accumulate = i > cutlass.Int32(0)
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16,
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        desc_a,
                        desc_b,
                        idesc,
                        accumulate,
                    )
            if prims.elect_sync():
                prims.tcgen05_commit(mma_done.subview(stage))
                prims.tcgen05_commit(x_done.subview(x_stage))
            stage = stage + cutlass.Int32(1)
            if stage == cutlass.Int32(ring):
                stage = cutlass.Int32(0)
                phase = phase ^ cutlass.Int32(1)
            x_stage = x_stage + cutlass.Int32(1)
            if x_stage == cutlass.Int32(x_ring):
                x_stage = cutlass.Int32(0)
                x_phase = x_phase ^ cutlass.Int32(1)
        if prims.elect_sync():
            prims.tcgen05_commit(acc_done)
    elif warp_id >= 4:
        # =====================================================================
        # Epilogue: TMEM -> registers; partials of each owner's tokens to that
        # owner (own into the own slot); sum this rank's tokens in rank order;
        # stage the warp's rows, one TMA store.
        # =====================================================================
        lane = tx % 32
        w = warp_id - 4  # TMEM lanes 32 w .. 32 w + 31: tile rows 32 w + lane
        row = w * cutlass.Int32(LONG_ROWS_PER_WARP) + lane
        while not cute.arch.mbarrier_try_wait(acc_done.data_ptr(), 0):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Float32), num=n_tile
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        swz = _chunk_swizzle(lane, owned // 4)
        row_slot = row * cutlass.Int32(owned)
        peer_slots = []
        peer_mbars = []
        for r in cutlass.range_constexpr(split):
            peer_slots.append(
                _mapa_u32(
                    mailbox.subview(rank * cutlass.Int32(slot_elems) + row_slot).data_ptr(),
                    cutlass.Int32(r),
                )
            )
            peer_mbars.append(_mapa_u32(mail_full.data_ptr(), cutlass.Int32(r)))
        # Chunk j for every owner in turn, so the owners' incoming traffic interleaves; the own chunks to the own slot.
        for j in cutlass.range_constexpr(owned // 4):
            for r in cutlass.range_constexpr(split):
                if cutlass.const_expr(r * owned + 4 * j < n_tile):
                    t = r * owned + 4 * j
                    if rank == cutlass.Int32(r):
                        own = (
                            cutlass.Int32(r * slot_elems)
                            + row_slot
                            + ((cutlass.Int32(j) ^ swz) << cutlass.Int32(2))
                        )
                        mailbox.store(
                            cutlass.Vector.from_elements(
                                tuple(cutlass.Float32(acc[t + e]) for e in range(4)), cutlass.Float32
                            ),
                            idx=own,
                            vector_size=4,
                            alignment=16,
                        )  # fmt: skip
                    else:
                        _st_async_v4(
                            peer_slots[r] + ((cutlass.Int32(j) ^ swz) << cutlass.Int32(4)),
                            acc[t], acc[t + 1], acc[t + 2], acc[t + 3], peer_mbars[r],
                        )  # fmt: skip
        if my_owned > cutlass.Int32(0):
            # test_wait spin: a warp suspended in try_wait on a barrier completed by peers wakes late.
            # Acquire at cluster scope: the bytes come from other CTAs' st.async (complete_tx releases at
            # cluster scope).
            while not _test_wait_cluster(mail_full.data_ptr(), 0):
                pass
            # This row's owned tokens over the SPLIT slots in rank order, as the long kernel.
            totals = [cutlass.Float32(0.0)] * owned
            for q in cutlass.range_constexpr(split):
                for j in cutlass.range_constexpr(owned // 4):
                    part = mailbox.load(
                        idx=cutlass.Int32(q * slot_elems) + row_slot + ((cutlass.Int32(j) ^ swz) << cutlass.Int32(2)),
                        vector_size=4,
                        alignment=16,
                    )  # fmt: skip
                    for e in cutlass.range_constexpr(4):
                        totals[4 * j + e] = totals[4 * j + e] + cutlass.Float32(part[e])
            # [token][rows] staging of the warp's rows below N, one TMA store (tokens >= M clipped).
            n0 = m_offset + w * cutlass.Int32(LONG_ROWS_PER_WARP)
            tail_block = n0 + cutlass.Int32(LONG_ROWS_PER_WARP) > cutlass.Int32(n_out)
            pitch = cutlass.Int32(LONG_ROWS_PER_WARP)
            if tail_block:
                pitch = cutlass.Int32(tail_rows)
            t0 = rank * cutlass.Int32(owned)
            if cutlass.const_expr(out_fp32):
                # In place of this rank's own slot, rows of this warp only (read above by this warp alone).
                staged = mailbox.subview(
                    rank * cutlass.Int32(slot_elems) + w * cutlass.Int32(LONG_ROWS_PER_WARP * owned)
                )
                cute.arch.sync_warp()
            else:
                staged = smem_b.subview(w * cutlass.Int32(LONG_ROWS_PER_WARP * owned))
            if n0 < cutlass.Int32(n_out):
                if lane < pitch:
                    if cutlass.const_expr(out_fp32):
                        for i in cutlass.range_constexpr(owned):
                            staged.store(totals[i], idx=cutlass.Int32(i) * pitch + lane)
                    else:
                        if n0 + lane >= sig_row0:
                            for i in cutlass.range_constexpr(owned):
                                staged.store(
                                    _sigmoid_bf16(_bf16_rn(totals[i])).to(io_dtype), idx=cutlass.Int32(i) * pitch + lane
                                )  # fmt: skip
                        else:
                            for i in cutlass.range_constexpr(owned):
                                staged.store(
                                    totals[i].to(io_dtype), idx=cutlass.Int32(i) * pitch + lane
                                )
                # Generic-proxy writes of the staged rows -> the TMA store's async-proxy reads.
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                cute.arch.sync_warp()
                if lane == 0:
                    if tail_block:
                        prims.cp_async_bulk_tensor_global_shared_cta(
                            tma_ptr_y_tail, staged, [n0, t0]
                        )
                    else:
                        prims.cp_async_bulk_tensor_global_shared_cta(tma_ptr_y, staged, [n0, t0])
                    prims.cp_async_bulk_commit_group()
                    prims.cp_async_bulk_wait_group(0, read=True)
        prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
        if warp_id == 4:
            prims.tcgen05_dealloc(tmem_ptr, tmem_cols)


def _activation_tensor_map_rows(x, cols, num_tokens, rows):
    """x [M, cols] (rows dense) as (cols, M) with a ``rows``-row box: rows past num_tokens arrive as zeros."""
    return cuda.create_tensor_map_tiled(
        global_address=x.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[cols, num_tokens],
        global_strides=[(cols * ELEM_BYTES) // 16],
        box_dims=[TMA_K_BOX, rows],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


def _output_tensor_map(y, n_out, num_tokens, tokens, out_fp32, rows=LONG_ROWS_PER_WARP):
    """y [M, N] (rows dense) as (N, M) with a rows x tokens box, no swizzle: a warp's staged [token][rows]."""
    elem_bytes = 4 if out_fp32 else ELEM_BYTES
    return cuda.create_tensor_map_tiled(
        global_address=y.iterator.toint(),
        dtype=cutlass.Float32 if out_fp32 else cutlass.BFloat16,
        global_dims=[n_out, num_tokens],
        global_strides=[(n_out * elem_bytes) // 16],
        box_dims=[rows, tokens],
        swizzle=cuda.TensorMapSwizzle.none,
    )


@cute.jit
def k3_ctm_gemv_wide(
    w: cute.Tensor,  # [N, K] bf16, K contiguous
    x: cute.Tensor,  # [M, K] bf16, K contiguous, M <= n_tile
    y: cute.Tensor,  # [M * N] bf16 (fp32 with out_fp32)
    num_tokens: cutlass.Int32,
    sig_row0: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    x_ring: cutlass.Constexpr[int],
    n_tile: cutlass.Constexpr[int],
    out_fp32: cutlass.Constexpr[bool],
    trigger_early: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """``y = x @ w^T`` for up to ``n_tile`` tokens; bf16 output rows >= sig_row0 hold bf16(sigmoid(bf16(.)))."""
    owned = wide_owned_tokens(n_tile, split)
    tma_desc_w = _weight_tensor_map(w, n_out, k_in)
    tma_desc_x = _activation_tensor_map_rows(x, k_in, num_tokens, n_tile)
    tma_desc_y = _output_tensor_map(y, n_out, num_tokens, owned, out_fp32)
    tma_desc_y_tail = _output_tensor_map(
        y, n_out, num_tokens, owned, out_fp32, n_out % LONG_ROWS_PER_WARP or LONG_ROWS_PER_WARP
    )
    k3_ctm_gemv_wide_kernel(
        tma_desc_w, tma_desc_x, tma_desc_y, tma_desc_y_tail, sig_row0, n_out, k_in, split, ring, x_ring, n_tile,
        out_fp32, trigger_early,
    ).launch(
        grid=(((n_out + CTA_M - 1) // CTA_M) * split, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(split, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )  # fmt: skip


# =============================================================================
# SiTU-and-mul with PDL: a = bf16(situ(g) * situ_lin(u)) for a gate_up output [M, 2K] (g its first K columns), in
# SituAndMul's fp32 order (beta tanh(g / beta) sigmoid(g), times linear_beta tanh(u / linear_beta) or u), one 16-byte
# vector of 8 columns per thread. The dependents launch at once, so a following projection streams its weights while
# this kernel and its predecessor run; the activation reads wait for the grid.
# =============================================================================
SITU_THREADS = 128
SITU_COLS_PER_CTA = SITU_THREADS * VEC


def _situ_mul(g, u, beta, linear_beta, has_linear: bool):
    """bf16(situ(g) * situ_lin(u)) in fp32, SituAndMul's order."""
    one = cutlass.Float32(1.0)
    sig = _div_rn(one, one + cute.math.exp(-g, fastmath=False))
    a = beta * cute.math.tanh(_div_rn(g, beta), fastmath=False) * sig
    if cutlass.const_expr(has_linear):
        u = linear_beta * cute.math.tanh(_div_rn(u, linear_beta), fastmath=False)
    return (a * u).to(io_dtype)


@cute.kernel
def k3_situ_mul_kernel(
    gu: cutlass.Array,  # bf16 [M * 2K]
    out: cutlass.Array,  # bf16 [M * K]
    num_tokens: cutlass.Int32,
    beta: cutlass.Float32,
    linear_beta: cutlass.Float32,
    k_in: cutlass.Constexpr[int],
    has_linear: cutlass.Constexpr[bool],
):
    tx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    prims.griddepcontrol(prims.GridDepAction.WAIT)
    col = bx * cutlass.Int32(SITU_COLS_PER_CTA) + tx * cutlass.Int32(VEC)
    if col < cutlass.Int32(k_in):
        if by < num_tokens:
            row = by * cutlass.Int32(2 * k_in)
            gv = gu.load(idx=row + col, vector_size=VEC, alignment=16)
            uv = gu.load(idx=row + cutlass.Int32(k_in) + col, vector_size=VEC, alignment=16)
            outs = [
                _situ_mul(
                    cutlass.Float32(gv[e]), cutlass.Float32(uv[e]), beta, linear_beta, has_linear
                )
                for e in range(VEC)
            ]
            out.store(
                cutlass.Vector.from_elements(tuple(outs), io_dtype),
                idx=by * cutlass.Int32(k_in) + col, vector_size=VEC, alignment=16,
            )  # fmt: skip


@cute.jit
def k3_situ_mul(
    gu: cute.Tensor,  # bf16 [M * 2K]
    out: cute.Tensor,  # bf16 [M * K]
    num_tokens: cutlass.Int32,
    beta: cutlass.Float32,
    linear_beta: cutlass.Float32,
    max_tokens: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    has_linear: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """``out = SituAndMul(beta, linear_beta)(gu)`` for rows < num_tokens (grid over max_tokens rows)."""
    k3_situ_mul_kernel(gu, out, num_tokens, beta, linear_beta, k_in, has_linear).launch(
        grid=((k_in + SITU_COLS_PER_CTA - 1) // SITU_COLS_PER_CTA, max_tokens, 1),
        block=(SITU_THREADS, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )

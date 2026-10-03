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
"""Kimi K3 post-attention sandwich: the attention output projection, the TP all-reduce and the residual update
(attention-residual selection + RMSNorm) in one kernel, for M <= 8 decode tokens.

    partial_r = core_r @ W_o,r^T                    (this rank's [M, 7168] share, rounded to bf16)
    updated   = bf16(prefix + bf16(sum_r partial_r))  (or the sum alone without a prefix)
    normed    = KimiK3RMSNorm(attn_res(snapshots..., updated))

as ``o_proj`` followed by ``trtllm::mnnvl_allreduce_attn_res``; the reduction and the epilogue follow
``oneshotAllreduceAttnResKernel``'s arithmetic (rank chunks of 8, the same statistics, summation orders and
roundings), so the outputs are bit-identical to that pair.

56 CTAs = 8 clusters of 7, 256 threads. Phase 1, every CTA: the o_proj rows [128 c, 128 c + 128) with the whole
weight slice (192 KB) TMA-loaded and copied on into TMEM (``tcgen05.cp``, 8 columns per K16 granule, the MMAs' own
descriptors) as each k-tile lands, all before ``griddepcontrol.wait`` under PDL; then the core [M, 768] and 48 tcgen05
MMAs (M 128, N 8, fp32 in TMEM) with A read from TMEM (the tensor core reads A out of shared memory at ~110 B/clk).
The bf16 rows are pushed as 16-byte stores through the multicast mapping of the all-reduce buffer into slot
[token][rank] of every rank, and the dependents are launched. Phase 2, cluster t for token t < M: every thread owns 8
columns (7 x 128 x 8 = 7168). The snapshots' statistics do not depend on the peers, so they are reduced over the
cluster and turned into the snapshots' logits and their maximum while the rows are in flight; then the thread polls
the ranks' slots until no word is empty, sums them, empties the slots it read, and runs the rest of the epilogue. Its
reductions over the cluster are st.async stores into every peer's shared memory that complete one-shot mailbox
mbarriers there (no cluster barrier): the snapshots' statistics (before the poll), the updated sum's statistics and
the selection's sum of squares. The mailbox waits spin on
``mbarrier.test_wait`` (a warp suspended in ``try_wait`` on a mailbox completed by remote st.async wakes late),
acquiring at cluster scope: the st.async complete_tx of the other CTAs releases at cluster scope.

The buffer: two alternating halves of [8 tokens][world][7168] bf16 per rank, as int32 words; a word of
0x80000000 is empty (pushes turn bf16 -0.0 into +0.0, so a pushed pair never has that pattern). ``flags[b]``
counts the calls of CTA b (its parity selects the half; every CTA runs every call, so all CTAs and ranks agree).
A reducing thread empties the words it read right after reading them: the next push into that half comes from a
peer's call after next, which starts only after this call has ended here.

Published output (``x_slab``, when given): the normed rows also go to a Lamport slab [3][8][7168] bf16 (int32
words, sentinel 0xFFFFFFFF, a computed all-ones word stored as 0x7FC07FC0) that the next kernel polls tile by tile
instead of waiting for this grid. Call ``slab_buf`` writes buffer ``slab_buf`` and, after its grid wait, re-arms
buffer ``(slab_buf + 1) % 3`` (every row, every CTA), whose last readers ran two grid completions earlier.

The pre-attention sandwich (``k3_sandwich_tail``) runs the same phases with the row-parallel MoE tail as phase 1:
``[rmsnorm(latent)[:, lo:lo+224] | act] @ [W_lat (zero-padded to 256) | W_act]^T``, K = 256 + 384, the latent and
the activation k-tiles in two TMEM accumulators and the latent RMS (over the whole reduced latent row) applied to
the first; its phase 2 adds the MoE partial sum to the running prefix sum. The latent RMS of token t is computed once
per cluster, by epilogue warp t // 7 of CTA t % 7, from a bulk copy of the row into shared memory (the sums of a
per-lane chain in row order, then a butterfly), and st.async'd into every cluster CTA's row scales.

Warps: 0 weight TMA, 1 the activation TMA (tail: first the bulk copies of the latent rows whose RMS the CTA owns), 2
TMEM allocation, the weight copies into TMEM and the MMA, 3 idle, 4-7 the phase 1 epilogue and phase 2. After cluster
formation only warps 4-7 synchronize (named barrier 1 and the mailboxes), so warps 0-3 finish once their work is
issued.

The CTA's work is ``sandwich_role`` (a ``cute.jit`` role function on shared memory carved from its ``smem``
pointer, ``smem_bytes`` of it, e.g. one ``SmemAllocator`` block), so a kernel with further roles in other clusters
can run it in 56 of its CTAs; its ``act_hook`` replaces the activation TMA by the caller's own delivery into
``smem_b`` (see ``sandwich_role``).
``k3_sandwich_kernel`` is the sandwich alone.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

try:
    from cutlass.memory.smem import SmemAllocator
except ImportError:  # DSL builds that keep it under cutlass.utils
    from cutlass.utils import SmemAllocator

H = 7168
K_IN = 768  # this rank's o_proj input: 96 heads x 128 / TP16
TAIL_LAT = 256  # the MoE tail's latent slice (224 columns at TP16) zero-padded to whole k-tiles
TAIL_ACT = 384  # the shared-expert activation: 2 x 3072 / TP16 / 2
LATENT = 3584
CTA_M = 128
MMA_N = 8
CTA_K = 128
MMA_K = 16
TMA_K_BOX = 64
TMA_COPY_ITERS = CTA_K // TMA_K_BOX
K_BLOCKS_PER_HALF = TMA_K_BOX // MMA_K
MAX_K_TILES = K_IN // CTA_K
# The plain form takes up to 7 k-tiles (the drafter's down projection, K 896): all of them staged in TMEM (64 + 7 x 64
# of 512 columns), but through a ring of W_RING shared-memory stages once they no longer all fit there.
W_RESIDENT = 6
W_RING = 5
PLAIN_MAX_K = 896
THREADS = 256
EPI_THREADS = 128
EPI_WARPS = EPI_THREADS // 32
ELTS = 8  # bf16 columns per phase 2 thread
ELEM_BYTES = 2
TMEM_COLS = (
    512  # the whole TMEM: accumulators at columns 0 (and 8), the weight slice from TMEM_A_COL
)
TMEM_A_COL = 64  # A (the weight slice) staged in TMEM: 8 columns (128 lanes x K16 bf16) per MMA
TMEM_A_COLS_PER_MMA = MMA_K * ELEM_BYTES // 4
EVICT_FIRST = 0x12F0000000000000

CLUSTER = H // (EPI_THREADS * ELTS)  # 7 CTAs own one token in phase 2
NUM_CTAS = H // CTA_M  # 56
MAX_TOKENS = NUM_CTAS // CLUSTER  # 8
LATENT_ROWS_PER_CTA = (
    MAX_TOKENS + CLUSTER - 1
) // CLUSTER  # tail: cluster CTA c owns the RMS of tokens c and c + 7
WORDS_PER_ROW = H // 2
ROW_WORDS_PER_CTA = CTA_M // 2
MAX_CANDIDATES = (
    9  # snapshots + the updated sum; Kimi K3 (93 layers, a snapshot every 12) needs at most 9
)
MAX_SNAPSHOTS = MAX_CANDIDATES - 1
SNAP_STATS = 2 * MAX_SNAPSHOTS  # [2 n] sum of squares, [2 n + 1] residual projection of snapshot n
FLAG_WORDS = 64  # per-CTA call counters (NUM_CTAS of them)
RANK_CHUNK = 8  # ranks summed per chunk, as the MNNVL one-shot does for more than 8 ranks
EMPTY_WORD = -2147483648  # 0x80000000
LOG2E = 1.4426950408889634
# The published slab (x_slab): [SLAB_BUFS][MAX_TOKENS][WORDS_PER_ROW] int32 words.
SLAB_BUFS = 3
SLAB_WORDS = MAX_TOKENS * WORDS_PER_ROW
SLAB_SENTINEL = -1  # 0xFFFFFFFF
SLAB_CANON = 0x7FC07FC0  # what a computed all-ones word is stored as
POLL_BACKOFF = (
    256  # cycles between two polling rounds of phase 1's input slab that found a sentinel
)
# The latent exchange (x_src 2): k3_moe pushes every rank's routed latent partial into two alternating halves of
# [8 tokens][world][3584] bf16 per rank (int32 words, EMPTY_WORD = not written); the tail sums them itself.
LAT_SLICE = 224  # latent columns of a rank's slice
LAT_SLICE_VECS = LAT_SLICE // 8  # its 16-byte vectors per token row
LAT_WORDS = 3584 // 2  # int32 words of one latent row
LAT_VECS = 3584 // 8  # 16-byte vectors of one latent row
LAT_FLAG_WORDS = (
    64  # int32: [0] the tail's call count mod 6, [LAT_SCALES + 8 b + t] buffer b of the latent scale slab
)
LAT_SCALES = 32
LAT_SCALE_BUFS = 3
SCALE_SENTINEL = -1  # 0xFFFFFFFF: an unwritten scale (a computed rsqrt is never this NaN pattern)

LEADING = 16
STRIDE = 8 * TMA_K_BOX * ELEM_BYTES
A_HALF_ELEMS = CTA_M * TMA_K_BOX
B_HALF_ELEMS = MMA_N * TMA_K_BOX
STEP = (MMA_K * ELEM_BYTES) >> 4
A_BOX = A_HALF_ELEMS >> 3
B_BOX = B_HALF_ELEMS >> 3
STAGE_A = (CTA_M * CTA_K * ELEM_BYTES) >> 4
STAGE_B = (MMA_N * CTA_K * ELEM_BYTES) >> 4

io_dtype = cutlass.BFloat16

assert NUM_CTAS == MAX_TOKENS * CLUSTER
assert NUM_CTAS <= FLAG_WORDS
assert SNAP_STATS * CLUSTER <= EPI_THREADS
assert TMEM_A_COL + MAX_K_TILES * (CTA_K // MMA_K) * TMEM_A_COLS_PER_MMA <= TMEM_COLS
assert TMEM_A_COL + (PLAIN_MAX_K // CTA_K) * (CTA_K // MMA_K) * TMEM_A_COLS_PER_MMA <= TMEM_COLS


def buffer_words(world: int) -> int:
    """Int32 words of one rank's all-reduce buffer: both halves of rows."""
    return 2 * MAX_TOKENS * world * WORDS_PER_ROW


def lat_buffer_words(world: int) -> int:
    """Int32 words of one rank's latent exchange buffer (both halves)."""
    return 2 * MAX_TOKENS * world * LAT_WORDS


@dsl_user_op
def _clock64(*, loc=None, ip=None):
    return cutlass.Int64(
        _llvm.inline_asm(
            _T.i64(), [], "mov.u64 $0, %clock64;", "=l", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _pack_bf16x2(hi, lo, *, loc=None, ip=None):
    """(bf16(hi) << 16) | bf16(lo), round to nearest even."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [hi.ir_value(loc=loc, ip=ip), lo.ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", has_side_effects=False,
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
def _st_async_f32(dst, value, mbar, *, loc=None, ip=None):
    """st.async of one fp32 to a shared::cluster address, completing ``mbar`` (a shared::cluster address) by 4 bytes."""
    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Float32(value).ir_value(loc=loc, ip=ip),
         cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];", "r,f,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _st_async_v4(dst, w0, w1, w2, w3, mbar, *, loc=None, ip=None):
    """st.async of four int32 words (16 bytes) to a shared::cluster address, completing ``mbar`` (a shared::cluster
    address) by 16 bytes."""
    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Int32(w0).ir_value(loc=loc, ip=ip),
         cutlass.Int32(w1).ir_value(loc=loc, ip=ip), cutlass.Int32(w2).ir_value(loc=loc, ip=ip),
         cutlass.Int32(w3).ir_value(loc=loc, ip=ip), cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];", "r,r,r,r,r,r",
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


@dsl_user_op
def _rcp_rn(x, *, loc=None, ip=None):
    """rcp.rn.f32: the correctly rounded 1 / x, the same value as the IEEE division div.rn.f32 1.0, x."""
    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)], "rcp.rn.f32 $0, $1;", "=f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _ld_acquire_sys(addr_i64, *, loc=None, ip=None):
    """ld.acquire.sys.global.u32: a flag word whose writer released its earlier writes at system scope."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip)], "ld.acquire.sys.global.u32 $0, [$1];", "=r,l",
            has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _fence_sys(*, loc=None, ip=None):
    """fence.acq_rel.sys: this thread's earlier writes (and those it has observed) before its later ones, for every
    observer."""
    _llvm.inline_asm(
        None, [], "fence.acq_rel.sys;", "", has_side_effects=True, is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


def _lo(word):
    return (word << cutlass.Int32(16)).bitcast(cutlass.Float32)


def _hi(word):
    return (word & cutlass.Int32(-65536)).bitcast(cutlass.Float32)


def _sanitize(word):
    """A pushed pair must never read as empty: bf16 -0.0 halves become +0.0."""
    w = cutlass.Int32(
        cutlass.select_(
            (word & cutlass.Int32(0xFFFF)) == cutlass.Int32(0x8000),
            word & cutlass.Int32(-65536),
            word,
        )
    )
    return cutlass.Int32(
        cutlass.select_(
            (w & cutlass.Int32(-65536)) == cutlass.Int32(EMPTY_WORD), w & cutlass.Int32(0xFFFF), w
        )
    )


def _slab_word(word):
    """A published word must never read as the slab sentinel: an all-ones word (two all-ones bf16 NaNs) is stored
    as two quiet NaNs."""
    return cutlass.Int32(
        cutlass.select_(word == cutlass.Int32(SLAB_SENTINEL), cutlass.Int32(SLAB_CANON), word)
    )


def _fmax(a, b):
    return cutlass.Float32(cutlass.select_(a > b, a, b))


def _warp_allsum(value):
    """Sum over the warp, in every lane. The butterfly adds the same pairs as a shfl.down tree does into lane 0
    (fp32 addition is commutative), so every lane holds lane 0's shfl.down result bit for bit."""
    for offset in (16, 8, 4, 2, 1):
        value = value + cute.arch.shuffle_sync_bfly(value, offset=offset)
    return value


def _warp_sum_scatter(values, lane):
    """Sum each of len(values) (a power of two, 2..32) values over the warp; value i ends in the lanes l with
    (l >> (5 - log2(len))) == i, reduced over the same pairs as a shfl.down tree (so bit-identical to it).
    Each level keeps half of the values and sends the other half: len - 1 shuffles in all, then a plain
    butterfly over the remaining lane bits."""
    vals = list(values)
    offset = 16
    while len(vals) > 1:
        half = len(vals) // 2
        upper = (lane & cutlass.Int32(offset)) != cutlass.Int32(0)
        nxt = []
        for j in range(half):
            keep = cutlass.Float32(cutlass.select_(upper, vals[j + half], vals[j]))
            send = cutlass.Float32(cutlass.select_(upper, vals[j], vals[j + half]))
            nxt.append(keep + cute.arch.shuffle_sync_bfly(send, offset=offset))
        vals = nxt
        offset //= 2
    value = vals[0]
    while offset >= 1:
        value = value + cute.arch.shuffle_sync_bfly(value, offset=offset)
        offset //= 2
    return value


def weight_stages(k_tiles: int) -> int:
    """Shared-memory stages of the weight: every k-tile, or a ring of W_RING past W_RESIDENT k-tiles."""
    return k_tiles if k_tiles <= W_RESIDENT else W_RING


def _role_layout(k_tiles: int, rms_cols: int, swiglu: int = 0):
    """(dtype, elements, alignment) of the role's shared arrays, in carve order."""
    stages = weight_stages(k_tiles)
    return (
        (
            io_dtype,
            stages * CTA_M * CTA_K,
            1024,
        ),  # smem_a: the weight k-tiles or the ring's stages (TMA, SW128)
        (
            io_dtype,
            k_tiles * MMA_N * CTA_K,
            1024,
        ),  # smem_b: the activation k-tiles (SW128, see act_word_index)
        (cutlass.Int32, MMA_N * ROW_WORDS_PER_CTA, 16),  # stage: the bf16 rows before the push
        (
            cutlass.Int32,
            LATENT_ROWS_PER_CTA * (rms_cols // 2) if rms_cols > 0 else 4,
            128,
        ),  # rms_row
        (cutlass.Int64, stages, 8),  # weight_full, per stage
        (cutlass.Int64, 1, 8),  # act_full
        (cutlass.Int64, 1, 8),  # acc_done
        (cutlass.Int64, 1, 8),  # mb_snap
        (cutlass.Int64, 1, 8),  # mb_upd
        (cutlass.Int64, 1, 8),  # mb_sq
        (cutlass.Int64, 1, 8),  # mb_rms
        (cutlass.Int64, LATENT_ROWS_PER_CTA, 8),  # rms_full
        (cutlass.Int32, 1, 16),  # the TMEM address
        (cutlass.Float32, EPI_WARPS * SNAP_STATS, 16),  # warp_snap
        (cutlass.Float32, CLUSTER * SNAP_STATS, 16),  # box_snap
        (cutlass.Float32, CLUSTER * EPI_WARPS * 2, 16),  # box_upd
        (cutlass.Float32, CLUSTER * EPI_WARPS, 16),  # box_sq
        (cutlass.Float32, MMA_N, 16),  # row_scale
        (cutlass.Int64, 1, 8),  # mb_ts (plain)
        (
            cutlass.Float32,
            CLUSTER * EPI_THREADS,
            16,
        ),  # box_ts (plain): every cluster thread's sum of squares
        (cutlass.Int64, 1, 8),  # lat_full (x_src 2): the latent slice by st.async
        (
            io_dtype,
            k_tiles * MMA_N * CTA_K if swiglu else 8,
            1024 if swiglu else 16,
        ),  # smem_g (swiglu): the gate tiles
        (cutlass.Int64, stages, 8),  # stage_free (ring): a stage's copies into TMEM have completed
        (cutlass.Int64, 1, 8),  # b_ready (swiglu): the activation rewritten in smem_b
    )


def smem_bytes(k_in: int, rms_cols: int, swiglu: int = 0) -> int:
    """Bytes of shared memory ``sandwich_role`` carves from its ``smem`` pointer (1024-byte aligned):
    ``smem_bytes(K_IN, 0)`` for the post-attention form, ``smem_bytes(TAIL_LAT + TAIL_ACT, LATENT)`` for the tail."""
    off = 0
    for dtype, n, align in _role_layout(k_in // CTA_K, rms_cols, swiglu):
        off = -(-off // align) * align + n * dtype.width // 8
    return off


def _view(raw, off: int, dtype, n: int, align: int):
    return cutlass.Array(
        cute.recast_ptr(raw if off == 0 else raw + off, dtype=dtype), shape=(n,), dtype=dtype, bounds_check=False,
        addrspace=cutlass.AddressSpace.smem.value, alignment=align,
    )  # fmt: skip


def _carve(raw, k_tiles: int, rms_cols: int, swiglu: int = 0):
    """The role's arrays as views into the shared-memory bytes at ``raw`` (1024-byte aligned), in ``_role_layout``
    order, then an int32 view of ``smem_b``. Every CTA of a cluster gets the same offsets, so a peer's address of a
    mailbox is its mapa."""
    views = []
    off = 0
    b_off = 0
    for i, (dtype, n, align) in enumerate(_role_layout(k_tiles, rms_cols, swiglu)):
        off = -(-off // align) * align
        if i == 1:
            b_off = off
        views.append(_view(raw, off, dtype, n, align))
        off += n * dtype.width // 8
    views.append(_view(raw, b_off, cutlass.Int32, k_tiles * MMA_N * CTA_K // 2, 1024))
    return views


def act_word_index(t, col):
    """Int32-word index in ``smem_b`` of the bf16 activation columns [col, col + 8) of token row t (col a multiple of
    8): the SW128 layout the activation TMA box lands, 16-byte unit j = (col % 64) // 8 of row t at unit j ^ t of
    the row's 128 bytes, 64-column halves 1 KB apart and k-tiles of 128 columns 2 KB apart."""
    return (col // 128) * 512 + ((col % 128) // 64) * 256 + t * 32 + (((col % 64) // 8) ^ t) * 4


@cute.jit
def swiglu_share(smem_b: cutlass.Array, smem_g: cutlass.Array, act_full, b_ready, tid: cutlass.Int32,
                 crank: cutlass.Int32):  # fmt: skip
    """The SwiGLU of k-tile ``crank``, this CTA's share of the cluster's 7 k-tiles: epilogue thread ``tid`` takes
    16-byte chunk ``tid`` of the tile, B = bf16((g * sigmoid(g)) * a) with silu_and_mul's fp32 order and sigmoid(g)
    = 1 / (1 + exp(-g)) correctly rounded (rcp.rn, the value of the IEEE division), and st.async's the 16 bytes into
    the same offsets of tile ``crank`` in every cluster CTA's ``smem_b`` (a and g share one swizzled layout),
    completing their ``b_ready`` by 16 bytes each."""
    while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
        pass
    c_el = crank * cutlass.Int32(MMA_N * CTA_K) + tid * cutlass.Int32(ELTS)
    av = smem_b.load(idx=c_el, vector_size=ELTS, alignment=16)
    gv = smem_g.load(idx=c_el, vector_size=ELTS, alignment=16)
    b_words = []
    for q in cutlass.range_constexpr(ELTS // 2):
        pair = []
        for e in (2 * q, 2 * q + 1):
            g_e = cutlass.Float32(gv[e])
            sig = _rcp_rn(cutlass.Float32(1.0) + cute.math.exp(-g_e, fastmath=False))
            pair.append((g_e * sig) * cutlass.Float32(av[e]))
        b_words.append(_pack_bf16x2(pair[1], pair[0]))
    for peer in cutlass.range_constexpr(CLUSTER):
        _st_async_v4(
            _mapa_u32(smem_b.subview(c_el).data_ptr(), peer), b_words[0], b_words[1], b_words[2], b_words[3],
            _mapa_u32(b_ready.data_ptr(), peer),
        )  # fmt: skip


@cute.jit
def sandwich_role(
    tma_desc_w,  # W [7168, k_in] bf16, 5-D, one call per k-tile
    tma_desc_x,  # core [M, 768] (tail: latent [M, 3584]), box 8 x 64 (unused with act_hook)
    tma_desc_x2,  # tail: act [M, 384], box 8 x 64
    rms_src: cutlass.Array,  # tail: int32 words of the latent rows whose RMS scales accumulator 0
    ws_uc: cutlass.Array,  # int32 words: this rank's all-reduce buffer
    ws_mc: cutlass.Array,  # int32 words: its multicast mapping
    ws_flags: cutlass.Array,  # int32 [FLAG_WORDS]: call count of CTA b at [b]
    prefix: cutlass.Array,  # int32 words of bf16 [M, 7168] (read only with add_prefix)
    snapshots: cutlass.Array,  # int32 words of bf16 [num_cand - 1, M, 7168] (at least one [M, 7168] row)
    res_w: cutlass.Array,  # int32 words of bf16 [7168]: the residual projection
    rms_w: cutlass.Array,  # int32 words of bf16 [7168]: its RMSNorm weight
    out_w: cutlass.Array,  # int32 words of bf16 [7168]: the output RMSNorm weight
    updated: cutlass.Array,  # int32 words of bf16 [M, 7168], out
    normed: cutlass.Array,  # int32 words of bf16 [M, 7168], out
    x_slab: cutlass.Array,  # int32 words [SLAB_BUFS, MAX_TOKENS, 3584]: the published normed rows (publish)
    lat_flags: cutlass.Array,  # int32 [LAT_FLAG_WORDS] (x_src 2): the tail's call count and the latent scale slab
    tap: cutlass.Array,  # int32 words of bf16 rows (tap_out), tap_stride words apart: the capture layer's tap, out
    smem,  # shared-memory pointer, 1024-byte aligned, at least smem_bytes(k_in, rms_cols) bytes (SmemAllocator)
    num_tokens: cutlass.Int32,
    rank: cutlass.Int32,
    num_cand: cutlass.Int32,  # snapshots + 1, at most MAX_CANDIDATES
    add_prefix: cutlass.Int32,  # nonzero: updated = prefix + the sum
    rms_eps: cutlass.Float32,
    out_eps: cutlass.Float32,
    x_col0: cutlass.Int32,  # tail: first latent column of this rank's slice
    lat_eps: cutlass.Float32,  # tail: the latent RMSNorm's epsilon
    slab_buf: cutlass.Int32,  # publish: the slab buffer this call writes (0-2)
    x_buf: cutlass.Int32,  # x_src 1: the slab buffer of phase 1's input (0-2)
    tap_stride: cutlass.Int32,  # tap_out: int32 words from one tap row to the next (3584 when contiguous)
    world: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],  # K of phase 1: 768 (o_proj) or 640 (tail)
    x_tiles: cutlass.Constexpr[
        int
    ],  # k-tiles from tma_desc_x; the rest from tma_desc_x2 into accumulator 1
    rms_cols: cutlass.Constexpr[
        int
    ],  # > 0 (tail): y = rsqrt(mean(rms_src row^2) + lat_eps) * acc0 + acc1
    publish: cutlass.Constexpr[
        int
    ],  # 1: also publish normed into x_slab (and re-arm the next buffer)
    x_src: cutlass.Constexpr[
        int
    ],  # 0: phase 1's input by TMA after the grid wait; 1: polled from a slab; 2 (tail):
    # the latent summed from every rank's pushed partial (below)
    plain: cutlass.Constexpr[
        int
    ],  # 1 (MNNVL order) / 2 (IPC order): residual + RMSNorm, not attn_res (below)
    tap_out: cutlass.Constexpr[
        int
    ],  # attn_res forms: also store into ``tap`` the pre-norm mixture rows (1) or updated (2)
    swiglu: cutlass.Constexpr[
        int
    ],  # plain: phase 1's B is silu_and_mul of x [M, 2 k_in] (gate columns first), below
    k_split: cutlass.Constexpr[
        int
    ],  # 2: accumulate the even and the odd k-tiles apart, then (0 + even) + odd
    cta_base: cutlass.Constexpr[int],  # grid index of the role's first CTA (a multiple of CLUSTER)
    act_hook: cutlass.Constexpr,  # None: the activation by TMA after the grid wait; else see below
):
    """One CTA of the sandwich: grid indices cta_base .. cta_base + 55 (8 clusters of 7, 256 threads). A kernel may
    run other roles in other CTAs (their own clusters) as long as they never touch ``ws_flags`` or the all-reduce
    buffer.

    ``act_hook`` (post-attention form only): warp 1 calls ``act_hook(smem_b_words, act_full, grid_bx, lane)`` (all 32
    lanes; grid_bx the CTA's grid index) instead of waiting on the grid and loading the activation by TMA. The hook
    must store every token row's 768 columns into ``smem_b_words`` (int32 view of the bf16 tile) at
    ``act_word_index`` (rows >= M zero), then every storing lane ``fence.proxy.async.shared::cta``, a warp sync, and
    one plain arrive of lane 0 on ``act_full`` (count 1, no transaction bytes). Warps 4-7 still wait on the grid
    before the flags, the prefix and the snapshots.

    ``x_src`` 1 (the producer publishes phase 1's input as a Lamport slab, int32 [3][8][x_cols / 2] in ``rms_src``,
    sentinel 0xFFFFFFFF, buffer ``x_buf``): warp 1 polls this rank's k-tiles of the live rows into ``smem_b`` (tail:
    and the whole rows of the CTA's RMS tokens into ``rms_row``; the shared-expert activation by TMA at once), and no
    warp waits on the grid before its reads. That needs the producer to launch its dependents only after its own
    grid wait: then this launch implies that every kernel before the producer has completed, which covers the call
    count, the prefix, the snapshots and the slab buffer this call re-arms (read L1-bypassing). Warps 4-7 wait on the
    grid at their end, so the grid completes only after its producer.

    ``x_src`` 2 (tail; the latent all-reduce folded in): k3_moe, the producer, pushes its routed latent partial rows
    through the multicast mapping into slot [rank] of half ``n & 1`` of every rank's exchange buffer (``rms_src``: int32
    [2][8][world][1792], EMPTY_WORD = not written, pushes never write that pattern) and exits; ``n = lat_flags[0]``
    counts this op's calls modulo 6. No warp waits on the grid before its reads (as for ``x_src`` 1, the producer
    launches its dependents right after its own grid wait). The epilogue warps first empty the other half (every row:
    its readers, the previous call, have completed, and the fence before this call's push orders the empties before any
    rank's next push into it, which follows that rank's phase 2 of this call), then sum the partials in the MNNVL
    one-shot's order (fp32 per chunk of 8 ranks in rank order, the chunks added from 0, one bf16 rounding), so the
    latent is the all-reduce's bit for bit:
    - every cluster: the rank's 224 slice columns of the live rows, one (token, vector, rank chunk) per thread, st.async
      into all 7 CTAs' ``smem_b`` (``lat_full``; the latent k-tiles are zeroed by warp 3 before the cluster forms);
    - cluster t: token t's whole row, st.async into CTA 0's ``rms_row``; CTA 0's warp 4 takes its RMS as the bulk-copy
      path does and publishes the scale into buffer ``n % 3`` of the scale slab in ``lat_flags``, which every
      epilogue polls after the MMA (each call re-arms buffer ``(n + 1) % 3``).
    The MMA runs the shared-expert k-tiles first and the latent ones once ``lat_full`` completes. CTA 0 stores
    ``n + 1`` after its final grid wait. Every k3_moe push call must be followed by exactly one such call.

    ``plain`` 1 (post-attention form, no snapshots): the epilogue is the MNNVL one-shot's residual + RMSNorm
    (``kARResidualRMSNorm``) with its arithmetic: updated = bf16(float(sum) + float(prefix)); per thread the sum of
    the bf16 squares of its 8 values in order; that kernel's reduction tree for a 7168-wide row (8 CTAs of 112
    threads: xor-butterfly warps, the partial fourth warp, warp 0's butterfly over the warp sums, then the block sums
    in order), replayed on every thread's sum gathered into each cluster CTA; normed = bf16(float(x) * r * float(w))
    with r = rsqrt(S / 7168 + out_eps). ``plain`` 2: the same in the order of the IPC one-shot
    (``allreduce_fusion_kernel_oneshot_lamport``, kARResidualRMSNorm, fp32 accumulation), which TRT-LLM runs within one
    node: the ranks' sum in rank order (world <= 8: one chunk), per thread the fp32 squares of its 8 values in order,
    and that kernel's tree for a 7168-wide row (4 CTAs of 224 threads: xor-butterfly warps, blockReduceSumV2's
    butterfly over the 7 warp sums, then the CTA sums in order); thread g's 8 columns are [8 g, 8 g + 8) in both.

    ``swiglu`` (plain, the drafter MLP's down projection after gate_up, K 896 = one k-tile per cluster CTA): warp 1
    lands the up half of k-tile ``crank`` of x [M, 2 k_in] in ``smem_b`` and its gate half in ``smem_g`` (identical
    SW128 layouts); after the grid wait the epilogue warps compute that tile's B = bf16((g * sigmoid(g)) * a) with
    k3_ctm_gemv_swiglu's arithmetic (``swiglu_share``) and st.async it into every cluster CTA's ``smem_b``, whose
    ``b_ready`` expects the 7 tiles' bytes; warp 2 then fences the generic-proxy stores for the MMA's async-proxy
    reads. ``k_split`` 2 accumulates the even and
    the odd k-tiles in two TMEM accumulators and adds them as (0 + even) + odd before the one bf16 rounding: the sums
    of k3_ctm_gemv split 2, whose rank r takes the k-tiles r, r + 2, .... Past W_RESIDENT k-tiles the weight goes
    through a ring of W_RING shared-memory stages: warp 2 commits each stage's copies into TMEM to ``stage_free``,
    which warp 0 waits on before it refills the stage."""
    tail = rms_cols > 0
    hooked = act_hook is not None
    k_tiles = k_in // CTA_K
    split_acc = x_tiles < k_tiles
    stages = weight_stages(k_tiles)
    ring = stages < k_tiles
    acc_pair = split_acc or k_split == 2
    assert not (ring and (x_src != 0 or hooked)), "the weight ring serves the plain form only"
    assert not swiglu or (plain and k_split == 2 and k_tiles == CLUSTER), (
        "swiglu: the plain form, split 2's order, K 896"
    )
    tx, _, _ = cute.arch.thread_idx()
    grid_bx, _, _ = cute.arch.block_idx()
    bx = grid_bx - cutlass.Int32(cta_base)  # the role's CTA index (0-55)
    warp_id = cute.arch.warp_idx()
    crank = cute.arch.block_idx_in_cluster()
    token = bx // cutlass.Int32(CLUSTER)
    tma_ptr_w = tma_desc_w.get_ptr()
    tma_ptr_x = tma_desc_x.get_ptr()
    m_offset = bx * cutlass.Int32(CTA_M)

    tma_ptr_x2 = tma_desc_x2.get_ptr()
    # Views into the role's shared memory (_role_layout order). One-shot mailboxes: the cluster's statistics
    # (snapshots CTA-summed, the updated sum per warp), the per-warp output sums of squares and (tail) the tokens'
    # latent RMS land by st.async in box_snap [source CTA rank][statistic], box_upd [source CTA rank][source warp][sum
    # of squares, residual projection], box_sq [source CTA rank][source warp] and row_scale, completing mb_snap,
    # mb_upd, mb_sq and mb_rms. warp_snap [epilogue warp][snapshot statistic] holds this CTA's warp sums before the
    # CTA sum is sent. Tail: rms_row holds the latent rows of the tokens whose RMS this CTA computes (bulk copies).
    (smem_a, smem_b, stage, rms_row, weight_full, act_full, acc_done, mb_snap, mb_upd, mb_sq, mb_rms, rms_full,
     tmem_ptr_i32, warp_snap, box_snap, box_upd, box_sq, row_scale, mb_ts, box_ts, lat_full, smem_g,
     stage_free, b_ready, smem_b_words) = _carve(smem, k_tiles, rms_cols, swiglu)  # fmt: skip

    if warp_id == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        if cutlass.const_expr(not hooked and x_src == 0):
            prims.prefetch_tensormap(tma_ptr_x)
        if cutlass.const_expr(split_acc):
            prims.prefetch_tensormap(tma_ptr_x2)
        if prims.elect_sync():
            for k in cutlass.range_constexpr(stages):
                prims.mbarrier_init(weight_full.subview(k), 1)
                if cutlass.const_expr(ring):
                    prims.mbarrier_init(stage_free.subview(k), 1)
            if cutlass.const_expr(swiglu):
                # Armed before cluster formation: every cluster CTA's share of B, by st.async.
                prims.mbarrier_init(b_ready, 1)
                prims.mbarrier_arrive_expect_tx(b_ready, CLUSTER * EPI_THREADS * 16)
            # x_src 1 with a split accumulator: the activation TMA's arrive and warp 1's arrive after the slab rows.
            prims.mbarrier_init(act_full, 2 if (x_src == 1 and split_acc) else 1)
            prims.mbarrier_init(acc_done, 1)
            prims.mbarrier_init(mb_snap, 1)
            prims.mbarrier_init(mb_upd, 1)
            prims.mbarrier_init(mb_sq, 1)
            # Armed before cluster formation, so no peer's bytes can arrive before the expectation. Each peer sends
            # its CTA sums of the num_cand - 1 snapshots' two statistics (none: the phase completes here) and, per
            # warp, the updated sum's two and the selection's sum of squares.
            prims.mbarrier_arrive_expect_tx(
                mb_snap, cutlass.Int32(CLUSTER * 4 * 2) * (num_cand - cutlass.Int32(1))
            )
            prims.mbarrier_arrive_expect_tx(mb_upd, CLUSTER * EPI_WARPS * 2 * 4)
            prims.mbarrier_arrive_expect_tx(mb_sq, CLUSTER * EPI_WARPS * 4)
            if cutlass.const_expr(plain):
                # Every cluster thread's sum of squares, from every cluster CTA.
                prims.mbarrier_init(mb_ts, 1)
                prims.mbarrier_arrive_expect_tx(mb_ts, CLUSTER * EPI_THREADS * 4)
            if cutlass.const_expr(tail):
                # The latent RMS of each live token, from the one cluster CTA that computes it (x_src 2: from the
                # scale slab instead).
                prims.mbarrier_init(mb_rms, 1)
                if cutlass.const_expr(x_src != 2):
                    prims.mbarrier_arrive_expect_tx(mb_rms, num_tokens * cutlass.Int32(4))
                for rj in cutlass.range_constexpr(LATENT_ROWS_PER_CTA):
                    prims.mbarrier_init(rms_full.subview(rj), 1)
            if cutlass.const_expr(x_src == 2):
                # The live rows' slice vectors from the cluster's reducers, and (CTA 0 of a live token's cluster) the
                # token's whole summed row.
                prims.mbarrier_init(lat_full, 1)
                prims.mbarrier_arrive_expect_tx(
                    lat_full, num_tokens * cutlass.Int32(LAT_SLICE_VECS * 16)
                )
                if (crank == cutlass.Int32(0)) & (token < num_tokens):
                    prims.mbarrier_arrive_expect_tx(
                        rms_full.subview(0), cutlass.Int32(LAT_VECS * 16)
                    )

    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    if cutlass.const_expr(x_src == 2):
        if warp_id == 3:
            # Zero the latent k-tiles of smem_b (rows >= M and the 32 padding columns stay zero; the reducers' st.async
            # land after cluster formation), for the MMA's async-proxy reads and before any peer's store.
            lane3 = tx % cutlass.Int32(32)
            zero = cutlass.Int32(0)
            for zi in cutlass.range_constexpr(x_tiles * MMA_N * CTA_K // (2 * 4 * 32)):
                smem_b_words.store((zero, zero, zero, zero), idx=cutlass.Int32(zi * 128) + lane3 * cutlass.Int32(4),
                                   alignment=16)  # fmt: skip
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
            prims.fence_acq_rel(prims.MemScope.CLUSTER)
    prims.fence_mbarrier_init()
    # Cluster formation: the peers' shared memory and armed mailboxes are addressable from here on.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    tmem_ptr = cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32)

    if warp_id == 0:
        if prims.elect_sync():
            for k in cutlass.range_constexpr(k_tiles):
                ws_ = k % stages
                if cutlass.const_expr(k >= stages):
                    # The ring: this stage's previous k-tile has gone on into TMEM (warp 2's commit).
                    while not cute.arch.mbarrier_test_wait(
                        stage_free.subview(ws_).data_ptr(), (k // stages - 1) % 2
                    ):
                        pass
                prims.mbarrier_arrive_expect_tx(
                    weight_full.subview(ws_), CTA_M * CTA_K * ELEM_BYTES
                )
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a.subview(ws_ * CTA_M * CTA_K),
                    tma_ptr_w,
                    (
                        cutlass.Int32(0),
                        m_offset,
                        cutlass.Int32(k * TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ),
                    weight_full.subview(ws_),
                    l2_cache_hint=EVICT_FIRST,
                )
    elif warp_id == 1:
        if cutlass.const_expr(x_src == 1):
            # Phase 1's input from the producer's Lamport slab: a 16-byte vector is ready when none of its words is the
            # sentinel (relaxed gpu-scope loads, no sleep).
            lane1 = tx % cutlass.Int32(32)
            sent1 = cutlass.Int32(SLAB_SENTINEL)
            x_words = (rms_cols if tail else k_in) // 2
            x_row0 = x_buf * cutlass.Int32(MAX_TOKENS * x_words)
            if cutlass.const_expr(split_acc):
                # The shared-expert activation (accumulator 1's k-tiles) is complete at launch.
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(
                        act_full, (k_tiles - x_tiles) * MMA_N * CTA_K * ELEM_BYTES
                    )
                    for k in cutlass.range_constexpr(x_tiles, k_tiles):
                        for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                smem_b.subview(k * MMA_N * CTA_K + half * B_HALF_ELEMS),
                                tma_ptr_x2,
                                (
                                    cutlass.Int32((k - x_tiles) * CTA_K + half * TMA_K_BOX),
                                    cutlass.Int32(0),
                                ),
                                act_full,
                            )
            # Poll once this CTA's weights are in (the MMA needs both; earlier polls only add L2 traffic beside the
            # producer). Each round loads every vector of the lane (all loads in flight at once), stores them, and
            # repeats after a short back-off while any live one still holds the sentinel.
            for k in cutlass.range_constexpr(k_tiles):
                while not cute.arch.mbarrier_try_wait(weight_full.subview(k).data_ptr(), 0):
                    pass
            # This rank's x_tiles k-tiles of the live rows (columns x_col0 + [0, 128 x_tiles)); rows >= M zero.
            x_vecs = x_tiles * CTA_K // 8
            x_col_word = x_col0 // cutlass.Int32(2)
            x_pending = cutlass.Boolean(True)
            while x_pending:
                x_pending = cutlass.Boolean(False)
                xv_in = []
                for i in cutlass.range_constexpr(MAX_TOKENS * x_vecs // 32):
                    xi = cutlass.Int32(i * 32) + lane1
                    x_idx = (
                        x_row0
                        + (xi // cutlass.Int32(x_vecs)) * cutlass.Int32(x_words)
                        + x_col_word
                        + (xi % cutlass.Int32(x_vecs)) * cutlass.Int32(4)
                    )
                    xv_in.append(
                        prims.load_ext(
                            rms_src.subview(x_idx),
                            dtype=cutlass.Int32,
                            count=4,
                            order="relaxed",
                            scope="gpu",
                        )
                    )
                for i in cutlass.range_constexpr(MAX_TOKENS * x_vecs // 32):
                    xi = cutlass.Int32(i * 32) + lane1
                    xt = xi // cutlass.Int32(x_vecs)
                    live = xt < num_tokens
                    xw = [
                        cutlass.Int32(
                            cutlass.select_(live, cutlass.Int32(xv_in[i][q]), cutlass.Int32(0))
                        )
                        for q in range(4)
                    ]
                    smem_b_words.store(
                        (xw[0], xw[1], xw[2], xw[3]),
                        idx=act_word_index(xt, (xi % cutlass.Int32(x_vecs)) * cutlass.Int32(8)),
                        alignment=16,
                    )
                    x_pending = x_pending | (
                        (xw[0] == sent1) | (xw[1] == sent1) | (xw[2] == sent1) | (xw[3] == sent1)
                    )
                if x_pending:
                    t_back = _clock64()
                    while _clock64() - t_back < cutlass.Int64(POLL_BACKOFF):
                        pass
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
            cute.arch.sync_warp()
            if lane1 == cutlass.Int32(0):
                prims.mbarrier_arrive(act_full)
            if cutlass.const_expr(tail):
                # The whole latent row of each token this CTA takes the RMS of (the last of the producer's tiles), in
                # the same rounds.
                for rj in cutlass.range_constexpr(LATENT_ROWS_PER_CTA):
                    rt_s = crank + cutlass.Int32(rj * CLUSTER)
                    if rt_s < num_tokens:
                        r_pending = cutlass.Boolean(True)
                        while r_pending:
                            r_pending = cutlass.Boolean(False)
                            rv_in = []
                            for i in cutlass.range_constexpr(x_words // (4 * 32)):
                                r_idx = (
                                    x_row0 + rt_s * cutlass.Int32(x_words) + cutlass.Int32(i * 32 * 4)
                                    + lane1 * cutlass.Int32(4)
                                )  # fmt: skip
                                rv_in.append(
                                    prims.load_ext(
                                        rms_src.subview(r_idx),
                                        dtype=cutlass.Int32,
                                        count=4,
                                        order="relaxed",
                                        scope="gpu",
                                    )  # fmt: skip
                                )
                            for i in cutlass.range_constexpr(x_words // (4 * 32)):
                                rw = [cutlass.Int32(rv_in[i][q]) for q in range(4)]
                                rms_row.store(
                                    (rw[0], rw[1], rw[2], rw[3]),
                                    idx=cutlass.Int32(rj * x_words + i * 32 * 4)
                                    + lane1 * cutlass.Int32(4),
                                    alignment=16,
                                )
                                r_pending = r_pending | (
                                    (rw[0] == sent1)
                                    | (rw[1] == sent1)
                                    | (rw[2] == sent1)
                                    | (rw[3] == sent1)
                                )
                            if r_pending:
                                t_rback = _clock64()
                                while _clock64() - t_rback < cutlass.Int64(POLL_BACKOFF):
                                    pass
                        cute.arch.sync_warp()
                        if lane1 == cutlass.Int32(0):
                            prims.mbarrier_arrive(rms_full.subview(rj))
        elif cutlass.const_expr(x_src == 2):
            # The shared-expert activation (accumulator 1's k-tiles) is complete at launch; the latent k-tiles come
            # from the epilogue warps' reducers (lat_full).
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(
                    act_full, (k_tiles - x_tiles) * MMA_N * CTA_K * ELEM_BYTES
                )
                for k in cutlass.range_constexpr(x_tiles, k_tiles):
                    for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_b.subview(k * MMA_N * CTA_K + half * B_HALF_ELEMS),
                            tma_ptr_x2,
                            (
                                cutlass.Int32((k - x_tiles) * CTA_K + half * TMA_K_BOX),
                                cutlass.Int32(0),
                            ),
                            act_full,
                        )
        elif cutlass.const_expr(hooked):
            act_hook(smem_b_words, act_full, grid_bx, tx % cutlass.Int32(32))
        else:
            prims.griddepcontrol(prims.GridDepAction.WAIT)
        if cutlass.const_expr(tail and x_src == 0):
            # The whole latent row of each token this CTA takes the RMS of, one bulk copy each (the TMA engine, not
            # the epilogue warps' load path, which their phase 2 operand loads fill at the same time).
            if prims.elect_sync():
                for rj in cutlass.range_constexpr(LATENT_ROWS_PER_CTA):
                    rt_w1 = crank + cutlass.Int32(rj * CLUSTER)
                    if rt_w1 < num_tokens:
                        prims.mbarrier_arrive_expect_tx(rms_full.subview(rj), rms_cols * ELEM_BYTES)
                        prims.cp_async_bulk_shared_cluster_global(
                            rms_row.subview(rj * (rms_cols // 2)),
                            rms_src.subview(rt_w1 * cutlass.Int32(rms_cols // 2)),
                            rms_full.subview(rj),
                            rms_cols * ELEM_BYTES,
                        )
        if cutlass.const_expr(not hooked and x_src == 0):
            if prims.elect_sync():
                if cutlass.const_expr(swiglu):
                    # This CTA's k-tile only: the up half (from column x_col0 = k_in) into smem_b, the gate half into
                    # smem_g (the cluster CTAs share the SwiGLU tile by tile).
                    prims.mbarrier_arrive_expect_tx(act_full, 2 * MMA_N * CTA_K * ELEM_BYTES)
                    for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_b.subview(
                                crank * cutlass.Int32(MMA_N * CTA_K)
                                + cutlass.Int32(half * B_HALF_ELEMS)
                            ),
                            tma_ptr_x,
                            (
                                x_col0
                                + crank * cutlass.Int32(CTA_K)
                                + cutlass.Int32(half * TMA_K_BOX),
                                cutlass.Int32(0),
                            ),
                            act_full,
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_g.subview(
                                crank * cutlass.Int32(MMA_N * CTA_K)
                                + cutlass.Int32(half * B_HALF_ELEMS)
                            ),
                            tma_ptr_x,
                            (
                                crank * cutlass.Int32(CTA_K) + cutlass.Int32(half * TMA_K_BOX),
                                cutlass.Int32(0),
                            ),
                            act_full,
                        )
                else:
                    prims.mbarrier_arrive_expect_tx(act_full, k_tiles * MMA_N * CTA_K * ELEM_BYTES)
                for k in cutlass.range_constexpr(0 if swiglu else k_tiles):
                    for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                        if cutlass.const_expr(k < x_tiles):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                smem_b.subview(k * MMA_N * CTA_K + half * B_HALF_ELEMS),
                                tma_ptr_x,
                                (
                                    x_col0 + cutlass.Int32(k * CTA_K + half * TMA_K_BOX),
                                    cutlass.Int32(0),
                                ),
                                act_full,
                            )
                        else:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                smem_b.subview(k * MMA_N * CTA_K + half * B_HALF_ELEMS),
                                tma_ptr_x2,
                                (
                                    cutlass.Int32((k - x_tiles) * CTA_K + half * TMA_K_BOX),
                                    cutlass.Int32(0),
                                ),
                                act_full,
                            )
    elif warp_id == 2:
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
        tmem_raw = tmem_ptr_i32.load()
        tmem_acc1 = tmem_ptr
        if cutlass.const_expr(acc_pair):
            tmem_acc1 = cutlass.inttoptr(tmem_raw + cutlass.Int32(MMA_N), 6, cutlass.Int32)
        # Before the grid wait (under PDL): each landed weight k-tile goes on into TMEM, one 128 x K16 granule per copy
        # with the MMA's own descriptor; this thread issues the MMAs that read those columns later, in order. (The ring:
        # stage k % stages, its fill k // stages; the copies of a stage that is refilled later are committed to
        # stage_free, which warp 0 waits on.)
        for k in cutlass.range_constexpr(k_tiles):
            ws2 = k % stages
            while not cute.arch.mbarrier_try_wait(
                weight_full.subview(ws2).data_ptr(), (k // stages) % 2
            ):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if prims.elect_sync():
                for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                    box = kb // K_BLOCKS_PER_HALF
                    within = kb % K_BLOCKS_PER_HALF
                    prims.tcgen05_cp(
                        prims.Tcgen05CpShape.SHAPE_128X256B,
                        cutlass.inttoptr(
                            tmem_raw
                            + cutlass.Int32(
                                TMEM_A_COL
                                + (k * TMA_COPY_ITERS * K_BLOCKS_PER_HALF + kb)
                                * TMEM_A_COLS_PER_MMA
                            ),
                            6,
                            cutlass.Int32,
                        ),
                        desc_a_base + (ws2 * STAGE_A + box * A_BOX + within * STEP),
                        group=prims.CTAGroup.CTA_1,
                    )
                if cutlass.const_expr(ring and k + stages < k_tiles):
                    prims.tcgen05_commit(stage_free.subview(ws2))
        if cutlass.const_expr(swiglu):
            # Completed by the cluster CTAs' st.async of B (test_wait: a suspended try_wait wakes late on those); their
            # generic-proxy stores are then made visible to the MMA's async-proxy reads.
            while not _test_wait_cluster(b_ready.data_ptr(), 0):
                pass
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
        elif cutlass.const_expr(hooked or x_src == 1):
            # Completed by a thread's arrive (not TMA bytes): a suspended try_wait wakes up to µs late on those.
            while not cute.arch.mbarrier_test_wait(act_full.data_ptr(), 0):
                pass
        else:
            while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
                pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        # x_src 2: the shared-expert k-tiles first; the latent ones once the reducers' slice vectors are in.
        mma_order = (
            list(range(x_tiles, k_tiles)) + list(range(x_tiles))
            if x_src == 2
            else list(range(k_tiles))
        )
        for ki in cutlass.range_constexpr(k_tiles):
            k = mma_order[ki]
            second = (split_acc and k >= x_tiles) or (k_split == 2 and k % 2 == 1)
            first_tile = k == 0 or (split_acc and k == x_tiles) or (k_split == 2 and k == 1)
            if cutlass.const_expr(x_src == 2 and k == 0):
                # Completed by the peers' st.async (test_wait: a suspended try_wait wakes late on those); their
                # generic-proxy stores are then made visible to the MMA's async-proxy reads.
                while not _test_wait_cluster(lat_full.data_ptr(), 0):
                    pass
                prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                box = kb // K_BLOCKS_PER_HALF
                within = kb % K_BLOCKS_PER_HALF
                a_staged = cutlass.inttoptr(
                    tmem_raw
                    + cutlass.Int32(
                        TMEM_A_COL
                        + (k * TMA_COPY_ITERS * K_BLOCKS_PER_HALF + kb) * TMEM_A_COLS_PER_MMA
                    ),
                    6,
                    cutlass.Int32,
                )
                desc_b = desc_b_base + (k * STAGE_B + box * B_BOX + within * STEP)
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, tmem_acc1 if second else tmem_ptr,
                        a_staged, desc_b, idesc, not (first_tile and kb == 0),
                    )  # fmt: skip
        if prims.elect_sync():
            prims.tcgen05_commit(acc_done)

    if warp_id >= 4:
        tid = tx - cutlass.Int32(EPI_THREADS)
        lane = tx % 32
        w = warp_id - 4
        # The flag word, the prefix and the snapshots are written upstream (x_src 1: see the docstring).
        if cutlass.const_expr(x_src == 0):
            prims.griddepcontrol(prims.GridDepAction.WAIT)
        if cutlass.const_expr(swiglu):
            swiglu_share(smem_b, smem_g, act_full, b_ready, tid, crank)
        # This CTA's call count. The previous call of this CTA has completed (every kernel between two calls waits
        # for its predecessor), and the next one reads it only after this call has completed.
        calls = ws_flags.load(idx=bx, is_volatile=True)
        cur = calls & cutlass.Int32(1)

        col_word = (crank * cutlass.Int32(EPI_THREADS) + tid) * cutlass.Int32(ELTS // 2)
        if cutlass.const_expr(publish):
            # Re-arm the slab buffer after next: its last readers (the consumer of the call two back) completed
            # before this grid's wait returned (x_src 1: before this grid launched). Every thread empties its own 16
            # bytes of its cluster's token row, live token or not, so every row of the buffer is re-armed whatever M
            # the next calls run.
            rearm = slab_buf + cutlass.Int32(1)
            rearm = cutlass.Int32(
                cutlass.select_(rearm == cutlass.Int32(SLAB_BUFS), cutlass.Int32(0), rearm)
            )
            sent = cutlass.Int32(SLAB_SENTINEL)
            x_slab.store(
                (sent, sent, sent, sent),
                idx=rearm * cutlass.Int32(SLAB_WORDS)
                + token * cutlass.Int32(WORDS_PER_ROW)
                + col_word,
                alignment=16,
            )

        # ---- Phase 2 operands, loaded while the MMA runs (cluster `token` reduces token `token`). Clusters past
        # the last token load a valid row and discard it.
        reducer = token < num_tokens
        row_tok = cutlass.Int32(cutlass.select_(reducer, token, num_tokens - cutlass.Int32(1)))
        row_word = row_tok * cutlass.Int32(WORDS_PER_ROW) + col_word
        if cutlass.const_expr(tap_out):
            tap_word = row_tok * tap_stride + col_word
        # (x_src 1 and 2: L1-bypassing, as no grid wait orders them.)
        pre = prefix.load(idx=row_word, vector_size=4, alignment=16, is_volatile=x_src >= 1)
        # Snapshots past the last one reload snapshot 0 (their candidates are masked below).
        snap = []
        rw = pre
        mw = pre
        if cutlass.const_expr(not plain):
            for n in cutlass.range_constexpr(MAX_SNAPSHOTS):
                n_row = cutlass.Int32(
                    cutlass.select_(
                        cutlass.Int32(n) < num_cand - cutlass.Int32(1),
                        cutlass.Int32(n),
                        cutlass.Int32(0),
                    )
                )
                snap.append(
                    snapshots.load(
                        idx=(n_row * num_tokens + row_tok) * cutlass.Int32(WORDS_PER_ROW)
                        + col_word,
                        vector_size=4,
                        alignment=16,
                        is_volatile=x_src >= 1,
                    )
                )
            rw = res_w.load(idx=col_word, vector_size=4, alignment=16)
            mw = rms_w.load(idx=col_word, vector_size=4, alignment=16)
        ow = out_w.load(idx=col_word, vector_size=4, alignment=16)

        n_calls = cutlass.Int32(0)
        if cutlass.const_expr(x_src == 2):
            # ---- The latent all-reduce, folded in (see the docstring). The previous call's CTA 0 stored this call's
            # count after its grid wait, and that call completed before this launch.
            n_calls = cutlass.Int32(lat_flags.load(idx=0, is_volatile=True))
            half_words = MAX_TOKENS * world * LAT_WORDS
            cur_h = n_calls & cutlass.Int32(1)
            h_base = cur_h * cutlass.Int32(half_words)
            # The other half (the next call's) emptied first: this CTA's 1/56 of it, 16-byte stores.
            e_base = (cur_h ^ cutlass.Int32(1)) * cutlass.Int32(half_words) + bx * cutlass.Int32(
                half_words // NUM_CTAS
            )
            empty_l = cutlass.Int32(EMPTY_WORD)
            for q in cutlass.range_constexpr(half_words // NUM_CTAS // (EPI_THREADS * 4)):
                rms_src.store((empty_l, empty_l, empty_l, empty_l),
                              idx=e_base + cutlass.Int32(q * EPI_THREADS * 4) + tid * cutlass.Int32(4),
                              alignment=16)  # fmt: skip
            if bx == cutlass.Int32(0):
                if tid < cutlass.Int32(MAX_TOKENS):
                    next_sb = (n_calls + cutlass.Int32(1)) % cutlass.Int32(LAT_SCALE_BUFS)
                    lat_flags.store(cutlass.Int32(SCALE_SENTINEL),
                                    idx=cutlass.Int32(LAT_SCALES) + next_sb * cutlass.Int32(MAX_TOKENS) + tid,
                                    is_volatile=True)  # fmt: skip
            # Rank chunks of 8 (world <= 8: one chunk of world ranks); a thread sums one chunk of one 16-byte vector,
            # the pair's even thread adds the two chunks from 0 as the one-shot does, rounds once and sends.
            chunks = (world + RANK_CHUNK - 1) // RANK_CHUNK
            per_chunk = min(RANK_CHUNK, world)
            g = crank * cutlass.Int32(EPI_THREADS) + tid
            g_chunk = g % cutlass.Int32(chunks)
            g_even = g_chunk == cutlass.Int32(0)

            # (1) The rank's slice of every live row, into all 7 CTAs' smem_b (the latent k-tiles).
            s_live = g < num_tokens * cutlass.Int32(LAT_SLICE_VECS * chunks)
            s_pair = g // cutlass.Int32(chunks)
            s_tok = cutlass.Int32(
                cutlass.select_(s_live, s_pair // cutlass.Int32(LAT_SLICE_VECS), cutlass.Int32(0))
            )
            s_vec = s_pair % cutlass.Int32(LAT_SLICE_VECS)
            s_base = (
                h_base + (s_tok * cutlass.Int32(world) + g_chunk * cutlass.Int32(RANK_CHUNK)) * cutlass.Int32(LAT_WORDS)
                + x_col0 // cutlass.Int32(2) + s_vec * cutlass.Int32(4)
            )  # fmt: skip
            s0 = cutlass.Float32(0.0)
            s1 = cutlass.Float32(0.0)
            s2 = cutlass.Float32(0.0)
            s3 = cutlass.Float32(0.0)
            s4 = cutlass.Float32(0.0)
            s5 = cutlass.Float32(0.0)
            s6 = cutlass.Float32(0.0)
            s7 = cutlass.Float32(0.0)
            s_pending = s_live
            while s_pending:
                s_dirty = cutlass.Boolean(False)
                sc = [cutlass.Float32(0.0)] * ELTS
                for rr in cutlass.range_constexpr(per_chunk):
                    spv = rms_src.load(idx=s_base + cutlass.Int32(rr * LAT_WORDS), vector_size=4, alignment=16,
                                       is_volatile=True)  # fmt: skip
                    for q in cutlass.range_constexpr(4):
                        sword = cutlass.Int32(spv[q])
                        s_dirty = s_dirty | (sword == cutlass.Int32(EMPTY_WORD))
                        sc[2 * q] = sc[2 * q] + _lo(sword)
                        sc[2 * q + 1] = sc[2 * q + 1] + _hi(sword)
                s0, s1, s2, s3, s4, s5, s6, s7 = sc
                s_pending = s_dirty
            cute.arch.sync_warp()
            s_mine = [s0, s1, s2, s3, s4, s5, s6, s7]
            s_tot = []
            for e in cutlass.range_constexpr(ELTS):
                if cutlass.const_expr(chunks == 2):
                    s_other = cute.arch.shuffle_sync_bfly(s_mine[e], offset=1)
                    s_c0 = cutlass.Float32(cutlass.select_(g_even, s_mine[e], s_other))
                    s_c1 = cutlass.Float32(cutlass.select_(g_even, s_other, s_mine[e]))
                    s_tot.append((cutlass.Float32(0.0) + s_c0) + s_c1)
                else:
                    s_tot.append(cutlass.Float32(0.0) + s_mine[e])
            if s_live & g_even:
                s_dst = act_word_index(s_tok, s_vec * cutlass.Int32(8))
                for pc in cutlass.range_constexpr(CLUSTER):
                    _st_async_v4(
                        _mapa_u32(smem_b_words.subview(s_dst).data_ptr(), cutlass.Int32(pc)),
                        _pack_bf16x2(s_tot[1], s_tot[0]), _pack_bf16x2(s_tot[3], s_tot[2]),
                        _pack_bf16x2(s_tot[5], s_tot[4]), _pack_bf16x2(s_tot[7], s_tot[6]),
                        _mapa_u32(lat_full.data_ptr(), cutlass.Int32(pc)),
                    )  # fmt: skip

            # (2) Cluster t: token t's whole row, into CTA 0's rms_row.
            r_live = (token < num_tokens) & (g < cutlass.Int32(LAT_VECS * chunks))
            r_tok = cutlass.Int32(cutlass.select_(token < num_tokens, token, cutlass.Int32(0)))
            r_vec = g // cutlass.Int32(chunks)
            r_base = (
                h_base + (r_tok * cutlass.Int32(world) + g_chunk * cutlass.Int32(RANK_CHUNK)) * cutlass.Int32(LAT_WORDS)
                + r_vec * cutlass.Int32(4)
            )  # fmt: skip
            r0 = cutlass.Float32(0.0)
            r1 = cutlass.Float32(0.0)
            r2 = cutlass.Float32(0.0)
            r3 = cutlass.Float32(0.0)
            r4 = cutlass.Float32(0.0)
            r5 = cutlass.Float32(0.0)
            r6 = cutlass.Float32(0.0)
            r7 = cutlass.Float32(0.0)
            r_pending = r_live
            while r_pending:
                r_dirty = cutlass.Boolean(False)
                rc = [cutlass.Float32(0.0)] * ELTS
                for rr in cutlass.range_constexpr(per_chunk):
                    rpv = rms_src.load(idx=r_base + cutlass.Int32(rr * LAT_WORDS), vector_size=4, alignment=16,
                                       is_volatile=True)  # fmt: skip
                    for q in cutlass.range_constexpr(4):
                        rword = cutlass.Int32(rpv[q])
                        r_dirty = r_dirty | (rword == cutlass.Int32(EMPTY_WORD))
                        rc[2 * q] = rc[2 * q] + _lo(rword)
                        rc[2 * q + 1] = rc[2 * q + 1] + _hi(rword)
                r0, r1, r2, r3, r4, r5, r6, r7 = rc
                r_pending = r_dirty
            cute.arch.sync_warp()
            r_mine = [r0, r1, r2, r3, r4, r5, r6, r7]
            r_tot = []
            for e in cutlass.range_constexpr(ELTS):
                if cutlass.const_expr(chunks == 2):
                    r_other = cute.arch.shuffle_sync_bfly(r_mine[e], offset=1)
                    r_c0 = cutlass.Float32(cutlass.select_(g_even, r_mine[e], r_other))
                    r_c1 = cutlass.Float32(cutlass.select_(g_even, r_other, r_mine[e]))
                    r_tot.append((cutlass.Float32(0.0) + r_c0) + r_c1)
                else:
                    r_tot.append(cutlass.Float32(0.0) + r_mine[e])
            if r_live & g_even:
                _st_async_v4(
                    _mapa_u32(rms_row.subview(r_vec * cutlass.Int32(4)).data_ptr(), cutlass.Int32(0)),
                    _pack_bf16x2(r_tot[1], r_tot[0]), _pack_bf16x2(r_tot[3], r_tot[2]),
                    _pack_bf16x2(r_tot[5], r_tot[4]), _pack_bf16x2(r_tot[7], r_tot[6]),
                    _mapa_u32(rms_full.subview(0).data_ptr(), cutlass.Int32(0)),
                )  # fmt: skip

            # (3) CTA 0's warp 4 of cluster t: token t's latent RMS, as the bulk-copy path computes it, published into
            # buffer n % 3 of the scale slab.
            if (crank == cutlass.Int32(0)) & (w == cutlass.Int32(0)) & (token < num_tokens):
                while not _test_wait_cluster(rms_full.subview(0).data_ptr(), 0):
                    pass
                cute.arch.sync_warp()
                lat_rows_o = []
                for rv in cutlass.range_constexpr(rms_cols // (8 * 32)):
                    lat_rows_o.append(
                        rms_row.load(
                            idx=cutlass.Int32(rv * 32 * 4) + lane * cutlass.Int32(4),
                            vector_size=4,
                            alignment=16,
                        )
                    )
                lat_sq_o = cutlass.Float32(0.0)
                for rv in cutlass.range_constexpr(rms_cols // (8 * 32)):
                    for ri in cutlass.range_constexpr(4):
                        lat_lo_o = _lo(cutlass.Int32(lat_rows_o[rv][ri]))
                        lat_hi_o = _hi(cutlass.Int32(lat_rows_o[rv][ri]))
                        lat_sq_o = lat_sq_o + lat_lo_o * lat_lo_o + lat_hi_o * lat_hi_o
                for offset in (16, 8, 4, 2, 1):
                    lat_sq_o = lat_sq_o + cute.arch.shuffle_sync_bfly(lat_sq_o, offset=offset)
                scale_o = cute.math.rsqrt(lat_sq_o * cutlass.Float32(1.0 / rms_cols) + lat_eps)
                if lane == cutlass.Int32(0):
                    lat_flags.store(
                        scale_o.bitcast(cutlass.Int32),
                        idx=cutlass.Int32(LAT_SCALES)
                        + (n_calls % cutlass.Int32(LAT_SCALE_BUFS)) * cutlass.Int32(MAX_TOKENS)
                        + token,
                        is_volatile=True,
                    )  # fmt: skip

        if cutlass.const_expr(tail and x_src != 2):
            # Per-token latent RMS while the MMA runs: cluster CTA c computes tokens c and c + 7 (epilogue warps 0 and
            # 1) from its bulk-copied row: each lane sums the squares of its 14 16-byte vectors in order, then a
            # butterfly; the scale goes by st.async into row_scale of every cluster CTA, completing mb_rms there.
            for rj in cutlass.range_constexpr(LATENT_ROWS_PER_CTA):
                rt_c = crank + cutlass.Int32(rj * CLUSTER)
                if w == cutlass.Int32(rj):
                    if rt_c < num_tokens:
                        if cutlass.const_expr(x_src == 1):
                            # Completed by warp 1's arrive after its polled stores (test_wait: see act_full).
                            while not cute.arch.mbarrier_test_wait(
                                rms_full.subview(rj).data_ptr(), 0
                            ):
                                pass
                        else:
                            while not cute.arch.mbarrier_try_wait(
                                rms_full.subview(rj).data_ptr(), 0
                            ):
                                pass
                        cute.arch.sync_warp()
                        lat_rows_c = []
                        for rv in cutlass.range_constexpr(rms_cols // (8 * 32)):
                            lat_rows_c.append(
                                rms_row.load(
                                    idx=cutlass.Int32(rj * (rms_cols // 2) + rv * 32 * 4)
                                    + lane * cutlass.Int32(4),
                                    vector_size=4,
                                    alignment=16,
                                )
                            )
                        lat_sq_c = cutlass.Float32(0.0)
                        for rv in cutlass.range_constexpr(rms_cols // (8 * 32)):
                            for ri in cutlass.range_constexpr(4):
                                lat_lo_c = _lo(cutlass.Int32(lat_rows_c[rv][ri]))
                                lat_hi_c = _hi(cutlass.Int32(lat_rows_c[rv][ri]))
                                lat_sq_c = lat_sq_c + lat_lo_c * lat_lo_c + lat_hi_c * lat_hi_c
                        for offset in (16, 8, 4, 2, 1):
                            lat_sq_c = lat_sq_c + cute.arch.shuffle_sync_bfly(
                                lat_sq_c, offset=offset
                            )
                        scale_c = cute.math.rsqrt(
                            lat_sq_c * cutlass.Float32(1.0 / rms_cols) + lat_eps
                        )
                        if lane < cutlass.Int32(CLUSTER):
                            _st_async_f32(
                                _mapa_u32(row_scale.subview(rt_c).data_ptr(), lane),
                                scale_c,
                                _mapa_u32(mb_rms.data_ptr(), lane),
                            )

        # ---- Phase 1 epilogue: TMEM -> bf16 pairs (even lane: its row and the next) -> 16-byte pushes.
        # (Every wait loop and lane-dependent branch before a warp shuffle ends in a warp sync: lanes may leave a
        # wait loop in different iterations, and a shuffle on a diverged warp takes the slow collective path.)
        while not cute.arch.mbarrier_try_wait(acc_done.data_ptr(), 0):
            pass
        cute.arch.sync_warp()
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Float32), num=MMA_N
        )
        acc1 = acc
        if cutlass.const_expr(acc_pair):
            acc1 = prims.tcgen05_ld(
                "32x32b",
                cutlass.inttoptr(tmem_ptr_i32.load() + cutlass.Int32(MMA_N), 6, cutlass.Float32),
                num=MMA_N,
            )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        scales = []
        if cutlass.const_expr(tail and x_src == 2):
            # The live tokens' scales from buffer n % 3 of the scale slab (published by each token's cluster).
            sb_idx = cutlass.Int32(LAT_SCALES) + (
                n_calls % cutlass.Int32(LAT_SCALE_BUFS)
            ) * cutlass.Int32(MAX_TOKENS)
            sw0 = cutlass.Int32(SCALE_SENTINEL)
            sw1 = cutlass.Int32(SCALE_SENTINEL)
            sw2 = cutlass.Int32(SCALE_SENTINEL)
            sw3 = cutlass.Int32(SCALE_SENTINEL)
            sw4 = cutlass.Int32(SCALE_SENTINEL)
            sw5 = cutlass.Int32(SCALE_SENTINEL)
            sw6 = cutlass.Int32(SCALE_SENTINEL)
            sw7 = cutlass.Int32(SCALE_SENTINEL)
            sc_pending = cutlass.Boolean(True)
            while sc_pending:
                sv_a = lat_flags.load(idx=sb_idx, vector_size=4, alignment=16, is_volatile=True)
                sv_b = lat_flags.load(
                    idx=sb_idx + cutlass.Int32(4), vector_size=4, alignment=16, is_volatile=True
                )
                sws = [cutlass.Int32(sv_a[q]) for q in range(4)] + [
                    cutlass.Int32(sv_b[q]) for q in range(4)
                ]
                sc_dirty = cutlass.Boolean(False)
                for t in cutlass.range_constexpr(MMA_N):
                    sc_dirty = sc_dirty | (
                        (cutlass.Int32(t) < num_tokens) & (sws[t] == cutlass.Int32(SCALE_SENTINEL))
                    )
                sw0, sw1, sw2, sw3, sw4, sw5, sw6, sw7 = sws
                sc_pending = sc_dirty
            cute.arch.sync_warp()
            for sword_t in (sw0, sw1, sw2, sw3, sw4, sw5, sw6, sw7):
                scales.append(cutlass.Int32(sword_t).bitcast(cutlass.Float32))
        elif cutlass.const_expr(tail):
            while not _test_wait_cluster(mb_rms.data_ptr(), 0):
                pass
            cute.arch.sync_warp()
            # Every token's scale at once (rows past the last token are scaled but never pushed).
            for sv in cutlass.range_constexpr(MMA_N // 4):
                scale_vec = row_scale.load(idx=sv * 4, vector_size=4, alignment=16)
                for si in cutlass.range_constexpr(4):
                    scales.append(cutlass.Float32(scale_vec[si]))
        row = w * cutlass.Int32(32) + lane
        for t in cutlass.range_constexpr(MMA_N):
            mine = cutlass.Float32(acc[t])
            if cutlass.const_expr(tail):
                mine = mine * scales[t]
            if cutlass.const_expr(split_acc):
                mine = mine + cutlass.Float32(acc1[t])
            if cutlass.const_expr(k_split == 2):
                # Split 2's sums: the rank partials added from zero in rank order.
                mine = (cutlass.Float32(0.0) + mine) + cutlass.Float32(acc1[t])
            odd = cute.arch.shuffle_sync_bfly(mine, offset=1)
            if lane % 2 == 0:
                stage.store(
                    _pack_bf16x2(odd, mine), idx=cutlass.Int32(t * ROW_WORDS_PER_CTA) + row // 2
                )
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
        push_tok = tid // cutlass.Int32(16)
        push_vec = tid % cutlass.Int32(16)
        if push_tok < num_tokens:
            v = stage.load(
                idx=push_tok * cutlass.Int32(ROW_WORDS_PER_CTA) + push_vec * cutlass.Int32(4),
                vector_size=4,
                alignment=16,
            )
            if cutlass.const_expr(x_src == 2):
                # The epilogue warps' empties of the latent half (ordered before this thread by the stage barrier)
                # before the push: a rank's next push into that half follows its phase 2 of this call.
                _fence_sys()
            slot = (
                (cur * cutlass.Int32(MAX_TOKENS) + push_tok) * cutlass.Int32(world) + rank
            ) * cutlass.Int32(WORDS_PER_ROW)
            ws_mc.store(
                (_sanitize(v[0]), _sanitize(v[1]), _sanitize(v[2]), _sanitize(v[3])),
                idx=slot + (m_offset // cutlass.Int32(2)) + push_vec * cutlass.Int32(4),
                alignment=16,
            )
            prims.fence_acq_rel(prims.MemScope.CLUSTER)
        # The dependents launch once every CTA has pushed: they stream their weights during the exchange without
        # competing with phase 1, and they wait for (or poll) this grid's outputs.
        if tid == 0:
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        # After the stage barrier: every TMEM read of this CTA is done and every epilogue thread has read the count.
        if warp_id == 4:
            prims.tcgen05_dealloc(tmem_ptr, TMEM_COLS)
        if tid == 0:
            ws_flags.store(calls + cutlass.Int32(1), idx=bx, is_volatile=True)

        # ---- Phase 2: reduce token `token` over the ranks, then the attn_res + RMSNorm epilogue.
        cute.arch.sync_warp()
        if reducer:
            if cutlass.const_expr(not plain):
                # res_w * rms_w of this thread's columns.
                qv = []
                for q in cutlass.range_constexpr(4):
                    rword = cutlass.Int32(rw[q])
                    mword = cutlass.Int32(mw[q])
                    qv.append(_lo(rword) * _lo(mword))
                    qv.append(_hi(rword) * _hi(mword))

                # The snapshots' statistics, while the peers' rows are in flight: warp sums (value i of 16 in lanes
                # 2 i, 2 i + 1), the CTA sum over the warps in order, sent to slot [this CTA] of every cluster CTA.
                snap_stats = []
                for n in cutlass.range_constexpr(MAX_SNAPSHOTS):
                    sum_sq = cutlass.Float32(0.0)
                    dot = cutlass.Float32(0.0)
                    for e in cutlass.range_constexpr(ELTS):
                        word = cutlass.Int32(snap[n][e // 2])
                        val = _lo(word) if e % 2 == 0 else _hi(word)
                        sum_sq = val * val + sum_sq
                        dot = val * qv[e] + dot
                    snap_stats.append(sum_sq)
                    snap_stats.append(dot)
                snap_sum = _warp_sum_scatter(snap_stats, lane)
                if lane % 2 == 0:
                    warp_snap.store(
                        snap_sum, idx=w * cutlass.Int32(SNAP_STATS) + lane // cutlass.Int32(2)
                    )
                prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
                if tid < cutlass.Int32(CLUSTER * SNAP_STATS):
                    peer = tid // cutlass.Int32(SNAP_STATS)
                    s = tid % cutlass.Int32(SNAP_STATS)
                    if s < cutlass.Int32(2) * (num_cand - cutlass.Int32(1)):
                        partial = cutlass.Float32(0.0)
                        for ww in cutlass.range_constexpr(EPI_WARPS):
                            partial = partial + warp_snap.load(
                                idx=cutlass.Int32(ww * SNAP_STATS) + s
                            )
                        _st_async_f32(
                            _mapa_u32(
                                box_snap.subview(crank * cutlass.Int32(SNAP_STATS) + s).data_ptr(),
                                peer,
                            ),
                            partial,
                            _mapa_u32(mb_snap.data_ptr(), peer),
                        )
                # The snapshots' logits and their maximum need only the cluster's snapshot statistics: taken here,
                # while the ranks' rows are in flight (lane n owns snapshot n).
                while not _test_wait_cluster(mb_snap.data_ptr(), 0):
                    pass
                cute.arch.sync_warp()
                snap_lane = cutlass.Int32(
                    cutlass.select_(lane < cutlass.Int32(MAX_SNAPSHOTS), lane, cutlass.Int32(0))
                )
                box_s = []
                for r in cutlass.range_constexpr(CLUSTER):
                    box_s.append(
                        box_snap.load(
                            idx=cutlass.Int32(r * SNAP_STATS) + cutlass.Int32(2) * snap_lane,
                            vector_size=2,
                            alignment=8,
                        )
                    )
                snap_tot_sq = cutlass.Float32(0.0)
                snap_tot_dot = cutlass.Float32(0.0)
                for r in cutlass.range_constexpr(CLUSTER):
                    snap_tot_sq = snap_tot_sq + cutlass.Float32(box_s[r][0])
                    snap_tot_dot = snap_tot_dot + cutlass.Float32(box_s[r][1])
                is_snap = lane < num_cand - cutlass.Int32(1)
                snap_logit = cutlass.Float32(
                    cutlass.select_(
                        is_snap,
                        snap_tot_dot * cute.math.rsqrt(snap_tot_sq / cutlass.Float32(H) + rms_eps),
                        cutlass.Float32(-3.4028234663852886e38),
                    )
                )
                max_snap = snap_logit
                for offset in (16, 8, 4, 2, 1):
                    max_snap = _fmax(max_snap, cute.arch.shuffle_sync_bfly(max_snap, offset=offset))

            slot0 = (cur * cutlass.Int32(MAX_TOKENS) + token) * cutlass.Int32(
                world * WORDS_PER_ROW
            ) + col_word
            a0 = cutlass.Float32(0.0)
            a1 = cutlass.Float32(0.0)
            a2 = cutlass.Float32(0.0)
            a3 = cutlass.Float32(0.0)
            a4 = cutlass.Float32(0.0)
            a5 = cutlass.Float32(0.0)
            a6 = cutlass.Float32(0.0)
            a7 = cutlass.Float32(0.0)
            pending = cutlass.Boolean(True)
            while pending:
                dirty = cutlass.Boolean(False)
                total = [cutlass.Float32(0.0)] * ELTS
                for rb in cutlass.range_constexpr(0, world, RANK_CHUNK):
                    chunk = [cutlass.Float32(0.0)] * ELTS
                    for rr in cutlass.range_constexpr(min(RANK_CHUNK, world - rb)):
                        pv = ws_uc.load(
                            idx=slot0 + cutlass.Int32((rb + rr) * WORDS_PER_ROW),
                            vector_size=4,
                            alignment=16,
                            is_volatile=True,
                        )
                        for q in cutlass.range_constexpr(4):
                            pword = cutlass.Int32(pv[q])
                            dirty = dirty | (pword == cutlass.Int32(EMPTY_WORD))
                            chunk[2 * q] = chunk[2 * q] + _lo(pword)
                            chunk[2 * q + 1] = chunk[2 * q + 1] + _hi(pword)
                    for e in cutlass.range_constexpr(ELTS):
                        total[e] = total[e] + chunk[e]
                a0, a1, a2, a3, a4, a5, a6, a7 = total
                pending = dirty
            cute.arch.sync_warp()
            acc_sum = [a0, a1, a2, a3, a4, a5, a6, a7]

            # updated = bf16(prefix + bf16(sum)) (bf16(sum) without the prefix).
            upd_words = []
            for q in cutlass.range_constexpr(4):
                red = _pack_bf16x2(acc_sum[2 * q + 1], acc_sum[2 * q])
                pw = cutlass.Int32(pre[q])
                with_prefix = _pack_bf16x2(_hi(pw) + _hi(red), _lo(pw) + _lo(red))
                upd_words.append(
                    cutlass.Int32(cutlass.select_(add_prefix != cutlass.Int32(0), with_prefix, red))
                )

            if cutlass.const_expr(plain):
                # The MNNVL one-shot's residual + RMSNorm (kARResidualRMSNorm), its arithmetic and reduction order.
                updated.store(
                    (upd_words[0], upd_words[1], upd_words[2], upd_words[3]),
                    idx=row_word,
                    alignment=16,
                )
                empty_p = cutlass.Int32(EMPTY_WORD)
                for r in cutlass.range_constexpr(world):
                    ws_uc.store((empty_p, empty_p, empty_p, empty_p), idx=slot0 + cutlass.Int32(r * WORDS_PER_ROW),
                                alignment=16)  # fmt: skip
                # This thread's sum of squares in element order: bf16 products (x * x rounded to bf16; plain 2: fp32,
                # exact for bf16 x).
                t_sq = cutlass.Float32(0.0)
                for e in cutlass.range_constexpr(ELTS):
                    word = upd_words[e // 2]
                    val = _lo(word) if e % 2 == 0 else _hi(word)
                    if cutlass.const_expr(plain == 2):
                        t_sq = t_sq + val * val
                    else:
                        sq_pair = _pack_bf16x2(val * val, val * val)
                        t_sq = t_sq + _lo(sq_pair)
                # To slot [this thread's group] of every cluster CTA.
                my_group = crank * cutlass.Int32(EPI_THREADS) + tid
                for pc in cutlass.range_constexpr(CLUSTER):
                    _st_async_f32(
                        _mapa_u32(box_ts.subview(my_group).data_ptr(), cutlass.Int32(pc)), t_sq,
                        _mapa_u32(mb_ts.data_ptr(), cutlass.Int32(pc)),
                    )  # fmt: skip
                while not _test_wait_cluster(mb_ts.data_ptr(), 0):
                    pass
                cute.arch.sync_warp()
                if w == cutlass.Int32(0):
                    # The one-shot's tree over the 896 thread sums. MNNVL (8 blocks of 112 threads): lane k takes warp
                    # k % 4 of block k // 4 (32 thread sums; the fourth warp has 16, the rest zero) through the xor
                    # butterfly; lane 4 b then forms block b's sum (s0 + s2) + (s1 + s3), and lane 0 adds the
                    # blocks. IPC (plain 2, 4 blocks of 224 threads): lane k < 28 takes warp k % 7 of block k // 7
                    # (thread sums [32 k, 32 k + 32)); lane 7 b forms block b's sum as blockReduceSumV2's butterfly
                    # over the 7 warp sums and 25 zeros, ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + s3); lane 0 adds the
                    # blocks.
                    vw_live = lane < cutlass.Int32(28 if plain == 2 else 32)
                    if cutlass.const_expr(plain == 2):
                        vw_base = lane * cutlass.Int32(32)
                        vw_count = cutlass.Int32(32)
                    else:
                        vw_base = (lane // cutlass.Int32(4)) * cutlass.Int32(112) + (
                            lane % cutlass.Int32(4)
                        ) * cutlass.Int32(32)
                        vw_count = cutlass.Int32(cutlass.select_(lane % cutlass.Int32(4) == cutlass.Int32(3),
                                                                 cutlass.Int32(16), cutlass.Int32(32)))  # fmt: skip
                    vals = []
                    for i in cutlass.range_constexpr(32):
                        v_i = box_ts.load(
                            idx=cutlass.Int32(
                                cutlass.select_(
                                    vw_live, vw_base + cutlass.Int32(i), cutlass.Int32(0)
                                )
                            )
                        )
                        vals.append(
                            cutlass.Float32(
                                cutlass.select_(
                                    cutlass.Int32(i) < vw_count, v_i, cutlass.Float32(0.0)
                                )
                            )
                        )
                    for m in (16, 8, 4, 2, 1):
                        vals = [vals[ln] + vals[ln ^ m] for ln in range(32)]
                    warp_sum = vals[0]
                    s_w = [warp_sum]
                    for j in cutlass.range_constexpr(1, 7 if plain == 2 else 4):
                        s_w.append(
                            cute.arch.shuffle_sync(
                                warp_sum, offset=(lane + cutlass.Int32(j)) % cutlass.Int32(32)
                            )
                        )
                    if cutlass.const_expr(plain == 2):
                        block_sum = ((s_w[0] + s_w[4]) + (s_w[2] + s_w[6])) + (
                            (s_w[1] + s_w[5]) + s_w[3]
                        )
                    else:
                        block_sum = (s_w[0] + s_w[2]) + (s_w[1] + s_w[3])
                    full_sum = cutlass.Float32(0.0)
                    for b in cutlass.range_constexpr(4 if plain == 2 else 8):
                        full_sum = full_sum + cute.arch.shuffle_sync(
                            block_sum, offset=(7 if plain == 2 else 4) * b
                        )
                    r_plain = cute.math.rsqrt(full_sum / cutlass.Float32(H) + out_eps)
                    if lane == cutlass.Int32(0):
                        row_scale.store(r_plain, idx=0)
                prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
                rsigma_p = row_scale.load(idx=0)
                out_words = []
                for q in cutlass.range_constexpr(4):
                    oword = cutlass.Int32(ow[q])
                    x_word = upd_words[q]
                    out_words.append(
                        _pack_bf16x2(
                            _hi(x_word) * rsigma_p * _hi(oword), _lo(x_word) * rsigma_p * _lo(oword)
                        )
                    )
                normed.store(
                    (out_words[0], out_words[1], out_words[2], out_words[3]),
                    idx=row_word,
                    alignment=16,
                )
                if cutlass.const_expr(publish):
                    x_slab.store(
                        (_slab_word(out_words[0]), _slab_word(out_words[1]), _slab_word(out_words[2]),
                         _slab_word(out_words[3])),
                        idx=slab_buf * cutlass.Int32(SLAB_WORDS) + row_word,
                        alignment=16,
                    )  # fmt: skip
            else:
                # The updated sum's statistics: warp sums (sum of squares in lanes 0-15, projection in 16-31), sent per
                # warp to slot [this CTA][warp] of every cluster CTA.
                u_sq = cutlass.Float32(0.0)
                u_dot = cutlass.Float32(0.0)
                for e in cutlass.range_constexpr(ELTS):
                    word = upd_words[e // 2]
                    val = _lo(word) if e % 2 == 0 else _hi(word)
                    u_sq = val * val + u_sq
                    u_dot = val * qv[e] + u_dot
                upd_sum = _warp_sum_scatter([u_sq, u_dot], lane)
                u_peer = lane % cutlass.Int32(16)
                if u_peer < cutlass.Int32(CLUSTER):
                    _st_async_f32(
                        _mapa_u32(
                            box_upd.subview(
                                (crank * cutlass.Int32(EPI_WARPS) + w) * cutlass.Int32(2)
                                + lane // cutlass.Int32(16)
                            ).data_ptr(),
                            u_peer,
                        ),
                        upd_sum,
                        _mapa_u32(mb_upd.data_ptr(), u_peer),
                    )
                updated.store(
                    (upd_words[0], upd_words[1], upd_words[2], upd_words[3]),
                    idx=row_word,
                    alignment=16,
                )
                if cutlass.const_expr(tap_out == 2):
                    # The running prefix sum (a capture layer that taps the prefix rather than the mixture).
                    tap.store(
                        (upd_words[0], upd_words[1], upd_words[2], upd_words[3]),
                        idx=tap_word,
                        alignment=16,
                    )
                # Every rank's single push of this call into these words has landed; empty them for the call after next.
                empty = cutlass.Int32(EMPTY_WORD)
                for r in cutlass.range_constexpr(world):
                    ws_uc.store(
                        (empty, empty, empty, empty),
                        idx=slot0 + cutlass.Int32(r * WORDS_PER_ROW),
                        alignment=16,
                    )
                # Candidate n < num_cand - 1 is snapshot n, candidate num_cand - 1 is updated; later ones are masked.
                cand = []
                for n in cutlass.range_constexpr(MAX_CANDIDATES):
                    vals = []
                    for q in cutlass.range_constexpr(4):
                        word = upd_words[q]
                        if cutlass.const_expr(n < MAX_CANDIDATES - 1):
                            word = cutlass.Int32(
                                cutlass.select_(
                                    cutlass.Int32(n) == num_cand - cutlass.Int32(1),
                                    upd_words[q],
                                    cutlass.Int32(snap[n][q]),
                                )
                            )
                        vals.append(_lo(word))
                        vals.append(_hi(word))
                    cand.append(vals)

                # Every warp: the candidates' logits (lane n owns candidate n: snapshot n < num_cand - 1, then updated),
                # each statistic summed over warps, then over the cluster's CTAs in rank order. The snapshots' logits
                # and their maximum were taken before the poll; now the updated sum's.
                while not _test_wait_cluster(mb_upd.data_ptr(), 0):
                    pass
                cute.arch.sync_warp()
                box_u = []
                for r in cutlass.range_constexpr(CLUSTER):
                    for hv in cutlass.range_constexpr(EPI_WARPS * 2 // 4):
                        box_u.append(
                            box_upd.load(
                                idx=r * EPI_WARPS * 2 + hv * 4, vector_size=4, alignment=16
                            )
                        )
                upd_tot_sq = cutlass.Float32(0.0)
                upd_tot_dot = cutlass.Float32(0.0)
                for r in cutlass.range_constexpr(CLUSTER):
                    part_sq = cutlass.Float32(0.0)
                    part_dot = cutlass.Float32(0.0)
                    for ww in cutlass.range_constexpr(EPI_WARPS):
                        part_sq = part_sq + cutlass.Float32(box_u[r * 2 + ww // 2][(ww % 2) * 2])
                        part_dot = part_dot + cutlass.Float32(
                            box_u[r * 2 + ww // 2][(ww % 2) * 2 + 1]
                        )
                    upd_tot_sq = upd_tot_sq + part_sq
                    upd_tot_dot = upd_tot_dot + part_dot
                upd_logit = upd_tot_dot * cute.math.rsqrt(upd_tot_sq / cutlass.Float32(H) + rms_eps)
                # The maximum over every lane's logit is exact whatever the order it is taken in.
                max_logit = _fmax(max_snap, upd_logit)
                live = lane < num_cand
                logit = cutlass.Float32(
                    cutlass.select_(
                        is_snap,
                        snap_logit,
                        cutlass.select_(
                            lane == num_cand - cutlass.Int32(1),
                            upd_logit,
                            cutlass.Float32(-3.4028234663852886e38),
                        ),
                    )
                )
                weight = cutlass.Float32(0.0)
                if live:
                    weight = cute.math.exp2(
                        (logit - max_logit) * cutlass.Float32(LOG2E), fastmath=True
                    )
                denominator = weight
                for offset in (16, 8, 4, 2, 1):
                    denominator = denominator + cute.arch.shuffle_sync_bfly(
                        denominator, offset=offset
                    )
                # Zero past num_cand, so the masked candidates add nothing below.
                lane_weight = weight * (cutlass.Float32(1.0) / denominator)
                weights = [
                    cute.arch.shuffle_sync(lane_weight, offset=n) for n in range(MAX_CANDIDATES)
                ]

                # Selection, rounded to bf16, and its RMS over the token (second cluster reduction).
                mixed = []
                out_sq = cutlass.Float32(0.0)
                for e in cutlass.range_constexpr(ELTS):
                    value = cutlass.Float32(0.0)
                    for n in cutlass.range_constexpr(MAX_CANDIDATES):
                        value = weights[n] * cand[n][e] + value
                    mixed.append(value)
                mixed_words = []
                for q in cutlass.range_constexpr(4):
                    mixed_words.append(_pack_bf16x2(mixed[2 * q + 1], mixed[2 * q]))
                if cutlass.const_expr(tap_out == 1):
                    # The mixture the RMSNorm below normalizes (a DSpark capture layer's tap).
                    tap.store(
                        (mixed_words[0], mixed_words[1], mixed_words[2], mixed_words[3]),
                        idx=tap_word,
                        alignment=16,
                    )
                for q in cutlass.range_constexpr(4):
                    lo = _lo(mixed_words[q])
                    hi = _hi(mixed_words[q])
                    out_sq = lo * lo + out_sq
                    out_sq = hi * hi + out_sq
                out_sq = _warp_allsum(out_sq)
                if lane < cutlass.Int32(CLUSTER):
                    _st_async_f32(
                        _mapa_u32(
                            box_sq.subview(crank * cutlass.Int32(EPI_WARPS) + w).data_ptr(), lane
                        ),
                        out_sq,
                        _mapa_u32(mb_sq.data_ptr(), lane),
                    )
                while not _test_wait_cluster(mb_sq.data_ptr(), 0):
                    pass
                cute.arch.sync_warp()
                box_q = [
                    box_sq.load(idx=r * EPI_WARPS, vector_size=EPI_WARPS, alignment=16)
                    for r in range(CLUSTER)
                ]
                total_sq = cutlass.Float32(0.0)
                for r in cutlass.range_constexpr(CLUSTER):
                    part = cutlass.Float32(0.0)
                    for ww in cutlass.range_constexpr(EPI_WARPS):
                        part = part + cutlass.Float32(box_q[r][ww])
                    total_sq = total_sq + part
                rsigma = cute.math.rsqrt(total_sq / cutlass.Float32(H) + out_eps)
                out_words = []
                for q in cutlass.range_constexpr(4):
                    oword = cutlass.Int32(ow[q])
                    # KimiK3RMSNorm: normalize in fp32, round to bf16, then apply the bf16 weight.
                    normed_pair = _pack_bf16x2(
                        _hi(mixed_words[q]) * rsigma, _lo(mixed_words[q]) * rsigma
                    )
                    out_words.append(
                        _pack_bf16x2(_hi(normed_pair) * _hi(oword), _lo(normed_pair) * _lo(oword))
                    )
                normed.store(
                    (out_words[0], out_words[1], out_words[2], out_words[3]),
                    idx=row_word,
                    alignment=16,
                )
                if cutlass.const_expr(publish):
                    # The same 16 bytes into the slab, where the next kernel's poll takes them as ready.
                    x_slab.store(
                        (
                            _slab_word(out_words[0]),
                            _slab_word(out_words[1]),
                            _slab_word(out_words[2]),
                            _slab_word(out_words[3]),
                        ),
                        idx=slab_buf * cutlass.Int32(SLAB_WORDS) + row_word,
                        alignment=16,
                    )
        if cutlass.const_expr(x_src >= 1):
            # The grid completes only after its producer (the dependents' own transitive reads rely on that).
            prims.griddepcontrol(prims.GridDepAction.WAIT)
        if cutlass.const_expr(x_src == 2):
            # The next call's count, kept in [0, 6) so that it never wraps: its parity picks the half, its value mod 3
            # the scale buffer. Every CTA of this call has read this one (CTA 0's phase 2 needed every CTA's push).
            if (bx == cutlass.Int32(0)) & (tid == cutlass.Int32(0)):
                count_period = cutlass.Int32(2 * LAT_SCALE_BUFS)
                lat_flags.store((n_calls % count_period + cutlass.Int32(1)) % count_period, idx=0, is_volatile=True)


@cute.kernel
def k3_sandwich_kernel(
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_x: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_x2: cutlass.GridConstant[cuda.TensorMap],
    rms_src: cutlass.Array,
    ws_uc: cutlass.Array,
    ws_mc: cutlass.Array,
    ws_flags: cutlass.Array,
    prefix: cutlass.Array,
    snapshots: cutlass.Array,
    res_w: cutlass.Array,
    rms_w: cutlass.Array,
    out_w: cutlass.Array,
    updated: cutlass.Array,
    normed: cutlass.Array,
    x_slab: cutlass.Array,
    lat_flags: cutlass.Array,
    tap: cutlass.Array,
    num_tokens: cutlass.Int32,
    rank: cutlass.Int32,
    num_cand: cutlass.Int32,
    add_prefix: cutlass.Int32,
    rms_eps: cutlass.Float32,
    out_eps: cutlass.Float32,
    x_col0: cutlass.Int32,
    lat_eps: cutlass.Float32,
    slab_buf: cutlass.Int32,
    x_buf: cutlass.Int32,
    tap_stride: cutlass.Int32,
    world: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    x_tiles: cutlass.Constexpr[int],
    rms_cols: cutlass.Constexpr[int],
    publish: cutlass.Constexpr[int],
    x_src: cutlass.Constexpr[int],
    plain: cutlass.Constexpr[int],
    tap_out: cutlass.Constexpr[int],
    swiglu: cutlass.Constexpr[int],
    k_split: cutlass.Constexpr[int],
):
    """The sandwich alone: 56 CTAs, each running ``sandwich_role`` on its own shared memory (arguments as there)."""
    smem = SmemAllocator().allocate(smem_bytes(k_in, rms_cols, swiglu), byte_alignment=1024)
    sandwich_role(
        tma_desc_w, tma_desc_x, tma_desc_x2, rms_src, ws_uc, ws_mc, ws_flags, prefix, snapshots, res_w, rms_w, out_w,
        updated, normed, x_slab, lat_flags, tap, smem, num_tokens, rank, num_cand, add_prefix, rms_eps, out_eps,
        x_col0, lat_eps, slab_buf, x_buf, tap_stride, world, k_in, x_tiles, rms_cols, publish, x_src, plain, tap_out,
        swiglu, k_split, 0, None,
    )  # fmt: skip


def weight_tensor_map(w, k_in):
    """W [7168, k_in] as five TMA dimensions (64-element column chunk, row, chunk index, 1, 1) so one call per
    k-tile lands both 128-byte-swizzled halves; strides in 16-byte units."""
    return cuda.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[TMA_K_BOX, H, k_in // TMA_K_BOX, 1, 1],
        global_strides=[
            (k_in * ELEM_BYTES) // 16,
            (TMA_K_BOX * ELEM_BYTES) // 16,
            (H * k_in * ELEM_BYTES) // 16,
            (H * k_in * ELEM_BYTES) // 16,
        ],
        box_dims=[TMA_K_BOX, CTA_M, TMA_COPY_ITERS, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


def activation_tensor_map(x, cols, num_tokens):
    """x [M, cols] as (cols, M) with an 8-row box: rows past num_tokens arrive as zeros."""
    return cuda.create_tensor_map_tiled(
        global_address=x.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[cols, num_tokens],
        global_strides=[(cols * ELEM_BYTES) // 16],
        box_dims=[TMA_K_BOX, MMA_N],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


@cute.jit
def k3_sandwich_oproj(
    w: cute.Tensor,
    x: cute.Tensor,
    ws_uc: cute.Tensor,
    ws_mc: cute.Tensor,
    ws_flags: cute.Tensor,
    prefix: cute.Tensor,
    snapshots: cute.Tensor,
    res_w: cute.Tensor,
    rms_w: cute.Tensor,
    out_w: cute.Tensor,
    updated: cute.Tensor,
    normed: cute.Tensor,
    x_slab: cute.Tensor,
    src_slab: cute.Tensor,
    num_tokens: cutlass.Int32,
    rank: cutlass.Int32,
    num_cand: cutlass.Int32,
    add_prefix: cutlass.Int32,
    rms_eps: cutlass.Float32,
    out_eps: cutlass.Float32,
    slab_buf: cutlass.Int32,
    src_buf: cutlass.Int32,
    world: cutlass.Constexpr[int],
    publish: cutlass.Constexpr[int],
    x_src: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """The post-attention sandwich: phase 1 is ``core @ W_o^T`` (x_src 1: core polled from ``src_slab``)."""
    tma_desc_w = weight_tensor_map(w, K_IN)
    tma_desc_x = activation_tensor_map(x, K_IN, num_tokens)
    k3_sandwich_kernel(
        tma_desc_w, tma_desc_x, tma_desc_x, src_slab, ws_uc, ws_mc, ws_flags, prefix, snapshots, res_w, rms_w, out_w,
        updated, normed, x_slab, ws_flags, ws_flags, num_tokens, rank, num_cand, add_prefix, rms_eps, out_eps,
        cutlass.Int32(0), cutlass.Float32(0.0), slab_buf, src_buf, cutlass.Int32(WORDS_PER_ROW), world, K_IN,
        MAX_K_TILES, 0, publish, x_src, 0, 0, 0, 1,
    ).launch(
        grid=(NUM_CTAS, 1, 1), block=(THREADS, 1, 1), cluster=(CLUSTER, 1, 1), stream=stream, use_pdl=use_pdl,
    )  # fmt: skip


@cute.jit
def k3_sandwich_tail(
    w: cute.Tensor,  # [7168, 256 + 384] bf16: latent-up columns of this rank's slice (zero-padded) | shared down
    latent: cute.Tensor,  # [M, 3584] bf16, the whole reduced latent row
    latent_words: cute.Tensor,  # the same memory as int32 words, for the RMS
    act: cute.Tensor,  # [M, 384] bf16, the shared-expert activation
    ws_uc: cute.Tensor,
    ws_mc: cute.Tensor,
    ws_flags: cute.Tensor,
    prefix: cute.Tensor,
    snapshots: cute.Tensor,
    res_w: cute.Tensor,
    rms_w: cute.Tensor,
    out_w: cute.Tensor,
    updated: cute.Tensor,
    normed: cute.Tensor,
    x_slab: cute.Tensor,
    lat_flags: cute.Tensor,  # x_src 2: int32 [LAT_FLAG_WORDS] (the call count and the scale slab)
    tap: cute.Tensor,  # tap_out: int32 words of the bf16 tap rows [M, 7168], tap_stride words apart
    num_tokens: cutlass.Int32,
    rank: cutlass.Int32,
    num_cand: cutlass.Int32,
    add_prefix: cutlass.Int32,
    rms_eps: cutlass.Float32,
    out_eps: cutlass.Float32,
    lat_col0: cutlass.Int32,
    lat_eps: cutlass.Float32,
    slab_buf: cutlass.Int32,
    src_buf: cutlass.Int32,
    tap_stride: cutlass.Int32,
    world: cutlass.Constexpr[int],
    publish: cutlass.Constexpr[int],
    x_src: cutlass.Constexpr[int],
    tap_out: cutlass.Constexpr[int],  # 1: the pre-norm mixture into ``tap``; 2: updated
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """The pre-attention sandwich: phase 1 is the row-parallel MoE tail (x_src 1: ``latent_words`` is the reduced
    latent's Lamport slab, int32 [3][8][1792], polled in buffer ``src_buf``; x_src 2: ``latent_words`` is this rank's
    latent exchange buffer, int32 [2][8][world][1792], and ``latent`` gives only the shape)."""
    k_in = TAIL_LAT + TAIL_ACT
    tma_desc_w = weight_tensor_map(w, k_in)
    tma_desc_x = activation_tensor_map(latent, LATENT, num_tokens)
    tma_desc_x2 = activation_tensor_map(act, TAIL_ACT, num_tokens)
    k3_sandwich_kernel(
        tma_desc_w, tma_desc_x, tma_desc_x2, latent_words, ws_uc, ws_mc, ws_flags, prefix, snapshots, res_w, rms_w,
        out_w, updated, normed, x_slab, lat_flags, tap, num_tokens, rank, num_cand, add_prefix, rms_eps, out_eps,
        lat_col0, lat_eps, slab_buf, src_buf, tap_stride, world, k_in, TAIL_LAT // CTA_K, LATENT, publish, x_src, 0,
        tap_out, 0, 1,
    ).launch(
        grid=(NUM_CTAS, 1, 1), block=(THREADS, 1, 1), cluster=(CLUSTER, 1, 1), stream=stream, use_pdl=use_pdl,
    )  # fmt: skip


@cute.jit
def k3_sandwich_plain(
    w: cute.Tensor,  # [7168, k_in] bf16: this rank's slice of the row-parallel projection
    x: cute.Tensor,  # [M, k_in] bf16
    ws_uc: cute.Tensor,
    ws_mc: cute.Tensor,
    ws_flags: cute.Tensor,
    residual: cute.Tensor,  # int32 words of bf16 [M, 7168]
    norm_w: cute.Tensor,  # int32 words of bf16 [7168]
    updated: cute.Tensor,
    normed: cute.Tensor,
    num_tokens: cutlass.Int32,
    rank: cutlass.Int32,
    eps: cutlass.Float32,
    world: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    order: cutlass.Constexpr[
        int
    ],  # 1: the MNNVL one-shot's arithmetic, 2: the IPC one-shot's (world <= 8)
    swiglu: cutlass.Constexpr[
        int
    ],  # 1: x is a gate_up output [M, 2 k_in] (gate columns first), B = silu_and_mul(x)
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    """The plain sandwich: ``x @ w^T``, the TP all-reduce, updated = sum + residual and normed = RMSNorm(updated),
    as a row-parallel GEMV followed by the one-shot all-reduce's RESIDUAL_RMS_NORM. swiglu: ``silu_and_mul(x) @ w^T``
    with k3_ctm_gemv_swiglu split 2's arithmetic (the drafter MLP's down projection)."""
    tma_desc_w = weight_tensor_map(w, k_in)
    tma_desc_x = activation_tensor_map(x, 2 * k_in if swiglu else k_in, num_tokens)
    k3_sandwich_kernel(
        tma_desc_w, tma_desc_x, tma_desc_x, ws_flags, ws_uc, ws_mc, ws_flags, residual, residual, norm_w, norm_w,
        norm_w, updated, normed, ws_flags, ws_flags, ws_flags, num_tokens, rank, cutlass.Int32(1), cutlass.Int32(1),
        eps, eps, cutlass.Int32(k_in if swiglu else 0), cutlass.Float32(0.0), cutlass.Int32(0), cutlass.Int32(0),
        cutlass.Int32(WORDS_PER_ROW), world, k_in, k_in // CTA_K, 0, 0, 0, order, 0, swiglu, 2 if swiglu else 1,
    ).launch(
        grid=(NUM_CTAS, 1, 1), block=(THREADS, 1, 1), cluster=(CLUSTER, 1, 1), stream=stream, use_pdl=use_pdl,
    )  # fmt: skip

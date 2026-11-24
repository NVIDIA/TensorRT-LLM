# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# ruff: noqa: E501
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Adapted for vLLM (Kimi K3 specialized decode) from tokenspeed-mla 2.0.2
# (https://github.com/lightseekorg/tokenspeed), tokenspeed_mla/mla_decode_fp8.py:
# BlackwellMultiHeadLatentAttentionForwardFP8.split_kv_kernel is copied below and
# changed where marked [tsport]; every other method is inherited from the installed
# tokenspeed-mla package. The original license notice is reproduced above.
"""tokenspeed's generic FP8 MLA decode kernel with the custom core's polled interface,
for the Rubin TP16 DCP16 B8 long-context step (mla.prepared.SWITCH, body 1).

Changes to tokenspeed's kernel (split_kv_kernel; launched without its reducer):

- q: MlaQuery does not gather (``gather=False``); after its PDL wait (MlaQuery's
  publish and KV insert are complete) this kernel polls its CTA's q rows of the DCP
  mailbox until no 32 B chunk holds the sentinel (the softmax and correction warps, 256
  threads, one round of 32 B loads), then the Q TMA follows (named barrier, async-proxy
  fence). The K and V TMA loads do not wait for q.
- x: the idle warp re-arms its share of the polled in-projection x rows after the PDL
  wait (MlaQuery, their only reader, is complete; the next writer is the next MLA
  layer's in-projection, after its own wait), so MlaQuery polls x (``poll_x``).
- The block_split_kvs argument (unused: ``get_k_tile_count`` ignores it) carries a
  table of int32 words: [q mailbox address lo/hi, x address lo/hi, x pitch in bytes].
- ``balanced`` (mla.dcp.FORK_BALANCE): the 2-CTA pair splits the heads evenly
  (``ts_shift``); the split partials and LSEs are bitwise the same.
- ``repoll`` (mla.dcp.FORK_REPOLL): each poller re-polls all of its late q chunks
  together (one round trip per pass) instead of spinning on each chunk in turn.
- ``valid`` (mla.dcp.FORK_VALID): no q pollers. The V-load warp copies the CTA's q rows
  with validity-reporting bulk copies (mla.qvalid) into the Q operand's shared memory
  until no 16 B chunk is armed, then issues the Q TMA as before; the softmax and
  correction warps go straight to the TMEM barrier. Needs a q writer that keeps
  mla.qvalid's contract (``MlaQuery(valid=True)``: one 16 B store per chunk, first word
  never 0x80000000) and the (0x80000000, Q_SENT x 3) arm pattern (mla.dcp).
"""

from types import SimpleNamespace  # noqa: F401  (used by the copied kernel)
from typing import Optional  # noqa: F401

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync  # noqa: F401
import cutlass.memory  # noqa: F401
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait  # noqa: F401
from tokenspeed_mla.mla_decode_fp8 import (
    BlackwellMultiHeadLatentAttentionForwardFP8,
    _get_max_tmem_alloc_cols,  # noqa: F401
)
from tokenspeed_mla.mla_helpers import (  # noqa: F401
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler,
)

from ..common import tensormap
from ..common.primitives import (
    asm,
    bits_float,
    float_bits,
    load_words,
    no_neg_zero,
    published,
    store_relaxed,
    store_words,
    zeros,
)
from .dcp import EMPTY_LSE, long_per, poll_b32x8
from .qvalid import (
    copy_valid,
    expect_valid,
    init_valid,
    signal_word,
    spin_ns,
    tma_valid3,
    wait_valid,
)

Q_SENT = 0xFFFFFFFF  # the q mailbox sentinel (mla.dcp.Q_SENT)
POLL_Q = True  # False only in the tests' positive control (q read unpolled)
X_SENT = 0xFFFFFFFF  # two BF16 0xFFFF (mla.query.X_SENT)
# ``wlist``: the address table's words holding the work list's address (mla.wlist)
WLIST_WORD = 6


@asm()
def fence_proxy_async_global():
    return "fence.proxy.async.global;", "~{memory}", ()


@asm()
def invalidate_barrier(barrier):
    return "mbarrier.inval.shared::cta.b64 [$0];", "r,~{memory}", (barrier,)


def polled_core(
    batch,
    x_columns=1536,
    balanced=False,
    repoll=False,
    cluster=0,
    valid=False,
    valid_delay=0,
    persistent=False,
    push=None,
    long=None,
    wlist=False,
):
    """The kernel class for ``batch`` requests (x rows) of x_columns BF16 x columns;
    ``balanced``: the 2-CTA pair splits the heads evenly (``ts_shift``); ``repoll``:
    the q pollers re-poll their late chunks together (``ts_poll_chunks``); ``valid``:
    the V-load warp polls q with validity-reporting copies (``ts_valid_q``), its first
    round ``valid_delay`` ns after the PDL wait. ``valid`` 2: the Q load itself is the
    validity copy (``ts_valid_tma``). ``persistent`` (``valid`` 2 only): tokenspeed's
    persistent scheduler; each work tile's Q load is its own validity copy, its
    barrier in the shared storage's padding (the epilogue exchange it uses otherwise is
    busy with the previous work tile's epilogue). ``push`` (split 1, packed verify rows,
    ``persistent``): (DCP, rank, heads per rank, q, early) of the core push
    (``ts_push_tail``; ``early``: the most first blocks of two-block CTAs one idle CTA
    pushes under the second wave, the rest going with their own CTA's last block):
    the core stores every row and LSE into its owner's DcpBox itself, as DcpValue's
    pushers would (the address table of ``core_q.push_table``), and DcpValue only
    merges. ``long`` (mla.dcp.LONG_SPLIT): (short split, long tiles) -- a request of at
    most ``long tiles`` 128-token K tiles keeps the short split's partition, the launched
    splits past it left empty (``get_k_tile_count``). ``wlist`` (persistent, no push):
    the work list -- persistent work block j runs word j of the list whose address is in
    the address table's words WLIST_WORD.. (``ts_item``, ``mla.wlist``), so a per-step schedule orders and places the items
    (e.g. longest first); the items and their partitions are unchanged."""
    assert not wlist or (persistent and not push), "the work list: persistent, no push"
    assert not persistent or valid == 2, "a persistent polled core needs FORK_VALID 2"
    assert not push or persistent, "the core push uses the persistent schedule"
    # ``valid`` 2: the q mailbox's TMA descriptors are grid-constant kernel parameters
    # (no tensormap proxy fence: ~0.9 us each on Rubin when global)
    assert valid != 2 or tensormap.AVAILABLE, "FORK_VALID 2 needs grid-constant maps"
    MAP = tensormap.GridMap if valid == 2 else cutlass.Constexpr

    class TsCore(BlackwellMultiHeadLatentAttentionForwardFP8):
        ts_batch, ts_x_columns, ts_balanced = batch, x_columns, balanced
        ts_repoll, ts_valid, ts_valid_delay = repoll, valid, valid_delay
        # (FORK_CLUSTER) launch clusters of the 2-CTA pair x ``cluster`` KV splits
        ts_cluster = cluster
        ts_persistent = persistent
        ts_push = push
        ts_long = long
        ts_wlist = wlist
        ts_pollers = 256  # the softmax and correction warps (tidx 0..255)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.is_persistent == self.ts_persistent and self.use_2cta_instrs
            if cluster:  # the pair's CTAs (x) x the request's splits (z)
                self.cluster_shape_mnk = (2, 1, cluster)
            if push:  # the core push's barriers (ids 9, 10: unused above)
                self.ts_publish_bar = pipeline.NamedBarrier(
                    barrier_id=9, num_threads=self.num_compute_warps * 32
                )
                self.ts_push_bar = pipeline.NamedBarrier(
                    barrier_id=10, num_threads=self.num_compute_warps * 32 * 2
                )
            self.ts_q_bar = pipeline.NamedBarrier(
                barrier_id=7, num_threads=self.ts_pollers + self.threads_per_warp
            )

        @property
        def ts_q_tx(self):
            """Bytes the Q full barrier expects: the Q TMAs', or (``valid`` 2) the two
            CTAs' 4 B signals."""
            return 8 if self.ts_valid == 2 else self.tma_copy_q_bytes

        @cute.jit
        def get_k_tile_count(self, split_kv, cache_seqs, block_split_kvs, blk_coord):
            K = cache_seqs[blk_coord[2]]
            k_tile_total = cute.ceil_div(K, self.mma_qk_tiler[1])
            k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
            if cutlass.const_expr(self.ts_long is not None):  # LONG_SPLIT
                k_tile_per_cta = long_per(k_tile_total, split_kv, self.ts_long)
            k_index = blk_coord[3] * k_tile_per_cta
            count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
            return k_index, count, split_kv

        if balanced:

            @property
            def ts_shift(self):
                """Rows CTA 1 of the pair moves down so both CTAs store half of
                the heads. 2-CTA M128 tiles put rows 0-63 in CTA 0 and 64-127 in CTA
                1, so H = 96 heads split 64 / 32, and CTA 0 stores twice the FP32
                split partials: the epilogue is bound by the SM's store egress
                (whatever the store pattern), 2.05 vs 1.02 us. With the shift
                CTA 1 loads heads 48-111 (96 and up: zero, out of bounds for the
                TMA) and stores 48-95; CTA 0 still loads and computes heads 0-63
                (all polled) but stores only 0-47. Per-row math is unchanged (the
                skip-correction vote only decides whether the rescale loop runs;
                every row multiplies by its own factor)."""
                return self.mma_qk_tiler[0] // 2 - (self.num_heads + 1) // 2

            @cute.jit
            def ts_shifted(self, tensor, cta):
                """``tensor`` with its head (mode 0) coordinates moved down by
                ts_shift on CTA 1 of the pair."""
                offset = (-cta * self.ts_shift,) + (0,) * (cute.rank(tensor) - 1)
                return cute.domain_offset(offset, tensor)

            @cute.jit
            def load_tma_q(self, common_params, q_params, load_q_producer_state):
                """tokenspeed's Q load, CTA 1's heads shifted (ts_shift)."""
                cta = common_params.blk_coord[0] % 2
                shifted = SimpleNamespace(**vars(q_params))
                shifted.mQL = self.ts_shifted(q_params.mQL, cta)
                shifted.mQR = self.ts_shifted(q_params.mQR, cta)
                return BlackwellMultiHeadLatentAttentionForwardFP8.load_tma_q(
                    self, common_params, shifted, load_q_producer_state
                )

            @cute.jit
            def epilogue(
                self, common_params, epilogue_params, mma_o_state, row_sum, row_max
            ):
                """tokenspeed's epilogue (2-CTA, split partials) with CTA 1's
                rows shifted (ts_shift) and each CTA storing only its half of the
                heads; the same values, LSEs and pipeline releases."""
                tidx = common_params.tidx % (self.num_compute_warps * 32)
                cta = common_params.blk_coord[0] % 2
                half: cutlass.Constexpr = self.mma_qk_tiler[0] // 2
                live: cutlass.Constexpr = half - self.ts_shift  # rows stored per CTA
                # exchange row_sum between warps (0, 2) and (1, 3)
                common_params.smem_exchange[tidx] = row_sum
                if (tidx // 32) % 2 == 0:
                    self.epilogue_exchange_sync_bar_pair02.arrive_and_wait()
                else:
                    self.epilogue_exchange_sync_bar_pair13.arrive_and_wait()
                row_sum = row_sum + common_params.smem_exchange[(tidx + 64) % 128]
                # partials skip output_scale; DcpValue's fold applies it
                epi_scale = cute.arch.rcp_approx(row_sum)
                shifted = SimpleNamespace(**vars(common_params))
                shifted.mAccO = self.ts_shifted(common_params.mAccO, cta)
                for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
                    common_params.mma_o_pipeline.consumer_wait(mma_o_state)
                    copy, _, t_acc, t_go, t_co, r_acc = self._tmem_load_partition(
                        shifted, common_params.tiled_mma_pv, iter_n
                    )
                    cute.copy(copy, t_acc, r_acc)
                    src = cute.make_fragment_like(t_go, self.partial_dtype)
                    # t_co: the unshifted coordinates, 64 cta + row (row < 64)
                    row = t_co[0][0] - cta * half
                    keep = cute.elem_less(row, live) & cute.elem_less(
                        row + cta * live, common_params.H
                    )
                    for d in cutlass.range_constexpr(r_acc.shape[2]):
                        vec = cute.size(cute.select(r_acc.shape, mode=[0]))
                        for i in cutlass.range(vec, vectorize=True, unroll_full=True):
                            r_acc[i, 0, d] = r_acc[i, 0, d] * epi_scale
                        src[None, None, d].store(
                            r_acc[None, None, d].load().to(self.partial_dtype)
                        )
                        if keep:
                            cute.autovec_copy(
                                src[None, None, d],
                                t_go[None, None, d],
                                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                            )
                    lse = (
                        cute.math.log2(row_sum, fastmath=True)
                        + epilogue_params.softmax_scale_log2 * row_max
                    )
                    split = self.ts_shifted(
                        epilogue_params.mAccLSE[
                            None, common_params.blk_coord[3], None, None
                        ],
                        cta,
                    )
                    coord = (
                        common_params.blk_coord[0],
                        common_params.blk_coord[1],
                        common_params.blk_coord[2],
                    )
                    g_lse = cute.local_tile(split, (half, 1, 1), coord, (1, 1, 1))
                    lrow = tidx % 64  # warps (0, 1) and (2, 3) hold rows tidx % 64
                    mine = (lrow < live) & (lrow + cta * live < common_params.H)
                    if mine & (tidx < 64):
                        g_lse[tidx] = lse
                    cute.arch.fence_view_async_tmem_load()
                    common_params.mma_o_pipeline.consumer_release(mma_o_state)
                    mma_o_state.advance()
                return mma_o_state

        @cute.jit
        def ts_push_row(self, table, blk_coord, row):
            """(row base, LSE slot) addresses in the owner's DcpBox of
            packed row ``row`` of tile blk_coord[1] of request blk_coord[2]: query
            row t, global head g (packed t x DCP x H + g), batch row request x q + t,
            owner g // H (``core_q.push_table``)."""
            D, rank, H, q, _ = self.ts_push
            packed = blk_coord[1] * self.mma_qk_tiler[0] + row
            t, g = packed // (D * H), packed % (D * H)
            owner, h = g // H, g % H
            slot = (blk_coord[2] * q + t) * H + h
            rows = self.ts_address(table, 2 + 4 * owner) + Int64(slot) * 1024
            lse = self.ts_address(table, 4 + 4 * owner) + Int64(slot * D + rank) * 4
            return rows, lse

        @cute.jit
        def ts_flag(self, table, block):
            """the ready flag of persistent work block ``block`` (table words
            2 + 4 DCP ..: ``core_q.push_table``'s flags, zeroed, reset by their reader)."""
            D = self.ts_push[0]
            base = self.ts_address(table, 2 + 4 * D) + Int64(block) * 4
            return cute.make_ptr(Uint32, base, cute.AddressSpace.gmem, assumed_align=4)

        @cute.jit
        def ts_publish(self, table, block, tidx):
            """the correction warps after work block ``block``'s epilogue
            (its rows and LSEs stored, or none: no KV): when this CTA has another
            block, flag it for the idle CTA that pushes it (``ts_push_tail``)."""
            P = cute.arch.grid_dim()[0]
            taken = self.ts_push[4] * (2 * P - self.ts_blocks())  # idle CTAs x early
            if (block + P < self.ts_blocks()) & (block < taken):
                self.ts_publish_bar.arrive_and_wait()
                if tidx % (self.num_compute_warps * 32) == 0:
                    cute.arch.fence_acq_rel_gpu()
                    store_words(self.ts_flag(table, block), (Uint32(1),), scope="gpu")

        def ts_blocks(self):
            """persistent work blocks: CTA pair halves x M128 tiles x
            requests (split 1)."""
            D, _, H, q, _ = self.ts_push
            tiles = -(-self.total_q_rows // self.mma_qk_tiler[0])
            return 2 * tiles * (self.ts_batch // q)

        @cute.jit
        def ts_push_block(self, table, mO, mLSE, cache_seqs, block, t):
            """pushers t < 256 (the compute and correction warps): work block
            ``block``'s 64 rows (1 KB each) and LSEs into their owners' DcpBox, as
            DcpValue pushes them (zeros and EMPTY_LSE for a request without local KV,
            no negative zero, EMPTY_LSE for NaN / +-inf). Thread t: 16 B chunk t % 64
            of rows t // 64 + 4 m, all 16 loads issued before the stores."""
            D, rank, H, q, _ = self.ts_push
            half: cutlass.Constexpr = self.mma_qk_tiler[0] // 2
            tiles: cutlass.Constexpr = -(-self.total_q_rows // self.mma_qk_tiler[0])
            rows_per: cutlass.Constexpr = q * D * H  # packed rows per request
            pair, rest = block % 2, block // 2
            tile, request = rest % tiles, rest // tiles
            blk = (pair, tile, request)
            live = cache_seqs[request] > 0
            dead = cache_seqs[request] <= 0
            k = t % 64
            loaded = []
            places = []
            for m in cutlass.range_constexpr(half // 4):
                row = pair * half + t // 64 + 4 * m  # in the M128 tile
                packed = tile * self.mma_qk_tiler[0] + row
                ok = packed < rows_per
                src = mO.iterator + (Int64(request) * rows_per + packed) * 512 + 8 * k
                words = (Uint32(0),) * 4
                if ok & live:
                    words = load_words(src, count=4, scope="gpu")
                loaded.append(words)
                places.append((row, ok))
            for (row, ok), words in zip(places, loaded):
                if ok:
                    rows, _ = self.ts_push_row(table, blk, row)
                    out = tuple(no_neg_zero(w) for w in words)
                    store_relaxed(rows + Int64(8 * k) * 2, out)
            if t < half:
                lrow = pair * half + t
                lpacked = tile * self.mma_qk_tiler[0] + lrow
                if lpacked < rows_per:
                    _, lse_row = self.ts_push_row(table, blk, lrow)
                    lse_src = mLSE.iterator + Int64(request) * rows_per + lpacked
                    value = bits_float(load_words(lse_src, count=1, scope="gpu")[0])
                    bad = (value != value) | (value <= Float32(EMPTY_LSE))
                    if dead | bad | (value >= Float32(-EMPTY_LSE)):
                        value = Float32(EMPTY_LSE)
                    store_relaxed(lse_row, (float_bits(value),))

        @cute.jit
        def ts_push_tail(self, table, mO, mLSE, cache_seqs, t):
            """the core push, after the compute (t < 128) and correction
            (t >= 128) warps' work loops. CTA c of the persistent grid (P CTAs, N
            blocks, E = N - P CTAs with two blocks) pushes its last block, then, when
            it had one block (c >= E: idle since its first wave ended), the first
            blocks of the two-block CTAs E' = c - E, c - E + (P - E), .. once their
            ready flags are up (``ts_publish``): those rows cross the fabric under the
            second wave instead of after it."""
            P = cute.arch.grid_dim()[0]
            c = cute.arch.block_idx()[0]
            N = self.ts_blocks()
            E = N - P
            taken = self.ts_push[4] * (P - E)  # first blocks [0, taken): idle CTAs
            self.ts_push_bar.arrive_and_wait()  # the last epilogue's rows are stored
            last = c
            if c < E:
                last = c + P
                if c >= taken:  # (its first block: not taken by an idle CTA)
                    self.ts_push_block(table, mO, mLSE, cache_seqs, c, t)
            self.ts_push_block(table, mO, mLSE, cache_seqs, last, t)
            if c >= E:
                for block in cutlass.range(c - E, min(E, taken), P - E, unroll=1):
                    if t == 0:
                        flag = self.ts_flag(table, block)
                        word = load_words(flag, count=1, scope="gpu")[0]
                        while word != Uint32(1):
                            word = load_words(flag, count=1, scope="gpu")[0]
                        cute.arch.fence_acq_rel_gpu()
                        store_words(flag, (Uint32(0),), scope="gpu")
                    self.ts_push_bar.arrive_and_wait()
                    self.ts_push_block(table, mO, mLSE, cache_seqs, block, t)

        @cute.jit
        def ts_item(self, tile_idx, tile_sched, table):
            """``wlist``: the work list's item for the current persistent work block
            (block = linear index // 2: the pair's CTAs are the fastest mode) --
            word q_tile | request << 8 | split << 16 (WLIST_EMPTY: no K tiles);
            otherwise the scheduler's own tile."""
            if cutlass.const_expr(not self.ts_wlist):
                return tile_idx
            block = tile_sched.current_work_linear_idx // 2
            address = self.ts_address(table, WLIST_WORD) + Int64(block) * 4
            ptr = cute.make_ptr(Int32, address, cute.AddressSpace.gmem, assumed_align=4)
            word = cute.make_tensor(ptr, cute.make_layout(1))[0]
            return (tile_idx[0], word & 0xFF, (word >> 8) & 0xFF, word >> 16)

        @cute.jit
        def ts_address(self, table, word):
            lo = Int64(Uint32(table[word]))
            return lo | (Int64(table[word + 1]) << 32)

        @cute.jit
        def ts_poll_q(self, table, split_kv, cache_seqs, tile_sched_params, poller):
            """Poll this CTA's q rows (the 2-CTA pair's half of the request's heads)
            until every 32 B chunk has landed, then arrive on ts_q_bar."""
            H: cutlass.Constexpr = self.num_heads
            half_rows: cutlass.Constexpr = self.mma_qk_tiler[0] // 2
            per: cutlass.Constexpr = -(-half_rows * 18 // self.ts_pollers)
            tile_sched = create_mla_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            if work_tile.is_valid_tile:
                blk_coord = self.ts_item(work_tile.tile_idx, tile_sched, table)
                _, count, _ = self.get_k_tile_count(
                    split_kv, cache_seqs, table, blk_coord
                )
                if count > 0:
                    first = (blk_coord[0] % 2) * half_rows
                    if cutlass.const_expr(self.ts_balanced):  # CTA 1: H - H/2 ..
                        first = (blk_coord[0] % 2) * (half_rows - self.ts_shift)
                    rows = min(half_rows, H - first)
                    if rows > 0 and cutlass.const_expr(POLL_Q):  # (fewer heads than
                        # an M tile: the 2nd CTA of the pair may have none)
                        last = rows * 18 - 1  # 32 B chunks of 576 B rows
                        q = self.ts_address(table, 0)
                        base = q + Int64(blk_coord[2] * H + first) * 576
                        for i0 in cutlass.range_constexpr(0, per, 8):
                            n: cutlass.Constexpr = min(8, per - i0)
                            ptrs = [
                                base
                                + Int64(min(poller + (i0 + i) * self.ts_pollers, last))
                                * 32
                                for i in range(n)
                            ]
                            if cutlass.const_expr(self.ts_repoll):
                                self.ts_poll_chunks(ptrs)
                            else:
                                loads = [poll_b32x8(p) for p in ptrs]
                                for i in cutlass.range_constexpr(n):
                                    words = loads[i]
                                    while not published(words, sentinel=Q_SENT):
                                        words = poll_b32x8(ptrs[i])
                    self.ts_q_bar.arrive()

        @cute.jit
        def ts_valid_q(self, table, blk_coord, q_latent, q_rope, spare):
            """[valid] One thread (the V-load warp's lane 0): rounds of validity-
            reporting bulk copies of this CTA's q rows into the Q operand's shared
            memory (``q_latent``, then ``q_rope``: scratch, the Q TMA rewrites them)
            until no copied 16 B chunk is armed; every copy of a round is re-issued.
            The layout::v1 barrier sits in ``spare`` (the epilogue's exchange buffer,
            first used after the first S, so after this) and is invalidated after.
            mla.qvalid: the report checks the first word of each 16 B chunk only, so
            this needs MlaQuery(valid=True)'s one-store-per-chunk writer."""
            H: cutlass.Constexpr = self.num_heads
            half_rows: cutlass.Constexpr = self.mma_qk_tiler[0] // 2
            latent: cutlass.Constexpr = half_rows * 512  # the CTA's Q latent bytes
            first = (blk_coord[0] % 2) * half_rows
            if cutlass.const_expr(self.ts_balanced):  # CTA 1: H - H/2 ..
                first = (blk_coord[0] % 2) * (half_rows - self.ts_shift)
            rows = min(half_rows, H - first)
            if rows > 0:
                base = self.ts_address(table, 0) + Int64(blk_coord[2] * H + first) * 576
                nbytes = rows * 576
                head = min(nbytes, latent)
                barrier = (Int32(spare.toint()) + 7) & Int32(-8)
                init_valid(barrier)
                cute.arch.mbarrier_init_fence()
                if cutlass.const_expr(self.ts_valid_delay > 0):  # let the rows land
                    spin_ns(self.ts_valid_delay)
                phase = Int32(0)
                armed = Int32(1)
                while armed != 0:
                    expect_valid(barrier, nbytes)
                    copy_valid(q_latent, base, head, barrier)
                    if nbytes > head:
                        copy_valid(q_rope, base + Int64(head), nbytes - head, barrier)
                    armed = wait_valid(barrier, phase)
                    phase = phase ^ 1
                invalidate_barrier(barrier)

        @cute.jit
        def ts_q_first(self, blk_coord):
            """[valid 2] The first q mailbox row (of its request) of this CTA's half of
            the Q tile: head 0, or CTA 1's first head."""
            half_rows: cutlass.Constexpr = self.mma_qk_tiler[0] // 2
            first = (blk_coord[0] % 2) * half_rows
            if cutlass.const_expr(self.ts_balanced):  # CTA 1: H - H/2 ..
                first = (blk_coord[0] % 2) * (half_rows - self.ts_shift)
            return first

        @cute.jit
        def ts_valid_tma(self, blk_coord, q_latent, q_rope, spare, full, latent, rope):
            """[valid 2] One thread (the V-load warp's lane 0): this CTA's half of the
            Q tile (64 head rows from the CTA's first head; balanced: CTA 1 from H/2;
            heads past H zero-filled) as CTA-local 3-D tensor TMAs (``latent`` / ``rope``
            descriptors) into the Q operand's shared memory on a layout::v1 barrier in
            ``spare``, re-issued while the report says a chunk was armed (POLL_Q False:
            one round, unchecked: the tests' control). Then one 4 B st.async completes
            on the pair leader's Q ``full`` barrier (it expects 2 x 4 bytes)."""
            half_rows: cutlass.Constexpr = self.mma_qk_tiler[0] // 2
            first = self.ts_q_first(blk_coord)
            barrier = (Int32(spare.toint()) + 7) & Int32(-8)
            init_valid(barrier)
            cute.arch.mbarrier_init_fence()
            if cutlass.const_expr(self.ts_valid_delay > 0):  # let the rows land
                spin_ns(self.ts_valid_delay)
            phase = Int32(0)
            armed = Int32(1)
            while armed != 0:
                expect_valid(barrier, half_rows * 576)
                for kb in cutlass.range_constexpr(4):
                    dst = q_latent + kb * half_rows * 128
                    tma_valid3(dst, latent, 128 * kb, first, blk_coord[2], barrier)
                tma_valid3(q_rope, rope, 512, first, blk_coord[2], barrier)
                armed = wait_valid(barrier, phase)
                phase = phase ^ 1
                if cutlass.const_expr(not POLL_Q):
                    armed = Int32(0)
            invalidate_barrier(barrier)
            rank = cute.arch.block_idx_in_cluster()
            signal_word(barrier + 16, full, rank - rank % 2)

        @cute.jit
        def ts_poll_chunks(self, ptrs):
            """``repoll``: the 32 B q chunks at ``ptrs`` polled until no word is the
            sentinel: every load in flight, then the late ones re-polled together, one
            round trip per pass. (Spinning on each chunk in turn costs a round trip
            per chunk that was still late at the first load, after the first lands.)"""
            n: cutlass.Constexpr = len(ptrs)
            raw = zeros((n, 8), Uint32)
            for i in cutlass.range_constexpr(n):
                loaded = poll_b32x8(ptrs[i])
                for e in cutlass.range_constexpr(8):
                    raw[i, e] = loaded[e]
            pending = Int32(0)
            for i in cutlass.range_constexpr(n):
                if not published((raw[i, e] for e in range(8)), sentinel=Q_SENT):
                    pending |= 1 << i
            while pending != 0:
                for i in cutlass.range_constexpr(n):
                    if (pending & (1 << i)) != 0:
                        loaded = poll_b32x8(ptrs[i])
                        for e in cutlass.range_constexpr(8):
                            raw[i, e] = loaded[e]
                for i in cutlass.range_constexpr(n):
                    if (pending & (1 << i)) != 0:  # noqa: SIM102
                        if published((raw[i, e] for e in range(8)), sentinel=Q_SENT):
                            pending &= ~(1 << i)

        @cute.jit
        def ts_arm_x(self, table, lane):
            """This CTA's share of x's 16 B chunks <- the sentinel (see the module doc),
            strided over the warp's lanes: small grids (few KV splits at a short
            max_model_len) hold more than 32 chunks per CTA."""
            chunks: cutlass.Constexpr = self.ts_batch * self.ts_x_columns // 8
            bx, by, bz = cute.arch.block_idx()
            gx, gy, gz = cute.arch.grid_dim()
            ctas = gx * gy * gz
            cta = bx + gx * (by + gy * bz)
            per = (chunks + ctas - 1) // ctas
            for j in cutlass.range(lane, per, 32, unroll=1):
                i = cta * per + j
                if i < chunks:
                    row, col = (
                        i // (self.ts_x_columns // 8),
                        i % (self.ts_x_columns // 8),
                    )
                    x = self.ts_address(table, 2) + Int64(row) * Int64(table[4])
                    ptr = cute.make_ptr(
                        Uint32,
                        x + Int64(col) * 16,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    store_words(ptr, (Uint32(X_SENT),) * 4)

        def split_kv_kernel(self, *args):
            """Host side (tokenspeed's ``__call__`` launches what this returns): the
            kernel, plus (``valid`` 2) the q mailbox's 3-D TMA descriptors (576 columns,
            rows, requests: [requests, q, H, 576] has q x H rows per request; Q latent
            boxes 128 B x 64 rows, 128 B swizzle; the rope box 64 B x 64 rows, 64 B
            swizzle)."""
            if self.ts_valid != 2:
                return self.ts_split_kv(*args, None, None)
            q = self.ts_query
            rows, requests = q.shape[1] * q.shape[2], q.shape[0]
            address = q.iterator.toint()
            u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
            l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE
            sw = cuda.CUtensorMapSwizzle
            dims, strides = (576, rows, requests), (576, rows * 576)
            latent = tensormap.encode_map(
                address,
                u8,
                dims,
                strides,
                (128, 64, 1),
                sw.CU_TENSOR_MAP_SWIZZLE_128B,
                l2,
            )
            rope = tensormap.encode_map(
                address,
                u8,
                dims,
                strides,
                (64, 64, 1),
                sw.CU_TENSOR_MAP_SWIZZLE_64B,
                l2,
            )
            return self.ts_split_kv(*args, latent, rope)

        if valid == 2:

            @cute.jit
            def __call__(
                self,
                q_latent: cute.Tensor,
                q_rope: cute.Tensor,
                c_latent: cute.Tensor,
                c_rope: cute.Tensor,
                page_table: cute.Tensor,
                o: cute.Tensor,
                lse: cute.Tensor | None,
                workspace: cute.Tensor,
                split_kv: cutlass.Int32,
                cache_seqs: cute.Tensor | None,
                causal_seqs: cute.Tensor | None,
                block_split_kvs: cute.Tensor | None,
                softmax_scale: cutlass.Float32,
                output_scale: cutlass.Float32,
                stream: cuda.CUstream,
                use_pdl: cutlass.Constexpr = True,
                local_visible_lens: cute.Tensor | None = None,
            ):
                """[valid 2] tokenspeed's ``__call__``, keeping the q tensor for the
                descriptors ``split_kv_kernel`` encodes."""
                self.ts_query = q_latent
                BlackwellMultiHeadLatentAttentionForwardFP8.__call__(
                    self,
                    q_latent,
                    q_rope,
                    c_latent,
                    c_rope,
                    page_table,
                    o,
                    lse,
                    workspace,
                    split_kv,
                    cache_seqs,
                    causal_seqs,
                    block_split_kvs,
                    softmax_scale,
                    output_scale,
                    stream,
                    use_pdl,
                    local_visible_lens,
                )

        @cute.kernel
        def ts_split_kv(
            self,
            tiled_mma_qk: cute.TiledMma,
            tiled_mma_pv: cute.TiledMma,
            tiled_mma_ws_epi: cute.TiledMma,
            tiled_mma_ws_splitn_epi: cute.TiledMma,
            tma_atom_q_latent: cute.CopyAtom | None,
            mQL: cute.Tensor,
            tma_atom_q_rope: cute.CopyAtom | None,
            mQR: cute.Tensor,
            tma_atom_c_latent: cute.CopyAtom | None,
            mCL: cute.Tensor,
            tma_atom_c_rope: cute.CopyAtom | None,
            mKR: cute.Tensor,
            tma_atom_c_latent_transpose: cute.CopyAtom | None,
            mCLT: cute.Tensor,
            mPT: cute.Tensor,
            mO: cute.Tensor | None,
            mLSE: cute.Tensor | None,
            mAccO: cute.Tensor | None,
            mAccLSE: cute.Tensor | None,
            split_kv: cutlass.Int32,
            cache_seqs: cute.Tensor,
            causal_seqs: cute.Tensor,  # per-request global causal bound (DCP)
            local_visible_lens: cute.Tensor | None,
            block_split_kvs: cute.Tensor,
            softmax_scale_log2: cutlass.Float32,
            output_scale: cutlass.Float32,
            q_latent_smem_layout_staged: cute.ComposedLayout,
            q_rope_smem_layout_staged: cute.ComposedLayout,
            kc_latent_smem_layout_staged: cute.ComposedLayout,
            kc_rope_smem_layout_staged: cute.ComposedLayout,
            p_smem_layout_staged: cute.ComposedLayout,
            vc_smem_layout_staged: cute.ComposedLayout,
            kc_latent_smem_layout_for_tma: cute.ComposedLayout | None,
            kc_rope_smem_layout_for_tma: cute.ComposedLayout | None,
            vc_smem_layout_for_tma: cute.ComposedLayout | None,
            cta_layout_vmnk: cute.Layout,
            tile_sched_params: MLAStaticTileSchedulerParams,
            SharedStorage: cutlass.Constexpr,
            ts_latent_map: MAP,
            ts_rope_map: MAP,
        ):
            """The device split_kv kernel implementation of the Multi-Head Latent Attention.

            This kernel coordinates multiple specialized warps to perform different phases of the MLA computation:
            1. Load warp: Loads Q/C latent/rope data from global memory to shared memory using TMA
            2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
            3. Compute warps: Compute softmax and do rescaling on accumulators, and store the intermediate/final results
            to global memory

            The kernel produces either intermediate or final results of the MLA computation based on the split_kv parameter.
            When split_kv is 1, the kernel generates the final results directly. Otherwise, it produces intermediate results
            that will later be combined by a reduction kernel.

            The kernel implements a complex pipeline with overlapping computation and memory operations,
            using tensor memory access (TMA) for efficient data loading, warp specialization for different
            computation phases.

            :param tiled_mma_qk: Tiled MMA for Q*K^T
            :type tiled_mma_qk: cute.TiledMma
            :param tiled_mma_pv: Tiled MMA for P*V
            :type tiled_mma_pv: cute.TiledMma
            :param tma_atom_q_latent: TMA copy atom for query latent tensor
            :type tma_atom_q_latent: cute.CopyAtom
            :param mQL: query latent tensor
            :type mQL: cute.Tensor
            :param tma_atom_q_rope: TMA copy atom for query rope tensor
            :type tma_atom_q_rope: cute.CopyAtom
            :param mKR: Compressed rope tensor
            :type mKR: cute.Tensor
            :param tma_atom_c_latent: TMA copy atom for c latent tensor
            :type tma_atom_c_latent: cute.CopyAtom
            :param mCL: Compressed latent tensor
            :type mCL: cute.Tensor
            :param tma_atom_c_rope: TMA copy atom for c rope tensor
            :type tma_atom_c_rope: cute.CopyAtom
            :param mCLT: Compressed latent transpose tensor
            :type mCLT: cute.Tensor
            :param mPT: Page table tensor
            :type mPT: cute.Tensor
            :param mO: Output tensor
            :type mO: cute.Tensor
            :param mLSE: Log-sum-exp tensor
            :type mLSE: cute.Tensor
            :param mAccO: Intermediate accumulator output tensor
            :type mAccO: cute.Tensor
            :param mAccLSE: Intermediate accumulator log-sum-exp tensor
            :type mAccLSE: cute.Tensor
            :param split_kv: The split_kv parameter
            :type split_kv: cutlass.Int32
            :param cache_seqs: The variable sequence length tensor
            :type cache_seqs: cute.Tensor
            :param block_split_kvs: The per-block split_kv values tensor
            :type block_split_kvs: cute.Tensor
            :param softmax_scale_log2: The log2 scale factor for softmax
            :type softmax_scale_log2: cutlass.Float32
            :param output_scale: The scale factor for the output
            :type output_scale: cutlass.Float32
            :param q_latent_smem_layout_staged: Shared memory layout for query tensor
            :type q_latent_smem_layout_staged: cute.ComposedLayout
            :param q_rope_smem_layout_staged: Shared memory layout for query rope tensor
            :type q_rope_smem_layout_staged: cute.ComposedLayout
            :param kc_latent_smem_layout_staged: Shared memory layout for key tensor
            :type kc_latent_smem_layout_staged: cute.ComposedLayout
            :param kc_rope_smem_layout_staged: Shared memory layout for key rope tensor
            :type kc_rope_smem_layout_staged: cute.ComposedLayout
            :param p_smem_layout_staged: Shared memory layout for probability matrix
            :type p_smem_layout_staged: cute.ComposedLayout
            :param vc_smem_layout_staged: Shared memory layout for value tensor
            :type vc_smem_layout_staged: cute.ComposedLayout
            :param cta_layout_vmnk: Layout for compute threads
            :type cta_layout_vmnk: cute.Layout
            :param tile_sched_params: Scheduling parameters for work distribution
            :type tile_sched_params: MLAStaticTileSchedulerParams
            :param SharedStorage: Shared storage for the kernel
            :type SharedStorage: cutlass.Constexpr
            """
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
            is_leader_cta = mma_tile_coord_v == 0

            # Prefetch TMA descriptors from load warps (not MMA warp) so the
            # MMA warp can proceed to TMEM allocation sooner.  The prefetch
            # populates the L2 cache and is not warp-specific.
            if warp_idx == self.load_tma_k_warp_id:
                cpasync.prefetch_descriptor(tma_atom_c_latent)
                cpasync.prefetch_descriptor(tma_atom_c_rope)
            if warp_idx == self.load_tma_v_warp_id:
                cpasync.prefetch_descriptor(tma_atom_q_latent)
                cpasync.prefetch_descriptor(tma_atom_q_rope)
                cpasync.prefetch_descriptor(tma_atom_c_latent_transpose)

            # Alloc
            smem = cutlass.memory.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            spare = storage.epilogue_smem_exchange.data_ptr()
            if cutlass.const_expr(self.ts_persistent):
                # [persistent] 32 B past the last pipeline barrier array (mma_o): the
                # padding before smem_p (1 KB aligned after 28 barriers = 224 B), which
                # nothing else touches in either CTA of the pair (the epilogue exchange
                # is busy with the previous work tile's epilogue)
                spare = storage.mma_o_mbar_ptr.data_ptr() + (self.mma_o_stage * 2 + 4)

            # Tensor memory dealloc barrier init
            tmem = cutlass.memory.TmemAllocator(
                storage.tmem_holding_buf.ptr,
                barrier_for_retrieve=self.tmem_ptr_sync_bar,
                allocator_warp_id=self.mma_warp_id,
                is_two_cta=self.use_2cta_instrs,
                two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
                arch=self.arch,
            )

            load_q_pipeline = self.make_and_init_load_qkv_pipeline(
                storage.load_q_mbar_ptr.data_ptr(),
                cta_layout_vmnk,
                self.load_q_stage,
                self.ts_q_tx,
            )
            load_k_pipeline = self.make_and_init_load_qkv_pipeline(
                storage.load_k_mbar_ptr.data_ptr(),
                cta_layout_vmnk,
                self.load_k_pipeline_stage,
                self.tma_copy_kc_latent_bytes,
            )
            if cutlass.const_expr(self.use_m64_ws):
                load_k_rope_pipeline = self.make_and_init_load_qkv_pipeline(
                    storage.load_k_rope_mbar_ptr.data_ptr(),
                    cta_layout_vmnk,
                    self.load_k_rope_pipeline_stage,
                    self.tma_copy_kc_rope_bytes,
                )
            else:
                load_k_rope_pipeline = load_k_pipeline
            load_v_pipeline = self.make_and_init_load_qkv_pipeline(
                storage.load_v_mbar_ptr.data_ptr(),
                cta_layout_vmnk,
                self.load_v_pipeline_stage,
                self.tma_copy_vc_bytes,
            )
            mma_s_pipeline = self.make_and_init_mma_s_pipeline(
                storage.mma_s_mbar_ptr.data_ptr(), cta_layout_vmnk
            )
            p_mma_pipeline = self.make_and_init_p_mma_pipeline(
                storage.p_mma_mbar_ptr.data_ptr(), cta_layout_vmnk
            )
            mma_o_pipeline = self.make_and_init_mma_o_pipeline(
                storage.mma_o_mbar_ptr.data_ptr(), cta_layout_vmnk
            )

            # Cluster arrive after cross-CTA barrier init.
            # p_cor_pipeline is intra-CTA only, so it can be initialized after
            # arrive to overlap with the cluster sync.
            pipeline_init_arrive(
                cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True
            )

            p_cor_pipeline = self.make_and_init_p_cor_pipeline(
                storage.p_cor_mbar_ptr.data_ptr()
            )

            # Generate smem tensor Q/KC/VC/exchange
            # (MMA, MMA_H, MMA_R, PIPE)
            sQ = storage.smem_q_latent.get_tensor(
                q_latent_smem_layout_staged.outer,
                swizzle=q_latent_smem_layout_staged.inner,
            )
            sQ_rope = storage.smem_q_rope.get_tensor(
                q_rope_smem_layout_staged.outer, swizzle=q_rope_smem_layout_staged.inner
            )
            # (MMA, MMA_K, MMA_R, PIPE)
            sKC = storage.smem_kc_latent.get_tensor(
                kc_latent_smem_layout_staged.outer,
                swizzle=kc_latent_smem_layout_staged.inner,
            )
            sKC_rope = storage.smem_kc_rope.get_tensor(
                kc_rope_smem_layout_staged.outer,
                swizzle=kc_rope_smem_layout_staged.inner,
            )
            sKC_for_tma = storage.smem_kc_latent.get_tensor(
                kc_latent_smem_layout_for_tma.outer,
                swizzle=kc_latent_smem_layout_for_tma.inner,
            )
            sKC_rope_for_tma = storage.smem_kc_rope.get_tensor(
                kc_rope_smem_layout_for_tma.outer,
                swizzle=kc_rope_smem_layout_for_tma.inner,
            )
            # (MMA, MMA_D, MMA_K, PIPE)
            sVC = storage.smem_vc.get_tensor(
                vc_smem_layout_staged.outer, swizzle=vc_smem_layout_staged.inner
            )
            sVC_for_tma = storage.smem_vc.get_tensor(
                vc_smem_layout_for_tma.outer, swizzle=vc_smem_layout_for_tma.inner
            )
            # (MMA, MMA_H, MMA_K)
            sP = storage.smem_p.get_tensor(
                p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner
            )
            # (compute_threads,)
            softmax_smem_exchange = storage.softmax_smem_exchange.get_tensor(
                cute.make_layout(self.num_compute_warps * self.threads_per_warp)
            )
            epilogue_smem_exchange = storage.epilogue_smem_exchange.get_tensor(
                cute.make_layout(self.num_compute_warps * self.threads_per_warp)
            )

            #
            # Cluster wait before tensor memory alloc
            #
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)

            # ///////////////////////////////////////////////////////////////////////////////
            #  Load warps, including page table and data tensors
            # ///////////////////////////////////////////////////////////////////////////////
            if (
                warp_idx >= self.empty_warp_ids[0]
                and warp_idx <= self.empty_warp_ids[-1]
            ):
                cute.arch.setmaxregister_decrease(self.other_reg_num)
                # [tsport] after the PDL wait: this CTA's share of x re-armed
                cute.arch.griddepcontrol_wait()
                self.ts_arm_x(block_split_kvs, tidx % 32)

            if warp_idx == self.load_tma_k_warp_id:
                cute.arch.setmaxregister_decrease(self.other_reg_num)
                cute.arch.griddepcontrol_wait()
                load_k_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.load_k_pipeline_stage
                )
                if cutlass.const_expr(self.use_m64_ws):
                    load_k_rope_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer,
                        self.load_k_rope_pipeline_stage,
                    )
                else:
                    load_k_rope_producer_state = load_k_producer_state
                tile_sched = create_mla_static_tile_scheduler(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    blk_coord = self.ts_item(
                        work_tile.tile_idx, tile_sched, block_split_kvs
                    )
                    k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                        split_kv,
                        cache_seqs,
                        block_split_kvs,
                        blk_coord,
                    )
                    if k_tile_count > 0:
                        tma_common_params = SimpleNamespace(
                            blk_coord=blk_coord,
                            local_split_kv=local_split_kv,
                            load_k_pipeline=load_k_pipeline,
                            load_k_rope_pipeline=load_k_rope_pipeline,
                            mPT=mPT,
                        )
                        tma_k_params = SimpleNamespace(
                            tiled_mma_qk=tiled_mma_qk,
                            tma_atom_c_latent=tma_atom_c_latent,
                            tma_atom_c_rope=tma_atom_c_rope,
                            mCL=mCL,
                            mKR=mKR,
                            sKC=sKC_for_tma,
                            sKC_full=sKC,
                            sKC_rope=sKC_rope_for_tma,
                        )
                        # Load K only
                        (
                            load_k_producer_state,
                            load_k_rope_producer_state,
                        ) = self.load_tma_k(
                            tma_common_params,
                            tma_k_params,
                            k_index,
                            k_tile_count,
                            load_k_producer_state,
                            load_k_rope_producer_state,
                        )
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                load_k_pipeline.producer_tail(load_k_producer_state)
                if cutlass.const_expr(self.use_m64_ws):
                    load_k_rope_pipeline.producer_tail(load_k_rope_producer_state)

            if warp_idx == self.load_tma_v_warp_id:
                cute.arch.setmaxregister_decrease(self.other_reg_num)
                cute.arch.griddepcontrol_wait()
                load_q_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.load_q_stage
                )
                load_v_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.load_v_pipeline_stage
                )
                tile_sched = create_mla_static_tile_scheduler(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    blk_coord = self.ts_item(
                        work_tile.tile_idx, tile_sched, block_split_kvs
                    )
                    k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                        split_kv,
                        cache_seqs,
                        block_split_kvs,
                        blk_coord,
                    )
                    if k_tile_count > 0:
                        # Load Q first (once per work tile)
                        tma_q_common_params = SimpleNamespace(
                            blk_coord=blk_coord,
                            load_q_pipeline=load_q_pipeline,
                        )
                        tma_q_params = SimpleNamespace(
                            tiled_mma_qk=tiled_mma_qk,
                            tma_atom_q_latent=tma_atom_q_latent,
                            tma_atom_q_rope=tma_atom_q_rope,
                            mQL=mQL,
                            mQR=mQR,
                            sQ=sQ,
                            sQ_rope=sQ_rope,
                        )
                        # [tsport] every q chunk of this CTA's rows has landed
                        if cutlass.const_expr(self.ts_valid == 2):
                            # [valid 2] the Q load is the poll (``ts_valid_tma``)
                            load_q_pipeline.producer_acquire(load_q_producer_state)
                            full = load_q_pipeline.producer_get_barrier(
                                load_q_producer_state
                            )
                            if cute.arch.lane_idx() == 0:
                                self.ts_valid_tma(
                                    blk_coord,
                                    storage.smem_q_latent.data_ptr(),
                                    storage.smem_q_rope.data_ptr(),
                                    spare,
                                    full,
                                    tensormap.map_address(ts_latent_map),
                                    tensormap.map_address(ts_rope_map),
                                )
                            cute.arch.sync_warp()
                            load_q_producer_state.advance()
                        elif cutlass.const_expr(self.ts_valid):
                            if cutlass.const_expr(POLL_Q):
                                if cute.arch.lane_idx() == 0:
                                    self.ts_valid_q(
                                        block_split_kvs,
                                        blk_coord,
                                        storage.smem_q_latent.data_ptr(),
                                        storage.smem_q_rope.data_ptr(),
                                        storage.epilogue_smem_exchange.data_ptr(),
                                    )
                                cute.arch.sync_warp()
                        else:
                            self.ts_q_bar.arrive_and_wait()
                            fence_proxy_async_global()
                        if cutlass.const_expr(self.ts_valid != 2):
                            load_q_producer_state = self.load_tma_q(
                                tma_q_common_params,
                                tma_q_params,
                                load_q_producer_state,
                            )
                        # Then load V
                        tma_v_common_params = SimpleNamespace(
                            blk_coord=blk_coord,
                            local_split_kv=local_split_kv,
                            load_v_pipeline=load_v_pipeline,
                            mPT=mPT,
                        )
                        tma_pv_params = SimpleNamespace(
                            tiled_mma_pv=tiled_mma_pv,
                            tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
                            mCLT=mCLT,
                            sVC=sVC_for_tma,
                        )
                        load_v_producer_state = self.load_tma_v(
                            tma_v_common_params,
                            tma_pv_params,
                            k_index,
                            k_tile_count,
                            load_v_producer_state,
                        )
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                load_q_pipeline.producer_tail(load_q_producer_state)
                load_v_pipeline.producer_tail(load_v_producer_state)

            # ///////////////////////////////////////////////////////////////////////////////
            #  MMA warp
            # ///////////////////////////////////////////////////////////////////////////////
            if warp_idx == self.mma_warp_id:
                cute.arch.setmaxregister_decrease(self.other_reg_num)
                # Fire off async TMEM allocation request early, then overlap
                # the allocation latency with pipeline state creation and
                # tile scheduler setup.
                tmem.allocate(_get_max_tmem_alloc_cols(self.arch))

                load_q_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.load_q_stage
                )
                load_k_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.load_k_pipeline_stage
                )
                if cutlass.const_expr(self.use_m64_ws):
                    load_k_rope_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer,
                        self.load_k_rope_pipeline_stage,
                    )
                else:
                    load_k_rope_consumer_state = load_k_consumer_state
                load_v_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.load_v_pipeline_stage
                )
                mma_s_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.mma_s_stage
                )
                p_mma_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.p_mma_stage
                )
                mma_o_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.mma_o_stage
                )
                tile_sched = create_mla_static_tile_scheduler(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()

                # Now wait for TMEM allocation and broadcast pointer
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                # The PDL wait before this warp's first global reads (the work-list
                # word, the request lengths): the first MLA layer's work list is
                # written by the kernel right before this one. (Free: the MMA warp
                # issues nothing before the loads, which follow their warps' waits.)
                cute.arch.griddepcontrol_wait()
                while work_tile.is_valid_tile:
                    blk_coord = self.ts_item(
                        work_tile.tile_idx, tile_sched, block_split_kvs
                    )
                    k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                        split_kv, cache_seqs, block_split_kvs, blk_coord
                    )
                    if k_tile_count > 0:
                        mma_common_params = SimpleNamespace(
                            blk_coord=blk_coord,
                            K=cache_seqs[blk_coord[2]],
                            k_index=k_index,
                            local_split_kv=local_split_kv,
                            load_q_pipeline=load_q_pipeline,
                            load_k_pipeline=load_k_pipeline,
                            load_k_rope_pipeline=load_k_rope_pipeline,
                            load_v_pipeline=load_v_pipeline,
                            tmem_ptr=tmem_ptr,
                            is_leader_cta=is_leader_cta,
                            L=mCL.shape[1],
                            tiled_mma_ws_epi=tiled_mma_ws_epi,
                            tiled_mma_ws_splitn_epi=tiled_mma_ws_splitn_epi,
                        )
                        mma_qk_params = SimpleNamespace(
                            mma_s_pipeline=mma_s_pipeline,
                            sQ=sQ,
                            sQ_rope=sQ_rope,
                            sKC=sKC,
                            sKC_rope=sKC_rope,
                        )
                        mma_pv_params = SimpleNamespace(
                            p_mma_pipeline=p_mma_pipeline,
                            mma_o_pipeline=mma_o_pipeline,
                            sP=sP,
                            sVC=sVC,
                        )
                        (
                            tiled_mma_qk,
                            tiled_mma_pv,
                            load_q_consumer_state,
                            load_k_consumer_state,
                            load_k_rope_consumer_state,
                            load_v_consumer_state,
                            mma_s_producer_state,
                            p_mma_consumer_state,
                            mma_o_producer_state,
                        ) = self.mma(
                            mma_common_params,
                            mma_qk_params,
                            mma_pv_params,
                            k_tile_count,
                            tiled_mma_qk,
                            tiled_mma_pv,
                            load_q_consumer_state,
                            load_k_consumer_state,
                            load_k_rope_consumer_state,
                            load_v_consumer_state,
                            mma_s_producer_state,
                            p_mma_consumer_state,
                            mma_o_producer_state,
                        )
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                mma_s_pipeline.producer_tail(mma_s_producer_state)
                mma_o_pipeline.producer_tail(mma_o_producer_state)
                cute.arch.griddepcontrol_launch_dependents()

                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)

            # ///////////////////////////////////////////////////////////////////////////////
            #  Compute warp
            # ///////////////////////////////////////////////////////////////////////////////
            if (
                warp_idx >= self.compute_warp_ids[0]
                and warp_idx <= self.compute_warp_ids[-1]
            ):
                cute.arch.setmaxregister_increase(self.softmax_reg_num)
                # [tsport] q polled after the PDL wait, before the first S
                cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(not self.ts_valid):
                    self.ts_poll_q(
                        block_split_kvs, split_kv, cache_seqs, tile_sched_params, tidx
                    )
                mma_s_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.mma_s_stage
                )
                p_mma_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.p_mma_stage
                )
                p_cor_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.p_cor_stage
                )
                mma_o_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.mma_o_stage
                )
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

                tile_sched = create_mla_static_tile_scheduler(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    blk_coord = self.ts_item(
                        work_tile.tile_idx, tile_sched, block_split_kvs
                    )
                    k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                        split_kv, cache_seqs, block_split_kvs, blk_coord
                    )
                    if k_tile_count > 0:
                        compute_common_params = SimpleNamespace(
                            blk_coord=blk_coord,
                            split_kv=split_kv,
                            local_split_kv=local_split_kv,
                            smem_exchange=softmax_smem_exchange,
                            mAccO=mAccO,
                            mO=mO,
                            K=cache_seqs[blk_coord[2]],
                            K_causal=causal_seqs[blk_coord[2]],
                            local_visible_lens=local_visible_lens,
                            L=mCL.shape[1],
                            tmem_ptr=tmem_ptr,
                            tidx=tidx,
                            tiled_mma_ws_epi=tiled_mma_ws_epi,
                            tiled_mma_ws_splitn_epi=tiled_mma_ws_splitn_epi,
                            p_cor_pipeline=p_cor_pipeline,
                        )
                        compute_softmax_params = SimpleNamespace(
                            tiled_mma_qk=tiled_mma_qk,
                            tiled_mma_ws_epi=tiled_mma_ws_epi,
                            tiled_mma_ws_splitn_epi=tiled_mma_ws_splitn_epi,
                            sP=sP,
                            mma_s_pipeline=mma_s_pipeline,
                            p_mma_pipeline=p_mma_pipeline,
                            softmax_scale_log2=softmax_scale_log2,
                        )
                        (
                            mma_s_consumer_state,
                            p_mma_producer_state,
                            p_cor_producer_state,
                        ) = self.compute(
                            compute_common_params,
                            compute_softmax_params,
                            k_index=k_index,
                            k_tile_count=k_tile_count,
                            mma_s_consumer_state=mma_s_consumer_state,
                            p_mma_producer_state=p_mma_producer_state,
                            p_cor_producer_state=p_cor_producer_state,
                        )
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                p_cor_pipeline.producer_tail(p_cor_producer_state)
                if cutlass.const_expr(self.ts_push is not None):
                    self.ts_push_tail(block_split_kvs, mO, mLSE, cache_seqs, tidx)

            # ///////////////////////////////////////////////////////////////////////////////
            #  Correction warp
            # ///////////////////////////////////////////////////////////////////////////////
            if (
                warp_idx >= self.correction_warp_ids[0]
                and warp_idx <= self.correction_warp_ids[-1]
            ):
                cute.arch.setmaxregister_increase(self.correction_reg_num)
                # [tsport] q polled after the PDL wait, before the first PV
                cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(not self.ts_valid):
                    self.ts_poll_q(
                        block_split_kvs, split_kv, cache_seqs, tile_sched_params, tidx
                    )
                p_cor_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.p_cor_stage
                )
                mma_o_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.mma_o_stage
                )
                # sync with mma warp before retrieving tmem ptr
                tmem.wait_for_alloc()

                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

                tile_sched = create_mla_static_tile_scheduler(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    blk_coord = self.ts_item(
                        work_tile.tile_idx, tile_sched, block_split_kvs
                    )
                    k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                        split_kv, cache_seqs, block_split_kvs, blk_coord
                    )
                    if k_tile_count > 0:
                        compute_common_params = SimpleNamespace(
                            blk_coord=blk_coord,
                            split_kv=split_kv,
                            local_split_kv=local_split_kv,
                            smem_exchange=epilogue_smem_exchange,
                            mAccO=mAccO,
                            mO=mO,
                            K=cache_seqs[blk_coord[2]],
                            K_causal=causal_seqs[blk_coord[2]],
                            local_visible_lens=local_visible_lens,
                            L=mCL.shape[1],
                            H=(
                                cutlass.min(
                                    self.mma_qk_tiler[0],
                                    self.total_q_rows
                                    - blk_coord[1] * self.mma_qk_tiler[0],
                                )
                                if cutlass.const_expr(self.pack_q)
                                else mQL.shape[0]
                            ),
                            tmem_ptr=tmem_ptr,
                            tidx=tidx,
                            tiled_mma_pv=tiled_mma_pv,
                            tiled_mma_ws_epi=tiled_mma_ws_epi,
                            tiled_mma_ws_splitn_epi=tiled_mma_ws_splitn_epi,
                            p_cor_pipeline=p_cor_pipeline,
                            mma_o_pipeline=mma_o_pipeline,
                        )
                        compute_epilogue_params = SimpleNamespace(
                            output_scale=output_scale,
                            softmax_scale_log2=softmax_scale_log2,
                            mAccLSE=mAccLSE,
                            mLSE=mLSE,
                        )
                        p_cor_consumer_state, mma_o_consumer_state = self.correction(
                            compute_common_params,
                            compute_epilogue_params,
                            k_tile_count=k_tile_count,
                            p_cor_consumer_state=p_cor_consumer_state,
                            mma_o_consumer_state=mma_o_consumer_state,
                        )
                    elif cutlass.const_expr(
                        mAccO is None and local_visible_lens is not None
                    ):
                        # No producer ran for this zero-K tile.
                        self.write_empty_output(mO, mLSE, blk_coord, tidx)
                    if cutlass.const_expr(self.ts_push is not None):
                        self.ts_publish(
                            block_split_kvs, tile_sched.current_work_linear_idx, tidx
                        )
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                if cutlass.const_expr(self.ts_push is not None):
                    self.ts_push_tail(block_split_kvs, mO, mLSE, cache_seqs, tidx)

            return

    return TsCore

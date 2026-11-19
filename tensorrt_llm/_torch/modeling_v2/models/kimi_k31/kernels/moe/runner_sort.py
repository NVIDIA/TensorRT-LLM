# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MoE runner sort (SM100/SM103): the TRTLLM-Gen runner's pre-routed layout of one
expert-parallel rank from the top-16 expert ids, deterministic, in one launch, up to
:data:`CLUSTER_MAX_ROWS` rows.

Input ``topk_ids`` int32 ``[M, 16]`` (global experts), the rank's expert window ``[local_offset,
local_offset + local_num)`` and the tile ``2^k`` rows. Outputs (int32): ``e2p [M * 16]`` (expanded
index ``token * 16 + slot`` -> permuted row, -1 for other ranks' experts), ``p2t`` (permuted row ->
token; the padding rows of the tiles are -1), ``total [1]`` (tiles * tile), ``t2e`` / ``t2l`` (tile ->
local expert / exclusive permuted-row limit) and ``nt [1]`` (tiles). Local experts ascending, each
expert's rows in ascending token order, ``ceil(count / tile)`` tiles per touched expert. Entries
past ``nt`` (tiles) and ``total`` (rows) are not written.

Rows of an expert follow the token order without atomics: a CTA marks, per local expert ``e`` and
32-token word ``w``, the bitmask of its tokens that chose ``e`` (``red.shared.or``); a thread per
expert turns its words into the rows before each word (``pre``) and its count, so a pick's row is
``base[e] + pre[e][w] + popc(mask[e][w] & below(token))``.

* ``M <= CLUSTER_MAX_ROWS``: one CTA up to 256 rows, else one cluster of 2..16 CTAs (about 256 rows
  each), each owning a contiguous block of tokens. The CTAs send their per-expert counts to every
  CTA of the cluster (``st.async`` completing on an mbarrier, no cluster barrier); each CTA then
  scans the expert tiles (``base[e]`` = the expert's first row + its rows in earlier CTAs), writes
  its share of ``t2e`` / ``t2l``, its tokens' ``e2p`` (16-byte stores) and rows, and a share of the
  padding rows (consecutive rows on consecutive threads); CTA 0 writes the totals. No global state:
  a CUDA graph replays it as is.
* The one-CTA launch (decode / verify rows, tiles >= 8) builds ``p2t`` (-1 filled) in shared memory
  and stores it coalesced. Only ``4 * words`` warps (one int4 of ids a thread) work after the wait,
  synchronized by a named barrier, the others clear the tables and exit: threads own ``EPT``
  consecutive local experts (vector loads of their mask words), take their warp's exclusive tile
  prefix by one ballot per bit (no shuffle chain), store it with the warp totals, and every thread
  reads an expert's first tile as the totals of the expert warps before it (one broadcast load) +
  that prefix. Three barriers after the wait (marks, prefixes, staged rows).

Every launch waits on the previous grid (programmatic dependent launch) only after clearing its
shared memory, and releases its dependents after its stores.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .route_ptx import (
    NUM_EXPERTS,
    TOP_K,
    _asm,
    _iv,
    _lds_b32,
    _lv,
    _popc,
    _red_shared_or,
    _st_global_b32,
    _sts_b32,
    _unpack,
    _warp_inclusive_sum,
)
from .route_tail import (
    _bar_sync_named,
    _cluster_arrive_relaxed,
    _cluster_wait_acquire,
    _griddepcontrol_launch_dependents,
    _griddepcontrol_wait,
    _lds_b32_or_zero,
    _shfl_idx_b32,
    _sts_b32_if,
    _sts_zero16,
)

#: Largest row count served.
CLUSTER_MAX_ROWS = 16384
#: Largest local expert count served.
MAX_LOCAL = 1024
#: Rows up to which the cluster path runs on one CTA; a cluster CTA owns at most 1024 tokens.
_ONE_CTA_ROWS = 256
#: Rows per CTA above that (cluster of up to _MAX_CLUSTER CTAs).
_CLUSTER_CTA_ROWS = 256
_MAX_CLUSTER = 16
#: One-CTA path: p2t rows staged in shared memory (the bound of max_tiles * tile).
_STAGE_ROWS = 16384
#: Bound of the (expert, word) mask + prefix tables of one CTA.
_MAX_TABLE_BYTES = 160 * 1024
_THREADS = 1024
#: One-CTA layout: 16-byte p2t copies a thread keeps in flight.
_COPY_UNROLL = 4


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def _ldg_v4_or_neg(n_left, addr, *, loc=None, ip=None):
    """``ld.global.v4.b32 [addr]`` if ``n_left > 0`` else four -1."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(n_left, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.gt.s32 p, $4, 0;\n"
        "mov.b32 $0, -1;\nmov.b32 $1, -1;\nmov.b32 $2, -1;\nmov.b32 $3, -1;\n"
        "@p ld.global.v4.b32 {$0, $1, $2, $3}, [$5];\n}",
        "=&r,=&r,=&r,=&r,r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _stg_v4_b32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_b32_if(valid, addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.global.b32 [$1], $2;\n}",
        "r,l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_v4_b32_if(valid, addr, *, loc=None, ip=None):
    """``ld.shared.v4.b32 [addr]`` if ``valid != 0`` else four zeros."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $4, 0;\n"
        "mov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\nmov.b32 $3, 0;\n"
        "@p ld.shared.v4.b32 {$0, $1, $2, $3}, [$5];\n}",
        "=&r,=&r,=&r,=&r,r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _lds_v2_b32_if(valid, addr, *, loc=None, ip=None):
    """``ld.shared.v2.b32 [addr]`` if ``valid != 0`` else two zeros."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $2, 0;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\n"
        "@p ld.shared.v2.b32 {$0, $1}, [$3];\n}",
        "=&r,=&r,r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 2, loc=loc, ip=ip)


@dsl_user_op
def _sts_v4_b32_if(valid, addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)]
        + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.shared.v4.b32 [$1], {$2, $3, $4, $5};\n}",
        "r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_v2_b32_if(valid, addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [
            _iv(valid, loc=loc, ip=ip),
            _iv(addr, loc=loc, ip=ip),
            _iv(v0, loc=loc, ip=ip),
            _iv(v1, loc=loc, ip=ip),
        ],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.shared.v2.b32 [$1], {$2, $3};\n}",
        "r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_v4_b32_if(valid, addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip)]
        + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.global.v4.b32 [$1], {$2, $3, $4, $5};\n}",
        "r,l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


def _lds_words_if(valid, addr, n: int) -> list:
    """``n`` (1, 2 or a multiple of 4) consecutive shared words at ``addr`` (aligned to the vector), zeros if not
    ``valid``."""
    if n == 1:
        return [_lds_b32_or_zero(valid, addr)]
    if n == 2:
        return list(_lds_v2_b32_if(valid, addr))
    out = []
    for q in range(n // 4):
        out += list(_lds_v4_b32_if(valid, addr + q * 16))
    return out


def _sts_words_if(valid, addr, values: list) -> None:
    """Store ``values`` (1, 2 or a multiple of 4 words) at ``addr`` (aligned to the vector) if ``valid``."""
    n = len(values)
    if n == 1:
        _sts_b32_if(valid, addr, values[0])
    elif n == 2:
        _sts_v2_b32_if(valid, addr, values[0], values[1])
    else:
        for q in range(n // 4):
            _sts_v4_b32_if(valid, addr + q * 16, *values[4 * q : 4 * q + 4])


@dsl_user_op
def _warp_scan_ballot(value, bits: int, *, loc=None, ip=None):
    """``(exclusive prefix over the lower lanes, warp total)`` of ``value`` in ``[0, 2^bits)``, one ballot per bit
    (independent, no shuffle chain). All 32 lanes must call it."""
    body = "".join(
        f"and.b32 t, x, {1 << b};\nsetp.ne.u32 p, t, 0;\nvote.sync.ballot.b32 m, p, 0xffffffff;\n"
        f"and.b32 t, m, lt;\npopc.b32 t, t;\nshl.b32 t, t, {b};\nadd.u32 ex, ex, t;\n"
        f"popc.b32 t, m;\nshl.b32 t, t, {b};\nadd.u32 tot, tot, t;\n"
        for b in range(bits)
    )
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_iv(value, loc=loc, ip=ip)],
        "{\n.reg .b32 x, t, m, lt, ex, tot;\n.reg .pred p;\nmov.b32 x, $2;\nmov.u32 lt, %lanemask_lt;\n"
        "mov.b32 ex, 0;\nmov.b32 tot, 0;\n" + body + "mov.b32 $0, ex;\nmov.b32 $1, tot;\n}",
        "=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 2, loc=loc, ip=ip)


@dsl_user_op
def _sts_neg16(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 z;\nmov.b32 z, -1;\nst.shared.v4.b32 [$0], {z, z, z, z};\n}",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_v4_b32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _mbar_init_expect(mbar, count, tx_bytes, *, loc=None, ip=None):
    """Init a one-arrival mbarrier, publish the init to the cluster, arrive expecting ``tx_bytes``."""
    _asm(
        None,
        [_iv(mbar, loc=loc, ip=ip), _iv(count, loc=loc, ip=ip), _iv(tx_bytes, loc=loc, ip=ip)],
        "mbarrier.init.shared::cta.b64 [$0], $1;\n"
        "fence.mbarrier_init.release.cluster;\n"
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [$0], $2;",
        "r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_async_b32(addr, rank, value, mbar, *, loc=None, ip=None):
    """``st.async`` of 4 bytes into cluster CTA ``rank``'s shared ``addr``, completing on its ``mbar``."""
    _asm(
        None,
        [
            _iv(addr, loc=loc, ip=ip),
            _iv(rank, loc=loc, ip=ip),
            _iv(value, loc=loc, ip=ip),
            _iv(mbar, loc=loc, ip=ip),
        ],
        "{\n.reg .b32 ra, rb;\nmapa.shared::cluster.u32 ra, $0, $1;\nmapa.shared::cluster.u32 rb, $3, $1;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [ra], $2, [rb];\n}",
        "r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_test_wait(mbar, parity, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(mbar, loc=loc, ip=ip), _iv(parity, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


def _min(a, b):
    return cutlass.select_(a < b, a, b)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class RunnerSortKernel:
    """One configuration of the runner sort (see the module docstring).

    Args:
        ctas: Cluster size (1, 2, 4, 8 or 16).
        words: 32-token mask words per expert and CTA (a power of two <= 32): a CTA owns at most
            ``32 * words`` tokens.
        local_cap: Local experts the shared tables hold (a power of two, 64..1024, <= threads).
        threads: Threads per CTA.
        early_release: release the dependent grid right after this grid's own wait on the previous one instead
            of after the stores, so a consumer that waits for this grid before reading the sort (the W4A8
            experts' 1-CTA kernel) launches and sets up while the sort runs.
        fast: (one CTA with the staged p2t; the caller guarantees tiles >= 8) the one-CTA layout of
            :meth:`one_cta` instead of the general one.
    """

    def __init__(
        self,
        ctas: int,
        words: int,
        local_cap: int,
        threads: int,
        stage_rows: int = 0,
        expanded: bool = False,
        early_release: bool = False,
        fast: bool = False,
    ):
        assert ctas in (1, 2, 4, 8, 16)
        assert stage_rows % 4 == 0 and (stage_rows == 0 or ctas == 1)
        assert words in (1, 2, 4, 8, 16, 32)
        assert local_cap & (local_cap - 1) == 0 and 64 <= local_cap <= threads and threads % 32 == 0
        self.ctas = ctas
        self.words = words
        self.words_log2 = words.bit_length() - 1
        self.local_cap = local_cap
        self.threads = threads
        # One-CTA path: p2t is built in shared memory (-1 filled before the PDL wait) and stored
        # coalesced, when the largest row count fits in stage_rows.
        self.stage_rows = stage_rows
        # p2t holds the expanded index ``token * 16 + slot`` of each row instead of the token.
        self.expanded = expanded
        self.early_release = early_release
        # int4 items (4 picks of one token) per thread: a CTA owns <= 32 * words tokens of 4 items.
        self.items = (128 * words + threads - 1) // threads
        # Warps of the per-expert scan.
        self.expert_warps = local_cap // 32
        # The one-CTA layout (one_cta): 4 * words working warps, one int4 item each; threads t < E_T own
        # EPT = local_cap / E_T consecutive experts (E_T = min(local_cap, working threads)); tiles >= 8.
        self.fast = fast and ctas == 1 and stage_rows > 0
        if self.fast:
            self.fast_warps = 4 * words
            assert self.fast_warps * 32 <= threads
            e_t = min(local_cap, self.fast_warps * 32)
            self.fast_ept = local_cap // e_t
            self.fast_ewarps = e_t // 32
            self.fast_ewarp_log2 = (32 * self.fast_ept).bit_length() - 1
            # Ballot-scan bits: a thread's tiles <= EPT * 4 * words, an expert warp's <= 32 times that.
            self.fast_bits = (self.fast_ept * 4 * words).bit_length()
            self.fast_wbits = (128 * self.fast_ept * words).bit_length()

    @cute.jit
    def __call__(
        self,
        m: Int32,
        per_cta: Int32,
        grid: Int32,
        ids_addr: Int64,
        e2p_addr: Int64,
        p2t_addr: Int64,
        total_addr: Int64,
        t2e_addr: Int64,
        t2l_addr: Int64,
        nt_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        tile_log2: Int32,
        stream: cuda.CUstream,
    ):
        lcap = self.local_cap
        cells = lcap * self.words

        @cute.struct
        class SharedStorage:
            mask: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, cells], 16]
            pre: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, cells], 16]
            # Per-CTA expert counts ([CTA][expert] over DSMEM).
            cnt: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.ctas * lcap], 16]
            base: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, lcap], 16]
            first: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, lcap], 16]
            tot: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, lcap], 16]
            stage: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, max(4, self.stage_rows)], 16
            ]
            wsum: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 32], 16]
            flag: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]

        self.shared_storage = SharedStorage
        kernel = self.kernel(
            m, per_cta, grid, ids_addr, e2p_addr, p2t_addr, total_addr, t2e_addr, t2l_addr, nt_addr, local_offset,
            local_num, tile_log2,
        )  # fmt: skip
        if cutlass.const_expr(self.ctas > 1):
            kernel.launch(
                grid=(self.ctas, 1, 1),
                block=(self.threads, 1, 1),
                cluster=(self.ctas, 1, 1),
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
                use_pdl=True,
            )
        else:
            kernel.launch(
                grid=(1, 1, 1),
                block=(self.threads, 1, 1),
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
                use_pdl=True,
            )

    @cute.kernel
    def kernel(
        self,
        m: Int32,
        per_cta: Int32,
        grid: Int32,
        ids_addr: Int64,
        e2p_addr: Int64,
        p2t_addr: Int64,
        total_addr: Int64,
        t2e_addr: Int64,
        t2l_addr: Int64,
        nt_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        tile_log2: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        mask_s = storage.mask.data_ptr().toint()
        pre_s = storage.pre.data_ptr().toint()
        cnt_s = storage.cnt.data_ptr().toint()
        base_s = storage.base.data_ptr().toint()
        first_s = storage.first.data_ptr().toint()
        tot_s = storage.tot.data_ptr().toint()
        stage_s = storage.stage.data_ptr().toint()
        wsum_s = storage.wsum.data_ptr().toint()
        flag_s = storage.flag.data_ptr().toint()

        if cutlass.const_expr(self.fast):
            self.one_cta(
                tidx, warp, lane, m, grid, ids_addr, e2p_addr, p2t_addr, total_addr, t2e_addr, t2l_addr, nt_addr,
                local_offset, local_num, tile_log2, mask_s, pre_s, first_s, stage_s, wsum_s,
            )  # fmt: skip
        else:
            self.general(
                tidx, warp, lane, cta, m, per_cta, grid, ids_addr, e2p_addr, p2t_addr, total_addr, t2e_addr,
                t2l_addr, nt_addr, local_offset, local_num, tile_log2, mask_s, pre_s, cnt_s, base_s, first_s, tot_s,
                stage_s, wsum_s, flag_s,
            )  # fmt: skip

    @cute.jit
    def general(
        self, tidx: Int32, warp: Int32, lane: Int32, cta: Int32, m: Int32, per_cta: Int32, grid: Int32,
        ids_addr: Int64, e2p_addr: Int64, p2t_addr: Int64, total_addr: Int64, t2e_addr: Int64, t2l_addr: Int64,
        nt_addr: Int64, local_offset: Int32, local_num: Int32, tile_log2: Int32, mask_s: Int32, pre_s: Int32,
        cnt_s: Int32, base_s: Int32, first_s: Int32, tot_s: Int32, stage_s: Int32, wsum_s: Int32, flag_s: Int32,
    ):  # fmt: skip
        """Every configuration but the one-CTA decode layout (see the module docstring)."""
        lcap = self.local_cap
        W = self.words
        # Shared tables do not depend on the input: clear them before waiting on the previous grid.
        for i in cutlass.range(tidx, lcap * W // 4, self.threads, unroll=1):
            _sts_zero16(mask_s + i * 16)
        if cutlass.const_expr(self.stage_rows > 0):
            # The rows of up to max_tiles tiles (grid carries max_tiles * tile / 4 here).
            for i in cutlass.range(tidx, grid, self.threads, unroll=1):
                _sts_neg16(stage_s + i * 16)
        if cutlass.const_expr(self.ctas > 1):
            # The counts arrive by st.async from every CTA ([CTA][expert] words) on this mbarrier.
            if tidx == 0:
                _mbar_init_expect(flag_s, Int32(1), local_num * (self.ctas * 4))
            _cluster_arrive_relaxed()
        _griddepcontrol_wait()
        if cutlass.const_expr(self.early_release):
            _griddepcontrol_launch_dependents()

        t_begin = cta * per_cta
        t_end = _min(t_begin + per_cta, m)
        n4 = (t_end - t_begin) * 4
        ids_cta = ids_addr + Int64(t_begin) * Int64(TOP_K * 4)
        cute.arch.barrier()

        # Picks: four per int4 item (one token's slots 4 (i % 4) ..), local expert or -1.
        # All loads first (the asm keeps program order), then the shared-memory marks.
        raws = []
        for q in cutlass.range_constexpr(self.items):
            i4 = tidx + q * self.threads
            raws.append(_ldg_v4_or_neg(n4 - i4, ids_cta + Int64(i4 * 16)))
        locs = []
        for q in cutlass.range_constexpr(self.items):
            i4 = tidx + q * self.threads
            raw = raws[q]
            tl = i4 >> 2
            bit = Int32(1) << (tl & 31)
            word = tl >> 5
            item = []
            for j in cutlass.range_constexpr(4):
                v = raw[j] - local_offset
                v = cutlass.select_((v >= 0) & (v < local_num), v, Int32(-1))
                if v >= 0:
                    _red_shared_or(mask_s + ((v << self.words_log2) + word) * 4, bit)
                item.append(v)
            locs.append(item)
        cute.arch.barrier()

        # Thread per local expert: its mask words -> its rows in the words before (pre) and its count.
        e = tidx
        eok = e < local_num
        cnt = Int32(0)
        if eok:
            words = []
            for w in cutlass.range_constexpr(W):
                words.append(_lds_b32(mask_s + ((e << self.words_log2) + w) * 4))
            for w in cutlass.range_constexpr(W):
                if cutlass.const_expr(W > 1):
                    _sts_b32(pre_s + ((e << self.words_log2) + w) * 4, cnt)
                cnt = cnt + _popc(words[w])
        tot = cnt
        before = Int32(0)
        if cutlass.const_expr(self.ctas > 1):
            # Every CTA of the cluster runs once the start barrier completes; broadcast the
            # counts ([CTA][expert]) over distributed shared memory.
            _cluster_wait_acquire()
            if eok:
                for d in cutlass.range_constexpr(self.ctas):
                    _st_async_b32(cnt_s + (cta * lcap + e) * 4, Int32(d), cnt, flag_s)
            done = _mbar_test_wait(flag_s, Int32(0))
            while done == 0:
                done = _mbar_test_wait(flag_s, Int32(0))
            ok = Int32(cutlass.select_(eok, Int32(1), Int32(0)))
            tot = Int32(0)
            for c in cutlass.range_constexpr(self.ctas):
                v = _lds_b32_or_zero(ok, cnt_s + (c * lcap + e) * 4)
                tot = tot + v
                before = before + cutlass.select_(Int32(c) < cta, v, Int32(0))
        num_tiles = self.expert_scan(
            tidx, warp, lane, cta, tot, before, base_s, first_s, tot_s, wsum_s, total_addr, t2e_addr,
            t2l_addr, nt_addr, local_num, tile_log2,
        )  # fmt: skip

        # Rows of this CTA's picks.
        for q in cutlass.range_constexpr(self.items):
            i4 = tidx + q * self.threads
            tl = i4 >> 2
            below = (Int32(1) << (tl & 31)) - 1
            word = tl >> 5
            outs = []
            for j in cutlass.range_constexpr(4):
                v = locs[q][j]
                perm = Int32(-1)
                if v >= 0:
                    cell = (v << self.words_log2) + word
                    perm = _lds_b32(base_s + v * 4) + _popc(_lds_b32(mask_s + cell * 4) & below)
                    if cutlass.const_expr(W > 1):
                        perm = perm + _lds_b32(pre_s + cell * 4)
                    row_val = t_begin + tl
                    if cutlass.const_expr(self.expanded):
                        row_val = row_val * TOP_K + ((i4 & 3) * 4 + j)
                    if cutlass.const_expr(self.stage_rows > 0):
                        _sts_b32(stage_s + perm * 4, row_val)
                    else:
                        _st_global_b32(p2t_addr + Int64(perm * 4), row_val)
                outs.append(perm)
            if i4 < n4:
                e2p_row = e2p_addr + Int64(t_begin) * Int64(TOP_K * 4) + Int64(i4 * 16)
                _stg_v4_b32(e2p_row, outs[0], outs[1], outs[2], outs[3])
        if cutlass.const_expr(self.stage_rows > 0):
            # p2t (rows and padding) from shared memory, 16 bytes a thread.
            cute.arch.barrier()
            for i in cutlass.range(tidx, num_tiles << (tile_log2 - 2), self.threads, unroll=1):
                quad = _lds_v4_b32(stage_s + i * 16)
                _stg_v4_b32(p2t_addr + Int64(i * 16), quad[0], quad[1], quad[2], quad[3])
        else:
            # Padding rows of each expert's last tile: thread (e, r), rows of a tile on consecutive
            # threads, shared by the cluster's CTAs.
            tile_m1 = (Int32(1) << tile_log2) - 1
            for idx in cutlass.range(
                cta * self.threads + tidx,
                local_num << tile_log2,
                self.ctas * self.threads,
                unroll=1,
            ):
                e = idx >> tile_log2
                r = idx & tile_m1
                tot = _lds_b32(tot_s + e * 4)
                last0 = ((tot - 1) >> tile_log2) << tile_log2
                if (tot > 0) & (r >= tot - last0):
                    _st_global_b32(
                        p2t_addr + Int64((_lds_b32(first_s + e * 4) + last0 + r) * 4), Int32(-1)
                    )
        if cutlass.const_expr(not self.early_release):
            _griddepcontrol_launch_dependents()

    @cute.jit
    def one_cta(
        self, tidx: Int32, warp: Int32, lane: Int32, m: Int32, grid: Int32, ids_addr: Int64, e2p_addr: Int64,
        p2t_addr: Int64, total_addr: Int64, t2e_addr: Int64, t2l_addr: Int64, nt_addr: Int64, local_offset: Int32,
        local_num: Int32, tile_log2: Int32, mask_s: Int32, pre_s: Int32, first_s: Int32, stage_s: Int32,
        wsum_s: Int32,
    ):  # fmt: skip
        """The one-CTA layout (rows <= 256, tiles >= 8, p2t staged in shared memory). Every warp clears the tables
        before the wait; then ``4 * words`` warps (one int4 item, the 4 picks of a token, a thread) do the work with
        named barriers among them and the other warps exit:

        1. Picks -> ``red.shared.or`` of the (expert, word) masks. Barrier.
        2. Thread ``t < E_T`` owns ``EPT`` consecutive local experts (vector loads of their mask words): counts,
           tiles, the exclusive tile prefix over the warp's experts (ballot scan); stores each expert's first tile
           within its warp and the warp total. Barrier.
        3. The first tile of expert ``v`` = the tiles of the expert warps before ``v``'s (from the warp totals) + its
           stored prefix: owner threads write ``t2e`` / ``t2l``, item threads their picks' rows (``e2p``, ``p2t`` rows
           into the -1 filled shared copy). Barrier. Coalesced ``p2t`` store."""
        lcap = self.local_cap
        W = self.words
        wl2 = self.words_log2
        NW = self.fast_warps
        EPT = self.fast_ept
        EW = self.fast_ewarps
        # The tables do not depend on the input: clear them (and -1 the staged p2t) before waiting on the previous
        # grid; the barrier orders the clears before every later shared access.
        for i in cutlass.range(tidx, lcap * W // 4, self.threads, unroll=1):
            _sts_zero16(mask_s + i * 16)
        for i in cutlass.range(tidx, grid, self.threads, unroll=1):
            _sts_neg16(stage_s + i * 16)
        cute.arch.barrier()
        if warp < NW:
            _griddepcontrol_wait()
            if cutlass.const_expr(self.early_release):
                _griddepcontrol_launch_dependents()

            # 1. This thread's item (token tl, slots 4 (i4 % 4) ..): local expert or -1, marked.
            n4 = m * 4
            i4 = tidx
            tl = i4 >> 2
            word = tl >> 5
            bit = Int32(1) << (tl & 31)
            raw = _ldg_v4_or_neg(n4 - i4, ids_addr + Int64(i4 * 16))
            locs = []
            for j in cutlass.range_constexpr(4):
                v = raw[j] - local_offset
                v = cutlass.select_((v >= 0) & (v < local_num), v, Int32(-1))
                if v >= 0:
                    _red_shared_or(mask_s + ((v << wl2) + word) * 4, bit)
                locs.append(v)
            self.fast_bar()

            # 2. Thread t < E_T: local experts [t * EPT, (t + 1) * EPT) (masks past local_num stay zero);
            # predicated, so the other working warps get zeros.
            nwd = EPT * W
            own = Int32(cutlass.select_(warp < EW, Int32(1), Int32(0)))
            words = _lds_words_if(own, mask_s + tidx * (nwd * 4), nwd)
            cnts = []
            pre_vals = []
            for k in cutlass.range_constexpr(EPT):
                c = Int32(0)
                for w in cutlass.range_constexpr(W):
                    pre_vals.append(c)
                    c = c + _popc(words[k * W + w])
                cnts.append(c)
            if cutlass.const_expr(W > 1):
                _sts_words_if(own, pre_s + tidx * (nwd * 4), pre_vals)
            tiles = []
            pref = []
            run = Int32(0)
            for k in cutlass.range_constexpr(EPT):
                tiles.append((cnts[k] + (Int32(1) << tile_log2) - 1) >> tile_log2)
                pref.append(run)
                run = run + tiles[k]
            wex, wtot = _warp_scan_ballot(run, self.fast_bits)
            _sts_words_if(own, first_s + tidx * (EPT * 4), [wex + pref[k] for k in range(EPT)])
            _sts_b32_if(
                Int32(cutlass.select_((lane == 0) & (warp < EW), Int32(1), Int32(0))),
                wsum_s + warp * 4,
                wtot,
            )
            self.fast_bar()

            # 3. Tiles before each expert warp (wpre) and in all (num_tiles).
            num_tiles = Int32(0)
            wpre = []
            wexl = Int32(0)
            if cutlass.const_expr(EW <= 8):
                tot = _lds_words_if(Int32(1), wsum_s, EW)
                for w in cutlass.range_constexpr(EW):
                    wpre.append(num_tiles)
                    num_tiles = num_tiles + tot[w]
            else:
                x = _lds_b32_or_zero(
                    Int32(cutlass.select_(lane < EW, Int32(1), Int32(0))), wsum_s + lane * 4
                )
                wexl, num_tiles = _warp_scan_ballot(x, self.fast_wbits)
            # First tile of this thread's experts' warp (owner threads); the rows of this thread's picks.
            base_own = self.warp_base(wpre, wexl, warp) + wex
            below = bit - 1
            firsts = []
            masks = []
            pre_rows = []
            for j in cutlass.range_constexpr(4):
                v = locs[j]
                ok = Int32(cutlass.select_(v >= 0, Int32(1), Int32(0)))
                vv = cutlass.select_(v >= 0, v, Int32(0))
                cell = (vv << wl2) + word
                firsts.append(
                    self.warp_base(wpre, wexl, vv >> self.fast_ewarp_log2)
                    + _lds_b32_or_zero(ok, first_s + vv * 4)
                )
                masks.append(_lds_b32_or_zero(ok, mask_s + cell * 4))
                if cutlass.const_expr(W > 1):
                    pre_rows.append(_lds_b32_or_zero(ok, pre_s + cell * 4))
            if tidx == 0:
                _st_global_b32(total_addr, num_tiles << tile_log2)
                _st_global_b32(nt_addr, num_tiles)
            outs = []
            for j in cutlass.range_constexpr(4):
                v = locs[j]
                ok = Int32(cutlass.select_(v >= 0, Int32(1), Int32(0)))
                perm = (firsts[j] << tile_log2) + _popc(masks[j] & below)
                if cutlass.const_expr(W > 1):
                    perm = perm + pre_rows[j]
                perm = cutlass.select_(v >= 0, perm, Int32(-1))
                row_val = tl
                if cutlass.const_expr(self.expanded):
                    row_val = tl * TOP_K + ((i4 & 3) * 4 + j)
                _sts_b32_if(ok, stage_s + perm * 4, row_val)
                outs.append(perm)
            if i4 < n4:
                _stg_v4_b32(e2p_addr + Int64(i4 * 16), outs[0], outs[1], outs[2], outs[3])
            self.fast_bar()

            # p2t (rows and padding) from shared memory, 16 bytes a thread, _COPY_UNROLL loads in flight; the owner
            # threads' t2e / t2l stores go out while the first loads are in flight.
            n_quads = num_tiles << (tile_log2 - 2)
            step = NW * 32
            quads = []
            for u in cutlass.range_constexpr(_COPY_UNROLL):
                i = tidx + u * step
                quads.append(
                    _lds_v4_b32_if(
                        Int32(cutlass.select_(i < n_quads, Int32(1), Int32(0))), stage_s + i * 16
                    )
                )
            if warp < EW:
                # Owner threads: every expert's first tile predicated, the further ones (more rows than a tile) in
                # one loop.
                fts = []
                ends = []
                most = Int32(0)
                for k in cutlass.range_constexpr(EPT):
                    ft = base_own + pref[k]
                    end = (ft << tile_log2) + cnts[k]
                    has = Int32(cutlass.select_(tiles[k] > 0, Int32(1), Int32(0)))
                    _st_global_b32_if(has, t2e_addr + Int64(ft * 4), tidx * EPT + k)
                    _st_global_b32_if(
                        has, t2l_addr + Int64(ft * 4), _min((ft + 1) << tile_log2, end)
                    )
                    fts.append(ft)
                    ends.append(end)
                    most = cutlass.select_(tiles[k] > most, tiles[k], most)
                for t in cutlass.range(1, most, unroll=1):
                    for k in cutlass.range_constexpr(EPT):
                        more = Int32(cutlass.select_(t < tiles[k], Int32(1), Int32(0)))
                        row = fts[k] + t
                        _st_global_b32_if(more, t2e_addr + Int64(row * 4), tidx * EPT + k)
                        _st_global_b32_if(
                            more, t2l_addr + Int64(row * 4), _min((row + 1) << tile_log2, ends[k])
                        )
            for u in cutlass.range_constexpr(_COPY_UNROLL):
                i = tidx + u * step
                q = quads[u]
                _stg_v4_b32_if(
                    Int32(cutlass.select_(i < n_quads, Int32(1), Int32(0))),
                    p2t_addr + Int64(i * 16),
                    q[0],
                    q[1],
                    q[2],
                    q[3],
                )
            for i0 in cutlass.range(
                tidx + step * _COPY_UNROLL, n_quads, step * _COPY_UNROLL, unroll=1
            ):
                quads2 = []
                for u in cutlass.range_constexpr(_COPY_UNROLL):
                    i = i0 + u * step
                    quads2.append(
                        _lds_v4_b32_if(
                            Int32(cutlass.select_(i < n_quads, Int32(1), Int32(0))),
                            stage_s + i * 16,
                        )
                    )
                for u in cutlass.range_constexpr(_COPY_UNROLL):
                    i = i0 + u * step
                    q = quads2[u]
                    _stg_v4_b32_if(
                        Int32(cutlass.select_(i < n_quads, Int32(1), Int32(0))),
                        p2t_addr + Int64(i * 16),
                        q[0],
                        q[1],
                        q[2],
                        q[3],
                    )
            if cutlass.const_expr(not self.early_release):
                _griddepcontrol_launch_dependents()

    def fast_bar(self):
        """Barrier of the working warps of the one-CTA layout."""
        _bar_sync_named(Int32(1), self.fast_warps * 32)

    def warp_base(self, wpre: list, wexl, w):
        """Tiles of the expert warps before expert warp ``w``: a select over ``wpre`` (<= 8 warps) or lane ``w``'s
        ballot-scan result ``wexl`` (a shuffle)."""
        if self.fast_ewarps <= 8:
            b = wpre[0]
            for x in range(1, self.fast_ewarps):
                b = cutlass.select_(w == x, wpre[x], b)
            return b
        return _shfl_idx_b32(wexl, w)

    @cute.jit
    def expert_scan(
        self, tidx: Int32, warp: Int32, lane: Int32, cta: Int32, tot: Int32, before: Int32, base_s: Int32,
        first_s: Int32, tot_s: Int32, wsum_s: Int32, total_addr: Int64, t2e_addr: Int64, t2l_addr: Int64,
        nt_addr: Int64, local_num: Int32, tile_log2: Int32,
    ):  # fmt: skip
        """Cluster path, thread ``e`` per local expert with its row count ``tot`` and its rows in
        earlier CTAs ``before``: the expert-order tile scan, ``base`` / ``first`` / ``tot``, this CTA's
        share (tiles ``t % C == cta``) of ``t2e`` / ``t2l``; CTA 0 the totals. Returns the tile count."""
        e = tidx
        tiles = (tot + (Int32(1) << tile_log2) - 1) >> tile_log2
        incl = _warp_inclusive_sum(tiles)
        if (lane == 31) & (warp < self.expert_warps):
            _sts_b32(wsum_s + warp * 4, incl)
        cute.arch.barrier()
        first_tile = incl - tiles
        num_tiles = Int32(0)
        for w in cutlass.range_constexpr(self.expert_warps):
            s = _lds_b32(wsum_s + w * 4)
            first_tile = first_tile + cutlass.select_(Int32(w) < warp, s, Int32(0))
            num_tiles = num_tiles + s
        if e < local_num:
            first_row = first_tile << tile_log2
            _sts_b32(base_s + e * 4, first_row + before)
            _sts_b32(first_s + e * 4, first_row)
            _sts_b32(tot_s + e * 4, tot)
            end = first_row + tot
            for t in cutlass.range(cta, tiles, self.ctas, unroll=1):
                _st_global_b32(t2e_addr + Int64((first_tile + t) * 4), e)
                _st_global_b32(
                    t2l_addr + Int64((first_tile + t) * 4),
                    _min((first_tile + t + 1) << tile_log2, end),
                )
        if (cta == 0) & (tidx == 0):
            _st_global_b32(total_addr, num_tiles << tile_log2)
            _st_global_b32(nt_addr, num_tiles)
        cute.arch.barrier()
        return num_tiles


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def max_tiles_for(rows: int, local_num: int, tile: int) -> int:
    """Tile-count bound of the runner sort: ``filled + (expanded - filled) // tile`` with
    ``filled = min(local_num, rows * 16)``."""
    expanded = rows * TOP_K
    filled = min(NUM_EXPERTS, local_num, expanded)
    return filled + (expanded - filled) // tile


def _pow2_at_least(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def plan(rows: int, local_num: int, tile: int = 8) -> tuple:
    """``(ctas, words, local_cap, per_cta, stage_rows)`` of the launch for ``rows``; raises ``ValueError``
    above :data:`CLUSTER_MAX_ROWS` rows or where the CTA's (expert, word) tables exceed the shared memory."""
    if rows > CLUSTER_MAX_ROWS:
        raise ValueError(f"rows={rows} must be at most {CLUSTER_MAX_ROWS}")
    lcap = max(64, _pow2_at_least(local_num))
    ctas = 1
    if rows > _ONE_CTA_ROWS:
        ctas = 2
        while rows > _CLUSTER_CTA_ROWS * ctas and ctas < _MAX_CLUSTER:
            ctas *= 2
    per_cta = -(-rows // ctas)
    words = _pow2_at_least(-(-per_cta // 32))
    if 8 * lcap * words > _MAX_TABLE_BYTES:
        raise ValueError(
            f"{local_num} local experts over {rows} rows exceed the {_MAX_TABLE_BYTES}-byte shared tables"
        )
    stage = 0
    if ctas == 1 and max_tiles_for(rows, local_num, tile) * tile <= _STAGE_ROWS:
        stage = _STAGE_ROWS
    return ctas, words, lcap, per_cta, stage


@functools.cache
def _compiled(
    ctas: int,
    words: int,
    local_cap: int,
    stage: int,
    expanded: bool,
    early_release: bool,
    fast: bool,
):
    kernel = RunnerSortKernel(
        ctas, words, local_cap, _THREADS, stage, expanded, early_release, fast
    )
    dummy = Int64(1 << 20)
    return cute.compile(
        kernel,
        Int32(64),
        Int32(64),
        Int32(1),
        *([dummy] * 7),
        Int32(0),
        Int32(local_cap),
        Int32(3),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def run_runner_sort(
    topk_ids: torch.Tensor,
    local_offset: int,
    local_num: int,
    tile: int,
    e2p: torch.Tensor,
    p2t: torch.Tensor,
    total: torch.Tensor,
    t2e: torch.Tensor,
    t2l: torch.Tensor,
    nt: torch.Tensor,
    expanded: bool = False,
    early_release: bool = False,
) -> None:
    """Launch the runner sort of ``topk_ids`` int32 ``[M, 16]`` (contiguous, 16-byte aligned) on the
    current stream into int32 outputs ``e2p [M * 16]`` (16-byte aligned), ``p2t [max_tiles * tile]``,
    ``total [1]``, ``t2e [max_tiles]``, ``t2l [max_tiles]``, ``nt [1]`` (``max_tiles`` =
    :func:`max_tiles_for`). Inputs are not re-validated, except that ``tile`` must be a power of two
    (the kernel tiles with shifts) and :func:`plan` must serve the shape. ``expanded``: ``p2t`` holds each
    row's expanded index ``token * 16 + slot`` instead of its token. ``early_release``: release the
    dependent grid early (:class:`RunnerSortKernel`); only for a consumer that waits for this grid before
    reading the outputs."""
    if tile < 1 or tile & (tile - 1):
        raise ValueError(f"tile={tile} must be a power of two")
    rows = topk_ids.shape[0]
    ctas, words, lcap, per_cta, stage = plan(rows, local_num, tile)
    tile_log2 = tile.bit_length() - 1
    stream = cuda.CUstream(torch.cuda.current_stream(topk_ids.device).cuda_stream)
    ptrs = [Int64(t.data_ptr()) for t in (topk_ids, e2p, p2t, total, t2e, t2l, nt)]
    grid = max_tiles_for(rows, local_num, tile) * tile // 4 if stage else ctas
    fast = ctas == 1 and stage > 0 and tile >= 8
    _compiled(ctas, words, lcap, stage, bool(expanded), bool(early_release), fast)(
        Int32(rows), Int32(per_cta), Int32(grid), *ptrs, Int32(local_offset), Int32(local_num),
        Int32(tile_log2), stream,
    )  # fmt: skip


def check_inputs(topk_ids: torch.Tensor, local_offset: int, local_num: int, tile: int) -> None:
    """Raise ``ValueError`` for inputs the kernel does not serve."""
    if topk_ids.dim() != 2 or topk_ids.shape[1] != TOP_K or topk_ids.dtype != torch.int32:
        raise ValueError(f"topk_ids must be int32 [M, {TOP_K}]")
    if not topk_ids.is_contiguous() or topk_ids.data_ptr() % 16:
        raise ValueError("topk_ids must be contiguous and 16-byte aligned")
    if not 1 <= topk_ids.shape[0] <= CLUSTER_MAX_ROWS:
        raise ValueError(f"rows={topk_ids.shape[0]} must be in [1, {CLUSTER_MAX_ROWS}]")
    if (
        not 0 <= local_num <= MAX_LOCAL
        or local_offset < 0
        or local_offset + local_num > NUM_EXPERTS
    ):
        raise ValueError(f"local experts [{local_offset}, {local_offset + local_num}) out of range")
    if tile < 8 or tile > 256 or tile & (tile - 1):
        raise ValueError(f"tile={tile} must be a power of two in [8, 256]")


def runner_sort(
    topk_ids: torch.Tensor,
    local_offset: int,
    local_num: int,
    tile: int,
    early_release: bool = False,
) -> list[torch.Tensor]:
    """Allocate the outputs and run :func:`run_runner_sort`; returns ``[e2p, p2t, total, t2e, t2l,
    nt]`` (the order of the W4A8 MoE op's ``prerouted``)."""
    check_inputs(topk_ids, local_offset, local_num, tile)
    rows = topk_ids.shape[0]
    max_tiles = max_tiles_for(rows, local_num, tile)
    i32 = dict(dtype=torch.int32, device=topk_ids.device)
    outs = [
        torch.empty(rows * TOP_K, **i32),
        torch.empty(max_tiles * tile, **i32),
        torch.empty(1, **i32),
        torch.empty(max_tiles, **i32),
        torch.empty(max_tiles, **i32),
        torch.empty(1, **i32),
    ]
    run_runner_sort(topk_ids, local_offset, local_num, tile, *outs, early_release=early_release)
    return outs


__all__ = [
    "CLUSTER_MAX_ROWS",
    "MAX_LOCAL",
    "RunnerSortKernel",
    "check_inputs",
    "runner_sort",
    "max_tiles_for",
    "plan",
    "run_runner_sort",
]

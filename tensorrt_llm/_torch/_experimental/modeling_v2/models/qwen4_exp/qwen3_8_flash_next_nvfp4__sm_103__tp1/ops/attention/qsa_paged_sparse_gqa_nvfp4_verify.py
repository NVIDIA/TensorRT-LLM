# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA over an NVFP4 paged K/V pool for MTP verify rows, request-grouped: CuTe DSL
(tcgen05) split-K, then `qsa_paged_sparse_gqa_nvfp4_decode`'s merge.

Tier 5 (CuTe DSL written for this target). A verify step's rows come `T`
(`rows_per_request`) consecutive rows a request, and the rows of one request select mostly
the same tokens. One CTA serves one (group of T rows, KV head, split): 128 MMA rows, row
m = t * 12 + h (verify row t, q head h of the KV head; rows past T * 12 idle), over the
group's *union* selection, so each selected K/V token is loaded and dequantized once for the
group's rows instead of once a row.

Union (every thread, one code copy per pass): per-row token bitmaps in shared memory (the
request's block-table row fetched in the same round trip), the union of four-token groups in
ascending order (an expanded selection keeps column % 4 == token % 4), each group's K/V row
offset and scale offset resolved; per tile a 64-bit selection word for each verify row. A
(row, position) pair the row did not select is masked to -inf before its max, so the softmax
offset is the row's own max and causality is the selection's. Splits take interleaved tiles of
the union. Rows of a group that belong to different requests (a partial or misaligned group)
are served in separate passes; rows of an invalid request give zeros.

Roles (11 warps):

* warps 5-6 `cp.async` each tile's raw NVFP4 K / V rows (eight threads a 128-byte row), K scales
  and V-scale groups into a ring of `RAW_SLOTS` raw slots, completing on the slot's mbarrier;
* warps 7-10 (one thread a position, K then V) load a raw row and its scales into registers,
  release the raw slot, and dequantize into f16 SW128 K and V slots (`cvt.rn.f16x2.e2m1x2` x
  `cvt.rn.f16x2.e4m3x2`, exact, as the per-row kernel);
* warp 4 issues the tcgen05 MMAs with precomputed descriptors (`qsa_nvfp4_verify_ptx`):
  S[j % 2] = Q K(j)^T (Q from tensor memory), then O += P(j - 1) V(j - 1) (P from tensor
  memory, over the S it came from);
* warps 0-3 (one thread an MMA row) copy their q row into a raw slot at once and into tensor
  memory (f16) after the first resolve; per tile load S, take the max over the row's own
  entries, move the row's reference max only when it grew past `RESCALE_TH` (log2), rescaling
  O in tensor memory then, and write P = 2^(s - max) as f16 (0 off the row's selection);
  finally normalize O (times the V global scale), optionally gate it, and store it (or an fp32
  partial and its log2-sum-exp for the merge).

Tensor memory: O in columns 0-255, S/P buffers at 256 and 320, Q at 384. Shared memory: two f16
K and two f16 V slots (32 KB each), three raw slots, the union tables (up to ~225 KB): one CTA
an SM.
"""

from __future__ import annotations

import functools
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import const_expr
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32, Int64

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from .qsa_nvfp4_verify_ptx import (
    ctz,
    gemm_ts,
    groups8,
    popc,
    red_or_shared,
    spread4,
    stg4,
    stg4f,
    tcgen05_fence_after,
    tcgen05_fence_before,
)
from .qsa_paged_sparse_gqa_nvfp4_decode import (
    MAX_SPLITS,
    _bf2_hi,
    _bf2_lo,
    _bf2_pack,
    _bf2_to_f2,
    _bits_f32,
    _cp16z,
    _dq_word,
    _f32_bits,
    _ldg1,
    _ldg4,
    _lds1,
    _lds4,
    _MergeKernel,
    _ptr,
    _round_bf16,
    _scale2s,
    _sigmoid,
    _stg1,
    _sts1,
    _sts4,
    nvfp4_sparse_gqa_decode,
    qsa_paged_sparse_gqa_nvfp4_decode,
)

try:
    from cutlass.memory import SmemAllocator
except ImportError:  # DSL releases before the memory namespace split
    from cutlass.utils import SmemAllocator

F16 = cutlass.Float16
LOG2E = 1.4426950408889634
HEAD_DIM = 256
#: MMA rows a CTA (T verify rows x the KV head's q heads) and positions a tile.
BM = 128
BN = 64
#: Four-token groups a tile.
GPT = BN // 4
#: Bytes of a raw K or V row in shared memory (128 + 16 of padding: conflict-free reads).
RAW_LD = 144
RAW_SLOTS = 3
RAW_K = BN * RAW_LD
RAW_KS = BN * 16
RAW_VS = GPT * 64
#: One raw slot: K rows, V rows, K-scale rows, V-scale groups.
RAW_BUF = 2 * RAW_K + RAW_KS + RAW_VS
#: Bytes of the four f16 K / V slots (two each, 64 positions x 256 dims).
KV_SLOTS_BYTES = 4 * BN * HEAD_DIM * 2
#: Shared memory a CTA may claim (227 KB), less barriers, scan scratch and alignment.
SMEM_BUDGET = 232448 - 4096
#: A row's reference max moves only past this many log2 units (P <= 2^8, exact in f16 range).
RESCALE_TH = 8.0
SOFT_WARPS = 4
MMA_WARP = 4
LOAD_WARP0 = 5
LOAD_WARPS = 2
DQ_WARP0 = 7
DQ_WARPS = 4
THREADS = 32 * (DQ_WARP0 + DQ_WARPS)
TMEM_O = 0
TMEM_S0 = 256
TMEM_Q = 384
TMEM_COLS = 512
CHUNK_COLS = 32
#: Bytes of a staged q row (bf16, 512 + 16 of padding).
Q_LD = 528
#: Most splits of one (group, KV head) union.
MAX_PLAN_SPLITS = 32
#: Fewer request groups than this take the per-row kernel (GB300: it is as fast or faster
#: below 32 rows at four rows a request, the grouped kernel from 32 rows up).
MIN_GROUPS = 8


class _Layout(NamedTuple):
    """The shared-memory plan of one compile: bitmap words (`ngw` groups of four), the union's
    group capacity, local tiles a CTA may own in one pass, union-scratch offsets in raw slot 0
    and the table size in int32 words."""

    ngw: int
    nw: int
    ug_max: int
    tmax: int
    u_ug: int
    u_bt: int
    tab_words: int


def _layout(rpr: int, group: int, width: int, bt_width: int, tpb: int, splits: int):
    """The kernel's shared-memory plan for `rpr` rows a group over a `width`-column expanded
    selection and a `bt_width`-page block table, or None where it does not fit: the union
    scratch (each row's token bitmap, the union's group list and the request's block-table
    row) shares raw slot 0, the staged q rows raw slots 1-2."""
    if tpb % 4 or tpb & (tpb - 1) or rpr < 1 or rpr * group > BM:
        return None
    ngw = -(-(bt_width * tpb) // 128)
    nw = 4 * ngw
    # an expanded selection names at most ceil(width / 4) four-token groups a row
    ug_max = min(rpr * -(-width // 4), 32 * ngw)
    tmax = -(-(-(-ug_max // GPT)) // splits)
    u_ug = -(-(rpr * nw * 4) // 16) * 16
    u_bt = -(-(u_ug + ug_max * 4) // 16) * 16
    tab_words = 3 * tmax * GPT + 2 * tmax * rpr
    fits = (
        ngw <= THREADS
        and u_bt + bt_width * 4 <= RAW_BUF
        and RAW_BUF + rpr * group * Q_LD <= RAW_SLOTS * RAW_BUF
        and 4 * tab_words + RAW_SLOTS * RAW_BUF + KV_SLOTS_BYTES <= SMEM_BUDGET
    )
    return _Layout(ngw, nw, ug_max, tmax, u_ug, u_bt, tab_words) if fits else None


def _phase(g, n):
    """Parity of use `g` of a ring of `n` slots."""
    return (g // n) % 2


class _VerifyKernel:
    def __init__(
        self,
        *,
        rpr,
        group,
        kv_heads,
        width,
        tpb,
        splits,
        scale,
        num_pages,
        bt_width,
        num_requests,
        k_strides,
        ks_strides,
        q_strides,
        req64,
        gated,
        pdl,
    ):
        lay = _layout(rpr, group, width, bt_width, tpb, splits)
        assert lay is not None, "no shared-memory plan for this geometry (see `_layout`)"
        self.T = rpr
        self.group, self.kvh, self.width, self.tpb = group, kv_heads, width, tpb
        self.splits = splits
        self.scale_log2 = scale * LOG2E
        self.num_pages, self.bt_width, self.num_requests = num_pages, bt_width, num_requests
        self.ks, self.kss, self.qs = k_strides, ks_strides, q_strides
        self.req64, self.gated, self.pdl = req64, gated, pdl
        self.heads = group * kv_heads
        self.rows_m = rpr * group
        #: softmax warps holding live MMA rows (the rest only keep the barriers' counts)
        self.live_warps = -(-self.rows_m // 32)
        self.ngw, self.nw, self.ug_max, self.tmax = lay.ngw, lay.nw, lay.ug_max, lay.tmax
        self.raw_k = RAW_K
        self.raw_v = RAW_K
        self.raw_ks = RAW_KS
        self.raw_vs = RAW_VS
        self.raw_buf = RAW_BUF
        self.raw_bytes = RAW_SLOTS * RAW_BUF
        # union scratch (raw slot 0, free between passes): bitmaps, group list, block-table row
        self.u_bm = 0
        self.u_ug = lay.u_ug
        self.u_bt = lay.u_bt
        # q rows (bf16, 528-byte rows: conflict-free) in raw slots 1-2 until Q is in tensor memory
        self.u_q = RAW_BUF
        # tables (int32 words). Per group: K/V row offset and scale offset of its first token
        # (16-byte units, -1 when the group does not resolve) and its (token, row) selection
        # bytes; per (tile, row): the 64-bit selection word.
        self.t_roff = 0
        self.t_soff = self.tmax * GPT
        self.t_gmask = 2 * self.tmax * GPT
        self.t_bits = 3 * self.tmax * GPT
        self.tab_words = lay.tab_words

    @cute.jit
    def __call__(
        self,
        sel_t: cute.Tensor,
        q_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        v_ptr: cute.Pointer,
        ks_ptr: cute.Pointer,
        vs_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        bt_ptr: cute.Pointer,
        req_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        lse_ptr: cute.Pointer,
        gate_ptr: cute.Pointer,
        rows: cutlass.Int32,
        gate_stride: cutlass.Int32,
        stream,
    ):
        def qk():
            return sm100_utils.make_trivial_tiled_mma(
                F16,
                F16,
                OperandMajorMode.K,
                OperandMajorMode.K,
                Float32,
                tcgen05.CtaGroup.ONE,
                (BM, BN),
                tcgen05.OperandSource.TMEM,
            )

        def pv():
            return sm100_utils.make_trivial_tiled_mma(
                F16,
                F16,
                OperandMajorMode.K,
                OperandMajorMode.MN,
                Float32,
                tcgen05.CtaGroup.ONE,
                (BM, HEAD_DIM),
                tcgen05.OperandSource.TMEM,
            )

        qk_mma, qk_mma_acc, pv_mma_acc = qk(), qk(), pv()
        q_layout = sm100_utils.make_smem_layout_a(qk_mma, (BM, BN, HEAD_DIM), F16, 1)
        k_layout = sm100_utils.make_smem_layout_b(qk_mma, (BM, BN, HEAD_DIM), F16, 2)
        v_layout = sm100_utils.make_smem_layout_b(pv_mma_acc, (BM, HEAD_DIM, BN), F16, 2)
        p_layout = sm100_utils.make_smem_layout_a(pv_mma_acc, (BM, HEAD_DIM, BN), F16, 1)

        @cute.struct
        class SharedStorage:
            q_full: cute.struct.MemRange[Int64, 1]
            raw_full: cute.struct.MemRange[Int64, RAW_SLOTS]
            raw_empty: cute.struct.MemRange[Int64, RAW_SLOTS]
            k_full: cute.struct.MemRange[Int64, 2]
            k_empty: cute.struct.MemRange[Int64, 2]
            v_full: cute.struct.MemRange[Int64, 2]
            v_empty: cute.struct.MemRange[Int64, 2]
            s_full: cute.struct.MemRange[Int64, 2]
            p_full: cute.struct.MemRange[Int64, 2]
            o_full: cute.struct.MemRange[Int64, 1]
            o_final: cute.struct.MemRange[Int64, 1]
            tmem_holding: Int32
            scan: cute.struct.MemRange[Int32, 32]
            tab: cute.struct.Align[cute.struct.MemRange[Int32, self.tab_words], 16]
            raw: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, self.raw_bytes], 128]
            sK: cute.struct.Align[cute.struct.MemRange[F16, cute.cosize(k_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[F16, cute.cosize(k_layout)], 1024]

        self.shared_storage = SharedStorage
        bt = cute.make_tensor(
            bt_ptr, cute.make_layout((self.num_requests, self.bt_width), stride=(self.bt_width, 1))
        )
        req = cute.make_tensor(req_ptr, cute.make_layout((rows,), stride=(1,)))
        gs = cute.make_tensor(gs_ptr, cute.make_layout((3,), stride=(1,)))
        n_groups = (rows + self.T - 1) // self.T
        self.kernel(
            qk_mma,
            qk_mma_acc,
            pv_mma_acc,
            q_layout,
            k_layout,
            v_layout,
            p_layout,
            sel_t.iterator.toint(),
            q_ptr.toint(),
            k_ptr.toint(),
            v_ptr.toint(),
            ks_ptr.toint(),
            vs_ptr.toint(),
            gs,
            bt,
            req,
            out_ptr.toint(),
            lse_ptr.toint(),
            gate_ptr.toint(),
            rows,
            gate_stride,
        ).launch(
            grid=[n_groups * self.kvh * self.splits, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.pdl,
        )

    # ---- pieces every role runs (every thread of the CTA, barrier 0) -----------------------

    @cute.jit
    def _pass_rows(self, rq: cute.Tensor, handled):
        lead = cutlass.Int32(-1)
        for t in tuple(range(self.T - 1, -1, -1)):
            lead = cutlass.Int32(cutlass.select_(((handled >> t) & 1) == 0, rq[t], lead))
        act = cutlass.Int32(0)
        for t in cutlass.range_constexpr(self.T):
            act = act | cutlass.Int32(
                cutlass.select_(
                    (((handled >> t) & 1) == 0) & (rq[t] == lead),
                    cutlass.Int32(1 << t),
                    cutlass.Int32(0),
                )
            )
        return lead, act

    @cute.jit
    def _union(
        self, act, lead, r0, rows, sel_base, tidx, warp, lane, s_bm, s_scan, s_ug, s_bt, mBT
    ):
        rpr: cutlass.Constexpr = self.T
        nw: cutlass.Constexpr = self.nw
        ngw: cutlass.Constexpr = self.ngw
        width: cutlass.Constexpr = self.width
        ntok: cutlass.Constexpr = 32 * self.nw
        n_q: cutlass.Constexpr = -(-width // 4)  # four-column quads a row
        n_i: cutlass.Constexpr = -(-n_q // THREADS)
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(-(-(rpr * nw) // THREADS)):
            idx = tidx + i * THREADS
            if idx < rpr * nw:
                _sts1(s_bm + idx * 4, cutlass.Int32(0))
        # thread: columns 4q .. 4q + 3 of each row (one selected group in an expanded selection);
        # every load first, then one bit set a distinct word
        toks = []
        for t in cutlass.range_constexpr(rpr):
            row_sel = sel_base + cutlass.Int64(cutlass.min(r0 + t, rows - 1)) * (width * 4)
            for i in cutlass.range_constexpr(n_i):
                for u in cutlass.range_constexpr(4):
                    colc = cutlass.min(4 * (tidx + i * THREADS) + u, width - 1)
                    toks.append(_ldg1(row_sel + cutlass.Int64(colc * 4)))
        # the request's block-table row, in the same round trip
        n_bt: cutlass.Constexpr = -(-self.bt_width // THREADS)
        btv = [mBT[lead, cutlass.min(tidx + i * THREADS, self.bt_width - 1)] for i in range(n_bt)]
        cute.arch.sync_threads()
        for t in cutlass.range_constexpr(rpr):
            for i in cutlass.range_constexpr(n_i):
                q4 = tidx + i * THREADS
                live_t = ((act >> t) & 1) != 0
                okv = []
                wds = []
                bv = []
                for u in cutlass.range_constexpr(4):
                    tok = toks[(t * n_i + i) * 4 + u]
                    okv.append(live_t & (4 * q4 + u < width) & (tok >= 0) & (tok < ntok))
                    wds.append(tok >> 5)
                    bv.append(
                        cutlass.Int32(
                            cutlass.select_(
                                okv[u], cutlass.Int32(1) << (tok & 31), cutlass.Int32(0)
                            )
                        )
                    )
                # bits sharing token 0's word go in one atomic; any other token on its own
                m0 = bv[0]
                for u in cutlass.range_constexpr(1, 4):
                    m0 = m0 | cutlass.Int32(
                        cutlass.select_(wds[u] == wds[0], bv[u], cutlass.Int32(0))
                    )
                if okv[0] | (m0 != 0):
                    if m0 != 0:
                        red_or_shared(s_bm + (t * nw + cutlass.max(wds[0], 0)) * 4, m0)
                for u in cutlass.range_constexpr(1, 4):
                    if okv[u] & (wds[u] != wds[0]):
                        red_or_shared(s_bm + (t * nw + wds[u]) * 4, bv[u])
        for i in cutlass.range_constexpr(n_bt):
            if tidx + i * THREADS < self.bt_width:
                _sts1(s_bt + (tidx + i * THREADS) * 4, btv[i])
        cute.arch.sync_threads()
        gmv = cutlass.Int32(0)
        if tidx < ngw:
            for k in cutlass.range_constexpr(4):
                xo = cutlass.Int32(0)
                for t in cutlass.range_constexpr(rpr):
                    xo = xo | _lds1(s_bm + (t * nw + 4 * tidx + k) * 4)
                gmv = gmv | (groups8(xo) << (8 * k))
        cnt = popc(gmv)
        incl = cnt
        for o in (1, 2, 4, 8, 16):
            up = cute.arch.shuffle_sync_up(incl, offset=o)
            if lane >= o:
                incl = incl + up
        if lane == 31:
            _sts1(s_scan + warp * 4, incl)
        cute.arch.sync_threads()
        wsum = cutlass.Int32(0)
        tot = cutlass.Int32(0)
        if lane < THREADS // 32:
            wv = _lds1(s_scan + lane * 4)
            tot = wv
            if lane < warp:
                wsum = wv
        before = cute.arch.warp_reduction_sum(wsum)
        n_ug = cute.arch.warp_reduction_sum(tot)
        excl = before + incl - cnt
        if tidx < ngw:
            xm = gmv
            for j in cutlass.range(cnt):
                b = ctz(xm)
                if excl + j < self.ug_max:
                    _sts1(s_ug + (excl + j) * 4, 32 * tidx + b)
                xm = xm & (xm - 1)
        cute.arch.sync_threads()
        return cutlass.min(n_ug, self.ug_max)

    @cute.jit
    def _resolve(self, lead, kvh, split, n_loc, n_ug, tidx, s_bm, s_ug, s_tab, s_bt):
        """Every local tile's groups (entry of the group's first token, (token, row) selection
        bytes), then each (tile, verify row)'s 64-bit selection word."""
        rpr: cutlass.Constexpr = self.T
        nw: cutlass.Constexpr = self.nw
        splits: cutlass.Constexpr = self.splits
        tmax: cutlass.Constexpr = self.tmax
        n_rs: cutlass.Constexpr = -(-(tmax * GPT) // THREADS)
        s_roff = s_tab + self.t_roff * 4
        s_soff = s_tab + self.t_soff * 4
        s_gmask = s_tab + self.t_gmask * 4
        s_bits = s_tab + self.t_bits * 4
        ks0, ks1, ks2 = self.ks
        kss0, kss1 = self.kss
        pages = []
        gids = []
        for i in cutlass.range_constexpr(n_rs):
            slot = tidx + i * THREADS
            kl = slot // GPT
            ugi = (split + kl * splits) * GPT + slot % GPT
            gid = _lds1(s_ug + cutlass.min(ugi, self.ug_max - 1) * 4)
            gid = cutlass.min(cutlass.max(gid, 0), 8 * nw - 1)
            gids.append(gid)
            pages.append(_lds1(s_bt + cutlass.min((gid * 4) // self.tpb, self.bt_width - 1) * 4))
        for i in cutlass.range_constexpr(n_rs):
            slot = tidx + i * THREADS
            kl = slot // GPT
            ugi = (split + kl * splits) * GPT + slot % GPT
            gid = gids[i]
            page = pages[i]
            wsel = gid >> 3
            sh = (gid & 7) * 4
            mw = cutlass.Int32(0)
            for t in cutlass.range_constexpr(rpr):
                nib = (_lds1(s_bm + (t * nw + wsel) * 4) >> sh) & 0xF
                mw = mw | (spread4(nib) << t)
            okp = (kl < n_loc) & (ugi < n_ug) & (page >= 0) & (page < self.num_pages)
            pg64 = cutlass.Int64(cutlass.max(page, 0))
            sl0 = (gid * 4) % self.tpb
            roff = (pg64 * ks0 + cutlass.Int64(kvh * ks1 + sl0 * ks2)) >> 4
            soff = (pg64 * kss0 + cutlass.Int64(kvh * kss1 + sl0 * 16)) >> 4
            roff = cutlass.Int32(cutlass.select_(okp, cutlass.Int32(roff), cutlass.Int32(-1)))
            soff = cutlass.Int32(cutlass.select_(okp, cutlass.Int32(soff), cutlass.Int32(-1)))
            mw = cutlass.Int32(cutlass.select_(okp, mw, cutlass.Int32(0)))
            if slot < tmax * GPT:
                _sts1(s_roff + slot * 4, roff)
                _sts1(s_soff + slot * 4, soff)
                _sts1(s_gmask + slot * 4, mw)
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(-(-(tmax * rpr) // THREADS)):
            w = tidx + i * THREADS
            if w < tmax * rpr:
                kl = w // rpr
                t = w % rpr
                lo = cutlass.Int32(0)
                hi = cutlass.Int32(0)
                for gq in cutlass.range_constexpr(GPT):
                    nb = (_lds1(s_gmask + (kl * GPT + gq) * 4) >> t) & 0x01010101
                    nb = (nb | (nb >> 7) | (nb >> 14) | (nb >> 21)) & 0xF
                    if const_expr(gq < 8):
                        lo = lo | (nb << (4 * gq))
                    else:
                        hi = hi | (nb << (4 * (gq - 8)))
                _sts1(s_bits + w * 8, lo)
                _sts1(s_bits + w * 8 + 4, hi)
        cute.arch.sync_threads()

    @cute.jit
    def _pass_tiles(
        self, act, lead, kvh, r0, rows, sel_base, split, tidx, warp, lane, s_raw, s_scan, s_tab, mBT
    ):
        """Union + resolve of one pass; returns the CTA's local tile count."""
        s_bm = s_raw + self.u_bm
        s_ug = s_raw + self.u_ug
        s_bt = s_raw + self.u_bt
        n_ug = self._union(
            act, lead, r0, rows, sel_base, tidx, warp, lane, s_bm, s_scan, s_ug, s_bt, mBT
        )
        n_tiles = (n_ug + GPT - 1) // GPT
        n_loc = cutlass.Int32(
            cutlass.select_(
                split < n_tiles,
                (n_tiles - split + self.splits - 1) // self.splits,
                cutlass.Int32(0),
            )
        )
        self._resolve(lead, kvh, split, n_loc, n_ug, tidx, s_bm, s_ug, s_tab, s_bt)
        return n_loc

    @cute.kernel
    def kernel(
        self,
        qk_mma: cute.TiledMma,
        qk_mma_acc: cute.TiledMma,
        pv_mma_acc: cute.TiledMma,
        q_layout: cute.ComposedLayout,
        k_layout: cute.ComposedLayout,
        v_layout: cute.ComposedLayout,
        p_layout: cute.ComposedLayout,
        sel_base: cutlass.Int64,
        q_base: cutlass.Int64,
        k_base: cutlass.Int64,
        v_base: cutlass.Int64,
        ks_base: cutlass.Int64,
        vs_base: cutlass.Int64,
        mGs: cute.Tensor,
        mBT: cute.Tensor,
        mReq: cute.Tensor,
        out_base: cutlass.Int64,
        lse_base: cutlass.Int64,
        gate_base: cutlass.Int64,
        rows: cutlass.Int32,
        gate_stride: cutlass.Int32,
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        qk_mma.set(tcgen05.Field.ACCUMULATE, False)
        qk_mma_acc.set(tcgen05.Field.ACCUMULATE, True)
        pv_mma_acc.set(tcgen05.Field.ACCUMULATE, True)
        rpr: cutlass.Constexpr = self.T
        group: cutlass.Constexpr = self.group
        kvh_n: cutlass.Constexpr = self.kvh
        splits: cutlass.Constexpr = self.splits
        gi = bidx // (kvh_n * splits)
        kvh = (bidx // splits) % kvh_n
        split = bidx % splits
        r0 = gi * rpr

        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        q_full = storage.q_full.data_ptr()
        raw_full = storage.raw_full.data_ptr()
        raw_empty = storage.raw_empty.data_ptr()
        k_full = storage.k_full.data_ptr()
        k_empty = storage.k_empty.data_ptr()
        v_full = storage.v_full.data_ptr()
        v_empty = storage.v_empty.data_ptr()
        s_full = storage.s_full.data_ptr()
        p_full = storage.p_full.data_ptr()
        o_full = storage.o_full.data_ptr()
        o_final = storage.o_final.data_ptr()
        s_scan = cutlass.Int32(storage.scan.data_ptr().toint())
        s_tab = cutlass.Int32(storage.tab.data_ptr().toint())
        s_raw = cutlass.Int32(storage.raw.data_ptr().toint())
        if tidx == 0:
            cute.arch.mbarrier_init(q_full, 32 * SOFT_WARPS)
            for i in cutlass.range_constexpr(RAW_SLOTS):
                cute.arch.mbarrier_init(raw_full + i, 32 * LOAD_WARPS)
                cute.arch.mbarrier_init(raw_empty + i, 32 * DQ_WARPS)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(k_full + i, BN)
                cute.arch.mbarrier_init(k_empty + i, 1)
                cute.arch.mbarrier_init(v_full + i, BN)
                cute.arch.mbarrier_init(v_empty + i, 1)
                cute.arch.mbarrier_init(s_full + i, 1)
                cute.arch.mbarrier_init(p_full + i, 32 * SOFT_WARPS)
            cute.arch.mbarrier_init(o_full, 1)
            cute.arch.mbarrier_init(o_final, 1)
        cute.arch.mbarrier_init_fence()
        if warp == MMA_WARP:
            cute.arch.alloc_tmem(TMEM_COLS, storage.tmem_holding.ptr)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, storage.tmem_holding.ptr)

        qk_thr = qk_mma.get_slice(0)
        pv_thr = pv_mma_acc.get_slice(0)
        sK = storage.sK.get_tensor(k_layout.outer, swizzle=k_layout.inner)
        sVr = storage.sV.get_tensor(k_layout.outer, swizzle=k_layout.inner)
        sV = cute.make_tensor(cute.recast_ptr(sVr.iterator, v_layout.inner), v_layout.outer)
        s_k16 = cutlass.Int32(storage.sK.data_ptr().toint())
        s_v16 = cutlass.Int32(storage.sV.data_ptr().toint())

        # ---- softmax-side tensor-memory views (warps 0-3: one thread an MMA row) -------------
        st = tidx
        tStS = qk_thr.make_fragment_C(qk_thr.partition_shape_C((BM, BN)))
        tStS0 = cute.make_tensor(tmem_ptr + TMEM_S0, tStS.layout)
        tOtO_f = pv_thr.make_fragment_C(pv_thr.partition_shape_C((BM, HEAD_DIM)))
        tOtO = cute.make_tensor(tmem_ptr + TMEM_O, tOtO_f.layout)
        tScS = qk_thr.partition_C(cute.make_identity_tensor((BM, BN)))
        load_s = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32),
            tStS0,
        )
        thr_load_s = load_s.get_slice(st)
        tLtS0 = thr_load_s.partition_S(tStS0)
        tLcS = thr_load_s.partition_D(tScS)
        p_cols = BN // 2
        tStP0 = cute.make_tensor(
            tStS0.iterator, cute.composition(tStS0.layout, cute.make_layout((BM, p_cols)))
        )
        tScP = cute.make_tensor(
            tScS.iterator, cute.composition(tScS.layout, cute.make_layout((BM, p_cols)))
        )
        store_p = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(p_cols)), Float32),
            tStP0,
        )
        thr_store_p = store_p.get_slice(st)
        tPtP0 = thr_store_p.partition_D(tStP0)
        tPcP = thr_store_p.partition_S(tScP)
        tOcO = pv_thr.partition_C(cute.make_identity_tensor((BM, HEAD_DIM)))
        tOtO_c = cute.make_tensor(
            tOtO.iterator, cute.composition(tOtO.layout, cute.make_layout((BM, CHUNK_COLS)))
        )
        tOcO_c = cute.make_tensor(
            tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((BM, CHUNK_COLS)))
        )
        load_o = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(CHUNK_COLS)), Float32
            ),
            tOtO_c,
        )
        store_o = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(CHUNK_COLS)), Float32
            ),
            tOtO_c,
        )
        thr_load_o = load_o.get_slice(st)
        thr_store_o = store_o.get_slice(st)
        tLtO = thr_load_o.partition_S(tOtO_c)
        tLcO = thr_load_o.partition_D(tOcO_c)
        tStO = thr_store_o.partition_D(tOtO_c)
        tview = (load_s, tLtS0, tLcS, store_p, tPtP0, tPcP, load_o, store_o, tLtO, tLcO, tStO)

        m = tidx
        t_row = m // group
        h = m % group
        live = (m < self.rows_m) & (r0 + t_row < rows)
        safe_row = cutlass.min(r0 + t_row, rows - 1)
        qs0, qs1 = self.qs
        q_row = q_base + (cutlass.Int64(safe_row) * qs0 + cutlass.Int64(kvh * group + h) * qs1) * 2
        q_s = s_raw + self.u_q + cutlass.min(m, self.rows_m - 1) * Q_LD

        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        if warp < SOFT_WARPS:
            # O = 0; this row's q copied into raw slots 1-2 now, into tensor memory (f16) after the
            # first pass's resolve. Dead rows stage zeros.
            rZ = cute.make_rmem_tensor(tLcO.shape, Float32)
            rZ.fill(0.0)
            for c in cutlass.range_constexpr(HEAD_DIM // CHUNK_COLS):
                cute.copy(
                    store_o, rZ, cute.make_tensor(tStO.iterator + c * CHUNK_COLS, tStO.layout)
                )
            cute.arch.fence_view_async_tmem_store()
            if m < self.rows_m:
                qn = cutlass.Int32(cutlass.select_(live, cutlass.Int32(16), cutlass.Int32(0)))
                for i in cutlass.range_constexpr(HEAD_DIM // 8):
                    _cp16z(q_s + 16 * i, q_row + cutlass.Int64(16 * i), qn)
            cute.arch.cp_async_commit_group()

        rq = cute.make_rmem_tensor(cute.make_layout((rpr,), stride=(1,)), cutlass.Int32)
        handled = cutlass.Int32(0)
        for t in cutlass.range_constexpr(rpr):
            row = r0 + t
            xr = cutlass.Int64(mReq[cutlass.min(row, rows - 1)])
            okr = (row < rows) & (xr >= 0) & (xr < self.num_requests)
            rq[t] = cutlass.Int32(cutlass.select_(okr, cutlass.Int32(xr), cutlass.Int32(-1)))
            handled = handled | cutlass.Int32(
                cutlass.select_(okr, cutlass.Int32(0), cutlass.Int32(1 << t))
            )
        full = (1 << rpr) - 1
        scale_log2 = Float32(self.scale_log2) * mGs[1]
        neg_inf = -Float32.inf
        row_max = neg_inf
        row_sum = Float32(0.0)
        q_pending = cutlass.Int32(1)
        g = cutlass.Int32(0)
        for _pass in cutlass.range(rpr):
            if handled != full:
                lead, act = self._pass_rows(rq, handled)
                n_loc = self._pass_tiles(
                    act,
                    lead,
                    kvh,
                    r0,
                    rows,
                    sel_base,
                    split,
                    tidx,
                    warp,
                    lane,
                    s_raw,
                    s_scan,
                    s_tab,
                    mBT,
                )
                if warp >= DQ_WARP0:
                    self.dequant_tiles(
                        tidx - 32 * DQ_WARP0,
                        g,
                        n_loc,
                        s_raw,
                        s_k16,
                        s_v16,
                        raw_full,
                        raw_empty,
                        k_full,
                        k_empty,
                        v_full,
                        v_empty,
                    )
                elif warp >= LOAD_WARP0:
                    self.load_tiles(
                        tidx - 32 * LOAD_WARP0,
                        g,
                        n_loc,
                        s_raw,
                        s_tab,
                        k_base,
                        v_base,
                        ks_base,
                        vs_base,
                        raw_full,
                        raw_empty,
                        q_full,
                    )
                elif warp == MMA_WARP:
                    self.mma_tiles(
                        g,
                        n_loc,
                        qk_mma,
                        pv_mma_acc,
                        qk_thr,
                        pv_thr,
                        q_layout,
                        p_layout,
                        sK,
                        sV,
                        tmem_ptr,
                        q_full,
                        k_full,
                        k_empty,
                        v_full,
                        v_empty,
                        s_full,
                        p_full,
                        o_full,
                    )
                else:
                    if q_pending != 0:
                        self.stage_q(q_s, live, tview, q_full)
                        q_pending = cutlass.Int32(0)
                    row_max, row_sum = self.softmax_tiles(
                        g,
                        n_loc,
                        row_max,
                        row_sum,
                        live,
                        t_row,
                        warp,
                        s_tab,
                        scale_log2,
                        tview,
                        s_full,
                        p_full,
                        o_full,
                    )
                g = g + n_loc
                cute.arch.sync_threads()  # [P] end of the pass
                handled = handled | act

        if warp == MMA_WARP:
            with cute.arch.elect_one():
                tcgen05.commit(o_final)
        if warp < SOFT_WARPS:
            cute.arch.cp_async_wait_group(0)
            self.epilogue(
                row_max,
                row_sum,
                live,
                warp,
                m,
                r0,
                rows,
                kvh,
                split,
                mGs,
                scale_log2,
                out_base,
                lse_base,
                gate_base,
                gate_stride,
                tview,
                o_final,
            )
        # every role done with tensor memory (the softmax warps read O last)
        cute.arch.sync_threads()
        if warp == MMA_WARP:
            cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS)

    # ------------------------------------------------------------------ loaders (warps 5-6)
    @cute.jit
    def load_tiles(
        self,
        lt,
        g,
        n_loc,
        s_raw,
        s_tab,
        k_base,
        v_base,
        ks_base,
        vs_base,
        raw_full,
        raw_empty,
        q_full,
    ):
        """Raw tile copies: K and V rows eight threads a row (16-byte chunk lt % 8 of rows
        lt // 8 + 8 i: whole 128-byte lines a warp instruction); the K-scale row lt and chunk
        lt % 4 of the V-scale group lt // 4 (both at the group's scale offset + lt % 4). Raw slots
        1-2 hold the q rows until Q is in tensor memory."""
        s_roff = s_tab + self.t_roff * 4
        s_soff = s_tab + self.t_soff * 4
        s_gmask = s_tab + self.t_gmask * 4
        tok16: cutlass.Constexpr = self.ks[2] // 16
        q8 = lt % 8
        for kl in cutlass.range(n_loc):
            gg = g + kl
            r = gg % RAW_SLOTS
            cute.arch.mbarrier_wait(raw_empty + r, _phase(gg, RAW_SLOTS) ^ 1)
            if gg >= 1:
                cute.arch.mbarrier_wait(q_full, 0)
            rb = s_raw + r * self.raw_buf
            rk = rb
            rv = rb + self.raw_k
            rks = rv + self.raw_v
            rvs = rks + self.raw_ks
            for i in cutlass.range_constexpr(BN // 8):
                n = lt // 8 + 8 * i
                slot = kl * GPT + n // 4
                roff = _lds1(s_roff + slot * 4)
                mw = _lds1(s_gmask + slot * 4)
                ok = (roff >= 0) & (((mw >> (8 * (n % 4))) & 0xFF) != 0)
                nb = cutlass.Int32(cutlass.select_(ok, cutlass.Int32(16), cutlass.Int32(0)))
                off = cutlass.Int64(cutlass.max(roff, 0) + (n % 4) * tok16 + q8) * 16
                _cp16z(rk + n * RAW_LD + 16 * q8, k_base + off, nb)
                _cp16z(rv + n * RAW_LD + 16 * q8, v_base + off, nb)
            slot = kl * GPT + lt // 4
            soff = _lds1(s_soff + slot * 4)
            mw = _lds1(s_gmask + slot * 4)
            ok = (soff >= 0) & (((mw >> (8 * (lt % 4))) & 0xFF) != 0)
            gok = (soff >= 0) & (mw != 0)
            nb = cutlass.Int32(cutlass.select_(ok, cutlass.Int32(16), cutlass.Int32(0)))
            ng = cutlass.Int32(cutlass.select_(gok, cutlass.Int32(16), cutlass.Int32(0)))
            soff64 = cutlass.Int64(cutlass.max(soff, 0) + lt % 4) * 16
            _cp16z(rks + lt * 16, ks_base + soff64, nb)
            _cp16z(rvs + (lt // 4) * 64 + 16 * (lt % 4), vs_base + soff64, ng)
            cute.arch.cp_async_mbarrier_arrive_noinc(raw_full + r)

    # ------------------------------------------------------------------ dequant (warps 7-10)
    @cute.jit
    def dequant_tiles(
        self,
        dt,
        g,
        n_loc,
        s_raw,
        s_k16,
        s_v16,
        raw_full,
        raw_empty,
        k_full,
        k_empty,
        v_full,
        v_empty,
    ):
        """Thread `dt` < 64 dequantizes K row `dt` of each tile, the others V row `dt - 64`: the raw
        row and its scales into registers (the raw slot is then free), then into the f16 SW128
        slot once the MMA has released it."""
        is_v = dt >= BN
        n = dt % BN
        for kl in cutlass.range(n_loc):
            gg = g + kl
            r = gg % RAW_SLOTS
            b = gg % 2
            cute.arch.mbarrier_wait(raw_full + r, _phase(gg, RAW_SLOTS))
            rb = s_raw + r * self.raw_buf
            row_off = (
                rb
                + cutlass.Int32(cutlass.select_(is_v, cutlass.Int32(self.raw_k), cutlass.Int32(0)))
                + n * RAW_LD
            )
            w = [_lds4(row_off + 16 * c) for c in range(8)]
            sc_off = rb + self.raw_k + self.raw_v
            sc_off = cutlass.Int32(
                cutlass.select_(is_v, sc_off + self.raw_ks + (n // 4) * 64, sc_off + n * 16)
            )
            sw = [_lds4(sc_off + 16 * i) for i in range(4)]
            cute.arch.mbarrier_arrive(raw_empty + r)
            if is_v:
                cute.arch.mbarrier_wait(v_empty + b, _phase(gg, 2) ^ 1)
                self._decode_row(w, sw, s_v16 + b * (BN * HEAD_DIM * 2), n, True)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(v_full + b)
            else:
                cute.arch.mbarrier_wait(k_empty + b, _phase(gg, 2) ^ 1)
                self._decode_row(w, sw, s_k16 + b * (BN * HEAD_DIM * 2), n, False)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(k_full + b)

    @cute.jit
    def _decode_row(self, w, sw, dst, n, is_v: cutlass.Constexpr):
        """Raw row words `w` (8 x 4) and scale words `sw` -> f16 SW128 row `n` at `dst`: 16-byte
        chunk ci of the row's 64-dim block ci // 8 at 8192 (ci // 8) + 128 n + 16 ((ci % 8) ^ (n % 8))."""
        sel_v = (n % 4) * 0x11
        for c in cutlass.range_constexpr(8):
            # blocks 2c, 2c + 1. K: bytes 2c, 2c + 1 of the token's 16 (word c // 2);
            # V: byte n % 4 of the group's words 2c, 2c + 1
            if const_expr(is_v):
                sc0 = _scale2s(sw[(2 * c) // 4][(2 * c) % 4], sel_v)
                sc1 = _scale2s(sw[(2 * c + 1) // 4][(2 * c + 1) % 4], sel_v)
            else:
                sc0 = _scale2s(sw[0][c // 2], (2 * (c % 2)) * 0x11)
                sc1 = _scale2s(sw[0][c // 2], (2 * (c % 2) + 1) * 0x11)
            for e in cutlass.range_constexpr(4):
                d4 = _dq_word(w[c][e], sc0 if e < 2 else sc1)
                ci = 4 * c + e
                _sts4(
                    dst + 8192 * (ci // 8) + 128 * n + 16 * ((ci % 8) ^ (n % 8)),
                    d4[0],
                    d4[1],
                    d4[2],
                    d4[3],
                )

    # ------------------------------------------------------------------ MMA (warp 4)
    @cute.jit
    def mma_tiles(
        self,
        g,
        n_loc,
        qk_mma,
        pv_mma_acc,
        qk_thr,
        pv_thr,
        q_layout,
        p_layout,
        sK,
        sV,
        tmem_ptr,
        q_full,
        k_full,
        k_empty,
        v_full,
        v_empty,
        s_full,
        p_full,
        o_full,
    ):
        """QK(g), then QK(j) / PV(j - 1) pairs, then the last PV: S/P buffer j % 2 at TMEM column
        TMEM_S0 + 64 (j % 2); QK(j) overwrites it only after PV(j - 2) was issued (tcgen05 MMAs of
        one thread execute in order)."""
        tQ = cute.make_tensor(tmem_ptr + TMEM_Q, q_layout.outer)
        tSrQ = qk_thr.make_fragment_A(tQ)[None, None, None, 0]
        tSrQ = cute.make_tensor(cute.recast_ptr(tmem_ptr + TMEM_Q, dtype=F16), tSrQ.layout)
        tSrK = qk_thr.make_fragment_B(sK)
        tOrV = pv_thr.make_fragment_B(sV)
        tP = cute.make_tensor(tmem_ptr + TMEM_S0, p_layout.outer)
        tOrP = pv_thr.make_fragment_A(tP)[None, None, None, 0]
        tOrP = cute.make_tensor(cute.recast_ptr(tmem_ptr + TMEM_S0, dtype=F16), tOrP.layout)
        tbase = cutlass.Int32(tmem_ptr.toint())
        qk_op = qk_mma.op
        pv_op = pv_mma_acc.op
        if n_loc > 0:
            cute.arch.mbarrier_wait(q_full, 0)
            cute.arch.mbarrier_wait(k_full + g % 2, _phase(g, 2))
            tcgen05_fence_after()
            self.issue_qk(qk_op, tbase, tSrQ, tSrK, sK, g % 2)
            with cute.arch.elect_one():
                tcgen05.commit(s_full + g % 2)
                tcgen05.commit(k_empty + g % 2)
            for j in cutlass.range(1, n_loc + 1, 1, unroll=1):
                gj = g + j
                if j < n_loc:
                    cute.arch.mbarrier_wait(k_full + gj % 2, _phase(gj, 2))
                    tcgen05_fence_after()
                    self.issue_qk(qk_op, tbase, tSrQ, tSrK, sK, gj % 2)
                    with cute.arch.elect_one():
                        tcgen05.commit(s_full + gj % 2)
                        tcgen05.commit(k_empty + gj % 2)
                gp = gj - 1
                cute.arch.mbarrier_wait(v_full + gp % 2, _phase(gp, 2))
                cute.arch.mbarrier_wait(p_full + gp % 2, _phase(gp, 2))
                tcgen05_fence_after()
                gemm_ts(
                    pv_op,
                    tbase + TMEM_O,
                    tbase + TMEM_S0 + (gp % 2) * BN,
                    tOrP,
                    tOrV[None, None, None, gp % 2],
                    sV[None, None, None, gp % 2],
                    False,
                )
                with cute.arch.elect_one():
                    tcgen05.commit(v_empty + gp % 2)
                    tcgen05.commit(o_full)

    @cute.jit
    def issue_qk(self, qk_op, tbase, tSrQ, tSrK, sK, slot):
        """S[slot] = Q K(slot)^T: Q from tensor memory, K from SW128 slot `slot`."""
        gemm_ts(
            qk_op,
            tbase + TMEM_S0 + slot * BN,
            tbase + TMEM_Q,
            tSrQ,
            tSrK[None, None, None, slot],
            sK[None, None, None, slot],
            True,
        )

    # ------------------------------------------------------------------ softmax + epilogue (warps 0-3)
    @cute.jit
    def stage_q(self, q_s, live, tview, q_full):
        """This MMA row's q (bf16 in raw slots 1-2) -> f16 into tensor memory (the QK A operand)."""
        load_s, tLtS0, tLcS, store_p, tPtP0, tPcP, load_o, store_o, tLtO, tLcO, tStO = tview
        p_cols: cutlass.Constexpr = BN // 2
        cute.arch.cp_async_wait_group(0)
        for c in cutlass.range_constexpr(HEAD_DIM // (2 * p_cols)):
            rQ = cute.make_rmem_tensor(tPcP.shape, Float32)
            for i in cutlass.range_constexpr(p_cols // 4):
                w4 = _lds4(q_s + 16 * (c * (p_cols // 4) + i))
                for e in cutlass.range_constexpr(4):
                    rQ[4 * i + e] = _bits_f32(
                        cutlass.Int32(cutlass.select_(live, _bf2_to_f2(w4[e]), cutlass.Int32(0)))
                    )
            cute.copy(
                store_p,
                rQ,
                cute.make_tensor(tPtP0.iterator + (TMEM_Q - TMEM_S0 + c * p_cols), tPtP0.layout),
            )
        cute.arch.fence_view_async_tmem_store()
        tcgen05_fence_before()
        cute.arch.mbarrier_arrive(q_full)

    @cute.jit
    def softmax_tiles(
        self,
        g,
        n_loc,
        row_max,
        row_sum,
        live,
        t_row,
        warp,
        s_tab,
        scale_log2,
        tview,
        s_full,
        p_full,
        o_full,
    ):
        """Per tile: S from tensor memory, the max over the row's own entries, the reference max
        moved (and O rescaled) only past RESCALE_TH, P = 2^(s - max) as f16 over the S."""
        load_s, tLtS0, tLcS, store_p, tPtP0, tPcP, load_o, store_o, tLtO, tLcO, tStO = tview
        rpr: cutlass.Constexpr = self.T
        s_bits = s_tab + self.t_bits * 4
        tau = Float32(RESCALE_TH)
        neg_inf = -Float32.inf
        live_warp = warp < self.live_warps
        for kl in cutlass.range(n_loc, unroll=1):
            gg = g + kl
            buf = gg % 2
            cute.arch.mbarrier_wait(s_full + buf, _phase(gg, 2))
            tcgen05_fence_after()
            if live_warp:
                lo = cutlass.Int32(0)
                hi = cutlass.Int32(0)
                if live:
                    lo = _lds1(s_bits + (kl * rpr + t_row) * 8)
                    hi = _lds1(s_bits + (kl * rpr + t_row) * 8 + 4)
                rS = cute.make_rmem_tensor(tLcS.shape, Float32)
                tLtS_b = cute.make_tensor(tLtS0.iterator + buf * BN, tLtS0.layout)
                for c in cutlass.range_constexpr(cute.size(rS, mode=[1])):
                    cute.copy(load_s, tLtS_b[None, c, 0, 0], rS[None, c, 0, 0])
                cute.arch.fence_view_async_tmem_load()
                tile_max = neg_inf
                for i in cutlass.range_constexpr(BN):
                    word = lo if i < 32 else hi
                    x = rS[i]
                    x = x if (word & (1 << (i % 32))) != 0 else neg_inf
                    rS[i] = x
                    tile_max = cute.arch.fmax(tile_max, x)
                new_max = cute.arch.fmax(row_max, tile_max)
                use_max = row_max
                factor = Float32(1.0)
                if row_max == neg_inf:
                    use_max = new_max
                elif (new_max - row_max) * scale_log2 > tau:
                    use_max = new_max
                    factor = cute.math.exp2((row_max - new_max) * scale_log2, fastmath=True)
                need = factor != Float32(1.0)
                if cute.arch.vote_any_sync(need):
                    # O holds PV(..gg - 1): wait for it, then rescale this warp's rows
                    cute.arch.mbarrier_wait(o_full, (gg - 1) % 2)
                    tcgen05_fence_after()
                    for c in cutlass.range_constexpr(HEAD_DIM // CHUNK_COLS):
                        rO = cute.make_rmem_tensor(tLcO.shape, Float32)
                        cute.copy(
                            load_o,
                            cute.make_tensor(tLtO.iterator + c * CHUNK_COLS, tLtO.layout),
                            rO,
                        )
                        for i in cutlass.range_constexpr(cute.size(rO)):
                            rO[i] = rO[i] * factor
                        cute.copy(
                            store_o,
                            rO,
                            cute.make_tensor(tStO.iterator + c * CHUNK_COLS, tStO.layout),
                        )
                    cute.arch.fence_view_async_tmem_store()
                row_max = use_max
                safe = use_max
                if use_max == neg_inf:
                    safe = Float32(0.0)
                neg = -safe * scale_log2
                rP = cute.make_rmem_tensor(tPcP.shape, Float32)
                rP16 = cute.make_tensor(cute.recast_ptr(rP.iterator, dtype=F16), rS.layout)
                acc0 = (Float32(0.0), Float32(0.0))
                acc1 = (Float32(0.0), Float32(0.0))
                for i in cutlass.range_constexpr(0, BN, 4):
                    e = cute.make_rmem_tensor((4,), Float32)
                    e[0], e[1] = cute.arch.fma_packed_f32x2(
                        (rS[i], rS[i + 1]), (scale_log2, scale_log2), (neg, neg)
                    )
                    e[2], e[3] = cute.arch.fma_packed_f32x2(
                        (rS[i + 2], rS[i + 3]), (scale_log2, scale_log2), (neg, neg)
                    )
                    for u in cutlass.range_constexpr(4):
                        rS[i + u] = cute.math.exp2(e[u], fastmath=True)
                    acc0 = cute.arch.add_packed_f32x2(acc0, (rS[i], rS[i + 1]))
                    acc1 = cute.arch.add_packed_f32x2(acc1, (rS[i + 2], rS[i + 3]))
                row_sum = row_sum * factor + ((acc0[0] + acc0[1]) + (acc1[0] + acc1[1]))
                rP16.store(rS.load().to(F16))
                cute.copy(store_p, rP, cute.make_tensor(tPtP0.iterator + buf * BN, tPtP0.layout))
                cute.arch.fence_view_async_tmem_store()
            tcgen05_fence_before()
            cute.arch.mbarrier_arrive(p_full + buf)
        return row_max, row_sum

    @cute.jit
    def epilogue(
        self,
        row_max,
        row_sum,
        live,
        warp,
        m,
        r0,
        rows,
        kvh,
        split,
        mGs,
        scale_log2,
        out_base,
        lse_base,
        gate_base,
        gate_stride,
        tview,
        o_final,
    ):
        """O / l (times the V global scale), gated or as an fp32 partial and its log2-sum-exp."""
        load_s, tLtS0, tLcS, store_p, tPtP0, tPcP, load_o, store_o, tLtO, tLcO, tStO = tview
        group: cutlass.Constexpr = self.group
        heads: cutlass.Constexpr = self.heads
        splits: cutlass.Constexpr = self.splits
        neg_inf = -Float32.inf
        cute.arch.mbarrier_wait(o_final, 0)
        tcgen05_fence_after()
        row = r0 + m // group
        h = m % group
        if warp < self.live_warps:
            gv = mGs[2]
            inv = Float32(0.0)
            if row_sum > Float32(0.0):
                inv = gv / row_sum
            head = kvh * group + h
            for c in cutlass.range_constexpr(HEAD_DIM // CHUNK_COLS):
                rO = cute.make_rmem_tensor(tLcO.shape, Float32)
                cute.copy(load_o, cute.make_tensor(tLtO.iterator + c * CHUNK_COLS, tLtO.layout), rO)
                if live:
                    for v8 in cutlass.range_constexpr(CHUNK_COLS // 8):
                        vals = [rO[8 * v8 + i] * inv for i in range(8)]
                        d0 = c * CHUNK_COLS + 8 * v8
                        if const_expr(splits == 1):
                            if const_expr(self.gated):
                                gaddr = (
                                    gate_base
                                    + (cutlass.Int64(row) * gate_stride + head * HEAD_DIM + d0) * 2
                                )
                                g4 = _ldg4(gaddr)
                                for e in cutlass.range_constexpr(4):
                                    vals[2 * e] = _round_bf16(vals[2 * e]) * _sigmoid(
                                        _bf2_lo(g4[e])
                                    )
                                    vals[2 * e + 1] = _round_bf16(vals[2 * e + 1]) * _sigmoid(
                                        _bf2_hi(g4[e])
                                    )
                            oaddr = (
                                out_base + ((cutlass.Int64(row) * heads + head) * HEAD_DIM + d0) * 2
                            )
                            stg4(
                                oaddr,
                                _bf2_pack(vals[0], vals[1]),
                                _bf2_pack(vals[2], vals[3]),
                                _bf2_pack(vals[4], vals[5]),
                                _bf2_pack(vals[6], vals[7]),
                            )
                        else:
                            paddr = (
                                out_base
                                + (
                                    ((cutlass.Int64(split) * rows + row) * heads + head) * HEAD_DIM
                                    + d0
                                )
                                * 4
                            )
                            stg4f(paddr, vals[0], vals[1], vals[2], vals[3])
                            stg4f(paddr + 16, vals[4], vals[5], vals[6], vals[7])
            if const_expr(splits > 1):
                if live:
                    lse = neg_inf
                    if row_sum > Float32(0.0):
                        lse = row_max * scale_log2 + cute.math.log2(row_sum)
                    _stg1(
                        lse_base + ((cutlass.Int64(split) * rows + row) * heads + head) * 4,
                        _f32_bits(lse),
                    )


# ---------------------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------------------

_COMPILED: dict = {}


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def plan_splits(n_groups: int, kv_heads: int, width: int, num_sms: int) -> int:
    """Splits so the grid fills the GPU once (one CTA an SM): measured on GB300, a second
    partial wave costs more than the extra splits save."""
    est_tiles = -(-int(width * 1.25) // BN)
    return max(1, min(MAX_PLAN_SPLITS, est_tiles, num_sms // (n_groups * kv_heads)))


def _plan(q, k_cache, block_table, selected_tokens, tokens_per_block, rows_per_request):
    """(splits, layout) of the grouped kernel for this call, or None where the per-row kernel
    takes it: fewer than `MIN_GROUPS` groups, one row a request, not SM103, or no shared-memory
    plan (more than six rows a request, or a block table wider than the union scratch holds:
    612 pages of 32 tokens at four rows a request)."""
    rows, num_q_heads, head_dim = q.shape
    rpr = int(rows_per_request)
    if rpr < 2 or -(-rows // rpr) < MIN_GROUPS or head_dim != HEAD_DIM:
        return None
    if torch.cuda.get_device_capability(q.device) != (10, 3):
        return None
    kv_heads = k_cache.shape[1]
    width = selected_tokens.shape[1]
    splits = plan_splits(-(-rows // rpr), kv_heads, width, _num_sms(q.device.index or 0))
    layout = _layout(
        rpr, num_q_heads // kv_heads, width, block_table.shape[1], tokens_per_block, splits
    )
    return None if layout is None else (splits, layout)


def takes_grouped(q, k_cache, block_table, selected_tokens, tokens_per_block, rows_per_request):
    """Whether `nvfp4_sparse_gqa_verify` runs this call on the grouped kernel (else on the
    per-row kernel). A host-side decision on shapes only: a per-capture constant."""
    plan = _plan(q, k_cache, block_table, selected_tokens, tokens_per_block, rows_per_request)
    return plan is not None


def nvfp4_sparse_gqa_verify_grouped(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    kv_scale_quant_orig: torch.Tensor,
    block_table: torch.Tensor,
    selected_tokens: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    softmax_scale: float,
    rows_per_request: int,
    output_gate: Optional[torch.Tensor] = None,
    pdl: bool = True,
    splits: Optional[int] = None,
) -> torch.Tensor:
    """The grouped kernel (and the merge with more than one split), whatever the row count;
    `splits` overrides the plan. `nvfp4_sparse_gqa_verify` is the routed entry."""
    rows, num_q_heads, head_dim = q.shape
    num_pages, num_kv_heads, tpb, packed = k_cache.shape
    rpr = int(rows_per_request)
    assert head_dim == HEAD_DIM and packed == HEAD_DIM // 2 and tpb == tokens_per_block
    assert q.dtype == torch.bfloat16 and q.stride(2) == 1 and q.stride(1) == HEAD_DIM
    assert (q.stride(0) * 2) % 16 == 0 and q.data_ptr() % 16 == 0, "16-byte aligned q rows"
    assert k_cache.dtype == torch.uint8 and k_cache.stride() == v_cache.stride()
    assert k_cache.stride(3) == 1 and v_cache.shape == k_cache.shape
    assert k_scale_cache.stride() == v_scale_cache.stride()
    assert k_scale_cache.stride(2) == HEAD_DIM // 16
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert block_table.stride(0) == block_table.shape[1], "row-contiguous block table"
    assert selected_tokens.dtype == torch.int32 and selected_tokens.is_contiguous()
    assert request_indices.is_contiguous() and request_indices.dtype in (torch.int32, torch.int64)
    group = num_q_heads // num_kv_heads
    width = selected_tokens.shape[1]
    out = torch.empty((rows, num_q_heads, head_dim), dtype=torch.bfloat16, device=q.device)
    if rows == 0:
        return out
    n_groups = -(-rows // rpr)
    if splits is None:
        splits = plan_splits(n_groups, num_kv_heads, width, _num_sms(q.device.index or 0))
    n_splits = max(1, min(MAX_SPLITS, int(splits)))
    assert _layout(rpr, group, width, block_table.shape[1], tpb, n_splits) is not None, (
        "no shared-memory plan for this geometry: route through nvfp4_sparse_gqa_verify"
    )
    use_pdl = bool(pdl and get_env_enable_pdl())
    gated = output_gate is not None
    if gated:
        assert output_gate.shape == (rows, num_q_heads * head_dim) and output_gate.stride(1) == 1
        assert output_gate.dtype == torch.bfloat16
        assert (output_gate.stride(0) * 2) % 16 == 0 and output_gate.data_ptr() % 16 == 0
    req64 = request_indices.dtype == torch.int64
    key = (
        rpr,
        group,
        num_kv_heads,
        width,
        tpb,
        n_splits,
        float(softmax_scale),
        num_pages,
        tuple(block_table.shape),
        tuple(k_cache.stride()[:3]),
        tuple(k_scale_cache.stride()[:2]),
        tuple(q.stride()[:2]),
        req64,
        gated and n_splits == 1,
        use_pdl,
    )
    if n_splits > 1:
        part = torch.empty(
            (n_splits, rows, num_q_heads, head_dim), dtype=torch.float32, device=q.device
        )
        lse = torch.empty((n_splits, rows, num_q_heads), dtype=torch.float32, device=q.device)
        o_t, l_t = part, lse
    else:
        o_t, l_t = out, out
    gate_t = output_gate if gated else q
    gate_stride = output_gate.stride(0) if gated else 0
    fn = _COMPILED.get(key)
    if fn is None:
        kern = _VerifyKernel(
            rpr=rpr,
            group=group,
            kv_heads=num_kv_heads,
            width=width,
            tpb=tpb,
            splits=n_splits,
            scale=float(softmax_scale),
            num_pages=num_pages,
            bt_width=block_table.shape[1],
            num_requests=block_table.shape[0],
            k_strides=tuple(k_cache.stride()[:3]),
            ks_strides=tuple(k_scale_cache.stride()[:2]),
            q_strides=tuple(q.stride()[:2]),
            req64=req64,
            gated=gated and n_splits == 1,
            pdl=use_pdl,
        )
        fn = cute.compile(
            kern,
            # read with scalar loads: a row slice of the selection needs no copy
            from_dlpack(selected_tokens, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            _ptr(q, cutlass.BFloat16),
            _ptr(k_cache, cutlass.Uint8),
            _ptr(v_cache, cutlass.Uint8),
            _ptr(k_scale_cache, cutlass.Uint8),
            _ptr(v_scale_cache, cutlass.Uint8),
            _ptr(kv_scale_quant_orig, cutlass.Float32, 4),
            _ptr(block_table, cutlass.Int32, 4),
            _ptr(request_indices, cutlass.Int64 if req64 else cutlass.Int32, 8 if req64 else 4),
            _ptr(o_t, cutlass.Float32),
            _ptr(l_t, cutlass.Float32),
            _ptr(gate_t, cutlass.BFloat16),
            cutlass.Int32(rows),
            cutlass.Int32(gate_stride),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[key] = fn
    fn(
        selected_tokens,
        q.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        k_scale_cache.data_ptr(),
        v_scale_cache.data_ptr(),
        kv_scale_quant_orig.data_ptr(),
        block_table.data_ptr(),
        request_indices.data_ptr(),
        o_t.data_ptr(),
        l_t.data_ptr(),
        gate_t.data_ptr(),
        rows,
        gate_stride,
    )
    if n_splits == 1:
        return out
    mkey = ("merge", num_q_heads, n_splits, gated, use_pdl)
    mfn = _COMPILED.get(mkey)
    if mfn is None:
        mfn = cute.compile(
            _MergeKernel(heads=num_q_heads, splits=n_splits, gated=gated, pdl=use_pdl),
            from_dlpack(gate_t, assumed_align=16).mark_layout_dynamic(leading_dim=gate_t.dim() - 1),
            _ptr(part, cutlass.Float32),
            _ptr(lse, cutlass.Float32, 4),
            _ptr(out, cutlass.BFloat16),
            cutlass.Int32(rows),
            cutlass.Int32(gate_stride),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[mkey] = mfn
    mfn(gate_t, part.data_ptr(), lse.data_ptr(), out.data_ptr(), rows, gate_stride)
    return out


def nvfp4_sparse_gqa_verify(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    kv_scale_quant_orig: torch.Tensor,
    block_table: torch.Tensor,
    selected_tokens: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    softmax_scale: float,
    rows_per_request: int,
    output_gate: Optional[torch.Tensor] = None,
    pdl: bool = True,
) -> torch.Tensor:
    """`nvfp4_sparse_gqa_decode` for rows that come `rows_per_request` consecutive rows a
    request: the grouped kernel where `takes_grouped`, the per-row kernel otherwise. Returns
    `[rows, Hq, D]` bf16 (contiguous); with `output_gate` (`[rows, Hq * D]` bf16, unit column
    stride) `bf16(attn) * sigmoid(gate)` rounded to bf16."""
    plan = _plan(q, k_cache, block_table, selected_tokens, tokens_per_block, rows_per_request)
    if plan is None:
        return nvfp4_sparse_gqa_decode(
            q,
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
            kv_scale_quant_orig,
            block_table,
            selected_tokens,
            request_indices,
            tokens_per_block,
            softmax_scale,
            output_gate=output_gate,
            pdl=pdl,
        )
    return nvfp4_sparse_gqa_verify_grouped(
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        kv_scale_quant_orig,
        block_table,
        selected_tokens,
        request_indices,
        tokens_per_block,
        softmax_scale,
        rows_per_request,
        output_gate=output_gate,
        pdl=pdl,
        splits=plan[0],
    )


def _verify(segments, overlap=0.05, **overrides) -> dict:
    """A cell spec: `(first position, rows)` per request in row order, four rows a request
    unless `rows_per_request` overrides it, this checkpoint's geometry (24 q heads over 2 KV
    heads of 256, 32 tokens a page, 2051 selected tokens). `overlap` is the spread of a
    request's rows' block scores around the request's (each row then takes its top 512 blocks
    in ascending order, as `qsa_topk_expand_decode` writes them); None selects every row
    independently at random, as the decode op's cells do."""
    return dict(segments=segments, rows_per_request=4, overlap=overlap, **overrides)


class QsaPagedSparseGqaNvfp4Verify(OpWrapper):
    """`qsa_paged_sparse_gqa_nvfp4_decode` for rows that come `rows_per_request` consecutive
    rows a request (an MTP verify step: T = draft length + 1 rows a generation request):
    sparse GQA over the explicitly selected tokens of an NVFP4 paged pool, `[rows, Hq, D]`
    bf16, each selected K/V token loaded and dequantized once for a group of T rows. With
    `output_gate` (`[rows, Hq * D]` bf16, unit column stride) the result is
    `bf16(attn) * sigmoid(gate)` rounded once more to bf16.

    Fusion boundary: the block-table gather, the union of the group's selections, the NVFP4
    dequantize, scale, online softmax, P.V, the split-K merge and (optionally) the output gate.
    Selection, the K/V append and q's projection, norm and RoPE are not inside.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTP3 verify, 8 requests x 4 rows: the fewest the grouped kernel takes, "
            "9 splits, gate fused into the merge",
            spec=_verify(tuple((4000 + 37 * i, 4) for i in range(8)), gated=True),
        ),
        Cell(
            why="32 rows whose selections share little (independent random rows, union "
            "~2.7x a row's)",
            spec=_verify(tuple((3000 + 211 * i, 4) for i in range(8)), overlap=None),
        ),
        Cell(
            why="MTP3 verify at 1024 rows (256 requests x 4): one split, gate fused into the "
            "kernel's store",
            spec=_verify(tuple((2100 + (37 * i) % 1900, 4) for i in range(256)), gated=True),
        ),
        Cell(
            why="groups spanning requests (3, 4, 4, 1, 4 x 5, 2 rows: misaligned and partial "
            "groups), early positions, int64 request indices, non-unit global scales, trunk "
            "layout",
            spec=_verify(
                (
                    (0, 3),
                    (5, 4),
                    (37, 4),
                    (4000, 1),
                    (4100, 4),
                    (4200, 4),
                    (4300, 4),
                    (4400, 4),
                    (4500, 4),
                    (4600, 2),
                ),
                request_dtype=torch.int64,
                global_k=0.37,
                global_v=2.5,
                slot_pages=24,
                layer=11,
                pool_pages=2048,
            ),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument (as `qsa_paged_sparse_gqa_nvfp4`).
    Call sites (the target's `modeling.py`): `_sparse_gqa` over an NVFP4 pool for rows without
    the prefill pair that come `rows_per_request` > 1 a request -- the decode target's
    attention layers on a verify step (with the output gate) and a mixed step's generation
    rows. Draft-layer and one-row-a-request calls stay on `qsa_paged_sparse_gqa_nvfp4_decode`.

    Routing (`takes_grouped`, host-side, on shapes only): the grouped kernel from
    `MIN_GROUPS` (8) groups of T >= 2 rows up, on SM103, where its shared-memory plan fits
    (T <= 6; a block table of at most 612 pages of 32 tokens at T = 4, 892 at T = 3, 1364 at
    T = 2); the per-row kernel otherwise. Splits fill the GPU once (one CTA an SM), each split
    followed by the decode op's merge.

    GB300, one layer, cold L2, T = 4, a request's rows sharing ~80% of their blocks (union
    1.2x a row's selection): 32 rows 18.7 us, 128 rows 36.4, 512 rows 91.7, 1024 rows 181 --
    against 22.8 / 65.9 / 230 / 448 us for the per-row kernel; at union 1.4x 19.2 / 39.5 /
    104 / 203 us. Rows that select independently (union 2.7x) gain less: 23.8 / 59.9 / 176 /
    346 us against 23.1 / 67.3 / 232 / 452. Below 32 rows the call is the per-row kernel's.

    Numerics: dequantized K/V values are exact in f16; scores on f16 MMAs with fp32
    accumulation; P rounded to f16 (as the per-row kernel); a row's softmax offset moves only
    when its max grew by more than 2^8, the sum stays fp32 unrounded; the second-level scales
    act on the scores and the normalized output. Within the decode op's band (4 ulp element,
    2 ulp RMS) of the float64 reference and of the per-row kernel; not bitwise the per-row
    kernel (tiles hold the group's union). Deterministic; graph-capturable (row count is a
    runtime value; one compile per (T, split) variant).

    Preconditions: as `qsa_paged_sparse_gqa_nvfp4_decode` (bf16 q with head_dim 256 and
    16-byte aligned rows; K and V planes, and their scale planes, share strides; power-of-two
    tokens per block; int32 row-contiguous block table; contiguous int32 selections), and
    expanded selections: each four-column quad of a row names one four-token group at its own
    offsets (column % 4 == token % 4), no group twice in a row -- what
    `qsa_topk_expand_decode` and `qsa_expand_block_indices` write. The rows of a group share
    the K/V tiles, so a NaN in a token one row selects reaches the group's other rows of that
    request.
    """

    def __call__(
        self,
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        kv_scale_quant_orig,
        block_table,
        selected_tokens,
        request_indices,
        tokens_per_block: int,
        softmax_scale: float,
        rows_per_request: int,
        output_gate=None,
    ) -> torch.Tensor:
        return nvfp4_sparse_gqa_verify(
            q,
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
            kv_scale_quant_orig,
            block_table,
            selected_tokens,
            request_indices,
            tokens_per_block,
            softmax_scale,
            rows_per_request,
            output_gate=output_gate,
        )

    def reference(
        self,
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        kv_scale_quant_orig,
        block_table,
        selected_tokens,
        request_indices,
        tokens_per_block: int,
        softmax_scale: float,
        rows_per_request: int,
        output_gate=None,
    ) -> torch.Tensor:
        """`qsa_paged_sparse_gqa_nvfp4_decode`'s reference (`rows_per_request` only groups the
        work, it does not change the result)."""
        return qsa_paged_sparse_gqa_nvfp4_decode.reference(
            q,
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
            kv_scale_quant_orig,
            block_table,
            selected_tokens,
            request_indices,
            tokens_per_block,
            softmax_scale,
            output_gate=output_gate,
        )

    def is_valid(
        self,
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        kv_scale_quant_orig,
        block_table,
        selected_tokens,
        request_indices,
        tokens_per_block: int,
        softmax_scale: float,
        rows_per_request: int,
        output_gate=None,
    ) -> None:
        qsa_paged_sparse_gqa_nvfp4_decode.is_valid(
            q,
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
            kv_scale_quant_orig,
            block_table,
            selected_tokens,
            request_indices,
            tokens_per_block,
            softmax_scale,
            output_gate=output_gate,
        )
        assert int(rows_per_request) >= 1, "rows_per_request >= 1"
        # The union is sized for expanded selections: a quad of columns names one four-token
        # group at its own offsets, and no group twice in a row. Anything else can overflow
        # the union, or count a token once where the reference counts it twice.
        rows, width = selected_tokens.shape
        quads = -(-width // 4)
        padded = torch.full((rows, 4 * quads), -1, dtype=torch.int64, device=selected_tokens.device)
        padded[:, :width] = selected_tokens.long()
        sel = padded.view(rows, quads, 4)
        valid = sel >= 0
        column = torch.arange(4, device=sel.device)
        assert bool(((sel % 4 == column) | ~valid).all()), "column % 4 == token % 4"
        group = torch.where(valid, sel // 4, torch.full_like(sel, -1)).amax(dim=2)
        assert bool(((sel // 4 == group[..., None]) | ~valid).all()), "one group a quad"
        listed = torch.where(group >= 0, group, torch.full_like(group, -1)).sort(dim=1).values
        repeat = (listed[:, 1:] == listed[:, :-1]) & (listed[:, 1:] >= 0)
        assert not bool(repeat.any()), "no four-token group twice in a row"


qsa_paged_sparse_gqa_nvfp4_verify = QsaPagedSparseGqaNvfp4Verify()

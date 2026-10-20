# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-launch MoE decode front for Qwen3.5 above 32 tokens (``TRTLLM_Q35_MOE_FRONT_BIG``): the MTP-6 target
verify at b16 / b32 (112 / 224 tokens).

Router GEMM + top-10-of-512 routing + trtllm-gen pre-routed tables + NVFP4 quantization of the MoE input in ONE
launch, like :mod:`q35_moe_front` (M <= 32), whose single routing CTA and one-tile-per-expert tables do not scale
to 112-224 tokens. Outputs, bitwise:

* logits [M, E] BF16 = those of the low-M BF16 router GEMM (:class:`LowMMmaGemmKernel`, tactic (16, 1): 16 K
  slices of 256, each a chain of ``mma.sync`` m16n8k16 over its 4 chunks x 4 steps with the low-M fragment
  mapping, the 16 slice partials summed in slice order from 0.0, times 1.0, ``cvt.rn.bf16``). Each output row
  depends only on its own x row, so the reference is that kernel run on row chunks of <= 64 rows.
* top-k ids / BF16 weights: the routing arithmetic of :mod:`q35_moe_front` (= the Triton route kernel).
* xq / xsf: ``fp4_quantize`` (linear scale layout).
* tables (``q35_prerouted_sizes`` order): those of the per-token route + tables kernel up to the row order inside
  an expert's block (token order here).

Layout (no thread-block clusters: on GB300 only 15 clusters of 8 are co-resident, so 16 would run in two waves;
a plain grid of <= #SMs CTAs with one CTA per SM is co-resident, which makes the spin waits below safe):

* GEMM: CTA (row group g, column block b, K slice s) computes rows [g RG, (g + 1) RG) x router columns
  [b CB, (b + 1) CB) x K [256 s, 256 (s + 1)). M = 224: RG 112, CB 128 (2 x 4 x 16 = 128 CTAs); M = 112: RG 112,
  CB 64 (1 x 8 x 16). Before ``griddepcontrol.wait``: the CTA's router weights (static) -> shared memory
  (cp.async). After it: the x slice in 4 chunk groups; the MMA warps start on chunk 0 while 1..3 land.
* The slice partial (FP32 [RG, CB]) goes through shared memory to a global scratch (coalesced 16-byte stores),
  then the CTA publishes a flag (= the launch's epoch + 1, release).
* Rows: CTA j of a row group's 16 NB CTAs owns rows j, j + 16 NB, ... of the group. After acquiring the group's
  flags (relaxed polls, then one acquire fence) it sums each owned row's 512 logits over the 16 slices in order
  (logits out, and into shared memory),
  then routes the row (one warp per row): top-k / renormalization (bitwise :mod:`q35_moe_front`), ids / weights
  out, local picks set the token's bit in a per-local-expert token mask (global, red.or).
* The CTAs count their routed rows on a ticket (acq_rel); the CTA that completes M builds the tables from the
  masks (counts, CTAs per expert for ``tile``, exclusive scan, CTA -> expert / row limit, totals, expanded ->
  permuted and permuted -> token), zeroes the masks and the ticket and bumps the epoch.
* Each CTA NVFP4-quantizes its share of x from shared memory (rows i % NB == b of its row group, its K slice).

The epoch makes the flags self-invalidating: every CTA reads it after the wait; the last CTA bumps it after every
CTA of the launch has passed its spin wait (the last token needs every CTA's partial). State and scratch are
zero-initialized; the masks and the ticket return to zero at the end of every launch.

PDL rule: before ``griddepcontrol.wait`` only static data is read and nothing global is written; every
CTA releases its dependents after a CTA barrier that follows its last global access.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .lowm_mma_gemm import CHUNK, N_TILE, _mma_bf16_m16n8k16
from .q35_moe_front import (_SORT16, _SmemAllocator, _add, _asm, _atom_add_acq_rel, _below, _bit, _div_bf16,
                            _keys2,
                            _exp_tr, _f32, _gdc_launch, _gdc_wait, _gtime, _i32, _i64, _key1, _key_val, _ld_f32,
                            _ld_shared, _max_u32, _min_u32, _nvfp4_block, _popc, _red_or_shared,
                            _redux_max_u32, _scan_add, _sel_eq, _sel_eq_f, _sel_lt, _smem_addr, _st_b16, _st_b32,
                            _st_shared, _st_u8, _st_u64, _st_v2, _struct, _sub, _unpack, max_num_tiles)

# State words (int32, zero-initialized, per layer): epoch, routed-row ticket, partial flags, token masks.
STATE_EPOCH = 0
STATE_TICKET = 1
STATE_FLAGS = 2
MAX_CTAS = 256
STATE_MASKS = STATE_FLAGS + MAX_CTAS
MAX_MASK_WORDS = 512 * 8  # num_local <= 512, M <= 256
STATE_WORDS = STATE_MASKS + MAX_MASK_WORDS
MAX_ROWS = 256
SLICES = 16  # the low-M router tactic (16, 1): 16 K slices
THREADS = 256
ROW_GROUP = 4  # row tiles whose activation fragments a warp holds at once
TRACE_SLOTS = 16  # u64 per CTA in trace builds: globaltimer at the phase ends (slot 12: built the tables)


# ---------------------------------------------------------------------------------------------------------
# Inline PTX helpers (memory-clobbering where they touch memory, so nothing moves across them)
# ---------------------------------------------------------------------------------------------------------
@dsl_user_op
def _cp_async16(saddr: Int32, gaddr: Int64, *, loc=None, ip=None) -> None:
    _asm(None, [_i32(saddr, loc, ip), _i64(gaddr, loc, ip)], "cp.async.cg.shared.global [$0], [$1], 16;",
         "r,l,~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _cp_async_commit(*, loc=None, ip=None) -> None:
    _asm(None, [], "cp.async.commit_group;", "~{memory}", loc=loc, ip=ip)


def _make_cp_async_wait(n: int):
    @dsl_user_op
    def _wait(*, loc=None, ip=None) -> None:
        _asm(None, [], f"cp.async.wait_group {n};", "~{memory}", loc=loc, ip=ip)

    return _wait


_CP_ASYNC_WAIT = [_make_cp_async_wait(n) for n in range(8)]


@dsl_user_op
def _ld_shared_v4(addr: Int32, *, loc=None, ip=None):
    out = _asm(_struct(4), [_i32(addr, loc, ip)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
               "=r,=r,=r,=r,r,~{memory}", loc=loc, ip=ip)
    return _unpack(out, 4, loc, ip)


@dsl_user_op
def _st_shared_v2f(addr: Int32, a: Float32, b: Float32, *, loc=None, ip=None) -> None:
    _asm(None, [_i32(addr, loc, ip), _f32(a, loc, ip), _f32(b, loc, ip)], "st.shared.v2.f32 [$0], {$1, $2};",
         "r,f,f,~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _st_shared_v2(addr: Int32, w0: Int32, w1: Int32, *, loc=None, ip=None) -> None:
    _asm(None, [_i32(addr, loc, ip), _i32(w0, loc, ip), _i32(w1, loc, ip)], "st.shared.v2.b32 [$0], {$1, $2};",
         "r,r,r,~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _st_global_v4(addr: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None) -> None:
    _asm(None, [_i64(addr, loc, ip)] + [_i32(w, loc, ip) for w in (w0, w1, w2, w3)],
         "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r,~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _ld_cg_v4f(addr: Int64, *, loc=None, ip=None):
    out = _asm(llvm.StructType.get_literal([T.f32()] * 4), [_i64(addr, loc, ip)],
               "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,l,~{memory}", loc=loc, ip=ip)
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ld_cg_b32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [_i64(addr, loc, ip)], "ld.global.cg.b32 $0, [$1];", "=r,l,~{memory}", loc=loc,
                      ip=ip))


@dsl_user_op
def _st_release_b32(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(None, [_i64(addr, loc, ip), _i32(value, loc, ip)], "st.release.gpu.global.b32 [$0], $1;",
         "l,r,~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _ld_relaxed_b32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [_i64(addr, loc, ip)], "ld.relaxed.gpu.global.b32 $0, [$1];", "=r,l,~{memory}",
                      loc=loc, ip=ip))


@dsl_user_op
def _fence_acq_rel_gpu(*, loc=None, ip=None) -> None:
    _asm(None, [], "fence.acq_rel.gpu;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _red_or_b32(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(None, [_i64(addr, loc, ip), _i32(value, loc, ip)], "red.relaxed.gpu.global.or.b32 [$0], $1;",
         "l,r,~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    """``cvt.rn.bf16`` of lo (low half) and hi (high half), one word."""
    return Int32(_asm(T.i32(), [_f32(lo, loc, ip), _f32(hi, loc, ip)], "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f",
                      False, loc=loc, ip=ip))


@dsl_user_op
def _vote_all(x: Int32, *, loc=None, ip=None) -> Int32:
    """1 if ``x != 0`` in every lane of the warp, else 0."""
    return Int32(_asm(T.i32(), [_i32(x, loc, ip)],
                      "{\n\t.reg .pred p, q;\n\tsetp.ne.u32 p, $1, 0;\n\tvote.sync.all.pred q, p, 0xffffffff;\n\t"
                      "selp.u32 $0, 1, 0, q;\n\t}", "=r,r", loc=loc, ip=ip))


def big_front_config(num_rows: int, num_experts: int, num_sms: int):
    """(row_group, col_block) of the big front for ``num_rows`` tokens, or None: 16 K-slice CTAs per (row group,
    column block), at most ``num_sms`` CTAs (one per SM, all co-resident), at most 2 rows per CTA to reduce."""
    if num_experts != 512 or num_rows % 16 or not 33 <= num_rows <= MAX_ROWS:
        return None
    for rg in (112, 128, 96, 80, 64, 48):
        if num_rows % rg:
            continue
        for cb in (32, 64, 128, 256):
            nb = num_experts // cb
            ctas = (num_rows // rg) * nb * SLICES
            if ctas <= num_sms and rg % nb == 0 and rg <= 2 * nb * SLICES:
                return rg, cb
    return None


class Q35MoeFrontBigKernel:
    """``num_rows`` (multiple of 16, <= 256) tokens: router logits [M, E] (BF16), NVFP4 x, top-k ids / BF16
    weights and the trtllm-gen pre-routed tables (flat int32 buffer in ``q35_prerouted_sizes`` order) in one
    launch (module doc). ``row_group`` / ``col_block``: the CTA tile (``big_front_config``). ``n_per_warp``: router
    n-tiles (8 columns) per MMA warp (each MMA warp takes all row tiles). ``a_stride``: row stride of x in
    elements. Local experts [local_start, local_start + num_local) (local_start is a runtime argument).
    ``trace``: debug builds only, %globaltimer at the phase ends of every CTA into ``trace_ptr``."""

    def __init__(self, *, num_rows: int, k_extent: int, num_experts: int, top_k: int, num_local: int, tile: int,
                 a_stride: int, row_group: int, col_block: int, use_pdl: bool, n_per_warp: int = 0,
                 trace: bool = False) -> None:
        if num_rows % 16 or not 1 <= num_rows <= MAX_ROWS:
            raise ValueError(f"num_rows={num_rows} must be a multiple of 16 in [16, {MAX_ROWS}]")
        if top_k != 10:
            raise ValueError("the renormalization tree is the route kernel's top-10 tree")
        if num_experts != 32 * 16:
            raise ValueError("one warp per token holds 16 logits per lane: 512 experts")
        if k_extent != SLICES * 256:
            raise ValueError("K must be 16 slices of 256 (the low-M router tactic (16, 1) at K = 4096)")
        if num_local % 32 or not 32 <= num_local <= num_experts:
            raise ValueError(f"num_local={num_local} must be a multiple of 32 in [32, {num_experts}]")
        if tile & (tile - 1) or not 8 <= tile <= 256:
            raise ValueError(f"tile={tile} must be a power of two in [8, 256]")
        if a_stride % 8 or a_stride < k_extent:
            raise ValueError("x rows must be 16-byte aligned")
        nb = num_experts // col_block
        if (num_rows % row_group or row_group % 16 or col_block % N_TILE or num_experts % col_block
                or row_group % nb):
            raise ValueError(f"bad tile rows {row_group} x cols {col_block} for M={num_rows}")
        self.num_rows = num_rows
        self.k_extent = k_extent
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local = num_local
        self.tile = tile
        self.a_stride = a_stride
        self.use_pdl = use_pdl
        self.trace = trace
        self.rg = row_group
        self.cb = col_block
        self.nrg = num_rows // row_group
        self.nb = nb
        self.group_ctas = nb * SLICES  # CTAs of a row group
        self.ctas = self.nrg * self.group_ctas
        if self.ctas > MAX_CTAS:
            raise ValueError(f"{self.ctas} CTAs > {MAX_CTAS}")
        self.rows_per_cta = -(-row_group // self.group_ctas)  # rows a CTA reduces and routes at most
        if self.rows_per_cta * (num_experts // 4) > THREADS:
            raise ValueError("one thread per 4 logits of every owned row")
        self.slice_k = k_extent // SLICES
        self.row_tiles = row_group // 16
        self.tile_groups = [(g0, min(g0 + ROW_GROUP, self.row_tiles)) for g0 in range(0, self.row_tiles, ROW_GROUP)]
        self.n_tiles = col_block // N_TILE
        self.tn = n_per_warp or max(1, self.n_tiles // 8)
        if self.n_tiles % self.tn:
            raise ValueError(f"n_per_warp={self.tn} must divide the {self.n_tiles} n-tiles of a CTA")
        self.mma_warps = self.n_tiles // self.tn
        self.warps = THREADS // 32
        if self.mma_warps > self.warps:
            raise ValueError(f"{self.mma_warps} MMA warps > {self.warps}")
        self.mask_words = -(-num_rows // 32)
        if num_local * self.mask_words > MAX_MASK_WORDS:
            raise ValueError("token masks do not fit the state")
        self.max_ctas = max_num_tiles(num_rows, top_k, num_local, tile)
        self.max_ctas_per_expert = -(-num_rows // tile)
        # q35_prerouted_sizes order: e2p [M*K], p2t [(max_ctas + 1) * tile], cta -> expert, cta -> limit,
        # CTA count, padded rows.
        self.off_p2t = num_rows * top_k
        self.off_cte = self.off_p2t + (self.max_ctas + 1) * tile
        self.off_ctl = self.off_cte + self.max_ctas
        self.off_nct = self.off_ctl + self.max_ctas
        self.off_npd = self.off_nct + 1
        # shared memory (bytes): x slice [RG][256 + 8] bf16, W slice [CB][256 + 8] bf16 (16-byte row pad: the
        # 16-byte fragment loads of a warp hit distinct banks), misc words. After the MMAs the W area holds the
        # FP32 partial [RG][CB + 4]; after the quantization the x area holds the owned rows' logits, then (the
        # table-building CTA) the masks / prefix counts / block offsets.
        self.row_bytes = (self.slice_k + 8) * 2
        self.sx_bytes = row_group * self.row_bytes
        self.sw_bytes = col_block * self.row_bytes
        self.part_stride = col_block + 4  # floats
        if row_group * self.part_stride * 4 > self.sw_bytes:
            raise ValueError("partial tile does not fit the weight area")
        self.misc_bytes = 64
        self.smem_bytes = self.sx_bytes + self.sw_bytes + self.misc_bytes
        if (num_local * self.mask_words * 8 + num_local * 4 > self.sx_bytes
                or self.rows_per_cta * num_experts * 2 > self.sx_bytes):
            raise ValueError("scratch does not fit the x area")

    @cute.jit
    def __call__(self, x_ptr: Int64, w_ptr: Int64, logits_ptr: Int64, gs_ptr: Int64, xq_ptr: Int64, xsf_ptr: Int64,
                 ids_ptr: Int64, wts_ptr: Int64, tab_ptr: Int64, state_ptr: Int64, part_ptr: Int64,
                 trace_ptr: Int64, local_start: Int32, stream) -> None:
        self.kernel(x_ptr, w_ptr, logits_ptr, gs_ptr, xq_ptr, xsf_ptr, ids_ptr, wts_ptr, tab_ptr, state_ptr,
                    part_ptr, trace_ptr, local_start).launch(
            grid=[self.ctas, 1, 1],
            block=[THREADS, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(self, x_ptr: Int64, w_ptr: Int64, logits_ptr: Int64, gs_ptr: Int64, xq_ptr: Int64, xsf_ptr: Int64,
               ids_ptr: Int64, wts_ptr: Int64, tab_ptr: Int64, state_ptr: Int64, part_ptr: Int64,
               trace_ptr: Int64, local_start: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tr = cute.make_rmem_tensor(cute.make_layout((TRACE_SLOTS, )), Int64)
        if const_expr(self.trace):
            tr[0] = _gtime()
        warp = tidx // 32
        lane = tidx % 32
        r = lane // 4
        q = lane % 4
        k_extent: cutlass.Constexpr = self.k_extent
        rg: cutlass.Constexpr = self.rg
        cb: cutlass.Constexpr = self.cb
        nb: cutlass.Constexpr = self.nb
        row_bytes: cutlass.Constexpr = self.row_bytes
        slice_k: cutlass.Constexpr = self.slice_k

        smem = _SmemAllocator()
        buf = smem.allocate_tensor(Int32, cute.make_layout((self.smem_bytes // 4, )), byte_alignment=128)
        sx = _smem_addr(buf.iterator)
        sw = sx + Int32(self.sx_bytes)
        sm = sw + Int32(self.sw_bytes)  # [0] epoch, [1] last-CTA flag

        s = bidx % SLICES
        blk = bidx // SLICES
        b = blk % nb
        g = blk // nb
        row0 = g * rg  # first token row of the group
        col0 = b * cb  # first router column of the block
        k0 = s * slice_k  # first K of the slice

        # ---- router weights of the slice -> shared memory (static: before the dependency wait), group 0 ----
        for idx in cutlass.range(tidx, cb * (slice_k // 8), THREADS):
            row = idx // (slice_k // 8)
            seg = idx % (slice_k // 8)
            _cp_async16(sw + row * row_bytes + seg * 16,
                        w_ptr + (Int64(col0 + row) * Int64(k_extent) + Int64(k0 + seg * 8)) * Int64(2))
        _cp_async_commit()
        gs = _ld_f32(gs_ptr)
        if const_expr(self.use_pdl):
            _gdc_wait()
        if const_expr(self.trace):
            tr[1] = _gtime()

        if tidx == 0:
            _st_shared(sm, _ld_cg_b32(state_ptr + Int64(STATE_EPOCH * 4)))
        # ---- x slice -> shared memory, one group per 64-element chunk ----
        for c in cutlass.range_constexpr(4):
            for idx in cutlass.range(tidx, rg * (CHUNK // 8), THREADS):
                row = idx // (CHUNK // 8)
                seg = idx % (CHUNK // 8)
                _cp_async16(sx + row * row_bytes + (c * CHUNK + seg * 8) * 2,
                            x_ptr + (Int64(row0 + row) * Int64(self.a_stride) + Int64(k0 + c * CHUNK + seg * 8))
                            * Int64(2))
            _cp_async_commit()

        # ---- the slice partial: low-M mma.sync chains (module doc) ----
        row_tiles: cutlass.Constexpr = self.row_tiles
        tn: cutlass.Constexpr = self.tn
        acc = cute.make_rmem_tensor(cute.make_layout((row_tiles * tn, 4), stride=(4, 1)), Float32)
        acc.fill(0.0)
        for c in cutlass.range_constexpr(4):
            _CP_ASYNC_WAIT[3 - c]()
            cute.arch.sync_threads()
            if const_expr(self.trace and c == 0):
                tr[2] = _gtime()
            if const_expr(self.trace and c == 3):
                tr[3] = _gtime()
            if warp < self.mma_warps:
                self._mma_chunk(acc, sx, sw, c, warp, r, q)
        cute.arch.sync_threads()  # every warp is done with the weights
        if const_expr(self.trace):
            tr[4] = _gtime()

        # ---- partial -> shared memory (weight area) -> global scratch, then the flag ----
        ps: cutlass.Constexpr = self.part_stride
        if warp < self.mma_warps:
            for i in cutlass.range_constexpr(row_tiles):
                for n in cutlass.range_constexpr(tn):
                    col = (warp * tn + n) * N_TILE + 2 * q
                    a = sw + ((16 * i + r) * ps + col) * 4
                    _st_shared_v2f(a, acc[i * tn + n, 0], acc[i * tn + n, 1])
                    _st_shared_v2f(a + 8 * ps * 4, acc[i * tn + n, 2], acc[i * tn + n, 3])
        cute.arch.sync_threads()
        pbase = part_ptr + Int64(bidx) * Int64(rg * cb * 4)
        for u in cutlass.range_constexpr(-(-(rg * cb // 4) // THREADS)):
            v = tidx + u * THREADS
            if v < rg * cb // 4:
                prow = v // (cb // 4)
                pc4 = v % (cb // 4)
                w0, w1, w2, w3 = _ld_shared_v4(sw + (prow * ps + 4 * pc4) * 4)
                _st_global_v4(pbase + (Int64(prow) * Int64(cb) + Int64(4 * pc4)) * Int64(4), w0, w1, w2, w3)
        cute.arch.sync_threads()
        epoch = _ld_shared(sm)
        ready = epoch + Int32(1)
        if tidx == 0:
            _st_release_b32(state_ptr + Int64((STATE_FLAGS + bidx) * 4), ready)
        if const_expr(self.trace):
            tr[5] = _gtime()

        # ---- NVFP4 quantization of this CTA's share of x (rows i % NB == b of the group, this K slice) ----
        self._quant_share(sx, gs, xq_ptr, xsf_ptr, row0, b, s, tidx)
        if const_expr(self.trace):
            cute.arch.sync_threads()
            tr[6] = _gtime()

        # ---- owned rows j, j + 16 NB, ... of the group: wait for the group's partials ----
        gc: cutlass.Constexpr = self.group_ctas
        j = bidx % gc
        n_rows = Int32(0)
        for u in cutlass.range_constexpr(self.rows_per_cta):
            n_rows = n_rows + _sel_lt(j + Int32(u * gc), Int32(rg), Int32(1), Int32(0))
        if warp == 0:
            # relaxed polls (all of a lane's flags in flight at once), then one acquire fence
            fbase = state_ptr + Int64((STATE_FLAGS + g * gc) * 4)
            pending = Int32(1)
            while pending != 0:
                ok = Int32(1)
                for u in cutlass.range_constexpr(-(-gc // 32)):
                    if const_expr((u + 1) * 32 <= gc):
                        f = _ld_relaxed_b32(fbase + Int64(lane + u * 32) * Int64(4))
                        ok = ok & _sel_eq(f, ready, Int32(1), Int32(0))
                    else:
                        if lane + u * 32 < gc:
                            f = _ld_relaxed_b32(fbase + Int64(lane + u * 32) * Int64(4))
                            ok = ok & _sel_eq(f, ready, Int32(1), Int32(0))
                pending = Int32(1) - _vote_all(ok)
            _fence_acq_rel_gpu()
        cute.arch.sync_threads()
        if const_expr(self.trace):
            tr[7] = _gtime()

        # ---- reduce the owned rows: ((0 + P_0) + P_1) + ... + P_15, times 1.0, BF16; 4 logits per thread ----
        slg = sx  # owned rows' logits [rows_per_cta][E] bf16 (the x slice is no longer needed)
        self._reduce_rows(part_ptr, logits_ptr, slg, g, j, tidx)
        cute.arch.sync_threads()
        if const_expr(self.trace):
            tr[8] = _gtime()

        # ---- route the owned rows, one warp per row ----
        if warp < n_rows:
            t = row0 + j + warp * gc
            self._route_token(t, slg + warp * (self.num_experts * 2), ids_ptr, wts_ptr, state_ptr, local_start, lane)
        cute.arch.sync_threads()
        if const_expr(self.trace):
            tr[9] = _gtime()

        # ---- routed-row ticket: the CTA that completes M builds the tables ----
        if tidx == 0:
            last = Int32(0)
            if n_rows > 0:
                old = _atom_add_acq_rel(state_ptr + Int64(STATE_TICKET * 4), n_rows)
                last = _sel_eq(old + n_rows, Int32(self.num_rows), Int32(1), Int32(0))
            _st_shared(sm + 4, last)
        cute.arch.sync_threads()
        if const_expr(self.trace):
            tr[10] = _gtime()
        if _ld_shared(sm + 4) != 0:
            self._tables(tab_ptr, ids_ptr, state_ptr, local_start, sx, epoch, tidx, warp, lane)
        cute.arch.sync_threads()
        if const_expr(self.trace):
            tr[11] = _gtime()
            if tidx == 0:
                for i in cutlass.range_constexpr(12):
                    _st_u64(trace_ptr + (Int64(bidx) * Int64(TRACE_SLOTS) + Int64(i)) * Int64(8), tr[i])
                _st_u64(trace_ptr + (Int64(bidx) * Int64(TRACE_SLOTS) + Int64(12)) * Int64(8),
                        Int64(_ld_shared(sm + 4)))
        # The barrier above follows this CTA's last global access.
        if const_expr(self.use_pdl):
            _gdc_launch()

    # ------------------------------------------------------------------------------------------------- GEMM
    @cute.jit
    def _mma_chunk(self, acc: cute.Tensor, sx: Int32, sw: Int32, c: cutlass.Constexpr, warp, r, q) -> None:
        """Chunk c (64 K) of the slice for all row tiles x this warp's n-tiles: lane (r, q) loads 16 consecutive K
        of its rows / weight row at chunk offset 16q (the low-M registers), then the four m16n8k16 steps j with
        the low-M mapping; step j of every accumulator of a row-tile group before step j + 1."""
        tn: cutlass.Constexpr = self.tn
        row_bytes: cutlass.Constexpr = self.row_bytes
        kb = (c * CHUNK + 16 * q) * 2
        bw = []
        for n in cutlass.range_constexpr(tn):
            addr = sw + ((warp * tn + n) * N_TILE + r) * row_bytes + kb
            b0, b1, b2, b3 = _ld_shared_v4(addr)
            b4, b5, b6, b7 = _ld_shared_v4(addr + 16)
            bw.append((b0, b1, b2, b3, b4, b5, b6, b7))
        for gi in cutlass.range_constexpr(len(self.tile_groups)):
            g0, g1 = self.tile_groups[gi]
            aw = []
            for i in cutlass.range_constexpr(g0, g1):
                alo = sx + (16 * i + r) * row_bytes + kb
                ahi = alo + 8 * row_bytes
                l0, l1, l2, l3 = _ld_shared_v4(alo)
                l4, l5, l6, l7 = _ld_shared_v4(alo + 16)
                h0, h1, h2, h3 = _ld_shared_v4(ahi)
                h4, h5, h6, h7 = _ld_shared_v4(ahi + 16)
                aw.append(((l0, l1, l2, l3, l4, l5, l6, l7), (h0, h1, h2, h3, h4, h5, h6, h7)))
            for jj in cutlass.range_constexpr(4):
                wi = 2 * jj
                for i in cutlass.range_constexpr(g0, g1):
                    lo, hi = aw[i - g0]
                    for n in cutlass.range_constexpr(tn):
                        a = i * tn + n
                        c0, c1, c2, c3 = _mma_bf16_m16n8k16(lo[wi], hi[wi], lo[wi + 1], hi[wi + 1], bw[n][wi],
                                                            bw[n][wi + 1], acc[a, 0], acc[a, 1], acc[a, 2],
                                                            acc[a, 3])
                        acc[a, 0] = c0
                        acc[a, 1] = c1
                        acc[a, 2] = c2
                        acc[a, 3] = c3

    @cute.jit
    def _reduce_rows(self, part_ptr: Int64, logits_ptr: Int64, slg: Int32, g, j, tidx) -> None:
        """Owned row u (j + u 16 NB of group g), logits 4 c4 .. 4 c4 + 3 by thread u E / 4 + c4: the 16 slice
        partials (column block 4 c4 // CB) summed in order from 0.0, times 1.0, BF16; logits out and into
        shared memory (slg [rows_per_cta][E])."""
        cb: cutlass.Constexpr = self.cb
        rg: cutlass.Constexpr = self.rg
        nb: cutlass.Constexpr = self.nb
        gc: cutlass.Constexpr = self.group_ctas
        num_experts: cutlass.Constexpr = self.num_experts
        per_row: cutlass.Constexpr = num_experts // 4
        u = tidx // per_row
        c4 = tidx % per_row
        brow = j + u * gc
        if u < self.rows_per_cta:
            if brow < rg:
                col = 4 * c4
                bb = col // cb
                cc = col % cb
                parts = []
                for p in cutlass.range_constexpr(SLICES):
                    a = part_ptr + ((Int64((g * nb + bb) * SLICES + p) * Int64(rg) + Int64(brow)) * Int64(cb)
                                    + Int64(cc)) * Int64(4)
                    parts.append(_ld_cg_v4f(a))
                v0 = Float32(0.0)
                v1 = Float32(0.0)
                v2 = Float32(0.0)
                v3 = Float32(0.0)
                for p in cutlass.range_constexpr(SLICES):
                    v0 = v0 + parts[p][0]
                    v1 = v1 + parts[p][1]
                    v2 = v2 + parts[p][2]
                    v3 = v3 + parts[p][3]
                alpha = Float32(1.0)
                w01 = _pack_bf16x2(v0 * alpha, v1 * alpha)
                w23 = _pack_bf16x2(v2 * alpha, v3 * alpha)
                grow = g * rg + brow
                _st_v2(logits_ptr + (Int64(grow) * Int64(num_experts) + Int64(col)) * Int64(2), w01, w23)
                _st_shared_v2(slg + (u * num_experts + col) * 2, w01, w23)

    # ------------------------------------------------------------------------------------------------- quant
    @cute.jit
    def _quant_share(self, sx: Int32, gs: Float32, xq_ptr: Int64, xsf_ptr: Int64, row0, b, s, tidx) -> None:
        """NVFP4 (arithmetic of ``_nvfp4_quant_row``) of rows i % NB == b of the row group over this K slice, one
        16-element block per thread, from the x slice in shared memory."""
        nb: cutlass.Constexpr = self.nb
        blocks: cutlass.Constexpr = self.slice_k // 16
        k_extent: cutlass.Constexpr = self.k_extent
        n_blocks: cutlass.Constexpr = (self.rg // nb) * blocks
        for u in cutlass.range_constexpr(-(-n_blocks // THREADS)):
            idx = tidx + u * THREADS
            if idx < n_blocks:
                i = b + (idx // blocks) * nb
                cblk = idx % blocks
                saddr = sx + i * self.row_bytes + cblk * 32
                w0, w1, w2, w3 = _ld_shared_v4(saddr)
                w4, w5, w6, w7 = _ld_shared_v4(saddr + 16)
                lo, hi, sf = _nvfp4_block(w0, w1, w2, w3, w4, w5, w6, w7, gs)
                grow = Int64(row0 + i)
                kblk = Int64(s * blocks + cblk)
                _st_v2(xq_ptr + grow * Int64(k_extent // 2) + kblk * Int64(8), lo, hi)
                _st_u8(xsf_ptr + grow * Int64(k_extent // 16) + kblk, sf)

    # ------------------------------------------------------------------------------------------------- routing
    @cute.jit
    def _route_token(self, t, srow: Int32, ids_ptr: Int64, wts_ptr: Int64, state_ptr: Int64, local_start: Int32,
                     lane) -> None:
        """One warp: token t's 512 BF16 logits from shared memory (16 per lane), top-k and renormalization
        (bitwise q35_moe_front), ids / BF16 weights out, the local picks' bits into the token masks."""
        top_k: cutlass.Constexpr = self.top_k
        num_local: cutlass.Constexpr = self.num_local
        mask_words: cutlass.Constexpr = self.mask_words
        e0 = lane * 16
        p0, p1, p2, p3 = _ld_shared_v4(srow + lane * 32)
        p4, p5, p6, p7 = _ld_shared_v4(srow + lane * 32 + 16)
        packed = (p0, p1, p2, p3, p4, p5, p6, p7)
        keys = cute.make_rmem_tensor(cute.make_layout((16, )), Int32)
        for jw in cutlass.range_constexpr(8):
            keys[2 * jw] = _key1(packed[jw], e0 + 2 * jw)
            keys[2 * jw + 1] = _key1(packed[jw] >> Int32(16), e0 + 2 * jw + 1)
        for pi in cutlass.range_constexpr(len(_SORT16)):
            i, jj, imax = _SORT16[pi]
            x = keys[i]
            y = keys[jj]
            hi = _max_u32(x, y)
            lo = _min_u32(x, y)
            if const_expr(imax):
                keys[i] = hi
                keys[jj] = lo
            else:
                keys[i] = lo
                keys[jj] = hi
        sel = cute.make_rmem_tensor(cute.make_layout((top_k, )), Int32)
        val = cute.make_rmem_tensor(cute.make_layout((top_k, )), Float32)
        for k in cutlass.range_constexpr(top_k):
            head = keys[0]
            gmax = _redux_max_u32(head)
            sel[k] = Int32(65535) - (gmax & Int32(65535))
            val[k] = _key_val(gmax)
            if const_expr(k < top_k - 1):
                # pop: the owner lane's sorted keys move up by one (keys are unique)
                for jk in cutlass.range_constexpr(top_k - 1):
                    keys[jk] = _sel_eq(head, gmax, keys[jk + 1], keys[jk])
        # renormalization (route kernel arithmetic, as q35_moe_front._finish_token_v2)
        vmax = val[0]
        ex = cute.make_rmem_tensor(cute.make_layout((top_k, )), Float32)
        for k in cutlass.range_constexpr(top_k):
            ex[k] = _exp_tr(_sub(val[k], vmax))
        ssum = _add(_add(_add(_add(ex[0], ex[8]), ex[4]), _add(ex[2], ex[6])),
                    _add(_add(_add(ex[1], ex[9]), ex[5]), _add(ex[3], ex[7])))
        my_sel = sel[0]
        my_e = ex[0]
        for k in cutlass.range_constexpr(top_k - 1):
            my_sel = _sel_eq(lane, Int32(k + 1), sel[k + 1], my_sel)
            my_e = _sel_eq_f(lane, Int32(k + 1), ex[k + 1], my_e)
        if lane < top_k:
            o = Int64(t) * Int64(top_k) + Int64(lane)
            _st_b32(ids_ptr + o * Int64(4), my_sel)
            _st_b16(wts_ptr + o * Int64(2), _div_bf16(my_e, ssum))
            lid = my_sel - local_start
            pick = _sel_lt(lid, Int32(num_local), lid, Int32(-1))
            if pick >= Int32(0):
                _red_or_b32(state_ptr + (Int64(STATE_MASKS) + Int64(pick) * Int64(mask_words) + Int64(t // 32))
                            * Int64(4), _bit(t % 32))

    # ------------------------------------------------------------------------------------------------- tables
    @cute.jit
    def _tables(self, tab_ptr: Int64, ids_ptr: Int64, state_ptr: Int64, local_start: Int32, sx: Int32, epoch,
                tidx, warp, lane) -> None:
        """The CTA that completed the ticket (every row routed, its ids and mask bits released before the
        ticket): masks -> shared memory (and zeroed) and the ids -> registers (all loads in flight at once), warp 0
        the per-expert rows / CTAs, exclusive scan, CTA tables and totals, then one thread per expanded pick writes
        expanded -> permuted and permuted -> token (token order inside an expert's block); ticket to zero, epoch
        + 1."""
        num_rows: cutlass.Constexpr = self.num_rows
        top_k: cutlass.Constexpr = self.top_k
        num_local: cutlass.Constexpr = self.num_local
        tile: cutlass.Constexpr = self.tile
        mw: cutlass.Constexpr = self.mask_words
        nlw: cutlass.Constexpr = num_local * mw
        picks: cutlass.Constexpr = num_rows * top_k
        mpt: cutlass.Constexpr = -(-nlw // THREADS)
        ipt: cutlass.Constexpr = -(-picks // THREADS)
        smask = sx
        spref = sx + Int32(nlw * 4)
        soffs = spref + Int32(nlw * 4)
        mk = cute.make_rmem_tensor(cute.make_layout((mpt, )), Int32)
        idv = cute.make_rmem_tensor(cute.make_layout((ipt, )), Int32)
        for u in cutlass.range_constexpr(mpt):
            mk[u] = Int32(0)
            if tidx + u * THREADS < nlw:
                mk[u] = _ld_cg_b32(state_ptr + (Int64(STATE_MASKS) + Int64(tidx + u * THREADS)) * Int64(4))
        for u in cutlass.range_constexpr(ipt):
            idv[u] = Int32(0)
            if tidx + u * THREADS < picks:
                idv[u] = _ld_cg_b32(ids_ptr + Int64(tidx + u * THREADS) * Int64(4))
        for u in cutlass.range_constexpr(mpt):
            if tidx + u * THREADS < nlw:
                _st_shared(smask + (tidx + u * THREADS) * 4, mk[u])
                _st_b32(state_ptr + (Int64(STATE_MASKS) + Int64(tidx + u * THREADS)) * Int64(4), Int32(0))
        cute.arch.sync_threads()
        if warp == 0:
            epl: cutlass.Constexpr = num_local // 32
            cnt = cute.make_rmem_tensor(cute.make_layout((epl, )), Int32)
            nct = cute.make_rmem_tensor(cute.make_layout((epl, )), Int32)
            e_first = lane * epl
            total = Int32(0)
            for jx in cutlass.range_constexpr(epl):
                e = e_first + jx
                run = Int32(0)
                for w in cutlass.range_constexpr(mw):
                    _st_shared(spref + (e * mw + w) * 4, run)
                    run = run + _popc(_ld_shared(smask + (e * mw + w) * 4))
                cnt[jx] = run
                nct[jx] = (run + Int32(tile - 1)) // Int32(tile)
                total = total + nct[jx]
            incl = total
            for d in cutlass.range_constexpr(5):
                incl = _scan_add(incl, 1 << d)
            offs = incl - total
            for jx in cutlass.range_constexpr(epl):
                e = e_first + jx
                _st_shared(soffs + e * 4, offs)
                for c in cutlass.range_constexpr(self.max_ctas_per_expert):
                    if Int32(c) < nct[jx]:
                        cta_i = offs + Int32(c)
                        _st_b32(tab_ptr + Int64(self.off_cte + cta_i) * Int64(4), e)
                        _st_b32(tab_ptr + Int64(self.off_ctl + cta_i) * Int64(4),
                                cutlass.min((cta_i + 1) * Int32(tile), offs * Int32(tile) + cnt[jx]))
                offs = offs + nct[jx]
            if lane == 31:
                _st_b32(tab_ptr + Int64(self.off_nct) * Int64(4), incl)
                _st_b32(tab_ptr + Int64(self.off_npd) * Int64(4), incl * Int32(tile))
        cute.arch.sync_threads()
        for u in cutlass.range_constexpr(ipt):
            i = tidx + u * THREADS
            if i < picks:
                tok = i // top_k
                lid = idv[u] - local_start
                pick = _sel_lt(lid, Int32(num_local), lid, Int32(-1))
                perm = Int32(-1)
                if pick >= Int32(0):
                    wslot = pick * mw + tok // 32
                    rank = _ld_shared(spref + wslot * 4) + _popc(_ld_shared(smask + wslot * 4) & _below(tok % 32))
                    perm = _ld_shared(soffs + pick * 4) * Int32(tile) + rank
                    _st_b32(tab_ptr + Int64(self.off_p2t + perm) * Int64(4), tok)
                _st_b32(tab_ptr + Int64(i) * Int64(4), perm)
        if tidx == 0:
            _st_b32(state_ptr + Int64(STATE_TICKET * 4), Int32(0))
            _st_b32(state_ptr + Int64(STATE_EPOCH * 4), epoch + Int32(1))


# ---------------------------------------------------------------------------------------------------------
# Route-only mode (TRTLLM_Q35_MOE_FRONT_BIG=route): the router logits come from the stock router GEMM (cuBLAS);
# routing + tables + NVFP4 x quantization in one launch of thread-block clusters.
# ---------------------------------------------------------------------------------------------------------
ROUTE_CLUSTER = 8
ROUTE_THREADS = 1024


@dsl_user_op
def _cluster_arrive_release(*, loc=None, ip=None) -> None:
    _asm(None, [], "barrier.cluster.arrive.release.aligned;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _cluster_wait_acquire(*, loc=None, ip=None) -> None:
    _asm(None, [], "barrier.cluster.wait.acquire.aligned;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _ld_dsmem_b32(addr: Int32, rank: Int32, *, loc=None, ip=None) -> Int32:
    """The word at shared-memory address ``addr`` of cluster CTA ``rank``."""
    return Int32(_asm(T.i32(), [_i32(addr, loc, ip), _i32(rank, loc, ip)],
                      "{\n\t.reg .b32 ra;\n\tmapa.shared::cluster.u32 ra, $1, $2;\n\t"
                      "ld.shared::cluster.b32 $0, [ra];\n\t}", "=r,r,r,~{memory}", loc=loc, ip=ip))


@dsl_user_op
def _ld_cg_v4w(addr: Int64, *, loc=None, ip=None):
    out = _asm(_struct(4), [_i64(addr, loc, ip)], "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
               "=r,=r,=r,=r,l,~{memory}", loc=loc, ip=ip)
    return _unpack(out, 4, loc, ip)


@dsl_user_op
def _cluster_arrive_relaxed(*, loc=None, ip=None) -> None:
    _asm(None, [], "barrier.cluster.arrive.relaxed.aligned;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _shfl_idx_f32(x: Float32, src: int, *, loc=None, ip=None) -> Float32:
    """``x`` of lane ``src`` (whole warp)."""
    return Float32(_asm(T.f32(), [_f32(x, loc, ip)], f"shfl.sync.idx.b32 $0, $1, {src}, 31, 0xffffffff;", "=f,f",
                        loc=loc, ip=ip))


@dsl_user_op
def _vote_any(x: Int32, *, loc=None, ip=None) -> Int32:
    """1 if ``x != 0`` in any lane of the warp, else 0."""
    return Int32(_asm(T.i32(), [_i32(x, loc, ip)],
                      "{\n\t.reg .pred p, q;\n\tsetp.ne.u32 p, $1, 0;\n\tvote.sync.any.pred q, p, 0xffffffff;\n\t"
                      "selp.u32 $0, 1, 0, q;\n\t}", "=r,r", loc=loc, ip=ip))


def _cas(a, b):
    """(max, min) of two keys (unsigned)."""
    return _max_u32(a, b), _min_u32(a, b)


def _merge22(x, y):
    """Two descending pairs -> the descending 4."""
    c0, m1 = _cas(x[0], y[0])
    m2, c3 = _cas(x[1], y[1])
    c1, c2 = _cas(m1, m2)
    return (c0, c1, c2, c3)


def _top4of8(a, b):
    """The 4 largest of two descending 4s, descending (bitonic: max(a_i, b_3-i), then a 4-element bitonic sort)."""
    t0 = _max_u32(a[0], b[3])
    t1 = _max_u32(a[1], b[2])
    t2 = _max_u32(a[2], b[1])
    t3 = _max_u32(a[3], b[0])
    u0, u2 = _cas(t0, t2)
    u1, u3 = _cas(t1, t3)
    v0, v1 = _cas(u0, u1)
    v2, v3 = _cas(u2, u3)
    return (v0, v1, v2, v3)


class Q35MoeRouteBigKernel:
    """Route-only big front: router logits [M, E] BF16 from the stock GEMM -> top-k ids / BF16 weights, the
    trtllm-gen pre-routed tables (``q35_prerouted_sizes`` order, token order inside an expert's block) and the
    launch of ``1 + quant_clusters`` clusters of 8 CTAs (``threads`` each, default 1024):

    * cluster 0 routes: CTA c takes tokens [c TPC, (c + 1) TPC) (TPC = ceil(M / 8) <= 32), one warp per token (two
      interleaved when TPC > #warps; the routing arithmetic of :mod:`q35_moe_front`); the local picks set the token's
      bit in the CTA's per-expert token mask (shared memory). After a cluster barrier every routing CTA reads the 8
      CTAs' per-expert counts through distributed shared memory, scans the CTAs per expert (``tile``) and writes
      the expanded -> permuted / permuted -> token entries of its own picks (CTA 0 also the CTA tables and totals);
    * the other clusters NVFP4-quantize x (``fp4_quantize`` arithmetic, linear scale layout).

    No CTA waits for another cluster, so nothing needs the clusters co-resident. PDL: everything after
    ``griddepcontrol.wait``; every CTA releases its dependents after a barrier that follows its last global
    access."""

    def __init__(self, *, num_rows: int, k_extent: int, num_experts: int, top_k: int, num_local: int, tile: int,
                 a_stride: int, use_pdl: bool, quant_clusters: int = 7, threads: int = ROUTE_THREADS,
                 trace: bool = False) -> None:
        if threads % 32 or not 256 <= threads <= 1024:
            raise ValueError(f"threads={threads} must be a multiple of 32 in [256, 1024]")
        if not 1 <= num_rows <= ROUTE_CLUSTER * 32:
            raise ValueError(f"num_rows={num_rows} must be in [1, {ROUTE_CLUSTER * 32}]")
        if top_k != 10:
            raise ValueError("the renormalization tree is the route kernel's top-10 tree")
        if num_experts != 32 * 16:
            raise ValueError("one warp per token holds 16 logits per lane: 512 experts")
        if num_local % 32 or not 32 <= num_local <= min(num_experts, threads):
            raise ValueError(f"num_local={num_local} must be a multiple of 32 in [32, {threads}]")
        if tile & (tile - 1) or not 8 <= tile <= 256:
            raise ValueError(f"tile={tile} must be a power of two in [8, 256]")
        if k_extent % 16 or a_stride % 8 or a_stride < k_extent:
            raise ValueError("x rows must be 16-byte aligned blocks of 16")
        if quant_clusters < 1:
            raise ValueError("at least one quantization cluster")
        self.num_rows = num_rows
        self.k_extent = k_extent
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local = num_local
        self.tile = tile
        self.a_stride = a_stride
        self.use_pdl = use_pdl
        self.trace = trace
        self.threads = threads
        self.tpc = -(-num_rows // ROUTE_CLUSTER)
        self.warps = threads // 32
        self.tpw = -(-self.tpc // self.warps)
        if self.tpw > 2:
            raise ValueError("at most two tokens per warp")
        self.quant_clusters = quant_clusters
        self.ctas = ROUTE_CLUSTER * (1 + quant_clusters)
        self.q_ctas = ROUTE_CLUSTER * quant_clusters
        self.blocks = num_rows * (k_extent // 16)
        self.q_passes = -(-self.blocks // (self.q_ctas * threads))
        self.max_ctas = max_num_tiles(num_rows, top_k, num_local, tile)
        self.max_ctas_per_expert = -(-num_rows // tile)
        self.off_p2t = num_rows * top_k
        self.off_cte = self.off_p2t + (self.max_ctas + 1) * tile
        self.off_ctl = self.off_cte + self.max_ctas
        self.off_nct = self.off_ctl + self.max_ctas
        self.off_npd = self.off_nct + 1
        # shared memory words: mask, count, prefix (earlier CTAs), total, block offset [NL] each, warp sums [32],
        # picks [TPC * K]
        nl = num_local
        self.o_mask = 0
        self.o_cnt = nl
        self.o_pre = 2 * nl
        self.o_tot = 3 * nl
        self.o_offs = 4 * nl
        self.o_wsum = 5 * nl
        self.o_pick = 5 * nl + 32
        self.smem_words = self.o_pick + self.tpc * top_k + 16

    @cute.jit
    def __call__(self, logits_ptr: Int64, x_ptr: Int64, gs_ptr: Int64, xq_ptr: Int64, xsf_ptr: Int64, ids_ptr: Int64,
                 wts_ptr: Int64, tab_ptr: Int64, trace_ptr: Int64, local_start: Int32, stream) -> None:
        self.kernel(logits_ptr, x_ptr, gs_ptr, xq_ptr, xsf_ptr, ids_ptr, wts_ptr, tab_ptr, trace_ptr,
                    local_start).launch(
            grid=[self.ctas, 1, 1],
            block=[self.threads, 1, 1],
            cluster=[ROUTE_CLUSTER, 1, 1],
            smem=self.smem_words * 4,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(self, logits_ptr: Int64, x_ptr: Int64, gs_ptr: Int64, xq_ptr: Int64, xsf_ptr: Int64, ids_ptr: Int64,
               wts_ptr: Int64, tab_ptr: Int64, trace_ptr: Int64, local_start: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tr = cute.make_rmem_tensor(cute.make_layout((TRACE_SLOTS, )), Int64)
        if const_expr(self.trace):
            for i in cutlass.range_constexpr(8):
                tr[i] = Int64(0)
            tr[0] = _gtime()
        smem = _SmemAllocator()
        buf = smem.allocate_tensor(Int32, cute.make_layout((self.smem_words, )), byte_alignment=16)
        base = _smem_addr(buf.iterator)
        if bidx < ROUTE_CLUSTER:
            self._route_cta(logits_ptr, ids_ptr, wts_ptr, tab_ptr, local_start, base, bidx, tidx, tr)
        else:
            self._quant_cta(x_ptr, gs_ptr, xq_ptr, xsf_ptr, bidx - ROUTE_CLUSTER, tidx, tr)
        if const_expr(self.trace):
            if tidx == 0:
                tr[7] = _gtime()
                for i in cutlass.range_constexpr(8):
                    _st_u64(trace_ptr + (Int64(bidx) * Int64(8) + Int64(i)) * Int64(8), tr[i])

    # ------------------------------------------------------------------------------------------------- quant
    @cute.jit
    def _quant_cta(self, x_ptr: Int64, gs_ptr: Int64, xq_ptr: Int64, xsf_ptr: Int64, qc, tidx, tr) -> None:
        bpr: cutlass.Constexpr = self.k_extent // 16
        gs = _ld_f32(gs_ptr)  # static (the experts' input scale): before the wait
        if const_expr(self.use_pdl):
            _gdc_wait()
        if const_expr(self.trace):
            tr[1] = _gtime()
        for p in cutlass.range_constexpr(self.q_passes):
            blk = (p * self.q_ctas + qc) * self.threads + tidx
            if blk < self.blocks:
                row = blk // bpr
                cb = blk % bpr
                src = x_ptr + (Int64(row) * Int64(self.a_stride) + Int64(cb) * Int64(16)) * Int64(2)
                w0, w1, w2, w3 = _ld_cg_v4w(src)
                w4, w5, w6, w7 = _ld_cg_v4w(src + Int64(16))
                lo, hi, sf = _nvfp4_block(w0, w1, w2, w3, w4, w5, w6, w7, gs)
                _st_v2(xq_ptr + Int64(row) * Int64(self.k_extent // 2) + Int64(cb) * Int64(8), lo, hi)
                _st_u8(xsf_ptr + Int64(row) * Int64(bpr) + Int64(cb), sf)
        cute.arch.sync_threads()
        if const_expr(self.use_pdl):
            _gdc_launch()

    # ------------------------------------------------------------------------------------------------- routing
    @cute.jit
    def _route_cta(self, logits_ptr: Int64, ids_ptr: Int64, wts_ptr: Int64, tab_ptr: Int64, local_start: Int32,
                   base: Int32, c, tidx, tr) -> None:
        num_local: cutlass.Constexpr = self.num_local
        top_k: cutlass.Constexpr = self.top_k
        tile: cutlass.Constexpr = self.tile
        tpc: cutlass.Constexpr = self.tpc
        warp = tidx // 32
        lane = tidx % 32
        smask = base + Int32(self.o_mask * 4)
        scnt = base + Int32(self.o_cnt * 4)
        spre = base + Int32(self.o_pre * 4)
        stot = base + Int32(self.o_tot * 4)
        soffs = base + Int32(self.o_offs * 4)
        swsum = base + Int32(self.o_wsum * 4)
        spick = base + Int32(self.o_pick * 4)
        if tidx < num_local:
            _st_shared(smask + tidx * 4, Int32(0))
        # this CTA's tokens [c TPC, c TPC + n_tok)
        t_first = c * tpc
        n_tok = cutlass.min(Int32(tpc), Int32(self.num_rows) - t_first)
        if const_expr(self.use_pdl):
            _gdc_wait()
        cute.arch.sync_threads()
        if const_expr(self.trace):
            tr[1] = _gtime()
        my_sel = Int32(0)
        my_w = Int32(0)
        if const_expr(self.tpw == 1):
            # one token per warp; its ids / weights are stored after the cluster barrier (the barrier's release
            # then only waits for shared memory)
            if warp < n_tok:
                my_sel, my_w = self._route_warp_fast(logits_ptr, local_start, smask, spick, t_first, warp, lane)
        else:
            if warp < n_tok:
                self._route_warp(logits_ptr, ids_ptr, wts_ptr, local_start, smask, spick, t_first, n_tok, warp,
                                 lane)
        cute.arch.sync_threads()
        if const_expr(self.trace):
            tr[2] = _gtime()
        if tidx < num_local:
            _st_shared(scnt + tidx * 4, _popc(_ld_shared(smask + tidx * 4)))
        # ---- the 8 CTAs' counts through distributed shared memory ----
        _cluster_arrive_release()
        _cluster_wait_acquire()
        if const_expr(self.trace):
            tr[3] = _gtime()
        if const_expr(self.tpw == 1):
            if warp < n_tok:
                if lane < top_k:
                    o = Int64(t_first + warp) * Int64(top_k) + Int64(lane)
                    _st_b32(ids_ptr + o * Int64(4), my_sel)
                    _st_b16(wts_ptr + o * Int64(2), my_w)
        tot = Int32(0)
        nct = Int32(0)
        e = tidx
        if e < num_local:
            cnts = []
            for k in cutlass.range_constexpr(ROUTE_CLUSTER):
                cnts.append(_ld_dsmem_b32(scnt + e * 4, Int32(k)))
            pre = Int32(0)
            for k in cutlass.range_constexpr(ROUTE_CLUSTER):
                tot = tot + cnts[k]
                pre = pre + _sel_lt(Int32(k), c, cnts[k], Int32(0))
            _st_shared(spre + e * 4, pre)
            _st_shared(stot + e * 4, tot)
            nct = (tot + Int32(tile - 1)) // Int32(tile)
        # every peer's counts are read (the matching wait is at the end: this CTA's counts stay until then)
        _cluster_arrive_relaxed()
        # exclusive scan of the CTAs per expert over the num_local experts (one per thread)
        incl = nct
        for d in cutlass.range_constexpr(5):
            incl = _scan_add(incl, 1 << d)
        if lane == 31:
            _st_shared(swsum + warp * 4, incl)
        cute.arch.sync_threads()
        nwe: cutlass.Constexpr = -(-num_local // 32)  # warps holding experts
        wpre = Int32(0)
        total = Int32(0)
        for w in cutlass.range_constexpr(nwe):
            ws = _ld_shared(swsum + w * 4)
            wpre = wpre + _sel_lt(Int32(w), warp, ws, Int32(0))
            total = total + ws
        offs = incl - nct + wpre
        if e < num_local:
            _st_shared(soffs + e * 4, offs)
            if c == 0:
                for cc in cutlass.range_constexpr(self.max_ctas_per_expert):
                    if Int32(cc) < nct:
                        cta_i = offs + Int32(cc)
                        _st_b32(tab_ptr + Int64(self.off_cte + cta_i) * Int64(4), e)
                        _st_b32(tab_ptr + Int64(self.off_ctl + cta_i) * Int64(4),
                                cutlass.min((cta_i + 1) * Int32(tile), offs * Int32(tile) + tot))
        if c == 0:
            if tidx == 0:
                _st_b32(tab_ptr + Int64(self.off_nct) * Int64(4), total)
                _st_b32(tab_ptr + Int64(self.off_npd) * Int64(4), total * Int32(tile))
        cute.arch.sync_threads()
        if const_expr(self.trace):
            tr[4] = _gtime()
        # ---- this CTA's picks: expanded -> permuted, permuted -> token (token order inside a block) ----
        for u in cutlass.range_constexpr(-(-(tpc * top_k) // self.threads)):
            i = tidx + u * self.threads
            if i < n_tok * top_k:
                tl = i // top_k
                tok = t_first + tl
                pick = _ld_shared(spick + i * 4)
                perm = Int32(-1)
                if pick >= Int32(0):
                    rank = _ld_shared(spre + pick * 4) + _popc(_ld_shared(smask + pick * 4) & _below(tl))
                    perm = _ld_shared(soffs + pick * 4) * Int32(tile) + rank
                    _st_b32(tab_ptr + Int64(self.off_p2t + perm) * Int64(4), tok)
                _st_b32(tab_ptr + (Int64(tok) * Int64(top_k) + Int64(i - tl * top_k)) * Int64(4), perm)
        if const_expr(self.trace):
            cute.arch.sync_threads()
            tr[5] = _gtime()
        _cluster_wait_acquire()  # every peer has read this CTA's counts
        cute.arch.sync_threads()
        if const_expr(self.trace):
            tr[6] = _gtime()
        if const_expr(self.use_pdl):
            _gdc_launch()

    @cute.jit
    def _route_warp_fast(self, logits_ptr: Int64, local_start: Int32, smask: Int32, spick: Int32, t_first, warp,
                         lane):
        """Token ``t_first + warp`` (one per warp): the same top-k / renormalization bits as :meth:`_route_warp`
        with fewer instructions. Each lane keeps its 4 largest keys (a selection network instead of the 16-key
        bitonic sort) and the 10 rounds pop from those; a lane that runs dry before the last round (it holds 5 or
        more of the top 10) sends the warp to the full sort. Lane k computes only exp(v_k - v_0) and the warp
        gathers the 10 terms by shuffles into the route kernel's summation tree. Local picks go into the CTA's
        token masks and pick list; returns (expert id, BF16 weight bits) of pick ``lane`` (lanes < top_k)."""
        top_k: cutlass.Constexpr = self.top_k
        num_local: cutlass.Constexpr = self.num_local
        num_experts: cutlass.Constexpr = self.num_experts
        e0 = lane * 16
        src = logits_ptr + (Int64(t_first + warp) * Int64(num_experts) + Int64(e0)) * Int64(2)
        p0, p1, p2, p3 = _ld_cg_v4w(src)
        p4, p5, p6, p7 = _ld_cg_v4w(src + Int64(16))
        packed = (p0, p1, p2, p3, p4, p5, p6, p7)
        pairs = []
        for jw in cutlass.range_constexpr(8):
            klo, khi = _keys2(packed[jw], e0 + 2 * jw)
            pairs.append(_cas(klo, khi))
        quads = []
        for qd in cutlass.range_constexpr(4):
            quads.append(_merge22(pairs[2 * qd], pairs[2 * qd + 1]))
        c0, c1, c2, c3 = _top4of8(_top4of8(quads[0], quads[1]), _top4of8(quads[2], quads[3]))
        sel = cute.make_rmem_tensor(cute.make_layout((top_k, )), Int32)
        val = cute.make_rmem_tensor(cute.make_layout((top_k, )), Float32)
        zero = Int32(0)
        bad = Int32(0)
        for k in cutlass.range_constexpr(top_k):
            gmax = _redux_max_u32(c0)
            sel[k] = Int32(65535) - (gmax & Int32(65535))
            val[k] = _key_val(gmax)
            if const_expr(k < top_k - 1):
                # the owner lane's candidates move up by one (keys are unique)
                c0, c1, c2, c3 = (_sel_eq(c0, gmax, c1, c0), _sel_eq(c0, gmax, c2, c1), _sel_eq(c0, gmax, c3, c2),
                                  _sel_eq(c0, gmax, zero, c3))
            if const_expr(k == top_k - 2):
                bad = _vote_any(_sel_eq(c0, zero, Int32(1), zero))
        if bad != 0:
            self._topk_full(packed, e0, sel, val)
        my_sel = sel[0]
        my_v = val[0]
        for k in cutlass.range_constexpr(top_k - 1):
            my_sel = _sel_eq(lane, Int32(k + 1), sel[k + 1], my_sel)
            my_v = _sel_eq_f(lane, Int32(k + 1), val[k + 1], my_v)
        my_e = _exp_tr(_sub(my_v, val[0]))
        ex = []
        for k in cutlass.range_constexpr(top_k):
            ex.append(_shfl_idx_f32(my_e, k))
        ssum = _add(_add(_add(_add(ex[0], ex[8]), ex[4]), _add(ex[2], ex[6])),
                    _add(_add(_add(ex[1], ex[9]), ex[5]), _add(ex[3], ex[7])))
        my_w = _div_bf16(my_e, ssum)
        if lane < top_k:
            lid = my_sel - local_start
            pick = _sel_lt(lid, Int32(num_local), lid, Int32(-1))
            if pick >= Int32(0):
                _red_or_shared(smask + pick * 4, _bit(warp))
            _st_shared(spick + (warp * top_k + lane) * 4, pick)
        return my_sel, my_w

    @cute.jit
    def _topk_full(self, packed, e0, sel: cute.Tensor, val: cute.Tensor) -> None:
        """The full per-lane sort + pop top-k (as :meth:`_route_warp`) into sel / val: the rare fallback of the
        candidate path."""
        top_k: cutlass.Constexpr = self.top_k
        keys = cute.make_rmem_tensor(cute.make_layout((16, )), Int32)
        for jw in cutlass.range_constexpr(8):
            keys[2 * jw] = _key1(packed[jw], e0 + 2 * jw)
            keys[2 * jw + 1] = _key1(packed[jw] >> Int32(16), e0 + 2 * jw + 1)
        for pi in cutlass.range_constexpr(len(_SORT16)):
            i, jj, imax = _SORT16[pi]
            x = keys[i]
            y = keys[jj]
            hi = _max_u32(x, y)
            lo = _min_u32(x, y)
            if const_expr(imax):
                keys[i] = hi
                keys[jj] = lo
            else:
                keys[i] = lo
                keys[jj] = hi
        for k in cutlass.range_constexpr(top_k):
            head = keys[0]
            gmax = _redux_max_u32(head)
            sel[k] = Int32(65535) - (gmax & Int32(65535))
            val[k] = _key_val(gmax)
            if const_expr(k < top_k - 1):
                for jk in cutlass.range_constexpr(top_k - 1):
                    keys[jk] = _sel_eq(head, gmax, keys[jk + 1], keys[jk])

    @cute.jit
    def _route_warp(self, logits_ptr: Int64, ids_ptr: Int64, wts_ptr: Int64, local_start: Int32, smask: Int32,
                    spick: Int32, t_first, n_tok, warp, lane) -> None:
        """Tokens warp and warp + #warps (when TPC > #warps; clamped to the first, not stored when absent):
        logits from global memory (16 per lane), top-k / renormalization bitwise q35_moe_front, ids / BF16 weights
        out, local picks into the CTA's token masks and pick list."""
        top_k: cutlass.Constexpr = self.top_k
        num_local: cutlass.Constexpr = self.num_local
        num_experts: cutlass.Constexpr = self.num_experts
        tpw: cutlass.Constexpr = self.tpw
        e0 = lane * 16
        tl1 = warp
        if const_expr(tpw == 2):
            tl1 = _sel_lt(warp + Int32(self.warps), n_tok, warp + Int32(self.warps), warp)
        keys = cute.make_rmem_tensor(cute.make_layout((16 * tpw, )), Int32)
        for rr in cutlass.range_constexpr(tpw):
            tl = warp
            if const_expr(rr == 1):
                tl = tl1
            src = logits_ptr + (Int64(t_first + tl) * Int64(num_experts) + Int64(e0)) * Int64(2)
            p0, p1, p2, p3 = _ld_cg_v4w(src)
            p4, p5, p6, p7 = _ld_cg_v4w(src + Int64(16))
            packed = (p0, p1, p2, p3, p4, p5, p6, p7)
            for jw in cutlass.range_constexpr(8):
                keys[16 * rr + 2 * jw] = _key1(packed[jw], e0 + 2 * jw)
                keys[16 * rr + 2 * jw + 1] = _key1(packed[jw] >> Int32(16), e0 + 2 * jw + 1)
            for pi in cutlass.range_constexpr(len(_SORT16)):
                i, jj, imax = _SORT16[pi]
                x = keys[16 * rr + i]
                y = keys[16 * rr + jj]
                hi = _max_u32(x, y)
                lo = _min_u32(x, y)
                if const_expr(imax):
                    keys[16 * rr + i] = hi
                    keys[16 * rr + jj] = lo
                else:
                    keys[16 * rr + i] = lo
                    keys[16 * rr + jj] = hi
        sel = cute.make_rmem_tensor(cute.make_layout((top_k * tpw, )), Int32)
        val = cute.make_rmem_tensor(cute.make_layout((top_k * tpw, )), Float32)
        for k in cutlass.range_constexpr(top_k):
            for rr in cutlass.range_constexpr(tpw):
                head = keys[16 * rr]
                gmax = _redux_max_u32(head)
                sel[top_k * rr + k] = Int32(65535) - (gmax & Int32(65535))
                val[top_k * rr + k] = _key_val(gmax)
                if const_expr(k < top_k - 1):
                    for jk in cutlass.range_constexpr(top_k - 1):
                        keys[16 * rr + jk] = _sel_eq(head, gmax, keys[16 * rr + jk + 1], keys[16 * rr + jk])
        for rr in cutlass.range_constexpr(tpw):
            tl = warp
            active = Int32(1)
            if const_expr(rr == 1):
                tl = tl1
                active = _sel_eq(tl1, warp, Int32(0), Int32(1))
            vmax = val[top_k * rr]
            ex = cute.make_rmem_tensor(cute.make_layout((top_k, )), Float32)
            for k in cutlass.range_constexpr(top_k):
                ex[k] = _exp_tr(_sub(val[top_k * rr + k], vmax))
            ssum = _add(_add(_add(_add(ex[0], ex[8]), ex[4]), _add(ex[2], ex[6])),
                        _add(_add(_add(ex[1], ex[9]), ex[5]), _add(ex[3], ex[7])))
            my_sel = sel[top_k * rr]
            my_e = ex[0]
            for k in cutlass.range_constexpr(top_k - 1):
                my_sel = _sel_eq(lane, Int32(k + 1), sel[top_k * rr + k + 1], my_sel)
                my_e = _sel_eq_f(lane, Int32(k + 1), ex[k + 1], my_e)
            if active != Int32(0):
                if lane < top_k:
                    o = Int64(t_first + tl) * Int64(top_k) + Int64(lane)
                    _st_b32(ids_ptr + o * Int64(4), my_sel)
                    _st_b16(wts_ptr + o * Int64(2), _div_bf16(my_e, ssum))
                    lid = my_sel - local_start
                    pick = _sel_lt(lid, Int32(num_local), lid, Int32(-1))
                    if pick >= Int32(0):
                        _red_or_shared(smask + pick * 4, _bit(tl))
                    _st_shared(spick + (tl * top_k + lane) * 4, pick)

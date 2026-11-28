# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA over an e4m3 paged K/V pool for MTP verify rows, request-grouped over a token
bitmap union: `qsa_paged_sparse_gqa_nvfp4_verify`'s kernel with an e4m3 operand path.

Derived from that op; read its docstring for the union, the launch plan and the roles. What
differs:

* a raw K or V row is 256 e4m3 bytes (16 `cp.async` chunks: sixteen loader threads a row) and
  there are no block scales, so a raw slot is K rows + V rows only (`RAW_LD` 272) and the ring
  has two raw slots instead of three; the f16 K/V slots and the union tables keep their sizes,
  the tables lose the scale-offset words;
* dequantize is `cvt.rn.f16x2.e4m3x2` alone (exact: every e4m3 value is an f16 value); the K and
  V global scales are 1;
* the group table's K/V row offsets are in units of the largest power of two (16..256 bytes)
  that divides the K strides (`_roff_shift`), so a word reaches 512 GiB into a plane;
* a block table wider than the union scratch of raw slot 0 (more than 612 pages of 32 tokens at
  four rows a request) takes the wide plan up to 8192 pages (an engine up to `max_seq_len`
  262016 with MTP-3: a wider table takes the per-row kernel): the rows'
  bitmaps live in the f16 K/V slots, idle while a pass builds its union, the block table is read
  from global memory and the union scan runs `rounds` rounds of `THREADS` 128-token units;
* a loader thread stays on one four-token group for its four tokens, so a group's table words
  are read once a tile instead of four times;
* the union scan stops at the pass's highest selected token: the rows' selections feed a
  `red.shared.max` of their tokens, and the rounds past it hold no selected token;
* the first pass's bitmap clear, O = 0, the q copies and the request indices run before the PDL
  wait. That relies on the launch in front of this one releasing its dependents only after its
  own wait (the family's top-k launches do), so that every earlier kernel has completed when a
  CTA starts;
* the routed floor is a caller's (`min_groups`, `SMALL_MIN_GROUPS` up to `MIN_GROUPS`), and a
  launch whose units fill more than half of the SMs takes `TWO_WAVE_SPLITS` splits a unit.

Not bitwise the per-row kernel (`qsa_paged_sparse_gqa`) nor the hash-union grouped op
(`qsa_sparse_gqa_grouped_cute__decode`): tiles hold the group's union and the softmax reference
max moves differently. A target has to call it.
"""

from __future__ import annotations

import os
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp import archs
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.activation.fused_sigmoid_mul_inplace import (  # noqa: E501
    fused_sigmoid_mul_inplace,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa import (  # noqa: E501
    qsa_paged_sparse_gqa,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_nvfp4_decode import (  # noqa: E501
    MAX_SPLITS,
    _asm,
    _bf2_hi,
    _bf2_lo,
    _bf2_pack,
    _cp16z,
    _e4m3x2,
    _f32_bits,
    _ir,
    _ldg1,
    _ldg4,
    _lds1,
    _lds4,
    _ptr,
    _stg1,
    _sts1,
    _sts4,
)

from .qsa_nvfp4_verify_ptx import (
    ctz,
    groups8,
    popc,
    red_or_shared,
    spread4,
    stg4,
    stg4f,
    tcgen05_fence_after,
    tcgen05_fence_before,
)
from .qsa_paged_sparse_gqa_nvfp4_verify import _compile_merge, _num_sms, _phase, _sigmoid_fast
from .qsa_paged_sparse_gqa_nvfp4_verify import _VerifyKernel as _Nvfp4VerifyKernel

try:
    from cutlass.memory import SmemAllocator
except ImportError:  # DSL releases before the memory namespace split
    from cutlass.utils import SmemAllocator

F16 = cutlass.Float16
LOG2E = 1.4426950408889634


@dsl_user_op
def _red_max_shared(saddr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, Int32, loc, ip), _ir(v, Int32, loc, ip)],
        "{ red.shared.max.s32 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def tmem_ld16x2(taddr, n: int, off: int, *, loc=None, ip=None):
    """`tcgen05.ld.16x32bx2.x<n>` (32-bit values): threads 0-15 of the warp read lanes
    taddr.lane + i at columns taddr.col .. + n - 1, threads 16-31 the same lanes at columns
    taddr.col + off .. + n - 1. `tcgen05.wait::ld` (`fence_view_async_tmem_load`) before use."""
    regs = ", ".join(f"${i}" for i in range(n))
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * n),
        [_ir(taddr, Int32, loc, ip)],
        f"tcgen05.ld.sync.aligned.16x32bx2.x{n}.b32 {{{regs}}}, [${n}], {off};",
        "=f," * n + "r",
        loc,
        ip,
        side=True,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(n)]


@dsl_user_op
def tmem_st16x2(taddr, vals, off: int, *, loc=None, ip=None):
    """`tcgen05.st.16x32bx2.x<len(vals)>`: the store of `tmem_ld16x2`'s layout."""
    n = len(vals)
    regs = ", ".join(f"${i + 2}" for i in range(n))
    _asm(
        T.i32(),
        [_ir(taddr, Int32, loc, ip)] + [_ir(v, Float32, loc, ip) for v in vals],
        f"{{ tcgen05.st.sync.aligned.16x32bx2.x{n}.b32 [$1], {off}, {{{regs}}}; mov.u32 $0, 0; }}",
        "=r,r" + ",f" * n,
        loc,
        ip,
        side=True,
    )


HEAD_DIM = 256
#: MMA rows a CTA (T verify rows x the KV head's q heads) and positions a tile.
BM = 128
BN = 64
#: Four-token groups a tile.
GPT = BN // 4
#: Bytes of a raw K or V row in shared memory (256 + 16 of padding: conflict-free reads).
RAW_LD = 272
RAW_SLOTS = 2
RAW_K = BN * RAW_LD
RAW_KS = 0
RAW_VS = 0
#: One raw slot: K rows, V rows.
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
#: Most splits of a last-wave unit (tail mode).
MAX_TAIL_SPLITS = 4
#: Request groups from which the routed entry takes the grouped kernel unless a caller asks for
#: fewer (`min_groups`): the other ops' floor.
MIN_GROUPS = 8
#: The fewest groups a caller may ask for. Measured on GB300 (two KV heads, four rows a request,
#: 512-block selections): the grouped kernel beat the per-row kernel and its gate launch by
#: 9.6-20.3 us a layer at 2..7 groups and lost 0.9 us at one group. A caller that asks below
#: `MIN_GROUPS` plans those launches on its first such call (`reachable_plans` from its floor).
SMALL_MIN_GROUPS = 2
#: Splits a unit of a launch whose units (group x KV head) fill more than half of the SMs but not
#: all of them, where the one-wave rule gives one CTA a unit and the launch lasts as long as its
#: widest union (39-76 request groups at 2 KV heads on 148-152 SMs): two waves of split units
#: and a merge. GB300, 45 requests a rank: -0.15 ms a step.
TWO_WAVE_SPLITS = 2
#: Opt-in: `TRTLLM_QWEN4_EXP_DP_GQA_SOFTMAX_SPLIT=1` puts verify row t's q heads on tensor-memory
#: lanes 32 t + h (h < 16), so all four softmax warps hold live rows, and splits each row's tile
#: between two threads (`tcgen05.ld/st .16x32bx2`: thread h takes positions 0-31, thread 16 + h
#: positions 32-63; `_VerifyKernel.softmax_tiles_s2`). The two threads agree on the tile max by one
#: shuffle, the second continues the first's row-sum chain, and the rescale and epilogue split O's
#: columns between them: S, the reference max, P, O and l are the default kernel's, bit for bit.
#: Read at import; the launch entries take `softmax_split=` to override it. Up to four rows a
#: request and 16 q heads a KV head; other geometries take the default kernel.
SOFTMAX_SPLIT_ENV = "TRTLLM_QWEN4_EXP_DP_GQA_SOFTMAX_SPLIT"
SOFTMAX_SPLIT = os.environ.get(SOFTMAX_SPLIT_ENV, "0").strip() == "1"


def _softmax_split_fits(rpr: int, group: int) -> bool:
    return rpr <= SOFT_WARPS and group <= 16


def _roff_shift(k_strides) -> int:
    """log2 of the unit (bytes) of the group table's K/V row offsets (int32 words): the largest
    power of two that divides the three K strides, 16 (the op's alignment contract) to 256 (a
    packed row). The engine's planes (page stride a whole number of 256-byte rows) get 256, so
    a table word reaches 512 GiB into a plane; at 16-byte units it wrapped past 32 GiB (a pool
    slot of 24 role pages: from page 87382), and the loader then dropped those tokens (word
    < 0) or, past 64 GiB, read another address."""
    sh = 4
    while sh < 8 and all(int(st) % (1 << (sh + 1)) == 0 for st in k_strides):
        sh += 1
    return sh


def _roff_fits(num_pages: int, k_strides, kv_heads: int, tpb: int) -> bool:
    """Whether every K/V row offset of the plane fits the group table's int32 word (in
    `_roff_shift` units). False: the grouped kernel cannot address the pool's last pages; the
    routed entry takes the per-row kernel."""
    ks0, ks1, ks2 = (int(st) for st in k_strides)
    last = (int(num_pages) - 1) * ks0 + (int(kv_heads) - 1) * ks1 + (int(tpb) - 1) * ks2
    return last >= 0 and (last >> _roff_shift(k_strides)) < 2**31 - 1


class _Layout(NamedTuple):
    """The shared-memory plan of one compile: bitmap words (`ngw` groups of four), the union's
    group capacity, local tiles a CTA may own in one pass, union-scratch offsets in raw slot 0
    and the table size in int32 words.
    """

    ngw: int
    nw: int
    ug_max: int
    tmax: int
    u_ug: int
    u_bt: int
    tab_words: int
    #: wide plan: the rows' bitmaps live in the f16 K slots (`rows_k` rows) and V slots, which
    #: are idle while a pass builds its union; the block table is read from global memory and
    #: the union scan takes `rounds` rounds of `THREADS` 128-token units.
    wide: bool = False
    rows_k: int = 0
    rounds: int = 1


def _layout(rpr: int, group: int, width: int, bt_width: int, tpb: int, splits: int):
    """The kernel's shared-memory plan for `rpr` rows a group over a `width`-column expanded
    selection and a `bt_width`-page block table, or None where it does not fit: the union
    scratch (each row's token bitmap, the union's group list and the request's block-table
    row) shares raw slot 0 and the staged q rows raw slot 1; where that does not fit, the wide
    plan (the module docstring).
    """
    if tpb % 4 or tpb & (tpb - 1) or rpr < 1 or rpr * group > BM:
        return None
    ngw = -(-(bt_width * tpb) // 128)
    nw = 4 * ngw
    # an expanded selection names at most ceil(width / 4) four-token groups a row
    ug_max = min(rpr * -(-width // 4), 32 * ngw)
    tmax = -(-(-(-ug_max // GPT)) // splits)
    u_ug = -(-(rpr * nw * 4) // 16) * 16
    u_bt = -(-(u_ug + ug_max * 4) // 16) * 16
    tab_words = 2 * tmax * GPT + 2 * tmax * rpr
    fits = (
        ngw <= THREADS
        and u_bt + bt_width * 4 <= RAW_BUF
        and RAW_BUF + rpr * group * Q_LD <= RAW_SLOTS * RAW_BUF
        and 4 * tab_words + RAW_SLOTS * RAW_BUF + KV_SLOTS_BYTES <= SMEM_BUDGET
    )
    if fits:
        return _Layout(ngw, nw, ug_max, tmax, u_ug, u_bt, tab_words)
    row_bytes = nw * 4
    rows_k = (KV_SLOTS_BYTES // 2) // row_bytes
    fits = (
        rpr <= 2 * rows_k
        and ug_max * 4 <= RAW_BUF
        and RAW_BUF + rpr * group * Q_LD <= RAW_SLOTS * RAW_BUF
        and 4 * tab_words + RAW_SLOTS * RAW_BUF + KV_SLOTS_BYTES <= SMEM_BUDGET
    )
    if not fits:
        return None
    return _Layout(
        ngw, nw, ug_max, tmax, 0, 0, tab_words, wide=True, rows_k=rows_k, rounds=-(-ngw // THREADS)
    )


class _VerifyKernel(_Nvfp4VerifyKernel):
    """The NVFP4 verify kernel's roles over e4m3 K/V rows (the module docstring). The pass
    rows, Q staging, the MMAs, softmax and epilogue are the base class's, unchanged."""

    def __init__(
        self,
        *,
        rpr,
        group,
        kv_heads,
        width,
        tpb,
        splits,
        tail,
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
        softmax_split=False,
    ):
        lay = _layout(rpr, group, width, bt_width, tpb, 1 if tail else splits)
        assert lay is not None, "no shared-memory plan for this geometry (see `_layout`)"
        self.T = rpr
        self.group, self.kvh, self.width, self.tpb = group, kv_heads, width, tpb
        self.splits = splits
        #: tail mode: CTAs below `n_whole` serve whole units, the rest `splits`-way splits
        self.tail = tail
        self.scale_log2 = scale * LOG2E
        self.num_pages, self.bt_width, self.num_requests = num_pages, bt_width, num_requests
        self.ks, self.kss, self.qs = k_strides, ks_strides, q_strides
        #: log2 of the unit of the table's K/V row offsets (`_roff_shift`)
        self.roff_sh = _roff_shift(k_strides)
        self.req64, self.gated, self.pdl = req64, gated, pdl
        self.heads = group * kv_heads
        self.rows_m = rpr * group
        #: softmax warps holding live MMA rows (the rest only keep the barriers' counts)
        self.live_warps = -(-self.rows_m // 32)
        #: lane split (`SOFTMAX_SPLIT`): MMA lane 32 t + h holds verify row t, q head h; thread
        #: 32 t + 16 s + h takes positions 32 s .. 32 s + 31 of each tile of that row
        self.split = bool(softmax_split)
        if self.split:
            assert _softmax_split_fits(rpr, group)
            self.live_warps = rpr
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
        self.wide, self.rows_k, self.rounds = lay.wide, lay.rows_k, lay.rounds
        # q rows (bf16, 528-byte rows: conflict-free) in raw slot 1 until Q is in tensor memory
        self.u_q = RAW_BUF
        # tables (int32 words). Per group: K/V row offset and scale offset of its first token
        # (`1 << roff_sh`-byte units, -1 when the group does not resolve) and its (token, row) selection
        # bytes; per (tile, row): the 64-bit selection word.
        self.t_roff = 0
        self.t_gmask = self.tmax * GPT
        self.t_bits = 2 * self.tmax * GPT
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
        part_ptr: cute.Pointer,
        lse_ptr: cute.Pointer,
        gate_ptr: cute.Pointer,
        rows: cutlass.Int32,
        gate_stride: cutlass.Int32,
        n_whole: cutlass.Int32,
        tail_row0: cutlass.Int32,
        n_ctas: cutlass.Int32,
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
            pass_done: cute.struct.MemRange[Int64, 1]
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
            part_ptr.toint(),
            lse_ptr.toint(),
            gate_ptr.toint(),
            rows,
            gate_stride,
            n_whole,
            tail_row0,
        ).launch(
            grid=[n_ctas, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.pdl,
        )

    # ---- pieces every role runs (every thread of the CTA, barrier 0) -----------------------

    def _bm_row(self, s_bm, s_bm2, t: int):
        """Shared-memory address of verify row `t`'s token bitmap (`nw` words): consecutive in
        the union scratch, or (wide plan) `rows_k` rows in the K slots and the rest in the V
        slots."""
        if not self.wide or t < self.rows_k:
            return s_bm + t * self.nw * 4
        return s_bm2 + (t - self.rows_k) * self.nw * 4

    @cute.jit
    def _union_wide_scan(self, tidx, warp, lane, s_bm, s_bm2, s_scan, s_ug):
        """The union's group list from the rows' bitmaps where the table has more 128-token
        units than the CTA has threads: `rounds` rounds of the one-round scan, ascending."""
        rpr: cutlass.Constexpr = self.T
        ngw: cutlass.Constexpr = self.ngw
        base = cutlass.Int32(0)
        # The rounds stop at the pass's highest selected token (scan word 31, `_union`): the
        # units past it hold no selected token, so the list is the full scan's.
        hi_tok = _lds1(s_scan + 124)
        n_rounds = cutlass.min(cutlass.Int32(self.rounds), ((hi_tok >> 7) + THREADS) // THREADS)
        for r in cutlass.range(n_rounds):
            unit = tidx + r * THREADS
            gmv = cutlass.Int32(0)
            if unit < ngw:
                for k in cutlass.range_constexpr(4):
                    xo = cutlass.Int32(0)
                    for t in cutlass.range_constexpr(rpr):
                        xo = xo | _lds1(self._bm_row(s_bm, s_bm2, t) + (4 * unit + k) * 4)
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
            n_r = cute.arch.warp_reduction_sum(tot)
            excl = base + before + incl - cnt
            if unit < ngw:
                xm = gmv
                for j in cutlass.range(cnt):
                    b = ctz(xm)
                    if excl + j < self.ug_max:
                        _sts1(s_ug + (excl + j) * 4, 32 * unit + b)
                    xm = xm & (xm - 1)
            base = base + n_r
            # the next round's warp totals overwrite `s_scan`
            cute.arch.sync_threads()
        return cutlass.min(base, self.ug_max)

    @cute.jit
    def _union(
        self,
        act,
        lead,
        r0,
        rows,
        sel_base,
        tidx,
        warp,
        lane,
        s_bm,
        s_bm2,
        s_scan,
        s_ug,
        s_bt,
        mBT,
        clr,
    ):
        rpr: cutlass.Constexpr = self.T
        nw: cutlass.Constexpr = self.nw
        width: cutlass.Constexpr = self.width
        ntok: cutlass.Constexpr = 32 * self.nw
        n_q: cutlass.Constexpr = -(-width // 4)  # four-column quads a row
        n_i: cutlass.Constexpr = -(-n_q // THREADS)
        # the highest selected token of the pass collects in scan word 31
        if tidx == 0:
            _sts1(s_scan + 124, cutlass.Int32(0))
        cute.arch.sync_threads()
        if const_expr(self.wide):
            if clr != 0:
                for i in cutlass.range(-(-nw // THREADS)):
                    idx = tidx + i * THREADS
                    if idx < nw:
                        for t in cutlass.range_constexpr(rpr):
                            _sts1(self._bm_row(s_bm, s_bm2, t) + idx * 4, cutlass.Int32(0))
        else:
            if clr != 0:
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
        # the request's block-table row, in the same round trip (wide plan: `_resolve` reads
        # its pages from global memory)
        n_bt: cutlass.Constexpr = 0 if self.wide else -(-self.bt_width // THREADS)
        btv = [mBT[lead, cutlass.min(tidx + i * THREADS, self.bt_width - 1)] for i in range(n_bt)]
        cute.arch.sync_threads()
        hi_tok = cutlass.Int32(0)
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
                    hi_tok = cutlass.max(
                        hi_tok, cutlass.Int32(cutlass.select_(okv[u], tok, cutlass.Int32(0)))
                    )
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
                        red_or_shared(self._bm_row(s_bm, s_bm2, t) + cutlass.max(wds[0], 0) * 4, m0)
                for u in cutlass.range_constexpr(1, 4):
                    if okv[u] & (wds[u] != wds[0]):
                        red_or_shared(self._bm_row(s_bm, s_bm2, t) + wds[u] * 4, bv[u])
        for i in cutlass.range_constexpr(n_bt):
            if tidx + i * THREADS < self.bt_width:
                _sts1(s_bt + (tidx + i * THREADS) * 4, btv[i])
        if hi_tok != 0:
            _red_max_shared(s_scan + 124, hi_tok)
        cute.arch.sync_threads()
        if const_expr(self.wide):
            n_ug = self._union_wide_scan(tidx, warp, lane, s_bm, s_bm2, s_scan, s_ug)
        else:
            n_ug = self._union_scan(tidx, warp, lane, s_bm, s_scan, s_ug)
        return n_ug

    @cute.jit
    def _union_scan(self, tidx, warp, lane, s_bm, s_scan, s_ug):
        """The union's group list from the rows' bitmaps, one thread a 128-token unit."""
        rpr: cutlass.Constexpr = self.T
        nw: cutlass.Constexpr = self.nw
        ngw: cutlass.Constexpr = self.ngw
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
    def _resolve(
        self, lead, kvh, split, nsplit, n_loc, n_ug, tidx, s_bm, s_bm2, s_ug, s_tab, s_bt, mBT
    ):
        """Every local tile's groups (entry of the group's first token, (token, row) selection
        bytes), then each (tile, verify row)'s 64-bit selection word.
        """
        rpr: cutlass.Constexpr = self.T
        nw: cutlass.Constexpr = self.nw
        tmax: cutlass.Constexpr = self.tmax
        n_rs: cutlass.Constexpr = -(-(tmax * GPT) // THREADS)
        s_roff = s_tab + self.t_roff * 4
        s_gmask = s_tab + self.t_gmask * 4
        s_bits = s_tab + self.t_bits * 4
        ks0, ks1, ks2 = self.ks
        pages = []
        gids = []
        for i in cutlass.range_constexpr(n_rs):
            slot = tidx + i * THREADS
            kl = slot // GPT
            ugi = (split + kl * nsplit) * GPT + slot % GPT
            gid = _lds1(s_ug + cutlass.min(ugi, self.ug_max - 1) * 4)
            gid = cutlass.min(cutlass.max(gid, 0), 8 * nw - 1)
            gids.append(gid)
            if const_expr(self.wide):
                pages.append(mBT[lead, cutlass.min((gid * 4) // self.tpb, self.bt_width - 1)])
            else:
                pages.append(
                    _lds1(s_bt + cutlass.min((gid * 4) // self.tpb, self.bt_width - 1) * 4)
                )
        for i in cutlass.range_constexpr(n_rs):
            slot = tidx + i * THREADS
            kl = slot // GPT
            ugi = (split + kl * nsplit) * GPT + slot % GPT
            gid = gids[i]
            page = pages[i]
            wsel = gid >> 3
            sh = (gid & 7) * 4
            mw = cutlass.Int32(0)
            for t in cutlass.range_constexpr(rpr):
                nib = (_lds1(self._bm_row(s_bm, s_bm2, t) + wsel * 4) >> sh) & 0xF
                mw = mw | (spread4(nib) << t)
            okp = (kl < n_loc) & (ugi < n_ug) & (page >= 0) & (page < self.num_pages)
            pg64 = cutlass.Int64(cutlass.max(page, 0))
            sl0 = (gid * 4) % self.tpb
            roff = (pg64 * ks0 + cutlass.Int64(kvh * ks1 + sl0 * ks2)) >> self.roff_sh
            roff = cutlass.Int32(cutlass.select_(okp, cutlass.Int32(roff), cutlass.Int32(-1)))
            mw = cutlass.Int32(cutlass.select_(okp, mw, cutlass.Int32(0)))
            if slot < tmax * GPT:
                _sts1(s_roff + slot * 4, roff)
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
        self,
        act,
        lead,
        kvh,
        r0,
        rows,
        sel_base,
        split,
        nsplit,
        tidx,
        warp,
        lane,
        s_raw,
        s_scan,
        s_tab,
        mBT,
        s_k16,
        s_v16,
        clr,
    ):
        """Union + resolve of one pass; returns the CTA's local tile count."""
        s_bm = s_raw + self.u_bm
        s_bm2 = s_bm
        if const_expr(self.wide):
            s_bm = s_k16
            s_bm2 = s_v16
        s_ug = s_raw + self.u_ug
        s_bt = s_raw + self.u_bt
        n_ug = self._union(
            act,
            lead,
            r0,
            rows,
            sel_base,
            tidx,
            warp,
            lane,
            s_bm,
            s_bm2,
            s_scan,
            s_ug,
            s_bt,
            mBT,
            clr,
        )
        n_tiles = (n_ug + GPT - 1) // GPT
        n_loc = cutlass.Int32(
            cutlass.select_(
                split < n_tiles,
                (n_tiles - split + nsplit - 1) // nsplit,
                cutlass.Int32(0),
            )
        )
        self._resolve(
            lead, kvh, split, nsplit, n_loc, n_ug, tidx, s_bm, s_bm2, s_ug, s_tab, s_bt, mBT
        )
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
        part_base: cutlass.Int64,
        lse_base: cutlass.Int64,
        gate_base: cutlass.Int64,
        rows: cutlass.Int32,
        gate_stride: cutlass.Int32,
        n_whole: cutlass.Int32,
        tail_row0: cutlass.Int32,
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
        if const_expr(self.tail):
            # units (group, KV head) below n_whole: one CTA each; the rest: `splits` CTAs each
            is_whole = bidx < n_whole
            t_cta = bidx - n_whole
            unit = cutlass.Int32(cutlass.select_(is_whole, bidx, n_whole + t_cta // splits))
            split = cutlass.Int32(cutlass.select_(is_whole, cutlass.Int32(0), t_cta % splits))
            nsplit = cutlass.Int32(
                cutlass.select_(is_whole, cutlass.Int32(1), cutlass.Int32(splits))
            )
        else:
            unit = bidx // splits
            split = bidx % splits
            nsplit = cutlass.Int32(splits)
        gi = unit // kvh_n
        kvh = unit % kvh_n
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
        pass_done = storage.pass_done.data_ptr()
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
            cute.arch.mbarrier_init(pass_done, 1)
        cute.arch.mbarrier_init_fence()
        if warp == MMA_WARP:
            cute.arch.alloc_tmem(TMEM_COLS, storage.tmem_holding.ptr)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, storage.tmem_holding.ptr)
        tb = cutlass.Int32(tmem_ptr.toint())

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
        # the thread that copies this MMA row's q into raw slot 1, and whether its own tensor
        # memory lane holds a live q row
        stager = m < self.rows_m
        q_live = live
        if const_expr(self.split):
            t_row = tidx // 32
            h = tidx % 16
            lane_ok = (h < group) & (t_row < rpr)
            m = t_row * group + cutlass.min(h, group - 1)
            live = lane_ok & (r0 + t_row < rows)
            stager = lane_ok & ((tidx % 32) < 16)
            q_live = live & ((tidx % 32) < 16)
        safe_row = cutlass.min(r0 + t_row, rows - 1)
        qs0, qs1 = self.qs
        q_head = kvh * group + cutlass.min(h, group - 1)
        q_row = q_base + (cutlass.Int64(safe_row) * qs0 + cutlass.Int64(q_head) * qs1) * 2
        q_s = s_raw + self.u_q + cutlass.min(m, self.rows_m - 1) * Q_LD

        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()

        if warp < SOFT_WARPS:
            # O = 0; this row's q copied into raw slot 1 now, into tensor memory (f16) after the
            # first pass's resolve. Dead rows stage zeros.
            rZ = cute.make_rmem_tensor(tLcO.shape, Float32)
            rZ.fill(0.0)
            for c in cutlass.range_constexpr(HEAD_DIM // CHUNK_COLS):
                cute.copy(
                    store_o, rZ, cute.make_tensor(tStO.iterator + c * CHUNK_COLS, tStO.layout)
                )
            cute.arch.fence_view_async_tmem_store()
            if stager:
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
        # prewait: the first pass's bitmaps cleared here (`_union` skips its clear once)
        if const_expr(self.wide):
            for i_pw in cutlass.range(-(-self.nw // THREADS)):
                idx_pw = tidx + i_pw * THREADS
                if idx_pw < self.nw:
                    for t_pw in cutlass.range_constexpr(rpr):
                        _sts1(
                            self._bm_row(s_k16, s_v16, t_pw) + idx_pw * 4,
                            cutlass.Int32(0),
                        )
        else:
            for i_pw in cutlass.range_constexpr(-(-(rpr * self.nw) // THREADS)):
                idx_pw = tidx + i_pw * THREADS
                if idx_pw < rpr * self.nw:
                    _sts1(s_raw + self.u_bm + idx_pw * 4, cutlass.Int32(0))
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
        n_done_pw = cutlass.Int32(0)
        neg_inf = -Float32.inf
        row_max = neg_inf
        row_sum = Float32(0.0)
        q_pending = cutlass.Int32(1)
        g = cutlass.Int32(0)
        n_pass = cutlass.Int32(0)
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
                    nsplit,
                    tidx,
                    warp,
                    lane,
                    s_raw,
                    s_scan,
                    s_tab,
                    mBT,
                    s_k16,
                    s_v16,
                    n_done_pw,
                )
                n_done_pw = n_done_pw + 1
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
                        self.stage_q(q_s, q_live, tview, q_full)
                        q_pending = cutlass.Int32(0)
                    if const_expr(self.split):
                        row_max, row_sum = self.softmax_tiles_s2(
                            g,
                            n_loc,
                            row_max,
                            row_sum,
                            live,
                            t_row,
                            tidx,
                            warp,
                            s_tab,
                            scale_log2,
                            tb,
                            s_full,
                            p_full,
                            o_full,
                        )
                    else:
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
                if const_expr(self.wide):
                    # the next pass's bitmaps overwrite the K/V slots: every MMA of this pass
                    # has read them once the MMA warp's commit arrives
                    if warp == MMA_WARP:
                        with cute.arch.elect_one():
                            tcgen05.commit(pass_done)
                    cute.arch.mbarrier_wait(pass_done, n_pass & 1)
                    n_pass = n_pass + 1
                cute.arch.sync_threads()  # [P] end of the pass
                handled = handled | act

        if warp == MMA_WARP:
            with cute.arch.elect_one():
                tcgen05.commit(o_final)
        if const_expr(self.split):
            if warp < SOFT_WARPS:
                cute.arch.cp_async_wait_group(0)
                self.epilogue_s2(
                    row_max,
                    row_sum,
                    live,
                    warp,
                    m,
                    r0,
                    rows,
                    kvh,
                    split,
                    nsplit,
                    tail_row0,
                    mGs,
                    scale_log2,
                    out_base,
                    part_base,
                    lse_base,
                    gate_base,
                    gate_stride,
                    tb,
                    o_final,
                    tidx,
                    s_k16,
                )
        else:
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
                    nsplit,
                    tail_row0,
                    mGs,
                    scale_log2,
                    out_base,
                    part_base,
                    lse_base,
                    gate_base,
                    gate_stride,
                    tview,
                    o_final,
                    tidx,
                    s_k16,
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
        """Raw tile copies: K and V rows sixteen threads a row (16-byte chunk lt % 16 of rows
        lt // 16 + 4 i: whole 256-byte rows a half-warp). Raw slot 1 holds the q rows until Q is
        in tensor memory.
        """
        s_roff = s_tab + self.t_roff * 4
        s_gmask = s_tab + self.t_gmask * 4
        tok16: cutlass.Constexpr = self.ks[2] // 16
        roff_unit: cutlass.Constexpr = 1 << self.roff_sh
        q16 = lt % 16
        for kl in cutlass.range(n_loc):
            gg = g + kl
            r = gg % RAW_SLOTS
            cute.arch.mbarrier_wait(raw_empty + r, _phase(gg, RAW_SLOTS) ^ 1)
            if gg >= 1:
                cute.arch.mbarrier_wait(q_full, 0)
            rb = s_raw + r * self.raw_buf
            rk = rb
            rv = rb + self.raw_k
            # ldhoist: thread (lt // 16, chunk q16) takes groups lt // 16 + 4 go, each for
            # its four tokens: the group's table words are read once
            for go in cutlass.range_constexpr(GPT // 4):
                gq = lt // 16 + 4 * go
                slot = kl * GPT + gq
                roff = _lds1(s_roff + slot * 4)
                mw = _lds1(s_gmask + slot * 4)
                okg = roff >= 0
                off0 = cutlass.Int64(cutlass.max(roff, 0)) * roff_unit + cutlass.Int64(q16 * 16)
                for tk in cutlass.range_constexpr(4):
                    n = 4 * gq + tk
                    ok = okg & (((mw >> (8 * tk)) & 0xFF) != 0)
                    nb = cutlass.Int32(cutlass.select_(ok, cutlass.Int32(16), cutlass.Int32(0)))
                    off = off0 + cutlass.Int64(tk * tok16 * 16)
                    _cp16z(rk + n * RAW_LD + 16 * q16, k_base + off, nb)
                    _cp16z(rv + n * RAW_LD + 16 * q16, v_base + off, nb)
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
        slot once the MMA has released it.
        """
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
            w = [_lds4(row_off + 16 * c) for c in range(16)]
            cute.arch.mbarrier_arrive(raw_empty + r)
            if is_v:
                cute.arch.mbarrier_wait(v_empty + b, _phase(gg, 2) ^ 1)
                self._decode_row(w, s_v16 + b * (BN * HEAD_DIM * 2), n)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(v_full + b)
            else:
                cute.arch.mbarrier_wait(k_empty + b, _phase(gg, 2) ^ 1)
                self._decode_row(w, s_k16 + b * (BN * HEAD_DIM * 2), n)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(k_full + b)

    @cute.jit
    def _decode_row(self, w, dst, n):
        """Raw e4m3 row words `w` (16 x 4) -> f16 SW128 row `n` at `dst`: 16-byte chunk ci (dims
        8 ci .. 8 ci + 7, raw words 2 ci and 2 ci + 1) of the row's 64-dim block ci // 8 at
        8192 (ci // 8) + 128 n + 16 ((ci % 8) ^ (n % 8)).
        """
        for c in cutlass.range_constexpr(16):
            for h in cutlass.range_constexpr(2):
                a = w[c][2 * h]
                b = w[c][2 * h + 1]
                ci = 2 * c + h
                _sts4(
                    dst + 8192 * (ci // 8) + 128 * n + 16 * ((ci % 8) ^ (n % 8)),
                    _e4m3x2(a),
                    _e4m3x2(a >> 16),
                    _e4m3x2(b),
                    _e4m3x2(b >> 16),
                )

    # ---- lane split (`SOFTMAX_SPLIT`): softmax and epilogue, two threads a row ------------
    @cute.jit
    def softmax_tiles_s2(
        self,
        g,
        n_loc,
        row_max,
        row_sum,
        live,
        t_row,
        tidx,
        warp,
        s_tab,
        scale_log2,
        tb,
        s_full,
        p_full,
        o_full,
    ):
        """`softmax_tiles` with each row's tile split between two threads (`.16x32bx2` tensor-memory
        accesses on lanes 32 t .. 32 t + 15): thread half s holds positions 32 s .. 32 s + 31.
        Bit for bit the default kernel's: the halves agree on the tile max (max is exact), apply
        the same reference-max rule, write P (f16) for their own positions, and the second half
        continues the first half's row-sum chain (the default kernel's order: two packed
        accumulators over positions 0, 4, 8, ...); the O rescale splits O's columns."""
        rpr: cutlass.Constexpr = self.T
        hb: cutlass.Constexpr = BN // 2
        s_bits = s_tab + self.t_bits * 4
        tau = Float32(RESCALE_TH)
        neg_inf = -Float32.inf
        live_warp = warp < self.live_warps
        half = (tidx // 16) % 2
        lanes = tb + ((warp * 32) << 16)
        for kl in cutlass.range(n_loc, unroll=1):
            gg = g + kl
            buf = gg % 2
            cute.arch.mbarrier_wait(s_full + buf, _phase(gg, 2))
            tcgen05_fence_after()
            if live_warp:
                word = cutlass.Int32(0)
                if live:
                    word = _lds1(s_bits + (kl * rpr + t_row) * 8 + 4 * half)
                sv = tmem_ld16x2(lanes + (TMEM_S0 + buf * BN), hb, hb)
                cute.arch.fence_view_async_tmem_load()
                rS = cute.make_rmem_tensor((hb,), Float32)
                tile_max = neg_inf
                for i in cutlass.range_constexpr(hb):
                    x = sv[i]
                    x = x if (word & (1 << i)) != 0 else neg_inf
                    rS[i] = x
                    tile_max = cute.arch.fmax(tile_max, x)
                tile_max = cute.arch.fmax(
                    tile_max, cute.arch.shuffle_sync_bfly(tile_max, offset=16)
                )
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
                    # O holds PV(..gg - 1): wait for it, then rescale this warp's rows, the first
                    # half's threads O columns 0-127, the second half's 128-255
                    cute.arch.mbarrier_wait(o_full, (gg - 1) % 2)
                    tcgen05_fence_after()
                    for c in cutlass.range_constexpr(HEAD_DIM // (2 * CHUNK_COLS)):
                        ov = tmem_ld16x2(
                            lanes + (TMEM_O + c * CHUNK_COLS), CHUNK_COLS, HEAD_DIM // 2
                        )
                        cute.arch.fence_view_async_tmem_load()
                        tmem_st16x2(
                            lanes + (TMEM_O + c * CHUNK_COLS),
                            [x * factor for x in ov],
                            HEAD_DIM // 2,
                        )
                    cute.arch.fence_view_async_tmem_store()
                row_max = use_max
                safe = use_max
                if use_max == neg_inf:
                    safe = Float32(0.0)
                neg = -safe * scale_log2
                for i in cutlass.range_constexpr(0, hb, 4):
                    e = cute.make_rmem_tensor((4,), Float32)
                    e[0], e[1] = cute.arch.fma_packed_f32x2(
                        (rS[i], rS[i + 1]), (scale_log2, scale_log2), (neg, neg)
                    )
                    e[2], e[3] = cute.arch.fma_packed_f32x2(
                        (rS[i + 2], rS[i + 3]), (scale_log2, scale_log2), (neg, neg)
                    )
                    for u in cutlass.range_constexpr(4):
                        rS[i + u] = cute.math.exp2(e[u], fastmath=True)
                # the row-sum chain: the first half from 0, the second from the first's partials
                a0 = (Float32(0.0), Float32(0.0))
                a1 = (Float32(0.0), Float32(0.0))
                for i in cutlass.range_constexpr(0, hb, 4):
                    a0 = cute.arch.add_packed_f32x2(a0, (rS[i], rS[i + 1]))
                    a1 = cute.arch.add_packed_f32x2(a1, (rS[i + 2], rS[i + 3]))
                b0 = (
                    cute.arch.shuffle_sync_bfly(a0[0], offset=16),
                    cute.arch.shuffle_sync_bfly(a0[1], offset=16),
                )
                b1 = (
                    cute.arch.shuffle_sync_bfly(a1[0], offset=16),
                    cute.arch.shuffle_sync_bfly(a1[1], offset=16),
                )
                for i in cutlass.range_constexpr(0, hb, 4):
                    b0 = cute.arch.add_packed_f32x2(b0, (rS[i], rS[i + 1]))
                    b1 = cute.arch.add_packed_f32x2(b1, (rS[i + 2], rS[i + 3]))
                ts = (b0[0] + b0[1]) + (b1[0] + b1[1])
                ts_other = cute.arch.shuffle_sync_bfly(ts, offset=16)
                tile_sum = ts if half != 0 else ts_other
                row_sum = row_sum * factor + tile_sum
                p16 = cute.make_rmem_tensor((hb,), F16)
                p16.store(rS.load().to(F16))
                pw = cute.make_tensor(
                    cute.recast_ptr(p16.iterator, dtype=Float32), cute.make_layout((hb // 2,))
                )
                tmem_st16x2(lanes + (TMEM_S0 + buf * BN), [pw[j] for j in range(hb // 2)], hb // 2)
                cute.arch.fence_view_async_tmem_store()
            tcgen05_fence_before()
            cute.arch.mbarrier_arrive(p_full + buf)
        return row_max, row_sum

    @cute.jit
    def epilogue_s2(
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
        nsplit,
        tail_row0,
        mGs,
        scale_log2,
        out_base,
        part_base,
        lse_base,
        gate_base,
        gate_stride,
        tb,
        o_final,
        st,
        s_stage,
    ):
        """`epilogue` over the lane split: the first half's thread of a row stores O columns
        0-127, the second 128-255 (O and l are the default kernel's, so the same values); the
        first writes the log2-sum-exp."""
        group: cutlass.Constexpr = self.group
        heads: cutlass.Constexpr = self.heads
        splits: cutlass.Constexpr = self.splits
        tailm: cutlass.Constexpr = self.tail
        has_whole: cutlass.Constexpr = tailm or splits == 1
        has_split: cutlass.Constexpr = tailm or splits > 1
        gated_whole: cutlass.Constexpr = has_whole and self.gated
        rows_m: cutlass.Constexpr = self.rows_m
        cpr: cutlass.Constexpr = HEAD_DIM // 8  # 16-byte chunks a row
        n_gc: cutlass.Constexpr = -(-(rows_m * cpr) // (32 * SOFT_WARPS))
        neg_inf = -Float32.inf
        half = (st // 16) % 2
        lane_ok = ((st % 16) < group) & (warp < self.T)
        lanes = tb + ((warp * 32) << 16)
        row = r0 + m // group
        h = m % group
        head = kvh * group + h
        prow = cutlass.Int64(row - tail_row0)
        prows = cutlass.Int64(rows - tail_row0)
        whole = nsplit == 1
        gregs = []
        if const_expr(gated_whole):
            for j in cutlass.range_constexpr(n_gc):
                k = st + 32 * SOFT_WARPS * j
                gm = cutlass.min(k // cpr, rows_m - 1)
                grow = cutlass.min(r0 + gm // group, rows - 1)
                ghead = kvh * group + gm % group
                gregs.append(
                    _ldg4(
                        gate_base
                        + (cutlass.Int64(grow) * gate_stride + ghead * HEAD_DIM + (k % cpr) * 8) * 2
                    )
                )
        cute.arch.mbarrier_wait(o_final, 0)
        tcgen05_fence_after()
        if warp < self.live_warps:
            gv = mGs[2]
            inv = Float32(0.0)
            if row_sum > Float32(0.0):
                inv = gv / row_sum
            for c in cutlass.range_constexpr(HEAD_DIM // (2 * CHUNK_COLS)):
                ov = tmem_ld16x2(lanes + (TMEM_O + c * CHUNK_COLS), CHUNK_COLS, HEAD_DIM // 2)
                cute.arch.fence_view_async_tmem_load()
                for v8 in cutlass.range_constexpr(CHUNK_COLS // 8):
                    vals = [ov[8 * v8 + i] * inv for i in range(8)]
                    d0 = half * (HEAD_DIM // 2) + c * CHUNK_COLS + 8 * v8
                    oaddr = out_base + ((cutlass.Int64(row) * heads + head) * HEAD_DIM + d0) * 2
                    paddr = (
                        part_base
                        + (((cutlass.Int64(split) * prows + prow) * heads + head) * HEAD_DIM + d0)
                        * 4
                    )
                    packed = [_bf2_pack(vals[2 * e], vals[2 * e + 1]) for e in range(4)]
                    saddr = s_stage + cutlass.min(m, rows_m - 1) * Q_LD + d0 * 2
                    if const_expr(has_whole):
                        if whole:
                            if const_expr(gated_whole):
                                if lane_ok:
                                    _sts4(saddr, packed[0], packed[1], packed[2], packed[3])
                            else:
                                if live:
                                    stg4(oaddr, packed[0], packed[1], packed[2], packed[3])
                    if const_expr(has_split):
                        if live & (nsplit > 1):
                            stg4f(paddr, vals[0], vals[1], vals[2], vals[3])
                            stg4f(paddr + 16, vals[4], vals[5], vals[6], vals[7])
            if const_expr(has_split):
                if live & (nsplit > 1) & (half == 0):
                    lse = neg_inf
                    if row_sum > Float32(0.0):
                        lse = row_max * scale_log2 + cute.math.log2(row_sum)
                    _stg1(
                        lse_base + ((cutlass.Int64(split) * prows + prow) * heads + head) * 4,
                        _f32_bits(lse),
                    )
        if const_expr(gated_whole):
            if whole:
                cute.arch.barrier(barrier_id=1, number_of_threads=32 * SOFT_WARPS)
                for j in cutlass.range_constexpr(n_gc):
                    k = st + 32 * SOFT_WARPS * j
                    gm = cutlass.min(k // cpr, rows_m - 1)
                    grow = r0 + gm // group
                    ghead = kvh * group + gm % group
                    a4 = _lds4(s_stage + gm * Q_LD + (k % cpr) * 16)
                    g4 = gregs[j]
                    res = [
                        _bf2_pack(
                            _bf2_lo(a4[e]) * _sigmoid_fast(_bf2_lo(g4[e])),
                            _bf2_hi(a4[e]) * _sigmoid_fast(_bf2_hi(g4[e])),
                        )
                        for e in range(4)
                    ]
                    if (k < rows_m * cpr) & (grow < rows):
                        stg4(
                            out_base
                            + ((cutlass.Int64(grow) * heads + ghead) * HEAD_DIM + (k % cpr) * 8)
                            * 2,
                            res[0],
                            res[1],
                            res[2],
                            res[3],
                        )

    # The MMAs (warp 4), softmax and epilogue (warps 0-3) are the base class's.


# ---------------------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------------------

_COMPILED: dict = {}
#: (floor, geometry) pairs whose every reachable launch plan is compiled.
_PLANNED: set = set()


def plan_splits(n_groups: int, kv_heads: int, width: int, num_sms: int) -> int:
    """Splits so the grid fills the GPU once (one CTA an SM): measured on GB300, a second
    partial wave costs more than the extra splits save, except where one wave gives one CTA a
    unit for more than half of the SMs: there `TWO_WAVE_SPLITS` splits a unit.
    """
    est_tiles = -(-int(width * 1.25) // BN)
    units = n_groups * kv_heads
    one_wave = num_sms // units
    if one_wave < 2 and units <= num_sms:
        one_wave = TWO_WAVE_SPLITS
    return max(1, min(MAX_PLAN_SPLITS, est_tiles, one_wave))


def plan_launch(n_groups: int, kv_heads: int, width: int, num_sms: int):
    """(splits, n_whole) of a launch: n_whole None = every unit (group, KV head) takes `splits`
    CTAs (one wave: `plan_splits`); else units below n_whole take one CTA and the units of the
    last, partial wave (whole groups) `splits` CTAs each, merged afterwards."""
    units = n_groups * kv_heads
    if units <= num_sms:
        return plan_splits(n_groups, kv_heads, width, num_sms), None
    waves = -(-units // num_sms)
    tail = units - (waves - 1) * num_sms
    tail = -(-tail // kv_heads) * kv_heads
    tail_splits = min(MAX_TAIL_SPLITS, num_sms // tail)
    if tail_splits < 2:
        return 1, None
    return tail_splits, units - tail


def _floor(min_groups: Optional[int]) -> int:
    """The routed floor a caller asks for (`MIN_GROUPS` when it does not)."""
    floor = MIN_GROUPS if min_groups is None else int(min_groups)
    if not SMALL_MIN_GROUPS <= floor <= MIN_GROUPS:
        raise ValueError(f"min_groups={min_groups}: expected {SMALL_MIN_GROUPS}..{MIN_GROUPS}")
    return floor


def reachable_plans(
    max_groups: int, kv_heads: int, width: int, num_sms: int, min_groups: Optional[int] = None
) -> list:
    """Every `(splits, tail)` pair `plan_launch` returns for `min_groups..max_groups` groups
    (`MIN_GROUPS` by default), `tail` whether the last wave is split: the kernel variants a
    geometry can need."""
    floor = _floor(min_groups)
    plans = set()
    for n_groups in range(floor, max(floor, max_groups) + 1):
        splits, n_whole = plan_launch(n_groups, kv_heads, width, num_sms)
        plans.add((max(1, min(MAX_SPLITS, int(splits))), n_whole is not None))
    return sorted(plans)


def _plan(
    q,
    k_cache,
    block_table,
    selected_tokens,
    tokens_per_block,
    rows_per_request,
    min_groups: Optional[int] = None,
    one_row: bool = False,
):
    """(splits, layout) of the grouped kernel for this call, or None where the per-row kernel
    takes it: fewer groups than the floor (`min_groups`, `MIN_GROUPS` by default), one row a
    request (unless `one_row`: a caller that groups one-row requests on purpose), not SM103, a
    plane its int32 row offsets cannot address, or no shared-memory plan (more than six rows a
    request; the wide plan takes tables up to 8192 pages of 32 tokens at four rows a request, the
    rows' bitmaps in the K/V slots).
    """
    rows, num_q_heads, head_dim = q.shape
    rpr = int(rows_per_request)
    if rpr < (1 if one_row else 2) or -(-rows // rpr) < _floor(min_groups) or head_dim != HEAD_DIM:
        return None
    if not archs.fast_path("qsa_fp8_verify", torch.cuda.get_device_capability(q.device)):
        return None
    kv_heads = k_cache.shape[1]
    if not _roff_fits(k_cache.shape[0], k_cache.stride()[:3], kv_heads, tokens_per_block):
        return None
    width = selected_tokens.shape[1]
    splits = plan_splits(-(-rows // rpr), kv_heads, width, _num_sms(q.device.index or 0))
    layout = _layout(
        rpr, num_q_heads // kv_heads, width, block_table.shape[1], tokens_per_block, splits
    )
    return None if layout is None else (splits, layout)


def takes_grouped(
    q,
    k_cache,
    block_table,
    selected_tokens,
    tokens_per_block,
    rows_per_request,
    min_groups: Optional[int] = None,
    one_row: bool = False,
):
    """Whether `fp8_sparse_gqa_verify` runs this call on the grouped kernel (else on the
    per-row kernel), from `min_groups` groups (`MIN_GROUPS` by default). With `one_row`, whether
    `fp8_sparse_gqa_verify_grouped` has a plan for a call of one row a request too (each request
    its own group: one launch with the gate fused instead of the per-row kernel's two and the gate
    launch); the routed entry never groups one-row requests. A host-side decision on shapes only:
    a per-capture constant.
    """
    plan = _plan(
        q,
        k_cache,
        block_table,
        selected_tokens,
        tokens_per_block,
        rows_per_request,
        min_groups,
        one_row,
    )
    return plan is not None


def _compile_verify(key: tuple, kern_kw: dict, compile_args: tuple):
    """The grouped kernel of `key` (its `splits` and `tail` at positions 5 and 6), compiled
    once."""
    fn = _COMPILED.get(key)
    if fn is None:
        kern = _VerifyKernel(splits=key[5], tail=key[6], **kern_kw)
        fn = cute.compile(kern, *compile_args, options="--opt-level 2 --enable-tvm-ffi")
        _COMPILED[key] = fn
    return fn


_UNIT: dict = {}


def _unit_scales(device) -> torch.Tensor:
    """`[1, 1, 1]` float32 on `device`: the kernel's K / V global scales over an fp8 pool."""
    t = _UNIT.get(device)
    if t is None:
        t = _UNIT[device] = torch.ones(4, dtype=torch.float32, device=device)
    return t


def fp8_sparse_gqa_verify_grouped(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected_tokens: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    softmax_scale: float,
    rows_per_request: int,
    output_gate: Optional[torch.Tensor] = None,
    pdl: bool = True,
    splits: Optional[int] = None,
    min_groups: Optional[int] = None,
    softmax_split: Optional[bool] = None,
) -> torch.Tensor:
    """The grouped kernel (and the merge with more than one split), whatever the row count;
    `splits` overrides the plan, `softmax_split` the `SOFTMAX_SPLIT` switch.
    `fp8_sparse_gqa_verify` is the routed entry. The first call of a geometry compiles the plans
    of `min_groups` (`MIN_GROUPS` by default) up to the block table's rows groups: the floor a
    caller routes from (`takes_grouped(..., min_groups)`).
    """
    rows, num_q_heads, head_dim = q.shape
    num_pages, num_kv_heads, tpb, packed = k_cache.shape
    rpr = int(rows_per_request)
    assert head_dim == HEAD_DIM and packed == HEAD_DIM and tpb == tokens_per_block
    assert q.dtype == torch.bfloat16 and q.stride(2) == 1 and q.stride(1) == HEAD_DIM
    assert (q.stride(0) * 2) % 16 == 0 and q.data_ptr() % 16 == 0, "16-byte aligned q rows"
    assert k_cache.dtype in (torch.float8_e4m3fn, torch.uint8)
    assert v_cache.dtype == k_cache.dtype and k_cache.stride() == v_cache.stride()
    assert k_cache.data_ptr() % 16 == 0 and v_cache.data_ptr() % 16 == 0
    assert all(st % 16 == 0 for st in k_cache.stride()[:3]), "16-byte aligned K/V rows"
    assert k_cache.stride(3) == 1 and v_cache.shape == k_cache.shape
    assert _roff_fits(num_pages, k_cache.stride()[:3], num_kv_heads, tpb), (
        "the K/V plane spans more than the group table's int32 row offsets address"
    )
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert block_table.stride(0) == block_table.shape[1], "row-contiguous block table"
    assert selected_tokens.dtype == torch.int32 and selected_tokens.is_contiguous()
    assert request_indices.is_contiguous() and request_indices.dtype in (torch.int32, torch.int64)
    group = num_q_heads // num_kv_heads
    width = selected_tokens.shape[1]
    use_split = SOFTMAX_SPLIT if softmax_split is None else bool(softmax_split)
    use_split = use_split and _softmax_split_fits(rpr, group)
    out = torch.empty((rows, num_q_heads, head_dim), dtype=torch.bfloat16, device=q.device)
    if rows == 0:
        return out
    n_groups = -(-rows // rpr)
    units = n_groups * num_kv_heads
    n_whole = None
    splits_given = splits
    if splits is None:
        splits, n_whole = plan_launch(n_groups, num_kv_heads, width, _num_sms(q.device.index or 0))
    n_splits = max(1, min(MAX_SPLITS, int(splits)))
    tail = n_whole is not None
    assert (
        _layout(rpr, group, width, block_table.shape[1], tpb, 1 if tail else n_splits) is not None
    ), "no shared-memory plan for this geometry: route through fp8_sparse_gqa_verify"
    use_pdl = bool(pdl and get_env_enable_pdl())
    unit = _unit_scales(q.device)
    gated = output_gate is not None
    if gated:
        assert output_gate.shape == (rows, num_q_heads * head_dim) and output_gate.stride(1) == 1
        assert output_gate.dtype == torch.bfloat16
        assert (output_gate.stride(0) * 2) % 16 == 0 and output_gate.data_ptr() % 16 == 0
    req64 = request_indices.dtype == torch.int64
    if tail:
        tail_row0 = (n_whole // num_kv_heads) * rpr
        n_ctas = n_whole + (units - n_whole) * n_splits
    elif n_splits > 1:
        n_whole, tail_row0, n_ctas = 0, 0, units * n_splits
    else:
        n_whole, tail_row0, n_ctas = units, rows, units
    merged = rows - tail_row0
    key = (
        rpr,
        group,
        num_kv_heads,
        width,
        tpb,
        n_splits,
        tail,
        float(softmax_scale),
        num_pages,
        tuple(block_table.shape),
        tuple(k_cache.stride()[:3]),
        (0, 0),
        tuple(q.stride()[:2]),
        req64,
        gated,
        use_pdl,
        use_split,
    )
    if merged > 0:
        part = torch.empty(
            (n_splits, merged, num_q_heads, head_dim), dtype=torch.float32, device=q.device
        )
        lse = torch.empty((n_splits, merged, num_q_heads), dtype=torch.float32, device=q.device)
    else:
        part, lse = out, out
    gate_t = output_gate if gated else q
    gate_stride = output_gate.stride(0) if gated else 0
    compile_args = (
        # read with scalar loads: a row slice of the selection needs no copy
        from_dlpack(selected_tokens, assumed_align=4).mark_layout_dynamic(leading_dim=1),
        _ptr(q, cutlass.BFloat16),
        _ptr(k_cache, cutlass.Uint8),
        _ptr(v_cache, cutlass.Uint8),
        _ptr(k_cache, cutlass.Uint8),
        _ptr(v_cache, cutlass.Uint8),
        _ptr(unit, cutlass.Float32, 4),
        _ptr(block_table, cutlass.Int32, 4),
        _ptr(request_indices, cutlass.Int64 if req64 else cutlass.Int32, 8 if req64 else 4),
        _ptr(out, cutlass.BFloat16),
        _ptr(part, cutlass.Float32),
        _ptr(lse, cutlass.Float32),
        _ptr(gate_t, cutlass.BFloat16),
        cutlass.Int32(rows),
        cutlass.Int32(gate_stride),
        cutlass.Int32(n_whole),
        cutlass.Int32(tail_row0),
        cutlass.Int32(n_ctas),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    merge_args = (
        from_dlpack(gate_t, assumed_align=16).mark_layout_dynamic(leading_dim=gate_t.dim() - 1),
        _ptr(part, cutlass.Float32),
        _ptr(lse, cutlass.Float32, 4),
        _ptr(out, cutlass.BFloat16, 8),
        cutlass.Int32(merged),
        cutlass.Int32(gate_stride),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    kern_kw = dict(
        rpr=rpr,
        group=group,
        kv_heads=num_kv_heads,
        width=width,
        tpb=tpb,
        scale=float(softmax_scale),
        num_pages=num_pages,
        bt_width=block_table.shape[1],
        num_requests=block_table.shape[0],
        k_strides=tuple(k_cache.stride()[:3]),
        ks_strides=(0, 0),
        q_strides=tuple(q.stride()[:2]),
        req64=req64,
        gated=gated,
        pdl=use_pdl,
        softmax_split=use_split,
    )
    # A launch plan depends on the call's row count: an eager step (a mixed step's
    # generation rows) meeting a new one would compile mid-serving, for seconds. The first
    # call of a geometry compiles every plan its row counts can reach from the floor instead.
    floor = _floor(min_groups)
    if splits_given is None and (floor,) + key[:5] + key[7:] not in _PLANNED:
        _PLANNED.add((floor,) + key[:5] + key[7:])
        # A group is one request's rows: at most a block-table row's worth of them.
        max_groups = block_table.shape[0]
        for plan_splits_, plan_tail in reachable_plans(
            max_groups, num_kv_heads, width, _num_sms(q.device.index or 0), floor
        ):
            if (
                _layout(
                    rpr, group, width, block_table.shape[1], tpb, 1 if plan_tail else plan_splits_
                )
                is None
            ):
                continue
            _compile_verify(key[:5] + (plan_splits_, plan_tail) + key[7:], kern_kw, compile_args)
            if plan_tail or plan_splits_ > 1:
                _compile_merge(num_q_heads, plan_splits_, gated, use_pdl, merge_args)
    fn = _compile_verify(key, kern_kw, compile_args)
    fn(
        selected_tokens,
        q.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        unit.data_ptr(),
        block_table.data_ptr(),
        request_indices.data_ptr(),
        out.data_ptr(),
        part.data_ptr(),
        lse.data_ptr(),
        gate_t.data_ptr(),
        rows,
        gate_stride,
        n_whole,
        tail_row0,
        n_ctas,
    )
    if merged <= 0:
        return out
    mfn = _compile_merge(num_q_heads, n_splits, gated, use_pdl, merge_args)
    mfn(
        gate_t[tail_row0:],
        part.data_ptr(),
        lse.data_ptr(),
        out[tail_row0:].data_ptr(),
        merged,
        gate_stride,
    )
    return out


def fp8_sparse_gqa_verify(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected_tokens: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    softmax_scale: float,
    rows_per_request: int,
    output_gate: Optional[torch.Tensor] = None,
    pdl: bool = True,
    min_groups: Optional[int] = None,
) -> torch.Tensor:
    """The per-row e4m3 sparse GQA (`qsa_paged_sparse_gqa`, decode rows) for rows that come
    `rows_per_request` consecutive rows a request: the grouped kernel where `takes_grouped` (from
    `min_groups` groups), the per-row kernel and a separate gate launch otherwise. Returns
    `[rows, Hq, D]` bf16 (contiguous); with `output_gate` (`[rows, Hq * D]` bf16, unit column
    stride) `bf16(attn) * sigmoid(gate)` rounded to bf16.
    """
    plan = _plan(
        q, k_cache, block_table, selected_tokens, tokens_per_block, rows_per_request, min_groups
    )
    if plan is None:
        out = qsa_paged_sparse_gqa(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            selected_tokens=selected_tokens,
            request_indices=request_indices,
            tokens_per_block=tokens_per_block,
            softmax_scale=softmax_scale,
            query_positions=None,
            compress_ratio=None,
            output_gate=None,
        )
        if output_gate is not None:
            fused_sigmoid_mul_inplace(torch.reshape(out, [q.shape[0], -1]), output_gate)
        return out
    return fp8_sparse_gqa_verify_grouped(
        q,
        k_cache,
        v_cache,
        block_table,
        selected_tokens,
        request_indices,
        tokens_per_block,
        softmax_scale,
        rows_per_request,
        output_gate=output_gate,
        pdl=pdl,
        splits=None,
        min_groups=min_groups,
    )


def _verify(segments, **overrides) -> dict:
    """A cell spec: `(first position, rows)` per request in row order, four rows a request, this
    checkpoint's dep geometry (24 q heads over 2 KV heads of 256, 32 tokens a page, 2051 selected
    tokens over a request's rows' overlapping top-512 blocks, the output gate on) and the
    block table of an engine at `max_seq_len` 262016 (8190 pages a request: the wide plan) unless
    `overrides` say otherwise."""
    spec = dict(
        segments=segments, rows_per_request=4, overlap=0.05, gated=True, pages=8190, min_groups=None
    )
    spec.update(overrides)
    return spec


class QsaPagedSparseGqaFp8Verify(OpWrapper):
    """`qsa_paged_sparse_gqa` (decode rows) for rows that come `rows_per_request` consecutive
    rows a request, over an e4m3 paged pool `[pages, kv_heads, tokens, 256]`: each selected K/V
    token loaded and widened once for a group of T rows, over the union of the group's
    selections kept as token bitmaps. `[rows, Hq, D]` bf16; with `output_gate` (`[rows, Hq * D]`
    bf16, unit column stride) `bf16(attn) * sigmoid(gate)` rounded to bf16.

    Fusion boundary: the block-table gather, the union of the group's selections, the e4m3
    widening, scale, online softmax, P.V, the split-K merge and (optionally) the output gate.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dep8 at 3 requests a rank (12 verify rows), routed from 2 groups: one pass a "
            "group over many splits, the engine's 8190-page table (wide plan)",
            spec=_verify(tuple((90000 + 4099 * i, 4) for i in range(3)), min_groups=2),
        ),
        Cell(
            why="dep8 at 12 requests a rank (48 rows), from the default 8 groups",
            spec=_verify(tuple((20000 + 9973 * i, 4) for i in range(12))),
        ),
        Cell(
            why="dep8 at 24 requests (96 rows): 48 units, three splits a unit",
            spec=_verify(tuple((4000 + 7919 * (i % 16), 4) for i in range(24))),
        ),
        Cell(
            why="dep4 at 45 requests (180 rows): 90 units take two splits each over two waves "
            "and the merge (`TWO_WAVE_SPLITS`)",
            spec=_verify(tuple((2100 + (53 * i) % 60000, 4) for i in range(45))),
        ),
        Cell(
            why="contexts past 131072 tokens: the union scan's rounds stop at the highest "
            "selected token, here two of four rounds skipped on some groups",
            spec=_verify(tuple((130000 + 30011 * (i % 4), 4) for i in range(8))),
        ),
        Cell(
            why="256 requests x 4 rows: one CTA a unit with the gate fused, the last wave's "
            "units split and merged",
            spec=_verify(tuple((2100 + (37 * i) % 1900, 4) for i in range(256)), pages=516),
        ),
        Cell(
            why="groups spanning requests (3, 4, 4, 1, 4 x 5, 2 rows), early positions, int64 "
            "request indices, no gate, a narrow table (the union scratch in raw slot 0)",
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
                gated=False,
                pages=516,
                min_groups=2,
            ),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument.
    Call sites: the NVFP4 target's trunk verify step at an attention-DP rank over an e4m3 pool,
    under `TRTLLM_QWEN4_EXP_DP_GQA_BITMAP` (`_sparse_gqa_decode`), through
    `fp8_sparse_gqa_verify_grouped` where `takes_grouped(..., min_groups)`; its draft passes of
    one row a request under `TRTLLM_QWEN4_EXP_DP_DRAFT_GQA_GROUPED1`, the same entry at
    `rows_per_request` 1 where `takes_grouped(..., one_row=True)`.

    Routing (`takes_grouped`, host-side, on shapes only): the grouped kernel from the caller's
    floor (`min_groups`, 2..8, default `MIN_GROUPS` 8) of T >= 2 rows a request up, on SM103,
    where its shared-memory plan fits (T <= 6) and the plane's last row offset fits the group
    table's int32 word; the per-row kernel otherwise. Launch (`plan_launch`): the NVFP4 verify
    op's, except that a launch whose units fill more than half of the SMs takes
    `TWO_WAVE_SPLITS` splits a unit.

    GB300 (dep4 / dep8 AgentX, e4m3 KV, MTP-3): the grouped kernel beat the per-row kernel and
    its gate launch by 9.6-20.3 us a layer at 2..7 groups; the two-wave plan saved 0.15 ms a
    step at 45 requests a rank; the loader's one table read a group and the scan bound before
    the PDL wait, with the input-side cuts of that bundle, 0.37-0.47 ms a step at dep4.

    Numerics: e4m3 values are exact in f16; scores on f16 MMAs with fp32 accumulation; P rounded
    to f16; a row's softmax offset moves only when its max grew by more than 2^8. Within the
    decode op's band of the reference; not bitwise the per-row kernel. Deterministic;
    graph-capturable (row count is a runtime value).

    Preconditions: bf16 q with head_dim 256 and 16-byte aligned rows; e4m3 K and V planes with
    the same strides and 16-byte aligned rows; power-of-two tokens per block; int32
    row-contiguous block table; contiguous int32 expanded selections (each four-column quad names
    one four-token group at its own offsets, no group twice in a row).

    PDL: the first pass's bitmap clear, O = 0, the q copies and the request indices run before
    the kernel's wait. The launch in front must release its dependents only after its own wait
    (`qsa_topk_expand_cluster`, `qsa_topk_expand_decode`).
    """

    def __call__(
        self,
        q,
        k_cache,
        v_cache,
        block_table,
        selected_tokens,
        request_indices,
        tokens_per_block: int,
        softmax_scale: float,
        rows_per_request: int,
        output_gate=None,
        min_groups=None,
    ) -> torch.Tensor:
        return fp8_sparse_gqa_verify(
            q,
            k_cache,
            v_cache,
            block_table,
            selected_tokens,
            request_indices,
            tokens_per_block,
            softmax_scale,
            rows_per_request,
            output_gate=output_gate,
            min_groups=min_groups,
        )

    def reference(
        self,
        q,
        k_cache,
        v_cache,
        block_table,
        selected_tokens,
        request_indices,
        tokens_per_block: int,
        softmax_scale: float,
        rows_per_request: int,
        output_gate=None,
        min_groups=None,
    ) -> torch.Tensor:
        """The per-row op's reference (`rows_per_request` and `min_groups` only group the work),
        then the gate."""
        out = qsa_paged_sparse_gqa.reference(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            selected_tokens=selected_tokens,
            request_indices=request_indices,
            tokens_per_block=tokens_per_block,
            softmax_scale=softmax_scale,
            query_positions=None,
            compress_ratio=None,
            output_gate=None,
        )
        if output_gate is None:
            return out
        gated = out.float() * torch.sigmoid(torch.reshape(output_gate, out.shape).float())
        return gated.to(torch.bfloat16)

    def is_valid(
        self,
        q,
        k_cache,
        v_cache,
        block_table,
        selected_tokens,
        request_indices,
        tokens_per_block: int,
        softmax_scale: float,
        rows_per_request: int,
        output_gate=None,
        min_groups=None,
    ) -> None:
        assert k_cache.dtype == torch.float8_e4m3fn and v_cache.dtype == k_cache.dtype
        assert int(rows_per_request) >= 1, "rows_per_request >= 1"
        _floor(min_groups)
        # The union is sized for expanded selections: a quad of columns names one four-token
        # group at its own offsets, and no group twice in a row.
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


qsa_paged_sparse_gqa_fp8_verify = QsaPagedSparseGqaFp8Verify()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA for decode and verify rows over an fp8 (e4m3) paged K/V pool, at up to eight
query heads a KV head: a split-K pass on half-height warp MMAs that reads its K and V operands
straight from the raw e4m3 rows, then the bf16 decode entry's merge (`qsa_paged_sparse_gqa_decode`,
imported unchanged), which carries the output gate.

Split-K pass. One CTA (eight warps) serves one (row, KV head, split) and walks the split's `tps`
tiles of `CAP` = 64 selected tokens:

1. before its PDL wait the CTA stages the KV head's query heads in shared memory (rows past the
   group are zero) and releases its dependents; q is all it reads before the wait;
2. after the wait it resolves every token of the split once (selection -> block table ->
   `page * tpb + slot`, or -1) into a shared token table;
3. per tile: the raw e4m3 K and V rows (256 bytes each, sixteen 16-byte `cp.async` chunks a row,
   at a 272-byte pitch; a token that does not resolve gets zeros through the zero-fill form) land
   in one of two raw slots, the next tile's rows issued into the other slot as soon as this one
   has landed;
4. scores = q K^T on m16n8k16 warp MMAs, bf16 in and fp32 out, a warp eight tokens: q's A
   fragments (MMA rows 0..7; rows 8..15 are the zero registers, the group being at most eight)
   are held in registers for the whole split, and each lane builds its B fragment from the raw
   K row of its token, widened exactly (`cvt.rn.f16x2.e4m3x2`, then f16 -> f32 -> bf16: an e4m3
   value has four significant bits and its exponent lies inside both formats). The scores are
   scaled into the log2 domain and masked; a softmax over the tile's tokens follows (a warp one
   head), online across the split's tiles (running maximum, the running sum and the fp32
   accumulator rescaled by `exp2(old max - new max)`). P V: P in bf16 as the A fragments, the
   B fragments built from the raw V rows with byte permutes and the same widen, a warp 32 head
   dims on four m16n8 accumulators;
5. each head's partial times its reciprocal sum is staged in shared memory and stored in 16-byte
   pieces with its log2-sum-exp, as the bf16 entry stores them.

The arithmetic is the bf16 entry's on the same operand values (the same MMA shape, k order and
softmax); rows 8..15 of the MMAs hold zeros, which add nothing. The pool is read at face value,
as the per-row kernel (`qsa_paged_sparse_gqa`) reads an fp8 pool: there is no KV scale argument,
so the pool must hold K/V written at unit scale.

Shared memory: a 16-row K scratch (the masked scores and the output staging alias it), q, P, the
per-head running state, the token table and two raw slots of 34 KB: two CTAs an SM.

Launch plan (`plan`): splits are chosen so a launch holds about one wave of CTAs (`WAVE_CTAS` =
304 = two an SM on GB300's 152 SMs). At the tep4 shard (one KV head) and 2051 selected tokens:
33 splits of one tile up to 8 rows, 17 of two at 12 and 16 rows, 9 of four at 32.

PDL (`TRTLLM_ENABLE_PDL`, every served row count): both launches release their dependents at
entry, before their own waits. The split-K pass reads q before its wait and the merge the output
gate before its own; the request index, the selection, the block table and the K/V rows are read
after the split-K pass's wait. So q and the gate must be complete when the grid ahead of this
entry releases it: a launch without PDL releases at its end, and the target's top-k releases only
after its own wait.
"""

from __future__ import annotations

import functools
import math
from typing import Optional

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa import (
    qsa_paged_sparse_gqa as _composition,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_decode import (  # noqa: E501
    MAX_SPLITS,
    _compiled_merge,
)

#: Rows this entry serves: one to eight requests of an MTP3 verify step (four rows each).
MAX_ROWS = 32
#: Selected tokens a tile.
CAP = 64
#: MMA rows; a group of query heads fills at most half of them.
M = 16
#: Most query heads a KV head: the MMA rows the A fragments hold (0..7).
MAX_GROUP = M // 2
#: bf16 elements of padding after each shared q and P row and the K scratch's rows (keeps the
#: MMA fragment reads free of bank conflicts).
PAD = 8
WARPS = 8
THREADS = WARPS * 32
HEAD_DIM = 256
#: Bytes a raw e4m3 K or V row takes in a slot: 256 + 16, so the lanes of one warp hit 32
#: distinct banks.
RAW_PITCH = HEAD_DIM + 16
#: Bytes of one raw slot: a tile's K rows then its V rows; two slots.
SLOT = 2 * CAP * RAW_PITCH
LOG2E = 1.4426950408889634
#: CTAs a launch aims at: two an SM on GB300 (152 SMs).
WAVE_CTAS = 304
#: Shared memory two CTAs an SM may each claim on sm_100 / sm_103 (228 KiB an SM, 1 KiB
#: reserved a CTA).
_SMEM_TWO_PER_SM = 233472 // 2 - 1024
#: Tiles a split may walk: the token table (256 bytes a tile) has to leave two CTAs an SM.
MAX_TPS = 16
#: Fewest splits a launch may have. The imported merge (`_MergeKernel` of the bf16 decode
#: entry) launches `min(8, splits)` warps and writes its 128 output elements a CTA from the
#: threads that exist: with 2 or 3 splits (64 or 96 threads) it would leave 64 or 32 of every
#: 128 elements unwritten. Four splits give it 128 threads.
MIN_SPLITS = 4


# ---------------------------------------------------------------------------------------
# inline PTX helpers
# ---------------------------------------------------------------------------------------


def _ir(x, cls, loc, ip):
    return cls(x).ir_value(loc=loc, ip=ip)


def _asm(ret, args, text, cons, loc, ip, side=False):
    return llvm.inline_asm(
        ret,
        args,
        text,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _unpack_i32(res, n, loc, ip):
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(n)]


@dsl_user_op
def _cp16z(saddr, gaddr, nbytes, *, loc=None, ip=None):
    """cp.async of 16 bytes, zero-filled past `nbytes` (0 or 16): no branch for a dead token."""
    _asm(
        T.i32(),
        [
            _ir(saddr, cutlass.Int32, loc, ip),
            _ir(gaddr, cutlass.Int64, loc, ip),
            _ir(nbytes, cutlass.Int32, loc, ip),
        ],
        "{ cp.async.cg.shared.global [$1], [$2], 16, $3; mov.u32 $0, 0; }",
        "=r,r,l,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _e4m3x4_bf16(word, *, loc=None, ip=None):
    """Four e4m3 bytes (one b32, byte 0 the lowest) -> two bf16x2 words: bytes 0, 1 and bytes
    2, 3, each with the lower byte in the low half. Exact: e4m3 -> f16 (`cvt.rn.f16x2.e4m3x2`,
    the widen `qsa_paged_sparse_gqa_fp8_verify` uses), f16 -> f32, f32 -> bf16 (four
    significant bits, exponents -9 .. 8: nothing to round); a NaN code stays a NaN."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_ir(word, cutlass.Int32, loc, ip)],
        "{ .reg .b16 p0, p1, a0, a1, a2, a3; .reg .b32 t0, t1; .reg .f32 f0, f1, f2, f3; "
        "mov.b32 {p0, p1}, $2; "
        "cvt.rn.f16x2.e4m3x2 t0, p0; cvt.rn.f16x2.e4m3x2 t1, p1; "
        "mov.b32 {a0, a1}, t0; mov.b32 {a2, a3}, t1; "
        "cvt.f32.f16 f0, a0; cvt.f32.f16 f1, a1; cvt.f32.f16 f2, a2; cvt.f32.f16 f3, a3; "
        "cvt.rn.bf16x2.f32 $0, f1, f0; cvt.rn.bf16x2.f32 $1, f3, f2; }",
        "=r,=r,r",
        loc,
        ip,
    )
    return _unpack_i32(res, 2, loc, ip)


@dsl_user_op
def _lds1(saddr, *, loc=None, ip=None):
    res = _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip)],
        "ld.shared.b32 $0, [$1];",
        "=r,r",
        loc,
        ip,
        side=True,
    )
    return cutlass.Int32(res)


@dsl_user_op
def _prmt(a, b, sel: int, *, loc=None, ip=None):
    res = _asm(
        T.i32(),
        [_ir(a, cutlass.Int32, loc, ip), _ir(b, cutlass.Int32, loc, ip)],
        f"prmt.b32 $0, $1, $2, {int(sel)};",
        "=r,r,r",
        loc,
        ip,
    )
    return cutlass.Int32(res)


@dsl_user_op
def _mma16816(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (what cute.gemm emits for
    MmaF16BF16Op(bf16, f32, (16, 8, 16)))."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_ir(x, cutlass.Int32, loc, ip) for x in (a0, a1, a2, a3, b0, b1)]
        + [_ir(x, cutlass.Float32, loc, ip) for x in (c0, c1, c2, c3)],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, "
        "{$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        loc,
        ip,
    )
    return [cutlass.Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _lds16x2(saddr_lo, saddr_hi, *, loc=None, ip=None):
    """Two 16-bit shared loads packed into one b32: the low half from `saddr_lo`."""
    res = _asm(
        T.i32(),
        [_ir(saddr_lo, cutlass.Int32, loc, ip), _ir(saddr_hi, cutlass.Int32, loc, ip)],
        "{ .reg .b16 x0, x1; ld.shared.b16 x0, [$1]; ld.shared.b16 x1, [$2]; mov.b32 $0, {x0, x1}; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )
    return cutlass.Int32(res)


# ---------------------------------------------------------------------------------------
# split-K kernel
# ---------------------------------------------------------------------------------------


def _r128(n: int) -> int:
    return -(-n // 128) * 128


def smem_bytes(tps: int) -> int:
    """Shared memory of a split-K CTA walking `tps` tiles."""
    ld = HEAD_DIM + PAD
    b = _r128(M * ld * 2)  # the K scratch: masked scores, then the output staging
    b += _r128(M * ld * 2)  # q
    b += _r128(M * (CAP + PAD) * 2)  # P
    b += _r128(M * 4 * 4)  # per head: running max, running sum, this tile's rescale
    b += _r128(tps * CAP * 4)  # token table
    return b + 2 * _r128(SLOT) + 256  # two raw slots


class _SplitKKernel:
    def __init__(
        self,
        *,
        group,
        width,
        splits,
        tps,
        tpb,
        scale,
        bt_width,
        num_pages,
        k_strides,
        v_strides,
        num_requests,
        pdl,
    ):
        assert group <= MAX_GROUP and CAP == 2 * 32 and CAP % WARPS == 0
        assert (CAP // WARPS) % 2 == 0
        assert tpb & (tpb - 1) == 0, "tokens_per_block: a power of two"
        assert 1 <= tps <= MAX_TPS and splits * tps * CAP >= width
        assert (splits - 1) * tps * CAP < width, "every split owns a tile"
        assert MIN_SPLITS <= splits <= MAX_SPLITS, (
            f"{MIN_SPLITS} .. {MAX_SPLITS} splits (the merge writes its whole chunk only from "
            "four warps up)"
        )
        self.group, self.width, self.splits, self.tps, self.tpb = group, width, splits, tps, tpb
        self.n_tiles = -(-width // CAP)
        self.scale = scale * LOG2E
        self.bt_width, self.num_pages, self.num_requests = bt_width, num_pages, num_requests
        self.k_strides, self.v_strides = k_strides, v_strides
        self.ld = HEAD_DIM + PAD
        self.pdl = pdl
        assert self.smem_bytes() <= _SMEM_TWO_PER_SM, "two CTAs an SM"

    def smem_bytes(self) -> int:
        return smem_bytes(self.tps)

    @cute.jit
    def __call__(self, mQ, pK, pV, mBT, mSel, mReq, pPart, mLse, stream: cuda_drv.CUstream):
        rows = cute.size(mSel, mode=[0])
        kv_heads = cute.size(mQ, mode=[1]) // self.group
        g2s = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        st = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
        )
        self.kernel(mQ, pK, pV, mBT, mSel, mReq, pPart, mLse, g2s, st).launch(
            grid=[rows * kv_heads * self.splits, 1, 1],
            block=[THREADS, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=self.pdl,
            # Two CTAs an SM: at most 128 registers a thread.
            min_blocks_per_mp=2,
        )

    @cute.jit
    def _issue(self, j, warp_i, lane, kvh, sTok, raw, k_base, v_base):
        """cp.async of tile `j`'s raw K and V rows into raw slot `j & 1`, one commit group: warp
        w the tokens 8w .. 8w + 7, a lane one 16-byte chunk (`lane % 16`) of four of them; a
        token that does not resolve gets zeros (source size 0, read from a mapped address)."""
        tpb: cutlass.Constexpr = self.tpb
        lt: cutlass.Constexpr = tpb.bit_length() - 1
        ks0, ks1, ks2, _ = self.k_strides
        vs0, vs1, vs2, _ = self.v_strides
        tpw: cutlass.Constexpr = CAP // WARPS
        ch = lane % 16
        for i in cutlass.range_constexpr(tpw // 2):
            t = warp_i * tpw + 2 * i + lane // 16
            ent = sTok[j * CAP + t]
            n = cutlass.Int32(cutlass.select_(ent >= 0, cutlass.Int32(16), cutlass.Int32(0)))
            e = cutlass.max(ent, 0)
            pg = cutlass.Int64(e >> lt)
            sl = e & (tpb - 1)
            ko = pg * ks0 + cutlass.Int64(kvh * ks1 + sl * ks2 + 16 * ch)
            vo = pg * vs0 + cutlass.Int64(kvh * vs1 + sl * vs2 + 16 * ch)
            slot = raw + (j & 1) * SLOT
            _cp16z(slot + t * RAW_PITCH + 16 * ch, k_base + ko, n)
            _cp16z(slot + CAP * RAW_PITCH + t * RAW_PITCH + 16 * ch, v_base + vo, n)
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(
        self,
        mQ,
        pK,
        pV,
        mBT,
        mSel,
        mReq,
        pPart,
        mLse,
        g2s: cute.CopyAtom,
        st: cute.CopyAtom,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_i = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        group: cutlass.Constexpr = self.group
        splits: cutlass.Constexpr = self.splits
        tps: cutlass.Constexpr = self.tps
        n_tiles: cutlass.Constexpr = self.n_tiles
        width: cutlass.Constexpr = self.width
        tpb: cutlass.Constexpr = self.tpb
        bt_width: cutlass.Constexpr = self.bt_width
        ld: cutlass.Constexpr = self.ld
        dim: cutlass.Constexpr = HEAD_DIM
        tpw: cutlass.Constexpr = CAP // WARPS
        heads = cute.size(mQ, mode=[1])
        kv_heads = heads // group
        rows = cute.size(mSel, mode=[0])
        row = bidx // (kv_heads * splits)
        kvh = (bidx // splits) % kv_heads
        split = bidx % splits
        t0 = split * tps
        n_t = cutlass.min(tps, n_tiles - t0)

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((M, dim), stride=(ld, 1)), byte_alignment=128
        )
        sQ = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((M, dim), stride=(ld, 1)), byte_alignment=128
        )
        sP = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((M, CAP), stride=(CAP + PAD, 1)), byte_alignment=128
        )
        sStat = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((M, 4), stride=(4, 1)), byte_alignment=128
        )
        sTok = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((tps * CAP,), stride=(1,)), byte_alignment=128
        )
        sRaw = smem.allocate_tensor(
            cutlass.Int8, cute.make_layout((2 * SLOT,), stride=(1,)), byte_alignment=128
        )
        # The masked scores ([M, CAP] fp32, 4352 bytes) live in the K scratch: the scores of a
        # tile are written after its MMAs (barrier [A]) and read by the softmax before the next
        # tile's scores.
        sS = cute.make_tensor(
            cute.recast_ptr(sK.iterator, dtype=cutlass.Float32),
            cute.make_layout((M, CAP), stride=(CAP + 4, 1)),
        )
        raw = cutlass.Int32(sRaw.iterator.toint())
        sp_addr = cutlass.Int32(sP.iterator.toint())
        sq_addr = cutlass.Int32(sQ.iterator.toint())

        # 1. Before the wait: the KV head's query heads (complete before the grid ahead of
        # this one released it) into shared memory, then the release.
        cpr: cutlass.Constexpr = dim // 8  # 16-byte pieces a row: one a lane
        q_vec = cute.logical_divide(mQ, (None, None, 8))
        sq_vec = cute.logical_divide(sQ, (None, 8))
        for qi in cutlass.range_constexpr(-(-(group * cpr) // THREADS)):
            qc = tidx + qi * THREADS
            qcc = cutlass.min(qc, group * cpr - 1)
            qh = qcc // cpr
            chq = qcc % cpr
            if qc < group * cpr:
                cute.copy(g2s, q_vec[row, kvh * group + qh, (None, chq)], sq_vec[qh, (None, chq)])
        cute.arch.cp_async_commit_group()
        for zi in cutlass.range_constexpr((M - group) * dim // THREADS):
            ze = tidx + zi * THREADS
            sQ[group + ze // dim, ze % dim] = cutlass.BFloat16(0.0)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if const_expr(self.pdl):
            # The merge waits before reading the partials and reads nothing before its wait
            # that the grid ahead of this one writes.
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        # 2. The split's tokens, resolved once: `page * tpb + slot`, or -1.
        req = mReq[row]
        req_ok = (req >= 0) & (req < self.num_requests)
        safe_req = cutlass.min(cutlass.max(req, 0), self.num_requests - 1)
        for ti in cutlass.range_constexpr(-(-(tps * CAP) // THREADS)):
            cl = tidx + ti * THREADS
            col = t0 * CAP + cl
            colc = cutlass.min(col, width - 1)
            tok = mSel[row, colc]
            ok = req_ok & (col < width) & (tok >= 0)
            safe_tok = cutlass.max(tok, 0)
            lp = safe_tok // tpb
            ok = ok & (lp < bt_width)
            lpc = cutlass.min(lp, bt_width - 1)
            page = mBT[safe_req, lpc]
            ok = ok & (page >= 0) & (page < self.num_pages)
            ent = cutlass.Int32(
                cutlass.select_(ok, cutlass.max(page, 0) * tpb + safe_tok % tpb, cutlass.Int32(-1))
            )
            if cl < tps * CAP:
                sTok[cl] = ent
        cute.arch.sync_threads()

        k_base = pK.toint()
        v_base = pV.toint()
        self._issue(0, warp_i, lane, kvh, sTok, raw, k_base, v_base)

        # The P V accumulator as four m16n8 C fragments (n-tile jn at [4 jn .. 4 jn + 3]): rows
        # h0 and h0 + 8 (the second always zero), the warp's 32 head dims.
        accd = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
        accd.fill(0.0)
        tq = lane % 4
        h0 = lane // 4  # an m16n8 C fragment's rows: h0 and h0 + 8
        neg_inf = cutlass.Float32(-math.inf)
        # A warp's head (warp_i): running max [0], running sum [1].
        run = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), cutlass.Float32)
        run[0] = neg_inf
        run[1] = cutlass.Float32(0.0)
        # q's A fragments for the whole split (rows 0..7; rows 8..15 are the zero registers):
        # k-block kb at [2 kb] (dims 16 kb + 2 tq, + 1) and [2 kb + 1] (+ 8), the m16n8k16 A
        # registers a0 and a2.
        qa = cute.make_rmem_tensor(
            cute.make_layout((2 * (HEAD_DIM // 16),), stride=(1,)), cutlass.Int32
        )
        sc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
        qb = sq_addr + (h0 * ld + 2 * tq) * 2
        for kb in cutlass.range_constexpr(HEAD_DIM // 16):
            qa[2 * kb] = _lds1(qb + 32 * kb)
            qa[2 * kb + 1] = _lds1(qb + 32 * kb + 16)
        zero = cutlass.Int32(0)

        for j in cutlass.range(n_t):
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()  # [A] tile j's raw rows landed; tile j - 1's P V is done
            if j + 1 < n_t:
                self._issue(j + 1, warp_i, lane, kvh, sTok, raw, k_base, v_base)

            # 3. Scores: warp w's tokens 8 w .. 8 w + 7 (lane (h0, tq) holds B[k][n = h0] for
            # k = 2 tq, 2 tq + 1 (b0) and 2 tq + 8, + 9 (b1) of every k-block, read from the raw
            # row of token 8 w + h0 and widened), into the MMA in k order; masked into the scratch.
            sc.fill(0.0)
            kr = raw + (j & 1) * SLOT + (warp_i * tpw + h0) * RAW_PITCH + 2 * tq
            for kb in cutlass.range_constexpr(HEAD_DIM // 16):
                kw = _lds16x2(kr + 16 * kb, kr + 16 * kb + 8)
                bb = _e4m3x4_bf16(kw)
                d4 = _mma16816(
                    qa[2 * kb], zero, qa[2 * kb + 1], zero, bb[0], bb[1], sc[0], sc[1], sc[2], sc[3]
                )
                sc[0] = d4[0]
                sc[1] = d4[1]
                sc[2] = d4[2]
                sc[3] = d4[3]
            for cc in cutlass.range_constexpr(2):
                stk = warp_i * tpw + 2 * tq + cc
                tent = sTok[j * CAP + stk]
                sS[h0, stk] = cutlass.Float32(
                    cutlass.select_(tent >= 0, sc[cc] * self.scale, neg_inf)
                )
            cute.arch.sync_threads()  # [C]
            # 4. The softmax: warp w the head w, a lane two tokens; padding heads get P = 0. With
            # one tile a split the running state starts at (-inf, 0), so the maximum, P and the
            # sum are the bf16 entry's.
            wh = warp_i
            sa = sS[wh, lane]
            sb = sS[wh, lane + 32]
            tmx = cute.arch.warp_reduction_max(cute.arch.fmax(sa, sb))
            nm = cute.arch.fmax(run[0], tmx)
            safe_mx = cutlass.Float32(cutlass.select_(nm == neg_inf, cutlass.Float32(0.0), nm))
            keep = wh < group
            pa = cutlass.Float32(
                cutlass.select_(keep, cute.math.exp2(sa - safe_mx), cutlass.Float32(0.0))
            )
            pb = cutlass.Float32(
                cutlass.select_(keep, cute.math.exp2(sb - safe_mx), cutlass.Float32(0.0))
            )
            tden = cute.arch.warp_reduction_sum(pa + pb)
            if const_expr(tps > 1):
                # exp2(-inf - x) = 0 on the first tile: the empty state drops out.
                corr = cute.math.exp2(run[0] - safe_mx)
                tden = run[1] * corr + tden
                if lane == 0:
                    sStat[wh, 2] = corr
            sP[wh, lane] = pa.to(cutlass.BFloat16)
            sP[wh, lane + 32] = pb.to(cutlass.BFloat16)
            if lane == 0:
                sStat[wh, 0] = nm
                sStat[wh, 1] = tden
            run[0] = nm
            run[1] = tden
            cute.arch.sync_threads()  # [D]

            # 5. P V: a warp 32 of the head dims (four 8-wide n-tiles), onto the rescaled sum;
            # V's B fragments from the raw V rows (rows 2 tq, + 1, + 8, + 9 of each k-block,
            # byte-permuted to one head dim each, widened).
            if const_expr(tps > 1):
                cr0 = sStat[h0, 2]
                for c4 in cutlass.range_constexpr(4):
                    accd[4 * c4] = accd[4 * c4] * cr0
                    accd[4 * c4 + 1] = accd[4 * c4 + 1] * cr0
            pbase = sp_addr + (h0 * (CAP + PAD) + 2 * tq) * 2
            vb = (
                raw + (j & 1) * SLOT + CAP * RAW_PITCH + (2 * tq) * RAW_PITCH + 32 * warp_i + 4 * h0
            )
            for kb in cutlass.range_constexpr(CAP // 16):
                p_lo = _lds1(pbase + 32 * kb)
                p_hi = _lds1(pbase + 32 * kb + 16)
                w0 = _lds1(vb + (16 * kb) * RAW_PITCH)
                w1 = _lds1(vb + (16 * kb + 1) * RAW_PITCH)
                w2 = _lds1(vb + (16 * kb + 8) * RAW_PITCH)
                w3 = _lds1(vb + (16 * kb + 9) * RAW_PITCH)
                for jn in cutlass.range_constexpr(4):
                    y01 = _prmt(w0, w1, jn | ((jn + 4) << 4))
                    y23 = _prmt(w2, w3, jn | ((jn + 4) << 4))
                    bb = _e4m3x4_bf16(_prmt(y01, y23, 0x5410))
                    d4 = _mma16816(
                        p_lo, zero, p_hi, zero, bb[0], bb[1],
                        accd[4 * jn], accd[4 * jn + 1], accd[4 * jn + 2], accd[4 * jn + 3],
                    )  # fmt: skip
                    accd[4 * jn] = d4[0]
                    accd[4 * jn + 1] = d4[1]
                    accd[4 * jn + 2] = d4[2]
                    accd[4 * jn + 3] = d4[3]

        # 6. Each head's partial times its reciprocal sum, staged over the K scratch (the scores
        # on it are dead since the last tile's softmax barrier) as [group, dim] fp32, then stored
        # in 16-byte pieces: the CTA's heads are contiguous in the partial tensor.
        sOut = cute.make_tensor(
            cute.recast_ptr(sK.iterator, dtype=cutlass.Float32),
            cute.make_layout((group, dim), stride=(dim, 1)),
        )
        d0 = sStat[h0, 1]
        r0 = cutlass.Float32(0.0)
        if d0 > 0.0:
            r0 = 1.0 / d0
        if h0 < group:
            for jn in cutlass.range_constexpr(4):
                for cc in cutlass.range_constexpr(2):
                    sOut[h0, 32 * warp_i + 4 * (2 * tq + cc) + jn] = accd[4 * jn + cc] * r0
        cute.arch.sync_threads()
        base = ((cutlass.Int64(split) * rows + row) * heads + kvh * group) * dim
        out_ptr = pPart.toint() + base * 4
        so_vec = cute.make_tensor(
            sOut.iterator, cute.make_layout((group * dim // 4, 4), stride=(4, 1))
        )
        reg = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
        for oi in cutlass.range_constexpr(-(-(group * dim // 4) // THREADS)):
            oc = tidx + oi * THREADS
            if oc < group * dim // 4:
                cute.autovec_copy(so_vec[oc, None], reg)
                dst = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        out_ptr + cutlass.Int64(oc * 16),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((4,), stride=(1,)),
                )
                cute.copy(st, reg, dst)
        if tidx < group:
            fmx = sStat[tidx, 0]
            fden = sStat[tidx, 1]
            lse = neg_inf
            if fden > 0.0:
                lse = fmx + cute.math.log2(fden)
            mLse[split, row, kvh * group + tidx] = lse


# ---------------------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------------------


def _dl(t: torch.Tensor, align: int = 4):
    return from_dlpack(t, assumed_align=align)


def _ptr(t: torch.Tensor, dtype):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def plan(rows: int, kv_heads: int, width: int) -> tuple[int, int]:
    """`(splits, tiles a split)` of a launch: about `WAVE_CTAS` CTAs, at least `MIN_SPLITS`
    splits (the imported merge needs four warps to write its whole chunk), at most one a tile
    and at most `MAX_TPS` tiles a split. Needs at least `MIN_SPLITS` tiles (`supports`)."""
    n_tiles = -(-width // CAP)
    assert n_tiles >= MIN_SPLITS, (
        f"at least {MIN_SPLITS} tiles ({MIN_SPLITS * CAP - CAP + 1} tokens)"
    )
    want = WAVE_CTAS // max(1, rows * kv_heads)
    want = max(MIN_SPLITS, -(-n_tiles // MAX_TPS), min(n_tiles, MAX_SPLITS, want))
    tps = -(-n_tiles // want)
    # Rounding the tiles a split up can leave fewer splits than wanted (9 tiles at 4 wanted:
    # 3 tiles a split, 3 splits): shorten the splits until the floor holds.
    while tps > 1 and -(-n_tiles // tps) < MIN_SPLITS:
        tps -= 1
    n_splits = -(-n_tiles // tps)
    assert MIN_SPLITS <= n_splits <= MAX_SPLITS and tps <= MAX_TPS, (n_tiles, n_splits, tps)
    return n_splits, tps


@functools.lru_cache(maxsize=None)
def _compiled_splitk(
    rows,
    q_shape,
    q_stride,
    group,
    width,
    n_splits,
    tps,
    tpb,
    scale,
    bt_shape,
    num_pages,
    k_strides,
    v_strides,
    pdl,
):
    heads = q_shape[1]
    kern = _SplitKKernel(
        group=group,
        width=width,
        splits=n_splits,
        tps=tps,
        tpb=tpb,
        scale=scale,
        bt_width=bt_shape[1],
        num_pages=num_pages,
        k_strides=k_strides,
        v_strides=v_strides,
        num_requests=bt_shape[0],
        pdl=pdl,
    )
    dev = "cuda"
    q = torch.empty_strided(q_shape, q_stride, dtype=torch.bfloat16, device=dev)
    kv = torch.empty(64, dtype=torch.uint8, device=dev)
    bt = torch.empty(bt_shape, dtype=torch.int32, device=dev)
    sel = torch.empty(rows, width, dtype=torch.int32, device=dev)
    req = torch.empty(rows, dtype=torch.int32, device=dev)
    part = torch.empty(64, dtype=torch.float32, device=dev)
    lse = torch.empty(n_splits, rows, heads, dtype=torch.float32, device=dev)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(
        kern,
        _dl(q, 16),
        _ptr(kv, cutlass.Uint8),
        _ptr(kv, cutlass.Uint8),
        _dl(bt),
        _dl(sel),
        _dl(req),
        _ptr(part, cutlass.Float32),
        _dl(lse),
        stream,
    )


def launch(
    q,
    k_cache,
    v_cache,
    block_table,
    selected_tokens,
    request_indices,
    tokens_per_block,
    softmax_scale,
    output_gate=None,
    pdl=True,
):
    """The launch behind `QsaPagedSparseGqaE4m3Decode.__call__`; returns `[rows, H, D]` in q's
    dtype."""
    rows, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group_size = num_q_heads // num_kv_heads
    width = selected_tokens.shape[1]
    output = torch.empty((rows, num_q_heads, head_dim), dtype=q.dtype, device=q.device)
    if rows == 0:
        return output
    n_splits, tps = plan(rows, num_kv_heads, width)
    dev = q.device
    use_pdl = bool(pdl and get_env_enable_pdl())
    partial_output = torch.empty(
        (n_splits, rows, num_q_heads, head_dim), dtype=torch.float32, device=dev
    )
    partial_lse = torch.empty((n_splits, rows, num_q_heads), dtype=torch.float32, device=dev)
    fn = _compiled_splitk(
        rows,
        tuple(q.shape),
        tuple(q.stride()),
        group_size,
        width,
        n_splits,
        tps,
        tokens_per_block,
        float(softmax_scale),
        tuple(block_table.shape),
        k_cache.shape[0],
        tuple(k_cache.stride()),
        tuple(v_cache.stride()),
        use_pdl,
    )
    stream = cuda_drv.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    fn(
        _dl(q, 16),
        _ptr(k_cache, cutlass.Uint8),
        _ptr(v_cache, cutlass.Uint8),
        _dl(block_table),
        _dl(selected_tokens),
        _dl(request_indices),
        _ptr(partial_output, cutlass.Float32),
        _dl(partial_lse),
        stream,
    )
    gate2d = None if output_gate is None else output_gate.reshape(rows, num_q_heads * head_dim)
    mfn = _compiled_merge(
        rows,
        num_q_heads,
        head_dim,
        n_splits,
        None if gate2d is None else gate2d.stride(0),
        q.dtype,
        use_pdl,
    )
    mfn(
        _dl(partial_output, 16),
        _dl(partial_lse),
        _dl(output.view(rows, num_q_heads * head_dim) if gate2d is None else gate2d, 2),
        _dl(output, 2),
        stream,
    )
    return output


def supports(
    q,
    k_cache,
    v_cache,
    block_table,
    selected_tokens,
    request_indices,
    tokens_per_block,
    max_rows: int = MAX_ROWS,
) -> bool:
    """Whether these operands are ones this entry serves (see `is_valid`); a caller keeps its
    other kernel where it says no."""
    if q.dim() != 3 or k_cache.dim() != 4 or selected_tokens.dim() != 2 or block_table.dim() != 2:
        return False
    rows, heads, dim = q.shape
    width = selected_tokens.shape[1]
    n_tiles = -(-width // CAP)
    tpb = int(tokens_per_block)
    return (
        0 < rows <= min(max_rows, MAX_ROWS)
        and dim == HEAD_DIM
        and q.dtype == torch.bfloat16
        and q.stride(2) == 1
        and k_cache.dtype == torch.float8_e4m3fn
        and v_cache.dtype == torch.float8_e4m3fn
        and tuple(v_cache.shape) == tuple(k_cache.shape)
        and k_cache.shape[3] == HEAD_DIM
        and k_cache.shape[2] == tpb
        and tpb > 0
        and tpb & (tpb - 1) == 0
        and heads % k_cache.shape[1] == 0
        and heads // k_cache.shape[1] <= MAX_GROUP
        # at least MIN_SPLITS tiles (the merge writes its whole chunk only from four splits
        # up), at most 64 splits of MAX_TPS tiles
        and MIN_SPLITS <= n_tiles <= MAX_SPLITS * MAX_TPS
        # the token table holds page * tpb + slot in an int32
        and k_cache.shape[0] * tpb < 2**31
        and all(t.data_ptr() % 16 == 0 for t in (q, k_cache, v_cache))
        and all(s % 8 == 0 for s in q.stride()[:2])
        # 16-byte cp.async chunks: e4m3 rows 16-byte aligned, unit inner stride
        and all(s % 16 == 0 for s in (*k_cache.stride()[:3], *v_cache.stride()[:3]))
        and k_cache.stride(3) == 1
        and v_cache.stride(3) == 1
        and block_table.dtype == torch.int32
        and block_table.is_contiguous()
        and selected_tokens.dtype == torch.int32
        and selected_tokens.is_contiguous()
        and selected_tokens.shape[0] == rows
        and request_indices.dtype == torch.int32
        and request_indices.dim() == 1
        and request_indices.is_contiguous()
        and request_indices.shape[0] == rows
    )


class QsaPagedSparseGqaE4m3Decode(OpWrapper):
    """`qsa_paged_sparse_gqa` without a prefill pair over an fp8 (e4m3) pool, for decode and
    verify rows of at most eight query heads a KV head: a split-K pass on half-height MMAs that
    builds its K and V fragments from the raw e4m3 rows, then the bf16 decode entry's merge --
    and, with `output_gate`, the gate -- as one launch. Returns a new `[rows, H, D]` tensor in
    q's dtype.

    Fusion boundary: the split-K attention, the merge, the optional sigmoid output gate.
    The selection, the K/V store and the output projection are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec keys: requests (default 1), rows_per_request (default len(pos)), pos (a row's
    # position, or each row's), heads_q / heads_kv (default the tep4 shard: 6 on 1), gated,
    # holes (half the selection -1).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tep4 verify, 1 request x 4 rows at position 100000, gated: 33 splits of a tile",
            spec=dict(pos=(100000, 100001, 100002, 100003), gated=True),
        ),
        Cell(
            why="tep4 verify, 2 requests x 4 rows, gated: 33 splits of a tile",
            spec=dict(requests=2, rows_per_request=4, pos=100000, gated=True),
        ),
        Cell(
            why="tep4 verify, 4 requests x 4 rows, gated: 17 splits of two tiles",
            spec=dict(requests=4, rows_per_request=4, pos=100000, gated=True),
        ),
        Cell(
            why="tep4 verify, 8 requests x 4 rows, gated: 9 splits of four tiles",
            spec=dict(requests=8, rows_per_request=4, pos=100000, gated=True),
        ),
        Cell(
            why="tep4 decode, 1 row with half the selection -1",
            spec=dict(pos=(8700,), gated=True, holes=True),
        ),
    )

    note = """
    Metadata consumed: as `qsa_paged_sparse_gqa` (block table, request indices). Preconditions:
    as that entry's, plus: no prefill pair; 1 .. `MAX_ROWS` (32) rows; bf16 q with a head dim
    of 256 and at most 8 query heads a KV head; an e4m3 K/V pool at unit scale whose rows are
    16-byte aligned (pointer, page / head / token strides) with unit inner stride; a
    power-of-two page size with `pages * tokens_per_block` below 2^31; more than 192 selected
    tokens a row (four tiles at least: the imported merge writes its whole output only from
    four splits up); a contiguous int32 block table, selection and request index;
    `output_gate` (when given) `[rows, H * D]` or `[rows, H, D]` with unit inner stride in q's
    dtype. `supports(...)` is the predicate a caller can check.

    PDL: the split-K pass reads q before its wait and the merge reads the output gate before
    its own; the request index, the selection, the block table and the K/V rows are read
    after the split-K pass's wait. So q and the gate must be complete when the grid ahead of
    this entry releases it: a launch without PDL releases at its end; a PDL producer of the
    selection (the target's top-k) must release only after its own wait.

    Numerics: the K/V widen is exact, so the operands are the per-row kernel's. One softmax
    over a 64-token tile, online across a split's tiles, and the merge sums the splits in split
    order, so the fp32 values can differ from `qsa_paged_sparse_gqa_splitk_cute`'s or the
    per-row kernel's in the last bits and the bf16 output by an ulp or two on some elements.
    The MMAs' zero rows add nothing: the scores are the full-height kernel's bit for bit.

    Cells: the tep4 shard (6 query heads on one KV head) on GB300 (SM103) at 1, 2, 4 and 8
    requests of four verify rows and one decode row, against the composition's reference. The
    kernel was measured as an engine overlay on that shape (outputs equal to the full-height
    kernel's at 4-32 rows); this file has not run on a GPU yet.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        output_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return launch(
            q,
            k_cache,
            v_cache,
            block_table,
            selected_tokens,
            request_indices,
            tokens_per_block,
            softmax_scale,
            output_gate,
        )

    def reference(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        output_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The composition entry's reference with the same operands."""
        return _composition.reference(
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
            output_gate=output_gate,
        )

    def is_valid(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        output_gate: Optional[torch.Tensor] = None,
    ) -> None:
        assert supports(
            q,
            k_cache,
            v_cache,
            block_table,
            selected_tokens,
            request_indices,
            tokens_per_block,
        ), "operands outside this entry's envelope (see `supports` and the note)"
        if output_gate is not None:
            assert output_gate.dtype == q.dtype and output_gate.stride(-1) == 1, "gate in q's dtype"
            assert output_gate.numel() == q.numel(), "gate covers q"


qsa_paged_sparse_gqa_e4m3_decode = QsaPagedSparseGqaE4m3Decode()

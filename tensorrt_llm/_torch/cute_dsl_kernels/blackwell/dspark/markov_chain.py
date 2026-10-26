# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark Markov block chain for a vocabulary-sharded (tensor-parallel) draft LM head, one launch.

For every draft position ``k`` of a block and every batch row ``b``::

    bias = bf16(W2 @ W1[tok[b, k - 1]])                (FP32 dot, BF16 result)
    tok[b, k] = argmax_v  fp32(base[b, k, v]) + fp32(bias[v])

with ``tok[b, -1]`` the bonus token; NaN ranks first, ties take the smallest vocabulary id and an
id outside the vocabulary embeds as zeros (the semantics of ``CompactTPArgmax``). ``base`` and
``W2`` hold this rank's contiguous vocabulary shard, ``W1`` is the full embedding.

One persistent launch runs all ``K`` positions. CTA ``c`` owns shard rows ``[c * R, (c + 1) * R)``
with one thread per row and keeps that ``W2`` slice in shared memory for the whole chain. Per
position each CTA reduces its rows to one candidate per batch row and pushes it to every rank;
every CTA then reduces the ranks' winners itself, so neither a grid barrier nor a second launch is
needed. The exchange runs on the triple buffer and flag / clear-size words of an all-reduce fusion
workspace of its own (as in ``MhcAllReduceBoundaryKernel``): one launch advances its flag once and
the ``K`` positions use disjoint slots of that launch's buffer.

It takes two hops: each CTA stores its candidate of a row as one 8-byte entry in a local slot of
its own rank, CTA 0 waits for the ``CTAs`` local entries, reduces them and stores the rank's winner
into slot (position, row, rank) of every rank, and every CTA reduces the ``ranks`` winners. Every
entry carries the launch's tag (the launch epoch + 1, kept in the flag word and advanced by the
kernel at the end of every launch, wrapping before 2^14 at a multiple of 3); a reader takes an
entry only once it carries the current tag, and each launch zeroes (tag 0) the extent of the
previous launch's slot.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

from tensorrt_llm.math_utils import ceil_div, pad_up

from ..mhc.mhc_split_boundary import (
    _atomic_add_s32,
    _fma,
    _ld_volatile_s32,
    _ld_volatile_s64,
    _st_s32,
    _st_s64,
    _st_v4,
)

VEC = 8  # BF16 elements per 16-byte vector
_ABS_MASK = 0x7FFFFFFF
_F32_INF_BITS = 0x7F800000
# Exchange entry (one 8-byte store): low word = the candidate's value bits, high word =
# (launch tag << _INDEX_BITS) | vocabulary id. Tags run 1 .. _TAG_MOD (the launch epoch + 1); 0 marks
# a cleared entry. _TAG_MOD is a multiple of 3, so the epoch's slot (epoch % 3) keeps rotating
# across the wrap.
_INDEX_BITS = 18
_INDEX_MASK = (1 << _INDEX_BITS) - 1
_TAG_MOD = (1 << 14) - 1


@dsl_user_op
def _f32_bits(x: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bits_f32(x: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``add.rn.f32`` (the boundary of the reference's FP32 logit + bias addition)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_relaxed_b64(addr: Int64, value: Int64, sys: bool, *, loc=None, ip=None) -> None:
    """One single-copy-atomic 8-byte store (``.sys`` scope for a peer's memory, ``.gpu`` for this
    GPU's)."""
    scope = "sys" if sys else "gpu"
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        f"st.relaxed.{scope}.global.b64 [$0], $1;",
        "l,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_relaxed_b64(addr: Int64, sys: bool, *, loc=None, ip=None) -> Int64:
    """One single-copy-atomic 8-byte load, re-read from memory every time."""
    scope = "sys" if sys else "gpu"
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            f"ld.relaxed.{scope}.global.b64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _entry(value: Float32, index: Int32, tag: Int32) -> Int64:
    """The 8-byte exchange entry of a candidate (see ``_INDEX_BITS``)."""
    high = (Int64(tag) << _INDEX_BITS) | (Int64(index) & Int64(_INDEX_MASK))
    return (high << 32) | (Int64(_f32_bits(value)) & Int64(0xFFFFFFFF))


@cute.jit
def _entry_tag(entry: Int64) -> Int32:
    return Int32((entry >> (32 + _INDEX_BITS)) & Int64(_TAG_MOD))


@cute.jit
def _await_entry(addr: Int64, tag: Int32, sys: bool) -> Int64:
    """The entry at ``addr`` once it carries ``tag`` (polls)."""
    entry = _ld_relaxed_b64(addr, sys)
    while _entry_tag(entry) != tag:
        entry = _ld_relaxed_b64(addr, sys)
    return entry


@cute.jit
def _is_nan(x: Float32) -> cutlass.Boolean:
    return (_f32_bits(x) & Int32(_ABS_MASK)) > Int32(_F32_INF_BITS)


@cute.jit
def _prefer(av: Float32, ai: Int32, bv: Float32, bi: Int32):
    """The preferred (value, id) of two candidates: NaN first, larger value, smaller id on ties."""
    a_nan = _is_nan(av)
    b_nan = _is_nan(bv)
    take_a = cutlass.Boolean(False)
    if a_nan:
        take_a = cutlass.Boolean(True)
        if b_nan:
            take_a = ai < bi
    if not a_nan:
        if not b_nan:
            if av > bv:
                take_a = cutlass.Boolean(True)
            if av == bv:
                take_a = ai < bi
    rv = bv
    ri = bi
    if take_a:
        rv = av
        ri = ai
    return rv, ri


@cute.jit
def _warp_prefer(v: Float32, i: Int32):
    """Butterfly reduction of one candidate per lane; every lane ends with the warp's best."""
    for s in cutlass.range_constexpr(5):
        ov = cute.arch.shuffle_sync_bfly(v, offset=16 >> s)
        oi = cute.arch.shuffle_sync_bfly(i, offset=16 >> s)
        v, i = _prefer(v, i, ov, oi)
    return v, i


class DSparkMarkovChainKernel:
    """``K`` Markov positions for ``B`` rows over a ``local_vocab`` shard, ``num_ctas`` CTAs.

    Tensors:
      base   [B * K, local_vocab]  bf16   this rank's draft-head logits (row ``b * K + k``)
      w2     [local_vocab, D]      bf16   this rank's Markov projection rows (D = 256)
      w1     [vocab, D]            bf16   full Markov embedding
      bonus  [B]                   int64  token preceding position 0
      out    [B, K]                int32  draft tokens (-1 when no candidate exists)
      ws     [nranks + 2]          int64  exchange buffers of every rank, flag words, layout words
    """

    threads = 256

    def __init__(
        self,
        *,
        batch: int,
        block: int,
        dim: int,
        local_vocab: int,
        vocab: int,
        nranks: int,
        num_ctas: int,
    ):
        threads = self.threads
        if dim != 32 * VEC:
            raise ValueError(f"unsupported Markov rank {dim}")
        if nranks not in (2, 4, 8):
            raise ValueError(f"unsupported rank count {nranks}")
        rows = ceil_div(local_vocab, num_ctas)
        if rows > threads:
            raise ValueError(f"{num_ctas} CTAs leave {rows} rows per CTA (> {threads} threads)")
        if vocab > _INDEX_MASK or num_ctas > threads:
            raise ValueError(
                f"the exchange needs vocab <= {_INDEX_MASK} and CTAs <= threads "
                f"(vocab {vocab}, {num_ctas} CTAs, {threads} threads)"
            )
        self.batch = batch
        self.block = block
        self.dim = dim
        self.local_vocab = local_vocab
        self.vocab = vocab
        self.nranks = nranks
        self.num_ctas = num_ctas
        self.num_warps = threads // 32
        self.rows_per_cta = rows
        # Padded shared-memory rows: 16-byte reads of consecutive rows hit distinct banks.
        self.row_stride = dim + VEC
        # 8-byte entries: local slots [K, B, CTAs], then rank slots [K, B, ranks] (16-byte
        # aligned); the 16-byte vectors one launch writes into this rank's buffer.
        self.local_bytes = pad_up(block * batch * num_ctas * 8, 16)
        rank_bytes = pad_up(block * batch * nranks * 8, 16)
        self.launch_slots = (self.local_bytes + rank_bytes) // 16

    def smem_bytes(self) -> int:
        w2 = self.rows_per_cta * self.row_stride * 2
        base = self.batch * self.block * self.rows_per_cta * 2
        lat = self.batch * self.dim * 4
        red = 4 * self.num_warps * self.batch * 4 + self.batch * 4
        ranks = 2 * self.nranks * self.batch * 4
        # Each of the 10 allocations may be padded to its 16-byte alignment.
        return w2 + base + lat + red + ranks + 10 * 16

    @cute.jit
    def __call__(
        self,
        gBase: cute.Tensor,
        gW2: cute.Tensor,
        gW1: cute.Tensor,
        gBonus: cute.Tensor,
        gOut: cute.Tensor,
        gWs: cute.Tensor,
        rank: Int32,
        offset: Int32,
        stream,
    ) -> None:
        copy_bf16 = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
        )
        copy_f32 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128)
        self.kernel(
            gBase, gW2, gW1, gBonus, gOut, gWs, rank, offset, copy_bf16, copy_vec, copy_f32
        ).launch(
            grid=[self.num_ctas, 1, 1],
            block=[self.threads, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gBase: cute.Tensor,
        gW2: cute.Tensor,
        gW1: cute.Tensor,
        gBonus: cute.Tensor,
        gOut: cute.Tensor,
        gWs: cute.Tensor,
        rank: Int32,
        offset: Int32,
        copy_bf16: cute.CopyAtom,
        copy_vec: cute.CopyAtom,
        copy_f32: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        B: cutlass.Constexpr = self.batch
        K: cutlass.Constexpr = self.block
        D: cutlass.Constexpr = self.dim
        RPC: cutlass.Constexpr = self.rows_per_cta
        NW: cutlass.Constexpr = self.num_warps
        NT: cutlass.Constexpr = self.threads
        NR: cutlass.Constexpr = self.nranks
        NC: cutlass.Constexpr = self.num_ctas
        V: cutlass.Constexpr = self.vocab
        VL: cutlass.Constexpr = self.local_vocab
        neg_inf = Float32(float("-inf"))

        row0 = bid * RPC
        rows = Int32(VL) - row0
        if rows > RPC:
            rows = Int32(RPC)

        smem = SmemAllocator()
        w2s = smem.allocate_tensor(
            BFloat16, cute.make_layout((RPC, D), stride=(self.row_stride, 1)), byte_alignment=16
        )
        bs = smem.allocate_tensor(
            BFloat16, cute.make_layout((B * K, RPC), stride=(RPC, 1)), byte_alignment=16
        )
        latf = smem.allocate_tensor(
            Float32, cute.make_layout((B, D), stride=(D, 1)), byte_alignment=16
        )
        # Per-warp candidates of the CTA's rows (push) and of the rank's local entries (poll).
        push_v = smem.allocate_tensor(
            Float32, cute.make_layout((NW, B), stride=(B, 1)), byte_alignment=16
        )
        push_i = smem.allocate_tensor(
            Int32, cute.make_layout((NW, B), stride=(B, 1)), byte_alignment=16
        )
        poll_v = smem.allocate_tensor(
            Float32, cute.make_layout((NW, B), stride=(B, 1)), byte_alignment=16
        )
        poll_i = smem.allocate_tensor(
            Int32, cute.make_layout((NW, B), stride=(B, 1)), byte_alignment=16
        )
        tok = smem.allocate_tensor(Int32, cute.make_layout((B,)), byte_alignment=16)
        # The ranks' winners of every row.
        rank_v = smem.allocate_tensor(
            Float32, cute.make_layout((NR, B), stride=(B, 1)), byte_alignment=16
        )
        rank_i = smem.allocate_tensor(
            Int32, cute.make_layout((NR, B), stride=(B, 1)), byte_alignment=16
        )

        gW2_v = cute.logical_divide(gW2, (None, VEC))
        gW1_v = cute.logical_divide(gW1, (None, VEC))
        w2s_v = cute.logical_divide(w2s, (None, VEC))
        latf_v = cute.logical_divide(latf, (None, 4))

        # ---- this CTA's W2 rows -> shared memory (not written by the producer kernel); all of a
        # thread's loads are issued before its stores so the slice streams at full bandwidth ----
        vecs_per_row: cutlass.Constexpr = D // VEC
        w2_iters: cutlass.Constexpr = ceil_div(RPC * vecs_per_row, NT)
        chunk: cutlass.Constexpr = 14
        wreg = cute.make_rmem_tensor(cute.make_layout((chunk, VEC), stride=(VEC, 1)), BFloat16)
        w2_vecs = rows * vecs_per_row
        for c0 in cutlass.range_constexpr(0, w2_iters, chunk):
            for i in cutlass.range_constexpr(chunk):
                if c0 + i < w2_iters:
                    # Unconditional (clamped) loads keep the chunk's loads in flight together.
                    v = tidx + (c0 + i) * NT
                    if v >= w2_vecs:
                        v = w2_vecs - 1
                    cute.copy(
                        copy_bf16,
                        gW2_v[row0 + v // vecs_per_row, (None, v % vecs_per_row)],
                        wreg[i, None],
                    )
            for i in cutlass.range_constexpr(chunk):
                if c0 + i < w2_iters:
                    v = tidx + (c0 + i) * NT
                    if v < w2_vecs:
                        cute.copy(
                            copy_vec,
                            wreg[i, None],
                            w2s_v[v // vecs_per_row, (None, v % vecs_per_row)],
                        )

        # ---- exchange state of this launch (read before any CTA bumps the arrival counter) ----
        # ``flag`` is the launch epoch (0 .. _TAG_MOD - 1: slot ``flag % 3``, entry tag
        # ``flag + 1``), advanced by this kernel at the end of every launch.
        flag_addr = Int64(gWs[NR])
        layout_addr = Int64(gWs[NR + 1])
        flag = _ld_volatile_s32(flag_addr + Int64(8))
        clear_size = _ld_volatile_s64(layout_addr)
        comm_size = _ld_volatile_s64(layout_addr + Int64(8))
        tag = flag + 1

        # ---- the CTA's base logits of every position and row ----
        bs_iters: cutlass.Constexpr = ceil_div(B * K * RPC, NT)
        breg = cute.make_rmem_tensor(cute.make_layout((bs_iters,)), BFloat16)
        bs_elems = B * K * rows
        for i in cutlass.range_constexpr(bs_iters):
            e = tidx + i * NT
            if e < bs_elems:
                breg[i] = gBase[e // rows, row0 + e % rows]
        for i in cutlass.range_constexpr(bs_iters):
            e = tidx + i * NT
            if e < bs_elems:
                bs[e // rows, e % rows] = breg[i]

        data_off = Int64(flag % 3) * comm_size
        clear_off = Int64((flag + 2) % 3) * comm_size
        own = Int64(gWs[rank])
        cute.arch.sync_threads()
        if tidx == 0:
            _atomic_add_s32(flag_addr, Int32(1))

        # ---- clear the slots the previous launch wrote into this rank's buffer (tag 0 is never a
        # launch's tag) ----
        cleared = Int32(0)
        clear_vecs = clear_size // Int64(VEC)
        c = Int64(bid) * Int64(NT) + Int64(tidx)
        while c < clear_vecs:
            _st_v4(own + clear_off + c * Int64(16), cleared, cleared, cleared, cleared)
            c = c + Int64(NC * NT)

        prev = cute.make_rmem_tensor(cute.make_layout((B,)), Int32)
        for b in cutlass.range_constexpr(B):
            prev[b] = Int32(gBonus[b])
        vec = cute.make_rmem_tensor(cute.make_layout((VEC,)), BFloat16)
        lfa = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        lfb = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        wv = cute.make_rmem_tensor(cute.make_layout((VEC,)), BFloat16)

        for k in cutlass.range_constexpr(K):
            # ---- latent rows W1[prev] in FP32 (zeros for ids outside the vocabulary) ----
            if tidx < B * vecs_per_row:
                b = tidx // vecs_per_row
                cv = tidx % vecs_per_row
                t = Int32(-1)
                for bb in cutlass.range_constexpr(B):
                    if b == bb:
                        t = prev[bb]
                for i in cutlass.range_constexpr(VEC):
                    vec[i] = BFloat16(0.0)
                if t >= 0:
                    if t < V:
                        cute.copy(copy_bf16, gW1_v[t, (None, cv)], vec)
                for i in cutlass.range_constexpr(4):
                    lfa[i] = vec[i].to(Float32)
                    lfb[i] = vec[4 + i].to(Float32)
                cute.copy(copy_f32, lfa, latf_v[b, (None, 2 * cv)])
                cute.copy(copy_f32, lfb, latf_v[b, (None, 2 * cv + 1)])
            cute.arch.sync_threads()

            # ---- one row per thread: bias = bf16(W2[row] . latent), logit = base + bias ----
            cand_v = cute.make_rmem_tensor(cute.make_layout((B,)), Float32)
            cand_i = cute.make_rmem_tensor(cute.make_layout((B,)), Int32)
            for b in cutlass.range_constexpr(B):
                cand_v[b] = neg_inf
                cand_i[b] = Int32(V)
            gid = offset + row0 + tidx
            active = cutlass.Boolean(False)
            if tidx < rows:
                if gid < V:
                    active = cutlass.Boolean(True)
            if active:
                acc = cute.make_rmem_tensor(cute.make_layout((B,)), Float32)
                for b in cutlass.range_constexpr(B):
                    acc[b] = Float32(0.0)
                for cc in cutlass.range(vecs_per_row):
                    cute.copy(copy_vec, w2s_v[tidx, (None, cc)], wv)
                    for b in cutlass.range_constexpr(B):
                        cute.copy(copy_f32, latf_v[b, (None, 2 * cc)], lfa)
                        cute.copy(copy_f32, latf_v[b, (None, 2 * cc + 1)], lfb)
                        for i in cutlass.range_constexpr(4):
                            acc[b] = _fma(lfa[i], wv[i].to(Float32), acc[b])
                        for i in cutlass.range_constexpr(4):
                            acc[b] = _fma(lfb[i], wv[4 + i].to(Float32), acc[b])
                for b in cutlass.range_constexpr(B):
                    bias = acc[b].to(BFloat16).to(Float32)
                    cand_v[b] = _add_rn(bs[b * K + k, tidx].to(Float32), bias)
                    cand_i[b] = gid

            # ---- CTA candidate per batch row ----
            for b in cutlass.range_constexpr(B):
                wv_b, wi_b = _warp_prefer(cand_v[b], cand_i[b])
                if lane == 0:
                    push_v[warp, b] = wv_b
                    push_i[warp, b] = wi_b
            cute.arch.sync_threads()

            self._exchange(
                k,
                push_v,
                push_i,
                poll_v,
                poll_i,
                rank_v,
                rank_i,
                tok,
                gOut,
                gWs,
                own,
                data_off,
                tag,
                rank,
                bid,
                tidx,
                warp,
                lane,
                neg_inf,
            )
            for b in cutlass.range_constexpr(B):
                prev[b] = tok[b]

        # ---- hand the triple buffer to the next launch once every CTA has read the flag ----
        if bid == 0:
            if tidx == 0:
                arrived = _ld_volatile_s32(flag_addr)
                while arrived != NC:
                    arrived = _ld_volatile_s32(flag_addr)
                _st_s32(flag_addr + Int64(8), (flag + 1) % _TAG_MOD)
                _st_s64(layout_addr, Int64(self.launch_slots * VEC))
                _st_s32(flag_addr, Int32(0))

    @cute.jit
    def _exchange(
        self,
        k,
        push_v,
        push_i,
        poll_v,
        poll_i,
        rank_v,
        rank_i,
        tok,
        gOut,
        gWs,
        own,
        data_off,
        tag,
        rank,
        bid,
        tidx,
        warp,
        lane,
        neg_inf,
    ) -> None:
        """Position ``k``'s winner of every row into ``tok`` (and ``gOut`` by CTA 0); ``push_v`` /
        ``push_i`` hold the CTA's per-warp candidates."""
        B: cutlass.Constexpr = self.batch
        NR: cutlass.Constexpr = self.nranks
        NC: cutlass.Constexpr = self.num_ctas
        NW: cutlass.Constexpr = self.num_warps
        V: cutlass.Constexpr = self.vocab
        rank_off = data_off + Int64(self.local_bytes)
        # ---- this CTA's candidate of every row -> its slot (k, row, CTA) of this rank ----
        if tidx < B:
            cv = push_v[0, tidx]
            ci = push_i[0, tidx]
            for w in cutlass.range_constexpr(1, NW):
                cv, ci = _prefer(cv, ci, push_v[w, tidx], push_i[w, tidx])
            local_slot = Int64((k * B + tidx) * NC) + Int64(bid)
            _st_relaxed_b64(own + data_off + local_slot * Int64(8), _entry(cv, ci, tag), False)
        # ---- CTA 0: this rank's winner of every row -> slot (k, row, rank) of every rank ----
        if bid == 0:
            for b in cutlass.range_constexpr(B):
                lv = neg_inf
                li = Int32(V)
                if tidx < NC:
                    entry = _await_entry(
                        own + data_off + (Int64((k * B + b) * NC) + Int64(tidx)) * Int64(8),
                        tag,
                        False,
                    )
                    lv = _bits_f32(Int32(entry & Int64(0xFFFFFFFF)))
                    li = Int32((entry >> 32) & Int64(_INDEX_MASK))
                wv, wi = _warp_prefer(lv, li)
                if lane == 0:
                    poll_v[warp, b] = wv
                    poll_i[warp, b] = wi
            cute.arch.sync_threads()
            if tidx < B * NR:
                row = tidx // NR
                dst = tidx % NR
                rv = poll_v[0, row]
                ri = poll_i[0, row]
                for w in cutlass.range_constexpr(1, NW):
                    rv, ri = _prefer(rv, ri, poll_v[w, row], poll_i[w, row])
                rank_slot = Int64((k * B + row) * NR) + Int64(rank)
                _st_relaxed_b64(
                    Int64(gWs[dst]) + rank_off + rank_slot * Int64(8), _entry(rv, ri, tag), True
                )
        # ---- every CTA: the ranks' winners of every row ----
        if tidx < B * NR:
            row = tidx // NR
            src = tidx % NR
            entry = _await_entry(
                own + rank_off + (Int64((k * B + row) * NR) + Int64(src)) * Int64(8), tag, True
            )
            rank_v[src, row] = _bits_f32(Int32(entry & Int64(0xFFFFFFFF)))
            rank_i[src, row] = Int32((entry >> 32) & Int64(_INDEX_MASK))
        cute.arch.sync_threads()
        if tidx < B:
            tv = rank_v[0, tidx]
            ti = rank_i[0, tidx]
            for r in cutlass.range_constexpr(1, NR):
                tv, ti = _prefer(tv, ti, rank_v[r, tidx], rank_i[r, tidx])
            if ti >= V:
                ti = Int32(-1)
            tok[tidx] = ti
            if bid == 0:
                gOut[tidx, k] = ti
        cute.arch.sync_threads()

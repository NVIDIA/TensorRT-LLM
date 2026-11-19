# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Threshold-pruned exchange and exact merge of the key-sharded DSA decode Top-K (CuTe DSL, SM100 / SM103).

Rank ``w`` of ``W`` holds each decode row's exact local Top-``L`` (local column indices from the local radix Top-K)
over the row's 64-token key blocks ``b % W == w``. Three launches per call, one CTA per row each, build the row's
global Top-``K`` on every rank:

* ladder: the CTA turns the list into keys, sorts them descending (bitonic: registers, warp shuffles and shared
  memory), keeps them for the next launch and sends every rank a ladder of ``P`` keys: the ``t_j``-th largest for
  ``t_j = (j + 1) * K / (4 W)`` (``j < P - 1``, while ``t_j`` is within the list) and the list's smallest key;
* push: it waits for every rank's ladder and takes the bound ``T``: the largest ladder key ``v`` with
  ``sum_w count_w(v) >= K``, where ``count_w(v)`` is the largest ladder rank of rank ``w`` whose key is ``>= v`` (rank
  ``w`` holds at least that many keys ``>= v``). So ``T`` is at most the row's ``K``-th key, and every rank computes
  the same ``T`` from the same ladders. It sends every rank its listed keys ``>= T`` (a prefix of the sorted list) and
  a header: their count, whether the list holds all of the rank's keys, and the score of the list's smallest key;
* merge: it waits for every rank's header and keys, selects the ``K`` largest received keys (MSB radix select in
  shared memory) and checks the certificate: a rank whose list does not hold all its keys and whose smallest listed
  score is not below the ``K``-th selected score may hold unsent members of the Top-``K``; such a row is re-selected
  exactly from every rank's local scores, read through the peer mappings. It writes the token indices (any order).

Keys: ``(ordered fp32 score << 32) | (2^32 - 1 - token)``; a larger key is a better candidate and ties at a score go
to the smaller token: the deterministic GVR order. The certificate compares scores only, so the local Top-K may break
ties at its boundary score in any order. The selected set is the row's exact global Top-``K`` (every token when the
row has at most ``K``), which the order + pool remap step turns into the replicated path's rows bit for bit.

Transport (Lamport): every exchanged 32-bit word of the workspace starts as ``0x80000000`` (empty). Senders store
their words into every rank's workspace (one multicast store, or one store per peer mapping); a receiver polls until
none of the words it needs is empty and re-arms the words it read once it has used them. A key's score word is never
``0x80000000``: that ordered value belongs to the score bit pattern ``0xFFFFFFFF`` (a negative NaN with a full
payload), which takes the next value. The receive slots alternate by the parity of a device call counter that the
merge's last CTA advances after every CTA has read it; a rank writes a slot again two calls later, after every peer
has re-armed it (it has received every peer's keys of the call in between).

Calls on a workspace must be made in the same order on every rank of the group. The local scores (fallback reads
them through the peer mappings) are overwritten by the next call's logits on each rank; the TP collectives between
two indexer calls order that after every peer's merge.
"""

import functools
import os
from typing import List

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm import lamport_ar as _lam

#: Tokens per index-K cache block: the unit of the key split.
TOKENS_PER_BLOCK = 64
#: Top-K of a row.
TOPK = 2048
#: Ladder keys per rank and row.
P = 8
#: Receive slots (alternating by call parity).
NUM_SLOTS = 2
#: Empty Lamport word.
SENTINEL = -2147483648
_MIN64 = -(2**63)
_MAX64 = 2**63 - 1
#: Ladder entry beyond a short list: below every real key, no empty word.
_ABSENT = 0x8000000100000001 - (1 << 64)
#: Threads of a merge CTA.
MERGE_THREADS = 512

_asm = _lam._asm
_iv = _lam._iv
_lv = _lam._lv


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def _lds64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.b64 $0, [$1];", "=l,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _sts64(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _lv(v, loc=loc, ip=ip)],
        "st.shared.b64 [$0], $1;",
        "r,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _sts32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_shared_add(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
        "red.shared.add.u32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atom_shared_add(addr, v, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
            "atom.shared.add.u32 $0, [$1], $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ldg32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_lv(addr, loc=loc, ip=ip)], "ld.global.b32 $0, [$1];", "=r,l", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _ldv32(addr, *, loc=None, ip=None):
    """32-bit read past L1 (the call counter, a peer's scores)."""
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.volatile.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ldg_v2(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _ldg64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(), [_lv(addr, loc=loc, ip=ip)], "ld.global.b64 $0, [$1];", "=l,l", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _stg32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mkkey(bits, token, *, loc=None, ip=None):
    """Order key of an fp32 score (raw bits) at ``token``."""
    return Int64(
        _asm(
            T.i64(),
            [_iv(bits, loc=loc, ip=ip), _iv(token, loc=loc, ip=ip)],
            "{\n.reg .b32 o, x, lo;\n.reg .pred p;\n"
            "setp.lt.s32 p, $1, 0;\nxor.b32 x, $1, 0x7fffffff;\nselp.b32 o, x, $1, p;\n"
            "setp.eq.u32 p, o, 0x80000000;\nselp.b32 o, 0x80000001, o, p;\n"
            "not.b32 lo, $2;\nmov.b64 $0, {lo, o};\n}",
            "=l,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _split(v, *, loc=None, ip=None):
    """(low word, high word) of a 64-bit value."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_lv(v, loc=loc, ip=ip)],
        "mov.b64 {$0, $1}, $2;",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _join(lo, hi, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_iv(lo, loc=loc, ip=ip), _iv(hi, loc=loc, ip=ip)],
            "mov.b64 $0, {$1, $2};",
            "=l,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _byte(v, shift, *, loc=None, ip=None):
    """Bits [shift, shift + 8) of a 64-bit value."""
    return Int32(
        _asm(
            T.i32(),
            [_lv(v, loc=loc, ip=ip), _iv(shift, loc=loc, ip=ip)],
            "{\n.reg .b64 x;\nshr.b64 x, $1, $2;\ncvt.u32.u64 $0, x;\nand.b32 $0, $0, 255;\n}",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shr64(v, shift, *, loc=None, ip=None):
    """Logical right shift (amounts >= 64 give 0)."""
    return Int64(
        _asm(
            T.i64(),
            [_lv(v, loc=loc, ip=ip), _iv(shift, loc=loc, ip=ip)],
            "shr.b64 $0, $1, $2;",
            "=l,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shl64(v, shift, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(v, loc=loc, ip=ip), _iv(shift, loc=loc, ip=ip)],
            "shl.b64 $0, $1, $2;",
            "=l,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _max64(a, b, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(a, loc=loc, ip=ip), _lv(b, loc=loc, ip=ip)],
            "max.s64 $0, $1, $2;",
            "=l,l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _min64(a, b, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(a, loc=loc, ip=ip), _lv(b, loc=loc, ip=ip)],
            "min.s64 $0, $1, $2;",
            "=l,l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sel64(flag, a, b, *, loc=None, ip=None):
    """``a`` if ``flag`` != 0 else ``b``."""
    return Int64(
        _asm(
            T.i64(),
            [_iv(flag, loc=loc, ip=ip), _lv(a, loc=loc, ip=ip), _lv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.b64 $0, $2, $3, p;\n}",
            "=l,r,l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_xor64(v, mask, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(v, loc=loc, ip=ip), _iv(mask, loc=loc, ip=ip)],
            "{\n.reg .b32 lo, hi;\nmov.b64 {lo, hi}, $1;\n"
            "shfl.sync.bfly.b32 lo, lo, $2, 0x1f, 0xffffffff;\n"
            "shfl.sync.bfly.b32 hi, hi, $2, 0x1f, 0xffffffff;\nmov.b64 $0, {lo, hi};\n}",
            "=l,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_xor32(v, mask, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(v, loc=loc, ip=ip), _iv(mask, loc=loc, ip=ip)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_idx32(v, src, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(v, loc=loc, ip=ip), _iv(src, loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ballot(flag, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(flag, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nvote.sync.ballot.b32 $0, p, 0xffffffff;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _popc(v, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_iv(v, loc=loc, ip=ip)], "popc.b32 $0, $1;", "=r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _top_bit(v, *, loc=None, ip=None):
    """Index of the highest set bit (v != 0)."""
    return Int32(
        _asm(T.i32(), [_iv(v, loc=loc, ip=ip)], "bfind.u32 $0, $1;", "=r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _low_bit(v, *, loc=None, ip=None):
    """Index of the lowest set bit (v != 0)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(v, loc=loc, ip=ip)],
            "{\n.reg .b32 x;\nbrev.b32 x, $1;\nbfind.shiftamt.u32 $0, x;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lanemask_lt(*, loc=None, ip=None):
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %lanemask_lt;", "=r", loc=loc, ip=ip))


@dsl_user_op
def _warp_suffix_sum(v, *, loc=None, ip=None):
    """Sum over this lane and the lanes above it."""
    body = "".join(
        f"shfl.sync.down.b32 t|p, x, {o}, 0x1f, 0xffffffff;\n@p add.s32 x, x, t;\n"
        for o in (1, 2, 4, 8, 16)
    )
    return Int32(
        _asm(
            T.i32(),
            [_iv(v, loc=loc, ip=ip)],
            "{\n.reg .b32 x, t;\n.reg .pred p;\nmov.b32 x, $1;\n" + body + "mov.b32 $0, x;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _clean_pair(w0, w1, w2, w3, second, *, loc=None, ip=None):
    """1 if words 0-1 have arrived and, when ``second`` != 0, words 2-3 too."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(x, loc=loc, ip=ip) for x in (w0, w1, w2, w3, second)],
            "{\n.reg .pred p, q;\n"
            "setp.ne.b32 p, $1, 0x80000000;\nsetp.ne.and.b32 p, $2, 0x80000000, p;\n"
            "setp.ne.b32 q, $3, 0x80000000;\nsetp.ne.and.b32 q, $4, 0x80000000, q;\n"
            "setp.eq.or.s32 q, $5, 0, q;\nand.pred p, p, q;\nselp.u32 $0, 1, 0, p;\n}",
            "=r,r,r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# Shared device code
# ---------------------------------------------------------------------------


@cute.jit
def _local_count(n: Int32, w, world: cutlass.Constexpr):
    """Tokens below ``n`` in the 64-token blocks ``b % world == w``."""
    full = n // TOKENS_PER_BLOCK
    owned = (full - w + (world - 1)) // world
    if owned < 0:
        owned = Int32(0)
    part = Int32(0)
    if full % world == w:
        part = n - full * TOKENS_PER_BLOCK
    return owned * TOKENS_PER_BLOCK + part


@cute.jit
def _token(i: Int32, w, world: cutlass.Constexpr):
    """Token of local column ``i`` of rank ``w``."""
    return ((i >> 6) * world + w) * TOKENS_PER_BLOCK + (i & (TOKENS_PER_BLOCK - 1))


@cute.jit
def _row_len(kv_lens: Int64, next_n: Int32, r):
    """Keys visible to decode row ``r`` (verify row ``r % next_n`` of sequence ``r // next_n``)."""
    seq = r // next_n
    return _ldg32(kv_lens + Int64(seq) * 4) - next_n + 1 + (r - seq * next_n)


@cute.jit
def _poll_v4(addr: Int64, second: Int32, max_spin: Int32, state: Int64):
    """The 16 bytes at ``addr`` once words 0-1 (and 2-3 when ``second``) have arrived; a wait longer than
    ``max_spin`` polls is counted in ``state[3]`` and given up."""
    w = _lam.ld_volatile_v4(addr)
    ok = _clean_pair(w[0], w[1], w[2], w[3], second)
    n = Int32(0)
    while ok == 0:
        w = _lam.ld_volatile_v4(addr)
        ok = _clean_pair(w[0], w[1], w[2], w[3], second)
        n = n + 1
        if n >= max_spin:
            _lam.atom_add_relaxed_u32(state + Int64(12), Int32(1))
            ok = Int32(1)
    return w


@cute.jit
def _send_v4(
    mcast: cutlass.Constexpr,
    world: cutlass.Constexpr,
    mc: Int64,
    peers: Int64,
    off: Int64,
    w0,
    w1,
    w2,
    w3,
):
    """16 bytes at offset ``off`` of every rank's workspace: one multicast store, or one store per peer mapping."""
    if cutlass.const_expr(mcast):
        _lam.multimem_st_v4(mc + off, w0, w1, w2, w3)
    else:
        for d in cutlass.range_constexpr(world):
            _lam.st_v4(_lam.ld_nc_u64(peers + Int64(d * 8)) + off, w0, w1, w2, w3)


@cute.jit
def _scan_bins(hist: Int32, lane, remaining: Int32):
    """Warp-wide: (b*, keys above b*, keys in b*) of a 256-bin histogram, b* holding the ``remaining``-th largest key
    counting from the top bin."""
    hs = []
    for q in cutlass.range_constexpr(8):
        hs.append(_lds32(hist + (lane * 8 + q) * 4))
    s = hs[0]
    for q in cutlass.range_constexpr(1, 8):
        s = s + hs[q]
    suf = _warp_suffix_sum(s)
    lb = _top_bit(_ballot(Int32(suf >= remaining)))
    bstar = Int32(0)
    above = Int32(0)
    cnt = Int32(0)
    if lane == lb:
        run = suf - s
        found = Int32(0)
        for q in cutlass.range_constexpr(8):
            qq = 7 - q
            if found == 0:
                if run + hs[qq] >= remaining:
                    bstar = lane * 8 + qq
                    above = run
                    cnt = hs[qq]
                    found = Int32(1)
                else:
                    run = run + hs[qq]
    bstar = _shfl_idx32(bstar, lb)
    above = _shfl_idx32(above, lb)
    cnt = _shfl_idx32(cnt, lb)
    return bstar, above, cnt


@cute.jit
def _emit_warp(sel: Int32, key: Int64, token: Int32, lane, ctr: Int32, orow: Int64):
    """Write ``token`` of every lane with ``sel`` to the next free slots of the output row (one shared atomic per
    warp)."""
    bal = _ballot(sel)
    pos0 = Int32(0)
    if bal != 0:
        lead = _low_bit(bal)
        if lane == lead:
            pos0 = _atom_shared_add(ctr, _popc(bal))
        pos0 = _shfl_idx32(pos0, lead)
    if sel != 0:
        _stg32(orow + Int64(pos0 + _popc(bal & _lanemask_lt())) * 4, token)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


class KeyShardLadderKernel:
    """Sorts each row's local list descending, keeps it and sends the ladder (see the module docstring)."""

    def __init__(self, world: int, local_k: int, mcast: bool):
        self.W, self.L, self.mcast = world, local_k, mcast
        self.threads = local_k // 2
        self.Q = _ladder_step(world)
        # (k, j, shared buffer) of every bitonic stage over pairs held by one thread; a stage with j >= 64 exchanges
        # through shared memory (two buffers, alternating).
        self.stages = []
        nsm = 0
        kk = 2
        while kk <= local_k:
            jj = kk // 2
            while jj >= 1:
                self.stages.append((kk, jj, nsm % 2))
                if jj >= 64:
                    nsm += 1
                jj //= 2
            kk *= 2

        @cute.struct
        class Smem:
            xa: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, local_k], 16]
            xb: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, local_k], 16]
            lad: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, P], 16]

        self.smem = Smem

    @cute.jit
    def __call__(
        self,
        rows: Int32,
        topk: Int64,
        scores: Int64,
        score_stride: Int32,
        sorted_out: Int64,
        kv_lens: Int64,
        next_n: Int32,
        ws_mc: Int64,
        peers: Int64,
        lad_off: Int64,
        state: Int64,
        rows_max: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            topk,
            scores,
            score_stride,
            sorted_out,
            kv_lens,
            next_n,
            ws_mc,
            peers,
            lad_off,
            state,
            rows_max,
            rank,
        ).launch(
            grid=(rows, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem.size_in_bytes(),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        topk: Int64,
        scores: Int64,
        score_stride: Int32,
        sorted_out: Int64,
        kv_lens: Int64,
        next_n: Int32,
        ws_mc: Int64,
        peers: Int64,
        lad_off: Int64,
        state: Int64,
        rows_max: Int32,
        rank: Int32,
    ):
        W, L = self.W, self.L
        t, _, _ = cute.arch.thread_idx()
        r, _, _ = cute.arch.block_idx()
        st = utils.SmemAllocator().allocate(self.smem)
        xa = st.xa.data_ptr().toint()
        xb = st.xb.data_ptr().toint()
        lad = st.lad.data_ptr().toint()
        _lam.griddepcontrol_wait()
        buf = (_ldv32(state) + 1) & 1
        n_row = _row_len(kv_lens, next_n, r)
        _lam.griddepcontrol_launch_dependents()
        if n_row > TOPK:
            nl = _local_count(n_row, rank, W)
            if nl > L:
                nl = Int32(L)
            i0, i1 = _ldg_v2(topk + (Int64(r) * L + Int64(2 * t)) * 4)
            k0 = Int64(_MIN64)
            k1 = Int64(_MIN64)
            srow = scores + Int64(r) * Int64(score_stride) * 4
            if i0 >= 0:
                k0 = _mkkey(_ldg32(srow + Int64(i0) * 4), _token(i0, rank, W))
            if i1 >= 0:
                k1 = _mkkey(_ldg32(srow + Int64(i1) * 4), _token(i1, rank, W))
            # Bitonic sort, descending; thread t holds positions 2t and 2t + 1.
            for si in cutlass.range_constexpr(len(self.stages)):
                kk, jj, sbuf = self.stages[si]
                desc = Int32(((2 * t) & kk) == 0)
                if cutlass.const_expr(jj == 1):
                    hi = _max64(k0, k1)
                    lo = _min64(k0, k1)
                    k0 = _sel64(desc, hi, lo)
                    k1 = _sel64(desc, lo, hi)
                else:
                    keep_max = Int32(Int32((t & (jj // 2)) == 0) == desc)
                    o0 = Int64(0)
                    o1 = Int64(0)
                    if cutlass.const_expr(jj <= 32):
                        o0 = _shfl_xor64(k0, Int32(jj // 2))
                        o1 = _shfl_xor64(k1, Int32(jj // 2))
                    else:
                        sbase = xa if sbuf == 0 else xb
                        _sts64(sbase + (2 * t) * 8, k0)
                        _sts64(sbase + (2 * t + 1) * 8, k1)
                        cute.arch.barrier()
                        o0 = _lds64(sbase + ((2 * t) ^ jj) * 8)
                        o1 = _lds64(sbase + ((2 * t + 1) ^ jj) * 8)
                    k0 = _sel64(keep_max, _max64(k0, o0), _min64(k0, o0))
                    k1 = _sel64(keep_max, _max64(k1, o1), _min64(k1, o1))
            a0, b0 = _split(k0)
            a1, b1 = _split(k1)
            _lam.st_v4(sorted_out + (Int64(r) * L + Int64(2 * t)) * 8, a0, b0, a1, b1)
            if t < P:
                _sts64(lad + t * 8, Int64(_ABSENT))
            cute.arch.barrier()
            for j in cutlass.range_constexpr(P - 1):
                pos = min(L, (j + 1) * self.Q) - 1
                if t == pos // 2:
                    if pos < nl:
                        _sts64(lad + j * 8, k1 if pos % 2 else k0)
            last = nl - 1
            if t == (last >> 1):
                _sts64(lad + (P - 1) * 8, _sel64(last & 1, k1, k0))
            cute.arch.barrier()
            if t < P // 2:
                c0, d0 = _split(_lds64(lad + (2 * t) * 8))
                c1, d1 = _split(_lds64(lad + (2 * t + 1) * 8))
                off = (
                    lad_off
                    + (
                        (Int64(buf) * Int64(rows_max) + Int64(r)) * W * P
                        + Int64(rank) * P
                        + Int64(2 * t)
                    )
                    * 8
                )
                _send_v4(self.mcast, W, ws_mc, peers, off, c0, d0, c1, d1)


class KeyShardPushKernel:
    """Waits for every rank's ladder, takes the bound T and sends this rank's keys >= T and its header."""

    def __init__(self, world: int, local_k: int, mcast: bool):
        self.W, self.L, self.mcast = world, local_k, mcast
        self.threads = max(256, local_k // 2, min(1024, 2 * world * P))
        self.Q = _ladder_step(world)
        # Threads per candidate bound (each counts the ladders of world / tpc ranks).
        self.tpc = max(1, min(self.threads // (world * P), world))

        @cute.struct
        class Smem:
            lad: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, world * P], 16]
            wbest: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 32], 16]
            nl: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, world], 16]
            wcnt: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 32], 16]

        self.smem = Smem

    @cute.jit
    def __call__(
        self,
        rows: Int32,
        sorted_in: Int64,
        kv_lens: Int64,
        next_n: Int32,
        ws_local: Int64,
        ws_mc: Int64,
        peers: Int64,
        lad_off: Int64,
        hdr_off: Int64,
        cand_off: Int64,
        state: Int64,
        rows_max: Int32,
        rank: Int32,
        max_spin: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            sorted_in,
            kv_lens,
            next_n,
            ws_local,
            ws_mc,
            peers,
            lad_off,
            hdr_off,
            cand_off,
            state,
            rows_max,
            rank,
            max_spin,
        ).launch(
            grid=(rows, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem.size_in_bytes(),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        sorted_in: Int64,
        kv_lens: Int64,
        next_n: Int32,
        ws_local: Int64,
        ws_mc: Int64,
        peers: Int64,
        lad_off: Int64,
        hdr_off: Int64,
        cand_off: Int64,
        state: Int64,
        rows_max: Int32,
        rank: Int32,
        max_spin: Int32,
    ):
        W, L, Q, tpc = self.W, self.L, self.Q, self.tpc
        nthr = self.threads
        t, _, _ = cute.arch.thread_idx()
        r, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        st = utils.SmemAllocator().allocate(self.smem)
        lad = st.lad.data_ptr().toint()
        wbest = st.wbest.data_ptr().toint()
        nls = st.nl.data_ptr().toint()
        wcnt = st.wcnt.data_ptr().toint()
        _lam.griddepcontrol_wait()
        buf = (_ldv32(state) + 1) & 1
        n_row = _row_len(kv_lens, next_n, r)
        _lam.griddepcontrol_launch_dependents()
        if n_row > TOPK:
            rb = Int64(buf) * Int64(rows_max) + Int64(r)
            row_lad = ws_local + lad_off + rb * W * P * 8
            for q in cutlass.range(t, W * P // 2, nthr, unroll=1):
                w4 = _poll_v4(row_lad + Int64(q) * 16, Int32(1), max_spin, state)
                _sts64(lad + (2 * q) * 8, _join(w4[0], w4[1]))
                _sts64(lad + (2 * q + 1) * 8, _join(w4[2], w4[3]))
            if t < W:
                c = _local_count(n_row, t, W)
                if c > L:
                    c = Int32(L)
                _sts32(nls + t * 4, c)
            cute.arch.barrier()
            # Candidate bound: ladder entry t // tpc, if it is a real entry with sum_w count_w >= K.
            cand = t // tpc
            part = t - cand * tpc
            live = Int32(cand < W * P)
            if cand >= W * P:
                cand = Int32(W * P - 1)
            wc = cand // P
            jc = cand - wc * P
            nlc = _lds32(nls + wc * 4)
            tjc = (jc + 1) * Q
            if tjc > L:
                tjc = Int32(L)
            valid = Int32(tjc <= nlc)
            if jc == P - 1:
                valid = Int32(nlc >= 1)
            v = _lds64(lad + cand * 8)
            lc = Int32(0)
            for ww in cutlass.range_constexpr(W // tpc):
                w = part + ww * tpc
                nlw = _lds32(nls + w * 4)
                cnt = Int32(0)
                for j in cutlass.range_constexpr(P):
                    x = _lds64(lad + (w * P + j) * 8)
                    if x >= v:
                        if cutlass.const_expr(j < P - 1):
                            if min(L, (j + 1) * Q) <= nlw:
                                cnt = Int32(min(L, (j + 1) * Q))
                        else:
                            cnt = nlw
                lc = lc + cnt
            for o in cutlass.range_constexpr(5):
                if cutlass.const_expr((1 << o) < tpc):
                    lc = lc + _shfl_xor32(lc, Int32(1 << o))
            best = Int64(_MIN64)
            if valid * live != 0:
                if lc >= TOPK:
                    best = v
            for o in cutlass.range_constexpr(5):
                best = _max64(best, _shfl_xor64(best, Int32(16 >> o)))
            if lane == 0:
                _sts64(wbest + warp * 8, best)
            cute.arch.barrier()
            bound = _lds64(wbest)
            for w in cutlass.range_constexpr(1, nthr // 32):
                bound = _max64(bound, _lds64(wbest + w * 8))
            # This rank's keys >= T: a prefix of its sorted list.
            s0 = Int32(0)
            s1 = Int32(0)
            row_sorted = sorted_in + Int64(r) * L * 8
            if 2 * t < L:
                k0 = _ldg64(row_sorted + Int64(2 * t) * 8)
                k1 = _ldg64(row_sorted + Int64(2 * t + 1) * 8)
                s0 = Int32(k0 >= bound) * Int32(k0 > Int64(_MIN64))
                s1 = Int32(k1 >= bound) * Int32(k1 > Int64(_MIN64))
                if s0 != 0:
                    a0, b0 = _split(k0)
                    a1, b1 = _split(k1)
                    if s1 == 0:
                        a1 = Int32(SENTINEL)
                        b1 = Int32(SENTINEL)
                    off = cand_off + ((rb * W + Int64(rank)) * L + Int64(2 * t)) * 8
                    _send_v4(self.mcast, W, ws_mc, peers, off, a0, b0, a1, b1)
            sent = _popc(_ballot(s0)) + _popc(_ballot(s1))
            if lane == 0:
                _sts32(wcnt + warp * 4, sent)
            cute.arch.barrier()
            if t == 0:
                total = Int32(0)
                for w in cutlass.range_constexpr(nthr // 32):
                    total = total + _lds32(wcnt + w * 4)
                complete = Int32(2)
                bound_hi = Int32(1)
                min_lo, min_hi = _split(_ldg64(row_sorted + Int64(L - 1) * 8))
                if _local_count(n_row, rank, W) > L:
                    complete = Int32(1)
                    bound_hi = min_hi
                off = hdr_off + (rb * W + Int64(rank)) * 16
                _send_v4(self.mcast, W, ws_mc, peers, off, total + 1, complete, bound_hi, Int32(1))


class KeyShardMergeKernel:
    """Waits for every rank's header and keys, writes the row's K largest keys' tokens (with the certificate and the
    exact fallback) and re-arms what it read."""

    def __init__(self, world: int, local_k: int):
        self.W, self.L = world, local_k
        self.threads = MERGE_THREADS
        cap = world * local_k

        @cute.struct
        class Smem:
            keys: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, cap], 16]
            hist: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 256], 16]
            hc: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, world], 16]
            hinc: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, world], 16]
            hb: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, world], 16]
            bc: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
            wmin: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 32], 16]
            ctr: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]

        self.smem = Smem

    @cute.jit
    def __call__(
        self,
        rows: Int32,
        out: Int64,
        out_stride: Int32,
        kv_lens: Int64,
        next_n: Int32,
        ws_local: Int64,
        peers: Int64,
        lad_off: Int64,
        hdr_off: Int64,
        cand_off: Int64,
        scores_off: Int64,
        score_stride: Int32,
        state: Int64,
        rows_max: Int32,
        max_spin: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            rows,
            out,
            out_stride,
            kv_lens,
            next_n,
            ws_local,
            peers,
            lad_off,
            hdr_off,
            cand_off,
            scores_off,
            score_stride,
            state,
            rows_max,
            max_spin,
        ).launch(
            grid=(rows, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem.size_in_bytes(),
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def _select(self, n_keys: Int32, keys: Int32, hist: Int32, bc: Int32, warp, lane, t):
        """Threshold key of the K largest of the ``n_keys`` keys in shared memory (MSB radix select)."""
        nthr = self.threads
        remaining = Int32(TOPK)
        prefix = Int64(0)
        done = Int32(0)
        for ps in cutlass.range_constexpr(8):
            if done == 0:
                sh = 56 - 8 * ps
                if t < 256:
                    _sts32(hist + t * 4, Int32(0))
                cute.arch.barrier()
                for i in cutlass.range(t, n_keys, nthr, unroll=1):
                    u = _lds64(keys + i * 8) ^ Int64(_MIN64)
                    if cutlass.const_expr(ps == 0):
                        _red_shared_add(hist + _byte(u, Int32(sh)) * 4, Int32(1))
                    else:
                        if _shr64(u, Int32(sh + 8)) == _shr64(prefix, Int32(sh + 8)):
                            _red_shared_add(hist + _byte(u, Int32(sh)) * 4, Int32(1))
                cute.arch.barrier()
                if warp == 0:
                    bstar, above, cnt = _scan_bins(hist, lane, remaining)
                    if lane == 0:
                        _sts32(bc, bstar)
                        _sts32(bc + 4, above)
                        _sts32(bc + 8, cnt)
                cute.arch.barrier()
                bstar = _lds32(bc)
                above = _lds32(bc + 4)
                cnt = _lds32(bc + 8)
                remaining = remaining - above
                prefix = prefix | _shl64(Int64(bstar), Int32(sh))
                if cnt == remaining:
                    done = Int32(1)
        return prefix ^ Int64(_MIN64)

    @cute.kernel
    def kernel(
        self,
        rows: Int32,
        out: Int64,
        out_stride: Int32,
        kv_lens: Int64,
        next_n: Int32,
        ws_local: Int64,
        peers: Int64,
        lad_off: Int64,
        hdr_off: Int64,
        cand_off: Int64,
        scores_off: Int64,
        score_stride: Int32,
        state: Int64,
        rows_max: Int32,
        max_spin: Int32,
    ):
        W, L = self.W, self.L
        nthr = self.threads
        t, _, _ = cute.arch.thread_idx()
        r, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        st = utils.SmemAllocator().allocate(self.smem)
        keys = st.keys.data_ptr().toint()
        hist = st.hist.data_ptr().toint()
        hcs = st.hc.data_ptr().toint()
        hincs = st.hinc.data_ptr().toint()
        hbs = st.hb.data_ptr().toint()
        bc = st.bc.data_ptr().toint()
        wmin = st.wmin.data_ptr().toint()
        ctr = st.ctr.data_ptr().toint()
        # The push launch of this rank has completed: its ladder reads are done before this CTA re-arms them.
        _lam.griddepcontrol_wait()
        buf = (_ldv32(state) + 1) & 1
        n_row = _row_len(kv_lens, next_n, r)
        _lam.griddepcontrol_launch_dependents()
        orow = out + Int64(r) * Int64(out_stride) * 4
        if n_row <= TOPK:
            for i in cutlass.range(t, TOPK, nthr, unroll=1):
                v = Int32(-1)
                if i < n_row:
                    v = i
                _stg32(orow + Int64(i) * 4, v)
        else:
            rb = Int64(buf) * Int64(rows_max) + Int64(r)
            if t < W:
                ha = ws_local + hdr_off + (rb * W + Int64(t)) * 16
                h = _poll_v4(ha, Int32(1), max_spin, state)
                _sts32(hcs + t * 4, h[0] - 1)
                _sts32(hincs + t * 4, h[1])
                _sts32(hbs + t * 4, h[2])
                _lam.st_sentinel_v4(ha)
            if t == 0:
                _sts32(ctr, Int32(0))
            cute.arch.barrier()
            counts = []
            offs = []
            poffs = []
            n_keys = Int32(0)
            n_pairs = Int32(0)
            for w in cutlass.range_constexpr(W):
                c = _lds32(hcs + w * 4)
                counts.append(c)
                offs.append(n_keys)
                poffs.append(n_pairs)
                n_keys = n_keys + c
                n_pairs = n_pairs + (c + 1) // 2
            # Every rank's keys (pairs of slots), re-armed once read.
            for p in cutlass.range(t, n_pairs, nthr, unroll=1):
                src = Int32(0)
                pfirst = Int32(0)
                kfirst = Int32(0)
                cnt = Int32(0)
                for w in cutlass.range_constexpr(W):
                    if p >= poffs[w]:
                        src = Int32(w)
                        pfirst = poffs[w]
                        kfirst = offs[w]
                        cnt = counts[w]
                m = p - pfirst
                second = Int32(2 * m + 1 < cnt)
                a = ws_local + cand_off + ((rb * W + Int64(src)) * L + Int64(2 * m)) * 8
                v4 = _poll_v4(a, second, max_spin, state)
                _sts64(keys + (kfirst + 2 * m) * 8, _join(v4[0], v4[1]))
                if second != 0:
                    _sts64(keys + (kfirst + 2 * m + 1) * 8, _join(v4[2], v4[3]))
                _lam.st_sentinel_v4(a)
            row_lad = ws_local + lad_off + rb * W * P * 8
            for q in cutlass.range(t, W * P // 2, nthr, unroll=1):
                _lam.st_sentinel_v4(row_lad + Int64(q) * 16)
            cute.arch.barrier()
            thr = self._select(n_keys, keys, hist, bc, warp, lane, t)
            # Smallest selected key: the certificate compares its score with the incomplete lists' smallest scores.
            kmin = Int64(_MAX64)
            for i in cutlass.range(t, n_keys, nthr, unroll=1):
                k = _lds64(keys + i * 8)
                if k >= thr:
                    kmin = _min64(kmin, k)
            for o in cutlass.range_constexpr(5):
                kmin = _min64(kmin, _shfl_xor64(kmin, Int32(16 >> o)))
            if lane == 0:
                _sts64(wmin + warp * 8, kmin)
            cute.arch.barrier()
            kmin = _lds64(wmin)
            for w in cutlass.range_constexpr(1, nthr // 32):
                kmin = _min64(kmin, _lds64(wmin + w * 8))
            kmin_lo, kmin_hi = _split(kmin)
            fail = Int32(0)
            for w in cutlass.range_constexpr(W):
                if _lds32(hincs + w * 4) == 1:
                    if _lds32(hbs + w * 4) >= kmin_hi:
                        fail = Int32(1)
            if fail == 0:
                for base in cutlass.range(0, n_keys, nthr, unroll=1):
                    i = base + t
                    k = Int64(_MIN64)
                    if i < n_keys:
                        k = _lds64(keys + i * 8)
                    lo, hi = _split(k)
                    _emit_warp(Int32(k >= thr), k, lo ^ Int32(-1), lane, ctr, orow)
            else:
                if t == 0:
                    _lam.atom_add_relaxed_u32(state + Int64(8), Int32(1))
                self._fallback(
                    n_row, r, orow, peers, scores_off, score_stride, hist, bc, ctr, warp, lane, t
                )
            cute.arch.barrier()
            n_out = _lds32(ctr)
            for i in cutlass.range(t, TOPK, nthr, unroll=1):
                if i >= n_out:
                    _stg32(orow + Int64(i) * 4, Int32(-1))
        # The last CTA advances the call counter (every CTA of the launch has read it).
        if t == 0:
            old = _lam.atom_add_acq_rel_u32(state + Int64(4), Int32(1))
            if old == rows - 1:
                _lam.st_u32(state + Int64(4), Int32(0))
                _lam.red_release_add(state, Int32(1))

    @cute.jit
    def _fallback(
        self, n_row, r, orow, peers, scores_off, score_stride, hist, bc, ctr, warp, lane, t
    ):
        """Exact Top-K of the row from every rank's local scores, read through the peer mappings."""
        W = self.W
        nthr = self.threads
        remaining = Int32(TOPK)
        prefix = Int64(0)
        done = Int32(0)
        for ps in cutlass.range_constexpr(8):
            if done == 0:
                sh = 56 - 8 * ps
                if t < 256:
                    _sts32(hist + t * 4, Int32(0))
                cute.arch.barrier()
                for w in cutlass.range_constexpr(W):
                    nw = _local_count(n_row, w, W)
                    srow = (
                        _lam.ld_nc_u64(peers + Int64(w * 8))
                        + scores_off
                        + Int64(r) * Int64(score_stride) * 4
                    )
                    for i in cutlass.range(t, nw, nthr, unroll=1):
                        u = _mkkey(_ldv32(srow + Int64(i) * 4), _token(i, w, W)) ^ Int64(_MIN64)
                        if cutlass.const_expr(ps == 0):
                            _red_shared_add(hist + _byte(u, Int32(sh)) * 4, Int32(1))
                        else:
                            if _shr64(u, Int32(sh + 8)) == _shr64(prefix, Int32(sh + 8)):
                                _red_shared_add(hist + _byte(u, Int32(sh)) * 4, Int32(1))
                cute.arch.barrier()
                if warp == 0:
                    bstar, above, cnt = _scan_bins(hist, lane, remaining)
                    if lane == 0:
                        _sts32(bc, bstar)
                        _sts32(bc + 4, above)
                        _sts32(bc + 8, cnt)
                cute.arch.barrier()
                bstar = _lds32(bc)
                above = _lds32(bc + 4)
                cnt = _lds32(bc + 8)
                remaining = remaining - above
                prefix = prefix | _shl64(Int64(bstar), Int32(sh))
                if cnt == remaining:
                    done = Int32(1)
        thr = prefix ^ Int64(_MIN64)
        for w in cutlass.range_constexpr(W):
            nw = _local_count(n_row, w, W)
            srow = (
                _lam.ld_nc_u64(peers + Int64(w * 8))
                + scores_off
                + Int64(r) * Int64(score_stride) * 4
            )
            for base in cutlass.range(0, nw, nthr, unroll=1):
                i = base + t
                k = Int64(_MIN64)
                if i < nw:
                    k = _mkkey(_ldv32(srow + Int64(i) * 4), _token(i, w, W))
                _emit_warp(Int32(k >= thr), k, _token(i, w, W), lane, ctr, orow)


def _ladder_step(world: int) -> int:
    """Ladder rank step K / (4 W)."""
    return TOPK // (4 * world)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@functools.cache
def _ladder_launch(world: int, local_k: int, mcast: bool):
    a = Int64(1 << 20)
    return cute.compile(
        KeyShardLadderKernel(world, local_k, mcast),
        Int32(1),
        a,
        a,
        Int32(256),
        a,
        a,
        Int32(5),
        a,
        a,
        Int64(0),
        a,
        Int32(1),
        Int32(0),
        _stream(),
    )


@functools.cache
def _push_launch(world: int, local_k: int, mcast: bool):
    a = Int64(1 << 20)
    return cute.compile(
        KeyShardPushKernel(world, local_k, mcast),
        Int32(1),
        a,
        a,
        Int32(5),
        a,
        a,
        a,
        Int64(0),
        Int64(0),
        Int64(0),
        a,
        Int32(1),
        Int32(0),
        Int32(1),
        _stream(),
    )


@functools.cache
def _merge_launch(world: int, local_k: int):
    a = Int64(1 << 20)
    return cute.compile(
        KeyShardMergeKernel(world, local_k),
        Int32(1),
        a,
        Int32(TOPK),
        a,
        Int32(5),
        a,
        a,
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int32(256),
        a,
        Int32(1),
        Int32(1),
        _stream(),
    )


class KeyShardPrunedExchange:
    """Workspace and launches of the pruned exchange + merge (``K = 2048``).

    ``buffers`` are this rank's views of every rank's symmetric buffer (rank order); ``mc`` the multicast address of
    the same allocation, or 0 to store to every peer mapping instead (also how plain local tensors emulate ``world``
    ranks on one GPU in tests). Every rank's buffer has the same layout:

      int64 ladder[2][rows][W][P]          int32 x4 header[2][rows][W]       int64 keys[2][rows][W][L]
      fp32  scores[rows][width]            this rank's local logits (``local_logits``; peers read them on fallback)

    The ladder, header and key regions must be all ``0x80000000`` words before the first call (``rearm``).
    """

    def __init__(
        self,
        buffers: List[torch.Tensor],
        rank: int,
        rows_max: int,
        top_k: int,
        local_width: int,
        local_k: int,
        mc: int = 0,
        max_spin: int = 1 << 26,
    ):
        self.world = len(buffers)
        W = self.world
        assert W & (W - 1) == 0 and 4 <= W <= 32, "a power-of-two group of 4 to 32 ranks"
        assert top_k == TOPK, f"the exchange serves Top-{TOPK}"
        assert local_k in (256, 512, 1024) and local_k * 2 >= TOPK // W and local_k <= top_k
        assert W * local_k * 8 <= 128 * 1024, "the merge keeps every received key in shared memory"
        self.rank = rank
        self.rows_max = rows_max
        self.top_k = top_k
        self.local_k = local_k
        self.local_width = -(-local_width // 256) * 256
        self.mc = int(mc)
        self.max_spin = max_spin
        self.lad_off = 0
        self.hdr_off = NUM_SLOTS * rows_max * W * P * 8
        self.cand_off = self.hdr_off + NUM_SLOTS * rows_max * W * 16
        self.scores_off = self.cand_off + NUM_SLOTS * rows_max * W * local_k * 8
        need = self.bytes_needed(rows_max, W, local_k, self.local_width)
        mine = buffers[rank]
        assert mine.numel() * mine.element_size() >= need and mine.data_ptr() % 16 == 0
        # The kernels address the peers' buffers through their base addresses: keep them alive.
        self.buffers = list(buffers)
        dev = mine.device
        self.base = mine.data_ptr()
        self.peers = torch.tensor([b.data_ptr() for b in buffers], dtype=torch.int64, device=dev)
        raw = mine.view(torch.uint8)
        self.lamport = raw[: self.scores_off].view(torch.int32)
        self.local_logits = (
            raw[self.scores_off : self.scores_off + rows_max * self.local_width * 4]
            .view(torch.float32)
            .view(rows_max, self.local_width)
        )
        self.sorted = torch.empty(rows_max, local_k, dtype=torch.int64, device=dev)
        # [call counter, CTA ticket, fallback rows, poll timeouts]
        self.state = torch.zeros(4, dtype=torch.int32, device=dev)

    @staticmethod
    def bytes_needed(rows_max: int, world: int, local_k: int, local_width: int) -> int:
        local_width = -(-local_width // 256) * 256
        return (
            NUM_SLOTS * rows_max * world * (P * 8 + 16 + local_k * 8) + rows_max * local_width * 4
        )

    def rearm(self) -> None:
        """Every exchanged word of this rank's buffer empty (before the group's first call)."""
        self.lamport.fill_(SENTINEL)

    def ladder(self, local_topk: torch.Tensor, kv_lens: torch.Tensor, next_n: int) -> None:
        """Launch 1: sort the rows' local Top-``local_k`` (local column indices of ``local_logits``, -1 padded) and send
        the ladders. ``kv_lens`` [rows / next_n] int32: each sequence's key count (row ``r`` sees ``kv_lens[r //
        next_n] - next_n + 1 + r % next_n`` keys)."""
        rows = local_topk.shape[0]
        assert (
            rows <= self.rows_max
            and local_topk.shape[1] == self.local_k
            and local_topk.is_contiguous()
            and local_topk.dtype == torch.int32
            and kv_lens.dtype == torch.int32
        )
        _ladder_launch(self.world, self.local_k, self.mc != 0)(
            Int32(rows),
            Int64(local_topk.data_ptr()),
            Int64(self.base + self.scores_off),
            Int32(self.local_width),
            Int64(self.sorted.data_ptr()),
            Int64(kv_lens.data_ptr()),
            Int32(next_n),
            Int64(self.mc),
            Int64(self.peers.data_ptr()),
            Int64(self.lad_off),
            Int64(self.state.data_ptr()),
            Int32(self.rows_max),
            Int32(self.rank),
            _stream(),
        )

    def push(self, rows: int, kv_lens: torch.Tensor, next_n: int) -> None:
        """Launch 2: the bound from every rank's ladder; send this rank's keys >= it and the header."""
        _push_launch(self.world, self.local_k, self.mc != 0)(
            Int32(rows),
            Int64(self.sorted.data_ptr()),
            Int64(kv_lens.data_ptr()),
            Int32(next_n),
            Int64(self.base),
            Int64(self.mc),
            Int64(self.peers.data_ptr()),
            Int64(self.lad_off),
            Int64(self.hdr_off),
            Int64(self.cand_off),
            Int64(self.state.data_ptr()),
            Int32(self.rows_max),
            Int32(self.rank),
            Int32(self.max_spin),
            _stream(),
        )

    def merge(self, out: torch.Tensor, kv_lens: torch.Tensor, next_n: int) -> None:
        """Launch 3: every row's global Top-K token indices (any order, -1 padded) into ``out`` [rows, top_k]."""
        rows = out.shape[0]
        assert out.stride(1) == 1 and out.shape[1] == self.top_k and out.dtype == torch.int32
        _merge_launch(self.world, self.local_k)(
            Int32(rows),
            Int64(out.data_ptr()),
            Int32(out.stride(0) if rows > 1 else self.top_k),
            Int64(kv_lens.data_ptr()),
            Int32(next_n),
            Int64(self.base),
            Int64(self.peers.data_ptr()),
            Int64(self.lad_off),
            Int64(self.hdr_off),
            Int64(self.cand_off),
            Int64(self.scores_off),
            Int32(self.local_width),
            Int64(self.state.data_ptr()),
            Int32(self.rows_max),
            Int32(self.max_spin),
            _stream(),
        )

    def exchange(
        self, local_topk: torch.Tensor, kv_lens: torch.Tensor, next_n: int, out: torch.Tensor
    ) -> None:
        """The three launches of one call (a collective: every rank of the group makes the same calls)."""
        rows = out.shape[0]
        if rows == 0:
            return
        self.ladder(local_topk, kv_lens, next_n)
        self.push(rows, kv_lens, next_n)
        self.merge(out, kv_lens, next_n)


_PRUNED_EXCHANGES: dict = {}


def get_key_shard_pruned_exchange(
    mapping, rows_max: int, top_k: int, local_width: int, local_k: int
) -> KeyShardPrunedExchange:
    """The TP group's pruned-exchange workspace (collective on first use; call outside CUDA graph capture)."""
    key = (mapping.tp_size, mapping.tp_rank, mapping.pp_rank, rows_max, top_k, local_width, local_k)
    ex = _PRUNED_EXCHANGES.get(key)
    if ex is not None:
        return ex
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_workspace_all_succeeded,
    )

    granularity = 1 << 21
    size = KeyShardPrunedExchange.bytes_needed(rows_max, mapping.tp_size, local_k, local_width)
    size = -(-size // granularity) * granularity
    comm = _get_mnnvl_workspace_comm(mapping)
    error = None
    try:
        use_fabric_handle = (
            os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
        )
        handle = _make_mnnvl_mcast_buffer(comm, size, mapping, use_fabric_handle)
        buffers = [handle.get_uc_buffer(r, (size,), torch.uint8, 0) for r in range(mapping.tp_size)]
        mc = handle.get_mc_buffer((size,), torch.uint8, 0).data_ptr()
        ex = KeyShardPrunedExchange(
            buffers, mapping.tp_rank, rows_max, top_k, local_width, local_k, mc=mc
        )
        ex.rearm()
        torch.cuda.synchronize()
    except (RuntimeError, ValueError) as exc:
        error = exc
    # Every rank has emptied its buffer before any peer sends into it.
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        raise RuntimeError("key-shard pruned exchange workspace setup failed on a peer") from error
    ex.handle = handle
    ex.comm = comm
    _PRUNED_EXCHANGES[key] = ex
    return ex


__all__ = [
    "KeyShardPrunedExchange",
    "NUM_SLOTS",
    "P",
    "SENTINEL",
    "TOKENS_PER_BLOCK",
    "get_key_shard_pruned_exchange",
]

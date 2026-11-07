# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL top-16 block selection for MiniMax-M3 MSA context rows (Blackwell).

Bitwise equal to ``torch.ops.trtllm.minimax_m3_select_blocks(scores, n_valid_blocks, 16,
init_blocks, local_blocks, True)`` and several times faster on context batches, except on rows
holding a negative-NaN score or more than 2048 valid blocks tied at the 16th place, where the C++
radix select fills the tie in atomic order and this kernel takes the smallest block ids. Scores
arrive token-contiguous and the C++ op reads each row on its own, strided by the token count; here
each CTA loads a [blocks x tokens] tile with coalesced loads and transposes it through shared
memory, one warp per row.

Lane l of a row holds blocks 128 j + 4 l + {0..3} as order-preserving int32 keys. The 16th largest
of the 32 lane maxima bounds the row's 16th largest key from below, so only keys at or above it
(about 22 of a random row) are ranked exactly. A row with more than 64 of them (heavy ties) is
resolved by 16 rounds of exact extraction instead.
"""

import struct
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL, griddepcontrol_wait

__all__ = ["minimax_m3_select_context_blocks"]

_TOPK = 16
# Wider rows keep the C++ op: past this, four rows of keys fill most of an SM's shared memory and
# the kernel falls behind it.
_MAX_BLOCKS = 4096
# (row capacity in 128-block slots, rows per CTA). A batch runs on the smallest capacity holding
# G, the scores' block count, and pays for the unused slots, so the edges sit at common widths:
# 8 slots is a 128K-token context, and 9 the widest that keeps two 16-row CTAs per SM (64
# registers per thread). Rows past 16 slots keep their keys in shared memory only.
_BUCKETS = ((2, 8), (4, 8), (6, 16), (8, 16), (9, 16), (12, 8), (16, 8), (32, 4))
_MAX_REGISTER_SLOTS = 16
_MAX_CANDIDATES = 64
_NO_BLOCK = 0x7FFFFFFF  # sorts after every block id; stored as -1


def _f32_key(x: float) -> int:
    """Float -> int32 key whose signed order is the float total order on bit patterns."""
    bits = struct.unpack("<i", struct.pack("<f", x))[0]
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


_KEY_NEG_INF = _f32_key(float("-inf"))
# The C++ op's forced scores for init and local blocks.
_KEY_INIT = _f32_key(1.0e30)
_KEY_LOCAL = _f32_key(1.0e29)


# --------------------------------------------------------------------------------------------
# Inline PTX
# --------------------------------------------------------------------------------------------
def _asm(outs, ins, code, constraints, side_effects=True, loc=None, ip=None):
    """Inline PTX over int32 ('r') and int64 ('l') operands; returns None, a value or a list."""
    tys = {"r": T.i32, "l": T.i64}
    dsl = {"r": Int32, "l": Int64}
    args = [dsl[k](v).ir_value(loc=loc, ip=ip) for v, k in ins]
    kw = dict(
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if not outs:
        llvm.inline_asm(None, args, code, constraints, **kw)
        return None
    if len(outs) == 1:
        return dsl[outs](llvm.inline_asm(tys[outs](), args, code, constraints, **kw))
    res = llvm.inline_asm(
        llvm.StructType.get_literal([tys[k]() for k in outs]), args, code, constraints, **kw
    )
    return [
        dsl[k](llvm.extractvalue(tys[k](), res, [i], loc=loc, ip=ip)) for i, k in enumerate(outs)
    ]


@dsl_user_op
def _ldg32(addr, *, loc=None, ip=None):
    return _asm("r", [(addr, "l")], "ld.global.nc.b32 $0, [$1];", "=r,l", loc=loc, ip=ip)


@dsl_user_op
def _ldg32_if(addr, pred, *, loc=None, ip=None):
    code = "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.b32 $0, 0; @p ld.global.nc.b32 $0, [$1]; }"
    return _asm("r", [(addr, "l"), (pred, "r")], code, "=r,l,r", loc=loc, ip=ip)


@dsl_user_op
def _ldg128_if(addr, pred, *, loc=None, ip=None):
    """16-byte load if pred != 0, else zeros."""
    code = (
        "{ .reg .pred p; setp.ne.s32 p, $5, 0;"
        " mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;"
        " @p ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4]; }"
    )
    return _asm("rrrr", [(addr, "l"), (pred, "r")], code, "=r,=r,=r,=r,l,r", loc=loc, ip=ip)


@dsl_user_op
def _store_block_id(addr, v, *, loc=None, ip=None):
    code = (
        "{ .reg .pred p; .reg .b32 t; setp.eq.s32 p, $1, 2147483647;"
        " selp.b32 t, -1, $1, p; st.global.b32 [$0], t; }"
    )
    _asm("", [(addr, "l"), (v, "r")], code, "l,r", loc=loc, ip=ip)


@dsl_user_op
def _sts32(addr, v, *, loc=None, ip=None):
    _asm("", [(addr, "r"), (v, "r")], "st.shared.b32 [$0], $1;", "r,r", loc=loc, ip=ip)


@dsl_user_op
def _lds32(addr, *, loc=None, ip=None):
    return _asm("r", [(addr, "r")], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip)


@dsl_user_op
def _lds128(addr, *, loc=None, ip=None):
    code = "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];"
    return _asm("rrrr", [(addr, "r")], code, "=r,=r,=r,=r,r", loc=loc, ip=ip)


@dsl_user_op
def _lds64(addr, *, loc=None, ip=None):
    return _asm("l", [(addr, "r")], "ld.shared.b64 $0, [$1];", "=l,r", loc=loc, ip=ip)


@dsl_user_op
def _lds128_max(addr, acc, *, loc=None, ip=None):
    """max(acc, the four keys at addr)."""
    code = (
        "{ .reg .b32 a, b, c, d; ld.shared.v4.b32 {a, b, c, d}, [$1];"
        " max.s32 $0, a, b; max.s32 $0, $0, c; max.s32 $0, $0, d; max.s32 $0, $0, $2; }"
    )
    return _asm("r", [(addr, "r"), (acc, "r")], code, "=r,r,r", loc=loc, ip=ip)


@dsl_user_op
def _imax(a, b, *, loc=None, ip=None):
    code = "max.s32 $0, $1, $2;"
    return _asm("r", [(a, "r"), (b, "r")], code, "=r,r,r", side_effects=False, loc=loc, ip=ip)


@dsl_user_op
def _imin(a, b, *, loc=None, ip=None):
    code = "min.s32 $0, $1, $2;"
    return _asm("r", [(a, "r"), (b, "r")], code, "=r,r,r", side_effects=False, loc=loc, ip=ip)


@dsl_user_op
def _max64(a, b, *, loc=None, ip=None):
    code = "max.s64 $0, $1, $2;"
    return _asm("l", [(a, "l"), (b, "l")], code, "=l,l,l", side_effects=False, loc=loc, ip=ip)


@dsl_user_op
def _redux_min(a, *, loc=None, ip=None):
    code = "redux.sync.min.s32 $0, $1, 0xffffffff;"
    return _asm("r", [(a, "r")], code, "=r,r", loc=loc, ip=ip)


@dsl_user_op
def _lt_and_lt(a, b, c, d, *, loc=None, ip=None):
    """1 if a < b and c < d, else 0."""
    code = (
        "{ .reg .pred p; setp.lt.s32 p, $1, $2; setp.lt.and.s32 p, $3, $4, p;"
        " selp.u32 $0, 1, 0, p; }"
    )
    ins = [(a, "r"), (b, "r"), (c, "r"), (d, "r")]
    return _asm("r", ins, code, "=r,r,r,r,r", side_effects=False, loc=loc, ip=ip)


@dsl_user_op
def _ge(a, b, *, loc=None, ip=None):
    code = "{ .reg .pred p; setp.ge.s32 p, $1, $2; selp.u32 $0, 1, 0, p; }"
    return _asm("r", [(a, "r"), (b, "r")], code, "=r,r,r", side_effects=False, loc=loc, ip=ip)


@dsl_user_op
def _key(x, *, loc=None, ip=None):
    """Raw score bits -> order-preserving int32 key."""
    code = "{ .reg .b32 t; shr.s32 t, $1, 31; and.b32 t, t, 2147483647; xor.b32 $0, $1, t; }"
    return _asm("r", [(x, "r")], code, "=r,r", side_effects=False, loc=loc, ip=ip)


@dsl_user_op
def _forced_key(x, b, local_start, valid, init, *, loc=None, ip=None):
    """Key of block b: local forcing over init forcing over the raw score; -inf if b >= valid."""
    code = (
        "{ .reg .b32 t; .reg .pred p;"
        " shr.s32 t, $1, 31; and.b32 t, t, 2147483647; xor.b32 t, t, $1;"
        f" setp.lt.s32 p, $2, $5; selp.b32 t, {_KEY_INIT}, t, p;"
        f" setp.ge.s32 p, $2, $3; selp.b32 t, {_KEY_LOCAL}, t, p;"
        f" setp.ge.s32 p, $2, $4; selp.b32 $0, {_KEY_NEG_INF}, t, p; }}"
    )
    ins = [(x, "r"), (b, "r"), (local_start, "r"), (valid, "r"), (init, "r")]
    return _asm("r", ins, code, "=r,r,r,r,r,r", side_effects=False, loc=loc, ip=ip)


@dsl_user_op
def _compare_exchange(x, p, take_min, *, loc=None, ip=None):
    code = (
        "{ .reg .pred q; .reg .b32 mn, mx; setp.ne.s32 q, $3, 0;"
        " min.s32 mn, $1, $2; max.s32 mx, $1, $2; selp.b32 $0, mn, mx, q; }"
    )
    ins = [(x, "r"), (p, "r"), (take_min, "r")]
    return _asm("r", ins, code, "=r,r,r,r", side_effects=False, loc=loc, ip=ip)


@dsl_user_op
def _set_bit_if_ge(mask, key, tau, bit: int, *, loc=None, ip=None):
    code = (
        "{ .reg .pred p; setp.ge.s32 p, $2, $3; mov.b32 $0, $1;"
        f" @p or.b32 $0, $1, {(1 << bit):#x}; }}"
    )
    ins = [(mask, "r"), (key, "r"), (tau, "r")]
    return _asm("r", ins, code, "=r,r,r,r", side_effects=False, loc=loc, ip=ip)


@dsl_user_op
def _popc(x, *, loc=None, ip=None):
    return _asm("r", [(x, "r")], "popc.b32 $0, $1;", "=r,r", side_effects=False, loc=loc, ip=ip)


@dsl_user_op
def _pop_lowest_bit(m, *, loc=None, ip=None):
    """(index of the lowest set bit of m, m with that bit cleared)."""
    code = (
        "{ .reg .b32 t; brev.b32 t, $2; bfind.shiftamt.u32 $0, t;"
        " add.s32 t, $2, -1; and.b32 $1, $2, t; }"
    )
    return _asm("rr", [(m, "r")], code, "=r,=r,r", side_effects=False, loc=loc, ip=ip)


@dsl_user_op
def _add_if_lane_ge(v, p, lane, offset: int, *, loc=None, ip=None):
    code = (
        f"{{ .reg .pred q; setp.ge.s32 q, $3, {offset}; mov.b32 $0, $1; @q add.s32 $0, $1, $2; }}"
    )
    ins = [(v, "r"), (p, "r"), (lane, "r")]
    return _asm("r", ins, code, "=r,r,r,r", side_effects=False, loc=loc, ip=ip)


@dsl_user_op
def _ballot_rank(pred, lanes_below, *, loc=None, ip=None):
    """(set lanes below this one, set lanes in the warp) of an int predicate."""
    code = (
        "{ .reg .pred p; .reg .b32 b, t; setp.ne.s32 p, $2, 0; vote.sync.ballot.b32 b, p, 0xffffffff;"
        " and.b32 t, b, $3; popc.b32 $0, t; popc.b32 $1, b; }"
    )
    return _asm("rr", [(pred, "r"), (lanes_below, "r")], code, "=r,=r,r,r", loc=loc, ip=ip)


@dsl_user_op
def _store_candidate_of_element(row, swz, lane4, e, e_offset: int, cand, pos, *, loc=None, ip=None):
    """cand[pos] = (key ^ 0x80000000) << 32 | (65535 - block) of this lane's element e + e_offset."""
    code = (
        "{ .reg .b32 ee, j, i, col, a, k, b, o;"
        f" add.s32 ee, $3, {e_offset}; shr.u32 j, ee, 2; and.b32 i, ee, 3;"
        " shl.b32 col, j, 7; add.s32 col, col, $2;"
        " xor.b32 a, col, $1; add.s32 a, a, i; shl.b32 a, a, 2; add.s32 a, a, $0;"
        " ld.shared.b32 k, [a]; xor.b32 k, k, 0x80000000; add.s32 b, col, i; sub.s32 b, 65535, b;"
        " shl.b32 o, $5, 3; add.s32 o, o, $4;"
        " st.shared.v2.b32 [o], {b, k}; }"
    )
    ins = [(row, "r"), (swz, "r"), (lane4, "r"), (e, "r"), (cand, "r"), (pos, "r")]
    _asm("", ins, code, "r,r,r,r,r,r", loc=loc, ip=ip)


@dsl_user_op
def _store_candidate(addr, key, block, *, loc=None, ip=None):
    code = (
        "{ .reg .b32 k, b; xor.b32 k, $1, 0x80000000; sub.s32 b, 65535, $2;"
        " st.shared.v2.b32 [$0], {b, k}; }"
    )
    _asm("", [(addr, "r"), (key, "r"), (block, "r")], code, "r,r,r", loc=loc, ip=ip)


@dsl_user_op
def _count_greater(addr, my, r, *, loc=None, ip=None):
    """r + 1 if the candidate at addr outranks my, else r."""
    code = (
        "{ .reg .b64 k; .reg .pred p; ld.shared.b64 k, [$1]; setp.gt.u64 p, k, $2;"
        " mov.b32 $0, $3; @p add.s32 $0, $3, 1; }"
    )
    return _asm("r", [(addr, "r"), (my, "l"), (r, "r")], code, "=r,r,l,r", loc=loc, ip=ip)


@dsl_user_op
def _count_greater_x2(addr, my1, r1, my2, r2, *, loc=None, ip=None):
    code = (
        "{ .reg .b64 k; .reg .pred p, q; ld.shared.b64 k, [$2];"
        " setp.gt.u64 p, k, $3; setp.gt.u64 q, k, $5;"
        " mov.b32 $0, $4; @p add.s32 $0, $4, 1; mov.b32 $1, $6; @q add.s32 $1, $6, 1; }"
    )
    ins = [(addr, "r"), (my1, "l"), (r1, "r"), (my2, "l"), (r2, "r")]
    return _asm("rr", ins, code, "=r,=r,r,l,r,l,r", loc=loc, ip=ip)


@dsl_user_op
def _block_of_candidate(k, *, loc=None, ip=None):
    code = "{ .reg .b32 t; cvt.u32.u64 t, $1; and.b32 t, t, 65535; sub.s32 $0, 65535, t; }"
    return _asm("r", [(k, "l")], code, "=r,l", side_effects=False, loc=loc, ip=ip)


def _extract_code() -> str:
    lines = [
        "{ .reg .b32 a0, a1, a2, a3, s, bb, t; .reg .b64 K, x, y, bst; .reg .pred el, p;",
        " ld.shared.v4.b32 {a0, a1, a2, a3}, [$1];",
        " mov.b64 bst, $5;",
    ]
    for i in range(4):
        lines += [
            f" mov.b32 s, a{i}; add.u32 bb, $2, {i};",
            f" setp.lt.s32 el, bb, $3; setp.gt.and.s32 el, s, {_KEY_NEG_INF}, el;",
            " cvt.s64.s32 x, s; add.s64 x, x, 2147483648; shl.b64 x, x, 16;",
            " sub.u32 t, 65535, bb; cvt.u64.u32 y, t; or.b64 K, x, y;",
            " setp.lt.and.s64 p, K, $4, el; @p max.s64 bst, bst, K;",
        ]
    lines.append(" mov.b64 $0, bst; }")
    return "\n".join(lines)


_EXTRACT_CODE = _extract_code()


@dsl_user_op
def _extract_step(addr, b0, valid, prev, best, *, loc=None, ip=None):
    """max(best, packed (key, 65535 - block) < prev) over the four selectable keys at addr."""
    ins = [(addr, "r"), (b0, "r"), (valid, "r"), (prev, "l"), (best, "l")]
    return _asm("l", ins, _EXTRACT_CODE, "=l,r,r,r,l,l", loc=loc, ip=ip)


@dsl_user_op
def _block_of_extracted(k, *, loc=None, ip=None):
    """Block id of an extracted packed key, or _NO_BLOCK once the row ran out (k < 0)."""
    code = (
        "{ .reg .pred p; .reg .b32 t; setp.ge.s64 p, $1, 0; cvt.u32.u64 t, $1;"
        " and.b32 t, t, 65535; sub.u32 t, 65535, t; selp.b32 $0, t, 2147483647, p; }"
    )
    return _asm("r", [(k, "l")], code, "=r,l", side_effects=False, loc=loc, ip=ip)


# --------------------------------------------------------------------------------------------
# Warp building blocks
# --------------------------------------------------------------------------------------------
@cute.jit
def _bit(lane, k: cutlass.Constexpr[int]):
    return (lane >> k) & 1


@cute.jit
def _sort_ascending(x, lane):
    """Bitonic sort over groups of 16 lanes; sorts lanes 0-15 ascending."""
    for sl in cutlass.range_constexpr(1, 5):
        for st in cutlass.range_constexpr(sl - 1, -1, -1):
            p = cute.arch.shuffle_sync_bfly(x, offset=1 << st)
            low = _bit(lane, st)
            x = _compare_exchange(x, p, Int32(1) - (_bit(lane, sl) ^ low))
    return x


@cute.jit
def _sixteenth_largest(x, lane):
    """16th largest of the 32 lane values: sort the halves, half-clean, take the upper minimum."""
    y = _sort_ascending(x, lane)
    hi = _imax(y, cute.arch.shuffle_sync_bfly(y, offset=16))
    return _redux_min(hi)


@cute.jit
def _warp_max64(x):
    for k in cutlass.range_constexpr(5):
        x = _max64(x, cute.arch.shuffle_sync_bfly(x, offset=16 >> k))
    return x


# --------------------------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------------------------
class _SelectContextBlocksKernel:
    """One CTA per (head, rows_per_cta consecutive tokens), one warp per row.

    The CTA's threads load a [128 * num_slots blocks x rows_per_cta tokens] tile, four tokens per
    thread, apply the forcing, and store it transposed into shared memory (XOR-swizzled so both
    directions are conflict free). Each warp then selects its row from there.
    """

    def __init__(self, num_slots: int, rows_per_cta: int, vec: bool):
        self.num_slots = num_slots
        self.rows_per_cta = rows_per_cta
        self.row_words = 128 * num_slots
        self.keys_in_registers = num_slots <= _MAX_REGISTER_SLOTS
        # 16-byte loads; a separate kernel, since a runtime switch slows the scalar loads by ~10%.
        self.vec = vec

    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,  # [H, G, T] float32, read as raw bits
        n_valid_blocks: cute.Tensor,  # [T] int32
        out: cute.Tensor,  # [T, H, 16] int32, unit last stride
        init: Int32,
        local: Int32,
        vec_ok: Int32,
        stream,
    ):
        num_heads = scores.shape[0]
        num_blocks = scores.shape[1]
        num_tokens = scores.shape[2]
        self.kernel(
            scores,
            n_valid_blocks,
            out,
            num_blocks,
            num_tokens,
            Int64(scores.stride[0]),
            Int64(scores.stride[1]),
            Int64(scores.stride[2]),
            Int64(out.stride[0]),
            Int64(out.stride[1]),
            init,
            local,
            vec_ok,
        ).launch(
            grid=((num_tokens + self.rows_per_cta - 1) // self.rows_per_cta, num_heads, 1),
            block=(self.rows_per_cta * 32, 1, 1),
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
        )

    @cute.jit
    def _store_slot(self, k, brow, x4, forcing, dst):
        """Force and store the keys of block k * 128 + brow for the thread's four tokens.

        forcing = (init, min_forced, valid4, local_start4) and dst = (row address, swizzle).
        """
        row_words = self.row_words
        init, min_forced, valid4, local4 = forcing
        rowt, swz = dst
        x0, x1, x2, x3 = x4
        b = k * 128 + brow
        k0 = Int32(0)
        k1 = Int32(0)
        k2 = Int32(0)
        k3 = Int32(0)
        if b >= init and b < min_forced:
            k0 = _key(x0)
            k1 = _key(x1)
            k2 = _key(x2)
            k3 = _key(x3)
        else:
            k0 = _forced_key(x0, b, local4[0], valid4[0], init)
            k1 = _forced_key(x1, b, local4[1], valid4[1], init)
            k2 = _forced_key(x2, b, local4[2], valid4[2], init)
            k3 = _forced_key(x3, b, local4[3], valid4[3], init)
        ea = rowt + (b ^ swz) * 4
        _sts32(ea, k0)
        _sts32(ea + row_words * 4, k1)
        _sts32(ea + row_words * 8, k2)
        _sts32(ea + row_words * 12, k3)

    @cute.jit
    def _load_tile(self, vec: cutlass.Constexpr[bool], src, brow, tok, block_limit, forcing, dst):
        """Load and store every slot, all loads first for memory-level parallelism.

        src = (scores address, head offset, block stride, token stride, token count).
        """
        p_s, hoff, sG, sT, num_tokens = src
        xs = []
        for k in cutlass.range_constexpr(self.num_slots):
            b = brow + k * 128
            off = hoff + Int64(b) * sG
            if cutlass.const_expr(vec):
                pred = _lt_and_lt(b, block_limit, tok, num_tokens)
                xs.append(_ldg128_if(p_s + (off + Int64(tok)) * 4, pred))
            else:
                x4 = []
                for i in cutlass.range_constexpr(4):
                    pred = _lt_and_lt(b, block_limit, tok + i, num_tokens)
                    x4.append(_ldg32_if(p_s + (off + Int64(tok + i) * sT) * 4, pred))
                xs.append(x4)
        for k in cutlass.range_constexpr(self.num_slots):
            self._store_slot(Int32(k), brow, xs[k], forcing, dst)

    @cute.jit
    def _load_store_slot(self, k, full4, src, brow, tok, block_limit, forcing, dst):
        """Load and store slot k of a tile too wide to unroll."""
        p_s, hoff, sG, sT, num_tokens = src
        b = k * 128 + brow
        off = hoff + Int64(b) * sG
        x0 = Int32(0)
        x1 = Int32(0)
        x2 = Int32(0)
        x3 = Int32(0)
        if full4 != 0:
            pred = _lt_and_lt(b, block_limit, tok, num_tokens)
            x0, x1, x2, x3 = _ldg128_if(p_s + (off + Int64(tok)) * 4, pred)
        else:
            p0 = _lt_and_lt(b, block_limit, tok, num_tokens)
            p1 = _lt_and_lt(b, block_limit, tok + 1, num_tokens)
            p2 = _lt_and_lt(b, block_limit, tok + 2, num_tokens)
            p3 = _lt_and_lt(b, block_limit, tok + 3, num_tokens)
            x0 = _ldg32_if(p_s + (off + Int64(tok) * sT) * 4, p0)
            x1 = _ldg32_if(p_s + (off + Int64(tok + 1) * sT) * 4, p1)
            x2 = _ldg32_if(p_s + (off + Int64(tok + 2) * sT) * 4, p2)
            x3 = _ldg32_if(p_s + (off + Int64(tok + 3) * sT) * 4, p3)
        self._store_slot(k, brow, (x0, x1, x2, x3), forcing, dst)

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        n_valid_blocks: cute.Tensor,
        out: cute.Tensor,
        G: Int32,
        num_tokens: Int32,
        sH: Int64,
        sG: Int64,
        sT: Int64,
        out_stride_t: Int64,
        out_stride_h: Int64,
        init: Int32,
        local: Int32,
        vec_ok: Int32,
    ):
        rows_per_cta = self.rows_per_cta
        row_words = self.row_words
        num_slots = self.num_slots
        threads_per_block_row = rows_per_cta // 4
        swizzle_step = 128 // rows_per_cta
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        lane = tid & 31
        w = tid >> 5
        h = Int32(by)
        t0 = Int32(bx) * rows_per_cta
        p_s = Int64(scores.iterator.toint())
        p_nvb = Int64(n_valid_blocks.iterator.toint())
        p_out = Int64(out.iterator.toint())
        tile = Int32(cute.arch.alloc_smem(Int32, rows_per_cta * row_words, alignment=16).toint())
        cand = Int32(
            cute.arch.alloc_smem(Int32, rows_per_cta * _MAX_CANDIDATES * 2, alignment=16).toint()
        )
        selected = Int32(cute.arch.alloc_smem(Int32, rows_per_cta * _TOPK, alignment=16).toint())
        hoff = Int64(h) * sH
        griddepcontrol_wait()

        # ---- cooperative transposing load of the [blocks x rows_per_cta tokens] tile ----
        max_valid = Int32(0)
        min_forced = Int32(_NO_BLOCK)  # below it, no token of the CTA forces or masks a block
        for q in cutlass.range_constexpr(rows_per_cta):
            if t0 + q < num_tokens:
                raw_q = _ldg32(p_nvb + Int64(t0 + q) * 4)
                valid_q = _imax(Int32(0), _imin(raw_q, G))
                max_valid = _imax(max_valid, valid_q)
                min_forced = _imin(min_forced, _imin(_imax(raw_q - local, Int32(0)), valid_q))
        tq = (tid % threads_per_block_row) * 4
        brow = tid // threads_per_block_row
        tok = t0 + tq
        valid4 = []
        local4 = []
        for i in cutlass.range_constexpr(4):
            raw = Int32(0)
            if tok + i < num_tokens:
                raw = _ldg32(p_nvb + Int64(tok + i) * 4)
            valid4.append(_imax(Int32(0), _imin(raw, G)))
            local4.append(_imax(raw - local, Int32(0)))
        src = (p_s, hoff, sG, sT, num_tokens)
        forcing = (init, min_forced, valid4, local4)
        dst = (tile + tq * (row_words * 4), (tq >> 2) * swizzle_step)
        block_limit = max_valid
        if cutlass.const_expr(self.keys_in_registers):
            self._load_tile(self.vec, src, brow, tok, block_limit, forcing, dst)
        else:
            full4 = Int32(0)
            if tok + 3 < num_tokens:
                full4 = vec_ok
            for k in cutlass.range((max_valid + 127) >> 7, unroll=2):
                self._load_store_slot(k, full4, src, brow, tok, block_limit, forcing, dst)
        cute.arch.sync_threads()

        # ---- per-row exact select, one warp per row ----
        if w < num_tokens - t0:
            t = t0 + w
            V = _imax(Int32(0), _imin(_ldg32(p_nvb + Int64(t) * 4), G))
            nsl = (V + 127) >> 7
            X = (w >> 2) * swizzle_step
            rowa = tile + w * (row_words * 4)
            cand_w = cand + w * (_MAX_CANDIDATES * 8)
            sel_w = selected + w * (_TOPK * 4)
            outa = p_out + (Int64(t) * out_stride_t + Int64(h) * out_stride_h) * 4
            lane4 = lane * 4
            keys = []
            lmax = Int32(_KEY_NEG_INF)
            if cutlass.const_expr(self.keys_in_registers):
                for j in cutlass.range_constexpr(num_slots):
                    keys += _lds128(rowa + ((j * 128 + lane4) ^ X) * 4)
                for j in cutlass.range_constexpr(num_slots):
                    for i in cutlass.range_constexpr(4):
                        lmax = _imax(lmax, keys[4 * j + i])
            else:
                for j in cutlass.range(nsl, unroll=2):
                    lmax = _lds128_max(rowa + ((j * 128 + lane4) ^ X) * 4, lmax)
            # Keys at or above tau cover the row's top 16; -inf, masked blocks and negative NaN
            # (keys <= _KEY_NEG_INF) are never selected.
            tau = _imax(_sixteenth_largest(lmax, lane), Int32(_KEY_NEG_INF + 1))

            c = Int32(0)
            if cutlass.const_expr(self.keys_in_registers):
                m0 = Int32(0)
                m1 = Int32(0)
                for j in cutlass.range_constexpr(num_slots):
                    for i in cutlass.range_constexpr(4):
                        e = 4 * j + i
                        if cutlass.const_expr(e < 32):
                            m0 = _set_bit_if_ge(m0, keys[e], tau, e)
                        else:
                            m1 = _set_bit_if_ge(m1, keys[e], tau, e - 32)
                mine = _popc(m0)
                if cutlass.const_expr(4 * num_slots > 32):
                    mine = mine + _popc(m1)
                incl = mine
                for o in cutlass.range_constexpr(5):
                    up = cute.arch.shuffle_sync_up(incl, offset=1 << o, mask_and_clamp=0)
                    incl = _add_if_lane_ge(incl, up, lane, 1 << o)
                c = cute.arch.shuffle_sync(incl, offset=31)
                if c <= _MAX_CANDIDATES:
                    pos = incl - mine
                    mm = m0
                    while mm != 0:
                        e0, mm = _pop_lowest_bit(mm)
                        _store_candidate_of_element(rowa, X, lane4, e0, 0, cand_w, pos)
                        pos = pos + 1
                    if cutlass.const_expr(4 * num_slots > 32):
                        mm = m1
                        while mm != 0:
                            e1, mm = _pop_lowest_bit(mm)
                            _store_candidate_of_element(rowa, X, lane4, e1, 32, cand_w, pos)
                            pos = pos + 1
            else:
                lanes_below = (Int32(1) << lane) - 1
                for j in cutlass.range(nsl):
                    a4 = _lds128(rowa + ((j * 128 + lane4) ^ X) * 4)
                    for i in cutlass.range_constexpr(4):
                        pi = _ge(a4[i], tau)
                        below, total = _ballot_rank(pi, lanes_below)
                        if pi != 0 and c + below < _MAX_CANDIDATES:
                            _store_candidate(cand_w + (c + below) * 8, a4[i], j * 128 + lane4 + i)
                        c = c + total
            cute.arch.sync_warp()

            if c <= _MAX_CANDIDATES:
                # Rank each candidate by (key, smaller block) among all; the top 16 are the answer.
                my1 = Int64(0)
                my2 = Int64(0)
                if lane < c:
                    my1 = _lds64(cand_w + lane * 8)
                if lane + 32 < c:
                    my2 = _lds64(cand_w + (lane + 32) * 8)
                r1 = Int32(0)
                r2 = Int32(0)
                if c <= 32:
                    for kk in cutlass.range(c, unroll=4):
                        r1 = _count_greater(cand_w + kk * 8, my1, r1)
                else:
                    for kk in cutlass.range(c, unroll=2):
                        r1, r2 = _count_greater_x2(cand_w + kk * 8, my1, r1, my2, r2)
                if lane < c and r1 < _TOPK:
                    _sts32(sel_w + r1 * 4, _block_of_candidate(my1))
                if lane + 32 < c and r2 < _TOPK:
                    _sts32(sel_w + r2 * 4, _block_of_candidate(my2))
                cute.arch.sync_warp()
                v = Int32(_NO_BLOCK)
                if lane < _imin(c, Int32(_TOPK)):
                    v = _lds32(sel_w + lane * 4)
                v = _sort_ascending(v, lane)
                if lane < _TOPK:
                    _store_block_id(outa + Int64(lane) * 4, v)
            else:
                mine_key = self._extract_top16(rowa, X, nsl, V, lane)
                block = _sort_ascending(_block_of_extracted(mine_key), lane)
                if lane < _TOPK:
                    _store_block_id(outa + Int64(lane) * 4, block)

    @cute.jit
    def _extract_top16(self, rowa, X, nsl, V, lane):
        """Lane r < 16 receives the r-th best packed (key, 65535 - block), or -1 past the end."""
        prev = Int64(0x7FFFFFFFFFFFFFFF)
        mine = Int64(-1)
        for r in cutlass.range(Int32(_TOPK)):
            best = Int64(-1)
            for j in cutlass.range(nsl):
                b0 = j * 128 + lane * 4
                best = _extract_step(rowa + (b0 ^ X) * 4, b0, V, prev, best)
            best = _warp_max64(best)
            if lane == r:
                mine = best
            prev = best
        return mine


# --------------------------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------------------------
# _compiled[bucket][vec]: the kernel for that capacity, with or without 16-byte loads.
_compiled: list[tuple[Callable[..., None], Callable[..., None]]] = []
# _BUCKET_OF_SLOTS[n]: index of the smallest _BUCKETS capacity that holds n 128-block slots.
_BUCKET_OF_SLOTS = tuple(
    next(i for i, (slots, _) in enumerate(_BUCKETS) if slots >= n)
    for n in range(_MAX_BLOCKS // 128 + 1)
)


def _compile_all() -> None:
    """Compile every bucket: rows grow while serving, and a compile then would stall the step."""
    scores = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (cute.sym_int(), cute.sym_int(), cute.sym_int()),
        stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
    )
    n_valid = cute.runtime.make_fake_tensor(Int32, (cute.sym_int(),), stride=(1,))
    out = cute.runtime.make_fake_tensor(
        Int32,
        (cute.sym_int(), cute.sym_int(), _TOPK),
        stride=(cute.sym_int64(), cute.sym_int64(), 1),
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    def compile_one(slots: int, rows_per_cta: int, vec: bool) -> Callable[..., None]:
        kernel = _SelectContextBlocksKernel(slots, rows_per_cta, vec)
        zero = Int32(0)
        args = (scores, n_valid, out, zero, zero, zero, stream)
        return cute.compile(kernel, *args, options="--enable-tvm-ffi")

    kernels = []
    for slots, rows_per_cta in _BUCKETS:
        scalar = compile_one(slots, rows_per_cta, False)
        if slots > _MAX_REGISTER_SLOTS:
            # Loads one slot at a time and switches on vec_ok at runtime.
            kernels.append((scalar, scalar))
        else:
            kernels.append((scalar, compile_one(slots, rows_per_cta, True)))
    _compiled.extend(kernels)


def minimax_m3_select_context_blocks(
    scores: torch.Tensor,
    n_valid_blocks: torch.Tensor,
    *,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Select each context row's top-16 blocks, as minimax_m3_select_blocks with head-major output.

    Block b < V = clamp(n_valid_blocks[t], 0, G) scores 1e29 if b >= n_valid_blocks[t] -
    local_blocks, else 1e30 if b < init_blocks, else its raw score; blocks >= V never compete.
    Ties go to the smaller block (the C++ op breaks a tie of more than 2048 blocks in atomic order,
    so such rows can differ from it), and a block whose score is -inf is never selected. A
    negative-NaN score ranks with -inf here, where the C++ op orders it differently per dispatch
    path; the indexer scorers cannot produce NaN from finite inputs.

    The first call compiles a kernel per row-width bucket and load width, so it must not run under
    CUDA graph capture. Scores with G > 4096 blocks go to the C++ op whole.

    Args:
        scores: [num_kv_heads, G, T] float32 CUDA, any strides; token-contiguous loads fastest.
        n_valid_blocks: [T] int32 contiguous, on the scores' device.
        init_blocks: Leading blocks forced into every row.
        local_blocks: Trailing valid blocks forced into every row.

    Returns:
        [T, num_kv_heads, 16] int32 ascending block ids with -1 tail padding, backed head-major
        so ``result.permute(1, 0, 2)`` is contiguous.
    """
    H, G, num_tokens = scores.shape
    if scores.dtype != torch.float32 or n_valid_blocks.dtype != torch.int32:
        raise ValueError("expected float32 scores and int32 n_valid_blocks")
    if n_valid_blocks.shape != (num_tokens,) or not n_valid_blocks.is_contiguous():
        raise ValueError("n_valid_blocks must be a contiguous [T] tensor")
    if init_blocks < 0 or local_blocks < 0:
        raise ValueError("init_blocks and local_blocks must be non-negative")
    if G > _MAX_BLOCKS:
        return torch.ops.trtllm.minimax_m3_select_blocks(
            scores, n_valid_blocks, _TOPK, init_blocks, local_blocks, True
        )
    out = torch.empty((H, num_tokens, _TOPK), dtype=torch.int32, device=scores.device)
    out = out.permute(1, 0, 2)
    if H == 0 or num_tokens == 0:
        return out
    if not _compiled:
        _compile_all()
    sH, sG, sT = scores.stride()
    vec = int(
        scores.data_ptr() % 16 == 0
        and sT == 1
        and sG % 4 == 0
        and sH % 4 == 0
        and num_tokens % 4 == 0
    )
    _compiled[_BUCKET_OF_SLOTS[(G + 127) // 128]][vec](
        scores, n_valid_blocks, out, init_blocks, local_blocks, vec
    )
    return out

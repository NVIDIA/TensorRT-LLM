# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pool gather + pre-expert NVFP4 quantization, with the router logits and the routing.

``pool_gather_fp4quant`` (the fused pool gather of ``projection_gathers``) gathers the eight ranks' pool slices
inside the pre-expert RMSNorm + NVFP4 quantization kernel. This entry is that
kernel with the MoE routing folded in, for a decode step of at most 8 rows:

* each rank brings only its own 32 router logits per row
  (``ops/moe/router_shard``), ``[M, 32]`` FP32;
* a CTA has two warp groups. Warps 0-3 are ``pool_gather_fp4quant``'s kernel
  (flashinfer's 128-thread geometry): the pool gather from words 0-255 of the
  512-word inbox rows, the RMSNorm and the quantization. Warps 4-7 send the
  rank's whole ``[M, 32]`` logit block in words 256-511 of the box of the
  CTA's first row (same tag and slot as that row's pool words) and receive
  the seven others, so every CTA holds all ``M x 256`` logits;
* warps 4-7 then compute, redundantly in every CTA, the routing of all M rows
  exactly as the built-in ``_full_routing_kernel`` does (sigmoid, + expert
  bias for the selection, top-8 by a unique 64-bit key, the unbiased scores
  renormalized with ``eps``) and, when asked, the ring-buddy placement of
  ``replicated_experts`` exactly as ``ops/moe/ring_buddy_assign`` computes it, and
  write the CTA's rows of the full logits, the selected ids and FP32 weights,
  the placed local ids and BF16 weights, and (CTA 0) the per-rank loads.
  ``PLACE_BALANCE3`` (``three_window_placement``) places over three ring windows instead,
  by the rule of ``ring_windows.py``.

The two groups share no data and run concurrently: the kernel takes the
longer of the two paths, not their sum. They meet at one named barrier only,
by which the routing warps tell the quantizing warps that they have read the
first row's state before the latter advance it.

The quantization, the gathered row and the pool protocol are
``pool_gather_fp4quant``'s, statement for statement; its CTA barriers are named
barriers of warps 0-3, and flashinfer's ``row_reduce`` is copied with that
barrier (the same additions in the same order).

Bits. The routing reproduces the built-in kernel's instructions on the values
that decide bits: the sigmoid is ``mul.f32 x, -log2(e)`` -> ``ex2.approx.f32``
-> ``add.f32 1`` -> ``div.full.f32 1, d``; the bias add and the zero
canonicalization are the built-in's; the selection order is the descending
order of keys that are unique by construction, which any correct sort
reproduces; the weight sum is the built-in tree
``((p0+p1)+(p4+p5)) + ((p2+p3)+(p6+p7))`` (the built-in's other summands are
exact zeros); ``eps + sum`` and ``div.rn.f32`` as there. Every float operation
is inline PTX, so no contraction or reassociation can intervene. The
placement is integer arithmetic, the weights' BF16 cast ``cvt.rn.bf16.f32``.

Ownership and progress: the logit words ride in the box of the first row of
their CTA under that row's tag and slot, written by one sender per (slot,
sender, row), read and cleared within the call by the one CTA of the
receiving rank that owns the row. That is the pool words' protocol
(``projection_inbox``): a sender reuses a slot two calls later only after its
own intermediate call received this rank's words of that call, which this
rank sends only after the clearing call completed (stream order). Both
groups take the tag and slot from the row state as the previous call left
it, which the quantizing warps advance only after the routing warps signalled
that they have read it.

Early expert ids (``early_expert_ids``, ``early=`` an ``ops/moe/early_expert_ids``
buffer; a separately compiled variant, the default kernel is unchanged): the
routing warps of CTA 0 also store every row's placed ids as tagged 64-bit
words as soon as the placement inputs exist, tagged with the expert kernel's
next launch epoch (its workspace counter + 1, read after this kernel's
dependency wait), and the CTAs trigger their dependents after their logit
exchange (``exchange``; ``entry``: right after their
wait), never before their own wait. In that variant the routing warps' trigger
is the CTA's only one: the quantizing warps do not trigger, so the expert
kernel never becomes resident while a CTA's cross-rank logit exchange is
still open, whichever input of the CTA arrives first. The published id and
the ``local_ids`` store are one expression (``_placed_expert``), for the
placements in ``EARLY_PLACES``. The persistent expert kernel launched behind
this one reads those words before its dependency wait
(``ops/moe/persistent_experts``).
"""

from __future__ import annotations

import functools
import operator
from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .pool_gather_fp4quant import _AVAILABLE as _POOL_AVAILABLE
from .pool_gather_fp4quant import BLOCK, HIDDEN, LOCAL_COLUMNS
from .pool_gather_fp4quant import pool_gather_rmsnorm_fp4quant as _pool_op
from .projection_inbox import MAX_ROWS, WORLD, ProjectionInbox

#: The router's experts, the 32 each expert-parallel rank owns, and the top-k.
EXPERTS = 256
LOCAL_EXPERTS = EXPERTS // WORLD
TOP_K = 8
#: Payload words per inbox row: the pool slice, then the logit block.
POOL_WORDS = LOCAL_COLUMNS // 2
LOGIT_WORDS = MAX_ROWS * LOCAL_EXPERTS
ROW_WORDS = POOL_WORDS + LOGIT_WORDS

#: Placement kinds (the replicated experts' modes): none (routing only), every expert on
#: its owner, the balanced ring-buddy flow, a fixed routing-independent move; and
#: ``three_window_placement``'s balanced flow over three ring windows (``ring_windows.py``),
#: whose local ids address a 96-expert ``[own | b-1 | b-2]`` stack.
PLACE_NONE = 0
PLACE_OWNER = 1
PLACE_BALANCE = 2
PLACE_STATIC = 3
PLACE_BALANCE3 = 4
#: The placements whose ids ``early_expert_ids`` publishes: those ``_placed_expert``
#: computes. A placement added later is refused with ``early=`` until the
#: publication and the ``local_ids`` store compute it through that one helper.
EARLY_PLACES = (PLACE_OWNER, PLACE_BALANCE, PLACE_STATIC, PLACE_BALANCE3)

#: The CTA: warps 0-3 run the pool gather + quantization (flashinfer's 128-thread
#: geometry), warps 4-7 the logit exchange, the routing and the placement. Named
#: barriers: one per warp group, and one by which the routing warps tell the
#: quantizing warps that they have read the row state the latter then advance.
QUANT_WARPS = 4
QUANT_THREADS = 32 * QUANT_WARPS
ROUTE_THREADS = 128
QUANT_BARRIER = 1
ROUTE_BARRIER = 2
STATE_BARRIER = 3


_AVAILABLE = _POOL_AVAILABLE


if _AVAILABLE:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    from .pool_gather_fp4quant import (
        _FlashinferKernel,
        _fp4,
        _load_shared_words,
        _store_global_words,
        _store_shared_words,
    )
    from .projection_inbox import (
        _asm,
        _i64,
        _u32,
        any_stale,
        clear_words,
        load_state,
        load_words,
        next_tag,
        poll_words,
        publish_words,
        store_state,
    )

    def _u64(value, loc, ip):
        return Uint64(value).ir_value(loc=loc, ip=ip)

    def _pure(result_type, operands, text, constraints, loc, ip):
        return _asm(result_type, operands, text, constraints, loc, ip, side_effects=False)

    # ---- memory -------------------------------------------------------------

    @dsl_user_op
    def _ld_shared_b32(ptr, *, loc=None, ip=None):
        return Uint32(
            _asm(
                T.i32(),
                [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
                "ld.shared.b32 $0, [$1];",
                "=r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _st_shared_b32(ptr, value, *, loc=None, ip=None):
        _asm(
            None,
            [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), _u32(value, loc, ip)],
            "st.shared.b32 [$0], $1;",
            "r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _ld_global_b32(address, *, loc=None, ip=None):
        return Uint32(
            _asm(T.i32(), [_i64(address, loc, ip)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip)
        )

    @dsl_user_op
    def _st_global_b32(address, value, *, loc=None, ip=None):
        _asm(
            None,
            [_i64(address, loc, ip), _u32(value, loc, ip)],
            "st.global.b32 [$0], $1;",
            "l,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _ld_relaxed_gpu_b32(address, *, loc=None, ip=None):
        """A 32-bit relaxed GPU-scope load (early expert ids: the expert kernel's counter)."""
        return Uint32(
            _asm(
                T.i32(),
                [_i64(address, loc, ip)],
                "ld.relaxed.gpu.global.b32 $0, [$1];",
                "=r,l",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _st_relaxed_gpu_pair(address, lo, hi, *, loc=None, ip=None):
        """One single-copy-atomic 64-bit relaxed GPU-scope store of ``(lo, hi)`` (early expert ids)."""
        _asm(
            None,
            [_i64(address, loc, ip), _u32(lo, loc, ip), _u32(hi, loc, ip)],
            "{\n.reg .b64 t;\nmov.b64 t, {$1, $2};\nst.relaxed.gpu.global.b64 [$0], t;\n}",
            "l,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _stamp_timer(address, *, loc=None, ip=None):
        """Store the global timer (ns) at ``address`` (the early-ids timeline mode)."""
        _asm(
            None,
            [_i64(address, loc, ip)],
            "{\n.reg .u64 c;\nmov.u64 c, %globaltimer;\nst.global.u64 [$0], c;\n}",
            "l",
            loc,
            ip,
        )

    @dsl_user_op
    def _st_global_b16(address, value, *, loc=None, ip=None):
        _asm(
            None,
            [_i64(address, loc, ip), _u32(value, loc, ip)],
            "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\nst.global.b16 [$0], h;\n}",
            "l,r",
            loc,
            ip,
        )

    # ---- the routing's float arithmetic, as the built-in kernel's PTX -------

    @dsl_user_op
    def _sigmoid(bits, *, loc=None, ip=None):
        """``1 / (1 + 2^(x * -log2 e))`` with the built-in kernel's four instructions."""
        return Uint32(
            _pure(
                T.i32(),
                [_u32(bits, loc, ip)],
                "{\n.reg .f32 x, t, e, d, one, s;\nmov.b32 x, $1;\n"
                "mul.rn.f32 t, x, 0fBFB8AA3B;\nex2.approx.f32 e, t;\n"
                "add.rn.f32 d, e, 0f3F800000;\nmov.b32 one, 0f3F800000;\n"
                "div.full.f32 s, one, d;\nmov.b32 $0, s;\n}",
                "=r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _route_key(score, bias, low, *, loc=None, ip=None):
        """``(ordered(+0-canonical(score + bias)) << 32) | low`` as the built-in kernel forms it."""
        return Uint64(
            _pure(
                T.i64(),
                [_u32(score, loc, ip), _u32(bias, loc, ip), _u32(low, loc, ip)],
                "{\n.reg .f32 s, b, r;\n.reg .b32 rb, m;\n.reg .pred pz, pp;\n"
                "mov.b32 s, $1;\nmov.b32 b, $2;\nadd.rn.f32 r, s, b;\n"
                "setp.eq.f32 pz, r, 0f00000000;\nmov.b32 rb, r;\nselp.b32 rb, 0, rb, pz;\n"
                "setp.gt.s32 pp, rb, -1;\nselp.b32 m, -2147483648, -1, pp;\n"
                "xor.b32 m, m, rb;\nmov.b64 $0, {$3, m};\n}",
                "=l,r,r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _f32_bits(value, *, loc=None, ip=None):
        return Uint32(
            _pure(
                T.i32(),
                [Float32(value).ir_value(loc=loc, ip=ip)],
                "mov.b32 $0, $1;",
                "=r,f",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _lo32(value, *, loc=None, ip=None):
        return Uint32(
            _pure(T.i32(), [_u64(value, loc, ip)], "cvt.u32.u64 $0, $1;", "=r,l", loc, ip)
        )

    @dsl_user_op
    def _hi32(value, *, loc=None, ip=None):
        return Uint32(
            _pure(
                T.i32(),
                [_u64(value, loc, ip)],
                "{\n.reg .b32 lo;\nmov.b64 {lo, $0}, $1;\n}",
                "=r,l",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _key64(hi, lo, *, loc=None, ip=None):
        return Uint64(
            _pure(
                T.i64(),
                [_u32(hi, loc, ip), _u32(lo, loc, ip)],
                "mov.b64 $0, {$2, $1};",
                "=l,r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _add_rn(a, b, *, loc=None, ip=None):
        return Uint32(
            _pure(
                T.i32(),
                [_u32(a, loc, ip), _u32(b, loc, ip)],
                "{\n.reg .f32 x, y, z;\nmov.b32 x, $1;\nmov.b32 y, $2;\n"
                "add.rn.f32 z, x, y;\nmov.b32 $0, z;\n}",
                "=r,r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _div_rn(a, b, *, loc=None, ip=None):
        return Uint32(
            _pure(
                T.i32(),
                [_u32(a, loc, ip), _u32(b, loc, ip)],
                "{\n.reg .f32 x, y, z;\nmov.b32 x, $1;\nmov.b32 y, $2;\n"
                "div.rn.f32 z, x, y;\nmov.b32 $0, z;\n}",
                "=r,r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _bf16_bits(a, *, loc=None, ip=None):
        """The BF16 rounding (to nearest even) of an FP32, in the low half of a word."""
        return Uint32(
            _pure(
                T.i32(),
                [_u32(a, loc, ip)],
                "{\n.reg .f32 x;\n.reg .b16 h;\nmov.b32 x, $1;\n"
                "cvt.rn.bf16.f32 h, x;\ncvt.u32.u16 $0, h;\n}",
                "=r,r",
                loc,
                ip,
            )
        )

    # ---- 64-bit keys, warp exchanges, integer helpers -----------------------

    @dsl_user_op
    def _cas_desc(a, b, *, loc=None, ip=None):
        """``(max, min)`` of two unsigned 64-bit keys."""
        result = _pure(
            ir.Type.parse("!llvm.struct<(i64,i64)>"),
            [_u64(a, loc, ip), _u64(b, loc, ip)],
            "max.u64 $0, $2, $3;\nmin.u64 $1, $2, $3;",
            "=l,=l,l,l",
            loc,
            ip,
        )
        return (
            Uint64(llvm.extractvalue(T.i64(), result, [0], loc=loc, ip=ip)),
            Uint64(llvm.extractvalue(T.i64(), result, [1], loc=loc, ip=ip)),
        )

    @dsl_user_op
    def _stamp(address, *, loc=None, ip=None):
        """Store the SM clock (64-bit) at ``address`` (the phase probe build)."""
        _asm(
            None,
            [_i64(address, loc, ip)],
            "{\n.reg .u64 c;\nmov.u64 c, %clock64;\nst.global.u64 [$0], c;\n}",
            "l",
            loc,
            ip,
        )

    @dsl_user_op
    def _red_shared_or(ptr, value, *, loc=None, ip=None):
        _asm(
            None,
            [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), _u32(value, loc, ip)],
            "red.shared.or.b32 [$0], $1;",
            "r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _shfl_xor_b32(value, mask: int, *, loc=None, ip=None):
        return Uint32(
            _asm(
                T.i32(),
                [_u32(value, loc, ip)],
                f"shfl.sync.bfly.b32 $0, $1, {mask}, 31, -1;",
                "=r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _shfl_idx_b32(value, source, *, loc=None, ip=None):
        return Uint32(
            _asm(
                T.i32(),
                [_u32(value, loc, ip), _u32(source, loc, ip)],
                "shfl.sync.idx.b32 $0, $1, $2, 31, -1;",
                "=r,r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _redux_max_u32(value, *, loc=None, ip=None):
        return Uint32(
            _asm(T.i32(), [_u32(value, loc, ip)], "redux.sync.max.u32 $0, $1, -1;", "=r,r", loc, ip)
        )

    @dsl_user_op
    def _redux_max_s32(value, *, loc=None, ip=None):
        return Uint32(
            _asm(T.i32(), [_u32(value, loc, ip)], "redux.sync.max.s32 $0, $1, -1;", "=r,r", loc, ip)
        )

    @dsl_user_op
    def _popc(value, *, loc=None, ip=None):
        return Uint32(_pure(T.i32(), [_u32(value, loc, ip)], "popc.b32 $0, $1;", "=r,r", loc, ip))

    @dsl_user_op
    def _sel(cond, a, b, *, loc=None, ip=None):
        """``a`` where ``cond`` (a word) is nonzero, else ``b``."""
        return Uint32(
            _pure(
                T.i32(),
                [_u32(cond, loc, ip), _u32(a, loc, ip), _u32(b, loc, ip)],
                "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nselp.b32 $0, $2, $3, p;\n}",
                "=r,r,r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _sel64(cond, a, b, *, loc=None, ip=None):
        return Uint64(
            _pure(
                T.i64(),
                [_u32(cond, loc, ip), _u64(a, loc, ip), _u64(b, loc, ip)],
                "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nselp.b64 $0, $2, $3, p;\n}",
                "=l,r,l,l",
                loc,
                ip,
            )
        )

    def _cmp(op: str):
        @dsl_user_op
        def compare(a, b, *, loc=None, ip=None):
            return Uint32(
                _pure(
                    T.i32(),
                    [_u32(a, loc, ip), _u32(b, loc, ip)],
                    f"{{\n.reg .pred p;\nsetp.{op} p, $1, $2;\nselp.u32 $0, 1, 0, p;\n}}",
                    "=r,r,r",
                    loc,
                    ip,
                )
            )

        return compare

    _eq = _cmp("eq.u32")
    _lt_u = _cmp("lt.u32")
    _ge_u = _cmp("ge.u32")

    @dsl_user_op
    def _max_s32(a, b, *, loc=None, ip=None):
        return Uint32(
            _pure(
                T.i32(),
                [_u32(a, loc, ip), _u32(b, loc, ip)],
                "max.s32 $0, $1, $2;",
                "=r,r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _min_u32(a, b, *, loc=None, ip=None):
        return Uint32(
            _pure(
                T.i32(),
                [_u32(a, loc, ip), _u32(b, loc, ip)],
                "min.u32 $0, $1, $2;",
                "=r,r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _div_u32(a, divisor: int, *, loc=None, ip=None):
        return Uint32(
            _pure(T.i32(), [_u32(a, loc, ip)], f"div.u32 $0, $1, {divisor};", "=r,r", loc, ip)
        )

    #: A 19-comparator sorting network for 8 keys (max to the lower index: descending).
    _SORT8 = (
        ((0, 2), (1, 3), (4, 6), (5, 7)),
        ((0, 4), (1, 5), (2, 6), (3, 7)),
        ((0, 1), (2, 3), (4, 5), (6, 7)),
        ((2, 4), (3, 5)),
        ((1, 4), (3, 6)),
        ((1, 2), (3, 4), (5, 6)),
    )

    def _sort8(keys):
        keys = list(keys)
        for stage in _SORT8:
            for i, j in stage:
                keys[i], keys[j] = _cas_desc(keys[i], keys[j])
        return keys

    def _top8_by_redux(lists, lane):
        """The warp's 8 largest keys of each list group: key ``s`` in lane ``s`` (else 0),
        and every round's winning low word (warp-uniform).

        ``lists``: per row, this lane's keys sorted descending. Eight rounds; in
        each the warp's largest head key is found by two ``redux.sync.max.u32``
        (high word, then the low word among the lanes holding that high word) and
        the one lane holding it (keys are unique) pops it. The rows' rounds are
        interleaved for instruction-level parallelism.
        """
        lists = [list(keys) for keys in lists]
        picks = [Uint64(0) for _ in lists]
        winners = [[] for _ in lists]
        lane_u = Uint32(lane)
        for s in range(TOP_K):
            is_slot = _eq(lane_u, Uint32(s))
            for r, keys in enumerate(lists):
                head_hi = _hi32(keys[0])
                head_lo = _lo32(keys[0])
                top_hi = _redux_max_u32(head_hi)
                at_top = _eq(head_hi, top_hi)
                top_lo = _redux_max_u32(_sel(at_top, head_lo, Uint32(0)))
                won = at_top & _eq(head_lo, top_lo)
                for i in range(TOP_K - 1):
                    keys[i] = _sel64(won, keys[i + 1], keys[i])
                keys[TOP_K - 1] = _sel64(won, Uint64(0), keys[TOP_K - 1])
                picks[r] = _sel64(is_slot, _key64(top_hi, top_lo), picks[r])
                winners[r].append(top_lo)
        return picks, winners

    def _placed_expert(g, words, xs, peer_rank: int, x2s=None):
        """The placement of selected expert ``g`` on rank ``peer_rank`` (two or three ring windows).

        The local id of the copy this rank runs: ``g``'s index in its own block
        (own experts, 0..31), in the ring predecessor's block (32..63) when the
        expert moves one rank on, else ``g``'s id past the 64-expert window (the
        expert runs on another rank). ``words`` are the census bits per block
        (which experts any row selected), ``xs`` per block how many of its
        highest selected experts move one rank on (the placement mode's
        result: 0 for owner-only, the balanced flow, or the static move).
        With ``x2s`` (three windows, ``PLACE_BALANCE3``),
        ``xs`` counts the experts that leave the block and ``x2s`` (<= ``xs``)
        those of them that go on two ranks: the ``x2s`` highest present ones run
        two ranks on (local ids 64..95, the second predecessor's block), the next
        ``xs - x2s`` one rank on, and an expert run elsewhere gets its id past
        the 96-expert window. ``x2s`` is a trace-time choice: without it the
        expression is the two-window one, operation for operation.
        One expression for the ``local_ids`` store and the early-id
        publication, so both carry the same id.
        """
        blk = g >> 5
        idx = g & Uint32(31)
        if x2s is not None:
            word_blk = words[0]
            m_blk = xs[0]
            x2_blk = x2s[0]
            for b in range(1, WORLD):
                at = _eq(blk, Uint32(b))
                word_blk = _sel(at, words[b], word_blk)
                m_blk = _sel(at, xs[b], m_blk)
                x2_blk = _sel(at, x2s[b], x2_blk)
            above = _popc((word_blk >> idx) >> 1)
            hops = _lt_u(above, m_blk) + _lt_u(above, x2_blk)
            runs_on = (blk + hops) & Uint32(WORLD - 1)
            rank_u = Uint32(peer_rank)
            here = idx + hops * Uint32(LOCAL_EXPERTS)
            wrap = Uint32(EXPERTS - 3 * LOCAL_EXPERTS)
            elsewhere = _sel(_ge_u(g, wrap), g - wrap, g) + Uint32(3 * LOCAL_EXPERTS)
            return _sel(_eq(runs_on, rank_u), here, elsewhere)
        word_blk = words[0]
        x_blk = xs[0]
        for b in range(1, WORLD):
            at = _eq(blk, Uint32(b))
            word_blk = _sel(at, words[b], word_blk)
            x_blk = _sel(at, xs[b], x_blk)
        above = _popc((word_blk >> idx) >> 1)
        runs_on = (blk + _lt_u(above, x_blk)) & Uint32(WORLD - 1)
        rank_u = Uint32(peer_rank)
        here = _sel(_eq(blk, rank_u), idx, idx + Uint32(LOCAL_EXPERTS))
        wrap = Uint32(EXPERTS - 2 * LOCAL_EXPERTS)
        elsewhere = _sel(_ge_u(g, wrap), g - wrap, g) + Uint32(2 * LOCAL_EXPERTS)
        return _sel(_eq(runs_on, rank_u), here, elsewhere)

    # ---- the CTA's two warp groups --------------------------------------------

    def _bar_sync(barrier: int, threads: int):
        cute.arch.barrier(barrier_id=barrier, number_of_threads=threads)

    @dsl_user_op
    def _bar_arrive(barrier: int, threads: int, *, loc=None, ip=None):
        _asm(None, [], f"bar.arrive {barrier}, {threads};", "", loc, ip)

    @cute.jit
    def _row_sum(x: cute.TensorSSA, reduction_buffer: cute.Tensor) -> Float32:
        """flashinfer's ``row_reduce`` (ADD, one CTA, 64 threads per row), its barrier
        a named one of the quantizing warps only: the same additions in the same order."""
        local_val = x.reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)
        warp_val = _fp4.warp_reduce(local_val, operator.add, width=32)
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        warps_per_row = cute.size(reduction_buffer.shape[1])
        row_idx = warp_idx // warps_per_row
        col_idx = warp_idx % warps_per_row
        if lane_idx == 0:
            reduction_buffer[row_idx, col_idx] = warp_val
        cute.arch.barrier(barrier_id=QUANT_BARRIER, number_of_threads=QUANT_THREADS)
        block_reduce_val = Float32(0.0)
        if lane_idx < warps_per_row:
            block_reduce_val = reduction_buffer[row_idx, lane_idx]
        return _fp4.warp_reduce(block_reduce_val, operator.add)

    class _PoolGatherRouteKernel(_FlashinferKernel):
        """flashinfer's RMSNorm + FP4 quantization from the pool inbox, plus the routing."""

        def __init__(
            self,
            peer_rank: int,
            place: int,
            static_from: int,
            route: bool = True,
            probe: bool = False,
            publish: int = 0,
            epoch_word: int = 0,
            trigger: int = 0,
            stamps: int = 0,
        ):
            super().__init__(
                dtype=cutlass.BFloat16,
                H=HIDDEN,
                block_size=BLOCK,
                output_swizzled=False,
                is_fp16=False,
                sm_version=103,
                scale_format="e4m3",
            )
            if not (
                self.cluster_n == 1
                and self.threads_per_row == 64
                and self.rows_per_block == 2
                and self.vec_size == 8
                and self.num_vec_blocks == WORLD
                and self.num_threads == 128
                and self.num_vec_blocks * self.vec_size * self.threads_per_row == HIDDEN
            ):
                raise ValueError(
                    "flashinfer's RMSNormFP4QuantKernel geometry changed: "
                    f"cluster {self.cluster_n}, {self.threads_per_row} threads per row, "
                    f"{self.rows_per_block} rows per CTA, {self.num_vec_blocks} vectors of "
                    f"{self.vec_size}, {self.num_threads} threads"
                )
            self.peer_rank = peer_rank
            self.place = place if route else PLACE_NONE
            self.static_from = static_from
            # False only in the phase microbenchmark: exchange the logits, skip
            # the routing and the placement (their cost is the difference).
            self.route = route
            # True only in the phase microbenchmark: thread 0 of every CTA stores
            # the SM clock at six points into the gathered-row output (which a
            # probe build does not otherwise write: store_gathered must be False).
            self.probe = probe
            # Early expert ids (ops/moe/early_expert_ids.py), compile-time addresses,
            # 0 = off: CTA 0's routing warps store the placed ids as tagged words
            # at ``publish``, tagged with 1 + the expert workspace counter at
            # ``epoch_word``; ``trigger`` moves the dependents' launch to after
            # the logit exchange (1) or the dependency wait (2), and then the
            # routing warps' trigger is the CTA's only one. ``stamps`` (with or
            # without a publication) receives the global timer at fixed points.
            if publish and (self.place not in EARLY_PLACES or not epoch_word):
                raise ValueError(
                    "the early-id publication needs one of the placements _placed_expert computes "
                    f"(kinds {EARLY_PLACES}, got {self.place}) and the expert counter"
                )
            self.publish = publish
            self.epoch_word = epoch_word
            self.trigger = trigger if publish else 0
            self.stamps = stamps

        def _route_smem_bytes(self) -> int:
            # logits [8, 256] FP32, selected ids [64] and weights [64], the placement's
            # census [8], with alignment slack.
            return MAX_ROWS * EXPERTS * 4 + 2 * MAX_ROWS * TOP_K * 4 + WORLD * 4 + 128

        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor,
            mY: cute.Tensor,
            mS: cute.Tensor,
            mGlobalScale: cute.Tensor,
            M: Int32,
            eps: Float32,
            renorm_eps: Float32,
            local: Int64,
            counters: Int64,
            inbox: Int64,
            stride: Int64,
            logits_local: Int64,
            bias: Int64,
            logits_full: Int64,
            selected: Int64,
            weights: Int64,
            local_ids: Int64,
            weights_bf16: Int64,
            stats: Int64,
            enable_pdl: cutlass.Constexpr[bool],
            store_gathered: cutlass.Constexpr[bool],
            stream,
        ):
            tv_shape, tv_stride = self._make_tv_layout(
                self.threads_per_row,
                self.rows_per_block,
                self.vec_size,
                self.num_vec_blocks,
            )
            tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
            tiler_mn = (self.rows_per_block, self.cols_per_tile)
            self.kernel(
                mX,
                mW,
                mY,
                mS,
                mGlobalScale,
                M,
                eps,
                renorm_eps,
                local,
                counters,
                inbox,
                stride,
                logits_local,
                bias,
                logits_full,
                selected,
                weights,
                local_ids,
                weights_bf16,
                stats,
                enable_pdl,
                store_gathered,
                tv_layout,
                tiler_mn,
            ).launch(
                grid=[cute.ceil_div(M, self.rows_per_block), 1, 1],
                block=[self.num_threads + ROUTE_THREADS, 1, 1],
                smem=self._smem_size_in_bytes() + self._route_smem_bytes(),
                stream=stream,
                use_pdl=enable_pdl,
            )

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor,
            mY: cute.Tensor,
            mS: cute.Tensor,
            mGlobalScale: cute.Tensor,
            M: Int32,
            eps: Float32,
            renorm_eps: Float32,
            local: Int64,
            counters: Int64,
            inbox: Int64,
            stride: Int64,
            logits_local: Int64,
            bias: Int64,
            logits_full: Int64,
            selected: Int64,
            weights: Int64,
            local_ids: Int64,
            weights_bf16: Int64,
            stats: Int64,
            enable_pdl: cutlass.Constexpr[bool],
            store_gathered: cutlass.Constexpr[bool],
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            lane = tidx % 32
            warp = tidx // 32

            H = self.H
            block_size = self.block_size
            num_sf_blocks_per_row = self.num_sf_blocks_per_row
            threads_per_row = tv_layout.shape[0][0]
            warps_per_row = max(threads_per_row // 32, 1)
            rows_per_block = tiler_mn[0]

            smem = cutlass.utils.SmemAllocator()
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            sLogits = smem.allocate_tensor(
                Uint32, cute.make_layout((MAX_ROWS * EXPERTS,)), byte_alignment=16
            )
            sSelected = smem.allocate_tensor(
                Uint32, cute.make_layout((MAX_ROWS * TOP_K,)), byte_alignment=16
            )
            sWeights = smem.allocate_tensor(
                Uint32, cute.make_layout((MAX_ROWS * TOP_K,)), byte_alignment=16
            )
            # The placement's census: bit i of word b is set when local expert i of
            # block b is selected by some row.
            sCensus = smem.allocate_tensor(Uint32, cute.make_layout((WORLD,)), byte_alignment=16)
            probe = mX.iterator.toint() + Int64(bidx * 64)

            if warp < QUANT_WARPS:
                # ==========================================================
                # Warps 0-3: pool_gather_fp4quant statement for statement,
                # in 512-word inbox rows (the pool payload in words 0-255), its
                # CTA barriers named ones of these four warps.
                # ==========================================================
                if enable_pdl:
                    cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.probe):
                    if tidx == 0:
                        _stamp(probe)
                lane_in_row = tidx % threads_per_row
                row_in_block = tidx // threads_per_row
                fp4_max_rcp = _fp4.rcp_approx_ftz(Float32(_fp4.FLOAT4_E2M1_MAX))

                gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
                copy_atom_load_async = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(),
                    mX.element_type,
                    num_bits_per_copy=_fp4.COPY_BITS,
                )
                tiled_copy_load = cute.make_tiled_copy(copy_atom_load_async, tv_layout, tiler_mn)
                thr_copy_X = tiled_copy_load.get_slice(tidx)
                tXgX = thr_copy_X.partition_S(gX)
                tXsX = thr_copy_X.partition_D(sX)
                tXrX = cute.make_fragment_like(tXgX)

                # Phase 1: the pool gather.
                actual_row_idx = bidx * rows_per_block + row_in_block
                state = counters + Int64(actual_row_idx) * 8
                tag = Uint32(0)
                slot = Uint32(0)
                if actual_row_idx < M:
                    last_tag, last_slot = load_state(state)
                    tag = next_tag(last_tag)
                    slot = (last_slot ^ Uint32(1)) & Uint32(1)
                    word = Int64(lane_in_row * 4)
                    row64 = Int64(actual_row_idx)
                    l0, l1, l2, l3 = load_words(local + (row64 * POOL_WORDS + word) * 4)
                    sent = (
                        (Int64(slot) * WORLD + self.peer_rank) * MAX_ROWS + row64
                    ) * ROW_WORDS + word
                    for b in cutlass.range_constexpr(WORLD):
                        if cutlass.const_expr(b != self.peer_rank):
                            publish_words(inbox + Int64(b) * stride + sent * 8, l0, l1, l2, l3, tag)
                    own = inbox + Int64(self.peer_rank) * stride
                    received = cute.make_rmem_tensor(cute.make_layout((4 * WORLD,)), Uint32)
                    stale = Uint32(1)
                    while stale != Uint32(0):
                        stale = Uint32(0)
                        for b in cutlass.range_constexpr(WORLD):
                            if cutlass.const_expr(b != self.peer_rank):
                                box = (
                                    (Int64(slot) * WORLD + b) * MAX_ROWS + row64
                                ) * ROW_WORDS + word
                                p0, t0, p1, t1, p2, t2, p3, t3 = poll_words(own + box * 8)
                                received[4 * b] = p0
                                received[4 * b + 1] = p1
                                received[4 * b + 2] = p2
                                received[4 * b + 3] = p3
                                stale = stale | any_stale(t0, t1, t2, t3, tag)
                    gathered_row = mX.iterator.toint() + row64 * (H * 2)
                    for b in cutlass.range_constexpr(WORLD):
                        if cutlass.const_expr(b != self.peer_rank):
                            box = ((Int64(slot) * WORLD + b) * MAX_ROWS + row64) * ROW_WORDS + word
                            clear_words(own + box * 8)
                            w0 = received[4 * b]
                            w1 = received[4 * b + 1]
                            w2 = received[4 * b + 2]
                            w3 = received[4 * b + 3]
                        else:
                            w0, w1, w2, w3 = l0, l1, l2, l3
                        column = b * LOCAL_COLUMNS + lane_in_row * 8
                        _store_shared_words(sX.iterator + row_in_block * H + column, w0, w1, w2, w3)
                        if cutlass.const_expr(store_gathered):
                            _store_global_words(gathered_row + Int64(column) * 2, w0, w1, w2, w3)

                # Phase 2: sum of squares (flashinfer's).
                _bar_sync(QUANT_BARRIER, QUANT_THREADS)
                if cutlass.const_expr(self.probe):
                    if tidx == 0:
                        _stamp(probe + Int64(8))
                cute.autovec_copy(tXsX, tXrX)
                x = tXrX.load().to(Float32)
                x_sq = x * x
                sum_sq = _row_sum(x_sq, reduction_buffer)
                mean_sq = sum_sq / H
                rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)
                global_scale_val = mGlobalScale[0]
                _bar_sync(QUANT_BARRIER, QUANT_THREADS)
                if cutlass.const_expr(self.probe):
                    if tidx == 0:
                        _stamp(probe + Int64(16))
                # The routing warps have read the CTA's first-row state too.
                _bar_sync(STATE_BARRIER, QUANT_THREADS + ROUTE_THREADS)

                # Phase 3: RMSNorm + quantize (flashinfer's).
                if actual_row_idx < M:
                    if lane_in_row == 0:
                        store_state(state, tag, slot)
                    num_sf_per_thread = (
                        num_sf_blocks_per_row + threads_per_row - 1
                    ) // threads_per_row
                    for sf_iter in range(num_sf_per_thread):
                        sf_idx = lane_in_row + sf_iter * threads_per_row
                        if sf_idx < num_sf_blocks_per_row:
                            block_start = sf_idx * block_size

                            x_h2 = cute.make_rmem_tensor((8,), Uint32)
                            w_h2 = cute.make_rmem_tensor((8,), Uint32)
                            x_tile = sX.iterator + row_in_block * H + block_start
                            x_h2[0], x_h2[1], x_h2[2], x_h2[3] = _load_shared_words(x_tile)
                            x_h2[4], x_h2[5], x_h2[6], x_h2[7] = _load_shared_words(x_tile + 8)
                            w_ptr0 = _fp4.get_ptr_as_int64(mW, block_start)
                            w_ptr1 = _fp4.get_ptr_as_int64(mW, block_start + Int32(8))
                            w_h2[0], w_h2[1], w_h2[2], w_h2[3] = _fp4.ld_global_v4_u32(w_ptr0)
                            w_h2[4], w_h2[5], w_h2[6], w_h2[7] = _fp4.ld_global_v4_u32(w_ptr1)

                            xw_h2 = _fp4.bfloat2_mul_8(x_h2, w_h2)
                            max_xw_h2 = _fp4.bfloat2_max_abs_8(xw_h2)
                            max_xw = _fp4.bfloat2_hmax_to_f32(max_xw_h2)
                            y_f32 = _fp4.bfloat2_to_float16(xw_h2, rstd)

                            max_abs = max_xw * rstd

                            scale_float = global_scale_val * max_abs * fp4_max_rcp
                            scale_float = _fp4.fmin_f32(scale_float, Float32(_fp4.FLOAT8_E4M3_MAX))

                            scale_fp8_u32 = _fp4.cvt_f32_to_e4m3(scale_float)
                            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))

                            inv_scale = (
                                _fp4.fp8_e4m3_to_f32_and_rcp(scale_fp8_u32) * global_scale_val
                            )

                            mS[actual_row_idx, sf_idx] = scale_fp8

                            packed64 = _fp4.quantize_and_pack_16(y_f32, inv_scale)

                            out_offset = block_start // 2
                            out_ptr = _fp4.get_ptr_as_int64(
                                mY, actual_row_idx * (H // 2) + out_offset
                            )
                            _fp4.st_global_u64(out_ptr, packed64)
                if cutlass.const_expr(self.probe):
                    if tidx == 0:
                        _stamp(probe + Int64(24))
                if cutlass.const_expr(self.stamps != 0):
                    if tidx == 0:
                        _stamp_timer(Int64(self.stamps) + Int64(bidx * 64 + 32))
                # Early ids with an early trigger: the routing warps' trigger (after the
                # logit exchange) is the CTA's only one, since the first trigger
                # in a CTA counts and these warps can finish first.
                if cutlass.const_expr(self.trigger == 0):
                    if enable_pdl:
                        cute.arch.griddepcontrol_launch_dependents()
            else:
                # ==========================================================
                # Warps 4-7: the logit blocks (in the box of the CTA's first row,
                # words 256-511, under that row's tag and slot), the routing of
                # all M rows (warp 4 + w takes rows w and w + 4; lane l holds
                # expert 32k + l of block k) and the placement.
                # ==========================================================
                rt = tidx - QUANT_THREADS
                rwarp = warp - QUANT_WARPS
                # The expert bias is a parameter: read before the wait.
                bias_k = [_ld_global_b32(bias + Int64((32 * k + lane) * 4)) for k in range(WORLD)]
                if enable_pdl:
                    cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.publish != 0):
                    # Early ids: every grid before this one has completed (the wait above
                    # returned), so the expert kernel's counter is final; its next
                    # launch uses counter + 1 (never 0). The counter is read before
                    # any trigger of this thread. The expert launch a trigger
                    # enables advances it only after its CTA saw CTA 0's words,
                    # which CTA 0 stores after the routing barrier that each of its
                    # routing threads reaches after this read (or after a fallback
                    # wait, i.e. once this kernel completed). Every trigger option
                    # comes after this kernel's own wait: the transitivity the
                    # expert kernel's early reads rely on.
                    pub_tag = _ld_relaxed_gpu_b32(Int64(self.epoch_word)) + Uint32(1)
                    pub_tag = _sel(_eq(pub_tag, Uint32(0)), Uint32(1), pub_tag)
                    if cutlass.const_expr(self.trigger == 2):
                        if enable_pdl:
                            cute.arch.griddepcontrol_launch_dependents()
                if cutlass.const_expr(self.stamps != 0):
                    if rt == 0:
                        _stamp_timer(Int64(self.stamps) + Int64(bidx * 64))
                if cutlass.const_expr(self.probe):
                    if rt == 0:
                        _stamp(probe + Int64(32))
                if cutlass.const_expr(self.place != PLACE_NONE):
                    if rt < WORLD:
                        _st_shared_b32(sCensus.iterator + rt, Uint32(0))
                first_row = Int64(bidx * rows_per_block)
                last_tag, last_slot = load_state(counters + first_row * 8)
                _bar_arrive(STATE_BARRIER, QUANT_THREADS + ROUTE_THREADS)
                tag = next_tag(last_tag)
                slot = (last_slot ^ Uint32(1)) & Uint32(1)
                if rt * 4 < M * LOCAL_EXPERTS:
                    word = Int64(rt * 4)
                    g0, g1, g2, g3 = load_words(logits_local + word * 4)
                    sent = (
                        ((Int64(slot) * WORLD + self.peer_rank) * MAX_ROWS + first_row) * ROW_WORDS
                        + POOL_WORDS
                        + word
                    )
                    for b in cutlass.range_constexpr(WORLD):
                        if cutlass.const_expr(b != self.peer_rank):
                            publish_words(inbox + Int64(b) * stride + sent * 8, g0, g1, g2, g3, tag)
                    own = inbox + Int64(self.peer_rank) * stride
                    received = cute.make_rmem_tensor(cute.make_layout((4 * WORLD,)), Uint32)
                    stale = Uint32(1)
                    while stale != Uint32(0):
                        stale = Uint32(0)
                        for b in cutlass.range_constexpr(WORLD):
                            if cutlass.const_expr(b != self.peer_rank):
                                box = (
                                    ((Int64(slot) * WORLD + b) * MAX_ROWS + first_row) * ROW_WORDS
                                    + POOL_WORDS
                                    + word
                                )
                                q0, s0, q1, s1, q2, s2, q3, s3 = poll_words(own + box * 8)
                                received[4 * b] = q0
                                received[4 * b + 1] = q1
                                received[4 * b + 2] = q2
                                received[4 * b + 3] = q3
                                stale = stale | any_stale(s0, s1, s2, s3, tag)
                    # This thread's words: row (4 rt) // 32 of the block, experts
                    # (4 rt) % 32 .. + 3 of each sender's 32.
                    logit_row = (rt * 4) // LOCAL_EXPERTS
                    logit_col = (rt * 4) % LOCAL_EXPERTS
                    for b in cutlass.range_constexpr(WORLD):
                        if cutlass.const_expr(b != self.peer_rank):
                            box = (
                                ((Int64(slot) * WORLD + b) * MAX_ROWS + first_row) * ROW_WORDS
                                + POOL_WORDS
                                + word
                            )
                            clear_words(own + box * 8)
                            v0 = received[4 * b]
                            v1 = received[4 * b + 1]
                            v2 = received[4 * b + 2]
                            v3 = received[4 * b + 3]
                        else:
                            v0, v1, v2, v3 = g0, g1, g2, g3
                        _store_shared_words(
                            sLogits.iterator + logit_row * EXPERTS + b * LOCAL_EXPERTS + logit_col,
                            v0,
                            v1,
                            v2,
                            v3,
                        )
                _bar_sync(ROUTE_BARRIER, ROUTE_THREADS)
                if cutlass.const_expr(self.publish != 0):
                    # Early ids: this CTA's logit exchange is complete.
                    if cutlass.const_expr(self.trigger == 1):
                        if enable_pdl:
                            cute.arch.griddepcontrol_launch_dependents()
                if cutlass.const_expr(self.stamps != 0):
                    if rt == 0:
                        _stamp_timer(Int64(self.stamps) + Int64(bidx * 64 + 8))
                if cutlass.const_expr(self.probe):
                    if rt == 0:
                        _stamp(probe + Int64(40))
                # This CTA's rows of the full logits: 64 threads per row.
                full_row = bidx * rows_per_block + rt // 64
                if full_row < M:
                    lr = full_row * EXPERTS + (rt % 64) * 4
                    f0, f1, f2, f3 = _load_shared_words(sLogits.iterator + lr)
                    _store_global_words(logits_full + Int64(lr) * 4, f0, f1, f2, f3)
                eps_bits = _f32_bits(renorm_eps)
                # Each lane's keys of the warp's two rows, sorted; then the warp's
                # top 8 of each row (key s in lane s).
                lists = []
                for half in cutlass.range_constexpr(2 if self.route else 0):
                    row = rwarp + 4 * half
                    keys = []
                    for k in cutlass.range_constexpr(WORLD):
                        expert = 32 * k + lane
                        logit = _ld_shared_b32(sLogits.iterator + row * EXPERTS + expert)
                        keys.append(
                            _route_key(_sigmoid(logit), bias_k[k], Uint32(255) - Uint32(expert))
                        )
                    lists.append(_sort8(keys))
                picks, winners = _top8_by_redux(lists, lane)
                for half in cutlass.range_constexpr(2 if self.route else 0):
                    row = rwarp + 4 * half
                    if cutlass.const_expr(self.place != PLACE_NONE):
                        # The census, from the rounds' winners (warp-uniform): one
                        # lane sets each selected expert's bit.
                        counts = _eq(Uint32(lane), Uint32(0)) & _lt_u(Uint32(row), Uint32(M))
                        if counts != Uint32(0):
                            for s in cutlass.range_constexpr(TOP_K):
                                chosen = Uint32(255) - winners[half][s]
                                _red_shared_or(
                                    sCensus.iterator + (chosen >> 5),
                                    Uint32(1) << (chosen & Uint32(31)),
                                )
                for half in cutlass.range_constexpr(2 if self.route else 0):
                    row = rwarp + 4 * half
                    mine = picks[half]
                    # Masked so that the unused rows' (unwritten) keys stay in bounds.
                    expert_id = (Uint32(255) - _lo32(mine)) & Uint32(255)
                    score = _sigmoid(_ld_shared_b32(sLogits.iterator + row * EXPERTS + expert_id))
                    pair = _add_rn(score, _shfl_xor_b32(score, 1))
                    quad = _add_rn(pair, _shfl_xor_b32(pair, 4))
                    total = _add_rn(quad, _shfl_xor_b32(quad, 2))
                    weight = _div_rn(score, _add_rn(eps_bits, total))
                    keep = _lt_u(Uint32(row), Uint32(M)) & _lt_u(Uint32(lane), Uint32(TOP_K))
                    if keep != Uint32(0):
                        _st_shared_b32(sSelected.iterator + row * TOP_K + lane, expert_id)
                        _st_shared_b32(sWeights.iterator + row * TOP_K + lane, weight)
                        if row // rows_per_block == bidx:
                            out = Int64(row * TOP_K + lane) * 4
                            _st_global_b32(selected + out, expert_id)
                            _st_global_b32(weights + out, weight)
                if cutlass.const_expr(self.probe):
                    if rt == 0:
                        _stamp(probe + Int64(48))

                # Placement (replicated experts) from all M rows' selections, in every
                # routing warp.
                if cutlass.const_expr(self.place != PLACE_NONE):
                    _bar_sync(ROUTE_BARRIER, ROUTE_THREADS)
                    w0, w1, w2, w3 = _load_shared_words(sCensus.iterator)
                    w4, w5, w6, w7 = _load_shared_words(sCensus.iterator + 4)
                    words = [w0, w1, w2, w3, w4, w5, w6, w7]
                    loads = [_popc(w) for w in words]
                    lane_r = Uint32(lane) & Uint32(WORLD - 1)
                    # Per block, xs experts leave it (its highest present ones) and,
                    # with three windows, x2s of them go on two ranks.
                    x2s = None
                    if cutlass.const_expr(self.place == PLACE_BALANCE):
                        # Lane r (mod 8) takes the arcs ending at rank r.
                        held = loads[0]
                        for b in cutlass.range_constexpr(1, WORLD):
                            held = _sel(_eq(lane_r, Uint32(b)), loads[b], held)
                        arc = Uint32(0)
                        arcs = []
                        need = Uint32(0)
                        for k in cutlass.range_constexpr(WORLD):
                            span = k + 1
                            arc = arc + _shfl_idx_b32(
                                held, (lane_r - Uint32(k)) & Uint32(WORLD - 1)
                            )
                            arcs.append(arc)
                            if cutlass.const_expr(span == WORLD):
                                need = _max_s32(need, _div_u32(arc + Uint32(WORLD - 1), WORLD))
                            else:
                                need = _max_s32(need, _div_u32(arc + Uint32(span), span + 1))
                        t = _redux_max_s32(need)
                        moved_r = Uint32(0)
                        for k in cutlass.range_constexpr(WORLD):
                            moved_r = _max_s32(moved_r, arcs[k] - Uint32(k + 1) * t)
                        xs = [_shfl_idx_b32(moved_r, Uint32(b)) for b in range(WORLD)]
                    elif cutlass.const_expr(self.place == PLACE_BALANCE3):
                        # Three windows (ring_windows.py): the smallest maximum
                        # T over arcs that reach n + 2 ranks, the edge carries
                        # c[r] (moved_r, lane r), then per block the experts
                        # leaving it (xs: m = min(c, load)) and those going on
                        # two ranks (x2s).
                        held = loads[0]
                        for b in cutlass.range_constexpr(1, WORLD):
                            held = _sel(_eq(lane_r, Uint32(b)), loads[b], held)
                        arc = Uint32(0)
                        arcs = []
                        need = Uint32(0)
                        for k in cutlass.range_constexpr(WORLD):
                            reach = min(k + 3, WORLD)
                            arc = arc + _shfl_idx_b32(
                                held, (lane_r - Uint32(k)) & Uint32(WORLD - 1)
                            )
                            arcs.append(arc)
                            need = _max_s32(need, _div_u32(arc + Uint32(reach - 1), reach))
                        t = _redux_max_s32(need)
                        moved_r = Uint32(0)
                        for k in cutlass.range_constexpr(WORLD):
                            moved_r = _max_s32(moved_r, arcs[k] - Uint32(k + 1) * t)
                        cs = [_shfl_idx_b32(moved_r, Uint32(b)) for b in range(WORLD)]
                        xs = [_min_u32(cs[b], loads[b]) for b in range(WORLD)]
                        x2s = [
                            _max_s32(Uint32(0), cs[(b + 1) % WORLD] - loads[(b + 1) % WORLD])
                            for b in range(WORLD)
                        ]
                    elif cutlass.const_expr(self.place == PLACE_STATIC):
                        xs = [_popc(w >> self.static_from) for w in words]
                        moved_r = xs[0]
                        for b in cutlass.range_constexpr(1, WORLD):
                            moved_r = _sel(_eq(lane_r, Uint32(b)), xs[b], moved_r)
                    else:
                        xs = [Uint32(0) for _ in range(WORLD)]
                        moved_r = Uint32(0)
                    from_prev = _shfl_idx_b32(
                        moved_r, (lane_r + Uint32(WORLD - 1)) & Uint32(WORLD - 1)
                    )
                    if cutlass.const_expr(self.publish != 0):
                        # Early ids: CTA 0 publishes every row's placed ids, before its own
                        # rows' stores. Routing warp w takes rows w and w + 4, lane =
                        # slot; the id is _placed_expert's, as for the own-row store
                        # below; rows past M get -1.
                        for half in cutlass.range_constexpr(2):
                            prow = rwarp + 4 * half
                            pubs = _eq(Uint32(bidx), Uint32(0)) & _lt_u(Uint32(lane), Uint32(TOP_K))
                            if pubs != Uint32(0):
                                pslot = prow * TOP_K + lane
                                pg = _ld_shared_b32(sSelected.iterator + pslot)
                                pplaced = _placed_expert(pg, words, xs, self.peer_rank, x2s)
                                pplaced = _sel(
                                    _lt_u(Uint32(prow), Uint32(M)), pplaced, Uint32(0xFFFFFFFF)
                                )
                                _st_relaxed_gpu_pair(
                                    Int64(self.publish) + Int64(pslot) * 8, pplaced, pub_tag
                                )
                        if cutlass.const_expr(self.stamps != 0):
                            if rt == 0:
                                _stamp_timer(Int64(self.stamps) + Int64(bidx * 64 + 16))
                    # This CTA's rows: routing warp w < 2 takes row 2 * bidx + w,
                    # lane = slot.
                    own_row = bidx * rows_per_block + rwarp
                    writes = (
                        _lt_u(Uint32(rwarp), Uint32(rows_per_block))
                        & _lt_u(Uint32(own_row), Uint32(M))
                        & _lt_u(Uint32(lane), Uint32(TOP_K))
                    )
                    if writes != Uint32(0):
                        slot_index = own_row * TOP_K + lane
                        g = _ld_shared_b32(sSelected.iterator + slot_index)
                        placed = _placed_expert(g, words, xs, self.peer_rank, x2s)
                        _st_global_b32(local_ids + Int64(slot_index) * 4, placed)
                        w_bits = _ld_shared_b32(sWeights.iterator + slot_index)
                        _st_global_b16(weights_bf16 + Int64(slot_index) * 2, _bf16_bits(w_bits))
                    held_load = loads[0]
                    for b in cutlass.range_constexpr(1, WORLD):
                        held_load = _sel(_eq(lane_r, Uint32(b)), loads[b], held_load)
                    reports = (
                        _eq(Uint32(bidx), Uint32(0))
                        & _eq(Uint32(rwarp), Uint32(2))
                        & _lt_u(Uint32(lane), Uint32(WORLD))
                    )
                    if reports != Uint32(0):
                        _st_global_b32(stats + Int64(lane) * 4, held_load)
                        _st_global_b32(
                            stats + Int64(lane + WORLD) * 4, held_load - moved_r + from_prev
                        )
                if cutlass.const_expr(self.probe):
                    if rt == 0:
                        _stamp(probe + Int64(56))
                if cutlass.const_expr(self.stamps != 0):
                    if rt == 0:
                        _stamp_timer(Int64(self.stamps) + Int64(bidx * 64 + 24))
                if enable_pdl:
                    cute.arch.griddepcontrol_launch_dependents()

    @functools.lru_cache(maxsize=64)
    def _compile(
        device_index: int,
        peer_rank: int,
        place: int,
        static_from: int,
        enable_pdl: bool,
        store_gathered: bool,
        route: bool = True,
        probe: bool = False,
        publish: int = 0,
        epoch_word: int = 0,
        trigger: int = 0,
        stamps: int = 0,
    ):
        kernel = _PoolGatherRouteKernel(
            peer_rank, place, static_from, route, probe, publish, epoch_word, trigger, stamps
        )
        rows = cute.sym_int()
        fake = cute.runtime.make_fake_compact_tensor
        with torch.cuda.device(device_index):
            return cute.compile(
                kernel,
                fake(cutlass.BFloat16, (rows, HIDDEN), stride_order=(1, 0), assumed_align=128),
                fake(cutlass.BFloat16, (HIDDEN,), assumed_align=128),
                fake(cutlass.Uint8, (rows, HIDDEN // 2), stride_order=(1, 0), assumed_align=128),
                fake(
                    cutlass.Uint8, (rows, HIDDEN // BLOCK), stride_order=(1, 0), assumed_align=128
                ),
                fake(cutlass.Float32, (1,), assumed_align=4),
                Int32(1),
                Float32(1e-6),
                Float32(1e-20),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                enable_pdl,
                store_gathered,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )


class RouteOutputs:
    """What one call returns (all newly allocated, ``M`` rows).

    ``fp4`` ``[M, 2048]`` uint8 and ``scale`` ``[M, 256]`` uint8: the NVFP4
    expert input (unswizzled scales); ``gathered`` ``[M, 4096]`` BF16 or None;
    ``logits`` ``[M, 256]`` FP32: the router logits; ``selected`` ``[M, 8]``
    int32 and ``weights`` ``[M, 8]`` FP32: the routing; and with a placement,
    ``local_ids`` ``[M, 8]`` int32, ``weights_bf16`` ``[M, 8]`` BF16 and
    ``stats`` ``[16]`` int32 (``ring_buddy_assign``'s three outputs), else None.
    """

    __slots__ = (
        "fp4",
        "scale",
        "gathered",
        "logits",
        "selected",
        "weights",
        "local_ids",
        "weights_bf16",
        "stats",
    )

    def __init__(self, **values):
        for name in self.__slots__:
            setattr(self, name, values.get(name))

    def as_tuple(self):
        return tuple(getattr(self, name) for name in self.__slots__)


class PoolGatherRoute(OpWrapper):
    """Gather the pool slices and the logit slices; quantize; route; place.

    ``local`` is this rank's ``[M, 512]`` BF16 pool output, ``logits_local`` its
    ``[M, 32]`` FP32 router logits (experts ``32 * rank ...``), ``inbox`` a
    512-word-row ``ProjectionInbox``, ``gain`` / ``global_scale`` / ``eps`` the
    pre-expert norm's (as ``pool_gather_fp4quant``), ``bias`` the routing's
    ``[256]`` FP32 expert bias and ``renorm_eps`` its renormalization epsilon.
    ``place`` is one of ``PLACE_*``; ``static_from`` the ``PLACE_STATIC``
    threshold (``ring_buddy_assign``'s ``static_from``). Returns a
    ``RouteOutputs``.

    A collective over the TP group: every rank calls it with the same row count
    and placement kind, in the same order relative to its other calls on
    ``inbox``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DFlash2 K=7 verify: 8 rows, balanced ring-buddy placement (the verify shape)",
            spec=dict(rows=8, place=PLACE_BALANCE),
        ),
        Cell(why="8 rows, routing only (no placement)", spec=dict(rows=8, place=PLACE_NONE)),
        Cell(
            why="decode without drafts: 1 row, the CTA's second row idle",
            spec=dict(rows=1, place=PLACE_BALANCE),
        ),
        Cell(
            why="an odd row count (3 rows): the second CTA carries one row",
            spec=dict(rows=3, place=PLACE_BALANCE),
        ),
        Cell(why="8 rows, owner-only placement", spec=dict(rows=8, place=PLACE_OWNER)),
        Cell(why="8 rows, static placement", spec=dict(rows=8, place=PLACE_STATIC)),
        Cell(
            why="8 rows, balanced placement over three ring windows (the 96-expert stack)",
            spec=dict(rows=8, place=PLACE_BALANCE3),
        ),
        Cell(
            why="1 row over three ring windows: at most 8 experts, the carry rarely moves",
            spec=dict(rows=1, place=PLACE_BALANCE3),
        ),
    )

    note = """
    Certified against the built-in chain bit for bit: the full router's logits,
    `_full_routing_kernel`'s ids and weights, `ring_buddy_assign`'s local ids,
    BF16 weights and stats (`ring_windows.place`'s for `PLACE_BALANCE3`, which
    has no built-in counterpart), and the built-in gather + flashinfer quantization's
    payload, scales and gathered row; with random and constructed near-tie
    logits, a delayed rank, row-count changes with tag wrap-around and CUDA-graph
    replays. Rows 1..8 only.
    """

    def __call__(
        self,
        local: torch.Tensor,
        logits_local: torch.Tensor,
        inbox: ProjectionInbox,
        gain: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        bias: torch.Tensor,
        renorm_eps: float,
        place: int = PLACE_NONE,
        static_from: int = LOCAL_EXPERTS,
        enable_pdl: bool = True,
        return_gathered: bool = True,
        early=None,
        stamps: Optional[torch.Tensor] = None,
    ) -> RouteOutputs:
        rows = local.shape[0]
        device = local.device
        gathered = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        fp4 = torch.empty((rows, HIDDEN // 2), dtype=torch.uint8, device=device)
        scale = torch.empty((rows, HIDDEN // BLOCK), dtype=torch.uint8, device=device)
        logits = torch.empty((rows, EXPERTS), dtype=torch.float32, device=device)
        selected = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
        weights = torch.empty((rows, TOP_K), dtype=torch.float32, device=device)
        placed = place != PLACE_NONE
        local_ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=device) if placed else None
        weights_bf16 = (
            torch.empty((rows, TOP_K), dtype=torch.bfloat16, device=device) if placed else None
        )
        stats = torch.empty((2 * WORLD,), dtype=torch.int32, device=device) if placed else None
        # Early expert ids: publish the placed ids to the expert kernel that follows,
        # unless an earlier publication has not been consumed yet.
        publish = early is not None and placed and bool(enable_pdl) and early.may_publish()
        extra = (
            dict(
                publish=early.words_address,
                epoch_word=early.epoch_address,
                trigger=early.trigger,
                stamps=early.stamps_address,
            )
            if publish
            else {}
        )
        if stamps is not None:
            # Timeline mode (the early-ids chain timing): global-timer stamps, 8 per CTA.
            extra["stamps"] = stamps.data_ptr()
        _compile(
            device.index,
            inbox.rank,
            int(place),
            int(static_from),
            bool(enable_pdl),
            bool(return_gathered),
            **extra,
        )(
            gathered,
            gain,
            fp4,
            scale,
            global_scale,
            rows,
            eps,
            renorm_eps,
            local.data_ptr(),
            inbox.counters.data_ptr(),
            inbox.base,
            inbox.stride,
            logits_local.data_ptr(),
            bias.data_ptr(),
            logits.data_ptr(),
            selected.data_ptr(),
            weights.data_ptr(),
            local_ids.data_ptr() if placed else 0,
            weights_bf16.data_ptr() if placed else 0,
            stats.data_ptr() if placed else 0,
        )
        if publish:
            early.arm(local_ids)
        return RouteOutputs(
            fp4=fp4,
            scale=scale,
            gathered=gathered if return_gathered else None,
            logits=logits,
            selected=selected,
            weights=weights,
            local_ids=local_ids,
            weights_bf16=weights_bf16,
            stats=stats,
        )

    def reference(
        self,
        per_rank_locals: Sequence[torch.Tensor],
        per_rank_logits: Sequence[torch.Tensor],
        inbox: Optional[ProjectionInbox],
        gain: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        bias: torch.Tensor,
        renorm_eps: float,
        place: int = PLACE_NONE,
        static_from: int = LOCAL_EXPERTS,
        enable_pdl: bool = True,
        return_gathered: bool = True,
        rank: int = 0,
    ) -> RouteOutputs:
        """The quantization as ``pool_gather_fp4quant.reference``; the routing in float64.

        The collective exception to mirroring: ``per_rank_locals`` and
        ``per_rank_logits`` are every rank's operands in rank order, and
        ``rank`` names the rank whose placement is formed; ``inbox`` and
        ``enable_pdl`` are ignored. Scores are float64 sigmoids; the selection
        ranks their FP32 rounding plus the bias in FP32 (the built-in's
        selection key), ties to the lower expert id; the weights are the
        float64 renormalization of the float64 scores. The placement is
        ``ring_buddy_assign.reference`` on that selection (``ring_windows.place``
        for ``PLACE_BALANCE3``).
        """
        fp4, scale, gathered = _pool_op.reference(
            per_rank_locals, None, gain, global_scale, eps, return_gathered=return_gathered
        )
        logits = torch.cat([chunk.float() for chunk in per_rank_logits], dim=1)
        scores = torch.sigmoid(logits.double())
        ranked = scores.float() + bias.float()
        ids = torch.argsort(ranked, dim=1, descending=True, stable=True)[:, :TOP_K]
        picked = scores.gather(1, ids)
        weights = (picked / (picked.sum(1, keepdim=True) + renorm_eps)).float()
        selected = ids.to(torch.int32)
        local_ids = weights_bf16 = stats = None
        if place == PLACE_BALANCE3:
            from ...kernels.replicated_experts import windows as ring_windows

            flat, placed_stats = ring_windows.place(selected.reshape(-1).tolist(), rank, 3)
            local_ids = torch.tensor(flat, dtype=torch.int32, device=selected.device).reshape(
                selected.shape
            )
            weights_bf16 = weights.to(torch.bfloat16)
            stats = torch.tensor(placed_stats, dtype=torch.int32, device=selected.device)
        elif place != PLACE_NONE:
            from ..moe.ring_buddy_assign import ring_buddy_assign

            local_ids, weights_bf16, stats = ring_buddy_assign.reference(
                selected,
                weights,
                rank,
                balance=place == PLACE_BALANCE,
                static_from=static_from if place == PLACE_STATIC else LOCAL_EXPERTS,
            )
        return RouteOutputs(
            fp4=fp4,
            scale=scale,
            gathered=gathered,
            logits=logits,
            selected=selected,
            weights=weights,
            local_ids=local_ids,
            weights_bf16=weights_bf16,
            stats=stats,
        )

    def is_valid(
        self,
        local: torch.Tensor,
        logits_local: torch.Tensor,
        inbox: ProjectionInbox,
        gain: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        bias: torch.Tensor,
        renorm_eps: float,
        place: int = PLACE_NONE,
        static_from: int = LOCAL_EXPERTS,
        enable_pdl: bool = True,
        return_gathered: bool = True,
        early=None,
        stamps: Optional[torch.Tensor] = None,
    ) -> None:
        """Raw vector accesses at fixed strides: a strided or misaligned operand is read
        as if packed; a row count outside 1..8 addresses outside the inbox. The early ids
        and the timeline stamps are written at raw addresses on the operands' device."""
        rows = local.shape[0] if local.dim() == 2 else -1
        if not 1 <= rows <= MAX_ROWS or tuple(local.shape) != (rows, LOCAL_COLUMNS):
            raise ValueError(
                f"local must be [1..{MAX_ROWS}, {LOCAL_COLUMNS}], got {tuple(local.shape)}"
            )
        if early is not None and early.device != local.device:
            raise ValueError(f"early ids on {early.device}, operands on {local.device}")
        if stamps is not None and (
            stamps.dtype != torch.int64
            or not stamps.is_contiguous()
            or stamps.device != local.device
            or stamps.numel() < 8 * ((rows + 1) // 2)
        ):
            raise ValueError(
                f"stamps must be a contiguous int64 tensor of 8 words per CTA on "
                f"{local.device}, got {stamps.dtype} {tuple(stamps.shape)}"
            )
        for tensor, shape, dtype in (
            (local, (rows, LOCAL_COLUMNS), torch.bfloat16),
            (logits_local, (rows, LOCAL_EXPERTS), torch.float32),
            (gain, (HIDDEN,), torch.bfloat16),
            (global_scale, (1,), torch.float32),
            (bias, (EXPERTS,), torch.float32),
        ):
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
                or tensor.device != local.device
            ):
                raise ValueError(
                    f"expected a contiguous 16-byte aligned {dtype} {shape} tensor on "
                    f"{local.device}, got {tensor.dtype} {tuple(tensor.shape)}"
                )
        if inbox.row_words != ROW_WORDS:
            raise ValueError(f"inbox carries {inbox.row_words}-word rows, this op {ROW_WORDS}")
        if place not in (PLACE_NONE, PLACE_OWNER, PLACE_BALANCE, PLACE_STATIC, PLACE_BALANCE3):
            raise ValueError(f"unknown placement kind {place}")
        if early is not None and place not in (PLACE_NONE,) + EARLY_PLACES:
            raise ValueError(
                f"early_expert_ids publishes only the placements _placed_expert computes {EARLY_PLACES}, "
                f"got placement kind {place}"
            )
        if not 0 <= static_from <= LOCAL_EXPERTS:
            raise ValueError(f"static_from must be in [0, {LOCAL_EXPERTS}]")


pool_gather_route = PoolGatherRoute()


def available() -> bool:
    """Whether this image can build the kernel (CuTe DSL and flashinfer's kernel)."""
    return _AVAILABLE

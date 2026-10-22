# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fabric inbox for the column-sharded pool/unpool projections at TP4 or TP8, and its PTX.

The sharded projections compute ``N/world`` output columns per rank and need every
rank's slice before the next op. The built-in path gathers them with a
separate kernel (``musespark_open_projection_gather._gather_columns``) whose
output the consumer then reads back. The fused ops in this directory instead
let the *consumer* kernel publish its own rank's slice and read the peers'
slices straight out of this inbox, so the gather's launch, its output round
trip and its per-peer sequential polling disappear.

Protocol (the built-in gather's, with per-row state):

* Every 32-bit payload word (two BF16) travels as a 64-bit word
  ``(tag << 32) | payload``, stored with one system-scope 8-byte access, so a
  reader that sees the tag also sees the payload, whatever bits the payload
  holds (NaN payloads and negative zeros included). No sentinel value is
  reserved.
* The tag is a per-row epoch that every rank advances identically because
  every rank issues the same calls in the same order with the same row counts.
  Zero is never a tag: it is the empty word, and a reader clears every word it
  consumed back to zero before the kernel ends.
* Two slots alternate per row. Writing slot ``s`` of row ``r`` for use ``k+2``
  happens only after the writer completed use ``k+1``, which needed this rank's
  use-``k+1`` words, which this rank published only after use ``k`` (and the
  clears of its slot-``s`` words) completed. A slot is therefore never
  rewritten while it holds unread words.
* Per-row state (``counters[row] = (last tag, last slot)``) is read by every
  CTA of the row and written back by one of them after a barrier that all of
  them reach only after reading it. Rows a call does not touch keep their
  state, so calls of different row counts interleave freely.

Calls on one inbox must form one ordered CUDA execution chain per rank (one
stream, or explicit joins), with identical row counts and order on every rank.
Two call sites that run on different streams need two inboxes.
"""

from __future__ import annotations

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.mapping import Mapping

#: Rank counts the inbox connects: the attention TP group of tep4 (one tray) or tep8
#: (two trays). An inbox's own count is ``ProjectionInbox.world``.
WORLDS = (4, 8)
#: Widths of the two sharded projections' outputs: the pool (8192 -> 4096) and
#: the unpool (4096 -> 8192).
PROJECTION_WIDTHS = (4096, 8192)
#: Rows (tokens) per call the inbox has room for: a decode step of one request with
#: up to seven DFlash drafts. The built-in sharded projections stop at the same bound.
MAX_ROWS = 8
#: Rows per call of a wide inbox (``ProjectionInbox(..., max_rows=WIDE_ROWS)``): a decode
#: step of up to four requests with seven DFlash drafts each, at TP4 only, where the
#: fused consumers' per-CTA columns still fall inside one rank's slice at those rows.
WIDE_ROWS = 32
#: The row capacities an inbox can be built with.
CAPACITIES = (MAX_ROWS, WIDE_ROWS)
#: Bytes of one tagged payload word.
_TAGGED_BYTES = 8

try:
    import cutlass  # noqa: F401
    from cutlass import Float32, Int64, Uint32
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op
except ImportError:
    HAS_CUTE_DSL = False
else:
    HAS_CUTE_DSL = True


class _InboxFabricMemory(MnnvlMemory):
    """A fabric allocation of its own, split by TP group and ordered by TP rank."""

    force_fabric_handle = True

    @classmethod
    def comm_split_color_key(cls, mapping: Mapping) -> tuple[int, int]:
        return mapping.pp_rank * mapping.cp_size + mapping.cp_rank, mapping.tp_rank


def row_words_of(width: int, world: int) -> int:
    """32-bit payload words one of ``world`` ranks contributes per row of a ``width``-wide output."""
    return width // world // 2


class ProjectionInbox:
    """The receive slots of one sharded-projection site on every rank of a TP4 or TP8 group.

    ``row_words`` is the number of 32-bit payload words one rank contributes per
    row: half its local column count (``row_words_of``: at TP8 512 for the
    unpool's 1024 local columns and 256 for the pool's 512, at TP4 1024 and 512).
    Construction is collective over the TP group.

    ``max_rows`` is the most rows one call may carry: ``MAX_ROWS`` (8), or
    ``WIDE_ROWS`` (32) at TP4 for the wide sites of ``wide_projection_gathers``.

    Attributes read by the fused ops: ``base`` (address of rank 0's segment
    through the fabric mapping), ``stride`` (bytes between ranks' segments),
    ``counters`` (``int32 [max_rows, 2]``, the per-row tag and slot), ``rank``,
    ``world`` (the TP size) and ``max_rows`` (the rows per slot), which the fused
    ops compile in.
    ``segment()`` is this rank's receive memory as ``int64`` words (all zero
    whenever no call is in flight).
    """

    def __init__(self, mapping: Mapping, row_words: int, max_rows: int = MAX_ROWS) -> None:
        if mapping.tp_size not in WORLDS or mapping.enable_attention_dp or mapping.cp_size != 1:
            raise ValueError("ProjectionInbox needs TP4 or TP8 without attention DP or CP")
        world = mapping.tp_size
        allowed = tuple(row_words_of(width, world) for width in PROJECTION_WIDTHS)
        if row_words not in allowed:
            raise ValueError(f"row_words must be one of {allowed} at TP{world}, got {row_words}")
        if max_rows not in CAPACITIES or (max_rows != MAX_ROWS and world != 4):
            raise ValueError(
                f"max_rows must be {MAX_ROWS}, or {WIDE_ROWS} at TP4; got {max_rows} at TP{world}"
            )
        if not all(Distributed.get(mapping).tp_allgather(MnnvlMemory.supports_mnnvl())):
            raise ValueError("ProjectionInbox needs NVLink fabric memory on every TP rank")
        MnnvlMemory.initialize()
        self.rank = mapping.tp_rank
        self.world = world
        self.row_words = row_words
        self.max_rows = max_rows
        self.slot_bytes = world * max_rows * row_words * _TAGGED_BYTES
        self._owner = _InboxFabricMemory(mapping, 2 * self.slot_bytes)
        record = type(self._owner).allocated_map[self._owner.ptr]
        if record.comm_size != world or record.comm_rank != mapping.tp_rank:
            raise RuntimeError("ProjectionInbox allocation does not match the TP group")
        self.base = int(self._owner.ptr)
        self.stride = int(self._owner.rank_stride)
        # Every rank's segment as an int64 tensor. Kept for the inbox's lifetime:
        # the DLPack record behind it lives in this tensor object, and a view
        # that outlives it would release freed memory.
        self._segments = self._owner.as_torch_strided_tensor(torch.int64)
        self._segments[self.rank].zero_()
        self.counters = torch.zeros((max_rows, 2), dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        Distributed.get(mapping).tp_allgather(True)

    def segment(self) -> torch.Tensor:
        """This rank's receive slots, ``int64 [2 * world * max_rows * row_words]``."""
        return self._segments[self.rank]


if HAS_CUTE_DSL:

    def _asm(result_type, operands, text, constraints, loc, ip, side_effects=True):
        return llvm.inline_asm(
            result_type,
            operands,
            text,
            constraints,
            has_side_effects=side_effects,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    def _u32(value, loc, ip):
        return Uint32(value).ir_value(loc=loc, ip=ip)

    def _i64(value, loc, ip):
        return Int64(value).ir_value(loc=loc, ip=ip)

    def _split(result, count, loc, ip):
        return tuple(
            Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(count)
        )

    # Every global access below is a side-effecting asm statement, so none of
    # them can be hoisted above griddepcontrol.wait or merged across polls.

    @dsl_user_op
    def load_words(address, *, loc=None, ip=None):
        """Four consecutive 32-bit words (16 bytes) of a kernel input."""
        result = _asm(
            ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
            [_i64(address, loc, ip)],
            "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l",
            loc,
            ip,
        )
        return _split(result, 4, loc, ip)

    @dsl_user_op
    def publish_words(address, w0, w1, w2, w3, tag, *, loc=None, ip=None):
        """Store four payload words as four tagged 64-bit words (two 16-byte stores)."""
        _asm(
            None,
            [_i64(address, loc, ip)] + [_u32(v, loc, ip) for v in (w0, w1, w2, w3, tag)],
            "{\n.reg .b64 a, b, c, d;\n"
            "mov.b64 a, {$1, $5};\nmov.b64 b, {$2, $5};\n"
            "mov.b64 c, {$3, $5};\nmov.b64 d, {$4, $5};\n"
            "st.relaxed.sys.global.v2.b64 [$0], {a, b};\n"
            "st.relaxed.sys.global.v2.b64 [$0+16], {c, d};\n}",
            "l,r,r,r,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def poll_words(address, *, loc=None, ip=None):
        """Four tagged 64-bit words as (payload0, tag0, payload1, tag1, ...)."""
        result = _asm(
            ir.Type.parse("!llvm.struct<(i32,i32,i32,i32,i32,i32,i32,i32)>"),
            [_i64(address, loc, ip)],
            "{\n.reg .b64 a, b, c, d;\n"
            "ld.relaxed.sys.global.v2.b64 {a, b}, [$8];\n"
            "ld.relaxed.sys.global.v2.b64 {c, d}, [$8+16];\n"
            "mov.b64 {$0, $1}, a;\nmov.b64 {$2, $3}, b;\n"
            "mov.b64 {$4, $5}, c;\nmov.b64 {$6, $7}, d;\n}",
            "=r,=r,=r,=r,=r,=r,=r,=r,l",
            loc,
            ip,
        )
        return _split(result, 8, loc, ip)

    @dsl_user_op
    def poll_words_if(address, enabled, p0, t0, p1, t1, p2, t2, p3, t3, *, loc=None, ip=None):
        """``poll_words`` when ``enabled``; else the given words, without an access or a branch."""
        result = _asm(
            ir.Type.parse("!llvm.struct<(i32,i32,i32,i32,i32,i32,i32,i32)>"),
            [_i64(address, loc, ip), _u32(enabled, loc, ip)]
            + [_u32(v, loc, ip) for v in (p0, t0, p1, t1, p2, t2, p3, t3)],
            "{\n.reg .pred q;\n.reg .b64 a, b, c, d;\n"
            "mov.b64 a, {$10, $11};\nmov.b64 b, {$12, $13};\n"
            "mov.b64 c, {$14, $15};\nmov.b64 d, {$16, $17};\n"
            "setp.ne.u32 q, $9, 0;\n"
            "@q ld.relaxed.sys.global.v2.b64 {a, b}, [$8];\n"
            "@q ld.relaxed.sys.global.v2.b64 {c, d}, [$8+16];\n"
            "mov.b64 {$0, $1}, a;\nmov.b64 {$2, $3}, b;\n"
            "mov.b64 {$4, $5}, c;\nmov.b64 {$6, $7}, d;\n}",
            "=r,=r,=r,=r,=r,=r,=r,=r,l,r,r,r,r,r,r,r,r,r",
            loc,
            ip,
        )
        return _split(result, 8, loc, ip)

    @dsl_user_op
    def clear_words(address, *, loc=None, ip=None):
        """Return four consumed tagged words to the empty state."""
        _asm(
            None,
            [_i64(address, loc, ip)],
            "{\n.reg .b64 z;\nmov.b64 z, 0;\n"
            "st.relaxed.gpu.global.v2.b64 [$0], {z, z};\n"
            "st.relaxed.gpu.global.v2.b64 [$0+16], {z, z};\n}",
            "l",
            loc,
            ip,
        )

    @dsl_user_op
    def load_state(address, *, loc=None, ip=None):
        """A row's (last tag, last slot)."""
        result = _asm(
            ir.Type.parse("!llvm.struct<(i32,i32)>"),
            [_i64(address, loc, ip)],
            "ld.relaxed.gpu.global.v2.b32 {$0, $1}, [$2];",
            "=r,=r,l",
            loc,
            ip,
        )
        return _split(result, 2, loc, ip)

    @dsl_user_op
    def store_state(address, tag, slot, *, loc=None, ip=None):
        _asm(
            None,
            [_i64(address, loc, ip), _u32(tag, loc, ip), _u32(slot, loc, ip)],
            "st.relaxed.gpu.global.v2.b32 [$0], {$1, $2};",
            "l,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def next_tag(tag, *, loc=None, ip=None):
        """``tag + 1``, skipping zero on wrap."""
        result = _asm(
            T.i32(),
            [_u32(tag, loc, ip)],
            "{\n.reg .b32 t;\n.reg .pred p;\nadd.u32 t, $1, 1;\n"
            "setp.eq.u32 p, t, 0;\nselp.b32 $0, 1, t, p;\n}",
            "=r,r",
            loc,
            ip,
            side_effects=False,
        )
        return Uint32(result)

    @dsl_user_op
    def any_stale(t0, t1, t2, t3, tag, *, loc=None, ip=None):
        """1 while any of the four tags differs from ``tag``, else 0."""
        result = _asm(
            T.i32(),
            [_u32(v, loc, ip) for v in (t0, t1, t2, t3, tag)],
            "{\n.reg .pred p0, p1, p2, p3;\n"
            "setp.ne.u32 p0, $1, $5;\nsetp.ne.u32 p1, $2, $5;\n"
            "setp.ne.u32 p2, $3, $5;\nsetp.ne.u32 p3, $4, $5;\n"
            "or.pred p0, p0, p1;\nor.pred p2, p2, p3;\nor.pred p0, p0, p2;\n"
            "selp.u32 $0, 1, 0, p0;\n}",
            "=r,r,r,r,r,r",
            loc,
            ip,
            side_effects=False,
        )
        return Uint32(result)

    @dsl_user_op
    def stale_bit(t0, t1, t2, t3, tag, bit, *, loc=None, ip=None):
        """``bit`` while any of the four tags differs from ``tag``, else 0."""
        result = _asm(
            T.i32(),
            [_u32(v, loc, ip) for v in (t0, t1, t2, t3, tag, bit)],
            "{\n.reg .pred p0, p1, p2, p3;\n"
            "setp.ne.u32 p0, $1, $5;\nsetp.ne.u32 p1, $2, $5;\n"
            "setp.ne.u32 p2, $3, $5;\nsetp.ne.u32 p3, $4, $5;\n"
            "or.pred p0, p0, p1;\nor.pred p2, p2, p3;\nor.pred p0, p0, p2;\n"
            "selp.b32 $0, $6, 0, p0;\n}",
            "=r,r,r,r,r,r,r",
            loc,
            ip,
            side_effects=False,
        )
        return Uint32(result)

    @dsl_user_op
    def unpack_bf16x2(word, *, loc=None, ip=None):
        """The FP32 values of the low and high BF16 halves of ``word`` (exact)."""
        result = _asm(
            ir.Type.parse("!llvm.struct<(f32,f32)>"),
            [_u32(word, loc, ip)],
            "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xffff0000;\n"
            "mov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
            "=f,=f,r",
            loc,
            ip,
            side_effects=False,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from vLLM (https://github.com/vllm-project/vllm) under the Apache-2.0 license.
# Original file: vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/draft/markov.py
# Original copyright: Copyright contributors to the vLLM project
"""DSpark greedy Markov block chain over a vocab-sharded LM head, one launch (SM100 / SM103 / SM107).

Every TP rank holds its vocab shard of the block's base logits ``[B, K, shard]`` (BF16) and the
matching rows of ``markov_w2``. For each block position ``i`` the chain adds
``markov_w2 @ markov_w1[prev_i]`` to the position's logits and takes the global argmax, which is
draft token ``i`` and ``prev_{i+1}``; ``prev_0`` is the anchor, and an anchor outside
``[0, vocab)`` contributes a zero bias.

Grid: one CTA of 512 threads per 256 shard rows, all resident at once (each CTA waits for every
other CTA's keys). Per CTA:

* before ``griddepcontrol.wait``: its 256 ``markov_w2`` rows are copied into shared memory
  (``cp.async``, L2 evict-first) and the exchange set of the launch before the previous one is
  cleared;
* per position: warp ``w`` multiplies rows ``[16 w, 16 w + 16)`` by the requests' ``markov_w1`` rows
  with ``mma.sync.m16n8k16`` (BF16 in, FP32 accumulated over the Markov rank in order), rounds the
  bias to BF16 (``F.linear``'s output), adds the BF16 base logit in FP32 (the reference adds the
  BF16 bias to the FP32-cast logits, without a further rounding) and reduces each request to a
  64-bit key ``ordered(value) << 32 | ~index``. The CTA's key per request is stored into the
  exchange of every rank (``multimem.st``); every CTA then polls all ``tp x CTAs`` keys of the
  position and takes their maximum. Larger keys win: the larger value, then the lower global
  index, NaN above everything and ``-0 == +0``, which is ``torch.argmax`` on the gathered logits.

Exchange: three sets of ``[position, entry = rank x CTAs + CTA, request]`` u64 keys, 0 where
nothing was written (a key never has a zero 32-bit word). Launch ``L`` (from a per-rank ordinal
that CTA 0 advances after its last poll) uses set ``L % 3`` and clears set ``(L + 2) % 3``, which
launch ``L - 1`` used: every CTA of this rank finished reading it before launch ``L - 1``
completed, and peers write it again only in launch ``L + 2``, which needs this rank's keys of
launch ``L + 1``. The kernel never triggers its dependents early, so the next launch starts only
after this one has completed.

Options (per request width, :data:`OPTIONS`): ``staged`` -- per position, warp ``r`` copies request
``r``'s ``markov_w1`` row into shared memory (one coalesced 16-byte load per lane) and every warp
takes its MMA B fragments with ``ldmatrix`` instead of 32 scattered 4-byte loads per lane;
``gate`` -- before the sweep over all keys, lane ``s`` of warp 0 waits for the chunk holding the key
of rank ``s``'s last CTA, so fewer polls are outstanding while the keys are still in flight.
"""

from __future__ import annotations

import functools
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: Shard rows per CTA: 16 warps x one 16-row MMA tile.
ROWS = 256
THREADS = 512
WARPS = THREADS // 32
#: Largest block size and request count a launch serves (the exchange set is sized for them).
MAX_STEPS = 16
MAX_WIDTH = 16
#: Exchange sets in rotation.
NUM_SETS = 3
#: Int32 words of the per-rank state: launch ordinal, then the bytes each set's last launch used.
STATE_WORDS = 4
#: ``(staged, gate)`` per request width. Rubin R200, TP4 on one node, K = 7, 640 keys per request
#: and position (40960-row shards), CUDA graph, us per call, plain / staged / gate / both:
#: n1 32.7 / 32.2 / 28.7 / 28.2, n2 43.7 / 41.7 / 32.3 / 30.3, n4 41.8 / 36.5 / 34.7 / 29.5,
#: n8 42.7 / 31.9 / 40.0 / 30.1, n16 57.5 / 36.4 / 56.1 / 35.3 (scripts/draftfuse/bench_markov.py).
OPTIONS = {1: (True, True), 2: (True, True), 4: (True, True), 8: (True, True), 16: (True, True)}


def set_bytes(entries: int) -> int:
    """Bytes of one exchange set for ``entries`` (TP ranks x CTAs per rank)."""
    return MAX_STEPS * entries * MAX_WIDTH * 8


def ctas_for(shard: int) -> int:
    return (shard + ROWS - 1) // ROWS


def width_for(num_reqs: int) -> int:
    """Requests per key row: the next power of two."""
    return 1 << max(num_reqs - 1, 0).bit_length()


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _fv(value, *, loc=None, ip=None):
    return Float32(value).ir_value(loc=loc, ip=ip)


def _unpack(res, ty, cls, n, *, loc=None, ip=None):
    return tuple(cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _cp_async_evict_first(smem_addr, gmem_addr, *, loc=None, ip=None):
    """16-byte ``cp.async`` of static weights with an L2 evict-first policy."""
    _asm(
        None,
        [_iv(smem_addr, loc=loc, ip=ip), _lv(gmem_addr, loc=loc, ip=ip)],
        "{\n.reg .b64 pol;\ncreatepolicy.fractional.L2::evict_first.b64 pol, 1.0;\n"
        "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, pol;\n}",
        "r,l,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async_commit(*, loc=None, ip=None):
    _asm(None, [], "cp.async.commit_group;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _cp_async_wait_all(*, loc=None, ip=None):
    _asm(None, [], "cp.async.wait_group 0;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _ld_volatile_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _st_zero_v4(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)],
        "st.global.v4.b32 [$0], {0, 0, 0, 0};",
        "l,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_anchor(addr, ok, *, loc=None, ip=None):
    """The anchor token at ``addr`` (an int32, or the low word of an int64) if ``ok``, else -1."""
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip), _iv(ok, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $2, 0;\nmov.b32 $0, -1;\n@p ld.global.b32 $0, [$1];\n}",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_bf16_bits(addr, ok, *, loc=None, ip=None):
    """The BF16 at ``addr`` zero-extended to 32 bits if ``ok``, else 0 (a logit the previous
    grid wrote: read after ``griddepcontrol.wait``)."""
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip), _iv(ok, loc=loc, ip=ip)],
            "{\n.reg .pred p;\n.reg .b16 h;\nsetp.ne.s32 p, $2, 0;\nmov.b16 h, 0;\n"
            "@p ld.global.b16 h, [$1];\ncvt.u32.u16 $0, h;\n}",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_nc_b32(addr, *, loc=None, ip=None):
    """A 32-bit word of a static weight (read-only path)."""
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.nc.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_nc_v4_masked(addr, mask, *, loc=None, ip=None):
    """16 bytes of a static weight (read-only path), each word ANDed with ``mask``."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip), _iv(mask, loc=loc, ip=ip)],
        "{\n.reg .b32 a, b, c, d;\nld.global.nc.v4.b32 {a, b, c, d}, [$4];\n"
        "and.b32 $0, a, $5;\nand.b32 $1, b, $5;\nand.b32 $2, c, $5;\nand.b32 $3, d, $5;\n}",
        "=r,=r,=r,=r,l,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _st_shared_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)] + [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldsm_x4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _mma_bf16(a, b0, b1, c, *, loc=None, ip=None):
    """``d[16x8] = a[16x16] @ b[16x8] + c`` (BF16 inputs, FP32 accumulate)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(x, loc=loc, ip=ip) for x in (*a, b0, b1)] + [_fv(x, loc=loc, ip=ip) for x in c],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _logit_key(acc, base_bits, index, ok, *, loc=None, ip=None):
    """Key of ``bf16(acc) + base`` (FP32 add, round to nearest) at vocab ``index``, or 0 if not
    ``ok``. High word: the value's bits mapped to an unsigned order (``-0`` as ``+0``, every NaN
    above +inf); low word: ``~index`` (the lower index wins a tie). Neither word is ever 0."""
    return Int64(
        _asm(
            T.i64(),
            [
                _fv(acc, loc=loc, ip=ip),
                _iv(base_bits, loc=loc, ip=ip),
                _iv(index, loc=loc, ip=ip),
                _iv(ok, loc=loc, ip=ip),
            ],
            "{\n.reg .pred pz, pn, pnan, pok;\n.reg .b16 hb;\n.reg .b32 bb, xb, b, h, l, m;\n"
            ".reg .f32 bf, xf, v;\n"
            "cvt.rn.bf16.f32 hb, $1;\ncvt.u32.u16 bb, hb;\nshl.b32 bb, bb, 16;\nmov.b32 bf, bb;\n"
            "shl.b32 xb, $2, 16;\nmov.b32 xf, xb;\nadd.rn.f32 v, xf, bf;\nmov.b32 b, v;\n"
            "setp.eq.b32 pz, b, 0x80000000;\nselp.b32 b, 0, b, pz;\n"
            "setp.lt.s32 pn, b, 0;\nnot.b32 h, b;\nor.b32 m, b, 0x80000000;\nselp.b32 h, h, m, pn;\n"
            "and.b32 m, b, 0x7fffffff;\nsetp.gt.u32 pnan, m, 0x7f800000;\n"
            "selp.b32 h, 0xffffffff, h, pnan;\n"
            "not.b32 l, $3;\nsetp.ne.s32 pok, $4, 0;\nselp.b32 h, h, 0, pok;\nselp.b32 l, l, 0, pok;\n"
            "mov.b64 $0, {l, h};\n}",
            "=l,f,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _max_u64(a, b, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(a, loc=loc, ip=ip), _lv(b, loc=loc, ip=ip)],
            "max.u64 $0, $1, $2;",
            "=l,l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shuffle_max_u64(key, offset: int, *, loc=None, ip=None):
    """``max(key, key of lane ^ offset)`` (unsigned); every lane of the warp participates."""
    return Int64(
        _asm(
            T.i64(),
            [_lv(key, loc=loc, ip=ip)],
            "{\n.reg .b32 lo, hi, olo, ohi;\n.reg .b64 o;\nmov.b64 {lo, hi}, $1;\n"
            f"shfl.sync.bfly.b32 olo, lo, {offset}, 0x1f, 0xffffffff;\n"
            f"shfl.sync.bfly.b32 ohi, hi, {offset}, 0x1f, 0xffffffff;\n"
            "mov.b64 o, {olo, ohi};\nmax.u64 $0, $1, o;\n}",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_key(lo, hi, ok, *, loc=None, ip=None):
    """``hi << 32 | lo`` if ``ok``, else 0."""
    return Int64(
        _asm(
            T.i64(),
            [_iv(lo, loc=loc, ip=ip), _iv(hi, loc=loc, ip=ip), _iv(ok, loc=loc, ip=ip)],
            "{\n.reg .pred p;\n.reg .b32 l, h;\nsetp.ne.s32 p, $3, 0;\n"
            "selp.b32 l, $1, 0, p;\nselp.b32 h, $2, 0, p;\nmov.b64 $0, {l, h};\n}",
            "=l,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _key_low(key, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(key, loc=loc, ip=ip)],
            "{\n.reg .b32 hi;\nmov.b64 {$0, hi}, $1;\n}",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_b64(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _lv(value, loc=loc, ip=ip)],
        "st.shared.b64 [$0], $1;",
        "r,l,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_b64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_iv(addr, loc=loc, ip=ip)],
            "ld.shared.b64 $0, [$1];",
            "=l,r,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _publish_key(addr, key, multicast: bool, *, loc=None, ip=None):
    """Store the 8-byte ``key`` at ``addr``: the same offset of every rank's exchange through the
    multicast mapping, or the local exchange."""
    store = (
        "multimem.st.relaxed.sys.global.v2.f32 [$0], {lo, hi};"
        if multicast
        else "st.global.v2.b32 [$0], {lo, hi};"
    )
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _lv(key, loc=loc, ip=ip)],
        "{\n.reg .b32 lo, hi;\nmov.b64 {lo, hi}, $1;\n" + store + "\n}",
        "l,l,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _chunk_ready(w0, w1, w2, w3, second, *, loc=None, ip=None):
    """1 if the 16-byte chunk's first key and (where ``second``) its second key are published."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3, second)],
            "{\n.reg .pred p, q;\n"
            "setp.ne.b32 p, $1, 0;\nsetp.ne.and.b32 p, $2, 0, p;\n"
            "setp.ne.b32 q, $3, 0;\nsetp.ne.and.b32 q, $4, 0, q;\n"
            "setp.eq.or.s32 q, $5, 0, q;\nand.pred p, p, q;\nselp.u32 $0, 1, 0, p;\n}",
            "=r,r,r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class DSparkMarkovGreedyKernel:
    """One launch of the chain for ``width`` request slots (a power of two up to 16, the batch
    padded), ``steps`` block positions, ``tp`` ranks of ``ctas`` CTAs and Markov rank
    ``rank_dim``. See the module docstring."""

    def __init__(
        self,
        width: int,
        steps: int,
        tp: int,
        ctas: int,
        rank_dim: int,
        multicast: bool,
        anchor64: bool,
        use_pdl: bool,
        staged: bool = False,
        gate: bool = False,
    ):
        if width not in (1, 2, 4, 8, 16) or not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"unsupported width {width} / steps {steps}")
        if rank_dim % 16 or rank_dim > 256:
            raise ValueError(f"unsupported Markov rank {rank_dim}")
        self.width = width
        self.steps = steps
        self.tp = tp
        self.ctas = ctas
        self.rank_dim = rank_dim
        self.multicast = multicast
        self.anchor64 = anchor64
        self.use_pdl = use_pdl
        self.staged = staged
        self.gate = gate
        self.ntiles = (width + 7) // 8
        self.ksteps = rank_dim // 16
        # Shared-memory row pitch of the staged markov_w2 rows: 16 bytes of padding put the eight
        # rows of an ldmatrix 8x8 tile in distinct banks.
        self.pitch = rank_dim * 2 + 16
        self.row_chunks = rank_dim * 2 // 16
        self.entries = tp * ctas
        self.keys = self.entries * width
        self.chunks = (self.keys + 1) // 2
        self.per_thread = (self.chunks + THREADS - 1) // THREADS
        if self.per_thread > 31:
            raise ValueError(f"{self.keys} keys per position exceed the poll capacity")
        self.set_bytes = set_bytes(self.entries)

    @cute.jit
    def __call__(
        self,
        w2: Int64,
        w1: Int64,
        base: Int64,
        stride_req: Int32,
        stride_step: Int32,
        anchors: Int64,
        tokens: Int64,
        stride_tok: Int32,
        state: Int64,
        box_uc: Int64,
        box_mc: Int64,
        num_reqs: Int32,
        shard: Int32,
        vocab: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            stage: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, ROWS * self.pitch], 128]
            # Per position: this CTA's key per (warp, request), every rank's per (warp, request),
            # and the winning token per request.
            best: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, WARPS * MAX_WIDTH], 16]
            polled: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, WARPS * MAX_WIDTH], 16]
            winners: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, MAX_WIDTH], 16]
            # ``staged``: the requests' markov_w1 rows of the position.
            w1s: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, MAX_WIDTH * self.pitch], 128]

        self.shared_storage = SharedStorage
        self.kernel(
            w2,
            w1,
            base,
            stride_req,
            stride_step,
            anchors,
            tokens,
            stride_tok,
            state,
            box_uc,
            box_mc,
            num_reqs,
            shard,
            vocab,
            rank,
        ).launch(
            grid=(self.ctas, 1, 1),
            block=(THREADS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        w2: Int64,
        w1: Int64,
        base: Int64,
        stride_req: Int32,
        stride_step: Int32,
        anchors: Int64,
        tokens: Int64,
        stride_tok: Int32,
        state: Int64,
        box_uc: Int64,
        box_mc: Int64,
        num_reqs: Int32,
        shard: Int32,
        vocab: Int32,
        rank: Int32,
    ):
        width, steps = self.width, self.steps
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        g = lane // 4
        q = lane % 4
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        stage = storage.stage.data_ptr().toint()
        best = storage.best.data_ptr().toint()
        polled = storage.polled.data_ptr().toint()
        winners = storage.winners.data_ptr().toint()
        w1s = storage.w1s.data_ptr().toint()
        first = cta * ROWS
        row_bytes = self.rank_dim * 2

        # Static weights before the wait: this CTA's markov_w2 rows.
        for it in cutlass.range_constexpr(ROWS * self.row_chunks // THREADS):
            c = tid + it * THREADS
            r = c // self.row_chunks
            j = c % self.row_chunks
            if first + r < shard:
                _cp_async_evict_first(
                    stage + r * self.pitch + j * 16,
                    w2 + Int64(first + r) * row_bytes + Int64(j * 16),
                )
        _cp_async_commit()
        if cutlass.const_expr(self.staged):
            # Rows of requests without a valid previous token stay zero.
            for it in cutlass.range_constexpr(
                (MAX_WIDTH * self.pitch // 16 + THREADS - 1) // THREADS
            ):
                c = tid + it * THREADS
                if c < MAX_WIDTH * self.pitch // 16:
                    _st_shared_v4(w1s + c * 16, Int32(0), Int32(0), Int32(0), Int32(0))

        # Launch ordinal and the bytes each set's last launch used. The previous launch has
        # completed: it never triggers its dependents early.
        st = _ld_volatile_v4(state)
        launch = st[0]
        cur_set = launch % NUM_SETS
        clear_set = (launch + 2) % NUM_SETS
        clear_bytes = st[1]
        if clear_set == 1:
            clear_bytes = st[2]
        if clear_set == 2:
            clear_bytes = st[3]
        clear_base = box_uc + Int64(clear_set) * self.set_bytes
        for v in cutlass.range(
            cta * THREADS + tid, clear_bytes // 16, self.ctas * THREADS, unroll=1
        ):
            _st_zero_v4(clear_base + Int64(v) * 16)

        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()

        # Lane (g, q): B fragments of request 8 t + g, accumulators of requests 8 t + 2 q + {0, 1}
        # at rows (g, g + 8) of its warp's tile.
        # Each request's previous token lives in ``winners``: the anchor (-1 for padding requests),
        # then each position's winner.
        anchor_bytes = 8 if self.anchor64 else 4
        if tid < width:
            ok = Int32(0)
            if tid < num_reqs:
                ok = Int32(1)
            _st_shared_b32(winners + tid * 4, _ld_anchor(anchors + Int64(tid) * anchor_bytes, ok))
        rows = [first + warp * 16 + g, first + warp * 16 + g + 8]
        row_ok = []
        for r in cutlass.range_constexpr(2):
            ok = Int32(0)
            if rows[r] < shard:
                ok = Int32(1)
            row_ok.append(ok)
        load_ok = []
        for t in cutlass.range_constexpr(self.ntiles):
            for r in cutlass.range_constexpr(2):
                for n in cutlass.range_constexpr(2):
                    ok = Int32(0)
                    if 8 * t + 2 * q + n < num_reqs:
                        ok = row_ok[r]
                    load_ok.append(ok)

        base_cur = self.load_base(base, stride_req, stride_step, rows, q, load_ok, 0)
        _cp_async_wait_all()
        cute.arch.barrier()

        tile = (
            stage + (warp * 16 + lane % 8 + 8 * ((lane // 8) % 2)) * self.pitch + 16 * (lane // 16)
        )
        index0 = rank * shard
        own = rank * self.ctas + cta
        pos_bytes = self.keys * 8
        set_base_uc = box_uc + Int64(cur_set) * self.set_bytes
        set_base_mc = box_mc + Int64(cur_set) * self.set_bytes
        for i in cutlass.range_constexpr(steps):
            # Bias of the warp's 16 rows for every request tile.
            accs = [[Float32(0.0)] * 4 for _ in range(self.ntiles)]
            if cutlass.const_expr(self.staged):
                # Warp r stages request r's markov_w1 row; B fragments by ldmatrix (matrix j of
                # x4 load u: k-step 2 u + j // 2, half j % 2).
                if warp < width:
                    p = _ld_shared_b32(winners + warp * 4)
                    valid = Int32(0)
                    if warp < num_reqs:
                        if p >= 0:
                            if p < vocab:
                                valid = Int32(1)
                    if lane < row_bytes // 16:
                        words = _ld_nc_v4_masked(
                            w1 + Int64(p * valid) * row_bytes + Int64(16 * lane), Int32(0) - valid
                        )
                        _st_shared_v4(
                            w1s + warp * self.pitch + 16 * lane,
                            words[0],
                            words[1],
                            words[2],
                            words[3],
                        )
                cute.arch.barrier()
                for u in cutlass.range_constexpr(self.ksteps // 2):
                    a0 = _ldsm_x4(tile + 64 * u)
                    a1 = _ldsm_x4(tile + 64 * u + 32)
                    for t in cutlass.range_constexpr(self.ntiles):
                        frag = _ldsm_x4(
                            w1s + (8 * t + lane % 8) * self.pitch + 16 * (lane // 8) + 64 * u
                        )
                        accs[t] = list(_mma_bf16(a0, frag[0], frag[1], accs[t]))
                        accs[t] = list(_mma_bf16(a1, frag[2], frag[3], accs[t]))
            else:
                srcs = []
                masks = []
                for t in cutlass.range_constexpr(self.ntiles):
                    p = _ld_shared_b32(winners + ((8 * t + g) % width) * 4)
                    valid = Int32(0)
                    if 8 * t + g < num_reqs:
                        if p >= 0:
                            if p < vocab:
                                valid = Int32(1)
                    srcs.append(w1 + Int64(p * valid) * row_bytes + Int64(4 * q))
                    masks.append(Int32(0) - valid)
                for s in cutlass.range_constexpr(self.ksteps):
                    a = _ldsm_x4(tile + 32 * s)
                    for t in cutlass.range_constexpr(self.ntiles):
                        b0 = _ld_nc_b32(srcs[t] + Int64(32 * s)) & masks[t]
                        b1 = _ld_nc_b32(srcs[t] + Int64(32 * s + 16)) & masks[t]
                        accs[t] = list(_mma_bf16(a, b0, b1, accs[t]))

            # Keys of this CTA per request: lane max, then warp max, then CTA max.
            for t in cutlass.range_constexpr(self.ntiles):
                for n in cutlass.range_constexpr(2):
                    key = Int64(0)
                    for r in cutlass.range_constexpr(2):
                        k = _logit_key(
                            accs[t][2 * r + n],
                            base_cur[(t * 2 + r) * 2 + n],
                            index0 + rows[r],
                            row_ok[r],
                        )
                        key = _max_u64(key, k)
                    for off in (4, 8, 16):
                        key = _shuffle_max_u64(key, off)
                    if g == 0:
                        _st_shared_b64(best + (warp * MAX_WIDTH + 8 * t + 2 * q + n) * 8, key)
            cute.arch.barrier()
            pos_off = Int64(i * pos_bytes)
            if tid < width:
                mine = _ld_shared_b64(best + tid * 8)
                for w in cutlass.range_constexpr(1, WARPS):
                    mine = _max_u64(mine, _ld_shared_b64(best + (w * MAX_WIDTH + tid) * 8))
                _publish_key(
                    set_base_mc + pos_off + Int64((own * width + tid) * 8), mine, self.multicast
                )

            # The next position's base logits travel while the keys do.
            base_nxt = base_cur
            if cutlass.const_expr(i + 1 < steps):
                base_nxt = self.load_base(base, stride_req, stride_step, rows, q, load_ok, i + 1)

            # Every rank's keys of this position.
            region = set_base_uc + pos_off
            if cutlass.const_expr(self.gate):
                if tid < self.tp:
                    gc = ((tid + 1) * self.ctas - 1) * width // 2
                    second = Int32(1)
                    if 2 * gc + 1 >= self.keys:
                        second = Int32(0)
                    gw = _ld_volatile_v4(region + Int64(gc) * 16)
                    while _chunk_ready(gw[0], gw[1], gw[2], gw[3], second) == 0:
                        gw = _ld_volatile_v4(region + Int64(gc) * 16)
                cute.arch.barrier()
            got = cute.make_rmem_tensor((self.per_thread * 4,), Int32)
            pending = Int32(0)
            for j in cutlass.range_constexpr(self.per_thread):
                if tid + j * THREADS < self.chunks:
                    pending = pending | Int32(1 << j)
            while pending != 0:
                for j in cutlass.range_constexpr(self.per_thread):
                    # Threads past the end re-load the last chunk.
                    c = tid + j * THREADS
                    if c > self.chunks - 1:
                        c = Int32(self.chunks - 1)
                    w = _ld_volatile_v4(region + Int64(c) * 16)
                    for e in cutlass.range_constexpr(4):
                        got[j * 4 + e] = w[e]
                    second = Int32(1)
                    if 2 * c + 1 >= self.keys:
                        second = Int32(0)
                    ready = _chunk_ready(w[0], w[1], w[2], w[3], second)
                    pending = pending & (Int32(-1) ^ (Int32(1 << j) * ready))
            lo_key = Int64(0)
            hi_key = Int64(0)
            for j in cutlass.range_constexpr(self.per_thread):
                c = tid + j * THREADS
                ok0 = Int32(0)
                if c < self.chunks:
                    ok0 = Int32(1)
                ok1 = Int32(0)
                if 2 * c + 1 < self.keys:
                    ok1 = Int32(1)
                lo_key = _max_u64(lo_key, _pack_key(got[j * 4], got[j * 4 + 1], ok0))
                hi_key = _max_u64(hi_key, _pack_key(got[j * 4 + 2], got[j * 4 + 3], ok1))
            # Thread tid's chunks hold requests (2 tid) % width and (2 tid + 1) % width.
            if cutlass.const_expr(width == 1):
                lo_key = _max_u64(lo_key, hi_key)
                for off in (1, 2, 4, 8, 16):
                    lo_key = _shuffle_max_u64(lo_key, off)
                if lane == 0:
                    _st_shared_b64(polled + warp * MAX_WIDTH * 8, lo_key)
            else:
                for off in (1, 2, 4, 8, 16):
                    if cutlass.const_expr(off >= width // 2):
                        lo_key = _shuffle_max_u64(lo_key, off)
                        hi_key = _shuffle_max_u64(hi_key, off)
                if lane < width // 2:
                    _st_shared_b64(polled + (warp * MAX_WIDTH + 2 * lane) * 8, lo_key)
                    _st_shared_b64(polled + (warp * MAX_WIDTH + 2 * lane + 1) * 8, hi_key)
            cute.arch.barrier()
            if tid < width:
                final = _ld_shared_b64(polled + tid * 8)
                for w in cutlass.range_constexpr(1, WARPS):
                    final = _max_u64(final, _ld_shared_b64(polled + (w * MAX_WIDTH + tid) * 8))
                token = Int32(-1) ^ _key_low(final)
                _st_shared_b32(winners + tid * 4, token)
                if cta == 0:
                    if tid < num_reqs:
                        _st_global_b32(tokens + (Int64(tid) * stride_tok + i) * 4, token)
            cute.arch.barrier()
            base_cur = base_nxt

        if cta == 0:
            if tid == 0:
                _st_global_b32(state, launch + 1)
                _st_global_b32(state + Int64(4 + 4 * cur_set), Int32(steps * self.keys * 8))

    def load_base(self, base, stride_req, stride_step, rows, q, load_ok, i: int):
        """The lane's base logits of position ``i`` (BF16 bits), ``[tile][row][request]``
        (traced inline into the kernel)."""
        vals = []
        for t in range(self.ntiles):
            for r in range(2):
                for n in range(2):
                    req = 8 * t + 2 * q + n
                    addr = (
                        base
                        + (Int64(req * load_ok[(t * 2 + r) * 2 + n]) * stride_req) * 2
                        + Int64(i * stride_step) * 2
                        + Int64(rows[r]) * 2
                    )
                    vals.append(_ld_bf16_bits(addr, load_ok[(t * 2 + r) * 2 + n]))
        return vals


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@functools.cache
def _compiled(width, steps, tp, ctas, rank_dim, multicast, anchor64, use_pdl, staged, gate):
    kernel = DSparkMarkovGreedyKernel(
        width, steps, tp, ctas, rank_dim, multicast, anchor64, use_pdl, staged, gate
    )
    a = Int64(1 << 20)
    i = Int32(1)
    return cute.compile(kernel, a, a, a, i, i, a, a, i, a, a, a, i, i, i, i, _stream())


def smem_bytes(rank_dim: int) -> int:
    """Shared memory one CTA needs (an upper bound)."""
    pitch = rank_dim * 2 + 16
    return (ROWS + MAX_WIDTH) * pitch + 2 * WARPS * MAX_WIDTH * 8 + MAX_WIDTH * 4 + 1024


def run_dspark_markov_greedy(
    local_block: torch.Tensor,
    anchors: torch.Tensor,
    markov_w1: torch.Tensor,
    w2_shard: torch.Tensor,
    tokens: torch.Tensor,
    *,
    vocab: int,
    tp: int,
    rank: int,
    state: torch.Tensor,
    box_uc: int,
    box_mc: int,
    multicast: bool,
    use_pdl: bool = True,
    staged: Optional[bool] = None,
    gate: Optional[bool] = None,
) -> None:
    """Launch the chain on the current stream; writes ``tokens`` (int32 ``[B, K]``).

    ``local_block``: ``[B, K, shard]`` BF16 base logits with a unit vocab stride;
    ``anchors``: ``[B]`` int32 / int64; ``markov_w1``: ``[vocab, R]`` BF16 contiguous;
    ``w2_shard``: this rank's ``[shard, R]`` rows of ``markov_w2`` (contiguous); ``state``: the
    rank's int32 ``[STATE_WORDS]`` (zero-initialized); ``box_uc`` / ``box_mc``: the local and
    multicast addresses of the exchange (``NUM_SETS`` sets of :func:`set_bytes`, zero-initialized).
    Every rank makes the same sequence of calls. ``staged`` / ``gate`` override :data:`OPTIONS`."""
    num_reqs, steps, shard = local_block.shape
    rank_dim = markov_w1.shape[1]
    width = width_for(num_reqs)
    default_staged, default_gate = OPTIONS[width]
    _compiled(
        width,
        steps,
        tp,
        ctas_for(shard),
        rank_dim,
        multicast,
        anchors.dtype == torch.int64,
        use_pdl,
        default_staged if staged is None else staged,
        default_gate if gate is None else gate,
    )(
        Int64(w2_shard.data_ptr()),
        Int64(markov_w1.data_ptr()),
        Int64(local_block.data_ptr()),
        Int32(local_block.stride(0)),
        Int32(local_block.stride(1)),
        Int64(anchors.data_ptr()),
        Int64(tokens.data_ptr()),
        Int32(tokens.stride(0)),
        Int64(state.data_ptr()),
        Int64(box_uc),
        Int64(box_mc),
        Int32(num_reqs),
        Int32(shard),
        Int32(vocab),
        Int32(rank),
        _stream(),
    )


__all__ = [
    "MAX_STEPS",
    "MAX_WIDTH",
    "NUM_SETS",
    "OPTIONS",
    "ROWS",
    "STATE_WORDS",
    "DSparkMarkovGreedyKernel",
    "ctas_for",
    "run_dspark_markov_greedy",
    "set_bytes",
    "smem_bytes",
    "width_for",
]

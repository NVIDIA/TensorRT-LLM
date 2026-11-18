# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused expert finalize + all-reduce as a two-phase transport (TP8 port, w256).

Ported from Iman Tabrizian's TEP16 target (user/itabrizian/musespark-tep16-feat @ 23f5e00ef6,
``muse_spark_open_816b_a42b_nvfp4__sm_107__tep16/builtin/musespark_open_tep16_finalize.py``, design B(ii)), whose
kernels already take ``ranks`` 8 (one CTA's 512 columns per owner). Changes: the package path (this sm107 TEP target's
``kernels.persistent_experts.rows``), ``MAX_TOKENS`` 256 (17..32 requests at K = 7; ``CMAX`` 128 clusters, so at most
two
tokens per cluster), and ``TwoPhaseWorkspace`` (the TP8 fabric workspace, two builds: the cluster-8 order on 1..16-token
calls and the flat order above, the TP8 one-shot transport's builds, ``finalize_allreduce_transport``), which
``FinalizeTransportWorkspace`` runs under ``configure(tep8_two_phase=True)``. Original notes follow.

klab/moe: the fused expert finalize + all-reduce at TP16 as a two-phase transport (design B(ii)).

Arithmetic: the TP8 finalize transport's (overlays/tep8x9, the built-in cluster-8 finalize statements): per token
a cluster of 8 CTAs x 64 threads, each thread one 16-byte column vector of all 8 slots; each selected expert row is
normalized on its own (weight-first RMSNorm, zero-centered gains), weighted and added in FP32, and this rank's
partial is rounded to BF16 with ``-0`` cleared. The ranks' partials are then summed in FP32 in rank order from +0
and rounded to BF16 once: the one-shot transport's result, bit for bit, by a different route:

* **Phase 1 (reduce-scatter by direct publication).** Rank ``j`` owns the 4096 / R columns of shard ``j`` (R = 16:
  256 columns = one warp of one cluster CTA). Every thread stores its 16-byte partial into slot ``(buffer, source =
  this rank, token, column)`` of the owner's inbox (one unicast ``st.relaxed.sys``; the owner keeps its own partial
  in registers). The owner's threads poll their R - 1 source slots with all loads in flight (stale slots only, one
  predicated load per slot), add the R partials in rank order and write the sentinel back.
* **Phase 2 (all-gather by multicast).** The owner's thread stores the reduced 16 bytes once to the multicast address
  of slot ``(buffer, token, column)`` of every rank's gather buffer (``multicast``), or with R - 1 unicast stores
  (``unicast``: the single-GPU emulation); every other rank's thread with that column polls its slot, writes the
  sentinel back and stores the output.

Protocol: the one-shot transport's Lamport scheme per slot (one writer and one reader per call; the sentinel
``0x80000000``; published words never equal it: partials have ``-0`` halves cleared, and a sum of R BF16 values none
of which is ``-0`` is never ``-0``, so the reduced words never equal it either), its buffer parity (a free-running
arrival counter advanced by ``GRID_STEP`` per call), no system-scope fence, PDL. Reuse: the inbox slot ``(b, s)`` of
call ``n`` is written again by ``s`` at call ``n + 2`` only after ``s`` completed call ``n + 1``, which needed the
owner's call ``n + 1`` gather words, published after the owner completed call ``n`` (its reset included); a gather slot
``b`` is written again by its owner at call ``n + 2`` only after the owner's phase 1 of call ``n + 2``, which needs this
rank's call ``n + 2`` partial, published after this rank completed call ``n + 1``. Progress: every thread publishes its
partial before it polls anything, owners publish their sums before they poll anything else, and a grid of
``tokens x 8`` CTAs must be resident at once on each rank (``MAX_TOKENS``; the caller splits larger calls).
"""

from __future__ import annotations

import functools

import torch

PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep"

HIDDEN = 4096
TOP_K = 8
CLUSTER = 8
VECTOR = 8
THREADS = HIDDEN // (CLUSTER * VECTOR)  # 64
WARPS = THREADS // 32
MAX_RANKS = 16
SENTINEL = 0x80000000
BUFFERS = 2
#: Tokens per launch (grid tokens x 8 CTAs, resident at once on every rank).
MAX_TOKENS = 256  # w256 (TP16: 384)
#: The token-loop build (``FinalizeTP16LoopKernel``): at most CMAX clusters (CMAX x 8 CTAs resident at once on a
#: GPU), each finalizing up to NMAX tokens (cluster c: tokens c, c + C, c + 2C, ...).
CMAX = 128
NMAX = (MAX_TOKENS + CMAX - 1) // CMAX
GRID_SHIFT = max(7, (MAX_TOKENS * CLUSTER - 1).bit_length())
GRID_STEP = 1 << GRID_SHIFT
assert GRID_STEP >= MAX_TOKENS * CLUSTER
CONTROL_WORDS = 4


def inbox_slab_bytes(ranks: int) -> int:
    """One source's slab of a rank's inbox: MAX_TOKENS rows of the owner's shard."""
    return MAX_TOKENS * (HIDDEN // ranks) * 2


def inbox_bytes(ranks: int) -> int:
    return BUFFERS * ranks * inbox_slab_bytes(ranks)


def gather_bytes() -> int:
    return BUFFERS * MAX_TOKENS * HIDDEN * 2


import importlib  # noqa: E402

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import cutlass.utils as cutlass_utils  # noqa: E402
from cutlass import BFloat16, Float32, Int32, Int64, Uint32  # noqa: E402
from cutlass._mlir import ir  # noqa: E402
from cutlass._mlir.dialects import llvm  # noqa: E402
from cutlass.cute.runtime import make_ptr  # noqa: E402
from cutlass.cutlass_dsl import T, dsl_user_op  # noqa: E402

from tensorrt_llm._torch.models.musespark_open_finalize_ar import (  # noqa: E402
    _FOUR_WORDS,
    _asm,
    _i64,
    _load_control,
    _load_input_x4,
    _pack_bf16x2,
    _pack_publishable_bf16x2,
    _split_words,
    _store_x4,
    _u32,
    _unpack_bf16x2,
    _zero_unless,
)

_expert_rows = importlib.import_module(PKG + ".kernels.persistent_experts.rows")


@dsl_user_op
def _poll_x4(address, enabled, o0, o1, o2, o3, *, loc=None, ip=None):
    """Relaxed system-scope 16-byte load when ``enabled``; else the given words, no access."""
    result = _asm(
        ir.Type.parse(_FOUR_WORDS),
        [_i64(address, loc, ip), _u32(enabled, loc, ip)]
        + [_u32(w, loc, ip) for w in (o0, o1, o2, o3)],
        "{\n.reg .pred p;\n.reg .b32 a, b, c, d;\n"
        "mov.b32 a, $6;\nmov.b32 b, $7;\nmov.b32 c, $8;\nmov.b32 d, $9;\n"
        "setp.ne.u32 p, $5, 0;\n"
        "@p ld.relaxed.sys.global.v4.b32 {a, b, c, d}, [$4];\n"
        "mov.b32 $0, a;\nmov.b32 $1, b;\nmov.b32 $2, c;\nmov.b32 $3, d;\n}",
        "=r,=r,=r,=r,l,r,r,r,r,r",
        loc,
        ip,
    )
    return _split_words(result, loc, ip)


@dsl_user_op
def _stale_bit(w0, w1, w2, w3, bit, *, loc=None, ip=None):
    """``bit`` while any of the four words is the sentinel, else 0 (no branch)."""
    result = _asm(
        T.i32(),
        [_u32(w, loc, ip) for w in (w0, w1, w2, w3, bit)],
        "{\n.reg .b32 a, b, c, d;\n"
        "xor.b32 a, $1, 0x80000000;\nxor.b32 b, $2, 0x80000000;\n"
        "xor.b32 c, $3, 0x80000000;\nxor.b32 d, $4, 0x80000000;\n"
        "min.u32 a, a, b;\nmin.u32 c, c, d;\nmin.u32 a, a, c;\n"
        "set.eq.u32.u32 a, a, 0;\nand.b32 $0, a, $5;\n}",
        "=r,r,r,r,r,r",
        loc,
        ip,
        side_effects=False,
    )
    return Uint32(result)


@dsl_user_op
def _load_s32(address, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_i64(address, loc, ip)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip)
    )


@dsl_user_op
def _load_f32(address, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_i64(address, loc, ip)], "ld.global.f32 $0, [$1];", "=f,l", loc, ip)
    )


@dsl_user_op
def _load_row_x4(address, enabled, *, loc=None, ip=None):
    """Coherent 16-byte row load; zeros without touching memory when disabled."""
    result = _asm(
        ir.Type.parse(_FOUR_WORDS),
        [_i64(address, loc, ip), _u32(enabled, loc, ip)],
        "{\n.reg .pred p;\n.reg .b32 a, b, c, d;\n"
        "mov.b32 a, 0;\nmov.b32 b, 0;\nmov.b32 c, 0;\nmov.b32 d, 0;\n"
        "setp.ne.u32 p, $5, 0;\n"
        "@p ld.global.v4.b32 {a, b, c, d}, [$4];\n"
        "mov.b32 $0, a;\nmov.b32 $1, b;\nmov.b32 $2, c;\nmov.b32 $3, d;\n}",
        "=r,=r,=r,=r,l,r",
        loc,
        ip,
    )
    return _split_words(result, loc, ip)


@dsl_user_op
def _reset_x4(address, enabled, *, loc=None, ip=None):
    """The sentinel into a consumed slot when ``enabled``."""
    _asm(
        None,
        [_i64(address, loc, ip), _u32(enabled, loc, ip)],
        "{\n.reg .pred p;\n.reg .b32 s;\nmov.b32 s, 0x80000000;\n"
        "setp.ne.u32 p, $1, 0;\n"
        "@p st.global.v4.b32 [$0], {s, s, s, s};\n}",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def _store_peer_x4_if(address, enabled, w0, w1, w2, w3, *, loc=None, ip=None):
    """Unicast publication (``st.relaxed.sys.v4``) when ``enabled``."""
    _asm(
        None,
        [_i64(address, loc, ip), _u32(enabled, loc, ip)]
        + [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\n"
        "@p st.relaxed.sys.global.v4.b32 [$0], {$2, $3, $4, $5};\n}",
        "l,r,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def _store_multicast_x4(address, w0, w1, w2, w3, *, loc=None, ip=None):
    """16-byte store to a multicast address: the fabric writes it to every member."""
    _asm(
        None,
        [_i64(address, loc, ip)] + [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def _red_add(address, amount, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(address, loc, ip), _u32(amount, loc, ip)],
        "red.relaxed.gpu.global.add.u32 [$0], $1;",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def _float_bits(value, *, loc=None, ip=None):
    bits = llvm.inline_asm(
        T.i32(),
        [Float32(value).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=r,f",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int32(bits)


class FinalizeTP16Kernel:
    """Cluster-8 finalize arithmetic (tep8x9's ``warp`` reduction, ``red`` arrival) on the two-phase transport."""

    def __init__(
        self,
        ranks: int = 16,
        multicast: bool = True,
        pdl: bool = True,
        flat: bool = False,
        trigger: str = "late",
        debug: bool = False,
    ):
        assert ranks in (8, 16), ranks
        self.ranks = ranks
        self.shard = HIDDEN // ranks  # columns per owner: 512 (one CTA) or 256 (one warp)
        self.multicast = multicast
        self.pdl = pdl
        self.flat_sum = flat
        self.entry_trigger = trigger == "entry"
        self.debug = debug
        self.cluster = CLUSTER
        self.threads = THREADS
        self.warps = WARPS
        self.slab = inbox_slab_bytes(ranks)

    def select_row(self, indices, rows, token, k):
        index = _load_s32(indices.toint() + (Int64(token) * TOP_K + k) * 4)
        return index, Uint32(_expert_rows.routed_row_valid(index, rows))

    @cute.jit
    def __call__(
        self,
        gemm: cute.Pointer,
        indices: cute.Pointer,
        weights: cute.Pointer,
        norm: cute.Pointer,
        out: cute.Pointer,
        control: cute.Pointer,
        inbox: cute.Pointer,
        gather: cute.Pointer,
        gather_mc: cute.Pointer,
        dbg: cute.Pointer,
        pi0: cute.Pointer,
        pi1: cute.Pointer,
        pi2: cute.Pointer,
        pi3: cute.Pointer,
        pi4: cute.Pointer,
        pi5: cute.Pointer,
        pi6: cute.Pointer,
        pi7: cute.Pointer,
        pi8: cute.Pointer,
        pi9: cute.Pointer,
        pi10: cute.Pointer,
        pi11: cute.Pointer,
        pi12: cute.Pointer,
        pi13: cute.Pointer,
        pi14: cute.Pointer,
        pi15: cute.Pointer,
        pg0: cute.Pointer,
        pg1: cute.Pointer,
        pg2: cute.Pointer,
        pg3: cute.Pointer,
        pg4: cute.Pointer,
        pg5: cute.Pointer,
        pg6: cute.Pointer,
        pg7: cute.Pointer,
        pg8: cute.Pointer,
        pg9: cute.Pointer,
        pg10: cute.Pointer,
        pg11: cute.Pointer,
        pg12: cute.Pointer,
        pg13: cute.Pointer,
        pg14: cute.Pointer,
        pg15: cute.Pointer,
        rows: Int32,
        row_stride: Int32,
        tokens: Int32,
        rank: Int32,
        eps: Float32,
        dry: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            gemm,
            indices,
            weights,
            norm,
            out,
            control,
            inbox,
            gather,
            gather_mc,
            dbg,
            pi0,
            pi1,
            pi2,
            pi3,
            pi4,
            pi5,
            pi6,
            pi7,
            pi8,
            pi9,
            pi10,
            pi11,
            pi12,
            pi13,
            pi14,
            pi15,
            pg0,
            pg1,
            pg2,
            pg3,
            pg4,
            pg5,
            pg6,
            pg7,
            pg8,
            pg9,
            pg10,
            pg11,
            pg12,
            pg13,
            pg14,
            pg15,
            rows,
            row_stride,
            tokens,
            rank,
            eps,
            dry,
        ).launch(
            grid=[tokens * self.cluster, 1, 1],
            block=[self.threads, 1, 1],
            cluster=[self.cluster, 1, 1],
            smem=2048,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        gemm: cute.Pointer,
        indices: cute.Pointer,
        weights: cute.Pointer,
        norm: cute.Pointer,
        out: cute.Pointer,
        control: cute.Pointer,
        inbox: cute.Pointer,
        gather: cute.Pointer,
        gather_mc: cute.Pointer,
        dbg: cute.Pointer,
        pi0: cute.Pointer,
        pi1: cute.Pointer,
        pi2: cute.Pointer,
        pi3: cute.Pointer,
        pi4: cute.Pointer,
        pi5: cute.Pointer,
        pi6: cute.Pointer,
        pi7: cute.Pointer,
        pi8: cute.Pointer,
        pi9: cute.Pointer,
        pi10: cute.Pointer,
        pi11: cute.Pointer,
        pi12: cute.Pointer,
        pi13: cute.Pointer,
        pi14: cute.Pointer,
        pi15: cute.Pointer,
        pg0: cute.Pointer,
        pg1: cute.Pointer,
        pg2: cute.Pointer,
        pg3: cute.Pointer,
        pg4: cute.Pointer,
        pg5: cute.Pointer,
        pg6: cute.Pointer,
        pg7: cute.Pointer,
        pg8: cute.Pointer,
        pg9: cute.Pointer,
        pg10: cute.Pointer,
        pg11: cute.Pointer,
        pg12: cute.Pointer,
        pg13: cute.Pointer,
        pg14: cute.Pointer,
        pg15: cute.Pointer,
        rows: Int32,
        row_stride: Int32,
        tokens: Int32,
        rank: Int32,
        eps: Float32,
        dry: Int32,
    ):
        peer_inbox = (
            pi0,
            pi1,
            pi2,
            pi3,
            pi4,
            pi5,
            pi6,
            pi7,
            pi8,
            pi9,
            pi10,
            pi11,
            pi12,
            pi13,
            pi14,
            pi15,
        )
        peer_gather = (
            pg0,
            pg1,
            pg2,
            pg3,
            pg4,
            pg5,
            pg6,
            pg7,
            pg8,
            pg9,
            pg10,
            pg11,
            pg12,
            pg13,
            pg14,
            pg15,
        )
        tid = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        blocks = cute.arch.grid_dim()[0]
        token = block // self.cluster
        crank = block % self.cluster
        column = (crank * self.threads + tid) * VECTOR
        lane = tid % 32
        warp = tid // 32

        smem = cutlass_utils.SmemAllocator()
        snapshot = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=16)
        sums = smem.allocate_tensor(Float32, cute.make_layout((TOP_K,)), byte_alignment=16)
        recv = smem.allocate_tensor(
            Float32, cute.make_layout((CLUSTER * WARPS * TOP_K,)), byte_alignment=16
        )
        bar = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((1,)), byte_alignment=8)

        if dry == 0:
            # The receive mbarrier of the cluster's warp sums, visible cluster-wide before any peer sends.
            if tid == 0:
                cute.arch.mbarrier_init(bar.iterator, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar.iterator, self.warps * CLUSTER * TOP_K * 4
                )
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.entry_trigger):
                    cute.arch.griddepcontrol_launch_dependents()

            # ---- the cluster-8 finalize arithmetic (tep8x9, warp reduction) ----
            selection = Int64(token) * TOP_K
            gemm_address = gemm.toint()
            row_index = []
            row_weight = []
            row_words = []
            for k in cutlass.range_constexpr(TOP_K):
                index, valid = self.select_row(indices, rows, token, k)
                row_index.append(index)
                row_weight.append(_load_f32(weights.toint() + (selection + k) * 4))
                offset = _expert_rows.routed_row_offset(
                    Int64(index), Int64(row_stride), Int64(column)
                )
                row_words.append(_load_row_x4(gemm_address + offset * 2, valid))
            norm_words = _load_input_x4(norm.toint() + Int64(column) * 2, Uint32(1))

            control_address = control.toint()
            if tid == 0:
                counter = _load_control(control_address + 8)
                snapshot[0] = Int32((counter >> Uint32(GRID_SHIFT)) & Uint32(1))
                amount = Uint32(1)
                if block == 0:
                    amount = Uint32(GRID_STEP + 1) - Uint32(blocks)
                _red_add(control_address + 8, amount)

            gains = []
            for w in cutlass.range_constexpr(4):
                low, high = _unpack_bf16x2(norm_words[w])
                gains.append(low + Float32(1.0))
                gains.append(high + Float32(1.0))
            partials = []
            for k in cutlass.range_constexpr(TOP_K):
                square_sum = Float32(0.0)
                for w in cutlass.range_constexpr(4):
                    low, high = _unpack_bf16x2(row_words[k][w])
                    scaled_low = low * gains[2 * w]
                    scaled_high = high * gains[2 * w + 1]
                    square_sum = square_sum + scaled_low * scaled_low + scaled_high * scaled_high
                partials.append(cute.arch.warp_reduction_sum(square_sum))
            sent = []
            for k in cutlass.range_constexpr(TOP_K):
                sent.append(_float_bits(cute.arch.shuffle_sync(partials[k], 0)))
            if lane < self.cluster:
                row = recv.iterator + (crank * self.warps + warp) * TOP_K
                cute.arch.store_async_dsmem(
                    row.align(16), (sent[0], sent[1], sent[2], sent[3]), bar.iterator, lane
                )
                cute.arch.store_async_dsmem(
                    (row + 4).align(16), (sent[4], sent[5], sent[6], sent[7]), bar.iterator, lane
                )
            cute.arch.mbarrier_wait(bar.iterator, 0)
            if tid < TOP_K:
                total = Float32(0.0)
                if cutlass.const_expr(self.flat_sum):
                    for c in cutlass.range_constexpr(self.cluster):
                        for w in cutlass.range_constexpr(self.warps):
                            total = total + recv[(c * self.warps + w) * TOP_K + tid]
                else:
                    for c in cutlass.range_constexpr(self.cluster):
                        cta_total = Float32(0.0)
                        for w in cutlass.range_constexpr(self.warps):
                            cta_total = cta_total + recv[(c * self.warps + w) * TOP_K + tid]
                        total = total + cta_total
                sums[tid] = total
            cute.arch.barrier()
            result = [Float32(0.0)] * VECTOR
            for k in cutlass.range_constexpr(TOP_K):
                valid = Uint32(_expert_rows.routed_row_valid(row_index[k], rows))
                scale = row_weight[k] * cute.math.rsqrt(sums[k] / Float32(HIDDEN) + eps)
                scale = _zero_unless(scale, valid)
                for w in cutlass.range_constexpr(4):
                    low, high = _unpack_bf16x2(row_words[k][w])
                    result[2 * w] = result[2 * w] + scale * (low * gains[2 * w])
                    result[2 * w + 1] = result[2 * w + 1] + scale * (high * gains[2 * w + 1])
            own = []
            for w in cutlass.range_constexpr(4):
                own.append(_pack_publishable_bf16x2(result[2 * w], result[2 * w + 1]))
            if cutlass.const_expr(self.debug):
                _store_x4(
                    dbg.toint() + (Int64(token) * HIDDEN + Int64(column)) * 2,
                    own[0],
                    own[1],
                    own[2],
                    own[3],
                )

            # ---- phase 1: the partial to its shard's owner ----
            buffer = Int64(snapshot[0])
            owner = column // self.shard
            in_slot = (Int64(token) * self.shard + Int64(column - owner * self.shard)) * 2
            src_slab = (buffer * self.ranks + Int64(rank)) * self.slab
            for p in cutlass.range_constexpr(self.ranks):
                _store_peer_x4_if(
                    peer_inbox[p].toint() + src_slab + in_slot,
                    Uint32((owner == p) & (owner != rank)),
                    own[0],
                    own[1],
                    own[2],
                    own[3],
                )
            out_slot = (Int64(token) * HIDDEN + Int64(column)) * 2
            gather_slot = (buffer * MAX_TOKENS + Int64(token)) * (HIDDEN * 2) + Int64(column) * 2
            if owner == rank:
                # The owner: every source's partial of this column vector, in flight, stale slots re-read.
                values = cute.make_rmem_tensor(cute.make_layout((self.ranks, 4)), Uint32)
                for p in cutlass.range_constexpr(self.ranks):
                    values[p, 0] = own[0]
                    values[p, 1] = own[1]
                    values[p, 2] = own[2]
                    values[p, 3] = own[3]
                inbox_address = inbox.toint()
                stale = Uint32((1 << self.ranks) - 1) & ~(Uint32(1) << Uint32(rank))
                while stale != Uint32(0):
                    polled = stale
                    for p in cutlass.range_constexpr(self.ranks):
                        address = inbox_address + (buffer * self.ranks + p) * self.slab + in_slot
                        v0, v1, v2, v3 = _poll_x4(
                            address,
                            (polled >> Uint32(p)) & Uint32(1),
                            values[p, 0],
                            values[p, 1],
                            values[p, 2],
                            values[p, 3],
                        )
                        values[p, 0] = v0
                        values[p, 1] = v1
                        values[p, 2] = v2
                        values[p, 3] = v3
                    stale = Uint32(0)
                    for p in cutlass.range_constexpr(self.ranks):
                        stale = stale | _stale_bit(
                            values[p, 0], values[p, 1], values[p, 2], values[p, 3], Uint32(1 << p)
                        )
                for p in cutlass.range_constexpr(self.ranks):
                    _reset_x4(
                        inbox_address + (buffer * self.ranks + p) * self.slab + in_slot,
                        Uint32(rank != p),
                    )
                total_out = [Float32(0.0)] * VECTOR
                for p in cutlass.range_constexpr(self.ranks):
                    for w in cutlass.range_constexpr(4):
                        low, high = _unpack_bf16x2(values[p, w])
                        total_out[2 * w] = total_out[2 * w] + low
                        total_out[2 * w + 1] = total_out[2 * w + 1] + high
                packed = []
                for w in cutlass.range_constexpr(4):
                    packed.append(_pack_bf16x2(total_out[2 * w], total_out[2 * w + 1]))
                # ---- phase 2: the sum to every rank's gather slot ----
                if cutlass.const_expr(self.multicast):
                    _store_multicast_x4(
                        gather_mc.toint() + gather_slot, packed[0], packed[1], packed[2], packed[3]
                    )
                else:
                    for p in cutlass.range_constexpr(self.ranks):
                        _store_peer_x4_if(
                            peer_gather[p].toint() + gather_slot,
                            Uint32(p != rank),
                            packed[0],
                            packed[1],
                            packed[2],
                            packed[3],
                        )
                _store_x4(out.toint() + out_slot, packed[0], packed[1], packed[2], packed[3])
            else:
                # Everyone else: the owner's sum from this rank's gather slot.
                gather_address = gather.toint() + gather_slot
                g0 = Uint32(SENTINEL)
                g1 = Uint32(SENTINEL)
                g2 = Uint32(SENTINEL)
                g3 = Uint32(SENTINEL)
                pending = Uint32(1)
                while pending != Uint32(0):
                    g0, g1, g2, g3 = _poll_x4(gather_address, pending, g0, g1, g2, g3)
                    pending = _stale_bit(g0, g1, g2, g3, Uint32(1))
                _reset_x4(gather_address, Uint32(1))
                _store_x4(out.toint() + out_slot, g0, g1, g2, g3)

            if cutlass.const_expr(self.pdl and not self.entry_trigger):
                cute.arch.griddepcontrol_launch_dependents()


class FinalizeTP16LoopKernel(FinalizeTP16Kernel):
    """The same arithmetic and transport over a bounded grid: C = min(tokens, CMAX) clusters, NMAX token slots each.

    Order inside a cluster: the arithmetic and the phase-1 publication of every token of the cluster first (each
    token's cluster-wide slot norms on its own mbarrier, armed before the dependency wait), then the owner warps'
    phase 1 consumption and phase 2 publication for every token, else the other warps' gather polls. Owner warps
    never poll a gather slot and no warp polls before its cluster published all its partials, so the loop adds no
    wait cycle; the grid is bounded so that it is resident at once on every rank.
    """

    @cute.jit
    def __call__(
        self,
        gemm: cute.Pointer,
        indices: cute.Pointer,
        weights: cute.Pointer,
        norm: cute.Pointer,
        out: cute.Pointer,
        control: cute.Pointer,
        inbox: cute.Pointer,
        gather: cute.Pointer,
        gather_mc: cute.Pointer,
        dbg: cute.Pointer,
        pi0: cute.Pointer,
        pi1: cute.Pointer,
        pi2: cute.Pointer,
        pi3: cute.Pointer,
        pi4: cute.Pointer,
        pi5: cute.Pointer,
        pi6: cute.Pointer,
        pi7: cute.Pointer,
        pi8: cute.Pointer,
        pi9: cute.Pointer,
        pi10: cute.Pointer,
        pi11: cute.Pointer,
        pi12: cute.Pointer,
        pi13: cute.Pointer,
        pi14: cute.Pointer,
        pi15: cute.Pointer,
        pg0: cute.Pointer,
        pg1: cute.Pointer,
        pg2: cute.Pointer,
        pg3: cute.Pointer,
        pg4: cute.Pointer,
        pg5: cute.Pointer,
        pg6: cute.Pointer,
        pg7: cute.Pointer,
        pg8: cute.Pointer,
        pg9: cute.Pointer,
        pg10: cute.Pointer,
        pg11: cute.Pointer,
        pg12: cute.Pointer,
        pg13: cute.Pointer,
        pg14: cute.Pointer,
        pg15: cute.Pointer,
        rows: Int32,
        row_stride: Int32,
        tokens: Int32,
        rank: Int32,
        eps: Float32,
        dry: Int32,
        stream: cuda.CUstream,
    ):
        clusters = tokens
        if clusters > CMAX:
            clusters = Int32(CMAX)
        self.kernel(
            gemm,
            indices,
            weights,
            norm,
            out,
            control,
            inbox,
            gather,
            gather_mc,
            dbg,
            pi0,
            pi1,
            pi2,
            pi3,
            pi4,
            pi5,
            pi6,
            pi7,
            pi8,
            pi9,
            pi10,
            pi11,
            pi12,
            pi13,
            pi14,
            pi15,
            pg0,
            pg1,
            pg2,
            pg3,
            pg4,
            pg5,
            pg6,
            pg7,
            pg8,
            pg9,
            pg10,
            pg11,
            pg12,
            pg13,
            pg14,
            pg15,
            rows,
            row_stride,
            tokens,
            rank,
            eps,
            dry,
        ).launch(
            grid=[clusters * self.cluster, 1, 1],
            block=[self.threads, 1, 1],
            cluster=[self.cluster, 1, 1],
            smem=4096,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        gemm: cute.Pointer,
        indices: cute.Pointer,
        weights: cute.Pointer,
        norm: cute.Pointer,
        out: cute.Pointer,
        control: cute.Pointer,
        inbox: cute.Pointer,
        gather: cute.Pointer,
        gather_mc: cute.Pointer,
        dbg: cute.Pointer,
        pi0: cute.Pointer,
        pi1: cute.Pointer,
        pi2: cute.Pointer,
        pi3: cute.Pointer,
        pi4: cute.Pointer,
        pi5: cute.Pointer,
        pi6: cute.Pointer,
        pi7: cute.Pointer,
        pi8: cute.Pointer,
        pi9: cute.Pointer,
        pi10: cute.Pointer,
        pi11: cute.Pointer,
        pi12: cute.Pointer,
        pi13: cute.Pointer,
        pi14: cute.Pointer,
        pi15: cute.Pointer,
        pg0: cute.Pointer,
        pg1: cute.Pointer,
        pg2: cute.Pointer,
        pg3: cute.Pointer,
        pg4: cute.Pointer,
        pg5: cute.Pointer,
        pg6: cute.Pointer,
        pg7: cute.Pointer,
        pg8: cute.Pointer,
        pg9: cute.Pointer,
        pg10: cute.Pointer,
        pg11: cute.Pointer,
        pg12: cute.Pointer,
        pg13: cute.Pointer,
        pg14: cute.Pointer,
        pg15: cute.Pointer,
        rows: Int32,
        row_stride: Int32,
        tokens: Int32,
        rank: Int32,
        eps: Float32,
        dry: Int32,
    ):
        peer_inbox = (
            pi0,
            pi1,
            pi2,
            pi3,
            pi4,
            pi5,
            pi6,
            pi7,
            pi8,
            pi9,
            pi10,
            pi11,
            pi12,
            pi13,
            pi14,
            pi15,
        )
        peer_gather = (
            pg0,
            pg1,
            pg2,
            pg3,
            pg4,
            pg5,
            pg6,
            pg7,
            pg8,
            pg9,
            pg10,
            pg11,
            pg12,
            pg13,
            pg14,
            pg15,
        )
        tid = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        blocks = cute.arch.grid_dim()[0]
        cid = block // self.cluster
        clusters = blocks // self.cluster
        crank = block % self.cluster
        column = (crank * self.threads + tid) * VECTOR
        lane = tid % 32
        warp = tid // 32

        smem = cutlass_utils.SmemAllocator()
        snapshot = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=16)
        sums = smem.allocate_tensor(Float32, cute.make_layout((NMAX * TOP_K,)), byte_alignment=16)
        recv = smem.allocate_tensor(
            Float32, cute.make_layout((NMAX * CLUSTER * WARPS * TOP_K,)), byte_alignment=16
        )
        bars = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((NMAX,)), byte_alignment=8)

        if dry == 0:
            if tid == 0:
                for i in cutlass.range_constexpr(NMAX):
                    cute.arch.mbarrier_init(bars.iterator + i, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bars.iterator + i, self.warps * CLUSTER * TOP_K * 4
                    )
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.entry_trigger):
                    cute.arch.griddepcontrol_launch_dependents()

            control_address = control.toint()
            if tid == 0:
                counter = _load_control(control_address + 8)
                snapshot[0] = Int32((counter >> Uint32(GRID_SHIFT)) & Uint32(1))
                amount = Uint32(1)
                if block == 0:
                    amount = Uint32(GRID_STEP + 1) - Uint32(blocks)
                _red_add(control_address + 8, amount)
            norm_words = _load_input_x4(norm.toint() + Int64(column) * 2, Uint32(1))
            gains = []
            for w in cutlass.range_constexpr(4):
                low, high = _unpack_bf16x2(norm_words[w])
                gains.append(low + Float32(1.0))
                gains.append(high + Float32(1.0))
            cute.arch.barrier()
            buffer = Int64(snapshot[0])
            owner = column // self.shard
            src_slab = (buffer * self.ranks + Int64(rank)) * self.slab
            own_t = cute.make_rmem_tensor(cute.make_layout((NMAX, 4)), Uint32)
            gemm_address = gemm.toint()

            # ---- per token of this cluster: the arithmetic and the phase-1 publication ----
            for i in cutlass.range_constexpr(NMAX):
                token = cid + i * clusters
                if token < tokens:
                    selection = Int64(token) * TOP_K
                    row_index = []
                    row_weight = []
                    row_words = []
                    for k in cutlass.range_constexpr(TOP_K):
                        index, valid = self.select_row(indices, rows, token, k)
                        row_index.append(index)
                        row_weight.append(_load_f32(weights.toint() + (selection + k) * 4))
                        offset = _expert_rows.routed_row_offset(
                            Int64(index), Int64(row_stride), Int64(column)
                        )
                        row_words.append(_load_row_x4(gemm_address + offset * 2, valid))
                    partials = []
                    for k in cutlass.range_constexpr(TOP_K):
                        square_sum = Float32(0.0)
                        for w in cutlass.range_constexpr(4):
                            low, high = _unpack_bf16x2(row_words[k][w])
                            scaled_low = low * gains[2 * w]
                            scaled_high = high * gains[2 * w + 1]
                            square_sum = (
                                square_sum + scaled_low * scaled_low + scaled_high * scaled_high
                            )
                        partials.append(cute.arch.warp_reduction_sum(square_sum))
                    sent = []
                    for k in cutlass.range_constexpr(TOP_K):
                        sent.append(_float_bits(cute.arch.shuffle_sync(partials[k], 0)))
                    if lane < self.cluster:
                        row = (
                            recv.iterator
                            + i * (CLUSTER * WARPS * TOP_K)
                            + (crank * self.warps + warp) * TOP_K
                        )
                        cute.arch.store_async_dsmem(
                            row.align(16),
                            (sent[0], sent[1], sent[2], sent[3]),
                            bars.iterator + i,
                            lane,
                        )
                        cute.arch.store_async_dsmem(
                            (row + 4).align(16),
                            (sent[4], sent[5], sent[6], sent[7]),
                            bars.iterator + i,
                            lane,
                        )
                    cute.arch.mbarrier_wait(bars.iterator + i, 0)
                    if tid < TOP_K:
                        total = Float32(0.0)
                        if cutlass.const_expr(self.flat_sum):
                            for c in cutlass.range_constexpr(self.cluster):
                                for w in cutlass.range_constexpr(self.warps):
                                    total = (
                                        total
                                        + recv[
                                            i * (CLUSTER * WARPS * TOP_K)
                                            + (c * self.warps + w) * TOP_K
                                            + tid
                                        ]
                                    )
                        else:
                            for c in cutlass.range_constexpr(self.cluster):
                                cta_total = Float32(0.0)
                                for w in cutlass.range_constexpr(self.warps):
                                    cta_total = (
                                        cta_total
                                        + recv[
                                            i * (CLUSTER * WARPS * TOP_K)
                                            + (c * self.warps + w) * TOP_K
                                            + tid
                                        ]
                                    )
                                total = total + cta_total
                        sums[i * TOP_K + tid] = total
                    cute.arch.barrier()
                    result = [Float32(0.0)] * VECTOR
                    for k in cutlass.range_constexpr(TOP_K):
                        valid = Uint32(_expert_rows.routed_row_valid(row_index[k], rows))
                        scale = row_weight[k] * cute.math.rsqrt(
                            sums[i * TOP_K + k] / Float32(HIDDEN) + eps
                        )
                        scale = _zero_unless(scale, valid)
                        for w in cutlass.range_constexpr(4):
                            low, high = _unpack_bf16x2(row_words[k][w])
                            result[2 * w] = result[2 * w] + scale * (low * gains[2 * w])
                            result[2 * w + 1] = result[2 * w + 1] + scale * (
                                high * gains[2 * w + 1]
                            )
                    for w in cutlass.range_constexpr(4):
                        own_t[i, w] = _pack_publishable_bf16x2(result[2 * w], result[2 * w + 1])
                    if cutlass.const_expr(self.debug):
                        _store_x4(
                            dbg.toint() + (Int64(token) * HIDDEN + Int64(column)) * 2,
                            own_t[i, 0],
                            own_t[i, 1],
                            own_t[i, 2],
                            own_t[i, 3],
                        )
                    in_slot = (Int64(token) * self.shard + Int64(column - owner * self.shard)) * 2
                    for p in cutlass.range_constexpr(self.ranks):
                        _store_peer_x4_if(
                            peer_inbox[p].toint() + src_slab + in_slot,
                            Uint32((owner == p) & (owner != rank)),
                            own_t[i, 0],
                            own_t[i, 1],
                            own_t[i, 2],
                            own_t[i, 3],
                        )

            if owner == rank:
                inbox_address = inbox.toint()
                for i in cutlass.range_constexpr(NMAX):
                    token = cid + i * clusters
                    if token < tokens:
                        in_slot = (
                            Int64(token) * self.shard + Int64(column - owner * self.shard)
                        ) * 2
                        values = cute.make_rmem_tensor(cute.make_layout((self.ranks, 4)), Uint32)
                        for p in cutlass.range_constexpr(self.ranks):
                            values[p, 0] = own_t[i, 0]
                            values[p, 1] = own_t[i, 1]
                            values[p, 2] = own_t[i, 2]
                            values[p, 3] = own_t[i, 3]
                        stale = Uint32((1 << self.ranks) - 1) & ~(Uint32(1) << Uint32(rank))
                        while stale != Uint32(0):
                            polled = stale
                            for p in cutlass.range_constexpr(self.ranks):
                                address = (
                                    inbox_address + (buffer * self.ranks + p) * self.slab + in_slot
                                )
                                v0, v1, v2, v3 = _poll_x4(
                                    address,
                                    (polled >> Uint32(p)) & Uint32(1),
                                    values[p, 0],
                                    values[p, 1],
                                    values[p, 2],
                                    values[p, 3],
                                )
                                values[p, 0] = v0
                                values[p, 1] = v1
                                values[p, 2] = v2
                                values[p, 3] = v3
                            stale = Uint32(0)
                            for p in cutlass.range_constexpr(self.ranks):
                                stale = stale | _stale_bit(
                                    values[p, 0],
                                    values[p, 1],
                                    values[p, 2],
                                    values[p, 3],
                                    Uint32(1 << p),
                                )
                        for p in cutlass.range_constexpr(self.ranks):
                            _reset_x4(
                                inbox_address + (buffer * self.ranks + p) * self.slab + in_slot,
                                Uint32(rank != p),
                            )
                        total_out = [Float32(0.0)] * VECTOR
                        for p in cutlass.range_constexpr(self.ranks):
                            for w in cutlass.range_constexpr(4):
                                low, high = _unpack_bf16x2(values[p, w])
                                total_out[2 * w] = total_out[2 * w] + low
                                total_out[2 * w + 1] = total_out[2 * w + 1] + high
                        packed = []
                        for w in cutlass.range_constexpr(4):
                            packed.append(_pack_bf16x2(total_out[2 * w], total_out[2 * w + 1]))
                        gather_slot = (buffer * MAX_TOKENS + Int64(token)) * (HIDDEN * 2) + Int64(
                            column
                        ) * 2
                        if cutlass.const_expr(self.multicast):
                            _store_multicast_x4(
                                gather_mc.toint() + gather_slot,
                                packed[0],
                                packed[1],
                                packed[2],
                                packed[3],
                            )
                        else:
                            for p in cutlass.range_constexpr(self.ranks):
                                _store_peer_x4_if(
                                    peer_gather[p].toint() + gather_slot,
                                    Uint32(p != rank),
                                    packed[0],
                                    packed[1],
                                    packed[2],
                                    packed[3],
                                )
                        _store_x4(
                            out.toint() + (Int64(token) * HIDDEN + Int64(column)) * 2,
                            packed[0],
                            packed[1],
                            packed[2],
                            packed[3],
                        )
            else:
                for i in cutlass.range_constexpr(NMAX):
                    token = cid + i * clusters
                    if token < tokens:
                        gather_slot = (buffer * MAX_TOKENS + Int64(token)) * (HIDDEN * 2) + Int64(
                            column
                        ) * 2
                        gather_address = gather.toint() + gather_slot
                        g0 = Uint32(SENTINEL)
                        g1 = Uint32(SENTINEL)
                        g2 = Uint32(SENTINEL)
                        g3 = Uint32(SENTINEL)
                        pending = Uint32(1)
                        while pending != Uint32(0):
                            g0, g1, g2, g3 = _poll_x4(gather_address, pending, g0, g1, g2, g3)
                            pending = _stale_bit(g0, g1, g2, g3, Uint32(1))
                        _reset_x4(gather_address, Uint32(1))
                        _store_x4(
                            out.toint() + (Int64(token) * HIDDEN + Int64(column)) * 2,
                            g0,
                            g1,
                            g2,
                            g3,
                        )

            if cutlass.const_expr(self.pdl and not self.entry_trigger):
                cute.arch.griddepcontrol_launch_dependents()


_POINTER_DTYPES = (
    BFloat16,
    Int32,
    Float32,
    BFloat16,
    BFloat16,
    Uint32,
    Uint32,
    Uint32,
    Uint32,
    BFloat16,
) + (Uint32,) * (2 * MAX_RANKS)
_POINTER_ALIGNMENTS = (16, 4, 4) + (16,) * (len(_POINTER_DTYPES) - 3)


@functools.lru_cache(maxsize=32)
def compiled_kernel(
    device_index: int,
    ranks: int = 16,
    multicast: bool = True,
    pdl: bool = True,
    flat: bool = False,
    trigger: str = "late",
    debug: bool = False,
    loop: bool = False,
):
    kernel = (FinalizeTP16LoopKernel if loop else FinalizeTP16Kernel)(
        ranks, multicast, pdl, flat, trigger, debug
    )
    with torch.cuda.device(device_index):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return cute.compile(
            kernel,
            *[
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=align)
                for dtype, align in zip(_POINTER_DTYPES, _POINTER_ALIGNMENTS)
            ],
            Int32(0),
            Int32(HIDDEN),
            Int32(1),
            Int32(0),
            Float32(1e-6),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )


def launch(
    fn,
    gemm,
    indices,
    weights,
    norm,
    out,
    control,
    inbox,
    gather,
    gather_mc,
    dbg,
    peer_inbox,
    peer_gather,
    rows,
    row_stride,
    tokens,
    rank,
    eps,
    dry=0,
    stream=None,
):
    """``fn`` from ``compiled_kernel``; every pointer an int; peer lists padded to MAX_RANKS."""
    pi = list(peer_inbox) + [inbox] * (MAX_RANKS - len(peer_inbox))
    pg = list(peer_gather) + [gather] * (MAX_RANKS - len(peer_gather))
    s = stream if stream is not None else torch.cuda.current_stream()
    fn(
        gemm,
        indices,
        weights,
        norm,
        out,
        control,
        inbox,
        gather,
        gather_mc,
        dbg,
        *pi,
        *pg,
        rows,
        row_stride,
        tokens,
        rank,
        eps,
        dry,
        cuda.CUstream(s.cuda_stream),
    )


def reference(per_rank_gemm, per_rank_indices, per_rank_weights, norm_weight, eps, num_tokens):
    """The rounding points (BF16 partial with -0 cleared, BF16 rank sum), float64 in between."""
    gain = norm_weight.double() + 1.0
    summed = torch.zeros((num_tokens, HIDDEN), dtype=torch.float64, device=gain.device)
    for gemm, index, weight in zip(per_rank_gemm, per_rank_indices, per_rank_weights):
        index = index.reshape(num_tokens, -1).long()
        weight = weight.reshape(num_tokens, -1).double()
        nrows = gemm.shape[0]
        valid = (index >= 0) & (index < nrows)
        picked = gemm[:, :HIDDEN].double()[index.clamp(0, max(nrows - 1, 0))] * gain
        rstd = torch.rsqrt(picked.square().mean(-1) + eps)
        scale = torch.where(valid, weight * rstd, torch.zeros_like(rstd))
        partial = (scale[..., None] * picked).sum(1).to(torch.bfloat16)
        summed = summed + partial.double()
    return summed.to(torch.bfloat16)


class TwoPhaseWorkspace:
    """(w256) The TP8 two-phase finalize of one rank: its own fabric allocation (inbox ``[2][8 sources][MAX_TOKENS]
    [512]`` + gather ``[2][MAX_TOKENS][4096]`` BF16, every word the sentinel), its own arrival counter, and two
    token-loop
    builds: cluster-8 order (calls of 1..``CLUSTER8_TOKENS`` tokens) and flat order (above), the TP8 one-shot
    transport's two builds, so a call's result is that transport's bit for bit (a different route: each rank reduces
    its 512 columns of every token in rank order, then multicasts the sums). Collective construction over the TP
    group (``_mnnvl_workspace_all_succeeded``); calls serialized on one stream in the same order on every rank."""

    CLUSTER8_TOKENS = 16

    def __init__(self, mapping, device: torch.device, pdl: bool = True, trigger: str = "late"):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )

        self.ranks = mapping.tp_size
        self.rank = mapping.tp_rank
        self.device = device
        if self.ranks != 8:
            raise ValueError(f"the TP8 two-phase finalize serves 8 ranks, got {self.ranks}")
        comm = _get_mnnvl_workspace_comm(mapping)
        error = None
        try:
            inbox_b = inbox_bytes(self.ranks)
            size = inbox_b + gather_bytes()
            self._handle = _make_mnnvl_mcast_buffer(comm, size, mapping, mapping.is_multi_node())
            words = size // 4
            uc = [
                self._handle.get_uc_buffer(p, (words,), torch.int32, 0) for p in range(self.ranks)
            ]
            mc = self._handle.get_mc_buffer((words,), torch.int32, 0)
            with torch.cuda.device(device):
                uc[self.rank].fill_(-(2**31))
                self.control = torch.zeros(CONTROL_WORDS, dtype=torch.int32, device=device)
                self._fn = {
                    flat: compiled_kernel(
                        device.index, self.ranks, True, bool(pdl), flat, trigger, False, True
                    )
                    for flat in (False, True)
                }
                # Dry launches load both modules now, not inside a capture.
                dummy = self.control.data_ptr()
                for tokens in (1, self.CLUSTER8_TOKENS + 1):
                    launch(
                        self._fn[tokens > self.CLUSTER8_TOKENS],
                        dummy,
                        dummy,
                        dummy,
                        dummy,
                        dummy,
                        dummy,
                        uc[self.rank].data_ptr(),
                        uc[self.rank].data_ptr(),
                        mc.data_ptr(),
                        0,
                        [t.data_ptr() for t in uc],
                        [t.data_ptr() for t in uc],
                        0,
                        HIDDEN,
                        tokens,
                        self.rank,
                        1.0,
                        dry=1,
                    )
                torch.cuda.synchronize(device)
            self._uc, self._mc = uc, mc
            self.inbox = uc[self.rank].data_ptr()
            self.gather = self.inbox + inbox_b
            self.gather_mc = mc.data_ptr() + inbox_b
            self.peer_inbox = [t.data_ptr() for t in uc]
            self.peer_gather = [t.data_ptr() + inbox_b for t in uc]
            self.nbytes = size
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(comm, error is None):
            raise RuntimeError(
                "two-phase finalize workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def takes(self, num_tokens: int) -> bool:
        return 1 <= int(num_tokens) <= MAX_TOKENS

    def launch(self, gemm, indices, weights, norm, out, rows, row_stride, tokens, eps):
        """One call (pointers as ints); the build by token count (cluster-8 up to 16 tokens, flat above)."""
        fn = self._fn[int(tokens) > self.CLUSTER8_TOKENS]
        launch(
            fn,
            gemm,
            indices,
            weights,
            norm,
            out,
            self.control.data_ptr(),
            self.inbox,
            self.gather,
            self.gather_mc,
            0,
            self.peer_inbox,
            self.peer_gather,
            rows,
            row_stride,
            tokens,
            self.rank,
            eps,
            stream=torch.cuda.current_stream(self.device),
        )

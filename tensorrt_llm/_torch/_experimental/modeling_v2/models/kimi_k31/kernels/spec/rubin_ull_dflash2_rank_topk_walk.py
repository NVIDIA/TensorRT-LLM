# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL kernels of the DFlash2 drafter's tail under TP16 vocabulary parallelism, one Lamport hop.

The greedy draft tokens of one request's block (``K`` slots, one row each) are the selector walk
over each slot's global top-32 candidates. The engine path takes seven launches for it: the
shard top-32 (two), the zero-padded MNNVL all-reduce of the ``(id, value)`` pairs, the merge, the
scoring operands, torch's TF32 einsum and the walk. Here it is two kernels and one exchange:

* :class:`RankTop32PublishKernel`: one CTA per (row, 8,192-element chunk) selects the chunk's
  top-32 (``dflash2_top32``'s chunk pass); the last CTA of a row to finish (a per-row arrival
  counter) runs the second pass over the chunks' candidates, which sorts the shard's top-32 in
  ``torch.topk``'s order, and multicasts the 32 pairs (64 FP32 words, ``-0.0`` sent as ``+0.0``)
  into this rank's columns of the row in every rank's current Lamport buffer.
* :class:`SelectorWalkMcKernel`: a cluster of ``K`` CTAs, CTA ``e`` for slot ``e``, each polls its
  row's ``world x 64`` words, merges them (``dflash2_top32``'s pairs pass: the candidates and
  unary logits of ``merge_top32``), gathers the codebook rows of its 32 candidates (successors)
  and, through distributed shared memory, of slot ``e - 1``'s (predecessors; the anchor for slot
  0), and computes the slot's ``32 x 32`` bilinear scores with TF32 ``mma.sync`` (FP32
  accumulation). The scores go to CTA 0, whose first warp walks the slots
  (``SelectorWalkKernel``'s greedy ``realized_tokens`` walk) and writes the tokens. The other
  clusters of the grid re-arm the previous Lamport call's buffer; the last CTA to have read the
  rotation state advances it. With ``publish_drafts``, rank 0 also multicasts its tokens after
  the pairs (the draft agreement's payload, :class:`AgreedRewindKernel`).

Exchange layout of one call in Lamport buffer ``cur`` (one stage): row ``t`` of the ``m`` block
rows is ``world x 64`` FP32 words, rank ``r``'s pairs at words ``[64 r, 64 r + 64)``; after the
``m`` rows, ``DRAFT_WORDS`` int32 words per request carry rank 0's drafts. The rotation state the
walk writes clears both regions at the next call on the workspace.

Bitwise: the candidates and unary logits equal the engine path's (the same selection code on the
same pairs; the engine all-reduce of the zero-padded pairs also delivers ``-0.0`` as ``+0.0``).
The bilinear scores are TF32 products with FP32 accumulation like the einsum's, in another
accumulation order, so a score can differ in its last bits from cuBLAS's; the walk on them is the
engine walk.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..comm import lamport_ar as _lam
from . import dflash2_top32 as _t32
from .dflash2_selector_walk import SelectorWalkKernel

TOP_K = _t32.TOP_K
THREADS = _t32.THREADS
#: FP32 words of one rank's pairs in a row.
PAIR_WORDS = 2 * TOP_K
#: int32 words per request of rank 0's drafts after the pairs (K <= DRAFT_WORDS, zero padded).
DRAFT_WORDS = 8
#: The selector rank the walk kernel is written for.
RANK = 256
#: Row stride (FP32 words) of the staged operand rows: 256 + 4 keeps the mma fragment loads
#: conflict-free.
OPERAND_PITCH = RANK + 4
#: Clusters of the walk grid that re-arm the previous Lamport call's buffer.
CLEAR_CLUSTERS = 4
#: Largest block a walk kernel serves (one cluster of K CTAs, one request).
MAX_SLOTS = 8

_OPTIONS = "--opt-level 2 --enable-tvm-ffi"


def call_bytes(rows: int, requests: int, world: int) -> int:
    """Bytes of one Lamport buffer a call of ``rows`` block rows and ``requests`` requests uses."""
    return rows * world * PAIR_WORDS * 4 + requests * DRAFT_WORDS * 4


# ---------------------------------------------------------------------------
# PTX helpers
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, *, effects=True, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "", loc=loc, ip=ip)


@dsl_user_op
def _cluster_sync(*, loc=None, ip=None):
    """Every thread of the cluster arrives (release) and waits for all (acquire)."""
    _asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;",
        "",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mapa(addr, rank, *, loc=None, ip=None):
    """The shared::cluster address of this CTA's shared ``addr`` in cluster CTA ``rank``."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(addr).ir_value(loc=loc, ip=ip), Uint32(rank).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_cluster(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Uint32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.b32 $0, [$1];",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_cluster(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Uint32(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.shared::cluster.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


#: FP32 -> TF32 operand rounding of the scores: ``rne`` (``cvt.rn.tf32.f32``, cuBLAS's on
#: SM90: its TF32 einsum equals an RNE emulation to 6e-7), ``rna`` (``cvt.rna``) or ``trunc``
#: (the MMA reads the FP32 bits and drops the low 13).
TF32_ROUNDINGS = ("rne", "rna", "trunc")


@dsl_user_op
def _tf32_round(bits, mode: str, *, loc=None, ip=None):
    """The FP32 bits of ``cvt.<mode>.tf32.f32`` of the FP32 ``bits`` (``mode`` rn / rna)."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(bits).ir_value(loc=loc, ip=ip)],
            f"{{\n.reg .f32 x;\nmov.b32 x, $1;\ncvt.{mode}.tf32.f32 $0, x;\n}}",
            "=r,r",
            effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mma_tf32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """``mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32``: ``d = a b + c``."""
    ins = [Int32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1)]
    ins += [Float32(v).ir_value(loc=loc, ip=ip) for v in (c0, c1, c2, c3)]
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        ins,
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0, $1, $2, $3}, "
        "{$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        effects=False,
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4))


# ---------------------------------------------------------------------------
# RankTop32: the shard's top-32 pairs, multicast
# ---------------------------------------------------------------------------


class RankTop32PublishKernel:
    """This rank's top-32 ``(global id, value)`` pairs of each row of its BF16 vocabulary shard
    (``top32(..., pairs=True)``'s pairs, in its order), multicast into its columns of the row in
    every rank's current Lamport buffer.

    Grid ``(chunks, rows)``: CTA ``(c, t)`` selects chunk ``c`` of row ``t`` into the scratch rows;
    the last CTA of row ``t`` to arrive (``counters[t]``, reset by it) selects and sorts the
    shard's 32 from the chunks' candidates and publishes them.

    Args:
        world: ranks of the exchange.
        chunks: 8,192-element chunks of a row (the shard's width over ``CHUNK``, rounded up).
        multicast: publish with ``multimem.st`` through the multicast mapping; else ``st.global``
            to ``mc_base`` (a unicast emulation for single-GPU tests).
    """

    def __init__(self, world: int, chunks: int, multicast: bool):
        if chunks < 2:
            raise ValueError("a row of one chunk takes the single-pass top32")
        self.world = world
        self.chunks = chunks
        self.multicast = multicast
        self.m = chunks * TOP_K
        self.chunk_pass = _t32.Top32Kernel(
            _t32.SRC_BITS16, True, _t32.CHUNK // THREADS, _t32.OUT_SCRATCH, False
        )
        self.merge_pass = _t32.Top32Kernel(
            _t32.SRC_SCRATCH, True, -(-self.m // THREADS), _t32.OUT_PAIRS, False
        )

    @cute.jit
    def __call__(
        self,
        mSrc: cute.Tensor,  # [rows, pitch / 2] int32: BF16 logits rows (two to a word)
        mScrK: cute.Tensor,  # [rows, chunks * 32] int32 scratch
        mScrI: cute.Tensor,
        mScrB: cute.Tensor,
        mCount: cute.Tensor,  # [rows] int32 arrival counters (zero between calls)
        mDummy: cute.Tensor,  # [rows, 1] int32
        width: Int32,
        id_offset: Int32,
        rank: Int32,
        mc_base: Int64,
        flags: Int64,
        stream,
    ):
        rows = mCount.shape[0]
        self.kernel(
            mSrc, mScrK, mScrI, mScrB, mCount, mDummy, width, id_offset, rank, mc_base, flags
        ).launch(
            grid=[self.chunks, rows, 1],
            block=[THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mSrc: cute.Tensor,
        mScrK: cute.Tensor,
        mScrI: cute.Tensor,
        mScrB: cute.Tensor,
        mCount: cute.Tensor,
        mDummy: cute.Tensor,
        width: Int32,
        id_offset: Int32,
        rank: Int32,
        mc_base: Int64,
        flags: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        chunk, row, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        s_flag = smem.allocate_tensor(Int32, cute.make_layout((4,), stride=(1,)), 16)
        s_out = smem.allocate_tensor(Int32, cute.make_layout((PAIR_WORDS,), stride=(1,)), 16)
        # The logits are the previous grid's output; the Lamport state is the previous call's.
        _lam.griddepcontrol_wait()
        _lam.griddepcontrol_launch_dependents()

        self.chunk_pass.body(mSrc, mDummy, mDummy, mScrK, mScrI, mScrB, width, Int32(0), chunk, row)

        # Publish this CTA's scratch columns, then count it; the row's last CTA goes on.
        _fence_gpu()
        cute.arch.sync_threads()
        if tidx == 0:
            old = _lam.atom_add_acq_rel_u32(
                mCount.iterator.toint() + Int64(row) * 4, Int32(1)
            )
            last = Int32(0)
            if old == self.chunks - 1:
                last = Int32(1)
                _lam.st_u32(mCount.iterator.toint() + Int64(row) * 4, Int32(0))
            s_flag[0] = last
        cute.arch.sync_threads()
        if s_flag[0] != 0:
            _fence_gpu()
            # The pairs in torch.topk's order, into s_out (row stride 0: every row is s_out).
            pairs = cute.make_tensor(
                s_out.iterator, cute.make_layout((MAX_SLOTS * 64, PAIR_WORDS), stride=(0, 1))
            )
            self.merge_pass.body(
                mScrK, mScrI, mScrB, pairs, mDummy, mDummy, Int32(self.m), id_offset, Int32(0), row
            )
            cute.arch.sync_threads()
            if tidx < PAIR_WORDS // 4:
                cur, _, nbytes, _, _, _, _, _ = _lam.lamport_state(flags)
                w = [_lam_sanitize(s_out[4 * tidx + j]) for j in range(4)]
                dst = (
                    _lam.stage_addr(mc_base, cur, Int32(0), 1, nbytes)
                    + Int64(row) * (self.world * PAIR_WORDS * 4)
                    + Int64(rank) * (PAIR_WORDS * 4)
                    + Int64(tidx) * 16
                )
                if cutlass.const_expr(self.multicast):
                    _lam.multimem_st_v4(dst, w[0], w[1], w[2], w[3])
                else:
                    _lam.st_v4(dst, w[0], w[1], w[2], w[3])


@dsl_user_op
def _lam_sanitize(word, *, loc=None, ip=None):
    """``-0.0`` (``0x80000000``, the Lamport sentinel) as ``+0.0``; every other word unchanged."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(word).ir_value(loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.eq.b32 p, $1, 0x80000000;\nselp.b32 $0, 0, $1, p;\n}",
            "=r,r",
            effects=False,
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# SelectorWalk: poll, merge, operands, TF32 scores, walk
# ---------------------------------------------------------------------------


class SelectorWalkMcKernel:
    """The greedy draft tokens of one request's ``slots`` block rows from every rank's published
    pairs (see the module doc).

    Grid ``slots * (1 + CLEAR_CLUSTERS)`` CTAs of 256 threads in clusters of ``slots``: cluster 0
    walks, the others re-arm the previous call's Lamport buffer.

    Args:
        slots: K (block rows of the request), 2..8.
        world: ranks of the exchange.
        block_bf16: the tokens stand for a BF16 block (else FP32), as ``selector_greedy_tokens``.
        tf32_round: the FP32 -> TF32 rounding of the operands (``TF32_ROUNDINGS``).
        publish_drafts: rank 0 multicasts its tokens after the pairs (the agreement payload).
        multicast: as :class:`RankTop32PublishKernel`'s.
        debug: also write every slot's candidates, unary logits and scores (tests).
    """

    def __init__(
        self,
        slots: int,
        world: int,
        block_bf16: bool,
        tf32_round: str,
        publish_drafts: bool,
        multicast: bool,
        debug: bool = False,
    ):
        if tf32_round not in TF32_ROUNDINGS:
            raise ValueError(f"tf32_round {tf32_round!r} not in {TF32_ROUNDINGS}")
        if not 2 <= slots <= MAX_SLOTS:
            raise ValueError(f"the walk kernel serves 2..{MAX_SLOTS} slots, got {slots}")
        self.slots = slots
        self.world = world
        self.tf32_round = tf32_round
        self.publish_drafts = publish_drafts
        self.multicast = multicast
        self.debug = debug
        self.pairs = world * TOP_K
        self.row_words = world * PAIR_WORDS
        self.merge_pass = _t32.Top32Kernel(
            _t32.SRC_PAIRS, False, -(-self.pairs // THREADS), _t32.OUT_VALUES_IDS, False
        )
        self.walker = SelectorWalkKernel(
            slots,
            False,
            True,
            True,
            False,
            False,
            False,
            realized=True,
            realized_tokens=True,
            block_bf16=block_bf16,
        )
        if self.row_words % (4 * THREADS) and self.row_words > 4 * THREADS:
            raise ValueError("a row of the exchange is polled one 16-byte vector a thread")

    @cute.jit
    def __call__(
        self,
        mPredCb: cute.Tensor,  # [vocab, RANK] BF16
        mSuccCb: cute.Tensor,  # [vocab, RANK] BF16
        mGate: cute.Tensor,  # [slots, RANK] BF16, any 16-byte row stride
        mAnchor: cute.Tensor,  # [1] int32 / int64
        mTok: cute.Tensor,  # [1, slots] int32
        mDbgCand: cute.Tensor,  # [slots, 32] int32 (debug)
        mDbgUnary: cute.Tensor,  # [slots, 32] int32 FP32 bits (debug)
        mDbgScore: cute.Tensor,  # [slots, 32, 32] FP32 (debug)
        rank: Int32,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        stream,
    ):
        self.kernel(
            mPredCb,
            mSuccCb,
            mGate,
            mAnchor,
            mTok,
            mDbgCand,
            mDbgUnary,
            mDbgScore,
            rank,
            uc_local,
            mc_base,
            flags,
        ).launch(
            grid=[self.slots * (1 + CLEAR_CLUSTERS), 1, 1],
            block=[THREADS, 1, 1],
            cluster=[self.slots, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def rotate(self, flags: Int64, cur: Int32, nbytes: Int32, tidx: Int32, total: Int32):
        """Count this CTA (it has read the rotation state); the last of ``total`` writes the next
        call's state: this call's buffer becomes the dirty one, ``call_bytes`` of it to clear."""
        cute.arch.sync_threads()
        if tidx == 0:
            old = _lam.atom_add_acq_rel_u32(_lam.arrivals_addr(flags), Int32(1))
            if old == total - 1:
                _lam.st_v4(flags, (cur + 1) % _lam.NUM_BUFFERS, cur, nbytes, Int32(1))
                _lam.st_v4(
                    flags + Int64(16),
                    Int32(self.slots * self.row_words * 4 + DRAFT_WORDS * 4),
                    Int32(0),
                    Int32(0),
                    Int32(0),
                )
                _lam.st_u32(_lam.arrivals_addr(flags), Int32(0))

    @cute.kernel
    def kernel(
        self,
        mPredCb: cute.Tensor,
        mSuccCb: cute.Tensor,
        mGate: cute.Tensor,
        mAnchor: cute.Tensor,
        mTok: cute.Tensor,
        mDbgCand: cute.Tensor,
        mDbgUnary: cute.Tensor,
        mDbgScore: cute.Tensor,
        rank: Int32,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        K: cutlass.Constexpr = self.slots
        total = Int32(K * (1 + CLEAR_CLUSTERS))

        smem = cutlass.utils.SmemAllocator()
        s_row = smem.allocate_tensor(Int32, cute.make_layout((1, self.row_words), stride=(0, 1)), 16)
        s_val = smem.allocate_tensor(Int32, cute.make_layout((1, TOP_K), stride=(0, 1)), 16)
        s_id64 = smem.allocate_tensor(Int64, cute.make_layout((1, TOP_K), stride=(0, 1)), 16)
        s_cid = smem.allocate_tensor(Int32, cute.make_layout((TOP_K,), stride=(1,)), 16)
        s_pid = smem.allocate_tensor(Int32, cute.make_layout((TOP_K,), stride=(1,)), 16)
        s_gate = smem.allocate_tensor(Float32, cute.make_layout((RANK,), stride=(1,)), 16)
        s_a = smem.allocate_tensor(
            Float32, cute.make_layout((TOP_K, RANK), stride=(OPERAND_PITCH, 1)), 16
        )
        s_b = smem.allocate_tensor(
            Float32, cute.make_layout((TOP_K, RANK), stride=(OPERAND_PITCH, 1)), 16
        )
        # CTA 0's: every slot's scores, candidate ids and unary logits (pushed by their CTAs).
        s_score = smem.allocate_tensor(
            Float32, cute.make_layout((K, TOP_K, TOP_K), stride=(TOP_K * TOP_K, TOP_K, 1)), 16
        )
        s_cand = smem.allocate_tensor(Int32, cute.make_layout((K, TOP_K), stride=(TOP_K, 1)), 16)
        s_unary = smem.allocate_tensor(Int32, cute.make_layout((K, TOP_K), stride=(TOP_K, 1)), 16)
        s_tok = smem.allocate_tensor(Int32, cute.make_layout((DRAFT_WORDS,), stride=(1,)), 16)

        # Every input is an earlier grid's; the Lamport state is the previous call's (the pairs'
        # producer has read the same).
        _lam.griddepcontrol_wait()
        _lam.griddepcontrol_launch_dependents()
        cur, dirty, nbytes, dstages, c0, c1, c2, c3 = _lam.lamport_state(flags)
        self.rotate(flags, cur, nbytes, tidx, total)

        if bidx >= K:
            # Re-arm the previous call's buffer.
            _lam.clear_dirty_stages(
                uc_local,
                dirty,
                nbytes,
                dstages,
                c0,
                c1,
                c2,
                c3,
                (bidx - K) * THREADS + tidx,
                Int32(K * CLEAR_CLUSTERS * THREADS),
            )
        else:
            e = bidx
            row_bytes: cutlass.Constexpr = self.row_words * 4
            stage = _lam.stage_addr(uc_local, cur, Int32(0), 1, nbytes)
            # 1. Poll this slot's row: world x 64 words, one 16-byte vector a thread.
            for v0 in cutlass.range_constexpr(0, self.row_words // 4, THREADS):
                v = v0 + tidx
                if v < self.row_words // 4:
                    w = _lam.poll_v4(stage + Int64(e) * row_bytes + Int64(v) * 16)
                    for j in cutlass.range_constexpr(4):
                        s_row[0, 4 * v + j] = w[j]
            # The gate row while the pairs travel: FP32 in shared memory.
            gGate = cute.zipped_divide(mGate, (1, 8))
            if tidx < RANK // 8:
                g = gGate[(None, (e, tidx))].load().to(Float32)
                sGate8 = cute.zipped_divide(
                    cute.make_tensor(s_gate.iterator, cute.make_layout((1, RANK), stride=(0, 1))),
                    (1, 8),
                )
                sGate8[(None, (0, tidx))].store(g)
            cute.arch.sync_threads()

            # 2. The slot's global top-32 (merge_top32 of the gathered pairs).
            self.merge_pass.body(
                s_row, s_row, s_row, s_val, s_id64, s_val, Int32(self.pairs), Int32(0), Int32(0),
                Int32(0),
            )
            cute.arch.sync_threads()
            if tidx < TOP_K:
                s_cid[tidx] = Int32(s_id64[0, tidx])
            cute.arch.sync_threads()

            # 3. Successor rows of the candidates (FP32, exact from BF16).
            gSucc = cute.zipped_divide(mSuccCb, (1, 8))
            gPred = cute.zipped_divide(mPredCb, (1, 8))
            sA8 = cute.zipped_divide(s_a, (1, 8))
            sB8 = cute.zipped_divide(s_b, (1, 8))
            sG8 = cute.zipped_divide(
                cute.make_tensor(s_gate.iterator, cute.make_layout((1, RANK), stride=(0, 1))),
                (1, 8),
            )
            vcol = tidx % (RANK // 8)
            r0 = tidx // (RANK // 8)
            rstep: cutlass.Constexpr = THREADS // (RANK // 8)
            for i in cutlass.range_constexpr(TOP_K // rstep):
                c = r0 + i * rstep
                sB8[(None, (c, vcol))].store(gSucc[(None, (s_cid[c], vcol))].load().to(Float32))

            # 4. The predecessors: slot e - 1's candidates (CTA e - 1's), or the anchor.
            _cluster_sync()
            if tidx < TOP_K:
                pid = Int32(mAnchor[0])
                if e > 0:
                    pid = _ld_cluster(_mapa(Uint32(s_cid.iterator.toint()) + Uint32(4 * tidx), e - 1))
                s_pid[tidx] = pid
            cute.arch.sync_threads()
            gv = sG8[(None, (0, vcol))].load()
            for i in cutlass.range_constexpr(TOP_K // rstep):
                p = r0 + i * rstep
                pv = gPred[(None, (s_pid[p], vcol))].load().to(Float32)
                sA8[(None, (p, vcol))].store(pv * gv)
            cute.arch.sync_threads()

            # 5. scores[p, c] = sum_r a[p, r] b[c, r]: warp w the 16 x 8 tile (w // 4, w % 4).
            mt = warp // 4
            nt = warp % 4
            gq = lane // 4
            tq = lane % 4
            acc0 = Float32(0.0)
            acc1 = Float32(0.0)
            acc2 = Float32(0.0)
            acc3 = Float32(0.0)
            ra = mt * 16 + gq
            cb = nt * 8 + gq
            for ks in cutlass.range_constexpr(RANK // 8):
                k0 = ks * 8
                a0 = self.tf32(s_a[ra, k0 + tq])
                a1 = self.tf32(s_a[ra + 8, k0 + tq])
                a2 = self.tf32(s_a[ra, k0 + tq + 4])
                a3 = self.tf32(s_a[ra + 8, k0 + tq + 4])
                b0 = self.tf32(s_b[cb, k0 + tq])
                b1 = self.tf32(s_b[cb, k0 + tq + 4])
                acc0, acc1, acc2, acc3 = _mma_tf32(a0, a1, a2, a3, b0, b1, acc0, acc1, acc2, acc3)
            # Into CTA 0's s_score[e]; with the slot's candidates and unary logits.
            col = nt * 8 + 2 * tq
            base = Uint32(s_score.iterator.toint()) + Uint32(e * TOP_K * TOP_K * 4)
            _st_cluster(_mapa(base + Uint32((ra * TOP_K + col) * 4), 0), _t32._f32_bits(acc0))
            _st_cluster(_mapa(base + Uint32((ra * TOP_K + col + 1) * 4), 0), _t32._f32_bits(acc1))
            _st_cluster(
                _mapa(base + Uint32(((ra + 8) * TOP_K + col) * 4), 0), _t32._f32_bits(acc2)
            )
            _st_cluster(
                _mapa(base + Uint32(((ra + 8) * TOP_K + col + 1) * 4), 0), _t32._f32_bits(acc3)
            )
            if tidx < TOP_K:
                _st_cluster(
                    _mapa(Uint32(s_cand.iterator.toint()) + Uint32((e * TOP_K + tidx) * 4), 0),
                    s_cid[tidx],
                )
                _st_cluster(
                    _mapa(Uint32(s_unary.iterator.toint()) + Uint32((e * TOP_K + tidx) * 4), 0),
                    s_val[0, tidx],
                )
            _cluster_sync()

            # 6. CTA 0, warp 0: the greedy walk (SelectorWalkKernel, realized tokens).
            if e == 0:
                if cutlass.const_expr(self.debug):
                    for i in cutlass.range_constexpr(K * TOP_K * TOP_K // THREADS):
                        q = i * THREADS + tidx
                        mDbgScore[q // (TOP_K * TOP_K), (q // TOP_K) % TOP_K, q % TOP_K] = s_score[
                            q // (TOP_K * TOP_K), (q // TOP_K) % TOP_K, q % TOP_K
                        ]
                    if tidx < K * TOP_K:
                        mDbgCand[tidx // TOP_K, tidx % TOP_K] = s_cand[tidx // TOP_K, tidx % TOP_K]
                        mDbgUnary[tidx // TOP_K, tidx % TOP_K] = s_unary[tidx // TOP_K, tidx % TOP_K]
                if warp == 0:
                    prev = Int32(0)
                    for s in cutlass.range_constexpr(K):
                        row = _t32._bits_f32(s_unary[s, lane]) + s_score[s, prev, lane]
                        cand = s_cand[s, lane]
                        token = self.walker.argmax_vocab(row, cand)
                        if lane == 0:
                            mTok[0, s] = token
                            s_tok[s] = token
                        prev = self.walker.argmax_lane(row, lane)
                    if cutlass.const_expr(self.publish_drafts):
                        if lane < DRAFT_WORDS - K:
                            s_tok[K + lane] = Int32(0)
                        cute.arch.sync_warp()
                        if rank == 0:
                            if lane < DRAFT_WORDS // 4:
                                dst = (
                                    _lam.stage_addr(mc_base, cur, Int32(0), 1, nbytes)
                                    + Int64(K * row_bytes)
                                    + Int64(lane) * 16
                                )
                                w0 = s_tok[4 * lane]
                                w1 = s_tok[4 * lane + 1]
                                w2 = s_tok[4 * lane + 2]
                                w3 = s_tok[4 * lane + 3]
                                if cutlass.const_expr(self.multicast):
                                    _lam.multimem_st_v4(dst, w0, w1, w2, w3)
                                else:
                                    _lam.st_v4(dst, w0, w1, w2, w3)

    @cute.jit
    def tf32(self, x: Float32) -> Int32:
        bits = _t32._f32_bits(x)
        if cutlass.const_expr(self.tf32_round == "rne"):
            bits = _tf32_round(bits, "rn")
        if cutlass.const_expr(self.tf32_round == "rna"):
            bits = _tf32_round(bits, "rna")
        return bits


# ---------------------------------------------------------------------------
# The agreement: rank 0's drafts into the next step's tokens, with the KV rewind
# ---------------------------------------------------------------------------


class AgreedRewindKernel:
    """Every rank's ``next_draft_tokens`` and ``next_new_tokens`` from rank 0's drafts, polled
    from the last walk call's Lamport buffer, with the drafter's ``kv_lens`` rewind: the engine
    agreement (rank 0's drafts through the zero-padded all-reduce, ``k31_draft_agreed_tokens``)
    after ``draft_next_new_tokens_rewind``, in one launch. One CTA of 32 threads; ``batch``
    generation requests (no context rows).

    ``next_new[b] = [accepted[b, num_accepted[b] - 1], *drafts0[b]]``, ``next_draft[b] =
    drafts0[b]``, ``kv_lens[g] = max(kv_lens[g] - rewind[g], 0)``.
    """

    def __init__(self, slots: int, rows: int, world: int):
        self.slots = slots
        self.tail = rows * world * PAIR_WORDS * 4

    @cute.jit
    def __call__(
        self,
        mAcc: cute.Tensor,  # [batch, >= K + 1] tokens, any row stride
        mNumAcc: cute.Tensor,  # [>= batch] int32
        mKv: cute.Tensor,  # [>= batch] kv lens
        mRewind: cute.Tensor,  # [batch]
        mDraft: cute.Tensor,  # [batch, K] int32 out
        mNew: cute.Tensor,  # [batch, K + 1] out
        uc_local: Int64,
        flags: Int64,
        stream,
    ):
        self.kernel(mAcc, mNumAcc, mKv, mRewind, mDraft, mNew, uc_local, flags).launch(
            grid=[1, 1, 1], block=[32, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self,
        mAcc: cute.Tensor,
        mNumAcc: cute.Tensor,
        mKv: cute.Tensor,
        mRewind: cute.Tensor,
        mDraft: cute.Tensor,
        mNew: cute.Tensor,
        uc_local: Int64,
        flags: Int64,
    ):
        lane, _, _ = cute.arch.thread_idx()
        K: cutlass.Constexpr = self.slots
        batch = mDraft.shape[0]
        smem = cutlass.utils.SmemAllocator()
        s_d = smem.allocate_tensor(Int32, cute.make_layout((DRAFT_WORDS,), stride=(1,)), 16)
        # The walk grid (and every grid before it) has completed: the state names its buffer
        # as the dirty one, which no call clears before the next one on the workspace.
        _lam.griddepcontrol_wait()
        _lam.griddepcontrol_launch_dependents()
        f = _lam.ld_volatile_v4(flags)
        tail = _lam.stage_addr(uc_local, f[1], Int32(0), 1, f[2]) + Int64(self.tail)
        for b in cutlass.range(batch, unroll=1):
            if lane < DRAFT_WORDS // 4:
                w = _lam.poll_v4(tail + Int64(b * DRAFT_WORDS * 4) + Int64(lane) * 16)
                for j in cutlass.range_constexpr(4):
                    s_d[4 * lane + j] = w[j]
            cute.arch.sync_warp()
            if lane < K:
                d = s_d[lane]
                mDraft[b, lane] = d
                mNew[b, lane + 1] = mNew.element_type(d)
            if lane == K:
                n = Int32(mNumAcc[b])
                mNew[b, 0] = mNew.element_type(mAcc[b, n - 1])
            cute.arch.sync_warp()
        for g in cutlass.range(lane, mRewind.shape[0], 32, unroll=1):
            kv = Int32(mKv[g]) - Int32(mRewind[g])
            if kv < 0:
                kv = Int32(0)
            mKv[g] = mKv.element_type(kv)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_DT = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}


def _row_major(dtype, shape, align=None):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=align
    )


def _fake_stream():
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


@functools.cache
def compiled_rank_top32(world: int, chunks: int, src_cols: int, multicast: bool):
    kernel = RankTop32PublishKernel(world, chunks, multicast)
    rows = cute.sym_int()
    m = chunks * TOP_K
    return cute.compile(
        kernel,
        _row_major(Int32, (rows, src_cols), 16),
        _row_major(Int32, (rows, m)),
        _row_major(Int32, (rows, m)),
        _row_major(Int32, (rows, m)),
        _row_major(Int32, (rows,), 4),
        _row_major(Int32, (rows, 1)),
        Int32(1),
        Int32(0),
        Int32(0),
        Int64(0),
        Int64(0),
        _fake_stream(),
        options=_OPTIONS,
    )


@functools.cache
def compiled_walk(
    slots: int,
    world: int,
    block_bf16: bool,
    tf32_round: str,
    publish_drafts: bool,
    multicast: bool,
    cb_dtype: torch.dtype,
    gate_dtype: torch.dtype,
    anchor_dtype: torch.dtype,
    debug: bool = False,
):
    kernel = SelectorWalkMcKernel(
        slots, world, block_bf16, tf32_round, publish_drafts, multicast, debug
    )
    vocab = cute.sym_int()
    gate = cute.runtime.make_fake_tensor(
        _DT[gate_dtype],
        (slots, RANK),
        stride=(cute.sym_int64(divisibility=16 // gate_dtype.itemsize), 1),
        assumed_align=16,
    )
    return cute.compile(
        kernel,
        _row_major(_DT[cb_dtype], (vocab, RANK), 16),
        _row_major(_DT[cb_dtype], (vocab, RANK), 16),
        gate,
        _row_major(_DT[anchor_dtype], (1,)),
        _row_major(Int32, (1, slots)),
        _row_major(Int32, (slots, TOP_K)),
        _row_major(Int32, (slots, TOP_K)),
        _row_major(cutlass.Float32, (slots, TOP_K, TOP_K)),
        Int32(0),
        Int64(0),
        Int64(0),
        Int64(0),
        _fake_stream(),
        options=_OPTIONS,
    )


@functools.cache
def compiled_agreed_rewind(
    slots: int,
    rows: int,
    world: int,
    acc_dtype: torch.dtype,
    new_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    rewind_dtype: torch.dtype,
    num_acc_dtype: torch.dtype,
):
    kernel = AgreedRewindKernel(slots, rows, world)
    batch = cute.sym_int()
    gens = cute.sym_int()
    kvn = cute.sym_int()
    nacc = cute.sym_int()
    acc = cute.runtime.make_fake_tensor(
        _DT[acc_dtype], (batch, slots + 1), stride=(cute.sym_int64(), 1)
    )
    return cute.compile(
        kernel,
        acc,
        _row_major(_DT[num_acc_dtype], (nacc,)),
        _row_major(_DT[kv_dtype], (kvn,)),
        _row_major(_DT[rewind_dtype], (gens,)),
        _row_major(Int32, (batch, slots)),
        _row_major(_DT[new_dtype], (batch, slots + 1)),
        Int64(0),
        Int64(0),
        _fake_stream(),
        options=_OPTIONS,
    )


def rank_chunks(width: int) -> int:
    """8,192-element chunks of a shard row (the publish kernel needs at least two)."""
    return -(-width // _t32.CHUNK)


__all__ = [
    "AgreedRewindKernel",
    "CLEAR_CLUSTERS",
    "DRAFT_WORDS",
    "MAX_SLOTS",
    "PAIR_WORDS",
    "RANK",
    "RankTop32PublishKernel",
    "SelectorWalkMcKernel",
    "TF32_ROUNDINGS",
    "call_bytes",
    "compiled_agreed_rewind",
    "compiled_rank_top32",
    "compiled_walk",
    "rank_chunks",
]

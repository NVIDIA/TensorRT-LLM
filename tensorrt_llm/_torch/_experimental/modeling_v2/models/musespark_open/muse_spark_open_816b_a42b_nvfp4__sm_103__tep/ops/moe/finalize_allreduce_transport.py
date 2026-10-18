# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused expert finalize + all-reduce at TP8, on the fused hop's transport.

The built-in ``FinalizeIpcWorkspace.finalize`` (``tensorrt_llm/_torch/models/
musespark_open_finalize_ar.py``) normalizes each selected expert row on its own
(weight-first RMSNorm with zero-centered gains), weights and adds a token's
rows in FP32, rounds this rank's partial to BF16 with ``-0`` cleared, and sums
the eight ranks' partials in FP32 in rank order before one more BF16 rounding.
This entry computes exactly that -- the same statements on the same thread map
(a cluster of 8 CTAs per token, 64 threads, one 16-byte column vector of all 8
slots per thread), so every rounding happens on the same operands in the same
order -- and changes only how the partials travel and when dependents start:

* **Publication.** One store per thread to the slot's multicast address: the
  fabric writes it into slot ``rank`` of every rank's receive buffer (the
  built-in issues seven unicast system-scope stores per thread). ``unicast``
  keeps the seven stores.
* **Polling.** Every peer slot is read once with all reads in flight, then only
  the slots that still hold a sentinel word are read again, all in flight, with
  one predicated load per slot and no per-slot branch (the built-in spins on
  one peer at a time, so each later peer costs a full fabric read after the
  previous one arrived). A slot is read until its four words are all valid and
  never again.
* **Buffer parity.** A free-running per-rank arrival counter that every call
  advances by exactly ``GRID_STEP`` with fire-and-forget ``red.add`` (CTA 0
  adds ``GRID_STEP + 1 - grid``, every other CTA one, each after its thread 0
  read the counter); ``(counter >> GRID_SHIFT) & 1`` is the call's buffer. The
  built-in reads a buffer index and arrives with ``atom.acq_rel.inc``, whose
  last arrival flips the index (``atom``).
* **Cluster reduction of the slot norms.** After its warp reductions, lane
  ``c`` of each warp sends that warp's eight slot sums into CTA ``c``'s shared
  memory with two ``st.async`` (``warp``); every CTA waits on its own mbarrier
  for all of them and adds them as the built-in does: per source CTA its warps
  in order, then the CTAs in cluster-rank order. No cluster barrier follows,
  at the end or in the middle, and no CTA barrier precedes the exchange.
  ``push`` sends the per-CTA totals instead (one ``st.async`` per slot and
  destination, after a CTA barrier); ``pull`` keeps the built-in's cluster
  barrier, its ``ld.shared::cluster`` reads and its exit cluster barrier.
* **Dependent launch.** ``late`` triggers dependents after the output store, the
  built-in's place. ``entry`` triggers them right after this kernel's own
  ``griddepcontrol.wait``, so a dependent that loads immutable weights before
  its own wait (the unpool's split-K GEMM) streams them while this kernel waits
  for its peers.

Protocol (one-shot Lamport; the slot layout is this workspace's own). Each rank
owns a receive area of ``2 buffers x 8 source ranks x 16 tokens x 4096`` BF16
filled with the sentinel word ``0x80000000``: a 32-bit word that holds it has
not arrived. Published words never equal it (``-0`` halves are cleared). In
one call a slot ``(buffer, source, token, column vector)`` has one writer, the
source rank's thread with that token and column, and one reader, this rank's
thread with that token and column, which writes the sentinel back once it has
the value -- before its grid completes, hence before this rank publishes its
next call. A peer writes the same buffer again (call ``n + 2``) only after it
read this rank's call ``n + 1``, which this rank publishes only after call
``n`` completed. So a slot never holds two calls' words, and a word that is not
the sentinel is this call's. This rank's own slot is never read (its value is
in registers) and never reset; the multicast store writes it, unread.

The parity argument: a CTA of call ``n`` reads the counter after every earlier
call completed (its ``griddepcontrol.wait`` returned and every kernel between
two calls waits on its predecessor before completing) and before its own CTA
arrived, so it reads ``GRID_STEP * n + k`` with ``0 <= k < GRID_STEP``. The
grid is at most ``GRID_STEP`` CTAs. ``2**32`` is a multiple of
``2 * GRID_STEP``, so the parity survives the counter's wrap.

Progress: every CTA publishes before it polls, and a publication waits for no
read, so every slot is written; a stale slot is read again every round with a
strong system-scope load, so the loop ends once every peer published. Every
CTA of the grid is resident before a dependent grid can launch (a dependent
launches only after every CTA of this grid triggered or exited), so an early
dependent never holds back this grid's CTAs.

The ``warp`` / ``push`` reductions' shared-memory lifetime: a CTA's ``recv``
slots and mbarrier are written by its cluster peers' ``st.async`` only, each
slot exactly once per launch, and the CTA waits on its mbarrier for all of
them (512 or 1024 bytes) before it reads any; no peer reads its shared memory
at all. A CTA therefore leaves the cluster with no access to its shared memory
pending, and every ``st.async`` it issued targets a CTA that waits for exactly
those bytes before it can exit. A sending CTA may exit while its own
``st.async`` to a peer is still in flight: the store's data are register
operands (no shared memory of the sender is read), and its completion is
counted by the receiver's mbarrier (``complete_tx`` against the receiver's
``expect_tx``), never by the sender, so the in-flight store depends only on
the receiver, which is still waiting for it. The mbarriers are initialized, fenced and made
visible with one relaxed cluster arrive / wait before the first ``st.async``,
ahead of the kernel's dependency wait.

Ordering contract (the built-in's and the fused hop's): calls on one workspace
are issued in the same order on every rank and serialized on one stream (graph
replays included); every kernel launched with programmatic dependent launch
between two calls waits on its predecessor before it completes.

``FinalizeTransportWorkspace.finalize`` has the built-in workspace's signature
and input contract (its validation is the built-in's): calls of 1..16 tokens
run this kernel, larger ones the built-in workspace it wraps.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...kernels.persistent_experts import rows as _expert_rows

#: Expert (pooled) width the thread map is written for.
HIDDEN = 4096
#: Selected experts per token.
TOP_K = 8
#: Tokens per call this kernel serves: the built-in's cluster-8 range at TP8.
MAX_TOKENS = 16
#: CTAs per token (one cluster), threads per CTA, BF16 values per thread.
CLUSTER = 8
VECTOR = 8
THREADS = HIDDEN // (CLUSTER * VECTOR)
#: The tensor-parallel group size the transport is laid out for.
RANKS = 8
SENTINEL = 0x80000000
BUFFERS = 2
SLAB_BYTES = MAX_TOKENS * HIDDEN * 2
WORKSPACE_BYTES = BUFFERS * RANKS * SLAB_BYTES
#: Control words: 0 the buffer index and 1 the counter of the built-in-style
#: ``atom`` arrival, 2 the free-running counter of the ``red`` arrival.
CONTROL_WORDS = 4
#: What one call adds to the ``red`` counter, whatever its grid (at most 128 CTAs).
GRID_STEP = 128
GRID_SHIFT = 7
assert GRID_STEP == 1 << GRID_SHIFT and GRID_STEP >= MAX_TOKENS * CLUSTER
#: Bytes one ``push`` value occupies in a receiving CTA (one 16-byte ``st.async``).
_PUSH_BYTES = 16


@dataclasses.dataclass(frozen=True)
class Transport:
    """One combination of the transport choices (module docstring)."""

    publish: str = "multicast"
    arrival: str = "red"
    reduction: str = "warp"
    trigger: str = "late"

    def __post_init__(self):
        choices = {
            "publish": ("multicast", "unicast"),
            "arrival": ("red", "atom"),
            "reduction": ("warp", "push", "pull"),
            "trigger": ("late", "entry"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"Transport.{name} must be one of {allowed}")

    def label(self) -> str:
        return f"{self.publish}/{self.arrival}/{self.reduction}/{self.trigger}"


#: The transport ``finalize_transport`` installs.
DEFAULT = Transport()
#: The same transport with the trigger at kernel entry.
ENTRY_TRIGGER = Transport(trigger="entry")


try:
    from tensorrt_llm._torch.models import musespark_open_finalize_ar as _legacy
except ImportError:
    _legacy = None

HAS_CUTE_DSL = _legacy is not None and _legacy.HAS_CUTEDSL


if HAS_CUTE_DSL:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cutlass_utils
    from cutlass import BFloat16, Float32, Int32, Int64, Uint32
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import make_ptr
    from cutlass.cutlass_dsl import T, dsl_user_op

    from tensorrt_llm._torch.models.musespark_open_finalize_ar import (
        _FOUR_WORDS,
        _arrive,
        _asm,
        _i64,
        _load_cluster_peer_f32,
        _load_control,
        _load_input_x4,
        _pack_bf16x2,
        _pack_publishable_bf16x2,
        _split_words,
        _store_control,
        _store_x4,
        _u32,
        _unpack_bf16x2,
        _zero_unless,
    )

    # Like the built-in helpers, every global access is a side-effecting asm
    # statement, so none moves above griddepcontrol.wait or merges across polls.

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
        """``bit`` while any of the four words is the sentinel, else 0.

        Integer operations only (no predicate registers, no branch): a word
        XOR the sentinel is zero exactly when it is the sentinel, so the
        unsigned minimum of the four is zero exactly when the slot is stale.
        """
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

    # The predecessor grids' outputs (expert rows, row map, routing weights) are
    # read with coherent ``ld.global``, not ``ld.global.nc``: ``.nc`` promises the
    # data is read-only for the kernel's whole lifetime, which a programmatic
    # dependent launch breaks (the producer may still be writing until this
    # kernel's dependency wait), and ``ptxas`` takes the promise at its word and
    # may schedule such a load above ``griddepcontrol.wait``. Only the norm gain,
    # a weight, keeps ``.nc``.

    @dsl_user_op
    def _load_s32(address, *, loc=None, ip=None):
        result = _asm(T.i32(), [_i64(address, loc, ip)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip)
        return Int32(result)

    @dsl_user_op
    def _load_f32(address, *, loc=None, ip=None):
        result = _asm(T.f32(), [_i64(address, loc, ip)], "ld.global.f32 $0, [$1];", "=f,l", loc, ip)
        return Float32(result)

    @dsl_user_op
    def _load_row_x4(address, enabled, *, loc=None, ip=None):
        """Coherent 16-byte row load; yields zeros without touching memory when disabled."""
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
        """The built-in reset (the sentinel into a consumed slot, ``st.global``) when ``enabled``."""
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
        """The built-in unicast publication (``st.relaxed.sys.v4``) when ``enabled``."""
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
        """Add ``amount`` to an arrival counter; nothing waits for the result."""
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
        """The int32 with the same bit pattern as an FP32 value."""
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

    class LamportExchange:
        """The publish / poll half of the transport, on one rank's 16-byte column vector.

        Kept apart from the finalize so that a producer that computes the BF16
        partial itself (an expert kernel's last epilogue) can publish into and
        poll the same workspace: it calls ``publish`` with its four sanitized
        words, then ``poll``, then ``retire``, and owns everything else. The
        caller supplies the call's buffer (``buffer_parity``) and keeps the
        module's ownership rules: one writer and one reader per slot per call.
        """

        ranks: int
        multicast: bool

        def slot_offset(self, token, column):
            """Byte offset of ``(token, column vector)`` inside one source slab."""
            return (Int64(token) * HIDDEN + Int64(column)) * 2

        def slab_offset(self, buffer, source):
            """Byte offset of ``source``'s slab of ``buffer`` in a receive area."""
            return (Int64(buffer) * RANKS + Int64(source)) * SLAB_BYTES

        def publish(self, own, buffer, slot, rank, mcast, peers):
            """Store this rank's four words into slot ``rank`` of every rank's buffer."""
            # A plain method (unrolled while tracing): Python control flow only.
            own_slab = self.slab_offset(buffer, rank)
            if self.multicast:
                _store_multicast_x4(mcast.toint() + own_slab + slot, own[0], own[1], own[2], own[3])
            else:
                for p in range(self.ranks):
                    _store_peer_x4_if(
                        peers[p].toint() + own_slab + slot,
                        Uint32(rank != p),
                        own[0],
                        own[1],
                        own[2],
                        own[3],
                    )

        @cute.jit
        def poll(
            self,
            values: cute.Tensor,
            own0: Uint32,
            own1: Uint32,
            own2: Uint32,
            own3: Uint32,
            buffer: Int64,
            slot: Int64,
            rank: Int32,
            receive: Int64,
        ):
            """Fill ``values[p, 0..3]`` with every rank's words; this rank's are ``own``.

            Bit ``p`` of ``stale`` is set while rank ``p``'s slot may still hold a
            sentinel word: every peer's bit at first, never this rank's. Each
            round reads exactly the slots whose bit is set, all loads in flight
            (one predicated load per slot; a slot whose bit is clear keeps its
            words), then recomputes the bits from the words. The first round is
            the loop's first iteration, so its loads are scheduled like every
            other round's: all of them before any of their words is tested.
            """
            for p in cutlass.range_constexpr(self.ranks):
                values[p, 0] = own0
                values[p, 1] = own1
                values[p, 2] = own2
                values[p, 3] = own3
            stale = Uint32((1 << self.ranks) - 1) & ~(Uint32(1) << Uint32(rank))
            while stale != Uint32(0):
                polled = stale
                for p in cutlass.range_constexpr(self.ranks):
                    address = receive + (Int64(buffer) * RANKS + p) * SLAB_BYTES + slot
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

        def retire(self, buffer, slot, rank, receive):
            """Write the sentinel back into every peer slot this thread consumed."""
            for p in range(self.ranks):
                address = receive + (Int64(buffer) * RANKS + p) * SLAB_BYTES + slot
                _reset_x4(address, Uint32(rank != p))

    class _FinalizeTransportKernel(LamportExchange):
        """The built-in cluster-8 finalize arithmetic on this module's transport.

        CTA ``b`` handles token ``b // 8`` and the ``b % 8``-th 512-column slice;
        each thread owns one 16-byte column vector of that slice for all 8
        slots -- the built-in kernel's map.
        """

        def __init__(self, transport: Transport, pdl: bool):
            self.ranks = RANKS
            self.cluster = CLUSTER
            self.threads = THREADS
            self.warps = THREADS // 32
            self.multicast = transport.publish == "multicast"
            self.atom = transport.arrival == "atom"
            self.push = transport.reduction == "push"
            self.warp_push = transport.reduction == "warp"
            self.entry_trigger = transport.trigger == "entry"
            self.pdl = pdl

        def select_row(self, indices, rows, token, k):
            """The expert row of slot ``(token, k)`` and whether this rank holds it.

            The routing metadata contract of ``expert_rows``: ``indices`` is the
            ``[tokens, 8]`` int32 row map of the expert op (TRTLLM-gen's, or
            the persistent experts' slot map); a slot is this rank's when
            ``routed_row_valid`` (``0 <= index < rows``).
            """
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
            receive: cute.Pointer,
            mcast: cute.Pointer,
            peer0: cute.Pointer,
            peer1: cute.Pointer,
            peer2: cute.Pointer,
            peer3: cute.Pointer,
            peer4: cute.Pointer,
            peer5: cute.Pointer,
            peer6: cute.Pointer,
            peer7: cute.Pointer,
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
                receive,
                mcast,
                peer0,
                peer1,
                peer2,
                peer3,
                peer4,
                peer5,
                peer6,
                peer7,
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
            receive: cute.Pointer,
            mcast: cute.Pointer,
            peer0: cute.Pointer,
            peer1: cute.Pointer,
            peer2: cute.Pointer,
            peer3: cute.Pointer,
            peer4: cute.Pointer,
            peer5: cute.Pointer,
            peer6: cute.Pointer,
            peer7: cute.Pointer,
            rows: Int32,
            row_stride: Int32,
            tokens: Int32,
            rank: Int32,
            eps: Float32,
            dry: Int32,
        ):
            peers = (peer0, peer1, peer2, peer3, peer4, peer5, peer6, peer7)
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
            warp_sums = smem.allocate_tensor(
                Float32, cute.make_layout((self.warps * TOP_K,)), byte_alignment=16
            )
            cta_sums = smem.allocate_tensor(Float32, cute.make_layout((TOP_K,)), byte_alignment=16)
            sums = smem.allocate_tensor(Float32, cute.make_layout((TOP_K,)), byte_alignment=16)
            recv = smem.allocate_tensor(
                Float32, cute.make_layout((CLUSTER * TOP_K * 4,)), byte_alignment=16
            )
            bar = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((1,)), byte_alignment=8)

            # A dry launch only makes the driver load the module.
            if dry == 0:
                if cutlass.const_expr(self.push or self.warp_push):
                    # The receive mbarrier, visible cluster-wide before any peer
                    # sends (no global memory is touched before the wait).
                    if tid == 0:
                        cute.arch.mbarrier_init(bar.iterator, 1)
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bar.iterator,
                            self.warps * CLUSTER * TOP_K * 4
                            if self.warp_push
                            else CLUSTER * TOP_K * _PUSH_BYTES,
                        )
                    cute.arch.mbarrier_init_fence()
                    cute.arch.barrier()
                    cute.arch.cluster_arrive_relaxed()
                    cute.arch.cluster_wait()
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()
                    if cutlass.const_expr(self.entry_trigger):
                        cute.arch.griddepcontrol_launch_dependents()

                # Selected rows and routing weights; invalid slots read nothing.
                selection = Int64(token) * TOP_K
                gemm_address = gemm.toint()
                row_index = []
                row_valid = []
                row_weight = []
                row_words = []
                for k in cutlass.range_constexpr(TOP_K):
                    index, valid = self.select_row(indices, rows, token, k)
                    row_index.append(index)
                    row_valid.append(valid)
                    row_weight.append(_load_f32(weights.toint() + (selection + k) * 4))
                    offset = _expert_rows.routed_row_offset(
                        Int64(index), Int64(row_stride), Int64(column)
                    )
                    address = gemm_address + offset * 2
                    row_words.append(_load_row_x4(address, valid))
                norm_words = _load_input_x4(norm.toint() + Int64(column) * 2, Uint32(1))

                # This call's buffer, then this CTA's arrival.
                control_address = control.toint()
                if tid == 0:
                    if cutlass.const_expr(self.atom):
                        current = _load_control(control_address)
                        snapshot[0] = Int32(current)
                        previous = _arrive(control_address + 4, Uint32(blocks - 1))
                        if previous == Uint32(blocks - 1):
                            _store_control(control_address, current ^ Uint32(1))
                    else:
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

                # Sum of squares of the scaled row, per slot (the built-in's statements).
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
                if cutlass.const_expr(self.warp_push):
                    # Each warp's 8 slot sums (lane 0's, as the built-in stores
                    # them) into row (crank, warp) of every cluster CTA, this one
                    # included: lane c sends to CTA c, slots 0-3 and 4-7.
                    sent = []
                    for k in cutlass.range_constexpr(TOP_K):
                        sent.append(_float_bits(cute.arch.shuffle_sync(partials[k], 0)))
                    if lane < self.cluster:
                        row = recv.iterator + (crank * self.warps + warp) * TOP_K
                        cute.arch.store_async_dsmem(
                            row.align(16), (sent[0], sent[1], sent[2], sent[3]), bar.iterator, lane
                        )
                        cute.arch.store_async_dsmem(
                            (row + 4).align(16),
                            (sent[4], sent[5], sent[6], sent[7]),
                            bar.iterator,
                            lane,
                        )
                    cute.arch.mbarrier_wait(bar.iterator, 0)
                    if tid < TOP_K:
                        # The built-in's per-CTA total (warps in order from +0),
                        # then the cluster total (CTAs in order from +0).
                        total = Float32(0.0)
                        for c in cutlass.range_constexpr(self.cluster):
                            cta_total = Float32(0.0)
                            for w in cutlass.range_constexpr(self.warps):
                                cta_total = cta_total + recv[(c * self.warps + w) * TOP_K + tid]
                            total = total + cta_total
                        sums[tid] = total
                else:
                    if lane == 0:
                        for k in cutlass.range_constexpr(TOP_K):
                            warp_sums[warp * TOP_K + k] = partials[k]
                    cute.arch.barrier()
                    if tid < TOP_K:
                        total = Float32(0.0)
                        for w in cutlass.range_constexpr(self.warps):
                            total = total + warp_sums[w * TOP_K + tid]
                        if cutlass.const_expr(self.push):
                            # This CTA's total of slot `tid` into slot (crank, tid) of
                            # every cluster CTA, this one included.
                            target = recv.iterator + (crank * TOP_K + tid) * 4
                            for c in cutlass.range_constexpr(self.cluster):
                                cute.arch.store_async_dsmem(
                                    target.align(16),
                                    (_float_bits(total), Int32(0), Int32(0), Int32(0)),
                                    bar.iterator,
                                    c,
                                )
                        else:
                            cta_sums[tid] = total
                    if cutlass.const_expr(self.push):
                        cute.arch.mbarrier_wait(bar.iterator, 0)
                        if tid < TOP_K:
                            total = Float32(0.0)
                            for c in cutlass.range_constexpr(self.cluster):
                                total = total + recv[(c * TOP_K + tid) * 4]
                            sums[tid] = total
                    else:
                        # The built-in: release the totals to the cluster, then add
                        # all of them in cluster-rank order.
                        cute.arch.cluster_arrive()
                        cute.arch.cluster_wait()
                        if tid < TOP_K:
                            total = Float32(0.0)
                            for c in cutlass.range_constexpr(self.cluster):
                                total = total + _load_cluster_peer_f32(cta_sums.iterator + tid, c)
                            sums[tid] = total
                cute.arch.barrier()

                # Normalize each slot, weight it, and accumulate in FP32.
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

                # Publish, poll, retire.
                buffer = Int64(snapshot[0])
                slot = self.slot_offset(token, column)
                receive_address = receive.toint()
                self.publish(own, buffer, slot, rank, mcast, peers)
                values = cute.make_rmem_tensor(cute.make_layout((self.ranks, 4)), Uint32)
                self.poll(
                    values, own[0], own[1], own[2], own[3], buffer, slot, rank, receive_address
                )
                self.retire(buffer, slot, rank, receive_address)

                # The built-in sum: every rank's partial in rank order, FP32 from +0.
                total_out = [Float32(0.0)] * VECTOR
                for p in cutlass.range_constexpr(self.ranks):
                    for w in cutlass.range_constexpr(4):
                        low, high = _unpack_bf16x2(values[p, w])
                        total_out[2 * w] = total_out[2 * w] + low
                        total_out[2 * w + 1] = total_out[2 * w + 1] + high
                packed = []
                for w in cutlass.range_constexpr(4):
                    packed.append(_pack_bf16x2(total_out[2 * w], total_out[2 * w + 1]))
                _store_x4(out.toint() + slot, packed[0], packed[1], packed[2], packed[3])

                if cutlass.const_expr(self.pdl and not self.entry_trigger):
                    cute.arch.griddepcontrol_launch_dependents()
                if cutlass.const_expr(not (self.push or self.warp_push)):
                    # Keep this CTA's shared memory alive until its cluster peers read it.
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()

    _POINTER_DTYPES = (
        BFloat16,  # gemm
        Int32,  # indices
        Float32,  # weights
        BFloat16,  # norm
        BFloat16,  # out
        Uint32,  # control
        Uint32,  # receive
        Uint32,  # mcast
    ) + (Uint32,) * RANKS
    #: The built-in's assumptions: 4 bytes for the row map and the routing weights.
    _POINTER_ALIGNMENTS = (16, 4, 4) + (16,) * (len(_POINTER_DTYPES) - 3)

    @functools.lru_cache(maxsize=32)
    def _compiled_kernel(device_index: int, transport: Transport, pdl: bool):
        kernel = _FinalizeTransportKernel(transport, pdl)
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


def _pdl_enabled() -> bool:
    return _legacy is not None and _legacy._pdl_enabled()


class FinalizeTransportWorkspace:
    """This transport's memory and state for one TP8 group, in front of a built-in workspace.

    ``finalize`` has the built-in ``FinalizeIpcWorkspace.finalize`` signature and
    input contract: calls of 1..16 tokens run this module's kernel, every other
    call the wrapped built-in workspace (whose own protocol state is separate,
    so the two interleave freely; every rank routes a call alike because the
    choice depends on ``num_tokens`` only). Construction is collective over the
    group (multicast allocation, sentinel fill, kernel compilation and a dry
    launch) and either succeeds on every rank or raises on every rank.
    """

    def __init__(self, legacy, mapping: Mapping, transport: Transport = DEFAULT):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        if mapping.tp_size != RANKS:
            raise ValueError(f"the finalize transport serves TP{RANKS}, got TP{mapping.tp_size}")
        self.legacy = legacy
        self.transport = transport
        self.rank = mapping.tp_rank
        self.pdl = _pdl_enabled()
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self._comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[BaseException] = None
        try:
            if not HAS_CUTE_DSL:
                raise RuntimeError("the CuTe DSL or the built-in finalize module is unavailable")
            if type(legacy) is not _legacy.FinalizeIpcWorkspace or legacy._closed:
                raise RuntimeError(f"expected an open FinalizeIpcWorkspace, got {type(legacy)!r}")
            if not legacy._use_mnnvl or legacy._ranks != RANKS:
                raise RuntimeError("the wrapped workspace is not the TP8 fabric one")
            if torch.cuda.get_device_capability(self.device) != (10, 3):
                raise RuntimeError("the finalize transport is validated on SM103 only")
            self._handle = _make_mnnvl_mcast_buffer(self._comm, WORKSPACE_BYTES, mapping, True)
            words = WORKSPACE_BYTES // 4
            self._uc = [
                self._handle.get_uc_buffer(peer, (words,), torch.int32, 0) for peer in range(RANKS)
            ]
            self._mc = self._handle.get_mc_buffer((words,), torch.int32, 0)
            with torch.cuda.device(self.device):
                self._uc[self.rank].fill_(-(2**31))
                self.control = torch.zeros(CONTROL_WORDS, dtype=torch.int32, device=self.device)
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "finalize transport workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        receive = self._uc[self.rank].data_ptr()
        self._fixed = (
            self.control.data_ptr(),
            receive,
            self._mc.data_ptr(),
            *[t.data_ptr() for t in self._uc],
        )
        error = None
        try:
            self._compiled = _compiled_kernel(self.device.index, transport, self.pdl)
            # A dry launch loads the module now, not inside a capture.
            dummy = self.control.data_ptr()
            self._launch(dummy, dummy, dummy, dummy, dummy, 0, HIDDEN, 1, 1.0, 1)
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "finalize transport kernel compilation failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def _launch(self, gemm, indices, weights, norm, out, rows, row_stride, tokens, eps, dry=0):
        self._compiled(
            gemm,
            indices,
            weights,
            norm,
            out,
            *self._fixed,
            rows,
            row_stride,
            tokens,
            self.rank,
            eps,
            dry,
            cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream),
        )

    @staticmethod
    def takes(num_tokens) -> bool:
        """Whether a call of ``num_tokens`` runs this kernel (else the built-in workspace)."""
        return (
            isinstance(num_tokens, int)
            and not isinstance(num_tokens, bool)
            and 1 <= num_tokens <= MAX_TOKENS
        )

    def finalize(
        self,
        gemm2_output: torch.Tensor,
        indices: torch.Tensor,
        routing_weights: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        num_tokens: int,
    ) -> torch.Tensor:
        """``FinalizeIpcWorkspace.finalize``, with 1..16 tokens on this transport."""
        if not self.takes(num_tokens):
            return self.legacy.finalize(
                gemm2_output, indices, routing_weights, norm_weight, eps, num_tokens
            )
        if self.legacy._closed:
            raise RuntimeError("the wrapped MuseSparkOpen finalize workspace is closed")
        rows, row_stride = self.legacy._validate(
            gemm2_output, indices, routing_weights, norm_weight, eps, num_tokens
        )
        out = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device)
        self._launch(
            gemm2_output.data_ptr() if rows > 0 else 0,
            indices.data_ptr(),
            routing_weights.data_ptr(),
            norm_weight.data_ptr(),
            out.data_ptr(),
            rows,
            row_stride,
            num_tokens,
            float(eps),
        )
        return out


class FinalizeAllReduceTransport(OpWrapper):
    """The fused expert finalize + all-reduce of a ``FinalizeTransportWorkspace``.

    Returns the ``[num_tokens, 4096]`` BF16 group sum of every rank's partial:
    per token, each valid expert row ``x`` (slot index in ``[0, R)``) scaled by
    ``w * rsqrt(mean((x * g)^2) + eps)`` times ``x * g`` with ``g = 1 + norm``,
    summed over the slots, rounded to BF16 with ``-0`` cleared, then summed over
    the ranks in FP32 in rank order and rounded to BF16.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="no speculation: one token", spec=dict(tokens=1)),
        Cell(why="DFlash2 K=7 verify: 8 tokens", spec=dict(tokens=8)),
        Cell(
            why="the widest call on this kernel (the ring-buddy decode bound)", spec=dict(tokens=16)
        ),
        Cell(
            why="every slot of a token on one rank, none on the others",
            spec=dict(tokens=8, placement="concentrated"),
        ),
        Cell(
            why="ring-buddy placement: each slot on its owner or successor",
            spec=dict(tokens=8, placement="buddy"),
        ),
        Cell(
            why="exact and negative zeros in the rows, tokens with no valid slot",
            spec=dict(tokens=8, zeros=True),
        ),
        Cell(
            why="invalid slots and cancelling rows: the rank-order sum is observable",
            spec=dict(tokens=8, invalid=True, cancel=True),
        ),
    )

    note = """
    Metadata consumed: ``indices`` (the expert op's ``[tokens, 8]`` row map,
    read through one function, ``select_row``) and the routing weights. Call
    site: the built-in MoE's ``workspace.finalize`` once per layer per step,
    unchanged; the feature puts a ``FinalizeTransportWorkspace`` in that
    attribute, which runs this kernel for 1..16 tokens and the built-in kernel
    otherwise.

    Certified bit for bit against the built-in kernel by its own multi-GPU test
    (8 ranks on two nodes; every cell, every transport variant, tokens 1..16, a
    delayed rank, CUDA-graph replays with changing inputs and token counts, a
    long burst, the arrival counter across its wrap, a producer that triggers
    its dependents before it writes their inputs) and against `reference`
    within a band; the alternation of the two receive buffers by the counter's
    parity is checked white-box. Not certified: TP4, more than 16 tokens
    (served by the built-in kernel).
    """

    def __call__(
        self,
        gemm2_output: torch.Tensor,
        indices: torch.Tensor,
        routing_weights: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        num_tokens: int,
        workspace: FinalizeTransportWorkspace,
    ) -> torch.Tensor:
        if not FinalizeTransportWorkspace.takes(num_tokens):
            raise ValueError(f"finalize_allreduce_transport serves 1..{MAX_TOKENS} tokens")
        return workspace.finalize(
            gemm2_output, indices, routing_weights, norm_weight, eps, num_tokens
        )

    def reference(
        self,
        per_rank_gemm2_outputs: Sequence[torch.Tensor],
        per_rank_indices: Sequence[torch.Tensor],
        per_rank_routing_weights: Sequence[torch.Tensor],
        norm_weight: torch.Tensor,
        eps: float,
        num_tokens: int,
        workspace,
    ) -> torch.Tensor:
        """The op's rounding points (the BF16 partial, the BF16 sum), float64 in between.

        The collective exception to mirroring: takes every rank's expert rows,
        slot indices and routing weights in ascending rank order. ``workspace``
        is ignored.
        """
        gain = norm_weight.double() + 1.0
        summed = torch.zeros((num_tokens, HIDDEN), dtype=torch.float64, device=gain.device)
        for gemm, index, weight in zip(
            per_rank_gemm2_outputs, per_rank_indices, per_rank_routing_weights
        ):
            index = index.reshape(num_tokens, -1).long()
            weight = weight.reshape(num_tokens, -1).double()
            rows = gemm.shape[0]
            valid = (index >= 0) & (index < rows)
            picked = gemm[:, :HIDDEN].double()[index.clamp(0, max(rows - 1, 0))] * gain
            rstd = torch.rsqrt(picked.square().mean(-1) + eps)
            scale = torch.where(valid, weight * rstd, torch.zeros_like(rstd))
            partial = (scale[..., None] * picked).sum(1).to(torch.bfloat16)
            summed = summed + partial.double()
        return summed.to(torch.bfloat16)


finalize_allreduce_transport = FinalizeAllReduceTransport()


def log_once_enabled(layers: int, transport: Transport) -> None:
    logger.info(
        f"[musespark_open] finalize_transport: fused expert finalize + all-reduce of all {layers} layers on the "
        f"{transport.label()} transport (publish/arrival/reduction/trigger) for calls of "
        f"1..{MAX_TOKENS} tokens; the built-in kernel above that"
    )

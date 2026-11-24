# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Adapted from vLLM MR8's ``OprojFused`` (the vendored copy
# ``kernels/vendor/vllm_mr8/.../oproj/consumer.py``, producer ``oproj/producer.py``) and from
# vllm-project/vllm@e3ddbdbd60 vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/mla/dcp.py
# (``DcpValue.kernel_rows`` / ``merge_row``, the vendored copy
# ``kernels/vendor/vllm_main/.../mla/dcp.py``). Changed: one grid runs both; the merge clusters
# write the o-projection's x into a Lamport buffer that every CTA polls; every CTA waits
# (``griddepcontrol.wait``) before it reads anything a kernel of this layer wrote (module doc).
"""The DSA layer's Helix value merge fused into MR8's o-projection (``FlatDsaOproj.bind_value``).

One kernel replaces ``DcpValue`` (pushed, ``kernel_rows``) + ``OprojFused`` (``x_shared``) for
the DSA layers' fused rows (at most ``ROWS``): the BF16 partials and base-2 LSEs every rank's
sparse decode pushed into this rank's ``DcpMailbox`` are merged per (local head, row), projected
through ``W_UV``, gated, and fed to MR8's o-projection + TP16 all-reduce + post-attention AttnRes.

Grid (8, 14), clusters of 8 along x, 512 threads, as ``OprojFused``. Block (x, y) is cluster rank
``x`` of physical cluster ``y``; its logical cluster is ``(y + 8) % 14``:

- physical clusters 0 .. 5 (logical 8 .. 13, ``OprojFused``'s producer-only clusters) are the
  merge clusters: cluster ``h`` merges local head ``h``, its CTA ``z`` request row ``z``;
- physical clusters 6 .. 13 are logical 0 .. 7, ``OprojFused``'s consumer clusters (token = the
  logical cluster, cluster ranks 0 .. 6 own 1024 output columns each);
- every CTA runs the producer: output rows ``64 (8 lc + x)`` .. + 63 of the o-projection, ``lc``
  the logical cluster.

Before ``griddepcontrol.wait`` a CTA reads static weights only: the o-projection rows into
registers (``OprojProducer.load_weight``), the consumer's three AttnRes weight vectors (cp.async),
and in a merge CTA its ``W_UV`` slice ``[512, 16]`` (columns ``16 z`` .. ``16 z + 15`` of head
``h``) into shared memory. After the wait:

1. a merge CTA re-arms its slice of the other parity's x buffer (below), then threads 0 .. 127
   merge row ``z`` of head ``h`` exactly as ``DcpValue.merge_row`` (warp 0 polls the sources'
   LSEs; source weights ``exp2(lse - max) / sum`` in FP32; the 16 partials polled, added in source
   order in FP32 and re-armed as read; one BF16 rounding), and store the 512-wide row into row
   ``z`` of every cluster CTA's latent tile (``st.shared::cluster``); a cluster barrier
   (``barrier.cluster.arrive`` release / ``wait`` acquire) makes the 8 rows visible;
2. warps 0 and 1 of a merge CTA project the 8 rows onto the CTA's 16 columns (8 each) with
   ``DcpValue``'s MMA sequence (m16n8k16 BF16, two FP32 chains over the 16 K chunks, summed), round
   to BF16, multiply by ``sigmoid(gate)`` and store BF16 pairs into x (``st.relaxed.gpu``, negative
   zero halves made +0);
3. the consumer CTAs prefetch the AttnRes prefix / blocks rows and run the AttnRes prologue;
4. every CTA polls x (``OprojProducer.repoll_x``, bounded: site ``SITE_X``), runs the producer's
   MMAs and multicasts its partials; the consumer CTAs then run ``OprojConsumer.post``.

Per value the operations of ``DcpValue`` then ``OprojFused(x_shared=True)`` in their order: the
merge, ``W_UV`` and gate are ``DcpValue``'s instruction for instruction (only rows 0 .. 7 of each
MMA are fed; a row's outputs depend on its own A row only), and the producer's MMA operands are
``x_shared``'s. x differs from ``DcpValue``'s output only in the sign of a zero.

Ordering (what each read relies on):

- The predecessor on this stream is the sparse decode (``SparseDcpDecode``, or the speculative
  ``SpecDcpDecode``, which never runs ``griddepcontrol.wait`` and releases its dependents at its
  start). So nothing here relies on its predecessor having waited before it triggered: every read
  of a buffer a kernel of this layer or an earlier one wrote (gate, totals, AttnRes prefix and
  blocks, the x buffers) comes after this kernel's ``griddepcontrol.wait``, i.e. after the decode
  grid completed and flushed. The decode's completion implies the in-projection that wrote the
  gate completed: ``SparseDcpDecode`` waits for the merge grid; ``SpecDcpDecode``'s CTAs each see
  their row's ``done`` pair, which the merge grid writes after its own wait for the select grid;
  the select completes after the scorer (its wait, or, fused, every scorer CTA's epoch word
  written after the scorer's wait); and the scorer waits for ``IndexFrontGemm``, which waits for
  the in-projection. (The query side joins the decode only after ``QueryWait``'s wait, and
  ``MlaQuery`` starts after ``IndexFrontGemm`` completes: a full dependency.)
- The partial / LSE mailbox: cross-rank Lamport slots, polled with ``common.spin.poll_words``
  (about 67 s; past it the trap record, sites ``SITE_LSE`` / ``SITE_PAIR``, and a trap) and
  re-armed by their only reader, this kernel, as ``DcpValue`` does; parity alternates by DSA layer.
  A source's next write of a slot is its decode two DSA layers later, which needs this rank's
  query rows of that layer, published by ``MlaQuery`` only after that layer's ``IndexFrontGemm``
  completed, which is after this grid completed (every kernel between them waits before it
  completes, the MoE right after this one for this grid). So the re-arm precedes the next write.
- The x buffers ``[2, ROWS, 768]`` (``x`` / ``x_other`` here), armed with ``PAIR_SENT`` words once
  at allocation. The call of parity ``p`` writes rows ``< B`` of buffer ``p`` (each merge CTA its
  ``[B, 16]`` slice, once) and re-arms all ``ROWS`` rows of buffer ``1 - p`` (each merge CTA the
  same slice), after its ``griddepcontrol.wait``. Buffer ``1 - p``'s readers are the 112 CTAs of
  the previous DSA layer's call (``polled``: they read x into shared memory before they publish);
  that grid completed before this call's wait returns (the chain above: the decode's completion
  implies this layer's ``IndexFrontGemm`` completed, which comes after the previous DSA layer's
  MoE waited for that grid). Its next writer is the next DSA layer's call, which writes only after
  its own wait, so after this grid completed. Buffer ``p`` was re-armed by the previous call and
  is read by this grid only; no word written here equals ``PAIR_SENT`` (no negative zero halves),
  so a reader never takes a stale word: it polls until each of its words is written. The sequence
  of parities is 0, 1, 0, 1, ...: every DSA layer of a step runs this kernel or none does, and the
  DSA layer count is even (``FlatMlaQuery``), so a step's last call and the next one's first differ.
- The o-projection mailbox: ``OprojFused``'s, polled by the consumer CTAs and re-armed after use;
  here every one of those reads comes after the wait (``OprojFused`` with ``polled_x`` reads it
  before its wait), so its existing argument holds a fortiori.

Residency (no deadlock): the waits between CTAs of this grid are the consumer clusters' and the
merge clusters' cluster barriers / DSMEM exchanges (co-scheduled by the cluster launch) and the x
poll of every CTA on the 48 merge CTAs. The merge CTAs wait on nothing of this grid: only on the
decode's completion (``griddepcontrol.wait``) and on the partials every rank's decode pushes,
which never wait on this kernel (on any rank). Other grids that can hold SMs while this one is
placed: the decode and the merge grid (resident before any CTA here launches: a PDL dependent
launches only once every CTA of its primary has triggered or exited, and the decode launches
only after the merge grid and ``QueryWait`` triggered), and the prefetch stream's L2 prefetches;
none waits on this grid, so they exit and free their SMs. This grid's dependents launch only
after all its 112 CTAs triggered, i.e. were placed. ``FlatDsaOproj`` checks at load that the 14
clusters of 8 can be co-resident (``cuOccupancyMaxActiveClusters`` >= 14). The merge clusters
have the lowest block indices, so the scheduler places them first when the decode (96 CTAs) +
merge grid (8) + this grid (112) exceed the SM count (Rubin ~212, GB300 152).

Shared memory: a merge CTA never runs the consumer, so its ``W_UV`` slice ``[16, 544]`` BF16, its
latent tile ``[8, 544]`` BF16 and its 16 source weights live in the consumer's ``pf``, ``parts``
and ``wpre`` buffers (24 / 12 / 0.25 KB): the grid's shared memory is ``OprojFused``'s.
"""

from __future__ import annotations

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..common import rubin_ull_spin_trap as spin_trap
from ..vendor.vllm_main import module as main_module
from ..vendor.vllm_mr8 import module

_consumer = module("cutedsl.oproj.consumer")
_producer = module("cutedsl.oproj.producer")
_host = module("cutedsl.common.host")
_m8 = module("cutedsl.common.primitives")
_mp = main_module("cutedsl.common.primitives")
_spin = main_module("cutedsl.common.spin")
_dcp = main_module("cutedsl.mla.dcp")

#: Request rows of the fused path: one merge cluster CTA per row.
ROWS = 8
#: Local heads per rank (96 DSA heads over the 16 Helix ranks): one merge cluster each.
HEADS = 6
#: The o-projection's K (6 heads x 128 value columns) and N (hidden size).
K, HIDDEN = 768, 7168
LATENT, V_DIM = 512, 128
#: Physical clusters of the grid; logical cluster = (physical + SHIFT) % CLUSTERS.
CLUSTERS, SHIFT = 14, 8
#: Value columns of a head per merge CTA.
COLUMNS = V_DIM // ROWS
#: Row pitch (BF16 elements) of the latent tile and the W_UV slice in shared memory: 1088 bytes,
#: so a quarter-warp's 16-byte fragment loads (rows g, g + 1, columns 8 q) hit distinct banks.
PITCH = 544
#: Named barrier of the merge's source weights (128 threads; the consumer uses 1, 2 and 5).
MERGE_BARRIER = 6
#: ``spin_trap`` sites (a / b / c): a timed-out LSE poll (source, row, head), partial poll
#: (source, row, head) and x poll (physical cluster, cluster rank, pending chunk mask).
SITE_LSE, SITE_PAIR, SITE_X = 195, 196, 197

EMPTY_LSE, LSE_SENT, PAIR_SENT = _dcp.EMPTY_LSE, _dcp.LSE_SENT, _dcp.PAIR_SENT
SPIN_LSE, SPIN_ROWS = _dcp.SPIN_LSE, _dcp.SPIN_ROWS

at = _mp.at
load_words, load_volatile, store_words = _mp.load_words, _mp.load_volatile, _mp.store_words
mma_into, zeros, unpack_bf16 = _mp.mma_into, _mp.zeros, _mp.unpack_bf16


@_mp.asm()
def store_cluster_words(pointer, words, peer):
    """``words`` (32-bit) at the shared-memory offset of ``pointer`` in cluster CTA ``peer``."""
    n = len(words)
    vector, registers = _mp._vector(n, 1)
    code = f"{{ .reg .b32 dst; mapa.shared::cluster.u32 dst,$0,${n + 1}; "
    code += f"st.shared::cluster{vector}.b32 [dst],{registers}; }}"
    constraints = "r" + ",r" * n + ",r,~{memory}"
    return code, constraints, (pointer, *(Uint32(w) for w in words), Int32(peer))


class DsaOprojFused:
    """``OprojFused(nb, has_delta, x_shared=True, polled_x=True, drain, vote)`` whose x is merged
    here from the DSA partials (module doc). Arguments: x (this call's parity of the x buffers,
    ``[batch, 768]``), the o-projection weight, ``OprojFused``'s mailbox / multicast address /
    rank and AttnRes rows (prefix, blocks, the three weights, out, xmoe, hints, two eps), then the
    partials ``[world, cap, 6, 512]`` and LSEs ``[world, cap, 6]`` of this parity, ``W_UV``
    ``[6, 512, 128]`` (latent stride 1), the gate ``[batch, 768]``, the per-row totals, the poll
    timeout counters ``[4]`` and the other parity's x buffer ``[ROWS, 768]``. ``record``: the
    ``spin_trap`` record address (0: timed-out polls are counted, not trapped)."""

    DRAIN_BARRIER = _consumer.OprojFused.DRAIN_BARRIER

    def __init__(self, nb, *, has_delta, world, local, batch, drain, vote, record=0):
        if local != HEADS or not 1 <= batch <= ROWS or not 1 <= world <= 32:
            raise ValueError(f"DsaOprojFused: {local} heads, {batch} rows, {world} ranks")
        self.world, self.local, self.batch = world, local, batch
        self.has_delta, self.drain, self.record = has_delta, drain, record
        self.producer = _producer.OprojProducer(K, staged_blocks=0, x_shared=True, vote=vote)
        self.consumer = _consumer.OprojConsumer(nb, has_delta=has_delta, vote=vote)
        # The merge CTAs' buffers alias the consumer's (module doc): its 4-group layout.
        assert self.consumer.groups == 4 and self.producer.x_shared

    @cute.jit
    def __call__(self, *args):
        _m8.launch(self.kernel, args, (8, CLUSTERS, 1), 512, 8, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, args):
        x, weight = args[0], args[1]
        rows = (args[2], *args[5:15])
        rank, physical, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        M: cutlass.Constexpr = x.shape[0]
        cluster = (physical + SHIFT) % CLUSTERS
        cta = cluster * 8 + rank
        merging = physical < HEADS
        active = (cluster < M) & (rank < 7)
        state = self.consumer.allocate()
        smem = self.producer.allocate(M)
        self.consumer.setup(state[5])
        cute.arch.griddepcontrol_launch_dependents()
        group0 = active & (tid < 128)
        if group0:
            self.prefetch_weights(rows, state[7], rank)
        if merging:
            self.stage_w_uv(args, state, physical, rank, tid)
        weights = self.producer.load_weight(weight, smem[2], cta)
        # Everything below may read what this layer's kernels wrote (module doc).
        cute.arch.griddepcontrol_wait()
        if group0:
            self.prefetch_rows(rows, state[7], rank, cluster)
        if merging:
            self.merge_project(args, state, physical, rank, tid)
        mix = zeros(8, Float32)
        maximum, exponent_sum, qpart = Float32(0), Float32(0), Float32(0)
        if group0:
            maximum, exponent_sum, qpart = self.consumer.pre(state, rank, rows[9], mix)
        self.produce(args, smem, weights, cta, physical, rank)
        # The last publish's threads (``OprojProducer.publish``) and the drain warp.
        _, gt, ng = self.producer.schedule(M)
        publishers: cutlass.Constexpr = 8 * min(M - (ng - 1) * gt * 8, gt * 8)
        drainers: cutlass.Constexpr = publishers + 32
        if cutlass.const_expr(self.drain):  # noqa: SIM102 (compile-time)
            if tid < publishers:
                if active:
                    _m8.named_barrier(self.DRAIN_BARRIER, drainers, arrive=True)
                else:
                    cute.arch.fence_acq_rel_gpu()
        if active:
            block_mix = (mix, maximum, exponent_sum, qpart)
            self.consumer.post(rows, state, block_mix, rank, cluster)
            if cutlass.const_expr(self.drain):  # noqa: SIM102 (compile-time)
                if tid >= 512 - 32:  # the last poll group's last warp
                    _m8.named_barrier(self.DRAIN_BARRIER, drainers)
                    cute.arch.fence_acq_rel_gpu()

    @cute.jit
    def prefetch_weights(self, rows, pf, rank):
        """``OprojConsumer.prefetch``'s static rows (the AttnRes weights), one cp.async group."""
        _, _, _, norm, query, out_norm = rows[:6]
        tid, _, _ = cute.arch.thread_idx()
        h = (rank * 128 + tid) * 8
        if cutlass.const_expr(self.consumer.nb > 0):
            _m8.copy_g2s(pf.iterator + tid * 4, norm.iterator + h)
            _m8.copy_g2s(pf.iterator + 512 + tid * 4, query.iterator + h)
        _m8.copy_g2s(pf.iterator + 1024 + tid * 4, out_norm.iterator + h)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def prefetch_rows(self, rows, pf, rank, token):
        """``OprojConsumer.prefetch``'s activation rows (the prefix, the blocks of ``token``), one
        cp.async group; ``OprojConsumer.pre`` waits for both groups."""
        _, prefix, blocks = rows[:3]
        tid, _, _ = cute.arch.thread_idx()
        h = (rank * 128 + tid) * 8
        if cutlass.const_expr(self.has_delta):
            _m8.copy_g2s(pf.iterator + 1536 + tid * 4, at(prefix, token, h))
        for block in cutlass.range_constexpr(self.consumer.nb):
            dst = pf.iterator + (4 + block) * 512 + tid * 4
            _m8.copy_g2s(dst, at(blocks, token, block, h))
        cute.arch.cp_async_commit_group()

    def merge_buffers(self, state):
        """(W_UV slice [16, PITCH] BF16, latent tile [ROWS, PITCH] BF16, source weights [32]
        FP32) of a merge CTA, in the consumer's ``pf`` / ``parts`` / ``wpre`` (module doc)."""
        wpre, parts, pf = state[0], state[6], state[7]
        w = _mp.view(pf.iterator, 0, BFloat16, (COLUMNS, PITCH), (PITCH, 1))
        latent = _mp.view(parts.iterator, 0, BFloat16, (ROWS, PITCH), (PITCH, 1))
        scales = _mp.view(wpre.iterator, 0, Float32, (32,), (1,))
        return w, latent, scales

    @cute.jit
    def stage_w_uv(self, args, state, head, z, tid):
        """W_UV[head][:, 16 z .. 16 z + 15] into the merge CTA's slice (column-major rows of 512
        latents), one cp.async group: 1024 16-byte chunks, two per thread."""
        w_uv = args[17]
        w, _, _ = self.merge_buffers(state)
        for i in cutlass.range_constexpr(COLUMNS * LATENT // 8 // 512):
            c = tid + 512 * i
            j, k = c // (LATENT // 8), c % (LATENT // 8)
            _m8.copy_g2s(w.iterator + j * PITCH + 8 * k, at(w_uv, head, 8 * k, COLUMNS * z + j))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def merge_project(self, args, state, head, z, tid):
        """Merge CTA (``head``, ``z``): steps 1 and 2 of the module doc."""
        x, x_other = args[0], args[21]
        partials, lses, gate, totals, spins = args[15], args[16], args[18], args[19], args[20]
        B: cutlass.Constexpr = self.batch
        w, latent, scales = self.merge_buffers(state)
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        column0 = head * V_DIM + COLUMNS * z
        # The other parity's slice: its last readers completed, its next writer waits for this
        # grid (module doc).
        if tid < 2 * ROWS:
            row, half = tid // 2, tid % 2
            store_words(at(x_other, row, column0 + 8 * half), (Uint32(PAIR_SENT),) * 4)
        # Warp n < 2 projects columns 8 n .. 8 n + 7 of the slice; lane (g, q) rows g, 2 q, 2 q + 1.
        column = column0 + 8 * warp + 2 * quarter
        gate_word = Uint32(0)
        if (tid < 64) & (group < B):
            gate_word = load_words(at(gate, group, column), count=1, mutable=True)[0]
        # Every CTA of the cluster has started (its setup's arrive): its tile may be written.
        cute.arch.cluster_wait()
        if (tid < 128) & (z < B):
            pairs = self.merge_row(partials, lses, spins, scales, head, z, tid)
            for peer in cutlass.range_constexpr(ROWS):
                store_cluster_words(latent.iterator + z * PITCH + 4 * tid, pairs, peer)
        cute.arch.cp_async_wait_group(0)  # the W_UV slice
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if tid < 64:
            v = 8 * warp + group  # W_UV column of the slice
            none = Uint32(0)  # MMA rows 8 .. 15: no request row
            parts = zeros((2, 4), Float32)  # two accumulation chains, as DcpValue
            for c in cutlass.range_constexpr(16):
                a_words = load_words(
                    latent.iterator + group * PITCH + 32 * c + 8 * quarter, count=4, shared=True
                )
                b_words = load_words(
                    w.iterator + v * PITCH + 32 * c + 8 * quarter, count=4, shared=True
                )
                for s in cutlass.range_constexpr(2):
                    a = (a_words[2 * s], none, a_words[2 * s + 1], none)
                    b = (b_words[2 * s], b_words[2 * s + 1])
                    mma_into(parts[s, None], a, b, bf16=True)
            acc = [parts[0, k] + parts[1, k] for k in range(2)]
            if group < B:
                gates = unpack_bf16(gate_word)
                # Padded rows may hold a NaN gate; keep inactive rows zero.
                lo, hi = Float32(0), Float32(0)
                if totals[group] > 0:
                    lo, hi = (
                        Float32(BFloat16(acc[e])) * _mp.sigmoid(gates[e], False) for e in range(2)
                    )
                word = _mp.no_neg_zero(_mp.bf16x2(lo, hi))
                store_words(at(x, group, column), (word,), scope="gpu")

    @cute.jit
    def merge_row(self, partials, lses, spins, scales, head, t, tid):
        """``DcpValue.merge_row`` (row slot 0) of request row ``t`` of local head ``head`` on
        threads 0 .. 127, 4 columns each: the merged BF16 pairs of columns ``4 tid`` .. + 3."""
        D: cutlass.Constexpr = self.world
        lane, warp = tid % 32, tid // 32
        c4 = 4 * tid
        # Every source's 8 bytes of the row in flight while warp 0 polls the LSEs.
        sources = [at(partials, s, t, head, c4) for s in range(D)]
        loads = [load_volatile(source, count=2) for source in sources]
        if warp == 0:  # the row's source weights
            lse = Float32(EMPTY_LSE)
            if lane < D:
                source = at(lses, lane, t, head)
                word, late = _spin.poll_words(source, load_volatile(source), LSE_SENT)
                _spin.count_late(at(spins, SPIN_LSE), late)
                if cutlass.const_expr(self.record != 0):
                    spin_trap.expired(late, SITE_LSE, self.record, lane, t, head)
                store_words(source, (Uint32(LSE_SENT),))
                lse = _mp.bits_float(word)
                if late != 0:
                    lse = Float32(EMPTY_LSE)
            top = lse
            for shift in cutlass.range_constexpr(4, -1, -1):
                top = max(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
            weight = Float32(0)
            if lse > Float32(EMPTY_LSE):
                weight = cute.math.exp2(lse - top, fastmath=True)
            total = _mp.warp_sum(weight)
            # As dcp_direct_a2a_lse_reduce: weight * (1 / sum) (IEEE reciprocal).
            inverse = Float32(0)
            if total > 0:
                inverse = _mp.div_rn(Float32(1), total)
            scale = weight * inverse
            if lse <= Float32(EMPTY_LSE):
                scale = Float32(-1)  # no partial row: neither polled nor re-armed
            if lane < D:
                scales[lane] = scale
        _mp.named_barrier(MERGE_BARRIER, 128)  # the weights, for warps 0 .. 3
        # The chunks still unpublished at their first load are reloaded together before the
        # polls in source order (DcpValue's refresh).
        weights = [scales[s] for s in range(D)]
        loads = [_spin.refresh_words(sources[s], loads[s], PAIR_SENT, weights[s]) for s in range(D)]
        merged = zeros(4, Float32)
        for s in cutlass.range_constexpr(D):
            w = weights[s]
            if w >= Float32(0):
                self.poll_pair(sources[s], loads[s], w, merged, spins, s, t, head)
        return tuple(
            _mp.pack_pair(merged[2 * e], merged[2 * e + 1], policy=None) for e in range(2)
        )

    @cute.jit
    def poll_pair(self, source, words, w, merged, spins, s, t, head):
        """``DcpValue.poll_pair``: polls a BF16 partial chunk until published, re-arms it and adds
        it, weighted by ``w``, to ``merged``."""
        n: cutlass.Constexpr = len(words)
        *words, late = _spin.poll_words(source, words, PAIR_SENT)
        _spin.count_late(at(spins, SPIN_ROWS), late)
        if cutlass.const_expr(self.record != 0):
            spin_trap.expired(late, SITE_PAIR, self.record, s, t, head)
        store_words(source, (Uint32(PAIR_SENT),) * n)
        for e in cutlass.range_constexpr(n):
            lo, hi = unpack_bf16(words[e])
            merged[2 * e] += w * lo
            merged[2 * e + 1] += w * hi

    @cute.jit
    def produce(self, args, smem, wr, cta, physical, rank):
        """``OprojProducer.compute`` with ``x_shared`` and ``polled_x``: x polled until every word
        is written (bounded: ``spin_trap`` site ``SITE_X``), into shared memory, the MMAs, the K
        parts reduced in order and multicast."""
        P = self.producer
        x = args[0]
        red, xs, _ = smem
        M: cutlass.Constexpr = x.shape[0]
        tid, _, _ = cute.arch.thread_idx()
        per_row: cutlass.Constexpr = P.k // 8
        chunks: cutlass.Constexpr = M * per_row
        rounds: cutlass.Constexpr = (chunks + 511) // 512
        held = cute.make_rmem_tensor((rounds, 4), Uint32)
        for i in cutlass.range_constexpr(rounds):
            chunk = tid + i * 512
            if chunk < chunks:
                token, col = chunk // per_row, chunk % per_row
                words = _m8.load_words(at(x, token, col * 8), count=4, volatile=True)
                for v in cutlass.range_constexpr(4):
                    held[i, v] = words[v]
        pending = Int32(0)
        for i in cutlass.range_constexpr(rounds):
            if tid + i * 512 < chunks:  # noqa: SIM102
                if not _m8.published(held[i, v] for v in range(4)):
                    pending |= 1 << i
        spins, since = spin_trap.start()
        # The loops below carry only DSL values (``pending``, ``spins``, ``since``): the producer is
        # reached through ``self``, not a local.
        if cutlass.const_expr(P.vote):
            while not cute.arch.vote_all_sync(pending == 0):
                if cutlass.const_expr(self.record != 0):
                    spins, since = spin_trap.tick(
                        spins, since, SITE_X, self.record, physical, rank, pending
                    )
                pending = self.producer.repoll_x(x, held, pending, rounds, False)
        else:
            while pending != 0:
                if cutlass.const_expr(self.record != 0):
                    spins, since = spin_trap.tick(
                        spins, since, SITE_X, self.record, physical, rank, pending
                    )
                pending = self.producer.repoll_x(x, held, pending, rounds, False)
        for i in cutlass.range_constexpr(rounds):
            chunk = tid + i * 512
            if chunk < chunks:
                token, col = chunk // per_row, chunk % per_row
                dst = xs.iterator + (token * P.x_stride + col) * 4
                _m8.store_shared_words(dst, tuple(held[i, v] for v in range(4)))
        cute.arch.sync_threads()
        xr = cute.make_rmem_tensor((1, P.kblocks, 4), Uint32)  # unused: x is in shared memory
        acc = P.multiply(smem, wr, xr, 0, 0, M)
        P.store_partials(red, acc, 1)
        cute.arch.sync_threads()
        P.publish(args, red, cta, 0)


def arm(buffer: torch.Tensor) -> torch.Tensor:
    """Fill a BF16 x buffer with ``PAIR_SENT`` words (every word unwritten)."""
    buffer.view(torch.int32).fill_(PAIR_SENT - (1 << 32))
    return buffer


@cache
def compile_value_oproj(
    batch, nb, has_delta, uv, pitch, world, local, cap, drain, vote, record, device
):
    """``DsaOprojFused`` of ``batch`` rows for an AttnRes of ``nb`` blocks (``has_delta``: the
    prefix is added), ``uv`` the (shape, stride, dtype) of ``W_UV``, ``pitch`` the gate's row
    pitch, the ``DcpMailbox`` of ``world`` ranks x ``cap`` rows x ``local`` heads; ``drain`` /
    ``vote``: MR8's ``DRAIN`` / ``OPROJ_VOTE`` at this batch; ``record``: the ``spin_trap``
    record address."""
    fake, dynamic = _host.fake, _host.dynamic_fake
    f32, i32 = torch.float32, torch.int32
    x, w = fake((batch, K), (K, 1)), fake((HIDDEN, K), (K, 1))
    box = fake((world, 64, HIDDEN), (64 * HIDDEN, HIDDEN, 1))
    row, vector, eps = dynamic(2), fake((HIDDEN,), (1,)), Float32(1e-6)
    rows = (row, dynamic(3), vector, vector, vector, row, None, None, eps, eps)
    merge = (
        fake((world, cap, local, LATENT), None, torch.bfloat16),
        fake((world, cap, local), None, f32, 4),
        fake(*uv),
        fake((batch, local * V_DIM), (pitch, 1)),
        fake((batch,), None, i32, 4),
        fake((4,), None, i32, 4),
        fake((ROWS, K), (K, 1)),
    )
    kernel = DsaOprojFused(
        nb,
        has_delta=has_delta,
        world=world,
        local=local,
        batch=batch,
        drain=drain,
        vote=vote,
        record=record,
    )
    with torch.accelerator.device_index(device):
        return cute.compile(
            kernel, x, w, box, Int64(0), Int32(0), *rows, *merge, make_fake_stream()
        )

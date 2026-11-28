# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 all-gather / all-to-all of rows in a group of ranks over an MNNVL Lamport workspace (SM100 / SM103).

One persistent kernel serves both collectives of a group of ``world`` ranks. Rank ``s`` holds ``x_s`` (FP32 words, a
row of ``width`` words split into ``sections`` equal sections of ``sw = width / sections`` words); every rank gets
``out [m, world * width]``::

    all-gather (bcast):  out_d[t, c * world * sw + s * sw + j] = x_s[t, c * sw + j]             x_s [m, width]
    all-to-all:          out_d[t, c * world * sw + s * sw + j] = x_s[t * world + d, c * sw + j]  x_s [m * world, width]

With one section the all-gather concatenates the ranks' rows along the last dim and the all-to-all returns column
shards to their rows' owners; with ``sections`` > 1 each section of an output row is the concatenation of every
rank's shard of that section (a column-parallel ``[q | k | v | g]`` comes back in ``[q | k | v | g]`` order).

``ctas`` CTAs (one per SM by default) stride over 16-byte vectors, ``unroll`` of them per thread in flight, the
unrolled vectors of a warp contiguous so that every store moves whole 32-byte sectors over NVLink. A thread first
sends: it loads its vectors of this rank's rows, writes this rank's share of ``out`` and stores the other ranks'
shares (``-0.0`` words as ``+0.0``: ``0x80000000`` is the Lamport sentinel) into their current Lamport buffer through
their unicast mappings. The receive buffer holds only the other ranks' vectors, contiguous per output row. Then
the thread re-arms the previous call's buffer while the peers' vectors are in flight, and polls its vectors of the
current buffer (``unroll`` reads in flight) until no word holds the sentinel, writing them into ``out``. Each CTA
counts itself on the workspace's arrival counter once all its threads have read the rotation state; the last one
counted advances the state when it finishes. Calls on a workspace must be made in the same order on every rank of
the group, ordered by stream; a call replays from a CUDA graph. Every CTA polls vectors any peer CTA may send, so
the grid must be co-resident: at most one CTA of up to 1024 threads per SM.

GB300, one pair of GPUs of a tray, CUDA graph, us per call (BF16 rows as FP32 words; 100 rows per rank): all-gather
of [100, 7168] 7.6, all-to-all of the [q | k | v | g] column halves of 200 rows (4.9 MB each way) 13.0; about 5 us
of each call is its latency chain (grid wait, loads, the NVLink hop, the receive), the rest moves at 750-820 GB/s.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm import lamport_ar as _lam
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm.lamport_allgather import (
    sanitize_f32,
)

#: Threads per CTA.
THREADS = 512
#: 16-byte vectors in flight per thread.
UNROLL = 2


@cute.jit
def _settle_v4(addr: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32):
    """The 16-byte Lamport vector at ``addr`` given its first read ``w0..w3``, re-polled until it has arrived."""
    w = (w0, w1, w2, w3)
    while _lam.clean4(*w) == 0:
        w = _lam.ld_volatile_v4(addr)
    return w


class LamportGroupExchangeKernel:
    """See the module docstring."""

    def __init__(
        self,
        world: int,
        width: int,
        sections: int,
        bcast: bool,
        use_pdl: bool,
        threads: int = THREADS,
        unroll: int = UNROLL,
    ):
        if width % sections or (width // sections) % 4:
            raise ValueError(
                f"width={width} must split into {sections} sections of whole 16-byte vectors"
            )
        self.W = world
        self.width = width
        self.sections = sections
        self.bcast = bcast
        self.vrow = width // 4  # vectors of a source row
        self.sw_vecs = self.vrow // sections
        self.ovecs = world * self.vrow  # vectors of an output row
        self.use_pdl = use_pdl
        self.threads = threads
        self.unroll = unroll

    @cute.jit
    def __call__(
        self,
        x: Int64,
        uc_local: Int64,
        peers: Int64,
        flags: Int64,
        out: Int64,
        m: Int32,
        rank: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(x, uc_local, peers, flags, out, m, rank, ctas).launch(
            grid=(ctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    def _out_vec(self, v, s):
        """Output vector of vector ``v`` of rank ``s``'s shard row."""
        c = v // self.sw_vecs
        return c * (self.W * self.sw_vecs) + s * self.sw_vecs + (v - c * self.sw_vecs)

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        uc_local: Int64,
        peers: Int64,
        flags: Int64,
        out: Int64,
        m: Int32,
        rank: Int32,
        ctas: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        arrivals = _lam.arrivals_addr(flags)
        row_bytes = self.ovecs * 16
        per_row = (self.W - 1) * self.vrow  # remote vectors of an output row
        if cutlass.const_expr(self.use_pdl):
            _lam.griddepcontrol_wait()
        cur, dirty, nbytes, dstages, c0, c1, c2, c3 = _lam.lamport_state(flags)
        if cutlass.const_expr(self.use_pdl):
            _lam.griddepcontrol_launch_dependents()
        gtid = bidx * self.threads + tidx
        nthr = ctas * self.threads
        local = _lam.stage_addr(uc_local, cur, Int32(0), 1, nbytes)
        peer_bufs = []
        for d in cutlass.range_constexpr(self.W):
            peer_bufs.append(
                _lam.stage_addr(
                    Int64(_lam.ld_nc_u64(peers + Int64(d) * 8)), cur, Int32(0), 1, nbytes
                )
            )
        # ---- send: this rank's shard vectors, `unroll` per thread in flight, warp-contiguous
        n_send = m * self.vrow
        for base in cutlass.range(gtid, n_send, nthr * self.unroll, unroll=1):
            words = []
            for u in cutlass.range_constexpr(self.unroll):
                i = base + u * nthr
                ii = i - (i - (n_send - 1)) * Int32(i >= n_send)  # clamped: loaded, not stored
                t = ii // self.vrow
                v = ii - t * self.vrow
                for d in cutlass.range_constexpr(1 if self.bcast else self.W):
                    row = t if self.bcast else t * self.W + d
                    words.append(_lam.ld_v4(x + (Int64(row) * self.width + Int64(v) * 4) * 4))
            for u in cutlass.range_constexpr(self.unroll):
                i = base + u * nthr
                if i < n_send:
                    t = i // self.vrow
                    v = i - t * self.vrow
                    for d in cutlass.range_constexpr(self.W):
                        w = words[u if self.bcast else u * self.W + d]
                        if d == rank:
                            off = Int64(t) * row_bytes + Int64(self._out_vec(v, rank)) * 16
                            _lam.st_v4(out + off, w[0], w[1], w[2], w[3])
                        else:
                            sidx = rank - Int32(rank > d)  # this rank's slot at peer d
                            boff = (Int64(t) * per_row + Int64(sidx) * self.vrow + Int64(v)) * 16
                            _lam.st_v4(
                                peer_bufs[d] + boff,
                                sanitize_f32(w[0]),
                                sanitize_f32(w[1]),
                                sanitize_f32(w[2]),
                                sanitize_f32(w[3]),
                            )
        # every thread of the CTA has read the state (its stores used it): count the CTA
        cute.arch.barrier()
        old = Int32(0)
        if tidx == 0:
            old = _lam.atom_add_acq_rel_u32(arrivals, Int32(1))
        # ---- re-arm the previous call's buffer while the peers' vectors are in flight
        _lam.clear_dirty_stages(uc_local, dirty, nbytes, dstages, c0, c1, c2, c3, gtid, nthr)
        # ---- receive: the other ranks' vectors (contiguous in the buffer), `unroll` polls per thread in flight
        n_recv = m * per_row
        for base in cutlass.range(gtid, n_recv, nthr * self.unroll, unroll=1):
            words = []
            for u in cutlass.range_constexpr(self.unroll):
                i = base + u * nthr
                ii = i - (i - (n_recv - 1)) * Int32(i >= n_recv)
                words.append(_lam.ld_volatile_v4(local + Int64(ii) * 16))
            for u in cutlass.range_constexpr(self.unroll):
                i = base + u * nthr
                if i < n_recv:
                    w = _settle_v4(local + Int64(i) * 16, *words[u])
                    t = i // per_row
                    r = i - t * per_row
                    sidx = r // self.vrow
                    v = r - sidx * self.vrow
                    s = sidx + Int32(sidx >= rank)
                    off = Int64(t) * row_bytes + Int64(self._out_vec(v, s)) * 16
                    _lam.st_v4(out + off, w[0], w[1], w[2], w[3])
        # ---- the CTA that read the state last hands the workspace to the next call
        if tidx == 0:
            if old == ctas - 1:
                _lam.st_v4(flags, (cur + 1) % _lam.NUM_BUFFERS, cur, nbytes, Int32(1))
                _lam.st_v4(flags + Int64(16), m * (per_row * 16), Int32(0), Int32(0), Int32(0))
                _lam.st_u32(arrivals, Int32(0))


def required_bytes(rows: int, width: int, world: int) -> int:
    """Bytes of one Lamport buffer a call of ``rows`` output rows of ``world * width`` FP32 words needs (the
    other ranks' shards only)."""
    return rows * (world - 1) * width * 4


@functools.cache
def _compiled(
    world: int, width: int, sections: int, bcast: bool, use_pdl: bool, threads: int, unroll: int
):
    kernel = LamportGroupExchangeKernel(world, width, sections, bcast, use_pdl, threads, unroll)
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        a,
        a,
        a,
        a,
        a,
        Int32(1),
        Int32(0),
        Int32(1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def run_group_exchange(
    ws: "_lam.LamportWorkspace",
    x: torch.Tensor,
    out: torch.Tensor,
    sections: int = 1,
    bcast: bool = False,
    use_pdl: bool = True,
    threads: int = THREADS,
    unroll: int = UNROLL,
) -> torch.Tensor:
    """All-gather (``bcast``: ``x [m, width]``) or all-to-all (``x [m * world, width]``, row ``t * world + d`` for
    rank ``d``) of FP32 words into ``out [m, world * width]``. One worker CTA per SM: every worker CTA must be
    resident at once, as each polls vectors any peer CTA may send."""
    world = ws.world
    rows, width = x.shape
    m = rows if bcast else rows // world
    if (not bcast and rows != m * world) or out.shape != (m, world * width):
        raise ValueError(
            f"x {tuple(x.shape)} / out {tuple(out.shape)} do not match a group of {world}"
        )
    if required_bytes(m, width, world) > ws.bytes_per_buffer:
        raise ValueError(
            f"Lamport buffer of {ws.bytes_per_buffer} B too small for {m} x {world} x {width}"
        )
    ctas = _num_sms(x.device.index if x.device.index is not None else torch.cuda.current_device())
    _compiled(world, width, sections, bcast, use_pdl, threads, unroll)(
        Int64(x.data_ptr()),
        Int64(ws.uc_local),
        Int64(ws.peer_ptrs.data_ptr()),
        Int64(ws.flags.data_ptr()),
        Int64(out.data_ptr()),
        Int32(m),
        Int32(ws.rank),
        Int32(ctas),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return out

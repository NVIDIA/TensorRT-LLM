# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MNNVL all-reduce whose input the producer kernel pushes (SM100 / SM103 / SM107).

The MNNVL all-reduce (``trtllm::mnnvl_fusion_allreduce``, ``mnnvlAllreduceKernels.cu``) starts
by copying every rank's BF16 input rows into Lamport buffers: the two-shot kernel sends token row
``t`` to slot ``(t // W, rank)`` of the scatter stage of its owner ``t % W``, the one-shot kernel
to slot ``(t, rank)`` of every rank's buffer (multicast). Here a producer kernel stores its output
straight into those slots (:func:`push_slot`, :func:`push_v4`; the FP8 block-scale low-M GEMM's
all-reduce push epilogue) and :class:`PushedAllReduceFinishKernel` runs the rest of the all-reduce
on the same workspace:

* two-shot: the owner of token ``t`` polls its ``W`` slots until no 32-bit word holds the sentinel
  ``0x80000000``, sums them in FP32 in rank order, rounds to BF16, replaces ``-0`` by ``+0`` and
  multicasts the row into the broadcast stage of every rank; every rank then polls the broadcast
  rows and copies them to the output (the gather);
* one-shot: every rank polls the ``W`` slots of each token and sums them (chunks of eight ranks
  above eight ranks, like the C++ kernel).

The finish kernel clears the previous call's buffer back to the sentinel and advances the
workspace's rotation state (current / dirty buffer, bytes to clear, CTA arrival counter) exactly
like the C++ kernels, so pushed and C++ calls can alternate on one workspace. Pushed values are
sanitized (``-0`` sent as ``+0``) like the C++ kernels' inputs, and the sums use the C++ order
with round-to-nearest-even, so the output is bitwise that of the C++ all-reduce.

A two-shot call may finish without the gather: the result rows then stay in every rank's
broadcast stage, where the kernel that runs next reads them with :func:`poll_result_v4` (or
:func:`run_gather` copies them into a tensor). They live until the next all-reduce on the
workspace re-arms that buffer.

A consumer kernel may also finish a call itself (its finish is then left pending): it reads the
rotation state, sums the slots of its rows with :func:`reduce_slots_v4` (one-shot) or
:func:`finish_two_shot_v4` (two-shot: the owner reduces and multicasts its rows, the others poll
them), re-arms the previous call's buffer with :func:`clear_dirty_stages` and counts its CTAs with
:func:`arrive_finish` (or :func:`arrive_count` once its polls are done and :func:`arrive_rotate`
after its grid wait), whose last arrival advances the state. The sums and the state are the finish
kernel's, so the consumer reads the bits the finish kernel would have stored.
:class:`PushedAllReduceCopyKernel` pushes a tensor for a producer without a push epilogue.

An all-gather call (one-shot) carries, after its ``W`` slots per row, one all-gather row per token
(:func:`all_gather_row`) into which every rank multicasts its columns of a column-parallel
output ``y``; its finish gives ``bf16(y + bf16(slot sum))``. The slot sums may follow the order
of another C++ transport (``chunk``; ``sanitize``: ``-0`` replaced by ``+0`` as the two-shot
kernel does), so the call reproduces the C++ all-reduce of the slots at their size.

Ordering: the producer reads the rotation state after its own ``griddepcontrol.wait`` and
releases its dependent (the finish kernel) only after that wait; the finish kernel's workers then
read the state and poll without waiting for the producer, and only its updater CTA waits for the
producer before it advances the state (details in :class:`PushedAllReduceFinishKernel`). Calls
that share a workspace must be ordered by streams (or events) on every rank, every rank must make
the same sequence of calls, and every PDL kernel between two calls must execute
``griddepcontrol.wait``.

The workspace layout and rotation protocol are those of ``LamportFlags`` and
``LamportBufferLayout`` in ``cpp/tensorrt_llm/common/lamportUtils.cuh``; the constants below mirror
them (``test_mnnvl_pushed_allreduce.py`` checks them against the state ``ops.py`` initializes).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op

from .raw_ptx import acc_bf16x2 as _acc_bf16x2
from .raw_ptx import add_f32 as _add_f32
from .raw_ptx import asm as _asm
from .raw_ptx import atom_add_acq_rel_u32 as _atom_add_acq_rel_u32
from .raw_ptx import atom_add_relaxed_u32 as _atom_add_relaxed_u32
from .raw_ptx import iv as _iv
from .raw_ptx import ld_acquire_u32 as _ld_acquire_u32
from .raw_ptx import ld_nc_u64 as _ld_nc_u64
from .raw_ptx import ld_v4 as _ld_v4
from .raw_ptx import ld_volatile_v4, st_v4
from .raw_ptx import lv as _lv
from .raw_ptx import multimem_st_v4 as _multimem_st_v4
from .raw_ptx import pack_bf16x2 as _pack_bf16x2
from .raw_ptx import red_release_add as _red_release_add
from .raw_ptx import sanitize_bf16x2 as _sanitize_bf16x2
from .raw_ptx import st_u32 as _st_u32
from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

if TYPE_CHECKING:
    from ...distributed.ops import LamportWorkspaceView

#: Threads per CTA of the finish and gather kernels; one thread owns one 16-byte vector.
THREADS = 128
#: Number of Lamport buffers in a workspace (rotation period).
NUM_BUFFERS = 3
#: World sizes the C++ MNNVL kernels dispatch (and the finish kernel's chunked sums assume).
WORLD_SIZES = (2, 4, 8, 16, 32, 64)
#: Word of the CTA arrival counter in the workspace's state flags (uint32 [9]: current buffer,
#: dirty buffer, bytes per buffer, dirty stages, bytes to clear per stage x4, arrivals).
FLAG_ARRIVALS = 8


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def _st_sentinel_v4(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nmov.b32 t, 0x80000000;\nst.global.v4.b32 [$0], {t, t, t, t};\n}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def arrived_v4(w0, w1, w2, w3, *, loc=None, ip=None):
    """1 if none of the four words is the empty-slot sentinel ``0x80000000`` (the vector arrived)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, $1, 0x80000000;\nsetp.ne.and.b32 p, $2, 0x80000000, p;\n"
            "setp.ne.and.b32 p, $3, 0x80000000, p;\nsetp.ne.and.b32 p, $4, 0x80000000, p;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# Device functions for producers and consumers
# ---------------------------------------------------------------------------


@cute.jit
def lamport_state(flags_addr: Int64):
    """``(cur, dirty, bytes_per_buffer, dirty_stages, clear0, clear1, clear2, clear3)`` of the
    workspace, read past L1. Valid once the previous call on the workspace has completed."""
    f = ld_volatile_v4(flags_addr)
    c = ld_volatile_v4(flags_addr + Int64(16))
    return f[0], f[1], f[2], f[3], c[0], c[1], c[2], c[3]


@cute.jit
def stage_addr(base: Int64, index: Int32, stage: Int32, num_stages: int, bytes_per_buffer: Int32):
    """Address of ``stage`` of Lamport buffer ``index`` (``LamportBufferLayout::getStagePtr``)."""
    stage_bytes = Int64(bytes_per_buffer // num_stages)
    return base + Int64(index * num_stages + stage) * stage_bytes


@cute.jit
def _clear_stage(addr: Int64, nbytes: Int32, gtid: Int32, nthreads: Int32):
    """Re-arm ``nbytes`` (rounded up to 16) at ``addr`` with the sentinel, grid-strided."""
    nvec = (nbytes + 15) // 16
    for i in cutlass.range(gtid, nvec, nthreads, unroll=1):
        _st_sentinel_v4(addr + Int64(i) * 16)


@cute.jit
def push_slot(
    peer_ptrs: Int64,
    mc_base: Int64,
    cur: Int32,
    nbytes: Int32,
    tok: Int32,
    rank: Int32,
    elem: Int32,
    world: int,
    hidden: int,
    one_shot: bool,
):
    """Address where ``rank`` stores its values ``[elem, elem + 8)`` of token ``tok`` in the
    current buffer ``cur`` (``nbytes`` per buffer, :func:`lamport_state`): slot ``(tok // W,
    rank)`` of the owner's scatter stage (two-shot, through the owner's unicast mapping), or slot
    ``(tok, rank)`` of every rank's buffer (one-shot, multicast address)."""
    row_bytes = hidden * 2
    addr = Int64(0)
    if cutlass.const_expr(one_shot):
        addr = stage_addr(mc_base, cur, Int32(0), 1, nbytes) + Int64(tok * world + rank) * row_bytes
    else:
        owner = _ld_nc_u64(peer_ptrs + Int64(tok % world) * 8)
        addr = (
            stage_addr(owner, cur, Int32(0), 2, nbytes)
            + Int64((tok // world) * world + rank) * row_bytes
        )
    return addr + Int64(elem) * 2


@cute.jit
def push_v4(addr: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32, one_shot: bool):
    """Store one 16-byte vector (eight BF16 values) into a Lamport slot, ``-0`` sent as ``+0``."""
    s = [_sanitize_bf16x2(w) for w in (w0, w1, w2, w3)]
    if cutlass.const_expr(one_shot):
        _multimem_st_v4(addr, *s)
    else:
        st_v4(addr, *s)


@cute.jit
def poll_result_v4(uc_local: Int64, flags: Int64, tok: Int32, byte_off: Int32, hidden: int):
    """16 bytes at ``byte_off`` of row ``tok`` of the result of the two-shot all-reduce that
    completed last on the workspace without its gather, polled until they arrived.

    For a kernel that runs after the finish kernel (after its own ``griddepcontrol.wait``): the
    rotation has then advanced, so the call's buffer is the state's dirty one. The next
    all-reduce on the workspace re-arms it, so only a kernel ordered before that call may read
    it."""
    f = ld_volatile_v4(flags)
    row = stage_addr(uc_local, f[1], Int32(1), 2, f[2]) + Int64(tok * hidden) * 2
    w = ld_volatile_v4(row + Int64(byte_off))
    while arrived_v4(*w) == 0:
        w = ld_volatile_v4(row + Int64(byte_off))
    return w


def reduce_chunk(world: int, one_shot: bool) -> int:
    """Ranks per partial sum of the C++ kernel of the ``one_shot`` (else two-shot) transport:
    chunks of eight ranks for the one-shot kernel above eight ranks, up to sixteen ranks in one
    chunk for the two-shot kernel."""
    return (world if world <= 8 else 8) if one_shot else min(world, 16)


#: Largest group whose slots :func:`reduce_slots_v4` polls in one round (``4 * world`` words in
#: registers); larger groups are polled one chunk of ranks at a time.
POLL_ALL_MAX_WORLD = 16


@cute.jit
def reduce_slots_v4(base: Int64, slot_bytes: int, world: int, chunk: int):
    """Sum the ``world`` 16-byte slots at ``base + r * slot_bytes`` in the C++ order (``chunk``
    ranks per partial sum, :func:`reduce_chunk`); returns eight FP32 values (four BF16 pairs).
    Up to :data:`POLL_ALL_MAX_WORLD` ranks every slot is polled in one round until every word has
    arrived, so the sum follows the last rank's data by one load round trip; larger groups poll
    one chunk at a time, so only one chunk's slots are live."""
    poll = world if world <= POLL_ALL_MAX_WORLD else chunk
    acc = [cutlass.Float32(0.0)] * 8
    for p0 in cutlass.range_constexpr(0, world, poll):
        words = cute.make_rmem_tensor((poll * 4,), Int32)
        ready = Int32(0)
        while ready == 0:
            ready = Int32(1)
            for r in cutlass.range_constexpr(poll):
                w = ld_volatile_v4(base + Int64((p0 + r) * slot_bytes))
                for e in cutlass.range_constexpr(4):
                    words[r * 4 + e] = w[e]
                ready = ready & arrived_v4(*w)
        for c0 in cutlass.range_constexpr(0, poll, chunk):
            part = [cutlass.Float32(0.0)] * 8
            for r in cutlass.range_constexpr(c0, c0 + chunk):
                for q in cutlass.range_constexpr(4):
                    part[2 * q], part[2 * q + 1] = _acc_bf16x2(
                        words[r * 4 + q], part[2 * q], part[2 * q + 1]
                    )
            for i in cutlass.range_constexpr(8):
                acc[i] = _add_f32(acc[i], part[i])
    return acc


@cute.jit
def all_gather_row(
    base: Int64, cur: Int32, nbytes: Int32, m: Int32, tok: Int32, world: int, hidden: int
):
    """Address of token ``tok``'s all-gather row in buffer ``cur`` of an all-gather call of ``m``
    rows (after the ``m * world`` one-shot slots); ``base``: unicast or multicast base."""
    return stage_addr(base, cur, Int32(0), 1, nbytes) + Int64(m * world + tok) * (hidden * 2)


@cute.jit
def poll_v4(addr: Int64):
    """16 bytes of a Lamport buffer, polled until they arrived."""
    w = ld_volatile_v4(addr)
    while arrived_v4(*w) == 0:
        w = ld_volatile_v4(addr)
    return w


@cute.jit
def add_bf16x8(a, b):
    """``bf16(a + b)`` of two vectors of eight BF16 values (four packed words each), FP32
    round-to-nearest add, as PyTorch adds two BF16 tensors."""
    out = []
    for q in cutlass.range_constexpr(4):
        lo, hi = _acc_bf16x2(a[q], cutlass.Float32(0.0), cutlass.Float32(0.0))
        lo, hi = _acc_bf16x2(b[q], lo, hi)
        out.append(_pack_bf16x2(lo, hi))
    return out


@cute.jit
def round_sum_v4(acc, sanitize: bool):
    """The BF16 words of the FP32 slot sum ``acc`` (``sanitize``: ``-0`` replaced by ``+0``)."""
    words = [_pack_bf16x2(acc[2 * q], acc[2 * q + 1]) for q in range(4)]
    if cutlass.const_expr(sanitize):
        words = [_sanitize_bf16x2(w) for w in words]
    return words


@cute.jit
def one_shot_slots(uc_local: Int64, cur: Int32, nbytes: Int32, tok: Int32, world: int, hidden: int):
    """Address of rank 0's slot of token ``tok`` in buffer ``cur`` of a one-shot call (the slots
    of the other ranks follow, ``hidden * 2`` bytes apart)."""
    return stage_addr(uc_local, cur, Int32(0), 1, nbytes) + Int64(tok * world) * (hidden * 2)


@cute.jit
def clear_dirty_stages(
    uc_local: Int64,
    dirty: Int32,
    nbytes: Int32,
    dstages: Int32,
    c0: Int32,
    c1: Int32,
    c2: Int32,
    c3: Int32,
    gtid: Int32,
    nthreads: Int32,
):
    """Re-arm every stage the previous call on the workspace used (buffer ``dirty``, ``dstages``
    stages of ``c0..c3`` bytes, from :func:`lamport_state`), grid-strided over ``nthreads``.
    Safe as soon as every grid before the call's producer has completed."""
    clears = (c0, c1, c2, c3)
    for s in cutlass.range_constexpr(4):
        if s < dstages:
            _clear_stage(
                uc_local + Int64(dirty * dstages + s) * Int64(nbytes // dstages),
                clears[s],
                gtid,
                nthreads,
            )


@cute.jit
def _write_next_state(
    flags: Int64,
    cur: Int32,
    nbytes: Int32,
    m: Int32,
    world: int,
    hidden: int,
    one_shot: bool,
    all_gather: bool = False,
):
    """The state after a call of ``m`` rows in buffer ``cur``: the next buffer is current, ``cur``
    dirty with the stages and bytes the transport used (``LamportFlags::waitAndUpdate``; an
    all-gather call also its all-gather rows)."""
    row_bytes = hidden * 2
    clear0 = m * ((world + (1 if all_gather else 0)) * row_bytes)
    clear1 = Int32(0)
    if cutlass.const_expr(not one_shot):
        clear0 = ((m + (world - 1)) // world) * (world * row_bytes)
        clear1 = m * row_bytes
    st_v4(flags, (cur + 1) % NUM_BUFFERS, cur, nbytes, Int32(1 if one_shot else 2))
    st_v4(flags + Int64(16), clear0, clear1, Int32(0), Int32(0))


@cute.jit
def arrive_finish(
    flags: Int64,
    total: Int32,
    cur: Int32,
    nbytes: Int32,
    m: Int32,
    world: int,
    hidden: int,
    one_shot: bool,
    all_gather: bool = False,
):
    """Count one CTA of a consumer grid of ``total`` CTAs that finishes a call of ``m`` rows (one
    thread per CTA, once every thread of the CTA has read the state and the call's producer has
    completed); the last CTA writes the next call's state and resets the counter, as
    :class:`PushedAllReduceFinishKernel`'s updater does."""
    old = _atom_add_acq_rel_u32(flags + Int64(FLAG_ARRIVALS * 4), Int32(1))
    arrive_rotate(flags, old, total, cur, nbytes, m, world, hidden, one_shot, all_gather)


@cute.jit
def arrive_count(flags: Int64):
    """The counting half of :func:`arrive_finish` for a consumer that counts its CTA before its
    grid wait (one thread per CTA, once every thread of the CTA has read the state): a relaxed
    atomic whose result only :func:`arrive_rotate` reads, so the thread does not stall on it."""
    return _atom_add_relaxed_u32(flags + Int64(FLAG_ARRIVALS * 4), Int32(1))


@cute.jit
def arrive_rotate(
    flags: Int64,
    old: Int32,
    total: Int32,
    cur: Int32,
    nbytes: Int32,
    m: Int32,
    world: int,
    hidden: int,
    one_shot: bool,
    all_gather: bool = False,
):
    """The rotation half of :func:`arrive_finish`: the CTA whose count ``old`` was the last of
    ``total`` writes the next call's state and resets the counter. Only after the call's
    producer has completed (the caller has passed ``griddepcontrol.wait``)."""
    if old == total - 1:
        _write_next_state(flags, cur, nbytes, m, world, hidden, one_shot, all_gather)
        _st_u32(flags + Int64(FLAG_ARRIVALS * 4), Int32(0))


@cute.jit
def finish_two_shot_v4(
    uc_local: Int64,
    mc_base: Int64,
    cur: Int32,
    nbytes: Int32,
    tok: Int32,
    byte_off: Int32,
    rank: Int32,
    store: Int32,
    world: int,
    hidden: int,
):
    """16 bytes at ``byte_off`` of row ``tok`` of the two-shot call in buffer ``cur``, finished by
    the caller: on the row's owner (``tok % world == rank``) the C++-order sum of its scatter
    slots, rounded, ``-0`` replaced by ``+0`` and (where ``store != 0``) multicast into every
    rank's broadcast stage; elsewhere the broadcast row, polled until it arrived. Both are the
    bits :class:`PushedAllReduceFinishKernel` broadcasts."""
    w0 = Int32(0)
    w1 = Int32(0)
    w2 = Int32(0)
    w3 = Int32(0)
    if tok % world == rank:
        scatter = stage_addr(uc_local, cur, Int32(0), 2, nbytes)
        acc = reduce_slots_v4(
            scatter + Int64((tok // world) * world * hidden) * 2 + Int64(byte_off),
            hidden * 2,
            world,
            reduce_chunk(world, False),
        )
        w0 = _sanitize_bf16x2(_pack_bf16x2(acc[0], acc[1]))
        w1 = _sanitize_bf16x2(_pack_bf16x2(acc[2], acc[3]))
        w2 = _sanitize_bf16x2(_pack_bf16x2(acc[4], acc[5]))
        w3 = _sanitize_bf16x2(_pack_bf16x2(acc[6], acc[7]))
        if store != 0:
            bcast_mc = stage_addr(mc_base, cur, Int32(1), 2, nbytes)
            _multimem_st_v4(bcast_mc + Int64(tok * hidden) * 2 + Int64(byte_off), w0, w1, w2, w3)
    else:
        src = stage_addr(uc_local, cur, Int32(1), 2, nbytes) + Int64(tok * hidden) * 2
        w = ld_volatile_v4(src + Int64(byte_off))
        while arrived_v4(*w) == 0:
            w = ld_volatile_v4(src + Int64(byte_off))
        w0 = w[0]
        w1 = w[1]
        w2 = w[2]
        w3 = w[3]
    return w0, w1, w2, w3


@cute.jit
def two_shot_token(v: Int32, m: Int32, rank: Int32, world: int):
    """Row of the ``v``-th CTA group of a grid that finishes a two-shot call of ``m`` rows on
    ``rank``: the rows this rank owns (``rank``, ``rank + world``, ...) first, then the others
    in order, so the owners' reductions are dispatched before the CTAs that poll for them."""
    n_own = (m + (world - 1) - rank) // world
    t = rank + v * world
    if v >= n_own:
        j = v - n_own
        q = j // (world - 1)
        rem = j - q * (world - 1)
        t = q * world + rem
        if rem >= rank:
            t = t + 1
    return t


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


class PushedAllReduceFinishKernel:
    """Rest of an MNNVL all-reduce of ``[m, hidden]`` BF16 rows the producer pushes.

    Grid ``m * ceil(hidden / 8 / 128) + 1`` CTAs of 128 threads. Worker CTA ``b`` owns token
    ``b // cols`` and its 16-byte vectors ``(b % cols) * 128 + tid``; the last CTA advances the
    rotation.

    Ordering. The workers read the rotation state and start polling as soon as they are
    resident, without ``griddepcontrol.wait``: the rows they poll validate themselves (no sentinel
    word), so they react the moment the last rank's data lands. Reading the state early is safe
    because the producer releases this kernel only after its own ``griddepcontrol.wait``: by then
    every earlier grid has completed, the previous all-reduce on the workspace included, so the
    state is this call's. Every CTA releases its dependents at its start, so they become resident
    once the producer has passed its wait; dependents that read data before their own wait read
    only data older than the producer. The updater CTA alone waits for the producer: the producer
    reads the state after its own wait, so the state may only change once it has completed; the
    updater then waits until every worker has counted itself (having read the state) and writes
    the next call's state, like ``LamportFlags::waitAndUpdate``. Its wait also makes this
    kernel's completion imply the producer's.

    Numerics: those of the C++ kernel of the same transport. The one-shot kernel sums chunks of
    eight ranks above eight ranks and stores the rounded sum as is; the two-shot kernel sums up to
    sixteen ranks in one chunk and replaces ``-0`` by ``+0`` (its result passes through a Lamport
    buffer). ``gather=False`` (two-shot only) leaves the result in the broadcast stages.
    ``all_gather`` (one-shot): the call carries all-gather rows ``y`` and the kernel stores
    ``bf16(y + bf16(sum))``, the sums ``chunk`` ranks at a time (``sanitize``: ``-0`` as ``+0``).
    """

    def __init__(
        self,
        world: int,
        hidden: int,
        one_shot: bool,
        gather: bool,
        use_pdl: bool,
        all_gather: bool = False,
        chunk: int = 0,
        sanitize: bool = False,
    ):
        if hidden % 8:
            raise ValueError(f"hidden={hidden} must be a multiple of 8")
        if world not in WORLD_SIZES:
            raise ValueError(f"world={world} is not one of {WORLD_SIZES}")
        if not gather and one_shot:
            raise ValueError("only the two-shot transport leaves its result in the workspace")
        if (all_gather or chunk or sanitize) and not one_shot:
            raise ValueError("all-gather calls and sum orders of another transport are one-shot")
        self.W = world
        self.H = hidden
        self.one_shot = one_shot
        self.gather = gather
        self.use_pdl = use_pdl
        self.all_gather = all_gather
        self.sanitize = sanitize
        self.vecs = hidden // 8
        self.cols = _ceil_div(self.vecs, THREADS)
        self.chunk = chunk or reduce_chunk(world, one_shot)

    @cute.jit
    def __call__(
        self,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        out: Int64,
        m: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(uc_local, mc_base, flags, out, m, rank).launch(
            grid=(m * self.cols + 1, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self, uc_local: Int64, mc_base: Int64, flags: Int64, out: Int64, m: Int32, rank: Int32
    ):
        W, H = self.W, self.H
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        workers = m * self.cols
        arrivals = flags + Int64(FLAG_ARRIVALS * 4)
        if bidx == workers:
            # Updater: the next call's state, once the producer is done with this call's.
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_launch_dependents()
                griddepcontrol_wait()
            if tidx == 0:
                cur, _, nbytes, _, _, _, _, _ = lamport_state(flags)
                arrived = _ld_acquire_u32(arrivals)
                while arrived < workers:
                    arrived = _ld_acquire_u32(arrivals)
                _write_next_state(flags, cur, nbytes, m, W, H, self.one_shot, self.all_gather)
                _st_u32(arrivals, Int32(0))
        else:
            tok = bidx // self.cols
            by = bidx - tok * self.cols
            v = by * THREADS + tidx
            cur, dirty, nbytes, dstages, c0, c1, c2, c3 = lamport_state(flags)
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_launch_dependents()
            # Count this CTA once every thread has read the state.
            cute.arch.barrier()
            if tidx == 0:
                _red_release_add(arrivals, Int32(1))
            row_bytes = H * 2
            out_v = out + Int64(tok * H + v * 8) * 2
            if cutlass.const_expr(self.one_shot):
                buf = stage_addr(uc_local, cur, Int32(0), 1, nbytes)
                if v < self.vecs:
                    acc = reduce_slots_v4(
                        buf + Int64(tok * W * H + v * 8) * 2, row_bytes, W, self.chunk
                    )
                    words = round_sum_v4(acc, self.sanitize)
                    if cutlass.const_expr(self.all_gather):
                        y = poll_v4(
                            all_gather_row(uc_local, cur, nbytes, m, tok, W, H) + Int64(v * 16)
                        )
                        words = add_bf16x8(y, words)
                    st_v4(out_v, *words)
            else:
                if v < self.vecs:
                    if tok % W == rank:
                        scatter = stage_addr(uc_local, cur, Int32(0), 2, nbytes)
                        acc = reduce_slots_v4(
                            scatter + Int64((tok // W) * W * H + v * 8) * 2,
                            row_bytes,
                            W,
                            self.chunk,
                        )
                        o = [
                            _sanitize_bf16x2(_pack_bf16x2(acc[2 * q], acc[2 * q + 1]))
                            for q in range(4)
                        ]
                        bcast_mc = stage_addr(mc_base, cur, Int32(1), 2, nbytes)
                        _multimem_st_v4(bcast_mc + Int64(tok * H + v * 8) * 2, *o)
            # Re-arm the previous call's buffer (every stage it used) while the rows travel.
            clear_dirty_stages(
                uc_local,
                dirty,
                nbytes,
                dstages,
                c0,
                c1,
                c2,
                c3,
                bidx * THREADS + tidx,
                workers * THREADS,
            )
            if cutlass.const_expr(not self.one_shot and self.gather):
                if v < self.vecs:
                    src = (
                        stage_addr(uc_local, cur, Int32(1), 2, nbytes) + Int64(tok * H + v * 8) * 2
                    )
                    w = ld_volatile_v4(src)
                    while arrived_v4(*w) == 0:
                        w = ld_volatile_v4(src)
                    st_v4(out_v, *w)


class PushedAllReduceGatherKernel:
    """Copy the result rows of the last two-shot call finished without its gather into a tensor
    (:func:`poll_result_v4` per 16-byte vector). Grid ``(m, ceil(hidden / 8 / 128))``; it waits
    for its predecessor (the finish kernel)."""

    def __init__(self, hidden: int, use_pdl: bool):
        if hidden % 8:
            raise ValueError(f"hidden={hidden} must be a multiple of 8")
        self.H = hidden
        self.use_pdl = use_pdl
        self.vecs = hidden // 8
        self.cols = _ceil_div(self.vecs, THREADS)

    @cute.jit
    def __call__(self, uc_local: Int64, flags: Int64, out: Int64, m: Int32, stream: cuda.CUstream):
        self.kernel(uc_local, flags, out, m).launch(
            grid=(m, self.cols, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=self.use_pdl
        )

    @cute.kernel
    def kernel(self, uc_local: Int64, flags: Int64, out: Int64, m: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        tok, by, _ = cute.arch.block_idx()
        v = by * THREADS + tidx
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()
            griddepcontrol_launch_dependents()
        if v < self.vecs:
            w = poll_result_v4(uc_local, flags, tok, v * 16, self.H)
            st_v4(out + Int64(tok * self.H + v * 8) * 2, *w)


class PushedAllReduceCopyKernel:
    """Push a BF16 ``[m, hidden]`` tensor into the slots of the next call on the workspace: the
    producer side of a pushed all-reduce for an output that no kernel epilogue pushes. Grid
    ``(m, ceil(width / 8 / 128))``; like the push epilogues it waits for its predecessor (which
    wrote the tensor), releases its dependent right after that wait and then reads the state.
    ``columns > 0`` (one-shot): the tensor is ``[m, columns]``, columns ``col0 ..`` of a
    column-parallel output, multicast into the all-gather rows of an all-gather call."""

    def __init__(self, world: int, hidden: int, one_shot: bool, use_pdl: bool, columns: int = 0):
        if hidden % 8 or columns % 8:
            raise ValueError(f"hidden={hidden} and columns={columns} must be multiples of 8")
        if world not in WORLD_SIZES:
            raise ValueError(f"world={world} is not one of {WORLD_SIZES}")
        if columns and not one_shot:
            raise ValueError("all-gather rows travel with the one-shot transport")
        self.W = world
        self.H = hidden
        self.one_shot = one_shot
        self.use_pdl = use_pdl
        self.columns = columns
        self.vecs = (columns or hidden) // 8
        self.cols = _ceil_div(self.vecs, THREADS)

    @cute.jit
    def __call__(
        self,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        inp: Int64,
        m: Int32,
        rank: Int32,
        col0: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(peer_ptrs, mc_base, flags, inp, m, rank, col0).launch(
            grid=(m, self.cols, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=self.use_pdl
        )

    @cute.kernel
    def kernel(
        self,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        inp: Int64,
        m: Int32,
        rank: Int32,
        col0: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tok, by, _ = cute.arch.block_idx()
        v = by * THREADS + tidx
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()
            griddepcontrol_launch_dependents()
        cur, _, nbytes, _, _, _, _, _ = lamport_state(flags)
        if v < self.vecs:
            if cutlass.const_expr(self.columns > 0):
                w = _ld_v4(inp + Int64(tok * self.columns + v * 8) * 2)
                addr = (
                    all_gather_row(mc_base, cur, nbytes, m, tok, self.W, self.H)
                    + Int64(col0 + v * 8) * 2
                )
                push_v4(addr, *w, True)
            else:
                w = _ld_v4(inp + Int64(tok * self.H + v * 8) * 2)
                addr = push_slot(
                    peer_ptrs, mc_base, cur, nbytes, tok, rank, v * 8, self.W, self.H, self.one_shot
                )
                push_v4(addr, *w, self.one_shot)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def all_gather_call_bytes(num_tokens: int, hidden: int, world: int) -> int:
    """Bytes of one Lamport buffer an all-gather call of ``[num_tokens, hidden]`` uses: the
    one-shot slots of every rank and the all-gather rows."""
    return num_tokens * hidden * (world + 1) * 2


def check_transport(
    view: LamportWorkspaceView,
    num_tokens: int,
    hidden: int,
    one_shot: bool,
    all_gather: bool = False,
) -> None:
    """Raise unless a pushed BF16 ``[num_tokens, hidden]`` call on ``view`` with the ``one_shot``
    (else two-shot) transport is the call the C++ MNNVL all-reduce makes
    (``ops.mnnvl_allreduce_transport``; an ``all_gather`` call is one-shot at every size) and
    fits the workspace."""
    from ...distributed.ops import mnnvl_allreduce_transport

    if hidden % 8:
        raise ValueError(f"hidden={hidden} must be a multiple of 8")
    cpp_one_shot, need = mnnvl_allreduce_transport(num_tokens, hidden, view.world, torch.bfloat16)
    if all_gather:
        if not one_shot:
            raise ValueError("all-gather calls are one-shot")
        cpp_one_shot, need = True, all_gather_call_bytes(num_tokens, hidden, view.world)
    if one_shot != cpp_one_shot:
        raise ValueError(
            f"{num_tokens} x {hidden} on {view.world} ranks takes the "
            f"{'one' if cpp_one_shot else 'two'}-shot transport"
        )
    if need > view.bytes_per_buffer:
        raise ValueError(
            f"Lamport buffer of {view.bytes_per_buffer} B too small for {num_tokens} x {hidden} "
            f"({need} B)"
        )


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@functools.cache
def _compiled_finish(
    world: int,
    hidden: int,
    one_shot: bool,
    gather: bool,
    use_pdl: bool,
    all_gather: bool = False,
    chunk: int = 0,
    sanitize: bool = False,
):
    kernel = PushedAllReduceFinishKernel(
        world, hidden, one_shot, gather, use_pdl, all_gather, chunk, sanitize
    )
    a = Int64(1 << 20)
    return cute.compile(kernel, a, a, a, a, Int32(1), Int32(0), _stream())


@functools.cache
def _compiled_gather(hidden: int, use_pdl: bool):
    kernel = PushedAllReduceGatherKernel(hidden, use_pdl)
    a = Int64(1 << 20)
    return cute.compile(kernel, a, a, a, Int32(1), _stream())


@functools.cache
def _compiled_push(world: int, hidden: int, one_shot: bool, use_pdl: bool, columns: int = 0):
    kernel = PushedAllReduceCopyKernel(world, hidden, one_shot, use_pdl, columns)
    a = Int64(1 << 20)
    return cute.compile(kernel, a, a, a, a, Int32(1), Int32(0), Int32(0), _stream())


def run_push(
    view: LamportWorkspaceView,
    inp: torch.Tensor,
    one_shot: bool,
    use_pdl: bool = True,
    all_gather: bool = False,
    columns: tuple[int, int] | None = None,
) -> None:
    """Push ``inp`` (contiguous BF16 ``[m, hidden]``) into the slots of the next call on
    ``view`` with the ``one_shot`` (else two-shot) transport, as a push epilogue would
    (``all_gather``: the call carries all-gather rows). ``columns = (col0, hidden)``: ``inp`` is
    ``[m, n]``, columns ``col0 .. col0 + n`` of the all-gather call's rows of ``hidden``."""
    m, n = inp.shape
    col0, hidden = columns if columns is not None else (0, n)
    if columns is not None and (not all_gather or col0 % 8 or col0 + n > hidden):
        raise ValueError(f"columns [{col0}, {col0 + n}) of {hidden} need an all-gather call")
    check_transport(view, m, hidden, one_shot, all_gather)
    _compiled_push(view.world, hidden, one_shot, use_pdl, n if columns is not None else 0)(
        Int64(view.peer_ptrs.data_ptr()),
        Int64(view.mc_base),
        Int64(view.flags.data_ptr()),
        Int64(inp.data_ptr()),
        Int32(m),
        Int32(view.rank),
        Int32(col0),
        _stream(),
    )


def run_finish(
    view: LamportWorkspaceView,
    out: torch.Tensor,
    one_shot: bool,
    use_pdl: bool = True,
    all_gather: bool = False,
    chunk: int = 0,
    sanitize: bool = False,
) -> torch.Tensor:
    """Complete the all-reduce whose input every rank pushed with the ``one_shot`` (else
    two-shot) transport into ``out`` (contiguous BF16 ``[m, hidden]``); returns ``out``.
    ``all_gather`` (one-shot): ``out = bf16(y + bf16(sum))`` with ``y`` the all-gather rows;
    ``chunk`` / ``sanitize``: the sum order of another transport (see
    :class:`PushedAllReduceFinishKernel`)."""
    m, hidden = out.shape
    check_transport(view, m, hidden, one_shot, all_gather)
    _compiled_finish(view.world, hidden, one_shot, True, use_pdl, all_gather, chunk, sanitize)(
        Int64(view.uc_local),
        Int64(view.mc_base),
        Int64(view.flags.data_ptr()),
        Int64(out.data_ptr()),
        Int32(m),
        Int32(view.rank),
        _stream(),
    )
    return out


def run_finish_in_place(
    view: LamportWorkspaceView, num_tokens: int, hidden: int, use_pdl: bool = True
) -> None:
    """Two-shot finish without the gather: the result of the pushed call stays in every rank's
    broadcast stage, where the kernel that runs next reads it with :func:`poll_result_v4`."""
    check_transport(view, num_tokens, hidden, False)
    _compiled_finish(view.world, hidden, False, False, use_pdl)(
        Int64(view.uc_local),
        Int64(view.mc_base),
        Int64(view.flags.data_ptr()),
        Int64(0),
        Int32(num_tokens),
        Int32(view.rank),
        _stream(),
    )


def run_gather(view: LamportWorkspaceView, out: torch.Tensor, use_pdl: bool = True) -> torch.Tensor:
    """Copy the result of the last two-shot call on ``view`` finished in place into ``out``
    (contiguous BF16 ``[m, hidden]``); run it before the next all-reduce on the workspace."""
    m, hidden = out.shape
    _compiled_gather(hidden, use_pdl)(
        Int64(view.uc_local),
        Int64(view.flags.data_ptr()),
        Int64(out.data_ptr()),
        Int32(m),
        _stream(),
    )
    return out


__all__ = [
    "NUM_BUFFERS",
    "WORLD_SIZES",
    "PushedAllReduceCopyKernel",
    "PushedAllReduceFinishKernel",
    "PushedAllReduceGatherKernel",
    "add_bf16x8",
    "all_gather_call_bytes",
    "all_gather_row",
    "arrive_count",
    "arrive_finish",
    "arrive_rotate",
    "arrived_v4",
    "check_transport",
    "clear_dirty_stages",
    "finish_two_shot_v4",
    "lamport_state",
    "ld_volatile_v4",
    "one_shot_slots",
    "poll_result_v4",
    "poll_v4",
    "push_slot",
    "push_v4",
    "reduce_chunk",
    "reduce_slots_v4",
    "round_sum_v4",
    "run_finish",
    "run_finish_in_place",
    "run_gather",
    "run_push",
    "st_v4",
    "two_shot_token",
]

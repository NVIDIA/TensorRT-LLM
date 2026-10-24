# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions adapted from vLLM (https://github.com/vllm-project/vllm) under the Apache-2.0
# license: the two-shot layout of
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/common/twoshot.py.
# Original copyright: Copyright contributors to the vLLM project
"""Standalone Lamport-mailbox all-reduce of BF16 rows (one-shot and two-shot), bitwise equal to
the MNNVL all-reduce (``MNNVLAllReduce``, mnnvlAllreduceKernels.cu).

Both kernels run one thread per 16-byte fragment of a row (8 BF16), ``M * H / 8`` threads, and
use one parity region of a :class:`~tensorrt_llm._torch.distributed.k3_lamport_mailbox.K3LamportArena`.

* **One-shot** (region layout ``[M][world][H]``, token-major like MNNVL's): after the PDL wait a
  thread loads its fragment, turns BF16 ``-0`` into ``+0`` and stores it to slot ``rank`` of every
  rank's region (one store to the multicast address); it then polls the ``world`` slots of its
  fragment in its own region, sums them in FP32 in MNNVL's rank order, rounds once, stores the
  output and re-arms the slots.
* **Two-shot** (column ownership: virtual rank ``o`` owns columns ``[o H / world, (o+1) H / world)``):
  every thread pushes its fragment into receive slot ``rank`` of the fragment's owner (a unicast
  store to that peer), the owner's thread polls the ``world`` receive slots, sums them, rounds,
  sanitizes and multicasts the result into every rank's gather row and re-arms its slots; every
  thread then polls its gather fragment, stores it and re-arms it. Receive layout on a real rank:
  ``[M][world][V * H / world]`` (its ``V`` owned column ranges side by side), gather ``[M][H]``
  right behind it.

Rank emulation (``world > real ranks``, for TP8/TP16 tests on one 4-GPU node): real rank ``r``
plays virtual ranks ``r, r + R, r + 2R, ...`` (``V = world / R`` of them): its input is
``[V, M, H]`` and it publishes every virtual rank's fragment, so every GPU still receives
``world`` payloads per fragment, as in a real ``world``-rank group, and every slot is published,
polled and re-armed.

Sum order (``k3_lamport.rank_order_sum``): ``acc = 0; acc += r_0; ...`` in FP32 with denormals,
rounded to BF16 with round-to-nearest-even; the one-shot sums chunks of 8 ranks above 8 ranks and
the two-shot one chunk up to 16 ranks, exactly like the MNNVL kernels.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ..blackwell.utils import TRTLLM_ENABLE_PDL
from .k3_lamport import (
    FRAGMENT_BYTES,
    FRAGMENT_ELEMS,
    PUBLISH_MULTIMEM,
    PUBLISH_ST,
    SPIN_LIMIT,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    load_v4,
    oneshot_chunk,
    poll_slots_v4,
    publish_v4,
    rank_order_sum,
    rearm_slots_v4,
    sanitize_v4,
    store_v4,
    twoshot_chunk,
)

#: When the kernel allows its PDL dependents to launch: at entry (they may run their pre-wait
#: work under the all-reduce), after the poll (where the MNNVL all-reduce triggers) or at exit.
TRIGGER_ENTRY, TRIGGER_POLL, TRIGGER_NONE = "entry", "poll", "none"
MODE_ONESHOT, MODE_TWOSHOT = "oneshot", "twoshot"
#: Largest group: a poll keeps one fragment (4 registers) per rank live.
MAX_WORLD = 16


def _check_common(
    world: int, real: int, hidden: int, threads: int, trigger: str, publish: str
) -> None:
    if world % real or world < real or world > MAX_WORLD:
        raise ValueError(
            f"world {world} must be a multiple of the real rank count {real}, at most {MAX_WORLD}"
        )
    if trigger not in (TRIGGER_ENTRY, TRIGGER_POLL, TRIGGER_NONE):
        raise ValueError(f"trigger {trigger!r}")
    if publish not in (PUBLISH_ST, PUBLISH_MULTIMEM):
        raise ValueError(f"publish {publish!r}")
    if hidden % FRAGMENT_ELEMS:
        raise ValueError(f"hidden {hidden} must be a multiple of {FRAGMENT_ELEMS}")
    frags = hidden // FRAGMENT_ELEMS
    if threads % 32 or frags % threads:
        raise ValueError(f"threads {threads} must be a multiple of 32 dividing {frags}")


class K3LamportOneShotKernel:
    """One-shot all-reduce; see the module docstring."""

    def __init__(
        self,
        world: int,
        real: int,
        hidden: int,
        threads: int,
        use_pdl: bool,
        trigger: str,
        publish: str,
        backoff_ns: int,
        spin_limit: int,
        vote: bool,
    ):
        _check_common(world, real, hidden, threads, trigger, publish)
        self.world, self.real, self.v = world, real, world // real
        self.hidden, self.threads = hidden, threads
        self.row_bytes = hidden * 2
        self.cpr = hidden // FRAGMENT_ELEMS // threads
        self.use_pdl, self.trigger, self.publish = use_pdl, trigger, publish
        self.backoff_ns, self.spin_limit, self.vote = backoff_ns, spin_limit, vote
        self.chunk = oneshot_chunk(world)

    @cute.jit
    def __call__(
        self,
        x_addr: Int64,
        out_addr: Int64,
        local_addr: Int64,
        mc_addr: Int64,
        m: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(x_addr, out_addr, local_addr, mc_addr, m, rank).launch(
            grid=(m * self.cpr, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x_addr: Int64,
        out_addr: Int64,
        local_addr: Int64,
        mc_addr: Int64,
        m: Int32,
        rank: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx // self.cpr
        frag = (bidx - row * self.cpr) * self.threads + tidx
        col = Int64(frag) * FRAGMENT_BYTES
        row_off = Int64(row) * self.row_bytes
        box_row = Int64(row) * (self.world * self.row_bytes)

        if cutlass.const_expr(self.trigger == TRIGGER_ENTRY):
            griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()

        # Publish: every virtual rank of this real rank, one 16-byte store to all ranks each.
        plane = Int64(m) * self.row_bytes
        for k in cutlass.range_constexpr(self.v):
            words = sanitize_v4(load_v4(x_addr + plane * k + row_off + col))
            slot = Int64(rank + k * self.real)
            publish_v4(mc_addr + box_row + slot * self.row_bytes + col, words, self.publish)

        # Poll this fragment's world slots, reduce in rank order, re-arm.
        base = local_addr + box_row + col
        slots = poll_slots_v4(
            base, self.world, self.row_bytes, self.backoff_ns, self.spin_limit, self.vote
        )
        if cutlass.const_expr(self.trigger == TRIGGER_POLL):
            griddepcontrol_launch_dependents()
        store_v4(out_addr + row_off + col, rank_order_sum(slots, self.chunk))
        rearm_slots_v4(base, self.world, self.row_bytes)


class K3LamportTwoShotKernel:
    """Two-shot all-reduce with column ownership; see the module docstring."""

    def __init__(
        self,
        world: int,
        real: int,
        hidden: int,
        threads: int,
        use_pdl: bool,
        trigger: str,
        publish: str,
        backoff_ns: int,
        spin_limit: int,
        vote: bool,
    ):
        _check_common(world, real, hidden, threads, trigger, publish)
        frags = hidden // FRAGMENT_ELEMS
        if frags % world:
            raise ValueError(f"{frags} fragments per row do not split over {world} owners")
        self.world, self.real, self.v = world, real, world // real
        self.hidden, self.threads = hidden, threads
        self.row_bytes = hidden * 2
        self.cpr = frags // threads
        self.fpo = frags // world  # fragments per owner per row
        # One receive row (all fragments this real rank owns) per (token, source slot).
        self.recv_row_bytes = self.v * self.fpo * FRAGMENT_BYTES
        self.use_pdl, self.trigger, self.publish = use_pdl, trigger, publish
        self.backoff_ns, self.spin_limit, self.vote = backoff_ns, spin_limit, vote
        self.chunk = twoshot_chunk(world)

    @cute.jit
    def __call__(
        self,
        x_addr: Int64,
        out_addr: Int64,
        local_addr: Int64,
        mc_addr: Int64,
        peer0_addr: Int64,
        peer_stride: Int64,
        m: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(x_addr, out_addr, local_addr, mc_addr, peer0_addr, peer_stride, m, rank).launch(
            grid=(m * self.cpr, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x_addr: Int64,
        out_addr: Int64,
        local_addr: Int64,
        mc_addr: Int64,
        peer0_addr: Int64,
        peer_stride: Int64,
        m: Int32,
        rank: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx // self.cpr
        frag = (bidx - row * self.cpr) * self.threads + tidx
        col = Int64(frag) * FRAGMENT_BYTES
        row_off = Int64(row) * self.row_bytes
        # Owner (virtual rank) of this fragment, the real rank hosting it, and the fragment's
        # index among that real rank's owned fragments.
        owner = frag // self.fpo
        host = owner % self.real
        local_frag = (owner // self.real) * self.fpo + (frag - owner * self.fpo)
        recv_tok = Int64(row) * (self.world * self.recv_row_bytes)
        lf_off = Int64(local_frag) * FRAGMENT_BYTES
        gather_off = Int64(m) * (self.world * self.recv_row_bytes)

        if cutlass.const_expr(self.trigger == TRIGGER_ENTRY):
            griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()

        # Scatter: each virtual rank's fragment into its receive slot at the owner.
        plane = Int64(m) * self.row_bytes
        peer = peer0_addr + Int64(host) * peer_stride
        for k in cutlass.range_constexpr(self.v):
            words = sanitize_v4(load_v4(x_addr + plane * k + row_off + col))
            slot = Int64(rank + k * self.real)
            publish_v4(peer + recv_tok + slot * self.recv_row_bytes + lf_off, words, PUBLISH_ST)

        # Owner: reduce the receive slots, multicast the sum into every gather row, re-arm.
        if host == rank:
            base = local_addr + recv_tok + lf_off
            slots = poll_slots_v4(
                base, self.world, self.recv_row_bytes, self.backoff_ns, self.spin_limit, False
            )
            total = sanitize_v4(rank_order_sum(slots, self.chunk))
            publish_v4(mc_addr + gather_off + row_off + col, total, self.publish)
            rearm_slots_v4(base, self.world, self.recv_row_bytes)

        # Everyone: the reduced fragment from the gather row.
        gather = local_addr + gather_off + row_off + col
        got = poll_slots_v4(gather, 1, 0, self.backoff_ns, self.spin_limit, self.vote)
        if cutlass.const_expr(self.trigger == TRIGGER_POLL):
            griddepcontrol_launch_dependents()
        store_v4(out_addr + row_off + col, got[0])
        rearm_slots_v4(gather, 1, 0)


def oneshot_region_bytes(m: int, hidden: int, world: int) -> int:
    """Region bytes a one-shot of ``m`` rows needs."""
    return m * world * hidden * 2


def twoshot_region_bytes(m: int, hidden: int, world: int, real: int) -> int:
    """Region bytes a two-shot of ``m`` rows needs (receive slots + gather rows)."""
    return m * (world // real) * hidden * 2 + m * hidden * 2


@functools.cache
def _compiled(
    mode: str,
    world: int,
    real: int,
    hidden: int,
    threads: int,
    use_pdl: bool,
    trigger: str,
    publish: str,
    backoff_ns: int,
    spin_limit: int,
    vote: bool,
):
    args = (world, real, hidden, threads, use_pdl, trigger, publish, backoff_ns, spin_limit, vote)
    addr = Int64(1 << 20)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if mode == MODE_ONESHOT:
        kernel = K3LamportOneShotKernel(*args)
        return cute.compile(kernel, addr, addr, addr, addr, Int32(1), Int32(0), stream)
    if mode == MODE_TWOSHOT:
        kernel = K3LamportTwoShotKernel(*args)
        return cute.compile(kernel, addr, addr, addr, addr, addr, addr, Int32(1), Int32(0), stream)
    raise ValueError(f"mode {mode}")


#: Fragments one one-shot CTA polls per pass (threads x world). Larger passes re-read more than an
#: SM's load path clears quickly: on Rubin 128 threads are fastest at 4 ranks, 64 at 8 and 32 at
#: 16 (8 and 16 measured with emulated ranks).
ONESHOT_CTA_POLL_FRAGMENTS = 512


@functools.cache
def default_threads(hidden: int, mode: str = MODE_TWOSHOT, world: int = 1) -> int:
    """Threads per CTA: at most 128, for one-shot at most ``512 / world`` (at least 32); the
    largest of 128 / 64 / 32 within that bound dividing the row's fragments."""
    frags = hidden // FRAGMENT_ELEMS
    cap = 128
    if mode == MODE_ONESHOT:
        cap = max(32, min(cap, ONESHOT_CTA_POLL_FRAGMENTS // world))
    for threads in (128, 64, 32):
        if threads <= cap and frags % threads == 0:
            return threads
    raise ValueError(f"hidden {hidden}: no CTA size of at most {cap} divides {frags} fragments")


def compile_k3_lamport_allreduce(
    *,
    mode: str,
    world: int,
    real: int,
    hidden: int,
    threads: int | None = None,
    use_pdl: bool = TRTLLM_ENABLE_PDL,
    trigger: str = TRIGGER_POLL,
    publish: str = PUBLISH_ST,
    backoff_ns: int = 0,
    spin_limit: int | None = None,
    vote: bool = False,
):
    """The compiled kernel for these options (cached; compile at warmup, before graph capture,
    and before the group's first launch so no rank polls while a peer still compiles)."""
    threads = threads or default_threads(hidden, mode, world)
    if spin_limit is None:
        spin_limit = SPIN_LIMIT
    return _compiled(
        mode, world, real, hidden, threads, use_pdl, trigger, publish, backoff_ns, spin_limit, vote
    )


def run_k3_lamport_allreduce(
    x: torch.Tensor,
    out: torch.Tensor,
    *,
    mode: str,
    world: int,
    real: int,
    rank: int,
    local_addr: int,
    mc_addr: int,
    peer0_addr: int = 0,
    peer_stride: int = 0,
    threads: int | None = None,
    use_pdl: bool = TRTLLM_ENABLE_PDL,
    trigger: str = TRIGGER_POLL,
    publish: str = PUBLISH_ST,
    backoff_ns: int = 0,
    spin_limit: int | None = None,
    vote: bool = False,
) -> None:
    """Launch on the current stream (tensor checks: see ``K3LamportAllReduce``).

    Args:
        x: ``bf16 [M, H]`` (``world == real``) or ``[world / real, M, H]`` (emulated ranks), contiguous.
        out: ``bf16 [M, H]``.
        mode: ``"oneshot"`` or ``"twoshot"``.
        world, real, rank: Virtual group size, real group size, this real rank.
        local_addr, mc_addr: This rank's and the multicast base address of the parity region.
        peer0_addr, peer_stride: Unicast base of real rank 0's region and the distance between
            ranks' unicast mappings (two-shot only).
        threads: CTA size (a multiple of 32 dividing ``H / 8``); default :func:`default_threads`.
        use_pdl, trigger: Programmatic dependent launch (default ``TRTLLM_ENABLE_PDL``) and when
            dependents may launch (default after the poll, like the MNNVL all-reduce; ``"entry"``
            lets a dependent run its pre-wait work under the all-reduce).
        publish: ``"st"`` (``st.relaxed.sys`` to the multicast address) or ``"multimem"``.
        backoff_ns, spin_limit, vote: Poll options (see ``k3_lamport.poll_slots_v4``; spin limit
            None = ``TRTLLM_K3_LAMPORT_SPIN_LIMIT``).
    """
    hidden = out.shape[-1]
    m = out.shape[0]
    compiled = compile_k3_lamport_allreduce(
        mode=mode,
        world=world,
        real=real,
        hidden=hidden,
        threads=threads,
        use_pdl=use_pdl,
        trigger=trigger,
        publish=publish,
        backoff_ns=backoff_ns,
        spin_limit=spin_limit,
        vote=vote,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(out.device).cuda_stream)
    if mode == MODE_ONESHOT:
        compiled(
            Int64(x.data_ptr()),
            Int64(out.data_ptr()),
            Int64(local_addr),
            Int64(mc_addr),
            Int32(m),
            Int32(rank),
            stream,
        )
    else:
        compiled(
            Int64(x.data_ptr()),
            Int64(out.data_ptr()),
            Int64(local_addr),
            Int64(mc_addr),
            Int64(peer0_addr),
            Int64(peer_stride),
            Int32(m),
            Int32(rank),
            stream,
        )


__all__ = [
    "MAX_WORLD",
    "MODE_ONESHOT",
    "MODE_TWOSHOT",
    "ONESHOT_CTA_POLL_FRAGMENTS",
    "PUBLISH_MULTIMEM",
    "PUBLISH_ST",
    "TRIGGER_ENTRY",
    "TRIGGER_NONE",
    "TRIGGER_POLL",
    "K3LamportOneShotKernel",
    "K3LamportTwoShotKernel",
    "compile_k3_lamport_allreduce",
    "default_threads",
    "oneshot_region_bytes",
    "run_k3_lamport_allreduce",
    "twoshot_region_bytes",
]

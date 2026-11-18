# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed MoE tail: one fused kernel finalizes this rank's top-16 rows, multicasts the
latent partial and peer-stores the shared shard (three rotating parities), all-reduces
and RMS-normalizes per K-slice, runs the K-split up projection and multicasts the output
row into the next AttnRes mailbox. At TP8 B5+ and TP16 B9+ the experts publish the
latent partial, WaveMoeTail.push the shared shard and WaveMoeTail does the rest. At
``twoshot_batches`` the latent all-reduce is ``common.twoshot``: the experts push each
partial chunk to its column owner, whose WaveMoeTail CTAs sum and multicast it."""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass.cute.runtime import make_fake_stream

from ..common.host import (
    check_registers,
    fake,
    prepare_shared_carveout,
    stream,
    tensor_maps,
    to_cute,
)
from ..common.policy import batch_table
from ..common.twoshot import TwoShot
from ..oproj.prepared import OprojMailbox
from .experts_runtime import every_sm
from .tail import MoeTail, WaveMoeTail

# MoE tail mailbox tokens (padded decode batches); small-batch front_poll G/W/S rows per
# parity.
CAP, POLL_CAP = 64, 8
# Two-shot latent all-reduce batches per TP (B9+): 2-/4-node GB300, us/layer of RS push
# + experts + WaveMoeTail, max over ranks, multicast vs two-shot: TP8 B16 41.74/42.80,
# B32 52.20/53.18, B64 79.27/81.50; TP16 B16 33.98/34.49, B32 42.19/42.52, B64
# 65.05/60.51.
TWOSHOT = {8: "", 16: "33-64:1"}
# WaveMoeTail batches that release the next AttnRes at entry.
EARLY = {8: "9-64:1", 16: "9-64:1"}
# WaveMoeTail batches whose up projection runs on tcgen05 (B9+; 1-GPU WaveMoeTail us
# mma.sync vs tcgen05, B16/32/64: TP8 4.53/5.80/9.05 vs 4.45/5.26/7.80, TP16 two-shot
# 4.50/5.39/7.75 vs 4.52/5.46/7.21).
UMMA = {8: "9-64:1", 16: "33-64:1"}
# Rubin (compute capability 10.7): published batches below UMMA's whose up projection
# runs on tcgen05 too, with the latent all-reduce two-shot (``RUBIN_TAIL_TWOSHOT``,
# which needs tcgen05's registers: mma.sync takes 172-174 of the 112 allowed). tcgen05
# is a stated reorder (as UMMA's rows): it sums each K step in another order than
# mma.sync; per call, AttnRes output within 7e-3 relative (1-2 BF16 ulp) on wide-range
# cancelling data, most calls identical on Gaussian data. Two-shot vs one-shot over the
# same up projection is bitwise. 16 ranks, 32-layer DL7 proxy, us/step served / tcgen05
# / tcgen05 + two-shot: B2 (16 rows) 2174.0 / 2179.7 / 2194.8 (both lose: rows 9-16
# stay as served), B4 (32 rows) 2760.5 / 2748.1 / 2741.1.
RUBIN_TAIL_UMMA = {16: "17-32:1"}
RUBIN_TAIL_TWOSHOT = {16: "17-32:1"}
# Rubin (compute capability 10.7): front-mode MoeTail latent / shared-shard poll backoff
# in ns (TP16 default 64). Component bench, us/call, B4/B8: 1 GPU (peers pre-filled)
# 5.44/6.07 vs 5.31/5.88 at 0, 4 GPUs 7.04/7.68 vs 6.82/7.48, bitwise identical.
RUBIN_POLL_NS = {16: 0}
# Rubin: front-mode MoeTail batches whose CTAs gate their 16-source latent and
# shared-shard poll on one fragment per source (``MoeTail.gate``). 16-rank component
# bench (4 nodes, every slot real, lockstep), us/call B4/B8: 7.11-7.19/8.14-8.20
# ungated vs 6.36-6.37/7.46-7.52 gated, outputs bitwise identical on all 16 ranks.
RUBIN_GATE = {16: "3-8:1"}
# Rubin: published WaveMoeTail batches that re-poll their latent / RS / reduced rows
# without sleeping (else 64 ns). 16-rank chain (experts -> tail -> AttnRes, every slot
# real), us/layer max over ranks, three alternating runs each: B8 28.49 / 28.68 / 28.52
# at 64 ns vs 28.36 / 28.34 / 28.50 at 0; B5 23.89 / 24.04 / 23.95 vs 23.62 / 23.63 /
# 23.74. Outputs bitwise identical.
RUBIN_WAVE_POLL = {16: "5-8:1,33-64:1"}
# Rubin: tail batches whose latent poll (``MoeTail.poll``) and two-shot settle reload
# every source each pass (``MoeTail(reload=True)``): a pending-only reload went through
# predicated temporaries that ptxas merged a few loads at a time (~10 / ~9 dependent
# round trips per 16-source pass). Bitwise identical; 16 ranks: B64 two-shot tail
# -0.6 us/call, 8-row direct tail -0.1..-0.35 us per KDA layer (real-tail proxy).
RUBIN_TAIL_RELOAD = {16: "1-64:1"}
# Rubin: two-shot WaveMoeTail batches (``TWOSHOT``) whose owned-slice reduce runs one
# token per CTA of each cluster and whose previous-layer re-arm spans the whole grid
# and runs at entry (``WaveMoeTail(spread=True, early_rearm=True)``; with the poll
# backoff 0 above). M64, 4 ranks, us/call max over ranks, every input pre-filled:
# 10.35 -> 8.57; behind an experts stand-in that releases the tail at entry and
# writes the receive / RS slots 6 us later: 17.30 -> 14.69. Outputs bitwise identical.
RUBIN_TWOSHOT_SPREAD = {16: "17-64:1"}
# Rubin: two-shot spread batches whose up projection runs on BF16(x * gamma) and whose
# line owners scale by the token's RMS reciprocal (``WaveMoeTail(late_rms=True)``, a
# stated reorder as at B8). M64 behind the experts stand-in, us/call max over ranks:
# 16 ranks 14.29 -> 13.90, 4 ranks 17.19 (exact order, no spread) -> 14.38; max error
# vs an FP32 reference 0.37-0.54 % of max |out| (exact order 0.42-0.51 %).
# From 17 rows (with RUBIN_TAIL_TWOSHOT / UMMA): 4 ranks + 12 emulated, tail alone
# (inputs pre-filled), us/call max over ranks without / with: B16 6.40 / 6.13, B24
# 6.50 / 6.16, B32 6.33 / 6.07; vs the FP32 reference 0.36-0.53 % of max |out|
# (0.39-0.59 % without).
RUBIN_TWOSHOT_LATE_RMS = {16: "17-64:1"}
# Rubin: spread two-shot batches whose previous-layer re-arm runs after the CTA's owned
# reduce and RS lines are issued, while the gathered rows are in flight
# (``WaveMoeTail(rearm_after=True)``), not before them. Bitwise. 16 ranks, b=64, tail
# alone (inputs pre-filled), us/call: 8.41 -> 8.15; 4 ranks behind an experts
# stand-in 14.13 -> 13.74.
RUBIN_TWOSHOT_REARM_AFTER = {16: "17-64:1"}
# Rubin: published one-group batches whose WaveMoeTail CTAs poll the sources directly
# (no cluster-shared reduced rows, ``WaveMoeTail`` direct). Outputs bitwise identical.
RUBIN_WAVE_DIRECT = {16: "5-8:1"}


# Rubin: published one-group batches whose WaveMoeTail multiplies the RMS
# reciprocal after the up projection (``WaveMoeTail`` late_rms): the squares
# exchange overlaps the MMA. A stated reorder, checked against the FP32 reference.
RUBIN_WAVE_LATE_RMS = {16: "5-8:1"}
# Rubin: direct one-group batches whose WaveMoeTail releases the next AttnRes once its
# latent is polled instead of after the output multicast (``WaveMoeTail``
# release_polled), when the AttnRes feeds a KdaHead (``early_head``): the AttnRes and
# then KdaHead launch earlier. 4 real ranks, us/layer B8: KdaHead next 37.33/37.26/37.44
# -> 36.85/36.78/36.81; an in-projection GEMM next loses (34.21/34.18 -> 34.62/34.66).
# Outputs bitwise identical.
RUBIN_WAVE_RELEASE_POLLED = {16: "8:1"}
# Rubin: direct one-group batches whose WaveMoeTail runs fewer instructions after the
# last latent lands (``WaveMoeTail`` lean_owner: BF16-FMA column sums, the line owner on
# a warp without MMA, per-lane output multicast). 4 real ranks, same routes, in-process,
# us/layer B8 max (mean) over ranks: KdaHead next 34.32 (34.23) -> 34.15 (34.03), an
# in-projection next 33.15 (33.00) -> 33.02 (32.87), none 27.53 -> 27.41. Outputs
# bitwise identical.
RUBIN_WAVE_LEAN_OWNER = {16: "8:1"}
# Rubin: direct one-group batches whose WaveMoeTail re-arms the previous layer's rows at
# entry instead of after its output (``WaveMoeTail`` early_rearm). 4 real ranks, same
# routes, in-process, us/layer B8 max (mean) over ranks: KdaHead next 34.35 (34.19) ->
# 34.27 (34.11), an in-projection next 32.87 (32.80) -> 32.88 (32.81). Outputs bitwise
# identical.
RUBIN_WAVE_EARLY_REARM = {16: "8:1"}
# Rubin: late-RMS batches whose WaveMoeTail pushes the RMS squares to the line owners
# only, so the other CTAs exit after their partials (``WaveMoeTail`` owner_squares).
# 4 real ranks, same routes, in-process, us/layer B8 max (mean) over ranks: KdaHead next
# 34.27 (34.12) -> 34.17 (34.04), an in-projection next 32.91 (32.83) -> 32.87 (32.78).
# Outputs bitwise identical.
RUBIN_WAVE_OWNER_SQUARES = {16: "8:1"}
# Rubin: direct one-group batches whose WaveMoeTail releases the next AttnRes at entry
# (``early``, as EARLY at B9+) when that AttnRes feeds the projection ``use_heads``
# marks (a KdaHead; takes precedence over ``RUBIN_WAVE_RELEASE_POLLED``), which then
# becomes resident as soon as the tail's non-owner CTAs exit. 4 real ranks, same routes,
# in-process, us/layer B8 max (mean) over ranks, release after the poll -> at entry:
# KdaHead next 34.25 (34.08) -> 33.97 (33.80); a tensor-memory MLA in-projection next
# 32.72 (32.34) -> 32.08 (31.70). Outputs bitwise identical.
RUBIN_WAVE_RELEASE_ENTRY = {16: "8:1"}


def wave_lean_owner(tp, batch, device=None):
    if not wave_direct(tp, batch, device):
        return False
    return bool(batch_table(RUBIN_WAVE_LEAN_OWNER.get(tp, "")).get(batch, 0))


def wave_owner_squares(tp, batch, device=None):
    if not wave_late_rms(tp, batch, device):
        return False
    return bool(batch_table(RUBIN_WAVE_OWNER_SQUARES.get(tp, "")).get(batch, 0))


def wave_early_rearm(tp, batch, device=None):
    if not wave_direct(tp, batch, device):
        return False
    return bool(batch_table(RUBIN_WAVE_EARLY_REARM.get(tp, "")).get(batch, 0))


def wave_release_entry(tp, batch, device=None):
    if not wave_direct(tp, batch, device):
        return False
    return bool(batch_table(RUBIN_WAVE_RELEASE_ENTRY.get(tp, "")).get(batch, 0))


def wave_release_polled(tp, batch, device=None):
    if not wave_direct(tp, batch, device):
        return False
    return bool(batch_table(RUBIN_WAVE_RELEASE_POLLED.get(tp, "")).get(batch, 0))


def wave_late_rms(tp, batch, device=None):
    if not rubin(device):
        return False
    return bool(batch_table(RUBIN_WAVE_LATE_RMS.get(tp, "")).get(batch, 0))


def twoshot_late_rms(tp, batch, device=None):
    if not rubin(device):
        return False
    return bool(batch_table(RUBIN_TWOSHOT_LATE_RMS.get(tp, "")).get(batch, 0))


def twoshot_rearm_after(tp, batch, device=None):
    if not rubin(device):
        return False
    return bool(batch_table(RUBIN_TWOSHOT_REARM_AFTER.get(tp, "")).get(batch, 0))


def twoshot_spread(tp, batch, device=None):
    if not rubin(device):
        return False
    return bool(batch_table(RUBIN_TWOSHOT_SPREAD.get(tp, "")).get(batch, 0))


# Rubin two-shot WaveMoeTail: registers per thread at most. The B64 KDA in-projections
# start early only where they fit beside the AttnRes CTAs, and the tail's SMs take more
# AttnRes CTAs the fewer registers it holds: at 112 no AttnRes register class (56-96)
# leaves an in-projection CTA waiting for the AttnRes to exit; at 120 the 72 and 96
# classes do (1-GPU placement census). ``compile_tail`` raises above it.
RUBIN_TWOSHOT_REGISTERS = 112


# Rubin: direct one-group batches whose latent / RS poll exits by a warp vote
# (``WaveMoeTail(vote=True)``) instead of per-lane loops that the warp sync after them
# waits to reconverge (~1000-1500 cycles after the last lane on a DcpValue poll).
RUBIN_TAIL_VOTE = {16: "5-8:1"}


def tail_vote(tp, batch, device=None):
    if not rubin(device):
        return False
    return bool(batch_table(RUBIN_TAIL_VOTE.get(tp, "")).get(batch, 0))


def tail_reload(tp, batch, device=None):
    if not rubin(device):
        return False
    return bool(batch_table(RUBIN_TAIL_RELOAD.get(tp, "")).get(batch, 0))


def wave_direct(tp, batch, device=None):
    if not rubin(device):
        return False
    return bool(batch_table(RUBIN_WAVE_DIRECT.get(tp, "")).get(batch, 0))


def rubin(device):
    return device is not None and torch.cuda.get_device_capability(device) == (10, 7)


def flat(t):
    return to_cute(t.reshape(-1), 16)


# Rubin: the batches above which the experts publish (else TP8 4, TP16 8). TP16 B5-8
# as at TP8: the experts' epilogue finalizes and multicasts the latent partial and
# WaveMoeTail polls it, instead of MoeTail finalizing after the experts grid completes.
# 1-GPU chain (experts -> tail -> AttnRes, ranks 1-15 emulated, cold weights), us per
# layer at B8: 28.19 front vs 26.77 published.
RUBIN_PUBLISH = {16: 4}


def published(tp, batch, device=None):
    """The experts publish the latent and WaveMoeTail runs the tail (else MoeTail
    front)."""
    above = 4 if tp == 8 else 8
    if rubin(device):
        above = RUBIN_PUBLISH.get(tp, above)
    return batch > above


# Rubin: published batches whose routed experts peer-store this rank's shared-expert
# rows into the RS mailboxes (``ExpertsKernel`` publish="push": the epilogue warps,
# after the PDL wait, which also covers the shared expert they join by an event;
# TP16 B5-8, grid = every SM), instead
# of a push kernel ending the side stream that the experts join before they launch.
# MoE front proxy (o-proj -> front / shared chain -> experts), us per layer: that push
# kernel adds +2.5 (B8) / +2.6 (B5) to the experts' end.
RUBIN_EXPERTS_PUSH = {16: "5-8:1"}


def experts_push(tp, batch, device=None):
    """The routed experts push the shared rows (no ``compile_push`` kernel). Only where
    their grid spans every SM (``experts_runtime.GRID``), so that they launch after
    the shared expert (``moe.prepared``); other SM counts keep the push kernel."""
    if not (published(tp, batch, device) and rubin(device)):
        return False
    if not every_sm(tp, batch, device):
        return False
    return bool(batch_table(RUBIN_EXPERTS_PUSH.get(tp, "")).get(batch, 0))


def twoshot_batches(tp, device=None):
    table = batch_table(TWOSHOT.get(tp, ""))
    if rubin(device):
        table.update(batch_table(RUBIN_TAIL_TWOSHOT.get(tp, "")))
    return {batch for batch, on in table.items() if on and batch > 8}


def early_batches(tp, device=None):
    table = batch_table(EARLY.get(tp, ""))
    return {b for b, on in table.items() if on and published(tp, b, device)}


def umma_batches(tp, device=None):
    table = batch_table(UMMA.get(tp, ""))
    if rubin(device):
        table.update(batch_table(RUBIN_TAIL_UMMA.get(tp, "")))
    return {batch for batch, on in table.items() if on and batch > 8}


def shares(tp, batch, twoshot=False, device=None):
    """WaveMoeTail clusters share reduced latent rows (all 112 CTAs must be
    co-resident)."""
    direct = wave_direct(tp, batch, device) and not twoshot
    wave = WaveMoeTail(batch, tp, twoshot, direct=direct)
    return published(tp, batch, device) and wave.reduce


@cache
def resident_clusters(device):
    """Co-resident 16-CTA clusters at one CTA per SM."""
    with torch.accelerator.device_index(device):
        return cutlass.utils.HardwareInfo(device).get_max_active_clusters(16)


class MoeTailBox:
    """Model-owned MoE tail mailboxes: latent [3, TP, CAP, 3584] (multicast), shared
    reduce-scatter [3, TP, CAP, 7168/TP] (peer stores), output [1, CAP, 7168], the
    local all-reduced latent rows [3, CAP, 3584] that WaveMoeTail clusters share and the
    two-shot latent buffers (``twoshot_args``/``twoshot_tail`` replace args/tail)."""

    def __init__(self, group, device, tp, rank):
        self.rank = rank
        self.latent = OprojMailbox(group, device, tp, rank, (3, tp, CAP, 3584))
        self.rs = OprojMailbox(group, device, tp, rank, (3, tp, CAP, 7168 // tp))
        self.out = OprojMailbox(group, device, tp, rank, (1, CAP, 7168))
        ptrs = self.rs.handle.buffer_ptrs
        self.peers = torch.tensor(ptrs, dtype=torch.int64, device=device)
        self.args = (flat(self.latent.buffer), flat(self.rs.buffer))
        self.tail: tuple = (to_cute(self.peers, 16), flat(self.out.buffer))
        self.tail += (Int64(self.latent.mc), Int64(self.out.mc))
        self.reduced = torch.empty((3, CAP, 3584), dtype=torch.bfloat16, device=device)
        self.reduced.view(torch.int32).fill_(-0x80000000)
        self.twoshot = box = TwoShot(group, device, tp, rank, 3584, CAP, parities=3)
        self.twoshot_args = (flat(box.recv), self.args[1])
        self.twoshot_tail = (*self.tail[:2], Int64(box.gather_mc), self.tail[3])
        # Experts' push: every parity's receive slots [3 x TP, CAP, 3584/TP] and the
        # owners' buffer bases (slot parity x TP + rank).
        self.push_box = (
            to_cute(box.recv.view(3 * tp, CAP, -1)),
            to_cute(box.peers[0], 16),
        )


@cache
def compile_tail(
    tp, batch, device, mode="front", trigger=2, early=False, umma=False, head=False
):  # fmt: skip
    """front: rows complete at the PDL wait; front_poll: three 0xffff-armed G/W/S parity
    slots polled per row and re-armed after use; publish (``published``): WaveMoeTail
    over the published mailboxes, reading the weight shard directly, not by TMA;
    twoshot: publish over the two-shot receive slots and gather."""
    m8, twoshot = mode in ("publish", "twoshot"), mode == "twoshot"
    vec = lambda n, dtype=BFloat16, align=16: fake((n,), None, dtype, align)
    slots = 3 * POLL_CAP if mode == "front_poll" else batch
    latent = 3 * tp * CAP * (3584 // tp if twoshot else 3584)
    tensors: tuple = (vec(latent), vec(3 * tp * CAP * (7168 // tp)))
    weights = vec(7168 // tp * 3584) if m8 and not umma else vec(128, Uint8, 64)
    tensors += (weights, vec(3584))
    tensors += (vec(slots * 16 * 3584), vec(slots * 16), vec(slots * 7168))
    tensors += (vec(tp, Int64), vec(CAP * 7168))
    scalars: tuple = (Int64(0), Int64(0), Int32(0), Int32(0), Int32(CAP), Float32(1e-6))
    scalars += (Int32(2), Int32(CAP))
    kernel: MoeTail
    if m8:
        table = batch_table(RUBIN_WAVE_POLL.get(tp, "")) if rubin(device) else {}
        poll_ns = 0 if table.get(batch) else 64
        direct = wave_direct(tp, batch, device) and not twoshot
        late = wave_late_rms(tp, batch, device) and not (twoshot or umma)
        early = early or (direct and head and wave_release_entry(tp, batch, device))
        polled = direct and not early and head
        polled = polled and wave_release_polled(tp, batch, device)
        lean = direct and wave_lean_owner(tp, batch, device)
        rearm = direct and wave_early_rearm(tp, batch, device)
        owners = late and wave_owner_squares(tp, batch, device)
        spread = twoshot and twoshot_spread(tp, batch, device)
        rearm = rearm or spread
        late = late or (spread and umma and twoshot_late_rms(tp, batch, device))
        after = spread and twoshot_rearm_after(tp, batch, device)
        kernel = wave = WaveMoeTail(
            batch, tp, twoshot, early, umma, poll_ns, direct, late, polled, lean, rearm,
            owners, spread, after, reload=tail_reload(tp, batch, device),
            vote=direct and tail_vote(tp, batch, device),
        )  # fmt: skip
        scalars += (vec(3 * CAP * 3584),) if wave.reduce else ()
    else:
        front = mode == "front" and rubin(device)
        poll_ns = RUBIN_POLL_NS.get(tp) if front else None
        gate = front and bool(batch_table(RUBIN_GATE.get(tp, "")).get(batch, 0))
        kernel = MoeTail(
            tp, batch, mode=mode, trigger=trigger, poll_ns=poll_ns, gate=gate,
            reload=tail_reload(tp, batch, device),
        )  # fmt: skip
    options = "" if m8 else "--ptxas-options=--maxrregcount=120"
    with torch.accelerator.device_index(device):
        args = (*tensors, *scalars, make_fake_stream())
        code = cute.compile(kernel, *args, options=options)
        if mode == "twoshot" and rubin(device):  # RUBIN_TWOSHOT_REGISTERS
            what = f"Rubin TP{tp} B{batch} two-shot WaveMoeTail"
            check_registers(code, device, RUBIN_TWOSHOT_REGISTERS, what)
        # MoE tail requests 100% shared carveout so the next large PDL kernel can
        # coexist.
        return prepare_shared_carveout(code, device, 100)


@cache
def compile_push(batch, device, tp=8):
    args = (fake((batch, 7168)), fake((tp,), None, Int64), Int32(0), Int32(0))
    with torch.accelerator.device_index(device):
        return cute.compile(
            WaveMoeTail(batch, tp).push, *args, Int32(CAP), make_fake_stream()
        )


class MoeTailWeights:
    """One layer's up-projection shard [7168/TP, 3584], RMS weight and TMA map."""

    def __init__(self, tp, weight, gamma, eps, ordinal, count, batches, twoshots=()):
        self.weight, self.gamma = weight, gamma
        # Three parities rotate over the model's MoE layers; each launch re-arms the
        # previous MoE layer's slots of its batch (wrapping across decode steps, where
        # the first layer re-arms the largest batch the last layer may have published).
        parity, rearm = ordinal % 3, (ordinal - 1) % count % 3
        if parity == rearm:  # a launch must never re-arm the slots it publishes to
            raise ValueError(f"MoE tail parities collide for {count} MoE layers")
        self.parity, rearm = Int32(parity), Int32(rearm)
        self.scalars = (Int32(CAP), Float32(eps), rearm)
        self.wrap = 0 if ordinal else max(batches)
        cols = 512 if tp == 8 else 448
        u16 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16
        dims = (cols // 2, 3584 // (cols // 2), 7168 // tp)
        spec: tuple = (weight.data_ptr(), u16, dims, (cols, 7168), (cols // 2, 2, 32))
        spec += (cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,)
        spec += (cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,)
        self.descriptor = tensor_maps([spec], weight.device)
        self.views = (to_cute(self.descriptor, 64), to_cute(gamma, 16))
        device = weight.device.index
        self.codes, early, self.compiles = {}, early_batches(tp, device), {}
        umma = umma_batches(tp, device)
        if any(published(tp, b, device) and b in umma for b in batches):
            # WaveMoeTail tcgen05: [64 x lines, 64] boxes of the shard, 128B swizzle
            spec = (weight.data_ptr(), u16, (3584, 7168 // tp), (7168,))
            spec += ((64, 128 if tp == 8 else 64),)
            spec += (cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,)
            spec += (cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,)
            self.umma_map = tensor_maps([spec], weight.device)
            self.umma_views = (to_cute(self.umma_map, 64), self.views[1])
        for b in batches:  # WaveMoeTail reads the shard directly, not by TMA
            twoshot = b in twoshots
            publish = published(tp, b, device)
            mode = "twoshot" if twoshot else "publish" if publish else "front"
            views = self.views if mode == "front" else (flat(weight), self.views[1])
            on = mode != "front" and b in umma
            views = self.umma_views if on else views
            self.compiles[b] = (tp, b, device, mode), dict(early=b in early, umma=on)
            code = compile_tail(*self.compiles[b][0], **self.compiles[b][1])
            self.codes[b] = code, views, shares(tp, b, twoshot, device), twoshot
        shared = any(shares(tp, b, b in twoshots, device) for b in batches)
        if shared and resident_clusters(device) < 7:
            raise RuntimeError("WaveMoeTail needs seven co-resident 16-CTA clusters")

    def use_heads(self, batches):
        """The next layer starts with a KdaHead at ``batches``: their codes release
        its AttnRes early where ``RUBIN_WAVE_RELEASE_POLLED`` says so."""
        for b in batches:
            if b in self.codes:
                args, options = self.compiles[b]
                code = compile_tail(*args, **options, head=True)
                self.codes[b] = (code, *self.codes[b][1:])

    def run(self, box, expanded, weights, shared, code=None, batch=None):
        batch = batch or weights.shape[0]
        if code is None:
            code, views, share, twoshot = self.codes[batch]
        else:
            views, share, twoshot = self.views, 0, False
        args, tail = box.args, box.tail
        if twoshot:
            args, tail = box.twoshot_args, box.twoshot_tail
        rows = (flat(expanded), flat(weights), flat(shared), *tail, self.parity)
        rows += (Int32(box.rank), *self.scalars, Int32(self.wrap or batch))
        if share:
            rows += (flat(box.twoshot.gather if twoshot else box.reduced),)
        code(*args, *views, *rows, stream(expanded.device))


class PreparedTail:
    """The layer's MoE tail; placement = (model MoeTailBox, MoE ordinal, MoE layer
    count). At WaveMoeTail batches the experts publish into latent slot ``publish[b]``
    (multicast, parity, rank; at ``twoshots`` the two-shot push box) and the side
    stream pushes the shared-expert output (``push``)."""

    def __init__(self, op, norm, up, batches, placement):
        tp, rank = op.contract.tp_size, op.rank
        local_up = up.narrow(0, rank * (7168 // tp), 7168 // tp)
        self.tail_box, ordinal, count = placement
        eps = op.contract.rms_eps
        m8 = [b for b in batches if published(tp, b, norm.device)]
        self.twoshots = twoshot_batches(tp, norm.device) & set(m8)
        args = (tp, local_up, norm, eps, ordinal, count, batches, self.twoshots)
        self.tail_weights = MoeTailWeights(*args)
        self.peers, _, mc, _ = self.tail_box.tail
        self.ring = (self.tail_weights.parity, Int32(rank))
        dev = norm.device
        pushed = [b for b in m8 if not experts_push(tp, b, dev)]
        self.pushers = {b: compile_push(b, dev.index, tp) for b in pushed}
        self.publish = {b: (mc, *self.ring) for b in m8}
        slot = Int32(ordinal % 3 * tp + rank)
        self.publish.update({b: (*self.tail_box.push_box, slot) for b in self.twoshots})

    def push(self, shared):
        if shared.shape[0] in self.pushers:
            args = (to_cute(shared), self.peers, *self.ring, Int32(CAP))
            self.pushers[shared.shape[0]](*args, stream(shared.device))

    def run(self, expanded, weights, shared, out, *, consume):
        self.tail_weights.run(self.tail_box, expanded, weights, shared)
        if consume(out) is not out:
            raise RuntimeError("The tail output must be consumed from its mailbox ring")
        return out

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output projection fused with its TP all-reduce and post-attention AttnRes: rank
partials go into Lamport slots of a multicast [TP, 64, 7168] mailbox (parity alternates
by layer) and the consumer sums and re-arms them. TP16 B<=8 fuses producer and consumer
(MLA layers below B5 keep them split); above B8 the producer runs 16-token groups.
At ``twoshot_batches``, given ``TwoShot`` buffers, the all-reduce is ``common.twoshot``
(chunks pushed to their column owner, owner sums + multicasts, all Lamport-polled), so
each GPU receives ~2 x 14 KB per token instead of TP x 14 KB."""

from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.runtime import make_fake_stream

from ...shapes import Choice, Path, Shape, rows_policy
from ..attn_res.sharded import compile_gather, owned
from ..common.host import (
    dynamic_fake,
    dynamic_view,
    fake,
    prepare_shared_carveout,
    stream,
    to_cute,
)
from ..common.policy import batch_table
from ..gemm.projection import weight_map
from .consumer import OprojConsumer, OprojFused
from .producer import OprojProducer, OprojUmma, OprojUmmaRows

# The largest batch (token rows) of the tuned tables, the 64-row mailboxes and the
# producers' owned-row reduce (``reduce_owned``).
OPROJ_ROWS = 64
# Rubin (compute capability 10.7) decode shapes above OPROJ_ROWS ("wide") whose
# o-projection runs the two-shot all-reduce with ``OprojUmmaRows`` (weight-stationary
# over 64-row chunks, each chunk's reduce-scatter overlapping the next chunk's MMAs)
# into a ``TwoShot.rows_box`` sized to the deployment's largest graph, and the
# reduced-gather consumer with its AttnRes; other wide shapes take seam S3 (the stock
# o-projection + all-reduce, then the unfused AttnRes). Keyed per TP on token rows
# (``shapes.rows_policy``: every q).
WIDE_TWOSHOT = {16: "65-768:1"}
# Rubin wide shapes whose ``OprojUmmaRows`` runs one chunk of this many token rows
# instead of 64-row chunks, per TP on token rows (``shapes.rows_policy``): at 65-80
# rows a second 64-row chunk (up to 16 real rows) repeats the chunk's MMAs, drain,
# pushes and owner reduce after the first one's. Per token the same MMAs over the
# same K blocks and the same rounding: bitwise. 192-256 rows (DL7 B24-32, DL3 B48-64):
# 96-row chunks, two at 192 rows and three at 200-256 instead of three / four 64-row
# ones (axhb, 16 ranks, 93 layers, 32k, us/step: B24 -16, B28 -54, B32 -79; B17 at
# 136 rows +13, so no row below 192).
WIDE_CHUNK = {16: "65-80:80,192-256:96"}


# Rubin wide shapes (per TP, token rows) whose reduced-gather consumer prefers the
# largest shared-memory carve-out, as CONSUMER_CARVEOUT at 32 / 64 rows: its CTAs need
# ~22 KB, and at the driver default the MoE front's big CTAs launched behind it cannot
# share its SMs, so the front only preloads as consumer CTAs exit. A launch attribute:
# bitwise.
WIDE_CONSUMER_CARVEOUT: dict[int, str] = {16: ""}


def wide_chunk(tp, shape):
    return rows_policy(WIDE_CHUNK.get(tp, ""), shape, OprojUmmaRows.CHUNK)


class OprojMailbox:
    """Sentinel-armed multicast mailbox; defaults to [parity, TP, 64, 7168] BF16."""

    def __init__(self, group, device, tp, rank, shape=None):
        self.tp, self.rank = tp, rank
        shape = (2, tp, 64, 7168) if shape is None else shape
        self.buffer = symm_mem.empty(shape, dtype=torch.bfloat16, device=device)
        self.handle = handle = symm_mem.rendezvous(self.buffer, group)
        self.mc = int(handle.multicast_ptr or 0)
        if not self.mc:  # slot() offsets this address
            raise RuntimeError(
                "Kimi K3 specialized o-projection needs multicast memory"
            )
        self.buffer.view(torch.int32).fill_(-0x80000000)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)

    def slot(self, parity):
        return self.buffer[parity], self.mc + parity * self.buffer[0].numel() * 2


# Two-shot all-reduce batches per TP (2-/4-node GB300, 32 MLA layers, us/layer, max over
# ranks, multicast vs two-shot: TP8 B16 8.85/11.40, B32 10.63/11.32, B64 18.06/13.18;
# TP16 B16 9.78/12.28, B32 14.96/13.05, B64 28.06/14.07).
TWOSHOT = {8: "33-64:1", 16: "17-64:1"}


def fused_schedule(tp, batch, mla, device):
    """TP16 B<=8 fuses producer and consumer (MLA layers below B5 keep them split). On
    Rubin (compute capability 10.7) the split pair wins below B5 (4-GPU component bench,
    other ranks' slots pre-filled, us/call B1/B2/B4 fused 5.85/6.37/6.75 vs split
    5.62/5.89/6.53), but in 16-rank serving at B8 the split consumer ends 0.40 us later
    per KDA layer and 0.24 us later per MLA layer behind the real KdaSplit predecessor
    (+33 us/step), so B5-8 stay fused."""
    if tp != 16 or batch > 8 or (mla and batch < 5):
        return False
    return batch >= 5 or torch.cuda.get_device_capability(device) != (10, 7)


# Rubin (compute capability 10.7) fused o-projection batches per TP whose producer
# copies x to shared memory once per CTA (``OprojFused(x_shared=True)``) instead of
# 16 warps x 112 CTAs re-reading the same 12 KB from L2. 1-GPU in-proj stand-in ->
# KdaSplit -> o-proj -> MoE front proxy (the other ranks' partials landing 1.6 us
# after this rank's), us/layer: B8 35.10 -> 34.08, B5 32.27 -> 31.72.
SHARED_X = {16: "5-8:1"}


def shared_x(tp, batch, device):
    if torch.cuda.get_device_capability(device) != (10, 7):
        return False
    return bool(batch_table(SHARED_X.get(tp, "")).get(batch))


# Rubin (compute capability 10.7) fused o-projection batches per TP that, behind
# KdaHead (``after_head``), launch a copy of the kernel with an explicit preferred
# shared-memory carve-out of 0 (prefer L1) instead of the driver default. The executed
# configuration is the same (64 KiB) and the outputs are bitwise identical, but behind
# a grid in the 228 KiB configuration (KdaHead: 96 x 227 KiB) the default-preference
# o-projection is placed late: 1-GPU component behind a 96 x 227 KiB predecessor 5.89
# -> 3.78 us/call; KdaHead -> o-proj -> MoE front layer proxy 39.85 -> 36.41 us/layer
# (with both L2 prefetches 40.98 -> 39.04). Behind small-SMEM predecessors (DcpValue,
# KdaSplit-like) it does not help, and with a 16-CTA-cluster one it costs 0.47 us, so
# only the KdaHead path takes it. 90-100 % lose: they leave L1 smaller than the 98 KB
# of weights each CTA loads into registers.
PREFER_L1 = {16: "8:1"}


def prefer_l1(tp, batch, device):
    if torch.cuda.get_device_capability(device) != (10, 7):
        return False
    return bool(batch_table(PREFER_L1.get(tp, "")).get(batch))


# Rubin (compute capability 10.7) fused o-projection batches per TP whose kernel
# fences its multicast publish (``OprojFused(drain=True)``) off the critical path. A
# grid that issued NVLink stores and never fenced them delays its PDL dependents'
# release by ~0.2 us (the grid-end flush; 1-GPU microbenchmark, 96-112 CTAs x 1 KB of
# multimem stores 0.3-2 us before the end: dependent released 0.90-0.96 us after the
# last CTA exit vs 0.73-0.75 without the stores or with a fence after them). Bitwise
# identical.
DRAIN = {16: "8:1"}


def drain(tp, batch, device):
    if torch.cuda.get_device_capability(device) != (10, 7):
        return False
    return bool(batch_table(DRAIN.get(tp, "")).get(batch))


# Rubin (compute capability 10.7) one-shot batches per TP (B<=8 fused, 9-16 split)
# whose consumer reloads every rank slot each poll pass and leaves the poll on a warp
# vote (``OprojConsumer(vote=True)``). Bitwise. 9-16: 4 ranks + 12 emulated, M16
# one-shot pair us/call 7.14 -> 7.02 (with ONESHOT_RING below 6.83 -> 6.70).
OPROJ_VOTE = {16: "1-16:1"}


def oproj_vote(tp, batch, device):
    if torch.cuda.get_device_capability(device) != (10, 7):
        return False
    return bool(batch_table(OPROJ_VOTE.get(tp, "")).get(batch))


@cache
def compile_oproj_l1(*args):
    """``compile_oproj(*args)`` as a separate kernel with the preferred carve-out 0."""
    code = compile_oproj.__wrapped__(*args)
    prepare_shared_carveout(code, torch.cuda.current_device(), 0)
    return code


# Rubin (compute capability 10.7) two-shot batches per TP whose column owners poll
# their TP receive slots with every load in flight (``OprojConsumer(batched=True)``)
# instead of one dependent round trip per slot. Bitwise identical.
TWOSHOT_BATCHED = {16: "17-64:1"}


def twoshot_batched(tp, batch, device):
    if torch.cuda.get_device_capability(device) != (10, 7):
        return False
    return bool(batch_table(TWOSHOT_BATCHED.get(tp, "")).get(batch))


# Rubin (compute capability 10.7) two-shot batches per TP whose consumer prefers the
# largest shared-memory carve-out (100 %): its CTAs need only ~22 KB, and at the driver
# default (132 KiB executed) the MoE front's ~170 KB CTAs (``MoeFrontTmem64``, launched
# behind it) cannot share an SM with them, so the front only becomes resident as
# consumer CTAs exit and streams its weights after its PDL wait. At 228 KiB both fit
# and the front preloads beside the consumer. The consumer itself is unchanged
# (4 GPUs, M64, executed 132 -> 228 KiB checked with nsys: pair 11.90 / 11.69 vs
# 11.58 / 11.76 us/call, consumer alone 6.96 / 7.00 / 6.99 vs 6.83 / 6.73 / 6.98);
# 1-GPU front proxy behind a consumer-shaped predecessor: -1.5 us per MoE layer.
# M32 too (16 ranks, 32-layer proxy, same nodes, median us/step 64 -> 32+64, graph
# shas equal: DL7 B4 2781.4 -> 2741.0, DL3 B8 2732.4 -> 2706.1).
CONSUMER_CARVEOUT = {16: "32:100,64:100"}


def consumer_carveout(tp, batch, device):
    """The consumer's preferred carve-out (percent) at this batch, or None."""
    if torch.cuda.get_device_capability(device) != (10, 7):
        return None
    return batch_table(CONSUMER_CARVEOUT.get(tp, "")).get(batch)


# Rubin (compute capability 10.7) split (one- and two-shot) batches per TP whose
# producer gets a preferred shared-memory carve-out (percent). Without one, the driver
# runs the 123-160 KB producer (N 16-40) at the 328 KiB configuration, where two CTAs
# fit on an SM. At 80 it runs at 196 KiB, one CTA per SM, as the M64 producer already
# does. Same kernel, bitwise. 1 GPU, real KDA chain, producer behind its predecessor,
# us to the consumer's PDL-wait return / chain graph us per iteration:
# - M32 behind the q4 fold verify: 13.35 -> 11.70 / 36.94 -> 36.86 (nsys: producer
#   6.53 -> 4.59 us);
# - M32 behind KdaSplit (q8): 13.97 -> 13.37 / 38.45 -> 38.52;
# - M24: 14.48 -> 14.43 / 39.81 -> 39.64;
# - M40: 15.55 -> 14.85 / 39.83 -> 39.80.
# 4 ranks + 12 emulated, pair us/call: one-shot M16 (ONESHOT_RING) 6.93 -> 6.56.
PRODUCER_CARVEOUT = {16: "9-40:80"}


def producer_carveout(tp, batch, device):
    """The split producer's preferred carve-out (percent) at this batch, or None."""
    if torch.cuda.get_device_capability(device) != (10, 7):
        return None
    return batch_table(PRODUCER_CARVEOUT.get(tp, "")).get(batch)


# Rubin (compute capability 10.7) two-shot batches per TP whose producer CTAs also
# reduce this rank's owned columns and multicast the gather (``OprojUmma(reduce=True)``)
# instead of the consumer's owner threads: the producer's 112 CTAs are resident from
# the start, while ~250 of the consumer's 448 CTAs are only placed as producer CTAs
# exit and then run their AttnRes prologue first. Same reduction, bitwise. 16 ranks,
# producer + consumer pair, us/call (max over ranks): B64 13.57 -> 12.57, B32 12.48
# -> 11.55 (with RESIDENT_RING + X_MULTICAST 4: 12.01 / 11.25).
PRODUCER_REDUCE = {16: "17-64:1"}


def producer_reduce(tp, batch, device):
    if torch.cuda.get_device_capability(device) != (10, 7):
        return False
    return bool(batch_table(PRODUCER_REDUCE.get(tp, "")).get(batch))


# Rubin (compute capability 10.7) two-shot batches per TP whose tcgen05 producer holds
# every K block in its ring (``OprojUmma(resident=True)``: all weights before the PDL
# wait, no stage reuse) and, at ``X_MULTICAST`` batches, whose clusters of that many
# CTAs share x by TMA multicast (each CTA reads 1 / size of x from L2). Same MMAs,
# bitwise. 16 ranks, with PRODUCER_REDUCE, us/call: B64 12.57 -> 12.33 (resident) ->
# 12.01 (multicast 4); B32 11.55 -> 11.55 -> 11.25.
RESIDENT_RING = {16: "17-64:1"}
X_MULTICAST = {16: "17-64:4"}
# Rubin (compute capability 10.7) one-shot split batches per TP (9-16 rows at TP16)
# whose tcgen05 producer takes the same resident ring and x multicast (value: the
# cluster size; N 16 holds 8-row x slices at 2). Same MMAs, bitwise. 4 ranks + 12
# emulated, M16 one-shot pair us/call 7.14 -> 6.83 (with OPROJ_VOTE at 9-16: 6.70).
ONESHOT_RING = {16: "9-16:2"}


def producer_ring(tp, batch, device):
    """(resident, x multicast cluster size) of the two-shot producer at ``batch``."""
    if torch.cuda.get_device_capability(device) != (10, 7):
        return False, 1
    resident = bool(batch_table(RESIDENT_RING.get(tp, "")).get(batch))
    xmc = batch_table(X_MULTICAST.get(tp, "")).get(batch, 1) if resident else 1
    n = (batch + 7) // 8 * 8
    return resident, xmc if n % (8 * xmc) == 0 else 1


def oneshot_ring(tp, batch, device):
    """(resident, x multicast cluster size) of the one-shot split producer."""
    if torch.cuda.get_device_capability(device) != (10, 7):
        return False, 1
    xmc = batch_table(ONESHOT_RING.get(tp, "")).get(batch)
    if not xmc:
        return False, 1
    n = (batch + 7) // 8 * 8
    return True, xmc if n % (8 * xmc) == 0 else 1


def twoshot_batches(tp):
    table = batch_table(TWOSHOT.get(tp, ""))
    return {batch for batch, on in table.items() if on}


def engine(batch, tp):
    """Producer engine above B8 (GB300): grouped mma.sync for TP8 up to B16, else
    tcgen05."""
    default = "mma" if tp == 8 and batch <= 16 else "umma"
    return default


def producer(
    k, batch, tp, mla, twoshot=False, kind=None, reduce=False, resident=False, xmc=1
):
    """B<=8: one mma.sync pass, TP8 stages 8 (MLA 4) of 12 K blocks per warp. Above
    B8 (``kind`` overrides ``engine``): the tcgen05 engine, or 2-tile warps over 8 K
    parts in 16-token groups, TP8 staging 2 of 6 K blocks per warp."""
    if batch <= 8:
        staged = 0 if tp == 16 else 4 if mla else 8
        return OprojProducer(k, staged_blocks=staged, twoshot=twoshot)
    if (kind or engine(batch, tp)) == "umma":
        return OprojUmma(k, twoshot=twoshot, reduce=reduce, resident=resident, xmc=xmc)
    grouped = dict(rows=2, group_tiles=2, twoshot=twoshot)
    return OprojProducer(k, staged_blocks=0 if tp == 16 else 2, **grouped)


@cache
def compile_oproj(
    kind,
    k,
    batch,
    tp,
    nb,
    has_delta,
    mla,
    front=False,
    cap=64,
    twoshot=False,
    x_shared=False,
    polled_x=False,
    drain=False,
    batched=False,
    reduced=False,
    resident=False,
    xmc=1,
    vote=False,
    chunk=64,
):
    """Producer, consumer or fused kernel over a [TP, cap, 7168] parity view;
    ``front``: the consumer also publishes x_moe [1, 4, 7168] and the epoch hint.
    ``twoshot``: the two-shot pair (arguments as ``TwoShot.views``). ``x_shared``:
    the Rubin fused producer (``shared_x``); ``polled_x``: it polls x (True: MLA
    layers, mla.dcp.POLLED_OUTPUT; "half": behind KdaHead, kda.head.POLLED_OUT);
    ``drain``: it fences its multicast publish (``DRAIN``); ``batched``: the
    two-shot consumer's owners poll with every load in flight (``TWOSHOT_BATCHED``);
    ``reduced``: the two-shot producer reduces the owned columns
    (``PRODUCER_REDUCE``); ``resident``: the tcgen05 producer holds every K block
    (``RESIDENT_RING``); ``xmc``: its clusters multicast x (``X_MULTICAST``);
    ``vote``: the B<=8 consumer's poll (``OPROJ_VOTE``); ``chunk``: the wide
    producer's token rows per chunk (``WIDE_CHUNK``)."""
    box = fake((1 if twoshot else tp, cap, 7168), (cap * 7168, 7168, 1))
    x, w = fake((batch, k), (k, 1)), fake((7168, k), (k, 1))
    mc, slot, eps = cutlass.Int64(0), cutlass.Int32(0), cutlass.Float32(1e-6)
    recv, peers = fake((tp, cap, 7168 // tp)), fake((tp,), None, torch.int64)
    if kind == "producer_owners":  # row-sharded AttnRes: the row owners' slots
        owners = fake((tp, cap, 7168))  # cap: rows per owner
        kernel = OprojUmmaRows(k, owners=True)
        return cute.compile(kernel, x, w, owners, peers, slot, mc, make_fake_stream())
    if kind.startswith("producer") and batch > OPROJ_ROWS:  # wide: OprojUmmaRows
        tail = (recv, peers, slot, box, mc, mc)  # + gather, its multicast, x's map
        kernel = OprojUmmaRows(k, chunk=chunk)
        return cute.compile(kernel, x, w, *tail, make_fake_stream())
    if kind.startswith("producer"):  # producer_mma / producer_umma: engine A/B
        kernel = producer(
            k, batch, tp, mla, twoshot, kind[9:] or None, reduced, resident, xmc
        )
        tail = (recv, peers, slot) if twoshot else (box, mc, slot)
        tail += (box, mc) if reduced else ()  # the gather and its multicast address
        tail += (mc,) if xmc > 1 else ()  # x's tensor map (device address)
        return cute.compile(kernel, x, w, *tail, make_fake_stream())
    row, weight = dynamic_fake(2), fake((7168,), (1,))
    xmoe = fake((1, 4, 7168), (4 * 7168, 7168, 1)) if front else None
    hints = fake((224,), (1,), torch.int32) if front else None
    rows = (row, dynamic_fake(3), weight, weight, weight, row, xmoe, hints, eps, eps)
    if kind == "consumer_owners":  # this rank's rows: TP slots in rank order
        kernel = OprojConsumer(nb, groups=1, has_delta=has_delta, out_mc=True)
        return cute.compile(kernel, box, *rows, mc, slot, make_fake_stream())
    if kind == "consumer":  # B64: one 128-thread group per CTA polls every rank
        options = dict(has_delta=has_delta)
        groups = 4 if batch <= 32 and not twoshot else 1
        options.update(batched=batched, reduced=reduced, vote=vote)
        kernel = OprojConsumer(nb, groups=groups, twoshot=twoshot, **options)
        extra = (recv, mc, slot) if twoshot else ()
        return cute.compile(kernel, box, *rows, *extra, make_fake_stream())
    options = dict(has_delta=has_delta, x_shared=x_shared, polled_x=polled_x)
    kernel = OprojFused(nb, drain=drain, vote=vote, **options)
    return cute.compile(kernel, x, w, box, mc, slot, *rows, make_fake_stream())


class PreparedOproj:
    """One decoder layer's o-projection + all-reduce + post-attention AttnRes.
    ``polled``: fused x_shared batches whose x is DcpValue's Lamport output
    (mla.dcp.POLLED_OUTPUT), or, with ``after_head``, KdaHead's polled output
    (kda.head.POLLED_OUT); ``run(..., polled=True)`` selects those kernels."""

    def __init__(
        self, decoder, mailbox, batches, mla, front=None, twoshot=None, polled=(),
        head_polled=(), x_buffer=None, sharded=None,
    ):  # fmt: skip
        weight = decoder.self_attn.o_proj.weight
        norm, out_norm = decoder.mlp_res_norm, decoder.post_attention_layernorm
        weights = (norm.weight, decoder.mlp_res_proj.weight.squeeze(0), out_norm.weight)
        self.views = tuple(to_cute(w.reshape(-1)) for w in weights)
        self.eps = tuple(cutlass.Float32(n.variance_epsilon) for n in (norm, out_norm))
        parity, rank = decoder.layer_idx % 2, cutlass.Int32(mailbox.rank)
        local_box, mc = mailbox.slot(parity)
        self.device, self.box = weight.device, to_cute(local_box)
        self.head = (self.box, cutlass.Int64(mc), rank)
        self.twoshot: Any = None
        if twoshot is not None:
            recv, peers, gather, gather_mc = twoshot.views(parity)
            recv = to_cute(recv)
            self.twoshot = (
                (recv, to_cute(peers), rank),
                to_cute(gather),
                (recv, cutlass.Int64(gather_mc), rank),
            )
        self.wview, tp, k = to_cute(weight), mailbox.tp, weight.shape[1]
        nb = decoder.prev_valid_blocks + decoder.is_block_write_layer
        delta = not decoder.is_block_write_layer
        self.front = None if front is None else (*map(to_cute, front[:2]), front[2])
        # Wide batches (``choose``): OprojUmmaRows + the reduced-gather consumer over
        # a two-shot box holding ``x_buffer``'s rows (the deployment's largest graph).
        self.wide: dict[int, tuple] = {}
        self.owners: dict[int, tuple] = {}  # row-sharded wide batches
        wide = [b for b in batches if b > OPROJ_ROWS]
        if wide:
            cap = x_buffer.shape[0] if x_buffer is not None else 0
            if twoshot is None or tp != 16 or cap < max(wide):
                raise ValueError(f"o-projection batches above {OPROJ_ROWS}: {wide}")
            self.wide_box = box = twoshot.rows_box(cap)
            recv, peers, gather, gather_mc = box.views(parity)
            recv, peers, gather = to_cute(recv), to_cute(peers), to_cute(gather)
            gather_mc = cutlass.Int64(gather_mc)
            for b in wide:
                rows = x_buffer[:b]
                # x tiles {64 K, chunk rows}, over whole chunks where x_buffer holds
                # them: the rows past b load in bounds and are never pushed (a partial
                # last tile costs the producer +2.5-3.3 us at M65-80, gemm64); kept
                # alive here.
                chunk = wide_chunk(tp, Shape(b))
                tile = min(-(-b // chunk) * chunk, cap)
                tma = weight_map(x_buffer[:tile], chunk)
                address = cutlass.Int64(tma.data_ptr())
                shape = (k, b, tp, nb, delta, mla, False, cap, True)
                codes = (  # the producer reduces; the consumer only polls the gather
                    compile_oproj("producer", *shape, chunk=chunk),
                    compile_oproj("consumer", *shape, False, False, False, False, True),
                )
                carve = rows_policy(WIDE_CONSUMER_CARVEOUT.get(tp, ""), Shape(b), None)
                if carve is not None:  # every layer's wide consumer at this batch
                    prepare_shared_carveout(codes[1], self.device.index, carve)
                head = (recv, peers, rank, gather, gather_mc, address)
                tail = (recv, gather_mc, rank)
                self.wide[b] = codes, head, gather, tail, rows, tma
            # Row-sharded batches (``sharded`` = (batches, attn_res.sharded.ShardBox)):
            # the producer pushes each token's rows to its row owner, whose consumer
            # sums the TP slots in rank order (one-shot: the same sums and rounding as
            # the column owners'), runs the AttnRes on its rows and multicasts them into
            # gather box 0; ``GatherRows`` copies every row into ``out``.
            for b in wide if sharded is not None and sharded[1] is not None else ():
                if b not in sharded[0]:
                    continue
                shard = sharded[1]
                shape = (k, b, tp, nb, delta, mla, False, shard.per_cap)
                codes = (
                    compile_oproj("producer_owners", *shape),
                    compile_oproj("consumer_owners", *shape),
                )
                address = self.wide[b][-1].data_ptr()  # x's map
                owners = to_cute(shard.owners.buffer)
                head = (
                    owners,
                    to_cute(shard.owner_peers),
                    rank,
                    cutlass.Int64(address),
                )
                self.owners[b] = codes, head, owners, shard
            batches = [b for b in batches if b <= OPROJ_ROWS]
        self.codes, self.twoshots = {}, twoshot_batches(tp) if self.twoshot else set()
        self.reduced: set[int] = set()  # two-shot batches reduced by the producer
        # X_MULTICAST batches: (x rows, the x tensor map) of ``x_buffer`` [:batch]
        self.xmaps: dict[int, tuple] = {}
        self.after_head: dict[int, tuple] = {}  # codes behind KdaHead (``PREFER_L1``)
        self.polled: dict[int, tuple] = {}  # codes polling x (``polled``)
        self.head_polled: dict[int, tuple] = {}  # behind KdaHead, polling its x
        for b in batches:
            publish = front is not None and b in front[2]
            fused = fused_schedule(tp, b, mla, self.device)
            kinds = ("fused",) if fused else ("producer", "consumer")
            twoshot = b in self.twoshots and not fused
            shape = (k, b, tp, nb, delta, mla)
            staged_x = fused and shared_x(tp, b, self.device)
            fence = fused and drain(tp, b, self.device)
            tail = (64, twoshot, staged_x, False, fence)
            batched = twoshot and twoshot_batched(tp, b, self.device)
            reduced = twoshot and producer_reduce(tp, b, self.device)
            if reduced:
                self.reduced.add(b)
            resident, xmc = (0, 1)
            if twoshot:
                resident, xmc = producer_ring(tp, b, self.device)
            elif not fused:
                resident, xmc = oneshot_ring(tp, b, self.device)
            if xmc > 1 and x_buffer is None:
                xmc = 1
            if xmc > 1:
                rows = x_buffer[:b]
                tma = weight_map(rows, (b + 7) // 8 * 8 // xmc)  # kept alive here
                self.xmaps[b] = rows, tma, cutlass.Int64(tma.data_ptr())
            vote = oproj_vote(tp, b, self.device) and not twoshot
            self.codes[b] = tuple(
                compile_oproj(
                    kind,
                    *shape,
                    publish and kind != "producer",
                    *tail,
                    batched and kind == "consumer",
                    reduced,
                    resident and kind == "producer",
                    xmc if kind == "producer" else 1,
                    vote and kind != "producer",
                )
                for kind in kinds
            )
            extra = (False, False, False, 1, vote)  # batched, reduced, resident, xmc
            carve = consumer_carveout(tp, b, self.device) if twoshot else None
            if carve is not None:  # every layer's consumer at this batch: one function
                prepare_shared_carveout(self.codes[b][1], self.device.index, carve)
            carve = None if fused else producer_carveout(tp, b, self.device)
            if carve is not None:  # likewise every layer's producer
                prepare_shared_carveout(self.codes[b][0], self.device.index, carve)
            if fused and not mla and prefer_l1(tp, b, self.device):
                args = ("fused", *shape, publish, 64, twoshot, staged_x, False, fence)
                self.after_head[b] = (compile_oproj_l1(*args, *extra),)
                if staged_x and b in head_polled:
                    args = (*args[:-2], "half", fence)
                    self.head_polled[b] = (compile_oproj_l1(*args, *extra),)
            if fused and staged_x and b in polled:
                args = ("fused", *shape, publish, 64, twoshot, staged_x, True, fence)
                self.polled[b] = (compile_oproj(*args, *extra),)

    def run(self, local, prefix, blocks, out, after_head=False, polled=False):
        batch, s = local.shape[0], stream(self.device)
        if batch in self.wide:
            return self.run_wide(local, prefix, blocks, out, s)
        codes, x = self.codes[batch], to_cute(local)
        if after_head:
            codes = self.after_head.get(batch, codes)
        if polled:
            codes = (self.head_polled if after_head else self.polled)[batch]
        front = (None, None)
        if self.front is not None and batch in self.front[2]:
            front = self.front[:2]
        rows = (dynamic_view(prefix), dynamic_view(blocks), *self.views)
        rows += (dynamic_view(out), *front, *self.eps, s)
        if len(codes) == 1:
            codes[0](x, self.wview, *self.head, *rows)
        elif batch in self.twoshots:
            head, gather, tail = self.twoshot
            extra = (gather, tail[1]) if batch in self.reduced else ()
            if batch in self.xmaps:
                mapped, _, address = self.xmaps[batch]
                if local.data_ptr() != mapped.data_ptr():
                    raise RuntimeError("o-proj x multicast: x is not the mapped buffer")
                extra += (address,)
            codes[0](x, self.wview, *head, *extra, s)
            codes[1](gather, *rows[:-1], *tail, s)
        else:
            extra = ()
            if batch in self.xmaps:
                mapped, _, address = self.xmaps[batch]
                if local.data_ptr() != mapped.data_ptr():
                    raise RuntimeError("o-proj x multicast: x is not the mapped buffer")
                extra = (address,)
            codes[0](x, self.wview, *self.head, *extra, s)
            codes[1](self.box, *rows)
        return out, prefix, blocks

    def run_wide(self, local, prefix, blocks, out, s):
        codes, head, gather, tail, mapped, _ = self.wide[local.shape[0]]
        if local.data_ptr() != mapped.data_ptr():
            raise RuntimeError("wide o-projection: x is not the mapped buffer")
        if local.shape[0] in self.owners:
            return self.run_owned(local, prefix, blocks, out, s)
        rows = (dynamic_view(prefix), dynamic_view(blocks), *self.views)
        rows += (dynamic_view(out), None, None, *self.eps)
        codes[0](to_cute(local), self.wview, *head, s)
        codes[1](gather, *rows, *tail, s)
        return out, prefix, blocks

    def run_owned(self, local, prefix, blocks, out, s):
        """Row-sharded: rows pushed to their owners, this rank's rows reduced and
        AttnRes'd into gather box 0, then every row gathered into ``out``."""
        codes, head, owners, shard = self.owners[local.shape[0]]
        rows_total = local.shape[0]
        first, count = owned(rows_total, shard.tp, shard.rank)
        codes[0](to_cute(local), self.wview, *head, s)
        if count:
            mine = slice(first, first + count)
            views = (dynamic_view(prefix[mine]), dynamic_view(blocks[mine]))
            rows = (*views, *self.views, views[0], None, None, *self.eps)
            mc = cutlass.Int64(shard.gathers[0].mc)
            codes[1](owners, *rows, mc, cutlass.Int32(first), s)
        gather = compile_gather(rows_total, shard.cap, self.device.index)
        gather(to_cute(shard.gathers[0].buffer), to_cute(out), s)
        return out, prefix, blocks


def choose(shape, ctx):
    """The fused o-projection (+ exchange + AttnRes) for a captured decode shape (the
    runtime's family hook), or None: the attention runs the stock o-projection, its
    all-reduce and the unfused AttnRes (seam S3). A Choice above 64 rows makes the
    runtime prepare ``PreparedOproj`` at that shape: on Rubin the ``WIDE_TWOSHOT``
    shapes run ``OprojUmmaRows`` + the reduced-gather consumer (``.wide``)."""
    if shape.rows <= OPROJ_ROWS:
        return Choice(Path.GENERIC)
    if ctx.arch != "rubin" or not rows_policy(WIDE_TWOSHOT.get(ctx.tp, ""), shape, 0):
        return None
    chunk = wide_chunk(ctx.tp, shape)
    return Choice(Path.GENERIC, ("rows", chunk), f"two-shot, {chunk}-row chunks")

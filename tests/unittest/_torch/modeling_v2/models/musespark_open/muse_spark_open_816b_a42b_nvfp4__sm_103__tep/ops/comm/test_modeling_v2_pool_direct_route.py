# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the ``pool_into_route`` ops (TP8, two GB300 nodes): the routed pool kernel on a direct transport.

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, e.g.
``srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>``;
anywhere else every test skips.

``pool_direct_route`` claims ``pool_gather_route``'s bits on the pool slices and
logits its producers published into the transport: the pool GEMM (the plan's
kernel at each row count) with the slot as its output, and the router shard's
publish twin (``router_shard_publish``) on the router fork's side stream. Checked as raw
bits, against the router fork's chain (the side router into a fresh tensor, the pool GEMM
into a fresh tensor, ``pool_gather_route`` on the tagged inbox):

* the publish twin writes every box of its record with ``router_shard_side``'s
  logits, every rank's record in every rank's copy, and nothing past its rows;
* the 60-layer chain as the decode graph runs it -- attention-hop stand-in (PDL,
  triggers before its stores, as the hop's early trigger), router on the side stream as a plain
  launch, pool GEMM on the main stream, joined before the routed kernel, an
  expert-kernel stand-in -- eagerly and in one CUDA graph with GPU sleeps on
  rotating ranks on both streams, replayed back to back with no host sync and
  with the inputs changed, for rows 1..8 and every placement;
* every cell ``pool_direct_route`` declares (rows, placement -- including routing
  only, owner-only and static -- and buffer), one layer, eagerly and in CUDA graphs;
* a concurrent poller that records the first value of every pool half and every
  logit word while the producers write: it is the final value;
* last: every word of the transport is back to the sentinel and no guard fired.

``MODELING_V2_TIMING=1`` also runs the chain microbenchmark (printed, never gated):
the 60-layer chain of cold weights per arm, ``router_fork`` (tagged) vs ``pool_direct`` (direct), rounds
in alternating order; per layer the chain time and the segment from the hop's end to
the expert stand-in's dependency release on each rank.
"""

import importlib
import json
import os
import statistics
import time

import pytest
import torch

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

__extra_import_path__ = ["."]
from _pool_direct_kernels import hop_standin, moe_standin  # noqa: E402

_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)
R = importlib.import_module(_TARGET + ".ops.comm.pool_gather_route")
D = importlib.import_module(_TARGET + ".ops.comm.pool_direct_route")
S = importlib.import_module(_TARGET + ".ops.moe.router_shard")
GEMM = importlib.import_module(_TARGET + ".ops.gemm.dense_gemm")
PLAN = importlib.import_module(_TARGET + ".kernels.dense_gemms.dispatcher")
INBOX = importlib.import_module(_TARGET + ".ops.comm.projection_inbox")
MP = importlib.import_module(_TARGET + ".kernels.moe_prologue.prologue")
RB = importlib.import_module(_TARGET + ".kernels.replicated_experts.ring_buddy")

WORLD = 8
#: The model's routed layers (the 60 NVFP4-expert layers) and their buffers.
LAYERS = 60
BUFFERS = D.buffer_sequence(LAYERS)
EPS, RENORM_EPS = 1e-5, 1e-15
SCALE = (8192 / 256.0) ** -0.5
MODEL_HIDDEN = 8192
#: The model's router launch warp count (the built-in launch with the prepare stream).
WARPS = 16
#: Hop stand-in: poll time before its trigger and boundary time after it (the early trigger's lead).
HOP_PRE_NS, HOP_POST_NS = 3_000, 1_500
#: Expert stand-in spin after its dependency release.
MOE_NS = 3_000
SENTINEL_I32 = D.DIRECT_SENTINEL - (1 << 32)
PLACES = {
    "balance3": R.PLACE_BALANCE3,
    "balance": R.PLACE_BALANCE,
    "none": R.PLACE_NONE,
    "owner": R.PLACE_OWNER,
    "static": R.PLACE_STATIC,
}
_OUTPUTS = (
    "logits",
    "selected",
    "weights",
    "local_ids",
    "weights_bf16",
    "stats",
    "fp4",
    "scale",
    "gathered",
)
POLLER_CALLS = int(os.environ.get("MODELING_V2_POOL_DIRECT_CALLS", "10240"))


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs eight MPI ranks with CUDA (two four-GPU nodes)")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
        pytest.skip("needs sm_103 or sm_107")
    if not D.available():
        pytest.skip("needs CuTe DSL and flashinfer's RMSNormFP4QuantKernel")
    return comm, rank


def _settle(comm):
    torch.cuda.synchronize()
    comm.Barrier()


def _agree(comm, ok: bool) -> bool:
    return comm.allreduce(int(bool(ok)), op=MPI.SUM) == WORLD


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({4: torch.int32, 2: torch.int16, 1: torch.uint8}[t.element_size()])


def _differing(got, want) -> dict:
    bad = {}
    for g_layer, w_layer in zip(got, want):
        for name, g, w in zip(_OUTPUTS, g_layer, w_layer):
            if g is None and w is None:
                continue
            assert g.shape == w.shape and g.dtype == w.dtype, (name, g.shape, w.shape)
            bad[name] = bad.get(name, 0) + int((_bits(g) != _bits(w)).sum().item())
    return bad


@pytest.fixture(scope="module")
def group():
    """The tagged inbox, the direct transport, the router fork's side stream and the weights (collective)."""
    comm, rank = _world()
    from tensorrt_llm.mapping import Mapping

    e = type("Env", (), {})()
    e.comm, e.rank = comm, rank
    e.mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    e.inbox = INBOX.ProjectionInbox(e.mapping, row_words=R.ROW_WORDS)
    e.ws = D.PoolDirectWorkspace(e.mapping)
    e.side = MP.dedicated_stream()
    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(1234)  # the same router on every rank
    # One router weight and one pool slice per layer: the chain streams cold weights.
    e.router = [
        torch.randn(256, MODEL_HIDDEN, generator=gen, device=device) * 0.02 for _ in range(LAYERS)
    ]
    e.bias = (torch.randn(256, generator=gen, device=device) * 0.05).contiguous()
    own = torch.Generator(device=device).manual_seed(4321 + rank)
    e.pool = [
        (torch.randn(R.LOCAL_COLUMNS, MODEL_HIDDEN, generator=own, device=device) * 0.02).to(
            torch.bfloat16
        )
        for _ in range(LAYERS)
    ]
    e.gain = torch.empty(R.HIDDEN, device=device, dtype=torch.bfloat16).normal_(1, 0.1)
    e.global_scale = torch.tensor([224.0], device=device, dtype=torch.float32)
    e.kernels = PLAN.default_kernels()["pool"]
    e.sink = torch.zeros(64, dtype=torch.int32, device=device)
    e.stamps = torch.zeros(LAYERS, 2, dtype=torch.int64, device=device)
    e.graphs = {}
    _settle(comm)
    yield e
    torch.cuda.synchronize()
    e.graphs.clear()
    comm.Barrier()


class _Chain:
    """Per-layer decoder rows (replicated on every rank): staged by the host, written by the hop."""

    def __init__(self, rows: int):
        self.rows = rows
        self.stage = torch.zeros(LAYERS, rows, MODEL_HIDDEN, device="cuda", dtype=torch.bfloat16)
        self.live = torch.zeros_like(self.stage)

    def fill(self, seed: int):
        gen = torch.Generator().manual_seed(seed * 16 + 99)
        scale = torch.exp(1.5 * torch.randn(LAYERS, self.rows, 1, generator=gen))
        self.stage.copy_((torch.randn(*self.stage.shape, generator=gen) * scale).bfloat16())


def _delays(rows: int, rank: int) -> dict:
    """GPU sleeps (cycles) on a rotating rank per layer: before the hop or the side router."""
    return {
        layer: (2_000, 20_000, 200_000)[(layer // 3) % 3]
        for layer in range(LAYERS)
        if layer % 3 == 0 and (layer // 3 + rows) % WORLD == rank
    }


def _build(e, rows_list, places):
    """Compile the direct variants a test launches, on every rank together (collective)."""
    e.ws.build_variants([D.variant(place) for place in places])
    del rows_list


def _layers(e, chain: _Chain, arm: str, place: int, delays=None, side_delays=None, stamps=True):
    """``LAYERS`` layers: hop stand-in -> (side router || pool GEMM) -> routed kernel -> expert
    stand-in. ``arm``: ``"router_fork"`` (tagged) or ``"pool_direct"`` (direct)."""
    main = torch.cuda.current_stream()
    delays = delays or {}
    side_delays = side_delays or {}
    rows = chain.rows
    kernel = e.kernels[rows - 1]
    outputs = []
    stamp = e.stamps.data_ptr()
    for layer in range(LAYERS):
        if layer in delays:
            torch.cuda._sleep(delays[layer])
        hop_standin(
            chain.stage[layer],
            chain.live[layer],
            HOP_PRE_NS,
            HOP_POST_NS,
            stamp + (2 * layer) * 8 if stamps else stamp,
        )
        hidden = chain.live[layer]
        if arm == "router_fork":
            logits_local = torch.empty(rows, S.LOCAL_EXPERTS, device="cuda")
            e.side.wait_stream(main)
            with torch.cuda.stream(e.side):
                if layer in side_delays:
                    torch.cuda._sleep(side_delays[layer])
                S.router_shard_side(hidden, e.router[layer], e.rank, SCALE, WARPS, logits_local)
            local = GEMM.dense_gemm(hidden, e.pool[layer], kernel, True)
            main.wait_stream(e.side)
            out = R.pool_gather_route(
                local,
                logits_local,
                e.inbox,
                e.gain,
                e.global_scale,
                EPS,
                e.bias,
                RENORM_EPS,
                place,
                R.LOCAL_EXPERTS,
                enable_pdl=True,
                return_gathered=True,
            )
        else:
            buffer = BUFFERS[layer]
            e.side.wait_stream(main)
            with torch.cuda.stream(e.side):
                if layer in side_delays:
                    torch.cuda._sleep(side_delays[layer])
                S.router_shard_publish(
                    hidden, e.router[layer], e.rank, SCALE, WARPS, e.ws.boxes(buffer, rows)
                )
            GEMM.dense_gemm(hidden, e.pool[layer], kernel, True, out=e.ws.slot(buffer, rows))
            main.wait_stream(e.side)
            out = D.pool_direct_route(
                e.ws,
                buffer,
                rows,
                e.gain,
                e.global_scale,
                EPS,
                e.bias,
                RENORM_EPS,
                place,
                R.LOCAL_EXPERTS,
                True,
                True,
            )
        moe_standin(out.fp4, stamp + (2 * layer + 1) * 8 if stamps else stamp + 8, MOE_NS, e.sink)
        outputs.append(tuple(getattr(out, name) for name in _OUTPUTS))
    return outputs


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("warps", [4, 16])
@torch.inference_mode()
def test_the_publish_twin_fills_every_box_of_every_record(group, rows, warps):
    """Every rank publishes one layer's logits into buffer ``rows % 3``: every rank's copy
    holds, in each rank's record, ``publish_copies(rows)`` boxes of ``router_shard_side``'s
    logits for that rank; every other word of the transport is the sentinel."""
    e = group
    buffer = rows % D.BUFFERS
    gen = torch.Generator().manual_seed(rows * 31 + warps)
    hidden = (torch.randn(rows, MODEL_HIDDEN, generator=gen) * 2).to(torch.bfloat16).cuda()
    _settle(e.comm)
    S.router_shard_publish(hidden, e.router[3], e.rank, SCALE, warps, e.ws.boxes(buffer, rows))
    _settle(e.comm)
    copy = e.ws.receive().view(D.BUFFERS, WORLD, D.RECORD_WORDS)
    expected = torch.full_like(copy, SENTINEL_I32)
    copies = S.publish_copies(rows)
    for r in range(WORLD):
        logits = torch.empty(rows, S.LOCAL_EXPERTS, device="cuda")
        S.router_shard_side(hidden, e.router[3], r, SCALE, warps, logits)
        boxes = expected[buffer, r, D.POOL_SLOT_WORDS :].view(D.CTAS, D.MAX_ROWS, S.LOCAL_EXPERTS)
        for c in range(copies):
            boxes[c, :rows] = logits.view(torch.int32)
    bad = int((copy != expected).sum().item())
    _settle(e.comm)
    e.ws.reset()
    _settle(e.comm)
    assert _agree(e.comm, bad == 0), f"rank {e.rank}: {bad} words differ"


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("place_name", ["balance3", "balance"])
@torch.inference_mode()
def test_the_direct_chain_is_the_tagged_chain_bit_for_bit(group, rows, place_name):
    """The 60-layer chain, eager, then captured and replayed with sleeps on rotating ranks
    (before the hop and before the side router), new inputs, and back to back."""
    e = group
    place = PLACES[place_name]
    _build(e, [rows], [place])
    chain = _Chain(rows)
    chain.fill(rows)
    delays = _delays(rows, e.rank)
    side_delays = _delays(rows + 3, e.rank)
    _settle(e.comm)
    want = _layers(e, chain, "router_fork", place)
    got = _layers(e, chain, "pool_direct", place)
    _settle(e.comm)
    results = [("eager", _differing(got, want))]
    graphs = {}
    outs = {}
    for arm in ("router_fork", "pool_direct"):
        graphs[arm] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[arm]):
            outs[arm] = _layers(e, chain, arm, place, delays, side_delays)
        _settle(e.comm)
    for repeat in range(3):
        chain.fill(100 * rows + repeat)
        _settle(e.comm)
        graphs["router_fork"].replay()
        graphs["pool_direct"].replay()
        _settle(e.comm)
        results.append((f"replay {repeat}", _differing(outs["pool_direct"], outs["router_fork"])))
    for _ in range(6):
        graphs["pool_direct"].replay()
    graphs["router_fork"].replay()
    _settle(e.comm)
    results.append(("6 back-to-back replays", _differing(outs["pool_direct"], outs["router_fork"])))
    bad = {what: {k: v for k, v in diff.items() if v} for what, diff in results}
    bad = {what: diff for what, diff in bad.items() if diff}
    clean = e.ws.is_clean()
    fired = e.ws.guard_fired()
    graphs.clear()
    assert _agree(e.comm, not bad and clean and not fired), (
        f"rank {e.rank}: differing {bad}, transport clean {clean}, guard fired {fired}"
    )


def _one_layer(e, hidden: torch.Tensor, arm: str, rows: int, place: int, static_from: int, buffer):
    """One routed layer on the current stream: router, pool GEMM, routed kernel.

    ``arm``: ``"tagged"`` (``router_shard_side`` and the pool GEMM into fresh tensors,
    ``pool_gather_route`` on the inbox) or ``"direct"`` (the publish twin and the pool GEMM
    into ``buffer`` of the transport, ``pool_direct_route``).
    """
    kernel = e.kernels[rows - 1]
    if arm == "tagged":
        logits_local = torch.empty(rows, S.LOCAL_EXPERTS, device="cuda")
        S.router_shard_side(hidden, e.router[0], e.rank, SCALE, WARPS, logits_local)
        local = GEMM.dense_gemm(hidden, e.pool[0], kernel, True)
        out = R.pool_gather_route(
            local,
            logits_local,
            e.inbox,
            e.gain,
            e.global_scale,
            EPS,
            e.bias,
            RENORM_EPS,
            place,
            static_from,
            enable_pdl=True,
            return_gathered=True,
        )
    else:
        S.router_shard_publish(hidden, e.router[0], e.rank, SCALE, WARPS, e.ws.boxes(buffer, rows))
        GEMM.dense_gemm(hidden, e.pool[0], kernel, True, out=e.ws.slot(buffer, rows))
        out = D.pool_direct_route(
            e.ws,
            buffer,
            rows,
            e.gain,
            e.global_scale,
            EPS,
            e.bias,
            RENORM_EPS,
            place,
            static_from,
            True,
            True,
        )
    return tuple(getattr(out, name) for name in _OUTPUTS)


def _cell_differences(got, want) -> dict:
    """Differing words per output; an output one side returns and the other does not is -1."""
    bad = {}
    for name, g, w in zip(_OUTPUTS, got, want):
        if g is None and w is None:
            continue
        if g is None or w is None or g.shape != w.shape or g.dtype != w.dtype:
            bad[name] = -1
            continue
        count = int((_bits(g) != _bits(w)).sum().item())
        if count:
            bad[name] = count
    return bad


@pytest.mark.parametrize("cell", D.PoolDirectRoute.CELLS, ids=lambda c: c.why[:40])
@torch.inference_mode()
def test_every_cell_routes_the_tagged_bits(group, cell):
    """Each declared cell (rows, placement, buffer), one layer: the publish twin and the pool
    GEMM write the cell's buffer and ``pool_direct_route`` returns ``pool_gather_route``'s
    outputs on the tagged inbox bit for bit, eagerly and through CUDA-graph replays with new
    inputs; afterwards the transport is back to the sentinel and no guard fired."""
    e = group
    rows, place, buffer = cell.spec["rows"], cell.spec["place"], cell.spec["buffer"]
    static_from = RB.STATIC_FROM if place == R.PLACE_STATIC else R.LOCAL_EXPERTS
    e.ws.build_variants([D.variant(place, static_from)])
    hidden = torch.zeros(rows, MODEL_HIDDEN, device="cuda", dtype=torch.bfloat16)

    def fill(seed: int):
        gen = torch.Generator().manual_seed(seed)
        scale = torch.exp(1.5 * torch.randn(rows, 1, generator=gen))
        hidden.copy_((torch.randn(rows, MODEL_HIDDEN, generator=gen) * scale).bfloat16())

    fill(7 * rows + 3 * place + buffer)
    _settle(e.comm)
    want = _one_layer(e, hidden, "tagged", rows, place, static_from, buffer)
    _settle(e.comm)
    got = _one_layer(e, hidden, "direct", rows, place, static_from, buffer)
    _settle(e.comm)
    results = [("eager", _cell_differences(got, want))]
    graphs, outs = {}, {}
    for arm in ("tagged", "direct"):
        graphs[arm] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[arm]):
            outs[arm] = _one_layer(e, hidden, arm, rows, place, static_from, buffer)
        _settle(e.comm)
    for repeat in range(4):
        fill(1000 * rows + 100 * place + 10 * buffer + repeat)
        _settle(e.comm)
        if e.rank == repeat:
            time.sleep(0.002)
        graphs["tagged"].replay()
        _settle(e.comm)
        if e.rank == WORLD - 1 - repeat:
            time.sleep(0.002)
        graphs["direct"].replay()
        _settle(e.comm)
        results.append((f"replay {repeat}", _cell_differences(outs["direct"], outs["tagged"])))
    bad = {what: diff for what, diff in results if diff}
    clean = e.ws.is_clean()
    fired = e.ws.guard_fired()
    graphs.clear()
    _settle(e.comm)
    assert _agree(e.comm, not bad and clean and not fired), (
        f"rank {e.rank} cell {cell.why!r}: differing {bad}, transport clean {clean}, "
        f"guard fired {fired}"
    )


# ---------------------------------------------------------------------------
# Concurrent poller: the first value of every pool half and logit word is the final one.
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

if triton is not None:
    #: Words per sender of the diagnostic history (a whole pool slot).
    HISTORY_WORDS = tl.constexpr(2048)

    @triton.jit
    def _first_seen_kernel(
        receive,
        record_words,
        buffer_words,
        start_words,
        counts,
        max_rounds,
        history,
        call_index,
        events,
        event_count,
        PART: tl.constexpr,
        BLOCK: tl.constexpr,
        HALVES: tl.constexpr,
        EVENTS: tl.constexpr,
        CAP: tl.constexpr,
    ):
        """Poll ``PART`` words of one sender's record (from ``start_words``) until none is
        pending; record each unit's first non-sentinel value (a 16-bit half with
        ``HALVES``, else a 32-bit word), compare it with a final read, reset the words.

        ``counts`` (int64): [0] units whose first value differs from the final one, [1]
        units the first read saw as the sentinel, [2] blocks that hit ``max_rounds``,
        [3] units still the sentinel in those blocks. With ``EVENTS`` (diagnostics) the
        final words go to ``history[call_index, sender, offset]`` (int32, ``HISTORY_WORDS``
        per sender) and each changed word to ``events`` (int32 ``[CAP, 6]``: call, sender,
        offset, first word, final word, whether the first read saw the sentinel).
        """
        sender = tl.program_id(1)
        lane = tl.arange(0, BLOCK)
        offsets = tl.program_id(0) * PART + lane
        mask = lane < PART
        base = receive + sender.to(tl.int64) * record_words + buffer_words + start_words
        words = tl.load(base + offsets, mask=mask, other=0, volatile=True)
        initial = words
        if HALVES:
            low = words & 0xFFFF
            high = (words >> 16) & 0xFFFF
            unseen = tl.sum(
                tl.where(mask, (low == 0xFFFE).to(tl.int32) + (high == 0xFFFE).to(tl.int32), 0),
                axis=0,
            )
            first_low = low
            first_high = high
            pending = unseen
            rounds = 0
            while (pending > 0) & (rounds < max_rounds):
                words = tl.load(base + offsets, mask=mask, other=0, volatile=True)
                low = words & 0xFFFF
                high = (words >> 16) & 0xFFFF
                first_low = tl.where(first_low == 0xFFFE, low, first_low)
                first_high = tl.where(first_high == 0xFFFE, high, first_high)
                pending = tl.sum(
                    tl.where(
                        mask,
                        (first_low == 0xFFFE).to(tl.int32) + (first_high == 0xFFFE).to(tl.int32),
                        0,
                    ),
                    axis=0,
                )
                rounds += 1
            final = tl.load(base + offsets, mask=mask, other=0, volatile=True)
            changed = ((final & 0xFFFF) != first_low).to(tl.int32) + (
                ((final >> 16) & 0xFFFF) != first_high
            ).to(tl.int32)
            first_word = first_low | (first_high << 16)
            pending_first = (initial & 0xFFFF) == 0xFFFE
        else:
            unseen = tl.sum(tl.where(mask, (words == -65538).to(tl.int32), 0), axis=0)
            first = words
            pending = unseen
            rounds = 0
            while (pending > 0) & (rounds < max_rounds):
                words = tl.load(base + offsets, mask=mask, other=0, volatile=True)
                first = tl.where(first == -65538, words, first)
                pending = tl.sum(tl.where(mask, (first == -65538).to(tl.int32), 0), axis=0)
                rounds += 1
            final = tl.load(base + offsets, mask=mask, other=0, volatile=True)
            changed = (final != first).to(tl.int32)
            first_word = first
            pending_first = initial == -65538
        if EVENTS:
            row = (sender + call_index * 8).to(tl.int64)
            hist = history + row * HISTORY_WORDS + offsets
            tl.store(hist, final, mask=mask)
            bad = mask & (changed > 0)
            slot = tl.atomic_add(
                event_count + tl.zeros([BLOCK], tl.int32), tl.full([BLOCK], 1, tl.int32), mask=bad
            )
            keep = bad & (slot < CAP)
            event = events + slot.to(tl.int64) * 6
            tl.store(event + 0, tl.zeros([BLOCK], tl.int32) + call_index, mask=keep)
            tl.store(event + 1, tl.zeros([BLOCK], tl.int32) + sender, mask=keep)
            tl.store(event + 2, offsets, mask=keep)
            tl.store(event + 3, first_word, mask=keep)
            tl.store(event + 4, final, mask=keep)
            tl.store(event + 5, pending_first.to(tl.int32), mask=keep)
        tl.atomic_add(counts + 0, tl.sum(tl.where(mask, changed, 0), axis=0).to(tl.int64))
        tl.atomic_add(counts + 1, unseen.to(tl.int64))
        tl.atomic_add(counts + 2, (pending > 0).to(tl.int64))
        tl.atomic_add(counts + 3, pending.to(tl.int64))
        tl.store(base + offsets, tl.full([BLOCK], -65538, tl.int32), mask=mask)


def _trace_events(rank, producer, history, events, count, where, is_router, limit=40):
    """Lines describing each changed word: the call and position, the first and final
    words, and the latest earlier call whose final word at that position was the first
    value (a store of that call seen late), or where else in the same call it was final."""
    import numpy as np

    hist = history.cpu().numpy()
    rows_of = events[: min(count, events.shape[0])].cpu().numpy()
    lines = [f"[pool_direct poller events] producer {producer} rank {rank}: {count} changed words"]
    for call, sender, offset, first, final, pending in rows_of[:limit]:
        rows, buffer = where(int(call))
        earlier = [
            j
            for j in range(int(call) - 1, -1, -1)
            if hist[j, sender, offset] == first and where(j)[1] == buffer
        ]
        same = np.argwhere(hist[int(call)] == first)
        as_float = np.array([first, final], dtype=np.int32).view(np.float32)
        lines.append(
            f"[pool_direct poller event] rank {rank} call {int(call)} rows {rows} buffer {buffer} "
            f"sender {int(sender)} offset {int(offset)} first {int(first) & 0xFFFFFFFF:#010x} "
            f"final {int(final) & 0xFFFFFFFF:#010x}"
            + (f" ({as_float[0]:.6g} -> {as_float[1]:.6g})" if is_router else "")
            + f" first-read-pending {int(pending)}; same position final in earlier calls of "
            f"this buffer {earlier[:3]} (rows {[where(j)[0] for j in earlier[:3]]}); "
            f"final elsewhere in this call {same[:3].tolist()}"
        )
    return lines


@pytest.mark.parametrize("producer", ["router", "splitk:64,8,4,12", "direct:128,2", "copy"])
@torch.inference_mode()
def test_a_concurrent_poller_never_sees_a_value_before_the_final_one(group, producer):
    """Every rank's poller, started before the producers, records the first non-sentinel
    value of every unit of all eight senders' records while the producers write them
    through the multicast address: pool halves (GEMM / copy producers) or logit words
    (the router's publish twin). It must be the final value. Rows 1..8, all buffers, at
    least 10,000 calls per producer. 4 programs per sender (32 CTAs at most).

    Each polled word is held by exactly one thread of the poller (its block of words is
    at least as wide as its threads): a word Triton spreads over several threads is
    loaded by each of them separately, so the copies can disagree while the block's
    reduced ``pending`` count, taken from one copy, says every word has arrived."""
    import numpy as np

    e = group
    if triton is None:
        pytest.skip("needs Triton")
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device=dev).manual_seed(9000 + e.rank)
    receive = e.ws.receive()
    max_rounds = 1 << 22
    side = torch.cuda.Stream(device=dev)
    is_router = producer == "router"

    def usable(rows):
        if producer in ("copy", "router"):
            return True
        return GEMM.kernel_error(producer, rows, MODEL_HIDDEN, R.LOCAL_COLUMNS) is None

    row_counts = [rows for rows in range(1, D.MAX_ROWS + 1) if usable(rows)]

    def produce(rows, buffer):
        x = torch.randn(rows, MODEL_HIDDEN, generator=g, device=dev).to(torch.bfloat16)
        if is_router:
            S.router_shard_publish(x, e.router[0], e.rank, SCALE, WARPS, e.ws.boxes(buffer, rows))
        elif producer == "copy":
            partial = torch.randn(rows, R.LOCAL_COLUMNS, generator=g, device=dev)
            D._direct.publish_copy(e.ws.slot(buffer, rows), partial.to(torch.bfloat16))
        else:
            GEMM.dense_gemm(x, e.pool[0], producer, True, out=e.ws.slot(buffer, rows))

    calls = POLLER_CALLS
    # Diagnostics (MODELING_V2_POOL_DIRECT_EVENTS=1): every call's final words and each
    # changed word, so a first value can be traced to the call that stored it.
    diag = os.environ.get("MODELING_V2_POOL_DIRECT_EVENTS", "0") == "1"
    cap = 4096
    history = torch.zeros((calls if diag else 1, WORLD, 2048), dtype=torch.int32, device=dev)
    events = torch.zeros((cap, 6), dtype=torch.int32, device=dev)
    event_count = torch.zeros(1, dtype=torch.int32, device=dev)

    def call(rows, buffer, counts, index=-1):
        if is_router:
            words, start, halves = S.publish_copies(rows) * D.BOX_WORDS, D.POOL_SLOT_WORDS, False
            # Only the rows written: rows past M of each box stay the sentinel.
            words = rows * S.LOCAL_EXPERTS
        else:
            words, start, halves = rows * D.POOL_ROW_WORDS, 0, True
        part = max(words // 4, 1)
        # One thread per word at most (see the docstring): 32 lanes or more, and no more
        # threads than lanes.
        block = max(32, triton.next_power_of_2(part))
        warps = max(1, min(4, block // 32))
        _settle(e.comm)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _first_seen_kernel[(4 if words >= 4 else 1, WORLD)](
                receive,
                D.RECORD_WORDS,
                buffer * WORLD * D.RECORD_WORDS,
                start,
                counts,
                max_rounds,
                history,
                max(index, 0),
                events,
                event_count,
                PART=part,
                BLOCK=block,
                HALVES=halves,
                EVENTS=diag and index >= 0,
                CAP=cap,
                num_warps=warps,
            )
        torch.cuda._sleep(20_000)  # let the pollers start before the stores
        produce(rows, buffer)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        if is_router:
            # The other boxes of the record (CTAs 1..) are not polled here: reset them.
            _settle(e.comm)
            e.ws.reset()

    warm = torch.zeros(4, dtype=torch.int64, device=dev)
    _settle(e.comm)
    e.ws.reset()
    for buffer in range(D.BUFFERS):
        for rows in row_counts:
            call(rows, buffer, warm)
    _settle(e.comm)
    e.ws.reset()
    counts = torch.zeros(D.MAX_ROWS, 4, dtype=torch.int64, device=dev)
    seconds = []

    def where(index):
        return row_counts[index % len(row_counts)], (index // len(row_counts)) % D.BUFFERS

    for index in range(calls):
        rows, buffer = where(index)
        began = time.perf_counter()
        call(rows, buffer, counts[rows - 1], index)
        seconds.append(time.perf_counter() - began)
    mine = counts.cpu().numpy()
    total = e.comm.allreduce(mine, op=MPI.SUM)
    _settle(e.comm)
    e.ws.reset()
    _settle(e.comm)
    if diag:
        lines = _trace_events(
            e.rank, producer, history, events, int(event_count.item()), where, is_router
        )
        gathered = e.comm.gather(lines, root=0)
        if e.rank == 0:
            for rank_lines in gathered:
                for text in rank_lines:
                    print(text)
    if e.rank == 0:
        print(
            f"\n[pool_direct poller] producer {producer}: {calls} calls x {WORLD} senders per rank, "
            f"rows {row_counts}; call p50 {1e6 * statistics.median(seconds):.0f} us"
        )
        for rows in row_counts:
            changed, watched, stuck, unseen = (int(v) for v in total[rows - 1])
            print(
                f"[pool_direct poller]   rows {rows}: units changed after first seen {changed}, "
                f"watched arriving {watched}, blocks stuck {stuck} ({unseen} never seen)"
            )
    assert int(np.asarray(total)[:, 2].sum()) == 0, f"blocks never completed: {total[:, 2]}"
    assert int(np.asarray(total)[:, 0].sum()) == 0, f"units changed: {total[:, 0]}"
    assert int(np.asarray(total)[:, 1].sum()) > 0, "no poller ever watched a store arrive"


@torch.inference_mode()
def test_the_transport_is_clean_and_no_guard_fired(group):
    e = group
    _settle(e.comm)
    clean = e.ws.is_clean()
    fired = e.ws.guard_fired()
    assert _agree(e.comm, clean and not fired), f"rank {e.rank}: clean {clean}, fired {fired}"


# ---------------------------------------------------------------------------
# Microbenchmark: MODELING_V2_TIMING=1 (printed, never gated here).
# ---------------------------------------------------------------------------


def _time(comm, graphs: dict, stamps: torch.Tensor, rounds: int, trials: int, replays: int):
    """Per graph on this rank: median us per replay over rounds x trials x replays (rounds in
    alternating order), and the median over trials of the per-layer segment hop end ->
    expert release (median over layers 1..59 of the trial's last replay)."""
    names = list(graphs)
    samples = {name: [] for name in names}
    segments = {name: [] for name in names}
    per_layer_seg = {name: [] for name in names}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for name in names:
        _settle(comm)
        graphs[name].replay()
    for r in range(rounds):
        for name in names if r % 2 == 0 else list(reversed(names)):
            for _ in range(trials):
                _settle(comm)
                start.record()
                for _ in range(replays):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000.0 / replays)
                seg = (stamps[:, 1] - stamps[:, 0]).double().cpu() / 1000.0
                segments[name].append(float(seg[1:].median()))
                per_layer_seg[name].append(seg.tolist())
    medians = {name: statistics.median(v) for name, v in samples.items()}
    seg_medians = {name: statistics.median(v) for name, v in segments.items()}
    return medians, seg_medians, samples, segments, per_layer_seg


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="MODELING_V2_TIMING=1")
@pytest.mark.parametrize("place_name", ["balance3", "balance"])
@torch.inference_mode()
def test_microbenchmark_pool_route_chain(group, place_name):
    """``router_fork`` (tagged routed kernel) vs ``pool_direct`` (direct routed kernel), 8 rows.

    The 60-layer chain of cold weights. Microseconds per layer (replay / 60) and the
    hop-end -> expert-release segment, each rank's median over rounds x trials; printed
    per arm as the maximum over ranks, and the per-rank pool_direct - router_fork deltas.
    JSON on rank 0 (``[pool_direct bench json]``)."""
    e = group
    rows = 8
    place = PLACES[place_name]
    rounds = int(os.environ.get("MODELING_V2_TIMING_ROUNDS", "6"))
    _build(e, [rows], [place])
    chain = _Chain(rows)
    chain.fill(5)
    graphs, outs = {}, {}
    for arm in ("router_fork", "pool_direct"):
        _layers(e, chain, arm, place)
        _settle(e.comm)
        graphs[arm] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[arm]):
            outs[arm] = _layers(e, chain, arm, place)
        _settle(e.comm)
    medians, seg_medians, samples, segments, per_layer = _time(
        e.comm, graphs, e.stamps, rounds=rounds, trials=5, replays=20
    )
    bad = _differing(outs["pool_direct"], outs["router_fork"])
    bad = sum(bad.values())
    clocks = None
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        clocks = (
            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
        )
    except Exception:  # noqa: BLE001 - clocks are informational
        clocks = None
    mine = {
        "rank": e.rank,
        "host": os.uname().nodename,
        "per_layer_us": {k: v / LAYERS for k, v in medians.items()},
        "segment_us": seg_medians,
        "samples_us": samples,
        "segment_samples_us": segments,
        "last_per_layer_segment_us": {k: v[-1] for k, v in per_layer.items()},
        "differing_bits": bad,
        "clocks": clocks,
    }
    results = e.comm.gather(mine, root=0)
    graphs.clear()
    if e.rank == 0:
        print(
            f"\n[pool_direct bench] place {place_name}, rows {rows}, {LAYERS} layers, {rounds} rounds"
        )
        for key, label in (("per_layer_us", "chain us/layer"), ("segment_us", "hop end->expert")):
            for arm in ("router_fork", "pool_direct"):
                values = [r[key][arm] for r in results]
                print(
                    f"[pool_direct bench] {label:16s} {arm}: max {max(values):8.3f} min {min(values):8.3f} "
                    f"per rank {' '.join(f'{v:.3f}' for v in values)}"
                )
            deltas = [r[key]["pool_direct"] - r[key]["router_fork"] for r in results]
            slowest = max(r[key]["pool_direct"] for r in results) - max(
                r[key]["router_fork"] for r in results
            )
            print(
                f"[pool_direct bench] {label:16s} pool_direct - router_fork: slowest rank {slowest:+.3f} us; per rank "
                f"{' '.join(f'{d:+.3f}' for d in deltas)}"
            )
        print(f"[pool_direct bench] differing bits {[r['differing_bits'] for r in results]}")
        print(f"[pool_direct bench] clocks {[r['clocks'] for r in results]}")
        print("[pool_direct bench json] " + json.dumps({"place": place_name, "ranks": results}))
    _settle(e.comm)
    assert _agree(e.comm, bad == 0), f"rank {e.rank}: {bad} differing bits"

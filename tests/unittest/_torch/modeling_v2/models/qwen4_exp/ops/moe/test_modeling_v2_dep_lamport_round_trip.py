# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`dep_lamport_round_trip` on 4 and 8 ranks against NCCL's ring collectives, bit for bit.

One process a GPU, an NCCL group with `NCCL_ALGO=Ring` inside one node. For each cell's world
size, row count and slot protocol (device counter, or host index under `LamportSteps`), eagerly
and then replayed from one CUDA graph (the three slots rotating across replays):

* the gather of every rank's rows equals `all_gather_into_tensor` of the rows with -0.0 lanes as
  +0.0 (the transport's marker), and the entry's reference;
* the reduce-scatter of every rank's partials equals the reference bit for bit (the sum order
  the test fixes: `nccl_ring_order`, bf16 after every add), and `reduce_scatter_tensor` of the
  same sanitized partials. `NCCL_ALGO` picks the ring but not the order its ranks are visited in,
  which NCCL derives from the topology: a probe at the start compares NCCL with the reference,
  and where NCCL's ring visits the ranks in rank order (the one-node trays the op is certified
  on) the NCCL comparison is bitwise; elsewhere it allows the rounding of a reordered bf16 sum and
  the run says so.

Every other eager gather is the one that releases its PDL dependents at its start. The
host-index cells run each round inside a region with a varying number of launches (one to
four round trips), so the slot base moves by every residue mod 3, and also one launch outside any
region (its own advance). The 8-rank cells run on one node of eight SM103 GPUs. Their workspace,
and once more the 4-rank host-index one (so a 4-GPU tray runs it too), is built the way the NVFP4
target's dep8 builds it over `PeerBuffers`: every rank's mapped buffer as a raw device pointer,
wrapped by `buffers_from_pointers`, then `build_on` (here the pointers are torch symmetric
memory's; the target's are MNNVL memory's).

The inputs carry -0.0 lanes on purpose. Past `MAX_ROWS` rows a rank `takes` refuses the call
(the caller keeps NCCL). `QWEN4_EXP_LAMPORT_REPLAYS` raises the graph replays (default 500 a cell)
for a longer stress run.

The row count also changes from one launch to the next, shrinking and growing (`SHRINK_ROWS`:
2, 1, 2, ... rows at hidden 2560, where a one-row launch re-arms 640 vectors and publishes 320),
in both slot protocols: eagerly, then replayed back to back from one CUDA graph with a rotating
rank held back before each launch, against the reference, under a bounded wait
(`QWEN4_EXP_LAMPORT_TIMEOUT`, default 900 s) so a lost publication fails instead of hanging. A
CPU model of the slot protocol checks every interleaving of two ranks over such a sequence: with
the kernel's re-arm slot (`_CLEAR_AHEAD`) no publication is lost, and with the next launch's slot
one is.
"""

import functools
import os
import socket
import time

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import dep_lamport_round_trip as op

HIDDEN = 2560
REPLAYS = int(os.environ.get("QWEN4_EXP_LAMPORT_REPLAYS", "500"))
TIMEOUT = float(os.environ.get("QWEN4_EXP_LAMPORT_TIMEOUT", "900"))
#: Rows a launch of the shrinking / growing sequence (each a gather and a reduce-scatter).
SHRINK_ROWS = (2, 1, 2, 1, 1, 3, 1, 2, 12, 1)
CELLS = op.dep_lamport_round_trip.CELLS
GROUPS = sorted({(c.spec["world"], c.spec["step"]) for c in CELLS})

pytestmark = [pytest.mark.no_xdist]


def _skip(world: int):
    return pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.device_count() < world
        or torch.cuda.get_device_capability() != (10, 3),
        reason=f"the Lamport round trip runs on {world} SM103 GPUs of one NVLink domain here",
    )


def _inputs(gen, rows: int, device) -> torch.Tensor:
    """bf16 `[rows, HIDDEN]` with about 1% -0.0 lanes."""
    x = torch.randn([rows, HIDDEN], generator=gen, device=device).to(torch.bfloat16)
    zero = torch.rand([rows, HIDDEN], generator=gen, device=device) < 0.01
    return torch.where(zero, torch.full_like(x, -0.0), x)


def _check(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
    same = got.view(torch.int16) == want.view(torch.int16)
    assert bool(same.all()), f"{name}: {int((~same).sum())} of {same.numel()} lanes differ"


def _ring_in_rank_order(gen, rank: int, world: int, device) -> bool:
    """Whether NCCL's ring reduce-scatter sums in `nccl_ring_order` on this host, on every rank
    (a probe of random partials against the reference)."""
    import torch.distributed as dist

    y = _inputs(gen, world * 64, device)
    _, nccl = _nccl(y[:64], y, world)
    ys = [torch.empty_like(y) for _ in range(world)]
    dist.all_gather(ys, y)
    want = op.dep_lamport_round_trip.reference(ys, rank, True)
    same = torch.tensor([int(torch.equal(nccl.view(torch.int16), want.view(torch.int16)))])
    same = same.to(device)
    dist.all_reduce(same, op=dist.ReduceOp.MIN)
    ring = bool(same.item())
    if not ring and rank == 0:
        print(
            "NCCL's ring does not sum in rank order here: the reduce-scatter vs NCCL allows rounding"
        )
    return ring


def _check_rs(name: str, got: torch.Tensor, nccl: torch.Tensor, ring: bool) -> None:
    """The reduce-scatter against NCCL: bitwise where NCCL sums in rank order, else within the
    rounding of a reordered four-term bf16 sum of the test's inputs."""
    if ring:
        _check(name, got, nccl)
    else:
        torch.testing.assert_close(got.float(), nccl.float(), rtol=0, atol=0.0625, msg=name)


def _nccl(x, y, world: int):
    import torch.distributed as dist

    rows = x.shape[0]
    gathered = torch.empty([world * rows, HIDDEN], dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(gathered, op._sanitized(x))
    reduced = torch.empty([rows, HIDDEN], dtype=y.dtype, device=y.device)
    dist.reduce_scatter_tensor(reduced, op._sanitized(y))
    return gathered, reduced


def _build(rank: int, world: int, step: bool, via_pointers: bool):
    """`build` (torch symmetric memory), or with `via_pointers` the dep8 target's entry: every
    rank's mapped buffer as a device pointer through `buffers_from_pointers`, then `build_on`."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    steps = op.LamportSteps() if step else None
    if not via_pointers:
        return op.build(HIDDEN, op.MAX_ROWS, rank, world, steps=steps), steps
    device = torch.device("cuda", rank)
    shape = op.workspace_shape(HIDDEN, op.MAX_ROWS, world)
    buffer = symm_mem.empty(shape, dtype=torch.bfloat16, device=device)
    handle = symm_mem.rendezvous(buffer, group=dist.group.WORLD.group_name)
    mapped = [handle.get_buffer(p, buffer.shape, buffer.dtype) for p in range(world)]
    buffers = op.buffers_from_pointers(
        [b.data_ptr() for b in mapped], HIDDEN, op.MAX_ROWS, world, rank
    )
    assert [b.data_ptr() for b in buffers] == [b.data_ptr() for b in mapped]
    trip = op.build_on(
        HIDDEN, op.MAX_ROWS, rank, world, buffers, steps=steps, keepalive=(handle, buffer, mapped)
    )
    torch.cuda.synchronize(device)
    dist.barrier()
    return trip, steps


def _rank_main(rank: int, port: int, world: int, step: bool, via_pointers: bool) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), NCCL_ALGO="Ring")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        device = torch.device("cuda", rank)
        trip, steps = _build(rank, world, step, via_pointers)
        assert (trip.steps is not None) == step
        probe = torch.empty([1, HIDDEN], dtype=torch.bfloat16, device=device)
        assert trip.takes(op.MAX_ROWS, probe) and not trip.takes(op.MAX_ROWS + 1, probe)
        gen = torch.Generator(device=device).manual_seed(20261125 + rank)
        ring = _ring_in_rank_order(gen, rank, world, device)
        rows_list = sorted(
            c.spec["rows"] for c in CELLS if (c.spec["world"], c.spec["step"]) == (world, step)
        )
        for rows in rows_list:
            # Eager, 1..4 round trips a region (host index) or one at a time.
            for trips in range(1, 5 if step else 2):
                with steps.region() if step else _nothing():
                    rounds = []
                    for _ in range(trips):
                        x, y = _inputs(gen, rows, device), _inputs(gen, world * rows, device)
                        # Every other gather is the releasing one (same rows and bytes).
                        release = bool((len(rounds) + rows) % 2)
                        gathered = op.dep_lamport_round_trip(x, trip, False, release)
                        reduced = op.dep_lamport_round_trip(y, trip, True)
                        rounds.append((x, y, gathered, reduced))
                for k, (x, y, gathered, reduced) in enumerate(rounds):
                    nccl_gathered, nccl_reduced = _nccl(x, y, world)
                    tag = f"{world} ranks, {rows} rows, {trips} trips, trip {k}"
                    _check(f"gather vs NCCL ({tag})", gathered, nccl_gathered)
                    _check_rs(f"reduce-scatter vs NCCL ({tag})", reduced, nccl_reduced, ring)
                    xs = [torch.empty_like(x) for _ in range(world)]
                    ys = [torch.empty_like(y) for _ in range(world)]
                    dist.all_gather(xs, x)
                    dist.all_gather(ys, y)
                    ref = op.dep_lamport_round_trip.reference
                    _check(f"gather vs reference ({tag})", gathered, ref(xs, rank, False))
                    _check(f"reduce-scatter vs reference ({tag})", reduced, ref(ys, rank, True))
            if step:
                # A launch outside any region takes index 0 and its own advance.
                x, y = _inputs(gen, rows, device), _inputs(gen, world * rows, device)
                gathered = op.dep_lamport_round_trip(x, trip, False)
                reduced = op.dep_lamport_round_trip(y, trip, True)
                nccl_gathered, nccl_reduced = _nccl(x, y, world)
                _check(f"gather outside a region ({rows} rows)", gathered, nccl_gathered)
                _check_rs(
                    f"reduce-scatter outside a region ({rows} rows)", reduced, nccl_reduced, ring
                )

            # Replayed from one graph: fresh inputs copied into its static buffers.
            sx = torch.zeros([rows, HIDDEN], dtype=torch.bfloat16, device=device)
            sy = torch.zeros([world * rows, HIDDEN], dtype=torch.bfloat16, device=device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    with steps.region() if step else _nothing():
                        g_out = op.dep_lamport_round_trip(sx, trip, False)
                        r_out = op.dep_lamport_round_trip(sy, trip, True)
            torch.cuda.current_stream().wait_stream(stream)
            for i in range(REPLAYS):
                x, y = _inputs(gen, rows, device), _inputs(gen, world * rows, device)
                sx.copy_(x)
                sy.copy_(y)
                graph.replay()
                nccl_gathered, nccl_reduced = _nccl(x, y, world)
                _check(f"replay {i}: gather {rows} rows", g_out, nccl_gathered)
                _check_rs(f"replay {i}: reduce-scatter {rows} rows", r_out, nccl_reduced, ring)
                ys = [torch.empty_like(y) for _ in range(world)]
                dist.all_gather(ys, y)
                _check(
                    f"replay {i}: reduce-scatter {rows} rows vs reference",
                    r_out,
                    op.dep_lamport_round_trip.reference(ys, rank, True),
                )
        torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


class _nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


#: `(world, step, via_pointers)`: every cell group; the 8-rank ones and one more 4-rank host-index
#: run over `buffers_from_pointers` + `build_on` (the dep8 target's entry).
_BUILDS = [(w, s, w == 8) for w, s in GROUPS] + [(4, True, True)]


@pytest.mark.parametrize(
    "world, step, via_pointers",
    [
        pytest.param(
            w,
            s,
            p,
            marks=_skip(w),
            id=f"w{w}-{'step' if s else 'counter'}{'-pointers' if p and w == 4 else ''}",
        )
        for w, s, p in _BUILDS
    ],
)
def test_round_trip_matches_nccl_ring(world, step, via_pointers):
    import torch.multiprocessing as mp

    mp.spawn(_rank_main, args=(_free_port(), world, step, via_pointers), nprocs=world, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device pointer")
def test_buffers_from_pointers_are_bf16_views_of_the_pointed_memory():
    """`buffers_from_pointers` hands `build_on` bf16 `workspace_shape` tensors over the very
    bytes it was pointed at, on any one GPU (the 4- and 8-rank cells above need a tray)."""
    world, rows = 2, 3
    shape = op.workspace_shape(HIDDEN, rows, world)
    owned = [torch.randn(shape, device="cuda").to(torch.bfloat16) for _ in range(world)]
    views = op.buffers_from_pointers(
        [b.data_ptr() for b in owned], HIDDEN, rows, world, torch.cuda.current_device()
    )
    for view, buf in zip(views, owned):
        assert view.dtype is torch.bfloat16 and tuple(view.shape) == shape
        assert view.data_ptr() == buf.data_ptr()
        assert torch.equal(view.view(torch.int16), buf.view(torch.int16))


def _shrink_main(rank: int, port: int, world: int, step: bool) -> None:
    """`SHRINK_ROWS` launches eagerly and then back to back from one graph, every result against
    the reference; mismatches of the replays are counted on the device."""
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), NCCL_ALGO="Ring")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        device = torch.device("cuda", rank)
        trip, steps = _build(rank, world, step, via_pointers=False)
        gen = torch.Generator(device=device).manual_seed(20261127 + rank)
        ref = op.dep_lamport_round_trip.reference

        def operands():
            return [(_inputs(gen, r, device), _inputs(gen, world * r, device)) for r in SHRINK_ROWS]

        def wanted(ops):
            out = []
            for x, y in ops:
                xs = [torch.empty_like(x) for _ in range(world)]
                ys = [torch.empty_like(y) for _ in range(world)]
                dist.all_gather(xs, x)
                dist.all_gather(ys, y)
                out.append((ref(xs, rank, False), ref(ys, rank, True)))
            return out

        # Eager: one region over the whole sequence (host index), or one launch at a time.
        for _ in range(3):
            ops = operands()
            with steps.region() if step else _nothing():
                got = [
                    (
                        op.dep_lamport_round_trip(x, trip, False),
                        op.dep_lamport_round_trip(y, trip, True),
                    )
                    for x, y in ops
                ]
            for k, ((g, r), (wg, wr)) in enumerate(zip(got, wanted(ops))):
                _check(f"eager {SHRINK_ROWS[k]} rows (launch {k}): gather", g, wg)
                _check(f"eager {SHRINK_ROWS[k]} rows (launch {k}): reduce-scatter", r, wr)

        # Back to back from one graph: launch i sleeps first on rank i mod world, so one rank
        # runs ahead into the next launch while another is still in the one before.
        static = operands()
        want = wanted(static)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            with steps.region() if step else _nothing():
                outs = []
                for i, (x, y) in enumerate(static):
                    if i % world == rank:
                        torch.cuda._sleep(20000)  # about 10 us
                    outs.append(
                        (
                            op.dep_lamport_round_trip(x, trip, False),
                            op.dep_lamport_round_trip(y, trip, True),
                        )
                    )
        torch.cuda.current_stream().wait_stream(stream)
        bad = torch.zeros((), dtype=torch.int64, device=device)
        for _ in range(REPLAYS):
            graph.replay()
            for (g, r), (wg, wr) in zip(outs, want):
                bad += (g.view(torch.int16) != wg.view(torch.int16)).sum()
                bad += (r.view(torch.int16) != wr.view(torch.int16)).sum()
        torch.cuda.synchronize(device)
        assert int(bad.item()) == 0, f"{int(bad.item())} lanes differ over {REPLAYS} replays"
    finally:
        dist.destroy_process_group()


def _spawn_bounded(fn, args, nprocs: int, timeout: float) -> None:
    """`mp.spawn` that fails, and stops the ranks, when they do not finish within `timeout`
    seconds (a lost publication leaves a rank polling forever)."""
    import torch.multiprocessing as mp

    context = mp.spawn(fn, args=args, nprocs=nprocs, join=False)
    deadline = time.monotonic() + timeout
    while not context.join(timeout=5):
        if time.monotonic() > deadline:
            for p in context.processes:
                if p.is_alive():
                    p.kill()
            pytest.fail(f"the ranks did not finish within {timeout:.0f} s (a rank is polling)")


@pytest.mark.parametrize(
    "step",
    [pytest.param(s, marks=_skip(4), id="step" if s else "counter") for s in (False, True)],
)
def test_round_trip_rows_shrink_and_grow(step):
    _spawn_bounded(_shrink_main, (_free_port(), 4, step), 4, TIMEOUT)


# ---- the slot protocol on the host: every interleaving of two ranks ----


def _protocol_races(vectors: tuple, clear_ahead: int, thread_order: bool = False):
    """An exhaustive model of the device-counter protocol between two ranks over launches of
    `vectors[L]` vectors each: the first schedule that loses a publication or overwrites an unread
    one, else None.

    A launch L of a rank re-arms `counts[(L + clear_ahead) mod 3]` vectors of that slot of its
    own buffer, publishes its vectors into the peer's slot `L mod 3`, receives the peer's from its
    own, and at its end records its vector count for its slot. Nothing orders the stores of one
    launch among themselves (relaxed stores to different addresses), or with `thread_order` the
    thread of vector v issues its re-arm of v, its publication of v and its receipt of v in that
    order and is otherwise unordered (a thread past the launch's vectors only re-arms). A rank
    starts launch L + 1 once its launch L has completed. A cell holds None (the marker) or the
    launch whose vector it carries."""
    order = {"clear": 0, "publish": 1, "receive": 2}
    n = len(vectors)

    def events(rank, launch, counts):
        slot = launch % 3
        clear = (slot + clear_ahead) % 3
        evs = [("clear", clear, v) for v in range(counts[clear])]
        evs += [("publish", slot, v) for v in range(vectors[launch])]
        evs += [("receive", slot, v) for v in range(vectors[launch])]
        return frozenset(evs)

    init_counts = (0, 0, 0)
    empty = tuple(tuple(None for _ in range(max(vectors))) for _ in range(3))
    # state: per rank (launch, pending events, counts) and per rank its buffer (the peer's region)
    start = tuple((0, events(r, 0, init_counts), init_counts) for r in range(2))

    @functools.lru_cache(maxsize=None)
    def explore(ranks, bufs):
        if all(launch == n for launch, _, _ in ranks):
            return None
        moved = False
        for r in range(2):
            launch, pending, counts = ranks[r]
            if launch == n:
                continue
            peer = 1 - r
            if not pending:  # the launch completes: record its count, start the next
                slot = launch % 3
                counts2 = tuple(vectors[launch] if s == slot else c for s, c in enumerate(counts))
                nxt = launch + 1
                pend = events(r, nxt, counts2) if nxt < n else frozenset()
                ranks2 = tuple((nxt, pend, counts2) if i == r else ranks[i] for i in range(2))
                found = explore(ranks2, bufs)
                if found is not None:
                    return found
                moved = True
                continue
            for ev in pending:
                kind, slot, v = ev
                if thread_order and any(
                    e[2] == v and order[e[0]] < order[kind] for e in pending if e != ev
                ):
                    continue  # this thread has not reached it
                b = [list(map(list, bufs[0])), list(map(list, bufs[1]))]
                if kind == "clear":
                    held = b[r][slot][v]
                    if held is not None and held >= launch:
                        return (
                            f"rank {r} launch {launch} re-armed slot {slot} vector {v} over "
                            f"launch {held}'s publication"
                        )
                    b[r][slot][v] = None
                elif kind == "publish":
                    held = b[peer][slot][v]
                    if held is not None and held >= ranks[peer][0]:
                        return (
                            f"rank {r} launch {launch} published over launch {held}'s vector, "
                            "which the peer has not received"
                        )
                    b[peer][slot][v] = launch  # over a received vector not yet re-armed: allowed
                else:
                    held = b[r][slot][v]
                    if held is None:
                        continue  # still polling
                    if held != launch:
                        return f"rank {r} launch {launch} received launch {held}'s vector"
                    # the device-counter form leaves the vector for the re-arm pass
                rest = pending - {ev}
                ranks2 = tuple((launch, rest, counts) if i == r else ranks[i] for i in range(2))
                bufs2 = tuple(tuple(map(tuple, b[i])) for i in range(2))
                found = explore(ranks2, bufs2)
                if found is not None:
                    return found
                moved = True
        if not moved:
            return f"deadlock: {[(launch, sorted(p)) for launch, p, _ in ranks]}"
        return None

    return explore(start, (empty, empty))


@pytest.mark.cpu_only
def test_slot_protocol_never_rearms_a_slot_a_peer_can_publish_into():
    """Rows shrinking and growing (as vectors: two, one, two, ...): with the kernel's re-arm slot
    (the previous launch's) every interleaving completes, whether or not a thread's own stores
    are ordered. With the next launch's slot one loses a publication: even with each thread's
    stores in program order, a clear-only thread of a shrunk launch can erase a publication of a
    peer already in the next launch; at a constant row count that needs the stores reordered."""
    for vectors in ((2, 2, 1, 2), (2, 1, 2, 2), (2, 2, 1, 2, 1), (1, 2, 1, 2, 2), (1, 1, 1, 1, 1)):
        for thread_order in (False, True):
            assert _protocol_races(vectors, op._CLEAR_AHEAD, thread_order) is None, vectors
    lost = _protocol_races((2, 2, 1, 2), 1, thread_order=True)
    assert lost is not None and "re-armed slot 0 vector 1 over launch 3" in lost, lost
    assert _protocol_races((2, 2, 2, 2), 1, thread_order=True) is None
    assert _protocol_races((2, 2, 2, 2), 1, thread_order=False) is not None

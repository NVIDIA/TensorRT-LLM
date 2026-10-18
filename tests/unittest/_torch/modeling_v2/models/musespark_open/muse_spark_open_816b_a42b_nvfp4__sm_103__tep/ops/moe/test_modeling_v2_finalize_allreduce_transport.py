# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open finalize_allreduce_transport entry (TP8 over two GB300 nodes).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, e.g.
``srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>``;
anywhere else every test skips.

The entry claims the built-in kernel's bits: a built-in fabric workspace and a
``FinalizeTransportWorkspace`` per transport variant get the same inputs, and
their outputs must be equal as raw bits on every rank. Covered: every cell
(tokens 1, 8, 16; every slot of a token on one rank; a ring-buddy placement;
exact zeros, negative zeros and BF16 subnormals whose partials round to ``-0``;
a cancelling cell with invalid slots), every token count 1..16 (and 17, 32, 128,
which go to the built-in kernel), a delayed rank, CUDA-graph replays with
changing inputs and token counts and eager calls between them, a burst of 200
calls, and the arrival counter across its 32-bit wrap. The cells are also
gated against the entry's float64 reference.

Two properties the outputs alone do not show under sequential calls are
tested directly: that the kernel reads its producer's outputs only after its
dependency wait (behind a producer that triggers its dependents first and
writes the inputs afterwards), and that consecutive calls alternate the two
receive buffers as the arrival counter's parity says (white-box, through the
own slot of the multicast publication, which the protocol never reads).

``MODELING_V2_TIMING=1`` also runs the microbenchmarks (printed, never gated):
the floor-bench chain (64 serial calls per graph) for every variant, a
62-layer chain with the real neighbours -- a GEMM standing for the expert down
projection before each finalize and the unpool slice GEMM (the ``dense_gemms`` split-K
kernel, programmatic dependent launch) after it -- and the multicast and the
unicast publication timed on the same allocations.
"""

import os
import statistics

import pytest
import torch

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the image has Triton
    triton = None

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
    finalize_allreduce_transport as fin,
)

__extra_import_path__ = ["../../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

WORLD = 8
HIDDEN = 4096
TOP_K = 8
EPS = 1e-8
op = fin.finalize_allreduce_transport

#: Every transport the tests certify: the two installable ones and the attribution variants.
VARIANTS = {
    "finalize_transport": fin.DEFAULT,
    "finalize_transport_entry": fin.ENTRY_TRIGGER,
    "uc_atom_pull": fin.Transport(publish="unicast", arrival="atom", reduction="pull"),
    "mc_atom_pull": fin.Transport(arrival="atom", reduction="pull"),
    "mc_red_pull": fin.Transport(reduction="pull"),
    "mc_red_push": fin.Transport(reduction="push"),
    "mc_red_push_e": fin.Transport(reduction="push", trigger="entry"),
    "uc_red_warp": fin.Transport(publish="unicast"),
    "mc_atom_warp": fin.Transport(arrival="atom"),
}


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs eight MPI ranks with CUDA (two four-GPU GB300 nodes)")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("needs GB300 (sm_103)")
    return comm, rank


def _agree(comm, ok: bool) -> bool:
    return all(comm.allgather(bool(ok)))


@pytest.fixture(scope="module")
def group():
    """A built-in fabric workspace and one transport workspace per variant, for the module."""
    comm, rank = _world()
    from tensorrt_llm._torch.models.musespark_open_finalize_ar import FinalizeIpcWorkspace
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    builtin = FinalizeIpcWorkspace(mapping, use_mnnvl=True)
    workspaces = {
        name: fin.FinalizeTransportWorkspace(builtin, mapping, transport)
        for name, transport in VARIANTS.items()
    }
    comm.Barrier()
    return comm, rank, builtin, workspaces


def _cancelling_row(rank: int) -> torch.Tensor:
    """A row whose partials, across the ranks, sum differently in different orders.

    Element ``j`` of every 8-element vector gets role ``(rank + j) % 8`` on
    ``rank``: +1, -1, ``2**-26`` (absorbed if added while +1 or -1 alone is in
    the FP32 accumulator) or noise of order ``2**-40``. Every rank's vectors hold
    the same values, so every rank's sum of squares -- and so its row scale --
    is the same in any summation order, and the +1 and -1 partials cancel.
    """
    roles = (rank + torch.arange(HIDDEN) % 8) % 8
    gen = torch.Generator().manual_seed(4242)
    noise = 2.0**-40 * (1 + torch.rand(HIDDEN, generator=gen))
    row = torch.where(roles == 0, 1.0, torch.where(roles == 1, -1.0, noise))
    row = torch.where(roles == 2, 2.0**-26, row)
    return row.to(torch.bfloat16)


def _owners(tokens: int, seed: int, placement: str) -> torch.Tensor:
    """The rank that holds slot ``(t, k)``, the same on every rank."""
    slots = torch.arange(TOP_K).expand(tokens, TOP_K)
    if placement == "modulo":
        return slots % WORLD
    if placement == "concentrated":
        return torch.arange(tokens)[:, None].expand(tokens, TOP_K) % WORLD
    gen = torch.Generator().manual_seed(10_000 + seed)
    experts = torch.stack([torch.randperm(256, generator=gen)[:TOP_K] for _ in range(tokens)])
    owner = experts // 32
    if placement == "buddy":
        moved = torch.rand(tokens, TOP_K, generator=gen) < 0.4
        owner = torch.where(moved, (owner + 1) % WORLD, owner)
    return owner


def _inputs(
    tokens: int,
    seed: int,
    rank: int,
    placement: str = "modulo",
    zeros: bool = False,
    invalid: bool = False,
    cancel: bool = False,
    permuted: bool = False,
):
    """One rank's expert rows, row map and routing weights; every rank can rebuild every rank's.

    A slot ``(t, k)`` held by this rank addresses row ``8 t + k`` (the identity
    row map of a persistent expert kernel) or, with ``permuted``, a shuffled
    local row as a permuting expert op emits; other slots are ``-1``.
    """
    gen = torch.Generator().manual_seed(seed * WORLD + rank)
    rows = max(64, tokens * TOP_K)
    gemm = torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16)
    owner = _owners(tokens, seed, placement)
    index = torch.arange(tokens * TOP_K, dtype=torch.int32).reshape(tokens, TOP_K)
    if permuted:
        index = torch.randperm(rows, generator=gen)[: tokens * TOP_K].to(torch.int32)
        index = index.reshape(tokens, TOP_K)
    index = torch.where(owner == rank, index, -1)
    shared = torch.Generator().manual_seed(seed)
    weights = torch.rand(tokens, TOP_K, generator=shared)
    weights /= weights.sum(-1, keepdim=True)
    norm = torch.randn(HIDDEN, generator=shared).to(torch.bfloat16) * 0.03
    if zeros:
        gemm[:, 0:64] = 0.0
        gemm[:, 64:128] = -0.0
        # Negative BF16 subnormals: normalized and weighted they round to -0 in the
        # BF16 partial, which the publication must turn into +0.
        tiny = torch.tensor(-(2.0**-133)).to(torch.bfloat16)
        gemm[:, 128:256:3] = tiny
        if tokens > 1:
            index[1] = -1  # a token with no valid slot on any rank
    if invalid:
        index[0, rank % TOP_K] = 10_000  # past the rows: an invalid slot
    if cancel:
        gemm = _cancelling_row(rank).expand(rows, HIDDEN).contiguous()
        weights = torch.full((tokens, TOP_K), 1.0 / TOP_K)
        norm = torch.zeros(HIDDEN, dtype=torch.bfloat16)
    return gemm.cuda(), index.contiguous().cuda(), weights.cuda(), norm.cuda()


def _sum_bf16(partials, order) -> torch.Tensor:
    """The FP32 sum from +0 over ``order``, rounded once to BF16 (on the CPU)."""
    acc = torch.zeros(partials[0].shape, dtype=torch.float32)
    for r in order:
        acc = acc + partials[r].cpu().float()
    return acc.to(torch.bfloat16)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _differing(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((_bits(a) != _bits(b)).sum().item())


def test_every_variant_built(group):
    comm, rank, builtin, workspaces = group
    ok = all(ws.legacy is builtin for ws in workspaces.values())
    assert _agree(comm, ok), f"rank {rank}: a variant is not on the built-in workspace"


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("cell", op.CELLS, ids=lambda c: c.why[:40].replace(" ", "_"))
@torch.inference_mode()
def test_cell_is_the_builtin_kernel_bit_for_bit(group, cell, variant, certifying):  # noqa: F811
    comm, rank, builtin, workspaces = group
    certifying(op)
    spec = dict(cell.spec)
    tokens = spec.pop("tokens")
    ws = workspaces[variant]
    gemm, index, weights, norm = _inputs(tokens, 100 + tokens, rank, **spec)
    got = op(gemm, index, weights, norm, EPS, tokens, ws)
    want = builtin.finalize(gemm, index, weights, norm, EPS, tokens)
    torch.cuda.synchronize()
    mismatch = _differing(got, want)
    parts = [_inputs(tokens, 100 + tokens, r, **spec) for r in range(WORLD)]
    order_observable = True
    if spec.get("cancel"):
        partials = [op.reference([g], [i], [w], norm, EPS, tokens, None) for g, i, w, _ in parts]
        forward = _sum_bf16(partials, range(WORLD))
        backward = _sum_bf16(partials, reversed(range(WORLD)))
        order_observable = _differing(forward, backward) > 1000
    identical = _agree(comm, mismatch == 0)
    observable = _agree(comm, order_observable)
    assert identical, f"rank {rank} {variant}: {mismatch} values differ from the built-in kernel"
    assert observable, "the cancelling cell does not make the summation order observable"
    ref = op.reference(
        [p[0] for p in parts],
        [p[1] for p in parts],
        [p[2] for p in parts],
        norm,
        EPS,
        tokens,
        None,
    )
    if not spec.get("cancel"):
        # Two BF16 roundings (the partial, the sum) and FP32 sums against float64.
        torch.testing.assert_close(got.float(), ref.float(), rtol=2**-6, atol=2**-10)
    if spec.get("zeros"):
        assert _agree(comm, bool((got[1] == 0).all()) and not bool(torch.signbit(got[1]).any()))


@pytest.mark.parametrize(
    "variant", ["finalize_transport", "finalize_transport_entry", "uc_atom_pull", "mc_red_push"]
)
@torch.inference_mode()
def test_every_token_count(group, variant):
    comm, rank, builtin, workspaces = group
    ws = workspaces[variant]
    bad = 0
    for tokens in list(range(1, 17)) + [17, 32, 128]:
        for permuted in (False, True):
            gemm, index, weights, norm = _inputs(
                tokens, 200 + tokens, rank, placement="buddy", permuted=permuted
            )
            got = ws.finalize(gemm, index, weights, norm, EPS, tokens)
            want = builtin.finalize(gemm, index, weights, norm, EPS, tokens)
            torch.cuda.synchronize()
            bad += _differing(got, want)
    assert _agree(comm, bad == 0), f"rank {rank} {variant}: {bad} values differ"


@pytest.mark.parametrize(
    "variant", ["finalize_transport", "finalize_transport_entry", "mc_red_push_e"]
)
@torch.inference_mode()
def test_a_delayed_rank_is_waited_for(group, variant):
    comm, rank, builtin, workspaces = group
    ws = workspaces[variant]
    bad = 0
    for late in range(WORLD):
        gemm, index, weights, norm = _inputs(8, 300 + late, rank, placement="buddy")
        want = builtin.finalize(gemm, index, weights, norm, EPS, 8)
        torch.cuda.synchronize()
        comm.Barrier()
        if rank == late:
            torch.cuda._sleep(2_000_000)  # ~1 ms
        got = ws.finalize(gemm, index, weights, norm, EPS, 8)
        torch.cuda.synchronize()
        bad += _differing(got, want)
    assert _agree(comm, bad == 0), f"rank {rank} {variant}: {bad} values differ"


@pytest.mark.parametrize(
    "variant", ["finalize_transport", "finalize_transport_entry", "mc_atom_warp", "mc_red_push"]
)
@torch.inference_mode()
def test_graph_replay_with_changing_inputs_and_token_counts(group, variant):
    comm, rank, builtin, workspaces = group
    ws = workspaces[variant]
    shapes = (8, 8, 1, 4, 16, 17, 8, 3)
    static = [_inputs(t, 400 + i, rank, placement="buddy") for i, t in enumerate(shapes)]

    def run(w):
        return [w.finalize(g, i, wt, n, EPS, t) for (g, i, wt, n), t in zip(static, shapes)]

    run(ws)
    run(builtin)
    torch.cuda.synchronize()
    comm.Barrier()
    graphs, outs = {}, {}
    for name, w in (("ws", ws), ("builtin", builtin)):
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            outs[name] = run(w)
    bad = 0
    for replay in range(30):
        for i, t in enumerate(shapes):
            fresh = _inputs(t, 500 + 100 * replay + i, rank, placement="buddy")
            for dst, src in zip(static[i], fresh):
                dst.copy_(src)
        graphs["ws"].replay()
        # Eager calls on the same workspace between replays, of other token counts.
        g, i, wt, n = _inputs(1 + replay % 16, 600 + replay, rank, placement="buddy")
        eager = ws.finalize(g, i, wt, n, EPS, 1 + replay % 16)
        eager_want = builtin.finalize(g, i, wt, n, EPS, 1 + replay % 16)
        graphs["builtin"].replay()
        torch.cuda.synchronize()
        bad += _differing(eager, eager_want)
        for got, want in zip(outs["ws"], outs["builtin"]):
            bad += _differing(got, want)
    assert _agree(comm, bad == 0), f"rank {rank} {variant}: {bad} values differ across replays"


@pytest.mark.parametrize("variant", ["finalize_transport_entry", "uc_red_warp", "mc_red_push_e"])
@torch.inference_mode()
def test_a_long_burst_of_mixed_token_counts_stays_identical(group, variant):
    comm, rank, builtin, workspaces = group
    ws = workspaces[variant]
    g = torch.Generator().manual_seed(77)
    pairs = []
    torch.cuda.synchronize()
    comm.Barrier()
    for call in range(200):
        tokens = int(torch.randint(1, 17, (1,), generator=g))
        if call % 50 == 49:
            tokens = 17
        gemm, index, weights, norm = _inputs(tokens, 700 + call, rank, placement="buddy")
        got = ws.finalize(gemm, index, weights, norm, EPS, tokens)
        want = builtin.finalize(gemm, index, weights, norm, EPS, tokens)
        pairs.append((got, want))
    torch.cuda.synchronize()
    bad = sum(_differing(got, want) for got, want in pairs)
    assert _agree(comm, bad == 0), f"rank {rank} {variant}: {bad} values differ over the burst"


@torch.inference_mode()
def test_the_arrival_counter_advances_by_one_step_per_call_across_its_wrap(group):
    comm, rank, builtin, workspaces = group
    ws = workspaces["finalize_transport"]
    torch.cuda.synchronize()
    comm.Barrier()
    start = (1 << 32) - 3 * fin.GRID_STEP  # three calls before the wrap; parity of the next: 1
    ws.control[2] = start - (1 << 32)
    torch.cuda.synchronize()
    comm.Barrier()
    bad, counts = 0, []
    for call in range(8):
        tokens = (8, 1, 16, 3)[call % 4]
        gemm, index, weights, norm = _inputs(tokens, 800 + call, rank, placement="buddy")
        got = ws.finalize(gemm, index, weights, norm, EPS, tokens)
        want = builtin.finalize(gemm, index, weights, norm, EPS, tokens)
        torch.cuda.synchronize()
        bad += _differing(got, want)
        counts.append(int(ws.control[2].item()) & 0xFFFFFFFF)
    expected = [(start + fin.GRID_STEP * (c + 1)) & 0xFFFFFFFF for c in range(8)]
    assert _agree(comm, bad == 0 and counts == expected), (rank, bad, counts, expected)


# ---------------------------------------------------------------------------
# What the outputs of sequential calls do not show.
# ---------------------------------------------------------------------------

if triton is not None:

    @triton.jit
    def _trigger_then_copy(src, dst, words, SPINS: tl.constexpr, BLOCK: tl.constexpr):
        """A PDL producer that releases its dependents first and writes their inputs ~30 us later."""
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
        x = tl.full((1,), 0, tl.int32)
        for _ in range(SPINS):
            x = tl.inline_asm_elementwise(
                "nanosleep.u32 1000; mov.u32 $0, $1;",
                "=r,r",
                [x],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < words
        tl.store(dst + offsets, tl.load(src + offsets, mask=mask, other=0) + x * 0, mask=mask)


class _Arena:
    """One call's expert rows, row map and routing weights in one int32 buffer (16-byte parts)."""

    def __init__(self, tokens: int):
        self.tokens = tokens
        self.rows = max(64, tokens * TOP_K)
        self.row_words = self.rows * HIDDEN // 2
        self.slots = tokens * TOP_K
        self.words = self.row_words + 2 * self.slots
        self.buf = torch.zeros(self.words, dtype=torch.int32, device="cuda")

    def views(self):
        gemm = self.buf[: self.row_words].view(torch.bfloat16).view(self.rows, HIDDEN)
        index = self.buf[self.row_words : self.row_words + self.slots].view(self.tokens, TOP_K)
        weights = self.buf[self.row_words + self.slots :].view(torch.float32)
        return gemm, index, weights.view(self.tokens, TOP_K)

    def load(self, gemm, index, weights):
        for dst, src in zip(self.views(), (gemm, index, weights)):
            dst.copy_(src)


@pytest.mark.skipif(triton is None, reason="needs Triton for the producer kernel")
@pytest.mark.parametrize("variant", ["finalize_transport", "finalize_transport_entry"])
@torch.inference_mode()
def test_inputs_are_read_only_after_the_dependency_wait(group, variant):
    """The finalize behind a producer that triggers its dependents before it writes their inputs.

    Each call's rows, row map and routing weights are filled with ``0x7FC07FC0``
    (BF16 / FP32 NaNs, an out-of-range row index), then a Triton kernel launched
    with programmatic dependent launch waits, triggers its dependents at once,
    sleeps ~30 us and only then copies the true inputs in; the finalize is
    launched behind it with programmatic dependent launch. An input read before
    the finalize's ``griddepcontrol.wait`` reads the fill. Eager, then a CUDA
    graph replayed with fresh inputs; a 17-token call (the built-in kernel)
    among them.
    """
    comm, rank, builtin, workspaces = group
    ws = workspaces[variant]
    if not ws.pdl:
        pytest.skip("programmatic dependent launch is off (TRTLLM_ENABLE_PDL=0)")
    shapes = (8, 1, 16, 8, 3, 17, 8)
    sources = [_Arena(t) for t in shapes]
    targets = [_Arena(t) for t in shapes]
    _, _, _, norm = _inputs(8, 1100, rank)

    def calls():
        outs = []
        for src, dst in zip(sources, targets):
            dst.buf.fill_(0x7FC07FC0)
            grid = (triton.cdiv(dst.words, 1024),)
            _trigger_then_copy[grid](
                src.buf, dst.buf, dst.words, SPINS=30, BLOCK=1024, launch_pdl=True
            )
            gemm, index, weights = dst.views()
            outs.append(ws.finalize(gemm, index, weights, norm, EPS, dst.tokens))
        return outs

    def refresh(round_):
        for n, src in enumerate(sources):
            gemm, index, weights, _ = _inputs(
                src.tokens, 1200 + 31 * round_ + n, rank, placement="buddy", permuted=n % 2 == 1
            )
            src.load(gemm, index, weights)
        torch.cuda.synchronize()
        want = [builtin.finalize(*src.views(), norm, EPS, src.tokens) for src in sources]
        torch.cuda.synchronize()
        return want

    want = refresh(0)
    comm.Barrier()
    got = calls()  # eager first: also loads the Triton module outside the capture
    torch.cuda.synchronize()
    bad = sum(_differing(g, w) for g, w in zip(got, want))
    comm.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = calls()
    for replay in range(1, 13):
        want = refresh(replay)
        comm.Barrier()
        graph.replay()
        torch.cuda.synchronize()
        bad += sum(_differing(g, w) for g, w in zip(outs, want))
    assert _agree(comm, bad == 0), (
        f"rank {rank} {variant}: {bad} values differ behind an early-triggering producer"
    )


@pytest.mark.parametrize("variant", ["finalize_transport", "finalize_transport_entry"])
@torch.inference_mode()
def test_consecutive_calls_alternate_the_buffers_by_the_counter_parity(group, variant):
    """White-box: which receive buffer each call publishes into.

    The multicast publication also writes this rank's partial into its own slot
    of its own receive area, which the protocol never reads or resets. Before
    each call both buffers' own slab gets a marker; after it exactly one of them
    must have changed, it must be the other one than the previous call's, and
    it must be ``(counter >> GRID_SHIFT) & 1`` read before the call. 17-token
    calls run the built-in kernel and must neither touch these buffers nor
    advance the parity. Every output is also checked against the built-in.
    """
    comm, rank, builtin, workspaces = group
    ws = workspaces[variant]
    assert ws.transport.publish == "multicast" and ws.transport.arrival == "red"
    marker = 0x7F7F7F7F
    area = ws._uc[rank]  # this rank's receive area, int32 words
    slab_words = fin.SLAB_BYTES // 4

    def own(buffer):
        start = (buffer * fin.RANKS + rank) * slab_words
        return area[start : start + slab_words]

    shapes = (8, 8, 1, 16, 3, 8, 17, 8, 12, 8, 8, 2, 16, 8, 17, 17, 8, 5)
    used, expected, problems, bad = [], [], [], 0
    torch.cuda.synchronize()
    comm.Barrier()
    for n, tokens in enumerate(shapes):
        own(0).fill_(marker)
        own(1).fill_(marker)
        counter = int(ws.control[2].item()) & 0xFFFFFFFF
        torch.cuda.synchronize()
        comm.Barrier()
        gemm, index, weights, norm = _inputs(tokens, 1300 + n, rank, placement="buddy")
        got = ws.finalize(gemm, index, weights, norm, EPS, tokens)
        want = builtin.finalize(gemm, index, weights, norm, EPS, tokens)
        torch.cuda.synchronize()
        bad += _differing(got, want)
        words = min(tokens, fin.MAX_TOKENS) * HIDDEN // 2
        changed = [bool((own(b)[:words] != marker).any().item()) for b in (0, 1)]
        after = int(ws.control[2].item()) & 0xFFFFFFFF
        if not ws.takes(tokens):
            if any(changed) or after != counter:
                problems.append((n, tokens, "built-in call touched the transport's state", changed))
            continue
        if changed.count(True) != 1:
            problems.append((n, tokens, "not exactly one buffer written", changed))
            continue
        used.append(changed.index(True))
        expected.append((counter >> fin.GRID_SHIFT) & 1)
    alternates = all(a != b for a, b in zip(used, used[1:]))
    ok = not problems and alternates and used == expected and bad == 0
    assert _agree(comm, ok), (
        f"rank {rank} {variant}: buffers {used}, parity {expected}, {problems}, {bad} values differ"
    )


# ---------------------------------------------------------------------------
# Microbenchmarks: MODELING_V2_TIMING=1 (printed, never gated).
# ---------------------------------------------------------------------------

_TIMING = os.environ.get("MODELING_V2_TIMING") == "1"
_PLACEMENT = os.environ.get("MODELING_V2_TIMING_PLACEMENT") == "1"
#: Workspaces the placement benchmark builds, kept until the process exits.
_KEPT = []
_CALLS, _REPLAYS, _TRIALS = 64, 40, 7
_LAYERS = 62


def _floor_inputs(tokens: int, local_per_token: int, rank: int):
    """The floor bench's finalize kind: ``local_per_token`` of 8 slots on each rank."""
    idx = torch.full((tokens * TOP_K,), -1, dtype=torch.int32)
    row = 0
    for t in range(tokens):
        for s in range(TOP_K):
            if (s + t + rank) % WORLD < local_per_token:
                idx[t * TOP_K + s] = row
                row += 1
    rows = max(row, 1)
    gen = torch.Generator().manual_seed(1234 + rank)
    g2 = (torch.randn(rows, HIDDEN, generator=gen) * 0.1).to(torch.bfloat16).cuda()
    wts = torch.rand(tokens * TOP_K, generator=gen, dtype=torch.float32).cuda()
    norm = (0.1 * torch.randn(HIDDEN, generator=gen)).to(torch.bfloat16).cuda()
    return g2, idx.cuda(), wts, norm


def _time_chain(comm, fn, calls):
    """The floor bench's method: ``calls`` serial calls in one graph; µs per call."""
    fn()
    torch.cuda.synchronize()
    comm.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(_TRIALS):
        comm.Barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(_REPLAYS):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / _REPLAYS / calls)
    del graph
    return statistics.median(samples), max(samples) - min(samples)


def _receive_address(ws) -> str:
    """The allocation a floor number belongs to: rank 0's receive area (built-in: its fabric slot)."""
    address = ws._uc[ws.rank].data_ptr() if hasattr(ws, "_uc") else ws._receive
    return f"0x{address:x}"


@pytest.mark.skipif(not _TIMING, reason="MODELING_V2_TIMING=1")
@torch.inference_mode()
def test_microbenchmark_floor_chain(group):
    """The floor bench's ``finalize_ar_<tokens>tok_<local>local`` kinds for the built-in kernel and every variant.

    Each arm: 64 serial calls in one CUDA graph, median of 7 trials x 40 replays;
    printed as the max and min over ranks with rank 0's receive address. The
    same kernel's floor differs between workspaces (between allocations of its
    receive area), so a floor number describes a kernel on
    one allocation; transports are compared on one allocation by
    ``test_microbenchmark_publication_on_one_allocation``, and the transport's
    gate is its own latency in the model.
    """
    comm, rank, builtin, workspaces = group
    arms = [("builtin", builtin)] + list(workspaces.items())
    for tokens, local in ((8, 2), (8, 4), (8, 0), (1, 2), (16, 2)):
        g2, idx, wts, norm = _floor_inputs(tokens, local, rank)
        # Alternate the arm order between kinds so no arm always runs first.
        order = arms if (tokens + local) % 2 == 0 else list(reversed(arms))
        for name, ws in order:
            med, spread = _time_chain(
                comm,
                lambda ws=ws: ws.finalize(g2, idx, wts, norm, 1e-5, tokens),
                _CALLS,
            )
            hi = comm.allreduce(med, op=MPI.MAX)
            lo = comm.allreduce(med, op=MPI.MIN)
            sp = comm.allreduce(spread, op=MPI.MAX)
            if rank == 0:
                print(
                    f"[finalize_transport floor] finalize_ar_{tokens}tok_{local}local {name:14s} "
                    f"max {hi:7.3f} min {lo:7.3f} (trial spread {sp:.3f}) "
                    f"receive {_receive_address(ws)}",
                    flush=True,
                )
    comm.Barrier()


def _sharing(ws, transport):
    """A workspace on ``ws``'s memory and protocol state that runs ``transport``'s kernel.

    The same receive area, peers, multicast address and arrival counter; only
    the kernel differs. Calls of the two interleave like calls of one workspace
    as long as both use the same arrival (the same counter and buffer parity).
    """
    assert transport.arrival == ws.transport.arrival
    twin = object.__new__(fin.FinalizeTransportWorkspace)
    twin.__dict__.update(ws.__dict__)
    twin.transport = transport
    twin._compiled = fin._compiled_kernel(ws.device.index, transport, ws.pdl)
    return twin


@pytest.mark.skipif(not _TIMING, reason="MODELING_V2_TIMING=1")
@torch.inference_mode()
def test_microbenchmark_publication_on_one_allocation(group):
    """Multicast vs unicast publication of the ``finalize_transport`` kernel on the same allocations.

    For the fixture's ``finalize_transport`` workspace and three more built here, a unicast
    twin on the same memory (``_sharing``). First the twins' bits: calls
    alternating between the two kernels on one allocation, against the
    built-in. Then per allocation the floor chain of both kernels (two rounds in
    opposite orders) and the 62-layer chain with the real neighbours; printed as
    the max over ranks (floor) and the median over ranks (chain).
    """
    from tensorrt_llm.mapping import Mapping

    comm, rank, builtin, workspaces = group
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    unicast = fin.Transport(publish="unicast")
    allocations = [("fixture", workspaces["finalize_transport"])]
    for copy in range(3):
        ws = fin.FinalizeTransportWorkspace(builtin, mapping, fin.DEFAULT)
        _KEPT.append(ws)
        allocations.append((f"new{copy}", ws))
    pairs = [(label, ws, _sharing(ws, unicast)) for label, ws in allocations]

    bad = 0
    for call in range(24):
        tokens = (8, 1, 16, 3, 8, 12)[call % 6]
        gemm, index, weights, norm = _inputs(tokens, 1400 + call, rank, placement="buddy")
        for _, mc, uc in pairs:
            got = (mc if call % 2 == 0 else uc).finalize(gemm, index, weights, norm, EPS, tokens)
            want = builtin.finalize(gemm, index, weights, norm, EPS, tokens)
            torch.cuda.synchronize()
            bad += _differing(got, want)
    assert _agree(comm, bad == 0), f"rank {rank}: {bad} values differ with alternating twins"

    for tokens, local in ((8, 2), (8, 0), (1, 2), (16, 2)):
        g2, idx, wts, norm = _floor_inputs(tokens, local, rank)
        for round_ in range(2):
            for label, mc, uc in pairs if round_ == 0 else list(reversed(pairs)):
                times = {}
                for name, ws in (("multicast", mc), ("unicast", uc))[:: 1 - 2 * round_]:
                    med, _ = _time_chain(
                        comm,
                        lambda ws=ws: ws.finalize(g2, idx, wts, norm, 1e-5, tokens),
                        _CALLS,
                    )
                    times[name] = comm.allreduce(med, op=MPI.MAX)
                if rank == 0:
                    print(
                        f"[finalize_transport publish] finalize_ar_{tokens}tok_{local}local round {round_} "
                        f"{label:8s} receive {_receive_address(mc)} multicast "
                        f"{times['multicast']:7.3f} unicast {times['unicast']:7.3f} "
                        f"unicast-multicast {times['unicast'] - times['multicast']:+.3f}",
                        flush=True,
                    )
    arms = []
    for label, mc, uc in pairs:
        arms += [(f"multicast#{label}", mc), (f"unicast#{label}", uc)]
    _neighbour_chains(comm, rank, arms, tag="finalize_transport publish chain")
    comm.Barrier()


@pytest.mark.skipif(not _TIMING, reason="MODELING_V2_TIMING=1")
@torch.inference_mode()
def test_microbenchmark_with_the_real_neighbours(group):
    """62 layers of: down GEMM -> finalize -> unpool slice GEMM (split-K, PDL), µs per layer.

    The down GEMM (``[64, 1024] x [1024, 4096]``, cuBLAS, launched without PDL)
    stands for the expert down projection and writes the finalize's rows; the
    unpool slice GEMM is ``dense_gemm``'s ``splitk:64,8,4,12`` on a cold
    ``[1024, 4096]`` weight per layer, the kernel ``dense_gemms`` runs at 8 rows, whose
    weight loads precede its ``griddepcontrol.wait``; its output feeds the next
    layer's down GEMM. Per arm the median of 16 alternating-order rounds of 10
    replays; printed as median, min and max over ranks.
    """
    comm, rank, builtin, workspaces = group
    arms = [("builtin", builtin)] + [
        (name, workspaces[name])
        for name in (
            "uc_atom_pull",
            "mc_red_push",
            "mc_red_push_e",
            "finalize_transport",
            "finalize_transport_entry",
        )
    ]
    _neighbour_chains(comm, rank, arms, tag="finalize_transport chain")
    comm.Barrier()


def _neighbour_chains(comm, rank, arms, tag):
    """The 62-layer chain of ``test_microbenchmark_with_the_real_neighbours`` for each arm."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.dense_gemm import (  # noqa: E501
        dense_gemm,
    )

    tokens = 8
    gemm, index, weights, norm = _inputs(tokens, 900, rank, placement="buddy")
    rows = gemm.shape[0]
    gen = torch.Generator().manual_seed(31 + rank)
    downs = [
        (torch.randn(HIDDEN, 1024, generator=gen) * 0.03).to(torch.bfloat16).cuda()
        for _ in range(_LAYERS)
    ]
    unpools = [
        (torch.randn(1024, HIDDEN, generator=gen) * 0.03).to(torch.bfloat16).cuda()
        for _ in range(_LAYERS)
    ]
    x0 = torch.randn(rows, 1024, generator=gen).to(torch.bfloat16).cuda()

    def chain(ws, with_unpool):
        x = x0
        out = None
        for layer in range(_LAYERS):
            g = torch.nn.functional.linear(x, downs[layer])
            out = ws.finalize(g, index, weights, norm, EPS, tokens)
            if with_unpool:
                y = dense_gemm(out, unpools[layer], "splitk:64,8,4,12", True)
                x = y.repeat(rows // tokens, 1)
            else:
                x = out[:, :1024].repeat(rows // tokens, 1)
        return out

    results = {}
    for with_unpool in (True, False):
        graphs = {}
        for name, ws in arms:
            chain(ws, with_unpool)
            torch.cuda.synchronize()
            comm.Barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                chain(ws, with_unpool)
            for _ in range(10):
                graph.replay()
            graphs[name] = graph
        torch.cuda.synchronize()
        samples = {name: [] for name in graphs}
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        names = list(graphs)
        for r in range(16):
            for name in names if r % 2 == 0 else list(reversed(names)):
                torch.cuda.synchronize()
                comm.Barrier()
                start.record()
                for _ in range(10):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000.0 / 10 / _LAYERS)
        for name in names:
            results[(with_unpool, name)] = statistics.median(samples[name])
        del graphs
    gathered = comm.gather(results, root=0)
    if rank == 0:
        for key in gathered[0]:
            values = sorted(r[key] for r in gathered)
            label = ("down + finalize + unpool " if key[0] else "down + finalize ") + key[1]
            print(
                f"[{tag}] {label:40s} median {statistics.median(values):8.3f} "
                f"min {values[0]:.3f} max {values[-1]:.3f} us per layer",
                flush=True,
            )


@pytest.mark.skipif(not _PLACEMENT, reason="MODELING_V2_TIMING_PLACEMENT=1")
@torch.inference_mode()
def test_microbenchmark_workspace_placement(group):
    """The floor chain of several workspaces of one transport: the spread between allocations.

    Four more workspaces each of ``finalize_transport`` and ``mc_red_push``, built interleaved,
    next to the fixture's; two rounds in opposite orders; printed as the max over
    ranks with this rank's receive address.
    """
    from tensorrt_llm.mapping import Mapping

    comm, rank, builtin, workspaces = group
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    arms = [(f"{name}#fixture", workspaces[name]) for name in ("finalize_transport", "mc_red_push")]
    for copy in range(4):
        for name in ("finalize_transport", "mc_red_push"):
            ws = fin.FinalizeTransportWorkspace(builtin, mapping, VARIANTS[name])
            _KEPT.append(ws)
            arms.append((f"{name}#{copy}", ws))
    for tokens, local in ((8, 2), (8, 0)):
        g2, idx, wts, norm = _floor_inputs(tokens, local, rank)
        for round_ in range(2):
            for name, ws in arms if round_ == 0 else list(reversed(arms)):
                med, _ = _time_chain(
                    comm,
                    lambda ws=ws: ws.finalize(g2, idx, wts, norm, 1e-5, tokens),
                    _CALLS,
                )
                hi = comm.allreduce(med, op=MPI.MAX)
                if rank == 0:
                    print(
                        f"[finalize_transport placement] finalize_ar_{tokens}tok_{local}local round {round_} "
                        f"{name:22s} max {hi:7.3f} receive 0x{ws._uc[ws.rank].data_ptr():x}",
                        flush=True,
                    )
    comm.Barrier()

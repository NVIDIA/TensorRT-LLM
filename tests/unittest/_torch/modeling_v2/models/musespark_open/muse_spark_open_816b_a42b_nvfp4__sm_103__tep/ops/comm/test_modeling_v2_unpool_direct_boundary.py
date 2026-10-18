# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open unpool_direct_boundary entry (``unpool_into_boundary``; TP8, two GB300 nodes).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, e.g.
``srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>``;
anywhere else every test skips.

The entry claims the gather boundary's bits (``unpool_gather_boundary``, itself certified
against the built-in gather + cluster boundary) on the slices the producers
published into the direct transport. Checked as raw bits:

* every cell through CUDA-graph replays with changing inputs, special BF16
  payloads, an infinite input and a delayed rank, published by an elementwise
  copy, against the gather boundary and the built-in pair on the same slices, and against the
  entry's float64 reference;
* the production producer: the unpool GEMM (the plan's kernel at each row count,
  split-K at every row count, cuBLAS) writing into the slot, against the gather boundary on the
  same GEMM's ordinary output, rows 1..8;
* the layer chain as the decode graph runs it -- 61 layers (the model's fused
  boundaries) of unpool GEMM into the slot of the layer's buffer
  (``buffer_sequence``: the last one on the third buffer), boundary (entry
  trigger, DFlash capture on the capture layers), next QKV GEMM -- with spin
  kernels on rotating ranks inside the graph, replayed back to back with no host
  sync (so the next replay's first call follows the last with no other
  collective in between), and interleaved with the gather boundary's eager calls on the
  tagged inbox, against the same chain on the gather boundary;
* a concurrent poller that records the first value of every 16-bit half of every
  slot while the producers write: it must be the final value (one store per
  element), at least 10,000 calls per producer;
* NaN / infinite GEMM inputs never make a producer store the sentinel half;
* the guard word under a broken buffer sequence (one rank consumes another
  buffer): set, and read by the host while the poll still spins (the guard
  watch's view), and that the default guard never fired;
* last: every word of the transport is back to the sentinel.

The negative control of the buffer discipline (two consecutive calls on one buffer
must be detected) is ``comm/test_modeling_v2_unpool_direct_parity.py``.

``MODELING_V2_TIMING=1`` also runs the chain microbenchmark (printed, never gated):
finalize stand-in -> unpool GEMM -> boundary -> QKV GEMM, 61 layers of cold weights in
one graph, the gather boundary vs the direct transport, rounds in alternating order.
"""

import importlib
import os
import statistics
import time

import pytest
import torch

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.unpool_direct_boundary import (  # noqa: E501
    UnpoolDirectBoundary,
)

__extra_import_path__ = ["../../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)
D = importlib.import_module(_TARGET + ".ops.comm.unpool_direct_boundary")
GATHER_BOUNDARY = importlib.import_module(_TARGET + ".ops.comm.unpool_gather_boundary")
INBOX = importlib.import_module(_TARGET + ".ops.comm.projection_inbox")
GEMM = importlib.import_module(_TARGET + ".ops.gemm.dense_gemm")
PLAN = importlib.import_module(_TARGET + ".kernels.dense_gemms.dispatcher")

op = D.unpool_direct_boundary
f1 = GATHER_BOUNDARY.unpool_gather_boundary

WORLD = 8
HIDDEN, LOCAL_COLUMNS, MAX_ROWS = D.HIDDEN, D.LOCAL_COLUMNS, D.MAX_ROWS
#: The checkpoint's norm epsilons: rms_post_norm_eps for the post-FFN norm, rms_norm_eps next.
POST_EPS, PRE_EPS = 1e-6, 1e-5
#: The unpool projection's K (the expert hidden width) and the QKV projection's N per rank.
UNPOOL_K, QKV_N = 4096, 2304
#: The unpool kernel of every row count at 8 rows and the split-K family at every row count.
SPLITK = "splitk:64,8,4,12"
#: The served drafter's capture layers (Muse-Spark-Open-DFlash2 ``target_layer_ids``).
CAPTURE_LAYERS = (1, 16, 30, 44, 59)
#: Layers of the chain graphs: the model's 61 fused boundaries, on the model's buffers.
LAYERS = 61
BUFFERS = D.buffer_sequence(LAYERS)
#: -0, the smallest subnormal, the largest finite, the smallest normal, the most negative finite.
_SPECIAL = (-32768, 1, 32639, 128, -129)
SENTINEL_I32 = D.DIRECT_SENTINEL - (1 << 32)
#: Calls of the concurrent-poller test per producer.
POLLER_CALLS = int(os.environ.get("MODELING_V2_UNPOOL_DIRECT_CALLS", "10240"))


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs eight MPI ranks with CUDA (two four-GPU GB300 nodes)")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("needs GB300 (sm_103)")
    return comm, rank


@pytest.fixture(scope="module")
def group():
    """The built-in gather workspace, the gather boundary's inbox and the direct transport (collective)."""
    comm, rank = _world()
    from tensorrt_llm._torch.models.musespark_open_projection_gather import (
        ProjectionGatherWorkspace,
    )
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    builtin = ProjectionGatherWorkspace(mapping)
    inbox = INBOX.ProjectionInbox(mapping, row_words=LOCAL_COLUMNS // 2)
    direct = D.UnpoolDirectWorkspace(mapping)
    comm.Barrier()
    yield comm, rank, mapping, builtin, inbox, direct
    torch.cuda.synchronize()
    comm.Barrier()


def _agree(comm, ok: bool) -> bool:
    return comm.allreduce(int(bool(ok)), op=MPI.SUM) == WORLD


def _local(seed: int, rank: int, rows: int) -> torch.Tensor:
    """Rank ``rank``'s unpool slice under ``seed``; every rank can rebuild every slice."""
    gen = torch.Generator().manual_seed(seed * WORLD + rank)
    return (torch.randn(rows, LOCAL_COLUMNS, generator=gen) * 3.0).to(torch.bfloat16)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({4: torch.int32, 2: torch.int16, 1: torch.uint8}[t.element_size()])


def _differing(got, want) -> int:
    total = 0
    for g, w in zip(got, want):
        assert g.dtype == w.dtype and g.shape == w.shape, (g.shape, w.shape)
        total += int((_bits(g) != _bits(w)).sum().item())
    return total


class _Join:
    """The built-in post-FFN join's modules: gated residual, next pre-QKV norm, gain."""

    def __init__(self, seed: int, with_gain: bool, norm_fp32_residual: bool, trigger="exit"):
        from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenGatedResidual
        from tensorrt_llm._torch.modules.rms_norm import RMSNorm

        torch.manual_seed(seed)
        self.gate = MuseSparkOpenGatedResidual(HIDDEN, torch.bfloat16).cuda()
        self.gate.gate.normal_(0, 0.5)
        self.gate.cache_derived_state()
        assert self.gate._fused_enabled, "MUSESPARK_OPEN_FUSED_RESIDUAL must be on for this test"
        self.gate._preqkv_norm_fp32 = norm_fp32_residual
        self.norm = RMSNorm(hidden_size=HIDDEN, eps=PRE_EPS, dtype=torch.bfloat16, use_gemma=True)
        self.norm = self.norm.cuda()
        self.norm.weight.normal_(0, 0.02)
        self.gain = (
            torch.empty(HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
            if with_gain
            else None
        )
        self.norm_fp32_residual = norm_fp32_residual
        self.trigger = trigger

    def builtin(self, workspace, local, residual):
        return self.gate._add_after_rmsnorm_and_norm(
            residual, workspace.gather(local), POST_EPS, self.norm, self.gain
        )

    def tagged(self, inbox, local, residual, capture=None):
        alpha, beta = self.gate.coefficients()
        return f1(
            local,
            inbox,
            residual,
            self.gain,
            alpha,
            beta,
            self.norm.weight,
            POST_EPS,
            PRE_EPS,
            self.norm_fp32_residual,
            self.trigger,
            capture=capture,
        )

    def direct(self, workspace, parity, published, residual, capture=None):
        alpha, beta = self.gate.coefficients()
        return op(
            published,
            workspace,
            parity,
            residual,
            self.gain,
            alpha,
            beta,
            self.norm.weight,
            POST_EPS,
            PRE_EPS,
            self.norm_fp32_residual,
            self.trigger,
            capture,
        )

    def reference(self, slices, residual, capture=None):
        alpha, beta = self.gate.coefficients()
        return op.reference(
            slices,
            None,
            0,
            residual,
            self.gain,
            alpha,
            beta,
            self.norm.weight,
            POST_EPS,
            PRE_EPS,
            self.norm_fp32_residual,
            self.trigger,
            capture,
        )


def _settle(comm):
    torch.cuda.synchronize()
    comm.Barrier()


def _build(workspace, join, rows: int, captures=(False,)):
    """Compile the boundary variants a test launches, on every rank together (collective):
    a rank still compiling must not leave its peers polling into the guard."""
    workspace.build_variants(
        [
            D.variant(
                rows,
                join.gain is not None,
                POST_EPS,
                PRE_EPS,
                join.norm_fp32_residual,
                join.trigger,
                capture,
            )
            for capture in captures
        ]
    )


@pytest.mark.parametrize("cell", UnpoolDirectBoundary.CELLS, ids=lambda c: c.why[:48])
@torch.inference_mode()
def test_cells_match_f1_and_builtin_bits_and_reference(group, cell, certifying):  # noqa: F811
    comm, rank, _, builtin, inbox, direct = group
    certifying(op)
    rows = cell.spec["rows"]
    parity = cell.spec.get("parity", 0)
    trigger = cell.spec.get("trigger", "exit")
    capture = cell.spec.get("capture", False)
    join = _Join(
        seed=1235 + rows,
        trigger=trigger,
        **{k: cell.spec[k] for k in ("with_gain", "norm_fp32_residual")},
    )
    local = torch.zeros(rows, LOCAL_COLUMNS, device="cuda", dtype=torch.bfloat16)
    residual = torch.zeros(rows, HIDDEN, device="cuda", dtype=torch.float32)
    slot = direct.slot(parity, rows)
    got_capture = torch.empty(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    want_capture = torch.empty_like(got_capture)
    _build(direct, join, rows, (capture,))

    def tagged_step():
        return join.tagged(inbox, local, residual, want_capture if capture else None)

    def direct_step():
        D.publish_copy(slot, local)
        return join.direct(direct, parity, slot, residual, got_capture if capture else None)

    def builtin_step():
        return join.builtin(builtin, local, residual)

    for step in (tagged_step, direct_step, builtin_step):  # compile / warm, eagerly
        step()
        _settle(comm)
    graphs, outs = {}, {}
    for name, step in (("tagged", tagged_step), ("direct", direct_step), ("builtin", builtin_step)):
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            outs[name] = step()
    _settle(comm)

    for repeat in range(8):
        seed = 3000 * rows + 10 * parity + repeat
        slices = [_local(seed, r, rows) for r in range(WORLD)]
        if repeat == 1:
            for s in slices:
                s.view(torch.int16)[0, : len(_SPECIAL)] = torch.tensor(_SPECIAL, dtype=torch.int16)
        if repeat == 5:
            for s in slices:
                s[rows - 1, 7] = float("inf")
        local.copy_(slices[rank].cuda())
        residual.normal_().mul_(10.0 ** (repeat % 3 - 1))
        join.norm.weight.normal_(0, 0.02)
        if join.gain is not None:
            join.gain.normal_(1, 0.1)
        _settle(comm)
        if rank == repeat:
            time.sleep(0.002)
        graphs["builtin"].replay()
        graphs["tagged"].replay()
        graphs["direct"].replay()
        torch.cuda.synchronize()
        bad = _differing(outs["direct"], outs["tagged"]) + _differing(
            outs["direct"], outs["builtin"]
        )
        if capture:
            bad += _differing([got_capture], [want_capture])
        assert _agree(comm, bad == 0), f"rows={rows} replay {repeat}: {bad} values differ"
        if repeat in (1, 5):
            continue
        ref_capture = torch.empty_like(got_capture) if capture else None
        ref = join.reference([s.cuda() for s in slices], residual, ref_capture)
        pairs = list(zip(outs["direct"], ref, (2e-4, 1e-3)))
        if capture:
            pairs.append((got_capture, ref_capture, 1e-3))
        for g, r, band in pairs:
            error = ((g.float() - r.float()).norm() / r.float().norm()).item()
            assert error < band, f"rows={rows} replay {repeat}: relative error {error:.2e}"
    comm.Barrier()


def _plan_kernels(site: str):
    return PLAN.default_kernels()[site]


def _gemm_kernels(rows: int):
    """The unpool kernels a call of ``rows`` rows is published by in this test."""
    names = {_plan_kernels("unpool")[rows - 1], "cublas"}
    if GEMM.kernel_error(SPLITK, rows, UNPOOL_K, LOCAL_COLUMNS) is None:
        names.add(SPLITK)
    return sorted(names)


@pytest.mark.parametrize("rows", range(1, MAX_ROWS + 1))
@torch.inference_mode()
def test_a_gemm_publishes_the_bits_f1_gathers(group, rows):
    """The unpool GEMM writing its product into the slot (one launch, its own epilogue
    stores) gives the boundary the gather boundary's bits on the GEMM's ordinary output."""
    comm, rank, _, _, inbox, direct = group
    join = _Join(seed=41 + rows, with_gain=True, norm_fp32_residual=False, trigger="entry")
    gen = torch.Generator(device="cuda").manual_seed(500 + rank)
    weight = (0.03 * torch.randn(LOCAL_COLUMNS, UNPOOL_K, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    _build(direct, join, rows)
    bad = []
    for kernel in _gemm_kernels(rows):
        for parity in range(D.BUFFERS):
            x = torch.randn(rows, UNPOOL_K, generator=gen, device="cuda").to(torch.bfloat16)
            residual = torch.randn(rows, HIDDEN, device="cuda")
            slot = direct.slot(parity, rows)
            GEMM.dense_gemm(x, weight, kernel, True, out=slot)
            got = join.direct(direct, parity, slot, residual)
            local = GEMM.dense_gemm(x, weight, kernel, True)
            want = join.tagged(inbox, local, residual)
            torch.cuda.synchronize()
            differ = _differing(got, want)
            if differ:
                bad.append((kernel, parity, differ))
    _settle(comm)
    assert _agree(comm, not bad), f"rank {rank} rows={rows}: {bad}"


def _delays(rows: int, rank: int) -> dict:
    """Spin kernels (cycles) captured into the chain graphs, on a rotating rank per layer:
    before the GEMM (a late producer) or between the GEMM and the boundary (a late consumer)."""
    delays = {}
    for layer in range(LAYERS):
        if layer % 3 == 0 and (layer // 3 + rows) % WORLD == rank:
            delays[("producer", layer)] = (2_000, 20_000, 200_000)[(layer // 3) % 3]
        if layer % 4 == 1 and (layer + 3 * rows) % WORLD == rank:
            delays[("consumer", layer)] = (5_000, 60_000)[(layer // 4) % 2]
    return delays


class _Delay:
    """The in-graph delay kernels. ``plain``: ``torch.cuda._sleep`` (cycles), a plain launch,
    i.e. a full barrier on its rank. ``pdl``: ``_Spin`` (cycles / 2 ns) launched with PDL --
    it waits on its predecessor, spins, then releases its dependent -- so the chain stays
    all-PDL: coverage of the transport on a chain with no plain launch anywhere, under a
    late writer and a sleeping rank. Not a negative control: on GB300 a PDL chain stays
    ordered even through a kernel that never waits (observed on that hardware), so a
    missing wait cannot make these cases fail. The control that can fail is the buffer
    discipline's (``test_modeling_v2_unpool_direct_parity.py``)."""

    def __init__(self, kind: str):
        self.kind = kind
        self.spinners = {}

    def prepare(self, values) -> None:
        """Compile the spinners up front: a rank compiling inside the chain would leave its
        peers polling into the guard."""
        if self.kind == "pdl":
            for cycles in sorted(set(values)):
                self.spinners.setdefault(cycles // 2, _stand_in(cycles // 2))

    def __call__(self, cycles: int) -> None:
        if self.kind == "plain":
            torch.cuda._sleep(cycles)
        else:
            self.spinners[cycles // 2]()


class _Chain:
    """62 layers of: unpool GEMM -> boundary -> next QKV GEMM on the boundary's output."""

    def __init__(self, rows: int, rank: int, kernel: str, cold: bool = False):
        self.rows, self.kernel = rows, kernel
        self.qkv_kernel = _plan_kernels("qkv")[rows - 1]
        gen = torch.Generator(device="cuda").manual_seed(7000 + rank)
        count = LAYERS if cold else 1
        self.unpool_w = [
            (0.03 * torch.randn(LOCAL_COLUMNS, UNPOOL_K, generator=gen, device="cuda")).to(
                torch.bfloat16
            )
            for _ in range(count)
        ]
        self.qkv_w = [
            (0.02 * torch.randn(QKV_N, HIDDEN, generator=gen, device="cuda")).to(torch.bfloat16)
            for _ in range(count)
        ]
        self.x = torch.zeros(LAYERS, rows, UNPOOL_K, device="cuda", dtype=torch.bfloat16)
        self.residual = torch.zeros(rows, HIDDEN, device="cuda", dtype=torch.float32)
        self.capture = torch.zeros(
            len(CAPTURE_LAYERS), rows, HIDDEN, device="cuda", dtype=torch.bfloat16
        )
        self.rank = rank

    def fill(self, seed: int, special: bool = False):
        gen = torch.Generator().manual_seed(seed * 16 + self.rank)
        self.x.copy_(torch.randn(*self.x.shape, generator=gen).to(torch.bfloat16))
        if special:
            values = torch.tensor(_SPECIAL, dtype=torch.int16, device="cuda")
            self.x.view(torch.int16)[:, 0, : values.numel()] = values
        replicated = torch.Generator().manual_seed(seed * 16 + 99)
        self.residual.copy_(
            torch.randn(*self.residual.shape, generator=replicated) * 10.0 ** (seed % 3 - 1)
        )

    def layers(
        self, join, inbox, direct, use_direct: bool, delays: dict, stand_in=None, delay=None
    ):
        delay = torch.cuda._sleep if delay is None else delay
        outputs = []
        residual = self.residual
        for layer in range(LAYERS):
            w = self.unpool_w[layer % len(self.unpool_w)]
            if stand_in is not None:
                stand_in()
            if ("producer", layer) in delays:
                delay(delays[("producer", layer)])
            capture = None
            if layer in CAPTURE_LAYERS:
                capture = self.capture[CAPTURE_LAYERS.index(layer)]
            if use_direct:
                parity = BUFFERS[layer]
                slot = direct.slot(parity, self.rows)
                GEMM.dense_gemm(self.x[layer], w, self.kernel, True, out=slot)
                if ("consumer", layer) in delays:
                    delay(delays[("consumer", layer)])
                residual, normed = join.direct(direct, parity, slot, residual, capture)
            else:
                local = GEMM.dense_gemm(self.x[layer], w, self.kernel, True)
                if ("consumer", layer) in delays:
                    delay(delays[("consumer", layer)])
                residual, normed = join.tagged(inbox, local, residual, capture)
            qkv = GEMM.dense_gemm(
                normed, self.qkv_w[layer % len(self.qkv_w)], self.qkv_kernel, True
            )
            outputs.append((residual, normed, qkv))
        return outputs


def _chain_bits(got, want, capture_got, capture_want) -> int:
    bad = sum(_differing(g, w) for g, w in zip(got, want))
    return bad + _differing([capture_got], [capture_want])


@pytest.mark.parametrize(
    "rows,kind", [(8, "plain"), (5, "plain"), (1, "plain"), (8, "pdl"), (3, "pdl")]
)
@torch.inference_mode()
def test_the_decode_chain_back_to_back_with_in_graph_delays(group, rows, kind):
    """61 layers (the model's buffer sequence) in one graph per transport, spin kernels on
    rotating ranks inside it, replays back to back without a host sync (a fast rank then
    publishes the next calls -- the next replay's first one right after the last -- while a
    slow peer still reads), and the gather boundary's eager calls on the tagged inbox in
    between: the direct chain must give the gather chain's bits. ``kind`` is the delay kernel (``_Delay``): ``pdl``
    keeps every kernel of the chain a PDL launch (coverage, not a control)."""
    comm, rank, _, _, inbox, direct = group
    if kind == "pdl" and cute is None:
        pytest.skip("needs CuTe DSL")
    delay = _Delay(kind)
    kernel = _plan_kernels("unpool")[rows - 1]
    join = _Join(seed=61 + rows, with_gain=True, norm_fp32_residual=False, trigger="entry")
    tagged_chain = _Chain(rows, rank, kernel)
    direct_chain = _Chain(rows, rank, kernel)
    delays = _delays(rows, rank)
    delay.prepare(delays.values())
    _build(direct, join, rows, (False, True))
    tagged_chain.layers(join, inbox, direct, False, delays, delay=delay)
    direct_chain.layers(join, inbox, direct, True, delays, delay=delay)
    _settle(comm)
    with torch.cuda.graph(tagged_graph := torch.cuda.CUDAGraph()):
        want = tagged_chain.layers(join, inbox, direct, False, delays, delay=delay)
    with torch.cuda.graph(direct_graph := torch.cuda.CUDAGraph()):
        got = direct_chain.layers(join, inbox, direct, True, delays, delay=delay)
    graphs = {"tagged": tagged_graph, "direct": direct_graph}
    _settle(comm)

    results = []
    for repeat in range(4):
        for chain in (tagged_chain, direct_chain):
            chain.fill(100 * rows + repeat, special=repeat == 1)
        _settle(comm)
        graphs["tagged"].replay()
        graphs["direct"].replay()
        torch.cuda.synchronize()
        results.append(
            (f"replay {repeat}", _chain_bits(got, want, direct_chain.capture, tagged_chain.capture))
        )
    # Six direct replays back to back, each rank at its own pace, then eager gather calls on
    # the tagged inbox, then the direct graph again: the two transports never share memory.
    for chain in (tagged_chain, direct_chain):
        chain.fill(9000 + rows)
    _settle(comm)
    for _ in range(6):
        graphs["direct"].replay()
    eager = tagged_chain.layers(join, inbox, direct, False, {})
    graphs["direct"].replay()
    graphs["tagged"].replay()
    torch.cuda.synchronize()
    results.append(
        ("back to back", _chain_bits(got, want, direct_chain.capture, tagged_chain.capture))
    )
    results.append(("eager gather between", sum(_differing(g, w) for g, w in zip(eager, want))))
    _settle(comm)
    assert _agree(comm, all(bad == 0 for _, bad in results)), f"rank {rank}: {results}"


@pytest.mark.parametrize("rows", [8, 1])
@torch.inference_mode()
def test_nan_and_infinite_inputs_never_store_the_sentinel(group, rows):
    """A GEMM on rows with NaN and infinite activations stores canonical NaNs and
    infinities, never the sentinel half; the boundary then gives the gather boundary's bits."""
    comm, rank, _, _, inbox, direct = group
    join = _Join(seed=71, with_gain=True, norm_fp32_residual=False)
    gen = torch.Generator(device="cuda").manual_seed(800 + rank)
    weight = (0.03 * torch.randn(LOCAL_COLUMNS, UNPOOL_K, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    _build(direct, join, rows)
    bad = []
    for kernel in _gemm_kernels(rows):
        x = torch.randn(rows, UNPOOL_K, generator=gen, device="cuda").to(torch.bfloat16)
        x[0, 3] = float("nan")
        x[rows - 1, 11] = float("inf") if rows > 1 else float("-inf")
        x.view(torch.int16)[rows // 2, 17] = -2  # 0xFFFE: a NaN carrying the sentinel half
        residual = torch.randn(rows, HIDDEN, device="cuda")
        # The same kernel's ordinary output holds the bits the slot receives.
        local = GEMM.dense_gemm(x, weight, kernel, True)
        sentinel_halves = int((local.view(torch.int16) == -2).sum().item())
        if sentinel_halves:
            bad.append((kernel, "sentinel halves", sentinel_halves))
        nans = int(torch.isnan(local).sum().item())
        if nans == 0:
            bad.append((kernel, "the NaN input produced no NaN", 0))
        _settle(comm)
        if not _agree(comm, sentinel_halves == 0):
            break  # a sentinel half in a slot would leave its consumers polling
        slot = direct.slot(1, rows)
        GEMM.dense_gemm(x, weight, kernel, True, out=slot)
        got = join.direct(direct, 1, slot, residual)
        want = join.tagged(inbox, local, residual)
        torch.cuda.synchronize()
        differ = _differing(got, want)
        if differ:
            bad.append((kernel, "differ", differ))
        _settle(comm)
    assert _agree(comm, not bad), f"rank {rank} rows={rows}: {bad}"


# ---------------------------------------------------------------------------
# Concurrent poller: the first value of every half a poller sees is the final one.
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

if triton is not None:

    @triton.jit
    def _first_seen_kernel(
        receive,
        slot_words,
        buffer_words,
        counts,
        max_rounds,
        PART: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Poll ``PART`` words of one slot until no 16-bit half is 0xFFFE; record each
        half's first non-sentinel value, compare it with a final read, reset the words.

        ``counts`` (int64): [0] halves whose first value differs from the final one, [1]
        halves the first read saw as the sentinel (stores the poll watched arrive), [2]
        blocks that hit ``max_rounds``, [3] halves still the sentinel in those blocks.
        """
        slot = tl.program_id(1)
        lane = tl.arange(0, BLOCK)
        offsets = tl.program_id(0) * PART + lane
        mask = lane < PART
        base = receive + slot.to(tl.int64) * slot_words + buffer_words
        words = tl.load(base + offsets, mask=mask, other=0, volatile=True)
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
        tl.atomic_add(counts + 0, tl.sum(tl.where(mask, changed, 0), axis=0).to(tl.int64))
        tl.atomic_add(counts + 1, unseen.to(tl.int64))
        tl.atomic_add(counts + 2, (pending > 0).to(tl.int64))
        tl.atomic_add(counts + 3, pending.to(tl.int64))
        tl.store(base + offsets, tl.full([BLOCK], -65538, tl.int32), mask=mask)


#: Producers of the poller test, writing the unpool shape into the slot.
POLL_PRODUCERS = ("copy", SPLITK, "direct:64,2", "cublas")


def _reset(comm, direct):
    _settle(comm)
    direct.reset()
    _settle(comm)


@pytest.mark.parametrize("producer", POLL_PRODUCERS)
@torch.inference_mode()
def test_a_concurrent_poller_never_sees_a_value_before_the_final_one(group, producer):
    """Every rank's poller, started before the producers, records the first non-sentinel
    value of every 16-bit half of all eight slots while the producers write them through
    the multicast address: it must be the final value. Rows 1..8, both buffers, at least
    10,000 calls. The poller takes 4 programs per slot (32 CTAs at most), so it cannot
    keep a cluster-launched GEMM off the SMs (a poller on every SM can hold the GEMM back
    until its polls give up); every variant runs once untimed first."""
    import numpy as np

    comm, rank, _, _, _, direct = group
    if triton is None:
        pytest.skip("needs Triton")
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device=dev).manual_seed(9000 + rank)
    weight = (0.03 * torch.randn(LOCAL_COLUMNS, UNPOOL_K, generator=g, device=dev)).to(
        torch.bfloat16
    )
    receive = direct.receive()
    slab_words = MAX_ROWS * LOCAL_COLUMNS // 2
    max_rounds = 1 << 22
    side = torch.cuda.Stream(device=dev)

    def usable(rows):
        if producer in ("copy", "cublas"):
            return True
        return GEMM.kernel_error(producer, rows, UNPOOL_K, LOCAL_COLUMNS) is None

    row_counts = [rows for rows in range(1, MAX_ROWS + 1) if usable(rows)]

    def produce(rows, slot):
        if producer == "copy":
            partial = torch.randn(rows, LOCAL_COLUMNS, generator=g, device=dev)
            D.publish_copy(slot, partial.to(torch.bfloat16))
        else:
            x = torch.randn(rows, UNPOOL_K, generator=g, device=dev).to(torch.bfloat16)
            GEMM.dense_gemm(x, weight, producer, True, out=slot)

    def call(rows, parity, counts):
        words = rows * LOCAL_COLUMNS // 2
        part = words // 4
        block = triton.next_power_of_2(part)
        _settle(comm)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _first_seen_kernel[(4, WORLD)](
                receive,
                slab_words,
                parity * WORLD * slab_words,
                counts,
                max_rounds,
                PART=part,
                BLOCK=block,
                num_warps=max(4, min(16, block // 512)),
            )
        torch.cuda._sleep(20_000)  # let the pollers start before the stores
        produce(rows, direct.slot(parity, rows))
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

    warm = torch.zeros(4, dtype=torch.int64, device=dev)
    _reset(comm, direct)
    for parity in range(D.BUFFERS):
        for rows in row_counts:
            call(rows, parity, warm)
    _reset(comm, direct)

    counts = torch.zeros(MAX_ROWS, 4, dtype=torch.int64, device=dev)
    seconds = []
    for index in range(POLLER_CALLS):
        rows = row_counts[index % len(row_counts)]
        began = time.perf_counter()
        call(rows, (index // len(row_counts)) % D.BUFFERS, counts[rows - 1])
        seconds.append(time.perf_counter() - began)
    mine = counts.cpu().numpy()
    total = comm.allreduce(mine, op=MPI.SUM)
    _reset(comm, direct)
    if rank == 0:
        print(
            f"\n[unpool_direct poller] producer {producer}: {POLLER_CALLS} calls x {WORLD} slots per rank, "
            f"rows {row_counts}; call p50 {1e6 * statistics.median(seconds):.0f} us"
        )
        for rows in row_counts:
            changed, watched, stuck, unseen = (int(v) for v in total[rows - 1])
            print(
                f"[unpool_direct poller]   rows {rows}: halves changed after first seen {changed}, "
                f"watched arriving {watched}, blocks stuck {stuck} ({unseen} halves never seen)"
            )
    assert POLLER_CALLS >= 10_000 or os.environ.get("MODELING_V2_UNPOOL_DIRECT_CALLS")
    assert int(np.asarray(total)[:, 2].sum()) == 0, f"blocks never completed: {total[:, 2]}"
    assert int(np.asarray(total)[:, 0].sum()) == 0, f"halves changed: {total[:, 0]}"
    assert int(np.asarray(total)[:, 1].sum()) > 0, "no poller ever watched a store arrive"


@torch.inference_mode()
def test_the_guard_word_flags_a_broken_parity_sequence(group):
    """Every rank publishes into buffer 0; rank 0 consumes buffer 1, where nothing
    arrives: its guard fires (50 ms, and its poll gives up); the other ranks' results are
    the gather boundary's. The buffers are reset afterwards."""
    comm, rank, mapping, _, inbox, _ = group
    guarded = D.UnpoolDirectWorkspace(mapping, guard_ns=50_000_000, guard_break=True)
    join = _Join(seed=91, with_gain=True, norm_fp32_residual=False)
    rows = 8
    _build(guarded, join, rows)
    local = _local(4242, rank, rows).cuda()
    residual = torch.randn(rows, HIDDEN, device="cuda")
    want = join.tagged(inbox, local, residual)
    _settle(comm)
    D.publish_copy(guarded.slot(0, rows), local)
    parity = 1 if rank == 0 else 0
    got = join.direct(guarded, parity, guarded.slot(parity, rows), residual)
    fired = guarded.guard_fired()
    ok = rank == 0 or _differing(got, want) == 0
    _settle(comm)
    guarded.reset()
    guarded.clear_guard()
    _settle(comm)
    assert _agree(comm, fired == (rank == 0)), f"rank {rank}: guard fired {fired}"
    assert _agree(comm, ok), f"rank {rank}: the undisturbed ranks' result differs"
    assert _agree(comm, guarded.is_clean())


@torch.inference_mode()
def test_the_host_sees_an_expired_guard_while_the_poll_spins(group, monkeypatch):
    """Rank 0 consumes buffer 2 before anyone published there, with a 50 ms guard that
    keeps polling (as in the model). Its host must read the guard word set while the kernel still
    spins -- no synchronization, as ``start_guard_watch``'s thread reads it, which logs it --
    and then, once every rank publishes into buffer 2, the poll completes with the gather boundary's bits."""
    comm, rank, mapping, _, inbox, _ = group
    waiting = D.UnpoolDirectWorkspace(mapping, guard_ns=50_000_000, guard_break=False)
    join = _Join(seed=93, with_gain=True, norm_fp32_residual=False)
    rows = 8
    _build(waiting, join, rows)
    local = _local(4343, rank, rows).cuda()
    residual = torch.randn(rows, HIDDEN, device="cuda")
    want = join.tagged(inbox, local, residual)
    ready = torch.cuda.Event()
    ready.record()
    errors = []
    monkeypatch.setattr(D.logger, "error", lambda message: errors.append(message))
    _settle(comm)
    seen = True
    if rank == 0:
        waiting.start_guard_watch(0.01)
        got = join.direct(waiting, 2, waiting.slot(2, rows), residual)
        deadline = time.monotonic() + 10.0
        while not waiting.guard_fired(sync=False) and time.monotonic() < deadline:
            time.sleep(0.005)
        seen = waiting.guard_fired(sync=False) and not torch.cuda.current_stream().query()
        while not errors and time.monotonic() < deadline:
            time.sleep(0.005)
    comm.Barrier()  # rank 0's poll is still spinning on the device here
    # From a side stream: on rank 0 the main stream is held by the spinning boundary,
    # which polls rank 0's own slot like the peers'.
    side = torch.cuda.Stream()
    side.wait_event(ready)
    with torch.cuda.stream(side):
        D.publish_copy(waiting.slot(2, rows), local)
    torch.cuda.synchronize()
    ok = True
    if rank == 0:
        ok = _differing(got, want) == 0 and len(errors) == 1 and "rank 0" in errors[0]
        waiting.stop_guard_watch()
    _settle(comm)
    waiting.reset()
    waiting.clear_guard()
    _settle(comm)
    assert _agree(comm, seen), f"rank {rank}: the host did not see the guard while polling"
    assert _agree(comm, ok), f"rank {rank}: result or watch log wrong ({errors})"
    assert _agree(comm, waiting.is_clean())


@torch.inference_mode()
def test_unsupported_inputs_are_refused(group):
    comm, rank, _, _, _, direct = group
    from _validating import validating

    join = _Join(seed=5, with_gain=True, norm_fp32_residual=False)
    rows = 8
    residual = torch.randn(rows, HIDDEN, device="cuda")
    alpha, beta = join.gate.coefficients()
    args = (residual, join.gain, alpha, beta, join.norm.weight, POST_EPS, PRE_EPS)
    with validating(op):
        for published, parity in (
            (torch.empty(rows, LOCAL_COLUMNS, device="cuda", dtype=torch.bfloat16), 0),
            (direct.slot(0, rows), 1),
            (direct.slot(1, 4), 1),
            (direct.slot(2, rows), 1),
        ):
            with pytest.raises(ValueError):
                op(published, direct, parity, *args)
        with pytest.raises(ValueError):
            op(direct.slot(0, rows), direct, 0, *args, trigger="later")
    with pytest.raises(ValueError):
        direct.slot(D.BUFFERS, rows)
    with pytest.raises(ValueError):
        D.variant(8, True, POST_EPS, PRE_EPS, False, "never", False)
    # A variant not built before a capture is refused inside it, not compiled there.
    key = D.variant(8, True, 0.5, 0.25, False, "exit", False)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError):
        with torch.cuda.graph(graph):
            direct.kernel(key)
    _settle(comm)


@torch.inference_mode()
def test_the_transport_is_clean_and_no_guard_fired(group):
    """Last: every word consumed in this module is back to the sentinel, on every rank."""
    comm, rank, _, _, _, direct = group
    _settle(comm)
    clean, fired = direct.is_clean(), direct.guard_fired()
    assert _agree(comm, clean and not fired), f"rank {rank}: clean {clean}, guard fired {fired}"


# ---------------------------------------------------------------------------
# Microbenchmark: MODELING_V2_TIMING=1 (printed, never gated).
# ---------------------------------------------------------------------------


try:
    import cuda.bindings.driver as _cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int64
    from cutlass.cute.runtime import make_ptr
except ImportError:  # pragma: no cover - the benchmark needs CuTe DSL
    cute = None

if cute is not None and D.available():

    class _Spin:
        """A PDL kernel that waits on its predecessor, spins ``ns`` nanoseconds on the
        global timer and then lets its dependent launch (64 CTAs x 256 threads)."""

        def __init__(self, ns: int):
            self.ns = int(ns)

        @cute.jit
        def __call__(self, sink: cute.Pointer, stream: _cuda_driver.CUstream):
            self.kernel(sink).launch(
                grid=[64, 1, 1], block=[256, 1, 1], stream=stream, use_pdl=True
            )

        @cute.kernel
        def kernel(self, sink: cute.Pointer):
            cute.arch.griddepcontrol_wait()
            start = D._globaltimer()
            now = D._globaltimer()
            while now - start < Int64(self.ns):
                now = D._globaltimer()
            cute.arch.griddepcontrol_launch_dependents()


def _stand_in(ns: int):
    """The expert finalize in front of the unpool GEMM, as a launcher of ``_Spin(ns)``."""
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    stream = _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        _Spin(ns),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
        stream,
        options="--enable-tvm-ffi",
    )

    def launch():
        compiled(sink.data_ptr(), _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream))

    return launch


def _time(comm, graphs: dict, rounds: int, trials: int = 5, replays: int = 20):
    """Median microseconds per replay of each graph on this rank, rounds in alternating order."""
    names = list(graphs)
    samples = {name: [] for name in names}
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
    return {name: statistics.median(v) for name, v in samples.items()}, samples


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="MODELING_V2_TIMING=1")
@torch.inference_mode()
def test_microbenchmark_decode_chain(group):
    """Finalize stand-in (6 us, trigger at its end) -> unpool split-K GEMM -> boundary
    (the early-release entry trigger, the capture fold on the capture layers) -> next QKV split-K GEMM,
    61 layers of cold weights in one graph per transport, 8 rows. Microseconds per layer
    (replay / 61), each rank's median over 6 rounds x 5 trials x 20 replays; printed per
    arm as the maximum and the spread over ranks, and the per-rank direct - tagged deltas."""
    comm, rank, _, _, inbox, direct = group
    rows = 8
    join = _Join(seed=99, with_gain=True, norm_fp32_residual=False, trigger="entry")
    stand_in = _stand_in(6_000)
    _build(direct, join, rows, (False, True))
    chains = {name: _Chain(rows, rank, SPLITK, cold=True) for name in ("tagged", "direct")}
    for chain in chains.values():
        chain.fill(5)
    graphs, outs = {}, {}
    for name, chain in chains.items():
        chain.layers(join, inbox, direct, name == "direct", {}, stand_in)
        _settle(comm)
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            outs[name] = chain.layers(join, inbox, direct, name == "direct", {}, stand_in)
        _settle(comm)
    medians, samples = _time(comm, graphs, rounds=6)
    per_layer = {name: value / LAYERS for name, value in medians.items()}
    bad = _chain_bits(
        outs["direct"], outs["tagged"], chains["direct"].capture, chains["tagged"].capture
    )
    results = comm.gather((per_layer, bad, {k: v[:5] for k, v in samples.items()}), root=0)
    if rank == 0:
        print(f"\n[unpool_direct bench] decode chain, rows={rows}, {LAYERS} layers (us per layer)")
        for name in ("tagged", "direct"):
            values = [r[0][name] for r in results]
            print(
                f"[unpool_direct bench] {name:7s} max {max(values):8.3f}  min {min(values):8.3f}  "
                f"per rank {' '.join(f'{v:.3f}' for v in values)}"
            )
        deltas = [r[0]["direct"] - r[0]["tagged"] for r in results]
        slowest = max(r[0]["direct"] for r in results) - max(r[0]["tagged"] for r in results)
        print(
            f"[unpool_direct bench] direct - tagged: slowest rank {slowest:+.3f} us/layer "
            f"({slowest * 61:+.1f} us per 61-boundary step); per rank "
            f"{' '.join(f'{d:+.3f}' for d in deltas)}; differing bits {[r[1] for r in results]}"
        )
    comm.Barrier()

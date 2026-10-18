# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of ``projection_gathers`` as a decode step runs it (TP8, two GB300 nodes).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes (four ranks
per node, ``python -m pytest <this file>`` under the MPI launcher); anywhere else
every test skips.

The op tests under ``comm/`` drive one fused call per CUDA-graph replay with a
host barrier between replays. This file drives both fused ops the way the
decode graphs do and checks, as raw bits against the built-in path, what one
call at a time cannot show:

* the layer chain on two streams -- the pool site on a preparation stream, the
  unpool site on the main stream, event joins between them -- for every row
  count 1..8, 62 layers in one CUDA graph, with rank-specific delays captured
  inside the graph, and fused replays issued back to back with no host sync:
  a fast rank then publishes call k+1 into the other slot while a slow peer
  still reads call k;
* the eight row counts' graphs replayed in a random interleaving, so the
  per-row tag and slot state crosses graphs of different row counts;
* the re-classed MoE's ``prepare_expert_input``, both branches;
* that the fused-or-built-in choice reads no pointer value (every rank must
  make it alike) and that a misaligned operand raises before any launch;
* that a site whose pinned built-in code changed on one rank is dropped on all;
* last: every inbox word consumed in this module was cleared back to zero, and
  the per-row counters agree on all ranks.

``MODELING_V2_TIMING=1`` also times the two-stream chain and the kernels' JIT
compilation (printed, never gated).
"""

import importlib
import os
import random
import statistics
import time
import types

import pytest
import torch

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)
inbox_mod = importlib.import_module(_TARGET + ".ops.comm.projection_inbox")
U = importlib.import_module(_TARGET + ".ops.comm.unpool_gather_boundary")
P = importlib.import_module(_TARGET + ".ops.comm.pool_gather_fp4quant")
PG = importlib.import_module(_TARGET + ".kernels.projection_gathers.gathers")

WORLD = 8
#: Pool sites per decode step (every layer); the unpool runs at 61 of them in
#: the model, all 62 here.
LAYERS = 62
#: The checkpoint's epsilons: rms_post_norm_eps (post-FFN norm), rms_norm_eps (the others).
POST_EPS, PRE_EPS, EPS = 1e-6, 1e-5, 1e-5
#: -0, the smallest subnormal, the largest finite, the smallest normal, the most negative finite.
_SPECIAL = (-32768, 1, 32639, 128, -129)


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
def env():
    """Built-in workspaces, inboxes, an aux stream and the join's modules (collective)."""
    comm, rank = _world()
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenGatedResidual
    from tensorrt_llm._torch.models.musespark_open_projection_gather import (
        ProjectionGatherWorkspace,
    )
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.mapping import Mapping

    e = types.SimpleNamespace(comm=comm, rank=rank)
    e.mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    e.builtin_pool = ProjectionGatherWorkspace(e.mapping)
    e.builtin_unpool = ProjectionGatherWorkspace(e.mapping)
    e.pool_inbox = inbox_mod.ProjectionInbox(e.mapping, row_words=P.LOCAL_COLUMNS // 2)
    e.unpool_inbox = inbox_mod.ProjectionInbox(e.mapping, row_words=U.LOCAL_COLUMNS // 2)
    e.aux = torch.cuda.Stream()
    torch.manual_seed(1234)  # the same weights on every rank, as under TP
    with torch.no_grad():
        e.gate = MuseSparkOpenGatedResidual(U.HIDDEN, torch.bfloat16).cuda()
        e.gate.gate.normal_(0, 0.5)
        e.gate.cache_derived_state()
        assert e.gate._fused_enabled, "MUSESPARK_OPEN_FUSED_RESIDUAL must be on for this test"
        e.norm = RMSNorm(hidden_size=U.HIDDEN, eps=PRE_EPS, dtype=torch.bfloat16, use_gemma=True)
        e.norm = e.norm.cuda()
        e.norm.weight.normal_(0, 0.02)
    e.post_gain = torch.empty(U.HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
    e.pool_gain = torch.empty(P.HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
    e.global_scale = torch.tensor([448.0 * 6.0 / 12.0], device="cuda", dtype=torch.float32)
    e.graphs = {}
    comm.Barrier()
    yield e
    torch.cuda.synchronize()
    e.graphs.clear()
    comm.Barrier()


def _bf16(shape, seed: int, scale: float) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)


def _bits(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    return t.view(torch.uint8)


#: What a layer produces, compared bit for bit.
_OUTPUTS = ("payload", "block scales", "gathered row", "residual", "next input")


def _differing(got, want) -> dict:
    bad = dict.fromkeys(_OUTPUTS, 0)
    for g_layer, w_layer in zip(got, want):
        for name, g, w in zip(_OUTPUTS, g_layer, w_layer):
            assert g.shape == w.shape and g.dtype == w.dtype, (name, g.shape, w.shape)
            bad[name] += int((_bits(g) != _bits(w)).sum().item())
    return bad


def _agree_zero(comm, bad: dict) -> bool:
    return comm.allreduce(sum(bad.values()), op=MPI.SUM) == 0


class _Chain:
    """Per-layer inputs of the two sites: each rank's pool and unpool slices, the residual."""

    def __init__(self, rows: int, rank: int):
        self.rows, self.rank = rows, rank
        self.pool = torch.zeros(LAYERS, rows, P.LOCAL_COLUMNS, device="cuda", dtype=torch.bfloat16)
        self.unpool = torch.zeros(
            LAYERS, rows, U.LOCAL_COLUMNS, device="cuda", dtype=torch.bfloat16
        )
        self.residual = torch.zeros(LAYERS, rows, U.HIDDEN, device="cuda", dtype=torch.float32)

    def fill(self, seed: int, special: bool = False):
        self.pool.copy_(_bf16(self.pool.shape, seed * 16 + self.rank, (0.05, 1.0, 20.0)[seed % 3]))
        self.unpool.copy_(_bf16(self.unpool.shape, seed * 16 + 8 + self.rank, 3.0))
        gen = torch.Generator().manual_seed(seed * 16 + 99)  # replicated on every rank
        self.residual.copy_(
            torch.randn(*self.residual.shape, generator=gen) * 10.0 ** (seed % 3 - 1)
        )
        if special:
            values = torch.tensor(_SPECIAL, dtype=torch.int16, device="cuda")
            self.pool.view(torch.int16)[:, 0, : values.numel()] = values
            self.unpool.view(torch.int16)[:, 0, : values.numel()] = values


def _delays(rows: int, rank: int) -> dict:
    """GPU sleeps (cycles) captured into the graph, on a rotating rank per layer and stream."""
    delays = {}
    for layer in range(LAYERS):
        if layer % 3 == 0 and (layer // 3 + rows) % WORLD == rank:
            delays[("aux", layer)] = (2_000, 20_000, 200_000)[(layer // 3) % 3]
        if layer % 4 == 1 and (layer + 3 * rows) % WORLD == rank:
            delays[("main", layer)] = (5_000, 60_000)[(layer // 4) % 2]
    return delays


def _layers(e, chain: _Chain, fused: bool, delays: dict):
    """62 layers of: (aux) pool site; join; (main) unpool site -- as the decode graphs order them."""
    from flashinfer.norm import rmsnorm_fp4quant

    main = torch.cuda.current_stream()
    alpha, beta = e.gate.coefficients()
    outputs = []
    for layer in range(LAYERS):
        e.aux.wait_stream(main)
        with torch.cuda.stream(e.aux):
            if ("aux", layer) in delays:
                torch.cuda._sleep(delays[("aux", layer)])
            pooled = chain.pool[layer].clone()
            if fused:  # as projection_gathers calls it
                fp4, scale, gathered = P.pool_gather_rmsnorm_fp4quant(
                    pooled, e.pool_inbox, e.pool_gain, e.global_scale, EPS, return_gathered=True
                )
            else:
                gathered = e.builtin_pool.gather(pooled)
                fp4, scale = rmsnorm_fp4quant(
                    gathered,
                    e.pool_gain,
                    eps=EPS,
                    global_scale=e.global_scale.reshape(1),
                    block_size=16,
                    scale_format="e4m3",
                    is_sf_swizzled_layout=False,
                )
                fp4, scale = fp4.view(torch.uint8), scale.view(torch.uint8)
        main.wait_stream(e.aux)
        if ("main", layer) in delays:
            torch.cuda._sleep(delays[("main", layer)])
        unpooled = chain.unpool[layer].clone()
        if fused:
            updated, normalized = U.unpool_gather_boundary(
                unpooled,
                e.unpool_inbox,
                chain.residual[layer],
                e.post_gain,
                alpha,
                beta,
                e.norm.weight,
                POST_EPS,
                PRE_EPS,
                False,
            )
        else:
            updated, normalized = e.gate._add_after_rmsnorm_and_norm(
                chain.residual[layer],
                e.builtin_unpool.gather(unpooled),
                POST_EPS,
                e.norm,
                e.post_gain,
            )
        outputs.append((fp4, scale, gathered, updated, normalized))
    return outputs


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@torch.inference_mode()
def test_two_stream_chain_back_to_back(env, rows):
    e = env
    chain = _Chain(rows, e.rank)
    chain.fill(1)
    delays = _delays(rows, e.rank)
    _layers(e, chain, False, delays)  # compiles, eagerly
    _layers(e, chain, True, delays)
    torch.cuda.synchronize()
    e.comm.Barrier()
    builtin_graph, fused_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(builtin_graph):
        want = _layers(e, chain, False, delays)
    with torch.cuda.graph(fused_graph):
        got = _layers(e, chain, True, delays)
    torch.cuda.synchronize()
    e.comm.Barrier()

    results = []
    for repeat in range(4):
        chain.fill(100 * rows + repeat, special=repeat == 1)
        torch.cuda.synchronize()
        e.comm.Barrier()
        builtin_graph.replay()
        fused_graph.replay()
        torch.cuda.synchronize()
        results.append((f"replay {repeat}", _differing(got, want)))
    # Six fused replays back to back, no host sync, each rank at its own pace.
    e.comm.Barrier()
    for _ in range(6):
        fused_graph.replay()
    builtin_graph.replay()
    torch.cuda.synchronize()
    results.append(("6 back-to-back replays", _differing(got, want)))
    e.graphs[rows] = (chain, builtin_graph, fused_graph, want, got)
    clean = [_agree_zero(e.comm, bad) for _, bad in results]
    e.comm.Barrier()
    assert all(clean), f"rows={rows}: differing bits on some rank; this rank: {results}"


@torch.inference_mode()
def test_interleaved_row_count_graphs(env):
    """The eight fused graphs replayed back to back in one random order (the same on all ranks)."""
    e = env
    assert sorted(e.graphs) == list(range(1, 9)), "needs every rows case of the chain test"
    order = random.Random(4242)
    results = []
    for round_ in range(3):
        for rows, (chain, *_) in e.graphs.items():
            chain.fill(7000 + 10 * round_ + rows, special=round_ == 2)
        sequence = [order.randint(1, 8) for _ in range(40)]
        torch.cuda.synchronize()
        e.comm.Barrier()
        for rows in sequence:
            e.graphs[rows][2].replay()
        for rows in sorted(set(sequence)):
            e.graphs[rows][1].replay()
        torch.cuda.synchronize()
        for rows in sorted(set(sequence)):
            _, _, _, want, got = e.graphs[rows]
            results.append((round_, rows, _differing(got, want)))
    clean = [_agree_zero(e.comm, bad) for *_, bad in results]
    e.comm.Barrier()
    assert all(clean), f"differing bits on some rank; this rank: {results}"


def _fake_op_backend():
    """An instance that passes ``isinstance(..., TRTLLMOpBackend)``, as the built-in checks."""
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend

    fake = type("FakeTRTLLMOpBackend", (TRTLLMOpBackend,), {})
    fake.__abstractmethods__ = frozenset()
    return object.__new__(fake)


def _stub_moe(e):
    """A re-classed MoE with what ``prepare_expert_input`` reads, wired to this module's inboxes."""
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    moe = PG.MuseSparkOpenTepGatherMoE.__new__(PG.MuseSparkOpenTepGatherMoE)
    torch.nn.Module.__init__(moe)
    torch.manual_seed(77)
    with torch.no_grad():
        moe.pre_expert_norm = RMSNorm(
            hidden_size=P.HIDDEN, eps=EPS, dtype=torch.bfloat16, use_gemma=True
        ).cuda()
        moe.pre_expert_norm.weight.normal_(0, 0.05)
    moe.register_buffer("_pre_expert_quant_gain", None, persistent=False)
    moe._pre_expert_quant_fused = True
    moe.expert_hidden_size = P.HIDDEN
    backend = types.SimpleNamespace(
        op_backend=_fake_op_backend(),
        has_nvfp4=True,
        fc31_act_scale=None,
        fc31_input_scale=torch.tensor([224.0], device="cuda", dtype=torch.float32),
    )
    moe.experts = types.SimpleNamespace(backend=backend, comm=None, moe_max_num_tokens=8192)
    moe._pool_gather_workspace = e.builtin_pool
    moe._pool_inbox = e.pool_inbox
    moe._defer_pool = False
    moe._defer_unpool = False
    return moe, backend


@torch.inference_mode()
def test_reclassed_moe_prepares_the_builtin_expert_input(env):
    """``prepare_expert_input(slice)`` equals the built-in's on the gathered row: the fused op
    where the built-in runs flashinfer's fused quantization, the built-in gather + norm where
    it does not (here: an activation scale is set)."""
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE

    e = env
    moe, backend = _stub_moe(e)
    bad, arms = 0, []
    for declined in (False, True):
        backend.fc31_act_scale = torch.ones(1, device="cuda") if declined else None
        for rows in (1, 3, 8):
            local = _bf16(
                (rows, P.LOCAL_COLUMNS), 80000 + rows * 16 + e.rank + 1000 * declined, 2.0
            )
            local = local.cuda()
            want = MuseSparkOpenMoE.prepare_expert_input(moe, e.builtin_pool.gather(local))
            got = moe.prepare_expert_input(local)
            torch.cuda.synchronize()
            if declined:
                assert isinstance(got, torch.Tensor) and isinstance(want, torch.Tensor)
                bad += int((_bits(got) != _bits(want)).sum().item())
            else:
                assert type(got) is type(want) and type(got).__name__ == "Fp4QuantizedTensor"
                assert got.is_sf_swizzled == want.is_sf_swizzled
                for g, w in (
                    (got.fp4_tensor, want.fp4_tensor),
                    (got.scaling_factor, want.scaling_factor),
                ):
                    assert g.dtype == w.dtype and g.shape == w.shape
                    bad += int((_bits(g) != _bits(w)).sum().item())
            arms.append((declined, rows))
    clean = e.comm.allreduce(bad, op=MPI.SUM) == 0
    e.comm.Barrier()
    assert clean, f"{bad} differing bytes on this rank over arms {arms}"


def _misaligned(rows: int, columns: int, dtype: torch.dtype) -> torch.Tensor:
    """A contiguous ``[rows, columns]`` tensor one element past a 16-byte boundary."""
    buffer = torch.zeros(rows * columns + 16, device="cuda", dtype=dtype)
    tensor = buffer[1 : 1 + rows * columns].view(rows, columns)
    assert tensor.is_contiguous() and tensor.data_ptr() % 16
    return tensor


@torch.inference_mode()
def test_the_fused_choice_reads_no_pointer_value(env):
    """A misaligned operand still selects the fused path -- the choice reads only what every
    rank shares -- and the fused path raises before launching anything (no counter moves).

    The built-in predicates would decline here (``cluster_boundary_supported`` checks alignment);
    taken on one rank only, that would have left the peers waiting in the inbox.
    """
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenGatedResidual
    from tensorrt_llm._torch.models.musespark_open_boundary import cluster_boundary_supported

    e = env
    rows = 8
    torch.manual_seed(1234)
    with torch.no_grad():
        gate = MuseSparkOpenGatedResidual(U.HIDDEN, torch.bfloat16).cuda()
        gate.cache_derived_state()
    gate.__class__ = PG.MuseSparkOpenTepGatedResidual
    gate._gather_inbox = e.unpool_inbox
    gate._gather = e.builtin_unpool
    alpha, beta = gate.coefficients()
    aligned_residual = torch.randn(rows, U.HIDDEN, device="cuda")
    aligned_local = _bf16((rows, U.LOCAL_COLUMNS), 5, 1.0).cuda()
    unpool_counters = e.unpool_inbox.counters.clone()
    for residual, local in (
        (_misaligned(rows, U.HIDDEN, torch.float32), aligned_local),
        (aligned_residual, _misaligned(rows, U.LOCAL_COLUMNS, torch.bfloat16)),
    ):
        assert gate._fits_fused(residual, local, e.norm, e.post_gain, alpha, beta)
        if residual is not aligned_residual:
            gathered = torch.zeros(rows, U.HIDDEN, device="cuda", dtype=torch.bfloat16)
            assert not cluster_boundary_supported(
                residual, gathered, e.post_gain, alpha, beta, e.norm.weight
            )
        with pytest.raises(RuntimeError, match="16-byte"):
            gate._add_after_rmsnorm_and_norm(residual, local, POST_EPS, e.norm, e.post_gain)

    moe, _ = _stub_moe(e)
    pool_counters = e.pool_inbox.counters.clone()
    pooled = _misaligned(rows, P.LOCAL_COLUMNS, torch.bfloat16)
    assert moe._pool_quant_args(pooled) is not None
    with pytest.raises(RuntimeError, match="16-byte"):
        moe.prepare_expert_input(pooled)

    torch.cuda.synchronize()
    assert torch.equal(e.unpool_inbox.counters, unpool_counters)
    assert torch.equal(e.pool_inbox.counters, pool_counters)
    e.comm.Barrier()


@torch.inference_mode()
def test_the_sites_are_agreed_on_every_rank(env, monkeypatch):
    """Both sites fuse where every rank can build their kernels; a rank that cannot build one
    makes every rank refuse."""
    e = env
    assert PG.fusable_sites(e.mapping, unpool=True, pool=True) == (True, True)
    assert PG.fusable_sites(e.mapping, unpool=False, pool=True) == (False, True)
    if e.rank == 3:
        monkeypatch.setattr(PG._unpool, "available", lambda: False)
    with pytest.raises(ValueError, match=r"projection_gathers \(unpool\) needs"):
        PG.fusable_sites(e.mapping, unpool=True, pool=True)
    monkeypatch.undo()
    if e.rank == 5:
        monkeypatch.setattr(PG._pool, "available", lambda: False)
    with pytest.raises(ValueError, match=r"projection_gathers \(pool\) needs"):
        PG.fusable_sites(e.mapping, unpool=True, pool=True)
    monkeypatch.undo()
    e.comm.Barrier()


# ---------------------------------------------------------------------------
# Microbenchmark: MODELING_V2_TIMING=1 (printed, never gated). Before the
# hygiene test, so that its fused calls are covered by it too.
# ---------------------------------------------------------------------------


def _time_graphs(comm, graphs: dict, calls: int, rounds: int = 20):
    samples = {name: [] for name in graphs}
    names = list(graphs)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        for name in names:
            comm.Barrier()
            graphs[name].replay()
            torch.cuda.synchronize()
    for r in range(rounds):
        for name in names if r % 2 == 0 else reversed(names):
            torch.cuda.synchronize()
            comm.Barrier()
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0 / calls)
    return {name: statistics.median(v) for name, v in samples.items()}


def _print(comm, rank, tag, values: dict, unit: str):
    results = comm.gather(values, root=0)
    if rank == 0:
        print(f"\n[projection_gathers timing] {tag} ({unit}; median over ranks, min..max)")
        for name in results[0]:
            column = sorted(r[name] for r in results)
            print(
                f"[projection_gathers timing] {tag} {name:34s} {statistics.median(column):9.3f} "
                f"({column[0]:.3f}..{column[-1]:.3f})"
            )


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="MODELING_V2_TIMING=1")
@torch.inference_mode()
def test_microbenchmark_jit_compile(env):
    """Seconds to JIT-compile each kernel variant a K<=7 decode step uses (lru caches cleared;
    CuTe DSL's own caches, if any, are not)."""
    e = env
    from tensorrt_llm._torch.models import musespark_open_boundary as builtin

    device = torch.cuda.current_device()
    seconds = {}
    for rows in range(1, 9):
        U._compile.cache_clear()
        start = time.perf_counter()
        U._compile(rows, True, POST_EPS, PRE_EPS, device, False, e.rank)
        seconds[f"fused unpool boundary rows={rows}"] = time.perf_counter() - start
        builtin._compile_boundary.cache_clear()
        start = time.perf_counter()
        builtin._compile_boundary(rows, True, POST_EPS, PRE_EPS, device, False)
        seconds[f"built-in boundary rows={rows}"] = time.perf_counter() - start
    P._compile.cache_clear()
    start = time.perf_counter()
    P._compile(device, e.rank, True, False)
    seconds["fused pool quant (all rows)"] = time.perf_counter() - start
    seconds["fused unpool boundary total (8 row counts)"] = sum(
        v for k, v in seconds.items() if k.startswith("fused unpool boundary rows=")
    )
    _print(e.comm, e.rank, "jit", seconds, "s")
    e.comm.Barrier()


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="MODELING_V2_TIMING=1")
@pytest.mark.parametrize("rows", [8, 1])
@torch.inference_mode()
def test_microbenchmark_two_stream_chain(env, rows):
    """The chain of the bit test without delays: per layer, built-in vs fused."""
    e = env
    chain = _Chain(rows, e.rank)
    chain.fill(3)
    graphs = {}
    for name, fused in (("built-in chain", False), ("fused chain", True)):
        _layers(e, chain, fused, {})
        torch.cuda.synchronize()
        e.comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _layers(e, chain, fused, {})
        graphs[name] = graph
    _print(
        e.comm, e.rank, f"two-stream rows={rows}", _time_graphs(e.comm, graphs, LAYERS), "us/layer"
    )
    torch.cuda.synchronize()
    e.comm.Barrier()


@torch.inference_mode()
def test_inboxes_are_clean_and_counters_agree(env):
    """After every fused call above: all words cleared, the same per-row state on every rank."""
    e = env
    torch.cuda.synchronize()
    e.comm.Barrier()
    nonzero, counters = {}, {}
    for name, inbox in (("pool", e.pool_inbox), ("unpool", e.unpool_inbox)):
        nonzero[name] = int((inbox.segment() != 0).sum().item())
        counters[name] = inbox.counters.cpu().tolist()
    everyone = e.comm.allgather(counters)
    agree = all(c == everyone[0] for c in everyone)
    dirty = e.comm.allreduce(sum(nonzero.values()), op=MPI.SUM)
    e.comm.Barrier()
    assert dirty == 0 and agree, (nonzero, counters)
    assert all(tag != 0 for name in counters for tag, _ in counters[name]), counters

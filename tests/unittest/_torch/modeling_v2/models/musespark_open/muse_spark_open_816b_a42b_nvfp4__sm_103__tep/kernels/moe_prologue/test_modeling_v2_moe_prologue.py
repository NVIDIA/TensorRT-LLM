# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the fast path ``moe_prologue`` as a decode step runs it (TP8, two GB300 nodes).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, e.g.
``srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>``;
anywhere else every test skips.

The op tests (``moe/test_modeling_v2_router_shard.py``,
``comm/test_modeling_v2_pool_gather_route.py``) check one call at a time. This
file checks, as raw bits against the built-in chain (full router, routing,
the replicated experts' placement on the main stream; pool gather + quantization on the
preparation stream; joined):

* the 62-layer chain on one stream -- router shard, routed pool op -- for every
  row count 1..8, in one CUDA graph with rank-specific delays captured inside
  it, replayed back to back without host sync, with and without the early
  trigger; then the row counts' graphs in a random interleaving;
* the re-classed MoE's ``prepare_moe_inputs``: its logits, the routing memo the
  experts and the finalize read, the replicated experts' placement and the expert input,
  against the built-in method (with the projection gathers and the replicated experts)
  on the same module; and the built-in path where the fast path does not apply;
* ``install`` on a stand-in model (class, inbox, the dropped dense GEMM prefetch);
* ``router_fork``: the same 62-layer chain with each router shard forked onto a
  side stream of its own as a plain launch (``router_shard_side``) and joined
  before the routed pool op, eager and in one CUDA graph, with GPU sleeps on the
  side stream in front of the router on a rotating rank: the built-in bits, the
  captured edge into each side launch a full one and the edge out of it a
  programmatic one into the routed pool op; the re-classed MoE's
  ``prepare_moe_inputs`` with the side stream set; and the negative control --
  without the join, a delayed side router is read stale;
* last: every inbox word consumed was cleared, and the counters agree.
"""

import importlib
import random
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
R = importlib.import_module(_TARGET + ".ops.comm.pool_gather_route")
P = importlib.import_module(_TARGET + ".ops.comm.pool_gather_fp4quant")
S = importlib.import_module(_TARGET + ".ops.moe.router_shard")
A = importlib.import_module(_TARGET + ".ops.moe.ring_buddy_assign")
PG = importlib.import_module(_TARGET + ".kernels.projection_gathers.gathers")
RB = importlib.import_module(_TARGET + ".kernels.replicated_experts.ring_buddy")
MP = importlib.import_module(_TARGET + ".kernels.moe_prologue.prologue")

#: The TP size under test: the MPI world when it is 4 (one node) or 8 (two nodes).
WORLD = MPI.COMM_WORLD.Get_size() if MPI is not None and MPI.COMM_WORLD.Get_size() in (4, 8) else 8
#: The layout of this TP size: pool slice columns, prologue inbox row words, router columns.
P_LOCAL = P.local_columns(WORLD)
R_ROW_WORDS = R.row_words(WORLD)
S_LOCAL = S.local_experts(WORLD)
#: The chain's placement: the balanced ring-buddy one against ``ring_buddy_assign`` at
#: TP8; at TP4 (no built-in placement kernel) the routing alone.
CHAIN_PLACE = R.PLACE_BALANCE if WORLD == 8 else R.PLACE_NONE
LAYERS = 62
EPS, RENORM_EPS = 1e-5, 1e-15
SCALE = (8192 / 256.0) ** -0.5
#: Router weights cycled over the layers (eight distinct ones are enough to
#: change the logits between neighbouring layers).
WEIGHTS = 8


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs eight MPI ranks (two four-GPU nodes) or four (one) with CUDA")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
        pytest.skip("needs sm_103 or sm_107")
    return comm, rank


def _routing(bias):
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenRoutingMethod

    return MuseSparkOpenRoutingMethod(
        top_k=8,
        num_experts=256,
        callable_e_score_correction_bias=lambda: bias,
        renorm_eps=RENORM_EPS,
    )


@pytest.fixture(scope="module")
def env():
    """Built-in pool workspace, the gather and prologue inboxes, an aux stream, router weights (collective)."""
    comm, rank = _world()
    from tensorrt_llm._torch.models.musespark_open_projection_gather import (
        ProjectionGatherWorkspace,
    )
    from tensorrt_llm.mapping import Mapping

    e = types.SimpleNamespace(comm=comm, rank=rank)
    e.mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    e.builtin_pool = ProjectionGatherWorkspace(e.mapping)
    e.pool_inbox = inbox_mod.ProjectionInbox(e.mapping, row_words=P_LOCAL // 2)
    e.inbox = inbox_mod.ProjectionInbox(e.mapping, row_words=R_ROW_WORDS)
    e.aux = torch.cuda.Stream()
    gen = torch.Generator(device="cuda").manual_seed(1234)  # the same weights on every rank
    e.router = [torch.randn(256, 8192, generator=gen, device="cuda") * 0.02 for _ in range(WEIGHTS)]
    e.bias = (torch.randn(256, generator=gen, device="cuda") * 0.05).contiguous()
    e.gain = torch.empty(P.HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
    e.global_scale = torch.tensor([224.0], device="cuda", dtype=torch.float32)
    e.graphs = {}
    e.held = []
    comm.Barrier()
    yield e
    torch.cuda.synchronize()
    e.graphs.clear()
    comm.Barrier()


def _bits(t: torch.Tensor) -> torch.Tensor:
    if t.element_size() == 4:
        return t.view(torch.int32)
    if t.element_size() == 2:
        return t.view(torch.int16)
    return t.view(torch.uint8)


_OUTPUTS = (
    ("logits", "selected", "weights", "local_ids", "weights_bf16", "stats", "fp4", "scale")
    if CHAIN_PLACE != R.PLACE_NONE
    else ("logits", "selected", "weights", "fp4", "scale")
)


def _pack(**outputs) -> tuple:
    """One layer's outputs in ``_OUTPUTS`` order."""
    return tuple(outputs[name] for name in _OUTPUTS)


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
    """Per-layer inputs: the replicated decoder rows and each rank's pool slice."""

    def __init__(self, rows: int, rank: int):
        self.rows, self.rank = rows, rank
        self.hidden = torch.zeros(LAYERS, rows, 8192, device="cuda", dtype=torch.bfloat16)
        self.pool = torch.zeros(LAYERS, rows, P_LOCAL, device="cuda", dtype=torch.bfloat16)

    def fill(self, seed: int):
        gen = torch.Generator().manual_seed(seed * 16 + 99)  # replicated on every rank
        scale = torch.exp(1.5 * torch.randn(LAYERS, self.rows, 1, generator=gen))
        self.hidden.copy_((torch.randn(*self.hidden.shape, generator=gen) * scale).bfloat16())
        gen = torch.Generator().manual_seed(seed * 16 + self.rank)
        self.pool.copy_((torch.randn(*self.pool.shape, generator=gen) * 2).bfloat16())


def _delays(rows: int, rank: int) -> dict:
    """GPU sleeps (cycles) captured into the graph on a rotating rank per layer."""
    return {
        layer: (2_000, 20_000, 200_000)[(layer // 3) % 3]
        for layer in range(LAYERS)
        if layer % 3 == 0 and (layer // 3 + rows) % WORLD == rank
    }


def _builtin_router(hidden, weight):
    from tensorrt_llm._torch.models.modeling_musespark_open import _router_projection_kernel

    rows = hidden.shape[0]
    tile = 1 if rows in (1, 2, 3, 5) else 2
    logits = torch.empty(rows, 256, device="cuda")
    _router_projection_kernel[(rows, 256 // tile)](
        hidden,
        weight,
        logits,
        SCALE,
        EXPERTS=256,
        HIDDEN=8192,
        ROWS=tile,
        num_warps=16,
        enable_fp_fusion=False,
    )
    return logits


def _layers(
    e,
    chain: _Chain,
    fused: bool,
    early: bool,
    delays: dict,
    routing,
    fork=None,
    side_delays=None,
    join: bool = True,
):
    """62 layers of the decode MoE prologue, built-in (two streams) or the prologue (one stream).

    ``fork``: the router fork's side stream; the router shard then runs there as a plain launch
    (after a GPU sleep where ``side_delays`` names the layer), the pool slice copy on
    the current stream, joined before the routed pool op (not with ``join=False``: the
    negative control, eager only; its logits buffer starts at a sentinel value).
    """
    from flashinfer.norm import rmsnorm_fp4quant

    main = torch.cuda.current_stream()
    side_delays = side_delays or {}
    outputs = []
    for layer in range(LAYERS):
        hidden = chain.hidden[layer]
        weight = e.router[layer % WEIGHTS]
        if layer in delays:
            torch.cuda._sleep(delays[layer])
        if fused:
            warps = S.builtin_warps(chain.rows, True)
            if fork is None:
                logits_local = S.router_shard(
                    hidden, weight, e.rank, SCALE, warps, True, early, ranks=WORLD
                )
            else:
                logits_local = (
                    torch.empty(chain.rows, S_LOCAL, device="cuda")
                    if join
                    else torch.full((chain.rows, S_LOCAL), 7.0, device="cuda")
                )
                if not join:
                    e.held.append(logits_local)
                fork.wait_stream(main)
                with torch.cuda.stream(fork):
                    if layer in side_delays:
                        torch.cuda._sleep(side_delays[layer])
                    S.router_shard_side(
                        hidden, weight, e.rank, SCALE, warps, logits_local, ranks=WORLD
                    )
            pooled = chain.pool[layer].clone()
            if fork is not None and join:
                main.wait_stream(fork)
            out = R.pool_gather_route(
                pooled,
                logits_local,
                e.inbox,
                e.gain,
                e.global_scale,
                EPS,
                e.bias,
                RENORM_EPS,
                CHAIN_PLACE,
                R.local_experts(WORLD),
            )
            outputs.append(
                _pack(
                    logits=out.logits,
                    selected=out.selected,
                    weights=out.weights,
                    local_ids=out.local_ids,
                    weights_bf16=out.weights_bf16,
                    stats=out.stats,
                    fp4=out.fp4,
                    scale=out.scale,
                )
            )
            continue
        e.aux.wait_stream(main)
        logits = _builtin_router(hidden, weight)
        routing.clear_routing_memo()
        selected, weights = routing.apply(logits)
        ids = weights_bf16 = stats = None
        if CHAIN_PLACE != R.PLACE_NONE:
            ids, weights_bf16, stats = A.ring_buddy_assign(selected, weights, e.rank, True)
        with torch.cuda.stream(e.aux):
            gathered = e.builtin_pool.gather(chain.pool[layer].clone())
            fp4, scale = rmsnorm_fp4quant(
                gathered,
                e.gain,
                eps=EPS,
                global_scale=e.global_scale.reshape(1),
                block_size=16,
                scale_format="e4m3",
                is_sf_swizzled_layout=False,
            )
        main.wait_stream(e.aux)
        outputs.append(
            _pack(
                logits=logits,
                selected=selected,
                weights=weights,
                local_ids=ids,
                weights_bf16=weights_bf16,
                stats=stats,
                fp4=fp4.view(torch.uint8),
                scale=scale.view(torch.uint8),
            )
        )
    return outputs


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@torch.inference_mode()
def test_one_stream_chain_back_to_back(env, rows):
    e = env
    early = rows % 2 == 0
    chain = _Chain(rows, e.rank)
    chain.fill(1)
    delays = _delays(rows, e.rank)
    routing = _routing(e.bias)
    _layers(e, chain, False, early, delays, routing)
    _layers(e, chain, True, early, delays, routing)
    torch.cuda.synchronize()
    e.comm.Barrier()
    builtin_graph, fused_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(builtin_graph):
        want = _layers(e, chain, False, early, delays, routing)
    with torch.cuda.graph(fused_graph):
        got = _layers(e, chain, True, early, delays, routing)
    torch.cuda.synchronize()
    e.comm.Barrier()

    results = []
    for repeat in range(3):
        chain.fill(100 * rows + repeat)
        torch.cuda.synchronize()
        e.comm.Barrier()
        builtin_graph.replay()
        fused_graph.replay()
        torch.cuda.synchronize()
        results.append((f"replay {repeat}", _differing(got, want)))
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
    for round_ in range(2):
        for rows, (chain, *_) in e.graphs.items():
            chain.fill(7000 + 10 * round_ + rows)
        sequence = [order.randint(1, 8) for _ in range(30)]
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


def _side_delays(rows: int, rank: int) -> dict:
    """GPU sleeps (cycles) on the router fork's side stream in front of the router, on a rotating rank."""
    return {
        layer: (20_000, 200_000, 2_000_000)[(layer // 3) % 3]
        for layer in range(LAYERS)
        if layer % 3 == 1 and (layer // 3 + rows + 3) % WORLD == rank
    }


_SIDE_KERNEL = "_router_shard_side_kernel"


def _fork_edge_problems(graph) -> list:
    """One side launch per layer, entered by full edges only: one from the current stream
    (none for at most one launch, the first, when nothing ran on the current stream before
    it) and at most one from the previous side launch (the side stream's own order); left by
    one programmatic edge, into the routed pool op, and at most one full edge (the side
    stream's next launch)."""
    from cuda.bindings import driver

    handle = driver.CUgraph(graph.raw_cuda_graph())
    _, _, count = driver.cuGraphGetNodes(handle, 0)
    _, nodes, _ = driver.cuGraphGetNodes(handle, int(count))
    names = {}
    for node in nodes:
        _, kind = driver.cuGraphNodeGetType(node)
        if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            names[int(node)] = f"<{kind.name}>"
            continue
        _, params = driver.cuGraphKernelNodeGetParams(node)
        status, raw = driver.cuFuncGetName(params.func)
        if status != driver.CUresult.CUDA_SUCCESS:
            status, raw = driver.cuKernelGetName(params.kern)
        names[int(node)] = raw.decode() if isinstance(raw, bytes) else str(raw)
    get = getattr(driver, "cuGraphGetEdges_v2", None) or driver.cuGraphGetEdges
    total = int(get(handle, 0)[-1])
    _, sources, targets, data, _ = get(handle, total)
    programmatic = int(driver.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC)
    edges = [
        (int(a), int(b), int(d.type) == programmatic) for a, b, d in zip(sources, targets, data)
    ]
    side = [n for n, name in names.items() if name == _SIDE_KERNEL]
    problems = [] if len(side) == LAYERS else [f"{len(side)} side launches, want {LAYERS}"]
    roots = 0
    for node in side:
        into = [(names[a], prog) for a, b, prog in edges if b == node]
        out = [(names[b], prog) for a, b, prog in edges if a == node]
        chained = sum(name == _SIDE_KERNEL for name, _ in into)
        roots += not into
        if into and (any(prog for _, prog in into) or chained > 1 or len(into) - chained != 1):
            problems.append(f"edges into a side launch: {into}")
        joins = [name for name, prog in out if prog]
        if len(joins) != 1 or "PoolGatherRoute" not in joins[0] or len(out) > 2:
            problems.append(f"edges out of a side launch: {out}")
    if roots > 1:
        problems.append(f"{roots} side launches with no edge into them")
    return problems[:6]


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@torch.inference_mode()
def test_forked_router_chain_back_to_back(env, rows):
    """The router fork: the chain with every router shard forked onto a side stream (sleeps in front of
    it on a rotating rank) gives the built-in bits eagerly and in a CUDA graph, replayed
    with new inputs and back to back; the graph's side launches are entered by full edges
    and leave by a programmatic edge into the routed pool op."""
    e = env
    chain = _Chain(rows, e.rank)
    chain.fill(2)
    delays, side_delays = _delays(rows, e.rank), _side_delays(rows, e.rank)
    routing = _routing(e.bias)
    side = MP.dedicated_stream()
    results = []
    want = _layers(e, chain, False, False, delays, routing)
    got = _layers(e, chain, True, False, delays, routing, fork=side, side_delays=side_delays)
    torch.cuda.synchronize()
    results.append(("eager", _differing(got, want)))
    e.comm.Barrier()
    builtin_graph = torch.cuda.CUDAGraph()
    fork_graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(builtin_graph):
        want = _layers(e, chain, False, False, delays, routing)
    with torch.cuda.graph(fork_graph):
        got = _layers(e, chain, True, False, delays, routing, fork=side, side_delays=side_delays)
    problems = _fork_edge_problems(fork_graph)
    torch.cuda.synchronize()
    e.comm.Barrier()
    for repeat in range(3):
        chain.fill(300 * rows + repeat)
        torch.cuda.synchronize()
        e.comm.Barrier()
        builtin_graph.replay()
        fork_graph.replay()
        torch.cuda.synchronize()
        results.append((f"replay {repeat}", _differing(got, want)))
    e.comm.Barrier()
    for _ in range(6):
        fork_graph.replay()
    builtin_graph.replay()
    torch.cuda.synchronize()
    results.append(("6 back-to-back replays", _differing(got, want)))
    clean = [_agree_zero(e.comm, bad) for _, bad in results]
    everyone = e.comm.allgather(problems)
    e.comm.Barrier()
    assert all(clean), f"rows={rows}: differing bits on some rank; this rank: {results}"
    assert not any(everyone), f"rows={rows}: captured edges per rank: {everyone}"


@torch.inference_mode()
def test_a_fork_without_the_join_is_caught(env):
    """The negative control of the forked chain: with no join, the routed pool op on the rank
    whose side router sleeps 4M cycles reads the logits buffer before the router wrote it,
    and the comparison above sees it on every rank."""
    e = env
    chain = _Chain(8, e.rank)
    chain.fill(5)
    routing = _routing(e.bias)
    side = MP.dedicated_stream()
    side_delays = {0: 4_000_000} if e.rank == 3 else {}
    want = _layers(e, chain, False, False, {}, routing)
    got = _layers(
        e, chain, True, False, {}, routing, fork=side, side_delays=side_delays, join=False
    )
    torch.cuda.synchronize()
    e.held.clear()
    bad = _differing(got[:1], want[:1])
    total = e.comm.allreduce(sum(bad.values()), op=MPI.SUM)
    e.comm.Barrier()
    assert total > 0, "a stale read of the forked router's logits went unnoticed"


def _fake_op_backend():
    """An instance that passes ``isinstance(..., TRTLLMOpBackend)``, as the built-in checks."""
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend

    fake = type("FakeTRTLLMOpBackend", (TRTLLMOpBackend,), {})
    fake.__abstractmethods__ = frozenset()
    return object.__new__(fake)


class _LinearDispatcher:
    """A deterministic low-M dispatcher for the stand-in pooler (one GEMM kernel on every call)."""

    def apply(self, module, input_tensor, weight, bias, force_active=False):
        return torch.nn.functional.linear(input_tensor, weight, bias)


def _stub_moe(e, place: str):
    """The gather MoE with what ``prepare_moe_inputs`` reads, the ring-buddy routing hook,
    then re-classed as the prologue MoE."""
    from tensorrt_llm._torch.models.modeling_musespark_open import LayerNvtx
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    moe = PG.MuseSparkOpenTepGatherMoE.__new__(PG.MuseSparkOpenTepGatherMoE)
    torch.nn.Module.__init__(moe)
    torch.manual_seed(77)
    with torch.no_grad():
        moe.pre_expert_norm = RMSNorm(
            hidden_size=P.HIDDEN, eps=EPS, dtype=torch.bfloat16, use_gemma=True
        ).cuda()
        moe.pre_expert_norm.weight.normal_(0, 0.05)
        moe.router = torch.nn.Linear(8192, 256, bias=False, device="cuda", dtype=torch.float32)
        moe.router.weight.copy_(e.router[0])
        moe.pooler = torch.nn.Linear(8192, P.HIDDEN, bias=False, device="cuda")
        moe.pooler = moe.pooler.to(torch.bfloat16)
        moe.pooler.weight.normal_(0, 0.02)
    moe.pooler._low_m_gemm_dispatcher = _LinearDispatcher()
    moe.register_buffer("_pre_expert_quant_gain", None, persistent=False)
    moe._pre_expert_quant_fused = True
    moe.expert_hidden_size = P.HIDDEN
    moe.num_experts = 256
    moe.metap_multiplier = SCALE
    moe._bf16x3_router_enabled = False
    moe.mapping = e.mapping
    moe.nvtx = LayerNvtx(
        "moe", 0, ("router", "pool", "experts", "finalize", "all_reduce", "unpool")
    )
    backend = types.SimpleNamespace(
        op_backend=_fake_op_backend(),
        has_nvfp4=True,
        fc31_act_scale=None,
        fc31_input_scale=torch.tensor([224.0], device="cuda", dtype=torch.float32),
    )
    routing = _routing(e.bias)
    moe.experts = types.SimpleNamespace(
        backend=backend, comm=None, moe_max_num_tokens=8192, routing_method=routing
    )
    moe._prepare_stream = torch.cuda.Stream()
    moe._pool_gather_workspace = e.builtin_pool
    moe._pool_inbox = e.pool_inbox
    moe._defer_pool = True
    moe._defer_unpool = False
    state = None
    if place != "none":
        state = RB._State("nvfp4", {}, e.rank, place == "balance", 28 if place == "static" else 32)
        RB._patch_routing(moe, state)
        moe._replicated_experts = state
    moe.__class__ = MP.MuseSparkOpenTepPrologueMoE
    moe._prologue_inbox = e.inbox
    moe._prologue_enable_pdl = True
    return moe, routing, state


def _expert_input_bits(result) -> list:
    logits, pooled = result
    return [_bits(pooled.fp4_tensor).clone(), _bits(pooled.scaling_factor).clone()]


@pytest.mark.parametrize("fork", [False, True], ids=["moe_prologue", "router_fork"])
@pytest.mark.parametrize("place", ["balance", "none", "owner", "static"])
@torch.inference_mode()
def test_reclassed_moe_prepares_the_builtin_moe_inputs(env, place, fork):
    """The prologue's ``prepare_moe_inputs`` against the built-in method (with the projection gathers
    and the replicated experts) on the same module: the logits, the routing memo, the placement and
    the NVFP4 expert input, bit for bit; the experts' and the finalize's routing calls then hit the
    memo and the placement. With the router fork's side stream set the router shard runs there
    (``router_shard_side``), with the same bits."""
    if WORLD != 8 and place != "none":
        pytest.skip("the ring-buddy placement (replicated_experts) is a TP8 install")
    e = env
    moe, routing, state = _stub_moe(e, place)
    if fork:
        moe._router_stream = MP.dedicated_stream()
    builtin = PG.MuseSparkOpenTepGatherMoE.prepare_moe_inputs
    side_calls = []
    side_call = S.RouterShardSide.__call__

    def counted_side(self, *args, **kwargs):
        side_calls.append(torch.cuda.current_stream().cuda_stream)
        return side_call(self, *args, **kwargs)

    bad, checked = 0, []
    for step, rows in enumerate((8, 1, 3, 8, 6, 2, 8)):
        gen = torch.Generator().manual_seed(9000 + step)
        hidden = (torch.randn(rows, 8192, generator=gen) * (1 + step)).bfloat16().cuda()
        routing.clear_routing_memo()
        want_logits, want_pooled = builtin(moe, hidden)
        torch.cuda.synchronize()
        want_selected, want_weights = routing.apply(want_logits)
        want = [want_logits, want_selected, want_weights]
        want += _expert_input_bits((want_logits, want_pooled))
        if state is not None:
            want += [t.clone() for t in state.current(want_logits)[2:]]
        routing.clear_routing_memo()
        calls = []
        original = R.PoolGatherRoute.__call__

        def counted(self, *args, **kwargs):
            calls.append(args[0].shape[0])
            return original(self, *args, **kwargs)

        R.PoolGatherRoute.__call__ = counted
        S.RouterShardSide.__call__ = counted_side
        side_calls.clear()
        try:
            got_logits, got_pooled = moe.prepare_moe_inputs(hidden)
        finally:
            R.PoolGatherRoute.__call__ = original
            S.RouterShardSide.__call__ = side_call
        torch.cuda.synchronize()
        assert calls == [rows], "the prologue path did not run"
        want_side = [moe._router_stream.cuda_stream] if fork else []
        assert side_calls == want_side, (
            "the router shard did not run on the router fork's side stream"
        )
        memo = routing._routing_memo
        assert memo is not None and memo[0] is got_logits
        selected, weights = routing.apply(got_logits)
        assert selected is memo[2] and weights is memo[3], "the experts' routing missed the memo"
        got = [got_logits, selected, weights] + _expert_input_bits((got_logits, got_pooled))
        if state is not None:
            placement = state.current(got_logits)
            assert placement is not None, "the ring-buddy placement was not filled for these logits"
            got += list(placement[2:])
        for g, w in zip(got, want):
            assert g.shape == w.shape and g.dtype == w.dtype
            bad += int((_bits(g) != _bits(w)).sum().item())
        checked.append(rows)
    clean = e.comm.allreduce(bad, op=MPI.SUM) == 0
    e.comm.Barrier()
    assert clean, f"{bad} differing elements on this rank over rows {checked}"


@torch.inference_mode()
def test_more_than_eight_rows_take_the_builtin_path(env):
    e = env
    moe, routing, state = _stub_moe(e, "balance" if WORLD == 8 else "none")
    hidden = torch.randn(9, 8192, device="cuda").bfloat16()
    assert moe._prologue_router_warps(hidden) is None
    assert moe._prologue_router_warps(hidden[:8].contiguous()) == 16
    moe._prepare_stream = None
    assert moe._prologue_router_warps(hidden[:8].contiguous()) == 4
    assert moe._prologue_router_warps(hidden[:2].contiguous()) == 16
    moe._defer_pool = False
    assert moe._prologue_router_warps(hidden[:8].contiguous()) is None
    e.comm.Barrier()


@torch.inference_mode()
def test_install_on_a_stand_in_model(env):
    e = env
    layers = []
    for index in range(3):
        moe = PG.MuseSparkOpenTepGatherMoE.__new__(PG.MuseSparkOpenTepGatherMoE)
        torch.nn.Module.__init__(moe)
        moe._pool_inbox = e.pool_inbox
        layers.append(types.SimpleNamespace(feed_forward=moe, layer_idx=index, mapping=e.mapping))
    model = types.SimpleNamespace(layers=layers)
    assert MP.install(model) == 3
    for layer in layers:
        assert type(layer.feed_forward) is MP.MuseSparkOpenTepPrologueMoE
        assert layer.feed_forward._prologue_inbox is layers[0].feed_forward._prologue_inbox
    with pytest.raises(ValueError, match="extends the fused pool site"):
        MP.install(model)
    e.comm.Barrier()


@torch.inference_mode()
def test_inbox_is_clean_and_counters_agree(env):
    """After every fused call above: all words cleared, the same per-row state on every rank."""
    e = env
    torch.cuda.synchronize()
    e.comm.Barrier()
    nonzero = int((e.inbox.segment() != 0).sum().item())
    counters = e.inbox.counters.cpu().tolist()
    everyone = e.comm.allgather(counters)
    agree = all(c == everyone[0] for c in everyone)
    dirty = e.comm.allreduce(nonzero, op=MPI.SUM)
    e.comm.Barrier()
    assert dirty == 0 and agree, (nonzero, counters)

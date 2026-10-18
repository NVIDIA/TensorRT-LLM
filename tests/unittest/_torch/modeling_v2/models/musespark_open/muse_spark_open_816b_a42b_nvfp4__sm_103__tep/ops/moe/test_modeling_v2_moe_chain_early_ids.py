# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of ``early_expert_ids`` in the tier-3 decode MoE chain (TP8, two GB300 nodes).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, e.g.
``srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>``;
anywhere else every test skips.

The chain of ``test_modeling_v2_moe_chain.py`` (the prologue's routed pool kernel
-> the persistent experts -> the ``finalize_transport`` finalize, one MoE layer of
synthetic NVFP4 experts on constructed logits) is run twice per layer and step: as
the tier-3 set runs it, and with early expert ids (the routed kernel publishes the
placed ids and triggers early; the expert kernel takes them before its dependency
wait). Two layers back to back, every kernel launched with PDL, each layer's
routed-kernel inputs written by a late PDL producer (triggers at entry, waits,
spins, then writes them) and one rank sleeping before each step: an all-PDL
transitivity negative control, since the one-chain prologue leaves no plain launch
between two expert launches. Claims:

* the early chain's finalized blocks, routed rows' inputs (expert input, routing,
  placement) are bitwise the tier-3 chain's, at rows 1..8, for every logit
  construction, with each trigger point (``exchange``, ``entry``, ``exit``),
  eagerly and in CUDA-graph replays with the inputs changed in place;
* every early expert launch actually ran early (the buffer was armed for it);
* with the ``exchange`` and ``exit`` triggers no polling thread fell back. With
  ``entry`` the expert kernel is resident while the routed kernel's logit
  exchange still waits for the sleeping rank: eagerly that rank's host launches
  trail the others' by more than the poll deadline, so threads fall back (the
  count is printed), and the bits must still be the tier-3 chain's -- the
  fallback path under a real cross-rank delay;
* after every early layer the 64 published words carry exactly the routed
  kernel's ``local_ids`` (rows < M) or -1 (rows >= M) as payload and the epoch
  of the paired expert launch as tag, for every placement the routed kernel
  publishes (owner, balanced, static from 24 and from 8, and the balanced
  placement over three ring windows, whose ids address a 96-expert
  ``[own | b-1 | b-2]`` stack), rows 1..8, three logit constructions, with the
  layer bitwise the tier-3 chain's (the three-window chain without early ids for
  the three-window placement) and no fallback;
* a CUDA graph holding only the early chain (so a late rank's skew reaches its
  routed kernel's exchange, not a tier-3 chain in front), replayed 2000 times
  with a randomly chosen rank sleeping on the host before each replay and the
  inputs changed every 50 replays, stays bitwise the tier-3 eager chain with no
  fallback (``exchange``), with the two-window placement and with three windows;
* the three-window chain in CUDA-graph replays with new inputs and a delayed
  rank is bitwise the same chain without early ids, every expert launch early,
  no fallback.
"""

import random
import time

import pytest
import torch

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

__extra_import_path__ = ["."]
import test_modeling_v2_moe_chain as base  # noqa: E402
from _early_ids_kernels import late_producer  # noqa: E402

_T = base._T
WORLD = base.WORLD
BLOCK, EXPERTS, POOL_COLUMNS = base.BLOCK, base.EXPERTS, base.POOL_COLUMNS
EPS, RENORM_EPS, FIN_EPS = base.EPS, base.RENORM_EPS, base.FIN_EPS
KINDS = base.KINDS
ROWS = base.ROWS
LAYERS = 2
#: The late producer's spin before it writes the routed kernel's inputs.
LATE_NS = 30_000
TRIGGERS = ("exchange", "entry", "exit")


@pytest.fixture(scope="module")
def env():
    """The tier-3 chain environment, plus the early-ids buffer and two layers of staged inputs."""
    comm, rank = base._world()
    import importlib

    from tensorrt_llm._torch.models.musespark_open_finalize_ar import FinalizeIpcWorkspace
    from tensorrt_llm.mapping import Mapping

    R = importlib.import_module(_T + ".ops.comm.pool_gather_route")
    inbox_mod = importlib.import_module(_T + ".ops.comm.projection_inbox")
    fat = importlib.import_module(_T + ".ops.moe.finalize_allreduce_transport")
    pe = importlib.import_module(_T + ".ops.moe.persistent_experts")
    ei = importlib.import_module(_T + ".ops.moe.early_expert_ids")
    if not (R.available() and fat.HAS_CUTE_DSL):
        pytest.skip("needs the CuTe DSL and flashinfer's RMSNormFP4QuantKernel")
    device = torch.device("cuda", torch.cuda.current_device())
    mapping = Mapping(
        world_size=WORLD, rank=rank, gpus_per_node=4, tp_size=WORLD, moe_ep_size=WORLD
    )
    e = type("Env", (), {})()
    e.comm, e.rank, e.R, e.fat, e.pe, e.ei = comm, rank, R, fat, pe, ei
    e.inbox = inbox_mod.ProjectionInbox(mapping, row_words=R.ROW_WORDS)
    e.transport = fat.FinalizeTransportWorkspace(
        FinalizeIpcWorkspace(mapping, use_mnnvl=True), mapping, fat.DEFAULT
    )
    own = base._block(7, rank, device)
    predecessor = base._block(7, (rank - 1) % WORLD, device)
    second = base._block(7, (rank - 2) % WORLD, device)
    e.stacked = {k: torch.cat([own[k], predecessor[k]]).contiguous() for k in own}
    # The [own | b-1 | b-2] stack the three-window placement addresses.
    e.stacked3 = {k: torch.cat([own[k], predecessor[k], second[k]]).contiguous() for k in own}
    del own, predecessor, second
    e.early = pe.early_expert_ids(device)
    for stacks in (e.stacked, e.stacked3):
        pe.persistent_experts.prepare(
            stacks["w1"], stacks["s1"], stacks["w2"], stacks["s2"], early=e.early
        )
    gen = torch.Generator(device=device).manual_seed(77)
    e.norm = (torch.randn(base.HIDDEN, device=device, generator=gen) * 0.1).to(torch.bfloat16)
    # Per layer: the staged values (host-written) and the live inputs the late producer writes.
    e.stage_local = torch.zeros(LAYERS, 8, POOL_COLUMNS, device=device, dtype=torch.bfloat16)
    e.stage_logits = torch.zeros(LAYERS, 8, BLOCK, device=device)
    e.live_local = torch.zeros(LAYERS, 8, POOL_COLUMNS, device=device, dtype=torch.bfloat16)
    e.live_logits = torch.zeros(LAYERS, 8, BLOCK, device=device)
    e.bias = torch.zeros(EXPERTS, device=device)
    e.gain = torch.ones(base.HIDDEN, device=device, dtype=torch.bfloat16)
    e.global_scale = torch.tensor([224.0], device=device)
    # Count the early expert launches (the buffer armed for exactly that call's ids).
    e.early_launches = 0
    original = ei.launches_early

    def counted(early, topk_ids):
        ran = original(early, topk_ids)
        e.early_launches += int(ran)
        return ran

    ei.launches_early = counted
    torch.cuda.synchronize()
    comm.Barrier()
    yield e
    ei.launches_early = original
    torch.cuda.synchronize()
    comm.Barrier()
    e.transport.legacy.close()


def _set_inputs(e, seed: int, rows: int, kind: str) -> None:
    for layer in range(LAYERS):
        s = seed * 7 + layer
        gen = torch.Generator().manual_seed(s * WORLD + e.rank)
        scale = (0.05, 1.0, 20.0)[s % 3]
        local = (torch.randn(rows, POOL_COLUMNS, generator=gen) * scale).to(torch.bfloat16)
        logits = base._logits(s, rows, kind)
        e.stage_local[layer, :rows].copy_(local)
        e.stage_logits[layer, :rows].copy_(logits[:, BLOCK * e.rank : BLOCK * (e.rank + 1)])
        # The live buffers hold the previous step's values (or zeros) until the late
        # producer writes them.
    e.bias.copy_(base._bias(seed, kind))
    e.gain.copy_(
        (1 + 0.1 * torch.randn(base.HIDDEN, generator=torch.Generator().manual_seed(seed))).to(
            torch.bfloat16
        )
    )
    e.global_scale.fill_(448.0 * 6.0 / 12.0)


def _layer(
    e, layer: int, rows: int, early, out: dict, tag: str, place=None, static_from=None
) -> None:
    """One MoE layer: late producer -> routed kernel -> persistent experts -> transport finalize.

    The placement is the balanced ring-buddy one unless ``place`` / ``static_from`` say
    otherwise; the three-window placement runs the experts over the 96-expert stack.
    """
    place = e.R.PLACE_BALANCE if place is None else place
    late_producer(
        [
            (e.stage_local[layer, :rows], e.live_local[layer, :rows]),
            (e.stage_logits[layer, :rows], e.live_logits[layer, :rows]),
        ],
        delay_ns=LATE_NS,
    )
    routed = e.R.pool_gather_route(
        e.live_local[layer, :rows],
        e.live_logits[layer, :rows],
        e.inbox,
        e.gain,
        e.global_scale,
        EPS,
        e.bias,
        RENORM_EPS,
        place,
        e.R.LOCAL_EXPERTS if static_from is None else static_from,
        return_gathered=True,
        **({} if early is None else {"early": early}),
    )
    x_sf = routed.scale.reshape(-1).view(torch.float8_e4m3fn)
    s = e.stacked3 if place == e.R.PLACE_BALANCE3 else e.stacked
    experts = e.pe.persistent_experts(
        routed.fp4, x_sf, routed.local_ids, routed.weights_bf16, s["w1"], s["s1"], s["w2"],
        s["s2"], s["scale_c"], s["alpha1"], s["alpha2"],
    )  # fmt: skip
    out[f"{tag}{layer}"] = e.transport.finalize(
        experts[0], experts[2], routed.weights, e.norm, FIN_EPS, rows
    )
    out[f"{tag}{layer}_rows"] = experts[0]
    out[f"{tag}{layer}_index"] = experts[2]
    out[f"{tag}{layer}_ids"] = routed.local_ids
    out[f"{tag}{layer}_fp4"] = routed.fp4


def _chains(e, rows: int, place=None) -> dict:
    out = {}
    for layer in range(LAYERS):
        _layer(e, layer, rows, None, out, "tier3_", place)
    for layer in range(LAYERS):
        _layer(e, layer, rows, e.early, out, "early_ids_", place)
    return out


def _problems(out: dict, rows: int, what: str) -> list:
    found = []
    for layer in range(LAYERS):
        for suffix in ("", "_index", "_ids", "_fp4"):
            a, b = out[f"early_ids_{layer}{suffix}"], out[f"tier3_{layer}{suffix}"]
            differ = int((base._bits(a) != base._bits(b)).sum())
            if differ:
                found.append(f"{what}: layer {layer}{suffix}: {differ}/{a.numel()} differ")
        index = out[f"tier3_{layer}_index"].reshape(-1)
        sel = index[index >= 0].long()
        a = out[f"early_ids_{layer}_rows"][sel].view(torch.int16)
        b = out[f"tier3_{layer}_rows"][sel].view(torch.int16)
        differ = int((a != b).sum())
        if differ:
            found.append(f"{what}: layer {layer} routed rows: {differ} differ")
    return found


def _use_trigger(e, trigger: str) -> None:
    e.early.trigger = e.ei.TRIGGERS[trigger]


@pytest.mark.parametrize("trigger", TRIGGERS)
@pytest.mark.parametrize("kind", KINDS)
@torch.inference_mode()
def test_eager_early_ids_chain_is_bitwise_tier3(env, kind, trigger):
    e = env
    _use_trigger(e, trigger)
    found = []
    fallbacks = e.early.fallback_count()
    launches = e.early_launches
    for step, rows in enumerate(ROWS):
        _set_inputs(e, 1000 * len(kind) + step, rows, kind)
        torch.cuda.synchronize()
        e.comm.Barrier()
        if e.rank == step % WORLD:
            time.sleep(0.002)
        out = _chains(e, rows)
        torch.cuda.synchronize()
        found += _problems(out, rows, f"eager {kind} {trigger} step {step} rows={rows}")
    if e.early_launches - launches != LAYERS * len(ROWS):
        found.append(f"{e.early_launches - launches} early launches, expected {LAYERS * len(ROWS)}")
    fell_back = e.early.fallback_count() - fallbacks
    if trigger == "entry":
        print(f"[early_ids chain] rank {e.rank} eager {kind} entry: {fell_back} polls fell back")
    elif fell_back:
        found.append(f"{fell_back} polls fell back")
    base._agree(e.comm, found, f"eager {kind} {trigger}")


@pytest.mark.parametrize("trigger", ("exchange", "entry"))
@pytest.mark.parametrize("rows", (1, 3, 8))
@torch.inference_mode()
def test_replayed_early_ids_chain_is_bitwise_tier3(env, rows, trigger):
    """One graph of both two-layer chains per row count, replayed with new inputs."""
    e = env
    _use_trigger(e, trigger)
    _set_inputs(e, 1, rows, "random")
    _chains(e, rows)
    torch.cuda.synchronize()
    e.comm.Barrier()
    launches = e.early_launches
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _chains(e, rows)
    torch.cuda.synchronize()
    e.comm.Barrier()
    found = []
    if e.early_launches - launches != LAYERS:
        found.append(f"capture: {e.early_launches - launches} early launches, expected {LAYERS}")
    fallbacks = e.early.fallback_count()
    for replay in range(2 * len(KINDS)):
        kind = KINDS[replay % len(KINDS)]
        _set_inputs(e, 7000 + 10 * rows + replay, rows, kind)
        torch.cuda.synchronize()
        e.comm.Barrier()
        if e.rank == replay % WORLD:
            time.sleep(0.002)
        graph.replay()
        torch.cuda.synchronize()
        found += _problems(out, rows, f"graph rows={rows} {trigger} replay {replay} {kind}")
    if e.early.fallback_count() != fallbacks:
        found.append(f"{e.early.fallback_count() - fallbacks} polls fell back in replays")
    base._agree(e.comm, found, f"graph rows={rows} {trigger}")


def _epoch0(e) -> int:
    device = torch.device("cuda", torch.cuda.current_device())
    num_ctas, options = e.pe._config(device)
    ws = e.pe._workspace(device, num_ctas, options["handoff"])
    return int(ws.epoch[0].item()) & 0xFFFFFFFF


#: (name, placement kind, static_from) of every placement the routed kernel publishes.
PLACES = (
    ("owner", 1, 32),
    ("balance", 2, 32),
    ("static24", 3, 24),
    ("static8", 3, 8),
    ("balance3", 4, 32),
)


@pytest.mark.parametrize("place_name,place,static_from", PLACES)
@pytest.mark.parametrize("kind", ("random", "ties", "extremes"))
@torch.inference_mode()
def test_published_words_are_the_local_ids(env, kind, place_name, place, static_from):
    e = env
    _use_trigger(e, "exchange")
    assert (e.R.PLACE_OWNER, e.R.PLACE_BALANCE, e.R.PLACE_STATIC, e.R.PLACE_BALANCE3) == (
        1,
        2,
        3,
        4,
    )
    assert {p for _, p, _ in PLACES} == set(e.R.EARLY_PLACES)
    found = []
    fallbacks = e.early.fallback_count()
    for step, rows in enumerate(range(1, 9)):
        _set_inputs(e, 5000 + 100 * len(kind) + 10 * place + static_from + step, rows, kind)
        torch.cuda.synchronize()
        e.comm.Barrier()
        out = {}
        for layer in range(LAYERS):
            _layer(e, layer, rows, None, out, "tier3_", place, static_from)
        for layer in range(LAYERS):
            _layer(e, layer, rows, e.early, out, "early_ids_", place, static_from)
            torch.cuda.synchronize()
            words = e.early.words.cpu()
            payload = words & 0xFFFFFFFF
            tags = (words >> 32) & 0xFFFFFFFF
            ids = out[f"early_ids_{layer}_ids"].reshape(-1).cpu().to(torch.int64) & 0xFFFFFFFF
            n = rows * 8
            what = f"{kind} {place_name} rows={rows} layer {layer}"
            if not torch.equal(payload[:n], ids):
                found.append(f"{what}: payload != local_ids ({int((payload[:n] != ids).sum())})")
            if not bool((payload[n:] == 0xFFFFFFFF).all()):
                found.append(f"{what}: rows >= M do not carry -1")
            epoch = _epoch0(e)
            if not bool((tags == epoch).all()):
                found.append(f"{what}: tags {sorted(set(tags.tolist()))} != epoch {epoch}")
        found += _problems(out, rows, f"{kind} {place_name} rows={rows}")
    if e.early.fallback_count() != fallbacks:
        found.append(f"{e.early.fallback_count() - fallbacks} polls fell back")
    base._agree(e.comm, found, f"words {kind} {place_name}")


@pytest.mark.parametrize("place_name", ("balance", "balance3"))
@pytest.mark.parametrize("rows", (1, 8))
@torch.inference_mode()
def test_replays_under_host_skew_are_bitwise_without_fallbacks(env, rows, place_name):
    e = env
    place = {"balance": e.R.PLACE_BALANCE, "balance3": e.R.PLACE_BALANCE3}[place_name]
    _use_trigger(e, "exchange")
    _set_inputs(e, 9000 + rows, rows, "random")
    out = {}
    for layer in range(LAYERS):
        _layer(e, layer, rows, e.early, out, "early_ids_", place)
    torch.cuda.synchronize()
    e.comm.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for layer in range(LAYERS):
            _layer(e, layer, rows, e.early, out, "early_ids_", place)
    torch.cuda.synchronize()
    e.comm.Barrier()
    rng = random.Random(4242 + rows)  # the same draws on every rank
    fallbacks = e.early.fallback_count()
    found = []
    slept = 0.0
    for block in range(40):
        kind = KINDS[block % len(KINDS)]
        _set_inputs(e, 11000 + 100 * rows + block, rows, kind)
        ref = {}
        for layer in range(LAYERS):
            _layer(e, layer, rows, None, ref, "tier3_", place)
        torch.cuda.synchronize()
        e.comm.Barrier()
        for _ in range(50):
            victim = rng.randrange(WORLD)
            delay = rng.choice((0.0, 0.0, 50e-6, 150e-6, 400e-6, 800e-6))
            if e.rank == victim and delay:
                time.sleep(delay)
                slept += delay
            graph.replay()
            torch.cuda.synchronize()
        both = dict(ref)
        both.update(out)
        found += _problems(both, rows, f"skew rows={rows} block {block} {kind}")
    fell_back = e.early.fallback_count() - fallbacks
    print(
        f"[early_ids chain] rank {e.rank} rows {rows} {place_name}: 2000 replays under host skew "
        f"(slept {slept * 1e3:.1f} ms), {fell_back} polls fell back",
        flush=True,
    )
    if fell_back:
        found.append(f"{fell_back} polls fell back under host skew")
    base._agree(e.comm, found, f"skew rows={rows} {place_name}")


@pytest.mark.parametrize("rows", (1, 3, 8))
@torch.inference_mode()
def test_replayed_early_ids_three_windows_chain_is_bitwise_three_windows(env, rows):
    """Three ring windows: one graph of both two-layer chains, replayed with new inputs
    and a delayed rank; the early-started chain equals the chain without early ids."""
    e = env
    place = e.R.PLACE_BALANCE3
    _use_trigger(e, "exchange")
    _set_inputs(e, 13000 + rows, rows, "random")
    _chains(e, rows, place)
    torch.cuda.synchronize()
    e.comm.Barrier()
    launches = e.early_launches
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _chains(e, rows, place)
    torch.cuda.synchronize()
    e.comm.Barrier()
    found = []
    if e.early_launches - launches != LAYERS:
        found.append(f"capture: {e.early_launches - launches} early launches, expected {LAYERS}")
    fallbacks = e.early.fallback_count()
    moved = 0
    for replay in range(4 * len(KINDS)):
        kind = KINDS[replay % len(KINDS)]
        _set_inputs(e, 13100 + 10 * rows + replay, rows, kind)
        torch.cuda.synchronize()
        e.comm.Barrier()
        if e.rank == replay % WORLD:
            time.sleep(0.002)
        graph.replay()
        torch.cuda.synchronize()
        found += _problems(out, rows, f"three-window graph rows={rows} replay {replay} {kind}")
        # Whether the third window is in use: slots running two ranks on (ids 64..95).
        ids = out["early_ids_0_ids"]
        moved += int(((ids >= 2 * BLOCK) & (ids < 3 * BLOCK)).sum())
    if e.early.fallback_count() != fallbacks:
        found.append(f"{e.early.fallback_count() - fallbacks} polls fell back in replays")
    print(
        f"[early_ids chain] rank {e.rank} three windows rows={rows}: {moved} slots ran two ranks on",
        flush=True,
    )
    base._agree(e.comm, found, f"three-window graph rows={rows}")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open pool_gather_route entry (TP8 over two GB300 nodes).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, e.g.
``srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>``;
anywhere else every test skips.

The entry claims the built-in chain's bits on every rank: the full router
logits (the eight ranks' 32-column slices), ``MuseSparkOpenRoutingMethod.apply``'s
selected ids and weights (the fused routing kernel), ``ring_buddy_assign``'s
local ids, BF16 weights and stats for the rank (``replicated_experts``; balanced,
owner-only and static) -- for ``PLACE_BALANCE3`` (``three_window_placement``, no built-in
kernel) ``ring_windows.place``'s on the built-in chain's selection -- and the
built-in pool gather + flashinfer
``rmsnorm_fp4quant``'s payload, block scales and gathered row. Every cell is
driven through CUDA-graph replays with changing inputs and a delayed rank;
constructed near-tie logits, row-count changes with tag wrap-around and the
real built-in router's logits are separate tests. The cells are also gated
against the entry's float64 reference.

``MODELING_V2_TIMING=1`` also runs the microbenchmark of the MoE prologue
chain this op ends (printed, never gated).
"""

import os
import statistics
import time

import pytest
import torch

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.pool_gather_fp4quant import (  # noqa: E501
    decode_nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.pool_gather_fp4quant import (  # noqa: E501
    pool_gather_rmsnorm_fp4quant as pool_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.pool_gather_route import (  # noqa: E501
    EXPERTS,
    HIDDEN,
    LOCAL_COLUMNS,
    LOCAL_EXPERTS,
    PLACE_BALANCE,
    PLACE_BALANCE3,
    PLACE_NONE,
    PLACE_OWNER,
    PLACE_STATIC,
    ROW_WORDS,
    TOP_K,
    PoolGatherRoute,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.pool_gather_route import (  # noqa: E501
    pool_gather_route as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.projection_inbox import (  # noqa: E501
    ProjectionInbox,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
    ring_buddy_assign,
)

__extra_import_path__ = ["../../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

WORLD = 8
#: The checkpoint's pre-expert norm epsilon (rms_norm_eps) and routing
#: renormalization epsilon (router_score_division_eps).
EPS = 1e-5
RENORM_EPS = 1e-15
#: The static placement's threshold (``ring_buddy.STATIC_FROM``).
STATIC_FROM = 28
PLACES = {
    "balance": PLACE_BALANCE,
    "none": PLACE_NONE,
    "owner": PLACE_OWNER,
    "static": PLACE_STATIC,
    "balance3": PLACE_BALANCE3,
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


@pytest.fixture(scope="module")
def group():
    """The built-in gather workspace, this op's 512-word inbox and the pool gather's 256-word one (collective)."""
    comm, rank = _world()
    from tensorrt_llm._torch.models.musespark_open_projection_gather import (
        ProjectionGatherWorkspace,
    )
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    builtin = ProjectionGatherWorkspace(mapping)
    inbox = ProjectionInbox(mapping, row_words=ROW_WORDS)
    pool_inbox = ProjectionInbox(mapping, row_words=LOCAL_COLUMNS // 2)
    comm.Barrier()
    return comm, rank, builtin, inbox, pool_inbox


def _static_from(place: int) -> int:
    return STATIC_FROM if place == PLACE_STATIC else LOCAL_EXPERTS


def _routing(bias: torch.Tensor):
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenRoutingMethod

    return MuseSparkOpenRoutingMethod(
        top_k=TOP_K,
        num_experts=EXPERTS,
        callable_e_score_correction_bias=lambda: bias,
        renorm_eps=RENORM_EPS,
    )


def _local(seed: int, rank: int, rows: int, scale: float = 2.0) -> torch.Tensor:
    """Rank ``rank``'s pool slice under ``seed``; every rank can rebuild every slice."""
    gen = torch.Generator().manual_seed(seed * WORLD + rank)
    return (torch.randn(rows, LOCAL_COLUMNS, generator=gen) * scale).to(torch.bfloat16)


def _logits(seed: int, rows: int, kind: str = "random") -> torch.Tensor:
    """``[rows, 256]`` FP32 logits every rank builds alike; ``kind`` picks the construction.

    ``random``: N(0, 2^2). ``ties``: a third of the experts share one logit.
    ``ulp``: every logit within 3 ulps of 0.75 (near-ties everywhere). ``zeros``:
    30 experts at logit 0 (score 0.5) whose bias -0.5 makes their selection key
    a zero. ``extremes``: +-inf logits and huge magnitudes.
    """
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, EXPERTS, generator=gen) * 2
    if kind == "ties":
        base[:, ::3] = 1.25
    elif kind == "ulp":
        steps = torch.randint(-3, 4, (rows, EXPERTS), generator=gen).float()
        base = 0.75 + steps * 2.0**-24
    elif kind == "zeros":
        base[:, 10:40] = 0.0
    elif kind == "extremes":
        base[0, 5] = float("inf")
        base[min(1, rows - 1), 7] = float("-inf")
        base[:, 100:104] = torch.tensor([1e30, -1e30, 88.0, -88.0])
    return base.float()


def _bias(seed: int, kind: str = "random") -> torch.Tensor:
    gen = torch.Generator().manual_seed(10_000 + seed)
    bias = torch.randn(EXPERTS, generator=gen) * 0.05
    if kind == "ties":
        bias[::3] = 0.0
    elif kind == "ulp":
        bias.zero_()
    elif kind == "zeros":
        bias[10:40] = -0.5
    return bias.float()


def _builtin(workspace, routing, local, logits, gain, global_scale, place, rank):
    """The built-in chain: gather + quantization, routing, the ring-buddy placement."""
    from flashinfer.norm import rmsnorm_fp4quant

    gathered = workspace.gather(local)
    fp4, sf = rmsnorm_fp4quant(
        gathered,
        gain,
        eps=EPS,
        global_scale=global_scale.reshape(1),
        block_size=16,
        scale_format="e4m3",
        is_sf_swizzled_layout=False,
    )
    routing.clear_routing_memo()
    selected, weights = routing.apply(logits)
    assert selected.dtype == torch.int32, "the built-in fused routing kernel did not run"
    out = dict(
        fp4=fp4.view(torch.uint8),
        scale=sf.view(torch.uint8),
        gathered=gathered,
        logits=logits,
        selected=selected,
        weights=weights,
    )
    if place not in (PLACE_NONE, PLACE_BALANCE3):
        ids, weights_bf16, stats = ring_buddy_assign(
            selected, weights, rank, place == PLACE_BALANCE, _static_from(place)
        )
        out.update(local_ids=ids, weights_bf16=weights_bf16, stats=stats)
    return out


def _with_place3(want: dict, place: int, rank: int) -> dict:
    """``want`` plus the three-window placement of its selection (``PLACE_BALANCE3`` only).

    No built-in kernel places over three windows; the rule is
    ``ring_windows.place`` (host, after the chain ran).
    """
    if place != PLACE_BALANCE3:
        return want
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
        windows as ring_windows,
    )

    selected = want["selected"]
    flat, stats = ring_windows.place(selected.reshape(-1).tolist(), rank, 3)
    return dict(
        want,
        local_ids=torch.tensor(flat, dtype=torch.int32, device=selected.device).reshape(
            selected.shape
        ),
        weights_bf16=want["weights"].to(torch.bfloat16),
        stats=torch.tensor(stats, dtype=torch.int32, device=selected.device),
    )


def _fused(inbox, local, logits_local, gain, global_scale, bias, place, return_gathered=True):
    return op(
        local,
        logits_local,
        inbox,
        gain,
        global_scale,
        EPS,
        bias,
        RENORM_EPS,
        place,
        _static_from(place),
        return_gathered=return_gathered,
    )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.element_size() == 4:
        return tensor.view(torch.int32)
    if tensor.element_size() == 2:
        return tensor.view(torch.int16)
    return tensor.view(torch.uint8)


def _same_bits(got, want: dict, what: str) -> list:
    """What differs between the op's outputs and the built-in chain's, as messages."""
    problems = []
    for name, expected in want.items():
        value = getattr(got, name)
        if value is None and name == "gathered":
            continue
        if value is None or value.shape != expected.shape:
            problems.append(f"{what}: {name} is {None if value is None else tuple(value.shape)}")
            continue
        differ = (_bits(value) != _bits(expected)).sum().item()
        if differ:
            first = (_bits(value) != _bits(expected)).nonzero()[0].tolist()
            problems.append(
                f"{what}: {differ} {name} elements differ from the built-in chain, first at "
                f"{first}: {value[tuple(first)].item()!r} vs {expected[tuple(first)].item()!r}"
            )
    if "local_ids" not in want and not (
        got.local_ids is None and got.weights_bf16 is None and got.stats is None
    ):
        problems.append(f"{what}: a placement output without a placement")
    return problems


def _agree(comm, problems: list, what: str) -> None:
    """Every rank raises when any rank found a problem (the ranks stay in step)."""
    everyone = comm.allgather(problems[:4])
    bad = {rank: found for rank, found in enumerate(everyone) if found}
    assert not bad, f"{what}: {bad}"


def _within_reference(got, ref, bias, global_scale, what: str) -> list:
    """The op against the float64 reference.

    Quantization: as ``pool_gather_fp4quant``'s band (at most 0.1 % of the
    dequantized elements and 0.5 % of the block scales differ, each value
    within one code). Routing: the reference ranks float64 sigmoids rounded to
    FP32; the kernel's sigmoid (``ex2.approx``, as the built-in kernel's) is
    within a few ulps of it, so the selected sets agree wherever the
    reference's 8th and 9th ranked keys differ by more than 2^-18 of the 8th,
    and there the weights are within 2^-20 relative of the float64
    renormalization. The logits and the placement of the reference's selection
    are exact. Returns the failed checks as messages.
    """
    try:
        _check_reference(got, ref, bias, global_scale, what)
    except AssertionError as error:
        return [f"{what}: {error}"]
    return []


def _check_reference(got, ref, bias, global_scale, what: str) -> None:
    assert torch.equal(_bits(got.logits), _bits(ref.logits)), what
    values = decode_nvfp4(got.fp4, got.scale, global_scale)
    ref_values = decode_nvfp4(ref.fp4, ref.scale, global_scale)
    assert (values != ref_values).double().mean().item() <= 1e-3, what
    assert (got.scale != ref.scale).double().mean().item() <= 5e-3, what
    block_max = ref_values.abs().reshape(values.shape[0], -1, 16).amax(-1)
    bound = 0.25 * block_max.repeat_interleave(16, dim=1) + 1e-30
    assert ((values - ref_values).abs() <= bound).all(), f"{what}: a value moved by over one code"

    keys = torch.sigmoid(ref.logits.double()).float() + bias.float()
    ranked = keys.sort(dim=1, descending=True).values
    clear = (ranked[:, TOP_K - 1] - ranked[:, TOP_K]).abs() > ranked[:, TOP_K - 1].abs() * 2.0**-18
    for row in clear.nonzero().flatten().tolist():
        assert sorted(got.selected[row].tolist()) == sorted(ref.selected[row].tolist()), (what, row)
        order = [ref.selected[row].tolist().index(e) for e in got.selected[row].tolist()]
        want_w = ref.weights[row, order].double()
        assert ((got.weights[row].double() - want_w).abs() <= want_w.abs() * 2.0**-20).all(), (
            what,
            row,
        )


def _reference(slices, logits, gain, global_scale, bias, place, rank):
    per_rank_logits = [
        logits[:, LOCAL_EXPERTS * r : LOCAL_EXPERTS * (r + 1)].contiguous() for r in range(WORLD)
    ]
    return op.reference(
        [s.cuda() for s in slices],
        per_rank_logits,
        None,
        gain,
        global_scale,
        EPS,
        bias,
        RENORM_EPS,
        place,
        _static_from(place),
        rank=rank,
    )


@pytest.mark.parametrize("cell", PoolGatherRoute.CELLS, ids=lambda c: c.why[:40])
@torch.inference_mode()
def test_cells_match_builtin_bits_and_reference(group, cell, certifying):  # noqa: F811
    comm, rank, builtin, inbox, _ = group
    certifying(op)
    rows, place = cell.spec["rows"], cell.spec["place"]
    torch.manual_seed(41 + rows + 10 * place)
    local = torch.zeros(rows, LOCAL_COLUMNS, device="cuda", dtype=torch.bfloat16)
    logits = torch.zeros(rows, EXPERTS, device="cuda")
    logits_local = torch.zeros(rows, LOCAL_EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    gain = torch.empty(HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
    global_scale = torch.tensor([224.0], device="cuda", dtype=torch.float32)
    routing = _routing(bias)

    def fused(return_gathered=True):
        return _fused(inbox, local, logits_local, gain, global_scale, bias, place, return_gathered)

    _builtin(builtin, routing, local, logits, gain, global_scale, place, rank)
    fused()
    fused(False)
    torch.cuda.synchronize()
    comm.Barrier()
    builtin_graph, fused_graph, lean_graph = (torch.cuda.CUDAGraph() for _ in range(3))
    with torch.cuda.graph(builtin_graph):
        want = _builtin(builtin, routing, local, logits, gain, global_scale, place, rank)
    with torch.cuda.graph(fused_graph):
        got = fused()
    with torch.cuda.graph(lean_graph):
        lean = fused(False)
    assert lean.gathered is None
    torch.cuda.synchronize()
    comm.Barrier()

    problems = []
    for repeat in range(8):
        seed = 3000 * rows + 100 * place + repeat
        scale = (0.05, 1.0, 20.0)[repeat % 3]
        slices = [_local(seed, r, rows, scale) for r in range(WORLD)]
        full = _logits(seed, rows) * (0.5 + repeat)
        local.copy_(slices[rank].cuda())
        logits.copy_(full.cuda())
        logits_local.copy_(logits[:, LOCAL_EXPERTS * rank : LOCAL_EXPERTS * (rank + 1)])
        bias.copy_(_bias(seed).cuda())
        gain.normal_(1, 0.1)
        global_scale.fill_(448.0 * 6.0 / (12.0 * scale))
        torch.cuda.synchronize()
        comm.Barrier()
        if rank == repeat:
            time.sleep(0.002)
        builtin_graph.replay()
        fused_graph.replay()
        lean_graph.replay()
        torch.cuda.synchronize()
        placed = _with_place3(want, place, rank)
        problems += _same_bits(got, placed, f"replay {repeat}")
        problems += _same_bits(lean, placed, f"replay {repeat}, no gathered")
        ref = _reference(slices, logits, gain, global_scale, bias, place, rank)
        problems += _within_reference(got, ref, bias, global_scale, f"replay {repeat} vs reference")
    _agree(comm, problems, f"rows={rows} place={place}")


@pytest.mark.parametrize("kind", ("ties", "ulp", "zeros", "extremes"))
@pytest.mark.parametrize("place", ("balance", "none", "balance3"))
@torch.inference_mode()
def test_near_ties_match_builtin_bits(group, kind, place):
    """Constructed near-ties: the selection key's order and its tie-break by expert id."""
    comm, rank, builtin, inbox, _ = group
    place = PLACES[place]
    gain = torch.empty(HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
    global_scale = torch.tensor([224.0], device="cuda", dtype=torch.float32)
    problems = []
    for trial, rows in enumerate((8, 8, 5, 1, 8, 3)):
        seed = 500 + 10 * trial + len(kind)
        logits = _logits(seed, rows, kind).cuda()
        bias = _bias(seed, kind).cuda()
        local = _local(seed, rank, rows).cuda()
        logits_local = logits[:, LOCAL_EXPERTS * rank : LOCAL_EXPERTS * (rank + 1)].contiguous()
        want = _builtin(builtin, _routing(bias), local, logits, gain, global_scale, place, rank)
        got = _fused(inbox, local, logits_local, gain, global_scale, bias, place)
        torch.cuda.synchronize()
        want = _with_place3(want, place, rank)
        problems += _same_bits(got, want, f"{kind} trial {trial} rows={rows}")
    _agree(comm, problems, f"near ties {kind} place={place}")


@torch.inference_mode()
def test_builtin_router_logits_match(group):
    """Logits of the real built-in router (its served launch) through every placement."""
    comm, rank, builtin, inbox, _ = group
    from tensorrt_llm._torch.models.modeling_musespark_open import _router_projection_kernel

    gen = torch.Generator(device="cuda").manual_seed(77)
    weight = torch.randn(EXPERTS, 8192, generator=gen, device="cuda") * 0.02
    bias = (torch.randn(EXPERTS, generator=gen, device="cuda") * 0.05).contiguous()
    gain = torch.empty(HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
    global_scale = torch.tensor([224.0], device="cuda", dtype=torch.float32)
    problems = []
    for step in range(30):
        rows = (8, 1, 3, 8, 6, 8, 2, 7)[step % 8]
        place = (PLACE_BALANCE, PLACE_NONE, PLACE_OWNER, PLACE_STATIC, PLACE_BALANCE3)[step % 5]
        x = (torch.randn(rows, 8192, generator=gen, device="cuda") * (1 + step)).to(torch.bfloat16)
        logits = torch.empty(rows, EXPERTS, device="cuda")
        tile_rows = 1 if rows in (1, 2, 3, 5) else 2
        _router_projection_kernel[(rows, EXPERTS // tile_rows)](
            x,
            weight,
            logits,
            (8192 / 256.0) ** -0.5,
            EXPERTS=EXPERTS,
            HIDDEN=8192,
            ROWS=tile_rows,
            num_warps=16,
            enable_fp_fusion=False,
        )
        local = _local(900 + step, rank, rows).cuda()
        logits_local = logits[:, LOCAL_EXPERTS * rank : LOCAL_EXPERTS * (rank + 1)].contiguous()
        want = _builtin(builtin, _routing(bias), local, logits, gain, global_scale, place, rank)
        got = _fused(inbox, local, logits_local, gain, global_scale, bias, place, step % 2 == 0)
        torch.cuda.synchronize()
        want = _with_place3(want, place, rank)
        problems += _same_bits(got, want, f"router step {step} rows={rows} place={place}")
    _agree(comm, problems, "built-in router logits")


@torch.inference_mode()
def test_row_count_changes_and_tag_wrap(group):
    comm, rank, builtin, inbox, _ = group
    inbox.counters[:, 0].fill_(-3)
    builtin.counters.fill_(2**32 - 3)
    gain = torch.empty(HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
    global_scale = torch.tensor([224.0], device="cuda", dtype=torch.float32)
    bias = _bias(3).cuda()
    problems = []
    for step, rows in enumerate((1, 8, 3, 8, 1, 2, 8, 1, 8, 5, 8, 8, 3)):
        place = (PLACE_BALANCE, PLACE_NONE, PLACE_BALANCE3)[step % 3]
        logits = _logits(800 + step, rows).cuda()
        local = _local(700 + step, rank, rows).cuda()
        logits_local = logits[:, LOCAL_EXPERTS * rank : LOCAL_EXPERTS * (rank + 1)].contiguous()
        want = _builtin(builtin, _routing(bias), local, logits, gain, global_scale, place, rank)
        if rank == step % WORLD:
            time.sleep(0.001)
        got = _fused(inbox, local, logits_local, gain, global_scale, bias, place, step % 3 != 0)
        torch.cuda.synchronize()
        want = _with_place3(want, place, rank)
        problems += _same_bits(got, want, f"step {step} rows={rows}")
    _agree(comm, problems, "row-count changes and tag wrap")


# ---------------------------------------------------------------------------
# Microbenchmark: MODELING_V2_TIMING=1 (printed, never gated).
# ---------------------------------------------------------------------------

_LAYERS = 62


def _time_graphs(comm, graphs: dict, calls: int, rounds: int = 16):
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


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="MODELING_V2_TIMING=1")
@pytest.mark.parametrize("rows", [8, 1])
@torch.inference_mode()
def test_microbenchmark(group, rows):
    """The decode MoE prologue per layer, 62 layers of distinct weights.

    ``tier1``: the built-in router on the main stream, then the routing and
    the ring-buddy placement; on the preparation stream the pool slice GEMM and the
    fused gather + quantization; joined. ``prologue``: the router shard, the pool
    slice GEMM and this op (routing + balanced placement) on one stream, with
    PDL; ``prologue early`` with the router releasing the GEMM at entry; ``prologue fork``
    with the GEMM on the preparation stream beside the router shard.
    """
    comm, rank, builtin, inbox, pool_inbox = group
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms.dispatcher import (  # noqa: E501
        DEFAULT_KERNELS,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.dense_gemm import (  # noqa: E501
        dense_gemm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.router_shard import (  # noqa: E501
        router_shard,
    )
    from tensorrt_llm._torch.models.modeling_musespark_open import _router_projection_kernel

    pool_kernel = next(kernel for last, kernel in DEFAULT_KERNELS["pool"] if rows <= last)
    gain = torch.empty(HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
    global_scale = torch.tensor([224.0], device="cuda", dtype=torch.float32)
    scale = (8192 / 256.0) ** -0.5
    hidden = (torch.randn(_LAYERS, rows, 8192, device="cuda") * 0.5).to(torch.bfloat16)
    pool_weights = [
        torch.randn(LOCAL_COLUMNS, 8192, device="cuda", dtype=torch.bfloat16) * 0.02
        for _ in range(_LAYERS)
    ]
    router_weights = [torch.randn(EXPERTS, 8192, device="cuda") * 0.01 for _ in range(_LAYERS)]
    bias = (torch.randn(EXPERTS, device="cuda") * 0.05).contiguous()
    routing = _routing(bias)
    aux = torch.cuda.Stream()
    tile_rows = 1 if rows in (1, 2, 3, 5) else 2

    def tier1():
        main = torch.cuda.current_stream()
        for layer in range(_LAYERS):
            aux.wait_stream(main)
            logits = torch.empty(rows, EXPERTS, device="cuda")
            _router_projection_kernel[(rows, EXPERTS // tile_rows)](
                hidden[layer],
                router_weights[layer],
                logits,
                scale,
                EXPERTS=EXPERTS,
                HIDDEN=8192,
                ROWS=tile_rows,
                num_warps=16,
                enable_fp_fusion=False,
            )
            routing.clear_routing_memo()
            selected, weights = routing.apply(logits)
            ring_buddy_assign(selected, weights, rank, True, LOCAL_EXPERTS)
            with torch.cuda.stream(aux):
                pooled = dense_gemm(hidden[layer], pool_weights[layer], pool_kernel, True)
                pool_op(pooled, pool_inbox, gain, global_scale, EPS, return_gathered=True)
            main.wait_stream(aux)

    def moe_prologue_on(
        early: bool, fork: bool, return_gathered: bool = True, place: int = PLACE_BALANCE
    ):
        def body():
            main = torch.cuda.current_stream()
            for layer in range(_LAYERS):
                if fork:
                    aux.wait_stream(main)
                logits_local = router_shard(
                    hidden[layer], router_weights[layer], rank, scale, 16, True, early
                )
                if fork:
                    with torch.cuda.stream(aux):
                        pooled = dense_gemm(hidden[layer], pool_weights[layer], pool_kernel, True)
                    main.wait_stream(aux)
                else:
                    pooled = dense_gemm(hidden[layer], pool_weights[layer], pool_kernel, True)
                _fused(
                    inbox,
                    pooled,
                    logits_local,
                    gain,
                    global_scale,
                    bias,
                    place,
                    return_gathered,
                )

        return body

    def pool_chain_tier1():
        for layer in range(_LAYERS):
            pooled = dense_gemm(hidden[layer], pool_weights[layer], pool_kernel, True)
            pool_op(pooled, pool_inbox, gain, global_scale, EPS, return_gathered=True)

    fixed_logits = torch.randn(rows, LOCAL_EXPERTS, device="cuda")

    def pool_chain_moe_prologue():
        for layer in range(_LAYERS):
            pooled = dense_gemm(hidden[layer], pool_weights[layer], pool_kernel, True)
            _fused(
                inbox,
                pooled,
                fixed_logits,
                gain,
                global_scale,
                bias,
                PLACE_BALANCE,
            )

    # The op alone, 64 calls per graph: the pool gather kernel, this one without the routing
    # (the logit exchange only), with the routing, with the routing and placement.
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.pool_gather_route import (  # noqa: E501
        _compile,
    )

    local = (torch.randn(rows, LOCAL_COLUMNS, device="cuda") * 2).to(torch.bfloat16)
    scratch = [
        torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda"),
        torch.empty(rows, HIDDEN // 2, dtype=torch.uint8, device="cuda"),
        torch.empty(rows, HIDDEN // 16, dtype=torch.uint8, device="cuda"),
        torch.empty(rows, EXPERTS, device="cuda"),
        torch.empty(rows, TOP_K, dtype=torch.int32, device="cuda"),
        torch.empty(rows, TOP_K, device="cuda"),
    ]

    def exchange_only():
        _compile(torch.cuda.current_device(), rank, PLACE_NONE, LOCAL_EXPERTS, True, True, False)(
            scratch[0],
            gain,
            scratch[1],
            scratch[2],
            global_scale,
            rows,
            EPS,
            RENORM_EPS,
            local.data_ptr(),
            inbox.counters.data_ptr(),
            inbox.base,
            inbox.stride,
            fixed_logits.data_ptr(),
            bias.data_ptr(),
            scratch[3].data_ptr(),
            scratch[4].data_ptr(),
            scratch[5].data_ptr(),
            0,
            0,
            0,
        )

    isolated = {}
    for name, body in (
        ("op: pool gather", lambda: pool_op(local, pool_inbox, gain, global_scale, EPS)),
        ("op: exchange only", exchange_only),
        (
            "op: routing",
            lambda: _fused(inbox, local, fixed_logits, gain, global_scale, bias, PLACE_NONE),
        ),
        (
            "op: routing + balance",
            lambda: _fused(inbox, local, fixed_logits, gain, global_scale, bias, PLACE_BALANCE),
        ),
        (
            "op: routing + balance3",
            lambda: _fused(inbox, local, fixed_logits, gain, global_scale, bias, PLACE_BALANCE3),
        ),
    ):
        body()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(64):
                body()
        isolated[name] = graph
        torch.cuda.synchronize()
        comm.Barrier()
    alone = _time_graphs(comm, isolated, calls=64)

    graphs = {}
    for name, body in (
        ("tier1 prologue", tier1),
        ("prologue chain", moe_prologue_on(False, False)),
        ("prologue chain, balance3", moe_prologue_on(False, False, place=PLACE_BALANCE3)),
        ("prologue chain, no gathered row", moe_prologue_on(False, False, False)),
        ("prologue early chain (early trigger)", moe_prologue_on(True, False)),
        ("prologue fork (GEMM beside shard)", moe_prologue_on(False, True)),
        ("tier1 pool chain only", pool_chain_tier1),
        ("prologue pool chain only (routed op)", pool_chain_moe_prologue),
    ):
        body()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        graphs[name] = graph
        torch.cuda.synchronize()
        comm.Barrier()
    per_layer = _time_graphs(comm, graphs, calls=_LAYERS)

    results = comm.gather({**alone, **per_layer}, root=0)
    if rank == 0:
        print(f"\n[moe_prologue bench] rows={rows} (us per layer; median over ranks, min..max)")
        for name in results[0]:
            values = sorted(r[name] for r in results)
            print(
                f"[moe_prologue bench] rows={rows} {name:36s} {statistics.median(values):8.3f} "
                f"({values[0]:.3f}..{values[-1]:.3f})"
            )
    comm.Barrier()

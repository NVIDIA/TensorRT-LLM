# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open pool_gather_fp4quant entry (TP8 over two GB300 nodes).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, e.g.
``srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>``;
anywhere else every test skips.

The entry claims the built-in path's bits: the built-in pool gather
(``ProjectionGatherWorkspace.gather``) followed by flashinfer's
``rmsnorm_fp4quant`` exactly as ``MuseSparkOpenMoE.prepare_expert_input``
calls it. Every cell is driven through CUDA-graph replays with changing inputs
and a delayed rank, with and without the gathered row's store; payload, block
scales and the gathered row must equal the built-in path's as raw bits on
every rank, and the payload is gated against the entry's float64 reference. A
separate test covers row-count changes with tag wrap-around, another a long
burst of mixed row counts. Every test runs with both polling schedules (every
peer each round, and ``stale_poll``: only the peers still stale).

``MODELING_V2_TIMING=1`` also runs the microbenchmark (printed, never gated).
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
    HIDDEN,
    LOCAL_COLUMNS,
    PoolGatherRmsnormFp4Quant,
    decode_nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.pool_gather_fp4quant import (  # noqa: E501
    pool_gather_rmsnorm_fp4quant as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.projection_inbox import (  # noqa: E501
    ProjectionInbox,
)

__extra_import_path__ = ["../../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

WORLD = 8
#: The checkpoint's pre-expert norm epsilon (rms_norm_eps).
EPS = 1e-5


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs eight MPI ranks with CUDA (two four-GPU nodes)")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
        pytest.skip("needs sm_103 or sm_107")
    return comm, rank


@pytest.fixture(scope="module")
def group():
    """One built-in gather workspace and one inbox for the whole module (collective)."""
    comm, rank = _world()
    from tensorrt_llm._torch.models.musespark_open_projection_gather import (
        ProjectionGatherWorkspace,
    )
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    builtin = ProjectionGatherWorkspace(mapping)
    inbox = ProjectionInbox(mapping, row_words=LOCAL_COLUMNS // 2)
    comm.Barrier()
    return comm, rank, builtin, inbox


def _local(seed: int, rank: int, rows: int, scale: float = 2.0) -> torch.Tensor:
    """Rank ``rank``'s pool slice under ``seed``; every rank can rebuild every slice."""
    gen = torch.Generator().manual_seed(seed * WORLD + rank)
    return (torch.randn(rows, LOCAL_COLUMNS, generator=gen) * scale).to(torch.bfloat16)


def _builtin(workspace, local, gain, global_scale):
    """``MuseSparkOpenMoE.prepare_expert_input``'s fused arm on the gathered row."""
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
    return fp4.view(torch.uint8), sf.view(torch.uint8), gathered


def _fused(inbox, local, gain, global_scale, return_gathered=True, stale_poll=False):
    return op(
        local,
        inbox,
        gain,
        global_scale,
        EPS,
        return_gathered=return_gathered,
        stale_poll=stale_poll,
    )


def _assert_same_bits(got, want, what: str):
    for g, w, name in zip(got, want, ("payload", "block scales", "gathered row")):
        if g is None:  # the fused op asked not to write the gathered row
            continue
        assert g.shape == w.shape, (what, name, g.shape, w.shape)
        g_bits = g.view(torch.int16) if g.dtype == torch.bfloat16 else g.view(torch.uint8)
        w_bits = w.view(torch.int16) if w.dtype == torch.bfloat16 else w.view(torch.uint8)
        differ = (g_bits != w_bits).sum().item()
        assert differ == 0, f"{what}: {differ} {name} bytes differ from the built-in path"


def _assert_within_reference(got, ref, global_scale, what: str):
    """The op's payload against the float64 reference.

    The two differ only where the kernel's approximate reciprocals and rsqrt
    (~2^-22 relative) move a value across an E2M1 or E4M3 rounding threshold;
    such a value moves by one code. So: at most 0.1 % of the dequantized
    elements and 0.5 % of the block scales may differ, and every element must
    sit within a quarter of its block's dequantized range of the reference
    (one E2M1 step is at most a third of the block maximum; an E4M3 scale
    rounding adds at most 1/16).
    """
    fp4, sf, gathered = got
    ref_fp4, ref_sf, ref_gathered = ref
    assert torch.equal(gathered.view(torch.int16), ref_gathered.view(torch.int16)), what
    values = decode_nvfp4(fp4, sf, global_scale)
    ref_values = decode_nvfp4(ref_fp4, ref_sf, global_scale)
    element_mismatch = (values != ref_values).double().mean().item()
    scale_mismatch = (sf != ref_sf).double().mean().item()
    assert element_mismatch <= 1e-3, f"{what}: {element_mismatch:.2e} of elements differ"
    assert scale_mismatch <= 5e-3, f"{what}: {scale_mismatch:.2e} of block scales differ"
    block_max = ref_values.abs().reshape(values.shape[0], -1, 16).amax(-1)
    bound = 0.25 * block_max.repeat_interleave(16, dim=1) + 1e-30
    assert ((values - ref_values).abs() <= bound).all(), f"{what}: a value moved by over one code"


@pytest.mark.parametrize("stale_poll", [False, True], ids=["poll_all", "stale_poll"])
@pytest.mark.parametrize("cell", PoolGatherRmsnormFp4Quant.CELLS, ids=lambda c: c.why[:40])
@torch.inference_mode()
def test_cells_match_builtin_bits_and_reference(group, cell, stale_poll, certifying):  # noqa: F811
    comm, rank, builtin, inbox = group
    certifying(op)
    rows = cell.spec["rows"]
    torch.manual_seed(31 + rows)
    local = torch.zeros(rows, LOCAL_COLUMNS, device="cuda", dtype=torch.bfloat16)
    gain = torch.empty(HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
    global_scale = torch.tensor([448.0 * 6.0 / 12.0], device="cuda", dtype=torch.float32)

    _builtin(builtin, local, gain, global_scale)
    _fused(inbox, local, gain, global_scale, stale_poll=stale_poll)
    _fused(inbox, local, gain, global_scale, return_gathered=False, stale_poll=stale_poll)
    torch.cuda.synchronize()
    comm.Barrier()
    builtin_graph, fused_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    lean_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(builtin_graph):
        want = _builtin(builtin, local, gain, global_scale)
    with torch.cuda.graph(fused_graph):
        got = _fused(inbox, local, gain, global_scale, stale_poll=stale_poll)
    with torch.cuda.graph(lean_graph):
        lean = _fused(
            inbox, local, gain, global_scale, return_gathered=False, stale_poll=stale_poll
        )
    assert lean[2] is None
    torch.cuda.synchronize()
    comm.Barrier()

    for repeat in range(8):
        seed = 2000 * rows + repeat
        scale = (0.05, 1.0, 20.0)[repeat % 3]
        slices = [_local(seed, r, rows, scale) for r in range(WORLD)]
        if repeat == 1:
            # Negative zeros, subnormals and the extremes of BF16.
            special = torch.tensor([-32768, 1, 32639, 128, -129], dtype=torch.int16)
            for s in slices:
                s.view(torch.int16)[0, : special.numel()] = special
        local.copy_(slices[rank].cuda())
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
        _assert_same_bits(got, want, f"rows={rows} replay {repeat}")
        _assert_same_bits(lean, want, f"rows={rows} replay {repeat}, no gathered row")
        if repeat == 1:
            continue
        ref = op.reference([s.cuda() for s in slices], None, gain, global_scale, EPS)
        _assert_within_reference(got, ref, global_scale, f"rows={rows} replay {repeat}")
    comm.Barrier()


@pytest.mark.parametrize("stale_poll", [False, True], ids=["poll_all", "stale_poll"])
@torch.inference_mode()
def test_row_count_changes_and_tag_wrap(group, stale_poll):
    comm, rank, builtin, inbox = group
    inbox.counters[:, 0].fill_(-3)
    builtin.counters.fill_(2**32 - 3)
    gain = torch.empty(HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
    global_scale = torch.tensor([224.0], device="cuda", dtype=torch.float32)
    for step, rows in enumerate((1, 8, 3, 8, 1, 2, 8, 1, 8)):
        local = _local(700 + step, rank, rows).cuda()
        want = _builtin(builtin, local, gain, global_scale)
        got = _fused(
            inbox, local, gain, global_scale, return_gathered=step % 2 == 0, stale_poll=stale_poll
        )
        torch.cuda.synchronize()
        _assert_same_bits(got, want, f"step {step} rows={rows}")
    comm.Barrier()


@pytest.mark.parametrize("stale_poll", [False, True], ids=["poll_all", "stale_poll"])
@torch.inference_mode()
def test_a_long_burst_of_mixed_row_counts_stays_identical(group, stale_poll):
    """200 back-to-back calls of 1..8 rows (both schedules interleaved), the built-in between."""
    comm, rank, builtin, inbox = group
    gain = torch.empty(HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
    global_scale = torch.tensor([224.0], device="cuda", dtype=torch.float32)
    g = torch.Generator().manual_seed(4242 + int(stale_poll))
    pairs = []
    torch.cuda.synchronize()
    comm.Barrier()
    for call in range(200):
        rows = int(torch.randint(1, 9, (1,), generator=g))
        local = _local(90_000 + call, rank, rows).cuda()
        schedule = stale_poll if call % 3 else not stale_poll
        got = _fused(inbox, local, gain, global_scale, stale_poll=schedule)
        want = _builtin(builtin, local, gain, global_scale)
        pairs.append((got, want, call, rows))
    torch.cuda.synchronize()
    for got, want, call, rows in pairs:
        _assert_same_bits(got, want, f"burst call {call} rows={rows}")
    comm.Barrier()


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
    """The MoE prologue this op sits in, per layer.

    The preparation stream runs the pool split-K GEMM then gather + quant (or
    the fused op); the main stream runs the built-in FP32 router kernel at its
    in-model launch shape; the layer ends when both are done. Also the pool
    chain alone and the op alone.
    """
    comm, rank, builtin, inbox = group
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
        default_tactic,
        run_splitk_dense,
    )

    from tensorrt_llm._torch.models.modeling_musespark_open import _router_projection_kernel

    gain = torch.empty(HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
    global_scale = torch.tensor([224.0], device="cuda", dtype=torch.float32)
    local = _local(9, rank, rows).cuda()

    isolated = {}
    for name, body in (
        ("builtin gather+quant", lambda: _builtin(builtin, local, gain, global_scale)),
        ("fused", lambda: _fused(inbox, local, gain, global_scale)),
        ("fused, stale poll", lambda: _fused(inbox, local, gain, global_scale, stale_poll=True)),
        ("fused, no gathered row", lambda: _fused(inbox, local, gain, global_scale, False)),
    ):
        body()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(64):
                body()
        isolated[name] = graph
    alone = _time_graphs(comm, isolated, calls=64)

    hidden = (torch.randn(_LAYERS, rows, 8192, device="cuda") * 0.5).to(torch.bfloat16)
    pool_weights = [
        torch.randn(LOCAL_COLUMNS, 8192, device="cuda", dtype=torch.bfloat16) * 0.02
        for _ in range(_LAYERS)
    ]
    router_weights = [torch.randn(256, 8192, device="cuda") * 0.01 for _ in range(_LAYERS)]
    tactic = default_tactic(rows, LOCAL_COLUMNS, 8192)
    pooled = torch.empty(_LAYERS, rows, LOCAL_COLUMNS, device="cuda", dtype=torch.bfloat16)
    logits = torch.empty(_LAYERS, rows, 256, device="cuda")
    aux_default = torch.cuda.Stream()
    # The highest priority torch exposes: blocks of this stream are scheduled
    # ahead of the router's when SM slots free up.
    aux_high = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
    router_rows, router_warps = (1, 16) if rows <= 2 else (2, 16)

    def prologue(
        fused: bool, with_router: bool, aux=aux_default, return_gathered=True, stale_poll=False
    ):
        main = torch.cuda.current_stream()
        for layer in range(_LAYERS):
            aux.wait_stream(main)
            if with_router:
                _router_projection_kernel[(rows, 256 // router_rows)](
                    hidden[layer],
                    router_weights[layer],
                    logits[layer],
                    1.0,
                    EXPERTS=256,
                    HIDDEN=8192,
                    ROWS=router_rows,
                    num_warps=router_warps,
                    enable_fp_fusion=False,
                )
            with torch.cuda.stream(aux):
                projected = run_splitk_dense(
                    hidden[layer], pool_weights[layer].t(), None, pooled[layer], True, tactic
                )
                if fused:
                    _fused(inbox, projected, gain, global_scale, return_gathered, stale_poll)
                else:
                    _builtin(builtin, projected, gain, global_scale)
            main.wait_stream(aux)

    graphs = {}
    for name, fused, with_router, aux, gathered_row, stale in (
        ("builtin pool chain", False, False, aux_default, True, False),
        ("fused pool chain", True, False, aux_default, True, False),
        ("fused pool chain, stale poll", True, False, aux_default, True, True),
        ("fused pool chain, no gathered row", True, False, aux_default, False, False),
        ("builtin prologue (+router)", False, True, aux_default, True, False),
        ("fused prologue (+router)", True, True, aux_default, True, False),
        ("fused prologue (+router), stale poll", True, True, aux_default, True, True),
        ("fused prologue, no gathered row", True, True, aux_default, False, False),
        ("builtin prologue, aux high prio", False, True, aux_high, True, False),
        ("fused prologue, aux high prio", True, True, aux_high, True, False),
    ):
        prologue(fused, with_router, aux, gathered_row, stale)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prologue(fused, with_router, aux, gathered_row, stale)
        graphs[name] = graph
    router_only = torch.cuda.CUDAGraph()
    with torch.cuda.graph(router_only):
        for layer in range(_LAYERS):
            _router_projection_kernel[(rows, 256 // router_rows)](
                hidden[layer],
                router_weights[layer],
                logits[layer],
                1.0,
                EXPERTS=256,
                HIDDEN=8192,
                ROWS=router_rows,
                num_warps=router_warps,
                enable_fp_fusion=False,
            )
    graphs["router only"] = router_only
    per_layer = _time_graphs(comm, graphs, calls=_LAYERS)

    results = comm.gather({**alone, **per_layer}, root=0)
    if rank == 0:
        print(
            f"\n[pool_gather bench] rows={rows} (us per call / per layer; median over ranks, min..max)"
        )
        for name in results[0]:
            values = sorted(r[name] for r in results)
            print(
                f"[pool_gather bench] rows={rows} {name:38s} {statistics.median(values):8.3f} "
                f"({values[0]:.3f}..{values[-1]:.3f})"
            )
    comm.Barrier()

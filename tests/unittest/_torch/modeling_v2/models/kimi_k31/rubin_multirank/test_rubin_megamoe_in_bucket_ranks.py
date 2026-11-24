# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rubin_megamoe_in_bucket's cells over the ranks of one tray, bit for bit against staging.

Each rank runs its rows of every call twice, as dep16's MoE block can: through the staging path
(the ``noaux_tc_op`` top-16, ``fp4_quantize`` and ``rubin_megamoe``, which stages the rows into the
chunk's bucket) and into the bucket (``rubin_route_topk_out`` and ``rubin_megamoe_quantize`` write
the rows there, ``rubin_megamoe_in_bucket`` launches on it as it is). A rank without rows in a call
stages its empty chunk either way, as dep16 does. Every call of a cell runs the staging path
first, then every call the bucket way, so a bucket call follows a bucket call and the tail it reads
is the one the producers reset (made stale on purpose before each). The outputs must be equal bit
for bit, and the bucket must hold the rows and routing the staging path's producers return.

The first cell (DEP16 generation b12 k7: 96 rows per rank, EP4 with all 896 experts, 224 per
rank, bucket 128) also counts the launches of one call (the launcher alone: the MegaMoE kernels;
the staging path: those plus its staging) and replays both chains from CUDA graphs captured after
a longer call left routing in the bucket's tail.

EP16 only (not covered on one tray): 56 experts per rank, the 16-rank exchange across trays, the
kernel instance of world 16 and the symmetric buffers over four trays. The in-bucket path changes
none of them: it changes who writes the bucket, not the kernel or its exchange.
"""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not _SM107, reason="certified on SM107 only"),
]

HIDDEN, INTER, TOP_K, NUM_EXPERTS = 3584, 3072, 16, 896
# The rows per rank of one DEP16 generation step: b12 k7 = 96, in bucket 128.
MAX_NUM_TOKENS = 128
# NVFP4 block scales are E4M3: bytes [0x30, 0x40) are 0.5 .. 1.875 (0x7f is NaN).
SF_BYTES = (0x30, 0x40)
STALE_ID, STALE_WEIGHT = 7, 3.0


def _entries():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
        rubin_megamoe,
        rubin_megamoe_in_bucket,
        rubin_megamoe_quantize,
        rubin_route_topk_out,
    )

    return rubin_megamoe, rubin_megamoe_in_bucket, rubin_megamoe_quantize, rubin_route_topk_out


def _experts(rank, local):
    """This rank's experts as the weight table stores them: random NVFP4 bytes, finite scales."""
    g = torch.Generator(device="cuda").manual_seed(29 + rank)
    u8 = dict(device="cuda", dtype=torch.uint8)
    return (
        torch.randint(0, 256, (local, 2 * INTER, HIDDEN // 2), generator=g, **u8),
        torch.randint(*SF_BYTES, (local, 2 * INTER, HIDDEN // 16), generator=g, **u8),
        torch.randint(0, 256, (local, HIDDEN, INTER // 2), generator=g, **u8),
        torch.randint(*SF_BYTES, (local, HIDDEN, INTER // 16), generator=g, **u8),
        torch.full((local,), 0.03, device="cuda"),
        torch.full((local,), 0.03, device="cuda"),
        torch.ones(local, device="cuda"),
    )


def _call_inputs(rank, n, seed):
    """One call's rows on this rank: the BF16 latent and the FP32 router logits."""
    g = torch.Generator(device="cuda").manual_seed(1000 * seed + rank)
    latent = (torch.randn(n, HIDDEN, generator=g, device="cuda") * 0.15).to(torch.bfloat16)
    logits = torch.randn(n, NUM_EXPERTS, generator=g, device="cuda") * 2
    return latent, logits


def _staged_rows(latent, logits, bias, gs):
    """The staging path's routing and quantization (dep16's): (x4, sf, ids, weights)."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
        fp4_quantize,
    )

    n = latent.shape[0]
    if not n:
        u8 = dict(dtype=torch.uint8, device=latent.device)
        return (
            torch.empty(0, HIDDEN // 2, **u8),
            torch.empty(0, HIDDEN // 16, **u8),
            torch.empty(0, TOP_K, dtype=torch.int32, device=latent.device),
            torch.empty(0, TOP_K, dtype=torch.float32, device=latent.device),
        )
    weights, ids = noaux_tc_op(logits, bias, 1, 1, TOP_K, 1.0)
    x4, sf = fp4_quantize(latent, gs, 16, False, False)
    return x4, sf.view(n, HIDDEN // 16), ids, weights


def _staging_path(latent, logits, bias, gs, experts, ws, chunk_max):
    M, _, _, _ = _entries()
    rows = _staged_rows(latent, logits, bias, gs)
    return M.rubin_megamoe(*rows, *experts, ws, chunk_max, latent.shape[0])


def _fill(latent, logits, bias, gs, ws, chunk_max):
    """The producers write this rank's rows into the chunk's bucket; returns the bucket."""
    M, _, Q, R = _entries()
    n = latent.shape[0]
    b = M.megamoe_bucket(ws, chunk_max, n)
    R.rubin_route_topk_out(
        logits, bias, 1.0, b.topk_idx[:n], b.topk_weights[:n, :TOP_K], route_approx=True
    )
    Q.rubin_megamoe_quantize(latent, gs, b.activation, b.activation_sf, b.topk_idx, b.topk_weights)
    return b


def _in_bucket(latent, logits, bias, gs, experts, ws, chunk_max):
    """The bucket path (dep16's, without its 32-row guard): rows go into the bucket; a rank
    without rows stages its empty chunk."""
    _, B, _, _ = _entries()
    n = latent.shape[0]
    if not n:
        return _staging_path(latent, logits, bias, gs, experts, ws, chunk_max)
    b = _fill(latent, logits, bias, gs, ws, chunk_max)
    return B.rubin_megamoe_in_bucket(b, *experts, ws, n)


def _kernels(comm, call):
    """The CUDA kernels one eager call launches (profiler), the same call on every rank."""
    comm.Barrier()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    comm.Barrier()
    return [e.name for e in prof.events() if e.device_type.name == "CUDA"]


def _capture(comm, call):
    """``call`` captured in a CUDA graph on a side stream after one eager call; (graph, output)."""
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        torch.cuda.synchronize()
        comm.Barrier()
        with torch.cuda.graph(graph, stream=stream):
            out = call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    comm.Barrier()
    return graph, out


def _replay(comm, graph):
    comm.Barrier()
    graph.replay()
    torch.cuda.synchronize()
    comm.Barrier()


def _stale_tail(ws, chunk_max, n):
    """What another call or graph may leave past the rows: routing that still routes."""
    M, _, _, _ = _entries()
    b = M.megamoe_bucket(ws, chunk_max, n)
    b.topk_idx[n:] = STALE_ID
    b.topk_weights[n:, :TOP_K] = STALE_WEIGHT


def _check_bucket(b, n, staged, bad, label):
    x4, sf, ids, weights = staged
    for name, got, want in (
        ("values", b.activation[:n], x4.view(torch.uint8)),
        ("scales", b.activation_sf[:n], sf.view(torch.uint8)),
        ("ids", b.topk_idx[:n], ids.to(torch.int32)),
        (
            "weights",
            b.topk_weights[:n, :TOP_K].view(torch.int32),
            weights.float().view(torch.int32),
        ),
    ):
        if not torch.equal(got, want):
            bad.append(f"{label}: the bucket's {name} are not the staging path's producers' bits")
    if not (bool((b.topk_idx[n:] == -1).all()) and not b.topk_weights[n:, :TOP_K].any()):
        bad.append(f"{label}: the bucket's routing past the rows is not empty")


def _measure(comm, rank, experts, ws, bias, gs, bad):
    """The first cell's launches and CUDA graph replays (rank-local results)."""
    M, B, _, _ = _entries()
    latent, logits = _call_inputs(rank, 96, 101)
    staged = _staged_rows(latent, logits, bias, gs)
    want = M.rubin_megamoe(*staged, *experts, ws, 96, 96).clone()
    launches = {
        "staging": len(_kernels(comm, lambda: M.rubin_megamoe(*staged, *experts, ws, 96, 96)))
    }
    b = _fill(latent, logits, bias, gs, ws, 96)
    launches["in_bucket"] = len(
        _kernels(comm, lambda: B.rubin_megamoe_in_bucket(b, *experts, ws, 96))
    )
    launches["staging_chain"] = len(
        _kernels(comm, lambda: _staging_path(latent, logits, bias, gs, experts, ws, 96))
    )
    launches["in_bucket_chain"] = len(
        _kernels(comm, lambda: _in_bucket(latent, logits, bias, gs, experts, ws, 96))
    )
    g_staging, out_staging = _capture(
        comm, lambda: _staging_path(latent, logits, bias, gs, experts, ws, 96)
    )
    g_bucket, out_bucket = _capture(
        comm, lambda: _in_bucket(latent, logits, bias, gs, experts, ws, 96)
    )
    # A longer call (120 rows, every rank) leaves 24 rows of routing in bucket 128's tail.
    long_latent, long_logits = _call_inputs(rank, 120, 102)
    _in_bucket(long_latent, long_logits, bias, gs, experts, ws, 120)
    torch.cuda.synchronize()
    comm.Barrier()
    _replay(comm, g_bucket)
    if not torch.equal(out_bucket, want):
        bad.append(
            "graph: the bucket chain's replay after a longer call differs from the staging path"
        )
    _check_bucket(b, 96, staged, bad, "graph")
    _replay(comm, g_staging)
    if not torch.equal(out_staging, want):
        bad.append("graph: the staging path's replay differs from its eager call")
    del g_staging, g_bucket
    return launches


def _run_rank(spec, measure):
    from tensorrt_llm.mapping import Mapping

    M, B, _, _ = _entries()
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    bad, launches = [], None
    try:
        local = spec["local"]
        mapping = Mapping(
            world_size=world,
            rank=rank,
            tp_size=world,
            enable_attention_dp=True,
            moe_ep_size=world,
            moe_tp_size=1,
            gpus_per_node=torch.cuda.device_count(),
        )
        ws = M.rubin_megamoe_workspace(
            M.megamoe_ep_group(mapping),
            rank,
            world,
            local,
            HIDDEN,
            INTER,
            MAX_NUM_TOKENS,
            MAX_NUM_TOKENS * world,
            torch.device("cuda"),
            rungs=spec["rungs"],
        )
        experts = _experts(rank, local)
        bias = (torch.rand(NUM_EXPERTS, generator=torch.Generator().manual_seed(5)) - 0.5) * 0.19
        bias = bias.cuda()
        gs = torch.ones(1, device="cuda")  # the routed experts' input scale is 1
        calls = [list(spec["counts"])] if "counts" in spec else [[n] * world for n in spec["rows"]]
        inputs = [_call_inputs(rank, ns[rank], c + 1) for c, ns in enumerate(calls)]
        staging_out = []
        for (latent, logits), ns in zip(inputs, calls):
            out = _staging_path(latent, logits, bias, gs, experts, ws, max(ns))
            torch.cuda.synchronize()
            staging_out.append(out.clone())
        for c, ((latent, logits), ns) in enumerate(zip(inputs, calls)):
            n, chunk_max = ns[rank], max(ns)
            label = f"call {c} ({n} rows, chunk max {chunk_max})"
            comm.Barrier()
            failed = None
            if n:
                # The producers and the guard first; the collective launch only when every rank
                # got there, so a failing rank never leaves the others waiting in the kernel.
                try:
                    _stale_tail(ws, chunk_max, n)
                    b = _fill(latent, logits, bias, gs, ws, chunk_max)
                    torch.cuda.synchronize()
                    _check_bucket(b, n, _staged_rows(latent, logits, bias, gs), bad, label)
                    B.rubin_megamoe_in_bucket.is_valid(b, *experts, ws, n)
                except Exception as exc:  # reported below, on every rank
                    traceback.print_exc()
                    failed = f"{label}: {type(exc).__name__}: {exc}"
            if not all(comm.allgather(failed is None)):
                bad.append(failed or f"{label}: another rank's producers failed; nothing launched")
                continue
            if n:
                out = B.rubin_megamoe_in_bucket(b, *experts, ws, n)
            else:
                out = _staging_path(latent, logits, bias, gs, experts, ws, chunk_max)
            torch.cuda.synchronize()
            if out.shape != staging_out[c].shape or not torch.equal(out, staging_out[c]):
                bad.append(f"{label}: the bucket path's rows differ from the staging path's")
            if n and not bool(torch.isfinite(out).all()):
                bad.append(f"{label}: non-finite rows")
        if measure and all(comm.allgather(not bad)):
            launches = _measure(comm, rank, experts, ws, bias, gs, bad)
    except Exception:
        traceback.print_exc()
        raise
    return dict(rank=rank, bad=bad, launches=launches)


@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_in_bucket_cells_match_the_staging_path_over_four_ranks(mpi_pool_executor):
    """RubinMegamoeInBucket.CELLS (4 ranks, one tray): the bucket path's bits are the staging
    path's."""
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    _, B, _, _ = _entries()
    cells = B.RubinMegamoeInBucket.CELLS
    for i, cell in enumerate(cells):
        results = list(mpi_pool_executor.map(_run_rank, *zip(*[(cell.spec, i == 0)] * world)))
        bad = [f"rank {r['rank']}: {m}" for r in results for m in r["bad"]]
        assert not bad, f"{cell.why}: {bad}"
        if i:
            continue
        launches = [r["launches"] for r in results]
        # The launcher alone: the MegaMoE kernels and no staging (3; the staging path adds six).
        assert all(lc["in_bucket"] == 3 for lc in launches), launches
        assert all(lc["staging"] == lc["in_bucket"] + 6 for lc in launches), launches

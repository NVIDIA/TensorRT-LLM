# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's sharded MoE front kernel (moe_front) on one GB300 tray (four MPI ranks).

Against the replicated front it replaces, on every rank: ``dsv3_router_gemm_op`` FP32 router logits, the BF16
latent_down GEMM and ``kimi_k3_noaux_tc_mxfp8_quant`` (top-16 + MXFP8 of the latent). The front sums the same
products in another order (split over ranks and K slices), so the comparison allows last-bit differences of the
MXFP8 bytes; the top-16 sets must match on these inputs and every rank must return the same bytes (they gather the
same head). With the shared expert's gate_up rows the front must return the same routing / MXFP8 bytes and a SiTU
activation within BF16-level error of the separate GEMM + SituAndMul. A CUDA graph of several calls (both workspace
buffers) replayed on new inputs must return what eager calls return. With ``sort=`` the front's runner routing must
equal a torch runner sort of its top-16 at every row count, eager and in graph replays.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
HIDDEN = 7168
LATENT = 3584
NUM_EXPERTS = 896
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
        or torch.cuda.device_count() < WORLD,
        reason="one GB300 / B200 tray: sm_100 / sm_103 and four GPUs",
    ),
]


def _trt():
    """``torch.ops.trtllm`` looked up by name (cloudpickle ships these functions by value)."""
    return getattr(getattr(torch, "ops"), "trtllm")


def _operands(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    down = (torch.randn(LATENT, HIDDEN, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    router = (torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=gen) * 0.02).to(
        torch.bfloat16
    )
    bias = (torch.rand(NUM_EXPERTS, device="cuda", generator=gen) - 0.5) * 0.19
    return down, router, bias


def _input(m, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(m, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)


def _reference(x, down, router, bias, scale):
    logits = _trt().dsv3_router_gemm_op(x, router.t(), None, torch.float32)
    latent = torch.nn.functional.linear(x, down)
    return _trt().kimi_k3_noaux_tc_mxfp8_quant(logits, bias, latent, scale)


def _compare(got, ref):
    """(hard mismatches, MXFP8 byte mismatches, elements): the top-16 sets per token and the weights where the
    order agrees must match; MXFP8 bytes / scales may differ in the last bit of the latent."""
    ids, weights, codes, scales = got[:4]
    r_ids, r_weights, r_codes, r_scales = ref
    hard = 0
    for t in range(ids.shape[0]):
        hard += len(set(ids[t].tolist()) ^ set(r_ids[t].tolist()))
    same = ids == r_ids
    hard += int(((weights.float() - r_weights.float()).abs() > 1e-2)[same].sum())
    soft = int((codes.view(torch.uint8) != r_codes.view(torch.uint8)).sum())
    soft += int(
        (scales.view(torch.uint8).reshape(-1) != r_scales.view(torch.uint8).reshape(-1)).sum()
    )
    return hard, soft, codes.numel()


def _sort_mismatches(prerouted, ids, offset, num, tile) -> int:
    """Entries of the front's runner routing (``sort=``) that differ from a torch runner sort of its top-16 ``ids``:
    this rank's experts ascending, each expert's rows in ascending token order, ceil(count / tile) tiles each. The
    front leaves the padding rows of ``permuted_idx_to_token_idx`` unwritten (the runner bounds every tile by its
    row limit), so only the filled rows are compared there."""
    e2p, p2t, total, t2e, t2l, nt = prerouted
    rows = ids.shape[0]
    loc = ids.reshape(-1).long() - offset
    x = torch.nonzero((loc >= 0) & (loc < num)).flatten()
    exp = loc[x]
    counts = torch.bincount(exp, minlength=num)[:num]
    tiles = (counts + tile - 1) // tile
    start = (torch.cumsum(tiles, 0) - tiles) * tile
    r_nt = int(tiles.sum())
    order = torch.argsort(exp * (rows * 16) + x)
    xs, es = x[order], exp[order]
    perm = (
        start[es]
        + torch.arange(xs.numel(), device=ids.device)
        - (torch.cumsum(counts, 0) - counts)[es]
    )
    r_e2p = torch.full((rows * 16,), -1, dtype=torch.int32, device=ids.device)
    r_e2p[xs] = perm.int()
    r_p2t = (xs // 16).int()
    r_t2e = torch.repeat_interleave(torch.arange(num, device=ids.device), tiles).int()
    r_t2l = torch.minimum(
        (torch.arange(r_nt, device=ids.device) + 1) * tile, start[r_t2e] + counts[r_t2e]
    ).int()
    bad = int(int(nt[0]) != r_nt) + int(int(total[0]) != r_nt * tile)
    bad += int((e2p != r_e2p).sum())
    if bad == 0:
        bad += int((p2t[perm] != r_p2t).sum())
        bad += int((t2e[:r_nt] != r_t2e).sum()) + int((t2l[:r_nt] != r_t2l).sum())
    return bad


def _run_rank(world_size):
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe import (  # noqa: E501
        moe_front as front,
    )
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    gpus_per_node = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus_per_node)
    try:
        mapping = Mapping(
            world_size=world_size, tp_size=world_size, rank=rank, gpus_per_node=gpus_per_node
        )
        ws = front.head_workspace(mapping)
        down, router, bias = _operands(7)
        weight = front.front_weight(front.head_weight(down, router, rank, world_size))
        scale = 2.5
        hard = soft = elems = 0
        for m in list(range(1, 9)) + [12, 14, 16, 24, 32]:
            x = _input(m, 100 + m)
            got = front.run_moe_front(x, weight, bias, scale, ws)
            torch.cuda.synchronize()
            h, s, n = _compare(got, _reference(x, down, router, bias, scale))
            hard, soft, elems = hard + h, soft + s, elems + n
            blob = (
                torch.cat([t.reshape(-1).view(torch.uint8) for t in got[:4]])
                .cpu()
                .numpy()
                .tobytes()
            )
            hard += sum(b != blob for b in comm.allgather(blob))

        # The shared expert's gate_up + SiTU (this rank's TP shard, up to 16 rows) vs the separate GEMM + SituAndMul.
        from tensorrt_llm._torch.modules.situ import SituAndMul

        inter = 6144 // world_size
        gen = torch.Generator(device="cuda").manual_seed(11 + rank)
        gate_up = (torch.randn(2 * inter, HIDDEN, device="cuda", generator=gen) * 0.02).to(
            torch.bfloat16
        )
        weight_s = front.front_weight(front.head_weight(down, router, rank, world_size), gate_up)
        act = SituAndMul(beta=4.0, linear_beta=25.0)
        for m in (1, 6, 8, 12, 16):
            x = _input(m, 700 + m)
            got = front.run_moe_front(
                x, weight_s, bias, scale, ws, shared_cols=inter, gate_cap=4.0, linear_cap=25.0
            )
            plain = front.run_moe_front(x, weight, bias, scale, ws)
            torch.cuda.synchronize()
            for a, b in zip(got[:4], plain[:4]):
                hard += int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
            ref = act(torch.nn.functional.linear(x, gate_up)).float()
            err = (got[4].float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
            hard += int(err > 2e-2)

        # One CUDA graph with several front calls (both workspace buffers; 24 rows = two token halves in one launch),
        # replayed on new inputs.
        m = 24
        xs = [_input(m, 500 + i) for i in range(3)]
        for x in xs:
            front.run_moe_front(x, weight, bias, scale, ws)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = [front.run_moe_front(x, weight, bias, scale, ws) for x in xs]
        for rep in range(3):
            for i, x in enumerate(xs):
                x.copy_(_input(m, 900 + 10 * rep + i))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for x, got in zip(xs, results):
                eager = front.run_moe_front(x, weight, bias, scale, ws)
                torch.cuda.synchronize()
                for a, b in zip(got[:4], eager[:4]):
                    hard += int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
                h, s, n = _compare(got, _reference(x, down, router, bias, scale))
                hard, soft, elems = hard + h, soft + s, elems + n
        del graph
        torch.cuda.synchronize()

        # The runner routing (sort=) at every row count: one-cluster launches hand the top-16 to the sort in shared
        # memory, the others through the handoff words of the call's buffer; eager, then a graph of several calls
        # (both buffers) replayed on new inputs.
        num, tile = NUM_EXPERTS // world_size, 8
        sort = (tile, rank * num, num, NUM_EXPERTS)
        for m in range(1, 33):
            got = front.run_moe_front(_input(m, 300 + m), weight, bias, scale, ws, sort=sort)
            torch.cuda.synchronize()
            hard += _sort_mismatches(got[5], got[0], rank * num, num, tile)
        for m in (5, 12, 24, 32):
            xs = [_input(m, 1300 + 10 * m + i) for i in range(3)]
            for x in xs:
                front.run_moe_front(x, weight, bias, scale, ws, sort=sort)
            torch.cuda.synchronize()
            comm.Barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                results = [front.run_moe_front(x, weight, bias, scale, ws, sort=sort) for x in xs]
            for rep in range(3):
                for i, x in enumerate(xs):
                    x.copy_(_input(m, 1700 + 100 * m + 10 * rep + i))
                torch.cuda.synchronize()
                comm.Barrier()
                graph.replay()
                torch.cuda.synchronize()
                for got in results:
                    hard += _sort_mismatches(got[5], got[0], rank * num, num, tile)
            del graph
            torch.cuda.synchronize()
        if rank == 0:
            print(
                f"[moe_front] MXFP8 bytes / scales differing from the replicated front: {soft} of {elems}"
            )
        assert comm.allreduce(hard) == 0, f"rank {rank}: {hard} hard mismatches"
        assert soft <= elems // 100, f"rank {rank}: {soft} of {elems} MXFP8 bytes differ"
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_moe_front(mpi_pool_executor):
    results = mpi_pool_executor.map(_run_rank, *zip(*[(WORLD,)] * WORLD))
    for r in results:
        assert r is True


@pytest.mark.parametrize("rows", [8, 72])
def test_routing_weights_are_the_runners(rows: int) -> None:
    """The front's routing (its route CTAs' device code, run alone by ``run_route_probe``) gives the TRTLLM-Gen
    runner's in-kernel routing bit for bit: the runner's top-16 order and its weights ``bf16((s + b - b) / sum)``.
    Ground truth: the NVFP4 runner returns its in-kernel routing weights unfinalized (the same routing code as the
    MXFP4 runner's); the experts are small dummies, only the routing is compared."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe.route_quant import (  # noqa: E501
        run_route_probe,
    )

    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    e, topk, h, inter, loc = NUM_EXPERTS, 16, 2560, 640, 8
    g = torch.Generator(device=dev).manual_seed(rows)
    w1 = torch.randint(0, 256, (loc, 2 * inter, h // 2), dtype=torch.uint8, device=dev, generator=g)
    s1 = torch.full((loc, 2 * inter, h // 16), 1.0, device=dev).to(torch.float8_e4m3fn)
    w2 = torch.randint(0, 256, (loc, h, inter // 2), dtype=torch.uint8, device=dev, generator=g)
    s2 = torch.full((loc, h, inter // 16), 1.0, device=dev).to(torch.float8_e4m3fn)
    ones = torch.ones(loc, dtype=torch.float32, device=dev)
    bias = (torch.randn(e, generator=g, device=dev) * 0.5).float()
    differing = compared = 0
    for _ in range(64_000 // (rows * topk) + 1):
        logits = (torch.randn(rows, e, generator=g, device=dev) * 2.0).float()
        xq = torch.randint(0, 256, (rows, h // 2), dtype=torch.uint8, device=dev, generator=g)
        xsf = torch.full((rows * h // 16,), 1.0, device=dev).to(torch.float8_e4m3fn)
        outs = torch.ops.trtllm.fp4_block_scale_moe_runner(
            logits, bias, xq, xsf, w1, s1, None, None, None, None, w2, s2, None, ones, ones, ones,
            e, topk, 1, 1, inter, 0, loc, 1.0, 2, False, 0,
        )  # fmt: skip
        runner_weights = outs[1].to(torch.bfloat16)
        _, runner_ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, topk, 1.0)
        ids, weights = run_route_probe(logits, bias, 1.0)
        assert torch.equal(ids, runner_ids.to(torch.int32)), "the top-16 order must be the runner's"
        differing += int((weights.view(torch.int16) != runner_weights.view(torch.int16)).sum())
        compared += weights.numel()
    assert differing == 0, f"{differing} of {compared} routing weights differ from the runner's"


@pytest.mark.parametrize("world", [4, 8, 16, 32])
def test_launch_rows_fit_the_gpu(world: int) -> None:
    """A launch serves 32 rows wherever its two token halves of GEMV clusters and the role clusters fit on the GPU
    at once (TP16 / TP32 in head half-tiles, TP8 in 128-row head tiles), else 16, and never asks for more clusters
    than the GPU holds at once."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe import (  # noqa: E501
        moe_front as kernel,
    )

    dev = torch.device("cuda", 0)
    capacity = kernel.max_clusters(dev)
    rows = kernel._chunk_rows(world, 0, HIDDEN, kernel.DEFAULT_RING, 0)
    cfg = kernel._launch_config(rows, HIDDEN, world, 0, kernel.DEFAULT_RING, dev)
    clusters, mma_n, halves = cfg[4], cfg[9], cfg[10]
    assert clusters * halves + kernel.role_clusters_of(mma_n * halves) <= capacity
    if capacity >= 15 and world >= 8:
        assert rows == 32, f"world {world}: {rows} rows per launch on {capacity} clusters"

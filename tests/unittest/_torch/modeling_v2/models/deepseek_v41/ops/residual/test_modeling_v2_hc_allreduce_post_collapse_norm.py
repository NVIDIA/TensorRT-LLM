# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-rank test of the hc_allreduce_post_collapse_norm entry: the fused launch against the engine's one-shot
`AllReduce` followed by `hc_post_collapse_norm`, bit for bit (the kernel's sum is the library kernel's: fp32 in
rank order, one bf16 rounding), eager and replayed from a CUDA graph, for a bf16 partial and for an unfinalized
MoE partial (exact inputs, so any finalize summation order gives the same bits).

Needs `world` visible GPUs of one tray; run under the repo's `mpi_pool_executor` fixture.
"""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_allreduce_post_collapse_norm import (  # noqa: E501
    hc_allreduce_post_collapse_norm,
)
from tensorrt_llm._torch.modules.mhc.mhc_lowm import LamportAllReduce, MoeUnfinalized
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 4
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
        reason="needs 4 SM100/SM103/SM107 GPUs",
    ),
]

D, EPS = 5120, 1e-6


def _operands(n, seed, rank, moe):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    res = torch.randn(n, 4, D, generator=gen, device="cuda").bfloat16()
    post = 2 * torch.sigmoid(torch.randn(n, 4, generator=gen, device="cuda"))
    comb = torch.softmax(torch.randn(n, 4, 4, generator=gen, device="cuda"), -1)
    pre = torch.sigmoid(torch.randn(n, 4, generator=gen, device="cuda")) + 1e-6
    w = (1 + 0.1 * torch.randn(D, generator=gen, device="cuda")).bfloat16()
    gx = torch.Generator(device="cuda").manual_seed(seed * 100 + rank + 1)
    x = torch.randn(n, D, generator=gx, device="cuda").bfloat16()
    x[:, :16] = 0.0
    x[:, 16:32] = -0.0  # -0.0 is the Lamport "not yet written" marker
    if moe:
        rows, k = n * 6 + 7, 6
        x = torch.randint(-64, 64, (n, D), generator=gx, device="cuda").bfloat16() / 16
        fc2 = torch.randint(-128, 128, (rows, D), generator=gx, device="cuda").bfloat16() / 16
        perm = (
            torch.randperm(rows, generator=gx, device="cuda")[: n * k]
            .view(n, k)
            .to(torch.int32)
            .contiguous()
        )
        # -1: the expert of that slot is on another rank; token 0 has none here
        perm[torch.rand(n, k, generator=gx, device="cuda") < 0.25] = -1
        perm[0] = -1
        scales = 2.0 ** torch.randint(-2, 2, (n, k), generator=gx, device="cuda").float()
        x = MoeUnfinalized(shared=x, fc2=fc2, scales=scales, perm=perm)
    return x, res, post, comb, pre, w, EPS


def _rank_body(world, n, moe):
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post_collapse_norm import (  # noqa: E501
        hc_post_collapse_norm,
    )

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world, tp_size=world, rank=rank)
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.ONESHOT, dtype=torch.bfloat16)
        lamport = LamportAllReduce(get_allreduce_workspace(mapping), rank, world)
        lamport.reference_sum = lambda t: ar(t)

        def fused(*a, **k):
            return hc_allreduce_post_collapse_norm(*a, lamport, **k)

        def ref(x, *a):
            return hc_post_collapse_norm(ar(x.finalize() if moe else x), *a)

        args = _operands(n, 3, rank, moe)
        hc_allreduce_post_collapse_norm.is_valid(*args, lamport)
        for g, w in zip(fused(*args), ref(*args)):
            torch.testing.assert_close(g, w, rtol=0, atol=0)
        # the op's own reference (float64 post/collapse/norm) within a band
        got, want = fused(*args), hc_allreduce_post_collapse_norm.reference(*args, lamport)
        torch.testing.assert_close(got[0].float(), want[0].float(), rtol=2e-2, atol=2e-2)
        # the NVFP4 copy of `h` (stock's FFN boundary ahead of the routed experts): the same launch's rows and scales
        # are `fp4_quantize(h, scale, 16, False, False)` bit for bit, and the other outputs do not change
        scale = (448.0 * 6.0 / ref(*args)[1].float().abs().max()).reshape(1).float()
        res_q, h_q, (fp4, sf) = fused(*args, nvfp4_scale=scale)
        for g, w in zip((res_q, h_q), ref(*args)):
            torch.testing.assert_close(g, w, rtol=0, atol=0)
        want_q, want_sf = torch.ops.trtllm.fp4_quantize(h_q, scale, 16, False, False)
        assert torch.equal(fp4.view(torch.uint8), want_q.view(torch.uint8))
        assert torch.equal(sf.view(torch.uint8), want_sf.reshape(-1).view(torch.uint8))
        if moe:
            # the permutation as the tep8 target hands it over: row 0 of a later tile's routing table (a view at
            # tile * 5 * width int32 entries into the table buffer), at every tile index the [3, 5, width] layout has
            for tile in range(3):
                width = -(-(args[0].perm.numel() + 1) // 4) * 4
                table = torch.full((3, 5, width), -7, dtype=torch.int32, device="cuda")
                table[tile, 0, : args[0].perm.numel()] = args[0].perm.reshape(-1)
                perm = table[tile, 0][: args[0].perm.numel()].view(args[0].perm.shape)
                viewed = MoeUnfinalized(
                    shared=args[0].shared, fc2=args[0].fc2, scales=args[0].scales, perm=perm
                )
                hc_allreduce_post_collapse_norm.is_valid(viewed, *args[1:], lamport)
                for g, w in zip(fused(viewed, *args[1:]), ref(*args)):
                    torch.testing.assert_close(g, w, rtol=0, atol=0)
        # graph replay on new data
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fused(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fused(*args)
        for step in range(10):
            # new partials in the captured buffers before each replay (rank- and step-specific), so a replay that
            # read an earlier round's Lamport payload would not match
            gx = torch.Generator(device="cuda").manual_seed(1000 * (step + 1) + rank)
            if moe:
                args[0].shared.copy_(
                    torch.randint(-64, 64, args[0].shared.shape, generator=gx, device="cuda") / 16
                )
                args[0].fc2.copy_(
                    torch.randint(-128, 128, args[0].fc2.shape, generator=gx, device="cuda") / 16
                )
            else:
                args[0].copy_(torch.randn(args[0].shape, generator=gx, device="cuda"))
            graph.replay()
            torch.cuda.synchronize()
            for g, w in zip(out, ref(*args)):
                torch.testing.assert_close(g, w, rtol=0, atol=0)
    except Exception:
        traceback.print_exc()
        raise
    return True


# Every certified cell is 4 ranks of one tray; the pool is sized to the cells' world.
_WORLDS = {c.spec["world"] for c in hc_allreduce_post_collapse_norm.CELLS}
assert _WORLDS == {4}, _WORLDS


@pytest.mark.parametrize(
    "cell",
    hc_allreduce_post_collapse_norm.CELLS,
    ids=[c.why[:40] for c in hc_allreduce_post_collapse_norm.CELLS],
)
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_certified_cells(mpi_pool_executor, cell):
    """The fused launch over each cell, bit for bit against the engine's all-reduce + `hc_post_collapse_norm`."""
    s = cell.spec
    world = mpi_pool_executor.num_workers
    assert world == s["world"] and s["d"] == D
    results = mpi_pool_executor.map(_rank_body, *zip(*[(world, s["n"], s["moe"])] * world))
    for r in results:
        assert r is True

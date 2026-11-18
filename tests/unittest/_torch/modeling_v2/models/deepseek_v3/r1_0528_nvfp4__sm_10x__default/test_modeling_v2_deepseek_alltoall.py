# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek-R1 default target's routed experts under wide expert parallelism, against the
attention-DP round trip they replace, on 4 GPUs of one NVLink domain (a GB200 or GB300 tray).

`_experts_alltoall` (each rank routes its own tokens and exchanges them) and the allgather
round trip (every rank routes every rank's tokens through `_experts`, and the ranks' partial
sums add up) run the same router and the same expert kernels over the same tokens, so they
differ only in where each token's rows are computed and in the order of the final sum. Every
rank builds the same layer from one seed, and its own tokens from its rank.

The layer is the R1 routed geometry (256 experts in 8 groups, top-8, hidden 7168,
intermediate 2048) at EP 4: the trunk's NVFP4 experts through the trtllm-gen runner, and the
MTP layer's bf16 experts through `fused_moe`.
"""

import pickle
import sys
from pathlib import Path
from typing import Iterator

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)

WORLD = 4
EXPERTS, HIDDEN, INTER, TOP_K = 256, 7168, 2048, 8
#: tests/unittest/_torch/modeling_v2, where the FP4 MoE test helpers live.
_HELPERS = str(Path(__file__).resolve().parents[3])


def _worker(case: dict) -> dict:
    """One rank: the allgather round trip and the alltoall over the same tokens."""
    import sys
    from types import SimpleNamespace

    import torch

    sys.path.insert(0, case["helpers"])
    from _fp4_moe_test_utils import _build, _calibrate_g2, _rand_activation, _scalars

    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
    from tensorrt_llm._torch.modeling_v2.catalog.comm.moe_alltoall import moe_alltoall_workspace
    from tensorrt_llm._torch.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v3.r1_0528_nvfp4__sm_10x__default import (  # noqa: E501
        modeling as T,
    )
    from tensorrt_llm.mapping import Mapping

    rank = MPI.COMM_WORLD.Get_rank()
    torch.cuda.set_device(rank)
    MnnvlMemory.initialize()
    if not all(MPI.COMM_WORLD.allgather(MnnvlMemory.supports_mnnvl())):
        return {"rank": rank, "skip": "the alltoall needs multi-node NVLink on every rank"}

    counts, cap, layer = list(case["counts"]), case["cap"], case["layer"]
    local = EXPERTS // WORLD
    window = slice(rank * local, (rank + 1) * local)
    cfg = SimpleNamespace(
        n_group=8, topk_group=4, num_experts_per_tok=TOP_K, routed_scaling_factor=2.5
    )
    dims = SimpleNamespace(
        num_slots=EXPERTS,
        moe_inter=INTER,
        slot_offset=rank * local,
        local_slots=local,
        moe_tp=1,
        moe_tp_rank=0,
        moe_ep=WORLD,
        moe_ep_rank=rank,
    )
    gen = torch.Generator(device="cuda").manual_seed(1234)
    router = (torch.randn(EXPERTS, HIDDEN, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    bias = (torch.randn(EXPERTS, device="cuda", generator=gen) * 0.01).float()
    g1 = 137.0
    xs = []
    for r, n in enumerate(counts):
        g = torch.Generator(device="cuda").manual_seed(4321 + r)
        xs.append(_rand_activation(n, HIDDEN, g1, g)[2])
    x_all = torch.cat(xs).to(torch.bfloat16)
    offsets = [sum(counts[:r]) for r in range(WORLD)]
    mine = slice(offsets[rank], offsets[rank] + counts[rank])

    mapping = Mapping(
        rank=rank, world_size=WORLD, tp_size=WORLD, moe_ep_size=WORLD, enable_attention_dp=True
    )
    payload = "NVFP4" if layer == "trunk" else "BF16"
    a2a = moe_alltoall_workspace(
        mapping,
        num_slots=EXPERTS,
        top_k=TOP_K,
        max_tokens_per_rank=cap,
        hidden_size=HIDDEN,
        payload=payload,
    )
    core = SimpleNamespace(
        model_config=SimpleNamespace(pretrained_config=cfg),
        dims=dims,
        rank=rank,
        alltoall=T._Alltoall(
            trunk=a2a if layer == "trunk" else None, mtp=a2a if layer == "mtp" else None
        ),
    )
    if layer == "trunk":
        args, ref = _build(EXPERTS, HIDDEN, INTER, seed=2048)
        _, ids_all = noaux_tc_op(x_all @ router.t(), bias, 8, 4, TOP_K, 2.5)
        g2 = _calibrate_g2(x_all.float(), ids_all, ref)
        s1, sg, s2 = _scalars(local, g1, g2)
        weights = SimpleNamespace(
            router=router.t(),
            router_bias=bias,
            moe_gscale=torch.tensor([g1], dtype=torch.float32, device="cuda"),
            fc1_w=args["gemm1_weights"][window].contiguous(),
            fc1_s=args["gemm1_weights_scale"][window].contiguous(),
            fc2_w=args["gemm2_weights"][window].contiguous(),
            fc2_s=args["gemm2_weights_scale"][window].contiguous(),
            out1_scale=s1,
            out1_gate_scale=sg,
            out2_scale=s2,
            eplb=None,
        )
        gathered = T._experts(core, weights, x_all, None)
        exchange = T._experts_alltoall
    else:
        g = torch.Generator(device="cuda").manual_seed(99)
        weights = SimpleNamespace(
            router=router.t(),
            router_bias=bias,
            fc1=(torch.randn(local, 2 * INTER, HIDDEN, device="cuda", generator=g) * 0.02).to(
                torch.bfloat16
            ),
            fc2=(torch.randn(local, HIDDEN, INTER, device="cuda", generator=g) * 0.02).to(
                torch.bfloat16
            ),
            eplb=None,
        )
        gathered = T._mtp_experts(core, weights, x_all, None)
        exchange = T._mtp_experts_alltoall

    # The allgather round trip: every rank's partial sums over every token, summed.
    total = torch.from_numpy(MPI.COMM_WORLD.allreduce(gathered.float().cpu().numpy(), op=MPI.SUM))
    want = total[mine.start : mine.stop]

    x_mine = x_all[mine].contiguous()
    got = exchange(core, weights, x_mine, tuple(counts)).float().cpu()
    result = {"rank": rank, "rows": got.shape[0], "want_rows": want.shape[0]}
    result["err"] = (got - want).abs().max().item() if got.numel() else 0.0
    result["scale"] = want.abs().max().item() if want.numel() else 1.0

    if case["graph"]:
        # Captured once, replayed on other tokens of the same counts.
        static_x = x_mine.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                exchange(core, weights, static_x, tuple(counts))
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = exchange(core, weights, static_x, tuple(counts))
        errs = []
        for trial in range(2):
            g = torch.Generator(device="cuda").manual_seed(777 + 10 * trial + rank)
            fresh = _rand_activation(counts[rank], HIDDEN, g1, g)[2].to(torch.bfloat16)
            static_x.copy_(fresh)
            graph.replay()
            replayed = static_out.float().cpu()
            eager = exchange(core, weights, fresh, tuple(counts)).float().cpu()
            errs.append((replayed - eager).abs().max().item() if eager.numel() else 0.0)
        result["graph_err"] = max(errs)
        del graph
    del core, a2a
    return result


@pytest.fixture(scope="module")
def pool() -> Iterator[MPIPoolExecutor]:
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"needs {WORLD} GPUs")
    executor = MPIPoolExecutor(
        WORLD,
        env={"TRTLLM_MOE_COMM_TIMEOUT_SEC": "120", "TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC": "300"},
    )
    try:
        yield executor
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _run(pool: MPIPoolExecutor, **case) -> list[dict]:
    case["helpers"] = _HELPERS
    results = sorted(pool.map(_worker, [case] * WORLD), key=lambda r: r["rank"])
    skipped = [r["skip"] for r in results if "skip" in r]
    if skipped:
        assert len(skipped) == WORLD, "platform support differs across ranks"
        pytest.skip(skipped[0])
    for r in results:
        assert r["rows"] == r["want_rows"], r
    return results


#: Every rank's token count: decode, uneven with a rank of no tokens, a prefill-like count
#: that takes several rounds of `cap` rows, and a round whose received rows (4 x 2600) take
#: more than one expert-kernel call.
COUNTS = {
    "decode": ((1, 1, 1, 1), 64),
    "uneven": ((37, 0, 64, 5), 64),
    "rounds": ((300, 17, 0, 129), 64),
    "large": ((2500, 2600, 2400, 2500), 4096),
}


@pytest.mark.no_xdist
@pytest.mark.parametrize("layer", ["trunk", "mtp"])
@pytest.mark.parametrize("counts", list(COUNTS), ids=list(COUNTS))
def test_the_alltoall_matches_the_allgather_round_trip(pool, layer, counts) -> None:
    """Each rank's tokens get the routed output the allgather round trip gives them: the same
    rows through the same kernels, summed in another order, so within a few bf16 ulps of the
    largest value -- a token missing a slot's share, or given one twice, is off by a whole
    expert's output."""
    rows, cap = COUNTS[counts]
    for r in _run(pool, counts=rows, cap=cap, layer=layer, graph=False):
        assert r["err"] <= 2**-5 * max(r["scale"], 1e-3), r


@pytest.mark.no_xdist
@pytest.mark.parametrize("layer", ["trunk", "mtp"])
def test_a_captured_alltoall_replays_like_an_eager_one(pool, layer) -> None:
    """Decode runs under CUDA graphs: a captured round trip replayed on new tokens gives what
    the eager one gives on them."""
    for r in _run(pool, counts=(8, 8, 8, 8), cap=64, layer=layer, graph=True):
        assert r["err"] <= 2**-5 * max(r["scale"], 1e-3), r
        assert r["graph_err"] <= 2**-5 * max(r["scale"], 1e-3), r

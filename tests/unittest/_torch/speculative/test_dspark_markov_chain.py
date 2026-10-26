# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused DSpark Markov chain over a vocabulary-sharded draft head (one launch, TP exchange in-kernel).

Weights are multiples of 1/16 in [-2/16, 2/16], so every Markov bias dot product is exact in FP32 in any
summation order and the reference (float64 dot -> FP32 -> BF16, FP32 logit addition, NaN-first and
smallest-id argmax) must match bit for bit. The chain's dedicated exchange workspace must not interact
with the model's all-reduce workspace: CUDA-graph replays mix batch sizes (so consecutive launches clear
different extents), interleave library one-shot all-reduces and cross the launch-tag wrap, every value
checked.
"""

import pickle
import platform
import random
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

VOCAB, RANK_DIM, BLOCK, HIDDEN = 129280, 256, 5, 5120


def _make_inputs(batch, seed, vocab):
    """Full [B, K, vocab] logits and [B] bonus ids (identical on every rank), with ties, NaN, -0.0
    and out-of-vocabulary bonus ids."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(batch, BLOCK, vocab, generator=gen, device="cuda") * 3
    for b in range(batch):
        for k in range(BLOCK):
            mode = (seed + 7 * b + k) % 5
            row = base[b, k]
            if mode == 1:
                j = int(row.argmax())
                row[(j + vocab // 4 + 17) % vocab] = row[j]
            elif mode == 2:
                row[(seed * 31 + b) % vocab] = float("nan")
                row[(seed * 57 + 5 * b + 3) % vocab] = float("nan")
            elif mode == 3:
                row.mul_(1e-3)
            elif mode == 4:
                row.clamp_(max=0.0)
                row[(seed * 13 + b) % vocab] = -0.0
    bonus = torch.randint(0, vocab, (batch,), generator=gen, device="cuda")
    if seed % 3 == 0:
        bonus[0] = -1
    if seed % 3 == 1 and batch > 1:
        bonus[batch - 1] = vocab + 5
    return base.bfloat16(), bonus


def _reference(base, bonus, w1, w2_double):
    vocab = w1.shape[0]
    prev = bonus.clone()
    out = torch.empty(base.shape[0], BLOCK, dtype=torch.int64, device="cuda")
    for k in range(BLOCK):
        valid = (prev >= 0) & (prev < vocab)
        latent = torch.where(
            valid[:, None], w1[prev.clamp(0, vocab - 1)].double(), torch.zeros_like(w2_double[:1])
        )
        logits = base[:, k].float() + (latent @ w2_double.T).float().bfloat16().float()
        nan = torch.isnan(logits)
        best = torch.where(nan, torch.full_like(logits, float("-inf")), logits).max(1).values
        first_max = (logits == best[:, None]).int().argmax(1)
        tok = torch.where(nan.any(1), nan.int().argmax(1), first_max)
        out[:, k] = tok
        prev = tok
    return out


@torch.inference_mode()
def _run(tp_size: int, fabric: bool) -> bool:
    from tensorrt_llm._torch.speculative.dspark_markov_chain import (
        MarkovWorkspace,
        markov_chain_fused,
        markov_chain_supported,
    )

    rank = tensorrt_llm.mpi_rank()
    gpus_per_node = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus_per_node)
    try:
        mapping = Mapping(
            world_size=tp_size, tp_size=tp_size, rank=rank, gpus_per_node=gpus_per_node
        )
        workspace = MarkovWorkspace(mapping, fabric=fabric)
        # The flag word holds the launch epoch: start close to the launch-tag wrap (2^14 - 1).
        workspace._buffers[3][2] = (1 << 14) - 30
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        # The model's one-shot all-reduces: IPC within a node, MNNVL across nodes.
        strategy = AllReduceStrategy.MNNVL if mapping.is_multi_node() else AllReduceStrategy.ONESHOT
        allreduce = AllReduce(mapping=mapping, strategy=strategy, dtype=torch.bfloat16)
        gen = torch.Generator(device="cuda").manual_seed(1234)
        w1 = (torch.randint(-2, 3, (VOCAB, RANK_DIM), generator=gen, device="cuda") / 16).bfloat16()
        w2 = (torch.randint(-2, 3, (VOCAB, RANK_DIM), generator=gen, device="cuda") / 16).bfloat16()
        w2_double = w2.double()
        local = VOCAB // tp_size
        offset = rank * local
        w2_shard = w2[offset : offset + local].contiguous()

        def fused(base, bonus):
            shard = base[:, :, offset : offset + local].contiguous()
            assert markov_chain_supported(shard, w2_shard, w1, workspace)
            return markov_chain_fused(shard, w2_shard, w1, bonus, offset, VOCAB, workspace)

        for batch in (1, 2, 4, 8):
            for trial in range(3):
                base, bonus = _make_inputs(batch, 100 * batch + trial, VOCAB)
                got = fused(base, bonus).long()
                torch.testing.assert_close(
                    got, _reference(base, bonus, w1, w2_double), rtol=0, atol=0
                )

        graphs = {}
        for batch in (1, 2, 4, 8):
            shard = torch.zeros(batch, BLOCK, local, dtype=torch.bfloat16, device="cuda")
            bonus = torch.zeros(batch, dtype=torch.int64, device="cuda")
            x1 = torch.zeros(6 * batch, HIDDEN, dtype=torch.bfloat16, device="cuda")
            x2 = torch.zeros(3 * batch, HIDDEN, dtype=torch.bfloat16, device="cuda")

            def body(shard=shard, bonus=bonus, x1=x1, x2=x2):
                return (
                    allreduce(x1),
                    markov_chain_fused(shard, w2_shard, w1, bonus, offset, VOCAB, workspace),
                    allreduce(x2),
                )

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                body()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = body()
            graphs[batch] = (graph, shard, bonus, x1, x2, outs)

        order = random.Random(99)  # same sequence on every rank
        for it in range(60):
            batch = order.choice(sorted(graphs))
            graph, shard, bonus, x1, x2, (y1, tokens, y2) = graphs[batch]
            base, bonus_ids = _make_inputs(batch, 1000 + 37 * it + batch, VOCAB)
            shard.copy_(base[:, :, offset : offset + local])
            bonus.copy_(bonus_ids)
            gen_x = torch.Generator(device="cuda").manual_seed(it)
            xs1 = [
                torch.randn(x1.shape, generator=gen_x, device="cuda").bfloat16()
                for _ in range(tp_size)
            ]
            xs2 = [
                torch.randn(x2.shape, generator=gen_x, device="cuda").bfloat16()
                for _ in range(tp_size)
            ]
            x1.copy_(xs1[rank])
            x2.copy_(xs2[rank])
            graph.replay()
            torch.testing.assert_close(
                tokens.long(), _reference(base, bonus_ids, w1, w2_double), rtol=0, atol=0
            )
            sum1, sum2 = xs1[0].float(), xs2[0].float()
            for r in range(1, tp_size):
                sum1 = sum1 + xs1[r].float()
                sum2 = sum2 + xs2[r].float()
            torch.testing.assert_close(y1, sum1.bfloat16(), rtol=0, atol=0)
            torch.testing.assert_close(y2, sum2.bfloat16(), rtol=0, atol=0)
    except Exception:
        traceback.print_exc()
        raise
    return True


def _fabric_supported() -> bool:
    from tensorrt_llm._mnnvl_utils import MnnvlMemory

    return platform.machine().lower() in ("aarch64", "arm64") and MnnvlMemory.supports_mnnvl()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 4
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs 4 SM100/SM103 GPUs",
)
@pytest.mark.parametrize("fabric", [False, True], ids=["ipc", "fabric"])
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_dspark_markov_chain_matches_reference(mpi_pool_executor, fabric):
    if fabric and not _fabric_supported():
        pytest.skip("needs fabric-mapped (MNNVL) memory")
    tp_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_run, *zip(*[(tp_size, fabric)] * tp_size))
    for r in results:
        assert r is True

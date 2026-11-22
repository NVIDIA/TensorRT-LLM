# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The DSpark draft head split by vocabulary over a node-local attention-DP group proposes exactly the
replicated head's tokens, eager and from CUDA-graph replays, with ties across slices and NaN rows."""

import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from tensorrt_llm._torch.models.modeling_dspark import CompactTPArgmax
from tensorrt_llm._torch.models.modeling_speculative import VanillaMarkov
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.speculative.group_sharded_draft_head import GroupShardedDraftHead
from tensorrt_llm._utils import get_sm_version, mpi_allgather, mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_WORLD = 4
VOCAB, HIDDEN, MARKOV_RANK, BLOCK = 129280, 1024, 256, 5


def _init_worker(paths: list[str]) -> None:
    sys.path[:] = paths


@torch.inference_mode()
def _run() -> list[str]:
    rank = mpi_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    mapping = Mapping(
        world_size=_WORLD,
        rank=rank,
        gpus_per_node=_WORLD,
        tp_size=_WORLD,
        enable_attention_dp=True,
        enable_lm_head_tp_in_adp=True,
    )
    # The same head and Markov weights on every rank, as attention DP replicates them.
    gen = torch.Generator(device="cuda").manual_seed(1234)
    weight = (torch.randn(VOCAB, HIDDEN, generator=gen, device="cuda") * 0.05).bfloat16()
    head = LMHead(
        VOCAB,
        HIDDEN,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
    )
    head.load_weights([dict(weight=weight)])
    head = head.cuda()
    markov = VanillaMarkov(vocab_size=VOCAB, markov_rank=MARKOV_RANK).cuda()
    markov.markov_w1.weight.data = (
        torch.randn(VOCAB, MARKOV_RANK, generator=gen, device="cuda") * 0.1
    ).bfloat16()
    markov.markov_w2.weight.data = (
        torch.randn(VOCAB, MARKOV_RANK, generator=gen, device="cuda") * 0.05
    ).bfloat16()
    # Exact ties across slices (a head row and its Markov row copied): the smaller id must win.
    tied = VOCAB - 3
    head.weight.data[tied] = head.weight.data[7]
    markov.markov_w2.weight.data[tied] = markov.markov_w2.weight.data[7]
    replicated, split = CompactTPArgmax(), CompactTPArgmax()
    split.group = GroupShardedDraftHead.create(mapping, VOCAB, HIDDEN, 25 * BLOCK)
    problems = [] if split.group is not None else ["no group head"]

    def inputs(requests, seed):
        g = torch.Generator(device="cuda").manual_seed(seed + 7919 * rank)
        x = torch.randn(requests, BLOCK, HIDDEN, generator=g, device="cuda").bfloat16()
        if seed % 3 == 0:
            x[0, seed % BLOCK] = head.weight[7] * 40
        if seed % 4 == 1:
            x[requests - 1, 2] = float("nan")
        bonus = torch.randint(-1, VOCAB + 2, (requests,), generator=g, device="cuda")
        return x, bonus

    def propose(model, x, bonus):
        return model.try_propose(
            head,
            markov,
            x,
            bonus,
            block_size=BLOCK,
            temperature=0.0,
            return_logits=False,
            all_rank_num_tokens=[x.shape[0] * BLOCK] * _WORLD,
        )

    # Every member runs every call with the same rows before any assertion.
    for step in range(30):
        x, bonus = inputs((1, 4, 25)[step % 3], 1000 + step)
        want, got = propose(replicated, x, bonus), propose(split, x, bonus)
        if got is None or not torch.equal(got, want):
            problems.append(f"eager step {step} differs")

    graphs = {}
    for requests in (1, 25):
        x, bonus = inputs(requests, 3000 + requests)
        propose(split, x, bonus)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = propose(split, x, bonus)
        graphs[requests] = (graph, x, bonus, out)
    for step in range(30):
        graph, x, bonus, got = graphs[(1, 25)[step % 2]]
        new_x, new_bonus = inputs(x.shape[0], 4000 + step)
        x.copy_(new_x)
        bonus.copy_(new_bonus)
        graph.replay()
        if not torch.equal(got, propose(replicated, new_x, new_bonus)):
            problems.append(f"replay {step} differs")
    torch.cuda.synchronize()
    return mpi_allgather(problems)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD,
    reason=f"Requires {_WORLD} GPUs",
)
@pytest.mark.skipif(get_sm_version() not in (100, 103), reason="MXFP8 draft head on SM100/SM103")
def test_group_split_draft_head_matches_replicated() -> None:
    if mpi_world_size() > 1:
        pytest.skip("Spawns its own worker processes")
    with MPIPoolExecutor(
        max_workers=_WORLD, initializer=_init_worker, initargs=(sys.path,)
    ) as pool:
        results = [f.result() for f in [pool.submit(_run) for _ in range(_WORLD)]]
    problems = [p for per_rank in results[0] for p in per_rank]
    assert not problems, problems

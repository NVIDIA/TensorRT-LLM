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
"""Four-rank GB300 finalize/IPC parity, including mutated CUDA graph inputs.

Run this file with four MPI ranks and one local GPU per rank.
"""

import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.models.modeling_musespark_open import post_expert_norm_combine
from tensorrt_llm._torch.models.musespark_open_finalize_ar import HAS_CUTEDSL, FinalizeIpcWorkspace
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping


@pytest.mark.parametrize("tokens", [1, 2, 8, 16, 32, 80, 128])
@pytest.mark.parametrize("case", ["normal", "upper", "empty_rank", "zero_rows"])
@torch.inference_mode()
def test_finalize_allreduce_graph_parity(tokens: int, case: str) -> None:
    if MPI.COMM_WORLD.Get_size() != 4 or not torch.cuda.is_available() or not HAS_CUTEDSL:
        pytest.skip("Requires four local MPI ranks, CUDA and CuTe DSL")
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("GB300 required")
    mapping = Mapping(world_size=4, tp_size=4, rank=rank, gpus_per_node=4)
    workspace = FinalizeIpcWorkspace(mapping)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    torch.manual_seed(1270 + tokens)
    weights = torch.rand(tokens, 8, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    norm = torch.randn(4096, device="cuda", dtype=torch.bfloat16) * 0.03
    torch.manual_seed(1270 + rank)
    source = torch.randn(max(64, tokens * 8), 4096, device="cuda", dtype=torch.bfloat16)
    gemm = source.clone()
    indices = torch.arange(tokens * 8, device="cuda", dtype=torch.int32).reshape(tokens, 8)
    indices = torch.where(indices % 4 == rank, indices, -1).contiguous()
    if case == "upper":
        indices[0, rank] = 10000
    if case == "empty_rank" and rank == 3:
        indices.fill_(-1)
        gemm.fill_(float("nan"))
    if case == "zero_rows" and rank == 3:
        indices.zero_()
        gemm = torch.empty(0, 4096, device="cuda", dtype=torch.bfloat16)

    def reference() -> torch.Tensor:
        return nccl(
            post_expert_norm_combine(
                gemm,
                indices,
                weights,
                norm,
                1e-8,
                num_tokens=tokens,
                top_k=8,
                hidden_size=4096,
            ).to(torch.bfloat16)
        )

    for _ in range(3):
        workspace.finalize(gemm, indices, weights, norm, 1e-8, tokens)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5):
            actual = workspace.finalize(gemm, indices, weights, norm, 1e-8, tokens)
    for repeat in range(3):
        if not (case in ("empty_rank", "zero_rows") and rank == 3):
            gemm.copy_(source * (-1 if repeat % 2 else 1) + 0.1 * repeat)
        expected = reference()
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.04, atol=0.02)
        assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.01
        for peer in MPI.COMM_WORLD.allgather(actual.cpu().float().numpy()):
            torch.testing.assert_close(actual.cpu().float(), torch.from_numpy(peer), rtol=0, atol=0)
    del graph
    MPI.COMM_WORLD.Barrier()

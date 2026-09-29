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
"""GB300 finalize parity, including mutated CUDA graph inputs.

Run with four local MPI ranks, or eight ranks across two NVLink-connected
four-GPU nodes to exercise the experimental fabric workspace.
"""

import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.models import musespark_open_finalize_ar
from tensorrt_llm._torch.models.modeling_musespark_open import post_expert_norm_combine
from tensorrt_llm._torch.models.musespark_open_finalize_ar import HAS_CUTEDSL, FinalizeIpcWorkspace
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping


@pytest.mark.parametrize("tokens", [1, 2, 8, 16, 32, 80, 115, 122, 123, 128])
@pytest.mark.parametrize("case", ["normal", "upper", "empty_rank", "zero_rows"])
@torch.inference_mode()
def test_finalize_allreduce_graph_parity(tokens: int, case: str) -> None:
    world_size = MPI.COMM_WORLD.Get_size()
    if world_size not in (4, 8) or not torch.cuda.is_available() or not HAS_CUTEDSL:
        pytest.skip("Requires four or eight MPI ranks, CUDA and CuTe DSL")
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("GB300 required")
    mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank, gpus_per_node=4)
    workspace = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    torch.manual_seed(1270 + tokens)
    weights = torch.rand(tokens, 8, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    norm = torch.randn(4096, device="cuda", dtype=torch.bfloat16) * 0.03
    torch.manual_seed(1270 + rank)
    source = torch.randn(max(64, tokens * 8), 4096, device="cuda", dtype=torch.bfloat16)
    gemm = source.clone()
    indices = torch.arange(tokens * 8, device="cuda", dtype=torch.int32).reshape(tokens, 8)
    indices = torch.where(indices % world_size == rank, indices, -1).contiguous()
    if case == "upper":
        indices[0, rank] = 10000
    if case == "empty_rank" and rank == world_size - 1:
        indices.fill_(-1)
        gemm.fill_(float("nan"))
    if case == "zero_rows" and rank == world_size - 1:
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
        if not (case in ("empty_rank", "zero_rows") and rank == world_size - 1):
            gemm.copy_(source * (-1 if repeat % 2 else 1) + 0.1 * repeat)
        expected = reference()
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.04, atol=0.02)
        # TP8 can mask every expert in the one-token upper-bound case. Require
        # exact zero error there instead of dividing two zero norms.
        assert (actual.float() - expected.float()).norm() <= 0.01 * expected.float().norm()
        for peer in MPI.COMM_WORLD.allgather(actual.cpu().float().numpy()):
            torch.testing.assert_close(actual.cpu().float(), torch.from_numpy(peer), rtol=0, atol=0)
    del graph
    MPI.COMM_WORLD.Barrier()


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 8), (2, 16), (8, 64), (16, 0)),
        ((20, 160), (24, 3), (32, 0)),
        ((40, 320), (48, 7), (64, 0)),
        ((80, 640), (123, 129), (122, 0), (115, 71), (128, 1024), (80, 8)),
    ],
)
@torch.inference_mode()
def test_finalize_reuses_compilation_across_mixed_shapes(
    shapes: tuple[tuple[int, int], ...],
) -> None:
    """New mixed-batch sizes and per-rank expert row counts must not compile."""
    world_size = MPI.COMM_WORLD.Get_size()
    if world_size not in (4, 8) or not torch.cuda.is_available() or not HAS_CUTEDSL:
        pytest.skip("Requires four or eight MPI ranks, CUDA and CuTe DSL")
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("GB300 required")
    mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank, gpus_per_node=4)
    workspace = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    compiled_keys = None
    # Each group retains its launch cluster size while token and expert-row
    # counts vary. Different launch cluster sizes must remain specialized.
    cluster_sizes = {
        next(size for size in (8, 4, 2, 1) if tokens * size <= workspace.controller_count)
        for tokens, _ in shapes
    }
    assert len(cluster_sizes) == 1
    for tokens, rows in shapes:
        torch.manual_seed(1729 + tokens)
        weights = torch.rand(tokens, 8, device="cuda")
        weights /= weights.sum(-1, keepdim=True)
        norm = torch.randn(4096, device="cuda", dtype=torch.bfloat16) * 0.03
        rank_rows = max(0, rows - rank)
        torch.manual_seed(1729 + rank)
        gemm = torch.randn(rank_rows, 4096, device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(tokens * 8, device="cuda", dtype=torch.int32).reshape(tokens, 8)
        indices = torch.where(indices % world_size == rank, indices, -1).contiguous()
        expected = nccl(
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
        actual = workspace.finalize(gemm, indices, weights, norm, 1e-8, tokens)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.04, atol=0.02)
        keys = set(musespark_open_finalize_ar._compile_cache)
        if compiled_keys is None:
            compiled_keys = keys
        else:
            assert keys == compiled_keys, "A new mixed shape triggered kernel compilation"
    MPI.COMM_WORLD.Barrier()

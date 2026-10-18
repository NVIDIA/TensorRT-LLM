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
"""``FinalizeIpcWorkspace.finalize_chunked``: batches above 128 tokens as 128-token calls.

Run with four local MPI ranks (TP4 IPC workspace), or eight ranks across two
NVLink-connected four-GPU nodes (fabric workspace). On SM107 the workspace
needs ``MUSESPARK_OPEN_SM107_FUSED_KERNELS=1``.

* At up to 128 tokens ``finalize_chunked`` is ``finalize``, bitwise.
* Above 128 tokens every row is bitwise the row the matching single
  ``finalize`` call over its 128-token chunk returns, and within the existing
  fused-vs-unfused gates of the NCCL reference.
* Under CUDA-graph replay with mutated inputs it equals the eager call bitwise.
"""

import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.models.modeling_musespark_open import (
    fused_kernels_supported,
    post_expert_norm_combine,
)
from tensorrt_llm._torch.models.musespark_open_finalize_ar import HAS_CUTEDSL, FinalizeIpcWorkspace
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

# The NCCL reference all-reduce starts communicator threads that outlive each test.
pytestmark = pytest.mark.threadleak(enabled=False)

#: Decode rows at MBS 8..64 (eight rows per request), and uneven chunk tails.
TOKENS = [1, 64, 127, 128, 129, 200, 256, 383, 384, 511, 512]


def _require_group():
    world_size = MPI.COMM_WORLD.Get_size()
    if world_size not in (4, 8) or not torch.cuda.is_available() or not HAS_CUTEDSL:
        pytest.skip("Requires four or eight MPI ranks, CUDA and CuTe DSL")
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank % 4)
    if not fused_kernels_supported(torch.cuda.get_device_capability()):
        pytest.skip("Requires SM103, or SM107 with MUSESPARK_OPEN_SM107_FUSED_KERNELS=1")
    mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank, gpus_per_node=4)
    return world_size, rank, mapping


def _inputs(tokens: int, rank: int, world_size: int, seed: int):
    """Routing shared by all ranks; every ``world_size``-th slot is local to ``rank``."""
    shared = torch.Generator(device="cuda").manual_seed(seed)
    weights = torch.rand(tokens, 8, device="cuda", generator=shared)
    weights /= weights.sum(-1, keepdim=True)
    norm = (torch.randn(4096, device="cuda", generator=shared) * 0.03).to(torch.bfloat16)
    local = torch.Generator(device="cuda").manual_seed(seed * 1009 + rank)
    gemm = torch.randn(tokens * 8, 4096, device="cuda", generator=local).to(torch.bfloat16)
    indices = torch.arange(tokens * 8, device="cuda", dtype=torch.int32).reshape(tokens, 8)
    indices = torch.where(indices % world_size == rank, indices, -1).contiguous()
    return gemm, indices, weights.contiguous(), norm


def _reference(nccl, gemm, indices, weights, norm, eps, tokens):
    partial = post_expert_norm_combine(
        gemm, indices, weights, norm, eps, num_tokens=tokens, top_k=8, hidden_size=4096
    )
    return nccl(partial.to(torch.bfloat16))


def _within_gates(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    actual, expected = actual.float(), expected.float()
    if not torch.allclose(actual, expected, rtol=0.04, atol=0.02):
        return False
    return bool((actual - expected).norm() <= 0.01 * expected.norm())


def _same_on_all_ranks(tensor: torch.Tensor) -> bool:
    gathered = MPI.COMM_WORLD.allgather(tensor.view(torch.int16).cpu().numpy())
    return all((peer == gathered[0]).all() for peer in gathered)


def _all_ranks(ok: bool) -> bool:
    return all(MPI.COMM_WORLD.allgather(bool(ok)))


def _per_chunk(workspace, gemm, indices, weights, norm, eps, tokens):
    """The same rows as explicit ``finalize`` calls of at most 128 tokens."""
    return torch.cat(
        [
            workspace.finalize(
                gemm,
                indices[a : a + 128].contiguous(),
                weights[a : a + 128].contiguous(),
                norm,
                eps,
                min(128, tokens - a),
            )
            for a in range(0, tokens, 128)
        ]
    )


@pytest.mark.parametrize("tokens", TOKENS)
@torch.inference_mode()
def test_chunked_rows_are_the_fused_calls_rows(tokens: int) -> None:
    world_size, rank, mapping = _require_group()
    workspace = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    ok = True
    for seed in range(3):
        gemm, indices, weights, norm = _inputs(tokens, rank, world_size, 4100 + 10 * tokens + seed)
        actual = workspace.finalize_chunked(gemm, indices, weights, norm, 1e-6, tokens)
        ok &= tuple(actual.shape) == (tokens, 4096)
        ok &= torch.equal(actual, _per_chunk(workspace, gemm, indices, weights, norm, 1e-6, tokens))
        if tokens <= 128:
            ok &= torch.equal(actual, workspace.finalize(gemm, indices, weights, norm, 1e-6, tokens))
        ok &= _within_gates(actual, _reference(nccl, gemm, indices, weights, norm, 1e-6, tokens))
        ok &= _same_on_all_ranks(actual)
    workspace.close()
    assert _all_ranks(ok)
    MPI.COMM_WORLD.Barrier()


@pytest.mark.parametrize("tokens", [200, 256, 512])
@torch.inference_mode()
def test_chunked_graph_replays_track_every_input(tokens: int) -> None:
    world_size, rank, mapping = _require_group()
    workspace = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    gemm, indices, weights, norm = _inputs(tokens, rank, world_size, 9000 + tokens)
    for _ in range(2):
        workspace.finalize_chunked(gemm, indices, weights, norm, 1e-6, tokens)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Interleave a small call, as consecutive layers of different batch shapes would.
        small = workspace.finalize(gemm, indices[:8], weights[:8], norm, 1e-6, 8)
        actual = workspace.finalize_chunked(gemm, indices, weights, norm, 1e-6, tokens)
    ok = True
    for repeat in range(4):
        fresh = _inputs(tokens, rank, world_size, 9100 + 10 * tokens + repeat)
        gemm.copy_(fresh[0])
        indices.copy_(torch.roll(fresh[1], shifts=repeat, dims=1))
        weights.copy_(fresh[2])
        norm.copy_(fresh[3])
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        graph.replay()
        torch.cuda.synchronize()
        replayed, replayed_small = actual.clone(), small.clone()
        eager = workspace.finalize_chunked(gemm, indices, weights, norm, 1e-6, tokens)
        eager_small = workspace.finalize(
            gemm, indices[:8].contiguous(), weights[:8].contiguous(), norm, 1e-6, 8
        )
        ok &= torch.equal(replayed, eager) and torch.equal(replayed_small, eager_small)
        ok &= _within_gates(replayed, _reference(nccl, gemm, indices, weights, norm, 1e-6, tokens))
        ok &= _same_on_all_ranks(replayed)
    del graph
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    workspace.close()
    assert _all_ranks(ok)
    MPI.COMM_WORLD.Barrier()


@torch.inference_mode()
def test_chunked_rejects_malformed_calls_before_launch() -> None:
    world_size, rank, mapping = _require_group()
    workspace = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    gemm, indices, weights, norm = _inputs(256, rank, world_size, 77)
    rejected = []
    for call in (
        (gemm, indices, weights, norm, 1e-6, 0),
        (gemm, indices, weights, norm, 1e-6, 255),
        (gemm, indices[:, :4].contiguous(), weights, norm, 1e-6, 256),
        (gemm, indices.long(), weights, norm, 1e-6, 256),
        (gemm.float(), indices, weights, norm, 1e-6, 256),
    ):
        try:
            workspace.finalize_chunked(*call)
        except ValueError:
            rejected.append(True)
        else:
            rejected.append(False)
    ok = all(rejected)
    actual = workspace.finalize_chunked(gemm, indices, weights, norm, 1e-6, 256)
    ok &= _same_on_all_ranks(actual)
    workspace.close()
    assert _all_ranks(ok), f"rejected per malformed call: {rejected}"
    MPI.COMM_WORLD.Barrier()

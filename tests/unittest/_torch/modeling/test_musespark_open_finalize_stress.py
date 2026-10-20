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
"""Long and skewed call sequences for the finalize+all-reduce workspace.

Run with four local MPI ranks, or eight ranks across two NVLink-connected
four-GPU nodes. Every call carries its own inputs, so a read of stale or
not-yet-written data shows up as a mismatch against the NCCL reference.
"""

import random
import time

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


def _reference(nccl, gemm, indices, weights, norm, tokens):
    partial = post_expert_norm_combine(
        gemm, indices, weights, norm, 1e-8, num_tokens=tokens, top_k=8, hidden_size=4096
    )
    return nccl(partial.to(torch.bfloat16))


def _within_gates(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    actual, expected = actual.float(), expected.float()
    if not torch.allclose(actual, expected, rtol=0.04, atol=0.02):
        return False
    return bool((actual - expected).norm() <= 0.01 * expected.norm())


def _all_ranks(ok: bool) -> bool:
    return all(MPI.COMM_WORLD.allgather(bool(ok)))


def _inputs(tokens: int, rank: int, world_size: int, seed: int, rows=None):
    shared = torch.Generator(device="cuda").manual_seed(seed)
    weights = torch.rand(tokens, 8, device="cuda", generator=shared)
    weights /= weights.sum(-1, keepdim=True)
    norm = (torch.randn(4096, device="cuda", generator=shared) * 0.03).to(torch.bfloat16)
    local = torch.Generator(device="cuda").manual_seed(seed * 1009 + rank)
    rows = max(64, tokens * 8) if rows is None else rows
    gemm = torch.randn(rows, 4096, device="cuda", generator=local).to(torch.bfloat16)
    indices = torch.arange(tokens * 8, device="cuda", dtype=torch.int32).reshape(tokens, 8)
    indices = torch.where(indices % world_size == rank, indices, -1).contiguous()
    return gemm, indices, weights, norm


def _refill(call, tokens, rank, world_size, seed):
    """Overwrite a captured call's inputs in place with a fresh payload."""
    fresh = _inputs(tokens, rank, world_size, seed, rows=call[0].shape[0])
    for target, source in zip(call, fresh):
        target.copy_(source)


def _capture(workspace, calls):
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [workspace.finalize(g, i, w, n, 1e-8, t) for (g, i, w, n), t in calls]
    return graph, outputs


@torch.inference_mode()
def test_finalize_random_call_sequences() -> None:
    """Eager calls and three graphs (5, 7 and 1 calls) interleaved in a seeded order."""
    world_size, rank, mapping = _require_group()
    workspace = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    schedule = random.Random(2026)  # identical on every rank
    graphs = []
    for count in (5, 7, 1):
        tokens = [schedule.randint(1, 128) for _ in range(count)]
        calls = [
            (_inputs(t, rank, world_size, 5000 + 100 * len(graphs) + i), t)
            for i, t in enumerate(tokens)
        ]
        for (gemm, indices, weights, norm), t in calls:
            workspace.finalize(gemm, indices, weights, norm, 1e-8, t)
        graphs.append((calls, *_capture(workspace, calls)))
    # Start with abrupt 128 <-> 1 switches and ramps, then random traffic.
    plan = [128, 1] * 8 + list(range(1, 129, 9)) + list(range(128, 0, -11))
    ok = True
    for step in range(1500):
        local = random.Random(step * 31 + rank)  # per-rank row counts may differ
        if step < len(plan) or schedule.random() < 0.7:
            tokens = plan[step] if step < len(plan) else schedule.randint(1, 128)
            rows = local.choice((0, 1, 5, None))
            gemm, indices, weights, norm = _inputs(tokens, rank, world_size, 10_000 + step, rows)
            actual = workspace.finalize(gemm, indices, weights, norm, 1e-8, tokens)
            ok &= _within_gates(actual, _reference(nccl, gemm, indices, weights, norm, tokens))
        else:
            calls, graph, outputs = graphs[schedule.randrange(len(graphs))]
            for index, (call, tokens) in enumerate(calls):
                _refill(call, tokens, rank, world_size, 20_000 + 10 * step + index)
            expected = [_reference(nccl, *call, tokens) for call, tokens in calls]
            graph.replay()
            for actual, want in zip(outputs, expected):
                ok &= _within_gates(actual, want)
    torch.cuda.synchronize()
    del graphs
    assert _all_ranks(ok)
    MPI.COMM_WORLD.Barrier()


@pytest.mark.parametrize("pattern", ["always_slow", "rotating", "random"])
@torch.inference_mode()
def test_finalize_with_delayed_ranks(pattern: str) -> None:
    """A lagging rank must never let the others reuse a buffer it has not drained."""
    world_size, rank, mapping = _require_group()
    workspace = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    schedule = random.Random(77)
    ok = True
    for step in range(200):
        tokens = schedule.randint(1, 128)
        slow = {
            "always_slow": 0,
            "rotating": step % world_size,
            "random": schedule.randrange(world_size),
        }[pattern]
        cycles = int(schedule.uniform(0, 4e6))
        host_pause = schedule.random() < 0.1
        gemm, indices, weights, norm = _inputs(tokens, rank, world_size, 30_000 + step)
        expected = _reference(nccl, gemm, indices, weights, norm, tokens)
        if rank == slow:
            torch.cuda._sleep(cycles)
            if host_pause:
                time.sleep(0.002)
        actual = workspace.finalize(gemm, indices, weights, norm, 1e-8, tokens)
        ok &= _within_gates(actual, expected)
    # The same skew inside a captured graph: the slow rank sleeps before every call.
    calls = [(_inputs(t, rank, world_size, 40_000 + t), t) for t in (3, 128, 17)]
    for (gemm, indices, weights, norm), t in calls:
        workspace.finalize(gemm, indices, weights, norm, 1e-8, t)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = []
        for (gemm, indices, weights, norm), t in calls:
            if rank == 0:
                torch.cuda._sleep(1_000_000)
            outputs.append(workspace.finalize(gemm, indices, weights, norm, 1e-8, t))
    for repeat in range(30):
        for index, (call, tokens) in enumerate(calls):
            _refill(call, tokens, rank, world_size, 50_000 + 10 * repeat + index)
        expected = [_reference(nccl, *call, tokens) for call, tokens in calls]
        graph.replay()
        for actual, want in zip(outputs, expected):
            ok &= _within_gates(actual, want)
    torch.cuda.synchronize()
    del graph
    assert _all_ranks(ok)
    MPI.COMM_WORLD.Barrier()


@torch.inference_mode()
def test_finalize_two_workspaces_interleave() -> None:
    """Independent workspaces share no state, eagerly or inside one graph."""
    world_size, rank, mapping = _require_group()
    first = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    second = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    schedule = random.Random(5)
    ok = True
    for step in range(120):
        workspace = first if step % 3 else second
        tokens = schedule.randint(1, 128)
        gemm, indices, weights, norm = _inputs(tokens, rank, world_size, 60_000 + step)
        actual = workspace.finalize(gemm, indices, weights, norm, 1e-8, tokens)
        ok &= _within_gates(actual, _reference(nccl, gemm, indices, weights, norm, tokens))
    calls = [(_inputs(t, rank, world_size, 70_000 + t), t) for t in (5, 64, 128)]
    owners = (first, second, first)
    for owner, ((gemm, indices, weights, norm), t) in zip(owners, calls):
        owner.finalize(gemm, indices, weights, norm, 1e-8, t)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [
            owner.finalize(g, i, w, n, 1e-8, t) for owner, ((g, i, w, n), t) in zip(owners, calls)
        ]
    for repeat in range(10):
        for index, (call, tokens) in enumerate(calls):
            _refill(call, tokens, rank, world_size, 80_000 + 10 * repeat + index)
        expected = [_reference(nccl, *call, tokens) for call, tokens in calls]
        graph.replay()
        for actual, want in zip(outputs, expected):
            ok &= _within_gates(actual, want)
    torch.cuda.synchronize()
    del graph
    # Retiring one workspace must not disturb the other.
    MPI.COMM_WORLD.Barrier()
    del second
    for step in range(20):
        tokens = schedule.randint(1, 128)
        gemm, indices, weights, norm = _inputs(tokens, rank, world_size, 90_000 + step)
        actual = first.finalize(gemm, indices, weights, norm, 1e-8, tokens)
        ok &= _within_gates(actual, _reference(nccl, gemm, indices, weights, norm, tokens))
    assert _all_ranks(ok)
    MPI.COMM_WORLD.Barrier()

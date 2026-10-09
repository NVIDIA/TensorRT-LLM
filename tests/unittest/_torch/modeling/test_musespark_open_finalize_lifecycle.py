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
"""Construction, failure and teardown behavior of the finalize+all-reduce workspace.

Run with four local MPI ranks, or eight ranks across two NVLink-connected
four-GPU nodes. The last test intentionally leaves a workspace alive at
interpreter exit; the launch must still finish with exit code 0 on every rank.

With four ranks the workspace shares CUDA IPC buffers, and no rank may free its
buffer while a peer still maps it. The tests record every unmap and free on the
node-wide monotonic clock, delay one rank's unmaps, and check that every unmap
finished before any free started.
"""

import gc
import time

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

# The NCCL reference all-reduce starts communicator threads that outlive each test.
pytestmark = pytest.mark.threadleak(enabled=False)

_LEAK_TOLERANCE = 32 << 20
_UNMAP_DELAY_S = 0.5
_kept_alive = []


def _require_group():
    world_size = MPI.COMM_WORLD.Get_size()
    if world_size not in (4, 8) or not torch.cuda.is_available() or not HAS_CUTEDSL:
        pytest.skip("Requires four or eight MPI ranks, CUDA and CuTe DSL")
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("GB300 required")
    mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank, gpus_per_node=4)
    return world_size, rank, mapping


def _check_call(workspace, nccl, rank, world_size, tokens, seed) -> bool:
    shared = torch.Generator(device="cuda").manual_seed(seed)
    weights = torch.rand(tokens, 8, device="cuda", generator=shared)
    weights /= weights.sum(-1, keepdim=True)
    norm = (torch.randn(4096, device="cuda", generator=shared) * 0.03).to(torch.bfloat16)
    local = torch.Generator(device="cuda").manual_seed(seed * 1009 + rank)
    gemm = torch.randn(max(64, tokens * 8), 4096, device="cuda", generator=local).to(torch.bfloat16)
    indices = torch.arange(tokens * 8, device="cuda", dtype=torch.int32).reshape(tokens, 8)
    indices = torch.where(indices % world_size == rank, indices, -1).contiguous()
    expected = nccl(
        post_expert_norm_combine(
            gemm, indices, weights, norm, 1e-8, num_tokens=tokens, top_k=8, hidden_size=4096
        ).to(torch.bfloat16)
    )
    actual = workspace.finalize(gemm, indices, weights, norm, 1e-8, tokens)
    torch.cuda.synchronize()
    actual, expected = actual.float(), expected.float()
    return bool(
        torch.allclose(actual, expected, rtol=0.04, atol=0.02)
        and (actual - expected).norm() <= 0.01 * expected.norm()
    )


def _all_ranks(ok: bool) -> bool:
    return all(MPI.COMM_WORLD.allgather(bool(ok)))


class _ReleaseLog:
    """Records this rank's IPC unmaps and buffer frees; can delay or fail its unmaps."""

    def __init__(self, monkeypatch, *, unmap_delay: float = 0.0, fail_unmap: bool = False):
        runtime = musespark_open_finalize_ar.cudart
        unmap_peer, free, malloc = (
            runtime.cudaIpcCloseMemHandle,
            runtime.cudaFree,
            runtime.cudaMalloc,
        )
        self.unmaps, self.frees, self.buffers = [], [], set()

        def recorded_unmap(pointer):
            if unmap_delay and not self.unmaps:
                time.sleep(unmap_delay)
            if fail_unmap:
                result = (runtime.cudaError_t.cudaErrorInvalidValue,)
            else:
                result = unmap_peer(pointer)
            self.unmaps.append(time.monotonic_ns())
            return result

        def recorded_free(pointer):
            if int(pointer) in self.buffers:
                self.frees.append(time.monotonic_ns())
            return free(pointer)

        def recorded_malloc(size):
            error, pointer = malloc(size)
            if error == runtime.cudaError_t.cudaSuccess:
                self.buffers.add(int(pointer))
            return error, pointer

        monkeypatch.setattr(runtime, "cudaIpcCloseMemHandle", recorded_unmap)
        monkeypatch.setattr(runtime, "cudaFree", recorded_free)
        monkeypatch.setattr(runtime, "cudaMalloc", recorded_malloc)

    def watch(self, workspace) -> None:
        if workspace._export:
            self.buffers.add(workspace._export)

    def clear(self) -> None:
        self.unmaps.clear()
        self.frees.clear()


def _release_order(log: _ReleaseLog) -> tuple[bool, int, int]:
    """Whether every rank's unmaps ended before any rank's frees began (IPC ranks share one
    node clock), plus the group's unmap and free counts."""
    logs = MPI.COMM_WORLD.allgather((log.unmaps, log.frees))
    unmaps = [stamp for done, _ in logs for stamp in done]
    frees = [stamp for _, done in logs for stamp in done]
    ordered = not unmaps or not frees or max(unmaps) < min(frees)
    return ordered, len(unmaps), len(frees)


def _free_device_memory() -> int:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0]


def _use_and_close_workspace(mapping, nccl, rank, world_size, cycle, log) -> bool:
    """One full lifetime: eager calls, a captured graph, quiescence, then close()."""
    ok = True
    workspace = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    log.watch(workspace)
    for step in range(3):
        ok &= _check_call(workspace, nccl, rank, world_size, 1 + 40 * step, 100 * cycle + step)
    gemm = torch.randn(64, 4096, device="cuda").to(torch.bfloat16)
    indices = torch.full((8, 8), -1, dtype=torch.int32, device="cuda")
    weights = torch.ones(8, 8, device="cuda")
    norm = torch.zeros(4096, dtype=torch.bfloat16, device="cuda")
    workspace.finalize(gemm, indices, weights, norm, 1e-8, 8)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = workspace.finalize(gemm, indices, weights, norm, 1e-8, 8)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    ok &= bool((output.float() == 0).all())
    del graph, output
    MPI.COMM_WORLD.Barrier()
    log.clear()
    workspace.close()
    ipc = world_size == 4
    ordered, unmaps, frees = _release_order(log)
    ok &= ordered
    ok &= (unmaps, frees) == ((world_size * (world_size - 1), world_size) if ipc else (0, 0))
    del workspace
    gc.collect()
    MPI.COMM_WORLD.Barrier()
    return ok


@torch.inference_mode()
def test_finalize_workspace_teardown_and_recreate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quiesce, close and rebuild ten times after eager and graph use, with one rank slow
    to unmap each time; no buffer is freed early and nothing leaks."""
    world_size, rank, mapping = _require_group()
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    # The first lifetime also pays one-time costs (NCCL buffers, kernel modules).
    with monkeypatch.context() as patch:
        ok = _use_and_close_workspace(mapping, nccl, rank, world_size, 0, _ReleaseLog(patch))
    MPI.COMM_WORLD.Barrier()
    baseline = _free_device_memory()
    for cycle in range(1, 11):
        with monkeypatch.context() as patch:
            delay = _UNMAP_DELAY_S if rank == cycle % world_size else 0.0
            log = _ReleaseLog(patch, unmap_delay=delay)
            ok &= _use_and_close_workspace(mapping, nccl, rank, world_size, cycle, log)
    leaked = baseline - _free_device_memory()
    assert _all_ranks(ok and leaked < _LEAK_TOLERANCE), f"leaked {leaked} bytes on rank {rank}"
    MPI.COMM_WORLD.Barrier()


_IPC_STAGES = ("allocate", "map_peers", "handle_raises", "open_raises")


@pytest.mark.parametrize(
    "stage",
    [
        "fingerprint",
        "compile",
        "allocate",
        "map_peers",
        "initialize",
        "setup_raises",
        "handle_raises",
        "open_raises",
        "launch_raises",
    ],
)
@torch.inference_mode()
def test_finalize_construction_failure_is_agreed(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure on one rank, reported as an error code or raised as any exception type, makes
    every rank raise the same type and release in group order, with rank 0 slow to unmap;
    nothing leaks and a retry then works."""
    world_size, rank, mapping = _require_group()
    if world_size == 8 and stage in _IPC_STAGES:
        pytest.skip("IPC allocation stages; the fabric allocator agrees on failures itself")
    failing = world_size - 1
    module = musespark_open_finalize_ar
    log = _ReleaseLog(monkeypatch, unmap_delay=_UNMAP_DELAY_S if rank == 0 else 0.0)
    if rank == failing:
        if stage == "fingerprint":
            monkeypatch.setattr(module, "_fingerprint", lambda *args: 12345)
        elif stage == "compile":

            def broken_compile(*args):
                raise TypeError("injected compilation failure")

            monkeypatch.setattr(module, "_compiled_kernel", broken_compile)
        elif stage == "allocate":
            monkeypatch.setattr(
                module.cudart,
                "cudaMalloc",
                lambda size: (module.cudart.cudaError_t.cudaErrorMemoryAllocation, 0),
            )
        elif stage == "map_peers":
            monkeypatch.setattr(
                module.cudart,
                "cudaIpcOpenMemHandle",
                lambda handle, flags: (module.cudart.cudaError_t.cudaErrorInvalidValue, 0),
            )
        elif stage == "setup_raises":

            def broken_prepare(self, mapping):
                raise KeyError("injected setup failure")

            monkeypatch.setattr(FinalizeIpcWorkspace, "_prepare", broken_prepare)
        elif stage == "handle_raises":

            def broken_export(handle):
                raise ValueError("injected handle export failure")

            monkeypatch.setattr(module, "_ipc_mem_handle_to_bytes", broken_export)
        elif stage == "open_raises":

            def broken_import(data):
                raise TypeError("injected handle import failure")

            monkeypatch.setattr(module, "_ipc_mem_handle_from_bytes", broken_import)
        elif stage == "launch_raises":
            # TVM-FFI reports launch failures as ValueError, TypeError and other types.
            def broken_launch(self, *args, **kwargs):
                raise ValueError("injected launch failure")

            monkeypatch.setattr(FinalizeIpcWorkspace, "_launch", broken_launch)
        else:
            monkeypatch.setattr(
                module.cuda,
                "cuMemsetD32",
                lambda *args: (module.cuda.CUresult.CUDA_ERROR_INVALID_VALUE,),
            )
    expected = ValueError if stage == "fingerprint" else RuntimeError
    MPI.COMM_WORLD.Barrier()
    baseline = _free_device_memory()
    raised = None
    try:
        FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    except (ValueError, RuntimeError) as error:
        raised = type(error)
    monkeypatch.undo()
    ordered, unmaps, frees = _release_order(log)
    # Group-wide IPC events the release must have produced; the failing rank maps nothing at
    # "map_peers" and allocates nothing at "allocate".
    events = {
        "allocate": (0, world_size - 1),
        "map_peers": ((world_size - 1) ** 2, world_size),
        "initialize": (world_size * (world_size - 1), world_size),
        "handle_raises": (0, world_size),
        "open_raises": ((world_size - 1) ** 2, world_size),
        "launch_raises": (world_size * (world_size - 1), world_size),
    }
    ordered &= (unmaps, frees) == (events.get(stage, (0, 0)) if world_size == 4 else (0, 0))
    leaked = baseline - _free_device_memory()
    MPI.COMM_WORLD.Barrier()
    workspace = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    works = _check_call(workspace, nccl, rank, world_size, 16, 4242)
    workspace.close()
    ok = raised is expected and ordered and leaked < _LEAK_TOLERANCE and works
    assert _all_ranks(ok), (
        f"rank {rank}: raised {raised}, ordered {ordered}, leaked {leaked}, retry ok {works}"
    )
    MPI.COMM_WORLD.Barrier()


@pytest.mark.parametrize("failure", ["sync", "unmap"])
@torch.inference_mode()
def test_finalize_failed_release_frees_nothing(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If one rank fails to synchronize or to unmap, every rank raises from close(), no IPC
    buffer is freed, and the group can still build a working workspace."""
    world_size, rank, mapping = _require_group()
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    workspace = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    ok = _check_call(workspace, nccl, rank, world_size, 8, 77)
    failing = world_size - 1
    log = _ReleaseLog(monkeypatch, fail_unmap=failure == "unmap" and rank == failing)
    log.watch(workspace)
    if rank == failing:
        if failure == "sync":
            monkeypatch.setattr(workspace, "_sync", lambda: False)
        elif world_size == 8:
            fabric_class = type(workspace._fabric)
            release = fabric_class.close_mnnvl_memory

            # Release for real, so the allocator's bookkeeping stays intact, but report failure.
            def failed_release(cls, pointer):
                release(pointer)
                raise RuntimeError("injected fabric release failure")

            monkeypatch.setattr(fabric_class, "close_mnnvl_memory", classmethod(failed_release))
    MPI.COMM_WORLD.Barrier()
    message = ""
    try:
        workspace.close()
    except RuntimeError as error:
        message = str(error)
    monkeypatch.undo()
    try:
        workspace.finalize(*_zero_call())
        rejected = False
    except RuntimeError:
        rejected = True
    retry = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    works = _check_call(retry, nccl, rank, world_size, 16, 4243)
    retry.close()
    ok &= f"rank(s) [{failing}]" in message and not log.frees and rejected and works
    assert _all_ranks(ok), (
        f"rank {rank}: message {message!r}, frees {len(log.frees)}, rejected {rejected}, "
        f"retry ok {works}"
    )
    MPI.COMM_WORLD.Barrier()


def _zero_call(tokens: int = 4):
    gemm = torch.zeros(64, 4096, device="cuda", dtype=torch.bfloat16)
    indices = torch.full((tokens, 8), -1, dtype=torch.int32, device="cuda")
    weights = torch.zeros(tokens, 8, device="cuda")
    norm = torch.zeros(4096, dtype=torch.bfloat16, device="cuda")
    return gemm, indices, weights, norm, 1e-8, tokens


@torch.inference_mode()
def test_finalize_drop_right_after_enqueue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropping a workspace without close(), right after a call, waits for the call, frees no
    IPC buffer, and leaves another live workspace working."""
    world_size, rank, mapping = _require_group()
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    other = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    log = _ReleaseLog(monkeypatch)
    ok = True
    for repeat in range(3):
        workspace = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
        log.watch(workspace)
        lagging = repeat % world_size
        tokens = 64
        shared = torch.Generator(device="cuda").manual_seed(repeat)
        weights = torch.rand(tokens, 8, device="cuda", generator=shared)
        weights /= weights.sum(-1, keepdim=True)
        norm = torch.zeros(4096, dtype=torch.bfloat16, device="cuda")
        gemm = torch.randn(512, 4096, device="cuda").to(torch.bfloat16)
        indices = torch.arange(tokens * 8, device="cuda", dtype=torch.int32).reshape(tokens, 8)
        indices = torch.where(indices % world_size == rank, indices, -1).contiguous()
        expected = nccl(
            post_expert_norm_combine(
                gemm, indices, weights, norm, 1e-8, num_tokens=tokens, top_k=8, hidden_size=4096
            ).to(torch.bfloat16)
        )
        if rank == lagging:
            torch.cuda._sleep(20_000_000)
        actual = workspace.finalize(gemm, indices, weights, norm, 1e-8, tokens)
        if rank == lagging:
            del workspace
            gc.collect()
        torch.cuda.synchronize()
        actual, expected = actual.float(), expected.float()
        ok &= bool(torch.allclose(actual, expected, rtol=0.04, atol=0.02))
        MPI.COMM_WORLD.Barrier()
        if rank != lagging:
            del workspace
            gc.collect()
        MPI.COMM_WORLD.Barrier()
        ok &= _check_call(other, nccl, rank, world_size, 24, 500 + repeat)
    monkeypatch.undo()
    other.close()
    assert _all_ranks(ok and not log.frees), f"rank {rank}: ok {ok}, frees {len(log.frees)}"
    MPI.COMM_WORLD.Barrier()


@torch.inference_mode()
def test_finalize_workspace_alive_at_exit() -> None:
    """Keep a used workspace until interpreter exit; the job must still exit cleanly."""
    world_size, rank, mapping = _require_group()
    workspace = FinalizeIpcWorkspace(mapping, use_mnnvl=world_size == 8)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    assert _all_ranks(_check_call(workspace, nccl, rank, world_size, 32, 7))
    _kept_alive.append(workspace)
    MPI.COMM_WORLD.Barrier()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construction, failure and teardown of the fused finalize + all-reduce workspace, on four ranks.

The four ranks share CUDA IPC buffers, and no rank may free its buffer while a
peer still maps it. The tests record every unmap and free on the node-wide
monotonic clock, delay one rank's unmaps, and check that every unmap finished
before any free started. Each test runs its body on four MPI workers spawned by
the test process.
"""

import gc
import pickle
import sys
import time

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.models import modeling_musespark_open, musespark_open_finalize_ar
from tensorrt_llm._torch.models.modeling_musespark_open import post_expert_norm_combine
from tensorrt_llm._torch.models.musespark_open_finalize_ar import HAS_CUTEDSL, FinalizeIpcWorkspace
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

RANKS = 4
_LEAK_TOLERANCE = 32 << 20
_UNMAP_DELAY_S = 0.5

pytestmark = [
    # The worker pool is reused across tests, and NCCL starts threads that outlive each test.
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        torch.cuda.device_count() < RANKS
        or not HAS_CUTEDSL
        or torch.cuda.get_device_capability() != (10, 3),
        reason="four GB300 GPUs and CuTe DSL required",
    ),
    pytest.mark.parametrize("mpi_pool_executor", [RANKS], indirect=True),
]


def _run_rank(body, *args):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=RANKS, tp_size=RANKS, rank=rank, gpus_per_node=RANKS)
    with torch.inference_mode():
        return body(rank, mapping, *args)


def _on_every_rank(executor, body, *args) -> list:
    """`body(rank, mapping, *args)` on each worker; the per-rank results in rank order."""
    return list(executor.map(_run_rank, [body] * RANKS, *[[arg] * RANKS for arg in args]))


def _check_call(workspace, nccl, rank, tokens, seed) -> bool:
    shared = torch.Generator(device="cuda").manual_seed(seed)
    weights = torch.rand(tokens, 8, device="cuda", generator=shared)
    weights /= weights.sum(-1, keepdim=True)
    norm = (torch.randn(4096, device="cuda", generator=shared) * 0.03).to(torch.bfloat16)
    local = torch.Generator(device="cuda").manual_seed(seed * 1009 + rank)
    gemm = torch.randn(max(64, tokens * 8), 4096, device="cuda", generator=local).to(torch.bfloat16)
    indices = torch.arange(tokens * 8, device="cuda", dtype=torch.int32).reshape(tokens, 8)
    indices = torch.where(indices % RANKS == rank, indices, -1).contiguous()
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

    def __init__(self, patch, *, unmap_delay: float = 0.0, fail_unmap: bool = False):
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

        patch.setattr(runtime, "cudaIpcCloseMemHandle", recorded_unmap)
        patch.setattr(runtime, "cudaFree", recorded_free)
        patch.setattr(runtime, "cudaMalloc", recorded_malloc)

    def watch(self, workspace) -> None:
        if workspace._export:
            self.buffers.add(workspace._export)

    def clear(self) -> None:
        self.unmaps.clear()
        self.frees.clear()


def _release_order(log: _ReleaseLog) -> tuple[bool, int, int]:
    """Whether every rank's unmaps ended before any rank's frees began (the ranks share one
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


def _use_and_close_workspace(mapping, nccl, rank, cycle, log) -> bool:
    """One full lifetime: eager calls, a captured graph, quiescence, then close()."""
    ok = True
    workspace = FinalizeIpcWorkspace(mapping)
    log.watch(workspace)
    for step in range(3):
        ok &= _check_call(workspace, nccl, rank, 1 + 40 * step, 100 * cycle + step)
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
    ordered, unmaps, frees = _release_order(log)
    ok &= ordered
    ok &= (unmaps, frees) == (RANKS * (RANKS - 1), RANKS)
    del workspace
    gc.collect()
    MPI.COMM_WORLD.Barrier()
    return ok


def _teardown_and_recreate(rank, mapping):
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    # The first lifetime also pays one-time costs (NCCL buffers, kernel modules).
    with pytest.MonkeyPatch.context() as patch:
        ok = _use_and_close_workspace(mapping, nccl, rank, 0, _ReleaseLog(patch))
    MPI.COMM_WORLD.Barrier()
    baseline = _free_device_memory()
    for cycle in range(1, 11):
        with pytest.MonkeyPatch.context() as patch:
            delay = _UNMAP_DELAY_S if rank == cycle % RANKS else 0.0
            log = _ReleaseLog(patch, unmap_delay=delay)
            ok &= _use_and_close_workspace(mapping, nccl, rank, cycle, log)
    leaked = baseline - _free_device_memory()
    return _all_ranks(ok and leaked < _LEAK_TOLERANCE)


def test_finalize_workspace_teardown_and_recreate(mpi_pool_executor) -> None:
    """Quiesce, close and rebuild ten times after eager and graph use, with one rank slow
    to unmap each time; no buffer is freed early and nothing leaks."""
    assert all(_on_every_rank(mpi_pool_executor, _teardown_and_recreate))


def _inject(patch, stage):
    module = musespark_open_finalize_ar
    if stage == "fingerprint":
        patch.setattr(module, "_fingerprint", lambda *args: 12345)
    elif stage == "compile":

        def broken_compile(*args):
            raise TypeError("injected compilation failure")

        patch.setattr(module, "_compiled_kernel", broken_compile)
    elif stage == "allocate":
        patch.setattr(
            module.cudart,
            "cudaMalloc",
            lambda size: (module.cudart.cudaError_t.cudaErrorMemoryAllocation, 0),
        )
    elif stage == "map_peers":
        patch.setattr(
            module.cudart,
            "cudaIpcOpenMemHandle",
            lambda handle, flags: (module.cudart.cudaError_t.cudaErrorInvalidValue, 0),
        )
    elif stage == "setup_raises":

        def broken_prepare(self, mapping):
            raise KeyError("injected setup failure")

        patch.setattr(FinalizeIpcWorkspace, "_prepare", broken_prepare)
    elif stage == "handle_raises":

        def broken_export(handle):
            raise ValueError("injected handle export failure")

        patch.setattr(module, "_ipc_mem_handle_to_bytes", broken_export)
    elif stage == "open_raises":

        def broken_import(data):
            raise TypeError("injected handle import failure")

        patch.setattr(module, "_ipc_mem_handle_from_bytes", broken_import)
    elif stage == "launch_raises":
        # TVM-FFI reports launch failures as ValueError, TypeError and other types.
        def broken_launch(self, *args, **kwargs):
            raise ValueError("injected launch failure")

        patch.setattr(FinalizeIpcWorkspace, "_launch", broken_launch)
    else:
        patch.setattr(
            module.cuda,
            "cuMemsetD32",
            lambda *args: (module.cuda.CUresult.CUDA_ERROR_INVALID_VALUE,),
        )


def _construction_failure(rank, mapping, stage):
    failing = rank == RANKS - 1
    with pytest.MonkeyPatch.context() as patch:
        log = _ReleaseLog(patch, unmap_delay=_UNMAP_DELAY_S if rank == 0 else 0.0)
        if failing:
            _inject(patch, stage)
        expected = ValueError if stage == "fingerprint" else RuntimeError
        MPI.COMM_WORLD.Barrier()
        baseline = _free_device_memory()
        raised = None
        try:
            FinalizeIpcWorkspace(mapping)
        except (ValueError, RuntimeError) as error:
            raised = type(error)
    ordered, unmaps, frees = _release_order(log)
    # Group-wide IPC events the release must produce; the failing rank maps nothing at
    # "map_peers" and allocates nothing at "allocate".
    events = {
        "allocate": (0, RANKS - 1),
        "map_peers": ((RANKS - 1) ** 2, RANKS),
        "initialize": (RANKS * (RANKS - 1), RANKS),
        "handle_raises": (0, RANKS),
        "open_raises": ((RANKS - 1) ** 2, RANKS),
        "launch_raises": (RANKS * (RANKS - 1), RANKS),
    }
    ordered &= (unmaps, frees) == events.get(stage, (0, 0))
    leaked = baseline - _free_device_memory()
    MPI.COMM_WORLD.Barrier()
    workspace = FinalizeIpcWorkspace(mapping)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    works = _check_call(workspace, nccl, rank, 16, 4242)
    workspace.close()
    return _all_ranks(raised is expected and ordered and leaked < _LEAK_TOLERANCE and works)


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
def test_finalize_construction_failure_is_agreed(mpi_pool_executor, stage: str) -> None:
    """A failure on one rank, reported as an error code or raised as any exception type,
    makes every rank raise the same type and release in group order, with rank 0 slow to
    unmap; nothing leaks and a retry then works."""
    assert all(_on_every_rank(mpi_pool_executor, _construction_failure, stage))


def _model_falls_back(rank, mapping):
    warnings = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(modeling_musespark_open.logger, "warning", warnings.append)
        if rank == RANKS - 1:
            _inject(patch, "allocate")
        workspace = modeling_musespark_open._build_finalize_allreduce_workspace(mapping)
    ok = workspace is None and len(warnings) == 1 and "allocation failed" in warnings[0]
    return _all_ranks(ok)


def test_a_failure_on_one_rank_leaves_every_rank_on_the_unfused_path(mpi_pool_executor) -> None:
    """The model's construction degrades on every rank alike instead of failing or hanging."""
    assert all(_on_every_rank(mpi_pool_executor, _model_falls_back))


def _zero_call(tokens: int = 4):
    gemm = torch.zeros(64, 4096, device="cuda", dtype=torch.bfloat16)
    indices = torch.full((tokens, 8), -1, dtype=torch.int32, device="cuda")
    weights = torch.zeros(tokens, 8, device="cuda")
    norm = torch.zeros(4096, dtype=torch.bfloat16, device="cuda")
    return gemm, indices, weights, norm, 1e-8, tokens


def _failed_release(rank, mapping, failure):
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    workspace = FinalizeIpcWorkspace(mapping)
    ok = _check_call(workspace, nccl, rank, 8, 77)
    failing = rank == RANKS - 1
    message = ""
    with pytest.MonkeyPatch.context() as patch:
        log = _ReleaseLog(patch, fail_unmap=failure == "unmap" and failing)
        log.watch(workspace)
        if failing and failure == "sync":
            patch.setattr(workspace, "_sync", lambda: False)
        MPI.COMM_WORLD.Barrier()
        try:
            workspace.close()
        except RuntimeError as error:
            message = str(error)
    try:
        workspace.finalize(*_zero_call())
        rejected = False
    except RuntimeError:
        rejected = True
    retry = FinalizeIpcWorkspace(mapping)
    works = _check_call(retry, nccl, rank, 16, 4243)
    retry.close()
    ok &= f"rank(s) [{RANKS - 1}]" in message and not log.frees and rejected and works
    return _all_ranks(ok)


@pytest.mark.parametrize("failure", ["sync", "unmap"])
def test_finalize_failed_release_frees_nothing(mpi_pool_executor, failure: str) -> None:
    """If one rank fails to synchronize or to unmap, every rank raises from close(), no IPC
    buffer is freed, and the group can still build a working workspace."""
    assert all(_on_every_rank(mpi_pool_executor, _failed_release, failure))


def _drop_after_enqueue(rank, mapping):
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    other = FinalizeIpcWorkspace(mapping)
    ok = True
    with pytest.MonkeyPatch.context() as patch:
        log = _ReleaseLog(patch)
        for repeat in range(3):
            workspace = FinalizeIpcWorkspace(mapping)
            log.watch(workspace)
            lagging = repeat % RANKS
            tokens = 64
            shared = torch.Generator(device="cuda").manual_seed(repeat)
            weights = torch.rand(tokens, 8, device="cuda", generator=shared)
            weights /= weights.sum(-1, keepdim=True)
            norm = torch.zeros(4096, dtype=torch.bfloat16, device="cuda")
            gemm = torch.randn(512, 4096, device="cuda").to(torch.bfloat16)
            indices = torch.arange(tokens * 8, device="cuda", dtype=torch.int32).reshape(tokens, 8)
            indices = torch.where(indices % RANKS == rank, indices, -1).contiguous()
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
            ok &= _check_call(other, nccl, rank, 24, 500 + repeat)
    other.close()
    return _all_ranks(ok and not log.frees)


def test_finalize_drop_right_after_enqueue(mpi_pool_executor) -> None:
    """Dropping a workspace without close(), right after a call, waits for the call, frees no
    IPC buffer, and leaves another live workspace working."""
    assert all(_on_every_rank(mpi_pool_executor, _drop_after_enqueue))

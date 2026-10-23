# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Two MPI-rank raw-NCCL numerical/event tests, including OFF and runtime-none.

Run each build/selector entry in a fresh pytest process. The existing MPI pool
fixture establishes the MPI communicator; torchrun alone is not sufficient.
Traces describe host launches at capture, never additional launches at replay.
"""

import json
import os
import pickle
import shutil
import socket
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cloudpickle
import pytest
import torch
import torch.distributed as dist
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from profiler_test_utils import decode_profiler_metadata

import tensorrt_llm
from tensorrt_llm.bindings import BuildInfo

if TYPE_CHECKING:
    from tensorrt_llm._profiler import MetadataValue

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)


def _events(path: Path) -> list[dict[str, "MetadataValue"]]:
    trace = json.loads(path.read_text())
    result = []
    for node in trace["nodes"]:
        if node["name"] == "trtllm::nccl_metadata":
            region, fields = decode_profiler_metadata(node["inputs"]["values"][0])
            assert region == "nccl"
            result.append(fields)
    return result


def _enabled() -> bool:
    assert type(BuildInfo.ENABLE_NCCL_PROFILING) is bool
    selection = os.environ.get("TLLM_PROFILE_CUSTOM_REGIONS")
    return BuildInfo.ENABLE_NCCL_PROFILING and (
        selection is None or "nccl" in {item.strip() for item in selection.split(",")}
    )


def _profile(
    operation: Callable[[], list[torch.Tensor]],
    expected: list[torch.Tensor],
    graph: bool,
) -> list[dict[str, "MetadataValue"]]:
    # Initialize communicators and allocator state outside capture and profiling.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    with tempfile.TemporaryDirectory(prefix="nccl-profiler-") as directory:
        path = Path(directory) / "execution.json"
        observer = torch.profiler.ExecutionTraceObserver().register_callback(str(path))
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                execution_trace_observer=observer,
            ):
                if graph:
                    captured = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(captured, stream=stream):
                        outputs = operation()
                else:
                    outputs = operation()
            if graph:
                captured.replay()
            torch.cuda.synchronize()
            for output, reference in zip(outputs, expected, strict=True):
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
        finally:
            observer.unregister_callback()
        fields = _events(path)
        if graph:
            # Replay-only profiling must not invent per-replay host metadata.
            replay_path = Path(directory) / "replay.json"
            replay_observer = torch.profiler.ExecutionTraceObserver().register_callback(
                str(replay_path)
            )
            try:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    record_shapes=True,
                    execution_trace_observer=replay_observer,
                ):
                    for _ in range(2):
                        captured.replay()
                        torch.cuda.synchronize()
                        for output, reference in zip(outputs, expected, strict=True):
                            torch.testing.assert_close(output, reference, rtol=0, atol=0)
            finally:
                replay_observer.unregister_callback()
            assert _events(replay_path) == []
        trace_root = os.environ.get("TLLM_NCCL_PROFILER_TEST_TRACE_DIR")
        if trace_root is not None:
            destination = (
                Path(trace_root) / f"rank-{tensorrt_llm.mpi_rank()}" / Path(directory).name
            )
            destination.mkdir(parents=True, exist_ok=True)
            for trace_path in Path(directory).glob("*.json"):
                shutil.copy2(trace_path, destination / trace_path.name)
    return fields


def _assert_fields(
    fields: list[dict[str, "MetadataValue"]],
    operation: str,
    logical_bytes: int,
    dimension: str,
    strategy: str,
    graph: bool,
) -> None:
    if not _enabled():
        assert fields == []
        return
    assert len(fields) == 1
    metadata = fields[0]
    assert metadata["nccl.operation"] == operation
    assert metadata["nccl.group"] == [0, 1]
    assert metadata["nccl.rank"] == tensorrt_llm.mpi_rank()
    assert metadata["nccl.logical_bytes"] == logical_bytes
    assert metadata["nccl.parallelism_dimension"] == dimension
    assert metadata["nccl.kind"] == ("p2p" if operation in {"send", "recv"} else "collective")
    assert metadata["nccl.selected_strategy"] == strategy
    assert metadata["nccl.cuda_graph_state"] == ("capture" if graph else "eager")
    assert not {"nccl.algorithm", "nccl.protocol", "nccl.sequence"} & metadata.keys()


def _run_collective(
    case: Literal["allreduce", "equal", "uneven", "list", "helix"], graph: bool
) -> bool:
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    group = [0, 1]
    local = torch.full((2, 4), rank + 1.0, device="cuda")
    gathered = torch.cat([torch.ones_like(local), torch.full_like(local, 2)])
    if case == "allreduce":
        # NCCL=0 and fusion NONE=0 are the checked-in native enum values.
        def operation() -> list[torch.Tensor]:
            return list(
                torch.ops.trtllm.allreduce(
                    local, None, None, None, None, None, group, 0, 0, 0.0, False
                )
            )

        fields = _profile(operation, [torch.full_like(local, 3)], graph)
        _assert_fields(fields, "all_reduce", 32, "tp", "NCCL", graph)
        if fields:
            assert fields[0]["nccl.reduce_op"] == "sum"
    elif case in {"equal", "list"}:

        def operation() -> list[torch.Tensor]:
            if case == "list":
                return list(torch.ops.trtllm.allgather_list([local, local], None, group))
            return [torch.ops.trtllm.allgather(local, None, group)]

        fields = _profile(operation, [gathered] * (2 if case == "list" else 1), graph)
        _assert_fields(
            fields, "all_gather", 64 if case == "list" else 32, "unknown", "nccl_allgather", graph
        )
    elif case == "uneven":
        local = torch.full((rank + 1, 4), rank + 1.0, device="cuda")
        reference = torch.cat(
            [torch.ones((1, 4), device="cuda"), torch.full((2, 4), 2, device="cuda")]
        )
        fields = _profile(
            lambda: [torch.ops.trtllm.allgather(local, [1, 2], group)], [reference], graph
        )
        _assert_fields(fields, "all_gather", (rank + 1) * 16, "unknown", "nccl_broadcast", graph)
        if fields:
            assert fields[0]["nccl.input_split_sizes"] == [1, 2]
            assert fields[0]["nccl.output_split_sizes"] == [1, 2]
    else:
        inputs = [
            torch.full((2, 4), 100 * rank + 10 * field + peer, device="cuda", dtype=torch.float32)
            for field in range(2)
            for peer in group
        ]
        expected = [
            torch.stack([torch.full_like(local, 100 * peer + 10 * field + rank) for peer in group])
            for field in range(2)
        ]
        fields = _profile(
            lambda: list(torch.ops.trtllm.alltoall_helix(inputs, group, 2)), expected, graph
        )
        _assert_fields(fields, "all_to_all", 128, "unknown", "nccl_send_recv", graph)
        if fields:
            assert fields[0]["nccl.input_split_sizes"] == [8, 8, 8, 8]
    return True


def _run_point_to_point() -> bool:
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    communicator = torch.classes.trtllm.NcclCommunicatorOp(2, rank)
    local = torch.full((2, 4), 17.0 if rank == 0 else 0.0, device="cuda")

    def operation() -> list[torch.Tensor]:
        if rank == 0:
            communicator.send(local, 1)
        else:
            communicator.recv(local, 0)
        return [local]

    fields = _profile(operation, [torch.full_like(local, 17)], False)
    _assert_fields(fields, "send" if rank == 0 else "recv", 32, "pp", "nccl_send_recv", False)
    if fields:
        assert fields[0]["nccl.src_rank"] == 0
        assert fields[0]["nccl.dst_rank"] == 1
    return True


def _run_process_group() -> bool:
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    port = None
    if rank == 0:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
    port = MPI.COMM_WORLD.bcast(port, root=0)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        group = dist.distributed_c10d._get_default_group().boxed()
        local = torch.full((2, 4), rank + 1.0, device="cuda")
        expected = torch.cat([torch.ones_like(local), torch.full_like(local, 2)])
        fields = _profile(
            lambda: [torch.ops.trtllm.allgather_pg(local, None, [0, 1], group)], [expected], False
        )
        assert fields == []
    finally:
        dist.destroy_process_group()
    return True


@pytest.mark.cpu_only
def test_nccl_profiling_build_info() -> None:
    """The build feature is a bool even when custom regions are compiled out."""
    assert type(BuildInfo.ENABLE_NCCL_PROFILING) is bool


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two real CUDA devices")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("case", ["allreduce", "equal", "uneven", "list", "helix"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "capture"])
def test_raw_nccl_events(
    mpi_pool_executor: MPIPoolExecutor,
    case: Literal["allreduce", "equal", "uneven", "list", "helix"],
    graph: bool,
) -> None:
    """Validate raw operations and graph replay with the real MPI-backed fixture."""
    assert list(mpi_pool_executor.map(_run_collective, [case] * 2, [graph] * 2)) == [True, True]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two real CUDA devices")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("process_group", [False, True], ids=["point-to-point", "process-group"])
def test_nccl_owner_boundaries(mpi_pool_executor: MPIPoolExecutor, process_group: bool) -> None:
    """P2P observes actual endpoints; ProcessGroup must not acquire a raw-NCCL label."""
    operation = _run_process_group if process_group else _run_point_to_point
    futures = [mpi_pool_executor.submit(operation) for _ in range(2)]
    assert [future.result() for future in futures] == [True, True]

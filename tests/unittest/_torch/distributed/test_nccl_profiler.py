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

"""Two MPI-rank collective and raw-NCCL P2P numerical/event tests.

Run each build/selector entry in a fresh pytest process. The existing MPI pool
fixture establishes the MPI communicator; torchrun alone is not sufficient.
Traces describe host launches at capture, never additional launches at replay.
"""

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
from profiler_test_utils import decode_node_metadata, execution_trace_observer, profiler_nodes

import tensorrt_llm
from tensorrt_llm import _profiler as profiler
from tensorrt_llm.bindings import BuildInfo

if TYPE_CHECKING:
    from tensorrt_llm._profiler import MetadataValue

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)


def _events(path: Path) -> list[dict[str, "MetadataValue"]]:
    """Decode NCCL metadata records from an exported execution-trace file."""
    result = []
    for node in profiler_nodes(path, "trtllm::nccl_send", "trtllm::nccl_recv"):
        region, fields = decode_node_metadata(node)
        assert region == "nccl"
        assert node["name"] == f"trtllm::nccl_{fields['nccl.operation']}"
        result.append(fields)
    return result


def _enabled() -> bool:
    """Require compiled NCCL probes and a current selection that includes
    nccl.
    """
    assert type(BuildInfo.ENABLE_NCCL_PROFILING) is bool
    return BuildInfo.ENABLE_NCCL_PROFILING and profiler.custom_region_enabled("nccl")


def _profile(
    operation: Callable[[], list[torch.Tensor]],
    expected: list[torch.Tensor],
    graph: bool,
    dispatcher_name: str | None = None,
    before_run: Callable[[], None] | None = None,
) -> list[dict[str, "MetadataValue"]]:
    # Initialize communicators and allocator state outside capture and profiling.
    """Warm up, execute, and validate a distributed operation under real
    profiler capture.
    """
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
        with execution_trace_observer(path) as observer:
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
                    if before_run is not None:
                        before_run()
                    outputs = operation()
            if graph:
                if before_run is not None:
                    before_run()
                captured.replay()
            torch.cuda.synchronize()
            for output, reference in zip(outputs, expected, strict=True):
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
        fields = _events(path)
        if dispatcher_name is not None:
            dispatchers = profiler_nodes(path, dispatcher_name)
            assert len(dispatchers) == 1
            attributes = {item["name"]: item["value"] for item in dispatchers[0]["attrs"]}
            assert attributes["scope"] == 0
            assert attributes["op_schema"]
            assert fields == []
        if graph:
            # Replay-only profiling must not invent per-replay host metadata.
            replay_path = Path(directory) / "replay.json"
            with execution_trace_observer(replay_path) as replay_observer:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    record_shapes=True,
                    execution_trace_observer=replay_observer,
                ):
                    for _ in range(2):
                        if before_run is not None:
                            before_run()
                        captured.replay()
                        torch.cuda.synchronize()
                        for output, reference in zip(outputs, expected, strict=True):
                            torch.testing.assert_close(output, reference, rtol=0, atol=0)
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
    """Check native NCCL fields or the absence of omitted build-time probes."""
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
    """Run the selected two-rank collective and validate its profiler
    metadata.
    """
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    group = [0, 1]
    local = torch.full((2, 4), rank + 1.0, device="cuda")
    gathered = torch.cat([torch.ones_like(local), torch.full_like(local, 2)])
    updates = [(local, rank + 1, 10)]
    run = 0

    def before_run() -> None:
        """Refresh inputs and independent expected buffers for every measured run."""
        nonlocal run
        run += 1
        for tensor, initial, increment in updates:
            tensor.fill_(initial + increment * run)

    if case == "allreduce":
        # NCCL=0 and fusion NONE=0 are the checked-in native enum values.
        def operation() -> list[torch.Tensor]:
            return list(
                torch.ops.trtllm.allreduce(
                    local, None, None, None, None, None, group, 0, 0, 0.0, False
                )
            )

        reference = torch.full_like(local, 3)
        updates.append((reference, 3, 20))
        _profile(operation, [reference], graph, "trtllm::allreduce", before_run=before_run)
    elif case in {"equal", "list"}:
        updates.extend([(gathered[:2], 1, 10), (gathered[2:], 2, 10)])

        def operation() -> list[torch.Tensor]:
            """Execute the selected tensor or list allgather for profiler
            validation.
            """
            if case == "list":
                return list(torch.ops.trtllm.allgather_list([local, local], None, group))
            return [torch.ops.trtllm.allgather(local, None, group)]

        _profile(
            operation,
            [gathered] * (2 if case == "list" else 1),
            graph,
            "trtllm::allgather_list" if case == "list" else "trtllm::allgather",
            before_run=before_run,
        )
    elif case == "uneven":
        local = torch.full((rank + 1, 4), rank + 1.0, device="cuda")
        reference = torch.cat(
            [torch.ones((1, 4), device="cuda"), torch.full((2, 4), 2, device="cuda")]
        )
        updates = [(local, rank + 1, 10), (reference[:1], 1, 10), (reference[1:], 2, 10)]
        _profile(
            lambda: [torch.ops.trtllm.allgather(local, [1, 2], group)],
            [reference],
            graph,
            "trtllm::allgather",
            before_run=before_run,
        )
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
        updates = [
            (inputs[2 * field + peer], 100 * rank + 10 * field + peer, 1000)
            for field in range(2)
            for peer in group
        ] + [
            (expected[field][peer], 100 * peer + 10 * field + rank, 1000)
            for field in range(2)
            for peer in group
        ]
        _profile(
            lambda: list(torch.ops.trtllm.alltoall_helix(inputs, group, 2)),
            expected,
            graph,
            "trtllm::alltoall_helix",
            before_run=before_run,
        )
    return True


def _run_point_to_point(graph: bool, expected: tuple[bool, bool]) -> bool:
    """Validate fresh two-rank send/receive payloads and native metadata."""
    import torch

    assert (BuildInfo.ENABLE_NCCL_PROFILING, profiler.custom_region_enabled("nccl")) == expected
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    communicator = torch.classes.trtllm.NcclCommunicatorOp(2, rank)
    local = torch.full((2, 4), 17.0 if rank == 0 else 0.0, device="cuda")

    def operation() -> list[torch.Tensor]:
        """Send from rank zero or receive in place on its peer."""
        if rank == 0:
            communicator.send(local, 1)
        else:
            communicator.recv(local, 0)
        return [local]

    def before_run() -> None:
        # Warmup sends 17; every measured run must transfer the new payload.
        """Refresh the measured payload so warmup data cannot satisfy
        validation.
        """
        if rank == 0:
            local.fill_(31)
        else:
            local.zero_()

    fields = _profile(operation, [torch.full_like(local, 31)], graph, before_run=before_run)
    _assert_fields(fields, "send" if rank == 0 else "recv", 32, "pp", "nccl_send_recv", graph)
    if fields:
        assert fields[0]["nccl.src_rank"] == 0
        assert fields[0]["nccl.dst_rank"] == 1
        assert fields[0]["nccl.input_shape" if rank == 0 else "nccl.output_shape"] == [2, 4]
        assert fields[0]["nccl.dtype"] == "Float"
    return True


def _run_process_group() -> bool:
    """Verify the process-group dispatcher path does not claim native probe
    metadata.
    """
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


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("selection", "selected"),
    [
        (None, True),
        ("", True),
        (" \t", True),
        ("unknown", True),
        ("nccl", True),
        ("none", False),
        ("mpi", False),
    ],
)
def test_nccl_enabled_uses_cached_startup_selection(
    monkeypatch: pytest.MonkeyPatch, selection: str | None, selected: bool
) -> None:
    """Use the product startup loader, including its default and invalid fallback."""
    if selection is None:
        monkeypatch.delenv("TLLM_PROFILE_CUSTOM_REGIONS", raising=False)
    else:
        monkeypatch.setenv("TLLM_PROFILE_CUSTOM_REGIONS", selection)
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", profiler._load_custom_regions())
    # The test expectation must follow the same immutable selection as the probes.
    monkeypatch.setenv("TLLM_PROFILE_CUSTOM_REGIONS", "none" if selected else "nccl")
    assert profiler.custom_region_enabled("nccl") is selected
    assert _enabled() is (BuildInfo.ENABLE_NCCL_PROFILING and selected)


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
@pytest.mark.parametrize(
    ("process_group", "graph"),
    [(False, False), (False, True), (True, False)],
    ids=["point-to-point", "point-to-point-capture", "process-group"],
)
def test_nccl_owner_boundaries(
    mpi_pool_executor: MPIPoolExecutor, process_group: bool, graph: bool
) -> None:
    """P2P observes actual endpoints; ProcessGroup must not acquire a raw-NCCL label."""
    expected = (BuildInfo.ENABLE_NCCL_PROFILING, profiler.custom_region_enabled("nccl"))
    futures = [
        mpi_pool_executor.submit(_run_process_group)
        if process_group
        else mpi_pool_executor.submit(_run_point_to_point, graph, expected)
        for _ in range(2)
    ]
    assert [future.result() for future in futures] == [True, True]

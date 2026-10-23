# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU request transport and retirement tests.

Run the MPI cases with ``mpirun -n 4 python -m pytest <this file>``.
"""

import contextlib
import gc
import os
import pickle
import time
from collections import namedtuple
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.distributed.communicator import MPIDist
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import ExecutorRequest
from tensorrt_llm._torch.pyexecutor.request_utils import (
    RequestBroadcaster,
    attach_py_objects_to_requests,
)
from tensorrt_llm._torch.pyexecutor.tp_bcast_handles import TpBroadcastHandles
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.mapping import Mapping

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
DIST = MPIDist(Mapping(world_size=COMM.Get_size(), rank=RANK, tp_size=COMM.Get_size()))
ROOT_PID: int | None = None
FD_COUNTS: list[int] = []
pytestmark = pytest.mark.cpu_only
Metadata = namedtuple("Metadata", ("label", "size"))


class Pause:
    pause = staticmethod(contextlib.nullcontext)


def _tensors(device: str = "cpu") -> list[torch.Tensor]:
    dtypes = (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
    tensors = [torch.arange(24).reshape(4, 6).to(dtype=dtype, device=device) for dtype in dtypes]
    tensors += [
        torch.arange(40, device=device).reshape(5, 8)[1:5:2, 1:7:2],
        torch.tensor(3.5, device=device),
        torch.tensor([0x7FC00123, 0x80000000, 0x7F800000], dtype=torch.int64, device=device)
        .to(torch.int32)
        .view(torch.float32),
    ]
    return tensors


def _bits(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _request(data: dict | None) -> list[RequestQueueItem]:
    request = ExecutorRequest(input_token_ids=[1, 2], max_tokens=4)
    request.py_multimodal_data = data
    request.py_conversation_params = None
    return [RequestQueueItem(id=41, request=request)]


def _collect() -> None:
    gc.collect()
    COMM.Barrier()


def _assert_clean() -> None:
    _collect()
    deadline = time.monotonic() + 15
    owned = []
    if RANK == 0:
        while time.monotonic() < deadline:
            owned = list(Path("/dev/shm").glob("torch_" + str(ROOT_PID) + "_*"))
            if not owned:
                break
            gc.collect()
            time.sleep(0.05)
    owned = COMM.bcast([str(path) for path in owned], root=0)
    assert not owned, owned


def _broadcaster(handles: bool = True) -> RequestBroadcaster:
    broadcaster = RequestBroadcaster(DIST, Pause())
    if not handles:
        broadcaster.tp_handles = None
    return broadcaster


def _transfer(handles: bool, variants: bool, delay: bool) -> None:
    broadcaster = _broadcaster(handles)
    if RANK == 0:
        values = _tensors() if variants else [torch.arange(4096)]
        data = dict(
            image=dict(pixel_values=values[0]), nested=values[1:], metadata=[1, None, "keep"]
        )
        requests = _request(data)
    else:
        requests = []
    requests, objects = broadcaster.broadcast(requests)
    attach_py_objects_to_requests(requests, objects)
    actual = requests[0].request.py_multimodal_data
    expected = _tensors() if variants else [torch.arange(4096)]
    tensors = [actual["image"]["pixel_values"], *actual["nested"]]
    for left, right in zip(expected, tensors):
        assert (left.dtype, left.shape, left.stride(), _bits(left)) == (
            right.dtype,
            right.shape,
            right.stride(),
            _bits(right),
        )
        if right.is_cuda:
            # Consumers may use a different GPU from the exporting rank.
            assert _bits(right.to(torch.cuda.current_device())) == _bits(left)
        if RANK and handles:
            assert right.is_shared()
    if delay and RANK == 3:
        # Other ranks discard their request while the final consumer keeps reading.
        time.sleep(0.2)
        assert _bits(tensors[0]) == _bits(expected[0])
    assert actual["metadata"] == [1, None, "keep"]


def _roundtrip(handles: bool, variants: bool = False, delay: bool = False) -> None:
    _transfer(handles, variants, delay)
    _assert_clean()
    FD_COUNTS.append(len(list(Path("/proc/self/fd").iterdir())))


def _wire(
    handles: bool = True,
    remote: bool = False,
    unsupported: bool = False,
    special: str | None = None,
) -> list[bytes]:
    broadcaster = _broadcaster(handles)
    if remote:
        broadcaster.tp_handles.same_host = False
    values = torch.arange(1024).to(torch.complex64 if unsupported else torch.int64)
    if special == "empty":
        values = torch.empty(0)
    elif special == "negative":
        values = torch._neg_view(values)
    requests = _request(dict(image=values)) if RANK == 0 else []
    objects = broadcaster._collect_py_objects(requests) if RANK == 0 else None
    payloads = []
    original = DIST.broadcast

    def capture(value: Any, **kwargs: Any) -> Any:
        if RANK == 0:
            payloads.append(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        return original(value, **kwargs)

    with patch.object(DIST, "broadcast", capture):
        result = broadcaster._broadcast_requests(requests, objects)
    assert torch.equal(dict(result[1])["py_multimodal_data"][41]["image"], values)
    return payloads


def test_default_and_fallback() -> None:
    default = _wire()
    byte_transport = _wire(handles=False)
    compact = len(default[0]) < len(byte_transport[0]) / 2 if RANK == 0 else None
    assert COMM.bcast(compact, root=0), "Default serialized tensor data"
    remote = _wire(remote=True)
    assert remote == byte_transport, "Multi-node fallback changed serialized bytes"
    # Exercise actual hostname detection independently of the transport fallback.
    helper = TpBroadcastHandles(DIST)
    with patch("socket.gethostname", return_value="host-" + str(RANK // 2)):
        result = helper.broadcast([], ())
    assert helper.same_host is False and result == ([], ())
    _wire(unsupported=True)
    for special in ("empty", "negative"):
        _wire(special=special)
    for pp_size in (1, 2):
        local = MPIDist(
            Mapping(
                world_size=1 if pp_size == 1 else 4,
                rank=0,
                tp_size=1 if pp_size == 1 else 2,
                pp_size=pp_size,
            )
        )
        assert RequestBroadcaster(local, Pause()).tp_handles is None
    _assert_clean()


def _abort(mapping: bool, after_open: bool = False) -> None:
    broadcaster = RequestBroadcaster(DIST, Pause())
    requests = _request(dict(image=_tensors())) if RANK == 0 else []
    calls = 0
    method = "get_local_view" if after_open else "from_dict" if mapping else "dump_to_dict"
    original = getattr(SharedTensorContainer, method)

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2 and RANK == (2 if mapping else 0):
            raise RuntimeError("Deliberate graceful abort")
        return original(*args, **kwargs)

    failed = False
    with patch.object(SharedTensorContainer, method, fail_once):
        try:
            broadcaster.broadcast(requests)
        except RuntimeError:
            failed = True
    assert failed, "An abort must reach every rank"
    requests.clear()
    _assert_clean()


def _serialization_abort(device: str = "cpu") -> None:
    broadcaster = RequestBroadcaster(DIST, Pause())
    requests = _request(dict(image=torch.arange(4096, device=device))) if RANK == 0 else []
    if RANK == 0:
        requests[0].request.py_logits_post_processors = lambda value: value
    failed = False
    try:
        broadcaster.broadcast(requests)
    except RuntimeError:
        failed = True
    assert failed, "Serialization failure must reach every rank"
    requests.clear()
    _assert_clean()


def _metadata_roundtrip(handles: bool) -> None:
    broadcaster = _broadcaster(handles)
    requests = []
    if RANK == 0:
        requests = _request(
            dict(
                image=torch.arange(6).reshape(2, 3),
                metadata=Metadata("image", torch.Size((2, 3))),
            )
        )
        requests[0].request.py_scheduling_params = Metadata("schedule", torch.Size((4,)))
    requests, objects = broadcaster.broadcast(requests)
    attach_py_objects_to_requests(requests, objects)
    metadata = requests[0].request.py_multimodal_data["metadata"]
    other = requests[0].request.py_scheduling_params
    assert isinstance(metadata, Metadata) and isinstance(metadata.size, torch.Size), (
        type(metadata),
        type(metadata.size),
    )
    assert isinstance(other, Metadata) and isinstance(other.size, torch.Size), (
        type(other),
        type(other.size),
    )
    assert metadata.label == "image" and metadata.size == (2, 3)
    assert other.label == "schedule" and other.size == (4,)


def test_metadata_types() -> None:
    for handles in (False, True):
        _metadata_roundtrip(handles)
        _assert_clean()


@pytest.mark.parametrize("handles", [False, True], ids=["bytes", "automatic_handles"])
def test_tensor_variants(handles: bool) -> None:
    _roundtrip(handles, variants=True)


def test_last_consumer_lifetime() -> None:
    _roundtrip(True, delay=True)


@pytest.mark.parametrize("mapping", [False, True])
def test_partial_abort(mapping: bool) -> None:
    _abort(mapping)


def test_serialization_abort() -> None:
    _serialization_abort()


def test_retirement_cycles() -> None:
    FD_COUNTS.clear()
    for cycle in range(20):
        _roundtrip(True, delay=cycle % 5 == 0)
    stable = FD_COUNTS[-20:]
    assert max(stable[2:]) <= stable[1], stable


def test_activation() -> None:
    for dist in (DIST, object()):
        enabled = dist is DIST and COMM.Get_size() > 1
        assert (RequestBroadcaster(dist, Pause()).tp_handles is not None) == enabled
    for tp, pp, cp in [(1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 1, 2)]:
        dist = MPIDist(Mapping(world_size=tp * pp * cp, rank=0, tp_size=tp, pp_size=pp, cp_size=cp))
        enabled = tp > 1 and pp == cp == 1
        assert (RequestBroadcaster(dist, Pause()).tp_handles is not None) == enabled


@pytest.mark.parametrize("data", [None, {"metadata": [1, "image", None]}])
def test_requests_without_tensor_payload(data: dict | None) -> None:
    requests = _request(data) if RANK == 0 else []
    with patch.object(SharedTensorContainer, "from_tensor", side_effect=AssertionError):
        requests, objects = RequestBroadcaster(DIST, Pause()).broadcast(requests)
    attach_py_objects_to_requests(requests, objects)
    if data is None:
        assert "py_multimodal_data" not in dict(objects)
    else:
        assert requests[0].request.py_multimodal_data == data


def test_view_failure_after_storage_open() -> None:
    # CPU from_dict has consumed the exported reference before view creation.
    # A failing view must not cause rank 0 to retire that reference a second time.
    retired = 0

    def record_retirement(_container: SharedTensorContainer) -> None:
        nonlocal retired
        retired += 1

    # Do not retain the container in mock call history during the lifetime check.
    with patch("tensorrt_llm._torch.pyexecutor.tp_bcast_handles._retire", record_retirement):
        _abort(mapping=True, after_open=True)
    retired = COMM.bcast(retired if RANK == 0 else None, root=0)
    assert retired == 0, "An opened CPU storage reference was retired twice"


@pytest.fixture(autouse=True)
def _mpi_case(request: pytest.FixtureRequest) -> Iterator[None]:
    global ROOT_PID
    if request.node.originalname != "test_activation" and COMM.Get_size() != 4:
        pytest.skip("Requires four MPI ranks; run with mpirun -n 4")
    if request.node.originalname != "test_activation":
        ROOT_PID = COMM.bcast(os.getpid() if RANK == 0 else None, root=0)
    with patch.dict(os.environ):
        yield

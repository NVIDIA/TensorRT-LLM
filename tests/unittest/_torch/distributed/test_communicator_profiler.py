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

"""MPI metadata contracts; real two-rank smoke runs under mpirun -n 2."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn, cast, get_args, get_type_hints
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch
from profiler_test_utils import decode_profiler_metadata

from tensorrt_llm import _profiler as profiler
from tensorrt_llm._torch.distributed import communicator

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


class _MpiGroup:
    """Model mpi4py's classmethod requiring an explicit source MPI group."""

    def __init__(self, members: list[int], failure: Exception | None = None) -> None:
        self.translate = Mock(return_value=members, side_effect=failure)
        self.free = Mock()

    @classmethod
    def Translate_ranks(
        cls,
        group1: "_MpiGroup",
        ranks1: Sequence[int],
        group2: "_MpiGroup | None" = None,
    ) -> list[int]:
        if not isinstance(group1, cls) or not isinstance(group2, cls):
            raise TypeError("Expected MPI.Group source and destination")
        return cast(list[int], group1.translate(group1, ranks1, group2))

    def Free(self) -> None:
        self.free()


def _dist() -> communicator.MPIDist:
    mapping = SimpleNamespace(
        rank=99,
        world_size=4,
        tp_size=2,
        cp_size=2,
        tp_group=[1, 3],
        pp_group=[3, 7],
        cp_group=[2, 3],
    )
    return communicator.MPIDist(cast("Mapping", mapping))


@pytest.fixture
def active(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, MagicMock, Mock]:
    scope = MagicMock()
    record = Mock(return_value=scope)
    parent = Mock()
    parent.Get_rank.return_value = 3
    parent.Get_size.return_value = 8
    parent.Get_group.return_value = _MpiGroup(list(range(8)))
    monkeypatch.setattr(communicator.MPI, "Group", _MpiGroup, raising=False)
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"mpi"}))
    monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: True)
    monkeypatch.setattr(torch.profiler, "record_function", record)
    monkeypatch.setattr(communicator, "mpi_comm", lambda: parent)
    return record, scope, parent


def _fields(record: Mock, index: int = -1) -> dict[str, profiler.MetadataValue]:
    region, metadata = decode_profiler_metadata(record.call_args_list[index].kwargs["args"])
    assert region == "mpi"
    return metadata


@pytest.mark.parametrize("selected,enabled", [(False, True), (True, False)])
def test_inactive_scope_inspects_neither_group_rank_nor_payload(
    monkeypatch: pytest.MonkeyPatch,
    selected: bool,
    enabled: bool,
) -> None:
    group = Mock(side_effect=AssertionError("group inspected"))
    rank = Mock(side_effect=AssertionError("rank inspected"))
    payload = MagicMock()
    record = Mock()
    monkeypatch.setattr(
        profiler, "_CUSTOM_REGIONS", frozenset({"mpi"}) if selected else frozenset()
    )
    monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: enabled)
    monkeypatch.setattr(torch.profiler, "record_function", record)
    with profiler.record_mpi_function(
        "allgather", communicator="world", group=group, rank=rank, payload=payload
    ) as value:
        assert value is None
    group.assert_not_called()
    rank.assert_not_called()
    record.assert_not_called()
    assert payload.mock_calls == []


def test_world_metadata_uses_active_parent_not_mapping(
    active: tuple[Mock, MagicMock, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, _, _ = active
    payload = np.ones((2, 3), dtype=np.float32)
    result = [payload]
    gather = Mock(return_value=result)
    monkeypatch.setattr(communicator, "mpi_allgather", gather)
    assert _dist().allgather(payload) is result
    gather.assert_called_once_with(payload)
    assert record.call_args.args == ("trtllm::mpi_allgather",)
    assert _fields(record) == {
        "mpi.operation": "allgather",
        "mpi.communicator": "world",
        "mpi.group": "0,1,2,3,4,5,6,7",
        "mpi.kind": "collective",
        "mpi.rank": 3,
        "mpi.payload.type": "ndarray",
        "mpi.payload.shape": [2, 3],
        "mpi.payload.dtype": "float32",
        "mpi.payload.numel": 6,
        "mpi.payload.bytes": 24,
    }


@pytest.mark.parametrize(
    "payload,want",
    [
        (None, {"type": "NoneType"}),
        (7, {"type": "int"}),
        (np.int64(7), {"type": "int64"}),
        ("é", {"type": "str", "length": 1}),
        (b"abc", {"type": "bytes", "bytes": 3}),
        (bytearray(3), {"type": "bytearray", "bytes": 3}),
        (memoryview(np.ones((2, 3), dtype=np.int64)), {"type": "memoryview", "bytes": 48}),
        ([object()], {"type": "list", "length": 1}),
        ((object(),), {"type": "tuple", "length": 1}),
        ({"nested": object()}, {"type": "dict", "length": 1}),
        ({7}, {"type": "set", "length": 1}),
    ],
)
def test_payload_description_is_shallow(
    active: tuple[Mock, MagicMock, Mock],
    payload: object,
    want: dict[str, profiler.MetadataValue],
) -> None:
    record, _, _ = active
    with profiler.record_mpi_function(
        "send", communicator="world", group=lambda: [3], rank=3, payload=payload
    ):
        pass
    assert {
        key.removeprefix("mpi.payload."): val
        for key, val in _fields(record).items()
        if key.startswith("mpi.payload.")
    } == want


def test_tensor_metadata_does_not_read_values(active: tuple[Mock, MagicMock, Mock]) -> None:
    record, _, _ = active
    payload = torch.ones((3, 2)).transpose(0, 1)
    with profiler.record_mpi_function(
        "send", communicator="world", group=lambda: [3], rank=3, payload=payload
    ):
        pass
    fields = _fields(record)
    assert {
        key.removeprefix("mpi.payload."): val
        for key, val in fields.items()
        if key.startswith("mpi.payload.")
    } == {
        "type": "Tensor",
        "shape": [2, 3],
        "dtype": "torch.float32",
        "stride": [1, 2],
        "device": "cpu",
        "numel": 6,
        "bytes": 24,
    }


def test_direct_mpi_scalars_normalize_without_schema_change(
    active: tuple[Mock, MagicMock, Mock],
) -> None:
    record, _, _ = active
    with profiler.record_mpi_function(
        "allreduce",
        communicator="tp",
        group=lambda: [1, 3],
        rank=np.int64(3),
        root=np.int32(1),
        reduce_op=communicator.ReduceOp.MAX,
        factor=np.float32(1.5),
    ):
        pass
    fields = _fields(record)
    assert fields["mpi.rank"] == 3
    assert fields["mpi.root"] == 1
    assert fields["mpi.reduce_op"] == 3
    assert fields["mpi.factor"] == 1.5


@pytest.mark.parametrize("group_name,members", [("tp", [3, 1]), ("pp", [3, 7]), ("cp", [2, 3])])
def test_subgroup_inspection_uses_cached_group_and_frees_temporaries(
    active: tuple[Mock, MagicMock, Mock],
    group_name: str,
    members: list[int],
) -> None:
    record, _, parent = active
    mpi_dist = _dist()
    subcomm = Mock()
    subcomm.Get_size.return_value = 2
    subcomm.Get_group.return_value = _MpiGroup(members)
    setattr(mpi_dist, f"_{group_name}_comm", subcomm)
    with mpi_dist._record_mpi("barrier", group_name):
        pass
    assert _fields(record)["mpi.group"] == ",".join(map(str, members))
    assert _fields(record)["mpi.rank"] == 3
    subcomm.Get_group.return_value.translate.assert_called_once_with(
        subcomm.Get_group.return_value,
        range(2),
        parent.Get_group.return_value,
    )
    subcomm.Get_group.return_value.free.assert_called_once_with()
    parent.Get_group.return_value.free.assert_called_once_with()
    parent.Create_group.assert_not_called()


def test_uninitialized_group_uses_mapping_without_creating_communicator(
    active: tuple[Mock, MagicMock, Mock],
) -> None:
    record, _, parent = active
    with _dist()._record_mpi("tp_barrier", "tp"):
        pass
    assert _fields(record)["mpi.group"] == "1,3"
    parent.Create_group.assert_not_called()
    parent.Get_group.assert_not_called()


def test_uninitialized_tp_metadata_does_not_inspect_other_mapping_groups(
    active: tuple[Mock, MagicMock, Mock],
) -> None:
    record, _, _ = active

    class PartialMapping:
        tp_group = [1, 3]

        @property
        def pp_group(self) -> list[int]:
            raise AssertionError("unrelated PP group inspected")

        @property
        def cp_group(self) -> list[int]:
            raise AssertionError("unrelated CP group inspected")

    mpi_dist = _dist()
    mpi_dist.mapping = cast("Mapping", PartialMapping())
    with mpi_dist._record_mpi("tp_barrier", "tp"):
        pass
    assert _fields(record)["mpi.group"] == "1,3"


def test_group_inspection_failure_does_not_replace_collective_exception(
    active: tuple[Mock, MagicMock, Mock],
) -> None:
    record, _, parent = active
    mpi_dist = _dist()
    subcomm = Mock()
    subcomm.Get_size.return_value = 2
    subcomm.Get_group.return_value = _MpiGroup([1, 3], RuntimeError("unsupported"))
    failure = ValueError("collective failed")
    subcomm.allgather.side_effect = failure
    mpi_dist._tp_comm = subcomm
    with pytest.raises(ValueError) as caught:
        mpi_dist.tp_allgather("x", small_payload=True)
    assert caught.value is failure
    subcomm.allgather.assert_called_once_with("x")
    subcomm.Get_group.return_value.free.assert_called_once_with()
    parent.Get_group.return_value.free.assert_called_once_with()
    record.assert_not_called()


def test_unmapped_cached_group_discards_metadata_not_result(
    active: tuple[Mock, MagicMock, Mock],
) -> None:
    record, _, parent = active
    mpi_dist = _dist()
    subcomm = Mock()
    subcomm.Get_size.return_value = 2
    subcomm.Get_group.return_value = _MpiGroup([1, -32766])
    mpi_dist._tp_comm = subcomm
    assert mpi_dist.tp_allgather("x", small_payload=True) is subcomm.allgather.return_value
    record.assert_not_called()
    subcomm.Get_group.return_value.free.assert_called_once_with()
    parent.Get_group.return_value.free.assert_called_once_with()


def test_nested_mpi_scalars_are_not_recursively_normalized(
    active: tuple[Mock, MagicMock, Mock],
) -> None:
    record, _, _ = active
    with profiler.record_mpi_function(
        "allreduce", communicator="world", group=lambda: [3], rank=3, nested=[np.int64(3)]
    ):
        result = 7
    assert result == 7
    record.assert_not_called()


def test_container_payload_never_iterates_nested_tensors(
    active: tuple[Mock, MagicMock, Mock],
) -> None:
    record, _, _ = active

    class ShallowDict(dict[str, object]):
        def items(self) -> NoReturn:
            raise AssertionError("payload traversed")

        def __iter__(self) -> NoReturn:
            raise AssertionError("payload iterated")

    with profiler.record_mpi_function(
        "send",
        communicator="world",
        group=lambda: [3],
        rank=3,
        payload=ShallowDict(tensor=torch.ones(3)),
    ):
        pass
    assert _fields(record)["mpi.payload.length"] == 1


@pytest.mark.parametrize(
    "operation,helper,direction",
    [
        ("send", "mpi_send", "send"),
        ("isend", "mpi_isend", "send"),
        ("recv", "mpi_recv", "recv"),
        ("send_object", "mpi_send_object", "send"),
        ("isend_object", "mpi_isend_object", "send"),
        ("recv_object", "mpi_recv_object", "recv"),
    ],
)
def test_p2p_arguments_results_and_exception_identity(
    active: tuple[Mock, MagicMock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    helper: str,
    direction: str,
) -> None:
    record, scope, _ = active
    failure = RuntimeError("transport failed")
    transport = Mock(side_effect=failure)
    monkeypatch.setattr(communicator, helper, transport)
    payload = np.ones(4, dtype=np.uint8)
    args = (7, 11) if operation == "recv_object" else (payload, 7, 11)
    with pytest.raises(RuntimeError) as caught:
        getattr(_dist(), operation)(*args)
    assert caught.value is failure
    transport.assert_called_once_with(*args)
    fields = _fields(record)
    assert fields["mpi.kind"] == "p2p"
    assert fields["mpi.direction"] == direction
    assert fields["mpi.peer"] == 7
    assert fields["mpi.tag"] == 11
    assert ("mpi.payload.type" in fields) is (operation != "recv_object")
    scope.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize(
    "operation,helper",
    [
        ("send", "mpi_send"),
        ("isend", "mpi_isend"),
        ("recv", "mpi_recv"),
        ("send_object", "mpi_send_object"),
        ("isend_object", "mpi_isend_object"),
        ("recv_object", "mpi_recv_object"),
    ],
)
def test_p2p_returns_preserve_submission_handle_or_receive_result(
    active: tuple[Mock, MagicMock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    helper: str,
) -> None:
    transport = Mock(return_value=object())
    monkeypatch.setattr(communicator, helper, transport)
    payload = np.ones(4, dtype=np.uint8)
    args = (7, 11) if operation == "recv_object" else (payload, 7, 11)
    result = getattr(_dist(), operation)(*args)
    assert result is (None if operation in ("send", "send_object") else transport.return_value)
    transport.assert_called_once_with(*args)


@pytest.mark.parametrize("group_name", ["tp", "cp"])
@pytest.mark.parametrize("small", [False, True])
def test_small_payload_retains_native_or_chunked_collective(
    active: tuple[Mock, MagicMock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    group_name: str,
    small: bool,
) -> None:
    record, _, _ = active
    mpi_dist = _dist()
    subcomm = Mock()
    subcomm.Get_size.return_value = 2
    subcomm.Get_group.return_value = _MpiGroup([1, 3])
    setattr(mpi_dist, f"_{group_name}_comm", subcomm)
    payload = {"x": 1}
    result = [payload]
    subcomm.allgather.return_value = result
    chunked = Mock(return_value=result)
    monkeypatch.setattr(communicator, "safe_allgather", chunked)
    assert (
        getattr(mpi_dist, f"{group_name}_allgather")(
            payload,
            chunk_size=128,
            small_payload=small,
        )
        is result
    )
    if small:
        subcomm.allgather.assert_called_once_with(payload)
        chunked.assert_not_called()
    else:
        chunked.assert_called_once_with(subcomm, payload, chunk_size=128)
        subcomm.allgather.assert_not_called()
    assert _fields(record)["mpi.operation"] == f"{group_name}_allgather"


def test_typed_broadcast_retains_tp_then_cp_and_int64_buffer(
    active: tuple[Mock, MagicMock, Mock],
) -> None:
    record, _, _ = active
    mpi_dist = _dist()
    calls: list[tuple[str, int, str]] = []
    for name, members in [("tp", [1, 3]), ("cp", [2, 3])]:
        comm = Mock()
        comm.Get_size.return_value = 2
        comm.Get_group.return_value = _MpiGroup(members)

        def bcast(buf: list[object], *, root: int, label: str = name) -> None:
            assert buf[1] == communicator.MPI.INT64_T
            arr = cast(np.ndarray, buf[0])
            calls.append((label, root, str(arr.dtype)))
            arr[:] += 1

        comm.Bcast.side_effect = bcast
        setattr(mpi_dist, f"_{name}_comm", comm)
    result = mpi_dist.tp_cp_broadcast_int64([3, 4], root=1)
    np.testing.assert_array_equal(result, [5, 6])
    assert calls == [("tp", 1, "int64"), ("cp", 1, "int64")]
    assert [_fields(record, i)["mpi.operation"] for i in range(2)] == [
        "tp_broadcast_int64",
        "cp_broadcast_int64",
    ]


@pytest.mark.parametrize(
    "operation,group_name,helper",
    [
        ("broadcast", "world", "safe_broadcast"),
        ("tp_broadcast", "tp", "safe_broadcast"),
        ("cp_broadcast", "cp", "safe_broadcast"),
        ("tp_gather", "tp", "safe_gather"),
        ("pp_gather", "pp", "safe_gather"),
        ("pp_allgather", "pp", "safe_allgather"),
    ],
)
def test_chunked_collective_preserves_arguments_and_result(
    active: tuple[Mock, MagicMock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    group_name: str,
    helper: str,
) -> None:
    record, _, parent = active
    mpi_dist = _dist()
    subcomm = Mock()
    subcomm.Get_size.return_value = 2
    subcomm.Get_group.return_value = _MpiGroup([3, 7])
    if group_name != "world":
        setattr(mpi_dist, f"_{group_name}_comm", subcomm)
    comm = parent if group_name == "world" else subcomm
    result = object()
    transport = Mock(return_value=result)
    monkeypatch.setattr(communicator, helper, transport)
    kwargs = {"chunk_size": 128}
    if operation != "pp_allgather":
        kwargs["root"] = 1
    assert getattr(mpi_dist, operation)("payload", **kwargs) is result
    transport.assert_called_once_with(comm, "payload", **kwargs)
    assert _fields(record)["mpi.operation"] == operation
    assert _fields(record)["mpi.chunk_bytes"] == 128


@pytest.mark.parametrize(
    "operation,group_name",
    [
        ("allreduce", "world"),
        ("tp_allreduce", "tp"),
        ("barrier", "world"),
        ("tp_barrier", "tp"),
        ("pp_broadcast", "pp"),
    ],
)
def test_direct_collective_preserves_reduction_and_root(
    active: tuple[Mock, MagicMock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    group_name: str,
) -> None:
    record, _, parent = active
    mpi_dist = _dist()
    subcomm = Mock()
    subcomm.Get_size.return_value = 2
    subcomm.Get_group.return_value = _MpiGroup([3, 7])
    setattr(mpi_dist, f"_{group_name}_comm", subcomm)
    comm = parent if group_name == "world" else subcomm
    if "allreduce" in operation:
        result = (
            mpi_dist.allreduce(7, communicator.ReduceOp.MAX)
            if group_name == "world"
            else mpi_dist.tp_allreduce(7, communicator.ReduceOp.MAX)
        )
        assert result is comm.allreduce.return_value
        comm.allreduce.assert_called_once_with(7, communicator.MPI.MAX)
        assert _fields(record)["mpi.reduce_op"] == 3
    elif operation == "pp_broadcast":
        assert mpi_dist.pp_broadcast("x", root=1) is comm.bcast.return_value
        comm.bcast.assert_called_once_with("x", 1)
        assert _fields(record)["mpi.root"] == 1
    elif operation == "barrier":
        transport = Mock()
        monkeypatch.setattr(communicator, "mpi_barrier", transport)
        assert mpi_dist.barrier() is None
        transport.assert_called_once_with()
    else:
        assert mpi_dist.tp_barrier() is None
        comm.Barrier.assert_called_once_with()
    assert _fields(record)["mpi.operation"] == operation


@pytest.mark.parametrize("group_name", ["tp", "cp"])
def test_typed_allgather_uses_one_int64_collective(
    active: tuple[Mock, MagicMock, Mock],
    group_name: str,
) -> None:
    record, _, _ = active
    mpi_dist = _dist()
    subcomm = Mock()
    subcomm.Get_size.return_value = 2
    subcomm.Get_group.return_value = _MpiGroup([1, 3])
    setattr(mpi_dist, f"_{group_name}_comm", subcomm)

    def allgather(send: list[object], recv: list[object]) -> None:
        assert send[1] == recv[1] == communicator.MPI.INT64_T
        np.testing.assert_array_equal(send[0], [3, 4])
        cast(np.ndarray, recv[0])[:] = [3, 4, 5, 6]

    subcomm.Allgather.side_effect = allgather
    result = getattr(mpi_dist, f"{group_name}_allgather_int64")([3, 4])
    np.testing.assert_array_equal(result, [[3, 4], [5, 6]])
    assert result.dtype == np.int64
    assert subcomm.Allgather.call_count == 1
    subcomm.allgather.assert_not_called()
    assert _fields(record)["mpi.operation"] == f"{group_name}_allgather_int64"


def test_inactive_wrapper_does_not_resolve_parent_for_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset())
    parent = Mock(side_effect=AssertionError("parent inspected"))
    monkeypatch.setattr(communicator, "mpi_comm", parent)
    result = object()
    monkeypatch.setattr(communicator, "mpi_allgather", lambda obj: result)
    assert _dist().allgather(object()) is result
    parent.assert_not_called()


@pytest.mark.parametrize(
    "operation,helper", [("isend", "mpi_isend"), ("isend_object", "mpi_isend_object")]
)
def test_nonblocking_send_return_contract_includes_disabled_none(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    helper: str,
) -> None:
    method = getattr(communicator.MPIDist, operation)
    assert type(None) in get_args(get_type_hints(method)["return"])
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset())
    monkeypatch.setattr(communicator, helper, Mock(return_value=None))
    assert getattr(_dist(), operation)(np.ones(2, dtype=np.uint8), 1) is None


def test_torch_backend_is_not_labeled_as_mpi(
    active: tuple[Mock, MagicMock, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, _, _ = active
    torch_dist = object.__new__(communicator.TorchDist)
    mapping = Mock()
    mapping.has_cp_ulysses.return_value = False
    torch_dist.mapping = mapping
    monkeypatch.setattr(communicator, "mpi_disabled", lambda: True)
    broadcast = Mock()
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    payload = torch.ones(3)
    assert torch_dist.broadcast(payload, root=1) is payload
    broadcast.assert_called_once_with(payload, src=1)
    record.assert_not_called()


def test_two_rank_collective_p2p_profiler_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate real MPI results and real CPU user scopes on every rank."""
    comm = communicator.mpi_comm()
    if comm.Get_size() != 2:
        pytest.skip("Requires exactly two real MPI ranks")
    from tensorrt_llm.mapping import Mapping

    rank = comm.Get_rank()
    mpi_dist = communicator.MPIDist(Mapping(world_size=2, rank=rank, tp_size=2))
    captured: list[str] = []
    real_record = torch.profiler.record_function

    def recording(name: str, args: str) -> torch.profiler.record_function:
        captured.append(args)
        return real_record(name, args=args)

    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"mpi"}))
    monkeypatch.setattr(torch.profiler, "record_function", recording)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        assert mpi_dist.tp_allgather(rank, small_payload=True) == [0, 1]
        np.testing.assert_array_equal(mpi_dist.tp_allgather_int64([rank]), [[0], [1]])
        if rank == 0:
            mpi_dist.send_object({"value": 7}, 1, tag=17)
        else:
            assert mpi_dist.recv_object(0, tag=17) == {"value": 7}
        mpi_dist.barrier()
    assert "trtllm::mpi_tp_allgather" in {event.name for event in trace.events()}
    decoded = [decode_profiler_metadata(value)[1] for value in captured]
    assert len(decoded) == 4
    assert all(fields["mpi.rank"] == rank and fields["mpi.group"] == "0,1" for fields in decoded)

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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.distributed import communicator


def _mapping(**overrides: object) -> SimpleNamespace:
    values = {
        "world_size": 8,
        "rank": 5,
        "moe_ep_group": list(range(8)),
        "moe_ep_size": 8,
        "moe_ep_rank": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _valid_topologies(parent_size: int = 8) -> list[dict[str, object]]:
    topologies = []
    for rank in range(parent_size):
        ep_group = tuple(range(parent_size))
        topologies.append(
            {
                "parent_rank": rank,
                "parent_size": parent_size,
                "world_size": parent_size,
                "mapping_rank": rank,
                "ep_group": ep_group,
                "ep_size": len(ep_group),
                "ep_rank": ep_group.index(rank),
                "health_size": None,
                "error_kind": None,
                "error_message": None,
            }
        )
    return topologies


def _valid_setup_statuses(parent_size: int = 8) -> list[dict[str, object]]:
    return [
        {
            "parent_rank": rank,
            "error_message": None,
            "ulfm_available": True,
        }
        for rank in range(parent_size)
    ]


def _communicators(*, parent_rank: int = 5, parent_size: int = 8) -> tuple[Mock, Mock]:
    ft_comm = Mock()
    ft_comm.Get_rank.return_value = 5
    ft_comm.Get_size.return_value = 8
    ft_comm.Is_revoked.return_value = False

    parent_comm = Mock()
    parent_comm.Get_rank.return_value = parent_rank
    parent_comm.Get_size.return_value = parent_size
    parent_comm.Split.return_value = ft_comm

    def _allgather(local_record: dict[str, object]) -> list[dict[str, object]]:
        if "ep_group" in local_record:
            records = _valid_topologies(parent_size)
        else:
            records = _valid_setup_statuses(parent_size)
        records[parent_rank] = local_record
        return records

    parent_comm.allgather.side_effect = _allgather
    return parent_comm, ft_comm


def _fake_mpi(parent_comm: Mock, *, thread_level: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        COMM_WORLD=parent_comm,
        ERRORS_RETURN=object(),
        THREAD_MULTIPLE=3,
        Exception=Exception,
        Query_thread=Mock(return_value=thread_level),
    )


def test_create_mpi_ft_subcomm_splits_ep_group_and_sets_error_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, ft_comm = _communicators()
    fake_mpi = _fake_mpi(parent_comm)
    monkeypatch.setattr(communicator, "MPI", fake_mpi)
    active_comm = Mock(return_value=parent_comm)
    monkeypatch.setattr(communicator, "mpi_comm", active_comm)
    initial_refs = len(communicator._MPI_FT_PROCESS_LIFETIME_REFS)

    try:
        result = communicator.create_mpi_ft_subcomm(_mapping())

        assert result.comm is ft_comm
        assert result.local_rank == 5
        assert result.ep_size == 8
        assert result.ulfm_available is True
        active_comm.assert_called_once_with()
        assert parent_comm.allgather.call_count == 2
        parent_comm.Split.assert_called_once_with(color=0, key=5)
        ft_comm.Set_errhandler.assert_called_once_with(fake_mpi.ERRORS_RETURN)
        ft_comm.Free.assert_not_called()
        assert communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:] == [ft_comm]
    finally:
        del communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:]


def test_create_mpi_ft_subcomm_requires_thread_multiple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, _ = _communicators()
    fake_mpi = _fake_mpi(parent_comm, thread_level=2)
    monkeypatch.setattr(communicator, "MPI", fake_mpi)

    with pytest.raises(RuntimeError, match=r"requires MPI\.THREAD_MULTIPLE"):
        communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

    parent_comm.allgather.assert_called_once()
    parent_comm.Split.assert_not_called()


@pytest.mark.parametrize(
    ("mapping_overrides", "error_match"),
    [
        ({"moe_ep_group": []}, "must not be empty"),
        ({"moe_ep_group": list(range(7))}, "size must match mapping.moe_ep_size"),
        ({"moe_ep_group": [0, 1, 2, 3, 4, 5, 6, 6]}, "contains duplicate ranks"),
        ({"moe_ep_group": [0, 1, 2, 3, 4, 5, 6, 8]}, "outside the parent communicator"),
        (
            {"moe_ep_group": [0, 1, 2, 3], "moe_ep_size": 4},
            "spanning the full MPI world",
        ),
        (
            {"moe_ep_group": [5, 0, 1, 2, 3, 4, 6, 7]},
            "group order must match mapping.moe_ep_rank",
        ),
    ],
)
def test_create_mpi_ft_subcomm_validates_ep_group(
    monkeypatch: pytest.MonkeyPatch,
    mapping_overrides: dict[str, object],
    error_match: str,
) -> None:
    parent_comm, _ = _communicators()
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))

    with pytest.raises(ValueError, match=error_match):
        communicator.create_mpi_ft_subcomm(_mapping(**mapping_overrides), parent_comm)

    parent_comm.allgather.assert_called_once()
    parent_comm.Split.assert_not_called()


@pytest.mark.parametrize(
    ("parent_rank", "parent_size", "error_match"),
    [
        (5, 7, "size must match mapping.world_size"),
        (6, 8, "rank must match mapping.rank"),
    ],
)
def test_create_mpi_ft_subcomm_validates_parent_mapping(
    monkeypatch: pytest.MonkeyPatch,
    parent_rank: int,
    parent_size: int,
    error_match: str,
) -> None:
    parent_comm, _ = _communicators(parent_rank=parent_rank, parent_size=parent_size)
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))

    with pytest.raises(RuntimeError, match=error_match):
        communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

    parent_comm.allgather.assert_called_once()
    parent_comm.Split.assert_not_called()


@pytest.mark.parametrize(
    ("mapping_error", "error_match"),
    [
        (IndexError("missing EP group"), "IndexError: missing EP group"),
        (ValueError("invalid EP group"), "ValueError: invalid EP group"),
    ],
    ids=["index-error", "value-error"],
)
def test_create_mpi_ft_subcomm_collectively_reports_mapping_accessor_error(
    monkeypatch: pytest.MonkeyPatch,
    mapping_error: Exception,
    error_match: str,
) -> None:
    class FailingMapping:
        world_size = 8
        rank = 5
        moe_ep_size = 8
        moe_ep_rank = 5

        @property
        def moe_ep_group(self) -> list[int]:
            raise mapping_error

    parent_comm, _ = _communicators()
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))

    with pytest.raises(ValueError, match=error_match):
        communicator.create_mpi_ft_subcomm(FailingMapping(), parent_comm)

    parent_comm.allgather.assert_called_once()
    parent_comm.Split.assert_not_called()


def test_create_mpi_ft_subcomm_uses_collectively_validated_ep_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SingleReadMapping:
        world_size = 8
        rank = 5
        moe_ep_group = list(range(8))

        def __init__(self) -> None:
            self.ep_size_reads = 0
            self.ep_rank_reads = 0

        @property
        def moe_ep_size(self) -> int:
            self.ep_size_reads += 1
            if self.ep_size_reads > 1:
                raise ValueError("moe_ep_size was read after collective validation")
            return 8

        @property
        def moe_ep_rank(self) -> int:
            self.ep_rank_reads += 1
            if self.ep_rank_reads > 1:
                raise ValueError("moe_ep_rank was read after collective validation")
            return 5

    parent_comm, ft_comm = _communicators()
    fake_mpi = _fake_mpi(parent_comm)
    monkeypatch.setattr(communicator, "MPI", fake_mpi)
    mapping = SingleReadMapping()

    result = communicator.create_mpi_ft_subcomm(mapping, parent_comm)

    assert result.comm is ft_comm
    assert result.local_rank == 5
    assert result.ep_size == 8
    assert result.ulfm_available is True
    assert mapping.ep_size_reads == 1
    assert mapping.ep_rank_reads == 1
    assert parent_comm.allgather.call_count == 2
    ft_comm.Set_errhandler.assert_called_once_with(fake_mpi.ERRORS_RETURN)


@pytest.mark.parametrize(
    ("failure_kind", "error_match"),
    [
        ("rank-mismatch", "unexpected WideEP FT communicator"),
        ("size-mismatch", "unexpected WideEP FT communicator"),
        ("get-rank-error", "RuntimeError: cannot inspect rank"),
        ("get-size-error", "RuntimeError: cannot inspect size"),
        ("errhandler", "RuntimeError: cannot set error handler"),
    ],
)
def test_create_mpi_ft_subcomm_retains_comm_after_post_split_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    error_match: str,
) -> None:
    parent_comm, ft_comm = _communicators()
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))
    initial_refs = len(communicator._MPI_FT_PROCESS_LIFETIME_REFS)
    if failure_kind == "rank-mismatch":
        ft_comm.Get_rank.return_value = 4
    elif failure_kind == "size-mismatch":
        ft_comm.Get_size.return_value = 7
    elif failure_kind == "get-rank-error":
        ft_comm.Get_rank.side_effect = RuntimeError("cannot inspect rank")
    elif failure_kind == "get-size-error":
        ft_comm.Get_size.side_effect = RuntimeError("cannot inspect size")
    else:
        ft_comm.Set_errhandler.side_effect = RuntimeError("cannot set error handler")

    try:
        with pytest.raises(RuntimeError, match=error_match):
            communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

        assert parent_comm.allgather.call_count == 2
        ft_comm.Free.assert_not_called()
        assert communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:] == [ft_comm]
    finally:
        del communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:]


def test_create_mpi_ft_subcomm_installs_error_handler_before_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, ft_comm = _communicators()
    fake_mpi = _fake_mpi(parent_comm)
    monkeypatch.setattr(communicator, "MPI", fake_mpi)
    calls = []
    ft_comm.Set_errhandler.side_effect = lambda _handler: calls.append("errhandler")
    ft_comm.Get_rank.side_effect = lambda: calls.append("rank") or 5
    ft_comm.Get_size.side_effect = lambda: calls.append("size") or 8

    communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

    assert calls[:3] == ["errhandler", "rank", "size"]


def test_create_mpi_ft_subcomm_collectively_disables_ulfm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, ft_comm = _communicators()
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))
    default_allgather = parent_comm.allgather.side_effect

    def _allgather(local_record: dict[str, object]) -> list[dict[str, object]]:
        records = default_allgather(local_record)
        if "ep_group" not in local_record:
            records[2]["ulfm_available"] = False
        return records

    parent_comm.allgather.side_effect = _allgather

    result = communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

    assert result.comm is ft_comm
    assert result.ulfm_available is False


def test_create_mpi_ft_subcomm_collectively_reports_ulfm_probe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, ft_comm = _communicators()
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))
    ft_comm.Is_revoked.side_effect = RuntimeError("probe failed")
    initial_refs = len(communicator._MPI_FT_PROCESS_LIFETIME_REFS)

    try:
        with pytest.raises(
            RuntimeError,
            match="setup failed on parent rank 5.*ULFM probe raised.*probe failed",
        ):
            communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

        ft_comm.Free.assert_not_called()
        assert communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:] == [ft_comm]
    finally:
        del communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:]


def test_create_mpi_ft_subcomm_reconciles_ulfm_error_classifier_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClassifierFailureMpiException(Exception):
        def Get_error_class(self) -> int:
            raise RuntimeError("cannot classify MPI error")

    parent_comm, ft_comm = _communicators()
    fake_mpi = _fake_mpi(parent_comm)
    fake_mpi.Exception = ClassifierFailureMpiException
    monkeypatch.setattr(communicator, "MPI", fake_mpi)
    ft_comm.Is_revoked.side_effect = ClassifierFailureMpiException("probe failed")
    initial_refs = len(communicator._MPI_FT_PROCESS_LIFETIME_REFS)

    try:
        with pytest.raises(
            RuntimeError,
            match="setup failed on parent rank 5.*ULFM probe raised.*probe failed",
        ):
            communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

        assert parent_comm.allgather.call_count == 2
        ft_comm.Free.assert_not_called()
        assert communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:] == [ft_comm]
    finally:
        del communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:]


def test_create_mpi_ft_subcomm_treats_unsupported_ulfm_probe_as_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, ft_comm = _communicators()
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))
    ft_comm.Is_revoked.side_effect = NotImplementedError("ULFM is not compiled")

    result = communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

    assert result.ulfm_available is False
    ft_comm.Free.assert_not_called()


def test_create_mpi_ft_subcomm_collectively_rejects_revoked_comm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, ft_comm = _communicators()
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))
    ft_comm.Is_revoked.return_value = True
    initial_refs = len(communicator._MPI_FT_PROCESS_LIFETIME_REFS)

    try:
        with pytest.raises(RuntimeError, match="already revoked at construction"):
            communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

        ft_comm.Free.assert_not_called()
        assert communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:] == [ft_comm]
    finally:
        del communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:]


def test_create_mpi_ft_subcomm_retains_comm_after_remote_post_split_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, ft_comm = _communicators()
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))
    default_allgather = parent_comm.allgather.side_effect
    initial_refs = len(communicator._MPI_FT_PROCESS_LIFETIME_REFS)

    def _allgather(local_record: dict[str, object]) -> list[dict[str, object]]:
        records = default_allgather(local_record)
        if "ep_group" not in local_record:
            records[2]["error_message"] = "remote communicator setup failed"
        return records

    parent_comm.allgather.side_effect = _allgather

    try:
        with pytest.raises(
            RuntimeError,
            match="setup failed on parent rank 2: remote communicator setup failed",
        ):
            communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

        ft_comm.Set_errhandler.assert_called_once()
        ft_comm.Free.assert_not_called()
        assert communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:] == [ft_comm]
    finally:
        del communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:]


def test_create_mpi_ft_subcomm_retains_comm_when_post_split_allgather_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, ft_comm = _communicators()
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))
    default_allgather = parent_comm.allgather.side_effect
    allgather_error = RuntimeError("post-split allgather failed")
    initial_refs = len(communicator._MPI_FT_PROCESS_LIFETIME_REFS)

    def _allgather(local_record: dict[str, object]) -> list[dict[str, object]]:
        if "ep_group" in local_record:
            return default_allgather(local_record)
        raise allgather_error

    parent_comm.allgather.side_effect = _allgather

    try:
        with pytest.raises(RuntimeError, match="post-split allgather failed") as raised:
            communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

        assert raised.value is allgather_error
        assert parent_comm.allgather.call_count == 2
        ft_comm.Free.assert_not_called()
        assert communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:] == [ft_comm]
    finally:
        del communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:]


def test_create_mpi_ft_subcomm_retains_comm_after_malformed_post_split_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, ft_comm = _communicators()
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))
    default_allgather = parent_comm.allgather.side_effect
    initial_refs = len(communicator._MPI_FT_PROCESS_LIFETIME_REFS)

    def _allgather(local_record: dict[str, object]) -> list[object]:
        records = default_allgather(local_record)
        if "ep_group" not in local_record:
            records[2] = None
        return records

    parent_comm.allgather.side_effect = _allgather

    try:
        with pytest.raises(
            RuntimeError,
            match="invalid status for parent rank 2: None",
        ):
            communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

        ft_comm.Free.assert_not_called()
        assert communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:] == [ft_comm]
    finally:
        del communicator._MPI_FT_PROCESS_LIFETIME_REFS[initial_refs:]


def test_create_mpi_ft_subcomm_propagates_remote_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, _ = _communicators()
    topologies = _valid_topologies()
    topologies[2]["error_kind"] = "value"
    topologies[2]["error_message"] = "mapping.moe_ep_group must not be empty"
    parent_comm.allgather.side_effect = None
    parent_comm.allgather.return_value = topologies
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))

    with pytest.raises(
        ValueError, match="startup validation failed on parent rank 2.*must not be empty"
    ):
        communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

    parent_comm.Split.assert_not_called()


def test_create_mpi_ft_subcomm_collectively_validates_health_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, _ = _communicators()
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))

    with pytest.raises(ValueError, match="EPGroupHealth size must match"):
        communicator.create_mpi_ft_subcomm(
            _mapping(),
            parent_comm,
            health_size=7,
        )

    parent_comm.allgather.assert_called_once()
    parent_comm.Split.assert_not_called()


def test_create_mpi_ft_subcomm_rejects_inconsistent_remote_ep_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, _ = _communicators()
    topologies = _valid_topologies()
    topologies[6]["ep_group"] = (6, 0, 1, 2, 3, 4, 5, 7)
    parent_comm.allgather.side_effect = None
    parent_comm.allgather.return_value = topologies
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))

    with pytest.raises(RuntimeError, match="startup topology is inconsistent"):
        communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

    parent_comm.Split.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value", "error_match"),
    [
        ("ep_group", None, "invalid EP group"),
        ("ep_group", "01234567", "invalid EP group"),
        ("ep_group", (0, 1, True, 3, 4, 5, 6, 7), "invalid EP group ranks"),
        ("ep_group", (0, 1, 2, 3, 4, 5, 6, 8), "invalid EP group ranks"),
        ("ep_group", (0, 1, 2, 3, 4, 5, 6, 6), "duplicate EP group ranks"),
        ("ep_size", None, "reports EP size"),
        ("parent_rank", 2.0, "reports parent rank"),
        ("parent_size", 8.0, "reports communicator size"),
        ("world_size", 8.0, "reports world size"),
        ("mapping_rank", 2.0, "reports mapping rank"),
        ("ep_size", 8.0, "reports EP size"),
        ("ep_rank", 2.0, "reports invalid EP rank"),
        ("health_size", 8.0, "reports health size"),
    ],
    ids=[
        "missing-group",
        "string-group",
        "bool-rank",
        "out-of-range-rank",
        "duplicate-rank",
        "missing-ep-size",
        "float-parent-rank",
        "float-parent-size",
        "float-world-size",
        "float-mapping-rank",
        "float-ep-size",
        "float-ep-rank",
        "float-health-size",
    ],
)
def test_create_mpi_ft_subcomm_rejects_malformed_remote_topology_schema(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    error_match: str,
) -> None:
    parent_comm, _ = _communicators()
    topologies = _valid_topologies()
    topologies[2][field] = value
    parent_comm.allgather.side_effect = None
    parent_comm.allgather.return_value = topologies
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))

    with pytest.raises(RuntimeError, match=error_match):
        communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

    parent_comm.Split.assert_not_called()


def test_create_mpi_ft_subcomm_rejects_float_local_ep_size_before_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_comm, _ = _communicators()
    monkeypatch.setattr(communicator, "MPI", _fake_mpi(parent_comm))

    with pytest.raises(RuntimeError, match=r"reports EP size 8\.0"):
        communicator.create_mpi_ft_subcomm(
            _mapping(moe_ep_size=8.0),
            parent_comm,
        )

    parent_comm.allgather.assert_called_once()
    parent_comm.Split.assert_not_called()


def test_create_mpi_ft_subcomm_uses_pkl5_compatible_object_allgather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ft_comm = Mock()
    ft_comm.Get_rank.return_value = 5
    ft_comm.Get_size.return_value = 8
    ft_comm.Is_revoked.return_value = False

    class Pkl5LikeParent:
        def __init__(self) -> None:
            self.gathered_records = []
            self.split_args = None

        def Get_rank(self) -> int:
            return 5

        def Get_size(self) -> int:
            return 8

        def allgather(self, local_record: dict[str, object]) -> list[dict[str, object]]:
            self.gathered_records.append(local_record)
            if "ep_group" in local_record:
                records = _valid_topologies()
            else:
                records = _valid_setup_statuses()
            records[5] = local_record
            return records

        def Split(self, *, color: int, key: int) -> Mock:
            self.split_args = (color, key)
            return ft_comm

    parent_comm = Pkl5LikeParent()
    fake_mpi = _fake_mpi(Mock())
    monkeypatch.setattr(communicator, "MPI", fake_mpi)

    result = communicator.create_mpi_ft_subcomm(_mapping(), parent_comm)

    assert result.comm is ft_comm
    assert result.local_rank == 5
    assert result.ep_size == 8
    assert result.ulfm_available is True
    assert len(parent_comm.gathered_records) == 2
    assert parent_comm.split_args == (0, 5)
    ft_comm.Set_errhandler.assert_called_once_with(fake_mpi.ERRORS_RETURN)

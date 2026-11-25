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
"""The checkpoint page-cache release after loading, across the ranks of a node."""

import os
from pathlib import Path

import pytest

from tensorrt_llm._torch.pyexecutor import model_loader

pytestmark = pytest.mark.cpu_only

_ENV = "TRTLLM_RELEASE_CHECKPOINT_PAGE_CACHE"


class _FakeNodeComm:
    """A node communicator whose collectives record the order they are called in."""

    def __init__(self, rank: int, others: list[list[str]], calls: list[str]) -> None:
        self.rank = rank
        self.others = others
        self.calls = calls

    def allgather(self, dirs: list[str]) -> list[list[str]]:
        self.calls.append("allgather")
        return [dirs, *self.others]

    def Get_rank(self) -> int:
        return self.rank

    def Barrier(self) -> None:
        self.calls.append("barrier")


def _unexpected() -> None:
    raise AssertionError("the node communicator must not be used")


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Turn the release on under MPI and record the drops instead of doing them."""
    recorded: list[str] = []
    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setattr(model_loader, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(model_loader, "mpi_disabled", lambda: False)
    monkeypatch.setattr(
        model_loader, "_drop_page_cache_of_dirs", lambda dirs: recorded.append(f"drop {dirs}")
    )
    return recorded


def test_node_rank_0_drops_every_dir_read_on_the_node_between_the_collectives(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    comm = _FakeNodeComm(0, [["/draft", "/ckpt"], []], calls)
    monkeypatch.setattr(model_loader, "local_mpi_comm", lambda: comm)

    seconds = model_loader._release_checkpoint_page_cache(["/ckpt"])

    assert calls == ["allgather", "drop ['/ckpt', '/draft']", "barrier"]
    assert seconds is not None and seconds >= 0


def test_other_ranks_only_join_the_collectives(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    comm = _FakeNodeComm(1, [["/ckpt"]], calls)
    monkeypatch.setattr(model_loader, "local_mpi_comm", lambda: comm)

    assert model_loader._release_checkpoint_page_cache(["/ckpt"]) is not None

    assert calls == ["allgather", "barrier"]


def test_ranks_join_the_collectives_when_no_rank_read_files(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    comm = _FakeNodeComm(0, [[]], calls)
    monkeypatch.setattr(model_loader, "local_mpi_comm", lambda: comm)

    model_loader._release_checkpoint_page_cache([])

    assert calls == ["allgather", "barrier"]


def test_the_barrier_runs_when_the_drop_fails(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    comm = _FakeNodeComm(0, [], calls)
    monkeypatch.setattr(model_loader, "local_mpi_comm", lambda: comm)

    def fail(dirs: list[str]) -> None:
        raise RuntimeError("drop failed")

    monkeypatch.setattr(model_loader, "_drop_page_cache_of_dirs", fail)

    with pytest.raises(RuntimeError, match="drop failed"):
        model_loader._release_checkpoint_page_cache(["/ckpt"])

    assert calls == ["allgather", "barrier"]


@pytest.mark.parametrize("value", [None, "0", "true"])
def test_off_unless_the_variable_is_1(
    monkeypatch: pytest.MonkeyPatch, calls: list[str], value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(_ENV)
    else:
        monkeypatch.setenv(_ENV, value)
    monkeypatch.setattr(model_loader, "local_mpi_comm", _unexpected)

    assert model_loader._release_checkpoint_page_cache(["/ckpt"]) is None
    assert calls == []


def test_off_when_mpi_is_disabled(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    monkeypatch.setattr(model_loader, "mpi_disabled", lambda: True)
    monkeypatch.setattr(model_loader, "local_mpi_comm", _unexpected)

    assert model_loader._release_checkpoint_page_cache(["/ckpt"]) is None
    assert calls == []


def test_a_single_device_build_drops_without_collectives(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    monkeypatch.setattr(model_loader, "ENABLE_MULTI_DEVICE", False)
    monkeypatch.setattr(model_loader, "local_mpi_comm", _unexpected)

    model_loader._release_checkpoint_page_cache(["/ckpt", "/ckpt", ""])

    assert calls == ["drop ['/ckpt']"]


@pytest.fixture
def released(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Record what the drop advises and whether it waits, instead of doing either.

    The advise reports ``record["advised"]`` bytes.
    """
    record: dict = {"files": None, "waited": False, "advised": 2 << 30}

    def release(paths: list[str]) -> int:
        record["files"] = sorted(paths)
        return record["advised"]

    def wait() -> float:
        record["waited"] = True
        return 1.0

    monkeypatch.setattr(model_loader, "release_file_page_cache", release)
    monkeypatch.setattr(model_loader, "wait_for_page_migration_to_settle", wait)
    return record


def _page_cache(monkeypatch: pytest.MonkeyPatch, before: int | None, after: int | None) -> None:
    values = iter([before, after])
    monkeypatch.setattr(model_loader, "page_cache_bytes", lambda: next(values))


def test_the_drop_covers_the_top_level_files_of_existing_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, released: dict
) -> None:
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "nested").mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"w")
    (checkpoint / "nested" / "other.safetensors").write_bytes(b"w")
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "draft.safetensors").write_bytes(b"w")
    _page_cache(monkeypatch, 5 << 30, 1 << 30)

    model_loader._drop_page_cache_of_dirs([str(checkpoint), str(draft), str(tmp_path / "missing")])

    assert released["files"] == [
        str(checkpoint / "model.safetensors"),
        str(draft / "draft.safetensors"),
    ]
    assert released["waited"]


@pytest.mark.parametrize(
    "advised, before, after, waits",
    [
        ((1 << 30) - 1, 5 << 30, 1 << 30, False),  # files short of 1 GiB
        (1 << 30, 5 << 30, 4 << 30, True),
        # Others filled the page cache meanwhile: an unchanged page cache
        # must not skip the wait.
        (5 << 30, 5 << 30, 5 << 30, True),
        (1 << 30, None, None, True),
    ],
)
def test_the_drop_waits_for_page_migration_when_the_files_total_1_gib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    released: dict,
    advised: int,
    before: int | None,
    after: int | None,
    waits: bool,
) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"w")
    released["advised"] = advised
    _page_cache(monkeypatch, before, after)

    model_loader._drop_page_cache_of_dirs([str(tmp_path)])

    assert released["waited"] is waits


def test_the_drop_is_skipped_on_filesystem_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, released: dict
) -> None:
    def scandir(path: str) -> None:
        raise OSError("stale file handle")

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", scandir)
        model_loader._drop_page_cache_of_dirs([str(tmp_path)])

    assert released["files"] is None
    assert not released["waited"]

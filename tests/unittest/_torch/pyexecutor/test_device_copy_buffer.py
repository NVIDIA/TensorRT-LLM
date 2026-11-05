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

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# Loaded by path: the module needs only torch and tensorrt_llm._utils, not the PyTorch backend's package.
_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "device_copy.py"
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    # Registered first: its dataclasses resolve their annotations through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


device_copy = _load("_device_copy_under_test")


class _Site(device_copy.CopySiteEnum):
    UPLOAD = ("test.upload", device_copy.Direction.H2D)
    OTHER = ("test.other", device_copy.Direction.H2D)
    TAIL = ("test.tail", device_copy.Direction.FILL)


def DeviceCopyBuffer(tensor: torch.Tensor, **kwargs):
    return device_copy.DeviceCopyBuffer(tensor, site=kwargs.pop("site", _Site.UPLOAD), **kwargs)


pytestmark = pytest.mark.cpu_only

SENTINEL = -7


@pytest.fixture(autouse=True)
def _no_skipped_copy_check(monkeypatch) -> None:
    monkeypatch.setattr(device_copy, "CHECK_MODE", False)


def _host(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def _tamper(dst: torch.Tensor) -> None:
    dst.fill_(SENTINEL)


def test_first_call_copies() -> None:
    dst = torch.zeros(8, dtype=torch.int32)
    buffer = DeviceCopyBuffer(dst)
    buffer.update(_host([3, 1, 2]))
    assert dst.tolist() == [3, 1, 2, 0, 0, 0, 0, 0]


def test_unchanged_values_skip_the_copy() -> None:
    dst = torch.zeros(8, dtype=torch.int32)
    buffer = DeviceCopyBuffer(dst)
    buffer.update(_host([3, 1, 2]))
    _tamper(dst)
    buffer.update(_host([3, 1, 2]))
    assert dst.eq(SENTINEL).all()


def test_changed_values_copy() -> None:
    dst = torch.zeros(8, dtype=torch.int32)
    buffer = DeviceCopyBuffer(dst)
    buffer.update(_host([3, 1, 2]))
    _tamper(dst)
    buffer.update(_host([3, 1, 4]))
    assert dst[:3].tolist() == [3, 1, 4]
    assert dst[3:].eq(SENTINEL).all(), "only the leading elements are written"


def test_prefix_of_the_last_values_skips_the_copy() -> None:
    dst = torch.zeros(8, dtype=torch.int32)
    buffer = DeviceCopyBuffer(dst)
    buffer.update(_host([5, 6, 7, 8]))
    _tamper(dst)
    buffer.update(_host([5, 6]))
    assert dst.eq(SENTINEL).all()


def test_longer_values_copy_and_extend_what_is_kept() -> None:
    dst = torch.zeros(8, dtype=torch.int32)
    buffer = DeviceCopyBuffer(dst)
    buffer.update(_host([5, 6]))
    buffer.update(_host([5, 6, 7, 8]))
    assert dst[:4].tolist() == [5, 6, 7, 8]
    _tamper(dst)
    buffer.update(_host([5, 6, 7, 8]))
    assert dst.eq(SENTINEL).all()


def test_shorter_changed_values_keep_the_tail() -> None:
    dst = torch.zeros(8, dtype=torch.int32)
    buffer = DeviceCopyBuffer(dst)
    buffer.update(_host([5, 6, 7, 8]))
    buffer.update(_host([9, 9]))
    assert dst[:4].tolist() == [9, 9, 7, 8]
    _tamper(dst)
    buffer.update(_host([9, 9, 7, 8]))
    assert dst.eq(SENTINEL).all()
    buffer.update(_host([9, 9, 7, 1]))
    assert dst[:4].tolist() == [9, 9, 7, 1]


def test_host_values_may_be_reused_right_away() -> None:
    dst = torch.zeros(4, dtype=torch.int32)
    buffer = DeviceCopyBuffer(dst)
    host = _host([1, 2, 3])
    buffer.update(host)
    host.fill_(0)
    assert dst[:3].tolist() == [1, 2, 3]
    buffer.update(_host([1, 2, 3]))
    assert dst[:3].tolist() == [1, 2, 3]


def test_a_new_buffer_starts_over() -> None:
    first = torch.zeros(4, dtype=torch.int32)
    first_buffer = DeviceCopyBuffer(first)
    first_buffer.update(_host([1, 2]))
    second = torch.full((4,), SENTINEL, dtype=torch.int32)
    second_buffer = DeviceCopyBuffer(second)
    second_buffer.update(_host([1, 2]))
    assert second[:2].tolist() == [1, 2]


def test_two_dimensional_values_are_flattened() -> None:
    dst = torch.zeros(6, dtype=torch.int32)
    buffer = DeviceCopyBuffer(dst)
    buffer.update(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
    assert dst[:4].tolist() == [1, 2, 3, 4]
    _tamper(dst)
    buffer.update(_host([1, 2, 3, 4]))
    assert dst.eq(SENTINEL).all()


def test_debug_check_passes_a_buffer_only_the_function_wrote(monkeypatch) -> None:
    monkeypatch.setattr(device_copy, "CHECK_MODE", True)
    dst = torch.zeros(8, dtype=torch.int64)
    buffer = DeviceCopyBuffer(dst)
    buffer.update(_host([5, 6, 7, 8]))
    buffer.update(_host([5, 6, 7, 8]))
    buffer.update(_host([5, 6]))
    assert dst[:4].tolist() == [5, 6, 7, 8]


def test_debug_check_reports_a_write_by_anything_else(monkeypatch) -> None:
    monkeypatch.setattr(device_copy, "CHECK_MODE", True)
    dst = torch.zeros(8, dtype=torch.int32)
    buffer = DeviceCopyBuffer(dst)
    buffer.update(_host([3, 1]))
    dst[:2].copy_(_host([5, 6]))
    with pytest.raises(
        RuntimeError, match=r"_Site\.UPLOAD was passed unchanged=True, but its destination"
    ):
        buffer.update(_host([3, 1]))


def test_two_consumers_share_changed_then_restored_copy_state(monkeypatch) -> None:
    storage = torch.full((4,), SENTINEL, dtype=torch.int32)
    manager = DeviceCopyBuffer(storage)
    metadata = manager
    manager_view = manager.tensor.view_as(storage)
    metadata_view = metadata.tensor[:2]
    address = storage.data_ptr()
    writes = []
    original = torch.Tensor.copy_

    def counted(dst, src, *args, **kwargs):
        if dst.data_ptr() == address:
            writes.append(src.tolist())
        return original(dst, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", counted)
    for consumer, values in (
        (manager, [1, 2]),
        (metadata, [9, 2]),
        (manager, [1, 2]),
    ):
        consumer.update(_host(values))
        assert manager_view.tolist() == values + [SENTINEL, SENTINEL]
        assert metadata_view.tolist() == values
        assert storage.data_ptr() == address
    metadata.update(_host([1, 2]))
    manager.update(_host([1, 2]))
    assert writes == [[1, 2], [9, 2], [1, 2]]


def test_forced_writer_keeps_a_shared_alias_cache_coherent(monkeypatch) -> None:
    storage = torch.zeros(2, dtype=torch.int32)
    manager = DeviceCopyBuffer(storage)
    metadata = manager
    writes = []
    original = torch.Tensor.copy_

    def counted(dst, src, *args, **kwargs):
        if dst.data_ptr() == storage.data_ptr():
            writes.append(src.tolist())
        return original(dst, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", counted)
    manager.update(_host([1, 2]), force=True)
    manager.update(_host([1, 2]), force=True)
    metadata.update(_host([9, 2]))
    manager.update(_host([1, 2]), force=True)
    metadata.update(_host([1, 2]))
    assert storage.tolist() == [1, 2]
    assert writes == [[1, 2], [1, 2], [9, 2], [1, 2]]


def test_specialized_copy_records_host_inputs_and_layout_key() -> None:
    buffer = DeviceCopyBuffer(torch.zeros(4, dtype=torch.int32))
    host = _host([1, 2, 3, 4]).view(2, 2)
    key = ((7, 9), 1, 2, 8)
    assert not buffer.matches(host, key=key, exact=True)
    buffer.record(host, key=key)
    assert buffer.matches(host, key=key, exact=True)
    assert not buffer.matches(host, key=((9, 7), 1, 2, 8), exact=True)
    assert not buffer.matches(host.flatten(), key=key, exact=True)
    assert not buffer.matches(host.to(torch.int64), key=key, exact=True)
    host[0, 0] = 99
    assert not buffer.matches(host, key=key, exact=True)


def test_invalidation_discards_a_recorded_copy() -> None:
    buffer = DeviceCopyBuffer(torch.zeros(4, dtype=torch.int32))
    host = _host([1, 2])
    buffer.record(host)
    assert buffer.matches(host)
    buffer.invalidate()
    assert not buffer.matches(host)
    buffer.update(host)
    assert buffer.tensor.tolist() == [1, 2, 0, 0]


def test_new_layout_does_not_retain_the_previous_prefix_tail() -> None:
    buffer = DeviceCopyBuffer(torch.zeros(4, dtype=torch.int32))
    buffer.record(_host([1, 2, 3, 4]), key="first")
    buffer.record(_host([1, 2]), key="second")
    assert buffer.matches(_host([1, 2]), key="second")
    assert not buffer.matches(_host([1, 2, 3, 4]), key="second")


def test_check_mode_is_one_switch(monkeypatch) -> None:
    """TLLM_DEBUG_MODE=1, which verified DeviceCopyBuffer's skipped copies, turns on the same CHECK_MODE as
    TLLM_DEVICE_COPY_CHECK=1."""

    def load():
        return _load("_device_copy_env_under_test").CHECK_MODE

    monkeypatch.delenv(device_copy.CHECK_ENV, raising=False)
    monkeypatch.delenv("TLLM_DEBUG_MODE", raising=False)
    assert load() is False
    monkeypatch.setenv("TLLM_DEBUG_MODE", "1")
    assert load() is True
    monkeypatch.delenv("TLLM_DEBUG_MODE")
    monkeypatch.setenv(device_copy.CHECK_ENV, "1")
    assert load() is True


def test_update_returns_whether_it_issued() -> None:
    buffer = DeviceCopyBuffer(torch.zeros(4, dtype=torch.int32))
    assert buffer.update(_host([1, 2])) is True
    assert buffer.update(_host([1, 2])) is False
    assert buffer.update(_host([1, 2]), force=True) is True


def test_updates_go_through_the_buffer_site(monkeypatch) -> None:
    calls = []
    original = device_copy.copy

    def spy(site, dst, src, **kwargs):
        calls.append((site, bool(kwargs.get("unchanged", False))))
        return original(site, dst, src, **kwargs)

    monkeypatch.setattr(device_copy, "copy", spy)
    buffer = DeviceCopyBuffer(torch.zeros(4, dtype=torch.int32), site=_Site.OTHER)
    buffer.update(_host([1, 2]))
    buffer.update(_host([1, 2]))
    assert calls == [(_Site.OTHER, False), (_Site.OTHER, True)]


def test_a_buffer_needs_an_upload_site_or_none() -> None:
    with pytest.raises(ValueError, match="not an h2d site"):
        DeviceCopyBuffer(torch.zeros(4, dtype=torch.int32), site=_Site.TAIL)
    record_only = DeviceCopyBuffer(torch.zeros(4, dtype=torch.int32), site=None)
    record_only.record(_host([1, 2]))
    assert record_only.matches(_host([1, 2]))
    with pytest.raises(ValueError, match="record-only"):
        record_only.update(_host([1, 2]))


def test_a_failed_copy_leaves_no_record(monkeypatch) -> None:
    buffer = DeviceCopyBuffer(torch.zeros(4, dtype=torch.int32))
    buffer.update(_host([1, 2, 3, 4]))

    def fail(*args, **kwargs):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(device_copy, "copy", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        buffer.update(_host([9, 9]))
    assert not buffer.matches(_host([1, 2, 3, 4]))
    assert not buffer.matches(_host([9, 9]))


def test_a_dropped_merged_copy_invalidates_the_record(monkeypatch) -> None:
    drops = []
    original = device_copy.copy

    def keep_on_drop(site, dst, src, **kwargs):
        drops.append(kwargs["on_drop"])
        return original(site, dst, src, **kwargs)

    monkeypatch.setattr(device_copy, "copy", keep_on_drop)
    buffer = DeviceCopyBuffer(torch.zeros(4, dtype=torch.int32))
    buffer.update(_host([1, 2]))
    assert buffer.matches(_host([1, 2]))
    # The pack the copy was merged into is discarded: the copy never lands.
    drops[-1]()
    assert not buffer.matches(_host([1, 2]))


def test_an_eager_copy_is_snapshotted_unless_the_owner_opts_out(monkeypatch) -> None:
    sources = []
    original = torch.Tensor.copy_

    def spy(dst, src, *args, **kwargs):
        sources.append(src)
        return original(dst, src, *args, **kwargs)

    host = _host([1, 2])
    monkeypatch.setattr(torch.Tensor, "copy_", spy)
    DeviceCopyBuffer(torch.zeros(4, dtype=torch.int32)).update(host)
    assert sources[-1] is not host and sources[-1].data_ptr() != host.data_ptr()
    DeviceCopyBuffer(torch.zeros(4, dtype=torch.int32), snapshot=False).update(host)
    assert sources[-1].data_ptr() == host.data_ptr()


def test_a_view_update_is_keyed_by_the_view() -> None:
    storage = torch.zeros(3, 4, dtype=torch.int32)
    buffer = DeviceCopyBuffer(storage)
    rows = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    assert buffer.update(rows, view=storage[:2, :2]) is True
    assert storage[:2, :2].tolist() == [[1, 2], [3, 4]]
    assert buffer.update(rows, view=storage[:2, :2]) is False
    # The same values into another view of the storage are a new copy.
    assert buffer.update(rows, view=storage[1:3, :2]) is True
    assert storage[1:3, :2].tolist() == [[1, 2], [3, 4]]
    # An exact match is required: a prefix of the recorded values is not enough for a view.
    assert buffer.update(rows[:1], view=storage[1:2, :2]) is True


def test_a_repeated_tail_fill_is_skipped_and_a_moved_start_refills(monkeypatch) -> None:
    buf = torch.zeros(6, dtype=torch.int32)
    record = DeviceCopyBuffer(buf, site=None)
    record.fill_tail(_Site.TAIL, 2, -1)
    assert buf.tolist() == [0, 0, -1, -1, -1, -1]
    buf[4].fill_(SENTINEL)
    record.fill_tail(_Site.TAIL, 2, -1)
    assert buf[4].item() == SENTINEL, "an identical fill is skipped"
    record.fill_tail(_Site.TAIL, 3, -1)
    assert buf.tolist() == [0, 0, -1, -1, -1, -1]
    # A write of the whole buffer (start == len) fills nothing and drops the record.
    record.fill_tail(_Site.TAIL, 6, -1)
    buf.fill_(SENTINEL)
    record.fill_tail(_Site.TAIL, 3, -1)
    assert buf[3:].tolist() == [-1, -1, -1]


def test_an_update_past_the_tail_start_drops_the_tail_record() -> None:
    buf = torch.zeros(6, dtype=torch.int32)
    record = DeviceCopyBuffer(buf)
    record.update(_host([1, 2]))
    record.fill_tail(_Site.TAIL, 2, -1)
    record.update(_host([1, 2, 3]))
    record.fill_tail(_Site.TAIL, 2, -1)
    assert buf.tolist() == [1, 2, -1, -1, -1, -1]


def test_check_mode_verifies_a_skipped_tail_fill(monkeypatch) -> None:
    monkeypatch.setattr(device_copy, "CHECK_MODE", True)
    buf = torch.zeros(4, dtype=torch.int32)
    record = DeviceCopyBuffer(buf, site=None)
    record.fill_tail(_Site.TAIL, 1, -1)
    buf[3].fill_(SENTINEL)
    with pytest.raises(RuntimeError, match=r"_Site\.TAIL was passed unchanged=True"):
        record.fill_tail(_Site.TAIL, 1, -1)

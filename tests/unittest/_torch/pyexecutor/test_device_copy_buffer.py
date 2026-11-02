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
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).resolve().parents[4] / "tensorrt_llm" / "_device_copy.py"
_SPEC = importlib.util.spec_from_file_location("device_copy", _MODULE_PATH)
device_copy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(device_copy)
DeviceCopyBuffer = device_copy.DeviceCopyBuffer

pytestmark = pytest.mark.cpu_only

SENTINEL = -7


@pytest.fixture(autouse=True)
def _no_skipped_copy_check(monkeypatch) -> None:
    monkeypatch.setattr(device_copy, "_verify_skipped_device_copies", False)


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
    monkeypatch.setattr(device_copy, "_verify_skipped_device_copies", True)
    dst = torch.zeros(8, dtype=torch.int64)
    buffer = DeviceCopyBuffer(dst)
    buffer.update(_host([5, 6, 7, 8]))
    buffer.update(_host([5, 6, 7, 8]))
    buffer.update(_host([5, 6]))
    assert dst[:4].tolist() == [5, 6, 7, 8]


def test_debug_check_reports_a_write_by_anything_else(monkeypatch) -> None:
    monkeypatch.setattr(device_copy, "_verify_skipped_device_copies", True)
    dst = torch.zeros(8, dtype=torch.int32)
    buffer = DeviceCopyBuffer(dst)
    buffer.update(_host([3, 1]))
    dst[:2].copy_(_host([5, 6]))
    with pytest.raises(AssertionError, match="holds \\[5, 6\\] instead of \\[3, 1\\]"):
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

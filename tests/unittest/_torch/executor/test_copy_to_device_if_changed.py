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
"""``copy_to_device_if_changed``: which calls copy and what the buffer then holds.

The skip decision does not depend on the device, so the buffer here is a host tensor: a skipped copy shows as a
sentinel written behind the function's back that survives the call.
"""

import pytest
import torch

from tensorrt_llm._utils import copy_to_device_if_changed

pytestmark = pytest.mark.cpu_only

SENTINEL = -7


def _host(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def _tamper(dst: torch.Tensor) -> None:
    """Overwrite the buffer without the function knowing, so only a real copy restores it."""
    dst.fill_(SENTINEL)


def test_first_call_copies() -> None:
    dst = torch.zeros(8, dtype=torch.int32)
    copy_to_device_if_changed(dst, _host([3, 1, 2]))
    assert dst.tolist() == [3, 1, 2, 0, 0, 0, 0, 0]


def test_unchanged_values_skip_the_copy() -> None:
    dst = torch.zeros(8, dtype=torch.int32)
    copy_to_device_if_changed(dst, _host([3, 1, 2]))
    _tamper(dst)
    copy_to_device_if_changed(dst, _host([3, 1, 2]))
    assert dst.eq(SENTINEL).all()


def test_changed_values_copy() -> None:
    dst = torch.zeros(8, dtype=torch.int32)
    copy_to_device_if_changed(dst, _host([3, 1, 2]))
    _tamper(dst)
    copy_to_device_if_changed(dst, _host([3, 1, 4]))
    assert dst[:3].tolist() == [3, 1, 4]
    assert dst[3:].eq(SENTINEL).all(), "only the leading elements are written"


def test_prefix_of_the_last_values_skips_the_copy() -> None:
    dst = torch.zeros(8, dtype=torch.int32)
    copy_to_device_if_changed(dst, _host([5, 6, 7, 8]))
    _tamper(dst)
    copy_to_device_if_changed(dst, _host([5, 6]))
    assert dst.eq(SENTINEL).all()


def test_longer_values_copy_and_extend_what_is_kept() -> None:
    dst = torch.zeros(8, dtype=torch.int32)
    copy_to_device_if_changed(dst, _host([5, 6]))
    copy_to_device_if_changed(dst, _host([5, 6, 7, 8]))
    assert dst[:4].tolist() == [5, 6, 7, 8]
    _tamper(dst)
    copy_to_device_if_changed(dst, _host([5, 6, 7, 8]))
    assert dst.eq(SENTINEL).all()


def test_shorter_changed_values_keep_the_tail() -> None:
    # A shorter copy rewrites only its leading elements; the buffer's tail still holds the earlier values, and what
    # the function keeps says so.
    dst = torch.zeros(8, dtype=torch.int32)
    copy_to_device_if_changed(dst, _host([5, 6, 7, 8]))
    copy_to_device_if_changed(dst, _host([9, 9]))
    assert dst[:4].tolist() == [9, 9, 7, 8]
    _tamper(dst)
    copy_to_device_if_changed(dst, _host([9, 9, 7, 8]))
    assert dst.eq(SENTINEL).all()
    copy_to_device_if_changed(dst, _host([9, 9, 7, 1]))
    assert dst[:4].tolist() == [9, 9, 7, 1]


def test_host_values_may_be_reused_right_away() -> None:
    dst = torch.zeros(4, dtype=torch.int32)
    host = _host([1, 2, 3])
    copy_to_device_if_changed(dst, host)
    host.fill_(0)
    assert dst[:3].tolist() == [1, 2, 3]
    copy_to_device_if_changed(dst, _host([1, 2, 3]))
    assert dst[:3].tolist() == [1, 2, 3]


def test_a_new_buffer_starts_over() -> None:
    first = torch.zeros(4, dtype=torch.int32)
    copy_to_device_if_changed(first, _host([1, 2]))
    second = torch.full((4,), SENTINEL, dtype=torch.int32)
    copy_to_device_if_changed(second, _host([1, 2]))
    assert second[:2].tolist() == [1, 2]


def test_two_dimensional_values_are_flattened() -> None:
    dst = torch.zeros(6, dtype=torch.int32)
    copy_to_device_if_changed(dst, torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
    assert dst[:4].tolist() == [1, 2, 3, 4]
    _tamper(dst)
    copy_to_device_if_changed(dst, _host([1, 2, 3, 4]))
    assert dst.eq(SENTINEL).all()

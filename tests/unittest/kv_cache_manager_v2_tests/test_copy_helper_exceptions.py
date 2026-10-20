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
"""The device copy helpers report a rejected request instead of terminating.

launchBatchedCopyImpl validates its byte count with TLLM_CHECK_WITH_INFO, which throws. The
three helpers that reach it are therefore not noexcept, and these check that the exception
survives the binding boundary rather than reaching std::terminate.
"""

import pytest

copy_helpers = pytest.importorskip(
    "tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils",
    reason="the device copy helpers are C++-backend only",
)

MemToMemTask = copy_helpers.MemToMemTask

# Rejected before any CUDA work: the addresses are never dereferenced, so they need not be real.
TASKS = [MemToMemTask(dst=0x1000, src=0x2000)]

# Not a multiple of sizeof(Grain) == 16, which is what the helper rejects.
UNALIGNED_NUM_BYTES = 1

DEFAULT_STREAM = 0


@pytest.mark.parametrize(
    "helper_name",
    ["copy_host_to_device", "copy_device_to_host", "copy_device_to_device"],
)
def test_copy_helper_raises_on_unaligned_byte_count(helper_name: str) -> None:
    helper = getattr(copy_helpers, helper_name)
    with pytest.raises(Exception, match="multiple of 16"):
        helper(TASKS, UNALIGNED_NUM_BYTES, DEFAULT_STREAM)


@pytest.mark.parametrize(
    "helper_name",
    ["copy_host_to_device", "copy_device_to_host", "copy_device_to_device"],
)
def test_copy_helper_accepts_an_empty_task_list(helper_name: str) -> None:
    """The byte count is only reached with work to do, so an empty list is not rejected."""
    helper = getattr(copy_helpers, helper_name)
    assert helper([], UNALIGNED_NUM_BYTES, DEFAULT_STREAM) == 0

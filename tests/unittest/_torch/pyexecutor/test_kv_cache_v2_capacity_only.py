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
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState


def _manager() -> KVCacheManagerV2:
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.enable_block_reuse = False
    manager.kv_cache_map = {}
    manager._stream = MagicMock()
    return manager


def _request(request_id: int, *, rewind: int = 0, complete: bool = False) -> MagicMock:
    request = MagicMock()
    request.py_request_id = request_id
    request.py_rewind_len = rewind
    request.max_beam_num_tokens = 201
    request.py_kv_cache_generation_capacity_only = False
    request.py_kv_cache_compaction = None
    request.state = (
        LlmRequestState.GENERATION_COMPLETE if complete else LlmRequestState.GENERATION_IN_PROGRESS
    )
    return request


def _cache(capacity: int = 256) -> MagicMock:
    cache = MagicMock()
    cache.is_active = True
    cache.capacity = capacity
    cache.resize.return_value = True
    return cache


def test_capacity_only_is_request_scoped() -> None:
    manager = _manager()
    regular = _request(1, rewind=3)
    compacted = _request(2, rewind=5)
    regular_cache = _cache()
    compacted_cache = _cache()
    manager.kv_cache_map = {1: regular_cache, 2: compacted_cache}
    compacted.py_kv_cache_generation_capacity_only = True

    manager.update_resources(SimpleNamespace(generation_requests=[regular, compacted]))

    regular_cache.resize.assert_called_once_with(253, 200)
    compacted_cache.resize.assert_called_once_with(251, None)


def test_missing_capacity_only_flag_is_fail_closed() -> None:
    manager = _manager()
    request = MagicMock()
    request.py_request_id = 1
    request.py_rewind_len = 3
    request.max_beam_num_tokens = 201
    request.py_kv_cache_compaction = None
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    cache = _cache()
    manager.kv_cache_map[1] = cache

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(253, 200)


def test_capacity_only_completion_preserves_history() -> None:
    manager = _manager()
    request = _request(1, complete=True)
    cache = _cache()
    manager.kv_cache_map[1] = cache
    request.py_kv_cache_generation_capacity_only = True

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(None, None)


def test_compaction_target_preserves_overlap_growth() -> None:
    manager = _manager()
    request = _request(1)
    event = MagicMock()
    request.py_kv_cache_compaction = (129, 256, event)
    cache = _cache(capacity=257)
    manager.kv_cache_map[1] = cache
    request.py_kv_cache_generation_capacity_only = True

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    manager._stream.wait_event.assert_called_once_with(event)
    cache.resize.assert_called_once_with(130, None)
    assert request.py_kv_cache_compaction is None


def test_failed_compaction_resize_keeps_target() -> None:
    manager = _manager()
    request = _request(1)
    target = (129, 256, MagicMock())
    request.py_kv_cache_compaction = target
    cache = _cache(capacity=256)
    cache.resize.return_value = False
    manager.kv_cache_map[1] = cache
    request.py_kv_cache_generation_capacity_only = True

    with pytest.raises(ValueError, match="Failed to resize KV cache"):
        manager.update_resources(SimpleNamespace(generation_requests=[request]))

    assert request.py_kv_cache_compaction is target

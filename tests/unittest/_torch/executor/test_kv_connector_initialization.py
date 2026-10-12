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

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    KvCacheConnectorSchedulerOutputRequest,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import KvCacheRetentionConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("windows", [(None,), (64,), (64, None), (32, 64, None)])
def test_v2_connector_requires_one_layer_group(
    monkeypatch: pytest.MonkeyPatch, windows: tuple[int | None, ...]
) -> None:
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = Mock(capacity_only=False)
    executor.kv_cache_transceiver = None
    executor.dist = SimpleNamespace(pp_size=1, cp_size=1)
    executor.max_beam_width = 1
    executor.max_draft_len = 0
    executor.enable_attention_dp = False
    executor.llm_args = SimpleNamespace(kv_cache_config=None, scheduler_config=None)
    executor.kv_cache_manager = Mock(spec=KVCacheManagerV2, enable_block_reuse=True)
    executor.kv_cache_manager.impl = SimpleNamespace(
        init_config=SimpleNamespace(cache_tiers=[SimpleNamespace(tier=CacheTier.GPU_MEM)])
    )
    executor.model_engine = Mock()
    executor.model_engine.model.named_modules.return_value = []
    layout = SimpleNamespace(groups=[SimpleNamespace(window_size=window) for window in windows])
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.build_kv_cache_layout_v2",
        lambda manager: layout,
    )

    connector = executor.kv_connector_manager
    if len(windows) > 1:
        with pytest.raises(NotImplementedError, match="requires a single layer group"):
            executor._maybe_init_kv_connector_manager()
        connector.worker.register_kv_cache_layout.assert_not_called()
        connector.wait_for_initialization.assert_not_called()
    else:
        executor._maybe_init_kv_connector_manager()
        connector.worker.register_kv_cache_layout.assert_called_once_with(layout)
        connector.warn_flat_scheduler_under_swa.assert_called_once_with(windows[0])
        connector.wait_for_initialization.assert_called_once()


def test_single_group_priorities_follow_new_page_ordinals() -> None:
    manager = Mock(spec=KVCacheManagerV2)
    cache = Mock()
    manager.kv_cache_map = {7: cache}
    manager.get_page_indices_by_layer_group.return_value = [[12, 4]]
    cache.get_page_priorities.return_value = [80, 10]
    req = Mock(
        request_id=7,
        py_request_id=7,
        state=LlmRequestState.CONTEXT_INIT,
        context_current_position=0,
        context_remaining_length=64,
        context_chunk_size=64,
        kv_cache_retention_config=KvCacheRetentionConfig([]),
        cache_salt=None,
    )
    req.get_tokens.return_value = list(range(64))
    tracker = KvCacheConnectorSchedulerOutputRequest()

    first = tracker.update_and_build_data(req, manager)
    assert first.new_block_ids == [12, 4]
    assert first.priorities == [80, 10]

    manager.get_page_indices_by_layer_group.return_value = [[12, 4, 9]]
    cache.get_page_priorities.return_value = [80, 10, 20]
    second = tracker.update_and_build_data(req, manager)
    assert second.new_block_ids == [9]
    assert second.priorities == [20]

    unchanged = tracker.update_and_build_data(req, manager)
    assert unchanged.new_block_ids == []
    assert unchanged.priorities == []

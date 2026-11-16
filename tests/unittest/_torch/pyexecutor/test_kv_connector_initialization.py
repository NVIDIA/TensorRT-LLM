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
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    CppMambaHybridCacheManager,
    MambaCacheManager,
    MixedMambaHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import KvCacheRetentionConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, CacheTier, SsmLayerConfig

pytestmark = pytest.mark.cpu_only

RECURRENT_STATE_REJECTION = (
    "KV Cache Connector is not supported with Mamba / hybrid "
    "linear-attention models. Mamba layers carry SSM state "
    "rather than per-token KV blocks, so the connector's "
    "per-layer load/save hooks have nothing meaningful to "
    "transfer for those layers."
)


def _connector_executor(kv_cache_manager: Mock, *, capacity_only: bool = False) -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = Mock(capacity_only=capacity_only, enable_attention_dp=False)
    executor.kv_cache_transceiver = None
    executor.dist = SimpleNamespace(pp_size=1, cp_size=1)
    executor.max_beam_width = 1
    executor.max_draft_len = 0
    executor.enable_attention_dp = False
    executor.llm_args = SimpleNamespace(kv_cache_config=None, scheduler_config=None)
    executor.kv_cache_manager = kv_cache_manager
    executor.model_engine = Mock()
    executor.model_engine.model.named_modules.return_value = []
    return executor


def _v2_cache(*layer_types: type, tier: CacheTier = CacheTier.GPU_MEM) -> Mock:
    cache = Mock(spec=KVCacheManagerV2, enable_block_reuse=True)
    cache.impl = SimpleNamespace(
        init_config=SimpleNamespace(
            cache_tiers=[SimpleNamespace(tier=tier)],
            layers=[Mock(spec=layer_type) for layer_type in layer_types],
        )
    )
    return cache


@pytest.mark.parametrize("windows", [(None,), (64,), (64, None), (32, 64, None)])
@pytest.mark.parametrize("capacity_only", [False, True])
def test_v2_connector_registers_layer_groups(
    monkeypatch: pytest.MonkeyPatch, windows: tuple[int | None, ...], capacity_only: bool
) -> None:
    executor = _connector_executor(_v2_cache(AttentionLayerConfig), capacity_only=capacity_only)
    layout = SimpleNamespace(groups=[SimpleNamespace(window_size=window) for window in windows])
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.build_kv_cache_layout_v2",
        lambda manager: layout,
    )

    connector = executor.kv_connector_manager
    executor._maybe_init_kv_connector_manager()
    connector.reject_flat_only_scheduler.assert_called_once_with(len(windows))
    connector.worker.register_kv_cache_layout.assert_called_once_with(layout)
    connector.warn_flat_scheduler_under_swa.assert_called_once_with(
        windows[0] if len(windows) == 1 else None
    )
    connector.wait_for_initialization.assert_called_once()


@pytest.mark.parametrize("tier", [CacheTier.GPU_MEM, CacheTier.HOST_MEM])
@pytest.mark.parametrize("capacity_only", [False, True])
def test_v2_connector_rejects_a_cache_with_state_layers(
    tier: CacheTier, capacity_only: bool
) -> None:
    # The cache is a plain KVCacheManagerV2: its layout, not its class, is asked.
    executor = _connector_executor(
        _v2_cache(AttentionLayerConfig, SsmLayerConfig, tier=tier), capacity_only=capacity_only
    )

    with pytest.raises(NotImplementedError) as error:
        executor._maybe_init_kv_connector_manager()

    assert str(error.value) == RECURRENT_STATE_REJECTION
    executor.kv_connector_manager.wait_for_initialization.assert_not_called()


@pytest.mark.parametrize(
    "manager_cls", [CppMambaHybridCacheManager, MixedMambaHybridCacheManager, MambaCacheManager]
)
def test_v1_connector_rejects_recurrent_managers(manager_cls: type) -> None:
    executor = _connector_executor(Mock(spec=manager_cls, is_vswa=False))

    with pytest.raises(NotImplementedError) as error:
        executor._maybe_init_kv_connector_manager()

    assert str(error.value) == RECURRENT_STATE_REJECTION
    executor.kv_connector_manager.wait_for_initialization.assert_not_called()


@pytest.mark.parametrize(
    "use_v2,connector_mode,num_groups,with_retention,rejected",
    [
        pytest.param(True, "transfer", 2, True, True, id="grouped-retention"),
        pytest.param(True, "transfer", 3, True, True, id="three-group-retention"),
        pytest.param(True, "transfer", 2, False, False, id="grouped-no-retention"),
        pytest.param(True, "capacity", 2, True, False, id="capacity-retention"),
        pytest.param(True, None, 2, True, False, id="native-grouped-retention"),
        pytest.param(True, "transfer", 1, True, False, id="single-group-retention"),
        pytest.param(False, "transfer", 1, True, False, id="v1-retention"),
    ],
)
def test_connector_retention_request_validation(
    use_v2: bool,
    connector_mode: str | None,
    num_groups: int,
    with_retention: bool,
    rejected: bool,
) -> None:
    executor = object.__new__(PyExecutor)
    executor._is_kv_manager_v2 = use_v2
    executor.model_engine = SimpleNamespace(_dspark_confidence_enabled=False)
    executor.kv_connector_manager = (
        None if connector_mode is None else Mock(capacity_only=connector_mode == "capacity")
    )
    executor.kv_cache_manager = SimpleNamespace(num_pools=num_groups)
    executor.kv_cache_transceiver = None
    executor._configure_csa2_remote_tail = Mock()
    executor._validate_token_id_range = Mock()
    executor._validate_request_budget = Mock()
    executor.sampler = Mock()
    request = SimpleNamespace(
        kv_cache_retention_config=KvCacheRetentionConfig([]) if with_retention else None,
        py_disaggregated_params=None,
        sampling_config=None,
    )

    if rejected:
        with pytest.raises(ValueError, match="retention.*multiple layer groups"):
            executor._validate_request(request)
    else:
        executor._validate_request(request)
        executor.sampler.validate_request.assert_called_once_with(request)


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

    cache.get_page_priorities.reset_mock()
    unchanged = tracker.update_and_build_data(req, manager)
    assert unchanged.new_block_ids == []
    assert unchanged.priorities == []
    cache.get_page_priorities.assert_not_called()

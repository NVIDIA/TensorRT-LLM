# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The speculative-decoding guard on KV connectors applies only where drafts can be rejected."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    KvCacheConnectorSchedulerOutputManager,
    KvCacheConnectorSchedulerOutputRequest,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig

pytestmark = pytest.mark.cpu_only
CONNECTOR_MODULE = "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"


class _PassedSpecGuard(Exception):
    """Raised by the first call after the guard, so a test can stop there."""


def _batch(context=(), generation=()):
    return SimpleNamespace(context_requests=list(context), generation_requests=list(generation))


def _no_async_loads():
    return SimpleNamespace(loading_ids=set())


def _request(request_id, *, dummy=False):
    return SimpleNamespace(request_id=request_id, is_dummy_request=dummy)


def test_context_only_output_rejects_a_generation_request() -> None:
    manager = KvCacheConnectorSchedulerOutputManager(context_only=True)
    with pytest.raises(RuntimeError, match="context_only"):
        manager.build_scheduler_output(
            _batch(generation=[_request(7)]), _no_async_loads(), MagicMock()
        )


def test_context_only_output_skips_attention_dp_dummies() -> None:
    manager = KvCacheConnectorSchedulerOutputManager(enable_attention_dp=True, context_only=True)
    output = manager.build_scheduler_output(
        _batch(generation=[_request(7, dummy=True)]), _no_async_loads(), MagicMock()
    )
    assert output.cached_requests == []


def test_generation_requests_reach_the_connector_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = object()
    monkeypatch.setattr(
        KvCacheConnectorSchedulerOutputRequest,
        "update_and_build_data",
        lambda self, req, kv_cache_manager: data,
    )
    manager = KvCacheConnectorSchedulerOutputManager()
    output = manager.build_scheduler_output(
        _batch(generation=[_request(7)]), _no_async_loads(), MagicMock()
    )
    assert output.cached_requests == [data]


def test_connector_config_context_only_defaults_off() -> None:
    fields = dict(
        connector_module=CONNECTOR_MODULE,
        connector_scheduler_class="KvCacheConnectorScheduler",
        connector_worker_class="KvCacheConnectorWorker",
    )
    assert KvCacheConnectorConfig(**fields).context_only is False
    assert KvCacheConnectorConfig(**fields, context_only=True).context_only is True


def _executor_with_connector(*, capacity_only: bool, context_only: bool) -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = MagicMock(
        capacity_only=capacity_only, context_only=context_only, enable_attention_dp=False
    )
    executor.kv_connector_manager.configure_prefix_reservations.side_effect = _PassedSpecGuard
    executor.kv_cache_transceiver = None
    executor.dist = SimpleNamespace(pp_size=1, cp_size=1)
    executor.max_beam_width = 1
    executor.enable_attention_dp = False
    executor.kv_cache_manager = MagicMock(spec=KVCacheManagerV2)
    executor.max_draft_len = 7
    executor.llm_args = None
    return executor


def test_spec_decoding_rejects_a_transferring_connector() -> None:
    executor = _executor_with_connector(capacity_only=False, context_only=False)
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        executor._maybe_init_kv_connector_manager()


@pytest.mark.parametrize(
    "capacity_only,context_only",
    [(True, False), (False, True)],
    ids=["capacity_only", "context_only"],
)
def test_spec_decoding_allows_a_connector_that_sees_no_rejection(
    capacity_only: bool, context_only: bool
) -> None:
    executor = _executor_with_connector(capacity_only=capacity_only, context_only=context_only)
    with pytest.raises(_PassedSpecGuard):
        executor._maybe_init_kv_connector_manager()

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

import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tensorrt_llm._torch.pyexecutor import kv_cache_transceiver as transceiver_module
from tensorrt_llm._torch.pyexecutor import py_executor as executor_module
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import BindKvCacheTransceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig


@pytest.fixture(autouse=True)
def _reset_inflight_cancel_env_cache(monkeypatch):
    monkeypatch.delenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, raising=False)
    monkeypatch.delenv(transceiver_module._NIXL_KVCACHE_BACKEND_ENV, raising=False)
    monkeypatch.delenv(transceiver_module._DISABLE_KV_CACHE_TRANSFER_OVERLAP_ENV, raising=False)
    monkeypatch.delenv(transceiver_module._DISAGG_LAYERWISE_ENV, raising=False)
    monkeypatch.delenv(transceiver_module._TRY_ZCOPY_FOR_KV_CACHE_TRANSFER_ENV, raising=False)
    for env_name, _ in transceiver_module._CACHE_TRANSCEIVER_BACKEND_ENV_VARS:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(transceiver_module, "_disagg_inflight_cancel_enabled_cache", None)


def _supported_mapping():
    return SimpleNamespace(pp_size=1, cp_size=1, enable_attention_dp=False)


def _make_timeout_request(request_id=7, in_progress=False):
    return SimpleNamespace(
        is_attention_dp_dummy=False,
        py_kv_transfer_timed_out=True,
        py_request_id=request_id,
        is_disagg_generation_transmission_in_progress=in_progress,
        state=object(),
    )


def _make_response_handler_stub(active_requests, tp_allgather_result):
    executor = object.__new__(PyExecutor)
    executor.active_requests = list(active_requests)
    executor.perf_manager = Mock()
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.cancel_request.return_value = True
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.return_value = True
    executor._disagg_inflight_cancel_unsupported_logged = False
    executor.enable_attention_dp = True
    executor.dist = SimpleNamespace(
        rank=0,
        world_size=2,
        tp_allgather=Mock(return_value=tp_allgather_result),
    )
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()
    executor._handle_errors = Mock()
    executor._timeout_cleanup_order = Mock()
    executor._timeout_cleanup_order.attach_mock(executor.dist.tp_allgather, "vote")
    executor._timeout_cleanup_order.attach_mock(executor._handle_errors, "handle")
    return executor


def test_flag_unset_short_circuits_before_capability_query(monkeypatch):
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor._disagg_inflight_cancel_unsupported_logged = False
    monkeypatch.setattr(executor_module, "is_disagg_inflight_cancel_enabled", lambda: False)

    assert not PyExecutor._is_disagg_inflight_cancel_active(executor)
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.assert_not_called()


def test_unsupported_transceiver_warns_once(monkeypatch):
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.return_value = False
    executor._disagg_inflight_cancel_unsupported_logged = False
    monkeypatch.setattr(executor_module, "is_disagg_inflight_cancel_enabled", lambda: True)
    warning = Mock()
    monkeypatch.setattr(executor_module.logger, "warning", warning)

    assert not PyExecutor._is_disagg_inflight_cancel_active(executor)
    assert not PyExecutor._is_disagg_inflight_cancel_active(executor)

    assert executor._disagg_inflight_cancel_unsupported_logged
    warning.assert_called_once()


def test_flag_unset_generation_timeout_uses_rank_uniform_cleanup():
    request = _make_timeout_request()
    executor = _make_response_handler_stub([request], [True, False])

    PyExecutor._handle_responses(executor)

    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    assert executor.active_requests == []
    executor.dist.tp_allgather.assert_called_once_with(True)
    executor._handle_errors.assert_called_once_with(
        error_msg="Request timed out (KV transfer)",
        requests=[request],
        charge_budget=False,
    )
    assert executor._timeout_cleanup_order.mock_calls == [
        call.vote(True),
        call.handle(
            error_msg="Request timed out (KV transfer)",
            requests=[request],
            charge_budget=False,
        ),
    ]


def test_flag_unset_generation_timeout_peer_enters_cleanup():
    executor = _make_response_handler_stub([], [False, True])

    PyExecutor._handle_responses(executor)

    executor.kv_cache_transceiver.cancel_request.assert_not_called()
    executor.dist.tp_allgather.assert_called_once_with(False)
    executor._handle_errors.assert_called_once_with(
        error_msg="Request timed out (KV transfer)",
        requests=[],
        charge_budget=False,
    )
    assert executor._timeout_cleanup_order.mock_calls == [
        call.vote(False),
        call.handle(
            error_msg="Request timed out (KV transfer)",
            requests=[],
            charge_budget=False,
        ),
    ]


def test_flag_unset_context_timeout_preserves_legacy_cleanup():
    request = _make_timeout_request()
    request.py_kv_transfer_start_time = 1.0
    request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = ([], [])
    executor.kv_cache_transceiver.cancel_request.return_value = True
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {
        request.py_request_id: request
    }
    executor._disagg_timed_out_ctx_cancelled_ids = set()
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.return_value = True
    executor._disagg_inflight_cancel_unsupported_logged = False
    executor._end_transfer_and_maybe_terminate = Mock()
    executor._check_cache_transfer_errors = Mock()

    PyExecutor._check_disagg_ctx_cache_transfer_status(executor, 0)

    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    assert request.py_kv_transfer_start_time is None
    assert request.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    executor._end_transfer_and_maybe_terminate.assert_called_once_with(request)
    assert request.py_request_id not in executor._disagg_timed_out_ctx_cancelled_ids


def test_flag_unset_generation_driver_skips_cancel_pipeline():
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.return_value = True
    executor._disagg_inflight_cancel_unsupported_logged = False
    executor._check_disagg_gen_cache_transfer_status = Mock()
    executor._cancel_timed_out_gen_transfers = Mock()
    executor._check_gen_cache_transfer_errors_consensus = Mock()

    PyExecutor._check_disagg_gen_transfer_status(executor)

    executor._check_disagg_gen_cache_transfer_status.assert_called_once_with(0)
    executor._cancel_timed_out_gen_transfers.assert_not_called()
    executor._check_gen_cache_transfer_errors_consensus.assert_not_called()


@pytest.mark.parametrize(
    "backend,runtime,nixl_backend",
    [
        ("UCX", None, "UCX"),
        ("MPI", None, "UCX"),
        ("MOONCAKE", None, "UCX"),
        ("NIXL", "PYTHON", "UCX"),
        ("NIXL", "CPP", "LIBFABRIC"),
    ],
)
def test_feature_opt_in_rejects_unqualified_config(monkeypatch, backend, runtime, nixl_backend):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    monkeypatch.setenv(transceiver_module._NIXL_KVCACHE_BACKEND_ENV, nixl_backend)
    config = CacheTransceiverConfig(backend=backend, transceiver_runtime=runtime)

    with pytest.raises(ValueError, match="currently supported only"):
        transceiver_module.create_kv_cache_transceiver(
            _supported_mapping(),
            Mock(),
            Mock(),
            Mock(),
            config,
            inflight_cancel_supported_by_executor=True,
        )


def test_feature_opt_in_accepts_cpp_nixl_ucx(monkeypatch):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    monkeypatch.delenv(transceiver_module._NIXL_KVCACHE_BACKEND_ENV, raising=False)
    config = CacheTransceiverConfig(backend="DEFAULT", max_tokens_in_buffer=512)
    expected = object()
    constructor = Mock(return_value=expected)
    monkeypatch.setattr(transceiver_module, "BindKvCacheTransceiver", constructor)

    result = transceiver_module.create_kv_cache_transceiver(
        _supported_mapping(),
        Mock(),
        Mock(),
        Mock(),
        config,
        inflight_cancel_supported_by_executor=True,
    )

    assert result is expected
    assert config.backend == "NIXL"
    constructor.assert_called_once()
    assert constructor.call_args.kwargs["_inflight_cancel_config_prevalidated"] is True


def test_feature_preflight_rejects_asymmetric_rank_capability(monkeypatch):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    mapping = _supported_mapping()
    mapping.world_size = 2
    config = CacheTransceiverConfig(backend="NIXL", max_tokens_in_buffer=512)
    dist = Mock()
    dist.allgather.return_value = [
        (True, None),
        (True, "rank-local capability is unsupported"),
    ]

    with pytest.raises(ValueError, match="unsupported on one or more worker ranks"):
        transceiver_module._validate_disagg_inflight_cancel_config(
            config,
            mapping,
            dist,
            inflight_cancel_supported_by_executor=True,
        )

    dist.allgather.assert_called_once_with((True, None))


def test_feature_preflight_rejects_asymmetric_rank_mode(monkeypatch):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    mapping = _supported_mapping()
    mapping.world_size = 2
    config = CacheTransceiverConfig(backend="NIXL", max_tokens_in_buffer=512)
    dist = Mock()
    dist.allgather.return_value = [(True, None), (False, None)]

    with pytest.raises(ValueError, match="must resolve identically"):
        transceiver_module._validate_disagg_inflight_cancel_config(
            config,
            mapping,
            dist,
            inflight_cancel_supported_by_executor=True,
        )


def test_feature_opt_in_rejects_ambiguous_legacy_backend_env(monkeypatch):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    monkeypatch.setenv("TRTLLM_USE_NIXL_KVCACHE", "1")
    monkeypatch.setenv("TRTLLM_USE_UCX_KVCACHE", "1")
    config = CacheTransceiverConfig(backend="DEFAULT")

    with pytest.raises(ValueError, match="multiple legacy backend selectors"):
        transceiver_module.create_kv_cache_transceiver(
            _supported_mapping(), Mock(), Mock(), Mock(), config
        )


def test_direct_cpp_wrapper_rejects_python_runtime_opt_in(monkeypatch):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")

    with pytest.raises(ValueError, match="currently supported only"):
        BindKvCacheTransceiver(
            _supported_mapping(),
            Mock(),
            Mock(),
            Mock(),
            config,
            inflight_cancel_supported_by_executor=True,
        )


def test_flag_unset_preserves_existing_backend_selection(monkeypatch):
    config = CacheTransceiverConfig(backend="UCX")
    expected = object()
    constructor = Mock(return_value=expected)
    monkeypatch.setattr(transceiver_module, "BindKvCacheTransceiver", constructor)

    result = transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)

    assert result is expected
    assert config.backend == "UCX"
    constructor.assert_called_once()


def test_flag_unset_preserves_python_transceiver(monkeypatch):
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")
    expected = object()
    constructor = Mock(return_value=expected)
    fake_module = SimpleNamespace(KvCacheTransceiverV2=constructor)
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.disaggregation.transceiver", fake_module)

    result = transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)

    assert result is expected
    constructor.assert_called_once()


def test_flag_unset_preserves_libfabric_selection(monkeypatch):
    monkeypatch.setenv(transceiver_module._NIXL_KVCACHE_BACKEND_ENV, "LIBFABRIC")
    config = CacheTransceiverConfig(backend="NIXL")
    expected = object()
    constructor = Mock(return_value=expected)
    monkeypatch.setattr(transceiver_module, "BindKvCacheTransceiver", constructor)

    result = transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)

    assert result is expected
    constructor.assert_called_once()


@pytest.mark.parametrize(
    "selector,expected_backend",
    [
        ("TRTLLM_USE_NIXL_KVCACHE", "NIXL"),
        ("TRTLLM_USE_UCX_KVCACHE", "UCX"),
        ("TRTLLM_USE_MOONCAKE_KVCACHE", "MOONCAKE"),
        ("TRTLLM_USE_MPI_KVCACHE", "MPI"),
    ],
)
def test_flag_unset_preserves_legacy_backend_env(monkeypatch, selector, expected_backend):
    monkeypatch.setenv(selector, "1")
    config = CacheTransceiverConfig(backend="DEFAULT")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(transceiver_module, "BindKvCacheTransceiver", constructor)

    transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)

    assert config.backend == expected_backend
    constructor.assert_called_once()


def test_flag_unset_preserves_legacy_backend_env_precedence(monkeypatch):
    for selector in (
        "TRTLLM_USE_MPI_KVCACHE",
        "TRTLLM_USE_MOONCAKE_KVCACHE",
        "TRTLLM_USE_UCX_KVCACHE",
        "TRTLLM_USE_NIXL_KVCACHE",
    ):
        monkeypatch.setenv(selector, "1")
    config = CacheTransceiverConfig(backend="DEFAULT")
    monkeypatch.setattr(transceiver_module, "BindKvCacheTransceiver", Mock())

    transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)

    assert config.backend == "NIXL"


@pytest.mark.parametrize(
    "backend,runtime,nixl_backend,disable_overlap,expected",
    [
        ("NIXL", None, "UCX", None, True),
        ("NIXL", "CPP", "UCX", None, True),
        ("NIXL", "CPP", "UCX", "1", False),
        ("NIXL", "PYTHON", "UCX", None, False),
        ("NIXL", "CPP", "LIBFABRIC", None, False),
        ("UCX", "CPP", "UCX", None, False),
    ],
)
def test_cpp_capability_is_config_scoped(
    monkeypatch, backend, runtime, nixl_backend, disable_overlap, expected
):
    monkeypatch.setenv(transceiver_module._NIXL_KVCACHE_BACKEND_ENV, nixl_backend)
    if disable_overlap is not None:
        monkeypatch.setenv(
            transceiver_module._DISABLE_KV_CACHE_TRANSFER_OVERLAP_ENV,
            disable_overlap,
        )
    config = CacheTransceiverConfig(
        backend=backend, transceiver_runtime=runtime, max_tokens_in_buffer=512
    )
    mapping = _supported_mapping()

    assert (
        transceiver_module._is_disagg_inflight_cancel_config_supported(
            config, mapping, inflight_cancel_supported_by_executor=True
        )
        is expected
    )

    monkeypatch.setattr(transceiver_module, "mapping_to_world_config", lambda mapping: object())
    constructor = Mock(return_value=Mock())
    monkeypatch.setattr(transceiver_module, "CacheTransceiverCpp", constructor)
    monkeypatch.setattr(CacheTransceiverConfig, "_to_pybind", lambda config: object())
    dist = Mock()
    dist.pp_allgather.return_value = [1]
    kv_cache_manager = SimpleNamespace(
        total_num_kv_heads_per_layer=[1],
        head_dim=64,
        tokens_per_block=32,
        dtype=object(),
        num_kv_heads_per_layer=[1],
        impl=object(),
    )

    transceiver = BindKvCacheTransceiver(
        mapping,
        dist,
        kv_cache_manager,
        Mock(),
        config,
        inflight_cancel_supported_by_executor=True,
    )

    assert transceiver.supports_inflight_request_cancellation() is expected
    constructor.assert_called_once()
    assert constructor.call_args.args[-1] is False


def test_cpp_constructor_receives_effective_opt_in(monkeypatch):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    monkeypatch.setattr(transceiver_module, "mapping_to_world_config", lambda mapping: object())
    constructor = Mock(return_value=Mock())
    monkeypatch.setattr(transceiver_module, "CacheTransceiverCpp", constructor)
    monkeypatch.setattr(CacheTransceiverConfig, "_to_pybind", lambda config: object())
    dist = Mock()
    dist.pp_allgather.return_value = [1]
    kv_cache_manager = SimpleNamespace(
        total_num_kv_heads_per_layer=[1],
        head_dim=64,
        tokens_per_block=32,
        dtype=object(),
        num_kv_heads_per_layer=[1],
        impl=object(),
    )

    BindKvCacheTransceiver(
        _supported_mapping(),
        dist,
        kv_cache_manager,
        Mock(),
        CacheTransceiverConfig(
            backend="NIXL",
            transceiver_runtime="CPP",
            max_tokens_in_buffer=512,
        ),
        inflight_cancel_supported_by_executor=True,
    )

    assert constructor.call_args.args[-1] is True


@pytest.mark.parametrize(
    "qualifier",
    [
        "max_tokens_in_buffer",
        "kv_transfer_timeout_ms",
        "pipeline_parallelism",
        "context_parallelism",
        "attention_dp",
        "layerwise",
        "zero_copy",
        "executor",
    ],
)
def test_feature_opt_in_rejects_unqualified_safety_path(monkeypatch, qualifier):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    config_kwargs = {
        "backend": "NIXL",
        "transceiver_runtime": "CPP",
        "max_tokens_in_buffer": 512,
    }
    mapping = _supported_mapping()
    supported_by_executor = True
    if qualifier == "max_tokens_in_buffer":
        config_kwargs["max_tokens_in_buffer"] = None
    elif qualifier == "kv_transfer_timeout_ms":
        config_kwargs["kv_transfer_timeout_ms"] = None
    elif qualifier == "pipeline_parallelism":
        mapping.pp_size = 2
    elif qualifier == "context_parallelism":
        mapping.cp_size = 2
    elif qualifier == "attention_dp":
        mapping.enable_attention_dp = True
    elif qualifier == "layerwise":
        monkeypatch.setenv(transceiver_module._DISAGG_LAYERWISE_ENV, "1")
    elif qualifier == "zero_copy":
        monkeypatch.setenv(transceiver_module._TRY_ZCOPY_FOR_KV_CACHE_TRANSFER_ENV, "1")
    elif qualifier == "executor":
        supported_by_executor = False

    with pytest.raises(ValueError, match="currently supported only"):
        transceiver_module.create_kv_cache_transceiver(
            mapping,
            Mock(),
            Mock(),
            Mock(),
            CacheTransceiverConfig(**config_kwargs),
            inflight_cancel_supported_by_executor=supported_by_executor,
        )


def test_flag_unset_keeps_unsupported_executor_inactive(monkeypatch):
    config = CacheTransceiverConfig(
        backend="NIXL", transceiver_runtime="CPP", max_tokens_in_buffer=512
    )
    monkeypatch.setattr(transceiver_module, "mapping_to_world_config", lambda mapping: object())
    constructor = Mock(return_value=Mock())
    monkeypatch.setattr(transceiver_module, "CacheTransceiverCpp", constructor)
    monkeypatch.setattr(CacheTransceiverConfig, "_to_pybind", lambda config: object())
    dist = Mock()
    dist.pp_allgather.return_value = [1]
    kv_cache_manager = SimpleNamespace(
        total_num_kv_heads_per_layer=[1],
        head_dim=64,
        tokens_per_block=32,
        dtype=object(),
        num_kv_heads_per_layer=[1],
        impl=object(),
    )

    transceiver = BindKvCacheTransceiver(
        _supported_mapping(), dist, kv_cache_manager, Mock(), config
    )

    assert not transceiver.supports_inflight_request_cancellation()
    assert constructor.call_args.args[-1] is False


def test_python_transceiver_capability_defaults_to_unsupported():
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = object.__new__(KvCacheTransceiverV2)

    assert not transceiver.supports_inflight_request_cancellation()
    assert not transceiver.has_poisoned_transfer_buffer()


def test_flag_unset_skips_cpp_poison_query():
    transceiver = SimpleNamespace(
        impl=Mock(),
        _inflight_request_cancellation_enabled=False,
    )

    assert not BindKvCacheTransceiver.has_poisoned_transfer_buffer(transceiver)
    transceiver.impl.has_poisoned_transfer_buffer.assert_not_called()

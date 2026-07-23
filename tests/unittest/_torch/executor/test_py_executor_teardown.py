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

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


class _ShutdownOwnershipStub:
    """Minimal owner of the attributes exercised by ``PyExecutor.shutdown``."""

    shutdown = PyExecutor.shutdown
    _requires_physical_transfer_drain = PyExecutor._requires_physical_transfer_drain
    _discard_pending_transfer_responses_after_shutdown = (
        PyExecutor._discard_pending_transfer_responses_after_shutdown
    )
    _discard_unpublishable_transfer_terminals_after_shutdown = (
        PyExecutor._discard_unpublishable_transfer_terminals_after_shutdown
    )
    _drain_discarded_transfer_terminals_after_shutdown = (
        PyExecutor._drain_discarded_transfer_terminals_after_shutdown
    )
    _get_pending_transfer_response_terminals = PyExecutor._get_pending_transfer_response_terminals
    _has_async_transfer_owner = PyExecutor._has_async_transfer_owner
    _has_pending_ownership_transition = PyExecutor._has_pending_ownership_transition
    _dist_size = staticmethod(PyExecutor._dist_size)
    _get_resource_manager_shutdown_progress = PyExecutor._get_resource_manager_shutdown_progress
    _shutdown_resource_managers_rank_uniform = PyExecutor._shutdown_resource_managers_rank_uniform
    _run_executor_shutdown_phase = PyExecutor._run_executor_shutdown_phase
    _prepare_final_executor_shutdown = PyExecutor._prepare_final_executor_shutdown
    _release_final_executor_objects = PyExecutor._release_final_executor_objects

    def __init__(self, terminal_shutdown_result):
        self.executor_request_queue = Mock()
        self.shutdown_event = threading.Event()
        self.shutdown_event.set()
        self.hang_detector = Mock()
        self.hang_detector.detected.return_value = False
        self.worker_thread = Mock()
        self.dist = SimpleNamespace(pp_size=1)
        self._shutdown_sleep_wakeup_listeners = Mock()
        self.worker_started = True
        self._pending_transfer_responses = []
        self._pending_transfer_terminals = {}
        self.model_engine = None
        self.draft_model_engine = None
        self.kv_cache_transceiver = Mock()
        self.kv_cache_transceiver.requires_physical_drain_before_request_release = True
        self.kv_cache_transceiver.shutdown.side_effect = [
            False,
            terminal_shutdown_result,
        ]
        self.manager = Mock()
        self.resource_manager = SimpleNamespace(resource_managers={"manager": self.manager})
        self.virtual_memory_pools = {}
        self.sampler = object()
        self.dwdp_manager = None


def test_native_shutdown_retries_before_releasing_resource_managers(monkeypatch):
    """A non-drained Python-native transceiver vetoes manager teardown."""
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    stub = _ShutdownOwnershipStub(True)

    with pytest.raises(RuntimeError, match="still owns active transfer targets"):
        stub.shutdown()

    stub.manager.shutdown.assert_not_called()
    stub.shutdown()

    stub.manager.shutdown.assert_called_once_with()
    assert stub.kv_cache_transceiver.shutdown.call_count == 2
    stub.executor_request_queue.enqueue_shutdown_request.assert_called_once_with()
    stub.worker_thread.join.assert_called_once_with()
    stub._shutdown_sleep_wakeup_listeners.assert_called_once_with()


def test_native_shutdown_rejects_legacy_none_result(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    stub = _ShutdownOwnershipStub(None)

    with pytest.raises(RuntimeError, match="still owns active transfer targets"):
        stub.shutdown()
    with pytest.raises(RuntimeError, match="did not prove physical drain"):
        stub.shutdown()

    stub.manager.shutdown.assert_not_called()


@pytest.mark.parametrize("failing_phase", ["sampler", "dwdp_manager"])
def test_native_finalization_failure_is_not_replayed(monkeypatch, failing_phase):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )

    class FailingSampler:
        def __init__(self):
            self.stop_calls = 0

        @staticmethod
        def async_worker_enabled():
            return True

        def async_worker_stop(self):
            self.stop_calls += 1
            raise RuntimeError("sampler finalization failed")

    class TestDwdpManager:
        def __init__(self, *, fail):
            self.exit_calls = 0
            self.fail = fail

        def __exit__(self, *_):
            self.exit_calls += 1
            if self.fail:
                raise RuntimeError("DWDP finalization failed")

    stub = _ShutdownOwnershipStub(True)
    stub.kv_cache_transceiver.shutdown.side_effect = None
    stub.kv_cache_transceiver.shutdown.return_value = True
    if failing_phase == "sampler":
        sampler = FailingSampler()
        monkeypatch.setattr(
            "tensorrt_llm._torch.pyexecutor.py_executor.AsyncWorkerMixin",
            FailingSampler,
        )
        stub.sampler = sampler
        dwdp_manager = TestDwdpManager(fail=False)
        stub.dwdp_manager = dwdp_manager
        failed_object = sampler
    else:
        dwdp_manager = TestDwdpManager(fail=True)
        stub.dwdp_manager = dwdp_manager
        failed_object = dwdp_manager

    with pytest.raises(RuntimeError, match="finalization failed"):
        stub.shutdown()
    with pytest.raises(RuntimeError, match="outcome is in doubt"):
        stub.shutdown()

    if failing_phase == "sampler":
        assert failed_object.stop_calls == 1
    else:
        assert failed_object.exit_calls == 1
    assert dwdp_manager.exit_calls == 1
    stub.manager.shutdown.assert_not_called()
    assert hasattr(stub, "model_engine")
    stub._shutdown_sleep_wakeup_listeners.assert_called_once_with()


def test_estimation_teardown_retries_only_after_transceiver_drains():
    """Lifecycle-capable teardown requires explicit positive drain proof."""
    transceiver = Mock()
    transceiver.requires_physical_drain_before_request_release = True
    transceiver.shutdown.side_effect = [False, True]
    py_executor = SimpleNamespace(kv_cache_transceiver=transceiver)
    kv_cache_creator = Mock()
    resources = {}

    with pytest.raises(RuntimeError, match="did not prove physical drain"):
        py_executor_creator._teardown_kv_cache_managers_after_transceiver_shutdown(
            py_executor, kv_cache_creator, resources
        )

    kv_cache_creator.teardown_managers.assert_not_called()
    py_executor_creator._teardown_kv_cache_managers_after_transceiver_shutdown(
        py_executor, kv_cache_creator, resources
    )

    kv_cache_creator.teardown_managers.assert_called_once_with(resources)
    assert transceiver.shutdown.call_count == 2


def test_estimation_teardown_rejects_native_none_result():
    transceiver = Mock()
    transceiver.requires_physical_drain_before_request_release = True
    transceiver.shutdown.return_value = None
    py_executor = SimpleNamespace(kv_cache_transceiver=transceiver)
    kv_cache_creator = Mock()

    with pytest.raises(RuntimeError, match="did not prove physical drain"):
        py_executor_creator._teardown_kv_cache_managers_after_transceiver_shutdown(
            py_executor, kv_cache_creator, {}
        )

    kv_cache_creator.teardown_managers.assert_not_called()


def test_estimation_teardown_preserves_legacy_none_result():
    transceiver = Mock()
    transceiver.requires_physical_drain_before_request_release = False
    transceiver.shutdown.return_value = None
    py_executor = SimpleNamespace(kv_cache_transceiver=transceiver)
    kv_cache_creator = Mock()
    resources = {}

    py_executor_creator._teardown_kv_cache_managers_after_transceiver_shutdown(
        py_executor, kv_cache_creator, resources
    )

    kv_cache_creator.teardown_managers.assert_called_once_with(resources)


@pytest.mark.parametrize("shutdown_result", [True, None])
def test_estimation_teardown_rejects_inflight_transfer_owner(shutdown_result):
    transceiver = Mock()
    transceiver.requires_physical_drain_before_request_release = False
    transceiver.shutdown.return_value = shutdown_result
    transfer_manager = Mock()
    transfer_manager.has_any_inflight_requests.return_value = True
    py_executor = SimpleNamespace(
        kv_cache_transceiver=transceiver,
        async_transfer_manager=transfer_manager,
    )
    kv_cache_creator = Mock()

    with pytest.raises(RuntimeError, match="transfer ownership is still active"):
        py_executor_creator._teardown_kv_cache_managers_after_transceiver_shutdown(
            py_executor, kv_cache_creator, {}
        )

    transceiver.shutdown.assert_called_once_with()
    transfer_manager.has_any_inflight_requests.assert_called_once_with()
    kv_cache_creator.teardown_managers.assert_not_called()

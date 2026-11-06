# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transfer drain must precede pool release, and deregistration must follow it."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import KvCacheTransceiver
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def worker() -> Iterator[TransferWorker]:
    worker = object.__new__(TransferWorker)
    worker._retirement_watchdog = Mock()
    worker._sender = Mock()
    worker._receiver = Mock()
    worker._rank_info_server = Mock()
    worker._agent = Mock(bounce_enabled=False)
    worker._registered_mem = [object()]
    yield worker
    worker._retirement_watchdog.require_retired.side_effect = None
    worker.shutdown()


def test_worker_drain_retains_registration_until_shutdown(worker: TransferWorker) -> None:
    agent = worker._agent
    registered = worker._registered_mem.copy()
    worker.prepare_shutdown()
    worker.prepare_shutdown()

    worker._sender.shutdown.assert_called_once_with()
    worker._receiver.shutdown.assert_called_once_with()
    worker._rank_info_server.shutdown.assert_called_once_with()
    worker._retirement_watchdog.require_retired.assert_called_once_with()
    assert worker._agent is agent
    assert worker._registered_mem == registered
    agent.deregister_memory.assert_not_called()
    agent.shutdown.assert_not_called()

    worker.shutdown()
    worker.shutdown()
    agent.deregister_memory.assert_called_once_with(registered[0])
    agent.shutdown.assert_called_once_with()
    worker._sender.shutdown.assert_called_once_with()
    assert worker._registered_mem == []


def test_worker_drain_refusal_preserves_registration_and_can_retry(worker: TransferWorker) -> None:
    agent = worker._agent
    registered = worker._registered_mem.copy()
    worker._retirement_watchdog.require_retired.side_effect = RuntimeError("DMA still active")
    with pytest.raises(RuntimeError, match="DMA still active"):
        worker.prepare_shutdown()
    assert not getattr(worker, "_shutdown_prepared", False)
    assert worker._registered_mem == registered
    worker._sender.shutdown.assert_not_called()
    agent.deregister_memory.assert_not_called()
    agent.shutdown.assert_not_called()

    worker._retirement_watchdog.require_retired.side_effect = None
    worker.prepare_shutdown()
    assert worker._shutdown_prepared
    agent.deregister_memory.assert_not_called()


def test_transceiver_drain_refusal_preserves_sessions_and_registration(
    worker: TransferWorker,
) -> None:
    session = SimpleNamespace(
        _enforce_physical_ownership=True,
        resources_drained=Mock(return_value=False),
        close=Mock(return_value=True),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {1: session}
    transceiver._recv_reqs = {1: object()}
    transceiver._transfer_worker = worker
    agent = worker._agent

    with pytest.raises(RuntimeError, match="physical resources remain active"):
        transceiver.prepare_shutdown()
    assert transceiver._recv_sessions[1] is session
    session.close.assert_not_called()
    worker._receiver.shutdown.assert_not_called()
    agent.deregister_memory.assert_not_called()

    session.resources_drained.return_value = True
    transceiver.prepare_shutdown()
    transceiver.prepare_shutdown()
    assert transceiver._recv_sessions == {}
    session.close.assert_called_once_with()
    worker._receiver.shutdown.assert_called_once_with()
    agent.deregister_memory.assert_not_called()
    assert not getattr(transceiver, "_shutdown_complete", False)

    transceiver.shutdown()
    assert transceiver._shutdown_complete
    agent.deregister_memory.assert_called_once()
    agent.shutdown.assert_called_once_with()


def test_legacy_transceiver_prepare_uses_full_shutdown() -> None:
    transceiver = SimpleNamespace(shutdown=Mock(side_effect=RuntimeError("DMA still active")))
    with pytest.raises(RuntimeError, match="DMA still active"):
        KvCacheTransceiver.prepare_shutdown(transceiver)
    transceiver.shutdown.assert_called_once_with()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How long an idle executor waits on its request queue."""

import datetime
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


def _executor(transceiver, disable_mpi: bool = False) -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = transceiver
    executor._disable_mpi = disable_mpi
    executor.kv_connector_manager = None
    executor.enable_attention_dp = False
    executor.control_requests = []
    executor.request_accumulated = []
    executor.is_shutdown = False
    executor.dist = SimpleNamespace(rank=0, tp_size=1, has_pp=False, cp_size=1)
    executor.hang_detector = SimpleNamespace(pause=nullcontext)
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_from_request_queue.return_value = []
    executor.request_broadcaster = Mock()
    executor.request_broadcaster.broadcast.return_value = ([], {})
    executor._handle_special_queue_items = Mock(return_value=[])
    return executor


def _waiting_queue() -> Mock:
    queue = Mock()
    queue.__len__ = Mock(return_value=0)
    return queue


@pytest.mark.parametrize(
    ("has_transceiver", "disable_mpi", "expected_s"),
    [
        (False, False, None),
        (False, True, 1200.0),
        (True, False, py_executor._TRANSCEIVER_IDLE_WAIT_S),
        (True, True, py_executor._TRANSCEIVER_IDLE_WAIT_S),
    ],
)
def test_idle_request_wait_timeout(has_transceiver, disable_mpi, expected_s):
    executor = _executor(object() if has_transceiver else None, disable_mpi)
    timeout = executor._idle_request_wait_timeout()
    if expected_s is None:
        assert timeout is None
    else:
        assert timeout == datetime.timedelta(seconds=expected_s)


def test_idle_executor_with_transceiver_does_not_block_indefinitely():
    """An idle disaggregated worker still iterates to settle its KV transfers.

    With every request handed to the transfer manager the executor is idle, and
    only its iterations answer a transfer request for a request it no longer
    holds. A blocking wait would leave that requester waiting until the next
    request arrives.
    """
    executor = _executor(transceiver=object())
    executor._fetch_and_enqueue_requests(_waiting_queue(), total_num_live_requests=0)
    timeout = executor.executor_request_queue.get_from_request_queue.call_args.args[0]
    assert timeout is not None
    assert 0 < timeout.total_seconds() <= py_executor._TRANSCEIVER_IDLE_WAIT_S


def test_idle_executor_without_transceiver_blocks():
    executor = _executor(transceiver=None)
    executor._fetch_and_enqueue_requests(_waiting_queue(), total_num_live_requests=0)
    assert executor.executor_request_queue.get_from_request_queue.call_args.args[0] is None


def test_busy_executor_does_not_wait():
    executor = _executor(transceiver=object())
    executor._fetch_and_enqueue_requests(_waiting_queue(), total_num_live_requests=1)
    timeout = executor.executor_request_queue.get_from_request_queue.call_args.args[0]
    assert timeout.total_seconds() == 0

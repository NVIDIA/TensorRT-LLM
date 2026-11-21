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

from collections.abc import Callable
from queue import Queue
from threading import Event
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor.base_worker import AwaitResponseHelper, BaseWorker
from tensorrt_llm.executor.rpc_worker_mixin import RpcWorkerMixin
from tensorrt_llm.executor.utils import ErrorResponse

pytestmark = pytest.mark.cpu_only


class _WorkerBaseStub:
    def await_responses(self, timeout):
        self.await_responses_timeout = timeout
        responses = self.engine_responses
        self.engine_responses = []
        if self.await_responses_hook is not None:
            self.await_responses_hook()
        return responses


class _RpcWorkerStub(RpcWorkerMixin, _WorkerBaseStub):
    _pop_result = BaseWorker._pop_result

    def __init__(self, engine_responses):
        self.engine_responses = engine_responses
        self.await_responses_hook: Callable[[], None] | None = None
        self.engine = SimpleNamespace(
            is_shutdown=False, _event_loop_error=None, _event_loop_error_delivered=Event()
        )
        self._results: dict[int, object] = {}
        self._client_id_to_request_id: dict[int, int] = {}
        self.rank = 0
        self._fetch_timeout = 0.1
        self._response_queue = Queue()
        self.enable_postprocess_parallel = False
        self._await_response_helper = AwaitResponseHelper(self)
        self._await_response_helper.responses_handler = self._responses_handler
        self.handler_responses = None

    def _register_pending_request(self, client_id: int, request_id: int) -> None:
        self._results[client_id] = object()
        self._client_id_to_request_id[client_id] = request_id

    def _responses_handler(self, responses):
        self.handler_responses = responses
        if responses:
            self._response_queue.put(responses)


@pytest.mark.parametrize(
    "engine_responses, expected_engine_responses",
    [
        (["first-response", "second-response"], ["first-response", "second-response"]),
        (
            [None, "first-response", None, "second-response", None],
            ["first-response", "second-response"],
        ),
        ([None, None], []),
        ([], []),
    ],
    ids=["normal", "mixed-none", "all-none", "empty"],
)
def test_fetch_responses_dispatches_engine_responses_and_submission_errors(
    engine_responses, expected_engine_responses
):
    worker = _RpcWorkerStub(engine_responses)
    original_responses = list(engine_responses)
    submission_error = ErrorResponse(client_id=1, error_msg="submit failed", request_id=2)
    worker._await_response_helper.temp_error_responses.put(submission_error)

    responses = worker.fetch_responses(timeout=0.25)

    assert worker.await_responses_timeout == 0.25
    expected = [*expected_engine_responses, submission_error]
    assert worker.handler_responses == expected
    assert responses == expected
    assert responses[-1] is submission_error
    assert engine_responses == original_responses
    assert worker._await_response_helper.temp_error_responses.empty()


def test_fetch_responses_fails_pending_requests_once_after_loop_crash() -> None:
    worker = _RpcWorkerStub([])
    worker.engine._event_loop_error = RuntimeError("loop crashed")
    worker.engine.is_shutdown = True
    worker._register_pending_request(1, 11)
    worker._register_pending_request(2, 12)

    assert worker.fetch_responses() == [
        ErrorResponse(1, "Event loop terminated with error: loop crashed", -1),
        ErrorResponse(2, "Event loop terminated with error: loop crashed", -1),
    ]
    assert not worker._results
    assert not worker._client_id_to_request_id
    assert not worker.engine._event_loop_error_delivered.is_set()
    assert worker.fetch_responses() == []


def test_fetch_responses_fails_request_registered_after_loop_crash() -> None:
    worker = _RpcWorkerStub([])
    worker.engine._event_loop_error = RuntimeError("loop crashed")
    worker.engine.is_shutdown = True
    assert worker.fetch_responses() == []

    worker._register_pending_request(3, 13)

    assert worker.fetch_responses() == [
        ErrorResponse(3, "Event loop terminated with error: loop crashed", -1),
    ]


def test_fetch_responses_dispatches_flushed_final_response_before_crash_errors() -> None:
    worker = _RpcWorkerStub([])
    worker.engine._event_loop_error = RuntimeError("loop crashed")
    worker._register_pending_request(1, 11)
    worker._register_pending_request(2, 12)
    final_response = SimpleNamespace(client_id=1)

    def finish_cleanup() -> None:
        worker.engine_responses = [final_response]
        worker.engine.is_shutdown = True
        worker.await_responses_hook = None

    def dispatch_final_responses(responses: list[SimpleNamespace]) -> None:
        worker._responses_handler(responses)
        for response in responses:
            worker._pop_result(response.client_id)

    worker.await_responses_hook = finish_cleanup
    worker._await_response_helper.responses_handler = dispatch_final_responses

    assert worker.fetch_responses() == []
    assert worker.fetch_responses() == [
        final_response,
        ErrorResponse(2, "Event loop terminated with error: loop crashed", -1),
    ]

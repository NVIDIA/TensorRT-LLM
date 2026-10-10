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

from queue import Queue
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor.base_worker import AwaitResponseHelper, BaseWorker
from tensorrt_llm.executor.rpc_worker_mixin import RpcWorkerMixin
from tensorrt_llm.executor.utils import ErrorResponse, RequestError
from tensorrt_llm.executor.worker import GenerationExecutorWorker

pytestmark = pytest.mark.cpu_only


class _WorkerBaseStub:
    def set_result_queue(self, queue):
        self.result_queue = queue

    def await_responses(self, timeout):
        self.await_responses_timeout = timeout
        return ["forward", "consume", None]


class _RpcWorkerStub(RpcWorkerMixin, _WorkerBaseStub):
    def __init__(self):
        self.init_rpc_worker(0, "ipc:///unused-test-address", b"test-key")
        self._fetch_timeout = 0.1
        self.enable_postprocess_parallel = False
        self._await_response_helper = AwaitResponseHelper(self)
        self._await_response_helper.responses_handler = self._responses_handler
        self.handler_responses = None
        self.callback_responses = []

    def _responses_handler(self, responses):
        self.handler_responses = responses
        if responses:
            self._response_queue.put(responses)

    def _engine_response_callback(self, response):
        self.callback_responses.append(response)
        if response in ("consume", None):
            return None
        return f"processed-{response}"


def test_fetch_responses_processes_and_filters_engine_responses():
    worker = _RpcWorkerStub()
    worker._await_response_helper.temp_error_responses.put("temporary-error")

    responses = worker.fetch_responses(timeout=0.25)

    assert worker.await_responses_timeout == 0.25
    assert worker.callback_responses == ["forward", "consume", None]
    assert worker.handler_responses == ["processed-forward", "temporary-error"]
    assert responses == ["processed-forward", "temporary-error"]


@pytest.mark.parametrize("postproc", [False, True])
def test_rejected_submission_reaches_response_stream(postproc, monkeypatch):
    worker = _RpcWorkerStub()
    worker._results = {}
    worker._client_id_to_request_id = {}
    worker._pop_result = BaseWorker._pop_result.__get__(worker)
    worker.result_queue = worker._response_queue
    worker.frontend_result_queues = None
    worker.postproc_queues = [Queue()] if postproc else None
    worker.postproc_config = SimpleNamespace(num_postprocess_workers=int(postproc))
    worker._await_response_helper.enable_postprocprocess_parallel = postproc
    del worker._await_response_helper.responses_handler

    def reject(self, request):
        self._results[request.id] = object()
        raise RequestError("Cannot enqueue requests while executor admission is parked")

    monkeypatch.setattr(_WorkerBaseStub, "submit", reject, raising=False)
    monkeypatch.setattr(_WorkerBaseStub, "await_responses", lambda self, timeout: [])
    with pytest.raises(RequestError, match="Cannot enqueue"):
        worker.submit(SimpleNamespace(id=42))

    responses = worker.fetch_responses()
    assert responses == [
        ErrorResponse(42, "Cannot enqueue requests while executor admission is parked", 42)
    ]
    assert not worker._results
    assert not worker._client_id_to_request_id
    assert worker.fetch_responses() == []
    if postproc:
        assert worker.postproc_queues[0].empty()


def test_mpi_worker_leaves_submission_errors_to_caller(monkeypatch):
    worker = object.__new__(GenerationExecutorWorker)
    worker.doing_shutdown = True
    worker._await_response_helper = SimpleNamespace(temp_error_responses=Queue())

    def reject(self, request):
        raise RequestError("Cannot enqueue requests while executor admission is parked")

    monkeypatch.setattr(BaseWorker, "submit", reject)
    with pytest.raises(RequestError, match="Cannot enqueue"):
        worker.submit(SimpleNamespace(id=42))

    # The MPI caller queues its own error; in-process callers receive it
    # synchronously. Neither uses the mixin's generation response stream.
    assert worker._await_response_helper.temp_error_responses.empty()

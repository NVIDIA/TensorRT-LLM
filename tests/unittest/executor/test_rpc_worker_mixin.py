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
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from threading import Event, RLock
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor.base_worker import AwaitResponseHelper, BaseWorker
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.rpc_worker_mixin import RpcWorkerMixin
from tensorrt_llm.executor.utils import ErrorResponse, RequestError
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


class _WorkerBaseStub:
    submit = BaseWorker.submit

    def start(self) -> None:
        pass

    def _handle_background_error(self) -> None:
        pass

    def _get_logprob_params(self, request: GenerationRequest) -> None:
        return None

    def _enqueue_request(self, request: GenerationRequest) -> int:
        return request.id + 100

    def set_postproc_queues(self, queues, coexist_with_result_queue: bool) -> None:
        pass

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
            shutdown_event=Event(),
            execution_error=None,
            _event_loop_error_delivered=Event(),
        )
        self._results: dict[int, object] = {}
        self._client_id_to_request_id: dict[int, int] = {}
        self._results_lock = RLock()
        self.rank = 0
        self._fetch_timeout = 0.1
        self._response_queue = Queue()
        self._error_queue = Queue()
        self.shutdown_event = Event()
        self.enable_postprocess_parallel = False
        self._await_response_helper = AwaitResponseHelper(self)
        self._await_response_helper.responses_handler = self._responses_handler
        self.handler_responses = None

    def _register_pending_request(self, client_id: int, request_id: int) -> None:
        self._results[client_id] = object()
        self._client_id_to_request_id[client_id] = request_id

    def _crash_engine(self, message: str) -> None:
        self.engine.execution_error = RuntimeError(message)
        self.engine.shutdown_event.set()

    def _responses_handler(self, responses):
        self.handler_responses = responses
        if responses:
            self._response_queue.put(responses)


class _FakeIpcQueue:
    def __init__(self, *args, **kwargs) -> None:
        self.address = "unused"
        self.socket = SimpleNamespace(poll=lambda timeout: True)

    def setup_lazily(self) -> None:
        pass

    def get(self) -> None:
        # The shutdown sentinel: the collector thread exits at once.
        return None


class _FakeProcessPool:
    def __init__(self, max_workers: int) -> None:
        pass

    def submit(self, *args) -> Future:
        return Future()


def _start_postproc_workers(
    worker: _RpcWorkerStub, monkeypatch: pytest.MonkeyPatch, num: int = 2
) -> list[Future]:
    monkeypatch.setattr("tensorrt_llm.executor.ipc.IpcQueue", _FakeIpcQueue)
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _FakeProcessPool)
    worker.postproc_config = SimpleNamespace(
        num_postprocess_workers=num, postprocess_tokenizer_dir=None, post_processor_hook=None
    )
    worker._postproc_collector_stop = Event()
    worker._postproc_futures = []
    worker.init_postproc_workers()
    worker._postproc_collector_thread.join(timeout=5)
    return worker._postproc_futures


def _request(client_id: int) -> GenerationRequest:
    return GenerationRequest([1, 2, 3], SamplingParams(max_tokens=1)).set_id(client_id)


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


def test_fetch_responses_fails_pending_and_late_requests_after_loop_crash() -> None:
    worker = _RpcWorkerStub([])
    worker._crash_engine("loop crashed")
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

    worker.submit(_request(3))

    assert worker.fetch_responses() == [
        ErrorResponse(3, "Event loop terminated with error: loop crashed", -1),
    ]
    assert not worker._results


def test_fetch_responses_dispatches_flushed_final_response_before_crash_errors() -> None:
    worker = _RpcWorkerStub([])
    worker.engine.execution_error = RuntimeError("loop crashed")
    worker._register_pending_request(1, 11)
    worker._register_pending_request(2, 12)
    final_response = SimpleNamespace(client_id=1)

    def finish_cleanup() -> None:
        worker.engine_responses = [final_response]
        worker.engine.shutdown_event.set()
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


def test_submit_rejection_reaches_client_on_response_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _RpcWorkerStub([])

    def reject(submitted: GenerationRequest) -> int:
        raise RequestError("PyExecutor has already been shutdown.")

    monkeypatch.setattr(worker, "_enqueue_request", reject)

    assert worker.submit(_request(3)) is None
    assert worker.fetch_responses() == [
        ErrorResponse(3, "PyExecutor has already been shutdown.", 3),
    ]
    assert not worker._results
    assert not worker._client_id_to_request_id


def test_submit_rejection_raises_without_response_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _RpcWorkerStub([])
    worker._response_queue = None

    def reject(submitted: GenerationRequest) -> int:
        raise RequestError("Failed to load LoRA adapter")

    monkeypatch.setattr(worker, "_enqueue_request", reject)

    with pytest.raises(RequestError, match="Failed to load LoRA adapter"):
        worker.submit(_request(3))
    assert worker._await_response_helper.temp_error_responses.empty()


def test_postproc_failure_fails_pending_requests_on_its_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _RpcWorkerStub([])
    futures = _start_postproc_workers(worker, monkeypatch)
    worker._register_pending_request(1, 11)
    worker._register_pending_request(2, 12)

    futures[1].set_exception(RuntimeError("postproc crashed"))

    assert worker.fetch_responses() == [
        ErrorResponse(1, "PostprocWorker process died: postproc crashed", -1),
    ]
    assert list(worker._results) == [2]
    assert worker._error_queue.empty()


def test_postproc_failure_without_pending_requests_on_its_shard_reports_background_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _RpcWorkerStub([])
    futures = _start_postproc_workers(worker, monkeypatch)
    worker._register_pending_request(2, 12)
    error = RuntimeError("postproc crashed")

    futures[1].set_exception(error)

    assert worker._error_queue.get_nowait() is error
    assert list(worker._results) == [2]
    assert worker.fetch_responses() == []


def test_postproc_failure_does_not_report_request_retired_by_loop_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _RpcWorkerStub([])
    futures = _start_postproc_workers(worker, monkeypatch)
    worker._register_pending_request(1, 11)
    worker._crash_engine("loop crashed")
    assert worker.fetch_responses() == [
        ErrorResponse(1, "Event loop terminated with error: loop crashed", -1),
    ]

    futures[1].set_exception(RuntimeError("postproc crashed"))

    assert worker.fetch_responses() == []
    assert worker._error_queue.empty()


class _ClientIdThatActsDuringShardScan(int):
    during_scan: Callable[[], None]

    def __mod__(self, other: int) -> int:
        # Stands in for an RPC thread changing _results while the callback picks shards.
        self.during_scan()
        return int(self) % other


def test_postproc_failure_tolerates_submission_during_shard_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _RpcWorkerStub([])
    futures = _start_postproc_workers(worker, monkeypatch)
    client_id = _ClientIdThatActsDuringShardScan(1)
    client_id.during_scan = lambda: worker._results.setdefault(99, object())
    worker._register_pending_request(client_id, 11)

    futures[1].set_exception(RuntimeError("postproc crashed"))

    assert worker.fetch_responses() == [
        ErrorResponse(1, "PostprocWorker process died: postproc crashed", -1),
    ]
    assert list(worker._results) == [99]


def test_postproc_failure_skips_request_rolled_back_during_shard_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _RpcWorkerStub([])
    futures = _start_postproc_workers(worker, monkeypatch)
    client_id = _ClientIdThatActsDuringShardScan(1)
    # submit answers its own rejected request on the stream after this rollback.
    client_id.during_scan = lambda: worker._results.pop(3, None)
    worker._register_pending_request(client_id, 11)
    worker._register_pending_request(3, 13)

    futures[1].set_exception(RuntimeError("postproc crashed"))

    assert worker.fetch_responses() == [
        ErrorResponse(1, "PostprocWorker process died: postproc crashed", -1),
    ]


def test_postproc_failure_claim_excludes_concurrent_retirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _RpcWorkerStub([])
    futures = _start_postproc_workers(worker, monkeypatch)
    worker._register_pending_request(1, 11)
    pop_result = worker._pop_result
    retired_elsewhere = []

    def try_retiring(client_id: int) -> None:
        # Non-blocking, as a blocking acquire would wait for this test's own claim.
        if worker._results_lock.acquire(blocking=False):
            try:
                retired_elsewhere.append(pop_result(client_id))
            finally:
                worker._results_lock.release()

    with ThreadPoolExecutor(max_workers=1) as other_thread:

        def pop_after_concurrent_attempt(client_id: int) -> object | None:
            other_thread.submit(try_retiring, client_id).result(timeout=5)
            return pop_result(client_id)

        monkeypatch.setattr(worker, "_pop_result", pop_after_concurrent_attempt)
        futures[1].set_exception(RuntimeError("postproc crashed"))

    assert retired_elsewhere == []
    assert worker.fetch_responses() == [
        ErrorResponse(1, "PostprocWorker process died: postproc crashed", -1),
    ]
    assert not worker._results
    assert not worker._client_id_to_request_id

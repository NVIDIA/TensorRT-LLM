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

from tensorrt_llm.executor.base_worker import AwaitResponseHelper
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin
from tensorrt_llm.executor.rpc_worker_mixin import RpcWorkerMixin
from tensorrt_llm.executor.utils import ErrorResponse, RequestError
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


class _WorkerBaseStub:
    def await_responses(self, timeout):
        self.await_responses_timeout = timeout
        return self.engine_responses


class _RpcWorkerStub(RpcWorkerMixin, _WorkerBaseStub):
    def __init__(self, engine_responses):
        self.engine_responses = engine_responses
        self.rank = 0
        self._fetch_timeout = 0.1
        self._response_queue = Queue()
        self.enable_postprocess_parallel = False
        self._await_response_helper = AwaitResponseHelper(self)
        self._await_response_helper.responses_handler = self._responses_handler
        self.handler_responses = None

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


class _ProxyStub:
    """An RPC proxy whose fire-and-forget submit calls the worker directly."""

    def __init__(self, worker: _RpcWorkerStub) -> None:
        self._results = {}
        self.rpc_client = SimpleNamespace(
            submit_from_proxy=lambda request: SimpleNamespace(
                remote=lambda need_response: worker.submit_from_proxy(request)
            )
        )

    def _get_next_client_id(self) -> int:
        return 7

    def _get_logprob_params(self, request: GenerationRequest) -> None:
        return None

    def _handle_background_error(self, error: object = None) -> None:
        GenerationExecutor._handle_background_error(self, error)


def test_submit_from_proxy_returns_rejection_to_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    # A request the worker rejects at admission fails the caller's result
    # instead of leaving it waiting on a send that expects no reply.
    def reject(self: _WorkerBaseStub, request: GenerationRequest) -> None:
        raise RequestError("prompt_embeds is not supported by this model.")

    monkeypatch.setattr(_WorkerBaseStub, "submit", reject, raising=False)
    worker = _RpcWorkerStub([])
    proxy = _ProxyStub(worker)
    request = GenerationRequest([1, 2], SamplingParams(max_tokens=1))

    result = RpcExecutorMixin.submit(proxy, request)
    responses = worker.fetch_responses(timeout=0)
    # The error crosses the RPC boundary as text.
    assert [type(response.error_msg) for response in responses] == [str]
    RpcExecutorMixin.handle_responses(proxy, responses)

    with pytest.raises(RequestError, match="prompt_embeds is not supported"):
        result.result(timeout=1)
    assert proxy._results == {}

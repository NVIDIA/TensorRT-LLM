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
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm.executor.base_worker import AwaitResponseHelper, BaseWorker
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin
from tensorrt_llm.executor.rpc_worker_mixin import RpcWorkerMixin
from tensorrt_llm.executor.utils import ErrorResponse, RequestError
from tensorrt_llm.executor.worker import GenerationExecutorWorker
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("send_fails", [False, True])
def test_proxy_registers_result_before_immediate_response(send_fails: bool) -> None:
    """Retain immediate rejection responses and clean up failed sends.

    Args:
        send_fails: Whether the RPC send fails instead of returning an error reply.
    """
    proxy = RpcExecutorMixin()
    proxy._results = {}
    proxy.rpc_client = MagicMock()
    proxy._get_next_client_id = MagicMock(return_value=17)
    proxy._get_logprob_params = MagicMock(return_value=None)
    proxy._handle_background_error = None
    rejection = ErrorResponse(17, "request rejected", -1)

    def dispatch(*, need_response: bool) -> None:
        """Exercise the real response handler before submit returns."""
        assert not need_response
        if send_fails:
            raise RuntimeError("send failed")
        proxy.handle_responses([rejection])

    proxy.rpc_client.submit.return_value.remote.side_effect = dispatch
    request = GenerationRequest([1], SamplingParams(max_tokens=1))
    if send_fails:
        with pytest.raises(RuntimeError, match="send failed"):
            proxy.submit(request)
    else:
        result = proxy.submit(request)
        assert result.queue.get_nowait() == rejection
    assert not proxy._results


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


@pytest.mark.parametrize("transport", ["ipc", "ipc-stats-rpc", "rpc"])
@pytest.mark.parametrize("error_type", [RequestError, RuntimeError])
def test_submit_rejection_preserves_transport_error_route(
    transport: str, error_type: type[Exception]
) -> None:
    """Keep admission errors on the configured transport without replacing them.

    Args:
        transport: Worker response transport, including stats-only RPC over IPC.
        error_type: Request validation or admission failure from the base worker.
    """
    # Bypass model loading, but retain the real worker MRO and RPC initialization.
    worker = object.__new__(GenerationExecutorWorker)
    worker.postproc_queues = None
    worker.frontend_result_queues = None
    if transport == "rpc":
        worker.init_rpc_worker(0, "ipc://unused", b"test")
    elif transport == "ipc-stats-rpc":
        worker.rpc_server = object()
    request = GenerationRequest(prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=1))
    request.set_id(17)
    error = error_type("request rejected")
    with (
        patch.object(BaseWorker, "submit", side_effect=error) as submit,
        pytest.raises(error_type, match="request rejected") as caught,
    ):
        worker.submit(request)

    assert caught.value is error
    submit.assert_called_once_with(request)
    if transport == "rpc":
        assert worker._response_queue.get_nowait() == [ErrorResponse(17, str(error), -1)]
        assert worker._response_queue.empty()
    else:
        assert worker._response_queue is None


@pytest.mark.parametrize("rpc_enabled", [False, True], ids=["ipc", "rpc"])
def test_successful_submit_does_not_emit_error(rpc_enabled: bool) -> None:
    """Return the base result without adding an RPC error response.

    Args:
        rpc_enabled: Whether the worker initializes an RPC response stream.
    """
    worker = object.__new__(GenerationExecutorWorker)
    worker.postproc_queues = None
    worker.frontend_result_queues = None
    if rpc_enabled:
        worker.init_rpc_worker(0, "ipc://unused", b"test")
    request = GenerationRequest(prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=1))
    request.set_id(17)
    expected_result = object()
    with patch.object(BaseWorker, "submit", return_value=expected_result) as submit:
        assert worker.submit(request) is expected_result
    submit.assert_called_once_with(request)
    if rpc_enabled:
        assert worker._response_queue.empty()
    else:
        assert worker._response_queue is None

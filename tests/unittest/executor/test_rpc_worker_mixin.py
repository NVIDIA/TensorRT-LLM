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

import pytest

from tensorrt_llm.executor.base_worker import AwaitResponseHelper
from tensorrt_llm.executor.rpc_worker_mixin import RpcWorkerMixin

pytestmark = pytest.mark.cpu_only


class _WorkerBaseStub:
    def await_responses(self, timeout):
        self.await_responses_timeout = timeout
        return ["forward", "consume", None]


class _RpcWorkerStub(RpcWorkerMixin, _WorkerBaseStub):
    def __init__(self):
        self.rank = 0
        self._fetch_timeout = 0.1
        self._response_queue = Queue()
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

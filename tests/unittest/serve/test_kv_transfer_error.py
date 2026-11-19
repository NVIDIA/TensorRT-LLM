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
"""HTTP status returned when distributed KV transfer fails during generation."""

from http import HTTPStatus

import pytest
from test_common.serve.app_factory import AggConfig
from test_common.serve.fakes import ScriptedOutput

from tensorrt_llm.executor.utils import RequestError

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("role", ["context", "generation"])
def test_chat_kv_transfer_failure_returns_service_unavailable(serve_env, role):
    server = serve_env.build_openai_server(AggConfig())
    message = f"Error in kv cache transfer for {role} requests"
    server.generator.script(ScriptedOutput(raise_after_steps=RequestError(message)))
    client = serve_env.client(server.app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["code"] == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["type"] == "ServiceUnavailableError"
    assert response.json()["message"] == message

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
"""The typed errors of the serving core (``tensorrt_llm/serve/_base/errors.py``).

``error_body`` is the one place that builds an OpenAI error body, so a
``ServingError`` carried out of the chat core and an error response rendered
by ``OpenAIServer.create_error_response`` can never differ.
"""

import json
from http import HTTPStatus

import pytest

from tensorrt_llm.executor.utils import context_length_exceeded_message
from tensorrt_llm.serve._base.errors import ServingError, error_body, serving_error
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only

CONTEXT_LENGTH_MESSAGE = context_length_exceeded_message(2048, 4096)


@pytest.mark.parametrize(
    "message, err_type, status",
    [
        ("bad", "BadRequestError", HTTPStatus.BAD_REQUEST),
        ("gone", "NotFoundError", HTTPStatus.NOT_FOUND),
        (CONTEXT_LENGTH_MESSAGE, "BadRequestError", HTTPStatus.BAD_REQUEST),
    ],
    ids=["bad-request", "not-found", "context-length"],
)
def test_error_body_matches_create_error_response(message, err_type, status):
    rendered = OpenAIServer.create_error_response(
        message=message, err_type=err_type, status_code=status
    )
    body = error_body(message, err_type, status)
    assert body.model_dump() == json.loads(rendered.body)
    assert rendered.status_code == status.value

    error = serving_error(message, err_type, status)
    assert error.status == status.value
    assert error.body == body
    assert str(error) == message

    if message is CONTEXT_LENGTH_MESSAGE:
        # The code names the condition; the HTTP status stays a 400.
        assert body.code == "context_length_exceeded"
        assert error.status == 400
    else:
        assert body.code == status.value


def test_serving_error_is_not_a_value_error():
    assert not issubclass(ServingError, ValueError), (
        "ServingError must not be a ValueError: openai_chat's core catches ValueError "
        "and would re-wrap it"
    )

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the over-length-prompt (context_length_exceeded) error.

An over-length prompt is rejected (never truncated) with HTTP 400. The
message uses OpenAI's wording so agent frameworks can pattern-match it to
trigger context compaction, and the error payload carries the
machine-readable ``code: "context_length_exceeded"``.

The message is raised in ``_deduce_max_tokens`` (executor/base_worker.py)
and crosses the executor IPC boundary as a plain string, so the server
recognizes it by text in ``OpenAIServer.create_error_response`` — the single
chokepoint through which the chat, completions, and responses endpoints all
render request errors. These tests cover the message builder, the detector,
and that chokepoint; they are CPU-only.
"""

import json
from http import HTTPStatus

import pytest

from tensorrt_llm.executor.utils import (
    CONTEXT_LENGTH_EXCEEDED_CODE,
    context_length_exceeded_message,
    is_context_length_exceeded_message,
)
from tensorrt_llm.serve.openai_server import OpenAIServer

# The CPU stage collects with `-m cpu_only`; unittest/conftest.py also skips
# collecting any file that does not contain the cpu_only marker.
pytestmark = pytest.mark.cpu_only


def test_message_matches_openai_wording():
    msg = context_length_exceeded_message(max_context_length=69632, num_prompt_tokens=69666)
    assert msg == (
        "This model's maximum context length is 69632 tokens. "
        "However, your messages resulted in 69666 tokens. "
        "Please reduce the length of the messages."
    )


def test_detector_matches_exact_and_wrapped():
    msg = context_length_exceeded_message(2048, 4096)
    assert is_context_length_exceeded_message(msg)
    # RequestError text may arrive wrapped in extra context.
    assert is_context_length_exceeded_message(f"Request failed: {msg} (id=7)")


def test_detector_rejects_other_errors():
    assert not is_context_length_exceeded_message(
        "`default_max_tokens` (-2785) must be greater than 0, "
        "`default_max_tokens` (-2785) = max_seq_len (69632) "
        "- `splited_prompt_len` (72417)"
    )
    assert not is_context_length_exceeded_message("`max_tokens` (0) must be greater than 0")
    assert not is_context_length_exceeded_message("")


def _body(response):
    return json.loads(response.body)


def test_error_response_carries_machine_readable_code():
    msg = context_length_exceeded_message(2048, 4096)
    response = OpenAIServer.create_error_response(msg)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    body = _body(response)
    assert body["code"] == CONTEXT_LENGTH_EXCEEDED_CODE
    assert body["type"] == "BadRequestError"
    assert body["message"] == msg
    assert "2048 tokens" in body["message"]
    assert "4096 tokens" in body["message"]


def test_error_response_default_code_unchanged():
    response = OpenAIServer.create_error_response("`max_tokens` (0) must be greater than 0")
    assert response.status_code == HTTPStatus.BAD_REQUEST
    body = _body(response)
    assert body["code"] == 400
    assert body["type"] == "BadRequestError"


def test_error_response_non_400_status_unchanged():
    response = OpenAIServer.create_error_response(
        "Response with id 'resp_x' not found.",
        err_type="InvalidRequestError",
        status_code=HTTPStatus.NOT_FOUND,
    )
    assert response.status_code == HTTPStatus.NOT_FOUND
    body = _body(response)
    assert body["code"] == 404
    assert body["type"] == "InvalidRequestError"

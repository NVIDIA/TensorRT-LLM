# Copyright (c) 2026, NVIDIA CORPORATION.
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
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from starlette.datastructures import Headers

from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    ConversationParams,
    DisaggregatedParams,
)


def _raw_request(headers: dict[str, str]):
    return SimpleNamespace(headers=Headers(headers=headers))


def test_extract_conversation_id_from_headers():
    cases = [
        ({"X-Session-ID": "session-id"}, "session-id"),
        ({"X-Correlation-ID": "correlation-id"}, "correlation-id"),
        ({"x-session-affinity": "session-affinity"}, "session-affinity"),
        ({"x-multi-turn-session-id": "multi-turn-session-id"}, "multi-turn-session-id"),
        (
            {
                "X-Correlation-ID": "correlation-id",
                "X-Session-ID": "session-id",
                "x-session-affinity": "session-affinity",
                "x-multi-turn-session-id": "multi-turn-session-id",
            },
            "session-id",
        ),
        (
            {
                "x-session-affinity": "session-affinity",
                "x-multi-turn-session-id": "multi-turn-session-id",
            },
            "session-affinity",
        ),
        (
            {
                "X-Session-ID": "",
                "X-Correlation-ID": "correlation-id",
            },
            "correlation-id",
        ),
    ]

    for headers, expected_conversation_id in cases:
        request = CompletionRequest(model="test-model", prompt="hello")

        OpenAIDisaggServer._extract_conversation_id(request, _raw_request(headers))

        assert request.disaggregated_params is None
        assert request.conversation_params.conversation_id == expected_conversation_id


def test_extract_conversation_id_ignores_empty_headers():
    request = CompletionRequest(model="test-model", prompt="hello")

    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request(
            {
                "X-Session-ID": "",
                "X-Correlation-ID": " ",
                "x-session-affinity": "",
                "x-multi-turn-session-id": " ",
            }
        ),
    )

    assert request.disaggregated_params is None
    assert request.conversation_params is None


def test_disaggregated_params_rejects_conversation_id():
    with pytest.raises(ValidationError):
        DisaggregatedParams(
            request_type="context_only",
            conversation_id="body-id",
        )


def test_conversation_params_requires_conversation_id():
    with pytest.raises(ValidationError):
        ConversationParams()


def test_extract_conversation_id_does_not_populate_disaggregated_params():
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        disaggregated_params=DisaggregatedParams(request_type="context_only"),
    )

    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request({"x-multi-turn-session-id": "multi-turn-session-id"}),
    )

    assert not hasattr(request.disaggregated_params, "conversation_id")
    assert request.conversation_params.conversation_id == "multi-turn-session-id"


def test_extract_conversation_id_preserves_body_conversation_params():
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        conversation_params=ConversationParams(conversation_id="body-id"),
        disaggregated_params=DisaggregatedParams(request_type="context_only"),
    )

    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request({"X-Session-ID": "header-id"}),
    )

    assert request.conversation_params.conversation_id == "body-id"

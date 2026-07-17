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
from starlette.datastructures import Headers

from tensorrt_llm.llmapi.disagg_utils import extract_disagg_cfg
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
    assert request.disaggregated_params.conversation_id is None


def test_extract_conversation_id_populates_conversation_params_with_existing_disaggregated_params():
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        disaggregated_params=DisaggregatedParams(request_type="context_only"),
    )

    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request({"x-multi-turn-session-id": "multi-turn-session-id"}),
    )

    assert request.conversation_params.conversation_id == "multi-turn-session-id"
    assert request.disaggregated_params.conversation_id is None


def test_disagg_config_allows_request_chat_template_opt_in():
    config = extract_disagg_cfg(
        context_servers={"num_instances": 0},
        generation_servers={"num_instances": 0},
        allow_request_chat_template=True,
    )

    assert config.allow_request_chat_template is True


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_disagg_config_rejects_non_bool_request_chat_template_opt_in(value):
    with pytest.raises(ValueError, match="allow_request_chat_template must be a boolean"):
        extract_disagg_cfg(
            context_servers={"num_instances": 0},
            generation_servers={"num_instances": 0},
            allow_request_chat_template=value,
        )

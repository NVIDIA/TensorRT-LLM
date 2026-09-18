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
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request
from starlette.datastructures import Headers

from tensorrt_llm.llmapi.disagg_utils import ServerRole, extract_disagg_cfg
from tensorrt_llm.serve import openai_disagg_server
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    ConversationParams,
    DisaggregatedParams,
)

pytestmark = pytest.mark.cpu_only


def _raw_request(headers: dict[str, str]):
    return SimpleNamespace(headers=Headers(headers=headers))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_kwargs", "expected_timeout"),
    [({}, 10), ({"server_keep_alive_timeout": 3600}, 3600)],
)
async def test_server_keep_alive_timeout_is_passed_to_uvicorn(
    monkeypatch, config_kwargs, expected_timeout
):
    config = extract_disagg_cfg(
        context_servers={"num_instances": 0},
        generation_servers={"num_instances": 0},
        **config_kwargs,
    )
    server = object.__new__(OpenAIDisaggServer)
    server._config = config
    server.app = object()

    uvicorn_config = object()
    config_factory = Mock(return_value=uvicorn_config)
    uvicorn_server = SimpleNamespace(serve=AsyncMock())
    server_factory = Mock(return_value=uvicorn_server)
    monkeypatch.setattr(openai_disagg_server.uvicorn, "Config", config_factory)
    monkeypatch.setattr(openai_disagg_server.uvicorn, "Server", server_factory)

    await server(host="localhost", port=8000)

    assert config_factory.call_args.kwargs["timeout_keep_alive"] == expected_timeout
    server_factory.assert_called_once_with(uvicorn_config)
    uvicorn_server.serve.assert_awaited_once_with(sockets=None)


@pytest.mark.asyncio
async def test_http_cluster_storage_request_is_proxied_to_coordinator():
    payload = b'{"key":"worker","value":"ready"}'

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/set",
            "query_string": b"source=worker",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._coordinator = SimpleNamespace(
        proxy_cluster_storage_request=AsyncMock(
            return_value=(b'{"result":true}', 200, "application/json")
        )
    )

    response = await server._proxy_cluster_storage_request(request)

    server._coordinator.proxy_cluster_storage_request.assert_awaited_once_with(
        "POST", "/set", [("source", "worker")], payload, "application/json"
    )
    assert response.status_code == 200
    assert response.body == b'{"result":true}'


def test_create_client_does_not_register_with_server_metrics_collector():
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._coordinator = SimpleNamespace(get_disagg_request_id=AsyncMock(return_value=1))
    server._req_timeout_secs = 30
    server._collect_perf_metrics = True
    server._config = SimpleNamespace(internal_request_auth_key="key")
    server._perf_metrics_collector = SimpleNamespace()

    with patch("tensorrt_llm.serve.openai_disagg_server.OpenAIHttpClient") as mock_client:
        client = server._create_client(SimpleNamespace(), ServerRole.GENERATION, max_retries=2)

    assert client is mock_client.return_value
    mock_client.assert_called_once()


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


# --- sub-agent conversation affinity (routing key, conversation_id NOT rewritten) ---

_PARENT_HEADER = "X-Dynamo-Parent-Session-ID"


def _routing_id(request):
    from tensorrt_llm.serve.conversation_id import get_request_routing_id

    return get_request_routing_id(request)


def _affinity_id(request):
    from tensorrt_llm.serve.conversation_id import get_request_subagent_affinity_id

    return get_request_subagent_affinity_id(request)


def test_subagent_affinity_sets_routing_key_without_rewriting_conversation_id():
    # A sub-agent request keeps its OWN conversation_id (linear history for the
    # worker's per-conversation KV bookkeeping); only the server-private routing
    # key is set to the parent, so the ConversationRouter co-locates it.
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        conversation_params=ConversationParams(conversation_id="own-id"),
    )
    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request({_PARENT_HEADER: "parent-id"}),
        _PARENT_HEADER,
    )
    assert request.conversation_params.conversation_id == "own-id"  # NOT rewritten
    assert _affinity_id(request) == "parent-id"
    assert _routing_id(request) == "parent-id"  # routes to the parent's instance


def test_subagent_affinity_main_agent_has_no_routing_key():
    # A main-agent request lacks the parent header -> no affinity; routes by its
    # own id.
    request = CompletionRequest(model="test-model", prompt="hello")
    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request({"X-Session-ID": "own-id"}),
        _PARENT_HEADER,
    )
    assert request.conversation_params.conversation_id == "own-id"
    assert _affinity_id(request) is None
    assert _routing_id(request) == "own-id"


def test_subagent_affinity_feature_off_ignores_parent_header():
    # No configured header name -> the parent header is inert.
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        conversation_params=ConversationParams(conversation_id="own-id"),
    )
    OpenAIDisaggServer._extract_conversation_id(
        request, _raw_request({_PARENT_HEADER: "parent-id"}), None
    )
    assert request.conversation_params.conversation_id == "own-id"
    assert _affinity_id(request) is None


def test_subagent_affinity_parent_only_synthesizes_own_id():
    # No body id and no session header: synthesize a distinct own id so the
    # worker's bookkeeping / gen fleet see a distinct linear session, while the
    # routing key still pins to the parent.
    request = CompletionRequest(model="test-model", prompt="hello")
    OpenAIDisaggServer._extract_conversation_id(
        request, _raw_request({_PARENT_HEADER: "parent-id"}), _PARENT_HEADER
    )
    assert request.conversation_params.conversation_id.startswith("subagent:")
    assert _affinity_id(request) == "parent-id"
    assert _routing_id(request) == "parent-id"


@pytest.mark.parametrize("header", [None, _PARENT_HEADER])
def test_subagent_affinity_clears_client_supplied_routing_key(header):
    # subagent_affinity_id is server-private: a client cannot enable affinity by
    # putting it in the request body. With the feature OFF (header=None) it ends
    # up None; with a configured header but no parent header present it is also
    # cleared (only the trusted parent header re-sets it).
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        conversation_params=ConversationParams(
            conversation_id="own-id", subagent_affinity_id="attacker-id"
        ),
    )
    OpenAIDisaggServer._extract_conversation_id(
        request, _raw_request({"X-Session-ID": "own-id"}), header
    )
    assert request.conversation_params.conversation_id == "own-id"
    assert _affinity_id(request) is None


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

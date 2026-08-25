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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request
from starlette.datastructures import Headers

from tensorrt_llm.llmapi.disagg_utils import (
    ServerRole,
    extract_disagg_cfg,
    parse_disagg_config_file,
)
from tensorrt_llm.serve import openai_disagg_server
from tensorrt_llm.serve.openai_disagg_server import (
    OpenAIDisaggServer,
    _validate_physical_ownership_preflight,
)
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    ConversationParams,
    DisaggregatedParams,
)

pytestmark = pytest.mark.cpu_only


def _raw_request(headers: dict[str, str]):
    return SimpleNamespace(headers=Headers(headers=headers))


def _qualified_physical_ownership_config():
    worker = {
        "num_instances": 1,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "context_parallel_size": 1,
        "enable_attention_dp": False,
        "cache_transceiver_config": {
            "backend": "NIXL",
            "transceiver_runtime": "PYTHON",
            "kv_cache_bounce_size_mb": 0,
        },
        "kv_cache_config": {"use_kv_cache_manager_v2": False},
    }
    return extract_disagg_cfg(
        backend="pytorch",
        context_servers=worker.copy(),
        generation_servers=worker.copy(),
        schedule_style="context_first",
    )


def _enable_physical_ownership(monkeypatch):
    monkeypatch.setenv("TRTLLM_ENABLE_KV_TRANSFER_PHYSICAL_OWNERSHIP", "1")
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)
    monkeypatch.delenv("TRTLLM_DISAGG_LAYERWISE", raising=False)


@pytest.mark.parametrize(
    ("ownership", "no_retry", "qualified", "error"),
    [
        ("0", "0", False, None),
        ("0", "1", False, None),
        ("1", "0", True, "requires TRTLLM_DISAGG_NO_RETRY=1"),
        ("1", "1", True, None),
    ],
)
def test_physical_ownership_preflight_flag_matrix(
    monkeypatch, ownership, no_retry, qualified, error
):
    monkeypatch.setenv("TRTLLM_ENABLE_KV_TRANSFER_PHYSICAL_OWNERSHIP", ownership)
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", no_retry)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)
    monkeypatch.delenv("TRTLLM_DISAGG_LAYERWISE", raising=False)
    config = (
        _qualified_physical_ownership_config()
        if qualified
        else extract_disagg_cfg(
            context_servers={"num_instances": 0},
            generation_servers={"num_instances": 0},
        )
    )

    if error is None:
        _validate_physical_ownership_preflight(config)
    else:
        with pytest.raises(ValueError, match=error):
            _validate_physical_ownership_preflight(config)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("backend", "tensorrt", "backend"),
        ("tensor_parallel_size", 2, "tensor_parallel_size"),
        ("pipeline_parallel_size", 2, "pipeline_parallel_size"),
        ("context_parallel_size", 2, "context_parallel_size"),
        ("enable_attention_dp", True, "attention_dp"),
        ("dwdp_config", {"dwdp_size": 2}, "dwdp"),
    ],
)
def test_physical_ownership_preflight_rejects_worker_config(monkeypatch, field, value, error):
    _enable_physical_ownership(monkeypatch)
    config = _qualified_physical_ownership_config()
    config.server_configs[0].other_args[field] = value

    with pytest.raises(ValueError, match=error):
        _validate_physical_ownership_preflight(config)


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("cache_transceiver_config", "backend", "UCX", "cache_transceiver_backend"),
        ("cache_transceiver_config", "transceiver_runtime", "CPP", "transceiver_runtime"),
        ("cache_transceiver_config", "kv_cache_bounce_size_mb", 64, "bounce"),
        ("kv_cache_config", "use_kv_cache_manager_v2", True, "use_kv_cache_manager_v2"),
    ],
)
def test_physical_ownership_preflight_rejects_transfer_config(
    monkeypatch, section, field, value, error
):
    _enable_physical_ownership(monkeypatch)
    config = _qualified_physical_ownership_config()
    config.server_configs[0].other_args[section][field] = value

    with pytest.raises(ValueError, match=error):
        _validate_physical_ownership_preflight(config)


@pytest.mark.parametrize(
    ("env_name", "config_field", "value", "error"),
    [
        ("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", None, None, "synchronous_transfer"),
        ("TRTLLM_DISAGG_LAYERWISE", None, None, "layerwise_transfer"),
        (None, "schedule_style", "generation_first", "schedule_style"),
        (None, "conditional_disagg_config", SimpleNamespace(), "conditional_disagg"),
        (None, "disagg_cluster_config", SimpleNamespace(), "autoscaling"),
        (None, "num_workers", 2, "num_workers=2"),
        (None, "disagg_coordinator_url", "http://coordinator:8332", "external_coordinator"),
    ],
)
def test_physical_ownership_preflight_rejects_deployment_mode(
    monkeypatch, env_name, config_field, value, error
):
    _enable_physical_ownership(monkeypatch)
    config = _qualified_physical_ownership_config()
    if env_name is not None:
        monkeypatch.setenv(env_name, "1")
    else:
        setattr(config, config_field, value)

    with pytest.raises(ValueError, match=error):
        _validate_physical_ownership_preflight(config)


def test_physical_ownership_preflight_rejects_non_one_to_one_topology(monkeypatch):
    _enable_physical_ownership(monkeypatch)
    config = _qualified_physical_ownership_config()
    config.server_configs.append(config.server_configs[0])

    with pytest.raises(ValueError, match="worker_topology=CTX2/GEN1"):
        _validate_physical_ownership_preflight(config)


def test_physical_ownership_preflight_runs_before_coordinator_start(monkeypatch):
    monkeypatch.setenv("TRTLLM_ENABLE_KV_TRANSFER_PHYSICAL_OWNERSHIP", "1")
    monkeypatch.delenv("TRTLLM_DISAGG_NO_RETRY", raising=False)
    config = _qualified_physical_ownership_config()

    with patch.object(openai_disagg_server, "DisaggCoordinatorService") as coordinator:
        with pytest.raises(ValueError, match="requires TRTLLM_DISAGG_NO_RETRY=1"):
            OpenAIDisaggServer(config)

    coordinator.assert_not_called()


def test_physical_ownership_rejects_constructor_coordinator_before_client_start(monkeypatch):
    _enable_physical_ownership(monkeypatch)
    config = _qualified_physical_ownership_config()

    with patch.object(openai_disagg_server, "CoordinatorClient") as coordinator:
        with pytest.raises(ValueError, match="external_coordinator"):
            OpenAIDisaggServer(config, coordinator_url="http://coordinator:8332")

    coordinator.assert_not_called()


def test_phase1_static_canary_config_passes_preflight(monkeypatch):
    _enable_physical_ownership(monkeypatch)
    config_path = (
        Path(__file__).parents[2]
        / "integration"
        / "defs"
        / "disaggregated"
        / "test_configs"
        / "disagg_config_physical_ownership_phase1.yaml"
    )

    config = parse_disagg_config_file(str(config_path))

    _validate_physical_ownership_preflight(config)


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

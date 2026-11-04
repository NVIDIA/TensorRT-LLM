# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The multimodal-encoder ``POST /v1/chat/completions`` through ``MMEncoderServing``.

Pins what the route does today on an MM_ENCODER server built by the real
composition root: the embedding handle in the response, the injected
``RenderResources`` (the server chat template reaches the prompt), the
request-template gate, the msgpack route class and the error shape.
"""

import asyncio

import pytest
from fastapi import FastAPI
from test_common.serve.app_factory import AggConfig
from test_common.serve.fakes import FakeLlmArgs, FakeMultimodalEncoder, ScriptedOutput

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.api.openai.mm_encoder import api_router
from tensorrt_llm.serve.api.openai.mm_encoder.serving import MMEncoderServing
from tensorrt_llm.serve.openai_client import MSGPACK_HEADERS, _msgpack_encoder
from tensorrt_llm.serve.postprocess_handlers import select_postproc_strategy
from tensorrt_llm.serve.render import RenderResources

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

REQUIRED = (
    "render_resources",
    "multimodal_server_config",
    "allow_request_chat_template",
    "model",
    "generator",
    "tokenizer",
    "model_config",
    "collect_perf_metrics",
    "metrics_collector",
    "adjusted_steady_clock",
    "iteration_stats_wakeup_event",
    "iteration_stats_collector_task_getter",
    "internal_disagg_auth_key",
    "server_role",
    "per_request_spec_decode_stats",
    "spec_decode_num_spec_tokens",
    "input_proc_executor",
    "postproc",
)
CHAT_BODY = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
# Distinct from the fake tokenizer's own template, so the rendered prompt tells
# which template produced it.
SERVER_TEMPLATE = "{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}{% endfor %}"
REQUEST_TEMPLATE = "{% for m in messages %}<{{ m['content'] }}>{% endfor %}"
HANDLE = {"tensor_size": [3, 16], "shm_name": "fake-handle"}


def _dependencies(tmp_path):
    generator = FakeMultimodalEncoder(args=FakeLlmArgs(model=str(tmp_path)))
    return {
        "render_resources": RenderResources(tokenizer=generator.tokenizer),
        "multimodal_server_config": None,
        "allow_request_chat_template": False,
        "model": "fake-model",
        "generator": generator,
        "tokenizer": generator.tokenizer,
        "model_config": None,
        "collect_perf_metrics": False,
        "metrics_collector": None,
        "adjusted_steady_clock": AdjustedSteadyClock(),
        "iteration_stats_wakeup_event": asyncio.Event(),
        "iteration_stats_collector_task_getter": lambda: None,
        "internal_disagg_auth_key": None,
        "server_role": None,
        "per_request_spec_decode_stats": False,
        "spec_decode_num_spec_tokens": None,
        "input_proc_executor": None,
        "postproc": select_postproc_strategy(postproc_worker_enabled=False),
    }


def _serving(server) -> MMEncoderServing:
    return getattr(server.app.state, api_router.APP_STATE_KEY)


def _mm_encoder_server(serve_env, **config):
    server = serve_env.build_openai_server(AggConfig(role=ServerRole.MM_ENCODER, **config))
    return server, serve_env.client(server.app)


def _script_handles(server, *handles):
    server.generator.script(
        ScriptedOutput(
            disaggregated_params=DisaggregatedParams(multimodal_embedding_handles=list(handles))
        )
    )


def test_constructor_requires_each_dependency(tmp_path):
    dependencies = _dependencies(tmp_path)
    # The control: with every dependency present, construction succeeds.
    MMEncoderServing(**dependencies)
    for name in REQUIRED:
        partial = {key: value for key, value in dependencies.items() if key != name}
        try:
            MMEncoderServing(**partial)
        except TypeError:
            continue
        pytest.fail(
            f"MMEncoderServing must require {name!r}: serving dependencies are keyword-only "
            "constructor arguments without defaults (serve/AGENTS.md)"
        )


def test_render_resources_are_the_servers(serve_env):
    server, _ = _mm_encoder_server(serve_env, chat_template=SERVER_TEMPLATE)
    serving = _serving(server)
    assert serving._render_resources is server._render_resources, (
        "MMEncoderServing must use the RenderResources the composition root built, never "
        "RenderResources.from_server(serving), which silently drops the server chat template"
    )
    assert server.chat_template == SERVER_TEMPLATE
    assert server._render_resources.default_chat_template == server.chat_template


def test_mm_encoder_route_returns_embedding_handle(serve_env):
    server, client = _mm_encoder_server(serve_env, chat_template=SERVER_TEMPLATE)
    _script_handles(server, HANDLE)
    response = client.post("/v1/chat/completions", json=CHAT_BODY)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "fake-model"
    choice = body["choices"][0]
    assert choice["mm_embedding_handle"] == HANDLE
    assert choice["finish_reason"] == "length"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "dummy"
    assert body["usage"]["prompt_tokens"] == 3  # tensor_size[0] of the one handle
    assert body["usage"]["completion_tokens"] == 1
    assert body["usage"]["total_tokens"] == 4
    # The prompt was rendered with the server template the injected resources carry.
    assert server.generator.generate_calls[0].inputs == {"prompt": "[user] hi"}


def test_mm_encoder_route_decodes_msgpack(serve_env):
    server, client = _mm_encoder_server(serve_env)
    _script_handles(server, HANDLE)
    response = client.post(
        "/v1/chat/completions", content=_msgpack_encoder.encode(CHAT_BODY), headers=MSGPACK_HEADERS
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["mm_embedding_handle"] == HANDLE


def test_missing_handle_is_an_error_response(serve_env):
    server, client = _mm_encoder_server(serve_env)
    # The default scripted output carries no disaggregated params, so no handle.
    response = client.post("/v1/chat/completions", json=CHAT_BODY)
    # Today's shape, kept as it is: the handler builds a 500 for the missing
    # handle, then calls model_dump() on that response, and the resulting
    # AttributeError is answered as a 400 BadRequestError.
    assert response.status_code == 400
    body = response.json()
    assert body["object"] == "error"
    assert body["type"] == "BadRequestError"
    assert len(server.generator.generate_calls) == 1


def test_request_chat_template_rejected_unless_allowed(serve_env):
    server, client = _mm_encoder_server(serve_env)
    response = client.post(
        "/v1/chat/completions", json={**CHAT_BODY, "chat_template": REQUEST_TEMPLATE}
    )
    assert response.status_code == 400
    assert "chat_template cannot be supplied per request" in response.json()["message"]
    assert server.generator.generate_calls == []


def test_request_chat_template_renders_when_allowed(serve_env):
    server, client = _mm_encoder_server(
        serve_env, chat_template=SERVER_TEMPLATE, allow_request_chat_template=True
    )
    _script_handles(server, HANDLE)
    response = client.post(
        "/v1/chat/completions", json={**CHAT_BODY, "chat_template": REQUEST_TEMPLATE}
    )
    assert response.status_code == 200, response.text
    assert server.generator.generate_calls[0].inputs == {"prompt": "<hi>"}


def test_attach_router_requires_the_serving_object():
    with pytest.raises(RuntimeError, match=api_router.APP_STATE_KEY):
        api_router.attach_router(FastAPI())

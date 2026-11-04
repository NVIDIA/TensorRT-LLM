# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/chat/completions`` through ``ChatServing`` and its router.

Pins that the route does on the LLM roles what the server did (one chat, a
stream ending in ``[DONE]``, msgpack bodies), that the endpoint keeps the name
the server gave it on harmony and non-harmony servers, that the serving object
is built from the composition root's own objects (render resources, clock,
stats event, input-processor pool, metrics collector, post-processing
strategy), that ``/health_generate`` and the Anthropic Messages serving run on
it, and that the internal disaggregated auth key and the server role reach it.
"""

import asyncio
import json
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from test_common.serve.app_factory import AggConfig, isolated_serve_env
from test_common.serve.fakes import FakeLLM, FakeLlmArgs
from test_common.serve.routes import effective_routes

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve._base.errors import ServingError, error_body
from tensorrt_llm.serve.api.openai.chat import api_router
from tensorrt_llm.serve.api.openai.chat.serving import ChatServing
from tensorrt_llm.serve.openai_client import MSGPACK_HEADERS, _msgpack_encoder
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.postprocess_handlers import select_postproc_strategy
from tensorrt_llm.serve.render import RenderResources

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

# Constructor argument -> the attribute the moved chat handlers read it as.
CHAT_ATTRIBUTES = {
    "tool_parser": "tool_parser",
    "tool_call_id_type": "tool_call_id_type",
    "multimodal_server_config": "multimodal_server_config",
    "allow_request_chat_template": "allow_request_chat_template",
    "use_harmony": "use_harmony",
    "harmony_adapter": "harmony_adapter",
    "render_resources": "_render_resources",
}
BASE_REQUIRED = (
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
REQUIRED = tuple(CHAT_ATTRIBUTES) + BASE_REQUIRED
BODY = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
# Protected disaggregated fields: a worker accepts them only from the frontend,
# which signs them with the internal auth key.
PROTECTED_BODY = {
    **BODY,
    "disaggregated_params": {
        "request_type": "generation_only",
        "ctx_request_id": 1,
        "disagg_request_id": 2,
        "ctx_info_endpoint": "tcp://10.0.0.1:5000",
    },
}
# The composition root's own objects a serving object must share, not copy.
SHARED = (
    "_adjusted_steady_clock",
    "_iteration_stats_wakeup_event",
    "_input_proc_executor",
    "metrics_collector",
    "_postproc",
)
# Distinct from the fake tokenizer's own template, so the rendered prompt tells
# which template produced it.
SERVER_TEMPLATE = "{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}{% endfor %}"
ROUTE_MESSAGE = (
    "/v1/chat/completions must route to the method the server's use_harmony selects: "
    "openai_chat, or chat_harmony on a gpt-oss server"
)


def _dependencies(tmp_path):
    generator = FakeLLM(args=FakeLlmArgs(model=str(tmp_path)))
    return {
        "tool_parser": None,
        "tool_call_id_type": "random",
        "multimodal_server_config": None,
        "allow_request_chat_template": False,
        "use_harmony": False,
        "harmony_adapter": None,
        "render_resources": RenderResources(tokenizer=generator.tokenizer),
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


def _serving(server) -> ChatServing:
    return getattr(server.app.state, api_router.APP_STATE_KEY)


def _llm_server(serve_env, **config):
    server = serve_env.build_openai_server(AggConfig(**config))
    return server, serve_env.client(server.app)


def _sse_frames(text):
    return [frame[len("data: ") :] for frame in text.split("\n\n") if frame.startswith("data: ")]


def _chat_route_names(app):
    return [route.name for path, route in effective_routes(app) if path == "/v1/chat/completions"]


def test_constructor_requires_each_dependency(tmp_path):
    dependencies = _dependencies(tmp_path)
    # The control: with every dependency present, construction succeeds.
    ChatServing(**dependencies)
    for name in REQUIRED:
        partial = {key: value for key, value in dependencies.items() if key != name}
        try:
            ChatServing(**partial)
        except TypeError:
            continue
        pytest.fail(
            f"ChatServing must require {name!r}: serving dependencies are keyword-only "
            "constructor arguments without defaults (serve/AGENTS.md)"
        )
    with pytest.raises(TypeError):
        ChatServing(*dependencies.values())  # keyword-only: no positional construction


def test_constructor_stores_each_dependency_where_the_handlers_read_it(tmp_path):
    dependencies = _dependencies(tmp_path)
    sentinels = {name: object() for name in CHAT_ATTRIBUTES}
    serving = ChatServing(**{**dependencies, **sentinels})
    for name, attribute in CHAT_ATTRIBUTES.items():
        assert getattr(serving, attribute) is sentinels[name], (
            f"ChatServing must store {name!r} as self.{attribute}, the name the moved chat "
            "handlers read"
        )


def test_chat_serving_is_the_roots_and_uses_its_render_resources(serve_env):
    server, _ = _llm_server(serve_env, chat_template=SERVER_TEMPLATE)
    serving = _serving(server)
    assert serving is server._chat_serving
    assert serving._render_resources is server._render_resources, (
        "ChatServing must use the RenderResources the composition root built, never "
        "RenderResources.from_server(serving), which silently drops the server chat template"
    )
    assert (
        serving._render_resources.default_chat_template == server.chat_template == SERVER_TEMPLATE
    )
    for name in ("tool_parser", "tool_call_id_type", "multimodal_server_config"):
        assert getattr(serving, name) == getattr(server, name)
    assert serving.allow_request_chat_template is server.allow_request_chat_template
    assert serving.use_harmony is server.use_harmony is False


def test_chat_serving_shares_the_roots_objects(serve_env):
    server, _ = _llm_server(serve_env, return_perf_metrics=True, role=ServerRole.CONTEXT)
    serving = _serving(server)
    for name in SHARED:
        assert getattr(serving, name) is getattr(server, name), (
            f"ChatServing must share the composition root's {name}, not a new one; pass the "
            "root's object through _generate_serving_kwargs()"
        )
    assert server.metrics_collector is not None  # the identity above compared real objects
    assert serving.server_role is server.server_role is ServerRole.CONTEXT, (
        "ChatServing must receive the server role: _get_scheduling_params reads it to decide "
        "whether to honor a signed subagent affinity"
    )
    assert serving.generator is server.generator


@pytest.mark.parametrize(
    "role",
    [None, ServerRole.CONTEXT, ServerRole.GENERATION],
    ids=["llm", "context", "generation"],
)
def test_chat_route_non_streaming(role, serve_env):
    server, client = _llm_server(serve_env, role=role)
    response = client.post("/v1/chat/completions", json=BODY)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "fake-model"
    assert [choice["index"] for choice in body["choices"]] == [0]
    message = body["choices"][0]["message"]
    assert (message["role"], message["content"], message["tool_calls"]) == ("assistant", "ok", [])
    assert body["choices"][0]["finish_reason"] == "stop"
    [call] = server.generator.generate_calls
    assert call.kwargs["streaming"] is False
    assert call.kwargs["kv_hints"] is None


def test_chat_route_streaming_ends_with_done(serve_env):
    server, client = _llm_server(serve_env)
    response = client.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _sse_frames(response.text)
    assert frames[-1] == "[DONE]"
    assert frames.count("[DONE]") == 1
    chunks = [json.loads(frame) for frame in frames[:-1]]
    assert chunks, "a streamed chat answers at least one chunk before [DONE]"
    assert {chunk["object"] for chunk in chunks} == {"chat.completion.chunk"}
    assert len({chunk["id"] for chunk in chunks}) == 1
    [call] = server.generator.generate_calls
    assert call.kwargs["streaming"] is True


def test_chat_route_decodes_msgpack(serve_env):
    _, client = _llm_server(serve_env)
    response = client.post(
        "/v1/chat/completions", content=_msgpack_encoder.encode(BODY), headers=MSGPACK_HEADERS
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "ok"


def test_endpoint_follows_use_harmony(tmp_path, monkeypatch):
    # One server per environment: both register collectors in the default
    # Prometheus registry, which the environment restores on exit. Both live
    # at once, so each app must keep its own endpoint although the routers
    # are module-level objects.
    with (
        isolated_serve_env(tmp_path / "text") as text_env,
        isolated_serve_env(tmp_path / "harmony") as harmony_env,
    ):
        text = text_env.build_openai_server(AggConfig())
        harmony = harmony_env.build_openai_server(AggConfig(model_type="gpt_oss"))
        assert text.use_harmony is False and harmony.use_harmony is True  # the fixture's premise
        assert _chat_route_names(text.app) == ["openai_chat"], ROUTE_MESSAGE
        assert _chat_route_names(harmony.app) == ["chat_harmony"], ROUTE_MESSAGE

        spies = {}
        for server, method in ((text, "openai_chat"), (harmony, "chat_harmony")):
            spies[method] = AsyncMock(return_value=Response(status_code=204))
            # The router looks the method up on the serving object per request.
            monkeypatch.setattr(server._chat_serving, method, spies[method])
        assert text_env.client(text.app).post("/v1/chat/completions", json=BODY).status_code == 204
        assert (
            harmony_env.client(harmony.app).post("/v1/chat/completions", json=BODY).status_code
            == 204
        )
        for method, spy in spies.items():
            spy.assert_awaited_once()
            assert isinstance(spy.await_args.args[0], ChatCompletionRequest), ROUTE_MESSAGE


def test_health_generate_ok(serve_env):
    server, client = _llm_server(serve_env)
    response = client.get("/health_generate")
    assert response.status_code == 200, response.text
    assert response.text == "Generation health check OK"
    # The probe is one chat through the serving object: one engine call.
    [call] = server.generator.generate_calls
    assert call.kwargs["streaming"] is False


def test_health_generate_reports_a_failed_chat_as_500(serve_env, monkeypatch):
    server, client = _llm_server(serve_env)
    monkeypatch.setattr(
        server._chat_serving,
        "openai_chat",
        AsyncMock(return_value=JSONResponse(status_code=400, content={"message": "rejected"})),
    )
    response = client.get("/health_generate")
    assert response.status_code == 500
    assert response.text == "Generation health check failed"


def test_health_generate_uses_beam_search_arguments(serve_env, monkeypatch):
    server, client = _llm_server(serve_env)
    server.generator.args.max_beam_width = 4
    spy = AsyncMock(return_value=Response(status_code=200))
    # Patched on the serving instance, which is where the root looks it up.
    monkeypatch.setattr(server._chat_serving, "openai_chat", spy)
    assert client.get("/health_generate").status_code == 200
    spy.assert_awaited_once()
    request = spy.await_args.args[0]
    assert isinstance(request, ChatCompletionRequest)
    assert (request.use_beam_search, request.best_of, request.n) == (True, 4, 1)
    assert request.max_completion_tokens == 1 and request.stream is False


def test_anthropic_backend_is_the_chat_serving(serve_env):
    # Distinct names for the server and the request: the stream announces the
    # served name, as the aggregated route always did.
    server, _ = _llm_server(serve_env, model_name="served-name")
    serving = server._chat_serving
    assert server.app.state.anthropic_serving_messages._backend is serving, (
        "the Anthropic Messages serving must run on the ChatServing the composition root built"
    )
    request = ChatCompletionRequest(**{**BODY, "model": "requested-name"})
    assert serving.stream_model(request) == "served-name"
    # error_message: the rejected chat route's body message, else the whole body.
    worded = error_body("too long", "BadRequestError", HTTPStatus.BAD_REQUEST)
    assert serving.error_message(ServingError(400, worded)) == "too long"
    blank = error_body("", "BadRequestError", HTTPStatus.BAD_REQUEST)
    assert serving.error_message(ServingError(400, blank)) == json.dumps(blank.model_dump())


def test_internal_disagg_auth_key_is_wired(serve_env):
    server, client = _llm_server(
        serve_env, internal_disagg_auth_key="secret", cache_transceiver_backend="UCX"
    )
    response = client.post("/v1/chat/completions", json=PROTECTED_BODY)
    assert response.status_code == 400 and "Invalid internal" in response.json()["message"], (
        "ChatServing did not receive the internal disagg auth key; protected fields would be "
        "accepted unauthenticated; pass internal_disagg_auth_key through "
        f"_generate_serving_kwargs(): {response.status_code} {response.text}"
    )
    assert server.generator.generate_calls == []


def test_without_an_auth_key_protected_fields_are_accepted(serve_env):
    # The control for the test above: today a worker without a key accepts
    # the protected fields (with a warning), so only the wired key rejects.
    # The request then reaches the engine; the fake engine's answer carries no
    # disaggregated params, which the response builder rejects afterwards, so
    # the control is the engine call, not the final status.
    server, client = _llm_server(serve_env, cache_transceiver_backend="UCX")
    response = client.post("/v1/chat/completions", json=PROTECTED_BODY)
    assert "Invalid internal" not in response.text, response.text
    assert len(server.generator.generate_calls) == 1


def test_chat_is_not_served_on_the_embedding_role(serve_env):
    server, client = _llm_server(serve_env, role=ServerRole.EMBEDDING)
    assert client.post("/v1/chat/completions", json=BODY).status_code == 404
    assert not hasattr(server.app.state, api_router.APP_STATE_KEY)


def test_the_encoder_role_serves_chat_completions_itself(serve_env):
    # The multimodal encoder answers /v1/chat/completions with its own
    # handler; no ChatServing is built for it and the chat router stays off.
    server, _ = _llm_server(serve_env, role=ServerRole.MM_ENCODER)
    assert not hasattr(server.app.state, api_router.APP_STATE_KEY)
    assert not hasattr(server, "_chat_serving")
    names = _chat_route_names(server.app)
    assert len(names) == 1 and names[0] not in ("openai_chat", "chat_harmony"), names


def test_attach_router_requires_the_serving_object():
    with pytest.raises(RuntimeError, match=api_router.APP_STATE_KEY):
        api_router.attach_router(FastAPI())


def test_attach_router_picks_the_endpoint_from_use_harmony():
    for use_harmony, name in ((False, "openai_chat"), (True, "chat_harmony")):
        app = FastAPI()
        setattr(app.state, api_router.APP_STATE_KEY, SimpleNamespace(use_harmony=use_harmony))
        api_router.attach_router(app)
        assert _chat_route_names(app) == [name], ROUTE_MESSAGE

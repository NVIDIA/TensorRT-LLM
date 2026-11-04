# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/completions`` through ``CompletionServing`` and its router.

Pins what the route does today on the LLM roles built by the real composition
root (one prompt, several streamed prompts, msgpack bodies), that the serving
object shares the root's objects instead of building its own, that the
internal disaggregated auth key reaches it, and that a finished request wakes
the root's iteration-stats collector.
"""

import asyncio
import json

import pytest
from fastapi import FastAPI
from test_common.serve.app_factory import AggConfig
from test_common.serve.fakes import FakeLLM, FakeLlmArgs

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.api.openai.completions import api_router
from tensorrt_llm.serve.api.openai.completions.serving import CompletionServing
from tensorrt_llm.serve.openai_client import MSGPACK_HEADERS, _msgpack_encoder
from tensorrt_llm.serve.postprocess_handlers import select_postproc_strategy

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

REQUIRED = (
    "model",
    "generator",
    "tokenizer",
    "model_config",
    "collect_perf_metrics",
    "metrics_collector",
    "adjusted_steady_clock",
    "iteration_stats_wakeup_event",
    "iteration_stats_collector_task",
    "internal_disagg_auth_key",
    "per_request_spec_decode_stats",
    "spec_decode_num_spec_tokens",
    "input_proc_executor",
    "postproc",
)
BODY = {"model": "fake-model", "prompt": "hello", "max_tokens": 4}
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


def _dependencies(tmp_path):
    generator = FakeLLM(args=FakeLlmArgs(model=str(tmp_path)))
    return {
        "model": "fake-model",
        "generator": generator,
        "tokenizer": generator.tokenizer,
        "model_config": None,
        "collect_perf_metrics": False,
        "metrics_collector": None,
        "adjusted_steady_clock": AdjustedSteadyClock(),
        "iteration_stats_wakeup_event": asyncio.Event(),
        "iteration_stats_collector_task": lambda: None,
        "internal_disagg_auth_key": None,
        "per_request_spec_decode_stats": False,
        "spec_decode_num_spec_tokens": None,
        "input_proc_executor": None,
        "postproc": select_postproc_strategy(postproc_worker_enabled=False),
    }


def _serving(server) -> CompletionServing:
    return getattr(server.app.state, api_router.APP_STATE_KEY)


def _llm_server(serve_env, **config):
    server = serve_env.build_openai_server(AggConfig(**config))
    return server, serve_env.client(server.app)


def _sse_frames(text):
    return [frame[len("data: ") :] for frame in text.split("\n\n") if frame.startswith("data: ")]


def test_constructor_requires_each_dependency(tmp_path):
    dependencies = _dependencies(tmp_path)
    # The control: with every dependency present, construction succeeds.
    CompletionServing(**dependencies)
    for name in REQUIRED:
        partial = {key: value for key, value in dependencies.items() if key != name}
        try:
            CompletionServing(**partial)
        except TypeError:
            continue
        pytest.fail(
            f"CompletionServing must require {name!r}: serving dependencies are keyword-only "
            "constructor arguments without defaults (serve/AGENTS.md)"
        )
    with pytest.raises(TypeError):
        CompletionServing(*dependencies.values())  # keyword-only: no positional construction


def test_completion_serving_shares_the_roots_objects(serve_env):
    server, _ = _llm_server(serve_env, return_perf_metrics=True)
    serving = _serving(server)
    for name in SHARED:
        assert getattr(serving, name) is getattr(server, name), (
            f"CompletionServing must share the composition root's {name}, not a new one; pass "
            "the root's object through _generate_serving_kwargs()"
        )
    assert server.metrics_collector is not None  # the identity above compared real objects
    # The collector task is created by the lifespan, after construction, and
    # bound on the root: the serving object reads the root's current binding.
    assert serving._iteration_stats_collector_task is server._iteration_stats_collector_task
    sentinel = object()
    server._iteration_stats_collector_task = sentinel
    try:
        assert serving._iteration_stats_collector_task is sentinel, (
            "CompletionServing must read the composition root's current iteration-stats "
            "collector task, not the value bound when it was constructed"
        )
    finally:
        server._iteration_stats_collector_task = None


@pytest.mark.parametrize(
    "role",
    [None, ServerRole.CONTEXT, ServerRole.GENERATION],
    ids=["llm", "context", "generation"],
)
def test_completions_route_single_prompt(role, serve_env):
    server, client = _llm_server(serve_env, role=role)
    response = client.post("/v1/completions", json=BODY)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "fake-model"
    assert len(body["choices"]) == 1
    assert body["choices"][0] == {
        "index": 0,
        "text": "ok",
        "token_ids": None,
        "logprobs": None,
        "context_logits": None,
        "finish_reason": "stop",
        "stop_reason": None,
        "disaggregated_params": None,
        "avg_decoded_tokens_per_iter": None,
    }
    assert body["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    [call] = server.generator.generate_calls
    assert call.inputs == {"prompt_token_ids": [6]}  # "hello" through the real input processor
    assert call.kwargs["streaming"] is False
    assert call.kwargs["kv_hints"] is None


def test_completions_route_streams_several_prompts(serve_env):
    server, client = _llm_server(serve_env)
    response = client.post(
        "/v1/completions", json={**BODY, "prompt": ["hello", "hi"], "stream": True}
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _sse_frames(response.text)
    assert frames[-1] == "[DONE]"
    assert frames.count("[DONE]") == 1
    chunks = [json.loads(frame) for frame in frames[:-1]]
    assert len(chunks) == 2  # one scripted step per prompt
    assert {chunk["choices"][0]["index"] for chunk in chunks} == {0, 1}
    assert len({chunk["id"] for chunk in chunks}) == 1  # one stream id for both prompts
    assert len({chunk["created"] for chunk in chunks}) == 1
    assert [call.kwargs["streaming"] for call in server.generator.generate_calls] == [True, True]


def test_completions_route_decodes_msgpack(serve_env):
    _, client = _llm_server(serve_env)
    response = client.post(
        "/v1/completions", content=_msgpack_encoder.encode(BODY), headers=MSGPACK_HEADERS
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["text"] == "ok"


def test_internal_disagg_auth_key_is_wired(serve_env):
    server, client = _llm_server(
        serve_env, internal_disagg_auth_key="secret", cache_transceiver_backend="UCX"
    )
    response = client.post("/v1/completions", json=PROTECTED_BODY)
    assert response.status_code == 400 and "Invalid internal" in response.json()["message"], (
        "CompletionServing did not receive the internal disagg auth key; protected fields "
        "would be accepted unauthenticated; pass internal_disagg_auth_key through "
        f"_generate_serving_kwargs(): {response.status_code} {response.text}"
    )
    assert server.generator.generate_calls == []


def test_without_an_auth_key_protected_fields_are_accepted(serve_env):
    # The control for the test above: today a worker without a key accepts
    # the protected fields (with a warning), so only the wired key rejects.
    server, client = _llm_server(serve_env, cache_transceiver_backend="UCX")
    response = client.post("/v1/completions", json=PROTECTED_BODY)
    assert response.status_code == 200, response.text
    assert len(server.generator.generate_calls) == 1


def test_finished_request_wakes_the_stats_collector(serve_env):
    server, client = _llm_server(serve_env, return_perf_metrics=True)
    assert server.metrics_collector is not None
    # Control: with no collector task running, a finished request wakes nothing.
    assert client.post("/v1/completions", json=BODY).status_code == 200
    assert not server._iteration_stats_wakeup_event.is_set()
    # The lifespan binds the task on the root once the collector runs.
    server._iteration_stats_collector_task = object()
    try:
        assert client.post("/v1/completions", json=BODY).status_code == 200
        assert server._iteration_stats_wakeup_event.is_set(), (
            "a finished completion must set the composition root's iteration-stats wake-up "
            "event; CompletionServing must share the root's event and read its collector task"
        )
    finally:
        server._iteration_stats_collector_task = None


@pytest.mark.parametrize(
    "role", [ServerRole.MM_ENCODER, ServerRole.EMBEDDING], ids=["mm_encoder", "embedding"]
)
def test_completions_is_not_served_on_other_roles(role, serve_env):
    _, client = _llm_server(serve_env, role=role)
    assert client.post("/v1/completions", json=BODY).status_code == 404


def test_attach_router_requires_the_serving_object():
    with pytest.raises(RuntimeError, match=api_router.APP_STATE_KEY):
        api_router.attach_router(FastAPI())

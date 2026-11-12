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
"""Self-tests of the serve test seam: the helpers other serve tests build on."""

import asyncio
import json
import os
import threading
from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI
from prometheus_client import REGISTRY
from test_common.serve.app_factory import AggConfig, isolated_serve_env
from test_common.serve.cpu_tests import select_entries
from test_common.serve.fakes import (
    FakeLLM,
    FakeLlmArgs,
    ScriptedOutput,
    ScriptedStep,
    UserProbeMiddleware,
)
from test_common.serve.middleware import middleware_snapshot
from test_common.serve.routes import _records_conflict, compare, effective_routes, snapshot_routes

import tensorrt_llm._utils
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.usage import usage_lib

# threadleak is off: the scripted-generation tests create TestClients and run
# requests inside the test body, so the client's portal thread and the fake
# server's worker-pool threads are still alive when the leak check runs; the
# serve_env fixture closes them in its teardown, which runs after that check
# (precedent: tests/unittest/llmapi/apps/test_serve_lazy_import.py).
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

REPO_ROOT = Path(__file__).resolve().parents[3]
_MISSING = object()


# ---------------------------------------------------------------------------
# effective_routes
# ---------------------------------------------------------------------------


def test_effective_routes_flattens_included_routers():
    app = FastAPI()
    app.add_api_route("/direct", lambda: None, methods=["GET"])
    router = APIRouter()
    router.add_api_route("/from_router", lambda: None, methods=["POST"])
    app.include_router(router)
    prefixed = APIRouter()
    prefixed.add_api_route("/suffix", lambda: None, methods=["GET"])
    app.include_router(prefixed, prefix="/pre")
    paths = [path for path, _ in effective_routes(app)]
    assert "/direct" in paths
    assert "/from_router" in paths, (
        "effective_routes must walk into included routers; FastAPI 0.139 keeps them as "
        "objects without a path attribute (duck-typed on original_router and "
        "include_context in test_common/serve/routes.py)."
    )
    assert "/pre/suffix" in paths, "an include_router prefix must be applied to the leaf paths"


def test_effective_routes_matches_app_routes_without_includes():
    app = FastAPI()
    app.add_api_route("/only", lambda: None, methods=["GET"])
    assert [route for _, route in effective_routes(app)] == list(app.routes)


# ---------------------------------------------------------------------------
# compare and the conflict rule
# ---------------------------------------------------------------------------


def test_compare_reports_each_difference_kind():
    expected = [
        "GET /a _MsgspecRoute",
        "GET /metrics _MsgspecRoute #get_iteration_stats",
        "GET /metrics _MsgspecRoute #metrics",
        "POST /v1/chat/completions _MsgspecRoute",
    ]
    actual = [
        "GET /metrics _MsgspecRoute #metrics",
        "GET /metrics _MsgspecRoute #get_iteration_stats",
        "POST /v1/chat/completions APIRoute",
        "POST /new _MsgspecRoute",
    ]
    differences = compare(expected, actual)
    assert any(d.startswith("added:") and "/new" in d for d in differences)
    assert any(d.startswith("removed:") and "GET /a" in d for d in differences)
    assert any(
        "route class changed" in d
        and "_MsgspecRoute -> APIRoute" in d
        and "(a plain APIRouter breaks msgpack bodies from the disaggregated frontend)" in d
        for d in differences
    )
    assert any("order changed" in d and "#get_iteration_stats" in d for d in differences)


def test_compare_ignores_order_of_non_conflicting_routes():
    expected = ["GET /a APIRoute", "GET /b APIRoute"]
    assert compare(expected, ["GET /b APIRoute", "GET /a APIRoute"]) == []
    assert compare(expected, list(expected)) == []


# The first six rows use records of the real apps, the rest are synthetic.
@pytest.mark.parametrize(
    "record_a,record_b,conflict",
    [
        (
            "GET /metrics _MsgspecRoute #get_iteration_stats",
            "GET /metrics _MsgspecRoute #metrics",
            True,
        ),
        (
            "GET /v1/responses/{response_id} _MsgspecRoute",
            "DELETE /v1/responses/{response_id} _MsgspecRoute",
            False,
        ),
        (
            "GET /v1/messages/batches _MsgspecRoute",
            "GET /v1/messages/batches/{batch_id} _MsgspecRoute",
            False,
        ),
        (
            "GET /v1/images/{image_id}/content _MsgspecRoute",
            "GET /v1/videos/{video_id}/content _MsgspecRoute",
            False,
        ),
        ("MOUNT /prometheus/metrics Mount", "GET /metrics _MsgspecRoute", False),
        ("GET,HEAD /docs Route", "GET,HEAD /docs/oauth2-redirect Route", False),
        ("GET /v1/models _MsgspecRoute", "GET /v1/{name} APIRoute", True),
        ("GET,HEAD /openapi.json Route", "GET /openapi.json APIRoute", True),
        ("GET /files/{rest:path} APIRoute", "GET /files/a/b APIRoute", True),
        ("MOUNT /prometheus/metrics Mount", "POST /prometheus/metrics/x APIRoute", True),
    ],
)
def test_conflict_rule(record_a, record_b, conflict):
    assert _records_conflict(record_a, record_b) is conflict


# ---------------------------------------------------------------------------
# middleware_snapshot format
# ---------------------------------------------------------------------------


def test_middleware_snapshot_format():
    from tensorrt_llm.serve.responses_utils import ServerArrivalTimeMiddleware

    app = FastAPI()
    app.add_middleware(ServerArrivalTimeMiddleware, adjusted_clock=None)
    app.add_middleware(UserProbeMiddleware)  # added last, so outermost
    snapshot = middleware_snapshot(app)
    assert snapshot == [
        {"class": "UserProbeMiddleware"},  # not a tensorrt_llm class: name only
        {"class": "ServerArrivalTimeMiddleware", "kwargs": ["adjusted_clock"]},
    ]


# ---------------------------------------------------------------------------
# isolated_serve_env
# ---------------------------------------------------------------------------


def test_isolated_serve_env_restores_global_state(tmp_path):
    environ_before = dict(os.environ)
    media_executor_before = BaseMediaIO._executor
    prometheus_dir_before = getattr(tensorrt_llm._utils, "prometheus_multiproc_dir", _MISSING)
    session_before = usage_lib._SESSION
    collectors_before = set(REGISTRY._collector_to_names.keys())

    with isolated_serve_env(tmp_path) as env:
        env.build_openai_server(AggConfig(return_perf_metrics=True))
        os.environ["A2_RESTORE_SENTINEL"] = "1"
        sink = env.record_terminations()
        assert usage_lib._SESSION is not None and sink == []

    assert dict(os.environ) == environ_before
    assert BaseMediaIO._executor is media_executor_before
    prometheus_dir_after = getattr(tensorrt_llm._utils, "prometheus_multiproc_dir", _MISSING)
    if prometheus_dir_before is _MISSING:
        assert prometheus_dir_after is _MISSING, (
            "isolated_serve_env must delete tensorrt_llm._utils.prometheus_multiproc_dir "
            "when it did not exist before entry"
        )
    else:
        assert prometheus_dir_after is prometheus_dir_before
    assert usage_lib._SESSION is session_before
    assert set(REGISTRY._collector_to_names.keys()) == collectors_before, (
        "isolated_serve_env must unregister every Prometheus collector the build added"
    )


def test_isolated_serve_env_rejects_use_without_with(tmp_path):
    with pytest.raises(AttributeError):
        isolated_serve_env(tmp_path).build_openai_server(AggConfig())


def test_isolated_serve_env_joins_worker_pools(tmp_path: Path) -> None:
    """Environment exit retires test-owned threads before the leak guard runs."""
    with isolated_serve_env(tmp_path) as env:
        server = env.build_openai_server(AggConfig())
        workers = [
            pool.submit(threading.current_thread).result(timeout=1)
            for pool in (
                server._input_proc_executor,
                server._media_load_executor,
                server._metrics_stats_executor,
            )
        ]
        assert all(worker.is_alive() for worker in workers)
    assert all(not worker.is_alive() for worker in workers)


def test_checkpoint_factory_has_read_only_native_boundary(tmp_path: Path) -> None:
    """Build real lifecycle routes without pretending to implement native mutations."""
    with isolated_serve_env(tmp_path) as env:
        server = env.build_openai_server(AggConfig(checkpoint=True))
        status = server.checkpoint_lifecycle.get_lifecycle_status()
        assert status.profile.ranks == [0]
        assert status.readiness.admission_open
        routes = [
            record
            for record in snapshot_routes(server.app)
            if record.split()[1] == "/ready" or record.split()[1].startswith("/checkpoint/")
        ]
        assert routes == [
            "GET /ready _MsgspecRoute",
            "GET /checkpoint/status _MsgspecRoute",
            "POST /checkpoint/prepare_capture _MsgspecRoute",
            "POST /checkpoint/validate_runtime _MsgspecRoute",
            "POST /checkpoint/activate _MsgspecRoute",
            "POST /checkpoint/abort_operation _MsgspecRoute",
        ]
        with pytest.raises(NotImplementedError, match="only supports.*checkpoint profile"):
            server.generator._collective_rpc("checkpoint_step", non_block=True)


# ---------------------------------------------------------------------------
# cpu_tests selection
# ---------------------------------------------------------------------------


def test_cpu_tests_selection():
    kept, _ = select_entries()
    targets = [target for entry in kept for target in entry.targets]
    expected_new_files = [
        "unittest/api_stability/test_serve_routes.py",
        "unittest/serve/test_agents_md.py",
        "unittest/serve/test_layering.py",
        "unittest/serve/test_middleware_order.py",
        "unittest/serve/test_no_object_new.py",
        "unittest/serve/test_serve_test_helpers.py",
        "unittest/serve/test_shim_identity.py",
        "unittest/serve/api/ops/system/test_system_endpoints.py",
        "unittest/serve/api/visual_gen/test_ops_system_visual_gen.py",
        "unittest/serve/api/visual_gen/test_routes_visual_gen.py",
        "unittest/serve/api/visual_gen/test_shim_identity_visual_gen.py",
    ]
    for target in expected_new_files:
        assert target in targets, f"the serve CPU runner must select {target}"
    assert "unittest/llmapi/apps/test_anthropic_routes.py" in targets
    for target in targets:
        file_part = target.split("::")[0]
        text = (REPO_ROOT / "tests" / file_part).read_text(errors="replace")
        assert "pytest.mark.cpu_only" in text, (
            f"{file_part} is selected but carries no cpu_only marker; the CPU stage would "
            "collect nothing from it"
        )


def test_cpu_tests_selection_keeps_node_ids(tmp_path):
    # A node-id entry stays a node id: only that node runs, never the whole file.
    node = "unittest/serve/test_serve_test_helpers.py::test_cpu_tests_selection"
    cpu_list = tmp_path / "l0_cpu.yml"
    cpu_list.write_text(f"l0_cpu:\n- tests:\n  - {node}\n")
    kept, notes = select_entries(cpu_list)
    assert [entry.targets for entry in kept] == [(node,)], kept
    assert not notes, notes


# ---------------------------------------------------------------------------
# Scripted generation through the production handlers
# ---------------------------------------------------------------------------

_CHAT_BODY = {
    "model": "fake-model",
    "messages": [{"role": "user", "content": "hi"}],
    "cache_salt": "salted",
}


def test_scripted_chat_non_streaming(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    client = serve_env.client(server.app)
    response = client.post("/v1/chat/completions", json=_CHAT_BODY)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "ok"  # the default scripted text
    generator = server.generator
    assert len(generator.preprocess_calls) == 1
    call = generator.generate_calls[0]
    assert isinstance(call.sampling_params, SamplingParams)
    assert call.kwargs["streaming"] is False
    assert call.kwargs["cache_salt"] == "salted"


def test_scripted_chat_streaming(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    server.generator.script(
        ScriptedOutput(
            steps=(
                ScriptedStep(text="he", token_ids=(6,)),
                ScriptedStep(text="llo", token_ids=(6,), finish_reason="stop"),
            )
        )
    )
    client = serve_env.client(server.app)
    with client.stream(
        "POST", "/v1/chat/completions", json={**_CHAT_BODY, "stream": True}
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    deltas = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    text = "".join(
        choice["delta"].get("content") or "" for delta in deltas for choice in delta["choices"]
    )
    assert text == "hello"
    assert server.generator.generate_calls[0].kwargs["streaming"] is True


def test_scripted_completion_through_real_input_processor(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    server.generator.script(
        ScriptedOutput(steps=(ScriptedStep(text="done", token_ids=(7,), finish_reason="stop"),))
    )
    client = serve_env.client(server.app)
    response = client.post("/v1/completions", json={"model": "fake-model", "prompt": "hello hi"})
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["text"] == "done"
    call = server.generator.generate_calls[0]
    # A text prompt goes through the real DefaultInputProcessor, which
    # tokenizes with the fake tokenizer: "hello hi" -> ids 6 and 5.
    assert call.inputs == {"prompt_token_ids": [6, 5]}


def test_postprocess_worker_path_answers_through_postprocess_result(serve_env):
    server = serve_env.build_openai_server(AggConfig(num_postprocess_workers=1))
    client = serve_env.client(server.app)
    response = client.post("/v1/chat/completions", json=_CHAT_BODY)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    promise = server.generator.generate_calls[0].result
    assert promise.outputs[0]._postprocess_result is not None
    with client.stream(
        "POST", "/v1/chat/completions", json={**_CHAT_BODY, "stream": True}
    ) as streaming_response:
        lines = [line for line in streaming_response.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    assert any('"content"' in line for line in lines[:-1])


def test_raise_on_call_returns_400_and_records_the_call(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    server.generator.script(ScriptedOutput(raise_on_call=RuntimeError("scripted failure")))
    client = serve_env.client(server.app)
    response = client.post("/v1/chat/completions", json=_CHAT_BODY)
    assert response.status_code == 400
    assert response.json()["message"] == "scripted failure"
    call = server.generator.generate_calls[0]
    assert call.result is None  # recorded before the scripted exception fired
    assert isinstance(call.sampling_params, SamplingParams)


def test_raise_after_steps_ends_a_stream_with_the_scripted_error(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    server.generator.script(
        ScriptedOutput(
            steps=(ScriptedStep(text="partial", token_ids=(5,)),),
            raise_after_steps=RuntimeError("stream exploded"),
        )
    )
    client = serve_env.client(server.app)
    received = []
    with pytest.raises(RuntimeError, match="stream exploded"):
        with client.stream(
            "POST", "/v1/chat/completions", json={**_CHAT_BODY, "stream": True}
        ) as response:
            for line in response.iter_lines():
                received.append(line)
    assert any("partial" in line for line in received), (
        "the first scripted chunk must reach the client before the scripted error"
    )


def test_hold_after_steps_release_and_cancellation():
    async def scenario():
        llm = FakeLLM(args=FakeLlmArgs(model="unused"))
        llm.script(
            ScriptedOutput(
                steps=(ScriptedStep(text="first", token_ids=(5,)),), hold_after_steps=True
            )
        )
        promise = llm.generate_async({"prompt_token_ids": [1, 2]}, SamplingParams())
        first = await promise.__anext__()
        assert first.outputs[0].text == "first"
        assert not promise.finished

        async def consume_rest():
            async for _ in promise:
                pass

        consumer = asyncio.create_task(consume_rest())
        await asyncio.sleep(0)  # the consumer reaches the hold
        promise.release()
        await consumer  # release ends the wait with a normal end of stream
        assert promise.finished

        llm.script(
            ScriptedOutput(steps=(ScriptedStep(text="x", token_ids=(5,)),), hold_after_steps=True)
        )
        held = llm.generate_async({"prompt_token_ids": [1]}, SamplingParams())
        await held.__anext__()
        hanging = asyncio.create_task(held.__anext__())
        await asyncio.sleep(0)
        hanging.cancel()  # a client disconnect cancels the consuming task
        with pytest.raises(asyncio.CancelledError):
            await hanging

    asyncio.run(scenario())

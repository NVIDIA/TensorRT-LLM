# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from tensorrt_llm.bindings.executor import RequestPerfMetrics
from tensorrt_llm.metrics.perf_utils import _input_processing_observer, _time_input_processing
from tensorrt_llm.serve.anthropic_protocol import AnthropicCreateBatchRequest
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, ResponsesRequest
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.perf_metrics import (
    RETURN_METRICS_HEADER,
    SERVER_TIMING_HEADER,
    PerfMetricsMiddleware,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def observer():
    observer = MagicMock()
    token = _input_processing_observer.set(observer)
    yield observer
    _input_processing_observer.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_processor_metrics_separate_queue_and_execution(fails, observer):
    server = object.__new__(OpenAIServer)

    def process():
        if fails:
            raise ValueError("bad input")
        return "processed"

    with ThreadPoolExecutor(max_workers=1) as pool:
        server._input_proc_executor = pool
        with patch(
            "tensorrt_llm.metrics.perf_utils.time.perf_counter", side_effect=[1.0, 1.5, 1.5, 1.75]
        ):
            if fails:
                with pytest.raises(ValueError, match="bad input"):
                    await server._run_input_processor(process)
            else:
                assert await server._run_input_processor(process) == "processed"

    assert observer.call_args_list == [
        call("processor_queue", 0.5, None),
        call("processor", 0.25, None),
    ]


@pytest.mark.asyncio
async def test_input_stage_preserves_failure_and_records_elapsed_time(observer):
    with patch("tensorrt_llm.metrics.perf_utils.time.perf_counter", side_effect=[1.0, 2.0]):
        with pytest.raises(ValueError, match="bad media"):
            with _time_input_processing("media_load"):
                raise ValueError("bad media")

    observer.assert_called_once_with("media_load", 1.0, None)


@pytest.mark.asyncio
async def test_disabled_input_metrics_do_not_read_clock():
    server = object.__new__(OpenAIServer)

    async def load():
        return "media"

    with ThreadPoolExecutor(max_workers=1) as pool:
        server._input_proc_executor = pool
        with patch("tensorrt_llm.metrics.perf_utils.time.perf_counter") as clock:
            assert await server._run_input_processor(lambda: "processed") == "processed"
            with _time_input_processing("media_load"):
                assert await load() == "media"
        clock.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_mm_encoder_prometheus_route_is_optional(enabled, tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    server = object.__new__(OpenAIServer)
    server.app = FastAPI()
    server.generator = SimpleNamespace(args=SimpleNamespace(return_perf_metrics=enabled))

    server.register_mm_encoder_routes()

    with TestClient(server.app) as client:
        response = client.get("/prometheus/metrics")
    assert response.status_code == (200 if enabled else 404)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["openai_mm_encoder", "openai_responses"])
@pytest.mark.parametrize("fails", [False, True])
async def test_total_includes_inline_preprocessing_but_not_generation(route, fails, observer):
    server = object.__new__(OpenAIServer)
    server.model = "test-model"
    server.allow_request_chat_template = False
    server.model_config = None
    server.multimodal_server_config = None
    server.enable_store = False
    server.conversation_store = None
    server.use_harmony = False
    server.tokenizer = None
    server.processor = None
    server.chat_template = None
    server.tool_parser = None
    server.await_disconnected = AsyncMock()
    if route != "openai_mm_encoder":
        server._extract_metrics = AsyncMock()
    now = [1.0]
    server._collect_perf_metrics = True
    server._adjusted_steady_clock = SimpleNamespace(
        now=lambda: now[0], to_reference_time=lambda value: value
    )
    server.metrics_collector = None

    async def complete():
        now[0] += 10.0

    promise = SimpleNamespace(
        request_id="1",
        finished=True,
        outputs=[SimpleNamespace(request_perf_metrics=RequestPerfMetrics())],
        candidate_metrics=[],
        metrics_dict={},
        prompt_token_ids=[7, 8],
        aresult=AsyncMock(side_effect=complete),
        disaggregated_params=SimpleNamespace(
            multimodal_embedding_handles=[{"tensor_size": [4, 8]}]
        ),
    )

    def generate_async(**kwargs):
        now[0] += 0.05
        if fails:
            raise ValueError("bad preprocessing")
        return promise

    server.generator = SimpleNamespace(
        args=SimpleNamespace(reasoning_parser=None, num_postprocess_workers=0),
        generate_async=generate_async,
    )
    if route == "openai_mm_encoder":
        request = ChatCompletionRequest(
            model=server.model,
            messages=[{"role": "user", "content": "hi"}],
            prompt_token_ids=[7, 8],
        )
    else:
        request = ResponsesRequest(model=server.model, input="hi")
    raw_request = server._synthetic_request(request.model_dump(exclude_none=True))
    raw_request.scope["headers"].append((RETURN_METRICS_HEADER.encode(), b"1"))
    writer = MagicMock()
    messages = []
    response = None

    async def app(scope, receive, send):
        nonlocal response
        response = await getattr(server, route)(request, raw_request)
        await response(scope, receive, send)

    async def send(message):
        messages.append(message)

    middleware = PerfMetricsMiddleware(
        app, expose_headers=True, writer=writer, input_processing_observer=observer
    )

    with (
        patch("tensorrt_llm.metrics.perf_utils.time.perf_counter", side_effect=lambda: now[0]),
        patch(
            "tensorrt_llm.serve.openai_server.parse_chat_messages_coroutines",
            side_effect=lambda *args, **kwargs: ([], asyncio.sleep(0, result=(None, None)), {}, []),
        ),
        patch(
            "tensorrt_llm.serve.openai_server.responses_api_request_preprocess",
            new=AsyncMock(return_value=({"prompt": "hi"}, None)),
        ),
        patch(
            "tensorrt_llm.serve.openai_server.responses_api_create_response",
            new=AsyncMock(return_value=SimpleNamespace(model_dump=lambda **kwargs: {})),
        ),
    ):
        await middleware(raw_request.scope, raw_request.receive, send)

    assert response.status_code == (400 if fails else 200)
    totals = [sample.args[1] for sample in observer.call_args_list if sample.args[0] == "total"]
    assert totals == pytest.approx([0.05])
    if route == "openai_mm_encoder" and not fails:
        headers = {name.decode(): value.decode() for name, value in messages[0]["headers"]}
        assert "server_input_total;dur=50.000000" in headers[SERVER_TIMING_HEADER]
        writer.submit.assert_called_once()
        record = writer.submit.call_args.args[0]
        assert record["request_id"] == "1"
        assert record["phases"]["server"]["input_processing_metrics"]["total"] == pytest.approx(
            0.05
        )
    else:
        writer.submit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("collect_metrics", [False, True])
async def test_background_batch_items_have_separate_input_metrics_and_jsonl(collect_metrics):
    server = object.__new__(OpenAIServer)
    server.model = "test-model"
    server._collect_perf_metrics = collect_metrics
    server.metrics_collector = MagicMock()
    server._perf_metrics_writer = MagicMock()
    gate = asyncio.Event()
    item_states = {}

    async def chat(request, raw_request):
        await gate.wait()
        key = str(request.max_completion_tokens)
        item_states[key] = raw_request.scope["state"]
        observer = _input_processing_observer.get()
        if collect_metrics:
            await asyncio.to_thread(observer, "processor", request.max_completion_tokens / 10, None)
            raw_request.state.perf_metrics_records.append(
                {"request_id": key, "input": dict(raw_request.state.input_processing_metrics)}
            )
        else:
            assert observer is None
        return JSONResponse(
            {
                "id": key,
                "object": "chat.completion",
                "created": 1,
                "model": server.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    server.openai_chat = chat
    request = AnthropicCreateBatchRequest(
        requests=[
            {
                "custom_id": str(index),
                "params": {
                    "model": server.model,
                    "max_tokens": index,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            }
            for index in (1, 2)
        ]
    )
    batch_ids = []

    async def app(scope, receive, send):
        response = await server.anthropic_create_batch(request)
        batch_ids.append(json.loads(response.body)["id"])
        await response(scope, receive, send)

    async def send(message):
        pass

    scope = {"type": "http", "headers": []}
    if collect_metrics:
        await PerfMetricsMiddleware(app, expose_headers=True)(scope, None, send)
    else:
        await app(scope, None, send)
    gate.set()
    store = server._batch_store()
    await asyncio.wait_for(store._batches[batch_ids[0]].task, timeout=5)

    assert store.get(batch_ids[0]).request_counts.succeeded == 2
    assert _input_processing_observer.get() is None
    if collect_metrics:
        assert scope["state"]["input_processing_metrics"] == {}
        assert item_states["1"]["input_processing_metrics"] == {"processor": 0.1}
        assert item_states["2"]["input_processing_metrics"] == {"processor": 0.2}
        server._perf_metrics_writer.submit.assert_has_calls(
            [
                call({"request_id": key, "input": {"processor": int(key) / 10}})
                for key in ("1", "2")
            ],
            any_order=True,
        )
        assert server._perf_metrics_writer.submit.call_count == 2
        server.metrics_collector.log_input_processing.assert_has_calls(
            [call("processor", 0.1, None), call("processor", 0.2, None)], any_order=True
        )
    else:
        assert all("input_processing_metrics" not in state for state in item_states.values())
        server._perf_metrics_writer.submit.assert_not_called()
        server.metrics_collector.log_input_processing.assert_not_called()

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
"""output_length_prediction wired through the disaggregated config, coordinator and service."""

import asyncio
import json
from unittest.mock import AsyncMock

import msgpack
import pytest
from fastapi.testclient import TestClient

from tensorrt_llm.llmapi.disagg_utils import (
    DisaggServerConfig,
    OutputLengthPredictionConfig,
    extract_disagg_cfg,
)
from tensorrt_llm.serve import output_length_predictor as olp
from tensorrt_llm.serve.coordinator_server import CoordinatorServer
from tensorrt_llm.serve.disagg_coordinator import CoordinatorClient, DisaggCoordinatorService
from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ConversationParams,
    DisaggregatedParams,
    UsageInfo,
)
from tensorrt_llm.serve.router import Router

pytestmark = pytest.mark.cpu_only


def _client_factory(*_args, **_kwargs):
    return AsyncMock()


def _config(predictor="ridge"):
    config = DisaggServerConfig(server_configs=[])
    if predictor is not None:
        config.output_length_prediction = OutputLengthPredictionConfig(predictor=predictor)
    return config


def test_config_parsing():
    assert extract_disagg_cfg().output_length_prediction is None
    assert extract_disagg_cfg(output_length_prediction={}).output_length_prediction.predictor == (
        "ridge"
    )
    config = extract_disagg_cfg(output_length_prediction={"predictor": "ridge"})
    assert config.output_length_prediction.predictor == "ridge"
    with pytest.raises(ValueError):
        extract_disagg_cfg(output_length_prediction={"predictor": "ema"})
    with pytest.raises(ValueError):
        extract_disagg_cfg(output_length_prediction={"model": "file.json"})
    # Not a per-role option: it is not copied into the server sections.
    config = extract_disagg_cfg(
        output_length_prediction={"predictor": "ridge"}, context_servers={"num_instances": 0}
    )
    assert all("output_length_prediction" not in s.other_args for s in config.server_configs)


@pytest.mark.asyncio
async def test_coordinator_predicts_and_learns_in_process():
    coordinator = DisaggCoordinatorService(_config(), client_factory=_client_factory)
    tokens, key = await coordinator.predict_output_length("c", 1000)
    assert tokens == olp.PRIOR_TOKENS
    assert await coordinator.record_output_lengths([("c", 1000, 64.0, key)]) == 1
    assert (await coordinator.predict_output_length("c", 1200))[0] == 64.0
    off = DisaggCoordinatorService(_config(None), client_factory=_client_factory)
    with pytest.raises(ValueError, match="not configured"):
        await off.predict_output_length("c", 1000)


def test_coordinator_server_route():
    coordinator = DisaggCoordinatorService(_config(), client_factory=_client_factory)
    # No lifespan: the coordinator is not started (it has no servers to wait for).
    http = TestClient(CoordinatorServer(coordinator).app)

    def post(body):
        raw = http.post("/output_length", content=msgpack.packb(body, use_bin_type=True))
        return raw.status_code, msgpack.unpackb(raw.content, raw=False)

    status, reply = post({"op": "predict", "conv": "c", "isl": 1000})
    assert status == 200 and reply["tokens"] == olp.PRIOR_TOKENS
    status, reply = post({"op": "record", "records": [["c", 1000, 80, reply["key"]]]})
    assert status == 200 and reply == {"learned": 1}
    assert post({"op": "predict", "conv": "c", "isl": 1100})[1]["tokens"] == 80.0
    nan, inf = float("nan"), float("inf")
    for bad in (
        {"op": "fit"},
        {"op": "predict", "conv": 3, "isl": 1},
        {"op": "predict", "conv": "c", "isl": True},
        {"op": "predict", "conv": "c", "isl": nan},
        {"op": "predict", "conv": "c", "isl": -inf},
        {"op": "record", "conv": "c", "isl": 1},
        {"op": "record", "records": [["c", 1, 5, "k"]]},
        {"op": "record", "records": [["c", 1, nan, None]]},
        {"op": "record", "records": [["c", inf, 5, None]]},
        {"op": "record", "records": [["c", 1, -5, None]]},
        {"op": "record", "records": [[None, 1000, 1e308, None]]},
        {"op": "record", "records": [["c", olp.MAX_TOKENS + 1, 5, None]]},
        {"op": "predict", "conv": "c", "isl": 1e308},
        {"op": "record", "records": [["c", 1, 5]]},
        {"op": "record", "records": [["c", 1, 5, None]] * (olp.RECORD_BATCH_MAX + 1)},
    ):
        assert post(bad)[0] == 400, bad
    # NaN never reached the models: predictions and priorities stay finite.
    tokens = post({"op": "predict", "conv": None, "isl": 1100})[1]["tokens"]
    assert tokens == tokens and 0.0 <= olp.priority_of(tokens) <= 1.0


@pytest.mark.asyncio
async def test_coordinator_client_round_trip():
    """A fleet worker's CoordinatorClient reaches the coordinator's predictor over /output_length."""
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    coordinator = DisaggCoordinatorService(_config(), client_factory=_client_factory)
    seen = []

    async def handler(request):
        body = msgpack.unpackb(await request.read(), raw=False)
        seen.append(body)
        if body["op"] == "predict":
            tokens, key = await coordinator.predict_output_length(body["conv"], body["isl"])
            reply = {"tokens": tokens, "key": key}
        else:
            reply = {
                "learned": await coordinator.record_output_lengths(
                    [tuple(record) for record in body["records"]]
                )
            }
        return web.Response(body=msgpack.packb(reply), content_type="application/msgpack")

    app = web.Application()
    app.router.add_post("/output_length", handler)
    async with TestServer(app) as server:
        client = CoordinatorClient(f"http://{server.host}:{server.port}", _config())
        try:
            tokens, key = await client.predict_output_length("c", 1000)
            assert tokens == olp.PRIOR_TOKENS
            assert await client.record_output_lengths([("c", 1000, 77.0, key)]) == 1
            assert (await client.predict_output_length("c", 1100))[0] == 77.0
        finally:
            await client.stop()
    assert [b["op"] for b in seen] == ["predict", "record", "predict"]


def _ctx_response(prompt_tokens):
    return CompletionResponse(
        model="m",
        usage=UsageInfo(
            prompt_tokens=prompt_tokens, completion_tokens=1, total_tokens=prompt_tokens + 1
        ),
        prompt_token_ids=[1, 2, 3],
        choices=[
            CompletionResponseChoice(
                index=0,
                text="",
                finish_reason="length",
                disaggregated_params=DisaggregatedParams(
                    request_type="context_only", disagg_request_id=42, ctx_request_id=42
                ),
            )
        ],
    )


@pytest.mark.asyncio
async def test_service_sets_priority_before_generation_placement(monkeypatch):
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    config = _config()
    coordinator = DisaggCoordinatorService(config, client_factory=_client_factory)
    coordinator._ctx_router = AsyncMock(spec=Router)
    coordinator._gen_router = AsyncMock(spec=Router)
    coordinator._ctx_router.get_next_server.return_value = ("ctx:1", {})
    placed = []

    async def place_generation(req, **kwargs):
        placed.append(req.priority)
        return "gen:1", {}

    coordinator._gen_router.get_next_server.side_effect = place_generation
    service = OpenAIDisaggregatedService(config, coordinator, client_factory=_client_factory)
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._ctx_client.send_request.return_value = _ctx_response(5000)
    final = f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': 300}})}\n\n"

    async def gen_stream():
        yield b'data: {"choices": [{"text": "x"}]}\n\n'
        yield final.encode()

    service._gen_client.send_request.return_value = gen_stream()

    async def send(conv):
        request = CompletionRequest(
            model="m",
            prompt="hi",
            max_tokens=7,
            stream=True,
            conversation_params=ConversationParams(conversation_id=conv),
        )
        response = await service._send_disagg_request_ctx_first(request)
        return [chunk async for chunk in response]

    chunks = await send("conv-1")
    # The client did not ask for usage: the generation leg did, and its usage chunk is dropped.
    gen_req = service._gen_client.send_request.call_args.args[0]
    assert gen_req.stream_options.include_usage
    assert final.encode() not in b"".join(chunks)
    assert placed == [pytest.approx(olp.priority_of(olp.PRIOR_TOKENS))]
    await asyncio.sleep(0.01)  # the record task
    predictor = coordinator._output_length_predictor
    assert predictor.table.recorded == 1
    service._gen_client.send_request.return_value = gen_stream()
    await send("conv-1")
    # The conversation's learned reply (300 tokens) now sets the priority; max_tokens=7 does not.
    assert placed[-1] == pytest.approx(olp.priority_of(300.0))


@pytest.mark.asyncio
async def test_service_without_prediction_leaves_priority_alone(monkeypatch):
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    config = _config(None)
    coordinator = DisaggCoordinatorService(config, client_factory=_client_factory)
    coordinator._ctx_router = AsyncMock(spec=Router)
    coordinator._gen_router = AsyncMock(spec=Router)
    coordinator._ctx_router.get_next_server.return_value = ("ctx:1", {})
    coordinator._gen_router.get_next_server.return_value = ("gen:1", {})
    service = OpenAIDisaggregatedService(config, coordinator, client_factory=_client_factory)
    assert service._output_length_priority is None
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._ctx_client.send_request.return_value = _ctx_response(5000)
    service._gen_client.send_request.return_value = "response"
    request = CompletionRequest(model="m", prompt="hi")
    assert await service._send_disagg_request_ctx_first(request) == "response"
    assert service._gen_client.send_request.call_args.args[0].priority is None


@pytest.mark.asyncio
async def test_select_predicts_the_output_length():
    """The coordinator predicts inside /select: the pooled placement orders by it, no extra call."""
    from tensorrt_llm.llmapi.disagg_utils import CtxGenServerConfig, RouterConfig, ServerRole

    config = _config()
    config.server_configs = [CtxGenServerConfig(type="gen", hostname="g", port=1)]
    config.ctx_router_config = RouterConfig(type="round_robin", server_role=ServerRole.CONTEXT)
    config.gen_router_config = RouterConfig(
        type="load_balancing",
        server_role=ServerRole.GENERATION,
        args={"max_requests_per_server": 1},
    )
    coordinator = DisaggCoordinatorService(config, client_factory=_client_factory)
    await coordinator.record_output_lengths(
        [("short", 1000, 50.0, None), ("long", 1000, 5000.0, None)]
    )
    server, info, _ = await coordinator.select("generation", {}, 1, None)
    assert server == "g:1" and olp.OUTPUT_LENGTH_KEY not in info
    waits = {
        rid: asyncio.create_task(
            coordinator.select(
                "generation", {olp.OUTPUT_LENGTH_KEY: {"conv": conv, "isl": 1500.0}}, rid, None
            )
        )
        for rid, conv in ((2, "long"), (3, "short"))
    }
    for _ in range(5):
        await asyncio.sleep(0)
    await coordinator.finish("generation", 1)
    for _ in range(5):
        await asyncio.sleep(0)
    assert waits[3].done() and not waits[2].done()  # the shorter predicted reply goes first
    _, info, _ = await waits[3]
    prediction = info[olp.OUTPUT_LENGTH_KEY]
    assert prediction["tokens"] == 50.0 and isinstance(prediction["key"], int)
    assert prediction["priority"] == pytest.approx(olp.priority_of(50.0))
    await coordinator.finish("generation", 3)
    assert (await waits[2])[1][olp.OUTPUT_LENGTH_KEY]["tokens"] == 5000.0
    await coordinator.finish("generation", 2)
    # Invalid inputs are ignored rather than predicted.
    _, info, _ = await coordinator.select(
        "generation", {olp.OUTPUT_LENGTH_KEY: {"conv": None, "isl": float("nan")}}, 4, None
    )
    assert olp.OUTPUT_LENGTH_KEY not in info
    await coordinator.finish("generation", 4)

    # A predictor error never fails the placement: the request gets the prior's priority.
    def overflow(*_args):
        raise OverflowError("math range error")

    coordinator._output_length_predictor.predict = overflow
    routing_key = {olp.OUTPUT_LENGTH_KEY: {"conv": "c", "isl": 9000.0}}
    server, info, _ = await coordinator.select("generation", routing_key, 5, None)
    assert server == "g:1" and olp.OUTPUT_LENGTH_KEY not in info
    assert routing_key["priority"] == pytest.approx(olp.priority_of(olp.PRIOR_TOKENS))
    assert coordinator._output_length_errors == 1
    await coordinator.finish("generation", 5)
    await coordinator.stop()


@pytest.mark.asyncio
async def test_fleet_worker_gets_the_prediction_from_select(monkeypatch):
    """A delegating generation router sends the prediction inputs and applies the returned priority."""
    from tensorrt_llm.llmapi.disagg_utils import RouterConfig, ServerRole
    from tensorrt_llm.serve.router import CoordinatorDelegatingRouter, create_router

    local = create_router(
        RouterConfig(type="load_balancing", server_role=ServerRole.GENERATION), ["g:1"]
    )
    router = CoordinatorDelegatingRouter("http://c:1", local, "generation", 5.0)
    assert router.enable_output_length_prediction()
    # A conversation router's key is the conversation id: it keeps the separate prediction call.
    conversation = create_router(
        RouterConfig(type="conversation", server_role=ServerRole.GENERATION), ["g:1"]
    )
    assert not CoordinatorDelegatingRouter(
        "http://c:1", conversation, "generation", 5.0
    ).enable_output_length_prediction()
    assert not CoordinatorDelegatingRouter(
        "http://c:1", local, "context", 5.0
    ).enable_output_length_prediction()
    sent = []

    async def post_select(payload):
        sent.append(payload)
        return {
            "server": "g:1",
            "info": {olp.OUTPUT_LENGTH_KEY: {"tokens": 1024.0, "key": 3, "priority": 0.5}},
        }

    monkeypatch.setattr(router, "_post_select", post_select)
    request = CompletionRequest(
        model="m",
        prompt="x",
        conversation_params=ConversationParams(conversation_id="c"),
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only",
            ctx_request_id=5,
            ctx_usage=UsageInfo(prompt_tokens=777),
        ),
    )
    server, info = await router.get_next_server(request)
    assert server == "g:1" and request.priority == 0.5 and info[olp.OUTPUT_LENGTH_KEY]["key"] == 3
    assert sent[0]["routing_key"][olp.OUTPUT_LENGTH_KEY] == {"conv": "c", "isl": 777.0}
    await router.close()


def test_prediction_requires_context_first():
    """generation_first requests reach the generation server without a prompt length to predict from."""
    with pytest.raises(ValueError, match="context_first"):
        extract_disagg_cfg(output_length_prediction={}, schedule_style="generation_first")
    config = extract_disagg_cfg(output_length_prediction={}, schedule_style="context_first")
    assert config.output_length_prediction.predictor == "ridge"


def test_prediction_refuses_a_generation_first_override_after_parsing():
    """`trtllm-serve disaggregated -s generation_first` sets schedule_style after the parse-time check."""
    config = _config()
    config.schedule_style = "generation_first"
    coordinator = DisaggCoordinatorService(config, client_factory=_client_factory)
    with pytest.raises(ValueError, match="context_first"):
        OpenAIDisaggregatedService(config, coordinator, client_factory=_client_factory)


@pytest.mark.asyncio
async def test_request_without_a_context_phase_gets_the_prior_priority(monkeypatch):
    """Conditional disaggregation served locally: a client-sent priority never passes through."""
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    config = _config()
    coordinator = DisaggCoordinatorService(config, client_factory=_client_factory)
    coordinator._ctx_router = AsyncMock(spec=Router)
    coordinator._gen_router = AsyncMock(spec=Router)
    service = OpenAIDisaggregatedService(config, coordinator, client_factory=_client_factory)
    service._check_conditional_disagg = AsyncMock(return_value=("gen:1", False))
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._gen_client.send_request.return_value = "response"
    request = CompletionRequest(model="m", prompt="hi", priority=1.0)
    assert await service._send_disagg_request_ctx_first(request) == "response"
    gen_req = service._gen_client.send_request.call_args.args[0]
    assert gen_req.priority == pytest.approx(olp.priority_of(olp.PRIOR_TOKENS))
    service._ctx_client.send_request.assert_not_called()


@pytest.mark.asyncio
async def test_generation_first_stream_releases_the_context_server_without_a_seat(monkeypatch):
    """No generation seat (503): the context reservation made before it is released at once."""
    from tensorrt_llm.llmapi.disagg_utils import ServerRole
    from tensorrt_llm.serve.router import LoadBalancingRouter
    from tensorrt_llm.serve.seat_pool import NoGenerationSeatError

    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    config = _config(None)
    config.schedule_style = "generation_first"
    coordinator = DisaggCoordinatorService(config, client_factory=_client_factory)
    ctx_router = LoadBalancingRouter(ServerRole.CONTEXT, ["ctx:1"])
    coordinator._ctx_router = ctx_router
    coordinator._gen_router = AsyncMock(spec=Router)
    service = OpenAIDisaggregatedService(config, coordinator, client_factory=_client_factory)
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._gen_client.send_request.side_effect = NoGenerationSeatError("no seat within 60 s")
    request = CompletionRequest(model="m", prompt="hi", stream=True)
    with pytest.raises(NoGenerationSeatError):
        await service._send_disagg_request_gen_first(request)
    assert ctx_router._get_server_load("ctx:1") == 0 and ctx_router._req_routing_table == {}
    service._ctx_client.send_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("fleet", [False, True], ids=["in_process", "fleet"])
async def test_responses_request_waits_for_a_seat_at_the_prior(fleet):
    """/v1/responses is not predicted: a pooled seat queue ranks it at the prior, not by a client value."""
    from tensorrt_llm.llmapi.disagg_utils import CtxGenServerConfig, RouterConfig, ServerRole
    from tensorrt_llm.serve.openai_protocol import ResponsesRequest
    from tensorrt_llm.serve.router import CoordinatorDelegatingRouter, PooledLoadBalancingRouter

    config = _config()
    config.server_configs = [CtxGenServerConfig(type="gen", hostname="g", port=1)]
    config.ctx_router_config = RouterConfig(type="round_robin", server_role=ServerRole.CONTEXT)
    config.gen_router_config = RouterConfig(
        type="load_balancing",
        server_role=ServerRole.GENERATION,
        args={"max_requests_per_server": 1},
    )
    coordinator = DisaggCoordinatorService(config, client_factory=_client_factory)
    if fleet:
        coordinator._gen_router = CoordinatorDelegatingRouter(
            "http://c:1", coordinator.gen_router, "generation", 5.0
        )
    service = OpenAIDisaggregatedService(config, coordinator, client_factory=_client_factory)
    gen_router = service._gen_router
    local = gen_router._local if fleet else gen_router
    assert isinstance(local, PooledLoadBalancingRouter)
    prior = olp.priority_of(olp.PRIOR_TOKENS)
    for client_priority in (0.0, 1.0):
        # ResponsesRequest keeps unknown fields: a client "priority" is an extra, not a priority.
        request = ResponsesRequest(model="m", input="hi", priority=client_priority)
        assert local.routing_key(request)["priority"] == pytest.approx(prior)
    # A request model with a priority field keeps its (orchestrator-set) value.
    assert local.routing_key(CompletionRequest(model="m", prompt="x", priority=0.9))[
        "priority"
    ] == pytest.approx(0.9)
    if not fleet:
        # The in-process placement orders by the same priority.
        first = CompletionRequest(model="m", prompt="x", priority=0.5)
        assert (await local.get_next_server(first))[0] == "g:1"
        responses = asyncio.create_task(
            local.get_next_server(ResponsesRequest(model="m", input="hi", priority=0.0))
        )
        completion = asyncio.create_task(
            local.get_next_server(CompletionRequest(model="m", prompt="x", priority=0.6))
        )
        for _ in range(5):
            await asyncio.sleep(0)
        await local.finish_request(first)
        for _ in range(5):
            await asyncio.sleep(0)
        assert responses.done() and not completion.done()
        completion.cancel()
        await asyncio.gather(completion, return_exceptions=True)
    await coordinator.stop()


@pytest.mark.asyncio
async def test_select_keeps_one_row_per_request_and_drops_it_on_failure():
    """A repeated /select reuses its prediction key; a select that gets no seat drops its row."""
    from tensorrt_llm.llmapi.disagg_utils import CtxGenServerConfig, RouterConfig, ServerRole
    from tensorrt_llm.serve.seat_pool import NoGenerationSeatError

    config = _config()
    config.server_configs = [CtxGenServerConfig(type="gen", hostname="g", port=1)]
    config.ctx_router_config = RouterConfig(type="round_robin", server_role=ServerRole.CONTEXT)
    config.gen_router_config = RouterConfig(
        type="load_balancing",
        server_role=ServerRole.GENERATION,
        args={"max_requests_per_server": 1},
    )
    coordinator = DisaggCoordinatorService(config, client_factory=_client_factory)
    predictor = coordinator._output_length_predictor

    def key_of(rid):
        return {olp.OUTPUT_LENGTH_KEY: {"conv": "c", "isl": 1000.0}}

    _, info, _ = await coordinator.select("generation", key_of(1), 1, None)
    _, again, _ = await coordinator.select("generation", key_of(1), 1, None)
    assert again[olp.OUTPUT_LENGTH_KEY]["key"] == info[olp.OUTPUT_LENGTH_KEY]["key"]
    assert len(predictor._pending) == 1
    parked = asyncio.create_task(coordinator.select("generation", key_of(2), 2, None))
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(predictor._pending) == 2
    await coordinator.stop()  # fails the parked select: no seat, so no reply will come
    with pytest.raises(NoGenerationSeatError):
        await parked
    assert len(predictor._pending) == 1 and predictor.dropped == 1
    await coordinator.finish("generation", 1)
    assert coordinator._prediction_keys == {}


@pytest.mark.asyncio
async def test_failed_generation_drops_the_prediction_row(monkeypatch):
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    config = _config()
    coordinator = DisaggCoordinatorService(config, client_factory=_client_factory)
    coordinator._ctx_router = AsyncMock(spec=Router)
    coordinator._gen_router = AsyncMock(spec=Router)
    coordinator._ctx_router.get_next_server.return_value = ("ctx:1", {})
    coordinator._gen_router.get_next_server.return_value = ("gen:1", {})
    service = OpenAIDisaggregatedService(config, coordinator, client_factory=_client_factory)
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._ctx_client.send_request.return_value = _ctx_response(5000)
    service._gen_client.send_request.side_effect = ConnectionResetError("generation server died")
    request = CompletionRequest(
        model="m", prompt="hi", conversation_params=ConversationParams(conversation_id="c")
    )
    with pytest.raises(ConnectionResetError):
        await service._send_disagg_request_ctx_first(request)
    await asyncio.sleep(0.01)  # the record task
    predictor = coordinator._output_length_predictor
    assert len(predictor._pending) == 0 and predictor.dropped == 1 and predictor.learned == 0

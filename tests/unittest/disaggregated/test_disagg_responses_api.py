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
"""Offline tests for the Responses API on the disaggregated path.

A Responses response has no ``choices``, so the context-to-generation handoff
that completions and chat carry on ``choices[0]`` sits at its top level.
"""

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
import numpy as np
import pytest
from fastapi import HTTPException

from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_disagg_service import (
    OpenAIDisaggregatedService,
    _ctx_handoff_slots,
    _ctx_usage_info,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    DisaggregatedParams,
    ModelCard,
    ModelList,
    PromptTokensDetails,
    ResponsesRequest,
    ResponsesResponse,
    UsageInfo,
)
from tensorrt_llm.serve.responses_utils import (
    create_response_non_store,
    finish_reason_mapping,
    responses_done_generator,
)

pytestmark = pytest.mark.cpu_only


def _responses_response(finish_reason="length", status="incomplete"):
    return ResponsesResponse(
        model="m",
        output=[],
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        background=False,
        service_tier="auto",
        status=status,
        top_logprobs=0,
        truncation="disabled",
        finish_reason=finish_reason,
        prompt_token_ids=[1, 2],
        prompt_token_ids_b64="AQAAAAIAAAA=",
        disaggregated_params=DisaggregatedParams(
            request_type="context_only", ctx_request_id=7, disagg_request_id=7
        ),
    )


def _service(**attrs):
    service = OpenAIDisaggregatedService.__new__(OpenAIDisaggregatedService)

    async def ready():
        return True

    service.is_ready = ready
    for name, value in attrs.items():
        setattr(service, name, value)
    return service


# ---------------------------------------------------------------------------
# The handoff
# ---------------------------------------------------------------------------


def test_the_handoff_is_read_per_choice_for_chat_and_top_level_for_responses():
    chat = ChatCompletionResponse(
        model="m",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "length",
                "disaggregated_params": {"request_type": "context_only", "ctx_request_id": 3},
            }
        ],
        usage={"prompt_tokens": 1, "total_tokens": 2, "completion_tokens": 1},
    )
    assert [s.disaggregated_params.ctx_request_id for s in _ctx_handoff_slots(chat)] == [3]
    response = _responses_response()
    assert _ctx_handoff_slots(response) == [response]


@pytest.mark.parametrize(
    "finish_reason, need_gen", [("length", True), ("not_finished", True), ("stop", False)]
)
def test_a_finished_context_response_goes_to_the_client_without_the_handoff(
    finish_reason, need_gen
):
    response = _responses_response(finish_reason=finish_reason)
    assert _service()._need_gen(response) is need_gen
    handoff = (
        response.disaggregated_params,
        response.prompt_token_ids,
        response.prompt_token_ids_b64,
        response.finish_reason,
    )
    if need_gen:
        assert all(field is not None for field in handoff)
    else:
        assert handoff == (None, None, None, None)


@pytest.mark.parametrize(
    "finish_reason, status",
    [("not_finished", "incomplete"), ("length", "incomplete"), ("stop", "completed")],
)
def test_finish_reason_mapping(finish_reason, status):
    assert finish_reason_mapping(finish_reason) == status


def test_an_unknown_finish_reason_names_itself():
    with pytest.raises(RuntimeError, match="wat"):
        finish_reason_mapping("wat")


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"prompt_token_ids": [1, 2, 3]}, [1, 2, 3]),
        (
            {
                "prompt_token_ids_b64": base64.b64encode(
                    np.asarray([5, 6, 7], dtype=np.int32).tobytes()
                ).decode("ascii")
            },
            [5, 6, 7],
        ),
        ({}, None),
    ],
)
def test_the_generation_worker_uses_the_relayed_prompt(fields, expected):
    request = ResponsesRequest(model="m", input="hi", **fields)
    assert request.relayed_prompt_token_ids() == expected
    assert request.prompt_token_ids == fields.get("prompt_token_ids")


@pytest.mark.parametrize(
    "usage, expected",
    [
        (
            SimpleNamespace(
                input_tokens=96,
                input_tokens_details=SimpleNamespace(cached_tokens=64),
                output_tokens=1,
                total_tokens=97,
            ),
            (96, 1, 64),
        ),
        (
            SimpleNamespace(
                input_tokens=96, input_tokens_details=None, output_tokens=1, total_tokens=97
            ),
            (96, 1, 0),
        ),
    ],
)
def test_the_context_usage_is_handed_off_as_usage_info(usage, expected):
    carried = _ctx_usage_info(SimpleNamespace(usage=usage))
    assert (
        carried.prompt_tokens,
        carried.completion_tokens,
        carried.prompt_tokens_details.cached_tokens,
    ) == expected
    already = UsageInfo(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
    )
    assert _ctx_usage_info(SimpleNamespace(usage=already)) is already


@pytest.mark.parametrize("use_harmony", [False, True])
def test_a_context_only_response_carries_the_handoff(use_harmony):
    """The single handoff token is not a complete Harmony message to parse."""
    request = ResponsesRequest(
        model="m", input="hi", disaggregated_params=DisaggregatedParams(request_type="context_only")
    )
    output = SimpleNamespace(
        index=0, text="", token_ids=[7], finish_reason="length", disaggregated_params=None
    )
    result = SimpleNamespace(outputs=[output], prompt_token_ids=[1, 2], cached_tokens=0)
    with patch(
        "tensorrt_llm.serve.responses_utils._create_output_content_harmony",
        side_effect=AssertionError("parsed a context-only output"),
    ):
        response = create_response_non_store(
            generation_result=result,
            request=request,
            sampling_params=request.to_sampling_params(),
            model_name="m",
            use_harmony=use_harmony,
        )
    assert response.output == []
    assert (response.finish_reason, response.prompt_token_ids) == ("length", [1, 2])


@pytest.mark.parametrize("as_b64", [False, True])
def test_a_context_worker_with_postprocessing_workers_returns_the_prompt(monkeypatch, as_b64):
    """A postprocessing worker's result has no prompt ids; the handoff still does."""
    import tensorrt_llm.serve.openai_server as server_module
    from tensorrt_llm.serve.openai_server import OpenAIServer

    request = ResponsesRequest(
        model="m",
        input="hi",
        store=False,
        disaggregated_params=DisaggregatedParams(
            request_type="context_only", return_prompt_token_ids_b64=as_b64
        ),
    )
    output = SimpleNamespace(
        index=0, text="", token_ids=[7], finish_reason="length", disaggregated_params=None
    )
    # What a postprocessing worker builds: its result carries no prompt_token_ids.
    postprocessed = create_response_non_store(
        generation_result=SimpleNamespace(outputs=[output], cached_tokens=0),
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="m",
        use_harmony=False,
        num_prompt_tokens=3,
    )
    assert postprocessed.prompt_token_ids is None

    async def nothing(*args, **kwargs):
        return None

    promise = SimpleNamespace(
        outputs=[SimpleNamespace(_postprocess_result=postprocessed)],
        prompt_token_ids=[5, 6, 7],
        aresult=nothing,
    )
    server = object.__new__(OpenAIServer)
    server.model = "m"
    server.enable_store = False
    server._is_visual_gen = False
    server.use_harmony = False
    server.tokenizer = None
    server.model_config = None
    server.processor = None
    server.tool_parser = None
    server.metrics_collector = None
    server.conversation_store = None
    server.generator = SimpleNamespace(
        args=SimpleNamespace(reasoning_parser=None, num_postprocess_workers=1),
        generate_async=lambda **kwargs: promise,
    )
    server._validate_internal_disagg_request = lambda *args: None
    server.await_disconnected = nothing
    server._extract_metrics = nothing

    async def preprocess(**kwargs):
        return [5, 6, 7], kwargs["request"].to_sampling_params()

    monkeypatch.setattr(server_module, "responses_api_request_preprocess", preprocess)
    raw_request = SimpleNamespace(
        state=SimpleNamespace(), headers={}, url=SimpleNamespace(path="/v1/responses")
    )

    body = json.loads(asyncio.run(server.openai_responses(request, raw_request)).body)

    if as_b64:
        ids = np.frombuffer(base64.b64decode(body["prompt_token_ids_b64"]), dtype=np.int32)
        assert (ids.tolist(), body["prompt_token_ids"]) == ([5, 6, 7], None)
    else:
        assert body["prompt_token_ids"] == [5, 6, 7]


# ---------------------------------------------------------------------------
# Streaming a request the context phase finished
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "finish_reason, status, terminal, details",
    [
        ("stop", "completed", "response.completed", None),
        ("length", "incomplete", "response.incomplete", {"reason": "max_output_tokens"}),
    ],
)
def test_a_context_finished_stream_replays_the_response(finish_reason, status, terminal, details):
    response = _responses_response(finish_reason=finish_reason, status=status)
    if details:
        response.incomplete_details = details

    async def collect():
        return [chunk.decode() async for chunk in responses_done_generator(response)]

    frames = asyncio.run(collect())
    assert all(f.startswith("event: ") and f.endswith("\n\n") for f in frames)
    events = [json.loads(f.split("data: ", 1)[1]) for f in frames]
    assert [e["type"] for e in events] == ["response.created", "response.in_progress", terminal]
    assert [e["sequence_number"] for e in events] == [0, 1, 2]
    assert events[-1]["response"]["incomplete_details"] == details


# ---------------------------------------------------------------------------
# What the orchestrator does with a Responses request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("store, warned", [(None, False), (False, False), (True, True)])
def test_store_is_forwarded_off_and_an_explicit_request_for_it_is_logged(store, warned):
    sent = []

    async def send(request, hooks):
        sent.append(request)
        return "relayed"

    request = ResponsesRequest(model="m", input="hi", **({} if store is None else {"store": store}))
    with patch("tensorrt_llm.serve.openai_disagg_service.logger") as mock_logger:
        assert asyncio.run(_service(_send_disagg_request=send).openai_responses(request)) == (
            "relayed"
        )
    assert sent[0].store is False
    assert mock_logger.warning_once.called is warned


def test_previous_response_id_is_rejected_with_a_400():
    request = ResponsesRequest(model="m", input="hi", previous_response_id="resp_1")
    with pytest.raises(HTTPException) as raised:
        asyncio.run(_service().openai_responses(request))
    assert raised.value.status_code == 400


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


class _StubCtxClient:
    def __init__(self, error=None):
        self.error = error
        self.servers = []

    async def get_json(self, endpoint, response_type, server):
        assert endpoint == "v1/models"
        self.servers.append(server)
        if self.error is not None:
            raise self.error
        return ModelList(data=[ModelCard(id="served-name")])


def _models_service(client, servers=("w0:8001", "w1:8001")):
    return _service(
        _ctx_client=client,
        _ctx_router=SimpleNamespace(servers=list(servers)),
        _count_tokens_rr_counter=0,
    )


def test_models_are_listed_by_a_context_worker_round_robin():
    client = _StubCtxClient()
    service = _models_service(client)
    assert asyncio.run(service.get_model()).data[0].id == "served-name"
    asyncio.run(service.get_model())
    assert client.servers == ["w0:8001", "w1:8001"]


@pytest.mark.parametrize(
    "client, servers, status",
    [
        (_StubCtxClient(), (), 503),
        (
            _StubCtxClient(
                aiohttp.ClientResponseError(
                    SimpleNamespace(real_url="http://w0:8001/v1/models"),
                    (),
                    status=404,
                    message="Not Found",
                )
            ),
            ("w0:8001",),
            404,
        ),
    ],
    ids=["no_context_worker", "worker_error"],
)
def test_a_models_failure_is_an_error_response(client, servers, status):
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._service = _models_service(client, servers)
    response = asyncio.run(server.get_model())
    assert response.status_code == status
    assert json.loads(response.body)["code"] == status

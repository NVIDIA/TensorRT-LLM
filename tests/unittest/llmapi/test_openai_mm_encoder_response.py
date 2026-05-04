# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.serve import openai_server
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.openai_server import OpenAIServer


class _FakePromise:
    request_id = 123
    finished = True
    disaggregated_params = DisaggregatedParams(
        multimodal_embedding_handles=[
            {
                "tensor_size": [2, 8],
                "ipc_handle": "handle-0",
            },
            {
                "tensor_size": [3, 8],
                "ipc_handle": "handle-1",
            },
        ],
        multimodal_hashes=[
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ],
        multimodal_item_runs=[[(1, 2, [])], [(3, 4, [])]],
        mrope_position_ids_handle={"tensor_size": [3, 1, 5]},
        mrope_position_deltas_handle={"tensor_size": [1, 1]},
    )

    async def aresult(self):
        return self

    def abort(self):
        raise AssertionError("unexpected abort")


class _FakeGenerator:
    def __init__(self):
        self.promise = _FakePromise()
        self.inputs = None

    def generate_async(self, *, inputs):
        self.inputs = inputs
        return self.promise


@pytest.mark.asyncio
async def test_openai_mm_encoder_response_carries_disaggregated_params(monkeypatch):
    async def mm_coroutines():
        return None, None

    def parse_chat_messages_coroutines(messages, model_config, multimodal_server_config):
        return [{"role": "user", "content": "describe image"}], mm_coroutines(), {}

    monkeypatch.setattr(
        openai_server, "parse_chat_messages_coroutines", parse_chat_messages_coroutines
    )
    monkeypatch.setattr(openai_server, "apply_chat_template", lambda **kwargs: "describe image")

    server = OpenAIServer.__new__(OpenAIServer)
    server.generator = _FakeGenerator()
    server.model = "qwen3-vl"
    server.model_config = SimpleNamespace(model_type="qwen3_vl")
    server.multimodal_server_config = None
    server.tokenizer = None
    server.processor = None

    request = ChatCompletionRequest(
        model="qwen3-vl",
        messages=[
            {
                "role": "user",
                "content": "describe image",
            }
        ],
    )

    response = await server.openai_mm_encoder(request, raw_request=None)
    body = json.loads(response.body)
    choice = body["choices"][0]

    assert choice["mm_embedding_handle"] == (
        _FakePromise.disaggregated_params.multimodal_embedding_handles
    )
    assert choice["disaggregated_params"]["multimodal_embedding_handles"] == (
        _FakePromise.disaggregated_params.multimodal_embedding_handles
    )
    assert choice["disaggregated_params"]["multimodal_hashes"] == (
        _FakePromise.disaggregated_params.multimodal_hashes
    )
    assert choice["disaggregated_params"]["multimodal_item_runs"] == [
        [[1, 2, []]],
        [[3, 4, []]],
    ]
    assert choice["disaggregated_params"]["mrope_position_ids_handle"] == {"tensor_size": [3, 1, 5]}
    assert choice["disaggregated_params"]["mrope_position_deltas_handle"] == {"tensor_size": [1, 1]}
    assert choice["disaggregated_params"]["request_type"] is None

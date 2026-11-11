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
"""A streamed Harmony request releases its parser state however the stream ends.

The process-wide HarmonyAdapter keeps one stateful parser per stream, holding
every token of the request until the state is cleaned up.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from llmapi.apps.render_helpers import use_harmony_vocab
from test_common.serve.app_factory import AggConfig
from test_common.serve.fakes import ScriptedOutput, ScriptedStep

from tensorrt_llm.serve.harmony_adapter import get_harmony_adapter
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, ResponsesRequest
from tensorrt_llm.serve.postprocess_handlers import (
    ResponsesAPIPostprocArgs,
    responses_api_streaming_post_processor,
)
from tensorrt_llm.serve.responses_utils import ResponsesStreamingProcessor

pytestmark = pytest.mark.cpu_only

_OUTPUT = (
    "<|channel|>analysis<|message|>Think it over.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>Done."
)


@pytest.fixture
def encoding(monkeypatch):
    use_harmony_vocab(monkeypatch)
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def _script(encoding, *, finished):
    """One engine response per token; an unfinished generation waits after its last token."""
    steps = [
        ScriptedStep(token_ids=(token_id,))
        for token_id in encoding.encode(_OUTPUT, allowed_special="all")
    ]
    if finished:
        steps[-1] = ScriptedStep(token_ids=steps[-1].token_ids, finish_reason="stop")
    return ScriptedOutput(steps=tuple(steps), hold_after_steps=not finished)


def _raw_request():
    return SimpleNamespace(state=SimpleNamespace(no_client_connection=True), headers={})


async def _read_until(frames, marker):
    async for frame in frames:
        if marker in frame:
            return
    raise AssertionError(f"the stream ended before {marker!r}")


@pytest.mark.parametrize("num_postprocess_workers", [0, 1], ids=["frontend", "postprocess_worker"])
def test_a_finished_responses_stream_releases_its_state(
    serve_env, encoding, num_postprocess_workers
):
    server = serve_env.build_openai_server(
        AggConfig(model_type="gpt_oss", num_postprocess_workers=num_postprocess_workers)
    )
    server.generator.script(_script(encoding, finished=True))
    client = serve_env.client(server.app)

    with client.stream(
        "POST", "/v1/responses", json={"model": "fake-model", "input": "hi", "stream": True}
    ) as response:
        assert response.status_code == 200
        events = [
            json.loads(line[len("data: ") :])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert events[-1]["type"] == "response.completed"
    response_id = events[-1]["response"]["id"]
    assert get_harmony_adapter().get_stream_state(f"responses-api-{response_id}") is None


def test_a_responses_client_that_disconnects_releases_the_state(serve_env, encoding):
    server = serve_env.build_openai_server(AggConfig(model_type="gpt_oss"))
    server.generator.script(_script(encoding, finished=False))
    request = ResponsesRequest(model="fake-model", input="hi", stream=True)
    stream_request_id = f"responses-api-{request.request_id}"

    async def disconnect_mid_stream():
        response = await server.openai_responses(request, _raw_request())
        frames = response.body_iterator
        await asyncio.wait_for(_read_until(frames, "response.output_text.delta"), timeout=30)
        assert get_harmony_adapter().get_stream_state(stream_request_id) is not None
        # What ClosingStreamingResponse does when the client goes away.
        await frames.aclose()

    asyncio.run(disconnect_mid_stream())

    assert get_harmony_adapter().get_stream_state(stream_request_id) is None


def test_a_chat_client_that_disconnects_releases_the_state(serve_env, encoding):
    server = serve_env.build_openai_server(AggConfig(model_type="gpt_oss"))
    server.generator.script(_script(encoding, finished=False))
    request = ChatCompletionRequest(
        model="fake-model", messages=[{"role": "user", "content": "hi"}], stream=True
    )

    async def disconnect_mid_stream():
        response = await server.chat_harmony(request, _raw_request())
        stream_request_id = str(server.generator.generate_calls[0].result.id)
        frames = response.body_iterator
        await asyncio.wait_for(_read_until(frames, '"content":"Done'), timeout=30)
        assert get_harmony_adapter().get_stream_state(stream_request_id) is not None
        await frames.aclose()
        return stream_request_id

    stream_request_id = asyncio.run(disconnect_mid_stream())

    assert get_harmony_adapter().get_stream_state(stream_request_id) is None


def test_a_failed_responses_post_processing_releases_the_state(monkeypatch):
    """A postprocess worker drops the request after a failure; nothing else releases the state."""
    use_harmony_vocab(monkeypatch)
    request = ResponsesRequest(model="fake-model", input="hi", stream=True)
    processor = ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="fake-model",
        use_harmony=True,
    )
    adapter = get_harmony_adapter()

    def fail_after_parsing(rsp):
        adapter.create_stream_state(processor.stream_request_id)
        raise ValueError("malformed output")

    monkeypatch.setattr(processor, "process_single_output", fail_after_parsing)
    args = ResponsesAPIPostprocArgs(
        model="fake-model",
        request=request,
        sampling_params=request.to_sampling_params(),
        use_harmony=True,
        streaming_processor=processor,
    )

    with pytest.raises(ValueError, match="malformed output"):
        responses_api_streaming_post_processor(SimpleNamespace(_done=False), args)

    assert adapter.get_stream_state(processor.stream_request_id) is None

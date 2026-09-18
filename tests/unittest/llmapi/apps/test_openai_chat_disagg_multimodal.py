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
"""What media reaches the engine on a relayed disagg chat request.

The router gives the generation worker the context worker's
``prompt_token_ids`` and relays the original ``messages`` unchanged, image
parts included. Those ids already have the placeholders expanded, so the media
must neither reach the engine again nor be fetched again.

The engine is stubbed, so what these tests read is the prompt dict as it leaves
``openai_chat``.
"""

import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import OpenAIServer

# No GPU, engine or sockets: this runs in the CPU-Generic CI stage, which
# selects with `-m cpu_only`.
pytestmark = pytest.mark.cpu_only

MODEL = "test-vlm"

# Stands in for what the context worker returns. The placeholders are already
# expanded in these ids; their values do not matter, only that they arrive.
RELAYED_TOKEN_IDS = [7, 11, 13, 17, 19]


class _StubModelConfig:
    """Enough of an HF config to pick the multimodal placeholder strategy.

    ``model_type`` has to be a class attribute: `resolve_top_level_model_type`
    reads it off the type, not the instance.
    """

    model_type = "qwen3_vl"
    vocab_size = 1024


def _image_url_part() -> dict:
    """One image as a `data:` URL, decoded in-process by the image media IO."""
    with io.BytesIO() as buf:
        Image.new("RGB", (8, 8), (0, 128, 255)).save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }


def _image_embeds_part() -> dict:
    """One precomputed embedding, in the form an encoder server hands back."""
    with io.BytesIO() as buf:
        torch.save(torch.zeros(4, 8), buf)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_embeds", "image_embeds": {"data": encoded}}


def _unreadable_image_part() -> dict:
    """An image reference that raises if anything tries to load it."""
    return {
        "type": "image_url",
        "image_url": {"url": "/nonexistent/only-the-context-worker-had-this.png"},
    }


MEDIA_PARTS = {
    "image_url": (_image_url_part, "multi_modal_data"),
    "image_embeds": (_image_embeds_part, "multi_modal_embeddings"),
}


def _chat_body(media_kind: str, disaggregated_params: dict | None) -> dict:
    """A relayed chat request: media still on the messages, ids alongside."""
    make_part, _ = MEDIA_PARTS[media_kind]
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image."},
                    make_part(),
                ],
            }
        ],
        "prompt_token_ids": RELAYED_TOKEN_IDS,
        "max_tokens": 4,
    }
    if disaggregated_params is not None:
        body["disaggregated_params"] = disaggregated_params
    return body


def _stub_response() -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-disagg-mm-test",
        model=MODEL,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="ok"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=len(RELAYED_TOKEN_IDS),
            completion_tokens=1,
            total_tokens=len(RELAYED_TOKEN_IDS) + 1,
        ),
    )


@pytest.fixture
def route_client():
    """`openai_chat` on a bare app, with everything past prompt-building cut.

    The server is built with `object.__new__` and hand-set attributes, so the
    handler runs for real and the engine never does.
    """
    server = object.__new__(OpenAIServer)
    captured = {}

    def generate_async(*, inputs, **kwargs):
        captured["inputs"] = inputs
        return SimpleNamespace(prompt_token_ids=RELAYED_TOKEN_IDS, finished=True)

    server.model = MODEL
    server.allow_request_chat_template = False
    server.model_config = _StubModelConfig()
    server.processor = None
    server.tokenizer = SimpleNamespace(
        tokenizer=SimpleNamespace(vocab_size=_StubModelConfig.vocab_size)
    )
    server.chat_template = None
    server.tool_parser = None
    server.tool_call_id_type = "random"
    server.multimodal_server_config = None
    # No `preprocess` attribute, so the handler skips the input-processor
    # executor; `num_postprocess_workers=0` keeps post-processing in-process.
    server.generator = SimpleNamespace(
        args=SimpleNamespace(
            gather_generation_logits=False,
            reasoning_parser=None,
            backend="pytorch",
            guided_decoding_backend=None,
            num_postprocess_workers=0,
        ),
        generate_async=generate_async,
    )
    server.await_disconnected = AsyncMock()
    server._create_chat_response = AsyncMock(return_value=_stub_response())

    app = FastAPI()
    app.add_api_route("/v1/chat/completions", server.openai_chat, methods=["POST"])
    return TestClient(app), captured


def _post(route_client, media_kind: str, disaggregated_params: dict | None):
    client, captured = route_client
    response = client.post(
        "/v1/chat/completions", json=_chat_body(media_kind, disaggregated_params)
    )
    assert response.status_code == 200, response.text
    prompt = captured["inputs"]
    assert prompt["prompt_token_ids"] == RELAYED_TOKEN_IDS
    return prompt


@pytest.mark.parametrize("media_kind", sorted(MEDIA_PARTS))
def test_generation_only_drops_relayed_media(route_client, media_kind):
    """The generation worker gets the ids and none of the media.

    Both keys, not just one: `LLM._preprocess` takes its generation-only
    branch only when neither is attached.
    """
    prompt = _post(route_client, media_kind, {"request_type": "generation_only"})

    assert "multi_modal_data" not in prompt
    assert "multi_modal_embeddings" not in prompt
    assert "mm_item_order" not in prompt


def test_generation_only_never_loads_relayed_media(route_client):
    """The media is not loaded at all, rather than loaded and then dropped.

    The reference belongs to the context worker and may be unreachable from
    here -- a node-local path, or a URL that was single-use or has expired. A
    reference that raises on load stands in for all three.
    """
    client, captured = route_client
    body = _chat_body("image_url", {"request_type": "generation_only"})
    body["messages"][0]["content"][1] = _unreadable_image_part()

    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200, response.text
    assert "multi_modal_data" not in captured["inputs"]


@pytest.mark.parametrize("media_kind", sorted(MEDIA_PARTS))
def test_context_only_keeps_media(route_client, media_kind):
    """The context worker is the one that expands, so it keeps its media."""
    _, prompt_key = MEDIA_PARTS[media_kind]
    prompt = _post(route_client, media_kind, {"request_type": "context_only"})

    assert prompt[prompt_key]


@pytest.mark.parametrize("media_kind", sorted(MEDIA_PARTS))
def test_aggregated_request_keeps_media(route_client, media_kind):
    """Pre-tokenized without disagg is an ordinary request: media stays.

    This is the control on the guard's key. It fires on the request type, not
    on the presence of `prompt_token_ids`, which aggregated clients also send.
    """
    _, prompt_key = MEDIA_PARTS[media_kind]
    prompt = _post(route_client, media_kind, None)

    assert prompt[prompt_key]

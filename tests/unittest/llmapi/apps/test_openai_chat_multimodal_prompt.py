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
"""How ``openai_chat`` hands decoded media to the engine.

A prompt the server rendered through the chat template already carries one
placeholder per media item, so it goes to the input processor marked
``chat_template_applied``; a pre-tokenized request does not.
"""

import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from tensorrt_llm.serve.openai_server import OpenAIServer

from .openai_server_stub import make_stub_openai_server

pytestmark = pytest.mark.cpu_only


def test_rendered_prompt_is_marked_as_templated():
    prompt = {"prompt": "<image> describe"}
    media = {"image": [object()]}
    OpenAIServer._attach_multimodal_data(prompt, SimpleNamespace(prompt_token_ids=None), media)
    assert prompt == {
        "prompt": "<image> describe",
        "multi_modal_data": media,
        "chat_template_applied": True,
    }


def test_pre_tokenized_prompt_is_not_marked():
    prompt = {"prompt_token_ids": [7, 11, 13]}
    media = {"image": [object()]}
    OpenAIServer._attach_multimodal_data(
        prompt, SimpleNamespace(prompt_token_ids=[7, 11, 13]), media
    )
    assert prompt == {"prompt_token_ids": [7, 11, 13], "multi_modal_data": media}


def test_item_order_is_attached_only_when_given():
    order = [{"modality": "image", "index": 0, "placeholder": "<image>"}]
    prompt = {"prompt": "<image> describe"}
    OpenAIServer._attach_multimodal_data(
        prompt, SimpleNamespace(prompt_token_ids=None), {"image": [object()]}, order
    )
    assert prompt["mm_item_order"] == order
    prompt = {"prompt": "describe"}
    OpenAIServer._attach_multimodal_data(
        prompt, SimpleNamespace(prompt_token_ids=None), {"image": [object()]}, []
    )
    assert "mm_item_order" not in prompt


class _StubModelConfig:
    model_type = "qwen3_vl"
    vocab_size = 1024


class _StubTokenizer:
    """Renders every conversation to one fixed prompt."""

    RENDERED = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>describe<|im_end|>\n"

    def __init__(self):
        self.tokenizer = self
        self.vocab_size = _StubModelConfig.vocab_size
        self.name_or_path = None
        self.template_calls = 0

    def apply_chat_template(self, *args, **kwargs) -> str:
        self.template_calls += 1
        return self.RENDERED

    def encode(self, text, **kwargs):
        return [1, 2, 3]


def test_rendered_image_request_reaches_the_engine_marked_templated(tmp_path):
    """Without ``prompt_token_ids`` the chat route renders the prompt itself.

    The engine must receive that rendered prompt, the decoded image, and
    ``chat_template_applied`` so the input processor does not render the
    template a second time.
    """
    captured = {}

    def generate_async(*, inputs, **kwargs):
        captured["inputs"] = inputs
        return SimpleNamespace(prompt_token_ids=[7, 11], finished=True)

    tokenizer = _StubTokenizer()
    server = make_stub_openai_server(
        tmp_path, "test-vlm", _StubModelConfig(), generate_async, tokenizer=tokenizer
    )
    server.await_disconnected = AsyncMock()
    server._create_chat_response = AsyncMock(
        return_value=SimpleNamespace(model_dump=lambda: {"ok": True})
    )
    app = FastAPI()
    app.add_api_route("/v1/chat/completions", server.openai_chat, methods=["POST"])

    with io.BytesIO() as buf:
        Image.new("RGB", (8, 8), (0, 128, 255)).save(buf, format="PNG")
        image_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    body = {
        "model": "test-vlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 4,
    }
    response = TestClient(app).post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text

    prompt = captured["inputs"]
    assert tokenizer.template_calls == 1
    assert prompt["prompt"] == _StubTokenizer.RENDERED
    assert "prompt_token_ids" not in prompt
    assert prompt["chat_template_applied"] is True
    assert len(prompt["multi_modal_data"]["image"]) == 1
    (item,) = prompt["mm_item_order"]
    assert (item["modality"], item["index"]) == ("image", 0)
    assert item["placeholder"] in _StubTokenizer.RENDERED

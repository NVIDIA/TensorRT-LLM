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

from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.openai_server import OpenAIServer

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

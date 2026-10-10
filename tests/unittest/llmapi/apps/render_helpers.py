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
"""Shared builders for the prompt-preparation tests: a tiny BPE tokenizer and requests."""

from __future__ import annotations

from tensorrt_llm.serve.render import RenderResources

TEMPLATE = (
    "{%- if ran is defined %}[ran:{{ ran }}]{% endif -%}"
    "{%- if tools %}[tools:{% for t in tools %}{{ t.function.name }}"
    "{{ ',' if not loop.last }}{% endfor %}]\n{% endif -%}"
    "{%- for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor -%}"
    "{%- if add_generation_prompt %}<assistant>{% endif -%}"
)
SERVER_TEMPLATE = "SERVER:{% for m in messages %}{{ m.content }}|{% endfor %}"
REQUEST_TEMPLATE = "REQUEST:{% for m in messages %}{{ m.content }}|{% endfor %}"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def make_tokenizer(chat_template: str = TEMPLATE):
    """A tiny byte-level BPE tokenizer whose post-processor adds BOS."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=340,
        special_tokens=["<s>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    corpus = [
        "hello world this is a test of the chat template for rendering prompts",
        "user assistant system tool weather city paris get_weather tools",
    ] * 30
    backend.train_from_iterator(corpus, trainer)
    backend.post_processor = TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", backend.token_to_id("<s>"))]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        chat_template=chat_template,
    )


def chat_request(**overrides):
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    fields = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "this is a test"},
            {"role": "user", "content": "get the weather"},
        ],
    }
    fields.update(overrides)
    return ChatCompletionRequest(**fields)


def resources(tokenizer, **overrides) -> RenderResources:
    fields = {"tokenizer": tokenizer, "model_type": "render-test-model"}
    fields.update(overrides)
    return RenderResources(**fields)

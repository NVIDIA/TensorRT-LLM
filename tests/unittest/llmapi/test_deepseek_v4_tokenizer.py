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

from tensorrt_llm.inputs.utils import apply_chat_template
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer


class _DummyTokenizer:
    all_special_tokens = []
    eos_token_id = 1
    pad_token_id = 0
    name_or_path = "dummy"

    def encode(self, text, *args, **kwargs):
        self.last_encoded_text = text
        return [1, 2, 3]


def test_deepseek_v4_chat_template_matches_reference_single_user_prompt():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Question: 1+1?\nAnswer:",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>Question: 1+1?\nAnswer:<｜Assistant｜></think>"
    )


def test_deepseek_v4_chat_template_tokenize_uses_rendered_prompt():
    dummy = _DummyTokenizer()
    tokenizer = DeepseekV4Tokenizer(dummy)

    token_ids = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
    )

    assert token_ids == [1, 2, 3]
    assert dummy.last_encoded_text == (
        "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
    )


def test_deepseek_v4_custom_tokenizer_reuses_loaded_wrapper():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    args = TorchLlmArgs(model="dummy", tokenizer=tokenizer, custom_tokenizer="deepseek_v4")

    assert args.tokenizer is tokenizer


def test_deepseek_v4_server_chat_template_path_uses_custom_tokenizer():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = apply_chat_template(
        model_type="deepseek_v4",
        tokenizer=tokenizer,
        processor=None,
        conversation=[
            {
                "role": "user",
                "content": "hello",
            }
        ],
        add_generation_prompt=True,
        mm_placeholder_counts=[{}],
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>")

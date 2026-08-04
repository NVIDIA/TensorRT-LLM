# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def make_context(
    tokenizer,
    query,
    history,
    system,
    max_input_length,
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return (f"{role}\n{content}",
                    tokenizer.encode(
                        role,
                        allowed_special=set(),
                    ) + nl_tokens + tokenizer.encode(
                        content,
                        allowed_special=set(),
                    ))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens

            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response)
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens
            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (len(system_tokens) +
                                    len(next_context_tokens) +
                                    len(context_tokens))
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (nl_tokens + im_start_tokens +
                           _tokenize_str("user", query)[1] + im_end_tokens +
                           nl_tokens + im_start_tokens +
                           tokenizer.encode("assistant") + nl_tokens)
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    # truncate to max_input_length, truncate from the front
    return raw_text, context_tokens[-max_input_length:]


def get_qwen_key_list(qwen_type):
    qwen_key_list = [
        "attn.c_attn",  # attention.qkv
        "attn.c_proj",  # attention.dense
        "mlp.w1",  # mlp.gate
        "mlp.w2",  # mlp.fc
        "mlp.c_proj",  # mlp.proj
        "ln_1",  # input_layernorm
        "ln_2",  # post_layernorm
        "transformer.wte",  # vocabulary embedding
        "transformer.ln_f",  # final layer norm
    ]
    qwen2_key_list = [
        "self_attn.",  # attention.qkv
        "self_attn.o_proj",  # attention.dense
        "mlp.up_proj",  # mlp.gate
        "mlp.gate_proj",  # mlp.fc
        "mlp.down_proj",  # mlp.proj
        "input_layernorm",  # input_layernorm
        "post_attention_layernorm",  # post_layernorm
        "model.embed_tokens",  # vocabulary embedding
        "model.norm",  # final layer norm
    ]
    key_list = []
    if qwen_type == 'qwen':
        key_list.extend(qwen_key_list)
    else:
        key_list.extend(qwen2_key_list)
    return key_list

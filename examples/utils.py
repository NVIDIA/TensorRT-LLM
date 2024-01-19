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

import json
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, T5Tokenizer

import tensorrt_llm

DEFAULT_HF_MODEL_DIRS = {
    'baichuan': 'baichuan-inc/Baichuan-13B-Chat',
    'bloom': 'bigscience/bloom-560m',
    'chatglm_6b': 'THUDM/chatglm-6b',
    'chatglm2_6b': 'THUDM/chatglm2-6b',
    'chatglm2_6b_32k': 'THUDM/chatglm2-6b-32k',
    'chatglm3_6b': 'THUDM/chatglm3-6b',
    'chatglm3_6b_base': 'THUDM/chatglm3-6b-base',
    'chatglm3_6b_32k': 'THUDM/chatglm3-6b-32k',
    'falcon': 'tiiuae/falcon-rw-1b',
    'glm_10b': 'THUDM/glm-10b',
    'gpt': 'gpt2-medium',
    'gptj': 'EleutherAI/gpt-j-6b',
    'gptneox': 'EleutherAI/gpt-neox-20b',
    'internlm': 'internlm/internlm-chat-7b',
    'llama': 'meta-llama/Llama-2-7b-hf',
    'mpt': 'mosaicml/mpt-7b',
    'phi': 'microsoft/phi-2',
    'opt': 'facebook/opt-350m',
    'qwen': 'Qwen/Qwen-7B',
}

DEFAULT_PROMPT_TEMPLATES = {
    'internlm':
    "<|User|>:{input_text}<eoh>\n<|Bot|>:",
    'qwen':
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
}


def read_model_name(engine_dir: str):
    engine_version = tensorrt_llm.runtime.engine.get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)

    if engine_version is None:
        return config['builder_config']['name']

    return config['pretrained_config']['architecture']


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


def load_tokenizer(tokenizer_dir: Optional[str] = None,
                   vocab_file: Optional[str] = None,
                   model_name: str = 'gpt',
                   tokenizer_type: Optional[str] = None):
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                  legacy=False,
                                                  padding_side='left',
                                                  truncation_side='left',
                                                  trust_remote_code=True,
                                                  tokenizer_type=tokenizer_type,
                                                  use_fast=use_fast)
    else:
        # For gpt-next, directly load from tokenizer.model
        assert model_name == 'gpt'
        tokenizer = T5Tokenizer(vocab_file=vocab_file,
                                padding_side='left',
                                truncation_side='left')

    if model_name == 'qwen':
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        chat_format = gen_config['chat_format']
        if chat_format == 'raw':
            pad_id = gen_config['pad_token_id']
            end_id = gen_config['eos_token_id']
        elif chat_format == 'chatml':
            pad_id = tokenizer.im_end_id
            end_id = tokenizer.im_end_id
        else:
            raise Exception(f"unknown chat format: {chat_format}")
    elif model_name == 'glm_10b':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id

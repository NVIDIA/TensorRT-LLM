# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from .baichuan.model import BaichuanForCausalLM
from .bert.model import BertForQuestionAnswering, BertModel
from .bloom.model import BloomForCausalLM, BloomModel
from .chatglm.model import ChatGLMHeadModel, ChatGLMModel
from .falcon.model import FalconForCausalLM, FalconModel
from .gpt.model import GPTLMHeadModel, GPTModel
from .gptj.model import GPTJForCausalLM, GPTJModel
from .gptneox.model import GPTNeoXForCausalLM, GPTNeoXModel
from .internlm.model import InternLMForCausalLM, InternLMModel
from .llama.model import LLaMAForCausalLM, LLaMAModel
from .opt.model import OPTLMHeadModel, OPTModel
from .quantized.quant import quantize_model  # noqa

__all__ = [
    'BertModel',
    'BertForQuestionAnswering',
    'BloomModel',
    'BloomForCausalLM',
    'FalconForCausalLM',
    'FalconModel',
    'GPTModel',
    'GPTLMHeadModel',
    'OPTLMHeadModel',
    'OPTModel',
    'LLaMAForCausalLM',
    'LLaMAModel',
    'GPTJModel',
    'GPTJForCausalLM',
    'GPTNeoXModel',
    'GPTNeoXForCausalLM',
    'quantize_model',
    'ChatGLMHeadModel',
    'ChatGLMModel',
    'BaichuanForCausalLM',
    'InternLMForCausalLM',
    'InternLMModel',
]

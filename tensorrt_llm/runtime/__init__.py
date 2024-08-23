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
from .enc_dec_model_runner import EncDecModelRunner
from .generation import SamplingConfig  # autoflake: skip
from .generation import (ChatGLMGenerationSession, GenerationSession,
                         LogitsProcessor, LogitsProcessorList, ModelConfig,
                         QWenForCausalLMGenerationSession, StoppingCriteria,
                         StoppingCriteriaList, decode_words_list)
from .kv_cache_manager import GenerationSequence, KVCacheManager
from .model_runner import ModelRunner
from .multimodal_model_runner import MultimodalModelRunner
from .session import Session, TensorInfo

try:
    import tensorrt_llm.bindings  # NOQA
    PYTHON_BINDINGS = True
except ImportError:
    PYTHON_BINDINGS = False

if PYTHON_BINDINGS:
    from .model_runner_cpp import ModelRunnerCpp

__all__ = [
    'ModelConfig',
    'GenerationSession',
    'GenerationSequence',
    'KVCacheManager',
    'SamplingConfig',
    'Session',
    'TensorInfo',
    'ChatGLMGenerationSession',
    'QWenForCausalLMGenerationSession',
    'decode_words_list',
    'LogitsProcessorList',
    'LogitsProcessor',
    'StoppingCriteriaList',
    'StoppingCriteria',
    'ModelRunner',
    'ModelRunnerCpp',
    'EncDecModelRunner',
    'MultimodalModelRunner',
    'PYTHON_BINDINGS',
]

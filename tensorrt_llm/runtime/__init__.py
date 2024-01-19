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
from .generation import SamplingConfig  # autoflake: skip
from .generation import (ChatGLMGenerationSession, GenerationSession,
                         LogitsProcessor, LogitsProcessorList, ModelConfig,
                         QWenForCausalLMGenerationSession, StoppingCriteria,
                         StoppingCriteriaList, to_word_list_format)
from .kv_cache_manager import GenerationSequence, KVCacheManager
from .lora_manager import LoraManager  # autoflake: skip
from .model_runner import ModelRunner
from .session import Session, TensorInfo

try:
    from .model_runner_cpp import ModelRunnerCpp
    PYTHON_BINDINGS = True
except ImportError:
    PYTHON_BINDINGS = False

__all__ = [
    'ModelConfig',
    'GenerationSession',
    'GenerationSequence',
    'KVCacheManager',
    'LoraManager'
    'SamplingConfig',
    'Session',
    'TensorInfo',
    'ChatGLMGenerationSession',
    'QWenForCausalLMGenerationSession',
    'to_word_list_format',
    'LogitsProcessorList',
    'LogitsProcessor',
    'StoppingCriteriaList',
    'StoppingCriteria',
    'ModelRunner',
    'ModelRunnerCpp',
    'PYTHON_BINDINGS',
]

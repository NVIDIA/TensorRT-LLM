# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Visual Generation Attention Backend

This module provides attention backend infrastructure for visual generation (diffusion) models.
It reuses existing TRT-LLM attention backends (TrtllmAttention, VanillaAttention) with
simplified metadata that doesn't require KV caching.
"""

from .interface import AttentionTensorLayout
from .parallel import UlyssesAttention
from .trtllm import TrtllmAttention, TrtllmAttentionMetadata
from .utils import create_attention, get_visual_gen_attention_backend
from .vanilla import VanillaAttention

__all__ = [
    "AttentionTensorLayout",
    "get_visual_gen_attention_backend",
    "create_attention",
    "TrtllmAttention",
    "TrtllmAttentionMetadata",
    "UlyssesAttention",
    "VanillaAttention",
]

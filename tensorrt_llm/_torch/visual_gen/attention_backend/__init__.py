# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .cute_dsl import (
    VSA_TILE_SIZE,
    CuTeDSLAttention,
    VSAAttention,
    VSAMetadata,
    VSAMetadataBuilder,
    get_vsa_forward_context,
    set_vsa_forward_context,
)
from .flash_attn4 import FlashAttn4Attention
from .interface import AttentionBackend, AttentionTensorLayout
from .parallel import Attention2DAttention, RingAttention, UlyssesAttention, wrap_parallel_attention
from .trtllm import TrtllmAttention, TrtllmAttentionMetadata
from .utils import create_attention, get_visual_gen_attention_backend
from .vanilla import VanillaAttention

__all__ = [
    "AttentionBackend",
    "Attention2DAttention",
    "AttentionTensorLayout",
    "get_visual_gen_attention_backend",
    "create_attention",
    "CuTeDSLAttention",
    "VSAAttention",
    "FlashAttn4Attention",
    "TrtllmAttention",
    "TrtllmAttentionMetadata",
    "UlyssesAttention",
    "VanillaAttention",
    "RingAttention",
    "wrap_parallel_attention",
    "VSAMetadata",
    "VSAMetadataBuilder",
    "VSA_TILE_SIZE",
    "get_vsa_forward_context",
    "set_vsa_forward_context",
]

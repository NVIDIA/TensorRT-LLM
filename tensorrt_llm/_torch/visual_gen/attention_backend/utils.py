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
Visual Generation Attention Backend Utilities

Factory functions for creating attention backends for visual generation models.
Uses diffusion-specific wrappers (TrtllmAttention, VanillaAttention)
that handle metadata preparation internally for simplified usage.
"""

from typing import TYPE_CHECKING, Optional, Type, Union

import torch

from tensorrt_llm.models.modeling_utils import QuantConfig

# Lazy imports to avoid circular dependency
if TYPE_CHECKING:
    from .trtllm import TrtllmAttention
    from .vanilla import VanillaAttention

    # Type alias for diffusion attention backends
    DiffusionAttentionBackend = Union[TrtllmAttention, VanillaAttention]


def get_visual_gen_attention_backend(
    backend_name: str,
) -> Type["DiffusionAttentionBackend"]:
    """
    Get diffusion attention backend class by name.

    Args:
        backend_name: Backend identifier ("VANILLA", "TRTLLM")

    Returns:
        Diffusion attention backend class

    Backend Selection Guide:
        - "VANILLA": Full support for cross-attention (different Q/KV seq lengths)
                     Uses torch SDPA backend
        - "TRTLLM": Optimized for self-attention (requires same Q/KV seq lengths)
                    Better performance but requires fused QKV
    """
    # Lazy imports to avoid circular dependency
    from .trtllm import TrtllmAttention
    from .vanilla import VanillaAttention

    backend_name = backend_name.upper()

    if backend_name == "VANILLA":
        return VanillaAttention
    elif backend_name == "TRTLLM":
        return TrtllmAttention
    else:
        # Default to VANILLA for maximum compatibility
        return VanillaAttention


def create_attention(
    backend: str,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: Optional[int] = None,
    quant_config: Optional[QuantConfig] = None,
    dtype: Optional[torch.dtype] = None,
    max_batch_size: int = 16,
    max_seq_len: int = 4096,
    **kwargs,
) -> "DiffusionAttentionBackend":
    """
    Factory function to create attention backend instance for visual generation.

    Creates diffusion-specific attention backends that handle metadata preparation
    internally, simplifying the forward() call.

    Args:
        backend: Backend identifier ("VANILLA", "TRTLLM")
        layer_idx: Layer index in the model
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_kv_heads: Number of KV heads (for GQA/MQA, defaults to num_heads)
        quant_config: Optional quantization configuration
        dtype: Data type for the attention
        max_batch_size: Initial batch size for metadata pre-allocation. The backend
            will automatically reallocate if larger batches are encountered.
        max_seq_len: Initial sequence length for metadata pre-allocation. The backend
            will automatically reallocate if longer sequences are encountered.
        **kwargs: Additional backend-specific arguments

    Returns:
        Diffusion attention backend instance (TrtllmAttention or VanillaAttention)
    """
    attn_cls = get_visual_gen_attention_backend(backend)

    return attn_cls(
        layer_idx=layer_idx,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        quant_config=quant_config,
        dtype=dtype,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        **kwargs,
    )

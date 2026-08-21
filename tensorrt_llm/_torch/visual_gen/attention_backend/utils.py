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
Visual Generation Attention Backend Utilities

Factory functions for creating attention backends for visual generation models.
Uses diffusion-specific wrappers (TrtllmAttention, VanillaAttention)
that handle metadata preparation internally for simplified usage.
"""

from typing import Optional, Type

import torch

from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig

from .interface import AttentionBackend


def get_visual_gen_attention_backend(
    backend_name: str,
) -> Type[AttentionBackend]:
    """
    Get diffusion attention backend class by name.

    Args:
        backend_name: Backend identifier ("VANILLA", "TRTLLM", "FA4", "CUTEDSL")

    Returns:
        Diffusion attention backend class

    Backend Selection Guide:
        - "VANILLA": Full support for cross-attention (different Q/KV seq lengths)
                     Uses torch SDPA backend
        - "TRTLLM": Optimized for self-attention (requires same Q/KV seq lengths).
        - "FA4": Flash Attention 4; provides higher speedup on Blackwell GPUs (sm100)
                 Requires flash-attn package with cute interface
        - "CUTEDSL": CuTe DSL kernels. create_attention selects dense FMHA or CuTe VSA
                      from AttentionConfig.sparse_attention_config.
    """

    backend_name = backend_name.upper()

    if backend_name == "VANILLA":
        from .vanilla import VanillaAttention

        return VanillaAttention
    elif backend_name == "TRTLLM":
        from .trtllm import TrtllmAttention

        return TrtllmAttention
    elif backend_name == "FA4":
        from .flash_attn4 import FlashAttn4Attention

        return FlashAttn4Attention
    elif backend_name == "CUTEDSL":
        from .cute_dsl import CuTeDSLAttention

        return CuTeDSLAttention
    else:
        # Default to VANILLA for maximum compatibility
        from .vanilla import VanillaAttention

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
    attention_config: Optional[AttentionConfig] = None,
    attention_metadata_state: Optional[dict] = None,
    **kwargs,
) -> AttentionBackend:
    """
    Factory function to create attention backend instance for visual generation.

    Creates diffusion-specific attention backends that handle metadata preparation
    internally, simplifying the forward() call.

    Args:
        backend: Backend identifier ("VANILLA", "TRTLLM", "FA4", "CUTEDSL")
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
        attention_config: Optional AttentionConfig used to select the attention algorithm and
            forward its quantization or sparsity configuration.
        attention_metadata_state: Optional per-component VisualGen TRTLLM state.
            It keeps shape-stable metadata and backend runtime resources alive
            across layers and CUDA Graph captures. Generic block-sparse FMHA
            uses it to share one PrimTS plan cache across serialized layers.
        **kwargs: Additional backend-specific arguments

    Returns:
        AttentionBackend instance
    """
    sparse_attention_config = (
        attention_config.sparse_attention_config if attention_config is not None else None
    )
    is_vsa = (
        sparse_attention_config is not None
        and getattr(sparse_attention_config, "algorithm", None) == "vsa"
    )

    backend_name = backend.upper()
    if is_vsa and backend_name == "CUTEDSL":
        from .sparse.vsa.cute_dsl import CuTeDSLVSAAdapter

        attn_cls = CuTeDSLVSAAdapter
    elif is_vsa and backend_name == "TRTLLM":
        from .sparse.vsa.trtllm import TrtllmVSAAdapter

        attn_cls = TrtllmVSAAdapter
    else:
        attn_cls = get_visual_gen_attention_backend(backend)

    # Forward the validated quantization recipe to TRTLLM or the dense CuTe DSL FMHA backend.
    if attention_config is not None and attention_config.quant_attention_config is not None:
        kwargs["quant_attention_config"] = attention_config.quant_attention_config
    if backend_name == "TRTLLM":
        if attention_metadata_state is None:
            raise ValueError(
                "TRTLLM backend requires `attention_metadata_state` from "
                "DiffusionModelConfig; creation path must not allocate metadata implicitly."
            )
        kwargs["attention_metadata_state"] = attention_metadata_state
    if is_vsa and backend_name in ("CUTEDSL", "TRTLLM"):
        kwargs["sparse_attention_config"] = sparse_attention_config

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

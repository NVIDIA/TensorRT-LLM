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

from typing import NamedTuple, Optional, Type

import torch

from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

from .interface import AttentionBackend


def get_visual_gen_attention_backend(
    backend_name: str,
    sparse_algorithm: Optional[str] = None,
    quant_attention_config: Optional[QuantAttentionConfig] = None,
) -> Type[AttentionBackend]:
    """
    Get diffusion attention backend class by name, sparse algorithm and quantization recipe.

    Args:
        backend_name: Backend identifier ("VANILLA", "TRTLLM", "CUDNN", "FLASHINFER", "CUTEDSL", "FA4")
        sparse_algorithm: ``algorithm`` of the sparse attention config, if any. VSA
            ("vsa") and SOL ("sol_attn") have their own TRTLLM and CUTEDSL classes;
            other algorithms, such as skip-softmax, run in the dense class.
        quant_attention_config: The quantized attention config, if any. A recipe with
            ``deco_attention`` has its own CUTEDSL class; the other recipes run in the class
            of their backend.

    Returns:
        Diffusion attention backend class

    Backend Selection Guide:
        - "VANILLA": Full support for cross-attention (different Q/KV seq lengths)
                     Uses torch SDPA backend
        - "TRTLLM": Optimized for self-attention (requires same Q/KV seq lengths).
        - "FLASHINFER": Dense prefill attention without a KV cache. Supports FP16/BF16
                        and architecture-specific NVFP4 attention recipes.
        - "FA4": Flash Attention 4; provides higher speedup on Blackwell GPUs (sm100)
                 Requires flash-attn package with cute interface
        - "CUTEDSL": CuTe DSL kernels: dense/SkipSoftmax FMHA, VSA or SOL depending on
                     ``sparse_algorithm``, or DeCo-Attention.
        - "CUDNN": cuDNN fused SDPA. Unquantized by default; quant_attention_config
                   selects per-tensor FP8 or block-scaled MXFP8 (Blackwell).
    """

    backend_name = backend_name.upper()
    is_deco = quant_attention_config is not None and quant_attention_config.deco_attention

    if sparse_algorithm == "vsa" and backend_name == "TRTLLM":
        from .sparse.vsa.backend import VSATrtllmAttention

        return VSATrtllmAttention
    elif sparse_algorithm == "vsa" and backend_name == "CUTEDSL":
        from .sparse.vsa.backend import VSACuTeDSLAttention

        return VSACuTeDSLAttention
    elif sparse_algorithm == "sol_attn" and backend_name == "TRTLLM":
        from .sparse.sol.backend import SOLTrtllmAttention

        return SOLTrtllmAttention
    elif sparse_algorithm == "sol_attn" and backend_name == "CUTEDSL":
        from .sparse.sol.backend import SOLCuTeDSLAttention

        return SOLCuTeDSLAttention
    elif is_deco and backend_name == "CUTEDSL":
        from .cute_dsl import DeCoAttention

        return DeCoAttention
    elif backend_name == "VANILLA":
        from .vanilla import VanillaAttention

        return VanillaAttention
    elif backend_name == "TRTLLM":
        from .trtllm import TrtllmAttention

        return TrtllmAttention
    elif backend_name == "FLASHINFER":
        from .flashinfer import FlashInferAttention

        return FlashInferAttention
    elif backend_name == "FA4":
        from .flash_attn4 import FlashAttn4Attention

        return FlashAttn4Attention
    elif backend_name == "CUTEDSL":
        from .cute_dsl import CuTeDSLAttention

        return CuTeDSLAttention
    elif backend_name == "CUDNN":
        from .cudnn import CuDNNAttention

        return CuDNNAttention
    else:
        # Default to VANILLA for maximum compatibility
        from .vanilla import VanillaAttention

        return VanillaAttention


class PaddingSupport(NamedTuple):
    """What the dense attention backend of a config takes for the padding of a sequence.

    Attributes:
        supports_mask: The backend takes ``key_padding_mask``.
        supports_length: The backend takes ``valid_seq_len``.
        requires_trailing: The padding must follow the tokens; also true for a backend that
            takes only the length.
    """

    supports_mask: bool
    supports_length: bool
    requires_trailing: bool

    @property
    def supports_padding(self) -> bool:
        """Whether the backend can attend a padded sequence at all."""
        return self.supports_mask or self.supports_length


def get_padding_support(attention_config: AttentionConfig) -> PaddingSupport:
    """Return what the dense attention backend of a config takes for the padding of a sequence.

    A model hands a backend that takes a mask the mask, and one that takes only the length the
    length; a backend that takes neither cannot attend a padded sequence.
    """
    # A config object without the field carries no quantization recipe.
    quant_attention_config = getattr(attention_config, "quant_attention_config", None)
    backend_cls = get_visual_gen_attention_backend(
        attention_config.backend, quant_attention_config=quant_attention_config
    )
    supports_mask = backend_cls.support_key_padding_mask()
    supports_length = backend_cls.support_valid_seq_len()
    # The quantized recipes of these backends run on kernels without padding: FLASHINFER rejects
    # a mask and TRTLLM the valid prefix.
    if quant_attention_config is not None:
        backend = attention_config.backend.upper()
        supports_mask = supports_mask and backend != "FLASHINFER"
        supports_length = supports_length and backend != "TRTLLM"
    requires_trailing = backend_cls.requires_trailing_padding() or (
        supports_length and not supports_mask
    )
    return PaddingSupport(supports_mask, supports_length, requires_trailing)


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
        backend: Backend identifier ("VANILLA", "TRTLLM", "CUDNN", "FLASHINFER", "CUTEDSL", "FA4")
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
        attention_metadata_state: Optional per-component VisualGen attention state.
            It keeps shape-stable attention metadata alive across layers and
            CUDA Graph captures. Required for TRTLLM and shared by FlashInfer layers.
        **kwargs: Additional backend-specific arguments

    Returns:
        AttentionBackend instance
    """
    sparse_config = (
        attention_config.sparse_attention_config if attention_config is not None else None
    )
    sparse_algorithm = sparse_config.algorithm if sparse_config is not None else None
    is_vsa = sparse_algorithm == "vsa"
    is_sol = sparse_algorithm == "sol_attn"

    quant_attention_config = (
        attention_config.quant_attention_config if attention_config is not None else None
    )

    backend_name = backend.upper()
    attn_cls = get_visual_gen_attention_backend(
        backend_name, sparse_algorithm, quant_attention_config
    )

    if is_vsa:
        sparse_params = kwargs.pop("sparse_params", None)
        if sparse_params is not None:
            raise ValueError("VSA does not lower through core SparseParams.")
    elif is_sol and kwargs.get("sparse_params") is None:
        # The attention module lowers the config once per layer; callers that
        # construct a backend directly get the same lowering here.
        kwargs["sparse_params"] = sparse_config.to_sparse_params()

    # Forward the validated quantization recipe to TRTLLM, cuDNN, FlashInfer, the dense CuTe DSL
    # FMHA backend, or DeCo-Attention.
    if quant_attention_config is not None:
        kwargs["quant_attention_config"] = quant_attention_config
    if backend_name == "TRTLLM":
        if attention_metadata_state is None:
            raise ValueError(
                "TRTLLM backend requires `attention_metadata_state` from "
                "DiffusionModelConfig; creation path must not allocate metadata implicitly."
            )
        kwargs["attention_metadata_state"] = attention_metadata_state
    elif backend_name == "FLASHINFER":
        if attention_metadata_state is None:
            raise ValueError(
                "FLASHINFER backend requires `attention_metadata_state` from "
                "DiffusionModelConfig for shared workspace allocation."
            )
        kwargs["attention_metadata_state"] = attention_metadata_state
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Hugging Face attention-interface adapter for the native RDNA4 HIP kernel."""

import importlib.util
import threading

import torch
from transformers import AttentionInterface, PretrainedConfig

from trtllm_profile import component

from .ops import attention
from .runtime import KernelBackend

_NATIVE_MODELS = frozenset(("llama", "mistral", "qwen2", "qwen3"))
_REGISTER_LOCK = threading.Lock()
_REGISTERED = False


def attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    sliding_window: int | None = None,
    backend: KernelBackend = "hip",
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Return [batch, query_seq, heads, dim] output and no materialized attention map.

    A 4D HF mask already encodes causal, padding, and sliding-window rules. A
    2D mask is a padding mask and needs an explicit causal rule. This distinction
    is essential for cached decode and for preallocated/static KV layouts.
    """
    if module.training or dropout != 0:
        raise ValueError("Native RDNA4 attention is inference-only and does not implement dropout")
    if kwargs.get("output_attentions"):
        raise NotImplementedError("Native RDNA4 attention does not materialize attention weights")
    if kwargs.get("softcap") not in (None, 0):
        raise NotImplementedError(
            "Logit soft-capping is not implemented by the native attention kernel"
        )
    if attention_mask is not None:
        attention_mask = attention_mask[..., : key.shape[-2]]
        if attention_mask.ndim == 2:
            mask = attention_mask[:, None, None, :].bool()
            needs_causal = bool(getattr(module, "is_causal", True))
        elif attention_mask.ndim == 4:
            mask = attention_mask
            needs_causal = False
        else:
            raise ValueError("HF attention mask must be a 2D padding mask or 4D logits mask")
    else:
        mask = None
        needs_causal = bool(getattr(module, "is_causal", True))
    if sliding_window is not None and (attention_mask is None or attention_mask.ndim == 2):
        if sliding_window < 1:
            raise ValueError("sliding_window must be positive")
        positions = (
            torch.arange(query.shape[-2], device=query.device) + key.shape[-2] - query.shape[-2]
        )
        allowed = torch.arange(key.shape[-2], device=query.device)[None, :] > (
            positions[:, None] - sliding_window
        )
        mask = allowed if mask is None else mask & allowed
    with component("attention"):
        output = attention(
            query, key, value, mask=mask, causal=needs_causal, scale=scaling, backend=backend
        )
    return output.transpose(1, 2).contiguous(), None


def configure_native_attention(config: PretrainedConfig) -> str:
    """Register an attention and mask implementation for compatible decoder models."""
    global _REGISTERED
    if config.model_type not in _NATIVE_MODELS:
        raise NotImplementedError(
            "Native HIP attention is enabled only for Llama, Mistral, Qwen2 and Qwen3; "
            "use SDPA/eager for other architectures"
        )
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if dim > 256:
        raise ValueError("Native HIP attention supports head dimensions <= 256; use SDPA")
    with _REGISTER_LOCK:
        if not _REGISTERED:
            AttentionInterface.register("rdna4_hip", attention_forward)
            # Newer Transformers suppress masks for unknown attention backends.
            # Register eager mask construction as well, so padding/causality are
            # not silently lost when using the native HIP callback.
            if importlib.util.find_spec("transformers.masking_utils") is not None:
                from transformers.masking_utils import (
                    ALL_MASK_ATTENTION_FUNCTIONS,
                    AttentionMaskInterface,
                )

                AttentionMaskInterface.register("rdna4_hip", ALL_MASK_ATTENTION_FUNCTIONS["eager"])
            _REGISTERED = True
    return "rdna4_hip"


__all__ = ["attention_forward", "configure_native_attention"]

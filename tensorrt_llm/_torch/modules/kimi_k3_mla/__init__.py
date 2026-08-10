# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA in-tree module for TensorRT-LLM's PyTorch backend.

K3 MLA is DeepSeek-V3-style multi-latent attention with three K3-specific
deltas that live at the module level (not the attention backend):

* **NoPE.** ``mla_use_nope=True`` in K3 config disables the rotary
  embedding; both the query and key rope slots pass through the backend
  unchanged.
* **Output gate before ``o_proj``.** When ``mla_use_output_gate=True`` an
  extra ``g_proj`` computes ``sigmoid(g_proj(hidden_states)) * attn_output``
  before the final projection.
* **Softmax scale.** ``(qk_nope + qk_rope) ** -0.5 = 192 ** -0.5`` for
  real K3 dims — matches ``TrtllmAttention`` default MLA q_scaling.

The module wraps the existing ``TrtllmAttention`` backend MLA path plus
``KVCacheManagerV2`` for both context and cached-decode.
"""

from .kimi_k3_mla_attention import KimiK3MLAAttention

__all__ = [
    "KimiK3MLAAttention",
]

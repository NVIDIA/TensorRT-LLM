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
Diffusion Vanilla Attention Backend

Simple attention implementation for visual generation (diffusion) models using
torch.nn.functional.scaled_dot_product_attention (SDPA).

Supports both self-attention and cross-attention (different Q/KV sequence lengths).
No KV cache - full recompute each diffusion step.
"""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionTensorLayout


class VanillaAttention(nn.Module):
    """
    Vanilla Attention for diffusion models using torch SDPA.

    Uses torch.nn.functional.scaled_dot_product_attention which:
    - Properly handles cross-attention (different Q/KV sequence lengths)
    - Uses Flash Attention 2 when available (via SDPA backend selection)
    - No KV cache needed for diffusion models

    This is simpler than the LLM VanillaAttention which has complex
    KV cache handling and uses flash_attn_varlen_func.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.scale = 1.0 / math.sqrt(head_dim)

        # SDPA expects [B, H, S, D] format
        self._preferred_layout = AttentionTensorLayout.HND

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: Union[int, torch.Tensor],
        seq_len_kv: Optional[int] = None,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass using torch SDPA.

        Args:
            q: Query tensor [num_tokens, num_heads * head_dim]
            k: Key tensor [num_kv_tokens, num_kv_heads * head_dim]
            v: Value tensor [num_kv_tokens, num_kv_heads * head_dim]
            batch_size: Batch size
            seq_len: Query sequence length
            seq_len_kv: KV sequence length (for cross-attention)
            attention_mask: Attention mask type (CAUSAL or FULL)

        Returns:
            Output tensor [num_tokens, num_heads * head_dim]
        """
        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL

        # Validate tensor shapes - flexible for Ulysses head sharding
        # Expected: [batch_size, num_heads, seq_len, head_dim]
        # Note: num_heads may be sharded (num_heads // ulysses_size) when using Ulysses
        assert (
            q.dim() == 4
            and q.shape[0] == batch_size
            and q.shape[2] == seq_len
            and q.shape[3] == self.head_dim
        ), (
            f"Invalid q shape: expected [B={batch_size}, H, S={seq_len}, D={self.head_dim}], got {q.shape}"
        )
        assert k.dim() == 4 and k.shape[0] == batch_size and k.shape[3] == self.head_dim, (
            f"Invalid k shape: expected [B={batch_size}, H_kv, S_kv, D={self.head_dim}], got {k.shape}"
        )
        assert v.dim() == 4 and v.shape[0] == batch_size and v.shape[3] == self.head_dim, (
            f"Invalid v shape: expected [B={batch_size}, H_kv, S_kv, D={self.head_dim}], got {v.shape}"
        )

        # TODO: Maybe we need to enforce cuDNN backend here
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=self.scale)

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

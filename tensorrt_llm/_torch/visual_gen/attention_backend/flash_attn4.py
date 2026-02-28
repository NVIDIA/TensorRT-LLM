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
"""
Flash Attention 4 Backend for Visual Generation Models

Uses flash_attn.cute.interface._flash_attn_fwd for optimized attention.
Expects NHD layout ([B, S, H, D]) and supports float16/bfloat16.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionTensorLayout

try:
    from flash_attn.cute.interface import _flash_attn_fwd
except Exception:
    _flash_attn_fwd = None


class FlashAttn4Attention(nn.Module):
    """
    Flash Attention 4 backend for diffusion models.

    Uses flash_attn.cute.interface._flash_attn_fwd which:
    - Expects [B, S, H, D] (NHD) format
    - Supports float16 and bfloat16 (auto-casts other dtypes)
    - Supports both self-attention and cross-attention (different Q/KV lengths)
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

        # FA4 expects [B, S, H, D] format
        self._preferred_layout = AttentionTensorLayout.NHD

    @torch.compiler.disable
    def _fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """Calls _flash_attn_fwd with torch.compile disabled."""
        output, _lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=self.scale,
            causal=causal,
            window_size_left=None,
            window_size_right=None,
            learnable_sink=None,
            softcap=0.0,
            pack_gqa=None,
            mask_mod=None,
            block_sparse_tensors=None,
            return_lse=True,
        )
        return output

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
        seq_len_kv: Optional[int] = None,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass using Flash Attention 4.

        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor [batch_size, seq_len_kv, num_kv_heads, head_dim]
            v: Value tensor [batch_size, seq_len_kv, num_kv_heads, head_dim]
            batch_size: Batch size
            seq_len: Query sequence length
            seq_len_kv: KV sequence length (may differ from seq_len for cross-attention)
            attention_mask: Attention mask type (CAUSAL or FULL)

        Returns:
            Output tensor [batch_size, seq_len, num_heads, head_dim]
        """
        if _flash_attn_fwd is None:
            raise ImportError(
                "FlashAttention 4 is not available. "
                "Clone the source and export PYTHONPATH as described in the prerequisites."
            )

        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL

        # FA4 only supports float16 and bfloat16
        origin_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        output = self._fwd(q, k, v, is_causal)

        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)

        return output

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

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
Diffusion TransformerEngine FP8 Attention Backend

FP8 self/cross-attention for visual generation (diffusion) models via
TransformerEngine's ``DotProductAttention`` under ``fp8_autocast`` with a
``DelayedScaling(fp8_dpa=True, fp8_mha=True)`` recipe. Ported from the vfly
``te-fp8`` reference (3rdparty/vfly/vfly/ops/attention.py).

Declares NHD layout ([B, S, H, D]) which maps directly to TE's
``qkv_format="bshd"`` — avoids the bhsd->bshd transpose. The op is
``torch.compiler.disable``-d because TE FP8 modules graph-break under
torch.compile (matches vfly).
"""

import math
from typing import Optional

import torch

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

try:
    from transformer_engine.common.recipe import DelayedScaling
    from transformer_engine.pytorch import DotProductAttention, fp8_autocast
except ImportError:  # TE absent (e.g. client-only env); fail at construction, not import.
    DotProductAttention = None
    DelayedScaling = None
    fp8_autocast = None


class TEAttention(AttentionBackend):
    """FP8 attention via TransformerEngine ``DotProductAttention``.

    No KV cache (diffusion: full recompute each step). FP8 is always enabled —
    this backend exists specifically to get FP8 attention; for BF16 use VANILLA.
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
        if DotProductAttention is None:
            raise ImportError(
                "TransformerEngine is required for the TE attention backend "
                "(transformer_engine.pytorch.DotProductAttention not importable)."
            )
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.scale = 1.0 / math.sqrt(head_dim)
        # NHD == [B, S, H, D] == TE qkv_format="bshd" (no transpose needed).
        self._preferred_layout = AttentionTensorLayout.NHD
        self.recipe = DelayedScaling(fp8_dpa=True, fp8_mha=True)
        # DotProductAttention is stateful (amax history); rebuild only when the
        # (heads, dim, gqa, mask) traits change — mirrors vfly's lazy init.
        self._attn_op = None
        self._traits = None

    def _lazy_init(self, num_gqa_groups: Optional[int], attn_mask_type: str) -> None:
        traits = (self.num_heads, self.head_dim, num_gqa_groups, attn_mask_type)
        if traits != self._traits:
            self._attn_op = DotProductAttention(
                self.num_heads,
                self.head_dim,
                num_gqa_groups=num_gqa_groups,
                attn_mask_type=attn_mask_type,
                softmax_scale=self.scale,
                qkv_format="bshd",
            )
            self._traits = traits

    @torch.compiler.disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """FP8 attention. q/k/v are NHD ([B, S, H, D]); returns [B, S, H, D]."""
        assert q.dim() == 4 and q.shape[-1] == self.head_dim, (
            f"Invalid q shape: expected [B, S, H, D={self.head_dim}], got {q.shape}"
        )
        if key_padding_mask is not None:
            # The Wan T2V forward (full self-attn + fixed-length cross-attn) does
            # not pass a padding mask; defer this rather than risk wrong masking.
            raise NotImplementedError(
                "TE attention backend does not yet support key_padding_mask."
            )

        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL
        enable_gqa = self.num_heads != self.num_kv_heads
        num_gqa_groups = k.shape[-2] if enable_gqa else None
        attn_mask_type = "causal" if is_causal else "no_mask"

        self._lazy_init(num_gqa_groups, attn_mask_type)

        with fp8_autocast(enabled=True, fp8_recipe=self.recipe):
            out = self._attn_op(q, k, v, attention_mask=None)
        # TE returns [B, S, H*D]; restore [B, S, H, D] (NHD).
        return out.unflatten(-1, (self.num_heads, self.head_dim))

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

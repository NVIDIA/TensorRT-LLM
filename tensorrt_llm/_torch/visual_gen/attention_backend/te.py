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
"""TransformerEngine FP8 attention backend for visual generation (diffusion) models.

Uses TransformerEngine's ``DotProductAttention`` under ``fp8_autocast`` with
``Float8CurrentScaling(fp8_dpa=True, fp8_mha=True)``.  Operates in NHD layout
([B, S, H, D]) which maps directly to TE's ``qkv_format="bshd"`` -- no
transpose overhead.

``forward`` is decorated with ``@torch.compiler.disable`` because TE FP8
modules graph-break under torch.compile.
"""

import math
from typing import Any, Optional, Tuple

import torch

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

try:
    from transformer_engine.common.recipe import Float8CurrentScaling
    from transformer_engine.pytorch import DotProductAttention, fp8_autocast

    _TE_AVAILABLE = True
except ImportError:
    _TE_AVAILABLE = False


class TEAttention(AttentionBackend):
    """FP8 attention via TransformerEngine ``DotProductAttention``.

    FP8 is always enabled -- this backend exists to get FP8 attention.
    No KV cache: diffusion models recompute attention each denoising step.
    For BF16 attention use ``VANILLA``.

    Does not support ``forward_with_lse``: TE's ``DotProductAttention`` does not
    expose softmax stats, so the LSE would have to be recomputed from a separate
    O(S^2) fp32 score matrix -- strictly more work than the attention itself.
    That rules TE out as an inner backend for ``Attention2DAttention`` and
    ``RingAttention``, which raise at construction on ``support_lse() is False``.
    ``UlyssesAttention`` does not need LSE and still works.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        quant_attention_config: Optional[Any] = None,
        fp8_group: Optional[Any] = None,
        **kwargs,
    ):
        if not _TE_AVAILABLE:
            raise ImportError(
                "TransformerEngine is required for the TE attention backend. "
                "Install transformer_engine before using backend='TE'."
            )
        if quant_attention_config is not None:
            # TE drives its own FP8 recipe; swallowing this would hide a config mistake.
            raise NotImplementedError(
                "TE attention backend does not honor quant_attention_config -- it always "
                "runs FP8 via its own Float8CurrentScaling recipe. Drop quant_attention_config "
                "or pick a backend that applies it."
            )
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.scale = 1.0 / math.sqrt(head_dim)
        # Current scaling: TE gates delayed scaling's update on grad, so it never calibrates here.
        self.recipe = Float8CurrentScaling(fp8_dpa=True, fp8_mha=True)
        # None = no amax reduction: per-rank scales, no world-wide collective per call.
        self.fp8_group = fp8_group
        # Cache per trait: avoids rebuilding a DotProductAttention on every forward.
        self._attn_ops: dict[tuple, Any] = {}

    def _get_attn_op(self, num_gqa_groups: Optional[int], attn_mask_type: str) -> Any:
        traits = (self.num_heads, self.head_dim, num_gqa_groups, attn_mask_type)
        if traits not in self._attn_ops:
            op = DotProductAttention(
                self.num_heads,
                self.head_dim,
                num_gqa_groups=num_gqa_groups,
                attn_mask_type=attn_mask_type,
                softmax_scale=self.scale,
                qkv_format="bshd",
            )
            # eval(): cuDNN's FP8 gate demands head_dim == 128 on sm < 100 while is_training.
            self._attn_ops[traits] = op.eval()
        return self._attn_ops[traits]

    def _parse_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_mask: PredefinedAttentionMask,
        key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[int], str]:
        if key_padding_mask is not None:
            raise NotImplementedError("TE attention backend does not yet support key_padding_mask.")
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            attn_mask_type = "causal"
        elif attention_mask is None or attention_mask == PredefinedAttentionMask.FULL:
            # None means "no mask" across the backends, and Attention2D forwards it as-is.
            attn_mask_type = "no_mask"
        else:
            raise NotImplementedError(
                f"TE attention backend does not support attention_mask={attention_mask!r}. "
                "Only PredefinedAttentionMask.FULL and CAUSAL are supported."
            )
        enable_gqa = self.num_heads != self.num_kv_heads
        num_gqa_groups = k.shape[-2] if enable_gqa else None
        return num_gqa_groups, attn_mask_type

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
        """FP8 self/cross attention. q/k/v shape: [B, S, H, D]. Returns [B, S, H, D]."""
        num_gqa_groups, attn_mask_type = self._parse_inputs(q, k, attention_mask, key_padding_mask)
        attn_op = self._get_attn_op(num_gqa_groups, attn_mask_type)
        with fp8_autocast(enabled=True, fp8_recipe=self.recipe, fp8_group=self.fp8_group):
            # TE returns [B, S, H*D]; restore to [B, S, H, D].
            out = attn_op(q, k, v, attention_mask=None)
        return out.unflatten(-1, (self.num_heads, self.head_dim))

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return AttentionTensorLayout.NHD

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @classmethod
    def support_lse(cls) -> bool:
        return False

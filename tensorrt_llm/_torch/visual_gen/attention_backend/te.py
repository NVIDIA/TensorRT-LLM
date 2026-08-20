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
``DelayedScaling(fp8_dpa=True, fp8_mha=True)``.  Operates in NHD layout
([B, S, H, D]) which maps directly to TE's ``qkv_format="bshd"`` -- no
transpose overhead.

``forward`` and ``forward_with_lse`` are decorated with
``@torch.compiler.disable`` because TE FP8 modules graph-break under
torch.compile.
"""

import math
from typing import Any, Optional, Tuple

import torch

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

try:
    from transformer_engine.common.recipe import DelayedScaling
    from transformer_engine.pytorch import DotProductAttention, fp8_autocast

    _TE_AVAILABLE = True
except ImportError:
    _TE_AVAILABLE = False


class TEAttention(AttentionBackend):
    """FP8 attention via TransformerEngine ``DotProductAttention``.

    FP8 is always enabled -- this backend exists to get FP8 attention.
    No KV cache: diffusion models recompute attention each denoising step.
    For BF16 attention use ``VANILLA``.

    Supports ``forward_with_lse`` (LSE computed from BF16 Q/K scores), enabling
    use with Attention2DAttention for x72 context parallelism.
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
            # create_attention forwards this for every backend; TE drives its own
            # FP8 recipe, so silently swallowing it would hide a config mistake.
            raise NotImplementedError(
                "TE attention backend does not honor quant_attention_config -- it always "
                "runs FP8 via its own DelayedScaling recipe. Drop quant_attention_config "
                "or pick a backend that applies it."
            )
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.scale = 1.0 / math.sqrt(head_dim)
        self.recipe = DelayedScaling(fp8_dpa=True, fp8_mha=True)
        # Amax reduction group for fp8_autocast. Left as None, TE reduces amax over
        # the default process group on every autocast exit -- a world-wide collective
        # per attention call, and a hang if ranks ever take different paths. Diffusion
        # inference wants per-rank scales, so default to no reduction.
        self.fp8_group = fp8_group
        # DotProductAttention is stateful (amax history), and rebuilding one throws
        # that history away. Key a cache on the traits that force a new module so
        # alternating mask types or GQA shapes reuse their own instance.
        self._attn_ops: dict[tuple, Any] = {}

    def _get_attn_op(self, num_gqa_groups: Optional[int], attn_mask_type: str) -> Any:
        traits = (self.num_heads, self.head_dim, num_gqa_groups, attn_mask_type)
        if traits not in self._attn_ops:
            self._attn_ops[traits] = DotProductAttention(
                self.num_heads,
                self.head_dim,
                num_gqa_groups=num_gqa_groups,
                attn_mask_type=attn_mask_type,
                softmax_scale=self.scale,
                qkv_format="bshd",
            )
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
        elif attention_mask == PredefinedAttentionMask.FULL:
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

    @torch.compiler.disable
    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FP8 attention returning output and log-sum-exp. Required for Attention2D.

        TE's DotProductAttention does not expose softmax stats, so LSE is
        computed separately from BF16 Q/K scores. This is numerically accurate
        and satisfies the partition property Attention2D relies on.

        Note: allocates an O(S^2) float32 score matrix [B, H, S, S] for the
        LSE pass. For CP use cases S is the local shard length per rank.

        Returns:
            output: [B, S, H, D]
            lse:    [B, H, S] float32 -- log-sum-exp per query position
        """
        num_gqa_groups, attn_mask_type = self._parse_inputs(q, k, attention_mask, key_padding_mask)
        attn_op = self._get_attn_op(num_gqa_groups, attn_mask_type)
        B, S = q.shape[0], q.shape[1]
        with fp8_autocast(enabled=True, fp8_recipe=self.recipe, fp8_group=self.fp8_group):
            out = attn_op(q, k, v, attention_mask=None)
        # out: [B, S, H*D] -> [B, S, H, D]
        out = out.unflatten(-1, (self.num_heads, self.head_dim))

        # Compute LSE from BF16 scores: [B, H, S, S] -> [B, H, S]
        # q: [B, S, H, D] -> [B, H, S, D]
        # k: [B, S, Hkv, D] -> [B, H, S, D] (expand KV heads for GQA)
        q_f = q.transpose(1, 2).float()
        k_f = k.transpose(1, 2).float()
        if self.num_heads != self.num_kv_heads:
            k_f = k_f.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        scores = torch.matmul(q_f, k_f.transpose(-1, -2)) * self.scale
        if attn_mask_type == "causal":
            causal_mask = torch.ones(S, S, device=q.device, dtype=torch.bool).triu(diagonal=1)
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)  # [B, H, S] float32
        return out, lse

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return AttentionTensorLayout.NHD

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @classmethod
    def support_lse(cls) -> bool:
        return True

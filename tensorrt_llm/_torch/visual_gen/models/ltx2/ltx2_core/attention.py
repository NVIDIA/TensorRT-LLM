# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.attention import (
    Attention as UpstreamAttention,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.feed_forward import (
    FeedForward as FeedForward,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.gelu_approx import (
    GELUApprox as GELUApprox,
)

from .rope import LTXRopeType


class Attention(UpstreamAttention):
    """Translate the legacy connector-attention constructor and RoPE layout."""

    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        norm_eps=1e-6,
        rope_type=LTXRopeType.INTERLEAVED,
        apply_gated_attention=False,
    ):
        super().__init__(
            query_dim=query_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            norm_eps=norm_eps,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
        )

    def forward(self, x, context=None, mask=None, pe=None, k_pe=None):
        def convert(freqs):
            if freqs is None or self.rope_type != LTXRopeType.SPLIT:
                return freqs
            return tuple(f[..., : self.dim_head // 2].transpose(1, 2) for f in freqs)

        return super().forward(x, context=context, mask=mask, pe=convert(pe), k_pe=convert(k_pe))

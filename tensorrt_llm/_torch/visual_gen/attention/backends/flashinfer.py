# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer attention backend for visual generation models."""

from typing import Any

import torch

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.visual_gen.attention.flashinfer_runner import FlashInferPrefillRunner
from tensorrt_llm._torch.visual_gen.catalog.attention.flashinfer_prefill import flashinfer_prefill
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from .interface import AttentionBackend, AttentionTensorLayout


class FlashInferAttention(AttentionBackend):
    """Dense or block-scaled prefill over VisualGen NHD tensors."""

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: int | None = None,
        dtype: torch.dtype | None = None,
        quant_attention_config: QuantAttentionConfig | None = None,
        attention_metadata_state: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> None:
        self.runner = FlashInferPrefillRunner(
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            quant_attention_config=quant_attention_config,
            attention_metadata_state=attention_metadata_state,
            static_global_scale=(attention_metadata_state or {}).get("_ltx23_static_global_scale")
            == 1.0,
        )
        self.layer_idx = layer_idx
        self.num_heads = self.runner.num_heads
        self.num_kv_heads = self.runner.num_kv_heads
        self.head_dim = self.runner.head_dim
        self.scale = self.runner.scale
        self.dtype = dtype
        self.quant_attention_config = quant_attention_config

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        output, _ = self.forward_with_lse(
            q,
            k,
            v,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        return output

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attention output and FP32 LSE in ``[B, H, S]`` layout."""
        return flashinfer_prefill(
            q,
            k,
            v,
            runner=self.runner,
            is_causal=attention_mask == PredefinedAttentionMask.CAUSAL,
            key_padding_mask=key_padding_mask,
        )

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return AttentionTensorLayout.NHD

    @classmethod
    def support_lse(cls) -> bool:
        return True

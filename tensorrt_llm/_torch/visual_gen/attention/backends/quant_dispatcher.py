# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select an attention quantization path before sequence-parallel wrapping."""

from typing import Any

import torch

from .interface import AttentionBackend, AttentionTensorLayout


class AttentionQuantDispatcher(AttentionBackend):
    """Dispatch unquantized and quantized instances of the selected backend per step."""

    def __init__(
        self, quantized: AttentionBackend, unquantized: AttentionBackend, state: dict[str, Any]
    ) -> None:
        if quantized.preferred_layout != unquantized.preferred_layout:
            raise ValueError("Attention quantization paths must use the same tensor layout.")
        self.quantized = quantized
        self.unquantized = unquantized
        self.state = state
        # Sequence-parallel wrappers size exchanges from the inner backend's heads.
        self.num_heads = unquantized.num_heads
        self.num_kv_heads = getattr(unquantized, "num_kv_heads", self.num_heads)
        self.head_dim = unquantized.head_dim

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self.unquantized.preferred_layout

    def _selected(self) -> AttentionBackend:
        return (
            self.unquantized if self.state.get("attention_quant_ignored", False) else self.quantized
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._selected().forward(q=q, k=k, v=v, **kwargs)

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._selected().forward_with_lse(q=q, k=k, v=v, **kwargs)

    def support_fused_qkv(self) -> bool:
        return self.unquantized.support_fused_qkv() and self.quantized.support_fused_qkv()

    def support_lse(self) -> bool:
        return self.unquantized.support_lse() and self.quantized.support_lse()

    def support_kv_cache(self) -> bool:
        return self.unquantized.support_kv_cache() and self.quantized.support_kv_cache()

    def support_kv_cache_page_size(
        self, tokens_per_page: int, dtype: torch.dtype | None = None
    ) -> bool:
        return self.unquantized.support_kv_cache_page_size(
            tokens_per_page, dtype
        ) and self.quantized.support_kv_cache_page_size(tokens_per_page, dtype)

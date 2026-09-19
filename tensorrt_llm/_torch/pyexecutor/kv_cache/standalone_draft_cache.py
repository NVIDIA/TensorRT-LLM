# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage and history contracts for standalone DSpark/DFlash drafters."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StandaloneDraftLayout:
    """Rank-local full-attention storage, independent of target KV geometry."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    extra_tokens: int
    attention_backend: str

    def __post_init__(self) -> None:
        if min(self.num_layers, self.num_kv_heads, self.head_dim) <= 0:
            raise ValueError("Standalone draft cache dimensions must be positive")
        if self.extra_tokens < 0:
            raise ValueError("Standalone draft scratch capacity must be nonnegative")
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Standalone draft KV supports FP16 and BF16 storage")
        if self.attention_backend not in ("VANILLA", "TRTLLM"):
            raise ValueError("Standalone draft KV requires VANILLA or TRTLLM attention")

    @property
    def bytes_per_layer_token(self) -> int:
        return 2 * self.num_kv_heads * self.head_dim * self.dtype.itemsize

    @property
    def bytes_per_token(self) -> int:
        return self.num_layers * self.bytes_per_layer_token

    def transfer_identity(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": str(self.dtype),
            "attention_backend": self.attention_backend,
        }


@dataclass(frozen=True)
class StandaloneDraftHistory:
    """Committed draft tokens and their next absolute sequence position.

    These are deliberately separate from the target cache's monotonic history
    watermark and from its speculative allocation capacity.
    """

    valid_length: int
    position: int

    def __post_init__(self) -> None:
        if type(self.valid_length) is not int or type(self.position) is not int:
            raise ValueError("Standalone draft history requires integer length and position")
        if self.valid_length < 0 or self.position < self.valid_length:
            raise ValueError("Invalid standalone draft history length or position")

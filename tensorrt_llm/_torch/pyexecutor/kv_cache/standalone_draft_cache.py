# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage and history contracts for manager-owned DSpark/DFlash drafters."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .kv_cache_manager_v2 import KVCacheManagerV2


@dataclass(frozen=True)
class StandaloneDraftLayout:
    """Rank-local draft storage, independent of target geometry and retention."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    extra_tokens: int
    attention_backend: str
    kv_factor: int = 2
    window_size: int | None = None

    def __post_init__(self) -> None:
        if min(self.num_layers, self.num_kv_heads, self.head_dim) <= 0:
            raise ValueError("Standalone draft cache dimensions must be positive")
        if self.extra_tokens < 0:
            raise ValueError("Standalone draft scratch capacity must be nonnegative")
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Standalone draft KV supports FP16 and BF16 storage")
        if self.attention_backend not in ("VANILLA", "TRTLLM", "DSv4"):
            raise ValueError("Unsupported managed draft attention backend")
        if self.kv_factor not in (1, 2):
            raise ValueError("Draft KV storage requires one or two planes")
        if self.window_size is not None and self.window_size <= 0:
            raise ValueError("Draft history window must be positive")

    @property
    def bytes_per_layer_token(self) -> int:
        return self.kv_factor * self.num_kv_heads * self.head_dim * self.dtype.itemsize

    @property
    def retention_window_size(self) -> int | None:
        # V2 retains window_size - 1 committed rows between forwards.
        return self.window_size + 1 if self.window_size is not None else None

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
            "kv_factor": self.kv_factor,
            "window_size": self.window_size,
        }


@dataclass(frozen=True)
class StandaloneDraftHistory:
    """Committed length and next absolute position, independent of target/scratch state."""

    valid_length: int
    position: int

    def __post_init__(self) -> None:
        if type(self.valid_length) is not int or type(self.position) is not int:
            raise ValueError("Standalone draft history requires integer length and position")
        if self.valid_length < 0 or self.position < self.valid_length:
            raise ValueError("Invalid standalone draft history length or position")


@dataclass(frozen=True)
class DraftHistoryUpdate:
    """One execution's draft history, published after its sampling event completes."""

    manager: "KVCacheManagerV2"
    request_ids: tuple[int, ...]
    cache_instances: tuple[object, ...]
    values_host: torch.Tensor

    @classmethod
    def capture(
        cls,
        manager: "KVCacheManagerV2",
        request_ids: tuple[int, ...],
        values: torch.Tensor,
    ) -> "DraftHistoryUpdate":
        """Queue an independent length/position readback on the execution stream."""
        if values.shape != (len(request_ids), 2):
            raise ValueError("Draft history update requires one length/position pair per request")
        values_host = torch.empty(
            values.shape, dtype=values.dtype, device="cpu", pin_memory=values.is_cuda
        )
        values_host.copy_(values, non_blocking=True)
        return cls(
            manager,
            tuple(request_ids),
            tuple(manager.kv_cache_map[request_id] for request_id in request_ids),
            values_host,
        )

    def publish(self) -> None:
        """Publish completed writes without reviving released or replaced requests."""
        for request_id, cache, (length, position) in zip(
            self.request_ids, self.cache_instances, self.values_host.tolist()
        ):
            current = self.manager.kv_cache_map.get(request_id)
            if current is cache and current.is_active:
                self.manager.set_draft_history(request_id, length, position)

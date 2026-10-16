# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage and history contracts for manager-owned DSpark/DFlash drafters."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .kv_cache_manager_v2 import KVCacheManagerV2


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

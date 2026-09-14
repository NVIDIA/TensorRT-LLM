# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logical sparse selections, before any GPU cache lookup."""

from dataclasses import dataclass
from typing import Protocol

import torch


def check_tensor(
    tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    """Check metadata only; never read tensor values on the CPU."""
    if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(
            f"Expected {shape}, {dtype}, {device}; got {tensor.shape}, {tensor.dtype}, {tensor.device}"
        )


@dataclass(frozen=True)
class SelectionContext:
    """Identity and bounds for a batch of queries, all tensors on the selection device.

    request_ids: int64 [requests], stable request IDs, not batch slot numbers.
    request_indices: int32 [queries], maps each query to a request_ids row.
    valid_lengths: int32 [queries], number of entries this query may read. The model
        supplies causal bounds and counts only completed compressed entries.
    layer_id: KVCM layer ID, which can differ from the model layer number.
    life_cycle_id: KVCM-assigned lifecycle ID, not the residencyGroup setting.

    Update tensor contents before graph replay when requests or lengths change.
    IDs must distinguish live request lifetimes. Padding uses request index -1.
    """

    request_ids: torch.Tensor
    request_indices: torch.Tensor
    valid_lengths: torch.Tensor
    layer_id: int
    life_cycle_id: int

    def __post_init__(self) -> None:
        if self.layer_id < 0 or self.life_cycle_id < 0:
            raise ValueError("Layer and lifecycle IDs must be nonnegative")
        if self.request_ids.ndim != 1 or self.request_indices.ndim != 1:
            raise ValueError("Request IDs and query-to-request indices must be vectors")
        device = self.request_indices.device
        check_tensor(self.request_ids, (self.request_ids.numel(),), torch.int64, device)
        check_tensor(self.request_indices, (self.request_indices.numel(),), torch.int32, device)
        check_tensor(self.valid_lengths, self.request_indices.shape, torch.int32, device)


@dataclass(frozen=True)
class SelectedEntries:
    """Borrowed int32 positions [queries, top_k] in each request's logical history.

    Positions count tokens, or native compressed entries for a compressed layout.
    Column order is attention output order. Duplicates keep separate columns;
    a later transfer may share their bytes but must restore this order.
    Negative/out-of-range positions and false valid_mask entries are invalid.
    valid_mask is optional bool [queries, top_k]; None means use position bounds.
    Neither an ordinary GPU page table nor GPU residency changes validity.
    The producer must keep these tensors unchanged until consumers finish.
    """

    context: SelectionContext
    positions: torch.Tensor
    valid_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (
            self.positions.ndim != 2
            or self.positions.shape[0] != self.context.request_indices.numel()
        ):
            raise ValueError("Positions must have one row per query")
        check_tensor(
            self.positions, self.positions.shape, torch.int32, self.context.request_ids.device
        )
        if self.valid_mask is not None:
            check_tensor(self.valid_mask, self.positions.shape, torch.bool, self.positions.device)


class SelectionPolicy(Protocol):
    """Adapt a model selector's output without resolving physical storage."""

    def select(
        self,
        positions: torch.Tensor,
        context: SelectionContext,
        valid_mask: torch.Tensor | None = None,
    ) -> SelectedEntries:
        """Keep model selection order and use model-provided entry bounds."""
        ...

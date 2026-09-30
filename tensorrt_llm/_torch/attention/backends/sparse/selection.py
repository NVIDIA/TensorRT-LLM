# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logical sparse selections, before any GPU cache lookup."""

from dataclasses import dataclass

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
    """Borrow existing batch metadata and explicit host-source row references.

    req_idx_per_token: int32 [queries], DSA's existing query-to-batch mapping.
    kv_lens_cuda: int32 [requests], DSA's existing sequence lengths (input tokens).
    host_source_rows: int32 [requests], existing IndexMapper rows, or -1 for padding.
    host_source_generations: uint64 [requests], row generations for this batch.

    Generation checks reject stale row reuse without GPU request-ID lookup.
    The selector enforces each query's causal bound. The future cache path also
    checks sequence length and completed host coverage. All tensors are borrowed;
    update graph inputs before replay and keep them unchanged through GPU use.
    """

    req_idx_per_token: torch.Tensor
    kv_lens_cuda: torch.Tensor
    host_source_rows: torch.Tensor
    host_source_generations: torch.Tensor

    def __post_init__(self) -> None:
        if self.req_idx_per_token.ndim != 1 or self.kv_lens_cuda.ndim != 1:
            raise ValueError("Query-to-request indices and KV lengths must be vectors")
        device = self.req_idx_per_token.device
        check_tensor(self.req_idx_per_token, self.req_idx_per_token.shape, torch.int32, device)
        check_tensor(self.kv_lens_cuda, self.kv_lens_cuda.shape, torch.int32, device)
        check_tensor(self.host_source_rows, self.kv_lens_cuda.shape, torch.int32, device)
        check_tensor(self.host_source_generations, self.kv_lens_cuda.shape, torch.uint64, device)


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
            or self.positions.shape[0] != self.context.req_idx_per_token.numel()
        ):
            raise ValueError("Positions must have one row per query")
        check_tensor(
            self.positions, self.positions.shape, torch.int32, self.context.req_idx_per_token.device
        )
        if self.valid_mask is not None:
            check_tensor(self.valid_mask, self.positions.shape, torch.bool, self.positions.device)

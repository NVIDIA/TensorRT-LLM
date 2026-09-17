# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt DSA's logical top-K before its ordinary page-table remap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..selection import SelectedEntries, SelectionContext

if TYPE_CHECKING:
    from .indexer import Indexer
    from .metadata import DSAtrtllmAttentionMetadata


def make_dsa_selection(
    positions: torch.Tensor,
    metadata: DSAtrtllmAttentionMetadata,
    host_source_rows: torch.Tensor,
    host_source_generations: torch.Tensor,
    *,
    is_generation: bool,
) -> SelectedEntries:
    """Borrow DSA's existing batch mapping and lengths; do not build duplicate tensors."""
    start = metadata.num_ctx_tokens if is_generation else 0
    context = SelectionContext(
        metadata.req_idx_per_token[start : start + positions.shape[0]],
        metadata.kv_lens_cuda[: metadata.num_seqs],
        host_source_rows,
        host_source_generations,
    )
    return SelectedEntries(context, positions)


def select_dsa_topk(
    indexer: Indexer | None,
    q: torch.Tensor,
    metadata: DSAtrtllmAttentionMetadata,
    indexer_intermediates: list[torch.Tensor] | None,
    *,
    is_generation: bool,
) -> torch.Tensor:
    """Run/reuse DSA top-K without reading physical KV page tables.

    Shared layers borrow the preceding indexer's output. Preserve the target
    selection during an MTP draft step that shares its indexer. This is also
    used by the ordinary GPU path before it remaps positions.
    """
    phase_start = metadata.num_ctx_tokens if is_generation else 0
    phase_end = metadata.num_tokens if is_generation else metadata.num_ctx_tokens
    shared_topk_indices = metadata.shared_topk_indices
    if indexer is None:
        topk_indices = shared_topk_indices[phase_start:phase_end]
    else:
        topk_indices = indexer.forward_from_projected(
            metadata,
            q,
            indexer_intermediates,
            is_generation=is_generation,
        )
        preserve_mtp_topk = metadata.in_mtp_draft_loop and indexer.mtp_index_share
        if shared_topk_indices is not None and not preserve_mtp_topk:
            shared_topk_indices[
                phase_start : phase_start + topk_indices.shape[0],
                : topk_indices.shape[1],
            ].copy_(topk_indices)

    return topk_indices

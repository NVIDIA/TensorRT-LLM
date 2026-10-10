# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Describe a standalone MLA draft pool alongside the target's transfer pools."""

from dataclasses import dataclass, replace

import numpy as np

from .page import CacheKind, KVCachePageTable, MapperKind
from .utils import get_pool_view_slot_bytes


@dataclass(frozen=True)
class DraftCacheInfo:
    first_layer_group: int
    first_layer: int
    num_layers: int
    head_dim: int
    dtype: str

    def validate_peer(self, peer: "DraftCacheInfo | None") -> None:
        if peer is None:
            return
        if (self.first_layer, self.num_layers, self.head_dim, self.dtype) != (
            peer.first_layer,
            peer.num_layers,
            peer.head_dim,
            peer.dtype,
        ):
            raise ValueError(f"Standalone draft cache layouts differ: local={self}, peer={peer}")


def append_draft_page_table(
    target: KVCachePageTable, draft: KVCachePageTable, first_layer: int
) -> KVCachePageTable:
    """Combine descriptors without changing either manager's indices or storage.

    Draft layer IDs follow target layer IDs in the transfer namespace. MLA
    latents are replicated across TP ranks, including when target KV is sharded.
    """
    if target.tokens_per_block != draft.tokens_per_block:
        raise ValueError("Target and draft transfer pools require the same page size.")
    groups = list(target.layer_groups)
    for group in draft.layer_groups:
        if group.kind != CacheKind.PAGED:
            raise ValueError("Standalone MLA draft transfer requires attention-only cache pools.")
        groups.append(
            replace(
                group,
                pool_group_idx=group.pool_group_idx + len(target.pool_groups),
                local_layers=[
                    replace(layer, global_layer_id=first_layer + layer.global_layer_id)
                    for layer in group.local_layers
                ],
                pool_views=[
                    replace(
                        view,
                        mapper_kind=MapperKind.REPLICATED,
                        pool_role=frozenset({"draft_context"}),
                    )
                    for view in group.pool_views
                ],
            )
        )
    return KVCachePageTable(
        target.tokens_per_block, groups, [*target.pool_groups, *draft.pool_groups]
    )


def omit_draft_destinations(chunk, page_table: KVCachePageTable, info: DraftCacheInfo) -> int:
    """Remove optional draft destinations when only the receiver has a drafter."""
    removed_bytes = 0
    for index in range(info.first_layer_group, len(page_table.layer_groups)):
        blocks = chunk.block_ids_per_layer_groups[index]
        count = int((blocks >= 0).sum())
        removed_bytes += count * sum(
            get_pool_view_slot_bytes(view) for view in page_table.layer_groups[index].pool_views
        )
        chunk.block_ids_per_layer_groups[index] = np.empty(0, dtype=np.int64)
    return removed_bytes

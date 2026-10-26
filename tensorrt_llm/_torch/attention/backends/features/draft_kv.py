# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate draft KV cache state of the attention metadata."""

from typing import Optional

import torch

from ..extensions import MetadataExtension


class DraftKVMetadata(MetadataExtension):
    """Block offsets of the draft model's KV cache.

    One-model speculative decoding with a separate draft KV cache layout keeps
    the draft cache's block offsets here, refreshed with the target's every
    step, so a draft forward can swap them in without staging. The swap itself
    is ``prepare_attn_metadata_for_draft_replay`` in the speculative package.
    """

    def __init__(self, metadata) -> None:
        super().__init__(metadata)
        self.kv_cache_block_offsets: Optional[torch.Tensor] = None

    def allocate(self, buffers) -> None:
        md = self.metadata
        manager = md.draft_kv_cache_manager
        num_pools = getattr(manager, "num_attention_op_pools", manager.num_pools)
        self.kv_cache_block_offsets = md.get_empty(
            buffers,
            [num_pools, md.max_num_sequences, 2, manager.max_blocks_per_seq],
            cache_name="draft_kv_cache_block_offsets",
            dtype=torch.int32,
            capture_graph=md.is_cuda_graph,
        )

    def needs_full_block_table_width(self) -> bool:
        # Draft sub-steps advance kv_lens_cuda on device past the host snapshot.
        return self.metadata.draft_kv_cache_manager is not None

    def stage_block_offsets(self, max_blocks: Optional[int]) -> None:
        """Copy this step's draft block offsets, as wide as the target's."""
        md = self.metadata
        if md.draft_kv_cache_manager is None:
            return
        md.draft_kv_cache_manager.copy_batch_block_offsets(
            self.kv_cache_block_offsets,
            md.request_ids,
            md.beam_width,
            md.num_contexts,
            md.num_seqs,
            max_blocks=max_blocks,
        )

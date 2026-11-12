# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashMLA state of the attention metadata."""

from typing import Optional

import torch

from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.bindings.internal import thop

from ..extensions import MetadataExtension


class FlashMlaMetadata(MetadataExtension):
    """Block IDs and tile-scheduler metadata of the FlashMLA generation kernels.

    Provides the core's ``block_ids_per_seq`` (generation rows, compacted) and
    ``kv_block_ids_per_seq`` (every row). The tile-scheduler metadata is
    computed once per forward by the first generation layer and reused until
    the KV lengths or block IDs change.
    """

    fields = frozenset(
        {
            "metadata_valid",
            "flash_mla_num_splits",
            "flash_mla_tile_scheduler_metadata",
        }
    )
    methods = frozenset(
        {
            "compute_scheduler_metadata",
            "prepare_block_ids",
        }
    )

    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        self.flash_mla_tile_scheduler_metadata: Optional[torch.Tensor] = None
        self.flash_mla_num_splits: Optional[torch.Tensor] = None
        self.metadata_valid = False
        buffers = metadata.cuda_graph_buffers
        md = self.metadata
        manager = md.kv_cache_manager
        if manager is None:
            return
        capture_graph = md.is_cuda_graph

        def block_ids(owner, name):
            return md.get_empty(
                buffers,
                [owner.max_batch_size, owner.max_blocks_per_seq],
                cache_name=name,
                dtype=torch.int32,
                capture_graph=capture_graph,
            )

        md.use_block_id_tables(
            block_ids(manager, "block_ids_per_seq"), block_ids(manager, "kv_block_ids_per_seq")
        )
        # Pre-allocated so their GPU addresses are stable across CUDA graph captures.
        sm_count = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self.flash_mla_tile_scheduler_metadata = md.get_empty(
            buffers,
            [sm_count * 8],  # TileSchedulerMetaDataSize = 8
            cache_name="flash_mla_tile_scheduler_metadata",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.flash_mla_num_splits = md.get_empty(
            buffers,
            [manager.max_batch_size + 1],
            cache_name="flash_mla_num_splits",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    adopts_target_state = True
    # Its tile schedule, and whether it is current, are computed lazily from
    # a draft's own lengths.
    draft_private = frozenset(
        {"flash_mla_tile_scheduler_metadata", "flash_mla_num_splits", "metadata_valid"}
    )

    def prepare(self) -> None:
        self.prepare_block_ids()

    def adopt_target_state(self, source: "MetadataExtension") -> None:
        super().adopt_target_state(source)
        # Its tile schedule follows its own lengths.
        self.metadata_valid = False

    def prepare_block_ids(self) -> None:
        """Stage this batch's block IDs from the current KV cache manager."""
        md = self.metadata
        self.metadata_valid = False
        # Request-specific fills and H2D copies must happen before replay, not
        # become fixed operations in the captured graph.
        if torch.cuda.is_current_stream_capturing():
            return

        block_ids_per_seq = maybe_pin_memory(
            md.kv_cache_manager.get_block_ids_per_seq(md.request_ids)
        )
        num_blocks = block_ids_per_seq.shape[1]
        md.kv_block_ids_per_seq.fill_(0)
        md.kv_block_ids_per_seq[: md.num_seqs, :num_blocks].copy_(
            block_ids_per_seq, non_blocking=True
        )
        md.block_ids_per_seq.fill_(0)
        md.block_ids_per_seq[: md.num_generations, :num_blocks].copy_(
            block_ids_per_seq[md.num_contexts :], non_blocking=True
        )

    def refresh_from_draft_lengths(self) -> None:
        super().refresh_from_draft_lengths()
        self.metadata_valid = False

    def compute_scheduler_metadata(self, num_heads: int, kv_lora_rank: int) -> None:
        """Compute the tile-scheduler metadata once per forward."""
        md = self.metadata
        if self.metadata_valid:
            return
        num_generations = md.num_generations
        if num_generations > 0:
            generation_slice = slice(md.num_contexts, md.num_contexts + num_generations)
            s_q = int(md.seq_lens[md.num_contexts].item())
            thop.compute_flash_mla_metadata(
                md.kv_lens_cuda_runtime[generation_slice],
                self.flash_mla_tile_scheduler_metadata,
                self.flash_mla_num_splits[: num_generations + 1],
                num_generations,
                s_q,
                num_heads,
                1,
                kv_lora_rank,
            )
        self.metadata_valid = True

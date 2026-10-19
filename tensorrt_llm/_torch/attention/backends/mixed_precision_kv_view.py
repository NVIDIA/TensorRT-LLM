# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert mixed-precision KV pools to and from native Attention views."""

import torch

from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned

from .interface import AttentionMetadata


def get_mixed_precision_kv_view(
    metadata: AttentionMetadata, layer_idx: int
) -> "MixedPrecisionKVView | None":
    """Return the optional mixed-precision view for this layer."""
    views = getattr(metadata, "mixed_precision_kv_views", None)
    return None if views is None else views.get(layer_idx)


class MixedPrecisionKVView:
    """Expose mixed-precision KV pools as a native Attention cache."""

    @classmethod
    def prepare_batch(cls, metadata: AttentionMetadata) -> None:
        """Update mixed-precision addresses before capture or replay."""
        if not metadata.kv_cache_manager.high_layers:
            return
        view = next(iter(metadata.mixed_precision_kv_views.values()), None)
        if view is None or not metadata.is_cuda_graph:
            view = cls(metadata)
            metadata.mixed_precision_kv_views = dict.fromkeys(view.pools, view)
        view.update(metadata)

    def __init__(self, metadata: AttentionMetadata) -> None:
        from .trtllm import TrtllmAttentionMetadata

        manager = metadata.kv_cache_manager
        trtllm = isinstance(metadata, TrtllmAttentionMetadata)
        self.layout = "HND" if trtllm else metadata.kv_layout
        self.block_size = manager.tokens_per_block
        # Native BF16 paged-context FMHA uses 32-token pages.
        self.tokens_per_block = min(self.block_size, 32) if trtllm else self.block_size
        self.max_seq_len = manager.max_seq_len
        self.max_blocks_per_seq = (
            self.max_seq_len + self.tokens_per_block - 1
        ) // self.tokens_per_block
        self.kv_factor = manager.kv_factor
        # Fixed per-request spans keep addresses unchanged as sequences grow.
        self.page_starts = [row * self.max_blocks_per_seq for row in range(metadata.num_seqs + 1)]
        storage = manager.mixed_precision_buffers.get_buffer(
            [manager._staging_bytes], torch.uint8, "mixed_kv_staging", True
        )
        self.pools = {}
        for layer_idx, local in manager.layer_offsets.items():
            if local not in manager.high_layers:
                continue
            pools = (
                manager.get_buffers(layer_idx, metadata.kv_layout, mixed_precision_pool=pool)
                for pool in ("low", "high")
            )
            pools = tuple(
                pool if self.layout == metadata.kv_layout else pool.transpose(2, 3)
                for pool in pools
            )
            shape = [self.page_starts[-1], *pools[-1].shape[1:]]
            shape[3 if self.layout == "HND" else 2] = self.tokens_per_block
            staging = manager.mixed_precision_buffers._view_as(storage, shape, pools[-1].dtype)
            self.pools[layer_idx] = (staging, *pools)
        self.layer_offsets = {layer: layer for layer in self.pools}
        self.max_attention_window_vec = (None,) * (max(self.pools) + 1)
        self.dtype = manager.dtype
        self.low_dtype = manager.low_dtype
        self.kv_cache_pool_pointers = torch.tensor([[storage.data_ptr(), 0]], dtype=torch.int64)
        self.kv_cache_pool_mapping = torch.zeros((max(self.pools) + 1, 2), dtype=torch.int32)
        self.kv_cache_block_offsets = None

    def update(self, metadata: AttentionMetadata) -> None:
        manager = metadata.kv_cache_manager
        cached_lengths = metadata.kv_cache_params.num_cached_tokens_per_seq
        seq_lens = metadata.seq_lens.tolist()
        lengths = [a + b for a, b in zip(cached_lengths, seq_lens)]
        self.page_counts = [
            (n + self.tokens_per_block - 1) // self.tokens_per_block for n in lengths
        ]
        batch = len(lengths)
        buffers = manager.mixed_precision_buffers
        max_pages = (self.max_seq_len + self.block_size - 1) // self.block_size
        self.page_tables = {}
        for layer_idx in self.pools:
            page_tables = manager.get_mixed_precision_page_tables(metadata.request_ids, layer_idx)
            tables = torch.full(
                (2, batch, max_pages), -1, dtype=torch.int32, pin_memory=prefer_pinned()
            )
            for role, rows in enumerate(page_tables):
                for row, pages in enumerate(rows):
                    count = min(len(pages), max_pages)
                    tables[role, row, :count] = torch.tensor(pages[:count], dtype=torch.int32)
            self.page_tables[layer_idx] = buffers.get_buffer(
                (2, manager.max_batch_size, max_pages),
                torch.int32,
                f"mixed_page_tables_{layer_idx}",
                metadata.is_cuda_graph,
            )[:, :batch]
            self.page_tables[layer_idx].copy_(tables, non_blocking=True)
        self.lengths = buffers.get_buffer(
            (2, manager.max_batch_size), torch.int32, "mixed_lengths", metadata.is_cuda_graph
        )[:, :batch]
        self.lengths.copy_(
            torch.tensor([cached_lengths, lengths], dtype=torch.int32, pin_memory=prefer_pinned()),
            non_blocking=True,
        )
        self.max_new_tokens_per_request = max(seq_lens, default=0)

        from .trtllm import TrtllmAttentionMetadata

        if isinstance(metadata, TrtllmAttentionMetadata):
            ordinals = torch.arange(self.max_blocks_per_seq, dtype=torch.int32)
            pages = torch.tensor(self.page_starts[:-1], dtype=torch.int32)[:, None] + ordinals
            offsets = torch.stack(
                (self.kv_factor * pages, self.kv_factor * pages + self.kv_factor - 1), dim=1
            )
            valid = ordinals < torch.tensor(self.page_counts, dtype=torch.int32)[:, None]
            block_offsets = buffers.get_buffer(
                (1, manager.max_batch_size, 2, self.max_blocks_per_seq),
                torch.int32,
                "mixed_block_offsets",
                metadata.is_cuda_graph,
            )[:, : len(lengths)]
            block_offsets.copy_(
                maybe_pin_memory(offsets.masked_fill(~valid[:, None], -1).unsqueeze(0)),
                non_blocking=True,
            )
            self.kv_cache_block_offsets = block_offsets

    def swap_staging(self, metadata: AttentionMetadata, kv=None):
        """Select staging KV or restore the supplied cache state."""
        saved = (metadata.kv_cache_manager,)
        metadata.kv_cache_manager = self if kv is None else kv[0]
        if self.kv_cache_block_offsets is not None:
            saved += (
                metadata.kv_layout,
                metadata.use_paged_context_fmha,
                metadata.kv_cache_block_offsets,
            )
            (
                metadata.kv_layout,
                metadata.use_paged_context_fmha,
                metadata.kv_cache_block_offsets,
            ) = (self.layout, True, self.kv_cache_block_offsets) if kv is None else kv[1:]
        return saved

    def convert(self, layer_idx: int, *, writeback: bool = False) -> None:
        from ...cute_dsl_kernels.mixed_precision_kv import convert_mixed_precision_kv

        convert_mixed_precision_kv(
            writeback,
            self.max_new_tokens_per_request
            if writeback
            else self.max_blocks_per_seq * self.tokens_per_block,
            *self.pools[layer_idx],
            self.page_tables[layer_idx],
            self.lengths,
            self.layout,
            low_dtype=self.low_dtype,
        )

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> torch.Tensor:
        pool = self.pools[layer_idx][0]
        return pool if kv_layout == self.layout else pool.transpose(2, 3)

    def get_attention_op_num_blocks(self, local_layer_idx: int) -> int:
        return self.page_starts[-1] * self.kv_factor

    def get_attention_op_kv_page_offset(self, local_layer_idx: int) -> int:
        return self.kv_factor - 1

    def get_batch_cache_indices(self, request_ids: list[int], layer_idx: int) -> list[list[int]]:
        return [
            list(range(start, start + count))
            for start, count in zip(self.page_starts, self.page_counts)
        ]

    def get_batch_cache_indices_flat(self, request_ids, num_blocks, layer_idx=None):
        return torch.tensor(
            [
                page
                for start, count in zip(self.page_starts, num_blocks)
                for page in range(start, start + count)
            ],
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
        )

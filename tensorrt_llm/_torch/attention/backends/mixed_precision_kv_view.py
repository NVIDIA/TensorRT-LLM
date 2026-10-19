# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert mixed-precision KV pools to and from native Attention views."""

import copy
from dataclasses import replace

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
        """Update addresses and native Attention plans before capture or replay."""
        manager = metadata.kv_cache_manager
        for layer_idx, local in manager.layer_offsets.items():
            if local not in manager.high_layers:
                continue
            view = metadata.mixed_precision_kv_views.get(layer_idx)
            if view is None or not metadata.is_cuda_graph:
                view = cls(metadata, layer_idx)
                metadata.mixed_precision_kv_views[layer_idx] = view
            view.update(metadata, layer_idx)

    def __init__(
        self,
        metadata: AttentionMetadata,
        layer_idx: int,
    ) -> None:
        from .trtllm import TrtllmAttentionMetadata

        manager = metadata.kv_cache_manager
        trtllm = isinstance(metadata, TrtllmAttentionMetadata)
        self.layout = "HND" if trtllm else metadata.kv_layout
        self.block_size = manager.tokens_per_block
        # Native BF16 paged-context FMHA uses 32-token pages.
        self.tokens_per_block = min(self.block_size, 32) if trtllm else self.block_size
        self.history = manager.get_buffers(layer_idx, kv_layout=metadata.kv_layout)
        (self.high, _), (self.scales, _) = manager.get_mixed_precision_buffers(
            layer_idx, metadata.request_ids, metadata.kv_layout
        )
        if self.layout != metadata.kv_layout:
            # Reinterpret pool strides without moving stored KV.
            self.history = self.history.transpose(2, 3)
            self.high = self.high.transpose(2, 3)
            self.scales = self.scales.transpose(2, 3)
        self.max_seq_len = manager.max_seq_len
        self.max_blocks_per_seq = (
            self.max_seq_len + self.tokens_per_block - 1
        ) // self.tokens_per_block
        self.kv_factor = manager.kv_factor
        self.layer_offsets = {layer_idx: 0}
        self.max_attention_window_vec = (None,)
        # Fixed per-request spans keep addresses unchanged as sequences grow.
        self.page_starts = [row * self.max_blocks_per_seq for row in range(metadata.num_seqs + 1)]
        shape = [self.page_starts[-1], *self.high.shape[1:]]
        shape[3 if self.layout == "HND" else 2] = self.tokens_per_block
        self.pool = manager.get_staging_buffer(shape, self.high.dtype)
        self.dtype = manager.high_dtype
        self.kv_cache_pool_pointers = torch.tensor([[self.pool.data_ptr(), 0]], dtype=torch.int64)
        self.kv_cache_pool_mapping = torch.zeros((1, 2), dtype=torch.int32)
        self.metadata = None

    def update(self, metadata: AttentionMetadata, layer_idx: int) -> None:
        manager = metadata.kv_cache_manager
        (_, high_ids), (_, scale_ids) = manager.get_mixed_precision_buffers(
            layer_idx, metadata.request_ids, metadata.kv_layout
        )
        history_ids = manager.get_batch_cache_indices(metadata.request_ids, layer_idx)
        cached_lengths = metadata.kv_cache_params.num_cached_tokens_per_seq
        seq_lens = metadata.seq_lens.tolist()
        lengths = [a + b for a, b in zip(cached_lengths, seq_lens)]
        self.page_counts = [
            (n + self.tokens_per_block - 1) // self.tokens_per_block for n in lengths
        ]
        batch = len(lengths)
        buffers = manager.mixed_precision_buffers
        max_pages = (self.max_seq_len + self.block_size - 1) // self.block_size
        tables = torch.full(
            (3, batch, max_pages), -1, dtype=torch.int32, pin_memory=prefer_pinned()
        )
        for role, rows in enumerate((history_ids, scale_ids, high_ids)):
            for row, pages in enumerate(rows):
                count = (lengths[row] + self.block_size - 1) // self.block_size
                if role < 2 and (len(pages) < count or any(page < 0 for page in pages[:count])):
                    raise ValueError("Mixed-precision primary KV and scales must cover every token")
                count = min(len(pages), max_pages)
                tables[role, row, :count] = torch.tensor(pages[:count], dtype=torch.int32)
        self.page_tables = buffers.get_buffer(
            (3, manager.max_batch_size, max_pages),
            torch.int32,
            f"mixed_page_tables_{layer_idx}",
            metadata.is_cuda_graph,
        )[:, :batch]
        self.page_tables.copy_(tables, non_blocking=True)
        self.lengths = buffers.get_buffer(
            (2, manager.max_batch_size), torch.int32, "mixed_lengths", metadata.is_cuda_graph
        )[:, :batch]
        self.lengths.copy_(
            maybe_pin_memory(torch.tensor([cached_lengths, lengths], dtype=torch.int32)),
            non_blocking=True,
        )
        self.max_new_tokens_per_request = max(seq_lens, default=0)

        from .flashinfer import FlashInferAttentionMetadata

        if isinstance(metadata, FlashInferAttentionMetadata):
            if self.metadata is None:
                self.metadata = replace(metadata, kv_cache_manager=self, cuda_graph_buffers=None)
            else:
                self.metadata.seq_lens = metadata.seq_lens
                self.metadata.seq_lens_kv = metadata.seq_lens_kv
                self.metadata.num_contexts = metadata.num_contexts
                self.metadata.kv_cache_params = metadata.kv_cache_params
                self.metadata.request_ids = metadata.request_ids
                self.metadata.prompt_lens = metadata.prompt_lens
                self.metadata.position_ids = metadata.position_ids
                self.metadata.padded_num_tokens = metadata.padded_num_tokens
            self.metadata.prepare()
        else:
            self.metadata = copy.copy(metadata)
            self.metadata.kv_cache_manager = self
            self.metadata.kv_layout = self.layout
            self.metadata.use_paged_context_fmha = True
            offsets = self.block_offsets()
            block_offsets = buffers.get_buffer(
                (1, manager.max_batch_size, 2, self.max_blocks_per_seq),
                torch.int32,
                f"mixed_block_offsets_{layer_idx}",
                metadata.is_cuda_graph,
            )[:, : len(lengths)]
            block_offsets.copy_(offsets, non_blocking=True)
            self.metadata.kv_cache_block_offsets = block_offsets
        self.metadata.mixed_precision_kv_views = None

    def materialize(self) -> None:
        from ...cute_dsl_kernels.mixed_precision_kv import materialize_mixed_precision_kv

        materialize_mixed_precision_kv(
            self.pool,
            self.history,
            self.scales,
            self.high,
            self.page_tables,
            self.lengths,
            self.layout,
        )

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> torch.Tensor:
        return self.pool if kv_layout == self.layout else self.pool.transpose(2, 3)

    def get_attention_op_num_blocks(self, local_layer_idx: int) -> int:
        return self.pool.shape[0] * self.pool.shape[1]

    def get_attention_op_kv_page_offset(self, local_layer_idx: int) -> int:
        return self.pool.shape[1] - 1

    def get_batch_cache_indices(self, request_ids: list[int], layer_idx: int) -> list[list[int]]:
        return [
            list(range(start, start + count))
            for start, count in zip(self.page_starts, self.page_counts)
        ]

    @property
    def blocks_in_primary_pool(self) -> int:
        return self.pool.shape[0]

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

    def block_offsets(self) -> torch.Tensor:
        ordinals = torch.arange(self.max_blocks_per_seq, dtype=torch.int32)
        pages = torch.tensor(self.page_starts[:-1], dtype=torch.int32)[:, None] + ordinals
        kv_factor = self.pool.shape[1]
        offsets = torch.stack((kv_factor * pages, kv_factor * pages + kv_factor - 1), dim=1)
        valid = ordinals < torch.tensor(self.page_counts, dtype=torch.int32)[:, None]
        return maybe_pin_memory(offsets.masked_fill(~valid[:, None], -1).unsqueeze(0))

    def writeback(self) -> None:
        """Write every new token to the primary pool and its retained high page."""
        from ...cute_dsl_kernels.mixed_precision_kv import writeback_mixed_precision_kv

        writeback_mixed_precision_kv(
            self.pool,
            self.history,
            self.scales,
            self.high,
            self.page_tables,
            self.lengths,
            self.max_new_tokens_per_request,
            self.layout,
        )

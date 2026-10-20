# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-dtype staging of mixed-precision KV layers for one attention batch."""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._utils import prefer_pinned

from ...memory_buffer_utils import Buffers


class MixedPrecisionKVStaging:
    """Present the mixed-precision pools of a batch as one native paged KV cache.

    ``KVCacheManagerV2`` stores each mixed-precision layer in two pools: the
    full history in a low-precision pool and the first_n/last_n tokens of every
    sequence in a high-precision pool. Attention kernels read a single cache in
    the attention dtype, so this object owns a staging copy for the current
    batch: :meth:`convert` gathers both pools into it before attention runs and
    writes the newly produced tokens back afterwards. It also implements the
    part of the cache manager interface that the attention backends read while
    the staging copy is in use.
    """

    def __init__(
        self,
        manager,
        buffers: Buffers,
        *,
        rows: int,
        layout: str,
        pool_layout: str,
        tokens_per_block: int,
    ) -> None:
        """Allocate the staging copy and the pool views for ``rows`` requests.

        ``buffers`` is the attention metadata's buffer pool; the staging copy is
        taken from it once, sized for this rank: ``rows`` requests (the
        metadata's sequence capacity) times the manager's ``max_seq_len`` times
        the largest local mixed-precision layer page, whose head count is
        already sharded by the manager. ``layout`` and ``tokens_per_block``
        describe the paged cache the attention kernels read; the kernels may
        need smaller pages than the manager stores (TRTLLM paged-context FMHA
        uses 32-token pages). ``pool_layout`` is the layout the backend writes
        the manager's pools in.
        """
        self.manager = manager
        self.buffers = buffers
        self.rows = rows
        self.layout = layout
        self.block_size = manager.tokens_per_block
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = manager.mixed_precision_max_seq_len
        self.max_blocks_per_seq = -(-self.max_seq_len // tokens_per_block)
        self.kv_factor = manager.kv_factor
        # Fixed per-request spans keep addresses unchanged as sequences grow.
        self.page_starts = [row * self.max_blocks_per_seq for row in range(rows + 1)]
        # layer_idx -> (staging shape, low pool, high pool), all presented in ``layout``.
        pools: Dict[int, Tuple[List[int], torch.Tensor, torch.Tensor]] = {}
        self.max_page_bytes = 0
        for layer_idx, local in manager.layer_offsets.items():
            if local not in manager.high_layers:
                continue
            low, high = (
                manager.get_buffers(layer_idx, pool_layout, mixed_precision_pool=pool)
                for pool in ("low", "high")
            )
            if layout != pool_layout:
                low, high = low.transpose(2, 3), high.transpose(2, 3)
            shape = [self.page_starts[-1], *high.shape[1:]]
            shape[3 if layout == "HND" else 2] = tokens_per_block
            pools[layer_idx] = (shape, low, high)
            self.max_page_bytes = max(
                self.max_page_bytes, math.prod(shape[1:]) * high.element_size()
            )
        self.staging_bytes = self.page_starts[-1] * self.max_page_bytes
        storage = buffers.get_buffer([self.staging_bytes], torch.uint8, "mixed_kv_staging", True)
        self.pools: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
            layer_idx: (Buffers._view_as(storage, shape, high.dtype), low, high)
            for layer_idx, (shape, low, high) in pools.items()
        }
        # Cache manager interface read by the attention backends.
        self.layer_offsets = {layer: layer for layer in self.pools}
        self.max_attention_window_vec = (None,) * (max(self.pools) + 1)
        self.dtype = manager.dtype
        self.low_precision_dtype = manager.low_precision_dtype
        self.kv_cache_pool_pointers = torch.tensor([[storage.data_ptr(), 0]], dtype=torch.int64)
        self.kv_cache_pool_mapping = torch.zeros((max(self.pools) + 1, 2), dtype=torch.int32)
        self.page_counts: List[int] = []
        self.page_tables: Dict[int, torch.Tensor] = {}
        self.lengths: Optional[torch.Tensor] = None
        self.max_new_tokens_per_request = 0

    def update(
        self,
        request_ids: Sequence[int],
        cached_lengths: Sequence[int],
        seq_lens: Sequence[int],
        *,
        capture_graph: bool,
    ) -> None:
        """Load this batch's page tables and lengths into device buffers."""
        manager = self.manager
        buffers = self.buffers
        lengths = [cached + new for cached, new in zip(cached_lengths, seq_lens)]
        self.page_counts = [-(-length // self.tokens_per_block) for length in lengths]
        batch = len(lengths)
        max_pages = -(-self.max_seq_len // self.block_size)
        self.page_tables = {}
        for layer_idx in self.pools:
            page_tables = manager.get_mixed_precision_page_tables(request_ids, layer_idx)
            tables = torch.full(
                (2, batch, max_pages), -1, dtype=torch.int32, pin_memory=prefer_pinned()
            )
            for role, rows in enumerate(page_tables):
                for row, pages in enumerate(rows):
                    count = min(len(pages), max_pages)
                    tables[role, row, :count] = torch.tensor(pages[:count], dtype=torch.int32)
            self.page_tables[layer_idx] = buffers.get_buffer(
                (2, self.rows, max_pages),
                torch.int32,
                f"mixed_page_tables_{layer_idx}",
                capture_graph,
            )[:, :batch]
            self.page_tables[layer_idx].copy_(tables, non_blocking=True)
        self.lengths = buffers.get_buffer(
            (2, self.rows), torch.int32, "mixed_lengths", capture_graph
        )[:, :batch]
        self.lengths.copy_(
            torch.tensor(
                [list(cached_lengths), lengths], dtype=torch.int32, pin_memory=prefer_pinned()
            ),
            non_blocking=True,
        )
        self.max_new_tokens_per_request = max(seq_lens, default=0)

    def convert(self, layer_idx: int, *, writeback: bool = False) -> None:
        """Gather both pools into staging, or write the new tokens back to them."""
        from ...cute_dsl_kernels.mixed_precision_kv import convert_mixed_precision_kv

        staging, low, high = self.pools[layer_idx]
        # The kernel derives each request's staging span from the page count it
        # is handed, so pass only the rows of this batch.
        staging = staging[: len(self.page_counts) * self.max_blocks_per_seq]
        convert_mixed_precision_kv(
            writeback,
            self.max_new_tokens_per_request
            if writeback
            else self.max_blocks_per_seq * self.tokens_per_block,
            staging,
            low,
            high,
            self.page_tables[layer_idx],
            self.lengths,
            self.layout,
            low_dtype=self.low_precision_dtype,
        )

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> torch.Tensor:
        staging = self.pools[layer_idx][0]
        return staging if kv_layout == self.layout else staging.transpose(2, 3)

    def get_attention_op_num_blocks(self, local_layer_idx: int) -> int:
        return self.page_starts[-1] * self.kv_factor

    def get_attention_op_kv_page_offset(self, local_layer_idx: int) -> int:
        return self.kv_factor - 1

    def get_batch_cache_indices(self, request_ids: List[int], layer_idx: int) -> List[List[int]]:
        return [
            list(range(start, start + count))
            for start, count in zip(self.page_starts, self.page_counts)
        ]

    def get_batch_cache_indices_flat(
        self, request_ids: List[int], num_blocks: List[int], layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        return torch.tensor(
            [
                page
                for start, count in zip(self.page_starts, num_blocks)
                for page in range(start, start + count)
            ],
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
        )

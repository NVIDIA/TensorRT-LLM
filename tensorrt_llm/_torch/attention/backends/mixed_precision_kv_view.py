# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare one ordinary paged cache for optimized mixed-precision Attention.

The native manager owns both REQUIRED history and SWA/sink pages. This
eager implementation stages only the current batch, one layer at a time. The
backend writes new, post-RoPE K/V into that staging cache; writeback then stores
both representations without quantizing the high copy through the low copy.
"""

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.bindings import DataType

from ...pyexecutor.kv_cache.kv_cache_manager_v2 import MixedPrecisionKVCacheManager

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata


class MixedPrecisionKVView:
    """Present temporary paged KV to Attention and return new writes to its pools."""

    @classmethod
    def create(
        cls,
        metadata: "AttentionMetadata",
        layer_idx: int,
        dtype: torch.dtype,
        *,
        history_dequant_scale: torch.Tensor | None = None,
        max_page_size: int | None = None,
    ) -> "MixedPrecisionKVView | None":
        manager = metadata.kv_cache_manager
        if not isinstance(manager, MixedPrecisionKVCacheManager):
            return None
        if manager.layer_offsets[layer_idx] not in manager.high_layers:
            return None
        if metadata.is_cuda_graph or torch.cuda.is_current_stream_capturing():
            raise NotImplementedError(
                "Mixed-precision paged-cache staging requires eager execution"
            )
        if metadata.is_cross or manager.kv_factor != 2 or manager.max_beam_width != 1:
            raise NotImplementedError("Mixed-precision staging requires single-beam dense self K/V")
        if manager.max_draft_len or metadata.kv_cache_params.use_full_generation_page_table:
            raise NotImplementedError(
                "Mixed-precision staging does not yet support speculative or context-parallel steps"
            )
        if manager.dtype not in (DataType.FP8, DataType.BF16, DataType.HALF):
            raise NotImplementedError("Mixed-precision staging requires FP8, BF16 or FP16 hot K/V")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Mixed-precision staging requires FP16 or BF16 computation")
        window = manager.get_high_precision_window(
            layer_idx, metadata.request_ids, metadata.kv_layout
        )
        return cls(metadata, layer_idx, dtype, window, history_dequant_scale, max_page_size)

    def __init__(
        self,
        metadata: "AttentionMetadata",
        layer_idx: int,
        dtype: torch.dtype,
        window: tuple[torch.Tensor, list[list[int]]],
        history_dequant_scale: torch.Tensor | None,
        max_page_size: int | None,
    ) -> None:
        self.manager = metadata.kv_cache_manager
        self.layer_idx = layer_idx
        self.request_ids = metadata.request_ids
        self.layout = metadata.kv_layout
        self.block_size = self.manager.tokens_per_block
        self.stage_block_size = min(self.block_size, max_page_size or self.block_size)
        if self.block_size % self.stage_block_size:
            raise ValueError("Staging pages must divide the owning cache's block size")
        self.history = self.manager.get_buffers(layer_idx, kv_layout=self.layout)
        self.high, self.high_ids = window
        self.history_ids = self.manager.get_batch_cache_indices(self.request_ids, layer_idx)
        self.cached_lengths = list(metadata.kv_cache_params.num_cached_tokens_per_seq)
        self.query_lengths = metadata.seq_lens.tolist()
        self.lengths = [a + b for a, b in zip(self.cached_lengths, self.query_lengths)]
        self.page_counts = [
            (length + self.stage_block_size - 1) // self.stage_block_size for length in self.lengths
        ]
        self.page_ids = []
        cursor = 0
        for count in self.page_counts:
            self.page_ids.append(list(range(cursor, cursor + count)))
            cursor += count
        shape = [max(1, cursor), *self.high.shape[1:]]
        shape[3 if self.layout == "HND" else 2] = self.stage_block_size
        self.pool = self.manager.get_staging_buffer(shape, dtype)
        self.pool.zero_()
        self.history_scale = (
            history_dequant_scale if self.history.dtype == torch.float8_e4m3fn else None
        )
        self.tokens_per_block = self.stage_block_size
        self.max_seq_len = self.manager.max_seq_len
        self.layer_offsets = {layer_idx: 0}
        self.kv_cache_pool_pointers = torch.tensor([[self.pool.data_ptr(), 0]], dtype=torch.int64)
        self.kv_cache_pool_mapping = torch.zeros((1, 2), dtype=torch.int32)
        self.max_attention_window_vec = (None,)
        for row, cached in enumerate(self.cached_lengths):
            if self.lengths[row] > len(self.history_ids[row]) * self.block_size:
                raise ValueError("Mixed-precision history does not cover the requested KV length")
            for begin in range(0, cached, self.stage_block_size):
                ordinal, offset = divmod(begin, self.block_size)
                stage_id = self.page_ids[row][begin // self.stage_block_size]
                count = min(self.stage_block_size, cached - begin)
                low_id = self.history_ids[row][ordinal]
                if low_id < 0:
                    raise ValueError("Mixed-precision Attention cannot read a missing history page")
                low = self._tokens(self.history, low_id, offset, count).to(dtype)
                if self.history_scale is not None:
                    low = (low.float() * self.history_scale).to(dtype)
                target = self._tokens(self.pool, stage_id, 0, count)
                target.copy_(low)
                high_id = self.high_ids[row][ordinal] if ordinal < len(self.high_ids[row]) else -1
                if high_id < 0:
                    continue
                target.copy_(self._tokens(self.high, high_id, offset, count))

    @property
    def dtype(self) -> DataType:
        return torch_dtype_to_binding(self.pool.dtype)

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> torch.Tensor:
        return self.pool if kv_layout == self.layout else self.pool.transpose(2, 3)

    def get_attention_op_num_blocks(self, local_layer_idx: int) -> int:
        return self.pool.shape[0] * 2

    def get_attention_op_kv_page_offset(self, local_layer_idx: int) -> int:
        return 1

    def _tokens(self, pool: torch.Tensor, page: int, offset: int, count: int) -> torch.Tensor:
        if self.layout == "HND":
            return pool[page, :, :, offset : offset + count, :]
        return pool[page, :, offset : offset + count, :, :]

    def block_offsets(self) -> torch.Tensor:
        offsets = torch.full(
            (
                1,
                len(self.page_ids),
                2,
                (self.manager.max_seq_len + self.stage_block_size - 1) // self.stage_block_size,
            ),
            -1,
            dtype=torch.int32,
        )
        for row, ids in enumerate(self.page_ids):
            indices = torch.tensor(ids, dtype=torch.int32)
            offsets[0, row, 0, : len(ids)] = 2 * indices
            offsets[0, row, 1, : len(ids)] = 2 * indices + 1
        return offsets.to(self.pool.device)

    def writeback(self) -> None:
        """Publish only newly computed tokens; never overwrite shared history."""
        for row, (begin, end) in enumerate(zip(self.cached_lengths, self.lengths)):
            while begin < end:
                ordinal, offset = divmod(begin, self.block_size)
                stage_ordinal, stage_offset = divmod(begin, self.stage_block_size)
                count = min(self.stage_block_size - stage_offset, end - begin)
                values = self._tokens(
                    self.pool, self.page_ids[row][stage_ordinal], stage_offset, count
                )
                low = values
                if self.history_scale is not None:
                    low = values.float() / self.history_scale
                if self.history.dtype == torch.float8_e4m3fn:
                    low = low.clamp(-448, 448)
                self._tokens(self.history, self.history_ids[row][ordinal], offset, count).copy_(low)
                if ordinal < len(self.high_ids[row]) and self.high_ids[row][ordinal] >= 0:
                    self._tokens(self.high, self.high_ids[row][ordinal], offset, count).copy_(
                        values.clamp(-448, 448)
                        if self.high.dtype == torch.float8_e4m3fn
                        else values
                    )
                begin += count

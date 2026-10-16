# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare one ordinary paged cache for optimized mixed-precision Attention.

The native manager owns both REQUIRED history and SWA/sink pages. This
eager implementation stages only the current batch, one layer at a time. The
backend writes new, post-RoPE K/V into that staging cache; writeback then stores
both representations without quantizing the high copy through the low copy.
"""

from itertools import accumulate
from typing import TYPE_CHECKING, NamedTuple

import torch

from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.bindings import DataType

from ...pyexecutor.kv_cache.kv_cache_manager_v2 import MixedPrecisionKVCacheManager
from ..kv_cache_conversion import dequantize_nvfp4_kv, encode_high_precision_kv, quantize_nvfp4_kv

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionBackend, AttentionMetadata


class _TokenLocations(NamedTuple):
    staging: torch.Tensor
    history: torch.Tensor
    scales: torch.Tensor
    high: torch.Tensor


class MixedPrecisionKVView:
    """Present temporary paged KV to Attention and return new writes to its pools."""

    @classmethod
    def create(
        cls,
        attention: "AttentionBackend",
        metadata: "AttentionMetadata",
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        *,
        history_dequant_scale: torch.Tensor | None = None,
        max_page_size: int | None = None,
    ) -> "MixedPrecisionKVView | None":
        manager = metadata.kv_cache_manager
        if not isinstance(manager, MixedPrecisionKVCacheManager):
            return None
        layer_idx = attention.layer_idx
        if manager.layer_offsets[layer_idx] not in manager.high_layers:
            return None
        if metadata.is_cuda_graph or torch.cuda.is_current_stream_capturing():
            raise NotImplementedError(
                "Mixed-precision paged-cache staging requires eager execution"
            )
        if metadata.is_cross or manager.kv_factor != 2:
            raise NotImplementedError("Mixed-precision staging requires single-beam dense self K/V")
        if q.dtype != manager.compute_dtype:
            raise ValueError("Attention input dtype differs from its reserved mixed-precision view")
        fused_qkv = (
            attention.support_fused_qkv()
            and k is None
            and v is None
            and q.shape[-1]
            == (attention.num_heads + 2 * attention.num_kv_heads) * attention.head_dim
        )
        if not (fused_qkv or (k is not None and v is not None)):
            raise ValueError("Mixed-precision staging requires new K/V inputs")
        return cls(metadata, layer_idx, q.dtype, history_dequant_scale, max_page_size)

    def __init__(
        self,
        metadata: "AttentionMetadata",
        layer_idx: int,
        dtype: torch.dtype,
        history_dequant_scale: torch.Tensor | None,
        max_page_size: int | None,
    ) -> None:
        manager = metadata.kv_cache_manager
        self.layout = metadata.kv_layout
        self.block_size = manager.tokens_per_block
        self.tokens_per_block = min(self.block_size, max_page_size or self.block_size)
        if self.block_size % self.tokens_per_block:
            raise ValueError("Staging pages must divide the owning cache's block size")
        self.history = manager.get_buffers(layer_idx, kv_layout=self.layout)
        (self.high, high_ids), (self.scales, scale_ids) = manager.get_mixed_precision_buffers(
            layer_idx, metadata.request_ids, self.layout
        )
        history_ids = manager.get_batch_cache_indices(metadata.request_ids, layer_idx)
        cached_lengths = metadata.kv_cache_params.num_cached_tokens_per_seq
        lengths = [a + b for a, b in zip(cached_lengths, metadata.seq_lens.tolist())]
        self.page_counts = [
            (length + self.tokens_per_block - 1) // self.tokens_per_block for length in lengths
        ]
        self.page_starts = list(accumulate(self.page_counts, initial=0))
        shape = [max(1, self.page_starts[-1]), *self.high.shape[1:]]
        shape[3 if self.layout == "HND" else 2] = self.tokens_per_block
        self.pool = manager.get_staging_buffer(shape, dtype)
        self.pool.zero_()
        self.history_scale = (
            torch.ones(1, dtype=torch.float32, device=self.pool.device)
            if history_dequant_scale is None
            else history_dequant_scale.to(device=self.pool.device, dtype=torch.float32).contiguous()
        )
        if self.history_scale.numel() != 1:
            raise ValueError("Dense NVFP4 KV conversion requires one per-tensor scale")
        self.max_seq_len = manager.max_seq_len
        self.layer_offsets = {layer_idx: 0}
        self.kv_cache_pool_pointers = torch.tensor([[self.pool.data_ptr(), 0]], dtype=torch.int64)
        self.kv_cache_pool_mapping = torch.zeros((1, 2), dtype=torch.int32)
        self.max_attention_window_vec = (None,)
        tables = (history_ids, scale_ids, high_ids)
        cached = self._locations([0] * len(lengths), cached_lengths, tables)
        self.new_tokens = self._locations(cached_lengths, lengths, tables)
        low_rows = torch.nonzero(cached.high < 0).flatten()
        if low_rows.numel():
            packed = self._gather(self.history, cached.history[low_rows])
            scales = self._gather(self.scales, cached.scales[low_rows])
            values = dequantize_nvfp4_kv(packed, scales, self.history_scale, dtype)
            self._scatter(self.pool, cached.staging[low_rows], values)
        high_rows = torch.nonzero(cached.high >= 0).flatten()
        if high_rows.numel():
            values = self._gather(self.high, cached.high[high_rows]).to(dtype)
            self._scatter(self.pool, cached.staging[high_rows], values)

    @property
    def dtype(self) -> DataType:
        return torch_dtype_to_binding(self.pool.dtype)

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> torch.Tensor:
        return self.pool if kv_layout == self.layout else self.pool.transpose(2, 3)

    def get_attention_op_num_blocks(self, local_layer_idx: int) -> int:
        return self.pool.shape[0] * 2

    def get_attention_op_kv_page_offset(self, local_layer_idx: int) -> int:
        return 1

    def _locations(
        self,
        begins: list[int],
        ends: list[int],
        page_tables: tuple[list[list[int]], list[list[int]], list[list[int]]],
    ) -> _TokenLocations:
        """Map logical tokens to flat token indices in each native pool."""
        tables = []
        for row, (begin, end, history, scales, high) in enumerate(zip(begins, ends, *page_tables)):
            if end > min(len(history), len(scales)) * self.block_size:
                raise ValueError("Mixed-precision history does not cover the requested KV length")
            positions = torch.arange(begin, end)
            ordinals = positions // self.block_size
            history_pages = torch.tensor(history, dtype=torch.int64)[ordinals]
            scale_pages = torch.tensor(scales, dtype=torch.int64)[ordinals]
            if torch.any(history_pages < 0) or torch.any(scale_pages < 0):
                raise ValueError(
                    "Mixed-precision Attention requires existing NVFP4 data and scales"
                )
            high_pages = torch.tensor(high + [-1] * (len(history) - len(high)), dtype=torch.int64)
            offsets = positions % self.block_size
            tables.append(
                torch.stack(
                    (
                        self.page_starts[row] * self.tokens_per_block + positions,
                        history_pages * self.block_size + offsets,
                        scale_pages * self.block_size + offsets,
                        high_pages[ordinals] * self.block_size + offsets,
                    )
                )
            )
        table = torch.cat(tables, dim=1) if tables else torch.empty((4, 0), dtype=torch.int64)
        return _TokenLocations(*table.to(self.pool.device).unbind())

    def _gather(self, pool: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        view = pool.transpose(2, 3) if self.layout == "HND" else pool
        # Byte indexing also works for FP8, without invoking arithmetic on it.
        return view.view(torch.uint8)[indices // view.shape[2], :, indices % view.shape[2]].view(
            pool.dtype
        )

    def _scatter(self, pool: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> None:
        view = pool.transpose(2, 3) if self.layout == "HND" else pool
        view.view(torch.uint8)[indices // view.shape[2], :, indices % view.shape[2]] = (
            values.contiguous().view(torch.uint8)
        )

    def block_offsets(self) -> torch.Tensor:
        width = (self.max_seq_len + self.tokens_per_block - 1) // self.tokens_per_block
        ordinals = torch.arange(width, dtype=torch.int32)
        pages = torch.tensor(self.page_starts[:-1], dtype=torch.int32)[:, None] + ordinals
        offsets = torch.stack((2 * pages, 2 * pages + 1), dim=1)
        valid = ordinals < torch.tensor(self.page_counts, dtype=torch.int32)[:, None]
        return offsets.masked_fill(~valid[:, None], -1).unsqueeze(0).to(self.pool.device)

    def writeback(self) -> None:
        """Publish only newly computed tokens; never overwrite shared history."""
        locations = self.new_tokens
        if not locations.history.numel():
            return
        values = self._gather(self.pool, locations.staging)
        packed, scales = quantize_nvfp4_kv(values, self.history_scale)
        self._scatter(self.history, locations.history, packed)
        self._scatter(self.scales, locations.scales, scales)
        rows = torch.nonzero(locations.high >= 0).flatten()
        if rows.numel():
            high_values = encode_high_precision_kv(values[rows], self.high.dtype)
            self._scatter(self.high, locations.high[rows], high_values)

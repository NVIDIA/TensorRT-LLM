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
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX, DataRole

from ...pyexecutor.kv_cache.kv_cache_manager_v2 import MixedPrecisionKVCacheManager
from ..kv_cache_conversion import dequantize_nvfp4_kv, encode_high_precision_kv, quantize_nvfp4_kv

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata


class _TokenLocations(NamedTuple):
    staging: torch.Tensor
    staging_offset: torch.Tensor
    history: torch.Tensor
    scales: torch.Tensor
    high: torch.Tensor
    offset: torch.Tensor
    high_rows: torch.Tensor


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
        if manager.kv_factor == 1 and manager._uses_native_mixed_precision_rows:
            return None
        if manager.layer_offsets[layer_idx] not in manager.high_layers:
            return None
        if metadata.is_cuda_graph or torch.cuda.is_current_stream_capturing():
            raise NotImplementedError(
                "Mixed-precision paged-cache staging requires eager execution"
            )
        if metadata.is_cross or manager.kv_factor != 2:
            raise NotImplementedError("Mixed-precision staging requires single-beam dense self K/V")
        if dtype != manager.compute_dtype:
            raise ValueError("Attention input dtype differs from its reserved mixed-precision view")
        return cls(metadata, layer_idx, dtype, history_dequant_scale, max_page_size)

    def __init__(
        self,
        metadata: "AttentionMetadata",
        layer_idx: int,
        dtype: torch.dtype,
        history_dequant_scale: torch.Tensor | None,
        max_page_size: int | None,
    ) -> None:
        self.manager = metadata.kv_cache_manager
        self.layout = metadata.kv_layout
        self.block_size = self.manager.tokens_per_block
        self.tokens_per_block = min(self.block_size, max_page_size or self.block_size)
        if self.block_size % self.tokens_per_block:
            raise ValueError("Staging pages must divide the owning cache's block size")
        self.history = self.manager.get_buffers(layer_idx, kv_layout=self.layout)
        (self.high, self.high_ids), (self.scales, self.scale_ids) = (
            self.manager.get_mixed_precision_buffers(layer_idx, metadata.request_ids, self.layout)
        )
        self.history_ids = self.manager.get_batch_cache_indices(metadata.request_ids, layer_idx)
        self.cached_lengths = list(metadata.kv_cache_params.num_cached_tokens_per_seq)
        self.lengths = [a + b for a, b in zip(self.cached_lengths, metadata.seq_lens.tolist())]
        self.page_counts = [
            (length + self.tokens_per_block - 1) // self.tokens_per_block for length in self.lengths
        ]
        self.page_starts = list(accumulate(self.page_counts, initial=0))
        shape = [max(1, self.page_starts[-1]), *self.high.shape[1:]]
        shape[3 if self.layout == "HND" else 2] = self.tokens_per_block
        self.pool = self.manager.get_staging_buffer(shape, dtype)
        self.pool.zero_()
        self.history_scale = (
            torch.ones(1, dtype=torch.float32, device=self.pool.device)
            if history_dequant_scale is None
            else history_dequant_scale.to(device=self.pool.device, dtype=torch.float32).contiguous()
        )
        if self.history_scale.numel() != 1:
            raise ValueError("Dense NVFP4 KV conversion requires one per-tensor scale")
        self.max_seq_len = self.manager.max_seq_len
        self.layer_offsets = {layer_idx: 0}
        self.kv_cache_pool_pointers = torch.tensor([[self.pool.data_ptr(), 0]], dtype=torch.int64)
        self.kv_cache_pool_mapping = torch.zeros((1, 2), dtype=torch.int32)
        self.max_attention_window_vec = (None,)
        cached = self._locations(new_tokens=False)
        self.new_tokens = self._locations(new_tokens=True)
        if cached.history.numel():
            packed = self._gather(self.history, cached.history, cached.offset)
            scales = self._gather(self.scales, cached.scales, cached.offset)
            values = dequantize_nvfp4_kv(packed, scales, self.history_scale, dtype)
            self._scatter(self.pool, cached.staging, cached.staging_offset, values)
        if cached.high_rows.numel():
            rows = cached.high_rows
            values = self._gather(self.high, cached.high[rows], cached.offset[rows]).to(dtype)
            self._scatter(self.pool, cached.staging[rows], cached.staging_offset[rows], values)

    @property
    def dtype(self) -> DataType:
        return torch_dtype_to_binding(self.pool.dtype)

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> torch.Tensor:
        return self.pool if kv_layout == self.layout else self.pool.transpose(2, 3)

    def get_attention_op_num_blocks(self, local_layer_idx: int) -> int:
        return self.pool.shape[0] * 2

    def get_attention_op_kv_page_offset(self, local_layer_idx: int) -> int:
        return 1

    def _locations(self, *, new_tokens: bool) -> _TokenLocations:
        """Build batched token addresses from native page tables on the host."""
        tables = []
        for row, (cached, length) in enumerate(zip(self.cached_lengths, self.lengths)):
            if length > min(len(self.history_ids[row]), len(self.scale_ids[row])) * self.block_size:
                raise ValueError("Mixed-precision history does not cover the requested KV length")
            positions = torch.arange(cached if new_tokens else 0, length if new_tokens else cached)
            ordinals = positions // self.block_size
            history = torch.tensor(self.history_ids[row], dtype=torch.int64)[ordinals]
            scales = torch.tensor(self.scale_ids[row], dtype=torch.int64)[ordinals]
            if torch.any(history < 0) or torch.any(scales < 0):
                raise ValueError(
                    "Mixed-precision Attention requires existing NVFP4 data and scales"
                )
            high = torch.full((len(self.history_ids[row]),), -1, dtype=torch.int64)
            high[: len(self.high_ids[row])] = torch.tensor(self.high_ids[row], dtype=torch.int64)
            tables.append(
                torch.stack(
                    (
                        self.page_starts[row] + positions // self.tokens_per_block,
                        positions % self.tokens_per_block,
                        history,
                        scales,
                        high[ordinals],
                        positions % self.block_size,
                    )
                )
            )
        table = torch.cat(tables, dim=1) if tables else torch.empty((6, 0), dtype=torch.int64)
        high_rows = torch.nonzero(table[4] >= 0).flatten().to(self.pool.device)
        return _TokenLocations(*table.to(self.pool.device).unbind(), high_rows)

    def _gather(
        self, pool: torch.Tensor, pages: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        view = pool.transpose(2, 3) if self.layout == "HND" else pool
        # Byte indexing also works for FP8, without invoking arithmetic on it.
        return view.view(torch.uint8)[pages, :, offsets].view(pool.dtype)

    def _scatter(
        self, pool: torch.Tensor, pages: torch.Tensor, offsets: torch.Tensor, values: torch.Tensor
    ) -> None:
        view = pool.transpose(2, 3) if self.layout == "HND" else pool
        view.view(torch.uint8)[pages, :, offsets] = values.contiguous().view(torch.uint8)

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
        values = self._gather(self.pool, locations.staging, locations.staging_offset)
        packed, scales = quantize_nvfp4_kv(values, self.history_scale)
        self._scatter(self.history, locations.history, locations.offset, packed)
        self._scatter(self.scales, locations.scales, locations.offset, scales)
        if locations.high_rows.numel():
            rows = locations.high_rows
            high_values = encode_high_precision_kv(values[rows], self.high.dtype)
            self._scatter(self.high, locations.high[rows], locations.offset[rows], high_values)


class MixedPrecisionKVRows:
    """Copy fresh model-native rows to SWA and select them after the native gather."""

    @classmethod
    def create(
        cls, metadata: "AttentionMetadata", layer_id: int, role: DataRole
    ) -> "MixedPrecisionKVRows | None":
        manager = metadata.kv_cache_manager
        if (
            not isinstance(manager, MixedPrecisionKVCacheManager)
            or layer_id not in manager.high_layers
        ):
            return None
        if metadata.is_cuda_graph or torch.cuda.is_current_stream_capturing():
            raise NotImplementedError("Mixed-precision row selection requires eager execution")
        return cls(manager, layer_id, role, metadata.request_ids)

    def __init__(
        self,
        manager: MixedPrecisionKVCacheManager,
        layer_id: int,
        role: DataRole,
        request_ids: list[int],
    ) -> None:
        self.pool, tables = manager.get_high_precision_rows(layer_id, role, request_ids)
        self.rows_per_block = manager.high_layouts[layer_id].rows_per_block
        width = max((len(row) for row in tables), default=0)
        table = torch.full((max(1, len(tables)), max(1, width)), BAD_PAGE_INDEX, dtype=torch.int64)
        for index, row in enumerate(tables):
            table[index, : len(row)] = torch.tensor(row, dtype=torch.int64)
        self.table = table.to(self.pool.device)

    def _locations(
        self, requests: torch.Tensor, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        requests, rows = torch.broadcast_tensors(requests, rows)
        requests, rows = requests.reshape(-1).long(), rows.reshape(-1).long()
        blocks = rows // self.rows_per_block
        valid = (requests >= 0) & (requests < self.table.shape[0]) & (rows >= 0)
        valid &= blocks < self.table.shape[1]
        pages = self.table[
            requests.clamp(0, self.table.shape[0] - 1), blocks.clamp(0, self.table.shape[1] - 1)
        ]
        valid &= pages >= 0
        return pages, rows % self.rows_per_block, valid

    def store(
        self,
        values: torch.Tensor,
        requests: torch.Tensor,
        rows: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Store only valid new writes, before any NVFP4 round trip."""
        pages, offsets, selected = self._locations(requests, rows)
        selected &= valid.reshape(-1)
        source = values.reshape(-1, *self.pool.shape[2:])[selected]
        encoded = encode_high_precision_kv(source, self.pool.dtype)
        self.pool.view(torch.uint8)[pages[selected], offsets[selected]] = encoded.view(torch.uint8)

    def overlay(
        self,
        output: torch.Tensor,
        output_rows: torch.Tensor,
        requests: torch.Tensor,
        rows: torch.Tensor,
    ) -> None:
        """Replace gathered low-precision rows only where native SWA retains high KV."""
        pages, offsets, selected = self._locations(requests, rows)
        destinations = output_rows.reshape(-1).long()
        selected &= destinations >= 0
        values = self.pool.view(torch.uint8)[pages[selected], offsets[selected]].view(
            self.pool.dtype
        )
        values = encode_high_precision_kv(values, output.dtype)
        output.view(-1, *self.pool.shape[2:]).view(torch.uint8)[destinations[selected]] = (
            values.view(torch.uint8)
        )

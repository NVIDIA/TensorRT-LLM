# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare one ordinary paged cache for optimized mixed-precision Attention.

The native manager owns both REQUIRED history and SWA/sink pages. Metadata
preparation reserves the current batch's buffers, one layer at a time. The
backend writes new, post-RoPE K/V into that staging cache; writeback then stores
both representations without quantizing the high copy through the low copy.
"""

import copy
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned, torch_dtype_to_binding
from tensorrt_llm.bindings import DataType

if TYPE_CHECKING:
    from .flashinfer import FlashInferAttentionMetadata
    from .interface import AttentionBackend
    from .trtllm import TrtllmAttentionMetadata


def configure_mixed_precision_attention(attention: "AttentionBackend") -> None:
    """Bind an eligible layer's backend to its FP16/BF16 staging cache."""
    from .trtllm import TrtllmAttention

    quant_config = attention.quant_config
    if quant_config is not None and quant_config.layer_quant_mode.has_kv_cache_quant():
        quant_config = copy.copy(quant_config)
        quant_config.kv_cache_quant_algo = None
        # Both cached properties must describe the staging representation.
        quant_config.__dict__.pop("quant_mode", None)
        quant_config.__dict__.pop("layer_quant_mode", None)
        attention.update_quant_config(quant_config)
    if isinstance(attention, TrtllmAttention):
        attention.local_layer_idx = 0


def encode_high_precision_kv(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Store the high copy without passing it through NVFP4."""
    if dtype == torch.float8_e4m3fn and values.dtype != dtype:
        values, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            values.contiguous(), torch.ones(1, dtype=torch.float32, device=values.device)
        )
        return values
    return values.to(dtype)


class MixedPrecisionKVView:
    """Present temporary paged KV to Attention and return new writes to its pools."""

    def prepare(
        self,
        attention: "AttentionBackend",
        *,
        history_dequant_scale: torch.Tensor | None = None,
    ) -> None:
        from .trtllm import TrtllmAttention

        configure_mixed_precision_attention(attention)
        if history_dequant_scale is None and isinstance(attention, TrtllmAttention):
            history_dequant_scale = attention.kv_scale_quant_orig
        if history_dequant_scale is not None:
            self.history_scale.copy_(history_dequant_scale)
        self.materialize()

    @classmethod
    def prepare_batch(
        cls, metadata: "TrtllmAttentionMetadata | FlashInferAttentionMetadata"
    ) -> None:
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
        metadata: "TrtllmAttentionMetadata | FlashInferAttentionMetadata",
        layer_idx: int,
    ) -> None:
        from .trtllm import TrtllmAttentionMetadata

        manager = metadata.kv_cache_manager
        trtllm = isinstance(metadata, TrtllmAttentionMetadata)
        self.layout = "HND" if trtllm else metadata.kv_layout
        self.block_size = manager.tokens_per_block
        self.tokens_per_block = min(self.block_size, 64) if trtllm else self.block_size
        self.history = manager.get_buffers(layer_idx, kv_layout=metadata.kv_layout)
        (self.high, _), (self.scales, _) = manager.get_mixed_precision_buffers(
            layer_idx, metadata.request_ids, metadata.kv_layout
        )
        if self.layout != metadata.kv_layout:
            # Keep the stored layout; only the Attention buffer is materialized in the new layout.
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
        self.pool = manager.get_staging_buffer(shape, manager.compute_dtype)
        self.history_scale = torch.ones(1, dtype=torch.float32, device=self.pool.device)
        self.kv_cache_pool_pointers = torch.tensor([[self.pool.data_ptr(), 0]], dtype=torch.int64)
        self.kv_cache_pool_mapping = torch.zeros((1, 2), dtype=torch.int32)
        self._indices: dict[str, torch.Tensor] = {}
        self.metadata = None

    def update(
        self, metadata: "TrtllmAttentionMetadata | FlashInferAttentionMetadata", layer_idx: int
    ) -> None:
        manager = metadata.kv_cache_manager
        (_, high_ids), (_, scale_ids) = manager.get_mixed_precision_buffers(
            layer_idx, metadata.request_ids, metadata.kv_layout
        )
        history_ids = manager.get_batch_cache_indices(metadata.request_ids, layer_idx)
        cached_lengths = metadata.kv_cache_params.num_cached_tokens_per_seq
        lengths = [a + b for a, b in zip(cached_lengths, metadata.seq_lens.tolist())]
        self.page_counts = [
            (n + self.tokens_per_block - 1) // self.tokens_per_block for n in lengths
        ]
        tables = (history_ids, scale_ids, high_ids)
        cached = self._locations([0] * len(lengths), cached_lengths, tables)
        new = self._locations(cached_lengths, lengths, tables)
        low = cached[:, cached[3] < 0]
        high = cached[:, cached[3] >= 0]
        capacity = len(lengths) * self.max_seq_len if metadata.is_cuda_graph else None
        for name, pool, tokens, source, size in (
            ("read_low", self.history, low[1], True, capacity),
            ("read_scales", self.scales, low[2], True, capacity),
            ("restore_low", self.pool, low[0], False, capacity),
            ("read_high", self.high, high[3], True, capacity),
            ("restore_high", self.pool, high[0], False, capacity),
            ("read_new", self.pool, new[0], True, None),
            ("write_low", self.history, new[1], False, None),
            ("write_scales", self.scales, new[2], False, None),
            ("write_high", self.high, new[3], False, None),
        ):
            indices = self._row_indices(pool, tokens)
            rows_per_token = self.kv_factor * self.high.shape[2 if self.layout == "HND" else 3]
            count = size * rows_per_token if size is not None else indices.numel()
            host = torch.full(
                (count,), 0 if source else -1, dtype=torch.int64, pin_memory=prefer_pinned()
            )
            host[: indices.numel()] = indices.clamp_min(0) if source else indices
            if name not in self._indices:
                self._indices[name] = torch.empty(count, dtype=torch.int64, device=self.pool.device)
            self._indices[name].copy_(host, non_blocking=True)

        # Reuse the existing workspace owner; layers execute sequentially.
        buffers = manager.mixed_precision_buffers

        def workspace(name, rows, width, dtype):
            return buffers.get_buffer((rows, width), dtype, name, metadata.is_cuda_graph)

        width = self.high.shape[-1]
        low_rows = self._indices["read_low"].numel()
        high_rows = self._indices["read_high"].numel()
        new_rows = self._indices["read_new"].numel()
        self.read_low = workspace("mixed_read_low", low_rows, width // 2, torch.uint8)
        self.read_scales = workspace("mixed_read_scales", low_rows, width // 16, torch.uint8)
        self.restored_low = workspace("mixed_restored_low", low_rows, width, self.pool.dtype)
        self.read_high = workspace("mixed_read_high", high_rows, width, self.high.dtype)
        self.restored_high = workspace("mixed_restored_high", high_rows, width, self.pool.dtype)
        self.new_values = workspace("mixed_new", new_rows, width, self.pool.dtype)
        self.new_low = workspace("mixed_new_low", new_rows, width // 2, torch.uint8)
        self.new_scales = workspace("mixed_new_scales", new_rows, width // 16, torch.uint8)

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
            if "block_offsets" not in self._indices:
                self._indices["block_offsets"] = torch.empty_like(offsets, device=self.pool.device)
            self._indices["block_offsets"].copy_(offsets, non_blocking=True)
            self.metadata.kv_cache_block_offsets = self._indices["block_offsets"]
        self.metadata.mixed_precision_kv_views = {}

    def materialize(self) -> None:
        from flashinfer.quantization.fp4_quantization import get_fp4_kv_dequantization_module

        self.pool.zero_()
        if self.read_low.numel():
            self._gather(self.history, "read_low", self.read_low)
            self._gather(self.scales, "read_scales", self.read_scales)
            get_fp4_kv_dequantization_module().nvfp4_kv_dequant(
                self.read_low, self.read_scales, self.history_scale, self.restored_low
            )
            self._scatter(self.pool, "restore_low", self.restored_low)
        if self.read_high.numel():
            self._gather(self.high, "read_high", self.read_high)
            self.restored_high.copy_(self.read_high)
            self._scatter(self.pool, "restore_high", self.restored_high)

    @property
    def dtype(self) -> DataType:
        return torch_dtype_to_binding(self.pool.dtype)

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

    def _locations(
        self,
        begins: list[int],
        ends: list[int],
        page_tables: tuple[list[list[int]], list[list[int]], list[list[int]]],
    ) -> torch.Tensor:
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
        return table

    def _row_indices(self, pool: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Prepare physical head-row addresses on the host, outside Attention."""
        token_axis, head_axis = (3, 2) if self.layout == "HND" else (2, 3)
        kv = torch.arange(pool.shape[1])[None, :, None]
        head = torch.arange(pool.shape[head_axis])[None, None, :]
        addresses = (
            (tokens // pool.shape[token_axis])[:, None, None] * pool.stride(0)
            + (tokens % pool.shape[token_axis])[:, None, None] * pool.stride(token_axis)
            + kv * pool.stride(1)
            + head * pool.stride(head_axis)
        ) // pool.shape[-1]
        return addresses.masked_fill(tokens[:, None, None] < 0, -1).flatten()

    @staticmethod
    def _rows(pool: torch.Tensor) -> torch.Tensor:
        width = pool.shape[-1]
        return pool.as_strided((pool.numel() // width, width), (width, 1)).view(torch.uint8)

    def _gather(self, pool: torch.Tensor, name: str, output: torch.Tensor) -> None:
        torch.index_select(self._rows(pool), 0, self._indices[name], out=output.view(torch.uint8))

    def _scatter(self, pool: torch.Tensor, name: str, values: torch.Tensor) -> None:
        from .sparse.csa2.kernel import scatter_packed_rows

        scatter_packed_rows(self._rows(pool), self._indices[name], values.view(torch.uint8))

    def block_offsets(self) -> torch.Tensor:
        width = (self.max_seq_len + self.tokens_per_block - 1) // self.tokens_per_block
        ordinals = torch.arange(width, dtype=torch.int32)
        pages = torch.tensor(self.page_starts[:-1], dtype=torch.int32)[:, None] + ordinals
        kv_factor = self.pool.shape[1]
        offsets = torch.stack((kv_factor * pages, kv_factor * pages + kv_factor - 1), dim=1)
        valid = ordinals < torch.tensor(self.page_counts, dtype=torch.int32)[:, None]
        return maybe_pin_memory(offsets.masked_fill(~valid[:, None], -1).unsqueeze(0))

    def writeback(self) -> None:
        """Publish only newly computed tokens; never overwrite shared history."""
        from flashinfer.quantization.fp4_quantization import get_fp4_kv_quantization_module

        if not self.new_values.numel():
            return
        self._gather(self.pool, "read_new", self.new_values)
        get_fp4_kv_quantization_module().nvfp4_kv_quant(
            self.new_values, self.history_scale, self.new_low, self.new_scales
        )
        self._scatter(self.history, "write_low", self.new_low)
        self._scatter(self.scales, "write_scales", self.new_scales)
        high_values = encode_high_precision_kv(self.new_values, self.high.dtype)
        self._scatter(self.high, "write_high", high_values)

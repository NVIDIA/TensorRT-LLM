# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full KV history plus a native SWA/sink pool retaining original-precision KV.

Both groups are REQUIRED: ordinary radix matching and SWA coverage checks
determine reuse together. Allocation, commits, offload and eviction remain
native. The window controls storage, not the model's Attention mask.
"""

import math
from collections.abc import Sequence
from dataclasses import replace

import torch

from tensorrt_llm._utils import TensorWrapper, binding_to_torch_dtype, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    AttentionReusePolicy,
    BufferConfig,
    GpuCacheTierConfig,
    KVCacheManagerConfig,
    PageIndexMode,
)

from .kv_cache_manager_v2 import KVCacheManagerV2, Role


class MixedPrecisionKVCacheManager(KVCacheManagerV2):
    """Internal eager prototype using existing full-history and SWA lifecycles.

    ``first_n`` is the sink length; ``last_n`` counts previously written tokens.
    Counts are block aligned. High windows are stored and restored unchanged,
    including when a cold-page codec compresses the full-history pool.
    """

    def __init__(
        self,
        *args,
        first_n: int = 256,
        last_n: int = 256,
        high_dtype: DataType = DataType.BF16,
        **kwargs,
    ) -> None:
        self.first_n = first_n
        self.last_n = last_n
        self.high_dtype = high_dtype
        self.high_layers: dict[int, int] = {}
        self._staging_bytes = 0
        self._staging_buffer: torch.Tensor | None = None
        self._staging_stream: int | None = None
        super().__init__(*args, **kwargs)

    def _build_cache_config(self, config: KVCacheManagerConfig) -> KVCacheManagerConfig:
        config = super()._build_cache_config(config)
        if any(n < 0 or n % self.tokens_per_block for n in (self.first_n, self.last_n)):
            raise ValueError("First/Last counts must be nonnegative multiples of tokens_per_block")
        if not (self.first_n or self.last_n):
            raise ValueError("Mixed precision requires a nonzero First or Last count")
        if (
            self.kv_factor != 2
            or self.max_beam_width != 1
            or self.max_draft_len
            or self.is_disagg
            or self.mapping.cp_size != 1
        ):
            raise NotImplementedError("Mixed precision currently requires single-beam dense K/V")
        if self.dtype not in (DataType.FP8, DataType.BF16, DataType.HALF):
            raise NotImplementedError("Mixed precision requires FP8, BF16 or FP16 hot K/V")
        if self.high_dtype not in (DataType.FP8, DataType.BF16, DataType.HALF):
            raise ValueError("The high pool must use FP8, BF16 or FP16")
        if config.enable_swa_scratch_reuse:
            raise NotImplementedError("Mixed-precision staging requires ordinary SWA page tables")
        high_bytes = torch.empty((), dtype=binding_to_torch_dtype(self.high_dtype)).element_size()
        low_bytes = torch.empty((), dtype=binding_to_torch_dtype(self.dtype)).element_size()
        if high_bytes < low_bytes:
            raise ValueError("The high pool cannot have a narrower dtype than the history pool")
        layers = list(config.layers)
        next_id = max(int(layer.layer_id) for layer in layers) + 1
        largest_page = 0
        for layer in config.layers:
            if (
                not isinstance(layer, AttentionLayerConfig)
                or layer.sliding_window_size is not None
                or layer.reuse_policy != AttentionReusePolicy.REQUIRED
                or {buffer.role for buffer in layer.buffers} != {Role.KEY, Role.VALUE}
                or any(buffer.is_sparse or not buffer.size for buffer in layer.buffers)
            ):
                raise NotImplementedError(
                    "Mixed precision currently requires full dense K/V layers"
                )
            buffers = [
                BufferConfig(role=buffer.role, size=buffer.size // low_bytes * high_bytes)
                for buffer in layer.buffers
            ]
            self.high_layers[int(layer.layer_id)] = next_id
            layers.append(
                AttentionLayerConfig(
                    layer_id=next_id,
                    buffers=buffers,
                    # Native SWA includes the next query's write position.
                    sliding_window_size=self.last_n + 1,
                    num_sink_tokens=self.first_n,
                    reuse_policy=AttentionReusePolicy.REQUIRED,
                )
            )
            next_id += 1
            largest_page = max(largest_page, sum(buf.size for buf in buffers) // high_bytes)

        # One reusable layer-sized BF16/FP16 view, including all reused tokens.
        pages = self.max_batch_size * math.ceil(self.max_seq_len / self.tokens_per_block)
        self._staging_bytes = pages * largest_page * 2
        reserve = self._staging_bytes + largest_page * 12
        tiers = []
        for tier in config.cache_tiers:
            if isinstance(tier, GpuCacheTierConfig):
                if tier.quota <= reserve:
                    raise ValueError("KV budget cannot cover mixed-precision Attention staging")
                tier = GpuCacheTierConfig(quota=tier.quota - reserve)
            tiers.append(tier)
        return replace(config, layers=layers, cache_tiers=tiers)

    def _cold_page_codec_config(self, config: KVCacheManagerConfig) -> KVCacheManagerConfig:
        # Only full history enters the quantizer. The native codec copies
        # undeclared SWA groups byte-for-byte, without changing any quantizer.
        high_ids = set(self.high_layers.values())
        return replace(
            config, layers=[layer for layer in config.layers if layer.layer_id not in high_ids]
        )

    def get_high_precision_window(
        self, layer_idx: int, request_ids: list[int], kv_layout: str
    ) -> tuple[torch.Tensor, list[list[int]]]:
        """Return native SWA storage and its logical-to-physical page mapping."""
        local = self.layer_offsets[layer_idx]
        high_id = self.high_layers[local]
        address = self.impl.get_mem_pool_base_address(high_id, Role.KEY, PageIndexMode.SHARED)
        shape = [
            self.impl.get_page_index_upper_bound(high_id, Role.KEY) // 2,
            2,
            self.tokens_per_block,
            self.num_kv_heads_per_layer[local],
            self.head_dim_per_layer[local],
        ]
        if kv_layout == "HND":
            shape[2], shape[3] = shape[3], shape[2]
        elif kv_layout != "NHD":
            raise ValueError(f"Unsupported KV layout: {kv_layout}")
        view = convert_to_torch_tensor(TensorWrapper(address, self.high_dtype, shape))
        indices = self._get_batch_cache_indices_by_pool_id(
            request_ids,
            pool_id=int(self.impl.get_layer_group_id(high_id)),
            index_scale=int(self.impl.get_page_index_converter(high_id, Role.KEY).scale),
        )
        return view, indices

    def get_staging_buffer(self, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """Share a bounded Attention workspace between layers on one stream."""
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if size > self._staging_bytes:
            raise ValueError("Mixed-precision Attention staging exceeds its reserved capacity")
        stream = torch.cuda.current_stream()
        if self._staging_buffer is None:
            self._staging_buffer = torch.empty(
                self._staging_bytes, dtype=torch.uint8, device=stream.device
            )
            self._staging_stream = stream.cuda_stream
        elif stream.cuda_stream != self._staging_stream:
            raise NotImplementedError("Mixed-precision staging requires one execution stream")
        return self._staging_buffer[:size].view(dtype).view(shape)

    def shutdown(self) -> None:
        super().shutdown()
        self._staging_buffer = None

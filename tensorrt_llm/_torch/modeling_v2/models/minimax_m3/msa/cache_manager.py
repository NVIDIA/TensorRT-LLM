# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MiniMax-M3.1 KV cache manager: main's MiniMax-M3 manager plus an NVFP4 paged index-K cache.

``indexer_kv_dtype="nvfp4"`` stores each sparse layer's index keys as one 128-token page in the layout of MSA's
Q8KV4 decode indexer: 8192 bytes of E2M1 codes (64 per token, low nibble = even element) then 1024 bytes of E4M3
16-element scales (8 per token), no global scale, so 72 bytes per token viewed as ``[pages, 1, 128, 72]`` uint8.
The fused QKV producer writes E4M3 index keys at each token's main-pool slot into one FP8 sink shared by every
sparse layer (:meth:`MiniMaxM31KVCacheManagerV2.get_nvfp4_index_k_sink`); each layer's NVFP4 pages are packed from it
after its producer (``kernels/nvfp4_index_k``). Every other index-K dtype is main's manager unchanged.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
    default_sparse_layer_ids,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.logger import logger

__all__ = [
    "NVFP4_INDEX_K_BYTES_PER_TOKEN",
    "NVFP4_INDEX_K_DIM",
    "MiniMaxM31KVCacheManagerV2",
    "get_sparse_disable_index_value_layer_ids",
]

NVFP4_INDEX_K_DIM = 128
NVFP4_INDEX_K_BYTES_PER_TOKEN = NVFP4_INDEX_K_DIM // 2 + NVFP4_INDEX_K_DIM // 16


def get_sparse_disable_index_value_layer_ids(pretrained_config) -> Optional[List[int]]:
    """The sparse layers whose index-V the checkpoint disables, or None when the checkpoint does not say."""
    sparse_config = getattr(pretrained_config, "sparse_attention_config", None)
    if sparse_config is None:
        return None
    flags = getattr(sparse_config, "disable_index_value", None)
    if flags is None and isinstance(sparse_config, dict):
        flags = sparse_config.get("disable_index_value")
    if flags is None:
        return None
    if isinstance(flags, bool):
        return None
    return [i for i, flag in enumerate(flags) if flag]


def _is_nvfp4_index_k(sparse_attention_config) -> bool:
    return str(getattr(sparse_attention_config, "indexer_kv_dtype", "bf16")) == "nvfp4"


class _IndexKViewShim(KVCacheManagerV2):
    """Sits between main's manager and the V2 base in the MRO, so main's index-K view lookups resolve the NVFP4
    row width (72 packed bytes per token) instead of the index head dim."""

    def get_index_k_buffer(self, layer_idx, *args, **kwargs):
        if getattr(self, "_nvfp4_index_k", False):
            kwargs["head_dim"] = NVFP4_INDEX_K_BYTES_PER_TOKEN
            kwargs["dtype"] = torch.uint8
        return super().get_index_k_buffer(layer_idx, *args, **kwargs)


class MiniMaxM31KVCacheManagerV2(MiniMaxM3KVCacheManagerV2, _IndexKViewShim):
    """Main's MiniMax-M3 KV cache manager with the NVFP4 index-K cache (see the module docstring)."""

    def __init__(self, *args, **kwargs):
        sparse_attention_config = kwargs.get("sparse_attention_config")
        # Read by the index-K sizing and view overrides while main's __init__ allocates the pools.
        self._nvfp4_index_k = _is_nvfp4_index_k(sparse_attention_config)
        if self._nvfp4_index_k:
            sparse_index_dim = kwargs.get("sparse_index_dim") or getattr(
                sparse_attention_config, "sparse_index_dim", None
            )
            if sparse_index_dim is not None and int(sparse_index_dim) != NVFP4_INDEX_K_DIM:
                raise ValueError(
                    f"MiniMax M3 NVFP4 index-K requires sparse_index_dim {NVFP4_INDEX_K_DIM}, got "
                    f"{sparse_index_dim}."
                )
            # Main's manager validates bf16 / fp8; the NVFP4 sizing and views are this class's overrides.
            kwargs["sparse_attention_config"] = sparse_attention_config.model_copy(
                update={"indexer_kv_dtype": "fp8"}
            )
        self._nvfp4_index_k_sink: Optional[torch.Tensor] = None
        super().__init__(*args, **kwargs)
        if self._nvfp4_index_k:
            self.indexer_kv_dtype = "nvfp4"
            self._build_nvfp4_index_k_sink()

    def _build_nvfp4_index_k_sink(self) -> None:
        """The FP8 index-K sink of the main pool's page count. The index-K pool must address the same pages as the
        main pool: the producer's slot is the index-K slot."""
        if not self._index_k_buffers:
            return
        kv_layout = self._main_kv_view_layout
        main_pages = {
            int(self.get_buffers(layer_idx, kv_layout).shape[0])
            for layer_idx in self._index_k_buffers
        }
        index_pages = {
            int(view.shape[0]) for view in self._index_k_buffers.values() if view is not None
        }
        if len(main_pages) != 1 or index_pages != main_pages:
            raise ValueError(
                "MiniMax M3 NVFP4 index-K needs the index-K pool to address the main pool's pages: main pages "
                f"{sorted(main_pages)}, index-K pages {sorted(index_pages)}"
            )
        self._nvfp4_index_k_sink = torch.zeros(
            (main_pages.pop(), 1, self.tokens_per_block, NVFP4_INDEX_K_DIM),
            dtype=torch.float8_e4m3fn,
            device=torch.device("cuda"),
        )
        logger.info(
            f"[M3.1] NVFP4 index-K: index-K pool NVFP4 ({NVFP4_INDEX_K_BYTES_PER_TOKEN} B per token, "
            f"{len(self._index_k_buffers)} layer(s)), FP8 sink {tuple(self._nvfp4_index_k_sink.shape)}"
        )

    def _index_k_bytes_per_token(self) -> int:
        if getattr(self, "_nvfp4_index_k", False):
            return NVFP4_INDEX_K_BYTES_PER_TOKEN
        return super()._index_k_bytes_per_token()

    def _torch_dtype_for_index_cache(self) -> torch.dtype:
        if getattr(self, "_nvfp4_index_k", False):
            return torch.uint8
        return super()._torch_dtype_for_index_cache()

    def get_nvfp4_index_k_sink(self) -> Optional[torch.Tensor]:
        """The FP8 index-K sink the fused producer writes for NVFP4 index-K layers (None otherwise)."""
        return self._nvfp4_index_k_sink

    @classmethod
    def _extra_bytes_per_token(cls, model_config, mapping, num_layers, kwargs) -> int:
        extra = super()._extra_bytes_per_token(model_config, mapping, num_layers, kwargs)
        sparse_config = getattr(model_config, "sparse_attention_config", None)
        if not _is_nvfp4_index_k(sparse_config):
            return extra
        config = model_config.pretrained_config
        config = getattr(config, "text_config", None) or config
        num_target_layers = int(num_layers if num_layers is not None else config.num_hidden_layers)
        local_layers = (
            list(mapping.pp_layers(num_target_layers))
            if mapping.pp_size > 1
            else list(range(num_target_layers))
        )
        sparse_layer_ids = kwargs.get("sparse_layer_ids")
        if sparse_layer_ids is None:
            sparse_layer_ids = default_sparse_layer_ids(config, num_target_layers)
        sparse_layer_ids = set(int(layer) for layer in sparse_layer_ids)
        num_sparse = sum(1 for layer in local_layers if layer in sparse_layer_ids)
        if num_sparse == 0:
            return extra
        index_dim = int(getattr(sparse_config, "sparse_index_dim", None) or NVFP4_INDEX_K_DIM)
        # Main's estimate priced each sparse layer's index-K row as BF16; NVFP4 packs it into 72 bytes, plus the
        # shared FP8 sink (one FP8 row per token).
        return (
            extra
            - num_sparse * index_dim * 2
            + num_sparse * NVFP4_INDEX_K_BYTES_PER_TOKEN
            + NVFP4_INDEX_K_DIM
        )

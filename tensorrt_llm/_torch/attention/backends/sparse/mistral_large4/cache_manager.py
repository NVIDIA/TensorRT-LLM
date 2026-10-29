# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV cache manager for Mistral-Large-4 checkpoints: an fp8_ds_mla latent pool with
cross-layer KV sharing.

The checkpoint names, per layer, the layer whose latent it attends over
(``kv_source_layers``); a layer is a KV *owner* when that is itself. Only owners get
a latent page, so a sharing layer allocates no latent at all and its readers address
its source's page. Every latent row is the fp8_ds_mla token the reference
implementation writes::

    [ 512 x e4m3 NoPE | 4 x fp32 power-of-two tile scale | 64 x bf16 RoPE ]  = 656 B

The indexer K cache is the DSA one (``index_head_dim`` e4m3 + one fp32 scale per key)
on every ``index_sharing == 'full'`` layer, owners and sharers alike. Indexer
pages belong to their logical layers independently of latent ownership.

Bytes per token = owners x 656 + full-indexer layers x 132.
The manager is selected by ``sparse.registry`` for a ``MistralLarge4Config``
checkpoint and requires KV cache manager V2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import LayerId

from ..dsa.cache_manager import (
    DSACacheManagerV2,
    _get_indexer_k_cache_bytes_per_token,
    derive_indexer_k_cache_layer_mask,
)
from ..dsa.params import DSAParams

if TYPE_CHECKING:
    import tensorrt_llm.bindings
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, SparseAttentionConfig

    ModelConfig = tensorrt_llm.bindings.ModelConfig

#: The latent row each owner layer stores per token, in bytes.
DIM_NOPE = 512
DIM_ROPE = 64
QUANT_TILE = 128
SCALE_BYTES = DIM_NOPE // QUANT_TILE * 4
ROPE_OFFSET = DIM_NOPE + SCALE_BYTES
TOKEN_BYTES = ROPE_OFFSET + DIM_ROPE * 2  # 656
#: The layout name targets check before they address the pool.
LATENT_LAYOUT = "fp8_ds_mla"
_INDEXER_QUANT_BLOCK = 128


def kv_source_layers(pretrained_config) -> Optional[List[int]]:
    """The checkpoint's per-layer KV source, validated, or None when it shares nothing."""
    sources = getattr(pretrained_config, "kv_source_layers", None)
    if sources is None:
        return None
    sources = [int(s) for s in sources]
    num_layers = int(pretrained_config.num_hidden_layers)
    if len(sources) != num_layers:
        raise ValueError(f"kv_source_layers has {len(sources)} entries for {num_layers} layers")
    for layer, source in enumerate(sources):
        if not 0 <= source <= layer or sources[source] != source:
            raise ValueError(
                f"kv_source_layers[{layer}] = {source}: a source must be an earlier or the "
                "same layer, and must own its latent"
            )
    return sources


def latent_cache_bytes_per_token(
    sources: List[int], indexer_mask: List[bool], indexer_bytes: int
) -> int:
    """Owners x 656 + full-indexer layers x the indexer bytes."""
    owners = sum(1 for i, s in enumerate(sources) if s == i)
    return owners * TOKEN_BYTES + sum(indexer_mask) * indexer_bytes


def _indexer_bytes(sparse_params: DSAParams) -> int:
    return _get_indexer_k_cache_bytes_per_token(
        sparse_params.index_head_dim,
        _INDEXER_QUANT_BLOCK,
        sparse_params.indexer_k_dtype == "fp4",
    )


class MistralLarge4CacheManager(DSACacheManagerV2):
    """DSA cache manager with an owner-only fp8_ds_mla latent pool (see module doc)."""

    latent_layout = LATENT_LAYOUT
    latent_token_bytes = TOKEN_BYTES

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional["ModelConfig"] = None,
        max_beam_width: int = 1,
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        pretrained_config=None,
        **kwargs,
    ) -> None:
        sources = kv_source_layers(pretrained_config)
        if sources is None:
            raise ValueError(f"{type(self).__name__} needs the checkpoint's kv_source_layers")
        # Separate-draft allocation passes an all-target-layer mask. Physical
        # latent ownership still comes exclusively from kv_source_layers.
        if layer_mask is not None and layer_mask != [True] * len(sources):
            raise NotImplementedError(
                f"{type(self).__name__} derives its layer mask from kv_source_layers"
            )
        if spec_config is not None and (
            not spec_config.spec_dec_mode.is_dspark()
            or spec_config.draft_is_embedded_in_target
            or spec_config._use_shared_kv_cache
        ):
            raise NotImplementedError(
                f"{type(self).__name__} supports DSpark with a separate draft KV pool only"
            )
        if mapping.pp_size != 1:
            raise NotImplementedError(f"{type(self).__name__} supports pp_size 1 only")
        owner_heads = [1 if source == layer else None for layer, source in enumerate(sources)]
        if num_layers != len(sources):
            raise ValueError("num_layers must match the checkpoint's kv_source_layers")
        if kv_cache_type != CacheTypeCpp.SELFKONLY or num_kv_heads not in (
            1,
            [1] * num_layers,
            owner_heads,
        ):
            raise ValueError("the latent cache is one SELFKONLY MLA head per layer")
        if head_dim != DIM_NOPE + DIM_ROPE:
            raise ValueError(f"expected a {DIM_NOPE}+{DIM_ROPE} latent, got head_dim {head_dim}")
        requested = getattr(kv_cache_config, "dtype", "auto")
        if dtype != DataType.FP8 or requested not in ("auto", "fp8", "fp8_ds_mla"):
            raise ValueError(
                "the Mistral-Large-4 latent cache is fp8_ds_mla (656 B/token): set "
                "kv_cache_config.dtype to auto (the checkpoint's fp8_kv_cache), fp8 or "
                f"fp8_ds_mla; got dtype={requested!r} ({dtype})"
            )
        if getattr(kv_cache_config, "enable_kv_pool_rebalance", False):
            raise ValueError(
                "the fp8_ds_mla latent cache does not support enable_kv_pool_rebalance"
            )
        if requested == "fp8_ds_mla":
            # The generic DSA check for this dtype string describes the SM120 FlashInfer
            # inline-scale pages (576-element head, 64-token pages). This manager owns the
            # 656-byte layout itself and takes any page size, so hand the base the plain
            # FP8 element type.
            kv_cache_config = kv_cache_config.model_copy(update={"dtype": "fp8"})
        if sparse_attention_config is None and model_config is not None:
            sparse_attention_config = model_config.sparse_attention_config

        self.kv_source_layers = sources
        self.kv_owner_layers = [i for i, s in enumerate(sources) if s == i]
        self.kv_sharer_layers = [i for i, s in enumerate(sources) if s != i]
        indexer_mask = derive_indexer_k_cache_layer_mask(
            sparse_attention_config, pretrained_config, len(sources)
        )
        if any(not indexer_mask[layer] for layer in self.kv_sharer_layers):
            raise ValueError("Every latent-sharing layer must own an indexer K-cache")

        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=owner_heads,
            head_dim=TOKEN_BYTES,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=DataType.FP8,
            spec_config=spec_config,
            layer_mask=None,
            max_num_tokens=max_num_tokens,
            model_config=model_config,
            max_beam_width=max_beam_width,
            sparse_attention_config=sparse_attention_config,
            pretrained_config=pretrained_config,
            **kwargs,
        )
        # The generic flag names the SM120 FlashInfer inline-scale path, which the DSA
        # metadata keys extra work on; this pool is read through `latent_layout`.
        self.use_fp8_ds_mla = False
        actual_owners = [
            layer for local, layer in enumerate(self.pp_layers) if self.has_buffer(local, Role.KEY)
        ]
        if actual_owners != self.kv_owner_layers:
            raise RuntimeError(f"Physical KEY owners {actual_owners} do not match KV sources")

        logger.info(
            f"[{type(self).__name__}] {len(self.kv_owner_layers)} latent owners x "
            f"{TOKEN_BYTES} B + {sum(indexer_mask)} indexer layers x "
            f"{self.get_layer_bytes_per_token(0, Role.INDEX_KEY)} B = "
            f"{self.get_cache_bytes_per_token()} B/token; "
            f"sharers {self.kv_sharer_layers} read their source's latent"
        )

    def _kv_pool_mapping_offset(
        self, layer_id: LayerId, layer_group_id: int, key_base_addr: int
    ) -> int:
        source = self.kv_source_layers[self.pp_layers[int(layer_id)]]
        return super()._kv_pool_mapping_offset(
            self.layer_offsets[source], layer_group_id, key_base_addr
        )

    def get_unique_primary_pool(self) -> torch.Tensor:
        """Return the contiguous latent pool in physical owner order."""
        if self._unique_primary_pool is not None:
            return self._unique_primary_pool
        owners = self.kv_owner_layers
        key_buffers = [buffer for buffer in self.impl.all_buffer_ids if buffer.role == Role.KEY]
        pages = self.impl.get_aggregated_pages(key_buffers)
        if len(pages) != 1:
            raise ValueError("ML4 latent kernels require a single contiguous KEY pool")
        page = pages[0]
        bytes_per_slot = len(owners) * self.tokens_per_block * TOKEN_BYTES
        if (
            int(page.size) != bytes_per_slot
            or int(page.stride) != bytes_per_slot
            or any(int(buffer.expansion) != 1 for buffer in page.buffers)
        ):
            raise ValueError("ML4 latent kernels require packed, unexpanded KEY pages")
        # The descriptor supplies the pool's physical origin. Logical layers
        # address their source's row through the page-index converter.
        self._unique_primary_pool = convert_to_torch_tensor(
            TensorWrapper(
                page.base,
                self.dtype,
                [self.blocks_in_primary_pool, len(owners), 1, self.tokens_per_block * TOKEN_BYTES],
            )
        )
        return self._unique_primary_pool

    @staticmethod
    def get_cache_size_per_token(
        model_config: "ModelConfig",
        mapping: Mapping,
        num_layers: Optional[int] = None,
        **kwargs,
    ) -> int:
        """Owners x 656 + full-indexer layers x the indexer bytes."""
        config = model_config.pretrained_config
        sources = kv_source_layers(config)
        if sources is None or num_layers is not None:
            raise ValueError("the Mistral-Large-4 cache is sized from its own kv_source_layers")
        sparse = model_config.sparse_attention_config
        indexer_mask = derive_indexer_k_cache_layer_mask(sparse, config, len(sources))
        return latent_cache_bytes_per_token(
            sources, indexer_mask, _indexer_bytes(sparse.to_sparse_params(pretrained_config=config))
        )

    # ---- accessors the targets read ----

    def latent_pool_row(self, layer_idx: int) -> int:
        """The latent page row (0..owners-1) layer `layer_idx` reads: its source's."""
        source = self.layer_offsets[self.kv_source_layers[layer_idx]]
        return self.get_primary_pool_page_index_params(source)[1]

    def get_latent_pool(self) -> torch.Tensor:
        """The whole latent pool as uint8 `[blocks, owners, tokens_per_block, 656]`;
        the token at position p of a request whose block-table entry is b lives, for
        layer L, at `[b, latent_pool_row(L), p % tokens_per_block]`."""
        pool = self.get_unique_primary_pool()
        return pool.view(torch.uint8).view(
            pool.shape[0], pool.shape[1], self.tokens_per_block, TOKEN_BYTES
        )

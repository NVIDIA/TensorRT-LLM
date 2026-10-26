# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV cache manager V2 for Step-5's sliding-window and sparse-attention layers.

Step-5's decoder repeats [sliding, sliding, sliding, full]. The sliding layers
keep a ``sliding_window``-token window (the engine derives the per-layer windows
from ``layer_types``, so V2 gives them a life cycle of their own); the
full-attention layers keep the whole sequence and, beside their K/V, the sparse
indexer's state. The K/V pools of both kinds hold heads zero-padded from
``head_dim`` 192 to :func:`step5_kv_head_dim` (256), the width the attention
kernels run at.

The indexer state of one block of a full-attention layer is one
``Role.INDEX_KEY`` page:

* ``tokens_per_block`` bf16 rows ``[k | z]``: each token's one-head proxy key
  and its per-dimension softmax logits, ``proxy_dim`` each;
* then ``tokens_per_block // region_block_size`` fp8 e4m3 rows: the summary of
  each region of the block, ``proxy_dim`` wide.

Registered as a V2 extra buffer of the full-attention layers, the page shares
those layers' life cycle: allocation, eviction, block reuse and transfer treat a
block's K/V and indexer state as one unit, and one slot id addresses both. The
two parts share one page because V2 coalesces the buffers of a life cycle by
page size alone, and either part alone is the size of a K/V or block-scale page
at some KV-head split; the attention op addresses a layer's K/V and block-scale
pages with one block offset, so a page coalesced into those pools would move
them under it. The packed page matches no such size on any layout.

Every view this manager serves is slot-major: index it with the raw slot ids of
the request's block table (``get_batch_cache_indices(..., raw_indices=True)``,
or ``kv_cache_block_offsets // scale`` on the device, see
:meth:`Step5CacheManager.get_block_offset_slot_scale`).

The state a block holds is complete for every token it covers under this
contract with the forward: a step that writes the token completing a region
writes that region's summary before any query reads it, and a query reads only
the summaries of regions that end before its own region starts. A region never
straddles two blocks (``tokens_per_block`` is a multiple of
``region_block_size``), so a reused block's summaries are exactly the ones a
cold prefill computes. A partially reused block can carry a summary for the
region straddling the reuse point, computed from the donor's tokens; by the
contract it is rewritten from the per-token ``[k | z]`` rows before it is read,
and those rows depend on their own token and position only. Partial reuse
therefore stays enabled.

The GPU budget is split between the two life cycles by the engine's limits,
not by a modelled average sequence: the sliding pool holds ``max_batch_size``
windows plus one step's ``max_num_tokens`` context tokens (what V2's
quota-to-token conversion reserves for these layers), and the full-attention
pool (K/V and indexer pages) takes the rest. A ``pool_ratio`` or ``avg_seq_len`` in the KV
cache config replaces that split with V2's own.
"""

import copy
import math
from dataclasses import dataclass, replace
from typing import Any, List, Optional

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    Role,
    _get_generation_kv_capacity,
    _gpu_alloc_granularity,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import get_pp_layers
from tensorrt_llm._utils import TensorWrapper, binding_to_torch_dtype, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, DataRole, PageIndexMode

from .params import (
    STEP5_INDEX_KZ_DTYPE,
    STEP5_INDEX_KZ_PLANES,
    STEP5_INDEX_SUMMARY_DTYPE,
    Step5IndexerCacheGeometry,
    step5_kv_head_dim,
    step5_sliding_pool_blocks,
    step5_text_config,
)

_MAIN_KV_ROLES = (Role.KEY, Role.VALUE)
_BLOCK_SCALE_ROLES = (Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE)
# Element widths of the main K/V pool per dtype, in bits; NVFP4 also keeps one
# e4m3 scale byte per 16 values.
_KV_ELEMENT_BITS = {DataType.HALF: 16, DataType.BF16: 16, DataType.FP8: 8, DataType.NVFP4: 4}
_NVFP4_BLOCK_SIZE = 16
_IDENTITY_PAGE_INDEX_EXPANSION = 1


@dataclass(frozen=True)
class Step5PoolSplit:
    """How the GPU budget is split between the life cycles."""

    # V2's initial_pool_ratio: one share per life cycle, in life-cycle order.
    ratio: tuple[float, ...]
    # Slots each sliding-window pool must hold, by window.
    sliding_blocks: dict[int, int]
    # Allocator grains of the sliding-window pools, together, and of the
    # full-attention pools (the rest of the budget).
    sliding_grains: int
    full_grains: int
    grain_bytes: int


class Step5CacheManager(KVCacheManagerV2):
    """V2 manager whose full-attention layers also hold the sparse indexer state."""

    def __init__(
        self,
        *args,
        pretrained_config: Optional[Any] = None,
        sparse_attention_config: Optional[Any] = None,
        **kwargs,
    ) -> None:
        if pretrained_config is None:
            raise ValueError("Step5CacheManager needs the checkpoint's pretrained_config")
        if sparse_attention_config is not None:
            raise ValueError(
                "Step-5 reads its sparse geometry from text_config.sparse_config; "
                "a sparse_attention_config would describe a different algorithm"
            )
        text_config = step5_text_config(pretrained_config)
        # Set before super().__init__: _build_base_config calls
        # _extra_buffers_per_layer and get_layer_bytes_per_token, which read it.
        self.step5_geometry = Step5IndexerCacheGeometry.from_config(text_config)
        head_dim = kwargs.get("head_dim")
        if isinstance(head_dim, int):
            kwargs["head_dim"] = step5_kv_head_dim(head_dim)
        elif head_dim is not None:
            kwargs["head_dim"] = [step5_kv_head_dim(int(width)) for width in head_dim]
        num_kv_heads = kwargs.get("num_kv_heads")
        num_groups = getattr(text_config, "num_attention_groups", None)
        if isinstance(num_kv_heads, int) and num_groups is not None and num_kv_heads != num_groups:
            raise ValueError(
                f"Step-5 has {num_groups} KV heads (num_attention_groups), but the cache "
                f"was asked for {num_kv_heads}"
            )
        super().__init__(*args, **kwargs)
        if self.kv_cache_type != CacheTypeCpp.SELF:
            raise ValueError(f"Step-5 keeps both K and V, got cache type {self.kv_cache_type}")
        self.step5_regions_per_block = self.step5_geometry.regions_per_block(self.tokens_per_block)
        self._check_index_page_mapping()

    # ------------------------------------------------------------------
    # Layer bookkeeping
    # ------------------------------------------------------------------
    @property
    def sparse_layer_ids(self) -> tuple[int, ...]:
        """Global ids of the layers that hold indexer state."""
        return self.step5_geometry.sparse_layer_ids

    @property
    def local_sparse_layer_ids(self) -> List[int]:
        """Sparse layers this rank holds, in model order."""
        return [layer for layer in self.sparse_layer_ids if layer in self.layer_offsets]

    def _is_local_sparse(self, local_layer_idx: int) -> bool:
        return self.step5_geometry.is_sparse_layer(self.pp_layers[local_layer_idx])

    # ------------------------------------------------------------------
    # Buffer registration and sizing
    # ------------------------------------------------------------------
    def _build_cache_config(self, config: Any) -> Any:
        """Pin the split of the GPU budget between the life cycles, unless the
        KV cache config chose one (``pool_ratio``, or an ``avg_seq_len`` V2
        models a typical step from)."""
        config = super()._build_cache_config(config)
        self.step5_pool_split = None
        if config.initial_pool_ratio is not None or config.typical_step is not None:
            return config
        split = self._derive_pool_split(config)
        if split is None:
            return config
        self.step5_pool_split = split
        return replace(config, initial_pool_ratio=list(split.ratio))

    def _derive_pool_split(self, config: Any) -> Optional[Step5PoolSplit]:
        """The split the engine's limits call for, in the GPU allocator's grains.

        Each sliding-window life cycle gets the grains that hold
        :func:`step5_sliding_pool_blocks` slots: ``max_batch_size`` windows plus
        one step's ``max_num_tokens`` context tokens. The full-attention life
        cycle -- K/V plus the indexer pages -- gets every other grain. V2 hands
        the smallest share its grains first and the last share whatever is
        left, so expressing the shares in whole grains makes each sliding pool
        exactly those grains.

        None, leaving V2's own split, when the layout has no full-attention and
        sliding life cycle to split between, or when the budget cannot hold the
        sliding pools beside one ``max_seq_len`` request on the full-attention
        layers.
        """
        life_cycles: dict[tuple, dict[int, int]] = {}
        windows: dict[tuple, Optional[int]] = {}
        for layer in config.layers:
            buffers = list(layer.buffers)
            key = (
                layer.sliding_window_size,
                layer.num_sink_tokens,
                bool(buffers) and bool(buffers[0].is_sparse),
                layer.reuse_policy,
            )
            windows[key] = layer.sliding_window_size
            pools = life_cycles.setdefault(key, {})
            for buffer in buffers:
                pools[int(buffer.size)] = pools.get(int(buffer.size), 0) + 1
        full = [key for key, window in windows.items() if window is None]
        sliding = [key for key, window in windows.items() if window is not None]
        if len(full) != 1 or not sliding:
            return None

        quota = int(config.cache_tiers[0].quota)
        grain = _gpu_alloc_granularity(quota)
        total_grains = -(-quota // grain)

        def grains(key: tuple, slots: int) -> int:
            return sum(
                -(-slots * page * count // grain) for page, count in life_cycles[key].items()
            )

        chunk_tokens = (
            self.max_num_tokens if self.max_num_tokens is not None else self.max_seq_len
        ) + self.num_extra_kv_tokens
        headroom = self._generation_kv_capacity_headroom
        sliding_blocks = {
            windows[key]: step5_sliding_pool_blocks(
                max_batch_size=self._get_generation_request_capacity(),
                window=int(windows[key]),
                chunk_tokens=int(chunk_tokens),
                tokens_per_block=self.tokens_per_block,
                headroom=headroom,
                beam_width=self.max_beam_width,
            )
            for key in sliding
        }
        sliding_grains = {key: grains(key, sliding_blocks[windows[key]]) for key in sliding}
        one_request = -(-(self.max_seq_len + headroom) // self.tokens_per_block)
        if sum(sliding_grains.values()) + grains(full[0], one_request) > total_grains:
            return None
        full_grains = total_grains - sum(sliding_grains.values())
        ratio = tuple(
            (full_grains if key == full[0] else sliding_grains[key]) / total_grains
            for key in life_cycles
        )
        return Step5PoolSplit(
            ratio=ratio,
            sliding_blocks=sliding_blocks,
            sliding_grains=sum(sliding_grains.values()),
            full_grains=full_grains,
            grain_bytes=grain,
        )

    def _kz_page_bytes(self, tokens_per_block: int) -> int:
        return self.step5_geometry.kz_bytes_per_token * tokens_per_block

    def _index_page_bytes(self, tokens_per_block: int) -> int:
        return self.step5_geometry.bytes_per_token * tokens_per_block

    def _extra_buffers_per_layer(self, *, tokens_per_block: int) -> dict[int, list[BufferConfig]]:
        """One ``Role.INDEX_KEY`` page per block on every local sparse layer."""
        self.step5_geometry.regions_per_block(tokens_per_block)
        page_bytes = self._index_page_bytes(tokens_per_block)
        main_sizes = self._main_page_sizes(tokens_per_block)
        if page_bytes in main_sizes:
            raise ValueError(
                f"A {page_bytes}-byte Step-5 indexer page would coalesce into a K/V or "
                f"block-scale pool (main page sizes {sorted(main_sizes)})"
            )
        return {
            local_idx: [BufferConfig(role=Role.INDEX_KEY, size=page_bytes)]
            for local_idx, layer_id in enumerate(self.pp_layers)
            if self.step5_geometry.is_sparse_layer(layer_id)
        }

    def _main_page_sizes(self, tokens_per_block: int) -> set[int]:
        """Every K/V or block-scale page size a local layer can have, at any
        split of its KV heads over ranks; the same on every TP and DP layout."""
        sizes = set()
        for local_idx, layer_id in enumerate(self.pp_layers):
            heads = self.num_kv_heads_per_layer[local_idx]
            total = (
                self.num_kv_heads
                if isinstance(self.num_kv_heads, int)
                else self.num_kv_heads[layer_id]
            )
            if not heads or not total:
                continue
            for role in self._get_buffer_roles_for_layer(local_idx):
                page = (
                    KVCacheManagerV2.get_layer_bytes_per_token(self, local_idx, role)
                    * tokens_per_block
                )
                sizes.update(page // heads * h for h in range(1, total + 1))
        return sizes

    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: DataRole) -> int:
        sparse = self._is_local_sparse(local_layer_idx)
        if data_role == Role.INDEX_KEY:
            return self.step5_geometry.bytes_per_token if sparse else 0
        cache_bytes = super().get_layer_bytes_per_token(local_layer_idx, data_role)
        if data_role == Role.ALL and sparse:
            cache_bytes += self.step5_geometry.bytes_per_token
        return cache_bytes

    def get_cache_bytes_per_token(self) -> int:
        return sum(
            self.get_layer_bytes_per_token(local_layer_idx, Role.ALL)
            for local_layer_idx in range(self.num_local_layers)
        )

    @staticmethod
    def _text_model_config(model_config: Any) -> Any:
        """``model_config`` as the pools see it, for V2's static cost model: the
        decoder's fields (the composite config nests them) with the padded K/V
        head width the manager allocates."""
        text_config = copy.copy(step5_text_config(model_config.pretrained_config))
        text_config.head_dim = step5_kv_head_dim(int(text_config.head_dim))
        text_model_config = copy.copy(model_config)
        text_model_config.pretrained_config = text_config
        return text_model_config

    @staticmethod
    def _local_sparse_layer_count(
        model_config: Any, mapping: Any, num_layers: Optional[int]
    ) -> int:
        geometry = Step5IndexerCacheGeometry.from_config(model_config.pretrained_config)
        total_layers = geometry.num_layers if num_layers is None else num_layers
        local_layers, _ = get_pp_layers(total_layers, mapping)
        return sum(geometry.is_sparse_layer(layer) for layer in local_layers)

    @classmethod
    def get_cache_size_per_token(
        cls,
        model_config: Any,
        mapping: Any,
        num_layers: Optional[int] = None,
        **kwargs,
    ):
        """V2's affine cost of the padded pools plus the indexer state of the
        local sparse layers.

        The indexer state grows with the sequence on the full-attention layers,
        so it adds to the per-token slope and not to the per-request intercept.
        """
        text_model_config = cls._text_model_config(model_config)
        slope, fixed_cost = super().get_cache_size_per_token(
            text_model_config, mapping, num_layers, **kwargs
        )
        geometry = Step5IndexerCacheGeometry.from_config(text_model_config.pretrained_config)
        num_sparse = cls._local_sparse_layer_count(text_model_config, mapping, num_layers)
        return slope + num_sparse * geometry.bytes_per_token, fixed_cost

    @classmethod
    def get_cache_bytes_per_request(
        cls,
        model_config: Any,
        mapping: Any,
        num_layers: Optional[int] = None,
        *,
        tokens_per_block: int,
        max_seq_len: Optional[int] = None,
        spec_config=None,
        is_draft: bool = False,
        **kwargs,
    ) -> Optional[int]:
        """V2's slope-funded spend of one max-length request, with the indexer
        state its full-attention layers pin for the whole sequence."""
        text_model_config = cls._text_model_config(model_config)
        spend = super().get_cache_bytes_per_request(
            text_model_config,
            mapping,
            num_layers,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            spec_config=spec_config,
            is_draft=is_draft,
            **kwargs,
        )
        if spend is None or max_seq_len is None:
            return spend
        _, headroom = _get_generation_kv_capacity(spec_config, is_draft=is_draft)
        full_tokens = math.ceil((int(max_seq_len) + headroom) / tokens_per_block) * tokens_per_block
        geometry = Step5IndexerCacheGeometry.from_config(text_model_config.pretrained_config)
        num_sparse = cls._local_sparse_layer_count(text_model_config, mapping, num_layers)
        return spend + num_sparse * geometry.bytes_per_token * full_tokens

    # ------------------------------------------------------------------
    # Page-mapping checks
    # ------------------------------------------------------------------
    def _check_index_page_mapping(self) -> None:
        """The indexer views assume what V2 is free to change: one slot id
        addresses a layer's K/V and indexer pages, and the indexer pages of all
        local sparse layers share one page-index mapping."""
        sparse_locals = [self.layer_offsets[layer] for layer in self.local_sparse_layer_ids]
        if not sparse_locals:
            return
        group_ids = {self.layer_to_pool_mapping_dict[local] for local in sparse_locals}
        if len(group_ids) != 1:
            raise RuntimeError(
                f"Step-5 sparse layers span V2 layer groups {sorted(group_ids)}; "
                "the indexer views need one slot space"
            )
        scales = set()
        for local in sparse_locals:
            converter = self.impl.get_page_index_converter(local, Role.INDEX_KEY)
            if int(converter.expansion) != _IDENTITY_PAGE_INDEX_EXPANSION:
                raise RuntimeError("Step-5 indexer pages do not support page expansion")
            scales.add(int(converter.scale))
        if len(scales) != 1:
            raise RuntimeError(
                f"Step-5 indexer pages have different page-index scales {sorted(scales)} "
                "across the sparse layers"
            )

    # ------------------------------------------------------------------
    # Views the forward reads
    # ------------------------------------------------------------------
    def _slot_pages(self, layer_idx: int, role: DataRole, planes: int = 1) -> torch.Tensor:
        """uint8 ``[slots, planes, page_bytes]``: ``role``'s page of
        ``layer_idx`` in every slot, with the ``planes - 1`` pages that follow it
        in the slot (V after K).

        V2 lays a slot of a coalesced pool out as ``scale`` consecutive pages, so
        slot ``s`` of this layer is page ``scale * s`` from the layer's own base
        address.
        """
        local = self.layer_offsets[layer_idx]
        address = self.impl.get_mem_pool_base_address(local, role, PageIndexMode.SHARED)
        page_bytes = self.impl.get_page_stride(local, role)
        converter = self.impl.get_page_index_converter(local, role)
        scale = int(converter.scale)
        if int(converter.expansion) != _IDENTITY_PAGE_INDEX_EXPANSION:
            raise RuntimeError(f"Step-5 {role} pages do not support page expansion")
        if planes > scale:
            raise RuntimeError(
                f"Step-5 {role} of layer {layer_idx}: {planes} pages per slot requested, "
                f"the pool holds {scale}"
            )
        pages = self.impl.get_page_index_upper_bound(local, role) + int(converter.layer_offset)
        if pages % scale:
            raise RuntimeError(
                f"Step-5 {role} of layer {layer_idx}: {pages} pages are not whole slots "
                f"of {scale} pages"
            )
        raw = convert_to_torch_tensor(
            TensorWrapper(address, torch.uint8, [pages // scale, scale, page_bytes])
        )
        return raw[:, :planes]

    @staticmethod
    def _typed(raw: torch.Tensor, dtype: torch.dtype, shape: List[int]) -> torch.Tensor:
        """``raw``'s last axis, a whole number of ``shape`` rows of ``dtype``, typed."""
        expected = math.prod(shape) * dtype.itemsize
        if raw.shape[-1] != expected:
            raise RuntimeError(f"a {raw.shape[-1]}-byte region is not {shape} {dtype}")
        return raw.view(dtype).unflatten(-1, shape)

    def get_index_kz_buffer(self, layer_idx: int) -> Optional[torch.Tensor]:
        """bf16 ``[slots, tokens_per_block, 2, proxy_dim]``: token ``t`` of slot
        ``s`` holds its proxy key at ``[s, t, 0]`` and its z logits at
        ``[s, t, 1]``. None for a layer without indexer state on this rank."""
        if layer_idx not in self.local_sparse_layer_ids:
            return None
        pages = self._slot_pages(layer_idx, Role.INDEX_KEY)[:, 0]
        kz_bytes = self._kz_page_bytes(self.tokens_per_block)
        return self._typed(
            pages[:, :kz_bytes],
            STEP5_INDEX_KZ_DTYPE,
            [self.tokens_per_block, STEP5_INDEX_KZ_PLANES, self.step5_geometry.proxy_dim],
        )

    def get_index_summary_buffer(self, layer_idx: int) -> Optional[torch.Tensor]:
        """fp8 e4m3 ``[slots, tokens_per_block // region_block_size, proxy_dim]``:
        ``[s, r]`` is the summary of region ``r`` of slot ``s``'s block, valid
        once the region's last token is written. None for a layer without
        indexer state on this rank."""
        if layer_idx not in self.local_sparse_layer_ids:
            return None
        pages = self._slot_pages(layer_idx, Role.INDEX_KEY)[:, 0]
        kz_bytes = self._kz_page_bytes(self.tokens_per_block)
        return self._typed(
            pages[:, kz_bytes:],
            STEP5_INDEX_SUMMARY_DTYPE,
            [self.step5_regions_per_block, self.step5_geometry.proxy_dim],
        )

    def _kv_page_shape(self, layer_idx: int, width: int, kv_layout: str) -> List[int]:
        if kv_layout not in ("NHD", "HND"):
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        heads = self.num_kv_heads_per_layer[self.layer_offsets[layer_idx]]
        if kv_layout == "NHD":
            return [self.tokens_per_block, heads, width]
        return [heads, self.tokens_per_block, width]

    def get_kv_slot_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> Optional[torch.Tensor]:
        """``[slots, 2, tokens_per_block, heads, width]`` (NHD) or ``[slots, 2,
        heads, tokens_per_block, width]`` (HND), K then V, for any local
        attention layer, slot-indexed like the indexer views.

        bf16 and fp8 pools are viewed in their element type at ``width`` = the
        padded head width; an NVFP4 pool as uint8 at half that (two e2m1 values
        per byte, element ``2i`` in the low nibble), with its block scales in
        :meth:`get_kv_block_scale_slot_buffers`.
        """
        if layer_idx not in self.layer_offsets:
            return None
        head_dim = self.head_dim_per_layer[self.layer_offsets[layer_idx]]
        bits = _KV_ELEMENT_BITS.get(self.dtype)
        if bits is None:
            raise ValueError(f"Step-5 does not serve a {self.dtype} KV cache")
        if self.dtype == DataType.NVFP4:
            dtype, width = torch.uint8, head_dim * bits // 8
        else:
            dtype, width = binding_to_torch_dtype(self.dtype), head_dim
        pages = self._slot_pages(layer_idx, Role.KEY, planes=len(_MAIN_KV_ROLES))
        return self._typed(pages, dtype, self._kv_page_shape(layer_idx, width, kv_layout))

    def get_kv_block_scale_slot_buffers(
        self, layer_idx: int, kv_layout: str = "NHD"
    ) -> Optional[torch.Tensor]:
        """An NVFP4 pool's e4m3 block scales, K then V, as uint8 ``[slots, 2,
        tokens_per_block, heads, head_dim // 16]`` (NHD) or the HND order, with
        the slot indexing of :meth:`get_kv_slot_buffers`. The view is the page as
        bytes; the order within a page is the attention op's."""
        if self.dtype != DataType.NVFP4:
            raise ValueError(f"block scales exist only for an NVFP4 pool, not {self.dtype}")
        if layer_idx not in self.layer_offsets:
            return None
        head_dim = self.head_dim_per_layer[self.layer_offsets[layer_idx]]
        pages = self._slot_pages(layer_idx, Role.KEY_BLOCK_SCALE, planes=len(_BLOCK_SCALE_ROLES))
        return self._typed(
            pages,
            torch.uint8,
            self._kv_page_shape(layer_idx, head_dim // _NVFP4_BLOCK_SIZE, kv_layout),
        )

    def get_block_offset_slot_scale(self, layer_idx: int) -> tuple[int, int]:
        """``(pool, scale)``: the K lane of ``kv_cache_block_offsets[pool]``
        holds ``slot * scale`` for this layer's blocks, so a forward turns the
        engine's device block table into the slot ids every view here takes
        without a host round trip. One pair serves all sparse layers."""
        local = self.layer_offsets[layer_idx]
        pool = self.layer_to_pool_mapping_dict[local]
        return pool, int(self.index_scales[pool])

    def get_slot_table(self, request_ids: List[int], layer_idx: int) -> List[List[int]]:
        """Host slot ids of each request's blocks in ``layer_idx``'s life cycle."""
        return self.get_batch_cache_indices(request_ids, layer_idx=layer_idx, raw_indices=True)


__all__ = ["Step5CacheManager", "Step5PoolSplit"]

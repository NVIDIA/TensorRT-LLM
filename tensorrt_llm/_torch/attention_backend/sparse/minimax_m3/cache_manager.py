# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV cache management for MiniMax-M3 sparse attention.

Two complementary classes live here:

  * :class:`MiniMaxM3SparseIndexCache` -- a standalone plain-tensor
    side cache used by focused unit tests that exercise the algorithm
    outside the pyexecutor.
  * :func:`get_minimax_m3_kv_cache_manager_cls` -- lazy factory for the
    :class:`MiniMaxM3KVCacheManagerV2` subclass that integrates with
    :class:`tensorrt_llm._torch.pyexecutor.resource_manager.KVCacheManagerV2`,
    registering a per-sparse-layer ``Role.INDEX_KEY`` buffer in the V2
    paged lifecycle and providing M3-specific overrides for
    multi-buffer-per-layer storage layouts.
"""

from __future__ import annotations

import functools
from typing import List, Optional

import torch


class MiniMaxM3SparseIndexCache:
    """Side cache for the MiniMax-M3 sparse index branch.

    Slot layout matches the main ``KVCacheManagerV2`` paged buffer
    geometry so the same ``req_to_token`` mapping addresses both
    caches. One ``[num_slots, 1, sparse_index_dim]`` buffer is
    allocated per layer; a parallel ``index_v`` buffer is allocated
    when ``disable_index_value=False`` for that layer.

    The M3 checkpoint sets ``disable_index_value=True`` on every
    sparse layer (the index V branch is omitted entirely). The
    ``disable_index_value`` constructor list lets a future config
    variant flip individual layers; passing an empty
    ``disable_index_value_layer_ids`` list allocates index V for all
    sparse layers.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        sparse_layer_ids: List[int],
        disable_index_value_layer_ids: List[int],
        num_slots: int,
        sparse_index_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if num_slots <= 0:
            raise ValueError(f"num_slots must be > 0, got {num_slots}")
        if sparse_index_dim <= 0:
            raise ValueError(f"sparse_index_dim must be > 0, got {sparse_index_dim}")
        self.num_layers = int(num_layers)
        self.sparse_layer_ids = sorted(int(i) for i in sparse_layer_ids)
        self.disable_index_value_layer_ids = set(int(i) for i in disable_index_value_layer_ids)
        self.num_slots = int(num_slots)
        self.sparse_index_dim = int(sparse_index_dim)
        self.dtype = dtype
        self.device = device

        # Layer -> [num_slots, 1, sparse_index_dim] buffer.
        self._index_k: dict[int, torch.Tensor] = {}
        self._index_v: dict[int, torch.Tensor] = {}
        for layer_idx in self.sparse_layer_ids:
            if not (0 <= layer_idx < self.num_layers):
                raise ValueError(f"sparse layer_idx {layer_idx} outside [0, {self.num_layers})")
            self._index_k[layer_idx] = torch.zeros(
                (self.num_slots, 1, self.sparse_index_dim),
                dtype=dtype,
                device=device,
            )
            if layer_idx not in self.disable_index_value_layer_ids:
                self._index_v[layer_idx] = torch.zeros(
                    (self.num_slots, 1, self.sparse_index_dim),
                    dtype=dtype,
                    device=device,
                )

    def has_index_value(self, layer_idx: int) -> bool:
        return layer_idx in self._index_v

    def get_index_k_buffer(self, layer_idx: int) -> torch.Tensor:
        if layer_idx not in self._index_k:
            raise KeyError(
                f"layer_idx {layer_idx} is not a sparse layer; "
                f"sparse layers: {self.sparse_layer_ids}"
            )
        return self._index_k[layer_idx]

    def get_index_v_buffer(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._index_v.get(layer_idx)

    def set_index_k(self, layer_idx: int, out_cache_loc: torch.Tensor, idx_k: torch.Tensor) -> None:
        """Write ``idx_k`` into the index-K cache at the given slot indices.

        Args:
            layer_idx: sparse layer index.
            out_cache_loc: ``[num_tokens]`` int32/int64 slot indices.
            idx_k: ``[num_tokens, 1, sparse_index_dim]`` source.
        """
        buf = self.get_index_k_buffer(layer_idx)
        if idx_k.shape[1] != 1:
            raise ValueError(
                f"index K is replicated single-head; expected shape "
                f"[N, 1, {self.sparse_index_dim}], got {tuple(idx_k.shape)}"
            )
        buf.index_copy_(0, out_cache_loc.to(torch.long), idx_k.to(buf.dtype))

    def set_index_v(self, layer_idx: int, out_cache_loc: torch.Tensor, idx_v: torch.Tensor) -> None:
        """Write ``idx_v`` into the index-V cache (only when allocated)."""
        buf = self.get_index_v_buffer(layer_idx)
        if buf is None:
            raise RuntimeError(
                f"layer {layer_idx} has disable_index_value=True; index V is not allocated"
            )
        if idx_v.shape[1] != 1:
            raise ValueError(
                f"index V is replicated single-head; expected shape "
                f"[N, 1, {self.sparse_index_dim}], got {tuple(idx_v.shape)}"
            )
        buf.index_copy_(0, out_cache_loc.to(torch.long), idx_v.to(buf.dtype))


@functools.lru_cache(maxsize=1)
def get_minimax_m3_kv_cache_manager_cls():
    """Return :class:`MiniMaxM3KVCacheManagerV2` (lazy import).

    The pyexecutor module imports CUDA / C++ bindings; importing it at
    module-load time would couple every consumer of this file to those
    bindings, including the algorithm-only test path. Returning the
    class lazily lets :func:`get_sparse_attn_kv_cache_manager` register
    it without forcing the import.
    """
    from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig

    from ....pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, Role

    class MiniMaxM3KVCacheManagerV2(KVCacheManagerV2):
        """KVCacheManagerV2 subclass with a V2-managed paged index-K
        cache per sparse layer.

        The index-K side cache participates in the native
        ``KVCacheManagerV2`` paged lifecycle (allocation, free, slot
        reuse, prefix reuse) sharing the same ``life_cycle_id`` as the
        main K/V buffers.

        The cache is registered through the
        :meth:`_extra_buffers_per_layer` hook on the base manager: each
        sparse local layer gets a per-layer
        ``BufferConfig(role=Role.INDEX_KEY,
        size=1 * sparse_index_dim * dtype_bytes * tokens_per_block)``
        added alongside the standard K/V buffers. The base manager
        groups buffers by ``(life_cycle_id, size)``, so INDEX_KEY ends
        up in its own pool (its size differs from K/V's) but shares the
        lifecycle of the sparse layer's main K/V.

        :meth:`get_index_k_buffer` returns the V2 4-D paged view
        ``[num_pages, tokens_per_block, 1, sparse_index_dim]``
        directly. The sparse forward path consumes this layout natively
        via the same ``(page, within)`` decomposition the main K/V
        path uses, so reads go through the paged gather helper
        (multi-dim fancy indexing) and writes go through the paged
        scatter helper (multi-dim fancy assignment).

        The optional index-V branch is kept as a plain CUDA tensor for
        the rare ``disable_index_value=False`` case (focused tests).
        The M3 production checkpoint sets ``disable_index_value=True``
        on every sparse layer, so ``_index_v_buffers`` stays empty in
        production.

        Construction arguments:

          ``sparse_layer_ids``                — list of layer indices that
                                                use sparse attention.
          ``disable_index_value_layer_ids``   — subset of
                                                ``sparse_layer_ids``
                                                whose index V is
                                                **omitted** (the bring-up
                                                target).
          ``sparse_index_dim``                — width of the index K/V
                                                vectors.

        All other constructor arguments are forwarded to
        :class:`KVCacheManagerV2`.
        """

        def __init__(
            self,
            *args,
            sparse_layer_ids=None,
            disable_index_value_layer_ids=None,
            sparse_index_dim: Optional[int] = None,
            **kwargs,
        ):
            # The standard ``_create_kv_cache_manager`` factory in
            # ``_util.py`` builds the manager with the production
            # kwargs (``kv_cache_config``, ``num_layers``,
            # ``num_kv_heads``, ``head_dim``, ``tokens_per_block``,
            # ``max_seq_len``, ``mapping``, ``sparse_attn_config``, ...).
            # It does *not* know the M3-specific per-layer split
            # (which layers are sparse, which omit the index V branch,
            # the index Q/K width), so the M3 manager fills these in
            # from two sources, in priority order:
            #
            #   1. Explicit kwargs (used by the focused unit tests
            #      under ``tests/unittest/_torch/attention/sparse/``,
            #      which construct the manager directly).
            #   2. The ``sparse_attn_config`` kwarg the factory does
            #      pass — its ``sparse_index_dim`` covers index width.
            #   3. The M3 checkpoint convention as a final fallback:
            #      layers 0-2 are dense, 3-(num_layers-1) are sparse,
            #      every sparse layer sets ``disable_index_value=True``,
            #      and ``sparse_index_dim == 128``. The M3
            #      ``configuration_minimax_m3_vl.py`` sets these values
            #      verbatim, so the production LLM-API construction
            #      path falls through to these defaults.
            sparse_attn_config = kwargs.get("sparse_attn_config")
            num_layers = kwargs.get("num_layers")

            if sparse_index_dim is None:
                sparse_index_dim = (
                    int(getattr(sparse_attn_config, "sparse_index_dim", 0) or 0) or 128
                )
            if sparse_layer_ids is None:
                if num_layers is not None:
                    sparse_layer_ids = list(range(3, int(num_layers)))
                else:
                    sparse_layer_ids = []
            if disable_index_value_layer_ids is None:
                disable_index_value_layer_ids = list(sparse_layer_ids)

            # Resolve M3-specific sparse-layer metadata BEFORE
            # ``super().__init__()``. The base ``_build_cache_config``
            # is called during ``__init__`` and invokes our
            # ``_extra_buffers_per_layer`` hook, which reads these
            # attributes to size the INDEX_KEY ``BufferConfig`` per
            # sparse local layer.
            self.sparse_layer_ids = sorted(int(i) for i in sparse_layer_ids)
            self.disable_index_value_layer_ids = set(int(i) for i in disable_index_value_layer_ids)
            self.sparse_index_dim = int(sparse_index_dim)

            super().__init__(*args, **kwargs)

            # Optional plain-tensor index-V side cache for non-disabled
            # sparse layers. The M3 production checkpoint sets
            # ``disable_index_value=True`` everywhere, so this dict
            # stays empty in production. Focused tests that exercise
            # ``disable_index_value=False`` still receive the V buffers
            # they expect, mirroring the legacy behavior.
            num_total_slots = self._compute_num_total_slots()
            torch_dtype = self._torch_dtype_for_index_cache()
            device = torch.device("cuda")
            self._index_v_buffers: dict[int, torch.Tensor] = {}
            for layer_idx in self.sparse_layer_ids:
                if layer_idx not in self.layer_offsets:
                    continue
                if layer_idx not in self.disable_index_value_layer_ids:
                    self._index_v_buffers[layer_idx] = torch.zeros(
                        (num_total_slots, 1, self.sparse_index_dim),
                        dtype=torch_dtype,
                        device=device,
                    )

        def _extra_buffers_per_layer(self, *, tokens_per_block):
            """Register a per-sparse-layer ``Role.INDEX_KEY``
            :class:`BufferConfig` so the M3 index-K cache participates
            in the V2 paged lifecycle.

            ``size`` is bytes per **block** (matching the contract used
            for the standard K/V/scale buffers):
            ``num_heads=1 * sparse_index_dim * elem_bytes *
            tokens_per_block``. The dtype here must match the dtype
            used when wrapping the pool as a torch view in
            :meth:`get_index_k_buffer` so the V2 page-stride contract
            check in the base accessor passes.

            The returned dict is keyed by **local** layer id (i.e.
            ``self.layer_offsets[global_layer_id]``), because the base
            :meth:`KVCacheManagerV2._build_cache_config` iterates
            ``for layer_id in typed_range(LayerId(self.num_local_layers))``
            and does ``extra_buffers_per_layer.get(int(layer_id), ())``.
            Keying by global sparse layer id would silently skip
            registration on every nontrivial PP rank where local and
            global ids differ (e.g. on ``pp_size=2`` rank 1 with
            ``pp_layers=[2, 3]`` no local layer id ``{0, 1}`` would ever
            match a global key ``{2, 3}``). Only local sparse layers
            register the buffer; non-local layers (different PP rank)
            are filtered out via the ``layer_offsets`` membership check.
            """
            torch_dtype = self._torch_dtype_for_index_cache()
            elem_bytes = torch.tensor([], dtype=torch_dtype).element_size()
            bytes_per_token = 1 * self.sparse_index_dim * elem_bytes
            size_per_block = bytes_per_token * tokens_per_block
            return {
                self.layer_offsets[layer_id]: [
                    BufferConfig(role=Role.INDEX_KEY, size=size_per_block)
                ]
                for layer_id in self.sparse_layer_ids
                if layer_id in self.layer_offsets
            }

        def _compute_num_total_slots(self) -> int:
            """Total token slots across all blocks in the main K pool.

            Used to size the optional plain-tensor index-V cache (the
            index-K cache is now V2-managed and obtains its slot count
            from ``Role.INDEX_KEY``'s page upper bound).
            ``KVCacheManagerV2``'s block pool size is determined by
            ``quota / bytes_per_token`` plus a per-rank allreduce; the
            side cache must be at least that large. Uses
            ``get_page_index_upper_bound`` for the first layer's KEY
            pool times ``tokens_per_block``.
            """
            from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp

            # Pick any allocated layer offset; if none exist (degenerate
            # case for tests), fall back to the configured cap.
            if not self.layer_offsets:
                return int(self.max_batch_size * self.max_seq_len)
            any_layer_offset = next(iter(self.layer_offsets.values()))
            page_upper = self.impl.get_page_index_upper_bound(any_layer_offset, Role.KEY)
            kv_factor = 1 if self.kv_cache_type == CacheTypeCpp.SELFKONLY else 2
            return int((page_upper // kv_factor) * self.tokens_per_block)

        def _torch_dtype_for_index_cache(self) -> torch.dtype:
            """Match the main cache dtype where possible, fall back to bf16.

            The M3 checkpoint stores ``index_k_proj`` in BF16 (it's in
            the MXFP8 ignored_layers set) so BF16 is the right default.
            Heavier-precision (FP16/FP32) is supported via the same
            mapping ``KVCacheManagerV2`` already implements for the
            main cache.
            """
            from tensorrt_llm.bindings import DataType

            if self.dtype == DataType.HALF:
                return torch.float16
            if self.dtype == DataType.FLOAT:
                return torch.float32
            return torch.bfloat16

        def get_index_k_buffer(self, layer_idx: int) -> Optional[torch.Tensor]:
            """Return the V2-managed index-K buffer for sparse layer
            ``layer_idx`` as a 4-D paged view, or ``None`` for dense
            / non-local layers.

            Returns the base V2 paged view
            ``[num_pages, tokens_per_block, 1, sparse_index_dim]``.
            The sparse forward path consumes this layout natively:

              * **Reads** decompose the flat slot id into
                ``(page = s // tokens_per_block,
                within = s % tokens_per_block)`` and use multi-dim
                fancy indexing to read directly from the V2 pool.
              * **Writes** use the same ``(page, within)``
                decomposition with multi-dim fancy assignment so the
                assignment propagates to the underlying pool storage.

            Writes through this view propagate to the underlying V2
            pool and survive slot free/reuse / prefix-reuse cycles.
            """
            return super().get_index_k_buffer(
                layer_idx,
                num_heads=1,
                head_dim=self.sparse_index_dim,
                dtype=self._torch_dtype_for_index_cache(),
            )

        def get_index_v_buffer(self, layer_idx: int) -> Optional[torch.Tensor]:
            """Plain-tensor index-V cache for non-disabled sparse
            layers, else ``None``. The M3 production checkpoint sets
            ``disable_index_value=True`` everywhere, so this returns
            ``None`` in production. Focused tests that exercise the V
            branch still receive the per-layer V buffer they expect.
            """
            return self._index_v_buffers.get(layer_idx)

        def has_index_value(self, layer_idx: int) -> bool:
            return layer_idx in self._index_v_buffers

        def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> Optional[torch.Tensor]:
            """Return the K + V multi-dim paged view for ``layer_idx``,
            with strides that correctly span the V2 multi-layer
            coalesced pool.

            The base :meth:`KVCacheManagerV2.get_buffers` computes the
            view shape as ``[page_upper_K // kv_factor, kv_factor,
            tokens_per_block, num_kv_heads, head_dim]`` and lets the
            torch :class:`TensorWrapper` produce contiguous strides.
            That works only when the per-layer slot in the underlying
            pool contains exactly ``kv_factor`` page-sized buffers
            (K + V). At production MiniMax-M3 geometry — ``num_kv_heads
            = 1, head_dim = 128, sparse_index_dim = 128, BF16`` — K,
            V, and INDEX_KEY all share ``single_buffer_size = 256
            bytes/token``, so V2 storage coalesces them into one pool
            indexed by ``(life_cycle_id, single_buffer_size)``. The
            resulting per-slot layout interleaves all layers' K, V,
            (and INDEX_K for sparse layers) — e.g. for a 4-layer
            M3 setup with 3 sparse layers, one slot = ``[K@0, V@0,
            K@1, V@1, INDEX_K@1, K@2, V@2, INDEX_K@2, K@3, V@3,
            INDEX_K@3]`` (11 page-sized buffers per slot). The
            ``scale`` value returned by
            :meth:`get_page_index_converter` is exactly that
            per-slot buffer count.

            With ``scale > 2``, the base view's logical-page-to-
            physical-page mapping is wrong: moving from logical page 0
            to logical page 1 with the contiguous ``[N, 2, ...]``
            shape jumps **two** physical pages, but the next K-V for
            this layer is ``scale`` physical pages away (because the
            other layers' buffers sit in between). At runtime this
            silently corrupts every read or write past slot 0:
            ``pool[1, 0, ...]`` returns another layer's data, not the
            same layer's slot-1 K data.

            The override builds a strided view that respects the real
            per-slot stride:

              * dim 0 (slot index): stride = ``scale * page_stride``
                bytes (skip the whole slot to reach the next slot's
                copy of this layer's K).
              * dim 1 (K vs V): stride = ``page_stride`` bytes (V sits
                immediately after K within the slot — V2's
                ``addr_K + page_stride == addr_V`` invariant, asserted
                below).
              * dims 2-4: contiguous within the page (tokens_per_block,
                num_kv_heads, head_dim).

            Implementation: wrap raw pool memory at ``addr_K`` with a
            torch view shape ``[num_slots, scale, tokens_per_block,
            num_kv_heads, head_dim]`` (the slot's full buffer layout),
            then slice ``[:, :2]`` to extract the K-V pair. The slice
            preserves the dim-0 stride (= ``scale * page_stride`` in
            elements), so ``pool[s, 0, w, h, d]`` reaches the right
            byte for any slot ``s`` in the coalesced pool.

            ``num_slots`` is recovered from the storage layout:
            ``page_upper_K + layer_offset_K_pages == scale * num_slots``,
            so ``num_slots = (page_upper_K + layer_offset_pages) //
            scale``. The arithmetic is exact for any valid V2 storage
            config.

            Read/write contract is unchanged for callers: the paged
            scatter helper decomposes the per-token slot id into
            ``(page = s // tokens_per_block,
            within = s % tokens_per_block)`` and uses multi-dim fancy
            assignment. With the corrected dim-0 stride that
            assignment now lands at the right slot of *this layer's*
            K data, not at some other layer's K or INDEX_K data.
            """
            from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
            from tensorrt_llm._utils import (
                TensorWrapper,
                binding_to_torch_dtype,
                convert_to_torch_tensor,
            )
            from tensorrt_llm.bindings import DataType

            if kv_layout not in ("NHD", "HND"):
                raise ValueError(f"Unsupported kv_layout: {kv_layout}")
            if self.kv_cache_type == CacheTypeCpp.SELFKONLY:
                # Selfk-only path has no V buffer, which would make the
                # kv_factor==2 contract meaningless. The base class
                # implementation also assumes kv_factor==2 for this
                # code path, so do not exercise this branch from the
                # M3 cache manager today.
                raise NotImplementedError(
                    "MiniMaxM3KVCacheManagerV2.get_buffers does not support SELFKONLY cache type"
                )

            layer_offset = self.layer_offsets[layer_idx]
            addr_key = self.impl.get_mem_pool_base_address(layer_offset, Role.KEY)
            addr_value = self.impl.get_mem_pool_base_address(layer_offset, Role.VALUE)
            page_stride_key = self.impl.get_page_stride(layer_offset, Role.KEY)
            page_stride_value = self.impl.get_page_stride(layer_offset, Role.VALUE)
            # V2 always lays V immediately after K within the per-layer
            # contribution to a slot. The override depends on this
            # invariant — if it breaks, the slice ``[:, :2]`` below
            # stops returning a K-V pair and the read/write semantics
            # silently drift.
            assert addr_key + page_stride_value == addr_value, (
                f"MiniMaxM3 get_buffers requires addr_K + page_stride "
                f"== addr_V (V immediately after K in slot); got "
                f"addr_K={addr_key} page_stride_V={page_stride_value} "
                f"addr_V={addr_value} for layer {layer_idx}."
            )
            assert page_stride_key == page_stride_value, (
                f"MiniMaxM3 get_buffers requires equal K and V page "
                f"strides; got K={page_stride_key} V="
                f"{page_stride_value}."
            )

            # ``scale`` = buffers-per-slot in the coalesced pool.
            # ``layer_offset_pages`` = this layer's K offset (in
            # page-sized units) within the slot.
            converter = self.impl.get_page_index_converter(layer_offset, Role.KEY)
            scale = int(converter.scale)
            layer_offset_pages = int(converter.layer_offset)
            page_upper_K = self.impl.get_page_index_upper_bound(layer_offset, Role.KEY)
            # ``scale * num_slots == page_upper_K + layer_offset_pages``
            # (the formula in
            # ``KVCacheManager.get_page_index_upper_bound``). Solve for
            # num_slots.
            num_slots_total = page_upper_K + layer_offset_pages
            assert num_slots_total % scale == 0, (
                f"V2 storage inconsistency: page_upper_K + "
                f"layer_offset_pages = {num_slots_total} is not "
                f"divisible by scale = {scale}."
            )
            num_slots = num_slots_total // scale

            element_per_container = 1
            dtype = self.dtype
            if dtype == DataType.NVFP4:
                element_per_container = 2
                torch_dtype = torch.int8
            else:
                torch_dtype = binding_to_torch_dtype(dtype)

            layer_head_dim = self.head_dim_per_layer[layer_offset]
            num_kv_heads = self.num_kv_heads_per_layer[layer_offset]

            # View the per-slot stripe at K's base as a contiguous
            # ``[num_slots, scale, ...]`` tensor: dim 1 indexes into the
            # ``scale`` page-sized buffers starting at K (so dim 1 = 0
            # is K, dim 1 = 1 is V, dim 1 = 2+ is the next layer's K
            # / V / INDEX_K, ignored by the K-V slice below).
            if kv_layout == "NHD":
                full_slot_shape = [
                    num_slots,
                    scale,
                    self.tokens_per_block,
                    num_kv_heads,
                    layer_head_dim // element_per_container,
                ]
            else:
                full_slot_shape = [
                    num_slots,
                    scale,
                    num_kv_heads,
                    self.tokens_per_block,
                    layer_head_dim // element_per_container,
                ]

            full_view = convert_to_torch_tensor(
                TensorWrapper(addr_key, torch_dtype, full_slot_shape)
            )
            # Slice the first two slot entries: K (slot-offset 0
            # relative to this layer's K base) and V (slot-offset 1).
            # The slice preserves dim-0 stride = ``scale * (page_size /
            # dtype_bytes)`` elements, dim-1 stride = ``page_size /
            # dtype_bytes`` elements, so ``view[s, 0, w, h, d]`` lands
            # on the correct byte for any ``s`` in [0, num_slots).
            return full_view[:, :2]

        def _build_pool_mapping_tensors(self):
            """Override the base K+V-only ``exact_div`` formula so the
            M3 INDEX_KEY buffer is allowed to coalesce with K/V in the
            sparse pool group.

            The base :meth:`KVCacheManagerV2._build_pool_mapping_tensors`
            computes ``offset = exact_div(addr_offset, key_bytes *
            kv_factor * tokens_per_block)`` and asserts the division
            is exact. That assumption holds when each layer's
            contribution to its pool group is exactly K + V (so the
            per-layer stride is ``2 * key_single_buffer_size``).

            With the ``Role.INDEX_KEY`` registration, the M3 sparse
            layers add a third per-block buffer. The V2 storage groups
            buffers by ``(life_cycle_id, single_buffer_size)``, so if
            INDEX_KEY's per-block size **coincidentally** equals K's
            per-block size, K, V, and INDEX_KEY end up in the SAME
            coalesced buffer. The per-layer stride in that coalesced
            buffer is then ``3 * single_buffer_size``, not the
            ``2 * single_buffer_size`` the base formula assumes — and
            ``exact_div`` asserts.

            This is exactly what happens at the production MiniMax-M3
            configuration with TP=8: ``num_kv_heads=4`` divided by
            ``tp_size=8`` rounds **up** to ``num_kv_heads_per_rank=1``,
            so ``key_bytes_per_token_per_rank = 1 * head_dim=128 * 2 =
            256`` matches ``index_k_bytes_per_token = 1 *
            sparse_index_dim=128 * 2 = 256``. K, V and INDEX_KEY all
            have the same per-block size, all three coalesce, and the
            base formula divides ``3 * single_buffer_size /
            (2 * single_buffer_size) = 1.5`` per layer step — which
            ``exact_div`` rejects.

            The fix is to compute ``offset`` directly from
            ``self.impl.layer_grouping[group_id]`` (the ordered list of
            layers in the pool group). The ``offset`` is then exactly
            ``layer_position_in_group`` regardless of whether INDEX_KEY
            shares a coalesced buffer with K/V. For dense layers (only
            K + V in the pool) this matches the base formula's result;
            for sparse layers it sidesteps the broken assertion while
            keeping the same semantic (layer position within group),
            because the M3 sparse forward path consumes the paged K /
            V / INDEX_KEY views via :meth:`get_buffers` and
            :meth:`get_index_k_buffer` rather than the
            ``kv_cache_pool_mapping`` tensor.

            The NVFP4 KEY_BLOCK_SCALE branch is preserved unchanged for
            non-M3 paths; M3 itself does not use NVFP4 for KV today.
            """
            from tensorrt_llm._utils import prefer_pinned
            from tensorrt_llm.bindings import DataType
            from tensorrt_llm.runtime.kv_cache_manager_v2 import LayerId
            from tensorrt_llm.runtime.kv_cache_manager_v2._utils import typed_range

            kv_cache_pool_pointers = torch.tensor(
                [
                    [
                        self.impl.get_mem_pool_base_address(
                            self.impl.layer_grouping[pool_id][0], Role.KEY
                        ),
                        0,
                    ]
                    for pool_id in range(self.num_pools)
                ],
                dtype=torch.int64,
                device="cpu",
                pin_memory=prefer_pinned(),
            )

            if self.dtype == DataType.NVFP4:
                kv_cache_pool_pointers = torch.stack(
                    [
                        kv_cache_pool_pointers,
                        torch.tensor(
                            [
                                [
                                    self.impl.get_mem_pool_base_address(
                                        self.impl.layer_grouping[pool_id][0], Role.KEY_BLOCK_SCALE
                                    ),
                                    0,
                                ]
                                for pool_id in range(self.num_pools)
                            ],
                            dtype=torch.int64,
                            device="cpu",
                            pin_memory=prefer_pinned(),
                        ),
                    ],
                    dim=-1,
                )

            kv_cache_pool_mapping_list = []
            for layer_id in typed_range(LayerId(self.num_local_layers)):
                layer_group_id = self.impl.get_layer_group_id(layer_id)
                layers_in_group = list(self.impl.layer_grouping[int(layer_group_id)])
                # ``layers_in_group`` is in storage layout order; the
                # layer's position there is what the base formula's
                # ``exact_div(addr_offset, key_stride)`` would have
                # produced for the K-only / K+V-only case. With INDEX_K
                # potentially coalesced, the direct position is the
                # only invariant left.
                offset = layers_in_group.index(int(layer_id))
                kv_cache_pool_mapping_list.append([int(layer_group_id), offset])

            kv_cache_pool_mapping = torch.tensor(
                kv_cache_pool_mapping_list,
                dtype=torch.int32,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            return kv_cache_pool_pointers, kv_cache_pool_mapping

        def _get_batch_cache_indices_by_pool_id(
            self,
            request_ids,
            *,
            pool_id: int = 0,
            is_kv_aggregate: bool = True,
        ):
            """Return per-request slot ids in ``[0, num_slots)`` directly.

            The base
            :meth:`KVCacheManagerV2._get_batch_cache_indices_by_pool_id`
            multiplies V2's ``base_page_indices`` (which are slot ids in
            ``[0, num_slots)``) by ``index_scales[pool_id]`` and divides
            by ``kv_factor`` to convert to a V1-style block id. That
            conversion is correct **only** when each layer in the pool
            group contributes exactly K+V (so
            ``scale == 2 * num_local_layers``). The downstream
            :meth:`KVCacheManagerV2.get_block_ids_per_seq` then does a
            second ``// num_local_layers`` step, which undoes the
            multiplication and recovers the slot id.

            With the ``Role.INDEX_KEY`` registration, the M3 sparse
            layers contribute K + V + INDEX_KEY (3 buffers) to their
            pool group. At production MiniMax-M3 geometry (TP=8,
            ``num_kv_heads_per_rank=1, head_dim=128,
            sparse_index_dim=128, BF16``) all three role sizes equal
            256 bytes/token, so V2 coalesces them and ``scale`` becomes
            ``6 (dense K/V) + 171 (sparse K/V/INDEX_K) = 177`` for 60
            layers. The V1-style math ``base_idx * 177 // 2 // 60``
            no longer equals ``base_idx`` — it overshoots, producing
            block ids well above ``num_slots`` and triggering CUDA
            ``IndexKernel.cu`` index-out-of-bounds aborts during V2
            warmup when the runtime decomposes
            ``slot_id = block_id * tokens_per_block + offset`` and the
            paged cache view :meth:`get_buffers` /
            :meth:`get_index_k_buffer` is indexed at the now-too-large
            ``block_id``.

            The fix bypasses the V1-style multiplication and returns
            raw slot ids. The M3 forward path
            (:meth:`_dense_forward`, :meth:`_sparse_forward`) and the
            paged views built by
            :meth:`KVCacheManagerV2.get_buffers` / the override above /
            :meth:`KVCacheManagerV2.get_index_k_buffer` use the slot id
            directly as the dim-0 index. ``BAD_PAGE_INDEX`` slots stay
            as 0 to match the legacy padding contract.

            ``is_kv_aggregate`` is ignored here because the slot id is
            already pool-group-local; there is no K/V interleaving to
            collapse.
            """
            from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX

            res = []
            for req_id in request_ids:
                idx_tensor = torch.as_tensor(
                    self.kv_cache_map[req_id].get_base_page_indices(pool_id)
                )
                res.append(
                    (
                        torch.where(
                            idx_tensor != BAD_PAGE_INDEX,
                            idx_tensor,
                            torch.full_like(idx_tensor, BAD_PAGE_INDEX),
                        )
                    ).tolist()
                )
            return res

        def get_block_ids_per_seq(self, request_ids):
            """Return per-request block ids that match the per-layer
            paged view's dim-0 (slot ids).

            Overrides
            :meth:`KVCacheManagerV2.get_block_ids_per_seq`'s final
            ``i // self.num_local_layers`` step. That step was paired
            with the base ``index_scales`` multiplication to recover
            slot ids in the standard K+V case; both have been removed
            here in favor of returning slot ids directly via
            :meth:`_get_batch_cache_indices_by_pool_id`. Pads with
            ``0`` (a safe block id) to match the legacy shape.
            """
            from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX

            block_ids_per_seq = self.get_batch_cache_indices(request_ids)
            block_ids_per_seq_tensors = [
                torch.tensor(
                    [i if i != BAD_PAGE_INDEX else 0 for i in sublist],
                    dtype=torch.int,
                )
                for sublist in block_ids_per_seq
            ]
            padded_tensor = torch.nn.utils.rnn.pad_sequence(
                block_ids_per_seq_tensors,
                batch_first=True,
                padding_value=0,
            )
            return padded_tensor

    return MiniMaxM3KVCacheManagerV2


__all__ = [
    "MiniMaxM3SparseIndexCache",
    "get_minimax_m3_kv_cache_manager_cls",
]

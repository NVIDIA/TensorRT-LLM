# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default-off residual-NVFP4 owner worklist using registered native operators.

Metadata owns graph-stable buffers. Native operators publish the worklist or
gather the next layer into its existing owner positions without allocation or
host/device synchronization in the gather hot path.
"""

from __future__ import annotations

import os
import weakref
from typing import TYPE_CHECKING

import torch

import tensorrt_llm.bindings

if TYPE_CHECKING:
    from .backend import DSATrtllmAttention
    from .metadata import DSAtrtllmAttentionMetadata

__all__ = ["Nvfp4OwnerWorklist"]

_MODE_ENV = "TRTLLM_DSA_NVFP4_OWNER_WORKLIST"
_HEAD = 576


def _tensor(tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype,
            device: torch.device) -> bool:
    return (isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == shape
            and tensor.dtype == dtype and tensor.device == device and tensor.is_contiguous())


class Nvfp4OwnerWorklist:
    """Own one graph-stable worklist, independent of shared ``Buffers`` names.

    Metadata must outlive every graph using these raw CUDA pointers. Recreating
    ordinary scratch buffers resets this object, but must not free/replace it.
    Whole captured forwards always contain a fresh clear/hash/list publication
    before their followers; isolated follower graphs are not supported.
    """

    def __init__(self, metadata: DSAtrtllmAttentionMetadata) -> None:
        self.mode = os.environ.get(_MODE_ENV, "off")
        if self.mode not in ("per_layer", "group"):
            raise ValueError(f"{_MODE_ENV} must be off, per_layer or group")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NVFP4 owner-worklist must be prepared before graph capture")
        device = metadata.nvfp4_mla_fp8_scratch.device
        self.device = device
        self._owner = weakref.ref(metadata)
        # Graph metadata is a shallow copy, including the native producer's
        # result dictionary. Isolate that mutable cache too, before any forward.
        metadata._group_remap_batched = {}
        self.map: torch.Tensor | None = None
        self.table: torch.Tensor | None = None
        self.owners: torch.Tensor | None = None
        self.count: torch.Tensor | None = None
        self._retained_buffers: list[tuple[torch.Tensor, ...]] = []
        self.ensure_capacity(metadata)

    def ensure_capacity(self, metadata: DSAtrtllmAttentionMetadata) -> None:
        """Grow before capture, retaining buffers referenced by older graphs."""
        scratch = metadata.nvfp4_mla_fp8_scratch
        rows, topk, _ = scratch.shape
        batch = metadata.max_num_sequences
        queries = max(1, metadata.max_draft_tokens + 1)
        # At most one key per selected pair enters each request's table. Keep
        # <= 50% worst-case load; unusual device query distributions can still
        # use the native exact self-owner fallback on a full table.
        slots = 1 << ((queries * topk - 1).bit_length() + 1)
        if (rows <= 0 or topk <= 0 or batch <= 0 or rows * topk > 2**31 - 1
                or slots > 2**30 or scratch.device != self.device):
            raise ValueError("NVFP4 owner-worklist capacity exceeds native index limits")
        if (self.map is not None and self.map.shape[0] >= rows
                and self.map.shape[1] == topk and self.table.shape[0] >= batch
                and self.table.shape[1] >= slots):
            self.reset()
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NVFP4 owner-worklist capacity must grow before graph capture")
        old_buffers = (
            (self.map, self.table, self.owners, self.count) if self.map is not None else None
        )
        with torch.cuda.device(self.device):
            # Deliberately do not use metadata.get_empty: Buffers reuses names
            # across metadata objects and would alias live A/B owner maps.
            new_map = torch.empty((rows, topk), dtype=torch.int32, device=self.device)
            new_table = torch.empty((batch, slots), dtype=torch.int64, device=self.device)
            new_owners = torch.empty((rows * topk,), dtype=torch.int32, device=self.device)
            new_count = torch.empty((), dtype=torch.int32, device=self.device)
        if old_buffers is not None:
            self._retained_buffers.append(old_buffers)
        self.map, self.table = new_map, new_table
        self.owners, self.count = new_owners, new_count
        self.reset()

    def owns(self, metadata: DSAtrtllmAttentionMetadata) -> bool:
        """Distinguish actual graph metadata from its shallow-copied parent."""
        return self._owner() is metadata

    def reset(self) -> None:
        """Invalidate host group state without reallocating graph-owned buffers."""
        self._active: tuple | None = None
        self._remap: tuple[int, int, torch.Tensor] | None = None
        self._eligible = False
        self._active_batch = 0
        self._active_rows = 0
        self._leader_call = False
        self.last_dispatch = "native"

    def begin_call(self, backend: DSATrtllmAttention,
                   metadata: DSAtrtllmAttentionMetadata, is_generation: bool) -> None:
        """Start one real backend invocation; new leaders always republish."""
        self._remap = None
        self._active_batch = 0
        self._active_rows = 0
        self.last_dispatch = "native"
        self._leader_call = backend.indexer is not None
        manager = metadata.kv_cache_manager
        batch = metadata.num_generations
        rows = metadata.num_tokens - metadata.num_ctx_tokens
        topk = self.map.shape[1]
        self._eligible = (
            is_generation and manager is not None
            and manager.dtype == tensorrt_llm.bindings.DataType.NVFP4
            and 0 < batch <= self.table.shape[0]
            and batch == metadata.num_seqs - metadata.num_contexts
            and 0 < rows <= self.map.shape[0]
            and metadata.num_sparse_topk == topk
            and manager.head_dim == _HEAD and manager.tokens_per_block > 0
            and manager.mla_kv_cache_residual_dim == 64
        )
        if self._eligible:
            self._active_batch = batch
            self._active_rows = rows
        # Always consume the native device request map, including live draft
        # updates. Saturation or an unexpected request ID falls back to an
        # independent owner in the native publisher, never a dropped KV row.
        if not self._eligible or self._leader_call:
            self._active = None

    def record_remap(self, leader: int, slot: int, batched: torch.Tensor) -> None:
        """Bind only the actual native grouped-remap result for this call."""
        self._remap = (leader, slot, batched)

    def _pool(self, metadata: DSAtrtllmAttentionMetadata,
              members: tuple[int, ...]) -> tuple | None:
        """Check one layer's layout, or a group's common affine namespace."""
        manager = metadata.kv_cache_manager
        pool = manager.get_unique_primary_pool()
        params = tuple(manager.get_primary_pool_page_index_params(i) for i in members)
        scale = params[0][0]
        block = manager.tokens_per_block
        extent = metadata._cached_num_pool_tokens
        # Native physical IDs add a layer offset within each physical block. A complete
        # common namespace, bounded offsets and int32-safe extent prove both
        # validity masks and equality classes are invariant between members.
        if (scale <= 0 or block <= 0
                or tuple(pool.shape) != (manager.blocks_in_primary_pool, scale, 1, block * 320)
                or pool.device != self.device or pool.dtype not in (torch.int8, torch.uint8)
                or not pool.is_contiguous() or pool.numel() != extent * 320
                or not 0 < extent <= 2**31 - 1 or extent % (scale * block)):
            return None
        offsets = tuple(offset for _, offset in params)
        if (any(page_scale != scale or not 0 <= offset < scale for page_scale, offset in params)
                or len(set(offsets)) != len(members)):
            return None
        pointers = metadata.host_kv_cache_pool_pointers
        mapping = metadata.host_kv_cache_pool_mapping
        if (pointers is None or mapping is None or pointers.device.type != "cpu"
                or mapping.device.type != "cpu" or pointers.dtype != torch.int64
                or mapping.dtype != torch.int32 or pointers.ndim != 3
                or tuple(pointers.shape[1:]) != (2, 2) or mapping.ndim != 2
                or mapping.shape[1] != 2 or not pointers.is_contiguous()
                or not mapping.is_contiguous() or min(members) < 0
                or max(members) >= mapping.shape[0]):
            return None
        pools = tuple(int(mapping[i, 0]) for i in members)
        if len(set(pools)) != 1 or not 0 <= pools[0] < pointers.shape[0]:
            return None
        data = int(pointers[pools[0], 0, 0])
        scales = int(pointers[pools[0], 0, 1])
        if (data != pool.data_ptr() or data <= 0 or scales <= 0
                or data % 16 or scales % 8 or scales in (data, data + 320)):
            return None
        return data, scales, extent, params

    def _group(self, backend: DSATrtllmAttention, metadata: DSAtrtllmAttentionMetadata,
               indices: torch.Tensor) -> tuple | None:
        if self._remap is None:
            return None
        leader, slot, batched = self._remap
        struct = metadata._group_remap_struct
        local_layer = backend.get_local_layer_idx(metadata)
        size = struct["group_size"].get(leader, 0) if struct else 0
        if (size < 2 or not 0 <= slot < size or self._leader_call != (slot == 0)
                or not _tensor(batched, (size, self._active_rows, self.map.shape[1]), torch.int32, self.device)
                or indices.data_ptr() != batched[slot].data_ptr()
                or struct["leader_of"][local_layer] != leader
                or struct["slot_of"][local_layer] != slot
                or metadata._group_remap_batched.get(leader) is not batched):
            return None
        members = tuple(i for i, value in enumerate(struct["leader_of"]) if value == leader)
        if len(members) != size or members[slot] != local_layer:
            return None
        pool = self._pool(metadata, members)
        if pool is None or struct["group_scale"][leader] != pool[3][0][0]:
            return None
        return leader, slot, batched, members, pool

    def try_gather(self, backend: DSATrtllmAttention, metadata: DSAtrtllmAttentionMetadata,
                   indices: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor | None:
        """Return separate owner IDs, or None before any write for native fallback.

        Args:
            indices: Actual native physical IDs, CUDA int32 [rows, topk].
            scratch: Existing native FP8 scratch [rows, topk, 576].
        """
        requests = metadata._cached_req_idx_gen
        scale = backend._nvfp4_mla_kv_scale_quant_orig
        rows = self._active_rows
        batch = self._active_batch
        topk = self.map.shape[1]
        if (not self._eligible or not _tensor(indices, (rows, topk), torch.int32, self.device)
                or not _tensor(scratch, (rows, topk, _HEAD), torch.float8_e4m3fn, self.device)
                or not _tensor(requests, (rows,), torch.int32, self.device)
                or scale.device != self.device or scale.dtype != torch.float32
                or not scale.is_contiguous() or scale.numel() != 1):
            self.reset()
            return None
        pool = self._pool(metadata, (backend.get_local_layer_idx(metadata),))
        if pool is None:
            self.reset()
            return None
        _, _, extent, _ = pool
        group = self._group(backend, metadata, indices) if self.mode == "group" else None
        stream = torch.cuda.current_stream(self.device).cuda_stream
        # The engine orders complete forwards using this metadata and native
        # scratch. Warmup and capture may use different, externally ordered
        # streams. A fresh leader is legal there, but a follower cannot cross
        # streams within one group. Concurrent unordered reuse of one metadata
        # object is unsupported, just as for its native FP8 scratch.
        key = None
        slot = -1
        if group is not None:
            leader, slot, batched, members, group_pool = group
            key = (batch, rows, leader, id(batched), members, group_pool, requests.data_ptr(),
                   metadata._cached_block_table_gen.data_ptr(), stream)
            if slot > 0 and self._active != (key, slot):
                key = None
        # A missing/interrupted group invalidates reuse, not basic per-layer
        # deduplication. Rebuild from this layer's current native physical IDs.
        reuse = key is not None and slot > 0
        if key is None:
            self._active = None
        # A prefix view allocates no GPU storage; captured ops retain the same
        # backing allocation owned by this metadata object.
        owner_map = self.map if rows == self.map.shape[0] else self.map[:rows]
        local_layer = backend.get_local_layer_idx(metadata)
        try:
            if not reuse:
                torch.ops.trtllm.nvfp4_mla_kv_cache_gather_worklist(
                    metadata.host_kv_cache_pool_pointers,
                    metadata.host_kv_cache_pool_mapping,
                    indices, requests, scratch, owner_map, self.table,
                    self.owners, self.count, scale, local_layer, batch, extent,
                )
                self.last_dispatch = "leader" if key is not None else "per_layer"
            else:
                torch.ops.trtllm.nvfp4_mla_kv_cache_gather_worklist_reuse(
                    metadata.host_kv_cache_pool_pointers,
                    metadata.host_kv_cache_pool_mapping,
                    indices, scratch, owner_map, self.owners, self.count,
                    scale, local_layer, extent,
                )
                self.last_dispatch = "follower"
        except RuntimeError:
            self.reset()
            raise
        self._active = (
            (key, slot + 1) if key is not None and slot + 1 < len(members) else None
        )
        return owner_map

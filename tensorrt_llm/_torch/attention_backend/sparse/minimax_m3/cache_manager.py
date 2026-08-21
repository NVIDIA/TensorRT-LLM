# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KV cache management for MiniMax-M3 sparse attention.

Provides:
  * :class:`MiniMaxM3SparseIndexCache` — plain-tensor side cache used by
    algorithm-only unit tests (no pyexecutor dependency required for
    construction).
  * :class:`MiniMaxM3KVCacheManagerV2` — :class:`KVCacheManagerV2`
    subclass that registers a per-sparse-layer ``Role.INDEX_KEY`` paged
    buffer alongside the standard K/V buffers.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._utils import (
    TensorWrapper,
    binding_to_torch_dtype,
    convert_to_torch_tensor,
    prefer_pinned,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, PageIndexMode
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX
from tensorrt_llm.runtime.kv_cache_manager_v2._config import DataRole

from ....pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, Role


class MiniMaxM3SparseIndexCache:
    """Plain-tensor side cache for the M3 sparse index branch.

    Slot layout matches the main ``KVCacheManagerV2`` paged buffer
    geometry so the same ``req_to_token`` mapping addresses both
    caches. One ``[num_slots, 1, sparse_index_dim]`` index-K buffer is
    allocated per sparse layer; index-V is allocated only for layers
    not listed in ``disable_index_value_layer_ids``.
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
        """Write ``idx_k`` into the index-K cache at ``out_cache_loc``."""
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


def derive_shared_draft_layout(
    num_layers: Optional[int],
    num_kv_heads,
    num_draft: int,
) -> tuple[list[int], Optional[int]]:
    """Locate the appended one-model draft tail in the manager's layer range.

    ``num_layers`` is ambiguous at the creation site: for M3 + Eagle3 it
    carries the pretrained TARGET count (60) while the per-layer
    ``num_kv_heads`` list is already extended with the draft entries
    (61); other flows pass the extended count directly. The heads list's
    length is the unambiguous total, so anchor on it and fall back to
    ``num_layers`` for scalar heads.

    Returns ``(draft_layer_ids, num_target_layers)``; the target range is
    ``[0, num_target_layers)`` and the draft tail sits directly above it.
    ``num_target_layers`` is ``None`` when neither input pins the range.
    """
    total = (
        len(num_kv_heads)
        if isinstance(num_kv_heads, (list, tuple))
        else (int(num_layers) if num_layers is not None else None)
    )
    if total is None:
        return [], None
    if num_layers is not None:
        total = max(total, int(num_layers))
    num_target = total - max(0, int(num_draft))
    draft_ids = list(range(num_target, total))
    return draft_ids, num_target


class MiniMaxM3KVCacheManagerV2(KVCacheManagerV2):
    """KVCacheManagerV2 subclass with a V2-managed paged index-K cache
    per sparse layer.

    Each sparse local layer registers a ``Role.INDEX_KEY``
    :class:`BufferConfig` via :meth:`_extra_buffers_per_layer` so the
    index-K cache participates in the V2 paged lifecycle (allocation,
    free, prefix reuse) and shares ``life_cycle_id`` with main K/V.

    The optional index-V branch is kept as a plain CUDA tensor for the
    rare ``disable_index_value=False`` test path. M3 production sets
    ``disable_index_value=True`` everywhere, so ``_index_v_buffers``
    stays empty in production.

    Constructor extras (forwarded kwargs go to :class:`KVCacheManagerV2`):
      * ``sparse_layer_ids`` — layer indices using sparse attention.
      * ``disable_index_value_layer_ids`` — subset whose index-V is
        omitted.
      * ``sparse_index_dim`` — width of the index-K/V vectors.
      * ``num_one_model_draft_layers`` — how many one-model draft layers
        the creation site appended after the target's (0 when the drafter
        is separate or speculation is off).
    """

    # One-model speculative draft layers share this manager (unified KV
    # cache): reuse, eviction, and disaggregated transfer then cover the
    # drafter's KV natively. Its attention addresses the physical P32 buffer
    # expansion through ``get_draft_subpage_view``.
    supports_shared_draft_layers = True

    # WAR: the Eagle draft kernels break at tokens_per_block=128 (the MSA
    # target's page size) — the SM103 context cubin is missing (its unfused
    # fallback demands a multi-TiB workspace) and the generation kernel hits
    # an illegal memory access — so the drafter runs at 32-token pages. This
    # value sizes both the separate draft manager and the view's sub-pages.
    # Retirement, once the kernels are fixed (WAR sites point here):
    #   1. Validate with TRTLLM_M3_DRAFT_KV_TOKENS_PER_BLOCK=128; the view
    #      degenerates to the identity expansion (unit-tested).
    #   2. Delete the WAR surface — MiniMaxM3DraftSubpageView,
    #      ``get_draft_subpage_view``, the ``add_dummy_requests`` override,
    #      and this attribute; the drafter then attends the shared manager
    #      directly (validated at acceptance parity, PR #17457).
    draft_manager_tokens_per_block = 32
    # The separately allocated Eagle layer cannot inherit an NVFP4 target
    # cache: the shipped TRTLLM-Gen set has no matching M3 P32 NVFP4 decode
    # cubin.  Keep the draft on its established FP8/P32 representation while
    # sparse target layers use NVFP4/P128.
    draft_manager_kv_cache_dtype = "fp8"
    nvfp4_dense_tokens_per_block = 32
    _main_kv_layout = "NHD"

    def __init__(
        self,
        *args,
        sparse_layer_ids=None,
        disable_index_value_layer_ids=None,
        sparse_index_dim: Optional[int] = None,
        num_one_model_draft_layers: int = 0,
        **kwargs,
    ):
        # Linear Eagle3 verification is a causal multi-token append and is
        # compatible with the NVFP4 data+scale pools below. Dynamic-tree
        # acceptance is different: its relocation op currently copies only
        # the packed K/V bytes, not the per-16-element NVFP4 scale bytes. A
        # relocated token would therefore pair new data with stale scales and
        # silently corrupt attention. Reject that configuration until the
        # relocation op accepts and moves the scale pools as well.
        spec_config = kwargs.get("spec_config")
        if (
            kwargs.get("dtype") == DataType.NVFP4
            and spec_config is not None
            and getattr(spec_config, "use_dynamic_tree", False)
        ):
            raise NotImplementedError(
                "MiniMax-M3 NVFP4 KV cache supports linear Eagle3, but not "
                "dynamic-tree Eagle: accepted-token relocation does not yet "
                "move NVFP4 K/V block scales."
            )

        # Resolve M3 sparse-layer metadata from explicit kwargs first,
        # then from ``sparse_attn_config``, then from the M3 checkpoint
        # convention (layers 0..2 dense, 3..N-1 sparse,
        # disable_index_value=True, sparse_index_dim=128).
        # num_layers / num_kv_heads / sparse_attn_config belong to the base
        # __init__: peeked here (get, not pop) so super() still receives them.
        sparse_attn_config = kwargs.get("sparse_attn_config") or kwargs.get(
            "sparse_attention_config"
        )
        num_layers = kwargs.get("num_layers")
        implementation = getattr(sparse_attn_config, "implementation", "triton")
        self._main_kv_layout = "HND" if implementation == "msa" else "NHD"

        if sparse_index_dim is None:
            sparse_index_dim = int(getattr(sparse_attn_config, "sparse_index_dim", 0) or 0) or 128
        # One-model speculative decoding with shared draft layers appends the
        # drafter's layers after the target's (dense, no MSA index cache);
        # ``_create_kv_cache_manager`` passes the appended count explicitly.
        self._shared_draft_layer_ids, num_target_layers = derive_shared_draft_layout(
            num_layers, kwargs.get("num_kv_heads"), num_one_model_draft_layers
        )
        if sparse_layer_ids is None:
            if num_target_layers is not None:
                sparse_layer_ids = list(range(3, num_target_layers))
            else:
                sparse_layer_ids = []
        if disable_index_value_layer_ids is None:
            disable_index_value_layer_ids = list(sparse_layer_ids)

        # Must be set BEFORE super().__init__ — the base
        # ``_build_base_config`` invokes ``_extra_buffers_per_layer``
        # which reads these attributes.
        self.sparse_layer_ids = sorted(int(i) for i in sparse_layer_ids)
        self.disable_index_value_layer_ids = set(int(i) for i in disable_index_value_layer_ids)
        self.sparse_index_dim = int(sparse_index_dim)
        self.indexer_kv_dtype = str(getattr(sparse_attn_config, "indexer_kv_dtype", "bf16"))
        if self.indexer_kv_dtype not in ("bf16", "fp8"):
            raise ValueError(
                "MiniMax M3 indexer_kv_dtype must be 'bf16' or 'fp8', got "
                f"{self.indexer_kv_dtype!r}."
            )
        if self.indexer_kv_dtype == "fp8" and (
            set(self.sparse_layer_ids) - self.disable_index_value_layer_ids
        ):
            raise ValueError(
                "MiniMax M3 FP8 index cache requires disable_index_value=True "
                "for every sparse layer."
            )

        super().__init__(*args, **kwargs)

        if self.dtype == DataType.NVFP4:
            dense_target_layers = [
                layer
                for layer in self.layer_offsets
                if layer not in self.sparse_layer_ids and layer not in self._shared_draft_layer_ids
            ]
            logger.info(
                "[m3-kv] hybrid cache active: "
                f"{len(self.sparse_layer_ids)} sparse target layer(s)=NVFP4/P128, "
                f"{len(dense_target_layers)} dense target layer(s)=FP8/P128, "
                f"{len(self._shared_draft_layer_ids)} shared Eagle layer(s)=FP8/P32"
            )

        self._draft_subpage_view_obj: Optional["MiniMaxM3DraftSubpageView"] = None
        if self._shared_draft_layer_ids and self.sparse_layer_ids and not self.is_draft:
            # Paired with the "view active" log at first dispatch.
            logger.info(
                f"[unified-kv] draft layers {self._shared_draft_layer_ids} "
                f"share the target KV cache manager (sparse layers "
                f"{self.sparse_layer_ids[0]}..{self.sparse_layer_ids[-1]})"
            )

        index_v_layer_ids = set(self.sparse_layer_ids) - self.disable_index_value_layer_ids
        if self.is_disagg and index_v_layer_ids:
            raise ValueError(
                "MiniMax M3 disaggregated serving requires disable_index_value=True "
                "for every sparse layer because the optional test-only index-V cache "
                "is not managed or transferred by KVCacheManagerV2; enabled layers="
                f"{sorted(index_v_layer_ids)}"
            )

        # Optional plain-tensor index-V cache for non-disabled sparse
        # layers (test-only; production has disable_index_value=True
        # on every sparse layer).
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

    def _build_cache_config(self, config):
        """Use NVFP4 only on MSA sparse layers and FP8 on dense/Eagle layers.

        M3's 57 sparse target layers have a native MSA NVFP4 consumer.  The
        three dense target layers and the appended one-model Eagle layer do
        not have a matching TRTLLM-Gen NVFP4 cubin, so those buffers retain
        the proven FP8 representation.  Target dense layers retain their
        established P128 layout; only the shared Eagle layer uses physical
        P32 pages because its SM100/SM103 kernels require that geometry.
        """
        if self.dtype != DataType.NVFP4:
            return super()._build_cache_config(config)

        physical_page = self.nvfp4_dense_tokens_per_block
        assert config.tokens_per_block % physical_page == 0, (
            f"M3 logical page P{config.tokens_per_block} must be divisible by "
            f"the dense/Eagle physical page P{physical_page}."
        )
        scale_roles = {Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE}
        for layer in config.layers:
            local_layer_idx = int(layer.layer_id)
            global_layer_idx = int(self.pp_layers[local_layer_idx])
            if global_layer_idx in self.sparse_layer_ids:
                continue
            layer.buffers[:] = [
                buffer for buffer in layer.buffers if buffer.role not in scale_roles
            ]
            for buffer in layer.buffers:
                if buffer.role not in (Role.KEY, Role.VALUE):
                    continue
                if global_layer_idx in self._shared_draft_layer_ids:
                    buffer.size = (
                        self.get_layer_bytes_per_token(local_layer_idx, buffer.role) * physical_page
                    )
                    buffer.tokens_per_block_override = physical_page
        return super()._build_cache_config(config)

    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: Role):
        """Report the hybrid sparse-NVFP4 / dense-FP8 storage footprint."""
        if self.dtype != DataType.NVFP4:
            return super().get_layer_bytes_per_token(local_layer_idx, data_role)
        global_layer_idx = int(self.pp_layers[int(local_layer_idx)])
        if global_layer_idx in self.sparse_layer_ids:
            return super().get_layer_bytes_per_token(local_layer_idx, data_role)

        if data_role in (Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE):
            return 0
        if data_role == Role.ALL:
            kv_factor = self.kv_factor
        elif data_role in (Role.KEY, Role.VALUE):
            kv_factor = 1
        else:
            return super().get_layer_bytes_per_token(local_layer_idx, data_role)
        return (
            kv_factor
            * self.num_kv_heads_per_layer[int(local_layer_idx)]
            * self.head_dim_per_layer[int(local_layer_idx)]
        )

    def is_nvfp4_layer(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` stores packed NVFP4 data and block scales."""
        return self.dtype == DataType.NVFP4 and int(layer_idx) in self.sparse_layer_ids

    def is_fp8_dense_layer(self, layer_idx: int) -> bool:
        """Whether a dense target/Eagle layer is the FP8 half of hybrid KV."""
        return self.dtype == DataType.NVFP4 and int(layer_idx) not in self.sparse_layer_ids

    def is_fp8_subpaged_layer(self, layer_idx: int) -> bool:
        """Whether the shared Eagle layer uses physical P32 FP8 pages."""
        return self.is_fp8_dense_layer(layer_idx) and int(layer_idx) in self._shared_draft_layer_ids

    @property
    def uses_hybrid_nvfp4_kv_cache(self) -> bool:
        return self.dtype == DataType.NVFP4

    def _build_pool_mapping_tensors(self):
        """Publish scale pointers only for physical pools that own them.

        The base NVFP4 implementation assumes every layer has a scale buffer.
        Hybrid M3 deliberately omits scales from dense/Eagle layers, while the
        target metadata still needs the nested NVFP4 pointer envelope for the
        sparse pools.  Dense consumers use the direct P32 views below and the
        shared Eagle view publishes itself as an ordinary FP8 manager.
        """
        if self.dtype != DataType.NVFP4:
            return super()._build_pool_mapping_tensors()

        pointer_rows = []
        mapping_rows = []
        if self.enable_swa_scratch_reuse:
            for local_layer_idx in range(self.num_local_layers):
                global_layer_idx = int(self.pp_layers[local_layer_idx])
                data_addr = self.impl.get_mem_pool_base_address(
                    local_layer_idx, Role.KEY, PageIndexMode.PER_LAYER
                )
                scale_addr = (
                    self.impl.get_mem_pool_base_address(
                        local_layer_idx, Role.KEY_BLOCK_SCALE, PageIndexMode.PER_LAYER
                    )
                    if self.is_nvfp4_layer(global_layer_idx)
                    else 0
                )
                pointer_rows.append([[data_addr, scale_addr], [0, 0]])
                mapping_rows.append([local_layer_idx, 0])
        else:
            for pool_id in range(self.num_pools):
                local_layer_idx = int(self.impl.layer_grouping[pool_id][0])
                group_layers = [int(layer) for layer in self.impl.layer_grouping[pool_id]]
                group_is_uniform_nvfp4 = all(
                    self.is_nvfp4_layer(int(self.pp_layers[layer])) for layer in group_layers
                )
                data_addr = self.impl.get_mem_pool_base_address(
                    local_layer_idx, Role.KEY, PageIndexMode.SHARED
                )
                scale_addr = (
                    self.impl.get_mem_pool_base_address(
                        local_layer_idx, Role.KEY_BLOCK_SCALE, PageIndexMode.SHARED
                    )
                    if group_is_uniform_nvfp4
                    else 0
                )
                pointer_rows.append([[data_addr, scale_addr], [0, 0]])

            for local_layer_idx in range(self.num_local_layers):
                pool_id = int(self.impl.get_layer_group_id(local_layer_idx))
                data_base = pointer_rows[pool_id][0][0]
                offset = self._kv_pool_mapping_offset(local_layer_idx, pool_id, data_base)
                if (
                    self.is_nvfp4_layer(int(self.pp_layers[local_layer_idx]))
                    and pointer_rows[pool_id][0][1] != 0
                ):
                    scale_base = pointer_rows[pool_id][0][1]
                    scale_addr = self.impl.get_mem_pool_base_address(
                        local_layer_idx, Role.KEY_BLOCK_SCALE, PageIndexMode.SHARED
                    )
                    scale_layers = sorted(
                        self.impl.layer_grouping[pool_id],
                        key=lambda lid: self.impl.get_mem_pool_base_address(
                            lid, Role.KEY_BLOCK_SCALE, PageIndexMode.SHARED
                        ),
                    )
                    scale_offset = scale_layers.index(local_layer_idx)
                    assert scale_addr >= scale_base and scale_offset == offset, (
                        "M3 hybrid NVFP4 data/scale layer ordering differs: "
                        f"layer={local_layer_idx} data_offset={offset} "
                        f"scale_offset={scale_offset}."
                    )
                mapping_rows.append([pool_id, offset])

        return (
            torch.tensor(pointer_rows, dtype=torch.int64, pin_memory=prefer_pinned()),
            torch.tensor(mapping_rows, dtype=torch.int32, pin_memory=prefer_pinned()),
        )

    def get_draft_subpage_view(self) -> Optional["MiniMaxM3DraftSubpageView"]:
        """Sub-page view over the shared drafter pool, or None.

        Only meaningful on a target manager carrying appended one-model
        draft layers; built lazily so the manager's page tables exist. A
        method rather than a property so ``getattr`` fetches it without
        executing it (see ``resolve_draft_kv_cache_manager``).

        Retires with the P128 Eagle kernel fixes; see
        ``draft_manager_tokens_per_block``.
        """
        if self.is_draft or not self._shared_draft_layer_ids:
            return None
        if self._draft_subpage_view_obj is None:
            subpage_tokens = (
                int(os.environ.get("TRTLLM_M3_DRAFT_KV_TOKENS_PER_BLOCK", 0) or 0)
                or self.draft_manager_tokens_per_block
            )
            self._draft_subpage_view_obj = MiniMaxM3DraftSubpageView(
                self,
                self._shared_draft_layer_ids,
                subpage_tokens,
            )
            logger.info(
                f"[unified-kv] draft sub-page view active "
                f"(tokens_per_block={self._draft_subpage_view_obj.tokens_per_block}, "
                f"flat_page_bound={self._draft_subpage_view_obj.blocks_in_primary_pool})"
            )
        return self._draft_subpage_view_obj

    def add_dummy_requests(self, *args, **kwargs):
        """Drop the draft sub-page view before delegating.

        The base method mirrors dummy KV caches into a *separate* draft
        manager. With shared draft layers a dummy request's blocks already
        span the drafter's pool (pools allocate in lockstep per logical
        block), and the view owns no block lifecycle.

        Retires with the P128 Eagle kernel fixes; see
        ``draft_manager_tokens_per_block``.
        """
        if isinstance(kwargs.get("draft_kv_cache_manager"), MiniMaxM3DraftSubpageView):
            kwargs["draft_kv_cache_manager"] = None
        return super().add_dummy_requests(*args, **kwargs)

    def _extra_buffers_per_layer(self, *, tokens_per_block):
        """Register a per-sparse-layer ``Role.INDEX_KEY`` :class:`BufferConfig`.

        ``size`` is bytes per **block**: ``1 * sparse_index_dim *
        elem_bytes * tokens_per_block``. Keyed by **local** layer id —
        the base ``_build_base_config`` iterates local ids, so keying
        by global ids would silently skip registration on non-trivial
        PP ranks.
        """
        torch_dtype = self._torch_dtype_for_index_cache()
        elem_bytes = torch.tensor([], dtype=torch_dtype).element_size()
        bytes_per_token = 1 * self.sparse_index_dim * elem_bytes
        size_per_block = bytes_per_token * tokens_per_block
        return {
            self.layer_offsets[layer_id]: [BufferConfig(role=Role.INDEX_KEY, size=size_per_block)]
            for layer_id in self.sparse_layer_ids
            if layer_id in self.layer_offsets
        }

    def get_disagg_role_mapper_kinds(self) -> dict[DataRole, MapperKind]:
        """Declare the backend's main K/V layout and replicated index-K."""
        main_kv_mapper = MapperKind.INDEXED if self._main_kv_layout == "HND" else MapperKind.NHD
        return {
            Role.ALL: main_kv_mapper,
            Role.INDEX_KEY: MapperKind.REPLICATED,
        }

    def _compute_num_total_slots(self) -> int:
        """Total token slots across all blocks in the main K pool.

        Sizes the plain-tensor index-V cache only; the V2-managed
        index-K cache pulls its slot count from ``Role.INDEX_KEY``'s
        page upper bound.
        """
        if not self.layer_offsets:
            return int(self.max_batch_size * self.max_seq_len)
        any_layer_offset = next(iter(self.layer_offsets.values()))
        page_upper = self.impl.get_page_index_upper_bound(any_layer_offset, Role.KEY)
        kv_factor = 1 if self.kv_cache_type == CacheTypeCpp.SELFKONLY else 2
        return int((page_upper // kv_factor) * self.tokens_per_block)

    def _torch_dtype_for_index_cache(self) -> torch.dtype:
        """Return the independently configured index-cache storage dtype."""
        if self.indexer_kv_dtype == "fp8":
            return torch.float8_e4m3fn
        if self.dtype == DataType.HALF:
            return torch.float16
        if self.dtype == DataType.FLOAT:
            return torch.float32
        return torch.bfloat16

    def get_index_k_buffer(
        self, layer_idx: int, kv_layout: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """Return the V2-managed paged index-K view for ``layer_idx``.

        NHD shape is ``[num_pages, tokens_per_block, 1, sparse_index_dim]``;
        HND shape is ``[num_pages, 1, tokens_per_block, sparse_index_dim]``.
        When omitted, ``kv_layout`` follows the selected sparse backend.
        """
        if kv_layout is None:
            kv_layout = self._main_kv_layout
        return super().get_index_k_buffer(
            layer_idx,
            num_heads=1,
            head_dim=self.sparse_index_dim,
            dtype=self._torch_dtype_for_index_cache(),
            kv_layout=kv_layout,
        )

    def get_index_v_buffer(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Plain-tensor index-V cache for non-disabled sparse layers."""
        return self._index_v_buffers.get(layer_idx)

    def has_index_value(self, layer_idx: int) -> bool:
        return layer_idx in self._index_v_buffers

    def _kv_slot_geometry(
        self, layer_idx: int, kv_layout: Optional[str]
    ) -> Tuple[int, torch.dtype, int, int, List[int]]:
        """Resolve one layer's position in the coalesced K/V pool.

        Returns ``(addr_key, torch_dtype, num_slots, scale, page_shape)``,
        where ``scale`` is the number of equal-sized sub-pages a slot packs
        and ``page_shape`` is one sub-page's shape in ``kv_layout``. This
        layer's K is sub-page 0 and its V sub-page 1, counting from
        ``addr_key``. When ``kv_layout`` is None it follows the selected
        sparse backend.
        """
        if kv_layout is None:
            kv_layout = self._main_kv_layout
        if kv_layout not in ("NHD", "HND"):
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        if self.is_fp8_subpaged_layer(layer_idx):
            raise RuntimeError(
                f"hybrid FP8 layer {layer_idx} uses four physical P32 pages; "
                "use get_fp8_dense_buffers/get_dense_kv_subpage_pool instead"
            )
        if self.kv_cache_type == CacheTypeCpp.SELFKONLY:
            raise NotImplementedError(
                "MiniMaxM3KVCacheManagerV2 does not support the SELFKONLY cache type"
            )

        layer_offset = self.layer_offsets[layer_idx]
        addr_key = self.impl.get_mem_pool_base_address(layer_offset, Role.KEY)
        addr_value = self.impl.get_mem_pool_base_address(layer_offset, Role.VALUE)
        page_stride_key = self.impl.get_page_stride(layer_offset, Role.KEY)
        page_stride_value = self.impl.get_page_stride(layer_offset, Role.VALUE)
        # V2 always lays V immediately after K within the per-layer
        # contribution to a slot. The slice ``[:, :2]`` depends on this.
        assert addr_key + page_stride_key == addr_value, (
            f"MiniMaxM3 requires addr_K + page_stride "
            f"== addr_V (V immediately after K in slot); got "
            f"addr_K={addr_key} page_stride_K={page_stride_key} "
            f"addr_V={addr_value} for layer {layer_idx}."
        )
        assert page_stride_key == page_stride_value, (
            f"MiniMaxM3 requires equal K and V page "
            f"strides; got K={page_stride_key} V="
            f"{page_stride_value}."
        )

        converter = self.impl.get_page_index_converter(layer_offset, Role.KEY)
        scale = int(converter.scale)
        layer_offset_pages = int(converter.layer_offset)
        page_upper_K = self.impl.get_page_index_upper_bound(layer_offset, Role.KEY)
        num_slots_total = page_upper_K + layer_offset_pages
        assert num_slots_total % scale == 0, (
            f"V2 storage inconsistency: page_upper_K + "
            f"layer_offset_pages = {num_slots_total} is not "
            f"divisible by scale = {scale}."
        )
        num_slots = num_slots_total // scale

        element_per_container = 1
        dtype = self.dtype
        if self.is_nvfp4_layer(layer_idx):
            element_per_container = 2
            torch_dtype = torch.int8
        else:
            dtype = DataType.FP8 if self.is_fp8_dense_layer(layer_idx) else dtype
            torch_dtype = binding_to_torch_dtype(dtype)

        layer_head_dim = self.head_dim_per_layer[layer_offset]
        num_kv_heads = self.num_kv_heads_per_layer[layer_offset]
        containers = layer_head_dim // element_per_container

        if kv_layout == "NHD":
            page_shape = [self.tokens_per_block, num_kv_heads, containers]
        else:
            page_shape = [num_kv_heads, self.tokens_per_block, containers]
        return addr_key, torch_dtype, num_slots, scale, page_shape

    def get_buffers(
        self, layer_idx: int, kv_layout: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """Return a paged K+V view with strides spanning the coalesced pool.

        The base :meth:`KVCacheManagerV2.get_buffers` produces a
        ``[num_pages, kv_factor, ...]`` view with contiguous strides
        that assume the slot holds exactly one layer's K+V. In M3's
        pool the slot packs K+V for *all* layers of the group
        (``scale >= 2 * num_layers_in_group``), so the base view's
        dim-0 stride does not reach the next slot's K for this layer.
        (When INDEX_KEY's per-block size coincides with K/V's, it is
        coalesced into the same pool and contributes to ``scale`` too.)

        The override builds a ``[num_slots, scale, ...]`` view rooted
        at K's base, then slices ``[:, :2]`` to extract K+V. The slice
        preserves the dim-0 stride (``scale * page_stride``), so
        ``view[s, 0/1, ...]`` lands on this layer's K/V at slot ``s``.
        When omitted, ``kv_layout`` follows the selected sparse backend.
        """
        if self.is_fp8_subpaged_layer(layer_idx):
            if kv_layout not in (None, "HND"):
                raise ValueError(
                    "hybrid FP8 dense/Eagle buffers have a physical P32 HND layout; "
                    f"requested {kv_layout}"
                )
            k, _v, slot_stride, pages_per_role = self._fp8_dense_data_buffers(layer_idx)
            num_slots, _pages, num_heads, page_size, head_dim = k.shape
            full = convert_to_torch_tensor(
                TensorWrapper(
                    k.data_ptr(),
                    k.dtype,
                    [num_slots, slot_stride, num_heads, page_size, head_dim],
                )
            )
            return full[:, : 2 * pages_per_role].unflatten(1, (2, pages_per_role))

        addr_key, torch_dtype, num_slots, scale, page_shape = self._kv_slot_geometry(
            layer_idx, kv_layout
        )
        full_slot_shape = [num_slots, scale, *page_shape]
        full_view = convert_to_torch_tensor(TensorWrapper(addr_key, torch_dtype, full_slot_shape))
        return full_view[:, :2]

    def _kv_scale_slot_geometry(
        self, layer_idx: int, kv_layout: Optional[str]
    ) -> Tuple[int, torch.dtype, int, int, List[int]]:
        """Resolve one layer's NVFP4 K/V scale pages in their coalesced pool.

        The scale pool mirrors the packed-data pool's K-then-V ordering, but
        its page unit is one E4M3 byte per 16 logical cache elements.  Keeping
        this geometry separate from :meth:`_kv_slot_geometry` is important:
        the two pools have different byte strides even when their page-index
        converters have the same scale.
        """
        if not self.is_nvfp4_layer(layer_idx):
            raise RuntimeError("NVFP4 block-scale buffers require an NVFP4 KV cache")
        if kv_layout is None:
            kv_layout = self._main_kv_layout
        if kv_layout not in ("NHD", "HND"):
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        if self.kv_cache_type == CacheTypeCpp.SELFKONLY:
            raise NotImplementedError(
                "MiniMaxM3KVCacheManagerV2 does not support the SELFKONLY cache type"
            )

        layer_offset = self.layer_offsets[layer_idx]
        k_role = Role.KEY_BLOCK_SCALE
        v_role = Role.VALUE_BLOCK_SCALE
        addr_key = self.impl.get_mem_pool_base_address(layer_offset, k_role)
        addr_value = self.impl.get_mem_pool_base_address(layer_offset, v_role)
        page_stride_key = self.impl.get_page_stride(layer_offset, k_role)
        page_stride_value = self.impl.get_page_stride(layer_offset, v_role)
        assert addr_key + page_stride_key == addr_value, (
            "MiniMaxM3 NVFP4 scale pool requires K scale immediately followed "
            f"by V scale; got K={addr_key} stride={page_stride_key} V={addr_value}."
        )
        assert page_stride_key == page_stride_value, (
            "MiniMaxM3 NVFP4 K/V scale page strides differ: "
            f"K={page_stride_key} V={page_stride_value}."
        )

        converter = self.impl.get_page_index_converter(layer_offset, k_role)
        scale = int(converter.scale)
        layer_offset_pages = int(converter.layer_offset)
        page_upper = self.impl.get_page_index_upper_bound(layer_offset, k_role)
        num_slots_total = page_upper + layer_offset_pages
        assert num_slots_total % scale == 0, (
            "NVFP4 scale storage inconsistency: page_upper + layer_offset = "
            f"{num_slots_total} is not divisible by scale={scale}."
        )
        num_slots = num_slots_total // scale

        head_dim = self.head_dim_per_layer[layer_offset]
        assert head_dim % 16 == 0, f"NVFP4 head_dim must be divisible by 16, got {head_dim}"
        num_kv_heads = self.num_kv_heads_per_layer[layer_offset]
        scale_cols = head_dim // 16
        if kv_layout == "NHD":
            page_shape = [self.tokens_per_block, num_kv_heads, scale_cols]
        else:
            page_shape = [num_kv_heads, self.tokens_per_block, scale_cols]
        return addr_key, torch.uint8, num_slots, scale, page_shape

    def get_block_scale_buffers(
        self, layer_idx: int, kv_layout: Optional[str] = None
    ) -> torch.Tensor:
        """Return paged NVFP4 K+V E4M3 scale-byte views for ``layer_idx``."""
        addr_key, torch_dtype, num_slots, scale, page_shape = self._kv_scale_slot_geometry(
            layer_idx, kv_layout
        )
        full_slot_shape = [num_slots, scale, *page_shape]
        full_view = convert_to_torch_tensor(TensorWrapper(addr_key, torch_dtype, full_slot_shape))
        return full_view[:, :2]

    def _fp8_dense_data_buffers(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return hybrid FP8 K/V views backed by physical P32 pages.

        Each logical P128 role page is laid out as four consecutive P32 HND
        pages.  ``slot_stride`` is measured in those physical pages and already
        includes the V2 converter's expansion factor.
        """
        if not self.is_fp8_subpaged_layer(layer_idx):
            raise RuntimeError(f"layer {layer_idx} is not a physical-P32 hybrid FP8 layer")
        local_layer_idx = self.layer_offsets[layer_idx]
        physical_page = self.nvfp4_dense_tokens_per_block
        pages_per_role = self.tokens_per_block // physical_page
        addr_key = self.impl.get_mem_pool_base_address(local_layer_idx, Role.KEY)
        addr_value = self.impl.get_mem_pool_base_address(local_layer_idx, Role.VALUE)
        page_stride_key = self.impl.get_page_stride(local_layer_idx, Role.KEY)
        page_stride_value = self.impl.get_page_stride(local_layer_idx, Role.VALUE)
        assert page_stride_key == page_stride_value
        assert addr_key + pages_per_role * page_stride_key == addr_value, (
            "M3 hybrid FP8 storage requires V immediately after K's physical "
            f"P{physical_page} pages; layer={layer_idx} K={addr_key} "
            f"stride={page_stride_key} V={addr_value}."
        )

        converter = self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
        assert int(converter.expansion) == pages_per_role, (
            f"layer {layer_idx} expected V2 expansion {pages_per_role}, got "
            f"{int(converter.expansion)}"
        )
        slot_stride = int(converter.scale) * pages_per_role
        layer_offset_pages = int(converter.layer_offset) * pages_per_role
        page_upper = self.impl.get_page_index_upper_bound(local_layer_idx, Role.KEY)
        total_pages = int(page_upper) + layer_offset_pages
        assert total_pages % slot_stride == 0
        num_slots = total_pages // slot_stride

        num_heads = self.num_kv_heads_per_layer[local_layer_idx]
        head_dim = self.head_dim_per_layer[local_layer_idx]
        full = convert_to_torch_tensor(
            TensorWrapper(
                addr_key,
                torch.float8_e4m3fn,
                [num_slots, slot_stride, num_heads, physical_page, head_dim],
            )
        )
        return (
            full[:, :pages_per_role],
            full[:, pages_per_role : 2 * pages_per_role],
            slot_stride,
            pages_per_role,
        )

    def get_fp8_dense_buffers(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return physical-P32 hybrid FP8 views as ``K, V``."""
        k, v, _slot_stride, _pages_per_role = self._fp8_dense_data_buffers(layer_idx)
        return k, v

    def get_dense_kv_subpage_pool(self, layer_idx: int) -> Tuple[torch.Tensor, int, int]:
        """Flat dense-attention pool, slot stride, and pages per K/V role."""
        if not self.is_fp8_subpaged_layer(layer_idx):
            pool, slot_stride = self.get_kv_subpage_pool(layer_idx, "HND")
            return pool, slot_stride, 1
        k, _v, slot_stride, pages_per_role = self._fp8_dense_data_buffers(layer_idx)
        num_slots, _pages, num_heads, page_size, head_dim = k.shape
        num_pages = (num_slots - 1) * slot_stride + 2 * pages_per_role
        addr = k.data_ptr()
        pool = convert_to_torch_tensor(
            TensorWrapper(addr, k.dtype, [num_pages, num_heads, page_size, head_dim])
        )
        return pool, slot_stride, pages_per_role

    def get_kv_subpage_pool(
        self, layer_idx: int, kv_layout: str = "HND"
    ) -> Tuple[torch.Tensor, int]:
        """Return ``(flat_pool, subpages_per_slot)`` for flat-block consumers.

        trtllm-gen addresses K and V pages independently, through a
        ``[batch, 2, max_blocks]`` block table into one flat
        ``[num_subpages, *page_shape]`` pool. That is expressible here even
        though the per-layer stride is not uniform: a slot packs ``scale``
        equal-sized sub-pages, of which this layer owns two adjacent ones, so
        rooting the flat pool at this layer's K puts slot ``s``'s K at
        ``s * scale`` and its V at ``s * scale + 1``.

        The view stops two sub-pages past the last slot's K rather than
        spanning ``num_slots * scale``, which would run off the pool by
        whatever this layer's K offset is inside a slot.
        """
        if self.is_fp8_subpaged_layer(layer_idx):
            if kv_layout != "HND":
                raise ValueError("hybrid FP8 dense/Eagle sub-pages are HND only")
            pool, slot_stride, _pages_per_role = self.get_dense_kv_subpage_pool(layer_idx)
            return pool, slot_stride
        addr_key, torch_dtype, num_slots, scale, page_shape = self._kv_slot_geometry(
            layer_idx, kv_layout
        )
        num_subpages = (num_slots - 1) * scale + 2
        flat = convert_to_torch_tensor(
            TensorWrapper(addr_key, torch_dtype, [num_subpages, *page_shape])
        )
        return flat, scale

    def get_kv_scale_subpage_pool(
        self, layer_idx: int, kv_layout: str = "HND"
    ) -> Tuple[torch.Tensor, int]:
        """Return the flat NVFP4 scale pool paired with ``get_kv_subpage_pool``.

        The returned factor must match the packed-data factor so one K/V block
        table addresses both pools. Token-size subdivision, when needed by the
        Eagle draft view, is expressed by that view's expanded block table.
        """
        addr_key, torch_dtype, num_slots, scale, page_shape = self._kv_scale_slot_geometry(
            layer_idx, kv_layout
        )
        _data_addr, _data_dtype, data_slots, data_scale, _data_shape = self._kv_slot_geometry(
            layer_idx, kv_layout
        )
        assert (num_slots, scale) == (data_slots, data_scale), (
            "MiniMaxM3 NVFP4 data and scale pools require identical page-index "
            f"geometry; data={(data_slots, data_scale)} scale={(num_slots, scale)}."
        )
        num_subpages = (num_slots - 1) * scale + 2
        flat = convert_to_torch_tensor(
            TensorWrapper(addr_key, torch_dtype, [num_subpages, *page_shape])
        )
        return flat, scale

    def _kv_pool_mapping_offset(self, layer_id, layer_group_id, key_base_addr) -> int:
        """Pool-mapping offset from the layer's physical position in its pool.

        The base formula ``exact_div(addr_offset, key_bytes * kv_factor *
        tokens_per_block)`` assumes each layer contributes exactly K+V to
        its pool slot. When index-K coalesces into the K/V pool the layer
        stride is non-uniform (sparse layers add an INDEX_KEY sub-page),
        so no uniform-stride offset exists. The M3 forward path uses
        :meth:`get_buffers` / :meth:`get_index_k_buffer` rather than this
        mapping, so the offset just needs to be a consistent per-layer
        position. Rank the group's layers by their K base address instead
        of by ``layer_grouping`` iteration order: the ordering of
        ``layer_grouping`` is not a V2 API contract, while the address
        rank always reflects the physical slot layout (and keeps the
        NVFP4 ``block_scale_offset == offset`` cross-check in the base
        pool-mapping loop meaningful).
        """
        layers_by_addr = sorted(
            self.impl.layer_grouping[int(layer_group_id)],
            key=lambda lid: self.impl.get_mem_pool_base_address(
                lid, Role.KEY, PageIndexMode.SHARED
            ),
        )
        return layers_by_addr.index(int(layer_id))

    def _get_batch_cache_indices_by_pool_id(
        self,
        request_ids,
        *,
        pool_id: int = 0,
        is_kv_aggregate: bool = True,
        num_blocks_per_seq: Optional[Sequence[int]] = None,
        index_scale: Optional[int] = None,
    ):
        """Return page indices; padded entries remain ``BAD_PAGE_INDEX`` (-1).

        The base method converts slot ids to V1-style block ids via
        ``base_idx * index_scales[pool_id] // kv_factor``, which is
        only correct when each layer contributes exactly K+V. M3's slot
        packs K+V for all layers of the group, so the scale breaks the
        V1 conversion and produces out-of-bounds block ids during V2
        warmup.

        Bypass the conversion: the M3 forward path indexes paged
        views (built by :meth:`get_buffers` /
        :meth:`get_index_k_buffer`) directly by slot id.
        ``BAD_PAGE_INDEX`` slots remain ``-1`` here because disaggregation's
        :class:`KVRegionExtractorV1` filters ``region_ids >= 0``.
        :meth:`get_block_ids_per_seq` maps them to zero for the attention
        metadata's padded tensor.

        Args:
            request_ids: Request IDs whose page-index rows are returned.
            pool_id: V2 pool whose page indices are requested.
            is_kv_aggregate: Kept for compatibility with the base virtual method.
            num_blocks_per_seq: Optional per-request truncation limits. When
                omitted, preserve the full padded width required by MiniMax
                CUDA-graph metadata initialization.
            index_scale: Kept for compatibility with the base virtual method;
                M3 bypasses the V1 block-id conversion entirely, so any
                caller-supplied scale is ignored alongside ``index_scales``.
        """
        res = []
        for req_idx, req_id in enumerate(request_ids):
            kv_cache = self.kv_cache_map[req_id]
            base_page_indices = kv_cache.get_base_page_indices(pool_id)
            if num_blocks_per_seq is not None:
                num_blocks = min(kv_cache.num_blocks, num_blocks_per_seq[req_idx])
                base_page_indices = base_page_indices[:num_blocks]
            res.append(list(base_page_indices))
        return res

    def get_block_ids_per_seq(self, request_ids):
        """Return per-request slot ids matching the per-layer paged view's dim-0.

        Drops the base's final ``i // num_local_layers`` step (paired
        with the base ``index_scales`` multiplication that's also
        bypassed here). Pads with ``0`` to preserve shape.

        The rows are written through a numpy view of a single zero-filled,
        pinned result, so the attention metadata builders ship it to the device
        in one asynchronous copy.
        """
        block_ids_per_seq = self.get_batch_cache_indices(request_ids)
        batch = len(block_ids_per_seq)
        max_blocks = max((len(block_ids) for block_ids in block_ids_per_seq), default=0)
        padded_tensor = torch.zeros(
            (batch, max_blocks), dtype=torch.int32, pin_memory=prefer_pinned()
        )
        rows = padded_tensor.numpy()
        for row, block_ids in zip(rows, block_ids_per_seq):
            row[: len(block_ids)] = block_ids
        # BAD_PAGE_INDEX marks padding, which this tensor reports as 0.
        rows[rows == BAD_PAGE_INDEX] = 0
        return padded_tensor


class MiniMaxM3DraftSubpageView:
    """Present the shared manager's draft-layer pool at a smaller kernel page size.

    With unified KV cache the drafter's KV lives inside the shared manager's
    128-token logical blocks, but the Eagle3 kernels are only healthy at
    32-token pages on this architecture. This view flows wherever a separate
    draft manager would (``get_draft_kv_cache_manager`` and the
    attention-metadata draft swap) and re-expresses the geometry only:

    * the single pool pointer is re-rooted at the drafter's K address, and
      the draft layer's row in the pool mapping points at that pool;
    * the block table expands each logical slot ``s`` into sub-pages —
      K at ``s*scale*subdiv + j`` (``j < subdiv``), V at ``+subdiv``
      (``scale`` = the drafter's pages per mega-slot, from
      ``_kv_slot_geometry``; same layout trick as the dense-layer
      trtllm-gen adapter).

    The attention op reads ``tokens_per_block`` and the pool pointers from
    the metadata's manager, so no attention-backend changes are needed. The
    view owns no blocks: lifecycle stays entirely with the shared manager.

    Retires with the P128 Eagle kernel fixes; see the retirement plan on
    ``MiniMaxM3KVCacheManagerV2.draft_manager_tokens_per_block``.
    """

    def __init__(self, manager, draft_layer_ids: Sequence[int], subpage_tokens: int):
        self._manager = manager
        self.tokens_per_block = int(subpage_tokens)
        layer_id = draft_layer_ids[0]
        is_hybrid_fp8 = bool(
            getattr(manager, "is_fp8_subpaged_layer", lambda _layer_idx: False)(layer_id)
        )
        if is_hybrid_fp8:
            assert self.tokens_per_block == manager.nvfp4_dense_tokens_per_block, (
                "hybrid FP8 Eagle draft attention must use the physical dense-cache "
                f"page size P{manager.nvfp4_dense_tokens_per_block}, got "
                f"P{self.tokens_per_block}"
            )
        assert manager.tokens_per_block % self.tokens_per_block == 0, (
            f"subpage size {subpage_tokens} must divide manager "
            f"tokens_per_block {manager.tokens_per_block}"
        )
        self._subdiv = manager.tokens_per_block // self.tokens_per_block
        # The hybrid layout puts dense/Eagle FP8 pages in a separate physical
        # pool from sparse NVFP4 data/scales.  Root this single-pool view at the
        # draft layer's K address, while sourcing raw logical slot IDs from the
        # draft layer's actual V2 pool.
        local = manager.layer_offsets[layer_id]
        self._source_pool_id = int(manager.kv_cache_pool_mapping[int(local), 0])
        if is_hybrid_fp8:
            k, _v, slot_stride, pages_per_role = manager._fp8_dense_data_buffers(layer_id)
            assert pages_per_role == self._subdiv
            addr_key = k.data_ptr()
            self._num_slots = int(k.shape[0])
            self._slot_units = slot_stride
            self.dtype = DataType.FP8
        else:
            addr_key, _dt, num_slots, scale, _shape = manager._kv_slot_geometry(layer_id, None)
            self._num_slots = int(num_slots)
            self._slot_units = scale * self._subdiv
        self.num_pools = 1
        self.num_attention_op_pools = 1
        self.max_blocks_per_seq = manager.max_blocks_per_seq * self._subdiv
        # Single-pool pointer rooted at the drafter's K; the op derives the
        # page stride from tokens_per_block, so unit indices below address
        # 32-token drafter pages directly.
        self.kv_cache_pool_pointers = torch.tensor(
            [[addr_key, 0]], dtype=torch.int64, pin_memory=prefer_pinned()
        )
        mapping = manager.kv_cache_pool_mapping.clone()
        mapping[int(local)] = torch.tensor([0, 0], dtype=mapping.dtype)
        self.kv_cache_pool_mapping = mapping
        # Placeholder host mirror: the dense TRTLLM path plans from device
        # offsets; nothing reads the host table during the draft window.
        self.host_kv_cache_block_offsets = torch.zeros(
            (1, 1, 2, 1), dtype=torch.int32, pin_memory=prefer_pinned()
        )
        self._slots_host: Optional[np.ndarray] = None
        self._arange: Optional[torch.Tensor] = None

    @property
    def blocks_in_primary_pool(self) -> int:
        """Flattened sub-page index bound relative to the draft K pointer.

        ``FlashInferTrtllmGenFmha`` uses this value to size the flat paged-KV
        tensor passed to FlashInfer. The wrapped V2 manager reports its bound
        in 128-token page units relative to a different pool base, so
        delegating that property through ``__getattr__`` under-describes this
        32-token, draft-K-rooted view. The final slot contributes only this
        layer's K and V pages; inter-layer padding after V is not addressable
        from the view and need not be included.
        """
        return (self._num_slots - 1) * self._slot_units + 2 * self._subdiv

    def __getattr__(self, name):
        manager = self.__dict__.get("_manager")
        if manager is None:
            raise AttributeError(name)
        return getattr(manager, name)

    @property
    def host_kv_cache_pool_pointers(self):
        return self.kv_cache_pool_pointers

    def free_resources(self, request) -> None:
        """No-op: block lifecycle belongs to the shared manager."""

    def _host_block_table(
        self,
        slot_rows: Sequence[Sequence[int]],
        num_seqs: int,
        max_slots: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Expand this batch's slot ids into a freshly allocated pinned table.

        The buffer is allocated per call on purpose: it is the source of an
        asynchronous H2D copy, whose source is read at copy execution time
        rather than enqueue time. A persistent buffer refilled in place would
        let the next iteration's refill clobber a still-pending copy — the
        drafter would then index another batch's blocks (nvbug 6293536, whose
        rationale the V1 manager spells out on
        ``KVCacheManager._stage_block_offsets_for_copy``). The caching host
        allocator keeps this block alive until the copy retires. The numpy
        scratch below stays persistent: it is only ever read synchronously.
        """
        sub = self._subdiv
        if (
            self._slots_host is None
            or self._slots_host.shape[0] < num_seqs
            or self._slots_host.shape[1] != max_slots
        ):
            self._slots_host = np.zeros((num_seqs, max_slots), dtype=np.int32)
            self._arange = torch.arange(sub, dtype=dtype)
        slots_np = self._slots_host[:num_seqs]
        slots_np.fill(0)
        # Ragged fill is the only per-row work (numpy parses each row's list at
        # C speed); the arithmetic below is one fused expansion across the
        # batch. Pad/BAD_PAGE_INDEX entries clamp to slot 0 (safe pages:
        # kernels never read past kv_lens).
        for i, row in enumerate(slot_rows[:num_seqs]):
            n = min(len(row), max_slots)
            if n > 0:
                slots_np[i, :n] = row[:n]
        np.clip(slots_np, 0, None, out=slots_np)
        slots = torch.from_numpy(slots_np).to(dtype)
        host = torch.empty(
            (num_seqs, 2, max_slots * sub),
            dtype=dtype,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        out = host.view(num_seqs, 2, max_slots, sub)
        torch.add(slots.unsqueeze(-1) * self._slot_units, self._arange, out=out[:, 0])
        torch.add(out[:, 0], sub, out=out[:, 1])
        return host

    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        max_blocks: Optional[int] = None,
    ) -> None:
        # Raw logical slot ids from the draft layer's physical pool.
        slot_rows = self._manager._get_batch_cache_indices_by_pool_id(
            request_ids, pool_id=self._source_pool_id
        )
        host = self._host_block_table(
            slot_rows, num_seqs, dst_tensor.shape[-1] // self._subdiv, dst_tensor.dtype
        )
        dst_tensor[0, :num_seqs].copy_(host, non_blocking=True)


def get_minimax_m3_kv_cache_manager_cls():
    """Backward-compatible accessor; prefer importing the class directly."""
    return MiniMaxM3KVCacheManagerV2


__all__ = [
    "MiniMaxM3KVCacheManagerV2",
    "MiniMaxM3SparseIndexCache",
    "get_minimax_m3_kv_cache_manager_cls",
]

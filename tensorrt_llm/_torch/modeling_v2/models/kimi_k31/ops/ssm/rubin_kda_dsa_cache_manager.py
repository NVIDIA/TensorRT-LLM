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
"""Kimi KDA hybrid cache manager with a DSA indexer K-cache on its MLA layers (Kimi K3.1 Rubin).

A runtime helper of the family's Rubin targets: their runtime hooks (``rubin_runtime_hooks``)
declare it in place of the KDA hybrid manager the engine resolves for DSA on a Kimi linear model, and
the engine builds it with what ``extra_init_kwargs`` returns. A target binds its construction
arguments (``bind_kv_cache_args``): the GPU pools zeroed at construction and the KV cache, index-K and
KDA state dtypes it was written for. The static estimate and ``extra_init_kwargs`` refuse a config
without them.

The manager derives from the family's KDA hybrid cache manager (``kda_cache_manager.py``, over
the shared ``tensorrt_llm/_torch/modules/kimi_kda/cache_manager.py``), so the shared KDA cache
handling applies to K3.1 as it does to K3, with K3.1's replay state: the fused verify's replay
caches live in each request's recurrent-state slot, where the Rubin verify kernels read them.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Tuple

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.speculative.interface import get_force_num_accepted_tokens_float
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    GPU_LEVEL,
    BufferConfig,
    DataRole,
    PageIndexMode,
)

from ...runtime import K31_RUNTIME_HOOKS
from ..attention.rubin_dsa_cache import (
    get_indexer_k_cache_bytes_per_token,
    get_indexer_k_cache_size_per_token,
)
from .kda_cache_manager import K31HybridCacheManager
from .unified_draft_cache import MlaDFlashContextPolicy

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


#: Where a target's construction arguments of the manager live in ``model_config.extra_attrs``.
KV_CACHE_ARGS_KEY = "rubin_kv_cache_args"

#: The KV cache dtypes a target may name, as the engine passes them (``dtype``).
_KV_DTYPES = {"fp8": DataType.FP8}


def bind_kv_cache_args(
    model_config,
    *,
    zero_gpu_pools: bool,
    kv_dtype: str,
    indexer_k_dtype: str,
    state_dtype: torch.dtype,
) -> None:
    """Record a target's construction arguments of the manager on its model config.

    The engine builds the manager itself, after the model, with the arguments ``extra_init_kwargs``
    returns for the model's config; a target calls this when it is constructed.

    Args:
        model_config: The target's model config (the engine's, not a copy).
        zero_gpu_pools: Zero every GPU pool byte once, before the guard page and the first request.
        kv_dtype: The KV cache dtype the target was written for: ``"fp8"``; construction fails
            for another.
        indexer_k_dtype: The index-K dtype the target was written for: ``"fp8"``.
        state_dtype: The KDA recurrent state dtype the target was written for.
    """
    if kv_dtype not in _KV_DTYPES:
        raise ValueError(f"kv_dtype must be one of {sorted(_KV_DTYPES)}, got {kv_dtype!r}")
    if indexer_k_dtype != "fp8":
        raise ValueError(f"indexer_k_dtype must be 'fp8', got {indexer_k_dtype!r}")
    if not isinstance(state_dtype, torch.dtype):
        raise TypeError(f"state_dtype must be a torch dtype, got {state_dtype!r}")
    model_config.extra_attrs[KV_CACHE_ARGS_KEY] = {
        "zero_gpu_pools": bool(zero_gpu_pools),
        "kv_dtype": kv_dtype,
        "indexer_k_dtype": indexer_k_dtype,
        "state_dtype": state_dtype,
    }


def _bound_kv_cache_args(model_config) -> dict:
    """A copy of the arguments the target bound on this config.

    Raises:
        ValueError: The target bound none. The manager would then run with no pool zero and no
            dtype checks.
    """
    bound = (getattr(model_config, "extra_attrs", None) or {}).get(KV_CACHE_ARGS_KEY)
    if bound is None:
        raise ValueError(
            "The Kimi K3.1 Rubin cache manager needs its target's arguments: the target calls "
            "bind_kv_cache_args(model_config, ...) when it is constructed."
        )
    return dict(bound)


def _indexer_k_cache_layer_mask(
    sparse_attention_config, pretrained_config, num_layers: int
) -> List[bool]:
    """Per layer, whether it is a full indexer layer (keeps index keys): the sparse attention
    config's per-layer parameters, as the engine's DSA cache manager derives its indexer mask."""
    return [
        bool(
            getattr(
                sparse_attention_config.to_sparse_params(
                    pretrained_config=pretrained_config, layer_idx=layer_idx
                ),
                "is_full_indexer_layer",
                True,
            )
        )
        for layer_idx in range(num_layers)
    ]


def placeholder_draft_requests(pending: Dict[int, bool], scheduled_batch) -> FrozenSet[int]:
    """The requests of ``scheduled_batch`` whose generation row keeps its own accepted-token count under forced
    acceptance: each request's first generation step after a context step on this server, or after it arrived
    through a disaggregated handoff. As for the engine's parallel-draft modes, that step's drafts count as
    placeholders. ``pending`` holds, per request this server has scheduled, whether its next generation step is
    such a step.

    A context step (every chunk, and a recompute resume past the prompt) marks its request; its next generation
    row is exempt and clears the mark, so with chunked prefill only the step after the last chunk is exempt. A
    request first seen in a generation step arrived through a handoff (``context_phase_params``) and is exempt
    once, whatever drafts it brought. Requests receiving their KV (``is_disagg_generation_init_state``) are
    scheduled with the context requests but run no context step and are not marked; dummy requests have no handoff
    and stay forced.

    Read on the host from the scheduled requests, before the step's metadata is prepared. Every rank that serves
    a request schedules it alike (attention TP and Helix CP ranks the same batch, an attention-DP rank its own
    requests), so the ranks select the same rows.
    """
    for request in scheduled_batch.context_requests:
        if not request.is_disagg_generation_init_state:
            pending[request.py_request_id] = True
    placeholders = set()
    for request in scheduled_batch.generation_requests:
        request_id = request.py_request_id
        first = pending.get(request_id)
        if first is None:
            first = request.context_phase_params is not None
        if first:
            placeholders.add(request_id)
        pending[request_id] = False
    return frozenset(placeholders)


class RubinKdaDsaHybridCacheManager(K31HybridCacheManager):
    """Kimi K3 KDA/MLA cache plus a DSA indexer K-cache (Kimi K3.1).

    Kimi K3.1 adds a DeepSeek sparse-attention indexer to its MLA layers. The
    indexer keys live in one extra ``Role.INDEX_KEY`` buffer per indexer layer,
    so they share allocation, reuse and release with the latent KV pages. The
    DSA accessors mirror :class:`DSACacheManagerV2`, except that local layer
    indices also cover the KDA layers, which own neither a K page nor an
    indexer buffer. A unified standalone drafter's layers share admission and
    transfer with the target's, but own neither a DSA K page nor an indexer
    buffer.
    """

    # Read by ``is_dsa_cache_manager``: the DSA backend and metadata may use
    # this manager's indexer K-cache and primary-pool page mapping.
    provides_dsa_indexer_k_cache = True

    @classmethod
    def extra_init_kwargs(cls, model_config) -> dict:
        """Constructor arguments the engine adds for this manager (``_create_kv_cache_manager``):
        the DSA config, the text config and what the target bound (``bind_kv_cache_args``,
        required).

        Raises:
            ValueError: the target bound no arguments.
        """
        pretrained_config = model_config.pretrained_config
        return {
            "sparse_attention_config": model_config.sparse_attention_config,
            "pretrained_config": getattr(pretrained_config, "text_config", None)
            or pretrained_config,
            **_bound_kv_cache_args(model_config),
        }

    @override
    def __init__(
        self,
        *args,
        sparse_attention_config: "SparseAttentionConfig",
        pretrained_config=None,
        zero_gpu_pools: bool = False,
        kv_dtype: Optional[str] = None,
        indexer_k_dtype: Optional[str] = None,
        state_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> None:
        """Build the hybrid manager with an indexer buffer per DSA layer.

        Args:
            sparse_attention_config: DSA sparse-attention config of the model.
            pretrained_config: Text config; resolves indexer sharing per layer.
            zero_gpu_pools: Zero every GPU pool byte once, before the guard page.
            kv_dtype: The KV cache dtype the target was written for (``bind_kv_cache_args``);
                None checks nothing.
            indexer_k_dtype: The index-K dtype the target was written for; None checks nothing.
            state_dtype: The KDA state dtype the target was written for; None checks nothing.
            **kwargs: ``K31HybridCacheManager`` arguments; ``layer_mask``
                marks the MLA layers.
        """
        if sparse_attention_config is None:
            raise ValueError("sparse_attention_config is required for the Kimi DSA cache")
        sparse_params = sparse_attention_config.to_sparse_params(
            pretrained_config=pretrained_config
        )
        if not isinstance(sparse_params, DSAParams):
            raise ValueError("The Kimi DSA cache requires DSA sparse parameters")
        layer_mask = kwargs.get("layer_mask")
        if layer_mask is None:
            raise ValueError("The Kimi DSA cache needs layer_mask to place the indexer buffers")
        if kv_dtype is not None and kwargs.get("dtype") != _KV_DTYPES.get(kv_dtype):
            raise ValueError(
                f"The target's Kimi DSA cache stores {kv_dtype} latent KV; the engine asks for "
                f"{kwargs.get('dtype')} (kv_cache_config.dtype)."
            )
        if indexer_k_dtype is not None and sparse_params.indexer_k_dtype != indexer_k_dtype:
            raise ValueError(
                f"The target's Kimi DSA cache stores {indexer_k_dtype} index keys; the DSA config "
                f"asks for {sparse_params.indexer_k_dtype}."
            )
        # Read by the guard-page reservation during the base construction.
        self._zero_gpu_pools = bool(zero_gpu_pools)

        # Read by _extra_buffers_per_layer during the base construction.
        self.quant_block_size = 128
        self.index_head_dim = sparse_params.index_head_dim
        self.indexer_k_dtype = sparse_params.indexer_k_dtype
        self.use_fp8_ds_mla = False
        self.mla_kv_cache_residual_dim = 0
        self._primary_pools: dict[int, torch.Tensor] = {}
        # get_primary_pool by local layer, so a per-layer call skips the binding lookups.
        self._primary_pool_by_layer: dict[int, torch.Tensor] = {}
        full_indexer_mask = _indexer_k_cache_layer_mask(
            sparse_attention_config, pretrained_config, len(layer_mask)
        )
        self._indexer_layer_ids = frozenset(
            layer_idx
            for layer_idx, (is_attention, has_indexer) in enumerate(
                zip(layer_mask, full_indexer_mask)
            )
            if is_attention and has_indexer
        )

        super().__init__(*args, **kwargs)
        # Under forced acceptance (TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS): per scheduled request, whether
        # its next generation step is its first after a context step or a handoff, and the requests whose
        # generation row in the step being prepared is such a step (placeholder_draft_requests). None / empty
        # otherwise.
        self.first_step_pending: Optional[Dict[int, bool]] = (
            {} if get_force_num_accepted_tokens_float() != 0.0 else None
        )
        self.placeholder_requests: FrozenSet[int] = frozenset()

        if state_dtype is not None and self.ssm_state_dtype != state_dtype:
            raise ValueError(
                f"The target's Kimi DSA cache keeps a {state_dtype} KDA state; the engine asks for "
                f"{self.ssm_state_dtype} (kv_cache_config.mamba_ssm_cache_dtype)."
            )
        # A unified standalone drafter (DSpark, DFlash2) registers its layers as model layers; they
        # own neither a coalesced MLA K page nor an indexer buffer, so the DSA views skip them.
        self._attention_local_layers = [
            local_layer_idx
            for local_layer_idx in range(self.num_local_layers)
            if not self._is_standalone_draft_layer(local_layer_idx)
            and not self._is_local_mamba_layer(local_layer_idx)
        ]
        if not self._attention_local_layers:
            raise ValueError("DSA requires at least one local MLA layer")
        self._primary_pool_page_index_params = {}
        for local_layer_idx in self._attention_local_layers:
            converter = self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
            self._primary_pool_page_index_params[local_layer_idx] = (
                int(converter.scale),
                int(converter.layer_offset),
            )
        indexer_local_layers = [
            local_layer_idx
            for local_layer_idx, has_indexer in enumerate(self.indexer_k_cache_local_layer_mask)
            if has_indexer
        ]
        indexer_converters = [
            self.impl.get_page_index_converter(local_layer_idx, Role.INDEX_KEY)
            for local_layer_idx in indexer_local_layers
        ]
        if any(int(converter.expansion) != 1 for converter in indexer_converters):
            raise ValueError("DSA indexer K-cache does not support page expansion")
        # Each cache view starts at its SHARED base address, so the converter's
        # layer offset is already represented by the base pointer.
        self.indexer_k_cache_page_scale = (
            int(indexer_converters[0].scale) if indexer_converters else 1
        )
        if any(
            int(converter.scale) != self.indexer_k_cache_page_scale
            for converter in indexer_converters[1:]
        ):
            raise ValueError("DSA requires a uniform shared INDEX_KEY page mapping across layers")
        self._attention_pool_id = self.layer_to_pool_mapping_dict[self._attention_local_layers[0]]
        if any(
            self.layer_to_pool_mapping_dict[layer] != self._attention_pool_id
            for layer in self._attention_local_layers
        ):
            raise ValueError("Kimi DSA MLA layers must share one request-slot lifecycle.")
        self._num_attention_slots = self._count_attention_slots()
        self.indexer_k_cache_pool_per_layer = {
            local_layer_idx: self._get_indexer_k_cache_pool_data(local_layer_idx)
            for local_layer_idx in indexer_local_layers
        }
        self._log_fp8_latent_cache_pools()
        self._log_fp32_kda_state()
        # Startup logs show which cache manager serves the model.
        logger.info_once(
            f"Kimi K3.1 cache manager: {type(self).__name__} (DSA indexer K-cache on "
            f"{len(indexer_local_layers)} of {len(self._attention_local_layers)} local MLA layers, "
            f"indexer_k_dtype={self.indexer_k_dtype}, zero_gpu_pools={self._zero_gpu_pools})",
            key="rubin_kda_dsa_cache_manager",
        )

    @override
    def _reserve_guard_page(self) -> None:
        """Zero the GPU pools first where the target asks for it, then reserve the guard page.

        The engine calls this once, at the end of its construction. A zero after the guard page
        would overwrite the guard page's fill (``TRTLLM_KV_GUARD_PAGE``).
        """
        if self._zero_gpu_pools:
            self._zero_gpu_pool_memory()
        super()._reserve_guard_page()

    def _gpu_pool_regions(self) -> List[Tuple[Tuple[int, int], int, int]]:
        """``((pool group, pool), base address, mapped bytes)`` of every GPU pool."""
        return [
            (
                (int(group.pool_group_index), int(pool.pool_index)),
                int(pool.base_address),
                int(group.num_slots) * int(pool.slot_bytes),
            )
            for group in self.impl.pool_group_descs
            for pool in group.pools
        ]

    def _zero_gpu_pool_memory(self) -> None:
        """Zero every byte of the GPU pools once, before the first request.

        The pools are mapped from physical memory that is never cleared, and pages pass from
        one request to the next without being scrubbed. Paged decode kernels load whole tiles
        and mask the keys past a request's length: the CuTe DSL MLA decode, for one, reads to
        the end of its 128-key tile, i.e. the rest of the request's last page and the pages
        the next page-table entries name. A masked score is replaced, but its value row is
        still multiplied by a zero probability, so a NaN or Inf bit pattern there (FP8 E4M3
        0x7F / 0xFF) makes the output NaN. After this every pool byte is zero or a value a
        cache writer stored, and the writers store finite values. Memory a pool maps later is
        not zeroed: routing refuses the pool rebalancing that would map it, and
        ``check_gpu_pool_growth`` tells whether a pool grew.
        """
        # Wait for this stream only: a device-wide synchronize would also wait for unrelated
        # work in flight on other streams. The events time the zero itself.
        start = time.perf_counter()
        stream = torch.cuda.current_stream()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        regions = self._gpu_pool_regions()
        begin.record(stream)
        for _, base, size in regions:
            if size:
                convert_to_torch_tensor(TensorWrapper(base, torch.uint8, [size])).zero_()
        end.record(stream)
        end.synchronize()
        self._zeroed_gpu_pool_bytes = {key: size for key, _, size in regions}
        self._gpu_pool_growth_warned = False
        total = sum(size for _, _, size in regions)
        logger.info(
            f"KVCacheManagerV2: zeroed {total / (1 << 30):.2f} GiB of GPU KV pools in "
            f"{begin.elapsed_time(end):.1f} ms on the GPU "
            f"({(time.perf_counter() - start) * 1e3:.1f} ms on the host, incl. earlier work on the stream)"
        )

    def check_gpu_pool_growth(self) -> bool:
        """Whether a GPU pool has mapped memory since ``_zero_gpu_pool_memory`` (pool
        rebalancing); warns once, since that memory was not zeroed. Without the zero there is
        nothing to compare: ``False``. The engine does not call it: the Rubin targets run without
        the KV pool rebalance (routing refuses it)."""
        if not hasattr(self, "_zeroed_gpu_pool_bytes"):
            return False
        grown = [
            key
            for key, _, size in self._gpu_pool_regions()
            if size > self._zeroed_gpu_pool_bytes.get(key, 0)
        ]
        if grown and not self._gpu_pool_growth_warned:
            self._gpu_pool_growth_warned = True
            logger.warning(
                f"KVCacheManagerV2: GPU pools (pool group, pool) {grown} mapped memory after the "
                "construction-time zero; it is not zeroed, so a decode that loads past a request's "
                "length can read NaN bit patterns from it"
            )
        return bool(grown)

    @override
    def _verify_max_num_spec(self) -> int:
        """The Rubin verify kernels' draft capacity. Imported here, as the kernel module needs the
        CuTe DSL; the sm_103 verify kernel module would load the Kimi K3 package, whose KDA kernels
        patch the CuTe DSL compile pipeline of the whole process."""
        from ...kernels.ssm.rubin_kda_mtp_decode import MAX_NUM_SPEC

        return MAX_NUM_SPEC

    @override
    def prepare_resources(self, scheduled_batch) -> None:
        """The shared preparation; under forced acceptance, also the step's placeholder-draft requests."""
        super().prepare_resources(scheduled_batch)
        if self.first_step_pending is not None:
            self.placeholder_requests = placeholder_draft_requests(
                self.first_step_pending, scheduled_batch
            )

    @override
    def free_resources(self, request, pin_on_release: bool = False) -> None:
        """The shared release; the request leaves ``first_step_pending``."""
        if self.first_step_pending is not None:
            self.first_step_pending.pop(request.py_request_id, None)
        super().free_resources(request, pin_on_release)

    @property
    def indexer_k_cache_local_layer_mask(self) -> list[bool]:
        """Indexer ownership of each local layer; KDA layers own none."""
        return [layer_idx in self._indexer_layer_ids for layer_idx in self.pp_layers]

    @override
    def _extra_buffers_per_layer(self, *, tokens_per_block: int) -> dict[int, list[BufferConfig]]:
        """One ``Role.INDEX_KEY`` buffer per local indexer layer."""
        return {
            self.layer_offsets[layer_idx]: [
                BufferConfig(
                    role=Role.INDEX_KEY,
                    size=self._indexer_bytes_per_token() * tokens_per_block,
                )
            ]
            for layer_idx in sorted(self._indexer_layer_ids)
            if layer_idx in self.layer_offsets
        }

    def _indexer_bytes_per_token(self) -> int:
        return get_indexer_k_cache_bytes_per_token(self.index_head_dim, self.quant_block_size)

    def get_kv_cache_dtype(self, layer_idx: int | None = None) -> DataType:
        """Return the storage dtype for a global MLA layer."""
        if layer_idx is None:
            return self.dtype
        return self._get_layer_dtype_by_local_id(self.layer_offsets[layer_idx])

    @override
    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: DataRole) -> int:
        index_bytes = (
            self._indexer_bytes_per_token()
            if local_layer_idx < len(self.pp_layers)
            and self.pp_layers[local_layer_idx] in self._indexer_layer_ids
            else 0
        )
        if data_role == Role.INDEX_KEY:
            return index_bytes
        cache_bytes = super().get_layer_bytes_per_token(local_layer_idx, data_role)
        return cache_bytes + index_bytes if data_role == Role.ALL else cache_bytes

    @classmethod
    @override
    def get_cache_size_per_token(
        cls,
        model_config,
        mapping: Mapping,
        *,
        max_batch_size: int,
        kv_cache_config: KvCacheConfig,
        tokens_per_block: int = 32,
        max_seq_len: Optional[int] = None,
        **kwargs,
    ) -> tuple[int, int]:
        # Refuses a config its target bound no arguments on, as construction does.
        _bound_kv_cache_args(model_config)
        slope, fixed_cost = super().get_cache_size_per_token(
            model_config,
            mapping,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        extra_bytes_per_token = get_indexer_k_cache_size_per_token(model_config, mapping)
        state_config = kv_cache_config.mamba_state_config
        seq_limit = max_seq_len if max_seq_len is not None else float("inf")
        has_snapshots = kv_cache_config.enable_block_reuse and (
            0 < state_config.periodic_snapshot_interval <= seq_limit
            or any(
                offset <= seq_limit
                for offset in state_config.additional_snapshot_offsets_from_start
            )
            or any(
                offset < seq_limit for offset in state_config.additional_snapshot_offsets_from_end
            )
        )
        # V2 reserves one partial attention page per resident lineage when
        # snapshots are reachable; include the indexer section of those pages.
        if has_snapshots:
            fixed_cost += (
                max_batch_size * mapping.pp_size * tokens_per_block * extra_bytes_per_token
            )
        return slope + extra_bytes_per_token, fixed_cost

    def get_primary_pool_page_index_params(self, local_layer_idx: int) -> tuple[int, int]:
        """Return the V2 page scale and layer offset of an MLA layer's K page."""
        return self._primary_pool_page_index_params[local_layer_idx]

    def _count_attention_slots(self) -> int:
        """Physical slot count of the MLA K pool (not its V2 page bound), from the first MLA layer."""
        local_layer_idx = self._attention_local_layers[0]
        converter = self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
        page_upper = self.impl.get_page_index_upper_bound(local_layer_idx, Role.KEY)
        expansion = int(converter.expansion)
        assert page_upper % expansion == 0
        num_pages_with_offset = page_upper // expansion + int(converter.layer_offset)
        scale = int(converter.scale)
        assert num_pages_with_offset % scale == 0
        return num_pages_with_offset // scale

    def _get_indexer_k_cache_pool_data(self, local_layer_idx: int) -> torch.Tensor:
        """Return a contiguous shared-page view for one indexer K-cache."""
        address = self.impl.get_mem_pool_base_address(
            local_layer_idx, Role.INDEX_KEY, PageIndexMode.SHARED
        )
        page_upper = self.impl.get_page_index_upper_bound(local_layer_idx, Role.INDEX_KEY)
        flat_page_size = self.tokens_per_block * self._indexer_bytes_per_token()
        return convert_to_torch_tensor(
            TensorWrapper(address, torch.uint8, [page_upper, flat_page_size])
        )

    def get_indexer_k_cache_buffers(self, layer_idx: int) -> torch.Tensor:
        """Return the page-indexed indexer K-cache view for a global layer."""
        local_layer_idx = self.layer_offsets[layer_idx]
        pool = self.indexer_k_cache_pool_per_layer.get(local_layer_idx)
        assert pool is not None, f"Layer {layer_idx} owns no DSA indexer K-cache."
        return pool.view(pool.shape[0], self.tokens_per_block, 1, self._indexer_bytes_per_token())

    def get_pool_block_indices(
        self,
        num_seqs: int,
        *,
        request_ids: Optional[list[int]] = None,
        num_contexts: int = 0,
        beam_width: int = 1,
    ) -> torch.Tensor:
        """Read the MLA pool's V2 stable slots in current-batch order."""
        if request_ids is None:
            raise ValueError("RubinKdaDsaHybridCacheManager requires request_ids to map slots")
        if len(request_ids) != num_seqs:
            raise ValueError(f"Expected {num_seqs} request IDs, got {len(request_ids)}")
        copy_idx = self.index_mapper.get_copy_index(list(request_ids), num_contexts, beam_width)
        copy_idx = copy_idx.to(device="cpu", dtype=torch.long)
        num_cols = self._batch_block_columns(request_ids)
        block_indices = self.host_kv_cache_block_offsets[
            self._attention_pool_id, copy_idx, 0, :num_cols
        ]
        return block_indices.clamp(min=0, max=self._num_attention_slots - 1).to(torch.int32)

    def _batch_block_columns(self, request_ids: list[int]) -> int:
        """Block-table columns that can hold a page of this batch.

        A request's pages cover its KV capacity (every token any kernel reads, draft slots
        included), so columns past the largest capacity hold no page of the batch. Reading only
        those keeps the per-step host gather proportional to the batch's context instead of
        ``max_seq_len``.
        """
        width = self.host_kv_cache_block_offsets.shape[-1]
        capacity = 0
        for request_id in request_ids:
            kv_cache = self.kv_cache_map.get(request_id)
            if kv_cache is None:
                return width
            capacity = max(capacity, int(kv_cache.capacity))
        return min(width, max(1, -(-capacity // self.tokens_per_block)))

    def get_primary_pool(self, local_layer_idx: int) -> torch.Tensor:
        """Return this layer's physical K pool as ``[slots, pages, 1, elements]``.

        Slot IDs are shared with the indexer; page scales and offsets describe the layers
        coalesced in the physical pool.
        """
        pool = self._primary_pool_by_layer.get(local_layer_idx)
        if pool is not None:
            return pool
        if self.kv_factor != 1:
            raise ValueError("DSA requires a SELFKONLY KV cache")
        converter = self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
        if int(converter.expansion) != 1:
            raise ValueError("DSA MLA K-cache does not support page expansion")
        base_address = self.impl.get_mem_pool_base_address(
            local_layer_idx, Role.KEY, PageIndexMode.PER_LAYER
        )
        if base_address in self._primary_pools:
            pool = self._primary_pools[base_address]
            self._primary_pool_by_layer[local_layer_idx] = pool
            return pool
        head_dim = self.head_dim_per_layer[local_layer_idx]
        num_heads = self.num_kv_heads_per_layer[local_layer_idx]
        dtype = self._get_layer_dtype_by_local_id(local_layer_idx)
        elements_per_layer = self.tokens_per_block * num_heads * head_dim
        shape = [self._num_attention_slots, int(converter.scale), 1, elements_per_layer]
        pool = convert_to_torch_tensor(TensorWrapper(base_address, dtype, shape))
        self._primary_pools[base_address] = pool
        self._primary_pool_by_layer[local_layer_idx] = pool
        return pool

    def get_unique_primary_pool(self) -> torch.Tensor:
        """Return the primary pool when all MLA layers share one storage layout."""
        pools = [self.get_primary_pool(layer) for layer in self._attention_local_layers]
        if any(pool.data_ptr() != pools[0].data_ptr() for pool in pools[1:]):
            raise ValueError("Mixed Kimi DSA storage requires a layer-specific primary pool.")
        return pools[0]

    def _get_page_index_params(self, local_layer_idx: int, role: DataRole) -> tuple[int, int, int]:
        """Return the (scale, expansion, layer offset) of a buffer's page-index converter."""
        converter = self.impl.get_page_index_converter(local_layer_idx, role)
        return int(converter.scale), int(converter.expansion), int(converter.layer_offset)

    def _log_fp8_latent_cache_pools(self) -> None:
        """With an FP8 latent cache, log the GPU slots and slot bytes of every pool group, on one line.

        Only the cache that serves logs, not the one the engine first builds for the cache size
        estimate (``is_estimating_kv_cache``).
        """
        if self.dtype != DataType.FP8 or self.is_estimating_kv_cache:
            return
        try:
            stats = self._get_storage_statistics(GPU_LEVEL)
        except (AttributeError, RuntimeError) as error:
            # A log line only: a cache that cannot report its pools still serves.
            logger.warning(f"Kimi K3.1 KV cache pools unavailable: {error}")
            return
        entries = []
        for pool_group_id, stat in enumerate(stats):
            slot_sizes = [int(size) for size in self._stats_slot_sizes(stat)]
            num_slots = int(stat.total)
            entries.append(
                f"pool_group_id={pool_group_id}, num_slots={num_slots}, slot_size={slot_sizes}, "
                f"bytes={num_slots * sum(slot_sizes)}"
            )
        logger.info(f"Kimi K3.1 KV cache pools (FP8 latent KV): {'; '.join(entries)}")

    def _log_fp32_kda_state(self) -> None:
        """With an FP32 KDA recurrent state (``kv_cache_config.mamba_ssm_cache_dtype: float32``), log
        the local KDA layers, their state shape and the state bytes of a slot, on one line.

        Only rank 0's ``LLM Args`` line names the configured dtype (``auto`` by default, not what it
        resolves to); this line names the resolved dtype on every rank. With an FP8 latent cache the
        pool line shows the same bytes as a slot size of the KDA pool group. A BF16 state logs
        nothing; only the cache that serves logs, as for the pool line.
        """
        if (
            self.ssm_state_dtype != torch.float32
            or self.is_estimating_kv_cache
            or self.local_num_mamba_layers == 0
        ):
            return
        logger.info(
            f"Kimi K3.1 KDA recurrent state: float32, {self.local_num_mamba_layers} local KDA "
            f"layers of {list(self.ssm_state_shape)}, "
            f"{self.local_num_mamba_layers * self.ssm_bytes} B of state per slot"
        )

    @override
    def shutdown(self) -> None:
        self.indexer_k_cache_pool_per_layer = {}
        self._primary_pools.clear()
        self._primary_pool_by_layer.clear()
        super().shutdown()


def rubin_runtime_hooks(attention_backend_cls: type, draft_model_cls: type):
    """Bind the cache and sampling policy to the drafter this target constructs."""
    return replace(
        K31_RUNTIME_HOOKS,
        kv_cache_manager_cls=RubinKdaDsaHybridCacheManager,
        attention_backend_cls=attention_backend_cls,
        dflash_policy=replace(
            K31_RUNTIME_HOOKS.dflash_policy,
            uses_paged_context=MlaDFlashContextPolicy.for_drafter(draft_model_cls),
        ),
    )

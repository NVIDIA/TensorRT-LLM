# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unified standalone draft cache of the Kimi K3.1 family's KV cache managers.

A standalone DFlash2 drafter keeps one context-KV entry per target token. ``UnifiedDraftCacheMixin`` stores those
entries as extra layers of the target's KV cache manager V2, so they share each request's allocation, reuse match and
disaggregated transfer. The draft layers keep the drafter's sliding window, which no target layer has, so they never
share a lifecycle or a pool with target layers; the manager checks this when it is built. The drafter's history (how
many entries are valid, and the next position) is tracked here because it advances independently of the target's
history: rejected target tokens do not revoke draft entries.

The mixin goes before the engine manager in the family manager's bases (``K31HybridCacheManager``). It overrides the
engine methods that lay out the draft layers (``_lay_out_regions``), describe them (``_explicit_layer_spec``,
``_get_pool_roles``, ``_is_local_mamba_layer``, ``get_layer_owner``, ``draft_layer_ids``, ``get_buffers``), size every
allocation and the pools for them (``_kv_alloc_extra_tokens``, ``_generation_kv_capacity_headroom``, the quota
conversions, ``_get_num_free_blocks_for_layers``) and keep the drafter's history over a request's life
(``_resume_context_cache``, ``_create_kv_cache``, ``update_resources``, ``free_resources``, ``_shutdown_caches``).
Before construction, the classmethod ``unified_draft_cache_kwargs`` declares the ``draft_*`` constructor arguments of
a deployment, or none, and refuses the drafter settings the cache cannot serve. The DFlash worker reads a unified
draft cache through the public names below (``has_standalone_draft_cache``, ``get_draft_buffers``,
``get_draft_block_table``, ``get_draft_history``, ...). Disaggregated serving moves the draft layers as it moves the
target's (they share the target's page and helix layout) and checks both peers keep them alike through
``get_draft_kv_cache_view``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Collection, Iterable, List, Optional, Sequence

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    Role,
    _estimate_cache_size_components,
)
from tensorrt_llm._utils import (
    TensorWrapper,
    convert_to_torch_tensor,
    get_size_in_bytes,
    prefer_pinned,
    torch_dtype_to_binding,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    BAD_PAGE_INDEX,
    AttentionLayerConfig,
    BatchDesc,
    BufferConfig,
    DataRole,
    KVCacheDesc,
    LayerId,
    PageIndexMode,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        KVCacheManagerConfig as KVCacheManagerConfigPy,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache


@dataclass(frozen=True)
class StandaloneDraftHistory:
    """Committed draft context length and the next absolute position.

    ``valid_length`` tokens ending just before ``position`` hold drafter
    context KV. Both are independent of the target's history and of any
    scratch capacity reserved for the next draft forward.
    """

    valid_length: int
    position: int

    def __post_init__(self) -> None:
        if type(self.valid_length) is not int or type(self.position) is not int:
            raise ValueError("Standalone draft history requires integer length and position")
        if self.valid_length < 0 or self.position < self.valid_length:
            raise ValueError("Invalid standalone draft history length or position")


@dataclass(frozen=True)
class DraftHistoryUpdate:
    """One forward's draft history, which its manager publishes once that forward is done."""

    manager: UnifiedDraftCacheMixin
    request_ids: tuple[int, ...]
    cache_instances: tuple[object, ...]
    values_host: torch.Tensor
    # Recorded on the current stream at capture, after the forward that wrote the draft pages and the
    # device-to-host copy of a CUDA ``values``.
    copied: Optional[torch.cuda.Event] = None

    @classmethod
    def capture(
        cls,
        manager: UnifiedDraftCacheMixin,
        request_ids: tuple[int, ...],
        values: torch.Tensor,
    ) -> DraftHistoryUpdate:
        """Queue a length/position readback on the current stream.

        Args:
            manager: The unified cache manager owning the draft layers.
            request_ids: Requests of the rows of ``values``.
            values: ``[len(request_ids), 2]`` integer tensor of
                ``(valid_length, position)`` per request.
        """
        if tuple(values.shape) != (len(request_ids), 2):
            raise ValueError("Draft history update requires one length/position pair per request")
        values_host = torch.empty(
            values.shape,
            dtype=values.dtype,
            device="cpu",
            pin_memory=values.is_cuda and prefer_pinned(),
        )
        values_host.copy_(values, non_blocking=True)
        copied = None
        if torch.cuda.is_available():
            copied = torch.cuda.Event()
            copied.record()
        return cls(
            manager,
            tuple(request_ids),
            tuple(manager.kv_cache_map[request_id] for request_id in request_ids),
            values_host,
            copied,
        )

    def is_ready(self) -> bool:
        """Whether the forward and the readback are done, so that publishing does not wait."""
        return self.copied is None or self.copied.query()

    def publish(self) -> None:
        """Publish the readback without reviving released or replaced requests."""
        if self.copied is not None:
            self.copied.synchronize()
        for request_id, cache, (length, position) in zip(
            self.request_ids, self.cache_instances, self.values_host.tolist()
        ):
            current = self.manager.kv_cache_map.get(request_id)
            if current is cache and current.is_active:
                self.manager.set_draft_history(request_id, length, position)


def draft_layer_sizes(
    *,
    draft_num_layers: int = 0,
    draft_num_kv_heads: int = 0,
    draft_head_dim: int = 0,
    draft_dtype: DataType = DataType.BF16,
    draft_kv_factor: int = 2,
    **kwargs,
) -> tuple[int, ...]:
    """Bytes per token of each standalone draft layer of a unified cache.

    Extra keyword arguments are accepted so callers can pass the whole draft
    keyword set of the manager constructor.
    """
    del kwargs
    if draft_num_layers < 0:
        raise ValueError("Draft layer count must be nonnegative")
    if not draft_num_layers:
        return ()
    if min(draft_num_kv_heads, draft_head_dim) <= 0:
        raise ValueError("Standalone draft cache dimensions must be positive")
    if draft_dtype not in (DataType.HALF, DataType.BF16, DataType.FP8):
        raise ValueError("Standalone draft KV supports FP16, BF16 and FP8 storage")
    if draft_kv_factor not in (1, 2):
        raise ValueError("Standalone draft KV stores one or two planes")
    size = get_size_in_bytes(draft_num_kv_heads * draft_head_dim * draft_kv_factor, draft_dtype)
    return (size,) * draft_num_layers


def estimate_draft_cache_size_components(
    layer_sizes: Sequence[int],
    tokens_per_block: int,
    *,
    generation_capacity_headroom: int,
    scratch_tokens: int = 0,
    window_size: int | None = None,
) -> tuple[int, int, int]:
    """Draft storage cost in the units of the engine's ``_estimate_cache_size_components``.

    Args:
        layer_sizes: Bytes per token of each draft layer.
        tokens_per_block: Physical tokens per page.
        generation_capacity_headroom: Capacity a generation request holds
            beyond its history.
        scratch_tokens: Draft scratch capacity per request.
        window_size: Draft history window, or None for full history.
    """
    retention_window = window_size + 1 if window_size is not None else None
    context, generation, per_request = _estimate_cache_size_components(
        layer_sizes,
        [retention_window] * len(layer_sizes),
        tokens_per_block,
        scratch=False,
        generation_capacity_headroom=generation_capacity_headroom,
    )
    return context, generation, per_request + scratch_tokens * sum(layer_sizes)


def draft_pool_group_conflicts(
    pool_group_layers: Sequence[Collection[int]], draft_layers: Collection[int]
) -> list[int]:
    """Indices of the pool groups that hold standalone draft layers beside other layers.

    Args:
        pool_group_layers: The local layer ids of each pool group, in pool group order.
        draft_layers: The local layer ids of the standalone draft layers.
    """
    draft = frozenset(draft_layers)
    return [
        index
        for index, layers in enumerate(pool_group_layers)
        if not draft.isdisjoint(layers) and not draft.issuperset(layers)
    ]


def _is_standalone_dflash(spec_config: Any) -> bool:
    return spec_config is not None and spec_config.spec_dec_mode.is_dflash()


def _carries_dflash_context(spec_config: Any, is_disagg: bool) -> bool:
    """Whether a DFlash drafter's context travels in a unified draft cache: always under disaggregated serving.

    A DFlash drafter rebuilds its context from the target's hidden states
    during prefill and keeps it in its own buffers, so without helix or an
    aggregated opt-in the draft layers only carry that context across the
    cache transceiver.
    """
    return _is_standalone_dflash(spec_config) and is_disagg


def _is_mla_dflash(spec_config: Any, draft_config: Any) -> bool:
    """Whether this is DFlash with a standalone MLA drafter."""
    if (
        spec_config is None
        or not spec_config.spec_dec_mode.is_dflash()
        or spec_config._use_shared_kv_cache
        or draft_config is None
    ):
        return False
    config = draft_config.pretrained_config
    return (getattr(config, "dflash_config", None) or {}).get("attention_mode") == "mla"


def _helix_sharded(mapping: Any) -> bool:
    return mapping.has_cp_helix() or mapping.helix_repurposed_cp_size > 1


@dataclass(frozen=True)
class MlaDFlashContextPolicy:
    """Whether this drafter stores its context in KV pages for a deployment."""

    when_aggregated: bool = False

    @classmethod
    def for_drafter(cls, draft_model: Any) -> MlaDFlashContextPolicy:
        return cls(
            when_aggregated=getattr(draft_model, "unified_draft_cache_when_aggregated", False)
            is True
        )

    def __call__(self, mapping: Any, is_disagg: bool) -> bool:
        return self.when_aggregated or is_disagg or _helix_sharded(mapping)


def _needs_unified_mla_dflash(
    spec_config: Any, draft_config: Any, mapping: Any, is_disagg: bool, draft_model: Any
) -> bool:
    """Whether an MLA DFlash drafter's context KV goes to the unified draft
    cache: under helix (its pages are sharded over CP), when the drafter
    keeps it there in every deployment, or when it carries it across
    disaggregated serving."""
    if not _is_mla_dflash(spec_config, draft_config):
        return False
    uses_pages = MlaDFlashContextPolicy.for_drafter(draft_model)
    return uses_pages(mapping, is_disagg)


def _draft_cache_kwargs(draft: Any, draft_config: Any, spec_config: Any) -> dict:
    """Draft storage properties of a unified draft cache, as manager kwargs.

    A GQA drafter shards its KV heads like its attention; an MLA drafter
    (``_kv_factor == 1``) stores one latent per token and layer, sharded
    by page over its own helix mapping (``_helix_mapping``) when it has one.
    The scratch is the next block of ``max_draft_len + 1`` tokens.
    """
    config = draft_config.pretrained_config
    draft_mapping = getattr(draft, "_helix_mapping", None)
    if draft_mapping is None:
        # Shard heads like the drafter's attention, which runs on the
        # repurposed (CP folded into TP) mapping under helix.
        draft_mapping = draft_config.mapping
        if draft_mapping.has_cp_helix():
            draft_mapping = draft_mapping.repurpose_helix_cp_to_tp()
    mla = getattr(draft, "_kv_factor", 2) == 1
    if mla:
        num_layers = draft._num_attn_layers
        num_kv_heads = attention_tp_size = 1
        head_dim = draft._head_dim
    else:
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // num_heads
        attention_tp_size = 1 if draft_mapping.enable_attention_dp else draft_mapping.tp_size
    dtype = draft.fc.weight.dtype if hasattr(draft, "fc") else torch.bfloat16
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Standalone draft KV supports FP16 and BF16 storage")
    if mla and getattr(draft, "context_kv_fp8", False):
        # The drafter keeps FP8 E4M3 latents.
        dtype = torch.float8_e4m3fn
    return dict(
        draft_num_layers=num_layers,
        draft_kv_factor=1 if mla else 2,
        draft_cp_size=(draft_mapping.cp_size if draft_mapping.has_cp_helix() else 1),
        draft_num_kv_heads=(num_kv_heads + attention_tp_size - 1) // attention_tp_size,
        draft_head_dim=head_dim,
        draft_dtype=torch_dtype_to_binding(dtype),
        draft_total_num_kv_heads=num_kv_heads,
        draft_scratch_tokens=spec_config.max_draft_len + 1,
    )


class UnifiedDraftCacheMixin:
    """A family manager's unified standalone draft cache (see the module docstring).

    Lists before the engine's KV cache manager V2 in a manager's bases. Constructed without ``draft_*`` arguments,
    or with ``draft_num_layers=0``, the manager has no draft layers and every seam keeps the engine's behaviour.
    """

    # Declared on the class for instances built without running __init__.
    _draft_layer_sizes: tuple[int, ...] = ()
    _standalone_draft_layer_ids: tuple[int, ...] = ()
    _standalone_draft_reserve: int = 0
    _draft_registered_as_model_layers: bool = False
    _unified_draft_layer_specs: dict = {}
    _pending_draft_history: Sequence[DraftHistoryUpdate] = ()
    _engine_generation_kv_capacity_headroom: int = 0
    draft_window_size: Optional[int] = None

    def __init__(
        self,
        *args,
        draft_num_layers: int = 0,
        draft_cp_size: int = 1,
        draft_num_kv_heads: int = 0,
        draft_head_dim: int = 0,
        draft_dtype: DataType = DataType.BF16,
        draft_kv_factor: int = 2,
        draft_total_num_kv_heads: Optional[int] = None,
        draft_scratch_tokens: int = 0,
        draft_window_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        self._init_standalone_draft_cache(
            draft_num_layers=draft_num_layers,
            draft_num_kv_heads=draft_num_kv_heads,
            draft_head_dim=draft_head_dim,
            draft_dtype=draft_dtype,
            draft_kv_factor=draft_kv_factor,
            draft_total_num_kv_heads=draft_total_num_kv_heads,
            draft_scratch_tokens=draft_scratch_tokens,
            draft_window_size=draft_window_size,
            layer_specs=kwargs.get("layer_specs"),
            is_draft=bool(kwargs.get("is_draft", False)),
        )
        super().__init__(*args, **kwargs)
        self._check_standalone_draft_isolation()
        if self._standalone_draft_layer_ids and draft_cp_size != self._helix_cp_size:
            # Disaggregated transfer moves the draft pages as it moves the target's: one
            # physical page per rank and ledger block, [cp_rank::cp_size] under helix.
            raise NotImplementedError(
                "A unified standalone draft cache under helix needs the drafter's context "
                f"sharded like the target's pages (draft CP size {draft_cp_size}, helix CP "
                f"size {self._helix_cp_size})."
            )

    @classmethod
    def unified_draft_cache_kwargs(
        cls,
        *,
        model_engine: Any,
        draft_config: Any,
        spec_config: Any,
        mapping: Any,
        kv_cache_config: Any,
        is_disagg: bool,
    ) -> dict:
        """The ``draft_*`` constructor arguments of this deployment's unified draft cache, or ``{}`` for none.

        A DFlash drafter keeps its context KV here under disaggregated serving, where it travels between the
        servers, and an MLA drafter also under helix and, when it sets ``unified_draft_cache_when_aggregated``, in
        every deployment (routing refuses the speculative and KV cache settings it cannot serve). Raises ``ValueError``
        when the target model has no ``draft_model`` to keep there.
        """
        draft_model = getattr(model_engine.model, "draft_model", None)
        needs_mla = _needs_unified_mla_dflash(
            spec_config, draft_config, mapping, is_disagg, draft_model
        )
        unified = _carries_dflash_context(spec_config, is_disagg) or needs_mla
        if not unified:
            return {}
        if draft_model is None:
            raise ValueError(
                "A DFlash deployment that keeps the drafter's context KV in the unified draft cache "
                "needs the target model's draft_model."
            )
        if is_disagg:
            logger.info_once(
                "The DFlash drafter's context KV travels with the prompt KV in the unified draft cache.",
                key="k31_unified_draft_cache_disagg",
            )
        return _draft_cache_kwargs(draft_model, draft_config, spec_config)

    # Draft layers through the engine manager: the methods that lay out, describe, size, cost and release them.

    def _lay_out_regions(self, config: KVCacheManagerConfigPy) -> KVCacheManagerConfigPy:
        """Append the draft layers to the final layer list, then lay out the engine's regions."""
        return super()._lay_out_regions(self._append_standalone_draft_layers(config))

    def _explicit_layer_spec(self, local_layer_id: int) -> Optional[KVCacheLayerSpec]:
        """A draft layer's storage, which the engine's per-layer accessors read; the engine's otherwise."""
        spec = self._unified_draft_layer_specs.get(int(local_layer_id))
        return spec if spec is not None else super()._explicit_layer_spec(local_layer_id)

    def _get_pool_roles(self, pool_id: int) -> tuple[DataRole, Optional[DataRole]]:
        """A draft pool's roles, from the layer of its first buffer: its draft layers fill a pool group of their
        own."""
        spec = self._unified_draft_layer_specs.get(
            int(self.impl.buffer_grouping[pool_id][0].layer_id)
        )
        if spec is not None:
            return Role.KEY, Role.VALUE if spec.kv_factor == 2 else None
        return super()._get_pool_roles(pool_id)

    def _is_local_mamba_layer(self, local_layer_idx: int) -> bool:
        """Draft layers are attention layers."""
        if local_layer_idx in self._unified_draft_layer_specs:
            return False
        return super()._is_local_mamba_layer(local_layer_idx)

    def get_layer_owner(self, global_layer_id: int) -> KVCacheLayerOwner:
        if global_layer_id in self._standalone_draft_layer_ids:
            return KVCacheLayerOwner.DRAFT
        return super().get_layer_owner(global_layer_id)

    def get_layer_dtype(self, global_layer_id: int) -> DataType:
        """A draft layer's storage dtype, the drafter's (which disaggregated peers compare); the engine's otherwise."""
        if global_layer_id in self._standalone_draft_layer_ids:
            return self._draft_dtype
        return super().get_layer_dtype(global_layer_id)

    @property
    def draft_layer_ids(self) -> tuple[int, ...]:
        """The draft layers' global ids, which need not be model layers of this manager; the engine's without them."""
        return self._standalone_draft_layer_ids or super().draft_layer_ids

    def get_buffers(
        self, layer_idx: int, kv_layout: str = "NHD", **kwargs
    ) -> Optional[torch.Tensor]:
        local_layer_idx = self.layer_offsets[layer_idx]
        if self._is_standalone_draft_layer(local_layer_idx):
            return self._get_draft_buffer_view(local_layer_idx, kv_layout)
        return super().get_buffers(layer_idx, kv_layout, **kwargs)

    @property
    def _kv_alloc_extra_tokens(self) -> int:
        """The engine's tokens beyond a request's history, plus the drafter's scratch for its next forward."""
        return super()._kv_alloc_extra_tokens + self._standalone_draft_reserve

    @property
    def _generation_kv_capacity_headroom(self) -> int:
        """The engine's generation headroom, plus the drafter's scratch that every allocation also covers.

        The engine assigns its value once in ``__init__`` (the setter keeps it); the reserve is known before then.
        """
        return self._engine_generation_kv_capacity_headroom + self._standalone_draft_reserve

    @_generation_kv_capacity_headroom.setter
    def _generation_kv_capacity_headroom(self, value: int) -> None:
        self._engine_generation_kv_capacity_headroom = value

    def _get_runtime_cache_size_layer_components(self) -> tuple[List[int], List[Optional[int]]]:
        """The engine's layers without the draft layers, which :meth:`_runtime_cache_size_components` costs.

        Draft layers registered as model layers take the local ids after the model's own, which have no attention
        window entry; without explicit layer specs the engine lists one entry per local layer.
        """
        if not self._draft_registered_as_model_layers:
            return super()._get_runtime_cache_size_layer_components()
        layer_sizes = []
        attention_windows = []
        for local_layer_idx in range(self.num_local_layers):
            if local_layer_idx in self._unified_draft_layer_specs:
                continue
            layer_sizes.append(
                self.get_layer_bytes_per_token(local_layer_idx=local_layer_idx, data_role=Role.ALL)
            )
            attention_windows.append(self.max_attention_window_vec[local_layer_idx])
        return layer_sizes, attention_windows

    def _runtime_cache_size_components(self) -> tuple[int, int, int]:
        """Context/generation bytes per rank-local token and generation bytes per request.

        The engine's estimate of the target layers (as in ``KVCacheManagerV2._get_max_tokens_from_quota_impl``),
        plus the draft layers and every request's draft scratch in the target's layers. Called before and after the
        draft layers are appended to the layer list alike.
        """
        layer_sizes, attention_windows = self._get_runtime_cache_size_layer_components()
        context, generation, per_request = _estimate_cache_size_components(
            layer_sizes,
            attention_windows,
            self.tokens_per_block,
            scratch=self.enable_swa_scratch_reuse,
            generation_capacity_headroom=self._generation_kv_capacity_headroom,
            layer_specs=self.layer_specs,
            scratch_group_keys=self._get_runtime_cache_size_scratch_group_keys(),
        )
        if not self._draft_layer_sizes:
            return context, generation, per_request
        draft_context, draft_generation, draft_per_request = estimate_draft_cache_size_components(
            self._draft_layer_sizes,
            self.tokens_per_block,
            generation_capacity_headroom=self._generation_kv_capacity_headroom,
            scratch_tokens=self._standalone_draft_reserve,
            window_size=self.draft_window_size,
        )
        return (
            context + draft_context,
            generation + draft_generation,
            per_request + self._standalone_draft_reserve * sum(layer_sizes) + draft_per_request,
        )

    def _get_max_tokens_from_quota_impl(self, quota: int) -> float:
        """``KVCacheManagerV2._get_max_tokens_from_quota_impl`` on :meth:`_runtime_cache_size_components`."""
        context_size_per_token, generation_size_per_token, generation_swa_size_per_request = (
            self._runtime_cache_size_components()
        )
        size_per_batch = self._get_generation_request_capacity() * generation_swa_size_per_request
        if quota < size_per_batch:
            return 0
        context_limit_quota = self.max_num_tokens * context_size_per_token + size_per_batch
        if quota <= context_limit_quota:
            if context_size_per_token <= 0:
                return float("inf")
            return (quota - size_per_batch) / context_size_per_token
        if generation_size_per_token <= 0:
            return float("inf")
        return self.max_num_tokens + (quota - context_limit_quota) / generation_size_per_token

    def _get_quota_from_max_tokens_impl(self, max_tokens: int) -> int:
        """``KVCacheManagerV2._get_quota_from_max_tokens_impl`` on :meth:`_runtime_cache_size_components`."""
        context_size_per_token, generation_size_per_token, generation_swa_size_per_request = (
            self._runtime_cache_size_components()
        )
        context_tokens = min(max_tokens, self.max_num_tokens)
        generation_tokens = max_tokens - context_tokens
        return int(
            context_tokens * context_size_per_token
            + generation_tokens * generation_size_per_token
            + self._get_generation_request_capacity() * generation_swa_size_per_request
        )

    def _get_num_free_blocks_for_layers(self, local_layer_ids: Iterable[int]) -> int:
        """Standalone draft layers follow the target's ledger blocks and do not bound warmup."""
        return super()._get_num_free_blocks_for_layers(
            layer for layer in local_layer_ids if layer not in self._unified_draft_layer_specs
        )

    def _resume_context_cache(self, req: LlmRequest, kv_cache: _KVCache) -> bool:
        """The engine's resume; once a request's first context chunk resumed, its draft history starts."""
        if not super()._resume_context_cache(req, kv_cache):
            return False
        if req.is_first_context_chunk:
            self._init_draft_history_after_resume(req, kv_cache)
        return True

    def _create_kv_cache(
        self,
        request_id: int,
        lora_task_id: Optional[int],
        input_tokens: Optional[Sequence[Any]],
        *,
        is_dummy: bool = False,
        **kwargs,
    ) -> Optional[_KVCache]:
        kv_cache = super()._create_kv_cache(
            request_id, lora_task_id, input_tokens, is_dummy=is_dummy, **kwargs
        )
        if kv_cache is not None and is_dummy and self._standalone_draft_layer_ids:
            # The drafter keeps no history for padding and warmup requests.
            self._draft_dummy_request_ids.add(request_id)
        return kv_cache

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False) -> None:
        super().free_resources(request, pin_on_release)
        if self._standalone_draft_layer_ids:
            self.draft_history.pop(request.py_request_id, None)
            self._draft_dummy_request_ids.discard(request.py_request_id)

    def _shutdown_caches(self) -> None:
        super()._shutdown_caches()
        if self._standalone_draft_layer_ids:
            self.draft_history.clear()
            self._draft_dummy_request_ids.clear()
            self._pending_draft_history.clear()

    def queue_draft_history_update(self, update: DraftHistoryUpdate) -> None:
        """Queue a forward's draft history; :meth:`update_resources` publishes it once that forward is done."""
        self._pending_draft_history.append(update)

    def update_resources(self, scheduled_batch, attn_metadata=None, kv_cache_dtype_byte_size=None):
        """Publish the queued draft histories whose forwards are done, oldest first, then the engine's update.

        The executor updates a batch's resources after its sampling, so the batch's own forward is done here. The
        engine's rewind keeps every published history and the drafter's next-forward scratch allocated: a forward
        moves the drafter's position by the target tokens it accepted, which the rewind keeps, and every allocation
        covers the scratch (``_kv_alloc_extra_tokens``).
        """
        pending = self._pending_draft_history
        while pending and pending[0].is_ready():
            pending.popleft().publish()
        return super().update_resources(scheduled_batch, attn_metadata, kv_cache_dtype_byte_size)

    # The draft layers, their pages and the drafter's history.

    def _init_standalone_draft_cache(
        self,
        *,
        draft_num_layers: int,
        draft_num_kv_heads: int,
        draft_head_dim: int,
        draft_dtype: DataType,
        draft_kv_factor: int,
        draft_total_num_kv_heads: Optional[int],
        draft_scratch_tokens: int,
        draft_window_size: Optional[int],
        layer_specs: Optional[Sequence[KVCacheLayerSpec]],
        is_draft: bool,
    ) -> None:
        """Validate and record the unified draft layers; a no-op without them."""
        self._draft_layer_sizes = draft_layer_sizes(
            draft_num_layers=draft_num_layers,
            draft_num_kv_heads=draft_num_kv_heads,
            draft_head_dim=draft_head_dim,
            draft_dtype=draft_dtype,
            draft_kv_factor=draft_kv_factor,
        )
        self.draft_history: dict[int, StandaloneDraftHistory] = {}
        self._draft_dummy_request_ids: set[int] = set()
        self._pending_draft_history: deque[DraftHistoryUpdate] = deque()
        self._unified_draft_layer_specs: dict[int, KVCacheLayerSpec] = {}
        if not self._draft_layer_sizes:
            return
        if layer_specs is not None or is_draft:
            raise ValueError(
                "A unified standalone draft cache belongs to the target manager "
                "and does not support explicit layer specs."
            )
        if draft_total_num_kv_heads is not None and draft_total_num_kv_heads < draft_num_kv_heads:
            raise ValueError("Total draft KV heads must cover the rank-local heads")
        if draft_scratch_tokens < 0:
            raise ValueError("Standalone draft scratch capacity must be nonnegative")
        if draft_window_size is None:
            raise ValueError(
                "A unified standalone draft cache needs the drafter's sliding window "
                "(draft_window_size): without one the draft layers would share the "
                "target's full-attention lifecycle."
            )
        if draft_window_size <= 0:
            raise ValueError("Draft history window must be positive")
        self._draft_num_kv_heads = draft_num_kv_heads
        self._draft_head_dim = draft_head_dim
        self._draft_dtype = draft_dtype
        self._draft_kv_factor = draft_kv_factor
        self._draft_total_num_kv_heads = draft_total_num_kv_heads
        self._standalone_draft_reserve = draft_scratch_tokens
        self.draft_window_size = draft_window_size

    def _check_standalone_draft_isolation(self) -> None:
        """Refuse a built manager whose draft layers share a pool group with other layers.

        The native manager merges layers of equal window, sinks and reuse policy into one lifecycle, and lifecycles
        of equal page sizes into one pool group. The draft layers stay apart only by their window and page size;
        sharing would give the target's pages, page scale and transfer group to the drafter.

        Raises:
            RuntimeError: a pool group holds draft and other layers.
        """
        if not self._standalone_draft_layer_ids:
            return
        layers_by_group = self.impl.layer_grouping
        pool_groups = [
            {
                int(layer_id)
                for variant in desc.slot_desc.variants
                for layer_id in layers_by_group[int(variant.layer_group_id)]
            }
            for desc in self.impl.pool_group_descs
        ]
        draft_layers = frozenset(self._unified_draft_layer_specs)
        conflicts = draft_pool_group_conflicts(pool_groups, draft_layers)
        if conflicts:
            raise RuntimeError(
                "Unified standalone draft layers share pool groups with target layers: "
                + "; ".join(
                    f"pool group {index}: draft layers "
                    f"{sorted(pool_groups[index] & draft_layers)}, other layers "
                    f"{sorted(pool_groups[index] - draft_layers)}"
                    for index in conflicts
                )
                + f" (draft window {self.draft_window_size})."
            )

    def get_draft_kv_cache_view(self):
        """This manager when it holds the standalone draft layers.

        Disaggregated serving publishes the draft layers' count and dtype from it
        (``RankInfo.num_draft_layers`` / ``draft_kv_dtype``), so a peer that keeps the
        drafter's context differently, or not at all, is refused at registration.
        """
        if self._standalone_draft_layer_ids:
            return self
        return super().get_draft_kv_cache_view()

    @property
    def has_standalone_draft_cache(self) -> bool:
        """Whether a standalone drafter's context KV lives in this manager."""
        return bool(self._standalone_draft_layer_ids)

    def _is_standalone_draft_layer(self, local_layer_idx: int) -> bool:
        return local_layer_idx in self._unified_draft_layer_specs

    @property
    def draft_retention_window_size(self) -> Optional[int]:
        """Native window of the draft layers: the history window plus the next write."""
        return self.draft_window_size + 1 if self.draft_window_size is not None else None

    def _draft_layer_bytes_per_token(self, data_role: DataRole) -> int:
        if data_role == Role.ALL:
            kv_factor = self._draft_kv_factor
        elif data_role == Role.KEY or (data_role == Role.VALUE and self._draft_kv_factor == 2):
            kv_factor = 1
        else:
            raise ValueError(f"Invalid standalone draft data role: {data_role}")
        return get_size_in_bytes(
            kv_factor * self._draft_num_kv_heads * self._draft_head_dim, self._draft_dtype
        )

    def _draft_cache_bytes_per_token(self) -> int:
        """Draft bytes this rank stores per rank-local target token; zero without draft layers."""
        if not self._draft_layer_sizes:
            return 0
        return sum(self._draft_layer_sizes)

    def _append_standalone_draft_layers(
        self, config: KVCacheManagerConfigPy
    ) -> KVCacheManagerConfigPy:
        """Append the draft layers after every target layer.

        The draft layers get global ids after the last model layer. They are
        registered as model layers of this manager only when no virtual
        layers (FP4 MLA HP-tail rings, mixed-precision twins) precede them,
        so ``pp_layers`` stays a prefix of the native layer list.
        """
        if not self._draft_layer_sizes:
            return config
        if self.mapping.pp_size > 1:
            raise NotImplementedError("A unified standalone draft cache requires pp_size == 1")
        first_global_id = max(self.pp_layers, default=-1) + 1
        first_local_id = len(config.layers)
        num_draft_layers = len(self._draft_layer_sizes)
        register_model_layers = first_local_id == len(self.pp_layers)
        self._draft_registered_as_model_layers = register_model_layers
        # Rank-local geometry, indexed by native layer id like every other layer.
        padding = first_local_id - len(self.num_kv_heads_per_layer)
        assert padding >= 0
        self.num_kv_heads_per_layer = list(self.num_kv_heads_per_layer) + (
            [0] * padding + [self._draft_num_kv_heads] * num_draft_layers
        )
        padding = first_local_id - len(self.head_dim_per_layer)
        assert padding >= 0
        self.head_dim_per_layer = list(self.head_dim_per_layer) + (
            [0] * padding + [self._draft_head_dim] * num_draft_layers
        )
        self._standalone_draft_layer_ids = tuple(
            range(first_global_id, first_global_id + num_draft_layers)
        )
        layers = list(config.layers)
        for offset, global_id in enumerate(self._standalone_draft_layer_ids):
            local_id = first_local_id + offset
            self._unified_draft_layer_specs[local_id] = KVCacheLayerSpec(
                global_layer_id=global_id,
                owner=KVCacheLayerOwner.DRAFT,
                dtype=self._draft_dtype,
                num_kv_heads=self._draft_num_kv_heads,
                head_dim=self._draft_head_dim,
                buffer_roles=(Role.KEY, Role.VALUE)[: self._draft_kv_factor],
            )
            layers.append(
                AttentionLayerConfig(
                    layer_id=LayerId(local_id),
                    buffers=[
                        BufferConfig(
                            role=role,
                            size=self._draft_layer_bytes_per_token(role) * self.tokens_per_block,
                        )
                        for role in (Role.KEY, Role.VALUE)[: self._draft_kv_factor]
                    ],
                    sliding_window_size=self.draft_retention_window_size,
                    num_sink_tokens=None,
                )
            )
            self.layer_offsets[global_id] = local_id
            if register_model_layers:
                self.pp_layers.append(global_id)
                self.total_num_kv_heads_per_layer.append(
                    self._draft_total_num_kv_heads or self._draft_num_kv_heads
                )
        if register_model_layers:
            self.num_local_layers = len(self.pp_layers)
            self.num_layers += num_draft_layers

        reserve = self._standalone_draft_reserve

        def reserve_scratch(batch: BatchDesc) -> BatchDesc:
            # Every allocation also covers the drafter's next-forward scratch. The engine's entries without
            # history (context requests) carry it in _kv_alloc_extra_tokens; the others get it here.
            return replace(
                batch,
                kv_caches=[
                    KVCacheDesc(
                        capacity=desc.capacity + (reserve if desc.history_length else 0),
                        history_length=desc.history_length,
                        beam_width=desc.beam_width,
                        prompt_length=desc.prompt_length,
                    )
                    for desc in batch.kv_caches
                ],
            )

        logger.info(
            f"[KVCacheManagerV2] unified standalone draft cache: {num_draft_layers} layer(s) "
            f"(global {first_global_id}..{first_global_id + num_draft_layers - 1}), "
            f"dtype={self._draft_dtype}, kv_factor={self._draft_kv_factor}, "
            f"kv_heads={self._draft_num_kv_heads}, head_dim={self._draft_head_dim}, "
            f"scratch={reserve}, "
            f"window={self.draft_window_size}"
        )
        return replace(
            config,
            layers=layers,
            constraints=[reserve_scratch(batch) for batch in config.constraints],
            typical_step=(
                reserve_scratch(config.typical_step) if config.typical_step is not None else None
            ),
        )

    def _get_draft_buffer_view(self, local_layer_idx: int, kv_layout: str) -> torch.Tensor:
        """View the shared pool pages of one draft layer."""
        if kv_layout not in ("NHD", "HND"):
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        layer_id = LayerId(local_layer_idx)
        kv_factor = self._draft_kv_factor
        addr_key = self.impl.get_mem_pool_base_address(layer_id, Role.KEY, PageIndexMode.SHARED)
        if kv_factor == 2:
            addr_value = self.impl.get_mem_pool_base_address(
                layer_id, Role.VALUE, PageIndexMode.SHARED
            )
            page_size_key = self.impl.get_page_stride(layer_id, Role.KEY)
            page_size_value = self.impl.get_page_stride(layer_id, Role.VALUE)
            if addr_key + page_size_key != addr_value or page_size_key != page_size_value:
                raise ValueError(
                    "Standalone draft K/V buffers must have adjacent equal-sized pages"
                )
        num_kv_heads = self._draft_num_kv_heads
        head_dim = self._draft_head_dim
        dimensions = (
            [self.tokens_per_block, num_kv_heads]
            if kv_layout == "NHD"
            else [num_kv_heads, self.tokens_per_block]
        )
        page_upper = self.impl.get_page_index_upper_bound(layer_id, Role.KEY)
        num_pages = page_upper // kv_factor
        return convert_to_torch_tensor(
            TensorWrapper(
                addr_key, self._draft_dtype, [num_pages, kv_factor, *dimensions, head_dim]
            )
        )

    def get_draft_buffers(self, draft_layer_idx: int, kv_layout: str = "HND") -> torch.Tensor:
        """Paged view of draft layer ``draft_layer_idx`` (0-based among the draft layers).

        The view is ``[pages, kv_factor, kv_heads, tokens_per_block, head_dim]``
        for HND, in the draft storage dtype; index it with
        :meth:`get_draft_block_table`.
        """
        if not 0 <= draft_layer_idx < len(self._standalone_draft_layer_ids):
            raise ValueError("No standalone draft cache layer at this index")
        layer_idx = self._standalone_draft_layer_ids[draft_layer_idx]
        return self._get_draft_buffer_view(self.layer_offsets[layer_idx], kv_layout)

    @property
    def draft_max_blocks_per_seq(self) -> int:
        """Maximum physical draft pages per sequence, one per ledger block."""
        return self.max_blocks_per_seq

    def get_draft_num_blocks(self, request_id: int) -> int:
        """Allocated physical draft pages covering the request's global token positions."""
        return self.kv_cache_map[request_id].num_blocks

    def get_draft_block_indices(
        self,
        request_ids: List[int],
        histories: Optional[Sequence[StandaloneDraftHistory]] = None,
    ) -> List[List[int]]:
        """Validated rank-local page indices of the allocated draft pages, without padding."""
        for request_id in request_ids:
            cache = self.kv_cache_map.get(request_id)
            if cache is None or not cache.is_active:
                raise ValueError(f"Standalone draft request {request_id} has no active cache")
        expanded = self._helix_cp_size > 1
        layer_idx = self._standalone_draft_layer_ids[0]
        batch_indices = self.get_batch_cache_indices(request_ids, layer_idx, raw_indices=expanded)
        if expanded:
            converter = self.impl.get_page_index_converter(self.layer_offsets[layer_idx], Role.KEY)
            batch_indices = [converter(indices, PageIndexMode.SHARED) for indices in batch_indices]
        for row, indices in enumerate(batch_indices):
            cache = self.kv_cache_map[request_ids[row]]
            if (
                len(indices) != self.get_draft_num_blocks(request_ids[row])
                or len(indices) > self.draft_max_blocks_per_seq
            ):
                raise ValueError("Standalone draft cache has an incomplete or oversized page table")
            first_required_block = 0
            if self.draft_window_size is not None:
                history = (
                    histories[row]
                    if histories is not None
                    else self.get_draft_history(request_ids[row])
                )
                history_start = (
                    history.position - history.valid_length
                    if history is not None
                    else max(0, cache.history_length - self.draft_window_size)
                )
                # One table entry per ledger block (tokens_per_block x the helix CP size).
                first_required_block = history_start // self._ledger_tokens_per_block
            if any(index == BAD_PAGE_INDEX for index in indices[first_required_block:]):
                raise ValueError(
                    "Draft cache contains missing pages in its required history or scratch"
                )
        return batch_indices

    def get_draft_block_table(
        self,
        request_ids: List[int],
        histories: Optional[Sequence[StandaloneDraftHistory]] = None,
    ) -> torch.Tensor:
        """Host ``[len(request_ids), draft_max_blocks_per_seq]`` int32 draft page table.

        Unused tail entries point to page zero. A padding or warmup request (``_draft_dummy_request_ids``) keeps no
        draft history: its slot's context is empty, so the drafter stores its tokens from column 0, and its row lists
        only the pages it holds (a windowed pool has released a long request's leading pages).
        """
        batch_indices = self.get_draft_block_indices(request_ids, histories)
        table = torch.zeros((len(request_ids), self.draft_max_blocks_per_seq), dtype=torch.int32)
        for row, indices in enumerate(batch_indices):
            if request_ids[row] in self._draft_dummy_request_ids:
                indices = [index for index in indices if index != BAD_PAGE_INDEX]
            table[row, : len(indices)] = torch.tensor(indices, dtype=torch.int32)
        return table

    def get_draft_history(self, request_id: int) -> Optional[StandaloneDraftHistory]:
        return self.draft_history.get(request_id)

    def set_draft_history(self, request_id: int, valid_length: int, position: int) -> None:
        cache = self.kv_cache_map.get(request_id)
        if cache is None or not cache.is_active:
            raise ValueError(
                f"Standalone draft request {request_id} has no active cache allocation"
            )
        history = StandaloneDraftHistory(valid_length, position)
        if history.position > cache.capacity:
            raise ValueError("Standalone draft history exceeds allocated capacity")
        if self.draft_window_size is not None and history.valid_length > self.draft_window_size:
            raise ValueError("Draft history exceeds its retention window")
        self.draft_history[request_id] = history

    def _prepare_disagg_gen_init_cache(self, req: LlmRequest) -> bool:
        """The engine's allocation, then the draft history of the prompt the context server sends.

        The context server's drafter covers the whole prompt and its pages arrive with the target's, so the history
        ends at the prompt and keeps its last ``draft_window_size`` tokens (all of them without a window).
        """
        if not super()._prepare_disagg_gen_init_cache(req):
            return False
        if self._standalone_draft_layer_ids:
            prompt_len = req.total_input_len_cp if self._has_cp_helix else req.prompt_len
            window = self.draft_window_size
            valid_length = prompt_len if window is None else min(prompt_len, window)
            self.set_draft_history(req.py_request_id, valid_length, prompt_len)
        return True

    def _init_draft_history_after_resume(self, req: LlmRequest, kv_cache: _KVCache) -> None:
        """Start a fresh request's draft history at its reused prefix.

        The reuse match covers every cache domain. This runs once, after the
        first resume succeeds, so overlap and chunk retries cannot rewind a
        history that has already advanced on the device.
        """
        if not self._standalone_draft_layer_ids or req.py_request_id in self.draft_history:
            return
        reused = kv_cache.num_committed_tokens
        valid_length = reused
        if self.draft_window_size is not None:
            valid_length = min(valid_length, self.draft_window_size)
        self.set_draft_history(req.py_request_id, valid_length, reused)

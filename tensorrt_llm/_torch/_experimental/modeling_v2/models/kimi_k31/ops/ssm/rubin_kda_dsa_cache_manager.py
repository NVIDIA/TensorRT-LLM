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

A runtime helper of the family's Rubin targets: the engine builds it from the class a target's
``select_kv_cache_manager_cls`` returns (``select_rubin_kv_cache_manager_cls``), with what
``extra_init_kwargs`` returns. A target binds its construction arguments (``bind_kv_cache_args``):
the GPU pools zeroed at construction, the KV cache, index-K and KDA state dtypes it was written for,
and the MLA layers whose latent KV is FP8 beside an NVFP4 cache. The selector, the static estimate
and ``extra_init_kwargs`` refuse a config without them.

The manager derives from the shared KDA hybrid cache manager
(``tensorrt_llm/_torch/modules/kimi_kda/cache_manager.py``), so the shared KDA cache handling
applies to K3.1 as it does to K3, except in the two places where the shared manager loads the
Kimi K3 package's KDA modules. K3.1 seeds the replay caches of transferred requests with plain
copies, and its mid-chunk recurrent-state snapshots depend on this family's KDA prefill. A K3.1
process therefore never imports the K3 package, whose KDA kernels patch the CuTe DSL compile
pipeline of the whole process.
"""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.draft_window_cache import (
    DraftWindowCacheMixin,
    set_draft_window,
    window_draft_cost,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import (
    derive_indexer_k_cache_layer_mask,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAParams
from tensorrt_llm._torch.modules.kimi_kda import cache_manager as _kda_cache_manager
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAIntermediateState,
    KDAReplayState,
    KimiK3HybridCacheManager,
    get_kda_replay_num_spec,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.dkv import (
    PeerRegion,
    gpu_tier_chunk_bytes,
    map_peer_regions,
    unmap_peer_regions,
    verify_peer_regions,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    Role,
    _BasePageTableMaterializer,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import _stack_state_views
from tensorrt_llm._torch.pyexecutor.step_upload_kernels import (
    block_offsets_triton_enabled,
    copy_batch_block_offsets_from_host,
)
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, prefer_pinned
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    copy_batch_block_offsets_to_device,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    GPU_LEVEL,
    BufferConfig,
    DataRole,
    LayerId,
    PageIndexMode,
    _KVCache,
    exact_div,
)

from ..attention.rubin_dsa_cache import (
    get_indexer_k_cache_bytes_per_token,
    get_indexer_k_cache_size_per_token,
)
from .rubin_cache_policy import get_kimi_k31_fp8_kv_layer_ids
from .rubin_kda_metadata import KDA_PREFILL_CHUNK_SIZE

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


#: Where a target's construction arguments of the manager live in ``model_config.extra_attrs``.
KV_CACHE_ARGS_KEY = "rubin_kv_cache_args"

#: The KV cache dtypes a target may name, as the engine passes them (``dtype``).
_KV_DTYPES = {"fp8": DataType.FP8, "nvfp4": DataType.NVFP4, "bf16": DataType.BF16}


def bind_kv_cache_args(
    model_config,
    *,
    zero_gpu_pools: bool,
    kv_dtype: str,
    indexer_k_dtype: str,
    state_dtype: torch.dtype,
    fp8_kv_layers: Optional[Tuple[int, ...]],
) -> None:
    """Record a target's construction arguments of the manager on its model config.

    The engine builds the manager itself, after the model, with the arguments ``extra_init_kwargs``
    returns for the model's config; a target calls this when it is constructed.

    Args:
        model_config: The target's model config (the engine's, not a copy).
        zero_gpu_pools: Zero every GPU pool byte once, before the guard page and the first request.
        kv_dtype: The KV cache dtype the target was written for: ``"fp8"``, ``"nvfp4"`` or
            ``"bf16"``; construction fails for another.
        indexer_k_dtype: The index-K dtype the target was written for: ``"fp8"`` or ``"fp4"``.
        state_dtype: The KDA recurrent state dtype the target was written for.
        fp8_kv_layers: MLA layers whose latent KV is FP8 beside an NVFP4 cache: None for the
            default (the last two MLA layers with an NVFP4 cache, none otherwise), ``()`` for none.
    """
    if kv_dtype not in _KV_DTYPES:
        raise ValueError(f"kv_dtype must be one of {sorted(_KV_DTYPES)}, got {kv_dtype!r}")
    if indexer_k_dtype not in ("fp8", "fp4"):
        raise ValueError(f"indexer_k_dtype must be 'fp8' or 'fp4', got {indexer_k_dtype!r}")
    if not isinstance(state_dtype, torch.dtype):
        raise TypeError(f"state_dtype must be a torch dtype, got {state_dtype!r}")
    model_config.extra_attrs[KV_CACHE_ARGS_KEY] = {
        "zero_gpu_pools": bool(zero_gpu_pools),
        "kv_dtype": kv_dtype,
        "indexer_k_dtype": indexer_k_dtype,
        "state_dtype": state_dtype,
        "fp8_kv_layers": fp8_kv_layers,
    }


#: ``model_kwargs: {text_config: {kimi_k31_draft_window: ...}}``: the unified draft cache keeps only
#: the DFlash2 drafter's sliding window of its context (True, the default) or all of it (False).
DRAFT_WINDOW_ATTR = "kimi_k31_draft_window"


def draft_cache_option(model_config, name: str, default: bool) -> bool:
    """A boolean draft-cache option from ``model_kwargs``: the text config's ``name``, else the
    checkpoint config's, else ``default``."""
    pretrained = model_config.pretrained_config
    for config in (getattr(pretrained, "text_config", None), pretrained):
        value = getattr(config, name, None) if config is not None else None
        if value is None:
            continue
        if not isinstance(value, bool):
            raise ValueError(f"model_kwargs {name} must be true or false, got {value!r}")
        return value
    return default


#: ``model_kwargs: {text_config: {kimi_k31_dkv_draft: ...}}``: under DKV a context server's unified
#: draft cache stores the drafter's history like the latent, each page once over the group (True,
#: the default; ``KVCacheManagerV2(dkv_draft=True)``), or every page on every rank (False).
DKV_DRAFT_ATTR = "kimi_k31_dkv_draft"


#: The index-key layouts under DKV: ``whole`` keeps every index page of a DKV block on
#: every rank (the indexer reads only local keys); ``sharded`` stores them once per group as the
#: latent is (a reader stages its rows' requests' keys from their owners).
DKV_INDEX_LAYOUTS = ("whole", "sharded")


def dkv_kv_cache_args(model_config) -> dict:
    """The manager's DKV arguments from ``TorchLlmArgs.dkv_config`` (``ModelConfig.dkv_config``):
    ``page_split`` stores each latent page once per group (``KVCacheManagerV2(dkv=True)``),
    with the index keys in ``index_layout`` and the drafter history as ``kimi_k31_dkv_draft``
    says; unset or ``replicated`` adds none."""
    config = getattr(model_config, "dkv_config", None)
    if config is None or config.kv_layout != "page_split":
        return {}
    return {
        "dkv": True,
        "dkv_index": config.index_layout,
        "dkv_draft": draft_cache_option(model_config, DKV_DRAFT_ATTR, True),
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


def _kda_prefill_stores_snapshots() -> bool:
    """Whether this family's KDA prefill, which stores the mid-chunk snapshots, runs here: an sm_100,
    sm_103 or sm_107 GPU and an importable CuTe DSL prefill kernel."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
    from tensorrt_llm._utils import get_sm_version

    if not IS_CUTLASS_DSL_AVAILABLE:
        return False
    try:
        if get_sm_version() not in (100, 103, 107):
            return False
    except RuntimeError:  # no usable CUDA device
        return False
    try:
        from ...kernels.ssm import rubin_kda_prefill_ops  # noqa: F401
    except ImportError:
        return False
    return True


# Replay buffers a transferred slot is seeded or cleared in, as the shared per-layer loop does.
_SEEDED_ROLES = (
    *_kda_cache_manager._KDA_REPLAY_CONV_ROLES,
    *_kda_cache_manager._KDA_REPLAY_DRAFT_ROLES,
)


@dataclass(frozen=True)
class _StackedReplaySeed:
    """``[layers, slots, ...]`` views of the local KDA layers' conv states and seeded replay
    buffers."""

    conv: torch.Tensor
    replay: dict[str, torch.Tensor]
    # Bounds of the [Q | K | V] conv-state sections: 0, D, 2D, 3D.
    section_bounds: tuple[int, ...]
    committed_window: int


def _seed_stacked_layers(plan: _StackedReplaySeed, state_indices: torch.Tensor) -> None:
    """Write what the shared per-layer loop writes, in all layers at once: one launch per replay
    buffer kind and conv section, whatever the layer count."""
    for buffer in plan.replay.values():
        buffer.index_fill_(1, state_indices, 0)
    selected_conv = plan.conv.index_select(1, state_indices)
    bounds = plan.section_bounds
    for section, name in enumerate(_kda_cache_manager._KDA_REPLAY_CONV_ROLES):
        buffer = plan.replay[name]
        buffer[..., : plan.committed_window].index_copy_(
            1,
            state_indices,
            selected_conv[:, :, bounds[section] : bounds[section + 1], :].to(buffer.dtype),
        )


# The shared seed_transferred_slots hands a plan to seed_kda_replay_layers of the object the
# planner returns with it.
_STACKED_SEEDER = SimpleNamespace(seed_kda_replay_layers=_seed_stacked_layers)


class RubinKdaReplayState(KDAReplayState):
    """The shared KDA fused-verify replay state, seeding transferred slots from stacked views.

    The shared state seeds the replay caches of transferred requests in all layers at once with
    the Kimi K3 package's seed kernels. Kimi K3.1 instead stacks each kind of per-layer pool view
    into one ``[layers, slots, ...]`` view, as the V2 pools space a kind's per-layer views evenly,
    and seeds all local KDA layers with one plain copy per buffer kind and conv section. Views
    that do not stack (CPU tensors, unevenly spaced layers) keep the shared state's per-layer loop
    of the same copies.
    """

    @override
    def _plan_batched_seed(self) -> tuple[SimpleNamespace, _StackedReplaySeed] | None:
        """Stacked views of every seeded buffer kind, or None when one of them does not stack.

        Reads only the views' pointers, shapes and strides. The shared ``_get_batched_seed_plan``
        keeps the result until the views are bound again or released.
        """
        conv_states = list(self._conv_states)
        stacked_conv = _stack_state_views(conv_states)
        stacked_replay = {}
        if stacked_conv is not None:
            layers = [self._layer_cache_fields(layer) for layer in range(len(conv_states))]
            for name in _SEEDED_ROLES:
                if layers[0].get(name) is not None:
                    stacked_replay[name] = _stack_state_views([layer[name] for layer in layers])
        if stacked_conv is None or any(view is None for view in stacked_replay.values()):
            if conv_states and conv_states[0].is_cuda:
                logger.warning_once(
                    f"The KDA conv states or replay buffers of the {len(conv_states)} local KDA "
                    "layers are not evenly spaced in their pools; transferred requests are seeded "
                    "layer by layer",
                    key="rubin_kda_replay_seed_per_layer",
                )
            return None
        logger.info_once(
            f"KDA replay caches of transferred requests are seeded in all {len(conv_states)} "
            "local KDA layers at once",
            key="rubin_kda_replay_seed_stacked",
        )
        section_bounds = tuple(itertools.accumulate(self._conv_section_dims, initial=0))
        return _STACKED_SEEDER, _StackedReplaySeed(
            stacked_conv, stacked_replay, section_bounds, self._committed_window
        )


class RubinKdaDsaHybridCacheManager(DraftWindowCacheMixin, KimiK3HybridCacheManager):
    """Kimi K3 KDA/MLA cache plus a DSA indexer K-cache (Kimi K3.1).

    Kimi K3.1 adds a DeepSeek sparse-attention indexer to its MLA layers. The
    indexer keys live in one extra ``Role.INDEX_KEY`` buffer per indexer layer,
    so they share allocation, reuse and release with the latent KV pages. The
    DSA accessors mirror :class:`DSACacheManagerV2`, except that local layer
    indices also cover the KDA layers, which own neither a K page nor an
    indexer buffer. A unified standalone drafter's layers share admission and
    transfer with the target's, but own neither a DSA K page nor an indexer
    buffer.

    With an NVFP4 cache, the FP8 KV layers of ``get_kimi_k31_fp8_kv_layer_ids``
    (by default the last two MLA layers) keep their latent KV as E4M3 pages.
    They keep their indexer buffer and the DSA lifecycle, so they share slot
    ids with the NVFP4 layers, while V2 puts their larger K pages in a second
    physical pool of the DSA layer group. The page tables stay one per
    lifecycle: one virtual attention-op pool, after every lifecycle pool,
    addresses that pool from the DSA slot ids.
    """

    # Read by ``is_dsa_cache_manager``: the DSA backend and metadata may use
    # this manager's indexer K-cache and primary-pool page mapping.
    provides_dsa_indexer_k_cache = True

    @classmethod
    def extra_init_kwargs(cls, model_config) -> dict:
        """Constructor arguments the engine adds for this manager (``_create_kv_cache_manager``):
        the DSA config, the text config and what the target bound (``bind_kv_cache_args``,
        required)."""
        pretrained_config = model_config.pretrained_config
        return {
            "sparse_attention_config": model_config.sparse_attention_config,
            "pretrained_config": getattr(pretrained_config, "text_config", None)
            or pretrained_config,
            **_bound_kv_cache_args(model_config),
            **dkv_kv_cache_args(model_config),
            "draft_window": draft_cache_option(model_config, DRAFT_WINDOW_ATTR, True),
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
        fp8_kv_layers: Optional[Tuple[int, ...]] = None,
        dkv: bool = False,
        dkv_index: str = "whole",
        dkv_draft: bool = False,
        draft_window: bool = False,
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
            fp8_kv_layers: MLA layers whose latent KV is FP8 beside an NVFP4 cache
                (``get_kimi_k31_fp8_kv_layer_ids``); None for the default.
            dkv_draft: Under DKV, store a context server's drafter history like the latent, each
                page once over the group (``KVCacheManagerV2(dkv_draft=True)``).
            draft_window: Keep only the DFlash2 drafter's sliding window of its context in a
                unified draft cache (``set_draft_window``); the engine passes the
                ``kimi_k31_draft_window`` option (default on).
            **kwargs: ``KimiK3HybridCacheManager`` arguments; ``layer_mask``
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
        if dkv_index not in DKV_INDEX_LAYOUTS:
            raise ValueError(f"dkv_index must be one of {DKV_INDEX_LAYOUTS}")
        # Read by _extra_buffers_per_layer and the byte accounting during the base construction.
        self._dkv_index_whole = dkv_index == "whole"
        if dkv:
            # KVCacheManagerV2(dkv=True): each rank keeps one latent page per DKV block,
            # and, with whole index keys, every index-key page of it (_extra_buffers_per_layer).
            kwargs["dkv"] = True
            kwargs["dkv_draft"] = dkv_draft
        if draft_window:
            set_draft_window(kwargs)

        # Read by _extra_buffers_per_layer during the base construction.
        self.quant_block_size = 128
        self.index_head_dim = sparse_params.index_head_dim
        self.indexer_k_dtype = sparse_params.indexer_k_dtype
        self.use_fp4 = self.indexer_k_dtype == "fp4"
        self.use_fp8_ds_mla = False
        self.mla_kv_cache_residual_dim = 0
        self._fp8_mla_global_layer_ids = get_kimi_k31_fp8_kv_layer_ids(
            pretrained_config,
            nvfp4_enabled=kwargs.get("dtype") == DataType.NVFP4,
            fp8_kv_layers=fp8_kv_layers,
        )
        if self._fp8_mla_global_layer_ids:
            if kwargs.get("dtype") != DataType.NVFP4:
                raise ValueError(
                    "Kimi K3.1 cache: fp8_kv_layers requires the NVFP4 KV cache storage."
                )
            logger.info(
                "Kimi K3.1 DSA layers with FP8 latent KV beside the NVFP4 cache: "
                f"{sorted(self._fp8_mla_global_layer_ids)}."
            )
        self._primary_pools: dict[int, torch.Tensor] = {}
        # get_primary_pool by local layer, so a per-layer call skips the binding lookups.
        self._primary_pool_by_layer: dict[int, torch.Tensor] = {}
        # Set by _prepare_page_table_tensor when this rank holds FP8 latent layers.
        self._fp8_latent_op_pool: Optional[int] = None
        self._fp8_latent_source_pool: Optional[int] = None
        self._fp8_latent_index_scales: Optional[torch.Tensor] = None
        self._fp8_latent_kv_offsets: Optional[torch.Tensor] = None
        self._fp8_latent_materializer: Optional[_BasePageTableMaterializer] = None
        full_indexer_mask = derive_indexer_k_cache_layer_mask(
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
            if self._get_layer_dtype_by_local_id(local_layer_idx) == DataType.NVFP4:
                scale_converter = self.impl.get_page_index_converter(
                    local_layer_idx, Role.KEY_BLOCK_SCALE
                )
                if (converter.scale, converter.layer_offset, converter.expansion) != (
                    scale_converter.scale,
                    scale_converter.layer_offset,
                    scale_converter.expansion,
                ):
                    raise ValueError("NVFP4 DSA data and scale pages must use the same indices.")
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
        # Under DKV every index-key page of a DKV block stays on every rank: the
        # INDEX_KEY buffer has one expanded page per physical page of the block.
        self._indexer_page_expansion = (
            self._kv_shard_size if self.dkv is not None and self._dkv_index_whole else 1
        )
        if any(
            int(converter.expansion) != self._indexer_page_expansion
            for converter in indexer_converters
        ):
            raise ValueError(
                "DSA indexer K-cache supports page expansion only as DKV's whole index "
                f"pages ({self._indexer_page_expansion} per block)"
            )
        # Each cache view starts at its SHARED base address, so the converter's
        # layer offset is already represented by the base pointer.
        self._indexer_slot_page_scale = (
            int(indexer_converters[0].scale) if indexer_converters else 1
        )
        if any(
            int(converter.scale) != self._indexer_slot_page_scale
            for converter in indexer_converters[1:]
        ):
            raise ValueError("DSA requires a uniform shared INDEX_KEY page mapping across layers")
        # With expanded index pages get_pool_block_indices returns the page indices themselves.
        self.indexer_k_cache_page_scale = (
            self._indexer_slot_page_scale if self._indexer_page_expansion == 1 else 1
        )
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
        # The MLA layers whose latent KV lives in the manager-dtype pool, and the FP8 ones.
        self._fp8_latent_local_layers = self._local_fp8_latent_layers()
        fp8_latent_local_layers = frozenset(self._fp8_latent_local_layers)
        self._nvfp4_latent_local_layers = [
            local_layer_idx
            for local_layer_idx in self._attention_local_layers
            if local_layer_idx not in fp8_latent_local_layers
        ]
        if self._fp8_latent_local_layers:
            self._validate_mixed_latent_layout()
        self._log_fp8_latent_cache_pools()
        self._log_fp32_kda_state()
        # (base, size) of this rank's pools mapped below, by the attribute of their mapping.
        self._dkv_own_regions: Dict[str, Tuple[int, int]] = {}
        # Under DKV, every rank's latent pool mapped into this rank (a collective of the
        # group, as the construction is): the reads of other ranks' pages go through it.
        self.dkv_latent_peers: Optional[PeerRegion] = (
            self._map_dkv_latent_peers() if self.dkv is not None else None
        )
        # Sharded index keys: every rank's index pool mapped too, for the indexer's staged reads.
        self.dkv_index_peers: Optional[PeerRegion] = (
            self._map_dkv_index_peers()
            if self.dkv is not None and not self._dkv_index_whole
            else None
        )
        # Startup logs show which cache manager serves the model.
        logger.info_once(
            f"Kimi K3.1 cache manager: {type(self).__name__} (DSA indexer K-cache on "
            f"{len(indexer_local_layers)} of {len(self._attention_local_layers)} local MLA layers, "
            f"indexer_k_dtype={self.indexer_k_dtype}, zero_gpu_pools={self._zero_gpu_pools})",
            key="rubin_kda_dsa_cache_manager",
        )

    def _map_dkv_index_peers(self) -> PeerRegion:
        """Map every rank's index-key pool (sharded index keys) and check the mapping."""
        view = next(iter(self.indexer_k_cache_pool_per_layer.values()))
        region = (view.data_ptr(), view.numel() * view.element_size())
        chunk = gpu_tier_chunk_bytes(self.kv_cache_manager_py_config.cache_tiers[0].quota)
        (peers,) = map_peer_regions(self.dkv, [region], chunk, view.shape[-1])
        verify_peer_regions(self.dkv, [region], [peers], chunk)
        self._dkv_own_regions["dkv_index_peers"] = region
        return peers

    def _map_dkv_latent_peers(self) -> PeerRegion:
        """Map every rank's FP8 latent pool (one pool for all DSA layers) and check the mapping."""
        if self._fp8_latent_local_layers and self._nvfp4_latent_local_layers:
            raise NotImplementedError("A DKV needs one latent pool for every DSA layer")
        pool = self.get_unique_primary_pool()
        region = (pool.data_ptr(), pool.numel() * pool.element_size())
        chunk = gpu_tier_chunk_bytes(self.kv_cache_manager_py_config.cache_tiers[0].quota)
        page_bytes = pool.shape[-1] * pool.element_size()
        (peers,) = map_peer_regions(self.dkv, [region], chunk, page_bytes)
        verify_peer_regions(self.dkv, [region], [peers], chunk)
        self._dkv_own_regions["dkv_latent_peers"] = region
        logger.info(
            f"DKV: rank {self.dkv.rank} of {self.dkv.size} maps every rank's latent "
            f"pool ({region[1] / (1 << 30):.2f} GiB, {chunk >> 20} MiB chunks), checked"
        )
        return peers

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
    def _initialize_spec_state(self) -> KDAIntermediateState | KDAReplayState | None:
        """The shared manager's speculative state, with the K3.1 replay state."""
        num_spec = self._requested_num_spec
        if num_spec is None:
            num_spec = get_kda_replay_num_spec(self.spec_config, manager_supports_replay=True)
        self.kda_context_only = self.kda_context_only and num_spec is not None
        if self.kda_context_only:
            return None
        if num_spec is not None:
            self._kda_replay = RubinKdaReplayState(
                num_spec,
                stochastic_rounding=self._mamba_ssm_stochastic_rounding,
                draft_records=self._requested_draft_records,
            )
            self._kda_replay.validate(self._state_layout)
            return self._kda_replay
        return KDAIntermediateState()

    @override
    def _setup_model_state(self) -> None:
        super()._setup_model_state()
        if self._kda_replay is not None:
            # Plan the seeding once the views are bound, so that the startup log tells whether
            # transferred requests are seeded in all layers at once or layer by layer.
            self._kda_replay._get_batched_seed_plan()

    @override
    def _align_fp4_mla_snapshot_points(self, request) -> None:
        """Round the recurrent snapshot points of an NVFP4 cache down to whole cache blocks.

        A reused context resumes at a recurrent-state snapshot. On a block boundary it resumes on
        whole NVFP4 latent pages, with their block scales and indexer keys, and never copies part
        of a page. Points inside the first block are dropped. The shared manager rounds only for a
        cache with an FP4 MLA layout policy, which this manager does not have.
        """
        unit = self._reuse_tokens_per_block
        if self.dtype != DataType.NVFP4 and unit == self.tokens_per_block:
            super()._align_fp4_mla_snapshot_points(request)
            return
        # Under DKV without partial reuse the unit is a DKV block, whatever the dtype:
        # reuse resumes on whole DKV blocks only.
        request.expect_snapshot_points = sorted(
            {point // unit * unit for point in request.expect_snapshot_points if point >= unit}
        )

    @override
    def update_context_resources(self, scheduled_batch) -> None:
        """With an NVFP4 cache, move each context request's history to its context position.

        The history of an NVFP4 cache follows every completed context chunk, also while a later
        snapshot point is pending; the shared manager does so only for a cache with an FP4 MLA
        layout policy. The snapshot points sit on block boundaries (above), so a commit at one of
        them ends at the history. A pending mid-chunk snapshot is committed first, as the shared
        KDA manager would: that commit moves the history to the snapshot point, and the history
        cannot move back.

        The shared method then runs for every request. For the requests handled here it must do
        nothing more: their pending snapshots are already committed and removed, and their history
        already stands at the context position.
        """
        if self.dtype == DataType.NVFP4:
            for request in scheduled_batch.context_requests:
                snapshot = self._kda_midchunk_snapshots.pop(request.py_request_id, None)
                if snapshot is not None:
                    self._commit_kda_midchunk_snapshot(request, snapshot)
                kv_cache = self.kv_cache_map.get(request.py_request_id)
                if kv_cache is None or not kv_cache.is_active:
                    continue
                position = request.context_current_position
                if kv_cache.history_length < position and not kv_cache.resize(None, position):
                    raise ValueError(
                        "Failed to resize history length of V2 Mamba cache for request "
                        f"{request.py_request_id} to {position} tokens at context update"
                    )
        super().update_context_resources(scheduled_batch)

    @override
    def _get_kda_midchunk_snapshot_alignment(self) -> int | None:
        """Step alignment of mid-chunk snapshot points, or None when disabled.

        On with ``TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1`` (read by the shared manager), block reuse, a
        non-draft manager, pending SSM snapshot pages in the V2 backend and this family's KDA
        prefill, which stores the snapshots. Depends only on configuration shared by all ranks,
        so that every rank merges the same context chunks. The family's KDA metadata plans the
        snapshots on the prefill's chunk grid.
        """
        if (
            not _kda_cache_manager._KDA_MIDCHUNK_SNAPSHOT
            or not self.enable_block_reuse
            or self.is_draft
        ):
            return None
        if not hasattr(_KVCache, "acquire_ssm_snapshot_slot"):
            logger.warning(
                "TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1 is ignored: the KV cache manager V2 "
                "backend has no pending SSM snapshot pages."
            )
            return None
        if not _kda_prefill_stores_snapshots():
            logger.warning(
                "TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1 is ignored: the optimized KDA "
                "prefill kernel is unavailable."
            )
            return None
        logger.info("KDA mid-chunk recurrent-state snapshots are enabled")
        if self.dkv is not None:
            # Under DKV a snapshot must sit on a DKV block, which the prefill's chunk
            # grid divides.
            unit = self._reuse_tokens_per_block
            if unit % KDA_PREFILL_CHUNK_SIZE:
                raise ValueError(
                    f"A DKV block of {unit} tokens is not on the KDA prefill's "
                    f"{KDA_PREFILL_CHUNK_SIZE}-token chunk grid"
                )
            return unit
        return KDA_PREFILL_CHUNK_SIZE

    def _is_fp8_latent_layer(self, local_layer_idx: int) -> bool:
        """Whether a local layer stores its latent KV as FP8 pages.

        Standalone draft layers keep their own dtype: their global ids follow this
        rank's last layer, so on an earlier pipeline stage they can repeat the id of
        a selected model layer.
        """
        return (
            bool(self._fp8_mla_global_layer_ids)
            and local_layer_idx < len(self.pp_layers)
            and local_layer_idx not in self.cp_as_tp_layer_ids
            and self.pp_layers[local_layer_idx] in self._fp8_mla_global_layer_ids
        )

    def _local_fp8_latent_layers(self) -> list[int]:
        """This rank's FP8 latent layers as local indices, in local order.

        Read from ``pp_layers``, so the base construction may call it before
        ``_attention_local_layers`` exists.
        """
        if not self._fp8_mla_global_layer_ids:
            return []
        return [
            local_layer_idx
            for local_layer_idx in range(len(self.pp_layers))
            if self._is_fp8_latent_layer(local_layer_idx)
        ]

    @override
    def _kv_pool_mapping_offset(
        self, layer_id: LayerId, layer_group_id: int, key_base_addr: int
    ) -> int:
        # An FP8 latent page lies in another physical pool than the NVFP4 pages that the
        # lifecycle's pointer row addresses. _prepare_page_table_tensor maps the layer to
        # the virtual attention-op pool instead; this 0 is a placeholder.
        if self._is_fp8_latent_layer(int(layer_id)):
            return 0
        return super()._kv_pool_mapping_offset(layer_id, layer_group_id, key_base_addr)

    @override
    def _prepare_page_table_tensor(self, index_mapper_capacity: int) -> None:
        """Keep one page table per lifecycle and add an attention-op pool for FP8 latent pages.

        The FP8 and NVFP4 K pages of the DSA layer group lie in two physical pools with
        different page strides, for which the base would give every local layer a page
        table of its own (``num_attention_op_pools`` = local layers): a host table that
        is expanded into one plane per layer every step, and planes whose ids no longer
        are the layer group ids a unified drafter decodes its block table from. Instead
        the DSA lifecycle's pointer row addresses the NVFP4 pool, and one virtual
        attention-op pool addresses the FP8 pool from the same slot ids, which
        :meth:`copy_batch_block_offsets` scales by the FP8 page scale. The virtual pool
        comes after every lifecycle pool, and ``index_scales``, ``kv_offset`` and the host
        block offsets keep one entry per lifecycle, so every other layer keeps its plane.
        """
        fp8_local_layers = self._local_fp8_latent_layers()
        if not fp8_local_layers:
            super()._prepare_page_table_tensor(index_mapper_capacity)
            return
        if self.enable_swa_scratch_reuse:
            raise NotImplementedError(
                "FP8 KV layers of the Kimi DSA cache do not support SWA scratch reuse."
            )
        fp8_layers = frozenset(fp8_local_layers)
        for pool_id, layer_ids in enumerate(self.impl.layer_grouping):
            role = self._get_pool_roles(pool_id)[0]
            page_strides = {
                self.impl.get_page_stride(layer_id, role)
                for layer_id in layer_ids
                if int(layer_id) not in fp8_layers
            }
            if len(page_strides) > 1:
                raise NotImplementedError(
                    f"Layer group {pool_id} holds {role} pages of strides {sorted(page_strides)} "
                    "besides the FP8 latent pages; one page table per lifecycle cannot "
                    "address them."
                )
        dsa_pool_id = int(self.impl.get_layer_group_id(LayerId(fp8_local_layers[0])))
        nvfp4_layers = [
            int(layer_id)
            for layer_id in self.impl.layer_grouping[dsa_pool_id]
            if int(layer_id) not in fp8_layers
        ]
        if not nvfp4_layers:
            raise NotImplementedError(
                "FP8 KV layers of the Kimi DSA cache need an NVFP4 DSA layer on every rank."
            )
        # The DSA lifecycle's row addresses the NVFP4 pool. Pin its representative to the
        # first NVFP4 layer: the descriptor order is not part of the V2 contract.
        self._pool_layer_ids_by_role[(dsa_pool_id, Role.KEY)] = LayerId(min(nvfp4_layers))
        self._use_per_layer_page_tables = False
        # The NVFP4 layers make the rows [primary, secondary] x [data, block scale].
        pool_pointers, pool_mapping = self._build_pool_mapping_tensors()
        op_pool_id = self.num_pools
        fp8_base_address = int(
            self.impl.get_mem_pool_base_address(
                LayerId(fp8_local_layers[0]), Role.KEY, PageIndexMode.PER_LAYER
            )
        )
        pointer_rows = pool_pointers.tolist()
        # E4M3 pages carry no block scales.
        pointer_rows.append([[fp8_base_address, 0], [0, 0]])
        mapping_rows = pool_mapping.tolist()
        for local_layer_idx in fp8_local_layers:
            converter = self.impl.get_page_index_converter(LayerId(local_layer_idx), Role.KEY)
            mapping_rows[local_layer_idx] = [op_pool_id, int(converter.layer_offset)]
        pinned = prefer_pinned()
        self.kv_cache_pool_pointers = torch.tensor(
            pointer_rows, dtype=pool_pointers.dtype, device="cpu", pin_memory=pinned
        )
        self.kv_cache_pool_mapping = torch.tensor(
            mapping_rows, dtype=pool_mapping.dtype, device="cpu", pin_memory=pinned
        )
        self.num_attention_op_pools = op_pool_id + 1

        # The lifecycle conversion and host table, as the base builds them; the virtual
        # pool has its own conversion below.
        self.index_scales = torch.empty(
            self.num_pools, dtype=torch.int32, pin_memory=pinned, device="cpu"
        )
        self.kv_offset = torch.empty(
            self.num_pools, dtype=torch.int32, pin_memory=pinned, device="cpu"
        )
        for pool_id in range(self.num_pools):
            role_a, role_b = self._get_pool_roles(pool_id)
            layer_id = self._pool_layer_ids_by_role[(pool_id, role_a)]
            self.index_scales[pool_id] = self.impl.get_page_index_scale(layer_id, role_a)
            if role_b is not None:
                self.kv_offset[pool_id] = exact_div(
                    self.impl.get_mem_pool_base_address(layer_id, role_b, PageIndexMode.SHARED)
                    - self.impl.get_mem_pool_base_address(layer_id, role_a, PageIndexMode.SHARED),
                    self.impl.get_page_stride(layer_id, role_a),
                )
            else:
                self.kv_offset[pool_id] = 0
        self._index_scale_ints = self.index_scales.tolist()
        self.host_kv_cache_block_offsets = torch.zeros(
            self.num_pools,
            index_mapper_capacity * self.max_beam_width,
            2,
            self.max_blocks_per_seq,
            dtype=torch.int32,
            pin_memory=pinned,
            device="cpu",
        )
        self._prepare_page_table_materializer(index_mapper_capacity)

        # The virtual pool's plane holds the DSA slot ids times the FP8 page scale.
        fp8_converter = self.impl.get_page_index_converter(LayerId(fp8_local_layers[0]), Role.KEY)
        self._fp8_latent_op_pool = op_pool_id
        self._fp8_latent_source_pool = dsa_pool_id
        self._fp8_latent_index_scales = torch.tensor(
            [int(fp8_converter.scale)], dtype=torch.int32, device="cpu", pin_memory=pinned
        )
        self._fp8_latent_kv_offsets = torch.zeros(
            1, dtype=torch.int32, device="cpu", pin_memory=pinned
        )
        self._fp8_latent_materializer = (
            _BasePageTableMaterializer(
                self.host_kv_cache_block_offsets[dsa_pool_id : dsa_pool_id + 1],
                self._stream,
                self._fp8_latent_index_scales,
                self._fp8_latent_kv_offsets,
                output_row_capacity=index_mapper_capacity * self.max_copy_beam_width,
            )
            if self._page_table_materializer.uses_device_expansion
            else None
        )

    @override
    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: list[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        max_blocks: Optional[int] = None,
    ) -> None:
        """Copy the batch's block offsets of every attention-op pool to the device.

        Args:
            dst_tensor: ``[attention_op_pools, sequences, 2, blocks]`` int32 device
                block offsets.
            request_ids: Requests of the batch, in batch order.
            beam_width: Beam width of the batch.
            num_contexts: Number of context requests, which lead the batch.
            num_seqs: Number of sequences of the batch.
            max_blocks: Accepted for signature parity with the base.
        """
        op_pool_id = self._fp8_latent_op_pool
        if op_pool_id is None:
            super().copy_batch_block_offsets(
                dst_tensor, request_ids, beam_width, num_contexts, num_seqs, max_blocks=max_blocks
            )
            return
        if dst_tensor.shape[0] <= op_pool_id:
            raise ValueError(
                f"The block offsets hold {dst_tensor.shape[0]} attention-op pools, but the FP8 "
                f"latent pages need {self.num_attention_op_pools}."
            )
        # The base conversion owns the lifecycle planes.
        super().copy_batch_block_offsets(
            dst_tensor[: self.num_pools],
            request_ids,
            beam_width,
            num_contexts,
            num_seqs,
            max_blocks=max_blocks,
        )
        # The FP8 plane: the DSA lifecycle's slot ids times the FP8 page scale.
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width)
        fp8_plane = dst_tensor[op_pool_id : op_pool_id + 1]
        if self._fp8_latent_materializer is not None:
            self._fp8_latent_materializer.copy_block_offsets_to(fp8_plane, copy_idx)
            return
        source_pool_id = self._fp8_latent_source_pool
        source = self.host_kv_cache_block_offsets[source_pool_id : source_pool_id + 1]
        # As the base does for the lifecycle planes (TRTLLM_KV_BLOCK_OFFSETS_TRITON, beam 1).
        if (
            beam_width == 1
            and block_offsets_triton_enabled()
            and copy_batch_block_offsets_from_host(
                source,
                fp8_plane,
                copy_idx,
                self._fp8_latent_index_scales,
                self._fp8_latent_kv_offsets,
                self._stream,
            )
        ):
            return
        copy_batch_block_offsets_to_device(
            source,
            fp8_plane,
            copy_idx,
            self._fp8_latent_index_scales,
            self._fp8_latent_kv_offsets,
            self._stream.cuda_stream,
        )

    @override
    def materialize_block_offsets_snapshot(self, dst_tensor: torch.Tensor, *args, **kwargs) -> None:
        if self._fp8_latent_op_pool is not None:
            raise NotImplementedError(
                "Block-offset snapshots do not cover the FP8 latent plane of the Kimi DSA cache."
            )
        super().materialize_block_offsets_snapshot(dst_tensor, *args, **kwargs)

    @property
    def indexer_k_cache_local_layer_mask(self) -> list[bool]:
        """Indexer ownership of each local layer; KDA layers own none."""
        return [layer_idx in self._indexer_layer_ids for layer_idx in self.pp_layers]

    @override
    def _extra_buffers_per_layer(self, *, tokens_per_block: int) -> dict[int, list[BufferConfig]]:
        """One ``Role.INDEX_KEY`` buffer per local indexer layer."""
        # Under DKV the manager's blocks are DKV blocks of several physical pages; the
        # index keys keep every one of them (a page expansion), as the indexer scans them all.
        override = (
            {"tokens_per_block_override": tokens_per_block}
            if getattr(self, "_dkv", None) is not None and getattr(self, "_dkv_index_whole", True)
            else {}
        )
        return {
            self.layer_offsets[layer_idx]: [
                BufferConfig(
                    role=Role.INDEX_KEY,
                    size=self._indexer_bytes_per_token() * tokens_per_block,
                    **override,
                )
            ]
            for layer_idx in sorted(self._indexer_layer_ids)
            if layer_idx in self.layer_offsets
        }

    def _indexer_bytes_per_token(self) -> int:
        return get_indexer_k_cache_bytes_per_token(
            self.index_head_dim, self.quant_block_size, self.indexer_k_dtype
        )

    @override
    def _get_layer_dtype_by_local_id(self, local_layer_id: int) -> DataType:
        if self._is_fp8_latent_layer(local_layer_id):
            return DataType.FP8
        return super()._get_layer_dtype_by_local_id(local_layer_id)

    def get_kv_cache_dtype(self, layer_idx: int | None = None) -> DataType:
        """Return the storage dtype for a global MLA layer."""
        if layer_idx is None:
            return self.dtype
        return self._get_layer_dtype_by_local_id(self.layer_offsets[layer_idx])

    @override
    def _get_buffer_roles_for_layer(self, local_layer_idx: int) -> list[DataRole]:
        if self._is_fp8_latent_layer(local_layer_idx):
            return [Role.KEY]
        return super()._get_buffer_roles_for_layer(local_layer_idx)

    @override
    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: DataRole) -> int:
        index_bytes = (
            self._indexer_bytes_per_token()
            if local_layer_idx < len(self.pp_layers)
            and self.pp_layers[local_layer_idx] in self._indexer_layer_ids
            else 0
        )
        if getattr(self, "_dkv", None) is not None and getattr(self, "_dkv_index_whole", True):
            # A rank stores every index key of a DKV block per page of latent it stores.
            index_bytes *= self._kv_shard_size
        if data_role == Role.INDEX_KEY:
            return index_bytes
        if self._is_fp8_latent_layer(local_layer_idx):
            cache_bytes = (
                self.num_kv_heads_per_layer[local_layer_idx]
                * self.head_dim_per_layer[local_layer_idx]
                if data_role in (Role.KEY, Role.ALL)
                else 0
            )
        else:
            cache_bytes = super().get_layer_bytes_per_token(local_layer_idx, data_role)
        return cache_bytes + index_bytes if data_role == Role.ALL else cache_bytes

    @override
    def _attention_cache_bytes_per_token(self) -> int:
        if self._draft_layer_sizes:
            # The base sums the data roles of the target layers (K, NVFP4 block scales) and the
            # standalone draft layers with their page expansion. The indexer K is a role of its
            # own on the MLA layers, which the base leaves out, so it is added here.
            return super()._attention_cache_bytes_per_token() + sum(
                self.get_layer_bytes_per_token(local_layer_idx, Role.INDEX_KEY)
                for local_layer_idx in range(self.num_local_layers)
            )
        return sum(
            self.get_layer_bytes_per_token(local_layer_idx, Role.ALL)
            for local_layer_idx in range(self.num_local_layers)
        )

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
        fp8_kv_layers = _bound_kv_cache_args(model_config)["fp8_kv_layers"]
        slope, fixed_cost = super().get_cache_size_per_token(
            model_config,
            mapping,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        if draft_cache_option(model_config, DRAFT_WINDOW_ATTR, True):
            slope, fixed_cost = window_draft_cost(
                slope,
                fixed_cost,
                tokens_per_block=tokens_per_block,
                max_batch_size=max_batch_size,
                mapping=mapping,
                **kwargs,
            )
        index_bytes_per_token = get_indexer_k_cache_size_per_token(model_config, mapping)
        pretrained_config = model_config.pretrained_config
        config = getattr(pretrained_config, "text_config", None) or pretrained_config
        quant_config = model_config.quant_config
        fp8_layers = get_kimi_k31_fp8_kv_layer_ids(
            config,
            nvfp4_enabled=quant_config is not None and quant_config.quant_mode.has_fp4_kv_cache(),
            fp8_kv_layers=fp8_kv_layers,
        )
        local_fp8_layers = fp8_layers.intersection(mapping.pp_layers(config.num_hidden_layers))
        head_dim = int(config.kv_lora_rank + config.qk_rope_head_dim)
        # The base slope sizes every MLA layer as a latent-only NVFP4 page (half a byte per value
        # plus one E4M3 scale per 16 values): a DSA config does not take the FP4 MLA formula, which
        # would size FP8 layers by itself. An FP8 KV layer stores one byte per latent value.
        fallback_bytes_per_token = len(local_fp8_layers) * (
            head_dim - head_dim // 2 - head_dim // 16
        )
        extra_bytes_per_token = index_bytes_per_token + fallback_bytes_per_token
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

    def _count_attention_slots(self, local_layer_idx: Optional[int] = None) -> int:
        """Physical slot count of an MLA K pool (not its V2 page bound).

        Counts the K pool of ``local_layer_idx``, by default the first MLA layer's.
        """
        if local_layer_idx is None:
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
        slots = block_indices.clamp(min=0, max=self._num_attention_slots - 1)
        expansion = getattr(self, "_indexer_page_expansion", 1)
        if expansion == 1:
            return slots.to(torch.int32)
        # The expanded INDEX_KEY page of physical page j of a DKV block (PageIndexConverter):
        # slot * scale * expansion + j, relative to the layer's SHARED base.
        pages = slots.to(torch.int64) * (self._indexer_slot_page_scale * expansion)
        pages = pages.unsqueeze(-1) + torch.arange(expansion, dtype=torch.int64)
        return pages.reshape(slots.shape[0], -1).to(torch.int32)

    @property
    def indexer_tokens_per_block(self) -> int:
        """Tokens of one column of ``get_pool_block_indices``: a page, except with sharded DKV
        index keys, where a rank keeps one index page per DKV block and a column is a block."""
        expansion = getattr(self, "_indexer_page_expansion", 1)
        shard = self._kv_shard_size if self.dkv is not None else 1
        return self.tokens_per_block * shard // expansion

    @property
    def dkv_index_sharded(self) -> bool:
        """DKV with sharded index keys: the model builds its own index-key insert and gather slots
        (the owned page of each DKV block); the engine's per-page slot mappings do not apply."""
        return self.dkv is not None and not self._dkv_index_whole

    @property
    def indexer_max_blocks_per_seq(self) -> int:
        """Columns of an index-key block table: a DKV block holds several index pages, and
        ``get_pool_block_indices`` may list every column of the block-offset table."""
        expansion = getattr(self, "_indexer_page_expansion", 1)
        table = getattr(self, "host_kv_cache_block_offsets", None)
        columns = self.max_blocks_per_seq if table is None else int(table.shape[-1])
        return max(self.max_blocks_per_seq, columns) * expansion

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
        return min(width, max(1, -(-capacity // self._ledger_tokens_per_block)))

    def get_primary_pool(self, local_layer_idx: int) -> torch.Tensor:
        """Return this layer's physical K pool as ``[slots, pages, 1, elements]``.

        Different storage dtypes have distinct physical pools in the same
        lifecycle. Slot IDs remain shared with the indexer; page scales and
        offsets describe the layers coalesced in each physical pool.
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
        # NVFP4 codes are packed two per byte; their E4M3 scales live in a
        # separate role. DSA gathers both through the manager's pool pointers.
        packed = dtype == DataType.NVFP4
        elements_per_layer = self.tokens_per_block * num_heads * head_dim // (2 if packed else 1)
        shape = [self._num_attention_slots, int(converter.scale), 1, elements_per_layer]
        pool = convert_to_torch_tensor(
            TensorWrapper(base_address, torch.int8 if packed else dtype, shape)
        )
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

    def _validate_mixed_latent_layout(self) -> None:
        """Check the NVFP4 / FP8 latent page layout once, before the first forward.

        The attention ops, the DSA gathers, the indexer and a unified drafter address
        pages through the tables that :meth:`_prepare_page_table_tensor` built for this
        layout; under any other V2 layout they would read or write the wrong pages.

        Raises:
            RuntimeError: V2 laid the pages out differently, or a page table does not
                address them as the attention ops expect.
        """
        nvfp4_layers = self._nvfp4_latent_local_layers
        fp8_layers = self._fp8_latent_local_layers
        dsa_pool_id = int(self._attention_pool_id)
        op_pool_id = self.num_pools

        def require(condition: bool, message: str) -> None:
            if not condition:
                raise RuntimeError(f"Kimi DSA FP8 KV layer layout: {message}.")

        require(bool(nvfp4_layers), "no NVFP4 MLA layer on this rank")
        # Each latent pool holds one page per layer in every slot, in local layer order.
        for ordinal, local_layer_idx in enumerate(nvfp4_layers):
            layer_idx = self.pp_layers[local_layer_idx]
            key = self._get_page_index_params(local_layer_idx, Role.KEY)
            expected = (len(nvfp4_layers), 1, ordinal)
            require(
                key == expected,
                f"NVFP4 layer {layer_idx} has K page index (scale, expansion, offset) "
                f"{key}, expected {expected}",
            )
            block_scale = self._get_page_index_params(local_layer_idx, Role.KEY_BLOCK_SCALE)
            require(
                block_scale == key,
                f"NVFP4 layer {layer_idx} has block-scale page index {block_scale}, "
                f"but K page index {key}",
            )
        for ordinal, local_layer_idx in enumerate(fp8_layers):
            layer_idx = self.pp_layers[local_layer_idx]
            key = self._get_page_index_params(local_layer_idx, Role.KEY)
            expected = (len(fp8_layers), 1, ordinal)
            require(
                key == expected,
                f"FP8 layer {layer_idx} has K page index (scale, expansion, offset) "
                f"{key}, expected {expected}",
            )
            elements = (
                self.tokens_per_block
                * self.num_kv_heads_per_layer[local_layer_idx]
                * self.head_dim_per_layer[local_layer_idx]
            )
            stride = int(self.impl.get_page_stride(local_layer_idx, Role.KEY))
            require(
                stride == elements,
                f"FP8 layer {layer_idx} has K pages of {stride} B, expected {elements}",
            )
        # Both latent pools share the DSA slot ids, which get_pool_block_indices bounds.
        nvfp4_slots = self._count_attention_slots(nvfp4_layers[0])
        fp8_slots = self._count_attention_slots(fp8_layers[0])
        require(
            nvfp4_slots == fp8_slots == self._num_attention_slots,
            f"the NVFP4 latent pool has {nvfp4_slots} slots and the FP8 latent pool "
            f"{fp8_slots}, expected {self._num_attention_slots}",
        )

        # The DSA lifecycle's row addresses the NVFP4 pool, the virtual attention-op pool
        # after every lifecycle pool addresses the FP8 pool.
        pointers = self.kv_cache_pool_pointers.tolist()
        mapping = self.kv_cache_pool_mapping.tolist()
        require(
            self._fp8_latent_op_pool == op_pool_id
            and self.num_attention_op_pools == len(pointers) == op_pool_id + 1
            and not self._use_per_layer_page_tables,
            f"{len(pointers)} pool pointer rows and {self.num_attention_op_pools} "
            f"attention-op pools, expected {op_pool_id + 1} with the FP8 pool last",
        )
        nvfp4_row = [
            [
                int(
                    self.impl.get_mem_pool_base_address(
                        nvfp4_layers[0], Role.KEY, PageIndexMode.SHARED
                    )
                ),
                int(
                    self.impl.get_mem_pool_base_address(
                        nvfp4_layers[0], Role.KEY_BLOCK_SCALE, PageIndexMode.SHARED
                    )
                ),
            ],
            [0, 0],
        ]
        require(
            pointers[dsa_pool_id] == nvfp4_row and 0 not in nvfp4_row[0],
            f"DSA pool row {pointers[dsa_pool_id]}, expected the NVFP4 pool {nvfp4_row}",
        )
        fp8_row = [
            [
                int(
                    self.impl.get_mem_pool_base_address(
                        fp8_layers[0], Role.KEY, PageIndexMode.PER_LAYER
                    )
                ),
                0,
            ],
            [0, 0],
        ]
        require(
            pointers[op_pool_id] == fp8_row and fp8_row[0][0] != 0,
            f"FP8 pool row {pointers[op_pool_id]}, expected {fp8_row}",
        )
        for pool_id, local_layers in ((dsa_pool_id, nvfp4_layers), (op_pool_id, fp8_layers)):
            for ordinal, local_layer_idx in enumerate(local_layers):
                require(
                    mapping[local_layer_idx] == [pool_id, ordinal],
                    f"layer {self.pp_layers[local_layer_idx]} maps to "
                    f"{mapping[local_layer_idx]}, expected {[pool_id, ordinal]}",
                )
        require(
            self._index_scale_ints[dsa_pool_id] == len(nvfp4_layers),
            f"DSA pool page scale {self._index_scale_ints[dsa_pool_id]}, expected "
            f"{len(nvfp4_layers)}",
        )

        # Every other layer keeps its layer group's plane, which the conversion tables
        # cover one entry per lifecycle; a unified drafter decodes its draft table from that plane
        # with the draft layers' page scale.
        require(
            self.index_scales.numel()
            == self.kv_offset.numel()
            == self.host_kv_cache_block_offsets.shape[0]
            == op_pool_id,
            "the page-index conversion must hold one entry per lifecycle pool",
        )
        fp8_layer_set = frozenset(fp8_layers)
        for local_layer_idx, (pool_id, _) in enumerate(mapping):
            if local_layer_idx not in fp8_layer_set:
                layer_group_id = int(self.layer_to_pool_mapping_dict[local_layer_idx])
                require(
                    pool_id == layer_group_id,
                    f"layer {self.pp_layers[local_layer_idx]} maps to attention-op pool "
                    f"{pool_id}, not to its layer group {layer_group_id}",
                )
        for layer_idx in self.draft_layer_ids:
            local_layer_idx = self.layer_offsets[layer_idx]
            pool_id = mapping[local_layer_idx][0]
            page_scale = int(self.impl.get_page_index_scale(local_layer_idx, Role.KEY))
            require(
                self._index_scale_ints[pool_id] == page_scale,
                f"draft layer {layer_idx} has page scale {page_scale}, its pool "
                f"{self._index_scale_ints[pool_id]}",
            )

        logger.info(
            f"Kimi DSA latent KV: NVFP4 layers "
            f"{[self.pp_layers[layer] for layer in nvfp4_layers]} (page scale "
            f"{len(nvfp4_layers)}), FP8 layers {[self.pp_layers[layer] for layer in fp8_layers]} "
            f"(page scale {len(fp8_layers)}, attention-op pool {op_pool_id} of "
            f"{self.num_attention_op_pools})"
        )

    def _log_fp8_latent_cache_pools(self) -> None:
        """With an FP8 latent cache, log the GPU slots and slot bytes of every pool group, on one line.

        FP8 latent pages are larger than NVFP4 ones, so the same memory holds fewer KV pages and KDA
        state snapshots. Only the cache that serves logs, not the one the engine first builds for
        the cache size estimate (``is_estimating_kv_cache``).
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
    def dkv_peer_pools(self) -> List[Tuple[int, int, PeerRegion]]:
        """The latent pool and, with sharded index keys, the index-key pool, with their mappings."""
        return [
            (*self._dkv_own_regions[name], peers)
            for name in ("dkv_latent_peers", "dkv_index_peers")
            if (peers := getattr(self, name, None)) is not None
        ]

    @override
    def shutdown(self) -> None:
        for name in ("dkv_latent_peers", "dkv_index_peers"):
            peers = getattr(self, name, None)
            if peers is not None:
                # Release the other ranks' chunks before the pools go.
                unmap_peer_regions([peers])
                setattr(self, name, None)
        self.indexer_k_cache_pool_per_layer = {}
        self._primary_pools.clear()
        self._primary_pool_by_layer.clear()
        super().shutdown()


def select_rubin_kv_cache_manager_cls(
    model_config, kv_cache_config, is_disagg: bool = False, cache_transceiver_config=None
) -> Optional[type]:
    """The cache manager of a Kimi K3.1 Rubin target: KDA recurrent state, MLA latent cache and one
    indexer K-cache buffer per MLA layer in one KV cache manager V2. None without DSA or KDA layers
    (the engine's routes then apply).

    Raises:
        ValueError: The engine config cannot run it: no KV cache manager V2, a disaggregated
            transceiver other than Python NIXL, block reuse with pool rebalancing, or
            ``TRTLLM_USE_PY_MAMBA``. Or the target bound no arguments (``bind_kv_cache_args``).
    """
    from tensorrt_llm._torch.pyexecutor._util import _resolve_disagg_transceiver_route
    from tensorrt_llm._torch.pyexecutor.config_utils import is_hybrid_linear
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
        use_py_mamba_cache_manager,
    )

    sparse_config = model_config.sparse_attention_config
    if getattr(sparse_config, "algorithm", None) != "dsa":
        return None
    if (
        not is_hybrid_linear(model_config.pretrained_config)
        or model_config.get_num_mamba_layers() == 0
    ):
        return None
    if kv_cache_config.use_kv_cache_manager_v2 is not True:
        raise ValueError(
            "Kimi K3.1 DSA requires use_kv_cache_manager_v2=True "
            "(the indexer K-cache is a V2 extra buffer)."
        )
    if is_disagg:
        # Only the Python NIXL transceiver moves KDA state and the INDEX_KEY buffers.
        backend, runtime = _resolve_disagg_transceiver_route(cache_transceiver_config)
        if runtime != "PYTHON" or backend != "NIXL":
            raise ValueError(
                "Kimi K3.1 disaggregated serving requires cache_transceiver_config "
                "backend='NIXL' with transceiver_runtime='PYTHON' (got "
                f"backend={backend!r}, transceiver_runtime={runtime!r})."
            )
    if kv_cache_config.enable_block_reuse and kv_cache_config.enable_kv_pool_rebalance:
        raise ValueError(
            "V2 Mamba block reuse is not compatible with enable_kv_pool_rebalance because the "
            "rebalancer does not yet model retained recurrent-state snapshots."
        )
    if use_py_mamba_cache_manager():
        raise ValueError(
            "Kimi K3.1 DSA supports only its V2 cache manager; unset TRTLLM_USE_PY_MAMBA."
        )
    _bound_kv_cache_args(model_config)  # raises when the target bound nothing
    return RubinKdaDsaHybridCacheManager

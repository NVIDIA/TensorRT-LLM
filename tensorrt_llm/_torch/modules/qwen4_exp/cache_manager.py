# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import math
import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.attention.backends.sparse.qsa.constants import (
    QSA_INDEX_K_CACHE_DTYPE,
    QSA_POSITION_CACHE_DTYPE,
    QSA_POSITION_COORDINATE_AXES,
    QSA_SPARSE_KV_CACHE_DTYPES,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.params import QSASparseParams
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.modules.fla.cache_manager import (
    GDNReplayState,
    create_gdn_state,
    validate_gdn_layout,
)
from tensorrt_llm._torch.pyexecutor.config_utils import resolve_hf_torch_dtype
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    IntermediateState,
    _get_local_mamba_cache_layout,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import get_pp_layers
from tensorrt_llm._utils import TensorWrapper, binding_to_torch_dtype, convert_to_torch_tensor
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, PageIndexMode
from tensorrt_llm.runtime.kv_cache_manager_v2._config import DataRole

if TYPE_CHECKING:
    import transformers

# Storage-layout constants; model geometry comes from QSASparseParams.
_INDEX_K_ELEMENT_BYTES = torch.empty((), dtype=QSA_INDEX_K_CACHE_DTYPE).element_size()
_POSITION_ELEMENT_BYTES = torch.empty((), dtype=QSA_POSITION_CACHE_DTYPE).element_size()

# Per-token RoPE/mRoPE coordinates used when raw index keys are compressed.
# They are request state shared by all local QSA layers.
QSA_INDEX_POSITION = DataRole("qsa_index_position")


PLE_CONV_STATE = DataRole("ple_conv_state")
PLE_NGRAM_CONTEXT = DataRole("ple_ngram_context")


def get_qwen4_exp_ple_layer_mask(config: transformers.PretrainedConfig) -> list[bool]:
    """Return the decoder-layer mask for the one-based PLE layer IDs."""
    ple_layer_ids = list(getattr(config, "ple_layer_ids", None) or [])
    invalid_ids = [
        layer_id
        for layer_id in ple_layer_ids
        if not isinstance(layer_id, int)
        or isinstance(layer_id, bool)
        or not 1 <= layer_id <= config.num_hidden_layers
    ]
    if invalid_ids:
        raise ValueError(
            "ple_layer_ids must contain one-based decoder-layer IDs in "
            f"[1, {config.num_hidden_layers}], got {invalid_ids}"
        )
    if len(ple_layer_ids) != len(set(ple_layer_ids)):
        raise ValueError("ple_layer_ids must not contain duplicate layer IDs")
    if len(ple_layer_ids) > 1:
        # Qwen4ExpModel._prepare_ple_state currently resolves one shared state
        # tuple. Supporting multiple PLE layers requires separate per-layer
        # metadata and pools; removing only this guard would silently reuse state.
        raise ValueError(
            "Qwen4-Exp currently supports at most one PLE decoder layer, "
            f"got ple_layer_ids={ple_layer_ids}"
        )
    ple_layer_id_set = set(ple_layer_ids)
    return [(layer_id + 1) in ple_layer_id_set for layer_id in range(config.num_hidden_layers)]


@dataclasses.dataclass
class Qwen4ExpPLECacheParams:
    """Shapes and dtypes for PLE recurrent-state pools."""

    ple_layer_mask: list[bool]
    num_ple_layers: int
    short_conv_channels: int
    short_conv_state_len: int
    ngram_context_len: int
    conv_state_dtype: torch.dtype


def extract_qwen4_exp_ple_cache_params(
    config: transformers.PretrainedConfig,
) -> Qwen4ExpPLECacheParams:
    """Derive PLE recurrent-state pool dimensions from the model config."""
    ple_layer_mask = get_qwen4_exp_ple_layer_mask(config)
    hc_count = getattr(config, "hc_count", 1) or 1
    return Qwen4ExpPLECacheParams(
        ple_layer_mask=ple_layer_mask,
        num_ple_layers=sum(ple_layer_mask),
        short_conv_channels=hc_count * config.hidden_size,
        short_conv_state_len=(config.ple_conv_kernel_size - 1) * config.ngram_size,
        ngram_context_len=config.ngram_size - 1,
        conv_state_dtype=resolve_hf_torch_dtype(config) or torch.bfloat16,
    )


def _qwen4_exp_ple_state_bytes_per_rank(
    model_config,
    params,
    mapping,
    *,
    spec_config,
    use_separate_draft_kv_cache: bool,
) -> int:
    """Bytes this rank needs for Qwen4-Exp PLE recurrent state; 0 otherwise."""
    from tensorrt_llm._torch.pyexecutor.config_utils import is_qwen4_exp

    # The estimator is also called with minimal model-config stubs that carry no
    # `pretrained_config`; such a config is definitionally not Qwen4-Exp.
    pretrained_config = getattr(model_config, "pretrained_config", None)
    if pretrained_config is None or not is_qwen4_exp(pretrained_config):
        return 0
    ple_params = extract_qwen4_exp_ple_cache_params(pretrained_config)
    mamba_layer_mask, full_attention_layer_mask = params.get_layer_masks(
        is_draft=False,
        use_separate_draft_kv_cache=use_separate_draft_kv_cache,
    )
    combined_layer_mask = [
        is_mamba or is_attention
        for is_mamba, is_attention in zip(mamba_layer_mask, full_attention_layer_mask)
    ]
    local_layer_indices, _ = get_pp_layers(
        sum(combined_layer_mask),
        mapping,
        spec_config=spec_config,
        layer_mask=combined_layer_mask,
    )
    local_ple_layers = sum(
        layer_id < len(ple_params.ple_layer_mask) and ple_params.ple_layer_mask[layer_id]
        for layer_id in local_layer_indices
    )
    ple_bytes_per_layer = (
        ple_params.short_conv_channels
        * ple_params.short_conv_state_len
        * ple_params.conv_state_dtype.itemsize
        + ple_params.ngram_context_len * torch.int64.itemsize
    )
    return local_ple_layers * ple_bytes_per_layer


class Qwen4ExpHybridCacheManagerV2(MambaHybridCacheManagerV2):
    """Hybrid GDN/KV manager with lifecycle-coupled sparse side buffers.

    QSA adds a per-layer index-K cache and request-wide position coordinates.
    These buffers share the main K/V allocation, eviction, and prefix-reuse
    lifecycle. The parent hybrid manager owns GDN/Mamba recurrent state.

    Index geometry is resolved from the same serving/checkpoint configuration
    as the indexer. A local fallback could otherwise give the indexer and its
    paged cache different shapes or page strides.
    """

    @override
    def get_disagg_role_mapper_kinds(self) -> dict[DataRole, MapperKind]:
        """PLE state and request-wide QSA positions are replicated across TP."""
        return {
            **super().get_disagg_role_mapper_kinds(),
            PLE_CONV_STATE: MapperKind.REPLICATED,
            PLE_NGRAM_CONTEXT: MapperKind.REPLICATED,
            QSA_INDEX_POSITION: MapperKind.REPLICATED,
        }

    @override
    def __init__(
        self,
        *args,
        sparse_attention_config=None,
        pretrained_config=None,
        layer_mask=None,
        use_replay_state_update: bool | None = None,
        qwen4_exp_ple_cache_params: Qwen4ExpPLECacheParams | None = None,
        **kwargs,
    ) -> None:
        self._requested_replay = use_replay_state_update
        self._ple_params = qwen4_exp_ple_cache_params
        self._pretrained_config = pretrained_config
        self.qsa_index_dim = self.qsa_index_kv_heads = 0
        self.qsa_sparse_layer_ids = []
        self.qsa_position_layer_id = None
        self._qsa_enabled = sparse_attention_config is not None
        if self._qsa_enabled:
            if layer_mask is None:
                raise ValueError("QSA cache allocation requires the full-attention layer mask")
            params = sparse_attention_config.to_sparse_params(pretrained_config=pretrained_config)
            if not isinstance(params, QSASparseParams):
                raise ValueError("Qwen4 QSA requires QSA sparse parameters")
            self.qsa_index_dim, self.qsa_index_kv_heads = (
                params.index_head_dim,
                params.index_kv_heads,
            )
            self.qsa_sparse_layer_ids = [i for i, active in enumerate(layer_mask) if active]
        self._is_ple_draft = kwargs.get("is_draft", False)
        kwargs.setdefault("conv_state_layout", "q_k_v")
        super().__init__(*args, layer_mask=layer_mask, **kwargs)

    @override
    def _extra_buffers_per_layer(
        self,
        *,
        tokens_per_block: int,
    ) -> dict[int, list[BufferConfig]]:
        """Register per-layer index K and lifecycle-aligned position pages."""
        if not self._qsa_enabled:
            return {}
        index_size = (
            self.qsa_index_kv_heads * self.qsa_index_dim * _INDEX_K_ELEMENT_BYTES * tokens_per_block
        )
        local_sparse_layers = [
            layer_id for layer_id in self.qsa_sparse_layer_ids if layer_id in self.layer_offsets
        ]
        # Coordinates are request-wide, so all local indexers use one view.
        # Register the position role once; duplicating it on every sparse layer
        # wastes one three-axis int32 page per layer without adding state.
        self.qsa_position_layer_id = next(iter(local_sparse_layers), None)
        result = {
            self.layer_offsets[layer_id]: [BufferConfig(role=Role.INDEX_KEY, size=index_size)]
            for layer_id in local_sparse_layers
        }
        if self.qsa_position_layer_id is not None:
            local_idx = self.layer_offsets[self.qsa_position_layer_id]
            result[local_idx].append(
                BufferConfig(
                    role=QSA_INDEX_POSITION,
                    size=(
                        QSA_POSITION_COORDINATE_AXES * _POSITION_ELEMENT_BYTES * tokens_per_block
                    ),
                )
            )
        return result

    @override
    def get_index_k_buffer(
        self,
        layer_idx: int,
        kv_layout: str = "NHD",
    ) -> Optional[torch.Tensor]:
        """Return the index-K view using the indexer's resolved geometry."""
        return super().get_index_k_buffer(
            layer_idx,
            num_heads=self.qsa_index_kv_heads,
            head_dim=self.qsa_index_dim,
            dtype=QSA_INDEX_K_CACHE_DTYPE,
            kv_layout=kv_layout,
        )

    def _get_view_by_role_and_layer(
        self,
        layer_idx: int,
        roles: tuple[DataRole, ...],
        *,
        dtype: torch.dtype,
        page_shape: list[int],
    ) -> torch.Tensor:
        """Return a zero-copy strided view of one layer's selected roles across the pool.

        The view has shape [slots, roles, *page_shape] and preserves the physical
        slot stride when layers or roles share a pool, e.g. Qwen DSA. Selected
        roles must be adjacent and equal-sized; expanded page mappings are unsupported.
        """
        if not roles:
            raise ValueError("A slot view requires at least one role")
        layer_offset = self.layer_offsets[layer_idx]
        first_role = roles[0]
        address = self.impl.get_mem_pool_base_address(
            layer_offset, first_role, PageIndexMode.SHARED
        )
        page_stride = self.impl.get_page_stride(layer_offset, first_role)
        expected_stride = math.prod(page_shape) * dtype.itemsize
        if page_stride != expected_stride:
            raise RuntimeError(
                f"Slot-view page stride mismatch: expected {expected_stride}, got {page_stride}"
            )
        for offset, role in enumerate(roles[1:], start=1):
            role_address = self.impl.get_mem_pool_base_address(
                layer_offset, role, PageIndexMode.SHARED
            )
            if (
                role_address != address + offset * page_stride
                or self.impl.get_page_stride(layer_offset, role) != page_stride
            ):
                raise RuntimeError(f"Slot-view roles are not adjacent equal-sized pages: {roles}")

        converter = self.impl.get_page_index_converter(layer_offset, first_role)
        scale = int(converter.scale)
        if scale < len(roles):
            raise RuntimeError(
                f"Invalid slot-view page-index scale: {scale} for {len(roles)} roles"
            )
        if int(converter.expansion) != 1:
            raise RuntimeError("Slot views do not support expanded page indices")
        page_upper = self.impl.get_page_index_upper_bound(layer_offset, first_role)
        num_pages_with_offset = page_upper + int(converter.layer_offset)
        if num_pages_with_offset % scale != 0:
            raise RuntimeError("Slot-view page mapping is inconsistent")
        view = convert_to_torch_tensor(
            TensorWrapper(address, dtype, [num_pages_with_offset // scale, scale, *page_shape])
        )
        return view[:, : len(roles)]

    @override
    def get_buffers(
        self,
        layer_idx: int,
        kv_layout: str = "NHD",
    ) -> Optional[torch.Tensor]:
        """Return this layer's adjacent K/V roles with the physical slot stride.

        QSA tables store lifecycle slot IDs. Preserving V2's coalesced stride is
        therefore required even when several layers share one physical pool.
        """
        if not self._qsa_enabled or self.dtype not in QSA_SPARSE_KV_CACHE_DTYPES:
            # The parent owns packed data and scale-page layouts such as
            # NVFP4. QSA returns to the regular backend for these formats.
            return super().get_buffers(layer_idx, kv_layout)
        if kv_layout not in ("NHD", "HND"):
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        if layer_idx not in self.layer_offsets:
            return None
        if self.kv_cache_type == CacheTypeCpp.SELFKONLY:
            raise NotImplementedError("QSA sparse attention requires both K and V cache buffers")

        layer_offset = self.layer_offsets[layer_idx]
        torch_dtype = binding_to_torch_dtype(self.dtype)
        head_dim = self.head_dim_per_layer[layer_offset]
        num_heads = self.num_kv_heads_per_layer[layer_offset]
        if kv_layout == "NHD":
            page_shape = [self.tokens_per_block, num_heads, head_dim]
        else:
            page_shape = [num_heads, self.tokens_per_block, head_dim]
        return self._get_view_by_role_and_layer(
            layer_idx, (Role.KEY, Role.VALUE), dtype=torch_dtype, page_shape=page_shape
        )

    @torch.compiler.disable
    def get_qsa_position_buffer(self) -> Optional[torch.Tensor]:
        """Return per-token three-axis RoPE/mRoPE position coordinates."""
        if self.qsa_position_layer_id is None:
            return None
        return self._get_view_by_role_and_layer(
            self.qsa_position_layer_id,
            (QSA_INDEX_POSITION,),
            dtype=QSA_POSITION_CACHE_DTYPE,
            page_shape=[self.tokens_per_block, QSA_POSITION_COORDINATE_AXES],
        )[:, 0]

    def get_qsa_attention_pool_layout(self) -> tuple[int, int]:
        """Require one block-table mapping for all local sparse layers."""
        if self.qsa_position_layer_id is None:
            raise RuntimeError("QSA cache manager has no local sparse layer")
        local_idx = self.layer_offsets[self.qsa_position_layer_id]
        pool_id = self.layer_to_pool_mapping_dict[local_idx]
        converter = self.impl.get_page_index_converter(local_idx, Role.KEY)
        mapping = (pool_id, int(converter.scale))
        for layer_id in self.qsa_sparse_layer_ids:
            if layer_id not in self.layer_offsets:
                continue
            local_idx = self.layer_offsets[layer_id]
            pool_id = self.layer_to_pool_mapping_dict[local_idx]
            if pool_id >= self.num_attention_op_pools:
                raise RuntimeError(
                    f"KV pool {pool_id} is not represented in the attention block table"
                )
            converter = self.impl.get_page_index_converter(local_idx, Role.KEY)
            scale = int(converter.scale)
            if scale <= 0:
                raise RuntimeError(f"Invalid attention page-index scale: {scale}")
            if int(converter.expansion) != 1:
                raise RuntimeError("Attention slot mapping does not support expanded page indices")
            candidate = (pool_id, scale)
            if candidate != mapping:
                raise RuntimeError(
                    "QSA local layers do not share one attention page mapping: "
                    f"layer {layer_id} uses pool/scale {candidate}, expected {mapping}"
                )
        return mapping

    def _init_qwen4_exp_ple_geometry(
        self,
        params: Optional["Qwen4ExpPLECacheParams"],
        total_layers: int,
        default_state_dtype: torch.dtype,
    ) -> None:
        """Validate and record the Qwen4-Exp PLE pool geometry.

        Disabled PLE contributes no recurrent buffers.
        """
        self._ple_layer_ids: list[int] = []
        self._ple_conv_state_shape: list[int] = []
        self._ple_ngram_context_shape: list[int] = []
        self._ple_conv_state_dtype = default_state_dtype
        self._ple_conv_states: dict[int, torch.Tensor] = {}
        self._ple_ngram_contexts: dict[int, torch.Tensor] = {}
        if params is None:
            return
        if len(params.ple_layer_mask) != total_layers:
            raise ValueError(
                "PLE layer mask length must match the hybrid layer mask: "
                f"got {len(params.ple_layer_mask)}, expected {total_layers}"
            )
        self._ple_layer_ids = [
            layer_id for layer_id, active in enumerate(params.ple_layer_mask) if active
        ]
        if len(self._ple_layer_ids) != params.num_ple_layers:
            raise ValueError(
                "PLE layer mask count does not match num_ple_layers: "
                f"got {len(self._ple_layer_ids)}, expected "
                f"{params.num_ple_layers}"
            )
        if any(not self._mamba_layer_mask[layer_id] for layer_id in self._ple_layer_ids):
            raise ValueError("PLE lifecycle state must belong to Mamba layers")
        if (
            params.short_conv_channels <= 0
            or params.short_conv_state_len <= 0
            or params.ngram_context_len <= 0
        ):
            raise ValueError("PLE recurrent-state dimensions must be positive")
        self._ple_conv_state_shape = [
            params.short_conv_channels,
            params.short_conv_state_len,
        ]
        self._ple_ngram_context_shape = [params.ngram_context_len]
        self._ple_conv_state_dtype = params.conv_state_dtype

    def _setup_ple_states(self, num_state_slots: int) -> None:
        """Bind each local PLE layer to V2-managed recurrent-state views."""
        if not self._ple_layer_ids:
            return
        for layer_id in self._ple_layer_ids:
            local_layer_idx = self.layer_offsets.get(layer_id)
            if local_layer_idx is None:
                continue
            conv_state = self._get_state_buffer(
                local_layer_idx,
                PLE_CONV_STATE,
                self._ple_conv_state_dtype,
                self._ple_conv_state_shape,
            )
            ngram_context = self._get_state_buffer(
                local_layer_idx,
                PLE_NGRAM_CONTEXT,
                dtype=torch.long,
                state_shape=self._ple_ngram_context_shape,
            )
            if conv_state.shape[0] != num_state_slots or ngram_context.shape[0] != num_state_slots:
                raise RuntimeError(
                    "PLE and GDN lifecycle buffers must have the same number "
                    f"of slots: layer={layer_id}, GDN={num_state_slots}, "
                    f"conv={conv_state.shape[0]}, ngram={ngram_context.shape[0]}"
                )
            self._ple_conv_states[layer_id] = conv_state
            self._ple_ngram_contexts[layer_id] = ngram_context
        logger.info(
            "PLE state views bound to V2 lifecycle buffers for local layers "
            f"{sorted(self._ple_conv_states)}: conv "
            f"[{num_state_slots}, {', '.join(map(str, self._ple_conv_state_shape))}] "
            f"({self._ple_conv_state_dtype}), n-gram context "
            f"[{num_state_slots}, {self._ple_ngram_context_shape[0]}] (int64)"
        )

    def ple_layer_cache(self, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return persistent ``(conv_state, ngram_context)`` pools for a layer."""
        conv = self._ple_conv_states.get(layer_idx)
        ngram = self._ple_ngram_contexts.get(layer_idx)
        if conv is None or ngram is None:
            return None
        return conv, ngram

    def _ple_buffer_configs(self) -> list[BufferConfig]:
        if not self._ple_layer_ids:
            return []
        return [
            BufferConfig(
                role=PLE_NGRAM_CONTEXT,
                size=(math.prod(self._ple_ngram_context_shape) * torch.int64.itemsize),
            ),
            BufferConfig(
                role=PLE_CONV_STATE,
                size=(math.prod(self._ple_conv_state_shape) * self._ple_conv_state_dtype.itemsize),
            ),
        ]

    @override
    def _initialize_model_state(self) -> IntermediateState | GDNReplayState:
        state = create_gdn_state(self, self._requested_replay)

        validate_gdn_layout(self, state)
        params = self._ple_params
        if self._is_ple_draft:
            params = None
        elif params is None and getattr(self._pretrained_config, "ple_layer_ids", None):
            params = get_qwen4_exp_ple_cache_params(
                self._pretrained_config,
                total_layers=len(self._mamba_layer_mask),
                is_draft=False,
            )
        self._init_qwen4_exp_ple_geometry(
            params, len(self._mamba_layer_mask), self.conv_state_dtype
        )
        return state

    @override
    def _get_recurrent_buffer_configs(self, global_layer_id: int) -> list[BufferConfig]:
        buffers = super()._get_recurrent_buffer_configs(global_layer_id)
        if global_layer_id in self._ple_layer_ids:
            buffers.extend(self._ple_buffer_configs())
        return buffers

    @override
    def _setup_model_state(self) -> None:
        super()._setup_model_state()
        if self.local_num_mamba_layers:
            self._setup_ple_states(self._state_layout.slot_capacity)

    @classmethod
    @override
    def _extra_state_bytes_per_rank(cls, model_config, mapping, **kwargs) -> int:
        if kwargs.get("is_draft", False):
            return 0
        params, _, _ = _get_local_mamba_cache_layout(
            model_config,
            mapping,
            spec_config=kwargs.get("spec_config"),
            is_draft=False,
            use_separate_draft_kv_cache=kwargs.get("use_separate_draft_kv_cache", False),
        )
        return _qwen4_exp_ple_state_bytes_per_rank(
            model_config,
            params,
            mapping,
            spec_config=kwargs.get("spec_config"),
            use_separate_draft_kv_cache=kwargs.get("use_separate_draft_kv_cache", False),
        )

    @override
    def _shutdown_model_state(self) -> None:
        self._ple_conv_states.clear()
        self._ple_ngram_contexts.clear()
        super()._shutdown_model_state()

    @property
    def use_gdn_cached_replay_all_layer_commit(self) -> bool:
        state = self._speculative_state
        return isinstance(state, GDNReplayState) and state.has_bound_states


def get_qwen4_exp_ple_cache_params(config, *, total_layers: int, is_draft: bool):
    """Align target-only PLE state with a target/draft cache layout."""
    if is_draft:
        return None

    params = extract_qwen4_exp_ple_cache_params(config)
    num_target_layers = len(params.ple_layer_mask)
    if num_target_layers > total_layers:
        raise ValueError(
            "PLE layer mask cannot exceed the hybrid cache layout: "
            f"got {num_target_layers}, expected at most {total_layers}"
        )
    if num_target_layers == total_layers:
        return params

    # Unified one-model caches append attention-only MTP layers.
    return replace(
        params,
        ple_layer_mask=params.ple_layer_mask + [False] * (total_layers - num_target_layers),
    )


__all__ = [
    "Qwen4ExpHybridCacheManagerV2",
    "Qwen4ExpPLECacheParams",
    "extract_qwen4_exp_ple_cache_params",
    "get_qwen4_exp_ple_layer_mask",
    "get_qwen4_exp_ple_cache_params",
    "PLE_CONV_STATE",
    "PLE_NGRAM_CONTEXT",
    "QSA_INDEX_POSITION",
]

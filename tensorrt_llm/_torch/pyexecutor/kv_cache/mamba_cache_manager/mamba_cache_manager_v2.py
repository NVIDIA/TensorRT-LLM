# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import math
import sys
from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.disaggregation.resource.page import MapperKind, RoleLayout
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _RESERVED_REQUEST_IDS,
    BlockReusePolicy,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_stats import KVCacheV2IterationStatsReport
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager,
    CacheTypeCpp,
    DataType,
    get_pp_layers,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, prefer_pinned
from tensorrt_llm.bindings.executor import KvCacheRetentionConfig
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    BatchDesc,
    BufferConfig,
    DataRole,
    KVCacheDesc,
    LayerId,
    PageIndexMode,
    SsmLayerConfig,
    TokenIdExt,
    _KVCache,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager as KVCacheManagerPy
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig as KVCacheManagerConfigPy

from .common import (
    MambaAcceptanceBatch,
    MambaHybridCacheManager,
    MambaLayerCache,
    MambaRole,
    MambaState,
    MambaStateLayout,
    _estimate_mamba_hybrid_cache_cost,
    _get_local_mamba_cache_layout,
    _get_num_cuda_graph_padding_dummy_slots,
    _mamba_effective_tp_size,
    _mamba_regular_snapshot_interval,
    _mamba_snapshot_rule_counts,
    _PrefixReuseDiagnostics,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig
    from tensorrt_llm.sampling_params import SamplingParams

GB = 1 << 30

# In-forward Mamba snapshots (GDN, hybrid block reuse, snapshot offsets from the
# prompt end): run the prompt through its end L in one chunk instead of an extra
# iteration for the tail [P, L). The GDN layers leave the state at P live and
# stage the state at L; see finish_inforward_snapshots. The staging rows (the
# requests per batch that can take the path, at most max_batch_size and
# _INFORWARD_SNAPSHOT_MAX_ROWS, within _INFORWARD_SNAPSHOT_STAGING_BYTES) are
# allocated after the pools, outside the GPU cache quota.
_INFORWARD_SNAPSHOT_MAX_ROWS = 8
_INFORWARD_SNAPSHOT_STAGING_BYTES = 512 << 20


class InforwardSnapshotEntry(NamedTuple):
    """One request's in-forward snapshot in this step's context chunk."""

    row: int  # staging row
    point: int  # snapshot point P
    num_before: int  # chunk tokens before P
    num_after: int  # chunk tokens from P to the prompt end


def _uses_fp4_mla_cache_layout(
    dtype: DataType, kv_cache_type: CacheTypeCpp, pretrained_config: object
) -> bool:
    """Whether a hybrid V2 manager stores its MLA latent cache as FP4 MLA."""
    return (
        dtype == DataType.NVFP4
        and kv_cache_type == CacheTypeCpp.SELFKONLY
        and getattr(pretrained_config, "kv_lora_rank", None) is not None
        and getattr(pretrained_config, "qk_rope_head_dim", None) is not None
    )


class MambaHybridCacheManagerV2(KVCacheManagerV2, MambaHybridCacheManager):
    """Hybrid Mamba cache manager backed by KVCacheManagerV2.

    Attention KV pages and Mamba recurrent-state pages are both owned by the
    Python V2 cache manager.  Mamba layers are represented as V2 SSM layers,
    while this wrapper exposes the state tensors and slot indices expected by
    the PyTorch Mamba kernels.
    """

    _supports_additional_snapshot_offsets = True
    # Whether the model's recurrent layers implement the in-forward snapshot
    # split (GDN).
    _supports_inforward_snapshot = False
    # Whether a decode step whose requests, SSM base blocks and dummy flags all
    # match the last upload keeps the uploaded state indices and dummy mask
    # without rebuilding them.
    _skips_steady_state_index_upload = False
    # In-forward Mamba snapshots (staging rows: inforward_snapshot_rows): longest
    # tail, and the fewest chunk tokens before P (the convolution window at P
    # must lie inside the chunk). See _setup_inforward_snapshot.
    inforward_snapshot_max_tail = 0
    inforward_snapshot_min_main = 0
    # This step's plan {request id: entry}; replaced, never mutated in place.
    inforward_snapshot_plan: Dict[int, InforwardSnapshotEntry] = {}
    _inforward_stage_ssm: Optional[torch.Tensor] = None
    _inforward_stage_conv: Optional[torch.Tensor] = None
    # Recurrent snapshots use their own commit/history endpoints without draft backoff.
    _supports_reuse_match_backoff = False

    # Recurrent (conv/SSM) state summarizes the whole matched prefix, so the
    # cache manager __init__ coerces a positive spec recompute tail to a full
    # re-prefill (see _has_recurrent_state on the attention-only base classes).
    _has_recurrent_state = True

    # Set per instance when the MLA latent cache uses the FP4 MLA layout.
    _attention_cache_layout_policy = None
    # Set per instance once the FP4 MLA layout has logged a snapshot move.
    _fp4_mla_snapshot_realignment_logged = False
    # Set per instance by the FP4 MLA layout on a disaggregated context
    # worker, which keeps no generation-only V scales or packed V.
    fp4_mla_context_only_cache = False

    # Construction and cache configuration

    @override
    def __init__(
        self,
        # mamba cache parameters
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_num_heads: int,
        mamba_n_groups: int,
        mamba_head_dim: int,
        mamba_num_layers: int,
        mamba_layer_mask: List[bool],
        mamba_cache_dtype: torch.dtype,
        mamba_ssm_cache_dtype: torch.dtype,
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
        is_estimating_kv_cache: bool = False,
        is_draft: bool = False,
        conv_state_layout: Literal["x_b_c", "q_k_v"] = "x_b_c",
        gdn_fused_verify: bool = False,
        **kwargs,
    ) -> None:
        """See the class docstring.

        Args:
            gdn_fused_verify: Whether the GDN layers verify drafts with the
                fused kernels (see gdn_replay_partitioned).
        """
        if conv_state_layout not in ("x_b_c", "q_k_v"):
            raise ValueError(f"Unsupported convolution state layout: {conv_state_layout!r}")
        if "model_type" in kwargs:
            # The V1 managers select the conv-state layout by model_type; this
            # class takes it explicitly. Silently absorbing model_type here
            # means a caller's layout request would be dropped on the floor.
            raise TypeError(
                "MambaHybridCacheManagerV2 does not accept 'model_type' "
                f"(got {kwargs['model_type']!r}); pass "
                "conv_state_layout='x_b_c' or 'q_k_v' instead"
            )
        total_layers = len(mamba_layer_mask)
        if layer_mask is None:
            full_attention_layer_mask = [False] * total_layers
        elif len(layer_mask) != total_layers:
            raise ValueError(
                f"layer_mask length ({len(layer_mask)}) must match "
                f"mamba_layer_mask length ({total_layers})"
            )
        else:
            full_attention_layer_mask = list(layer_mask)

        combined_layer_mask = [
            mamba_layer_mask[i] or full_attention_layer_mask[i] for i in range(total_layers)
        ]

        self._mamba_layer_mask = list(mamba_layer_mask)
        self.spec_config = spec_config
        self.gdn_fused_verify = gdn_fused_verify
        self._recurrent_evicted_blocks_total = 0
        self._recurrent_onboarded_blocks_total = 0
        self._recurrent_dropped_blocks_total = 0
        self._recurrent_status_logged = False
        # Branch points keyed by request id. prepare_expect_snapshot_points()
        # overwrites the per-request list on every scheduler pass, so the point
        # has to be re-merged from here rather than stored only on the request.
        self._branch_snapshot_points: Dict[int, int] = {}
        self._snapshot_pruned_tokens_total = 0
        self._page_pruned_tokens_total = 0
        self._branch_snapshots_taken_total = 0
        self._branch_snapshots_skipped_total: Dict[str, int] = {}
        num_cuda_graph_padding_dummy_slots = _get_num_cuda_graph_padding_dummy_slots(
            spec_config, max_batch_size
        )
        self._num_reserved_dummy_slots = num_cuda_graph_padding_dummy_slots + int(
            mapping.enable_attention_dp
        )
        self.ssm_state_dtype = (
            mamba_ssm_cache_dtype if mamba_ssm_cache_dtype is not None else mamba_cache_dtype
        )
        self.conv_state_dtype = mamba_cache_dtype
        self._global_n_groups = mamba_n_groups

        self.pp_layers, _ = get_pp_layers(
            mamba_num_layers + num_layers,
            mapping,
            spec_config=spec_config,
            layer_mask=combined_layer_mask,
        )
        self.mamba_pp_layers = [
            layer_idx for layer_idx in self.pp_layers if mamba_layer_mask[layer_idx]
        ]
        self.local_num_mamba_layers = len(self.mamba_pp_layers)

        if self.local_num_mamba_layers > 0:
            tp_size = _mamba_effective_tp_size(mapping)
            d_inner = mamba_head_dim * mamba_num_heads
            grouped_state_dim = mamba_n_groups * mamba_d_state
            conv_dim = d_inner + 2 * grouped_state_dim
            nheads = mamba_num_heads
            assert nheads % tp_size == 0, "mamba_num_heads must be divisible by tp_size"
            assert conv_dim % tp_size == 0, "conv_dim must be divisible by tp_size"
            if kwargs.get("is_disagg", False) and grouped_state_dim % tp_size != 0:
                raise ValueError(
                    "Disaggregated Mamba transfer requires each convolution "
                    "state section to be divisible by tp_size"
                )
            self._n_groups_per_rank = mamba_n_groups // tp_size
            d_inner_local = d_inner // tp_size
            grouped_state_dim_local = grouped_state_dim // tp_size
            conv_dim = conv_dim // tp_size
            nheads = nheads // tp_size
            self.conv_state_shape = [conv_dim, mamba_d_conv - 1]
            self.ssm_state_shape = [nheads, mamba_head_dim, mamba_d_state]
            # TP-mismatch disaggregated transfers split the flat convolution
            # state at the semantic boundaries selected by the layout.
            if conv_state_layout == "x_b_c":
                self.conv_section_dims = [
                    d_inner_local,
                    grouped_state_dim_local,
                    grouped_state_dim_local,
                ]
            else:
                self.conv_section_dims = [
                    grouped_state_dim_local,
                    grouped_state_dim_local,
                    d_inner_local,
                ]
            self.ssm_bytes = math.prod(self.ssm_state_shape) * self.ssm_state_dtype.itemsize
            self.conv_bytes = math.prod(self.conv_state_shape) * self.conv_state_dtype.itemsize
        else:
            logger.info("No local mamba layers for this rank, skipping mamba state views")
            self._n_groups_per_rank = 0
            self.conv_state_shape = []
            self.ssm_state_shape = []
            self.conv_section_dims = []
            self.ssm_bytes = 0
            self.conv_bytes = 0

        self._state_layout = MambaStateLayout(
            layer_mask=tuple(self._mamba_layer_mask),
            pp_layers=tuple(self.pp_layers),
            mamba_pp_layers=tuple(self.mamba_pp_layers),
            mapping=mapping,
            conv_state_shape=tuple(self.conv_state_shape),
            ssm_state_shape=tuple(self.ssm_state_shape),
            conv_state_dtype=self.conv_state_dtype,
            ssm_state_dtype=self.ssm_state_dtype,
            n_groups_per_rank=self._n_groups_per_rank,
            max_batch_size=max_batch_size,
            state_index_capacity=(max_batch_size + self._num_reserved_dummy_slots),
            slot_capacity=None,
            spec_config=spec_config,
            conv_section_dims=tuple(self.conv_section_dims),
            conv_state_layout=conv_state_layout,
        )
        self._speculative_state: MambaState | None = self._initialize_spec_state()

        if isinstance(num_kv_heads, int):
            per_layer_kv_heads = [num_kv_heads] * total_layers
        else:
            if len(num_kv_heads) != total_layers:
                raise ValueError(
                    f"num_kv_heads list length ({len(num_kv_heads)}) does not "
                    f"match total layers ({total_layers})"
                )
            per_layer_kv_heads = list(num_kv_heads)
        for i, is_mamba in enumerate(mamba_layer_mask):
            if is_mamba:
                per_layer_kv_heads[i] = 0

        kv_cache_config = kv_cache_config.model_copy(deep=True)
        if any(mamba_layer_mask) and kv_cache_config.enable_block_reuse:
            block_reuse_config = kv_cache_config.block_reuse_config
            block_reuse_policy = BlockReusePolicy(block_reuse_config.policy)
            if block_reuse_policy == BlockReusePolicy.ALL_REUSABLE:
                # SSM reuse is valid only at explicit snapshot boundaries.
                kv_cache_config.block_reuse_config = block_reuse_config.model_copy(
                    update={"policy": BlockReusePolicy.PER_REQUEST.value}
                )
        self._attention_cache_layout_policy = None
        pretrained_config = kwargs.get("pretrained_config")
        if _uses_fp4_mla_cache_layout(dtype, kv_cache_type, pretrained_config):
            # NVFP4 MLA latent cache (Kimi K3): compose the FP4 MLA V2 layout
            # (K, K scales, V scales, BF16 HP ring) with the recurrent-state
            # lifecycle. Attention-layout operations are forwarded to the
            # policy; SSM operations stay on this manager.
            from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import (
                Fp4MlaV2CacheLayoutPolicy,
            )

            kv_cache_config = Fp4MlaV2CacheLayoutPolicy.configure_manager(
                self,
                kv_cache_config,
                kv_cache_type,
                dtype=dtype,
                tokens_per_block=tokens_per_block,
                head_dim=head_dim,
                pretrained_config=pretrained_config,
                spec_config=spec_config,
                is_disagg=kwargs.get("is_disagg", False),
                mapping=mapping,
            )
        self.kv_cache_config = kv_cache_config

        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=mamba_num_layers + num_layers,
            num_kv_heads=per_layer_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=combined_layer_mask,
            is_draft=is_draft,
            is_estimating_kv_cache=is_estimating_kv_cache,
            num_reserved_index_slots=self._num_reserved_dummy_slots,
            **kwargs,
        )
        if self._attention_cache_layout_policy is not None:
            self._attention_cache_layout_policy.finalize_manager(self)

        self._setup_state_pool()

    @override
    def _disagg_transfer_overwrites_whole_cached_prefix(self) -> bool:
        # A transferred recurrent state summarizes and overwrites the whole prefix.
        return (
            self.local_num_mamba_layers > 0
            or super()._disagg_transfer_overwrites_whole_cached_prefix()
        )

    def _initialize_spec_state(self) -> MambaState | None:
        """Return validated model-owned state before constructing the pool."""
        if self.spec_config is not None:
            raise NotImplementedError(
                "Speculative decoding requires a model-specific Mamba cache manager"
            )
        return None

    @classmethod
    @override
    def get_cache_size_per_token(
        cls,
        model_config,
        mapping: Mapping,
        *,
        max_batch_size: int,
        kv_cache_config: KvCacheConfig,
        num_layers: Optional[int] = None,
        tokens_per_block: int = 32,
        max_seq_len: Optional[int] = None,
        **kwargs,
    ):
        spec_config = kwargs.get("spec_config")
        num_reserved_dummy_slots = _get_num_cuda_graph_padding_dummy_slots(
            spec_config, max_batch_size
        ) + int(mapping.enable_attention_dp)
        attention_slope_override = None
        attention_intercept = 0
        attention_page_bytes_override = 0
        # Estimator callers may pass a bare config stand-in; only an FP4 MLA
        # model (which always carries a pretrained_config) takes this branch.
        config = getattr(model_config, "pretrained_config", None)
        quant_config = getattr(model_config, "quant_config", None)
        if (
            getattr(config, "kv_lora_rank", None) is not None
            and getattr(config, "qk_rope_head_dim", None) is not None
            and quant_config is not None
            and quant_config.quant_mode.has_fp4_kv_cache()
        ):
            # FP4 MLA pages carry K scales and V scales, and every resident
            # sequence pins one BF16 HP ring page per FP4 MLA layer.
            from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import (
                Fp4MlaKVCacheManagerV2,
            )

            _, _, local_attention_layers = _get_local_mamba_cache_layout(
                model_config,
                mapping,
                spec_config=spec_config,
                is_draft=kwargs.get("is_draft", False),
                use_separate_draft_kv_cache=kwargs.get("use_separate_draft_kv_cache", False),
            )
            attention_slope_override, attention_intercept = (
                Fp4MlaKVCacheManagerV2.get_cache_size_per_token(
                    model_config,
                    mapping,
                    num_layers=local_attention_layers,
                    tokens_per_block=tokens_per_block,
                    max_batch_size=max_batch_size,
                    spec_config=spec_config,
                    is_disagg=kwargs.get("is_disagg", False),
                )
            )
            max_resident_sequences = max_batch_size * mapping.pp_size
            if max_resident_sequences > 0:
                attention_page_bytes_override = attention_intercept // max_resident_sequences

        slope, intercept = _estimate_mamba_hybrid_cache_cost(
            model_config,
            mapping,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            num_reserved_dummy_slots=num_reserved_dummy_slots,
            extra_state_bytes_per_rank=cls._extra_state_bytes_per_rank(
                model_config, mapping, **kwargs
            ),
            include_explicit_snapshots=True,
            cap_partial_attention_snapshots=True,
            attention_slope_override=attention_slope_override,
            attention_page_bytes_override=attention_page_bytes_override,
            **kwargs,
        )
        return slope, intercept + attention_intercept

    @classmethod
    def _extra_state_bytes_per_rank(cls, model_config, mapping: Mapping, **kwargs) -> int:
        return 0

    @override
    def get_disagg_role_mapper_kinds(self) -> dict[DataRole, MapperKind]:
        """Convolution sections shard independently; SSM state shards by head.

        With the FP4 MLA cache layout, the MLA roles take the policy's
        explicit kinds. They are declared per role rather than through the
        ``Role.ALL`` fallback, which SSM state still needs for INDEXED.
        """
        kinds = {
            **super().get_disagg_role_mapper_kinds(),
            MambaRole.CONV_STATE: MapperKind.SECTIONED,
        }
        layout_policy = self._attention_cache_layout_policy
        if layout_policy is not None:
            kinds.update(layout_policy.DISAGG_ROLE_MAPPER_KINDS)
        return kinds

    def get_disagg_global_layer_ids(self, layer_group_id: int) -> List[int]:
        """Transfer IDs of a layer group's layers.

        Without the FP4 MLA cache layout every layer uses its global model
        layer. With it, the MLA latent and HP-ring layers use the policy's
        ``2 * L`` / ``2 * L + 1`` encoding, and recurrent layers use ``2 * L``
        so that the IDs stay collision-free across all of the manager's
        layers and identical under any pipeline split.
        """
        layout_policy = self._attention_cache_layout_policy
        result = []
        for layer_id in self.impl.layer_grouping[layer_group_id]:
            internal_layer = int(layer_id)
            if layout_policy is None:
                global_layer_id = int(self.pp_layers[internal_layer])
            elif self._is_local_mamba_layer(internal_layer):
                global_layer_id = 2 * int(self.pp_layers[internal_layer])
            else:
                global_layer_id = self._fp4_mla_disagg_global_layer_id(internal_layer)
                if global_layer_id is None:
                    raise ValueError(f"Unknown hybrid FP4 MLA internal layer {internal_layer}.")
            result.append(global_layer_id)
        return result

    @override
    def get_disagg_ignored_roles(self) -> frozenset[DataRole]:
        """Keep FP4 MLA V scales and packed V out of disaggregated transfers.

        The generation worker rebuilds them from the transferred K and K
        scales.
        """
        roles = super().get_disagg_ignored_roles()
        layout_policy = self._attention_cache_layout_policy
        if layout_policy is not None:
            roles |= frozenset(layout_policy.DISAGG_IGNORED_ROLES)
        return roles

    @override
    def get_disagg_role_layouts(self) -> dict[DataRole, RoleLayout]:
        """Describe per-layer recurrent geometry for resharding across TP.

        With the FP4 MLA cache layout, the policy's HP ring geometry joins
        them.
        """
        layouts = {}
        if self._attention_cache_layout_policy is not None:
            layouts.update(self._fp4_mla_disagg_role_layouts())
        if self.local_num_mamba_layers == 0:
            return layouts
        d_conv_m1 = int(self.conv_state_shape[1])
        _, head_dim, d_state = self.ssm_state_shape
        layouts.update(
            {
                MambaRole.CONV_STATE: RoleLayout(
                    section_bytes=tuple(
                        int(dim) * d_conv_m1 * self.conv_state_dtype.itemsize
                        for dim in self.conv_section_dims
                    )
                ),
                MambaRole.SSM_STATE: RoleLayout(
                    bytes_per_head=int(head_dim) * int(d_state) * self.ssm_state_dtype.itemsize
                ),
            }
        )
        return layouts

    def _extra_scratch_bytes_per_slot(self) -> int:
        return 0

    @override
    def _get_external_cache_bytes(self, cache: KVCacheManagerPy) -> int:
        bytes_per_slot = self._extra_scratch_bytes_per_slot()
        if not bytes_per_slot or not self.mamba_pp_layers:
            return 0
        layer_id = LayerId(self.layer_offsets[self.mamba_pp_layers[0]])
        scale = cache.get_page_index_scale(layer_id, MambaRole.SSM_STATE)
        pages = cache.get_page_index_upper_bound(layer_id, MambaRole.SSM_STATE)
        return ((pages + scale - 1) // scale) * bytes_per_slot

    def _get_recurrent_buffer_configs(self, global_layer_id: int) -> list[BufferConfig]:
        """Declare all buffers sharing one recurrent layer's slot lifecycle."""
        return [
            BufferConfig(role=MambaRole.SSM_STATE, size=self.ssm_bytes),
            BufferConfig(role=MambaRole.CONV_STATE, size=self.conv_bytes),
        ]

    def _is_local_mamba_layer(self, local_layer_idx: int) -> bool:
        if local_layer_idx >= len(self.pp_layers):
            # Virtual layers past the model's own (the FP4 MLA HP ring) are
            # attention layers.
            return False
        return self._mamba_layer_mask[self.pp_layers[local_layer_idx]]

    @override
    def _get_pool_roles(self, pool_id: int) -> Tuple[DataRole, Optional[DataRole]]:
        layer_id = int(self.impl.layer_grouping[pool_id][0])
        if layer_id >= self.num_local_layers:
            # Virtual layers past the model's own exist only for the FP4 MLA
            # HP ring.
            if (
                self._attention_cache_layout_policy is not None
                and LayerId(layer_id) in self._hp_manager_layer_ids
            ):
                return Role.MLA_HP_TAIL, None
            raise ValueError(f"Unknown hybrid V2 virtual layer {layer_id}.")
        if self._is_local_mamba_layer(layer_id):
            return MambaRole.SSM_STATE, None
        return super()._get_pool_roles(pool_id)

    def _max_resident_sequences(self) -> int:
        return self.max_batch_size * self.mapping.pp_size

    def _mamba_state_bytes_per_slot(self) -> int:
        """Return persistent bytes per slot allocated by the native pools."""
        return sum(
            buffer.size
            for layer_id in self.mamba_pp_layers
            for buffer in self._get_recurrent_buffer_configs(layer_id)
        )

    def _num_ssm_snapshots_for_capacity(
        self,
        capacity: int,
        kv_cache_config: KvCacheConfig,
    ) -> int:
        if capacity <= 0 or not kv_cache_config.enable_block_reuse:
            return 0

        fixed_rules, _ = _mamba_snapshot_rule_counts(
            kv_cache_config, self.max_seq_len, self.tokens_per_block
        )
        interval = _mamba_regular_snapshot_interval(kv_cache_config, self.max_seq_len)
        regular_snapshots = capacity // interval if interval is not None else 0
        return self._max_resident_sequences() * fixed_rules + regular_snapshots

    def _num_ssm_states_per_typical_request(
        self,
        capacity: int,
        kv_cache_config: KvCacheConfig,
    ) -> int:
        """Estimate live state plus retained snapshots for the pool ratio.

        Periodic and from-start snapshot placement treats capacity (avg_seq_len, or its
        fallback) as prompt length. Each unique from-end offset below max_seq_len adds a
        snapshot independently of that estimate, since its actual position depends on the
        request's prompt length.

        Because avg_seq_len includes output tokens, periodic retention is an upper-bound
        estimate for decode-heavy workloads. Longer prompts can still exceed this estimate.

        Data-dependent divergence points are not modeled, only the branch prompt-end fallback.

        Args:
            capacity: Estimated total sequence length of one typical request, in tokens (prompt plus
                output). Serves as the estimated prompt length for snapshot placement.
            kv_cache_config: Cache configuration controlling block reuse and Mamba snapshot
                placement.
        """
        if not kv_cache_config.enable_block_reuse:
            return 1

        state_config = kv_cache_config.mamba_state_config
        # From-start positions are fixed, so model only those within the typical
        # length and deduplicate them against each other and periodic boundaries.
        fixed_points = {
            offset
            for offset in state_config.additional_snapshot_offsets_from_start
            if offset <= capacity
        }
        interval = _mamba_regular_snapshot_interval(kv_cache_config, capacity)
        if interval is None:
            num_snapshots = len(fixed_points)
        else:
            # Runtime includes the endpoint when it is a periodic boundary and deduplicates
            # coincident from-start / periodic points, whose alignment is length-independent.
            num_snapshots = capacity // interval + sum(
                point % interval != 0 for point in fixed_points
            )
        # From-end positions depend on the actual prompt length. Do not assume
        # they coincide with from-start or periodic points at the estimated length.
        # Include offsets beyond capacity when longer prompts can reach them.
        # Offsets at or above max_seq_len cannot produce a positive position.
        # Repeated identical offsets always coincide, so count each offset only once.
        num_snapshots += len(
            {
                offset
                for offset in state_config.additional_snapshot_offsets_from_end
                if offset < self.max_seq_len
            }
        )
        if kv_cache_config.block_reuse_config.enable_branch_snapshot and num_snapshots == 0:
            num_snapshots = 1
        return 1 + num_snapshots

    def _typical_request_descs(
        self,
        capacity: int,
        kv_cache_config: KvCacheConfig,
    ) -> List[KVCacheDesc]:
        """Model desired retention, independently of minimum live capacity.

        Each retained state is modeled as an equal slice of the sequence.
        """
        num_states = self._num_ssm_states_per_typical_request(capacity, kv_cache_config)
        capacity_per_state, capacity_remainder = divmod(capacity, num_states)
        capacities = [capacity_per_state + int(i < capacity_remainder) for i in range(num_states)]
        return [
            KVCacheDesc(
                capacity=state_capacity,
                history_length=max(0, state_capacity - 1),
            )
            for state_capacity in capacities
        ]

    def _get_typical_request_capacity(
        self,
        kv_cache_config: KvCacheConfig,
    ) -> int:
        if kv_cache_config.avg_seq_len is not None:
            return kv_cache_config.avg_seq_len

        fallback_capacity = max(1, self.max_seq_len // 2)
        logger.warning(
            "'kv_cache_config.avg_seq_len' is not set for a hybrid Mamba "
            "model using KV cache manager V2. Falling back to "
            f"max_seq_len / 2={fallback_capacity} for cache-pool sizing. Set "
            "'kv_cache_config.avg_seq_len' in the YAML configuration to the "
            "workload's average total sequence length for KV/SSM pool sizing. "
            "This fallback also estimates snapshot retention using "
            "the total sequence length as a prompt-length upper bound."
        )
        return fallback_capacity

    @override
    def _get_quota_from_max_tokens(self, max_tokens: int) -> int:
        attention_quota = super()._get_quota_from_max_tokens(max_tokens)
        num_request_lineages = self._max_resident_sequences()
        snapshot_slots = self._num_ssm_snapshots_for_capacity(max_tokens, self.kv_cache_config)
        state_slots = num_request_lineages + self._num_reserved_dummy_slots + snapshot_slots
        state_quota = state_slots * self._mamba_state_bytes_per_slot()
        # Once the plan contains any non-live SSM capacity, reserve one partial
        # attention page per request lineage. This remains conservative when
        # the plan contains fewer than one non-live slot per lineage.
        extra_attention_quota = (
            num_request_lineages * self._attention_cache_bytes_per_token() * self.tokens_per_block
            if snapshot_slots > 0
            else 0
        )
        return attention_quota + state_quota + extra_attention_quota

    @override
    def _get_max_tokens_from_quota(self, quota: int) -> float:
        if self._get_quota_from_max_tokens(0) > quota:
            return 0

        low = 0
        high = 1
        while self._get_quota_from_max_tokens(high) <= quota:
            low = high
            high *= 2
            if high >= 1 << 62:
                return float("inf")

        while low + 1 < high:
            mid = (low + high) // 2
            if self._get_quota_from_max_tokens(mid) <= quota:
                low = mid
            else:
                high = mid
        return low

    def _minimum_live_gpu_quota(self) -> int:
        """Return the minimum quota for live states and one attention page."""
        attention_block_quota = self._attention_cache_bytes_per_token() * self.tokens_per_block
        num_state_slots = self._max_resident_sequences() + self._num_reserved_dummy_slots
        state_quota = num_state_slots * self._mamba_state_bytes_per_slot()
        return max(
            self._get_quota_from_max_tokens(0),
            state_quota + attention_block_quota,
        )

    @override
    def _build_cache_config(self, config: KVCacheManagerConfigPy) -> KVCacheManagerConfigPy:
        kv_cache_config = self.kv_cache_config
        gpu_quota = config.cache_tiers[0].quota
        minimum_live_quota = self._minimum_live_gpu_quota()
        if minimum_live_quota > gpu_quota:
            raise ValueError(
                "The V2 Mamba GPU cache quota is too small for live recurrent "
                f"states and attention pages: got {gpu_quota} bytes, need at "
                f"least {minimum_live_quota} bytes."
            )
        # _build_base_config already constructed every attention layer,
        # including dtype-specific scale and subclass-provided side buffers.
        # Preserve those configs and replace only the local Mamba layers.
        layers = list(config.layers)
        # Register pool-owned roles here, then bind their typed views in
        # _get_state_buffer. Model scratch must not participate in warmup cleanup.
        self._recurrent_buffers: dict[tuple[int, DataRole], torch.Tensor | None] = {}
        for local_layer_idx, global_layer_idx in enumerate(self.pp_layers):
            if self._mamba_layer_mask[global_layer_idx]:
                layer_id = LayerId(local_layer_idx)
                buffers = self._get_recurrent_buffer_configs(global_layer_idx)
                for buffer in buffers:
                    self._recurrent_buffers[local_layer_idx, buffer.role] = None
                layers[local_layer_idx] = SsmLayerConfig(
                    layer_id=layer_id,
                    buffers=buffers,
                )

        dummy_requests = [
            KVCacheDesc(capacity=0, history_length=0) for _ in range(self._num_reserved_dummy_slots)
        ]
        constraints = [
            replace(
                batch,
                kv_caches=[*batch.kv_caches, *dummy_requests],
            )
            for batch in config.constraints
        ]

        typical_step = config.typical_step
        if config.initial_pool_ratio is None:
            typical_capacity = self._get_typical_request_capacity(kv_cache_config)
            request_descs = self._typical_request_descs(typical_capacity, kv_cache_config)
            typical_step = BatchDesc(
                request_descs * self._max_resident_sequences() + dummy_requests
            )
        # The recurrent (SSM) state pool must hold one slot per resident
        # sequence plus every reserved dummy slot. Unlike attention pages, a
        # Mamba state is fixed-size per sequence, so this floor is independent
        # of sequence length. The base config only emits constraints when
        # ``avg_seq_len`` is set, and speculative decoding inflates the reserved
        # dummy slots (CUDA-graph padding), so without an explicit floor the SSM
        # pool can be undersized (see the live/dummy-slot check in _setup_states
        # / __init__). Add a min-slots constraint of zero-capacity requests:
        # these cost no attention pages but reserve one SSM slot each.
        if any(isinstance(layer, SsmLayerConfig) for layer in layers):
            ssm_floor_slots = self._max_resident_sequences() + self._num_reserved_dummy_slots
            constraints = [
                *constraints,
                BatchDesc(
                    [KVCacheDesc(capacity=0, history_length=0) for _ in range(ssm_floor_slots)]
                ),
            ]
        result = replace(
            config,
            layers=layers,
            typical_step=typical_step,
            constraints=constraints,
            # SSM lifecycles require minimum-snapshot commit semantics. The
            # flag is harmless when reuse is disabled because no commits are
            # attempted, while the runtime config still needs the invariant.
            commit_min_snapshot=True,
        )
        layout_policy = self._attention_cache_layout_policy
        if layout_policy is not None:
            # Adds the FP4 MLA HP-ring virtual layers and the fallback-layer
            # lifecycle; SSM layers are left unchanged.
            result = layout_policy._build_cache_config(self, result)
        return result

    def _setup_state_pool(self) -> None:
        """Resolve logical slots, bind base views, then initialize model state."""
        self.mamba_layer_offsets = {
            layer_id: offset for offset, layer_id in enumerate(self.mamba_pp_layers)
        }
        self._request_id_to_state_index = {}
        self._request_id_to_is_dummy = {}
        # (request ids, SSM base blocks, dummy flags) of the last upload while
        # it is known to be current.
        self._steady_state_index_key: (
            Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[bool, ...]] | None
        ) = None

        state_index_capacity = self.max_batch_size + self._num_reserved_dummy_slots
        self._state_indices_buffer = DeviceCopyBuffer(
            torch.zeros([state_index_capacity], dtype=torch.int32, device="cuda"),
            pin_memory=prefer_pinned(),
        )
        self._host_state_indices = torch.zeros(
            [state_index_capacity], dtype=torch.int32, pin_memory=prefer_pinned()
        )

        self._dummy_request_mask_buffer = DeviceCopyBuffer(
            torch.zeros(state_index_capacity, dtype=torch.bool, device="cuda"),
            pin_memory=prefer_pinned(),
        )
        self._dummy_request_mask_host = torch.zeros(
            state_index_capacity, dtype=torch.bool, pin_memory=prefer_pinned()
        )
        self._generation_state_indices = torch.arange(
            self.max_batch_size, dtype=torch.int32, device="cuda"
        )

        if self.local_num_mamba_layers > 0:
            first_mamba_local_layer = self.layer_offsets[self.mamba_pp_layers[0]]
            self.ssm_layer_group_id = self.impl.get_layer_group_id(LayerId(first_mamba_local_layer))
            self._ssm_page_index_scale = self.impl.get_page_index_scale(
                LayerId(first_mamba_local_layer), MambaRole.SSM_STATE
            )
            num_ssm_pages = self.impl.get_page_index_upper_bound(
                LayerId(first_mamba_local_layer), MambaRole.SSM_STATE
            )
            num_ssm_slots = (
                num_ssm_pages + self._ssm_page_index_scale - 1
            ) // self._ssm_page_index_scale
            required_live_slots = self._max_resident_sequences() + self._num_reserved_dummy_slots
            if num_ssm_slots < required_live_slots:
                KVCacheManagerV2.shutdown(self)
                raise ValueError(
                    "The V2 Mamba state pool has only "
                    f"{num_ssm_slots} slots but needs at least "
                    f"{required_live_slots} live/dummy slots. Increase the "
                    "KV cache budget or allocate a larger Mamba pool_ratio."
                )
            self._state_layout = self._state_layout.with_slot_capacity(num_ssm_slots)
            self._setup_states()
            self._setup_inforward_snapshot()
        else:
            self.ssm_layer_group_id = None
            self._ssm_page_index_scale = 1
            self.all_ssm_states = []
            self.all_conv_states = []
            self._state_layout = self._state_layout.with_slot_capacity(0)

        self._setup_model_state()

    def _get_state_buffer(
        self, local_layer_idx: int, role, dtype: torch.dtype, state_shape: List[int]
    ) -> torch.Tensor:
        key = (local_layer_idx, role)
        if key not in self._recurrent_buffers:
            raise ValueError(
                f"Recurrent buffer was not registered: layer={local_layer_idx}, role={role}"
            )
        addr = self.impl.get_mem_pool_base_address(
            LayerId(local_layer_idx), role, PageIndexMode.SHARED
        )
        num_pages = self.impl.get_page_index_upper_bound(LayerId(local_layer_idx), role)
        raw = convert_to_torch_tensor(TensorWrapper(addr, dtype, [num_pages] + state_shape))
        page_index_scale = self.impl.get_page_index_scale(LayerId(local_layer_idx), role)
        num_slots = (num_pages + page_index_scale - 1) // page_index_scale
        # V2 coalesces same-size per-layer buffers inside each slot.  Kernels
        # index Mamba states by logical slot id, so expose only this layer's
        # sub-page from each coalesced slot instead of the raw page-index view.
        view = raw.as_strided(
            [num_slots] + state_shape,
            [raw.stride(0) * page_index_scale] + list(raw.stride()[1:]),
        )
        self._recurrent_buffers[key] = view
        return view

    def _setup_states(self) -> None:
        local_layer_ids = [self.layer_offsets[layer_id] for layer_id in self.mamba_pp_layers]
        self.all_ssm_states = [
            self._get_state_buffer(
                local_layer_idx, MambaRole.SSM_STATE, self.ssm_state_dtype, self.ssm_state_shape
            )
            for local_layer_idx in local_layer_ids
        ]
        self.all_conv_states = [
            self._get_state_buffer(
                local_layer_idx, MambaRole.CONV_STATE, self.conv_state_dtype, self.conv_state_shape
            )
            for local_layer_idx in local_layer_ids
        ]

    def _inforward_snapshot_max_tail(self) -> int:
        """The longest tail the in-forward snapshots cover (the largest
        ``mamba_state_config.additional_snapshot_offsets_from_end``)."""
        return max(
            self.kv_cache_config.mamba_state_config.additional_snapshot_offsets_from_end,
            default=0,
        )

    def _inforward_snapshot_layout(self) -> Tuple[int, int]:
        """(staging rows, bytes per row) of the in-forward snapshots; no rows
        where the path does not apply.

        The recurrent layers must implement the in-forward split (GDN with the
        q_k_v convolution layout). Every rank of a server must schedule the
        same chunks, so PP stages without local GDN layers rule the path out.
        """
        row_bytes = self.local_num_mamba_layers * (self.ssm_bytes + self.conv_bytes)
        if not (
            self._supports_inforward_snapshot
            and self._state_layout.conv_state_layout == "q_k_v"
            and self._inforward_snapshot_max_tail() > 0
            and self.kv_cache_config.enable_block_reuse
            and not self.is_draft
            and self.mapping.pp_size == 1
            and row_bytes > 0
        ):
            return 0, row_bytes
        rows = max(
            1,
            min(
                self.max_batch_size,
                _INFORWARD_SNAPSHOT_MAX_ROWS,
                _INFORWARD_SNAPSHOT_STAGING_BYTES // row_bytes,
            ),
        )
        return rows, row_bytes

    def _setup_inforward_snapshot(self) -> None:
        """Allocate the in-forward snapshot staging rows when the path applies."""
        rows, row_bytes = self._inforward_snapshot_layout()
        if rows == 0 or not self.enable_block_reuse:
            return
        layers = self.local_num_mamba_layers
        max_tail = self._inforward_snapshot_max_tail()
        device = self.all_ssm_states[0].device
        # [layer, row, ...]: one dense tensor per layer, written by the GDN
        # layer with index_copy_.
        self._inforward_stage_ssm = torch.empty(
            [layers, rows] + self.ssm_state_shape, dtype=self.ssm_state_dtype, device=device
        )
        self._inforward_stage_conv = torch.empty(
            [layers, rows] + self.conv_state_shape, dtype=self.conv_state_dtype, device=device
        )
        self.inforward_snapshot_rows = rows
        self.inforward_snapshot_max_tail = max_tail
        self.inforward_snapshot_min_main = max(1, self.conv_state_shape[-1])
        logger.info(
            f"In-forward Mamba snapshots: {rows} staging rows "
            f"({rows * row_bytes / 2**20:.0f} MiB), tail <= {max_tail} tokens"
        )

    def inforward_snapshot_stage_views(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """A layer's staging rows: SSM ``[rows, H, V, K]`` and conv ``[rows, conv_dim, d_conv - 1]``."""
        offset = self.mamba_layer_offsets[layer_idx]
        return (self._inforward_stage_ssm[offset], self._inforward_stage_conv[offset])

    def state_index(self, request_id: int) -> int:
        """The live state slot of a request scheduled in this step."""
        return self._request_id_to_state_index[request_id]

    def _plan_inforward_snapshots(self, scheduled_batch: ScheduledRequests) -> None:
        """Turn the scheduler's in-forward snapshot marks into this step's plan."""
        plan: Dict[int, InforwardSnapshotEntry] = {}
        context_only = not scheduled_batch.generation_requests
        for req in scheduled_batch.context_requests:
            point = req.py_mamba_inforward_snapshot_point
            if point is None:
                continue
            req.py_mamba_inforward_snapshot_point = None
            pos = req.context_current_position
            end = pos + req.context_chunk_size
            if (
                context_only
                and self.inforward_snapshot_rows > len(plan)
                and not req.is_dummy_request
                and pos + self.inforward_snapshot_min_main <= point < end
                and end == req.prompt_len
                and end - point <= self.inforward_snapshot_max_tail
                and req.expect_snapshot_points
                and point == req.expect_snapshot_points[-1]
            ):
                plan[req.py_request_id] = InforwardSnapshotEntry(
                    len(plan), point, point - pos, end - point
                )
            else:
                # The scheduler checks the same conditions, so this is not
                # expected. The chunk still runs through the prompt end, so no
                # snapshot at P is recorded for this request.
                logger.warning(
                    f"Request {req.py_request_id} "
                    f"dropped from the in-forward snapshot plan (P={point}, "
                    f"chunk [{pos}, {end}), prompt {req.prompt_len}, "
                    f"context-only batch {context_only})"
                )
        self.inforward_snapshot_plan = plan

    def finish_inforward_snapshots(self, scheduled_batch: ScheduledRequests) -> None:
        """Commit each planned request up to P from its live state, then copy its staged prompt-end state back.

        Called on the execution stream right after the forward, so the copies precede the batch's sampler event.
        """
        plan = self.inforward_snapshot_plan
        if not plan:
            return
        self.inforward_snapshot_plan = {}
        dst, src = [], []
        for request in scheduled_batch.context_requests:
            entry = plan.get(request.py_request_id)
            kv_cache = self.kv_cache_map.get(request.py_request_id)
            if entry is None or kv_cache is None or not kv_cache.is_active:
                continue
            point = entry.point
            if (
                not request.is_dummy_request
                and kv_cache.num_committed_tokens < point
                and kv_cache.history_length <= point
            ):
                if kv_cache.history_length < point and not kv_cache.resize(None, point):
                    raise ValueError(
                        "Failed to resize history length of V2 Mamba cache "
                        f"for request {request.py_request_id} to the in-forward "
                        f"snapshot point {point}"
                    )
                kv_cache.commit(
                    self._augment_tokens_for_block_reuse(
                        self._reuse_token_source(request),
                        request,
                        start=kv_cache.num_committed_tokens,
                        end=point,
                    )
                )
                if self._branch_snapshot_points.get(request.py_request_id) == point:
                    self._branch_snapshots_taken_total += 1
            slot = kv_cache.get_ssm_block_base_index(self.ssm_layer_group_id)
            if slot < 0:
                raise RuntimeError(
                    f"In-forward Mamba snapshot: request {request.py_request_id} lost its live SSM slot"
                )
            dst += [state[slot] for state in self.all_ssm_states]
            dst += [state[slot] for state in self.all_conv_states]
            src += list(self._inforward_stage_ssm[:, entry.row].unbind(0))
            src += list(self._inforward_stage_conv[:, entry.row].unbind(0))
        if dst:
            with torch.cuda.stream(self._stream):
                torch._foreach_copy_(dst, src, non_blocking=True)

    def _setup_model_state(self) -> None:
        """Bind views only after layout, pool and slot capacity are final."""
        if self._speculative_state is not None:
            self._speculative_state.bind(
                self._state_layout, self.all_ssm_states, self.all_conv_states
            )

    @property
    def intermediate_state_indices(self) -> torch.Tensor | None:
        state = self._speculative_state
        return state.intermediate_indices if state is not None else None

    @property
    def intermediate_ssm_states(self) -> torch.Tensor | None:
        state = self._speculative_state
        return state.intermediate_ssm if state is not None else None

    @property
    def intermediate_conv_states(self) -> torch.Tensor | None:
        state = self._speculative_state
        return state.intermediate_conv if state is not None else None

    @override
    def mamba_layer_cache(self, layer_idx: int) -> MambaLayerCache:
        if self._speculative_state is not None and self.spec_config is not None:
            return self._speculative_state.make_layer_cache(
                self.mamba_layer_offsets[layer_idx],
                self.get_conv_states(layer_idx),
                self.get_ssm_states(layer_idx),
            )
        return MambaLayerCache(
            conv=self.get_conv_states(layer_idx), temporal=self.get_ssm_states(layer_idx)
        )

    def _attention_cache_bytes_per_token(self) -> int:
        layout_policy = self._attention_cache_layout_policy
        if layout_policy is not None:
            return layout_policy.get_cache_bytes_per_token(self)
        # Mamba layers have zero KV heads, so the generic calculation naturally
        # returns only bytes owned by local attention layers.
        return super().get_cache_bytes_per_token()

    @override
    def get_cache_bytes_per_token(self) -> int:
        cache_bytes = self._attention_cache_bytes_per_token()

        interval = self.kv_cache_config.mamba_state_config.periodic_snapshot_interval
        if self.kv_cache_config.enable_block_reuse and interval is not None and interval > 0:
            cache_bytes += self._mamba_state_bytes_per_slot() // interval
        if cache_bytes == 0 and self.local_num_mamba_layers > 0:
            cache_bytes = self._mamba_state_bytes_per_slot()
        return max(1, cache_bytes)

    @override
    def get_num_free_blocks(self) -> int:
        reserved = set(self.kv_cache_map) & _RESERVED_REQUEST_IDS
        assert not set(self.kv_cache_map) - reserved, (
            "get_num_free_blocks is only used when the kv cache manager is empty"
        )
        if self._attention_cache_layout_policy is not None:
            return self.blocks_in_primary_pool
        attention_pages = []
        ssm_pages = []
        for local_layer_idx in range(self.num_local_layers):
            layer_id = LayerId(local_layer_idx)
            if self._is_local_mamba_layer(local_layer_idx):
                ssm_pages.append(
                    self.impl.get_page_index_upper_bound(layer_id, MambaRole.SSM_STATE)
                    // self._ssm_page_index_scale
                )
            else:
                attention_pages.append(
                    self.impl.get_page_index_upper_bound(layer_id, Role.KEY) // self.kv_factor
                )
        if attention_pages:
            return max(attention_pages)
        return max(ssm_pages) if ssm_pages else 0

    @property
    @override
    def blocks_in_primary_pool(self) -> int:
        layout_policy = self._attention_cache_layout_policy
        if layout_policy is not None:
            return layout_policy.blocks_in_primary_pool.fget(self)
        for local_layer_idx in range(self.num_local_layers):
            if self._is_local_mamba_layer(local_layer_idx):
                continue
            return self.impl.get_page_index_upper_bound(LayerId(local_layer_idx), Role.KEY)
        return 0

    @override
    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> Optional[torch.Tensor]:
        local_layer_idx = self.layer_offsets[layer_idx]
        if self._is_local_mamba_layer(local_layer_idx):
            return None
        return super().get_buffers(layer_idx, kv_layout)

    @override
    def _create_kv_cache(
        self,
        request_id: int,
        lora_task_id: Optional[int],
        input_tokens: Optional[Sequence[TokenIdExt]],
        *,
        cache_salt: Optional[str] = None,
        is_dummy: bool = False,
        enable_request_stats: bool = False,
        expected_prompt_length: Optional[int] = None,
        kv_cache_retention_config: Optional[KvCacheRetentionConfig] = None,
    ) -> Optional[_KVCache]:
        kv_cache = super()._create_kv_cache(
            request_id,
            lora_task_id,
            input_tokens,
            cache_salt=cache_salt,
            is_dummy=is_dummy,
            enable_request_stats=enable_request_stats,
            expected_prompt_length=expected_prompt_length,
            kv_cache_retention_config=kv_cache_retention_config,
        )
        if (
            self.mapping.rank == 0
            and kv_cache is not None
            and input_tokens is not None
            and not is_dummy
            and self.local_num_mamba_layers > 0
        ):
            # Prefix lookup excludes the final prompt token because it must be
            # recomputed by prefill. Restore it for the request-length metric.
            request_total_tokens = len(input_tokens) + 1
            prefix_reuse_diagnostics = cast(_PrefixReuseDiagnostics, kv_cache)
            logger.debug(
                f"[MambaHybridCacheManagerV2] prefix reuse rank={self.mapping.rank} "
                f"request_id={request_id} "
                f"request_total_tokens={request_total_tokens} "
                f"longest_attention_match_tokens="
                f"{prefix_reuse_diagnostics._get_num_reusable_tokens_before_hybrid_pruning()} "
                f"content_divergence_tokens="
                f"{prefix_reuse_diagnostics._get_num_reusable_tokens_before_pruning()} "
                f"latest_recurrent_snapshot_tokens="
                f"{kv_cache.num_committed_tokens}"
            )
        return kv_cache

    @override
    def add_dummy_requests(
        self,
        request_ids: List[int],
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        max_num_draft_tokens: int = 0,
        kv_reserve_draft_tokens: Optional[int] = None,
        use_mrope: bool = False,
        max_beam_width: int = 1,
        encoder_output_lens: Optional[List[int]] = None,
        draft_kv_cache_manager: Optional[BaseResourceManager] = None,
        capture_sampling_params: Optional["SamplingParams"] = None,
    ) -> List[LlmRequest]:
        requests = super().add_dummy_requests(
            request_ids=request_ids,
            token_nums=token_nums,
            is_gen=is_gen,
            max_num_draft_tokens=max_num_draft_tokens,
            kv_reserve_draft_tokens=kv_reserve_draft_tokens,
            use_mrope=use_mrope,
            max_beam_width=max_beam_width,
            encoder_output_lens=encoder_output_lens,
            draft_kv_cache_manager=draft_kv_cache_manager,
            capture_sampling_params=capture_sampling_params,
        )
        if requests:
            self._setup_state_indices(
                requests,
                num_contexts=0 if is_gen else len(requests),
            )
            if not is_gen:
                self._reset_context_mamba_slots(len(requests))
        return requests

    @override
    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        super().prepare_resources(scheduled_batch)
        if self.inforward_snapshot_rows > 0:
            self._plan_inforward_snapshots(scheduled_batch)
        if self.local_num_mamba_layers == 0:
            return
        requests = scheduled_batch.context_requests + scheduled_batch.generation_requests
        num_contexts = len(scheduled_batch.context_requests)
        self._setup_state_indices(requests, num_contexts=num_contexts)
        self._reset_context_mamba_slots(num_contexts)

    def _setup_state_indices(
        self,
        requests: List[LlmRequest],
        num_contexts: int = 0,
    ) -> None:
        if self.local_num_mamba_layers == 0:
            return
        generation_requests = requests[num_contexts:]
        old_state_values = [
            self._request_id_to_state_index.get(request.py_request_id, -1)
            for request in generation_requests
        ]
        n = len(requests)
        assert n <= self._host_state_indices.shape[0], (
            f"State-index batch size {n} exceeds max_batch_size {self._host_state_indices.shape[0]}"
        )
        if self._skips_steady_state_index_upload:
            if n > 0 and self._is_steady_state_index_batch(requests):
                return
        self._host_state_indices.zero_()
        if n > 0:
            for i, req in enumerate(requests):
                kv_cache = self.kv_cache_map.get(req.py_request_id)
                if kv_cache is None:
                    raise RuntimeError(f"Missing V2 KV cache for request {req.py_request_id}")
                base_index = kv_cache.get_ssm_block_base_index(self.ssm_layer_group_id)
                if base_index < 0:
                    raise RuntimeError(
                        f"Invalid SSM state block index {base_index} for "
                        f"request {req.py_request_id}"
                    )
                self._host_state_indices[i] = base_index

        self._state_indices_buffer.update(self._host_state_indices)
        is_dummy = [req.is_dummy for req in requests]
        self._refresh_dummy_request_mask(is_dummy)
        state_values = self._host_state_indices[:n].tolist()
        for req, value, dummy in zip(requests, state_values, is_dummy):
            self._request_id_to_state_index[req.py_request_id] = value
            self._request_id_to_is_dummy[req.py_request_id] = dummy

        new_state_values = [
            self._request_id_to_state_index[request.py_request_id]
            for request in generation_requests
        ]
        self._on_state_slots_relocated(old_state_values, new_state_values)

    def _on_state_slots_relocated(self, old_slots: list[int], new_slots: list[int]) -> None:
        """Allow model-owned scratch state to follow the updated slot mapping."""

    def _is_steady_state_index_batch(self, requests: List[LlmRequest]) -> bool:
        """Whether the last upload (same requests, SSM base blocks and dummy flags) still holds.

        Records the batch as the new reference otherwise.
        """
        base_indices = []
        for req in requests:
            kv_cache = self.kv_cache_map.get(req.py_request_id)
            if kv_cache is None:
                self._steady_state_index_key = None
                return False
            base_indices.append(kv_cache.get_ssm_block_base_index(self.ssm_layer_group_id))
        if min(base_indices) < 0:
            self._steady_state_index_key = None
            return False
        key = (
            tuple(req.py_request_id for req in requests),
            tuple(base_indices),
            tuple(req.is_dummy for req in requests),
        )
        # Request ids recur (warmup and CUDA-graph dummies are freed and re-added
        # under the same id), so the per-request maps, which free_resources()
        # prunes, are checked as well.
        if key == self._steady_state_index_key and all(
            self._request_id_to_state_index.get(req.py_request_id) == base
            and self._request_id_to_is_dummy.get(req.py_request_id) == req.is_dummy
            for req, base in zip(requests, base_indices)
        ):
            return True
        self._steady_state_index_key = key
        return False

    def _reset_context_mamba_slots(self, num_contexts: int) -> None:
        if num_contexts and self._speculative_state is not None:
            self._speculative_state.reset_slots(
                self.cuda_state_indices[:num_contexts].long(),
                self._host_state_indices[:num_contexts].tolist(),
            )

    def _refresh_dummy_request_mask(self, is_dummy: list[bool]) -> None:
        count = len(is_dummy)
        if count > self._dummy_request_mask_host.shape[0]:
            raise ValueError("Dummy-request batch exceeds state-index capacity")
        self._dummy_request_mask_host.zero_()
        if count:
            self._dummy_request_mask_host[:count].copy_(torch.tensor(is_dummy, dtype=torch.bool))
        self._dummy_request_mask_buffer.update(self._dummy_request_mask_host)

    @override
    def get_state_indices(
        self, request_ids: Optional[List[int]] = None, is_padding: Optional[List[bool]] = None
    ):
        if self.local_num_mamba_layers == 0:
            # Mamba metadata is still prepared on attention-only PP ranks,
            # but no local kernel consumes these indices. Return harmless
            # placeholders instead of consulting an intentionally empty map.
            if request_ids is not None:
                return [0] * len(request_ids)
            return self.cuda_state_indices
        if request_ids is not None:
            indices = [self._request_id_to_state_index[rid] for rid in request_ids]
            if is_padding is None:
                is_padding = [False] * len(request_ids)
            assert len(request_ids) == len(is_padding)
            is_dummy = [
                self._request_id_to_is_dummy.get(rid, False) or padding
                for rid, padding in zip(request_ids, is_padding)
            ]
            self._refresh_dummy_request_mask(is_dummy)
            return indices
        return self.cuda_state_indices

    @override
    def get_max_resource_count(self) -> int:
        return self.max_batch_size

    @override
    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        kv_cache = self.kv_cache_map.get(request.py_request_id)
        if kv_cache is not None and kv_cache.is_active:
            self.try_commit_blocks(request, kv_cache)
        self._request_id_to_state_index.pop(request.py_request_id, None)
        self._request_id_to_is_dummy.pop(request.py_request_id, None)
        self._branch_snapshot_points.pop(request.py_request_id, None)
        # A freed request ends any steady decode run.
        self._steady_state_index_key = None
        super().free_resources(request, pin_on_release)

    def update_mamba_states(
        self,
        attn_metadata: "AttentionMetadata",
        num_accepted_tokens: torch.Tensor,
        state_indices: Optional[torch.Tensor] = None,
        accepted_leaf_positions: Optional[torch.Tensor] = None,
    ):
        if self.local_num_mamba_layers == 0:
            return
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        num_accepted_draft_tokens = (
            num_accepted_tokens[num_contexts : num_contexts + num_gens] - 1
        ).to(torch.int32)
        # Dynamic tree selects a tree node rather than a linear draft depth.
        accepted_positions = (
            accepted_leaf_positions.to(torch.int32)
            if accepted_leaf_positions is not None
            else num_accepted_draft_tokens
        )
        if state_indices is None:
            state_indices = self.get_state_indices()
        state_indices_d = state_indices[num_contexts : num_contexts + num_gens].to(torch.int32)
        src_state_indices = self._generation_state_indices[:num_gens]
        is_dummy_request = self._dummy_request_mask[num_contexts : num_contexts + num_gens]
        if self._speculative_state is None:
            raise NotImplementedError("The model manager must provide speculative state")
        self._speculative_state.update(
            MambaAcceptanceBatch(
                attention_metadata=attn_metadata,
                num_contexts=num_contexts,
                num_generations=num_gens,
                num_accepted_tokens=num_accepted_tokens[num_contexts : num_contexts + num_gens],
                accepted_positions=accepted_positions,
                source_state_indices=src_state_indices,
                destination_state_indices=state_indices_d,
                is_dummy_request=is_dummy_request,
            ),
        )

    @override
    def try_commit_blocks(self, request: LlmRequest, kv_cache=None) -> None:
        should_block_reuse = (
            self.enable_block_reuse and not self.is_draft and not request.is_dummy_request
        )
        if not should_block_reuse:
            return

        if kv_cache is None:
            kv_cache = self.kv_cache_map.get(request.py_request_id)
        if kv_cache is None:
            return

        snapshot_points = request.expect_snapshot_points
        commit_limit = (
            min(max(snapshot_points), request.prompt_len) if snapshot_points else request.prompt_len
        )
        commit_end = min(request.context_current_position, commit_limit)
        if (
            request.context_current_position in request.expect_snapshot_points
            and commit_end > kv_cache.num_committed_tokens
        ):
            tokens = self._augment_tokens_for_block_reuse(
                self._reuse_token_source(request),
                request,
                start=kv_cache.num_committed_tokens,
                end=commit_end,
            )
            kv_cache.commit(tokens)
            if (
                self.local_num_mamba_layers > 0
                and self._branch_snapshot_points.get(request.py_request_id) == commit_end
            ):
                self._branch_snapshots_taken_total += 1
        if request.context_current_position >= commit_limit:
            self._mark_context_position_as_history(request, kv_cache)
        if request.context_remaining_length == 0:
            kv_cache.stop_committing()

    def _mark_context_position_as_history(self, request: LlmRequest, kv_cache) -> None:
        """Advance history without making later recurrent state reusable."""
        history_length = request.context_current_position
        if history_length <= kv_cache.history_length:
            return
        capacity = max(kv_cache.capacity, history_length)
        if not kv_cache.resize(capacity, history_length=history_length):
            raise ValueError(
                "Failed to resize history length of V2 Mamba cache for "
                f"request {request.py_request_id} to {history_length} tokens"
            )

    @override
    def update_context_resources(self, scheduled_batch: ScheduledRequests) -> None:
        for request in scheduled_batch.context_requests:
            kv_cache = self.kv_cache_map.get(request.py_request_id)
            if kv_cache is None or not kv_cache.is_active:
                continue

            should_block_reuse = (
                self.enable_block_reuse and not self.is_draft and not request.is_dummy_request
            )
            is_all_reusable = self.block_reuse_policy == BlockReusePolicy.ALL_REUSABLE
            is_snapshot_boundary = (
                request.context_current_position in request.expect_snapshot_points
            )
            has_pending_snapshot = any(
                point > request.context_current_position for point in request.expect_snapshot_points
            )
            # The FP4 MLA HP ring is positioned by the history length, so it
            # needs each completed chunk's watermark even while a later
            # recurrent-state snapshot is still pending.
            # commit_min_snapshot accepts a commit only if it starts or ends at
            # history_length. While a snapshot point is pending, history stays
            # at the committed end, so the next commit starts there. Once
            # history has passed it (chunks without a pending point, then a
            # waiter's point arrived, see request_prefix_snapshot), the next
            # commit can only end at history, so history follows the context.
            should_resize = (
                self._attention_cache_layout_policy is not None
                or not should_block_reuse
                or (
                    not is_all_reusable
                    and (
                        not has_pending_snapshot
                        or kv_cache.history_length > kv_cache.num_committed_tokens
                    )
                )
            )
            should_commit = (
                is_all_reusable or is_snapshot_boundary or request.context_remaining_length == 0
            )

            if should_resize and not kv_cache.resize(None, request.context_current_position):
                raise ValueError(
                    "Failed to resize history length of V2 Mamba cache for "
                    f"request {request.py_request_id} to "
                    f"{request.context_current_position} tokens at context "
                    "update"
                )
            if should_commit:
                self.try_commit_blocks(request, kv_cache)
            if request.context_remaining_length == 0:
                if self.conversation_manager is not None:
                    self.conversation_manager.save_drop_plan(request, kv_cache)
                kv_cache.enable_swa_scratch_reuse = False

    @override
    def prepare_expect_snapshot_points(self, requests: List[LlmRequest]) -> None:
        super().prepare_expect_snapshot_points(requests)
        for request in requests:
            self._align_fp4_mla_snapshot_points(request)
        block_reuse_config = self.kv_cache_config.block_reuse_config
        if self.enable_block_reuse and block_reuse_config.enable_branch_snapshot:
            for request in requests:
                self._apply_branch_snapshot_point(request)
        if self.enable_block_reuse and self._waiter_snapshot_points:
            for request in requests:
                self._merge_waiter_snapshot_points(request)

    @override
    def _can_share_inflight_prefix(self) -> bool:
        """Snapshots requested for waiting requests need no branch snapshot config."""
        return self.enable_block_reuse and self._can_publish_block_reuse

    @override
    def request_prefix_snapshot(self, owner: LlmRequest, num_tokens: int) -> bool:
        """Snapshot ``owner`` at ``num_tokens`` for a request waiting for that prefix.

        The recurrent state reaches the radix tree only at snapshot points, so
        ``owner`` always needs one there; its context chunk ends there too. The
        point may arrive after ``owner``'s history has passed its committed
        tokens; update_context_resources then advances history to each chunk
        end, so the commit at the point ends at history as commit_min_snapshot
        requires.
        """
        return self._add_waiter_snapshot_point(owner, num_tokens)

    def _align_fp4_mla_snapshot_points(self, request: LlmRequest) -> None:
        """Move recurrent-state snapshots onto full cache blocks.

        The FP4 MLA cache quantizes the latent cache in 16-token tiles and
        disables partial-block reuse, so a reused prefix always ends on a
        cache block. A snapshot inside a block would restore the recurrent
        state of a later token than the reused prefix ends at; one below the
        first block can never be reused. Snapshot points also end the
        context chunks, which keeps every non-final chunk tile-aligned.
        """
        if self._attention_cache_layout_policy is None:
            return
        block = self.tokens_per_block
        points = request.expect_snapshot_points
        aligned = sorted({point // block * block for point in points if point >= block})
        if aligned != sorted(set(points)) and not self._fp4_mla_snapshot_realignment_logged:
            self._fp4_mla_snapshot_realignment_logged = True
            logger.info(
                "FP4 MLA moves Mamba snapshot points down to "
                f"{block}-token cache blocks and drops points below the "
                f"first block (logged once; request {request.py_request_id}: "
                f"{sorted(set(points))} -> {aligned})."
            )
        request.expect_snapshot_points = aligned

    def _skip_branch_snapshot(self, reason: str) -> None:
        self._branch_snapshots_skipped_total[reason] = (
            self._branch_snapshots_skipped_total.get(reason, 0) + 1
        )

    @override
    def _record_branch_snapshot_point(
        self, req: LlmRequest, kv_cache: _KVCache, num_lookup_tokens: Optional[int]
    ) -> None:
        """Snapshot where this request leaves the reuse tree, for its siblings."""
        if not self.kv_cache_config.block_reuse_config.enable_branch_snapshot:
            return
        # Generation-only requests receive their prompt state from the context
        # worker and never prefill here, so they cannot take a snapshot.
        if req.is_dummy_request or req.is_disagg_generation_init_state:
            return
        if num_lookup_tokens is None:
            self._skip_branch_snapshot("no_fresh_match")
            # A fresh lookup can record a point before resume() fails. On the
            # next attempt, reapply that point against the cache's authoritative
            # reuse depth: the request cursor is not rewound until preparation
            # succeeds and may still describe a deeper, abandoned attempt.
            if req.py_request_id in self._branch_snapshot_points:
                self._apply_branch_snapshot_point(
                    req, context_current_position=kv_cache.num_committed_tokens
                )
            return

        diagnostics = cast(_PrefixReuseDiagnostics, kv_cache)
        divergence = diagnostics._get_num_reusable_tokens_before_pruning()
        reused = kv_cache.num_committed_tokens
        if self.local_num_mamba_layers > 0:
            hybrid_depth = diagnostics._get_num_reusable_tokens_before_hybrid_pruning()
            # Loss attributable to a missing recurrent snapshot, versus loss
            # attributable to attention pages having been evicted. These call
            # for different fixes, so they are counted apart.
            self._snapshot_pruned_tokens_total += max(0, hybrid_depth - reused)
            self._page_pruned_tokens_total += max(0, divergence - hybrid_depth)

        # The whole lookup range matched, so there is no fork here.
        if divergence >= num_lookup_tokens:
            self._skip_branch_snapshot("no_divergence")
            return
        # Align down: tokens past a block boundary come from a partial-match
        # child rather than a confirmed shared prefix, and an unaligned commit
        # takes the partial-block snapshot path.
        point = (divergence // self.tokens_per_block) * self.tokens_per_block
        if point == 0:
            self._skip_branch_snapshot("aligned_to_zero")
            return
        # Reuse already reached the fork, so a snapshot there exists.
        if point <= reused:
            self._skip_branch_snapshot("already_reused")
            return
        # A branch point at or beyond the prompt end cannot improve sibling
        # reuse.
        if point >= req.prompt_len:
            self._skip_branch_snapshot("at_prompt_end")
            return

        self._branch_snapshot_points[req.py_request_id] = point
        self._apply_branch_snapshot_point(req, context_current_position=reused)

    def _apply_branch_snapshot_point(
        self,
        request: LlmRequest,
        context_current_position: Optional[int] = None,
    ) -> None:
        points = set(request.expect_snapshot_points)
        # Keep the flag usable without any configured snapshot placement. If a
        # placement is configured, preserve it: adding prompt_len can replace a
        # nearby recurrent checkpoint in the same radix-tree block because a
        # block stores one recurrent page per life cycle.
        if not points:
            points.add(request.prompt_len)
        point = self._branch_snapshot_points.get(request.py_request_id)
        current_position = (
            request.context_current_position
            if context_current_position is None
            else context_current_position
        )
        if point is not None and point > current_position:
            points.add(point)
        request.expect_snapshot_points = sorted(points)
        self._align_fp4_mla_snapshot_points(request)

    def _format_branch_snapshot_counters(self) -> str:
        """Attribute reuse loss between missing snapshots and evicted pages.

        A flat cache hit rate means something different depending on whether
        branch points were never found or were found and not reused, so the
        skip reasons are reported alongside the totals.
        """
        skipped = self._branch_snapshots_skipped_total
        return (
            f"snapshot_pruned_tokens={self._snapshot_pruned_tokens_total} "
            f"page_pruned_tokens={self._page_pruned_tokens_total} "
            f"branch_snapshots_taken={self._branch_snapshots_taken_total} "
            f"branch_snapshots_skipped={sum(skipped.values())} "
            f"branch_snapshots_skipped_by_reason="
            f"{dict(sorted(skipped.items()))}"
        )

    @override
    def get_iteration_stats(self) -> Optional[KVCacheV2IterationStatsReport]:
        """Log recurrent-cache movement across V2 storage tiers."""
        report = super().get_iteration_stats()
        if report is None:
            return None

        pool_group_ids = sorted(
            {
                pool_group_id
                for pool_group_id, _, kind in self._stats_life_cycle_metadata().values()
                if kind == "ssm"
            }
        )
        if not pool_group_ids:
            return report
        pool_group_reports = [
            report.by_pool_group[pool_group_id] for pool_group_id in pool_group_ids
        ]

        stats = [pool_group_report.stats for pool_group_report in pool_group_reports]
        evicted_blocks = sum(stat.iter_offload_blocks for stat in stats)
        evicted_bytes = sum(stat.iter_offload_bytes for stat in stats)
        onboarded_blocks = sum(stat.iter_onboard_blocks for stat in stats)
        onboarded_bytes = sum(stat.iter_onboard_bytes for stat in stats)
        dropped_blocks = sum(stat.iter_host_dropped_blocks for stat in stats)
        dropped_bytes = sum(stat.iter_host_dropped_bytes for stat in stats)
        has_movement = bool(evicted_blocks or onboarded_blocks or dropped_blocks)
        if has_movement:
            self._recurrent_evicted_blocks_total += evicted_blocks
            self._recurrent_onboarded_blocks_total += onboarded_blocks
            self._recurrent_dropped_blocks_total += dropped_blocks
        if self.mapping.rank == 0 and (has_movement or not self._recurrent_status_logged):
            logger.debug(
                f"[MambaHybridCacheManagerV2] recurrent cache status "
                f"rank={self.mapping.rank} pool_group_ids={pool_group_ids} "
                f"evicted_recurrent_blocks={evicted_blocks} "
                f"evicted_recurrent_bytes={evicted_bytes} "
                f"onboarded_recurrent_blocks={onboarded_blocks} "
                f"onboarded_recurrent_bytes={onboarded_bytes} "
                f"dropped_recurrent_blocks={dropped_blocks} "
                f"dropped_recurrent_bytes={dropped_bytes} "
                f"total_evicted_recurrent_blocks="
                f"{self._recurrent_evicted_blocks_total} "
                f"total_onboarded_recurrent_blocks="
                f"{self._recurrent_onboarded_blocks_total} "
                f"total_dropped_recurrent_blocks="
                f"{self._recurrent_dropped_blocks_total} "
                f"gpu_used_recurrent_blocks="
                f"{sum(stat.primary_used_num_blocks for stat in stats)} "
                f"gpu_free_recurrent_blocks="
                f"{sum(stat.primary_free_num_blocks for stat in stats)} "
                f"gpu_evictable_recurrent_blocks="
                f"{sum(stat.primary_evictable_num_blocks for stat in stats)} "
                f"host_used_recurrent_blocks="
                f"{sum(stat.secondary_used_num_blocks for stat in stats)} "
                f"host_free_recurrent_blocks="
                f"{sum(stat.secondary_free_num_blocks for stat in stats)} "
                f"{self._format_branch_snapshot_counters()}"
            )
            self._recurrent_status_logged = True
        return report

    @override
    def _iter_cache_buffers_for_invalid_check(self) -> Iterable[tuple[int, torch.Tensor]]:
        if self._attention_cache_layout_policy is not None:
            # FP4 MLA roles have role-specific dtypes and page geometries, so
            # scan each physical pool once through a page-major view instead
            # of a logical per-layer KV view.
            for buffer in self._iter_fp4_mla_physical_pool_views():
                yield -1, buffer
        else:
            for global_layer_id, local_layer_id in self.layer_offsets.items():
                if self._is_local_mamba_layer(local_layer_id):
                    continue
                # A layer group is a lifecycle, not a physical memory pool.
                # Differently sized attention buffers can share one
                # lifecycle, so scan every attention layer in this
                # diagnostic path.
                yield global_layer_id, KVCacheManagerV2.get_buffers(self, global_layer_id)

        for (layer_id, role), buffer in self._recurrent_buffers.items():
            if buffer is None:
                raise RuntimeError(
                    f"Recurrent buffer view was not bound: layer={layer_id}, role={role}"
                )
            # Recurrent roles have no attention guard page.
            yield -1, buffer

    @override
    def shutdown(self):
        if self._speculative_state is not None:
            self._speculative_state.shutdown()
        self._recurrent_buffers.clear()
        self.all_ssm_states = []
        self.all_conv_states = []
        self.inforward_snapshot_rows = 0
        self.inforward_snapshot_plan = {}
        self._inforward_stage_ssm = None
        self._inforward_stage_conv = None
        self._dummy_request_mask_buffer = None
        self._dummy_request_mask_host = None
        self._generation_state_indices = None
        self._branch_snapshot_points.clear()
        super().shutdown()


__all__ = ["InforwardSnapshotEntry", "MambaHybridCacheManagerV2"]

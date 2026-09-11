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

import math
from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import torch

from tensorrt_llm._torch.modules.mamba.cache_manager import (
    build_default_mamba_state_update_strategy,
    mamba_seed_rank_offset,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
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
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    DEFAULT_BEAM_INDEX,
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
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig as KVCacheManagerConfigPy

from .common import (
    MambaAuxCacheExtension,
    MambaCacheBuildContext,
    MambaCacheFeatures,
    MambaHybridCacheManager,
    MambaRole,
    MambaStateUpdateBatch,
    MambaStateUpdateStrategy,
    _estimate_mamba_hybrid_cache_cost,
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


class MambaHybridCacheManagerV2(KVCacheManagerV2, MambaHybridCacheManager):
    """Hybrid Mamba cache manager backed by KVCacheManagerV2.

    Attention KV pages and Mamba recurrent-state pages are both owned by the
    Python V2 cache manager.  Mamba layers are represented as V2 SSM layers,
    while this wrapper exposes the state tensors and slot indices expected by
    the PyTorch Mamba kernels.
    """

    _supports_additional_snapshot_offsets = True

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
        use_replay_state_update: bool = False,
        mamba_ssm_stochastic_rounding: bool = False,
        conv_state_layout: Literal["x_b_c", "q_k_v"] = "x_b_c",
        state_update_strategy: MambaStateUpdateStrategy | None = None,
        aux_cache_extensions: Sequence[MambaAuxCacheExtension] = (),
        **kwargs,
    ) -> None:
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
        if state_update_strategy is None:
            state_update_strategy = build_default_mamba_state_update_strategy(
                spec_config=spec_config,
                use_replay=use_replay_state_update,
            )
        elif use_replay_state_update and not state_update_strategy.uses_replay:
            raise ValueError(
                "use_replay_state_update=True conflicts with the supplied "
                f"state-update strategy {state_update_strategy.key!r}"
            )
        self.spec_config = spec_config
        self._mamba_ssm_stochastic_rounding = mamba_ssm_stochastic_rounding
        self._seed_rank_offset = mamba_seed_rank_offset(mapping)
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
            if state_update_strategy.uses_replay:
                assert mamba_n_groups % tp_size == 0, (
                    "replay state update requires mamba_n_groups divisible by tp_size"
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

        self._cache_features = MambaCacheFeatures(
            state_update=state_update_strategy,
            auxiliary=tuple(aux_cache_extensions),
        )
        self._cache_features.validate_unique((MambaRole.SSM_STATE, MambaRole.CONV_STATE))
        self._aux_cache_extensions = {
            extension.key: extension for extension in self._cache_features.auxiliary
        }
        self._cache_feature_context = MambaCacheBuildContext(
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
            backend="v2",
            stochastic_rounding=mamba_ssm_stochastic_rounding,
            seed_rank_offset=self._seed_rank_offset,
            conv_section_dims=tuple(self.conv_section_dims),
            conv_state_layout=conv_state_layout,
        )
        state_update_strategy.validate(self._cache_feature_context)
        for extension in self._cache_features.auxiliary:
            extension.validate(self._cache_feature_context)

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

        self.mamba_layer_offsets = {
            layer_id: offset for offset, layer_id in enumerate(self.mamba_pp_layers)
        }
        self._request_id_to_state_index = {}
        self._request_id_to_is_dummy = {}

        state_index_capacity = self.max_batch_size + self._num_reserved_dummy_slots
        self.cuda_state_indices = torch.zeros(
            [state_index_capacity], dtype=torch.int32, device="cuda"
        )
        self._host_state_indices = torch.zeros(
            [state_index_capacity], dtype=torch.int32, pin_memory=prefer_pinned()
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
            self._setup_states()
            self._cache_feature_context = self._cache_feature_context.with_slot_capacity(
                num_ssm_slots
            )
            for extension in self._cache_features.auxiliary:
                extension.bind(self._cache_feature_context, self)
            state_update_strategy.bind(self._cache_feature_context, self)
        else:
            self.ssm_layer_group_id = None
            self._ssm_page_index_scale = 1
            self.all_ssm_states = []
            self.all_conv_states = []
            self._cache_feature_context = self._cache_feature_context.with_slot_capacity(0)
            state_update_strategy.bind(self._cache_feature_context, self)

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
    ) -> Optional[_KVCache]:
        kv_cache = super()._create_kv_cache(
            request_id,
            lora_task_id,
            input_tokens,
            cache_salt=cache_salt,
            is_dummy=is_dummy,
            enable_request_stats=enable_request_stats,
            expected_prompt_length=expected_prompt_length,
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

    def _skip_branch_snapshot(self, reason: str) -> None:
        self._branch_snapshots_skipped_total[reason] = (
            self._branch_snapshots_skipped_total.get(reason, 0) + 1
        )

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

    def _record_branch_snapshot_point(
        self, req: LlmRequest, kv_cache: _KVCache, num_lookup_tokens: Optional[int]
    ) -> None:
        """Snapshot where this request leaves the reuse tree, for its siblings."""
        if not self.kv_cache_config.mamba_state_config.enable_branch_snapshot:
            return
        if req.is_dummy_request:
            return
        if num_lookup_tokens is None:
            self._skip_branch_snapshot("no_fresh_match")
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
        if point <= req.context_current_position:
            self._skip_branch_snapshot("already_reused")
            return
        # A branch point at or beyond the prompt end cannot improve sibling
        # reuse.
        if point >= req.prompt_len:
            self._skip_branch_snapshot("at_prompt_end")
            return

        self._branch_snapshot_points[req.py_request_id] = point
        self._apply_branch_snapshot_point(req)

    def prepare_expect_snapshot_points(self, requests: List[LlmRequest]) -> None:
        super().prepare_expect_snapshot_points(requests)
        if (
            not self.enable_block_reuse
            or not self.kv_cache_config.mamba_state_config.enable_branch_snapshot
        ):
            return
        for request in requests:
            self._apply_branch_snapshot_point(request)

    def _apply_branch_snapshot_point(self, request: LlmRequest) -> None:
        points = set(request.expect_snapshot_points)
        # Keep the flag usable without any configured snapshot placement. If a
        # placement is configured, preserve it: adding prompt_len can replace a
        # nearby recurrent checkpoint in the same radix-tree block because a
        # block stores one recurrent page per life cycle.
        if not points:
            points.add(request.prompt_len)
        point = self._branch_snapshot_points.get(request.py_request_id)
        if point is not None and point > request.context_current_position:
            points.add(point)
        request.expect_snapshot_points = sorted(points)

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

    @staticmethod
    def get_cache_size_per_token(
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
        return _estimate_mamba_hybrid_cache_cost(
            model_config,
            mapping,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            num_reserved_dummy_slots=num_reserved_dummy_slots,
            include_explicit_snapshots=True,
            cap_partial_attention_snapshots=True,
            **kwargs,
        )

    def _is_local_mamba_layer(self, local_layer_idx: int) -> bool:
        return self._mamba_layer_mask[self.pp_layers[local_layer_idx]]

    def _get_pool_roles(self, pool_id: int) -> Tuple[DataRole, Optional[DataRole]]:
        layer_id = int(self.impl.layer_grouping[pool_id][0])
        if self._is_local_mamba_layer(layer_id):
            return MambaRole.SSM_STATE, None
        return super()._get_pool_roles(pool_id)

    def _max_resident_sequences(self) -> int:
        return self.max_batch_size * self.mapping.pp_size

    def _mamba_state_bytes_per_slot(self) -> int:
        base_bytes = self.local_num_mamba_layers * (self.ssm_bytes + self.conv_bytes)
        state_update_bytes = sum(
            self._cache_features.state_update.bytes_per_slot(self._cache_feature_context, layer_id)
            for layer_id in self.mamba_pp_layers
        )
        extension_bytes = sum(
            extension.bytes_per_slot(self._cache_feature_context, layer_id)
            for extension in self._cache_features.auxiliary
            for layer_id in self.mamba_pp_layers
        )
        return base_bytes + state_update_bytes + extension_bytes

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
        fixed_rules, _ = _mamba_snapshot_rule_counts(
            kv_cache_config,
            capacity,
            self.tokens_per_block,
        )
        # Additional snapshots are stable boundaries that must remain alive.
        # Periodic snapshots are evictable cache entries and therefore do not
        # increase the guaranteed state count represented by BatchDesc.
        return 1 + fixed_rules

    def _typical_request_descs(
        self,
        capacity: int,
        kv_cache_config: KvCacheConfig,
    ) -> List[KVCacheDesc]:
        """Model one request with one descriptor per live SSM state."""
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
            "workload's average total sequence length for an accurate KV/SSM "
            "pool ratio."
        )
        return fallback_capacity

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

    def _build_cache_config(self, config: KVCacheManagerConfigPy) -> KVCacheManagerConfigPy:
        kv_cache_config = self.kv_cache_config
        cache_tiers = config.cache_tiers
        gpu_quota = cache_tiers[0].quota
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
        for local_layer_idx, global_layer_idx in enumerate(self.pp_layers):
            if self._mamba_layer_mask[global_layer_idx]:
                layer_id = LayerId(local_layer_idx)
                buffers = [
                    BufferConfig(role=MambaRole.SSM_STATE, size=self.ssm_bytes),
                    BufferConfig(role=MambaRole.CONV_STATE, size=self.conv_bytes),
                ]
                buffers.extend(
                    self._cache_features.state_update.buffer_configs(
                        self._cache_feature_context,
                        global_layer_idx,
                    )
                )
                for extension in self._cache_features.auxiliary:
                    buffers.extend(
                        extension.buffer_configs(
                            self._cache_feature_context,
                            global_layer_idx,
                        )
                    )
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
        return replace(
            config,
            layers=layers,
            typical_step=typical_step,
            constraints=constraints,
            # SSM lifecycles require minimum-snapshot commit semantics. The
            # flag is harmless when reuse is disabled because no commits are
            # attempted, while the runtime config still needs the invariant.
            commit_min_snapshot=True,
        )

    def _get_state_buffer(
        self, local_layer_idx: int, role, dtype: torch.dtype, state_shape: List[int]
    ) -> torch.Tensor:
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
        return raw.as_strided(
            [num_slots] + state_shape,
            [raw.stride(0) * page_index_scale] + list(raw.stride()[1:]),
        )

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

    def _attention_cache_bytes_per_token(self) -> int:
        # Mamba layers have zero KV heads, so the generic calculation naturally
        # returns only bytes owned by local attention layers.
        return super().get_cache_bytes_per_token()

    def get_cache_bytes_per_token(self) -> int:
        cache_bytes = self._attention_cache_bytes_per_token()

        interval = self.kv_cache_config.mamba_state_config.periodic_snapshot_interval
        if self.kv_cache_config.enable_block_reuse and interval is not None and interval > 0:
            cache_bytes += self._mamba_state_bytes_per_slot() // interval
        if cache_bytes == 0 and self.local_num_mamba_layers > 0:
            cache_bytes = self._mamba_state_bytes_per_slot()
        return max(1, cache_bytes)

    def get_num_free_blocks(self) -> int:
        assert len(self.kv_cache_map) == 0, (
            "get_num_free_blocks is only used when the kv cache manager is empty"
        )
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
    def blocks_in_primary_pool(self) -> int:
        for local_layer_idx in range(self.num_local_layers):
            if self._is_local_mamba_layer(local_layer_idx):
                continue
            return self.impl.get_page_index_upper_bound(LayerId(local_layer_idx), Role.KEY)
        return 0

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> Optional[torch.Tensor]:
        local_layer_idx = self.layer_offsets[layer_idx]
        if self._is_local_mamba_layer(local_layer_idx):
            return None
        return super().get_buffers(layer_idx, kv_layout)

    def _iter_cache_buffers_for_invalid_check(self) -> Iterable[torch.Tensor]:
        for global_layer_id, local_layer_id in self.layer_offsets.items():
            if self._is_local_mamba_layer(local_layer_id):
                continue
            # A layer group is a lifecycle, not a physical memory pool.
            # Differently sized attention buffers can share one lifecycle,
            # so scan every attention layer in this diagnostic path.
            yield KVCacheManagerV2.get_buffers(self, global_layer_id)

        yield from self.all_ssm_states
        yield from self.all_conv_states
        yield from self._cache_features.state_update.iter_buffers()
        for extension in self._cache_features.auxiliary:
            yield from extension.iter_buffers()

    def get_aux_cache_extension(self, key: str) -> MambaAuxCacheExtension | None:
        """Return an installed lifecycle-coupled auxiliary extension."""
        return self._aux_cache_extensions.get(key)

    def on_state_transfer_complete(self, request_ids: list[int]) -> None:
        """Notify cache features that base recurrent state is now available."""
        if self._cache_features.state_update is not None:
            self._cache_features.state_update.on_state_transfer_complete(request_ids, self)
        for extension in self._cache_features.auxiliary:
            extension.on_state_transfer_complete(request_ids, self)

    def add_dummy_requests(
        self,
        request_ids: List[int],
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
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
            prepare_resource=prepare_resource,
            max_num_draft_tokens=max_num_draft_tokens,
            kv_reserve_draft_tokens=kv_reserve_draft_tokens,
            use_mrope=use_mrope,
            max_beam_width=max_beam_width,
            encoder_output_lens=encoder_output_lens,
            draft_kv_cache_manager=draft_kv_cache_manager,
            capture_sampling_params=capture_sampling_params,
        )
        if requests and prepare_resource:
            self._setup_state_indices(
                requests,
                num_contexts=0 if is_gen else len(requests),
            )
            if not is_gen:
                self._reset_context_mamba_slots(len(requests))
        return requests

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        kv_cache = self.kv_cache_map.get(request.py_request_id)
        if kv_cache is not None and kv_cache.is_active:
            self.try_commit_blocks(request, kv_cache)
        self._request_id_to_state_index.pop(request.py_request_id, None)
        self._request_id_to_is_dummy.pop(request.py_request_id, None)
        self._branch_snapshot_points.pop(request.py_request_id, None)
        super().free_resources(request, pin_on_release)

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        super().prepare_resources(scheduled_batch)
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

        self.cuda_state_indices.copy_(self._host_state_indices, non_blocking=True)
        is_dummy = [req.is_dummy for req in requests]
        self._cache_features.state_update.refresh_dummy_request_mask(is_dummy)
        state_values = self._host_state_indices[:n].tolist()
        for req, value, dummy in zip(requests, state_values, is_dummy):
            self._request_id_to_state_index[req.py_request_id] = value
            self._request_id_to_is_dummy[req.py_request_id] = dummy

        new_state_values = [
            self._request_id_to_state_index[request.py_request_id]
            for request in generation_requests
        ]
        strategy = self._cache_features.state_update
        strategy.relocate_slots(old_state_values, new_state_values)
        fresh_slots = [new for old, new in zip(old_state_values, new_state_values) if old < 0]
        if fresh_slots:
            slots = torch.tensor(
                fresh_slots,
                dtype=torch.long,
                device=self.cuda_state_indices.device,
            )
            strategy.reset_slots(slots, fresh_slots)

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
            self._cache_features.state_update.refresh_dummy_request_mask(is_dummy)
            return indices
        return self.cuda_state_indices

    def get_max_resource_count(self) -> int:
        return self.max_batch_size

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
        strategy = self._cache_features.state_update
        src_state_indices = strategy.source_state_indices(num_gens)
        is_dummy_request = strategy.dummy_request_mask(num_contexts, num_contexts + num_gens)
        strategy.update(
            MambaStateUpdateBatch(
                attention_metadata=attn_metadata,
                num_contexts=num_contexts,
                num_generations=num_gens,
                num_accepted_tokens=num_accepted_tokens[num_contexts : num_contexts + num_gens],
                accepted_positions=accepted_positions,
                source_state_indices=src_state_indices,
                destination_state_indices=state_indices_d,
                is_dummy_request=is_dummy_request,
            ),
            self,
        )

    def update_resources(
        self,
        scheduled_batch: ScheduledRequests,
        attn_metadata: "AttentionMetadata" = None,
        kv_cache_dtype_byte_size: float | None = None,
    ) -> None:
        super().update_resources(
            scheduled_batch,
            attn_metadata,
            kv_cache_dtype_byte_size,
        )
        if self.local_num_mamba_layers > 0:
            self._cache_features.state_update.on_resources_updated(
                scheduled_batch,
                self,
            )

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
                request.get_tokens(DEFAULT_BEAM_INDEX),
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
            should_resize = not should_block_reuse or (
                not is_all_reusable and not has_pending_snapshot
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

    def shutdown(self):
        if self._cache_features.state_update is not None:
            self._cache_features.state_update.shutdown(self)
        self.all_ssm_states = []
        self.all_conv_states = []
        for extension in self._cache_features.auxiliary:
            extension.shutdown()
        self._aux_cache_extensions.clear()
        self._branch_snapshot_points.clear()
        super().shutdown()

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
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, List, Literal, Optional, Protocol, Tuple

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override
import triton
import triton.language as tl

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager,
    KVCacheManager,
    get_pp_layers,
)
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig


GB = 1 << 30


@dataclass(frozen=True, kw_only=True)
class MambaLayerCache:
    """Persistent recurrent-state views for one local layer."""

    conv: torch.Tensor
    temporal: torch.Tensor

    def at_layer_idx(self, layer: int) -> "MambaLayerCache":
        """Return a per-layer view while preserving shared slot metadata."""
        values = {}
        for cache_field in fields(self):
            value = getattr(self, cache_field.name)
            values[cache_field.name] = (
                value
                if value is None or cache_field.metadata.get("slot_shared", False)
                else value[layer]
            )
        return type(self)(**values)


@dataclass(frozen=True, kw_only=True)
class MambaStateLayout:
    """Resolved recurrent geometry; slot capacity is filled after pool allocation."""

    layer_mask: tuple[bool, ...]
    pp_layers: tuple[int, ...]
    mamba_pp_layers: tuple[int, ...]
    mapping: "Mapping"
    conv_state_shape: tuple[int, ...]
    ssm_state_shape: tuple[int, ...]
    conv_state_dtype: torch.dtype
    ssm_state_dtype: torch.dtype
    n_groups_per_rank: int
    max_batch_size: int
    state_index_capacity: int
    slot_capacity: int | None
    spec_config: object | None
    conv_section_dims: tuple[int, ...] = field(default_factory=tuple)
    conv_state_layout: Literal["x_b_c", "q_k_v"] = "x_b_c"

    def with_slot_capacity(self, slot_capacity: int) -> "MambaStateLayout":
        """Return the post-allocation context for tensor-view binding."""
        return replace(self, slot_capacity=slot_capacity)


@dataclass(frozen=True, kw_only=True)
class MambaAcceptanceBatch:
    """Accepted-token inputs for model-specific recurrent updates."""

    attention_metadata: object
    num_contexts: int
    num_generations: int
    num_accepted_tokens: torch.Tensor
    accepted_positions: torch.Tensor
    source_state_indices: torch.Tensor
    destination_state_indices: torch.Tensor
    is_dummy_request: torch.Tensor | None


class MambaState(Protocol):
    """Model-selected recurrent state; persistent views remain pool-owned."""

    @property
    def intermediate_indices(self) -> torch.Tensor | None: ...

    @property
    def intermediate_ssm(self) -> torch.Tensor | None: ...

    @property
    def intermediate_conv(self) -> torch.Tensor | None: ...

    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None: ...

    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None: ...

    def update(self, batch: MambaAcceptanceBatch) -> None: ...

    def shutdown(self) -> None: ...

    def make_layer_cache(
        self, layer_offset: int, conv: torch.Tensor, temporal: torch.Tensor
    ) -> MambaLayerCache: ...


class BaseMambaCacheManager(ABC):
    """Abstract interface for accessing Mamba/recurrent state caches."""

    @abstractmethod
    def get_state_indices(self, *args, **kwargs) -> torch.Tensor:
        """Return slot indices of each request with shape ``[max_batch_size]``."""

    @abstractmethod
    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        """Return conv states with shape ``[slot_size, conv_dim, d_conv - 1]``."""

    @abstractmethod
    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        """Return SSM states with shape ``[slot_size, heads, head_dim, d_state]``."""

    @abstractmethod
    def is_speculative(self) -> bool:
        """Whether speculative state is allocated."""

    @abstractmethod
    def mamba_layer_cache(self, layer_idx: int) -> MambaLayerCache | None:
        """Return the typed recurrent cache payload for one layer."""

    def on_state_transfer_complete(self, request_ids: list[int]) -> None:
        """Notify a model manager after a disaggregated transfer."""


class _PrefixReuseDiagnostics(Protocol):
    def _get_num_reusable_tokens_before_hybrid_pruning(self) -> int: ...

    def _get_num_reusable_tokens_before_pruning(self) -> int: ...


def _get_num_cuda_graph_padding_dummy_slots(
    spec_config: Optional["DecodingBaseConfig"],
    max_batch_size: int,
) -> int:
    """Return the number of persistent CUDA-graph padding dummy IDs.

    This is computed before ``ModelEngine`` exists and covers draft lengths
    reachable at every batch size, including the zero-length acceptance-rate
    fallback. ``ModelEngine._compute_dynamic_draft_len_mapping`` is created
    later and covers only configured CUDA-graph batch sizes, so it cannot size
    this persistent ID set.
    """
    if spec_config is None:
        return 1

    draft_len_schedule = getattr(spec_config, "draft_len_schedule", None)
    spec_dec_mode = getattr(spec_config, "spec_dec_mode", None)
    supports_dynamic_draft_len = (
        spec_dec_mode is not None
        and hasattr(spec_dec_mode, "support_dynamic_draft_len")
        and spec_dec_mode.support_dynamic_draft_len()
    )
    if draft_len_schedule and supports_dynamic_draft_len:
        runtime_draft_lengths = set()
        first_uncovered_batch_size = 1
        for batch_size_threshold, draft_len in draft_len_schedule.items():
            if first_uncovered_batch_size > max_batch_size:
                break
            if batch_size_threshold >= first_uncovered_batch_size:
                runtime_draft_lengths.add(draft_len)
                first_uncovered_batch_size = batch_size_threshold + 1
        if first_uncovered_batch_size <= max_batch_size:
            runtime_draft_lengths.add(0)
    else:
        max_draft_len = getattr(spec_config, "max_draft_len", 0) or 0
        max_total_draft_tokens = getattr(spec_config, "max_total_draft_tokens", 0) or 0
        is_linear_tree = getattr(
            spec_config,
            "is_linear_tree",
            max_draft_len == max_total_draft_tokens,
        )
        static_draft_len = max_draft_len if is_linear_tree else max_total_draft_tokens
        runtime_draft_lengths = {static_draft_len or 0}

    if (getattr(spec_config, "acceptance_rate_window_size", 0) or 0) > 0 and (
        getattr(spec_config, "acceptance_rate_threshold", 0) or 0
    ) > 0:
        runtime_draft_lengths.add(0)
    return len(runtime_draft_lengths)


class MambaRole:
    """V2 buffer roles owned only by the hybrid Mamba manager."""

    SSM_STATE = DataRole("ssm_state")
    CONV_STATE = DataRole("conv_state")


def _mamba_effective_tp_size(mapping: Mapping) -> int:
    """TP degree for sizing per-rank recurrent-state pools.

    Attention-DP replicates the state and takes precedence; helix
    repurposes CP ranks as plain TP for recurrent-state layers.
    """
    if mapping.enable_attention_dp:
        return 1
    if mapping.has_cp_helix():
        return mapping.tp_size * mapping.cp_size
    return mapping.tp_size


def get_tensor_size_bytes(tensor):
    """Calculate tensor size in bytes."""
    if isinstance(tensor, torch.Tensor):
        return tensor.element_size() * tensor.nelement()
    elif isinstance(tensor, list):
        return sum(get_tensor_size_bytes(t) for t in tensor)
    return 0


def use_py_mamba_cache_manager() -> bool:
    """Check if PythonMambaCacheManager should be forced (agg mode override).

    Returns True if TRTLLM_USE_PY_MAMBA='1' is set, False otherwise.

    Agg-mode-only override: forces the V1-route MixedMambaHybridCacheManager
    with PythonMambaCacheManager inside instead of the configured manager.
    Disagg mode is unaffected — its compatibility routing selects Mixed or
    Cpp based on the transceiver configuration.
    """
    return os.environ.get("TRTLLM_USE_PY_MAMBA", "0") == "1"


class MambaHybridCacheManager(BaseResourceManager, BaseMambaCacheManager):
    """Common recurrent views and snapshot planning, independent of algorithms."""

    _supports_additional_snapshot_offsets = False

    def prepare_expect_snapshot_points(self, requests: List[LlmRequest]) -> None:
        """Set reusable Mamba snapshot boundaries before scheduling."""
        if not self.enable_block_reuse:
            for request in requests:
                request.expect_snapshot_points = []
            return

        state_config = self.kv_cache_config.mamba_state_config
        interval = state_config.periodic_snapshot_interval
        for request in requests:
            snapshot_points = set()
            if interval is not None and interval > 0:
                snapshot_points.update(range(interval, request.prompt_len + 1, interval))
            if self._supports_additional_snapshot_offsets:
                for offset in state_config.additional_snapshot_offsets_from_start:
                    if offset <= request.prompt_len:
                        snapshot_points.add(offset)
                for offset in state_config.additional_snapshot_offsets_from_end:
                    point = request.prompt_len - offset
                    if point > 0:
                        snapshot_points.add(point)
            request.expect_snapshot_points = sorted(snapshot_points)

    @override
    def is_speculative(self) -> bool:
        return self.spec_config is not None

    @override
    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        return self.all_ssm_states[self.mamba_layer_offsets[layer_idx]]

    @override
    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        return self.all_conv_states[self.mamba_layer_offsets[layer_idx]]


def _get_mamba_hybrid_pool_size(max_batch_size: int, mapping: Mapping) -> int:
    """Return the internal Mamba state pool size for MixedMambaHybridCacheManager."""
    pool_size = max_batch_size
    # One permanent slot is shared by every CUDA-graph padding sentinel.
    pool_size += 1
    if mapping.enable_attention_dp:
        # Attention-DP can insert a transient dummy request on an otherwise
        # idle rank. Keep this headroom internal so scheduler-visible
        # max_batch_size still limits real requests.
        pool_size += 1
    return pool_size


@triton.jit
def _promote_mamba_state_kernel(
    src_ptr,
    dst_ptr,
    src_idx_ptr,
    accepted_position_source_ptr,
    num_accepted_tokens_ptr,
    blk_ptr,
    replay_pnat_ptr,
    replay_cache_buf_idx_ptr,
    dummy_request_mask_ptr,
    num_gens,
    count,
    src_s_layer,
    src_s_row,
    src_s_step,
    dst_s_layer,
    dst_s_block,
    POSITION_SOURCE_IS_TOKEN_COUNT: tl.constexpr,
    REPLAY_STEP_WIDTH: tl.constexpr,
    REPLAY_HISTORY_SIZE: tl.constexpr,
    UPDATE_REPLAY_BOOKKEEPING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program copies a BLOCK-sized tile of the contiguous inner state for a
    # single (layer, gen) pair. grid = (num_layers * num_gens, ceil(count/BLOCK)).
    pair = tl.program_id(0)
    tile = tl.program_id(1)
    layer = (pair // num_gens).to(tl.int64)
    g = pair % num_gens
    row = tl.load(src_idx_ptr + g).to(tl.int64)
    accepted_position = tl.load(accepted_position_source_ptr + g)
    if POSITION_SOURCE_IS_TOKEN_COUNT:
        accepted_position -= 1
    acc = accepted_position.to(tl.int64)
    blk = tl.load(blk_ptr + g).to(tl.int64)

    # The replay metadata update is batch-scoped, so layer 0's first copy tile
    # owns it. This folds PNAT and active-buffer maintenance into the existing
    # accepted-convolution-state promotion launch.
    if UPDATE_REPLAY_BOOKKEEPING:
        if (layer == 0) & (tile == 0):
            num_accepted_tokens = tl.load(num_accepted_tokens_ptr + g)
            is_dummy_request = tl.load(dummy_request_mask_ptr + g)
            pnat = tl.load(replay_pnat_ptr + blk)
            cache_buf_idx = tl.load(replay_cache_buf_idx_ptr + blk)
            wrote_checkpoint = pnat + REPLAY_STEP_WIDTH > REPLAY_HISTORY_SIZE
            next_pnat = tl.where(wrote_checkpoint, num_accepted_tokens, pnat + num_accepted_tokens)
            next_cache_buf_idx = tl.where(wrote_checkpoint, 1 - cache_buf_idx, cache_buf_idx)
            tl.store(replay_pnat_ptr + blk, next_pnat, mask=~is_dummy_request)
            tl.store(replay_cache_buf_idx_ptr + blk, next_cache_buf_idx, mask=~is_dummy_request)

    # int64 throughout: per-layer strides are O(1e8), so layer*stride overflows
    # int32 and would corrupt addresses / fault.
    src_base = layer * src_s_layer + row * src_s_row + acc * src_s_step
    dst_base = layer * dst_s_layer + blk * dst_s_block
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < count
    v = tl.load(src_ptr + src_base + offs, mask=mask)
    tl.store(dst_ptr + dst_base + offs, v, mask=mask)


def _promote_mamba_state_triton(
    dst: torch.Tensor,
    intermediate: torch.Tensor,
    src_state_indices: torch.Tensor,
    accepted_position_source: torch.Tensor,
    dst_state_indices: torch.Tensor,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    position_source_is_token_count: bool = False,
    replay_pnat: Optional[torch.Tensor] = None,
    replay_cache_buf_idx: Optional[torch.Tensor] = None,
    dummy_request_mask: Optional[torch.Tensor] = None,
    replay_step_width: int = 0,
    replay_history_size: int = 0,
    BLOCK: int = 2048,
) -> None:
    """Scatter each generation request's accepted draft-step recurrent state
    from the per-request intermediate buffer into the unified C++ pool view.

    Args:
        dst: ``[num_layers, num_blocks, *state_shape]`` view of the C++ pool.
            The block dim (dim 1) is strided (the pool interleaves
            ``ssm_bytes | conv_bytes`` per block), so ``dst`` is non-contiguous.
            The kernel writes through ``dst``'s real strides, touching only this
            state's bytes -- the case that defeats torch.compile / aot_autograd
            (dtype-view mutation) and Inductor codegen (``XBLOCK`` on the uint8
            pool).
        intermediate: ``[num_layers, max_batch_size, T, *state_shape]`` dense.
        src_state_indices, accepted_position_source, dst_state_indices:
            ``[num_gens]`` int.
        num_accepted_tokens: ``[num_gens]`` accepted-token counts. Required
            when replay bookkeeping is enabled.
        position_source_is_token_count: Whether ``accepted_position_source``
            contains accepted-token counts instead of zero-based intermediate
            state positions. When true, the kernel subtracts one while
            computing the source position.
        replay_pnat, replay_cache_buf_idx, dummy_request_mask: Optional replay
            metadata. When supplied together, the kernel advances PNAT and the
            active history buffer while promoting the convolution state.

    For each gen ``g``::

        accepted_position = (
            accepted_position_source[g] - 1
            if position_source_is_token_count
            else accepted_position_source[g]
        )
        dst[:, dst_state_indices[g]] = intermediate[:, src_state_indices[g], accepted_position]

    Pure gather->scatter copy; bandwidth-bound (~85% of HBM peak), one launch.
    """
    num_layers = dst.shape[0]
    num_gens = src_state_indices.shape[0]
    if num_gens == 0:
        return
    update_replay_bookkeeping = replay_pnat is not None
    if update_replay_bookkeeping:
        assert num_accepted_tokens is not None
        assert replay_cache_buf_idx is not None
        assert dummy_request_mask is not None
        assert replay_step_width > 0
        assert replay_history_size >= replay_step_width
        assert replay_pnat.dtype == torch.int32
        assert replay_cache_buf_idx.dtype == torch.int32
        assert dummy_request_mask.dtype == torch.bool
        assert dummy_request_mask.numel() >= num_gens
    else:
        # Compile-time-disabled pointer arguments still require tensors.
        num_accepted_tokens = accepted_position_source
        assert replay_cache_buf_idx is None
        assert dummy_request_mask is None
        replay_pnat = dst_state_indices
        replay_cache_buf_idx = dst_state_indices
        dummy_request_mask = dst_state_indices
    count = 1
    for s in dst.shape[2:]:
        count *= s
    # Grid dim order matters: dim 0 -> CUDA grid.x (limit 2^31-1), dim 1 ->
    # grid.y (limit 65535). Put the (layer, gen) pairs in dim 0 since that is
    # what grows with batch size (num_layers*num_gens stays far below 2^31 for
    # any real config); the tile count in dim 1 is small (~count/BLOCK). Do NOT
    # swap them: pairs would hit the 65535 y-limit at num_layers*num_gens>65535.
    grid = (num_layers * num_gens, triton.cdiv(count, BLOCK))
    _promote_mamba_state_kernel[grid](
        intermediate,
        dst,
        src_state_indices,
        accepted_position_source,
        num_accepted_tokens,
        dst_state_indices,
        replay_pnat,
        replay_cache_buf_idx,
        dummy_request_mask,
        num_gens,
        count,
        intermediate.stride(0),
        intermediate.stride(1),
        intermediate.stride(2),
        dst.stride(0),
        dst.stride(1),
        POSITION_SOURCE_IS_TOKEN_COUNT=position_source_is_token_count,
        REPLAY_STEP_WIDTH=replay_step_width,
        REPLAY_HISTORY_SIZE=replay_history_size,
        UPDATE_REPLAY_BOOKKEEPING=update_replay_bookkeeping,
        BLOCK=BLOCK,
    )


def _mamba_snapshot_rule_counts(
    kv_cache_config: KvCacheConfig,
    max_seq_len: Optional[int],
    tokens_per_block: int,
) -> Tuple[int, int]:
    """Return reachable fixed rules and their partial-block upper bound."""
    if not kv_cache_config.enable_block_reuse:
        return 0, 0

    num_rules = 0
    num_unaligned_rules = 0
    state_config = kv_cache_config.mamba_state_config
    for offset in set(state_config.additional_snapshot_offsets_from_start):
        if max_seq_len is not None and offset > max_seq_len:
            continue
        num_rules += 1
        num_unaligned_rules += int(offset % tokens_per_block != 0)
    for offset in set(state_config.additional_snapshot_offsets_from_end):
        # A from-end offset is reachable iff some valid prompt is longer than
        # the offset. Its absolute alignment depends on that prompt.
        if max_seq_len is not None and offset >= max_seq_len:
            continue
        num_rules += 1
        num_unaligned_rules += 1
    return num_rules, num_unaligned_rules


def _mamba_regular_snapshot_interval(
    kv_cache_config: KvCacheConfig,
    max_seq_len: Optional[int],
) -> Optional[int]:
    if not kv_cache_config.enable_block_reuse:
        return None
    interval = kv_cache_config.mamba_state_config.periodic_snapshot_interval
    if interval is None or interval <= 0:
        return None
    if max_seq_len is not None and interval > max_seq_len:
        return None
    return interval


def _get_local_mamba_cache_layout(
    model_config,
    mapping: Mapping,
    *,
    spec_config=None,
    is_draft: bool = False,
    use_separate_draft_kv_cache: bool = False,
):
    """Return normalized params and local Mamba/attention layer counts.

    Cache construction and affine sizing must follow the model's PP layout:
    partition base layers first, then place appended speculative layers on the
    last PP rank. The normalized params retain target masks and the appended
    draft-layer count so estimation selects the same combined or per-manager
    layout as runtime.
    """
    from tensorrt_llm._torch.pyexecutor.config_utils import extract_mamba_kv_cache_params

    params = extract_mamba_kv_cache_params(
        model_config.pretrained_config,
        spec_config=spec_config,
        quant_config=model_config.quant_config,
    )
    mamba_layer_mask, full_attention_layer_mask = params.get_layer_masks(
        is_draft=is_draft,
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
    local_mamba_layers = sum(mamba_layer_mask[layer_idx] for layer_idx in local_layer_indices)
    local_attention_layers = sum(
        full_attention_layer_mask[layer_idx] for layer_idx in local_layer_indices
    )
    return params, local_mamba_layers, local_attention_layers


def _estimate_mamba_hybrid_cache_cost(
    model_config,
    mapping: Mapping,
    *,
    max_batch_size: int,
    kv_cache_config: KvCacheConfig,
    tokens_per_block: int,
    max_seq_len: Optional[int],
    num_reserved_dummy_slots: int,
    include_explicit_snapshots: bool,
    cap_partial_attention_snapshots: bool,
    extra_state_bytes_per_rank: int = 0,
    is_draft: bool = False,
    use_separate_draft_kv_cache: bool = False,
    **kwargs,
) -> Tuple[int, int]:
    spec_config = kwargs.get("spec_config")
    params, local_mamba_layers, local_attention_layers = _get_local_mamba_cache_layout(
        model_config,
        mapping,
        spec_config=spec_config,
        is_draft=is_draft,
        use_separate_draft_kv_cache=use_separate_draft_kv_cache,
    )
    attention_slope = (
        KVCacheManager.get_cache_size_per_token(
            model_config,
            mapping,
            num_layers=local_attention_layers,
            **kwargs,
        )
        if local_attention_layers > 0
        else 0
    )
    state_bytes_per_rank = local_mamba_layers * params.get_states_bytes_per_layer(mapping)
    state_bytes_per_rank += extra_state_bytes_per_rank
    max_resident_sequences = max_batch_size * mapping.pp_size

    if include_explicit_snapshots:
        fixed_rules, unaligned_fixed_rules = _mamba_snapshot_rule_counts(
            kv_cache_config, max_seq_len, tokens_per_block
        )
    else:
        fixed_rules = 0
        unaligned_fixed_rules = 0
    fixed_state_slots = (
        max_resident_sequences + num_reserved_dummy_slots + max_resident_sequences * fixed_rules
    )
    attention_block_bytes = attention_slope * tokens_per_block

    interval = _mamba_regular_snapshot_interval(kv_cache_config, max_seq_len)
    has_unaligned_periodic_snapshot = interval is not None and interval % tokens_per_block != 0
    if cap_partial_attention_snapshots:
        # Snapshot alignment is unknown while estimating cache cost. Once a
        # snapshot is possible, reserve one retained partial attention page
        # per resident lineage. Dummy requests carry no attention capacity.
        has_non_live_ssm_capacity = fixed_rules > 0 or interval is not None
        partial_attention_slots = max_resident_sequences if has_non_live_ssm_capacity else 0
    else:
        partial_attention_slots = max_resident_sequences * unaligned_fixed_rules
    intercept = (
        fixed_state_slots * state_bytes_per_rank + partial_attention_slots * attention_block_bytes
    )

    if interval is None:
        regular_slope = 0
    else:
        regular_slope = math.ceil(state_bytes_per_rank / interval)
        if has_unaligned_periodic_snapshot and not cap_partial_attention_snapshots:
            regular_slope += math.ceil(attention_block_bytes / interval)
    return attention_slope + regular_slope, intercept


# The split legacy modules import the original implementation helpers as a
# group. Keep private helpers in this internal export list while the package
# facade exposes only its compatibility surface.
__all__ = [
    "BaseMambaCacheManager",
    "MambaHybridCacheManager",
    "MambaLayerCache",
    "MambaStateLayout",
    "MambaAcceptanceBatch",
    "MambaRole",
    "use_py_mamba_cache_manager",
    "get_tensor_size_bytes",
]


def _stack_state_views(states: List[torch.Tensor]) -> Optional[torch.Tensor]:
    """Returns one ``[num_layers, num_slots, *state_shape]`` view of ``states``.

    A stacked view collapses per-layer promotion loop into a single launch of _promote_mamba_state_triton.
    This requires per-layer pool views to be *affine*: identical dtype/device/shape/strides, and base
    pointers spaced by a constant. returns ``None`` otherwise.
    """
    if not states:
        return None
    ref = states[0]
    # TensorWrapper exposes CUDA pointers; CPU contract tests use per-layer views.
    if ref.device.type != "cuda":
        return None
    if len(states) == 1:
        return ref.unsqueeze(0)
    for state in states[1:]:
        if (
            state.dtype != ref.dtype
            or state.device != ref.device
            or state.shape != ref.shape
            or state.stride() != ref.stride()
        ):
            return None
    itemsize = ref.element_size()
    layer_stride_bytes = states[1].data_ptr() - states[0].data_ptr()
    if layer_stride_bytes <= 0 or layer_stride_bytes % itemsize != 0:
        return None
    base = states[0].data_ptr()
    if any(
        state.data_ptr() != base + layer * layer_stride_bytes for layer, state in enumerate(states)
    ):
        return None
    layer_stride = layer_stride_bytes // itemsize
    # Highest element one layer's view can reach, so the flat wrapper below
    # covers every byte the stacked view addresses and no more.
    layer_extent = 1 + sum((size - 1) * stride for size, stride in zip(ref.shape, ref.stride()))
    total_elems = (len(states) - 1) * layer_stride + layer_extent
    flat = convert_to_torch_tensor(TensorWrapper(base, ref.dtype, [total_elems]))
    return flat.as_strided([len(states)] + list(ref.shape), [layer_stride] + list(ref.stride()))


def _promote_intermediate_states(
    destination_states: tuple[torch.Tensor, ...],
    intermediate_states: torch.Tensor,
    batch: MambaAcceptanceBatch,
    stacked: torch.Tensor | None = None,
) -> None:
    """Copy accepted positions from [layer, batch, step, ...] into per-layer pools."""
    # Preserve the established kernel monkeypatch point without coupling the
    # concrete intermediate and replay algorithms through inheritance.
    from tensorrt_llm._torch.pyexecutor.kv_cache import mamba_cache_manager

    if stacked is not None:
        mamba_cache_manager._promote_mamba_state_triton(
            stacked,
            intermediate_states,
            batch.source_state_indices,
            batch.accepted_positions,
            batch.destination_state_indices,
        )
        return
    for layer_offset, destination in enumerate(destination_states):
        mamba_cache_manager._promote_mamba_state_triton(
            destination.unsqueeze(0),
            intermediate_states[layer_offset : layer_offset + 1],
            batch.source_state_indices,
            batch.accepted_positions,
            batch.destination_state_indices,
        )

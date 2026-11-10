# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba2-owned speculative and replay cache state."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaAcceptanceBatch,
    MambaLayerCache,
    MambaStateLayout,
    _get_local_mamba_cache_layout,
    _get_num_cuda_graph_padding_dummy_slots,
    _host_slots_to_device,
    _mamba_effective_tp_size,
    _promote_intermediate_states,
    _stack_state_views,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig

MIN_REPLAY_HISTORY_SIZE = 16


def _mamba_replay_disabled_reason(
    spec_config: object | None,
    ssm_dtype: torch.dtype,
    stochastic_rounding: bool,
    sm: int,
) -> str | None:
    """Why Nemotron's linear replay path is disabled, or None if enabled.

    Shared by the cache manager factory and the cache budget estimator.
    """
    if spec_config is None:
        return "speculative decoding is not configured"
    if sm < 80:
        return "replay requires SM80 or newer"
    if getattr(spec_config, "use_dynamic_tree", False):
        return "dynamic tree verification is unsupported"
    if os.environ.get("TRTLLM_USE_MAMBA_REPLAY", "1") == "0":
        return "TRTLLM_USE_MAMBA_REPLAY=0"
    if stochastic_rounding and ssm_dtype == torch.float16 and not (100 <= sm < 120):
        return f"FP16 replay stochastic rounding is unsupported on SM{sm}"
    return None


def _mamba_replay_capacity(
    max_batch_size: int, mapping: Mapping, spec_config: object | None, headroom: int
) -> int:
    """Replay slots for admitted requests, preemption headroom and padding dummies.

    Shared by the cache manager and the cache budget estimator.
    """
    return (
        max_batch_size * mapping.pp_size
        + headroom
        + _get_num_cuda_graph_padding_dummy_slots(spec_config, max_batch_size)
        + int(mapping.enable_attention_dp)
    )


def _mamba_replay_history_bytes(
    num_layers: int,
    capacity: int,
    history_size: int,
    nheads: int,
    head_dim: int,
    d_state: int,
    n_groups: int,
    dtype: torch.dtype,
) -> int:
    """Exact tensor bytes for double-buffered history and its two counters."""
    values_per_token = nheads * head_dim + n_groups * d_state
    history_bytes = (
        num_layers
        * capacity
        * 2
        * history_size
        * (values_per_token * dtype.itemsize + 2 * nheads * torch.float32.itemsize)
    )
    return history_bytes + capacity * 2 * torch.int32.itemsize


class ReplayStateUpdateMetadata(NamedTuple):
    """Shared tensors and fixed sizes for replay state updates."""

    prev_num_accepted_tokens: torch.Tensor
    cache_buf_idx: torch.Tensor
    replay_step_width: int
    replay_history_size: int


def _advance_replay_state(
    replay_metadata: ReplayStateUpdateMetadata,
    state_indices: torch.Tensor,
    accepted_tokens: torch.Tensor,
    is_dummy_request: torch.Tensor | None = None,
) -> None:
    """Advance double-buffered replay bookkeeping after a verify step."""
    slots = state_indices.long()
    accepted_tokens = accepted_tokens.to(replay_metadata.prev_num_accepted_tokens.dtype)
    previous = replay_metadata.prev_num_accepted_tokens[slots]
    wrote_checkpoint = (
        previous + replay_metadata.replay_step_width > replay_metadata.replay_history_size
    )
    next_accepted = torch.where(wrote_checkpoint, accepted_tokens, previous + accepted_tokens)
    cache_buffer = replay_metadata.cache_buf_idx[slots]
    next_cache_buffer = torch.where(wrote_checkpoint, 1 - cache_buffer, cache_buffer)
    if is_dummy_request is not None:
        next_accepted = torch.where(is_dummy_request, previous, next_accepted)
        next_cache_buffer = torch.where(is_dummy_request, cache_buffer, next_cache_buffer)
    replay_metadata.prev_num_accepted_tokens[slots] = next_accepted
    replay_metadata.cache_buf_idx[slots] = next_cache_buffer


_MAMBA_SSM_SEED_MASK = (1 << 62) - 1
_MAMBA_SSM_UINT64_MASK = (1 << 64) - 1
_MAMBA_SSM_SEED_BASE = 0x6A09E667F3BCC908
_MAMBA_SSM_SEED_MIX_COUNTER = 0x2545F4914F6CDD1D
_MAMBA_SSM_SEED_MIX_SLOT = 0x1B873593CC9E2D51
_MAMBA_SSM_SEED_MIX_RANK = 0x9E3779B97F4A7C15


def _splitmix64(value: int) -> int:
    """Return the SplitMix64 finalizer for ``value``."""
    value = (value + 0x9E3779B97F4A7C15) & _MAMBA_SSM_UINT64_MASK
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MAMBA_SSM_UINT64_MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MAMBA_SSM_UINT64_MASK
    value ^= value >> 31
    return value & _MAMBA_SSM_UINT64_MASK


def compute_deterministic_mamba_seed(counter: int, slot: int, rank_offset: int) -> int:
    """Return a reproducible, nonzero int64 Philox seed."""
    folded = (
        _MAMBA_SSM_SEED_BASE
        + counter * _MAMBA_SSM_SEED_MIX_COUNTER
        + slot * _MAMBA_SSM_SEED_MIX_SLOT
        + rank_offset * _MAMBA_SSM_SEED_MIX_RANK
    ) & _MAMBA_SSM_UINT64_MASK
    seed = _splitmix64(folded) & _MAMBA_SSM_SEED_MASK
    return seed or 1


def allocate_mamba_seed_buffer(
    cache_size: int, rank_offset: int, device: torch.device
) -> torch.Tensor:
    """Allocate the graph-stable per-slot Philox seed buffer."""
    seeds = [compute_deterministic_mamba_seed(0, slot, rank_offset) for slot in range(cache_size)]
    return torch.tensor(seeds, dtype=torch.int64, device=device)


def _reset_mamba_seed_buffer(
    seed_buffer: torch.Tensor,
    slots: torch.Tensor,
    host_slots: list[int],
    *,
    counter: int,
    rank_offset: int,
) -> None:
    """Refresh selected slots in the graph-stable int64 Philox buffer."""
    seeds = [compute_deterministic_mamba_seed(counter, slot, rank_offset) for slot in host_slots]
    seed_buffer[slots] = torch.tensor(seeds, dtype=torch.int64, pin_memory=prefer_pinned()).to(
        seed_buffer.device, non_blocking=True
    )


@dataclass(frozen=True, kw_only=True)
class Mamba2LayerCache(MambaLayerCache):
    """Per-layer Mamba2 intermediate or replay views consumed by its mixer."""

    intermediate_conv_window: torch.Tensor | None = None
    intermediate_ssm: torch.Tensor | None = None
    mamba_ssm_rand_seed: torch.Tensor | None = field(
        default=None,
        metadata={"slot_shared": True},
    )
    prev_num_accepted_tokens: torch.Tensor | None = field(
        default=None,
        metadata={"slot_shared": True},
    )
    cache_buf_idx: torch.Tensor | None = field(
        default=None,
        metadata={"slot_shared": True},
    )
    old_x: torch.Tensor | None = None
    old_B: torch.Tensor | None = None
    old_dt: torch.Tensor | None = None
    old_dA_cumsum: torch.Tensor | None = None


def mamba_seed_rank_offset(mapping: Mapping) -> int:
    """Return a distinct deterministic seed offset for this parallel rank."""
    return mapping.tp_rank * 1_000_003 + mapping.pp_rank * 1_000_033 + mapping.rank * 1_009


class Mamba2State:
    """Intermediate promotion plus Mamba2 stochastic-rounding seeds."""

    uses_replay = False

    def __init__(self, *, stochastic_rounding: bool = False) -> None:
        self._stochastic_rounding = stochastic_rounding
        self._ssm_states: tuple[torch.Tensor, ...] = ()
        self._conv_states: tuple[torch.Tensor, ...] = ()
        self._stacked_ssm: torch.Tensor | None = None
        self._stacked_conv: torch.Tensor | None = None
        self.intermediate_ssm: torch.Tensor | None = None
        self.intermediate_conv: torch.Tensor | None = None
        self.intermediate_indices: torch.Tensor | None = None
        self.rand_seed: torch.Tensor | None = None
        self._seed_request_counter = 0
        self._seed_rank_offset = 0

    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        self._ssm_states = tuple(ssm_states)
        self._conv_states = tuple(conv_states)
        self._stacked_ssm = _stack_state_views(ssm_states)
        self._stacked_conv = _stack_state_views(conv_states)
        if not context.mamba_pp_layers:
            return

        device = ssm_states[0].device

        if context.spec_config is not None:
            tokens_per_step = context.spec_config.tokens_per_gen_step
            common_shape = [
                len(context.mamba_pp_layers),
                context.max_batch_size,
                tokens_per_step,
            ]
            self.intermediate_ssm = torch.zeros(
                common_shape + list(context.ssm_state_shape),
                dtype=context.ssm_state_dtype,
                device=device,
            )
            self.intermediate_conv = torch.zeros(
                common_shape + list(context.conv_state_shape),
                dtype=context.conv_state_dtype,
                device=device,
            )
            self.intermediate_indices = torch.arange(
                context.max_batch_size, dtype=torch.int32, device=device
            )
        self._seed_rank_offset = (
            context.mapping.tp_rank * 1_000_003
            + context.mapping.pp_rank * 1_000_033
            + context.mapping.rank * 1_009
        )
        if self._stochastic_rounding and ssm_states:
            states = ssm_states[0]
            self.rand_seed = allocate_mamba_seed_buffer(
                states.shape[0], self._seed_rank_offset, states.device
            )

    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None:
        if self.rand_seed is None:
            return
        self._seed_request_counter += 1
        _reset_mamba_seed_buffer(
            self.rand_seed,
            slots,
            host_slots,
            counter=self._seed_request_counter,
            rank_offset=self._seed_rank_offset,
        )

    def _layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        fields: dict[str, torch.Tensor | None] = {}
        if self.intermediate_conv is not None:
            fields["intermediate_conv_window"] = self.intermediate_conv[layer_offset]
        if self.intermediate_ssm is not None:
            fields["intermediate_ssm"] = self.intermediate_ssm[layer_offset]
        if self.rand_seed is not None:
            fields["mamba_ssm_rand_seed"] = self.rand_seed
        return fields

    def shutdown(self) -> None:
        self._ssm_states = ()
        self._conv_states = ()
        self._stacked_ssm = None
        self._stacked_conv = None
        self.intermediate_ssm = None
        self.intermediate_conv = None
        self.intermediate_indices = None
        self.rand_seed = None

    def update(self, batch: MambaAcceptanceBatch) -> None:
        if self.intermediate_indices is None:
            return
        _promote_intermediate_states(
            self._ssm_states, self.intermediate_ssm, batch, self._stacked_ssm
        )
        _promote_intermediate_states(
            self._conv_states, self.intermediate_conv, batch, self._stacked_conv
        )

    def make_layer_cache(
        self, layer_offset: int, conv: torch.Tensor, temporal: torch.Tensor
    ) -> MambaLayerCache:
        return Mamba2LayerCache(
            conv=conv, temporal=temporal, **self._layer_cache_fields(layer_offset)
        )


class ReplayHistory:
    """Own Mamba2 replay history, seeds, bookkeeping, and conv promotion."""

    @property
    def intermediate_ssm(self) -> torch.Tensor | None:
        return None

    def __init__(self, tokens_per_gen_step: int) -> None:
        self.capacity: int | None = None
        self.ssm_to_replay_slot: torch.Tensor | None = None
        self._ssm_states: tuple[torch.Tensor, ...] = ()
        self._conv_states: tuple[torch.Tensor, ...] = ()
        self._stacked_conv: torch.Tensor | None = None
        self.intermediate_conv: torch.Tensor | None = None
        self.intermediate_indices: torch.Tensor | None = None
        self.rand_seed: torch.Tensor | None = None
        self._seed_request_counter = 0
        self._seed_rank_offset = 0
        self.replay_step_width = tokens_per_gen_step
        self.replay_history_size = max(MIN_REPLAY_HISTORY_SIZE, tokens_per_gen_step)
        self.prev_num_accepted_tokens: torch.Tensor | None = None
        self.cache_buf_idx: torch.Tensor | None = None
        self.old_x: torch.Tensor | None = None
        self.old_B: torch.Tensor | None = None
        self.old_dt: torch.Tensor | None = None
        self.old_dA_cumsum: torch.Tensor | None = None

    @property
    def uses_replay(self) -> bool:
        return True

    def validate(self, context: MambaStateLayout) -> None:
        if context.spec_config is None:
            raise ValueError("Mamba2 replay requires speculative decoding")
        if context.n_groups_per_rank <= 0 and context.mamba_pp_layers:
            raise ValueError("Mamba2 replay requires at least one state group per rank")

    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        self._ssm_states = tuple(ssm_states)
        self._conv_states = tuple(conv_states)
        self._stacked_conv = _stack_state_views(conv_states)
        self._seed_rank_offset = (
            context.mapping.tp_rank * 1_000_003
            + context.mapping.pp_rank * 1_000_033
            + context.mapping.rank * 1_009
        )
        if not context.mamba_pp_layers:
            return

        states = ssm_states
        cache_size = states[0].shape[0]
        device = states[0].device
        tokens_per_step = context.spec_config.tokens_per_gen_step
        common_intermediate_shape = [
            len(context.mamba_pp_layers),
            context.max_batch_size,
            tokens_per_step,
        ]
        self.intermediate_conv = torch.zeros(
            common_intermediate_shape + list(context.conv_state_shape),
            dtype=context.conv_state_dtype,
            device=device,
        )
        self.intermediate_indices = torch.arange(
            context.max_batch_size, dtype=torch.int32, device=device
        )
        self.rand_seed = allocate_mamba_seed_buffer(cache_size, self._seed_rank_offset, device)

        num_heads, head_dim, d_state = context.ssm_state_shape
        history_capacity = self.capacity if self.capacity is not None else cache_size
        if self.capacity is not None:
            self.ssm_to_replay_slot = torch.zeros(cache_size, dtype=torch.int32, device=device)
        common_replay_shape = [len(context.mamba_pp_layers), history_capacity, 2]
        self.prev_num_accepted_tokens = torch.zeros(
            history_capacity, dtype=torch.int32, device=device
        )
        self.cache_buf_idx = torch.zeros(history_capacity, dtype=torch.int32, device=device)
        self.old_x = torch.zeros(
            common_replay_shape + [self.replay_history_size, num_heads, head_dim],
            dtype=context.conv_state_dtype,
            device=device,
        )
        self.old_B = torch.zeros(
            common_replay_shape + [self.replay_history_size, context.n_groups_per_rank, d_state],
            dtype=context.conv_state_dtype,
            device=device,
        )
        self.old_dt = torch.zeros(
            common_replay_shape + [num_heads, self.replay_history_size],
            dtype=torch.float32,
            device=device,
        )
        self.old_dA_cumsum = torch.zeros_like(self.old_dt)

    def _layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        return {
            "mamba_ssm_rand_seed": self.rand_seed,
            "intermediate_conv_window": self.intermediate_conv[layer_offset],
            "prev_num_accepted_tokens": self.prev_num_accepted_tokens,
            "cache_buf_idx": self.cache_buf_idx,
            "old_x": self.old_x[layer_offset],
            "old_B": self.old_B[layer_offset],
            "old_dt": self.old_dt[layer_offset],
            "old_dA_cumsum": self.old_dA_cumsum[layer_offset],
        }

    def make_layer_cache(
        self,
        layer_offset: int,
        conv: torch.Tensor,
        temporal: torch.Tensor,
    ) -> MambaLayerCache:
        return Mamba2LayerCache(
            conv=conv,
            temporal=temporal,
            **self._layer_cache_fields(layer_offset),
        )

    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None:
        if self.prev_num_accepted_tokens is None:
            return
        replay_slots = self.get_replay_state_indices(slots).long()
        self.prev_num_accepted_tokens.index_fill_(0, replay_slots, 0)
        self.cache_buf_idx.index_fill_(0, replay_slots, 0)
        for buffer in (self.old_x, self.old_B, self.old_dt, self.old_dA_cumsum):
            buffer.index_fill_(1, replay_slots, 0)
        if self.rand_seed is not None:
            self._seed_request_counter += 1
            _reset_mamba_seed_buffer(
                self.rand_seed,
                slots,
                host_slots,
                counter=self._seed_request_counter,
                rank_offset=self._seed_rank_offset,
            )

    def update(self, batch: MambaAcceptanceBatch) -> None:
        metadata = self.get_replay_metadata()
        if metadata is None:
            raise RuntimeError("Mamba2 replay buffers are not bound")
        _advance_replay_state(
            metadata,
            self.get_replay_state_indices(batch.destination_state_indices),
            batch.num_accepted_tokens,
            batch.is_dummy_request,
        )
        _promote_intermediate_states(
            self._conv_states, self.intermediate_conv, batch, self._stacked_conv
        )

    def get_replay_state_indices(self, state_indices: torch.Tensor) -> torch.Tensor:
        if self.ssm_to_replay_slot is None:
            return state_indices
        return self.ssm_to_replay_slot[state_indices.long()]

    def get_replay_metadata(self) -> ReplayStateUpdateMetadata | None:
        if self.prev_num_accepted_tokens is None or self.cache_buf_idx is None:
            return None
        return ReplayStateUpdateMetadata(
            prev_num_accepted_tokens=self.prev_num_accepted_tokens,
            cache_buf_idx=self.cache_buf_idx,
            replay_step_width=self.replay_step_width,
            replay_history_size=self.replay_history_size,
        )

    def shutdown(self) -> None:
        self._ssm_states = ()
        self._conv_states = ()
        self._stacked_conv = None
        self.intermediate_conv = None
        self.intermediate_indices = None
        self.rand_seed = None
        self.prev_num_accepted_tokens = None
        self.cache_buf_idx = None
        self.old_x = None
        self.old_B = None
        self.old_dt = None
        self.old_dA_cumsum = None
        self.ssm_to_replay_slot = None


def create_mamba2_state(
    *, spec_config: object | None, use_replay: bool, stochastic_rounding: bool = False
) -> Mamba2State | ReplayHistory:
    """Build Mamba2 intermediate state or compact replay history."""
    if use_replay:
        if spec_config is None:
            raise ValueError("Mamba replay requires speculative decoding")
        return ReplayHistory(spec_config.tokens_per_gen_step)
    return Mamba2State(stochastic_rounding=stochastic_rounding)


def select_mamba2_state(
    *,
    spec_config: object | None,
    ssm_cache_dtype: torch.dtype,
    stochastic_rounding: bool,
) -> Mamba2State | ReplayHistory:
    """Apply Mamba2 replay gates and return its selected update policy."""
    disabled_reason = _mamba_replay_disabled_reason(
        spec_config, ssm_cache_dtype, stochastic_rounding, get_sm_version()
    )
    use_replay = disabled_reason is None
    status = "ENABLED" if use_replay else f"DISABLED ({disabled_reason})"
    logger.info(f"Mamba replay state update: {status}")
    return create_mamba2_state(
        spec_config=spec_config, use_replay=use_replay, stochastic_rounding=stochastic_rounding
    )


class NemotronHybridCacheManager(MambaHybridCacheManagerV2):
    """Nemotron/Mamba2 geometry, replay selection and seed ownership."""

    _speculative_state: Mamba2State | ReplayHistory

    @override
    def __init__(self, *args, use_replay_state_update: bool | None = None, **kwargs) -> None:
        self._mamba_ssm_stochastic_rounding = kwargs.pop("mamba_ssm_stochastic_rounding", False)
        self._requested_replay = use_replay_state_update
        self._replay_headroom = kwargs.get("gen_preemption_headroom", 0)
        self._replay_capacity = 0
        self._replay_history_bytes = 0
        self._request_id_to_replay_slot: dict[int, int] = {}
        self._free_replay_slots: list[int] = []
        self._ssm_to_replay_slot_host: dict[int, int] = {}
        kwargs.setdefault("conv_state_layout", "x_b_c")
        super().__init__(*args, **kwargs)

    @override
    def _initialize_spec_state(self) -> Mamba2State | ReplayHistory:
        if self._requested_replay is not None:
            state = create_mamba2_state(
                spec_config=self.spec_config,
                use_replay=self._requested_replay,
                stochastic_rounding=self._mamba_ssm_stochastic_rounding,
            )
        else:
            state = select_mamba2_state(
                spec_config=self.spec_config,
                ssm_cache_dtype=self.ssm_state_dtype,
                stochastic_rounding=self._mamba_ssm_stochastic_rounding,
            )

        if isinstance(state, ReplayHistory):
            if self.local_num_mamba_layers and self._global_n_groups % _mamba_effective_tp_size(
                self._state_layout.mapping
            ):
                raise ValueError("Replay state groups must be divisible by the effective TP size")
            state.validate(self._state_layout)
            if self.local_num_mamba_layers:
                layout = self._state_layout
                self._replay_capacity = _mamba_replay_capacity(
                    layout.max_batch_size, layout.mapping, layout.spec_config, self._replay_headroom
                )
                state.capacity = self._replay_capacity
                self._free_replay_slots = list(reversed(range(self._replay_capacity)))
                nheads, head_dim, d_state = self.ssm_state_shape
                self._replay_history_bytes = _mamba_replay_history_bytes(
                    self.local_num_mamba_layers,
                    self._replay_capacity,
                    state.replay_history_size,
                    nheads,
                    head_dim,
                    d_state,
                    self._n_groups_per_rank,
                    self.conv_state_dtype,
                )
        return state

    def get_replay_state_update_metadata(self) -> ReplayStateUpdateMetadata | None:
        state = self._speculative_state
        return state.get_replay_metadata() if isinstance(state, ReplayHistory) else None

    def get_mamba_ssm_rand_seed(self) -> torch.Tensor | None:
        return self._speculative_state.rand_seed

    @property
    def _compact_replay(self) -> bool:
        return self._replay_capacity > 0

    @override
    def get_replay_state_indices(self, state_indices: torch.Tensor) -> torch.Tensor:
        if not self._compact_replay:
            return state_indices
        return self._speculative_state.get_replay_state_indices(state_indices)

    @override
    def _fixed_external_cache_bytes(self) -> int:
        return self._replay_history_bytes

    def _release_replay_slot(self, request_id: int) -> None:
        slot = self._request_id_to_replay_slot.pop(request_id, None)
        if slot is not None:
            self._free_replay_slots.append(slot)

    def _setup_compact_replay_slots(
        self, requests: list[LlmRequest], state_values: list[int], num_contexts: int
    ) -> None:
        changed_states = []
        changed_slots = []
        new_generation_slots = []
        state = self._speculative_state
        device = state.ssm_to_replay_slot.device
        for i, (request, state_slot) in enumerate(zip(requests, state_values)):
            request_id = request.py_request_id
            slot = self._request_id_to_replay_slot.get(request_id)
            if slot is None:
                assert self._free_replay_slots, (
                    "Mamba replay capacity exhausted: admitted requests "
                    "(including suspended requests) and padding dummies must "
                    f"fit within {self._replay_capacity} replay slots"
                )
                slot = self._free_replay_slots.pop()
                self._request_id_to_replay_slot[request_id] = slot
                if i >= num_contexts:
                    new_generation_slots.append(slot)
            if self._ssm_to_replay_slot_host.get(state_slot) == slot:
                continue
            changed_states.append(state_slot)
            changed_slots.append(slot)
            self._ssm_to_replay_slot_host[state_slot] = slot
        # Stable resident batches need no mapping transfers or metadata resets.
        # Fresh pinned staging is owned by the async copy until DMA completes.
        if changed_states:
            state.ssm_to_replay_slot.index_copy_(
                0,
                _host_slots_to_device(changed_states, device),
                _host_slots_to_device(changed_slots, device, dtype=torch.int32),
            )
        if new_generation_slots:
            slots = _host_slots_to_device(new_generation_slots, device)
            state.prev_num_accepted_tokens.index_fill_(0, slots, 0)
            state.cache_buf_idx.index_fill_(0, slots, 0)

    @override
    def release_index_slot(self, request_id: int) -> None:
        """Release context-only replay ownership while retaining transfer KV."""
        super().release_index_slot(request_id)
        # Final context forward is complete; transfers read the recurrent
        # KV state, not replay history. New prefills can reuse that history.
        self._release_replay_slot(request_id)

    @override
    def _setup_state_indices(self, requests: list[LlmRequest], num_contexts: int = 0) -> None:
        super()._setup_state_indices(requests, num_contexts)
        if self._compact_replay:
            self._setup_compact_replay_slots(
                requests,
                [self._request_id_to_state_index[req.py_request_id] for req in requests],
                num_contexts,
            )

    @override
    def free_resources(self, request: LlmRequest, pin_on_release: bool = False) -> None:
        super().free_resources(request, pin_on_release)
        self._release_replay_slot(request.py_request_id)

    @override
    def _build_cache_config(self, config: KVCacheManagerConfig) -> KVCacheManagerConfig:
        if self._compact_replay and self.kv_cache_config.enable_kv_pool_rebalance:
            raise ValueError(
                "Replay slot maps are sized once from the SSM pool: set "
                "kv_cache_config.enable_kv_pool_rebalance=False."
            )
        return super()._build_cache_config(config)

    @classmethod
    @override
    def get_cache_size_per_token(cls, model_config, mapping: Mapping, **kwargs) -> tuple[int, int]:
        slope, intercept = super().get_cache_size_per_token(model_config, mapping, **kwargs)
        spec = kwargs.get("spec_config")
        params, layers, _ = _get_local_mamba_cache_layout(
            model_config,
            mapping,
            spec_config=spec,
            is_draft=kwargs.get("is_draft", False),
            use_separate_draft_kv_cache=kwargs.get("use_separate_draft_kv_cache", False),
        )
        quant = model_config.quant_config
        stochastic = quant is not None and quant.mamba_ssm_stochastic_rounding
        if (
            layers
            and _mamba_replay_disabled_reason(
                spec, params.mamba_ssm_cache_dtype, stochastic, get_sm_version()
            )
            is None
        ):
            capacity = _mamba_replay_capacity(
                kwargs["max_batch_size"],
                mapping,
                spec,
                kwargs.get("gen_preemption_headroom", 0),
            )
            tp_size = _mamba_effective_tp_size(mapping)
            intercept += _mamba_replay_history_bytes(
                layers,
                capacity,
                max(MIN_REPLAY_HISTORY_SIZE, spec.tokens_per_gen_step),
                params.num_heads // tp_size,
                params.head_dim,
                params.state_size,
                params.n_groups // tp_size,
                params.dtype,
            )
        return slope, intercept


def get_nemotron_cache_params(config, *, spec_config=None, quant_config=None):
    """Derive Mamba2 state geometry from the Nemotron hybrid pattern."""
    from tensorrt_llm._torch.pyexecutor.config_utils import build_mamba_kv_cache_params

    pattern = config.hybrid_override_pattern
    return build_mamba_kv_cache_params(
        config,
        state_size=config.ssm_state_size,
        conv_kernel=config.conv_kernel,
        num_heads=config.mamba_num_heads,
        n_groups=config.n_groups,
        head_dim=config.mamba_head_dim,
        mamba_mask=[layer == "M" for layer in pattern],
        target_full_attn_mask=[layer == "*" for layer in pattern],
        spec_config=spec_config,
        quant_config=quant_config,
    )


__all__ = [
    "Mamba2LayerCache",
    "Mamba2State",
    "ReplayHistory",
    "NemotronHybridCacheManager",
    "create_mamba2_state",
    "select_mamba2_state",
    "get_nemotron_cache_params",
    "mamba_seed_rank_offset",
]

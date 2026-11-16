# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba2-owned speculative and replay cache state."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.modules.mamba.exact_replay import pin_ssd_autotune_configs
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaAcceptanceBatch,
    MambaLayerCache,
    MambaStateLayout,
    _get_local_mamba_cache_layout,
    _mamba_effective_tp_size,
    _promote_intermediate_states,
    _stack_state_views,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
    ReplayLayout,
)
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole

MIN_REPLAY_HISTORY_SIZE = 16
# Per-layer replay history roles, in ReplayHistory attribute order.
_REPLAY_HISTORY_ROLES = {
    "old_x": DataRole("mamba2_old_x"),
    "old_B": DataRole("mamba2_old_B"),
    "old_dt": DataRole("mamba2_old_dt"),
    "old_dA_cumsum": DataRole("mamba2_old_dA_cumsum"),
}
# Held by the first local layer. KVCM copies 16-byte payloads; a row holds
# [prev_num_accepted_tokens, cache_buf_idx, padding, padding] as int32.
_REPLAY_META_ROLE = DataRole("mamba2_replay_meta")
_REPLAY_META_BYTES = 16
_REPLAY_META_PNAT = 0
_REPLAY_META_CACHE_BUF_IDX = 1
# Exact replay (FORCE_DETERMINISTIC): per-layer inputs of the partial chunk, in
# Mamba2LayerCache attribute order.
_EXACT_REPLAY_ROLES = {
    "replay_x": DataRole("mamba2_exact_replay_x"),
    "replay_dt": DataRole("mamba2_exact_replay_dt"),
    "replay_B": DataRole("mamba2_exact_replay_B"),
}
# Held by the first local layer: a row holds [tokens run, padding x3] as int32.
_EXACT_REPLAY_META_ROLE = DataRole("mamba2_exact_replay_meta")


class ReplayStateUpdateMetadata(NamedTuple):
    """Shared tensors and fixed sizes for replay state updates."""

    prev_num_accepted_tokens: torch.Tensor
    cache_buf_idx: torch.Tensor
    replay_step_width: int
    replay_history_size: int


def _advance_replay_state(
    replay_metadata: ReplayStateUpdateMetadata,
    replay_indices: torch.Tensor,
    accepted_tokens: torch.Tensor,
    is_dummy_request: torch.Tensor | None = None,
) -> None:
    """Advance double-buffered replay bookkeeping of ``replay_indices`` after a verify step."""
    slots = replay_indices.long()
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
    # Exact replay (FORCE_DETERMINISTIC): inputs of the partial chunk after
    # the boundary state held in `temporal`, and the slot's token count.
    replay_x: torch.Tensor | None = None
    replay_dt: torch.Tensor | None = None
    replay_B: torch.Tensor | None = None
    replay_num_tokens: torch.Tensor | None = field(
        default=None,
        metadata={"slot_shared": True},
    )


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
        # Stacked [layer, slot, ...] tensors, or per-layer views of PRIVATE
        # pool buffers that are not evenly spaced across layers.
        self.old_x: torch.Tensor | list[torch.Tensor] | None = None
        self.old_B: torch.Tensor | list[torch.Tensor] | None = None
        self.old_dt: torch.Tensor | list[torch.Tensor] | None = None
        self.old_dA_cumsum: torch.Tensor | list[torch.Tensor] | None = None
        self._replay_meta: torch.Tensor | None = None
        self._pool_layers: list[dict[DataRole, torch.Tensor]] | None = None

    @property
    def uses_replay(self) -> bool:
        return True

    @property
    def replay_shares_state_slot(self) -> bool:
        """The replay kernels address the history by its own replay slot."""
        return False

    def replay_layout(self, context: MambaStateLayout, layer_id: int) -> ReplayLayout:
        """Per-slot shapes of one layer's replay history.

        The first local layer also holds the replay metadata row.
        """
        if not context.mamba_pp_layers:
            return {}
        num_heads, head_dim, d_state = context.ssm_state_shape
        history = self.replay_history_size
        layout = {
            _REPLAY_HISTORY_ROLES["old_x"]: (
                context.conv_state_dtype,
                [2, history, num_heads, head_dim],
            ),
            _REPLAY_HISTORY_ROLES["old_B"]: (
                context.conv_state_dtype,
                [2, history, context.n_groups_per_rank, d_state],
            ),
            _REPLAY_HISTORY_ROLES["old_dt"]: (torch.float32, [2, num_heads, history]),
            _REPLAY_HISTORY_ROLES["old_dA_cumsum"]: (torch.float32, [2, num_heads, history]),
        }
        if layer_id == context.mamba_pp_layers[0]:
            layout[_REPLAY_META_ROLE] = (
                torch.int32,
                [_REPLAY_META_BYTES // torch.int32.itemsize],
            )
        return layout

    def bind_replay_pool(self, layers: list[dict[DataRole, torch.Tensor]]) -> None:
        """Use the manager's per-layer pool views instead of allocating history.

        Each view's first dimension is a replay slot.
        """
        self._pool_layers = layers

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

        if self._pool_layers is not None:
            self._bind_pool_history()
            return

        num_heads, head_dim, d_state = context.ssm_state_shape
        common_replay_shape = [len(context.mamba_pp_layers), cache_size, 2]
        self.prev_num_accepted_tokens = torch.zeros(cache_size, dtype=torch.int32, device=device)
        self.cache_buf_idx = torch.zeros(cache_size, dtype=torch.int32, device=device)
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

    def _bind_pool_history(self) -> None:
        layers = self._pool_layers
        self._replay_meta = layers[0][_REPLAY_META_ROLE]
        self.prev_num_accepted_tokens = self._replay_meta[:, _REPLAY_META_PNAT]
        self.cache_buf_idx = self._replay_meta[:, _REPLAY_META_CACHE_BUF_IDX]
        for name, role in _REPLAY_HISTORY_ROLES.items():
            views = [layer[role] for layer in layers]
            stacked = _stack_state_views(views)
            setattr(self, name, stacked if stacked is not None else views)

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

    def reset_replay_slots(self, replay_slots: torch.Tensor) -> None:
        """Clear the replay history and bookkeeping of ``replay_slots``."""
        if self.prev_num_accepted_tokens is None:
            return
        if self._replay_meta is not None:
            self._replay_meta.index_fill_(0, replay_slots, 0)
        else:
            self.prev_num_accepted_tokens.index_fill_(0, replay_slots, 0)
            self.cache_buf_idx.index_fill_(0, replay_slots, 0)
        for buffer in (self.old_x, self.old_B, self.old_dt, self.old_dA_cumsum):
            if isinstance(buffer, torch.Tensor):
                buffer.index_fill_(1, replay_slots, 0)
            else:
                for layer_buffer in buffer:
                    layer_buffer.index_fill_(0, replay_slots, 0)

    def reset_slots(
        self,
        slots: torch.Tensor,
        host_slots: list[int],
        replay_slots: torch.Tensor | None = None,
    ) -> None:
        """Reset context requests' replay history and refresh their state-slot seeds."""
        if self.prev_num_accepted_tokens is None:
            return
        self.reset_replay_slots(slots if replay_slots is None else replay_slots)
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
            (
                batch.destination_state_indices
                if batch.replay_indices is None
                else batch.replay_indices
            ),
            batch.num_accepted_tokens,
            batch.is_dummy_request,
        )
        _promote_intermediate_states(
            self._conv_states, self.intermediate_conv, batch, self._stacked_conv
        )

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
        self._replay_meta = None
        self._pool_layers = None


class Mamba2ExactReplayState:
    """Boundary-state semantics for FORCE_DETERMINISTIC (see exact_replay.py).

    `temporal` holds each slot's FP32 state at its last chunk boundary; the
    replay buffers hold the post-conv `x`, raw `dt` and post-conv `B` of the
    tokens after it, and `replay_num_tokens` counts the tokens a slot has run.
    They are buffers of the recurrent-state slot, so a snapshot, a relocation
    or a suspension of the slot carries them together with the boundary state.
    """

    uses_replay = False
    provides_layer_cache = True
    replay_shares_state_slot = True
    intermediate_indices = None
    intermediate_ssm = None
    intermediate_conv = None

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self._pool_layers: list[dict[DataRole, torch.Tensor]] | None = None
        self.replay_num_tokens: torch.Tensor | None = None

    def replay_layout(self, context: MambaStateLayout, layer_id: int) -> ReplayLayout:
        """Per-slot shapes of one layer's partial-chunk buffers.

        The first local layer also holds the token-count row.
        """
        if not context.mamba_pp_layers:
            return {}
        num_heads, head_dim, d_state = context.ssm_state_shape
        dtype = context.conv_state_dtype
        chunk = self.chunk_size
        layout = {
            _EXACT_REPLAY_ROLES["replay_x"]: (dtype, [chunk, num_heads, head_dim]),
            _EXACT_REPLAY_ROLES["replay_dt"]: (dtype, [chunk, num_heads]),
            _EXACT_REPLAY_ROLES["replay_B"]: (
                dtype,
                [chunk, context.n_groups_per_rank, d_state],
            ),
        }
        if layer_id == context.mamba_pp_layers[0]:
            layout[_EXACT_REPLAY_META_ROLE] = (
                torch.int32,
                [_REPLAY_META_BYTES // torch.int32.itemsize],
            )
        return layout

    def bind_replay_pool(self, layers: list[dict[DataRole, torch.Tensor]]) -> None:
        """Take the manager's per-layer views; their first dimension is the state slot."""
        self._pool_layers = layers

    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        if not context.mamba_pp_layers:
            return
        if self._pool_layers is None:
            raise RuntimeError("Mamba2 exact replay needs the cache manager's slot buffers")
        self.replay_num_tokens = self._pool_layers[0][_EXACT_REPLAY_META_ROLE][:, 0]
        pin_ssd_autotune_configs()

    def reset_replay_slots(self, replay_slots: torch.Tensor) -> None:
        """A request that starts in generation has run no tokens.

        No prefill has cleared its buffers either, and its slot may hold
        arbitrary bytes that a decode scan would read.
        """
        if self.replay_num_tokens is not None:
            self.replay_num_tokens.index_fill_(0, replay_slots, 0)
            for views in self._pool_layers:
                for name in ("replay_x", "replay_dt", "replay_B"):
                    views[_EXACT_REPLAY_ROLES[name]].index_fill_(0, replay_slots, 0)

    def reset_slots(
        self,
        slots: torch.Tensor,
        host_slots: list[int],
        replay_slots: torch.Tensor | None = None,
    ) -> None:
        # Context requests include resumed prefills and requests that reuse a
        # snapshot, whose counts must survive; a request without cached tokens
        # restarts its count in Mamba2Metadata.prepare instead.
        return

    def update(self, batch: MambaAcceptanceBatch) -> None:
        return

    def shutdown(self) -> None:
        self._pool_layers = None
        self.replay_num_tokens = None

    def make_layer_cache(
        self, layer_offset: int, conv: torch.Tensor, temporal: torch.Tensor
    ) -> MambaLayerCache:
        views = self._pool_layers[layer_offset]
        return Mamba2LayerCache(
            conv=conv,
            temporal=temporal,
            replay_x=views[_EXACT_REPLAY_ROLES["replay_x"]],
            replay_dt=views[_EXACT_REPLAY_ROLES["replay_dt"]],
            replay_B=views[_EXACT_REPLAY_ROLES["replay_B"]],
            replay_num_tokens=self.replay_num_tokens,
        )


def _exact_replay_disabled_reason(
    spec_config: object | None,
    ssm_cache_dtype: torch.dtype,
    chunk_size: int | None,
    is_disagg: bool,
) -> str | None:
    """Why FORCE_DETERMINISTIC keeps the default Mamba2 kernels, or None."""
    if spec_config is not None:
        return "speculative decoding is enabled"
    if is_disagg:
        return "disaggregated serving transfers the recurrent state without the replay buffers"
    if ssm_cache_dtype != torch.float32:
        return f"mamba_ssm_cache_dtype is {ssm_cache_dtype}; set it to float32"
    if get_sm_version() not in (100, 103):
        return "it is implemented for SM100 and SM103 only"
    if not chunk_size:
        return "the Mamba2 chunk size is unknown"
    return None


def _estimate_exact_replay_bytes_per_slot(model_config, mapping: Mapping, **kwargs) -> int:
    """Bytes the exact-replay buffers add to every recurrent-state slot of a manager not yet built."""
    if os.environ.get("FORCE_DETERMINISTIC", "0") != "1":
        return 0
    spec_config = kwargs.get("spec_config")
    params, local_layers, _ = _get_local_mamba_cache_layout(
        model_config,
        mapping,
        spec_config=spec_config,
        is_draft=kwargs.get("is_draft", False),
        use_separate_draft_kv_cache=kwargs.get("use_separate_draft_kv_cache", False),
    )
    if not local_layers:
        return 0
    ssm_cache_dtype = params.mamba_ssm_cache_dtype or params.dtype
    if _exact_replay_disabled_reason(
        spec_config, ssm_cache_dtype, params.mamba_chunk_size, kwargs.get("is_disagg", False)
    ):
        return 0
    tp_size = _mamba_effective_tp_size(mapping)
    layer_ids = tuple(range(local_layers))
    context = MambaStateLayout(
        layer_mask=(True,) * local_layers,
        pp_layers=layer_ids,
        mamba_pp_layers=layer_ids,
        mapping=mapping,
        conv_state_shape=(),
        ssm_state_shape=(params.num_heads // tp_size, params.head_dim, params.state_size),
        conv_state_dtype=params.dtype,
        ssm_state_dtype=ssm_cache_dtype,
        n_groups_per_rank=params.n_groups // tp_size,
        max_batch_size=0,
        state_index_capacity=0,
        slot_capacity=None,
        spec_config=spec_config,
    )
    state = Mamba2ExactReplayState(params.mamba_chunk_size)
    return sum(
        dtype.itemsize * math.prod(shape)
        for layer_id in layer_ids
        for dtype, shape in state.replay_layout(context, layer_id).values()
    )


def create_mamba2_state(
    *, spec_config: object | None, use_replay: bool, stochastic_rounding: bool = False
) -> Mamba2State | ReplayHistory:
    """Build Mamba2 intermediate state or compact replay history."""
    if use_replay:
        if spec_config is None:
            raise ValueError("Mamba replay requires speculative decoding")
        return ReplayHistory(spec_config.tokens_per_gen_step)
    return Mamba2State(stochastic_rounding=stochastic_rounding)


def _mamba2_replay_gate(
    spec_config: object | None, ssm_cache_dtype: torch.dtype, stochastic_rounding: bool
) -> tuple[bool, list[str]]:
    """Return whether Mamba2 replay is selected, with the reasons it is not."""
    sm = get_sm_version()
    use_replay = spec_config is not None and sm >= 80
    reasons = []
    if spec_config is None:
        reasons.append("Replay kernel requires speculative decoding; using non-replay path")
    if spec_config is not None and (getattr(spec_config, "use_dynamic_tree", False)):
        reasons.append("Replay kernel incompatible with tree attention; using legacy MTP path")
        use_replay = False
    if stochastic_rounding and ssm_cache_dtype == torch.float16 and not (100 <= sm < 120):
        reasons.append(
            "Replay kernel Philox requires 100 <= sm < 120; "
            "using legacy MTP path for stochastic rounding support"
        )
        use_replay = False
    if os.environ.get("TRTLLM_USE_MAMBA_REPLAY", "1") == "0":
        reasons.append("Replay kernel is disabled by TRTLLM_USE_MAMBA_REPLAY=0")
        use_replay = False
    return use_replay, reasons


def select_mamba2_state(
    *,
    spec_config: object | None,
    ssm_cache_dtype: torch.dtype,
    stochastic_rounding: bool,
) -> Mamba2State | ReplayHistory:
    """Apply Mamba2 replay gates and return its selected update policy."""
    use_replay, reasons = _mamba2_replay_gate(spec_config, ssm_cache_dtype, stochastic_rounding)
    for reason in reasons:
        logger.info(reason)
    if os.environ.get("TRTLLM_USE_MAMBA_REPLAY", "1") != "0":
        logger.info("Replay kernel is not changed since TRTLLM_USE_MAMBA_REPLAY=1")
    return create_mamba2_state(
        spec_config=spec_config, use_replay=use_replay, stochastic_rounding=stochastic_rounding
    )


class NemotronHybridCacheManager(MambaHybridCacheManagerV2):
    """Nemotron/Mamba2 geometry, replay selection and seed ownership."""

    _speculative_state: Mamba2State | ReplayHistory | Mamba2ExactReplayState

    @override
    def __init__(self, *args, use_replay_state_update: bool | None = None, **kwargs) -> None:
        self._mamba_ssm_stochastic_rounding = kwargs.pop("mamba_ssm_stochastic_rounding", False)
        self._mamba_chunk_size = kwargs.pop("mamba_chunk_size", None)
        self._requested_replay = use_replay_state_update
        kwargs.setdefault("conv_state_layout", "x_b_c")
        super().__init__(*args, **kwargs)

    def _exact_replay_state(self) -> Mamba2ExactReplayState | None:
        """The FORCE_DETERMINISTIC replay state, or None with the reason logged."""
        if os.environ.get("FORCE_DETERMINISTIC", "0") != "1":
            return None
        reason = _exact_replay_disabled_reason(
            self.spec_config, self.ssm_state_dtype, self._mamba_chunk_size, self._is_disagg
        )
        if reason is None:
            logger.info_once(
                "FORCE_DETERMINISTIC: Mamba2 decode replays the chunked-scan arithmetic from "
                "each sequence's last chunk boundary",
                key="mamba2_exact_replay",
            )
            return Mamba2ExactReplayState(self._mamba_chunk_size)
        logger.warning_once(
            f"FORCE_DETERMINISTIC: Mamba2 keeps the default prefill and decode kernels because {reason}",
            key="mamba2_exact_replay_disabled",
        )
        return None

    @property
    def mamba_replay_num_tokens(self) -> torch.Tensor | None:
        state = getattr(self, "_speculative_state", None)
        return state.replay_num_tokens if isinstance(state, Mamba2ExactReplayState) else None

    @classmethod
    @override
    def _extra_state_bytes_per_rank(cls, model_config, mapping: Mapping, **kwargs) -> int:
        # The exact-replay buffers share every recurrent-state slot.
        return _estimate_exact_replay_bytes_per_slot(model_config, mapping, **kwargs)

    @override
    def _initialize_spec_state(self) -> Mamba2State | ReplayHistory | Mamba2ExactReplayState:
        replay_state = self._exact_replay_state()
        if replay_state is not None:
            return replay_state
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
            from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
                _mamba_effective_tp_size,
            )

            if self.local_num_mamba_layers and self._global_n_groups % _mamba_effective_tp_size(
                self._state_layout.mapping
            ):
                raise ValueError("Replay state groups must be divisible by the effective TP size")
            state.validate(self._state_layout)
        return state

    @classmethod
    @override
    def _estimate_private_replay_bytes_per_slot(
        cls, model_config, mapping: Mapping, **kwargs
    ) -> int:
        spec_config = kwargs.get("spec_config")
        if spec_config is None:
            return 0
        params, local_layers, _ = _get_local_mamba_cache_layout(
            model_config,
            mapping,
            spec_config=spec_config,
            is_draft=kwargs.get("is_draft", False),
            use_separate_draft_kv_cache=kwargs.get("use_separate_draft_kv_cache", False),
        )
        if not local_layers:
            return 0
        ssm_cache_dtype = params.mamba_ssm_cache_dtype or params.dtype
        quant_config = getattr(model_config, "quant_config", None)
        stochastic_rounding = kwargs.get(
            "mamba_ssm_stochastic_rounding",
            getattr(quant_config, "mamba_ssm_stochastic_rounding", False)
            and ssm_cache_dtype == torch.float16,
        )
        use_replay, _ = _mamba2_replay_gate(spec_config, ssm_cache_dtype, stochastic_rounding)
        if not use_replay:
            return 0
        tp_size = _mamba_effective_tp_size(mapping)
        layer_ids = list(range(local_layers))
        context = MambaStateLayout(
            layer_mask=(True,) * local_layers,
            pp_layers=tuple(layer_ids),
            mamba_pp_layers=tuple(layer_ids),
            mapping=mapping,
            conv_state_shape=(),
            ssm_state_shape=(params.num_heads // tp_size, params.head_dim, params.state_size),
            conv_state_dtype=params.dtype,
            ssm_state_dtype=ssm_cache_dtype,
            n_groups_per_rank=params.n_groups // tp_size,
            max_batch_size=0,
            state_index_capacity=0,
            slot_capacity=None,
            spec_config=spec_config,
        )
        state = ReplayHistory(spec_config.tokens_per_gen_step)
        return sum(
            dtype.itemsize * math.prod(shape)
            for layer_id in layer_ids
            for dtype, shape in state.replay_layout(context, layer_id).values()
        )

    def get_replay_state_update_metadata(self) -> ReplayStateUpdateMetadata | None:
        state = self._speculative_state
        return state.get_replay_metadata() if isinstance(state, ReplayHistory) else None

    def get_mamba_ssm_rand_seed(self) -> torch.Tensor | None:
        return self._speculative_state.rand_seed


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
        chunk_size=getattr(config, "chunk_size", None),
    )


__all__ = [
    "Mamba2ExactReplayState",
    "Mamba2LayerCache",
    "Mamba2State",
    "ReplayHistory",
    "NemotronHybridCacheManager",
    "create_mamba2_state",
    "select_mamba2_state",
    "get_nemotron_cache_params",
    "mamba_seed_rank_offset",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated-DeltaNet replay and in-forward snapshot integration for Mamba cache managers."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, NamedTuple, Optional

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
    _is_disagg_context_server,
    _mamba_effective_tp_size,
    _promote_intermediate_states,
    _stack_state_views,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.page_table import PageTableSpec
from tensorrt_llm._torch.pyexecutor.kv_cache.regions import RegionProvider
from tensorrt_llm._torch.utils import (
    is_gdn_flashinfer_replay_enabled,
    is_gdn_flashinfer_replay_required,
    is_gdn_replay_enabled,
)
from tensorrt_llm._utils import get_sm_version, is_sm_100f, prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, DataRole

from .cached_replay import (
    CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
    build_gdn_history_descriptors,
    commit_gdn_cached_replay_history_layers,
)
from .gdn_metadata import GDNMambaMetadata
from .inforward_snapshot import InforwardSnapshotHost, InforwardSnapshots
from .replay_region import REPLAY_STATE_REGION, GDNReplayRegion, ReplayLayout

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

MIN_REPLAY_HISTORY_SIZE = 16
# Per-layer replay roles, by GDNReplayState attribute. The Triton kernel keeps
# a double-buffered history; FlashInfer keeps one ring per slot.
_GDN_REPLAY_HISTORY_ROLES = {
    "old_x": DataRole("gdn_old_x"),
    "old_B": DataRole("gdn_old_B"),
    "old_dt": DataRole("gdn_old_dt"),
}
_GDN_REPLAY_RING_ROLES = {
    "k_ring": DataRole("gdn_k_ring"),
    "u_ring": DataRole("gdn_u_ring"),
    "g_ring": DataRole("gdn_g_ring"),
}
# Held by the first local layer. KVCM copies 16-byte payloads; a row holds
# [prev_num_accepted_tokens, cache_buf_idx, cache_base, padding] as int32.
_GDN_REPLAY_META_ROLE = DataRole("gdn_replay_meta")
_GDN_REPLAY_META_BYTES = 16
_GDN_REPLAY_META_PNAT = 0
_GDN_REPLAY_META_CACHE_BUF_IDX = 1
_GDN_REPLAY_META_CACHE_BASE = 2
# Per-layer roles of the GDN MTP replay (GDNMtpReplayState): one history per
# replay slot holding the accepted window followed by the step's candidates.
_GDN_MTP_REPLAY_HISTORY_ROLES = {
    "old_x": DataRole("gdn_mtp_u"),
    "old_B": DataRole("gdn_mtp_k"),
    "old_dt": DataRole("gdn_mtp_g"),
}
GDN_MTP_REPLAY_HISTORY_SIZE = 16


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


def _advance_ring_start(
    cache_base: torch.Tensor,
    replay_metadata: ReplayStateUpdateMetadata,
    replay_indices: torch.Tensor,
    is_dummy_request: torch.Tensor | None = None,
) -> None:
    """Move each flushed replay slot's ring start past its folded history rows.

    Reads the pre-step history lengths, so it must run before
    ``_advance_replay_state`` overwrites them.
    """
    from .flashinfer_cached_replay import FLASHINFER_REPLAY_RING_SIZE

    slots = replay_indices.long()
    prev_len = replay_metadata.prev_num_accepted_tokens[slots]
    flushed = prev_len + replay_metadata.replay_step_width > replay_metadata.replay_history_size
    if is_dummy_request is not None:
        flushed = flushed & ~is_dummy_request
    base = cache_base[slots]
    cache_base[slots] = torch.where(flushed, (base + prev_len) % FLASHINFER_REPLAY_RING_SIZE, base)


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
class GDNLayerCache(MambaLayerCache):
    """Per-layer GDN intermediate or replay views consumed by its mixer."""

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
    cache_base: torch.Tensor | None = field(
        default=None,
        metadata={"slot_shared": True},
    )
    k_ring: torch.Tensor | None = None
    u_ring: torch.Tensor | None = None
    g_ring: torch.Tensor | None = None


class GDNIntermediateState:
    """Own conventional speculative intermediate states and promotion."""

    def __init__(self) -> None:
        self._ssm_states: tuple[torch.Tensor, ...] = ()
        self._conv_states: tuple[torch.Tensor, ...] = ()
        self._stacked_ssm: torch.Tensor | None = None
        self._stacked_conv: torch.Tensor | None = None
        self.intermediate_ssm: torch.Tensor | None = None
        self.intermediate_conv: torch.Tensor | None = None
        self.intermediate_indices: torch.Tensor | None = None

    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        """Borrow persistent per-layer views until shutdown; allocate owned scratch."""
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

    def _layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        fields: dict[str, torch.Tensor | None] = {}
        if self.intermediate_conv is not None:
            fields["intermediate_conv_window"] = self.intermediate_conv[layer_offset]
        if self.intermediate_ssm is not None:
            fields["intermediate_ssm"] = self.intermediate_ssm[layer_offset]
        return fields

    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None:
        """Plain intermediate buffers have no request-persistent bookkeeping."""

    def update(self, batch: MambaAcceptanceBatch) -> None:
        if self.intermediate_indices is None:
            return
        _promote_intermediate_states(
            self._ssm_states, self.intermediate_ssm, batch, self._stacked_ssm
        )
        _promote_intermediate_states(
            self._conv_states, self.intermediate_conv, batch, self._stacked_conv
        )

    def shutdown(self) -> None:
        self._ssm_states = ()
        self._conv_states = ()
        self._stacked_ssm = None
        self._stacked_conv = None
        self.intermediate_ssm = None
        self.intermediate_conv = None
        self.intermediate_indices = None

    def make_layer_cache(
        self, layer_offset: int, conv: torch.Tensor, temporal: torch.Tensor
    ) -> MambaLayerCache:
        return GDNLayerCache(conv=conv, temporal=temporal, **self._layer_cache_fields(layer_offset))


class GDNReplayState:
    """Own GDN replay buffers, bookkeeping, and checkpoint commits.

    Standalone, the history and bookkeeping are allocated here with one row per
    recurrent-state slot. A V2 manager instead binds views of its own replay
    buffers through ``bind_replay_pool()``; their rows are addressed by the
    replay slots the manager hands to the kernels.
    """

    def __init__(self, tokens_per_gen_step: int, use_flashinfer_ring: bool = False) -> None:
        self._ssm_states: tuple[torch.Tensor, ...] = ()
        self._conv_states: tuple[torch.Tensor, ...] = ()
        self._stacked_conv: torch.Tensor | None = None
        self.intermediate_conv: torch.Tensor | None = None
        self.intermediate_indices: torch.Tensor | None = None
        self.rand_seed: torch.Tensor | None = None
        self._seed_request_counter = 0
        self._seed_rank_offset = 0
        self.use_flashinfer_ring = use_flashinfer_ring
        self.replay_step_width = tokens_per_gen_step
        self.replay_history_size = max(MIN_REPLAY_HISTORY_SIZE, tokens_per_gen_step)
        # Whether the GDN layers verify drafts with the fused kernels, which commit
        # in-layer at every batch size (the manager's gdn_fused_verify).
        self.fused_verify = False
        self.prev_num_accepted_tokens: torch.Tensor | None = None
        self.cache_buf_idx: torch.Tensor | None = None
        # Stacked [layer, slot, ...] tensors, or per-layer views of pool
        # buffers that are not evenly spaced across layers.
        self.old_x: torch.Tensor | list[torch.Tensor] | None = None
        self.old_B: torch.Tensor | list[torch.Tensor] | None = None
        self.old_dt: torch.Tensor | list[torch.Tensor] | None = None
        self.cache_base: torch.Tensor | None = None
        self.k_ring: torch.Tensor | list[torch.Tensor] | None = None
        self.u_ring: torch.Tensor | list[torch.Tensor] | None = None
        self.g_ring: torch.Tensor | list[torch.Tensor] | None = None
        self._replay_meta: torch.Tensor | None = None
        self._pool_layers: list[dict[DataRole, torch.Tensor]] | None = None
        self._history_descriptors: torch.Tensor | None = None
        self._state_descriptors: torch.Tensor | None = None
        self._state_strides: tuple[int, int, int] | None = None
        # Linear draft chains: the accepted position is the accepted token count
        # minus one (see update).
        self._linear_steps = True

    @property
    def intermediate_ssm(self) -> torch.Tensor | None:
        return None

    @property
    def uses_replay(self) -> bool:
        return True

    @property
    def has_bound_states(self) -> bool:
        return self._state_descriptors is not None or self._state_strides is not None

    @property
    def replay_shares_state_slot(self) -> bool:
        """Whether the replay buffers live in the recurrent-state slot.

        The Triton history has replay slots of its own. FlashInfer addresses
        its rings by the recurrent-state index, so ring buffers share the
        recurrent-state slot.
        """
        return self.use_flashinfer_ring

    def validate(self, context: MambaStateLayout) -> None:
        if context.spec_config is None:
            raise ValueError("GDN replay requires speculative decoding")
        if context.n_groups_per_rank <= 0 and context.mamba_pp_layers:
            raise ValueError("GDN replay requires at least one state group per rank")

    def replay_layout(self, context: MambaStateLayout, layer_id: int) -> ReplayLayout:
        """Per-slot shapes of one layer's replay history.

        The first local layer also holds the replay metadata row.
        """
        if not context.mamba_pp_layers:
            return {}
        num_heads, head_dim, d_state = context.ssm_state_shape
        num_groups = context.n_groups_per_rank
        if self.use_flashinfer_ring:
            from .flashinfer_cached_replay import FLASHINFER_REPLAY_RING_SIZE

            ring = FLASHINFER_REPLAY_RING_SIZE
            layout = {
                _GDN_REPLAY_RING_ROLES["k_ring"]: (
                    context.conv_state_dtype,
                    [num_groups, ring, d_state],
                ),
                _GDN_REPLAY_RING_ROLES["u_ring"]: (
                    context.conv_state_dtype,
                    [num_heads, ring, head_dim],
                ),
                _GDN_REPLAY_RING_ROLES["g_ring"]: (torch.float32, [num_heads, ring]),
            }
        else:
            history = self.replay_history_size
            layout = {
                _GDN_REPLAY_HISTORY_ROLES["old_x"]: (
                    context.conv_state_dtype,
                    [2, history, num_heads, head_dim],
                ),
                _GDN_REPLAY_HISTORY_ROLES["old_B"]: (
                    context.conv_state_dtype,
                    [2, history, num_groups, d_state],
                ),
                _GDN_REPLAY_HISTORY_ROLES["old_dt"]: (torch.float32, [2, num_heads, history]),
            }
        if layer_id == context.mamba_pp_layers[0]:
            layout[_GDN_REPLAY_META_ROLE] = (
                torch.int32,
                [_GDN_REPLAY_META_BYTES // torch.int32.itemsize],
            )
        return layout

    def bind_replay_pool(self, layers: list[dict[DataRole, torch.Tensor]]) -> None:
        """Use the manager's per-layer pool views instead of allocating history.

        Each view's first dimension is a replay slot.
        """
        self._pool_layers = layers

    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        self._ssm_states = tuple(ssm_states)
        self._conv_states = tuple(conv_states)
        self._stacked_conv = _stack_state_views(conv_states)
        self._linear_steps = not getattr(context.spec_config, "use_dynamic_tree", False)
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
        else:
            self._allocate_history(context, cache_size, device)
        if not self.use_flashinfer_ring:
            self._bind_state_addressing(context)

    def _allocate_history(
        self, context: MambaStateLayout, cache_size: int, device: torch.device
    ) -> None:
        """Allocate standalone history with one row per recurrent-state slot."""
        layer_ids = context.mamba_pp_layers
        self.prev_num_accepted_tokens = torch.zeros(cache_size, dtype=torch.int32, device=device)
        self.cache_buf_idx = torch.zeros(cache_size, dtype=torch.int32, device=device)
        if self.use_flashinfer_ring:
            self.cache_base = torch.zeros(cache_size, dtype=torch.int32, device=device)
        layout = self.replay_layout(context, layer_ids[0])
        roles = _GDN_REPLAY_RING_ROLES if self.use_flashinfer_ring else _GDN_REPLAY_HISTORY_ROLES
        for name, role in roles.items():
            dtype, shape = layout[role]
            setattr(
                self,
                name,
                torch.zeros([len(layer_ids), cache_size, *shape], dtype=dtype, device=device),
            )

    def _bind_pool_history(self) -> None:
        layers = self._pool_layers
        self._replay_meta = layers[0][_GDN_REPLAY_META_ROLE]
        self.prev_num_accepted_tokens = self._replay_meta[:, _GDN_REPLAY_META_PNAT]
        self.cache_buf_idx = self._replay_meta[:, _GDN_REPLAY_META_CACHE_BUF_IDX]
        if self.use_flashinfer_ring:
            self.cache_base = self._replay_meta[:, _GDN_REPLAY_META_CACHE_BASE]
        roles = _GDN_REPLAY_RING_ROLES if self.use_flashinfer_ring else _GDN_REPLAY_HISTORY_ROLES
        for name, role in roles.items():
            views = [layer[role] for layer in layers]
            stacked = _stack_state_views(views)
            setattr(self, name, stacked if stacked is not None else views)
        if not self.use_flashinfer_ring and any(
            isinstance(getattr(self, name), list) for name in _GDN_REPLAY_HISTORY_ROLES
        ):
            self._history_descriptors = build_gdn_history_descriptors(
                *([layer[role] for layer in layers] for role in _GDN_REPLAY_HISTORY_ROLES.values())
            )

    def _bind_state_addressing(self, context: MambaStateLayout) -> None:
        """Resolve how the all-layer commit addresses every layer's checkpoint."""
        states = self._ssm_states
        if not states:
            return
        state_shape = list(context.ssm_state_shape)
        expected_inner_strides = (
            state_shape[1] * state_shape[2],
            state_shape[2],
            1,
        )
        reference = states[0]
        for state in states:
            if (
                state.dtype != reference.dtype
                or state.device != reference.device
                or list(state.shape[1:]) != state_shape
                or state.stride()[1:] != expected_inner_strides
            ):
                raise RuntimeError(
                    "GDN cached replay requires V2 SSM layers with matching "
                    "dtype, device, shape, and dense inner dimensions."
                )
        pointers = [state.data_ptr() for state in states]
        slot_stride = reference.stride(0)
        layer_stride_bytes = pointers[1] - pointers[0] if len(pointers) > 1 else 0
        has_affine_layout = (
            layer_stride_bytes % reference.element_size() == 0
            and all(state.stride(0) == slot_stride for state in states)
            and all(
                pointer == pointers[0] + layer * layer_stride_bytes
                for layer, pointer in enumerate(pointers)
            )
        )
        if has_affine_layout:
            self._state_strides = (
                len(pointers),
                layer_stride_bytes // reference.element_size(),
                slot_stride,
            )
        else:
            logger.warning_once(
                "V2 GDN state views are not affine; using indirect replay checkpoint addressing",
                key="gdn_cached_replay_v2_indirect_state_layout",
            )
            self._state_descriptors = torch.tensor(
                [(state.data_ptr(), state.stride(0)) for state in states],
                dtype=torch.int64,
                device=reference.device,
            )
        state_layout = "affine" if self._state_strides is not None else "indirect"
        logger.info_once(
            "Configured GDN cached replay commit mode for V2: small-batch fused, "
            f"large-batch all-layer; state layout: {state_layout}",
            key="gdn_cached_replay_v2_commit_mode_fused",
        )

    def _history_buffers(self) -> tuple:
        if self.use_flashinfer_ring:
            return (self.k_ring, self.u_ring, self.g_ring)
        return (self.old_x, self.old_B, self.old_dt)

    def _layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        fields = {
            "mamba_ssm_rand_seed": self.rand_seed,
            "intermediate_conv_window": self.intermediate_conv[layer_offset],
            "prev_num_accepted_tokens": self.prev_num_accepted_tokens,
            "cache_buf_idx": self.cache_buf_idx,
        }
        if self.use_flashinfer_ring:
            fields.update(
                cache_base=self.cache_base,
                k_ring=self.k_ring[layer_offset],
                u_ring=self.u_ring[layer_offset],
                g_ring=self.g_ring[layer_offset],
            )
        else:
            fields.update(
                old_x=self.old_x[layer_offset],
                old_B=self.old_B[layer_offset],
                old_dt=self.old_dt[layer_offset],
            )
        return fields

    def make_layer_cache(
        self,
        layer_offset: int,
        conv: torch.Tensor,
        temporal: torch.Tensor,
    ) -> MambaLayerCache:
        return GDNLayerCache(
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
            if self.cache_base is not None:
                self.cache_base.index_fill_(0, replay_slots, 0)
        for buffer in self._history_buffers():
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

    def update(self, batch: MambaAcceptanceBatch, replay_slots: torch.Tensor | None = None) -> None:
        """Commit, advance the bookkeeping of ``replay_slots`` and promote the accepted windows.

        ``replay_slots`` are the generation rows' replay slots; ``None`` means
        their destination state slots (the FlashInfer ring shares them).
        """
        num_decodes = batch.num_generations
        # Same rule as the GDN managers' gdn_replay_partitioned.
        if (
            not self.use_flashinfer_ring
            and not self.fused_verify
            and num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
        ):
            self.commit_all_layers(batch.attention_metadata, num_decodes)
        metadata = self.get_replay_metadata()
        if metadata is None:
            raise RuntimeError("GDN replay buffers are not bound")
        replay_indices = batch.destination_state_indices if replay_slots is None else replay_slots
        if self.use_flashinfer_ring:
            _advance_ring_start(
                self.cache_base,
                metadata,
                replay_indices,
                batch.is_dummy_request,
            )
        if self._linear_steps and self._stacked_conv is not None:
            self._promote_conv_states_and_advance_replay(batch, metadata, replay_indices)
            return
        _advance_replay_state(
            metadata,
            replay_indices,
            batch.num_accepted_tokens,
            batch.is_dummy_request,
        )
        _promote_intermediate_states(
            self._conv_states, self.intermediate_conv, batch, self._stacked_conv
        )

    def _promote_conv_states_and_advance_replay(
        self,
        batch: MambaAcceptanceBatch,
        metadata: ReplayStateUpdateMetadata,
        replay_indices: torch.Tensor,
    ) -> None:
        """Replay-path update after a linear speculative step, in one launch.

        The cached replay keeps no intermediate SSM states, so only the
        accepted convolution windows of all layers are promoted; the same
        launch advances PNAT and the active history buffer of
        ``replay_indices`` (the arithmetic of ``_advance_replay_state``).
        """
        from tensorrt_llm._torch.modules.fla import replay_promotion

        replay_promotion.promote_conv_and_advance_replay_triton(
            self._stacked_conv,
            self.intermediate_conv,
            batch.source_state_indices,
            batch.num_accepted_tokens,
            batch.destination_state_indices,
            num_accepted_tokens=batch.num_accepted_tokens,
            position_source_is_token_count=True,
            replay_pnat=metadata.prev_num_accepted_tokens,
            replay_cache_buf_idx=metadata.cache_buf_idx,
            dummy_request_mask=batch.is_dummy_request,
            replay_step_width=metadata.replay_step_width,
            replay_history_size=metadata.replay_history_size,
            replay_indices=replay_indices,
        )

    def commit_all_layers(self, attention_metadata: object, num_decodes: int) -> None:
        """Commit every local GDN checkpoint using one partitioned launch."""
        states = self._ssm_states
        if (
            not states
            or (self._state_descriptors is None and self._state_strides is None)
            or self.old_x is None
            or self.old_B is None
            or self.old_dt is None
        ):
            raise RuntimeError(
                "GDN cached replay all-layer commit requires V2 replay state buffers."
            )

        mamba_metadata = attention_metadata.mamba_metadata
        if mamba_metadata.replay_num_decodes != num_decodes:
            raise RuntimeError(
                "GDN replay metadata contains "
                f"{mamba_metadata.replay_num_decodes} decode requests, "
                f"but state update received {num_decodes}."
            )
        state_strides = self._state_strides
        history = (self.old_x, self.old_B, self.old_dt)
        if self._history_descriptors is not None:
            # The descriptors address every layer; the first layer's views give
            # the dtypes and shapes.
            history = tuple(buffer[0] for buffer in history)
        commit_gdn_cached_replay_history_layers(
            ssm_states=states[0],
            ssm_state_descriptors=self._state_descriptors,
            ssm_state_num_layers=(state_strides[0] if state_strides is not None else None),
            ssm_state_layer_stride=(state_strides[1] if state_strides is not None else None),
            ssm_state_slot_stride=(state_strides[2] if state_strides is not None else None),
            old_u=history[0],
            old_k=history[1],
            old_G=history[2],
            replay_work_items=mamba_metadata.replay_work_items[:num_decodes],
            n_writes=mamba_metadata.replay_n_writes,
            history_size=self.replay_history_size,
            history_descriptors=self._history_descriptors,
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
        self.cache_base = None
        self.k_ring = None
        self.u_ring = None
        self.g_ring = None
        self._replay_meta = None
        self._pool_layers = None
        self._history_descriptors = None
        self._state_descriptors = None
        self._state_strides = None


class GDNMtpReplayState(GDNReplayState):
    """GDN MTP replay with the Conv4 update fused into the replay kernel.

    Every layer keeps one history per replay slot: the accepted window of
    ``GDN_MTP_REPLAY_HISTORY_SIZE`` rows followed by the step's candidate rows
    (U and normalized K in BF16, cumulative G in FP32). The first layer's
    metadata row holds the accepted history length. Verification
    (``gdn_mtp_replay``) leaves both checkpoints untouched and keeps the raw
    candidate Conv inputs; after acceptance one ``gdn_mtp_commit`` launch
    advances every layer's Conv window and history length and, when the
    history fills, folds it into the recurrent checkpoint.
    """

    def __init__(self, tokens_per_gen_step: int) -> None:
        if not 1 <= tokens_per_gen_step <= 8:
            raise ValueError("GDN MTP replay supports 1 to 8 tokens per generation step")
        super().__init__(tokens_per_gen_step)
        self.replay_history_size = GDN_MTP_REPLAY_HISTORY_SIZE
        self._key_heads = 0
        self._value_heads = 0
        self._state_view: torch.Tensor | None = None
        self._conv_view: torch.Tensor | None = None
        self._conv_descriptors: torch.Tensor | None = None

    @property
    def has_bound_states(self) -> bool:
        return self._state_view is not None or self._state_descriptors is not None

    @override
    def replay_layout(self, context: MambaStateLayout, layer_id: int) -> ReplayLayout:
        if not context.mamba_pp_layers:
            return {}
        num_heads, head_dim, d_state = context.ssm_state_shape
        capacity = self.replay_history_size + self.replay_step_width
        layout = {
            _GDN_MTP_REPLAY_HISTORY_ROLES["old_x"]: (
                context.conv_state_dtype,
                [capacity, num_heads, head_dim],
            ),
            _GDN_MTP_REPLAY_HISTORY_ROLES["old_B"]: (
                context.conv_state_dtype,
                [capacity, context.n_groups_per_rank, d_state],
            ),
            _GDN_MTP_REPLAY_HISTORY_ROLES["old_dt"]: (torch.float32, [num_heads, capacity]),
        }
        if layer_id == context.mamba_pp_layers[0]:
            layout[_GDN_REPLAY_META_ROLE] = (
                torch.int32,
                [_GDN_REPLAY_META_BYTES // torch.int32.itemsize],
            )
        return layout

    @override
    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        self._ssm_states = tuple(ssm_states)
        self._conv_states = tuple(conv_states)
        self._linear_steps = True
        self._seed_rank_offset = (
            context.mapping.tp_rank * 1_000_003
            + context.mapping.pp_rank * 1_000_033
            + context.mapping.rank * 1_009
        )
        if not context.mamba_pp_layers:
            return
        num_heads, _, _ = context.ssm_state_shape
        self._key_heads = context.n_groups_per_rank
        self._value_heads = num_heads
        cache_size = ssm_states[0].shape[0]
        device = ssm_states[0].device
        # The commit rebuilds the accepted Conv window from the raw candidate
        # inputs, so one row of Conv channels per candidate token suffices.
        self.intermediate_conv = torch.zeros(
            [
                len(context.mamba_pp_layers),
                context.max_batch_size,
                self.replay_step_width,
                context.conv_state_shape[0],
            ],
            dtype=context.conv_state_dtype,
            device=device,
        )
        self.intermediate_indices = torch.arange(
            context.max_batch_size, dtype=torch.int32, device=device
        )
        self.rand_seed = allocate_mamba_seed_buffer(cache_size, self._seed_rank_offset, device)
        if self._pool_layers is not None:
            self._bind_pool_history()
        else:
            self._allocate_history(context, cache_size, device)
        from tensorrt_llm._torch.custom_ops.cute_dsl_gdn_mtp_commit import (
            build_gdn_mtp_history_descriptors,
        )

        self._history_descriptors = build_gdn_mtp_history_descriptors(
            *(self._layer_views(name) for name in _GDN_MTP_REPLAY_HISTORY_ROLES)
        )
        self._bind_state_addressing(context)

    def _layer_views(self, name: str) -> list[torch.Tensor]:
        buffer = getattr(self, name)
        return list(buffer.unbind(0)) if isinstance(buffer, torch.Tensor) else buffer

    @override
    def _allocate_history(
        self, context: MambaStateLayout, cache_size: int, device: torch.device
    ) -> None:
        """Allocate standalone history with one row per recurrent-state slot."""
        layer_ids = context.mamba_pp_layers
        self.prev_num_accepted_tokens = torch.zeros(cache_size, dtype=torch.int32, device=device)
        layout = self.replay_layout(context, layer_ids[0])
        for name, role in _GDN_MTP_REPLAY_HISTORY_ROLES.items():
            dtype, shape = layout[role]
            setattr(
                self,
                name,
                torch.zeros([len(layer_ids), cache_size, *shape], dtype=dtype, device=device),
            )

    @override
    def _bind_pool_history(self) -> None:
        layers = self._pool_layers
        self._replay_meta = layers[0][_GDN_REPLAY_META_ROLE]
        self.prev_num_accepted_tokens = self._replay_meta[:, _GDN_REPLAY_META_PNAT]
        for name, role in _GDN_MTP_REPLAY_HISTORY_ROLES.items():
            views = [layer[role] for layer in layers]
            stacked = _stack_state_views(views)
            setattr(self, name, stacked if stacked is not None else views)

    @override
    def _bind_state_addressing(self, context: MambaStateLayout) -> None:
        """Resolve how the all-layer commit addresses the checkpoints and Conv windows."""
        if not self._ssm_states:
            return
        self._state_view, self._state_descriptors = _gdn_mtp_layer_addressing(
            self._ssm_states, tuple(context.ssm_state_shape), "SSM state"
        )
        self._conv_view, self._conv_descriptors = _gdn_mtp_layer_addressing(
            self._conv_states, tuple(context.conv_state_shape), "Conv state"
        )
        logger.info_once(
            "Configured GDN MTP replay: fused Conv4 replay and all-layer commit; "
            f"state layout: {'affine' if self._state_view is not None else 'indirect'}",
            key="gdn_mtp_replay_v2_layout",
        )

    @override
    def _history_buffers(self) -> tuple:
        return (self.old_x, self.old_B, self.old_dt)

    @override
    def _layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        # old_x / old_B / old_dt carry the U / K / cumulative-G histories and
        # prev_num_accepted_tokens the accepted history length.
        return {
            "mamba_ssm_rand_seed": self.rand_seed,
            "intermediate_conv_window": self.intermediate_conv[layer_offset],
            "prev_num_accepted_tokens": self.prev_num_accepted_tokens,
            "old_x": self.old_x[layer_offset],
            "old_B": self.old_B[layer_offset],
            "old_dt": self.old_dt[layer_offset],
        }

    @override
    def reset_replay_slots(self, replay_slots: torch.Tensor) -> None:
        """Mark ``replay_slots`` empty.

        Only the length is reset: the kernels read no history row at or past
        it and overwrite candidate rows before accepting them.
        """
        if self.prev_num_accepted_tokens is None:
            return
        if self._replay_meta is not None:
            self._replay_meta.index_fill_(0, replay_slots, 0)
        else:
            self.prev_num_accepted_tokens.index_fill_(0, replay_slots, 0)

    @override
    def update(self, batch: MambaAcceptanceBatch, replay_slots: torch.Tensor | None = None) -> None:
        """Commit the accepted Conv windows and histories of every layer in one launch.

        The replay slots come from the step's work items, which the metadata
        snapshots with the history lengths before the layer loop.
        """
        from tensorrt_llm._torch.custom_ops.cute_dsl_gdn_mtp_commit import gdn_mtp_commit

        num_decodes = batch.num_generations
        if num_decodes == 0 or not self._ssm_states:
            return
        if self._history_descriptors is None or not self.has_bound_states:
            raise RuntimeError("GDN MTP replay buffers are not bound")
        if batch.is_dummy_request is None:
            raise RuntimeError("GDN MTP replay commit requires the dummy-request mask")
        mamba_metadata = batch.attention_metadata.mamba_metadata
        if mamba_metadata.replay_num_decodes != num_decodes:
            raise RuntimeError(
                "GDN replay metadata contains "
                f"{mamba_metadata.replay_num_decodes} decode requests, "
                f"but state update received {num_decodes}."
            )
        gdn_mtp_commit(
            ssm_states=(self._state_view if self._state_view is not None else self._ssm_states[0]),
            conv_states=(self._conv_view if self._conv_view is not None else self._conv_states[0]),
            candidate_x=self.intermediate_conv,
            history_descriptors=self._history_descriptors,
            history_len=self.prev_num_accepted_tokens,
            replay_work_items=mamba_metadata.replay_work_items[:num_decodes],
            accepted_tokens=batch.num_accepted_tokens.to(torch.int32),
            is_dummy=batch.is_dummy_request,
            key_heads=self._key_heads,
            value_heads=self._value_heads,
            history_window=self.replay_history_size,
            ssm_state_descriptors=self._state_descriptors,
            conv_state_descriptors=self._conv_descriptors,
        )

    @override
    def commit_all_layers(self, attention_metadata: object, num_decodes: int) -> None:
        raise NotImplementedError("GDN MTP replay commits in update()")

    @override
    def get_replay_metadata(self) -> ReplayStateUpdateMetadata | None:
        if self.prev_num_accepted_tokens is None:
            return None
        return ReplayStateUpdateMetadata(
            prev_num_accepted_tokens=self.prev_num_accepted_tokens,
            cache_buf_idx=None,
            replay_step_width=self.replay_step_width,
            replay_history_size=self.replay_history_size,
        )

    @override
    def shutdown(self) -> None:
        super().shutdown()
        self._state_view = None
        self._conv_view = None
        self._conv_descriptors = None


def _gdn_mtp_layer_addressing(
    states: tuple[torch.Tensor, ...], inner_shape: tuple[int, ...], name: str
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """A slot-major [S, L, ...] view of evenly spaced layer views, else [L, 2] descriptors.

    Descriptor rows are (base address in bytes, slot stride in elements).
    """
    reference = states[0]
    dense = tuple(math.prod(inner_shape[i + 1 :]) for i in range(len(inner_shape)))
    for state in states:
        if (
            state.dtype != reference.dtype
            or state.device != reference.device
            or tuple(state.shape[1:]) != inner_shape
            or state.stride()[1:] != dense
        ):
            raise RuntimeError(
                f"GDN MTP replay requires {name} layers with matching dtype, device, "
                "shape and dense inner dimensions"
            )
    stacked = _stack_state_views(list(states))
    if stacked is not None:
        return stacked.transpose(0, 1), None
    logger.warning_once(
        f"V2 GDN {name} views are not affine; using indirect GDN MTP commit addressing",
        key=f"gdn_mtp_replay_indirect_{name}",
    )
    descriptors = torch.tensor(
        [(state.data_ptr(), state.stride(0)) for state in states],
        dtype=torch.int64,
        device=reference.device,
    )
    return None, descriptors


def _gdn_mtp_replay_unsupported_reason(
    layout: MambaStateLayout, spec_config: object | None
) -> Optional[str]:
    """Why the GDN MTP replay kernels cannot serve ``layout``, or None when they can."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    num_heads, head_dim, d_state = layout.ssm_state_shape
    groups = layout.n_groups_per_rank
    if not is_sm_100f(get_sm_version()):
        return "requires an SM100/SM103 GPU"
    if not IS_CUTLASS_DSL_AVAILABLE:
        return "requires CUTLASS DSL"
    if layout.conv_state_layout != "q_k_v":
        return "requires the q_k_v convolution state layout"
    if layout.conv_state_dtype != torch.bfloat16 or layout.ssm_state_dtype != torch.bfloat16:
        return "requires bf16 model and SSM state dtypes"
    if head_dim != 128 or d_state != 128:
        return "requires key and value head size 128"
    if groups <= 0 or num_heads % groups != 0:
        return "requires an integral value/key head ratio"
    if layout.conv_state_shape and tuple(layout.conv_state_shape) != (
        (2 * groups * d_state + num_heads * head_dim),
        3,
    ):
        return "requires a 4-tap convolution over the q|k|v channels"
    if spec_config is not None and spec_config.tokens_per_gen_step > 8:
        return "supports at most 8 tokens per generation step"
    return None


def _gdn_replay_gate(
    spec_config: object | None, ssm_cache_dtype: torch.dtype
) -> tuple[bool, list[str]]:
    """Return whether GDN replay is selected, with the reasons it is not."""
    use_replay = spec_config is not None and get_sm_version() >= 80
    reasons = []
    if spec_config is None:
        reasons.append("GDN replay kernel requires speculative decoding; using non-replay path")
    elif spec_config.tokens_per_gen_step > 8:
        reasons.append(
            "GDN cached replay supports at most 8 tokens per generation step; using non-replay path"
        )
        use_replay = False
    if spec_config is not None and (getattr(spec_config, "use_dynamic_tree", False)):
        reasons.append("GDN replay kernel incompatible with tree attention; using legacy MTP path")
        use_replay = False
    if ssm_cache_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        reasons.append(
            "GDN replay kernel does not support quantized SSM cache dtype "
            f"{ssm_cache_dtype}; using legacy MTP path"
        )
        use_replay = False
    if not is_gdn_replay_enabled():
        use_replay = False
    return use_replay, reasons


def select_gdn_replay_state(
    *,
    spec_config: object | None,
    ssm_cache_dtype: torch.dtype,
    manager_cls: type,
) -> GDNReplayState | None:
    """Apply GDN replay feature gates and return the policy when eligible."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        CppMambaHybridCacheManager,
        MambaHybridCacheManagerV2,
    )

    use_replay, reasons = _gdn_replay_gate(spec_config, ssm_cache_dtype)
    for reason in reasons:
        logger.info(reason)
    if use_replay and not issubclass(
        manager_cls,
        (CppMambaHybridCacheManager, MambaHybridCacheManagerV2),
    ):
        logger.info(
            "GDN replay requires C++ V1 or V2 Mamba cache manager; "
            f"{manager_cls.__name__} was selected, so the non-replay MTP path will be used"
        )
        use_replay = False
    logger.info(f"GDN replay state update: {'ENABLED' if use_replay else 'DISABLED'}")
    return GDNReplayState(spec_config.tokens_per_gen_step) if use_replay else None


def _flashinfer_ring_replay_reason(
    layout: MambaStateLayout, spec_config: object | None, replay_enabled: bool
) -> tuple[bool, Optional[str]]:
    """Return whether ring replay is requested, and why it cannot run if it cannot."""
    # Managers without local GDN layers (e.g. the one-model MTP draft) hold no history.
    if not is_gdn_flashinfer_replay_enabled() or not layout.mamba_pp_layers:
        return False, None
    from .flashinfer_cached_replay import is_flashinfer_cached_replay_available

    _, head_dim, d_state = layout.ssm_state_shape
    reason = None
    if not replay_enabled:
        reason = "requires GDN replay to be enabled"
    elif layout.conv_state_layout != "q_k_v":
        reason = "requires the q_k_v convolution state layout"
    elif not is_flashinfer_cached_replay_available():
        reason = "needs FlashInfer and an SM90/SM100/SM103 GPU"
    elif layout.conv_state_dtype != torch.bfloat16 or layout.ssm_state_dtype != torch.bfloat16:
        reason = "requires bf16 model and SSM state dtypes"
    elif d_state != 128 or head_dim != 128:
        reason = "requires key and value head size 128"
    elif spec_config.tokens_per_gen_step not in (4, 8):
        reason = (
            f"supports 4 or 8 tokens per generation step, got {spec_config.tokens_per_gen_step}"
        )
    return True, reason


def _use_flashinfer_ring_replay(manager: MambaHybridCacheManagerV2, replay_enabled: bool) -> bool:
    """Whether GDN replay should use the FlashInfer ring-buffer kernel.

    Decided once at load: the ring and double-buffer history layouts differ,
    so the kernel cannot be switched per call.
    """
    requested, reason = _flashinfer_ring_replay_reason(
        manager._state_layout, manager.spec_config, replay_enabled
    )
    if not requested:
        return False
    if reason is None:
        return True
    if is_gdn_flashinfer_replay_required():
        raise RuntimeError(
            f"TRTLLM_USE_GDN_FLASHINFER_REPLAY=force but FlashInfer GDN replay {reason}"
        )
    logger.warning(
        f"TRTLLM_USE_GDN_FLASHINFER_REPLAY=1 ignored: FlashInfer GDN replay {reason}; "
        "using the Triton replay kernel"
    )
    return False


def create_gdn_state(
    manager: MambaHybridCacheManagerV2,
    use_replay: bool | None,
    allow_flashinfer_ring: bool = False,
) -> GDNIntermediateState | GDNReplayState:
    """Shared selection for the Qwen3.5 and Qwen4 cache managers."""
    if use_replay is False:
        state = None
    elif use_replay is True:
        if manager.spec_config is None:
            raise ValueError("GDN replay requires speculative decoding")
        state = GDNReplayState(manager.spec_config.tokens_per_gen_step)
    else:
        state = select_gdn_replay_state(
            spec_config=manager.spec_config,
            ssm_cache_dtype=manager.ssm_state_dtype,
            manager_cls=type(manager),
        )
    if allow_flashinfer_ring and _use_flashinfer_ring_replay(manager, state is not None):
        logger.info("GDN replay kernel: FlashInfer ring buffer")
        state = GDNReplayState(manager.spec_config.tokens_per_gen_step, use_flashinfer_ring=True)
    elif state is not None and manager.gdn_mtp_replay:
        # The model's GDN layers can verify with the GDN MTP replay kernels;
        # configurations they cannot serve keep the cached replay.
        reason = _gdn_mtp_replay_unsupported_reason(manager._state_layout, manager.spec_config)
        if reason is None:
            logger.info("GDN replay kernel: GDN MTP replay")
            state = GDNMtpReplayState(manager.spec_config.tokens_per_gen_step)
        else:
            logger.info(f"GDN MTP replay {reason}; using the cached replay")
    if state is None:
        return GDNIntermediateState()
    state.fused_verify = manager.gdn_fused_verify
    return state


def validate_gdn_layout(
    manager: MambaHybridCacheManagerV2, state: GDNIntermediateState | GDNReplayState
) -> None:
    if isinstance(state, GDNReplayState):
        if manager.local_num_mamba_layers and manager._global_n_groups % _mamba_effective_tp_size(
            manager._state_layout.mapping
        ):
            raise ValueError("GDN replay groups must be divisible by the effective TP size")
        state.validate(manager._state_layout)


def _estimated_gdn_replay_state(
    model_config, mapping: Mapping, **kwargs
) -> tuple[GDNReplayState, MambaStateLayout] | None:
    """The replay state and layout a GDN manager built from these arguments declares."""
    spec_config = kwargs.get("spec_config")
    if spec_config is None or _is_disagg_context_server(kwargs.get("is_disagg", False)):
        return None
    params, local_layers, _ = _get_local_mamba_cache_layout(
        model_config,
        mapping,
        spec_config=spec_config,
        is_draft=kwargs.get("is_draft", False),
        use_separate_draft_kv_cache=kwargs.get("use_separate_draft_kv_cache", False),
    )
    if not local_layers:
        return None
    ssm_cache_dtype = params.mamba_ssm_cache_dtype or params.dtype
    use_replay, _ = _gdn_replay_gate(spec_config, ssm_cache_dtype)
    if not use_replay:
        return None
    tp_size = _mamba_effective_tp_size(mapping)
    layer_ids = tuple(range(local_layers))
    layout = MambaStateLayout(
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
        conv_state_layout="q_k_v",
    )
    requested, reason = _flashinfer_ring_replay_reason(layout, spec_config, use_replay)
    if requested and reason is None:
        state = GDNReplayState(spec_config.tokens_per_gen_step, use_flashinfer_ring=True)
    elif (
        kwargs.get("gdn_mtp_replay", False)
        and _gdn_mtp_replay_unsupported_reason(layout, spec_config) is None
    ):
        state = GDNMtpReplayState(spec_config.tokens_per_gen_step)
    else:
        state = GDNReplayState(spec_config.tokens_per_gen_step)
    return state, layout


def estimate_gdn_replay_bytes_per_slot(
    model_config, mapping: Mapping, *, flashinfer_ring: bool, **kwargs
) -> int:
    """Estimate the replay bytes per slot of the Triton history or of the FlashInfer rings.

    Returns 0 unless the manager selects the requested kind of replay.
    """
    estimate = _estimated_gdn_replay_state(model_config, mapping, **kwargs)
    if estimate is None:
        return 0
    state, layout = estimate
    if state.use_flashinfer_ring != flashinfer_ring:
        return 0
    return sum(
        dtype.itemsize * math.prod(shape)
        for layer_id in layout.mamba_pp_layers
        for dtype, shape in state.replay_layout(layout, layer_id).values()
    )


class Qwen35HybridCacheManager(InforwardSnapshotHost, MambaHybridCacheManagerV2):
    """Qwen3.5/Qwen3Next recurrent state with GDN replay and in-forward snapshots.

    ``GDNReplayRegion`` declares and indexes the replay history; the FlashInfer
    ring buffers are buffers of the recurrent layers instead. The GDN mixer
    implements the in-forward snapshot split, whose staging rows and per-step
    plan ``InforwardSnapshots`` keeps.
    """

    _skips_steady_state_index_upload = True
    mamba_metadata_cls = GDNMambaMetadata
    _replay_region: GDNReplayRegion | None = None
    _context_only = False
    # Whether the model's GDN layers verify drafts with the fused kernels, which
    # commit in-layer at every batch size.
    gdn_fused_verify = False
    # Whether the model's GDN layers can verify drafts with the GDN MTP replay
    # kernels (modules/mamba/gdn_mixer.py) instead of the cached replay.
    gdn_mtp_replay = False

    @override
    def __init__(
        self,
        *args,
        use_replay_state_update: bool | None = None,
        gdn_fused_verify: bool = False,
        gdn_mtp_replay: bool = False,
        **kwargs,
    ) -> None:
        kwargs.pop("mamba_ssm_stochastic_rounding", False)
        self._requested_replay = use_replay_state_update
        self.gdn_fused_verify = gdn_fused_verify
        self.gdn_mtp_replay = gdn_mtp_replay
        # A disaggregated context server never decodes and keeps no replay history.
        self._context_only = _is_disagg_context_server(kwargs.get("is_disagg", False))
        self._replay_region = None
        self._inforward_snapshots = InforwardSnapshots()
        kwargs.setdefault("conv_state_layout", "q_k_v")
        super().__init__(*args, **kwargs)

    def get_replay_dummy_request_mask(self) -> torch.Tensor | None:
        """The batch-ordered dummy-request mask of the last state-index setup.

        The GDN MTP replay skips dummy rows; the buffer address stays fixed.
        """
        return self._dummy_request_mask

    def gdn_replay_partitioned(self, num_decodes: int) -> bool:
        """Whether a GDN decode batch of num_decodes requests takes the
        partitioned cached replay and one all-layer checkpoint commit; other
        batches commit inside each layer, as the fused target verify of the
        model's GDN layers (gdn_fused_verify) does at every batch size."""
        return not self.gdn_fused_verify and num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE

    @override
    def prepare_resources(self, scheduled_batch: ScheduledRequests) -> None:
        super().prepare_resources(scheduled_batch)
        self._plan_inforward_snapshots(scheduled_batch)

    @override
    def _initialize_spec_state(self) -> GDNIntermediateState | GDNReplayState:
        state = create_gdn_state(self, self._requested_replay, allow_flashinfer_ring=True)

        validate_gdn_layout(self, state)
        return state

    @override
    def _region_providers(self) -> Sequence[RegionProvider]:
        providers = super()._region_providers()
        if (
            not isinstance(self._speculative_state, GDNReplayState)
            or self._context_only
            or not self.mamba_pp_layers
        ):
            return providers
        self._replay_region = GDNReplayRegion(
            self._speculative_state,
            self._state_layout,
            self.layer_offsets,
            self._index_mapper_capacity,
            self.kv_cache_config.max_util_for_resume,
        )
        return (*providers, self._replay_region)

    @override
    def _get_recurrent_buffer_configs(self, global_layer_id: int) -> list[BufferConfig]:
        buffers = super()._get_recurrent_buffer_configs(global_layer_id)
        if self._replay_region is not None:
            buffers += self._replay_region.shared_buffers(global_layer_id)
        return buffers

    @override
    def page_table_spec(self) -> PageTableSpec:
        """The generation worker starts the replay history afresh."""
        spec = super().page_table_spec()
        if self._replay_region is None:
            return spec
        return replace(
            spec, untransferred_roles=spec.untransferred_roles | self._replay_region.roles
        )

    @override
    def get_disagg_ignored_roles(self) -> frozenset[DataRole]:
        """The generation worker starts the replay history afresh."""
        roles = super().get_disagg_ignored_roles()
        if self._replay_region is not None:
            roles |= self._replay_region.roles
        return roles

    @override
    def _setup_model_state(self) -> None:
        region = self._replay_region
        if region is not None:
            region.bind(
                self._get_state_buffer,
                self.layer_offsets,
                self.get_region_group_ids(REPLAY_STATE_REGION),
                self._host_state_indices.shape[0],
                self.cuda_state_indices.device,
            )
        super()._setup_model_state()
        self._setup_inforward_snapshots()

    @property
    def replay_layer_group_id(self) -> int | None:
        """Layer group of the replay history's own slots; None when it shares the state slots."""
        region = self._replay_region
        return region.group_id if region is not None else None

    @property
    def cuda_replay_state_indices(self) -> torch.Tensor:
        region = self._replay_region
        return (
            region.cuda_indices(self.cuda_state_indices)
            if region is not None
            else self.cuda_state_indices
        )

    def get_replay_state_indices(
        self, request_ids: list[int] | None = None
    ) -> torch.Tensor | list[int]:
        """Replay slots like get_state_indices(), without touching the dummy mask."""
        if request_ids is None:
            return self.cuda_replay_state_indices
        region = self._replay_region
        if region is None or region.group_id is None:
            return [self._request_id_to_state_index[rid] for rid in request_ids]
        return region.request_indices(request_ids)

    @override
    def _setup_state_indices(self, requests, num_contexts: int = 0) -> None:
        super()._setup_state_indices(requests, num_contexts)
        if self._replay_region is not None and self.local_num_mamba_layers:
            self._replay_region.prepare(
                requests,
                self.kv_cache_map,
                self.cuda_state_indices,
                self._host_state_indices,
                num_contexts,
                self._skips_steady_state_index_upload,
            )

    @override
    def _reset_context_mamba_slots(self, num_contexts: int) -> None:
        if self._replay_region is None:
            super()._reset_context_mamba_slots(num_contexts)
        else:
            self._replay_region.reset_context(
                num_contexts, self.cuda_state_indices, self._host_state_indices
            )

    @override
    def _update_model_state(self, batch: MambaAcceptanceBatch) -> None:
        if self._replay_region is None:
            super()._update_model_state(batch)
        else:
            self._replay_region.update(batch, self.cuda_state_indices)

    @override
    def free_resources(self, request, pin_on_release: bool = False) -> None:
        if self._replay_region is not None:
            self._replay_region.release(request.py_request_id)
        super().free_resources(request, pin_on_release)

    @override
    def shutdown(self) -> None:
        if self._replay_region is not None:
            self._replay_region.shutdown()
        self._inforward_snapshots.shutdown()
        super().shutdown()

    @classmethod
    @override
    def _estimate_fixed_region_bytes(
        cls,
        model_config,
        mapping: Mapping,
        *,
        max_batch_size: int,
        kv_cache_config,
        **kwargs,
    ) -> int:
        if _is_disagg_context_server(kwargs.get("is_disagg", False)):
            return 0
        return GDNReplayRegion.estimate_fixed_bytes(
            estimate_gdn_replay_bytes_per_slot(
                model_config, mapping, flashinfer_ring=False, **kwargs
            ),
            mapping,
            max_batch_size,
            kv_cache_config.max_util_for_resume,
            **kwargs,
        )

    @classmethod
    @override
    def _extra_state_bytes_per_rank(cls, model_config, mapping: Mapping, **kwargs) -> int:
        # FlashInfer rings share every recurrent-state slot.
        return estimate_gdn_replay_bytes_per_slot(
            model_config, mapping, flashinfer_ring=True, **kwargs
        )

    def get_replay_state_update_metadata(self) -> ReplayStateUpdateMetadata | None:
        state = self._speculative_state
        return state.get_replay_metadata() if isinstance(state, GDNReplayState) else None

    @property
    def use_gdn_flashinfer_ring_replay(self) -> bool:
        state = self._speculative_state
        return isinstance(state, GDNReplayState) and state.use_flashinfer_ring

    @property
    def use_gdn_cached_replay_all_layer_commit(self) -> bool:
        state = self._speculative_state
        return (
            isinstance(state, GDNReplayState)
            and not isinstance(state, GDNMtpReplayState)
            and state.has_bound_states
        )

    @property
    def use_gdn_mtp_replay(self) -> bool:
        return isinstance(self._speculative_state, GDNMtpReplayState)


def get_gdn_cache_params(config, *, spec_config=None, quant_config=None):
    """Shared Qwen3.5/Qwen4 GDN geometry; attention and PLE stay model-owned."""
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        build_mamba_kv_cache_params,
        get_qwen3_hybrid_layer_masks,
    )

    attention, recurrent = get_qwen3_hybrid_layer_masks(config)
    return build_mamba_kv_cache_params(
        config,
        state_size=config.linear_key_head_dim,
        conv_kernel=config.linear_conv_kernel_dim,
        num_heads=config.linear_num_value_heads,
        n_groups=config.linear_num_key_heads,
        head_dim=config.linear_value_head_dim,
        mamba_mask=recurrent,
        target_full_attn_mask=attention,
        spec_config=spec_config,
        quant_config=quant_config,
    )


__all__ = [
    "GDNIntermediateState",
    "GDNLayerCache",
    "GDNMtpReplayState",
    "GDNReplayState",
    "Qwen35HybridCacheManager",
    "create_gdn_state",
    "estimate_gdn_replay_bytes_per_slot",
    "select_gdn_replay_state",
    "validate_gdn_layout",
    "get_gdn_cache_params",
]

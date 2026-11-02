# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated-DeltaNet replay integration for Mamba cache managers."""

from __future__ import annotations

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
    _mamba_effective_tp_size,
    _promote_intermediate_states,
    _stack_state_views,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.utils import (
    is_gdn_flashinfer_replay_enabled,
    is_gdn_flashinfer_replay_required,
    is_gdn_replay_enabled,
)
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.logger import logger

from .cached_replay import (
    CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
    commit_gdn_cached_replay_history_layers,
)

MIN_REPLAY_HISTORY_SIZE = 16


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


def _advance_ring_start(
    cache_base: torch.Tensor,
    replay_metadata: ReplayStateUpdateMetadata,
    state_indices: torch.Tensor,
    is_dummy_request: torch.Tensor | None = None,
) -> None:
    """Move each flushed slot's ring start past its folded history rows.

    Reads the pre-step history lengths, so it must run before
    ``_advance_replay_state`` overwrites them.
    """
    from .flashinfer_cached_replay import FLASHINFER_REPLAY_RING_SIZE

    slots = state_indices.long()
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
    """Own GDN replay buffers, bookkeeping, and checkpoint commits."""

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
        self.old_x: torch.Tensor | None = None
        self.old_B: torch.Tensor | None = None
        self.old_dt: torch.Tensor | None = None
        self.old_dA_cumsum: torch.Tensor | None = None
        self.cache_base: torch.Tensor | None = None
        self.k_ring: torch.Tensor | None = None
        self.u_ring: torch.Tensor | None = None
        self.g_ring: torch.Tensor | None = None
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

    def validate(self, context: MambaStateLayout) -> None:
        if context.spec_config is None:
            raise ValueError("GDN replay requires speculative decoding")
        if context.n_groups_per_rank <= 0 and context.mamba_pp_layers:
            raise ValueError("GDN replay requires at least one state group per rank")

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

        num_heads, head_dim, d_state = context.ssm_state_shape
        self.prev_num_accepted_tokens = torch.zeros(cache_size, dtype=torch.int32, device=device)
        self.cache_buf_idx = torch.zeros(cache_size, dtype=torch.int32, device=device)
        if self.use_flashinfer_ring:
            from .flashinfer_cached_replay import FLASHINFER_REPLAY_RING_SIZE

            ring_shape = [len(context.mamba_pp_layers), cache_size]
            self.cache_base = torch.zeros(cache_size, dtype=torch.int32, device=device)
            self.k_ring = torch.zeros(
                ring_shape + [context.n_groups_per_rank, FLASHINFER_REPLAY_RING_SIZE, d_state],
                dtype=context.conv_state_dtype,
                device=device,
            )
            self.u_ring = torch.zeros(
                ring_shape + [num_heads, FLASHINFER_REPLAY_RING_SIZE, head_dim],
                dtype=context.conv_state_dtype,
                device=device,
            )
            self.g_ring = torch.zeros(
                ring_shape + [num_heads, FLASHINFER_REPLAY_RING_SIZE],
                dtype=torch.float32,
                device=device,
            )
            return
        common_replay_shape = [len(context.mamba_pp_layers), cache_size, 2]
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
                old_dA_cumsum=self.old_dA_cumsum[layer_offset],
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

    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None:
        if self.prev_num_accepted_tokens is None:
            return
        self.prev_num_accepted_tokens.index_fill_(0, slots, 0)
        self.cache_buf_idx.index_fill_(0, slots, 0)
        if self.use_flashinfer_ring:
            self.cache_base.index_fill_(0, slots, 0)
            history_buffers = (self.k_ring, self.u_ring, self.g_ring)
        else:
            history_buffers = (self.old_x, self.old_B, self.old_dt, self.old_dA_cumsum)
        for buffer in history_buffers:
            buffer.index_fill_(1, slots, 0)
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
        num_decodes = batch.num_generations
        # Same rule as MambaHybridCacheManager.gdn_replay_partitioned.
        if (
            not self.use_flashinfer_ring
            and not self.fused_verify
            and num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
        ):
            self.commit_all_layers(batch.attention_metadata, num_decodes)
        metadata = self.get_replay_metadata()
        if metadata is None:
            raise RuntimeError("GDN replay buffers are not bound")
        if self.use_flashinfer_ring:
            _advance_ring_start(
                self.cache_base,
                metadata,
                batch.destination_state_indices,
                batch.is_dummy_request,
            )
        if self._linear_steps and self._stacked_conv is not None:
            self._promote_conv_states_and_advance_replay(batch, metadata)
            return
        _advance_replay_state(
            metadata,
            batch.destination_state_indices,
            batch.num_accepted_tokens,
            batch.is_dummy_request,
        )
        _promote_intermediate_states(
            self._conv_states, self.intermediate_conv, batch, self._stacked_conv
        )

    def _promote_conv_states_and_advance_replay(
        self, batch: MambaAcceptanceBatch, metadata: ReplayStateUpdateMetadata
    ) -> None:
        """Replay-path update after a linear speculative step, in one launch.

        The cached replay keeps no intermediate SSM states, so only the
        accepted convolution windows of all layers are promoted; the same
        launch advances PNAT and the active history buffer (the arithmetic of
        ``_advance_replay_state``).
        """
        from tensorrt_llm._torch.pyexecutor.kv_cache import mamba_cache_manager

        mamba_cache_manager._promote_mamba_state_triton(
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
        commit_gdn_cached_replay_history_layers(
            ssm_states=states[0],
            ssm_state_descriptors=self._state_descriptors,
            ssm_state_num_layers=(state_strides[0] if state_strides is not None else None),
            ssm_state_layer_stride=(state_strides[1] if state_strides is not None else None),
            ssm_state_slot_stride=(state_strides[2] if state_strides is not None else None),
            old_u=self.old_x,
            old_k=self.old_B,
            old_G=self.old_dt,
            replay_work_items=mamba_metadata.replay_work_items[:num_decodes],
            n_writes=mamba_metadata.replay_n_writes,
            history_size=self.replay_history_size,
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
        self.cache_base = None
        self.k_ring = None
        self.u_ring = None
        self.g_ring = None
        self._state_descriptors = None
        self._state_strides = None


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

    use_replay = spec_config is not None and get_sm_version() >= 80
    if spec_config is None:
        logger.info("GDN replay kernel requires speculative decoding; using non-replay path")
    elif spec_config.tokens_per_gen_step > 8:
        logger.info(
            "GDN cached replay supports at most 8 tokens per generation step; using non-replay path"
        )
        use_replay = False
    if spec_config is not None and (getattr(spec_config, "use_dynamic_tree", False)):
        logger.info("GDN replay kernel incompatible with tree attention; using legacy MTP path")
        use_replay = False
    if ssm_cache_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        logger.info(
            "GDN replay kernel does not support quantized SSM cache dtype "
            f"{ssm_cache_dtype}; using legacy MTP path"
        )
        use_replay = False
    if not is_gdn_replay_enabled():
        use_replay = False
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


def _use_flashinfer_ring_replay(manager: MambaHybridCacheManagerV2, replay_enabled: bool) -> bool:
    """Whether GDN replay should use the FlashInfer ring-buffer kernel.

    Decided once at load: the ring and double-buffer history layouts differ,
    so the kernel cannot be switched per call.
    """
    layout = manager._state_layout
    # Managers without local GDN layers (e.g. the one-model MTP draft) hold no history.
    if not is_gdn_flashinfer_replay_enabled() or not layout.mamba_pp_layers:
        return False
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
    elif manager.spec_config.tokens_per_gen_step not in (4, 8):
        reason = (
            "supports 4 or 8 tokens per generation step, got "
            f"{manager.spec_config.tokens_per_gen_step}"
        )
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


class Qwen35HybridCacheManager(MambaHybridCacheManagerV2):
    """Qwen3.5/Qwen3Next recurrent state with shared GDN replay."""

    # The GDN mixer implements the in-forward snapshot split.
    _supports_inforward_snapshot = True
    _skips_steady_state_index_upload = True

    @override
    def __init__(self, *args, use_replay_state_update: bool | None = None, **kwargs) -> None:
        kwargs.pop("mamba_ssm_stochastic_rounding", False)
        self._requested_replay = use_replay_state_update
        kwargs.setdefault("conv_state_layout", "q_k_v")
        super().__init__(*args, **kwargs)

    @override
    def _initialize_spec_state(self) -> GDNIntermediateState | GDNReplayState:
        state = create_gdn_state(self, self._requested_replay, allow_flashinfer_ring=True)

        validate_gdn_layout(self, state)
        return state

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
        return isinstance(state, GDNReplayState) and state.has_bound_states


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
    "GDNReplayState",
    "Qwen35HybridCacheManager",
    "create_gdn_state",
    "select_gdn_replay_state",
    "validate_gdn_layout",
    "get_gdn_cache_params",
]

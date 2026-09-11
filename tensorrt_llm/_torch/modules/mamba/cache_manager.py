# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba2-owned speculative and replay cache state."""

import os
from dataclasses import dataclass, field
from typing import Iterable

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaCacheBuildContext,
    MambaLayerCache,
    MambaStateUpdateBatch,
    MambaStateUpdateStrategy,
    ReplayStateUpdateMetadata,
    SpeculativeMambaLayerCache,
)
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

MIN_REPLAY_HISTORY_SIZE = 16

_MAMBA_SSM_SEED_MASK = (1 << 62) - 1
_MAMBA_SSM_UINT64_MASK = (1 << 64) - 1
_MAMBA_SSM_SEED_BASE = 0x6A09E667F3BCC908
_MAMBA_SSM_SEED_MIX_COUNTER = 0x2545F4914F6CDD1D
_MAMBA_SSM_SEED_MIX_SLOT = 0x1B873593CC9E2D51
_MAMBA_SSM_SEED_MIX_RANK = 0x9E3779B97F4A7C15


@dataclass(frozen=True, kw_only=True)
class Mamba2ReplayLayerCache(SpeculativeMambaLayerCache):
    """Mamba2 compact-replay tensors for one recurrent layer."""

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


def mamba_seed_rank_offset(mapping: Mapping) -> int:
    """Return a distinct deterministic seed offset for this parallel rank."""
    return mapping.tp_rank * 1_000_003 + mapping.pp_rank * 1_000_033 + mapping.rank * 1_009


def advance_replay_state(
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


class IntermediateMambaStateUpdateStrategy(MambaStateUpdateStrategy):
    """Own conventional speculative intermediate states and promotion."""

    def __init__(self) -> None:
        self.intermediate_ssm: torch.Tensor | None = None
        self.intermediate_conv: torch.Tensor | None = None
        self.intermediate_indices: torch.Tensor | None = None
        self.rand_seed: torch.Tensor | None = None
        self._seed_request_counter = 0
        self._seed_rank_offset = 0

    @property
    def key(self) -> str:
        return "mamba_intermediate_state"

    def bind(self, context: MambaCacheBuildContext, manager: object) -> None:
        self._seed_rank_offset = context.seed_rank_offset
        if not context.mamba_pp_layers:
            self._publish_compatibility_views(manager)
            return

        states = getattr(manager, "all_ssm_states")
        cache_size = states[0].shape[0]
        device = states[0].device
        if context.stochastic_rounding:
            self.rand_seed = allocate_mamba_seed_buffer(
                cache_size, context.seed_rank_offset, device
            )

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
        self._publish_compatibility_views(manager)

    def _publish_compatibility_views(self, manager: object) -> None:
        # These aliases preserve the established kernel-facing manager API.
        setattr(manager, "_use_replay_state_update", self.uses_replay)
        setattr(manager, "intermediate_ssm_states", self.intermediate_ssm)
        setattr(manager, "intermediate_conv_states", self.intermediate_conv)
        setattr(manager, "intermediate_state_indices", self.intermediate_indices)
        setattr(manager, "mamba_ssm_rand_seed", self.rand_seed)

    def layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        fields: dict[str, torch.Tensor | None] = {}
        if self.rand_seed is not None:
            fields["mamba_ssm_rand_seed"] = self.rand_seed
        if self.intermediate_conv is not None:
            fields["intermediate_conv_window"] = self.intermediate_conv[layer_offset]
        if self.intermediate_ssm is not None:
            fields["intermediate_ssm"] = self.intermediate_ssm[layer_offset]
        return fields

    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None:
        if self.rand_seed is None:
            return
        self._seed_request_counter += 1
        seeds = [
            compute_deterministic_mamba_seed(
                self._seed_request_counter, slot, self._seed_rank_offset
            )
            for slot in host_slots
        ]
        self.rand_seed[slots] = torch.tensor(seeds, dtype=torch.int64, device=self.rand_seed.device)

    def update(self, batch: MambaStateUpdateBatch, manager: object) -> bool:
        if self.intermediate_indices is None:
            return False
        # Import through the compatibility facade so existing test patches and
        # out-of-tree diagnostics keep observing the promotion call.
        from tensorrt_llm._torch.pyexecutor.kv_cache import mamba_cache_manager

        for layer_offset, destination in enumerate(getattr(manager, "all_ssm_states")):
            mamba_cache_manager._promote_mamba_state_triton(
                destination.unsqueeze(0),
                self.intermediate_ssm[layer_offset : layer_offset + 1],
                batch.source_state_indices,
                batch.accepted_positions,
                batch.destination_state_indices,
            )
        self._promote_conv(batch, manager)
        return True

    def source_state_indices(self, count: int) -> torch.Tensor:
        if self.intermediate_indices is None:
            raise RuntimeError("Mamba intermediate-state strategy is not bound")
        return self.intermediate_indices[:count]

    def _promote_conv(self, batch: MambaStateUpdateBatch, manager: object) -> None:
        from tensorrt_llm._torch.pyexecutor.kv_cache import mamba_cache_manager

        for layer_offset, destination in enumerate(getattr(manager, "all_conv_states")):
            mamba_cache_manager._promote_mamba_state_triton(
                destination.unsqueeze(0),
                self.intermediate_conv[layer_offset : layer_offset + 1],
                batch.source_state_indices,
                batch.accepted_positions,
                batch.destination_state_indices,
            )

    def iter_buffers(self) -> Iterable[torch.Tensor]:
        for buffer in (
            self.intermediate_ssm,
            self.intermediate_conv,
            self.intermediate_indices,
            self.rand_seed,
        ):
            if buffer is not None:
                yield buffer

    def shutdown(self, manager: object | None = None) -> None:
        self.intermediate_ssm = None
        self.intermediate_conv = None
        self.intermediate_indices = None
        self.rand_seed = None
        if manager is not None:
            self._publish_compatibility_views(manager)


class Mamba2ReplayStateUpdateStrategy(IntermediateMambaStateUpdateStrategy):
    """Own compact Mamba2 replay history, seeds, and bookkeeping."""

    def __init__(self, tokens_per_gen_step: int) -> None:
        super().__init__()
        self.replay_step_width = tokens_per_gen_step
        self.replay_history_size = max(MIN_REPLAY_HISTORY_SIZE, tokens_per_gen_step)
        self.prev_num_accepted_tokens: torch.Tensor | None = None
        self.cache_buf_idx: torch.Tensor | None = None
        self.old_x: torch.Tensor | None = None
        self.old_B: torch.Tensor | None = None
        self.old_dt: torch.Tensor | None = None
        self.old_dA_cumsum: torch.Tensor | None = None
        self._dummy_mask: torch.Tensor | None = None
        self._dummy_mask_host: torch.Tensor | None = None

    @property
    def key(self) -> str:
        return "mamba2_replay"

    @property
    def uses_replay(self) -> bool:
        return True

    def validate(self, context: MambaCacheBuildContext) -> None:
        if context.spec_config is None:
            raise ValueError("Mamba2 replay requires speculative decoding")
        if context.n_groups_per_rank <= 0 and context.mamba_pp_layers:
            raise ValueError("Mamba2 replay requires at least one state group per rank")

    def bind(self, context: MambaCacheBuildContext, manager: object) -> None:
        self._seed_rank_offset = context.seed_rank_offset
        if not context.mamba_pp_layers:
            self._publish_compatibility_views(manager)
            self._publish_replay_views(manager)
            return

        states = getattr(manager, "all_ssm_states")
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
        self.rand_seed = allocate_mamba_seed_buffer(cache_size, context.seed_rank_offset, device)

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
        self._dummy_mask = torch.zeros(
            context.state_index_capacity, dtype=torch.bool, device=device
        )
        self._dummy_mask_host = torch.zeros(
            context.state_index_capacity,
            dtype=torch.bool,
            pin_memory=prefer_pinned(),
        )
        self._publish_compatibility_views(manager)
        self._publish_replay_views(manager)

    def _publish_replay_views(self, manager: object) -> None:
        for name in (
            "prev_num_accepted_tokens",
            "cache_buf_idx",
            "old_x",
            "old_B",
            "old_dt",
            "old_dA_cumsum",
        ):
            setattr(manager, name, getattr(self, name))
        setattr(manager, "_dummy_request_mask", self._dummy_mask)
        setattr(manager, "_dummy_request_mask_host", self._dummy_mask_host)
        setattr(manager, "replay_step_width", self.replay_step_width)
        setattr(manager, "replay_history_size", self.replay_history_size)

    def layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        fields = super().layer_cache_fields(layer_offset)
        fields.update(
            prev_num_accepted_tokens=self.prev_num_accepted_tokens,
            cache_buf_idx=self.cache_buf_idx,
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
        return Mamba2ReplayLayerCache(
            conv=conv,
            temporal=temporal,
            **self.layer_cache_fields(layer_offset),
        )

    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None:
        self.prev_num_accepted_tokens[slots] = 0
        self.cache_buf_idx[slots] = 0
        for buffer in (self.old_x, self.old_B, self.old_dt, self.old_dA_cumsum):
            buffer[:, slots] = 0
        super().reset_slots(slots, host_slots)

    @torch.inference_mode()
    def refresh_dummy_request_mask(self, is_dummy: list[bool]) -> None:
        if self._dummy_mask is None:
            return
        count = len(is_dummy)
        if count > self._dummy_mask_host.shape[0]:
            raise ValueError("Dummy-request batch exceeds the replay mask capacity")
        self._dummy_mask_host.zero_()
        if count:
            self._dummy_mask_host[:count].copy_(torch.tensor(is_dummy, dtype=torch.bool))
        self._dummy_mask.copy_(self._dummy_mask_host, non_blocking=True)

    def dummy_request_mask(self, start: int, end: int) -> torch.Tensor | None:
        if self._dummy_mask is None:
            return None
        return self._dummy_mask[start:end]

    def update(self, batch: MambaStateUpdateBatch, manager: object) -> bool:
        metadata = self.get_replay_metadata()
        if metadata is None:
            raise RuntimeError("Mamba2 replay buffers are not bound")
        # Import through the facade to retain the established monkeypatch point.
        from tensorrt_llm._torch.pyexecutor.kv_cache import mamba_cache_manager

        mamba_cache_manager._advance_replay_state(
            metadata,
            batch.destination_state_indices,
            batch.num_accepted_tokens,
            batch.is_dummy_request,
        )
        self._promote_conv(batch, manager)
        return True

    def get_replay_metadata(self) -> ReplayStateUpdateMetadata | None:
        if self.prev_num_accepted_tokens is None or self.cache_buf_idx is None:
            return None
        return ReplayStateUpdateMetadata(
            prev_num_accepted_tokens=self.prev_num_accepted_tokens,
            cache_buf_idx=self.cache_buf_idx,
            replay_step_width=self.replay_step_width,
            replay_history_size=self.replay_history_size,
        )

    def iter_buffers(self) -> Iterable[torch.Tensor]:
        yield from super().iter_buffers()
        for buffer in (
            self.prev_num_accepted_tokens,
            self.cache_buf_idx,
            self.old_x,
            self.old_B,
            self.old_dt,
            self.old_dA_cumsum,
            self._dummy_mask,
        ):
            if buffer is not None:
                yield buffer

    def shutdown(self, manager: object | None = None) -> None:
        super().shutdown(manager)
        self.prev_num_accepted_tokens = None
        self.cache_buf_idx = None
        self.old_x = None
        self.old_B = None
        self.old_dt = None
        self.old_dA_cumsum = None
        self._dummy_mask = None
        self._dummy_mask_host = None
        if manager is not None:
            self._publish_replay_views(manager)


def build_default_mamba_state_update_strategy(
    *, spec_config: object | None, use_replay: bool
) -> MambaStateUpdateStrategy:
    """Build the default conventional or compact-replay strategy."""
    if use_replay:
        if spec_config is None:
            raise ValueError("Mamba replay requires speculative decoding")
        return Mamba2ReplayStateUpdateStrategy(spec_config.tokens_per_gen_step)
    return IntermediateMambaStateUpdateStrategy()


def build_mamba2_state_update_strategy(
    *,
    spec_config: object | None,
    ssm_cache_dtype: torch.dtype,
    stochastic_rounding: bool,
) -> MambaStateUpdateStrategy:
    """Apply Mamba2 replay gates and return its selected update strategy."""
    sm = get_sm_version()
    use_replay = spec_config is not None and sm >= 80
    if spec_config is None:
        logger.info("Replay kernel requires speculative decoding; using non-replay path")
    if spec_config is not None and (
        getattr(spec_config, "eagle_choices", None) is not None
        or getattr(spec_config, "use_dynamic_tree", False)
    ):
        logger.info("Replay kernel incompatible with tree attention; using legacy MTP path")
        use_replay = False
    if stochastic_rounding and ssm_cache_dtype == torch.float16 and (sm < 100 or sm in (120, 121)):
        logger.info(
            "Replay kernel Philox requires 100 <= sm < 120; "
            "using legacy MTP path for stochastic rounding support"
        )
        use_replay = False
    if os.environ.get("TRTLLM_USE_MAMBA_REPLAY", "1") == "0":
        logger.info("Replay kernel is disabled by TRTLLM_USE_MAMBA_REPLAY=0")
        use_replay = False
    else:
        logger.info("Replay kernel is not changed since TRTLLM_USE_MAMBA_REPLAY=1")
    return build_default_mamba_state_update_strategy(spec_config=spec_config, use_replay=use_replay)


__all__ = [
    "MIN_REPLAY_HISTORY_SIZE",
    "IntermediateMambaStateUpdateStrategy",
    "Mamba2ReplayLayerCache",
    "Mamba2ReplayStateUpdateStrategy",
    "advance_replay_state",
    "allocate_mamba_seed_buffer",
    "build_default_mamba_state_update_strategy",
    "build_mamba2_state_update_strategy",
    "compute_deterministic_mamba_seed",
    "mamba_seed_rank_offset",
]

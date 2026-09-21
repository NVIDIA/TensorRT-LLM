# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba2-owned speculative and replay cache state."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    IntermediateState,
    MambaAcceptanceBatch,
    MambaLayerCache,
    MambaStateLayout,
    ReplayStateUpdateMetadata,
    _promote_intermediate_states,
    _stack_state_views,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
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
class ReplayLayerCache(MambaLayerCache):
    """Shared Mamba2/GDN replay views consumed by their recurrent mixers."""

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


class Mamba2State(IntermediateState):
    """Intermediate promotion plus Mamba2 stochastic-rounding seeds."""

    uses_replay = False

    @override
    def __init__(self) -> None:
        super().__init__()
        self.rand_seed: torch.Tensor | None = None
        self._seed_request_counter = 0
        self._seed_rank_offset = 0

    @override
    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        super().bind(context, ssm_states, conv_states)
        self._seed_rank_offset = context.seed_rank_offset
        if context.stochastic_rounding and ssm_states:
            states = ssm_states[0]
            self.rand_seed = allocate_mamba_seed_buffer(
                states.shape[0], context.seed_rank_offset, states.device
            )

    @override
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

    @override
    def _layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        fields = super()._layer_cache_fields(layer_offset)
        if self.rand_seed is not None:
            fields["mamba_ssm_rand_seed"] = self.rand_seed
        return fields

    @override
    def shutdown(self) -> None:
        super().shutdown()
        self.rand_seed = None


class ReplayHistory:
    """Own compact Mamba2 replay history, seeds, and bookkeeping."""

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
        self._seed_rank_offset = context.seed_rank_offset
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
        return ReplayLayerCache(
            conv=conv,
            temporal=temporal,
            **self._layer_cache_fields(layer_offset),
        )

    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None:
        if self.prev_num_accepted_tokens is None:
            return
        self.prev_num_accepted_tokens.index_fill_(0, slots, 0)
        self.cache_buf_idx.index_fill_(0, slots, 0)
        for buffer in (self.old_x, self.old_B, self.old_dt, self.old_dA_cumsum):
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


def create_mamba2_state(
    *, spec_config: object | None, use_replay: bool
) -> Mamba2State | ReplayHistory:
    """Build Mamba2 intermediate state or compact replay history."""
    if use_replay:
        if spec_config is None:
            raise ValueError("Mamba replay requires speculative decoding")
        return ReplayHistory(spec_config.tokens_per_gen_step)
    return Mamba2State()


def select_mamba2_state(
    *,
    spec_config: object | None,
    ssm_cache_dtype: torch.dtype,
    stochastic_rounding: bool,
) -> Mamba2State | ReplayHistory:
    """Apply Mamba2 replay gates and return its selected update policy."""
    sm = get_sm_version()
    use_replay = spec_config is not None and sm >= 80
    if spec_config is None:
        logger.info("Replay kernel requires speculative decoding; using non-replay path")
    if spec_config is not None and (getattr(spec_config, "use_dynamic_tree", False)):
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
    return create_mamba2_state(spec_config=spec_config, use_replay=use_replay)


class NemotronHybridCacheManagerV2(MambaHybridCacheManagerV2):
    """Nemotron/Mamba2 geometry, replay selection and seed ownership."""

    _speculative_state: Mamba2State | ReplayHistory

    @override
    def __init__(self, *args, use_replay_state_update: bool | None = None, **kwargs) -> None:
        self._requested_replay = use_replay_state_update
        kwargs.setdefault("conv_state_layout", "x_b_c")
        super().__init__(*args, **kwargs)

    @override
    def _initialize_model_state(self) -> Mamba2State | ReplayHistory:
        if self._requested_replay is not None:
            state = create_mamba2_state(
                spec_config=self.spec_config, use_replay=self._requested_replay
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
    )


__all__ = [
    "Mamba2State",
    "ReplayHistory",
    "ReplayLayerCache",
    "NemotronHybridCacheManagerV2",
    "create_mamba2_state",
    "select_mamba2_state",
    "get_nemotron_cache_params",
    "allocate_mamba_seed_buffer",
    "compute_deterministic_mamba_seed",
    "advance_replay_state",
    "mamba_seed_rank_offset",
    "MIN_REPLAY_HISTORY_SIZE",
]

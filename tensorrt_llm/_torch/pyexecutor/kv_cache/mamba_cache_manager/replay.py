# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared double-buffered replay storage for recurrent cache algorithms.

Model modules choose this storage and supply any model-specific checkpoint commit.
This module does not import model implementations or dispatch their kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .common import (
    MIN_REPLAY_HISTORY_SIZE,
    MambaAcceptanceBatch,
    MambaLayerCache,
    MambaStateLayout,
    ReplayStateUpdateMetadata,
    _promote_intermediate_states,
    _stack_state_views,
)
from .seeds import _reset_mamba_seed_buffer, allocate_mamba_seed_buffer


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


class ReplayHistory:
    """Own shared replay history, seeds, bookkeeping, and conv promotion."""

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
            raise ValueError("Replay requires speculative decoding")
        if context.n_groups_per_rank <= 0 and context.mamba_pp_layers:
            raise ValueError("Replay requires at least one state group per rank")

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
            raise RuntimeError("Replay buffers are not bound")
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


__all__ = ["ReplayHistory", "ReplayLayerCache", "advance_replay_state"]

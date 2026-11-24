# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 KDA state cache: the target-owned KDA hybrid cache manager with the K3.1 differences.

The local ``cache_manager`` module owns the KDA conv / recurrent state pools on KV
cache manager V2, the fused-verify replay caches and their lifecycle (pending-draft commit on suspend, seeding of
transferred and resumed requests, slot relocation, disaggregated transfer roles). Kimi K3.1 runs that manager with:

* no per-draft records: the replay caches (k / g / v / beta and the extended conv windows) hold the drafts, and a
  pending-draft commit replays them through the fused verify kernel's commit (``kda_mtp_commit_pending_drafts``);
* the fused-verify replay windows copied and the pending drafts committed through the family operations
  (``copy_kda_replay_conv_window``, ``kda_mtp_commit_pending_drafts`` then ``store_kda_replay_conv_window``);
* no stochastic rounding of the recurrent state (it applies to an fp16 state only; the KDA state is fp32 or bf16);
* the replay caches of transferred or resumed requests seeded layer by layer with plain copies (the engine's batched
  seeding launches kernels of another family's package);
* no mid-chunk recurrent-state snapshots (an engine opt-in that needs another family's prefill kernel).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_mtp_commit_pending_drafts import (
    kda_mtp_commit_pending_drafts,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaLayerCache
from tensorrt_llm.mapping import Mapping

from ....ops.ssm.copy_kda_replay_conv_window import copy_kda_replay_conv_window
from ....ops.ssm.store_kda_replay_conv_window import store_kda_replay_conv_window
from . import cache_manager as _base_cm
from .cache_manager import KDAIntermediateState, get_kda_replay_num_spec


@dataclass(frozen=True, kw_only=True)
class KDAReplayLayerCache(_base_cm.KDAReplayLayerCache):
    """The engine's per-layer replay views; seeding the replay conv windows goes through the family operation."""

    @override
    def commit_conv_window(self, slot_indices: torch.Tensor, conv_pool: torch.Tensor) -> None:
        """Seed the replay convolution windows from the live pool."""
        copy_kda_replay_conv_window(
            conv_pool, self.kda_conv_q, self.kda_conv_k, self.kda_conv_v, slot_indices
        )


class KDAReplayState(_base_cm.KDAReplayState):
    """The engine's fused-verify replay state without draft records, stochastic rounding or batched seeding."""

    @classmethod
    @override
    def estimate_bytes_per_slot(cls, model_config, mapping: Mapping, **kwargs) -> int:
        kwargs["mamba_ssm_stochastic_rounding"] = False
        kwargs["kda_draft_records"] = False
        return super().estimate_bytes_per_slot(model_config, mapping, **kwargs)

    @override
    def _get_batched_seed_plan(self) -> None:
        """Seed transferred slots layer by layer (``seed_transferred_slots``' copy loop)."""
        return None

    @override
    def make_layer_cache(
        self,
        layer_offset: int,
        conv: torch.Tensor,
        temporal: torch.Tensor,
    ) -> MambaLayerCache:
        return KDAReplayLayerCache(
            conv=conv, temporal=temporal, **self._layer_cache_fields(layer_offset)
        )

    @override
    def commit_pending_drafts(self, slot: int) -> None:
        """Fold a slot's accepted drafts and its replay conv window into the persistent states, every layer.

        The drafts are replayed from the replay caches by the fused verify kernel's own replay steps; the
        committed conv window is then rebuilt from the replay windows. Without draft records the commit takes
        no other path.
        """
        assert self.prev_num_accepted_tokens is not None
        slots = torch.full(
            (1,), slot, dtype=torch.int32, device=self.prev_num_accepted_tokens.device
        )
        num_accepted = self.prev_num_accepted_tokens.index_select(0, slots)
        for layer, (ssm_states, conv_states) in enumerate(zip(self._ssm_states, self._conv_states)):
            replay = self._layer_cache_fields(layer)
            assert replay.get("kda_state_tok") is None, "K3.1 keeps no per-draft records"
            conv_q, conv_k, conv_v = (
                replay["kda_conv_q"],
                replay["kda_conv_k"],
                replay["kda_conv_v"],
            )
            kda_mtp_commit_pending_drafts(
                ssm_states,
                conv_q,
                conv_k,
                conv_v,
                replay.get("kda_k_cache"),
                replay.get("kda_g_cache"),
                replay.get("kda_v_cache"),
                replay.get("kda_beta_cache"),
                slots,
                num_accepted,
                conv_q.shape[2] - conv_states.shape[2],
                conv_states.shape[2] + 1,
            )
            store_kda_replay_conv_window(conv_states, conv_q, conv_k, conv_v, slots, num_accepted)
        # zero_() on the element view stays stream-ordered; assigning a Python scalar through the index stages
        # it on the host and waits for the step in flight.
        self.prev_num_accepted_tokens[slot].zero_()


class KimiK3HybridCacheManager(_base_cm.KimiK3HybridCacheManager):
    """KDA/MLA cache layout and K3-only replay lifecycle handling, with K3.1's replay state."""

    @override
    def __init__(
        self,
        *args,
        kda_replay_num_spec: int | None = None,
        use_replay_state_update: bool = False,
        **kwargs,
    ) -> None:
        # Stochastic rounding applies to an fp16 recurrent state only; the KDA
        # state is fp32 or bf16 (see validate_kimi_state_cache_dtype).
        kwargs.pop("mamba_ssm_stochastic_rounding", None)
        super().__init__(
            *args,
            kda_replay_num_spec=kda_replay_num_spec,
            kda_draft_records=False,
            use_replay_state_update=use_replay_state_update,
            **kwargs,
        )

    @override
    def _get_kda_midchunk_snapshot_alignment(self) -> None:
        """Mid-chunk snapshots stay off: snapshot points split context chunks."""
        return None

    @override
    def _initialize_spec_state(self) -> KDAIntermediateState | KDAReplayState | None:
        num_spec = self._requested_num_spec
        if num_spec is None:
            num_spec = get_kda_replay_num_spec(self.spec_config, manager_supports_replay=True)
        self.kda_context_only = self.kda_context_only and num_spec is not None
        if self.kda_context_only:
            return None
        if num_spec is not None:
            self._kda_replay = KDAReplayState(num_spec)
            self._kda_replay.validate(self._state_layout)
            return self._kda_replay
        return KDAIntermediateState()

    @classmethod
    @override
    def _extra_state_bytes_per_rank(cls, model_config, mapping: Mapping, **kwargs) -> int:
        return KDAReplayState.estimate_bytes_per_slot(model_config, mapping, **kwargs)


__all__ = ["KDAReplayLayerCache", "KDAReplayState", "KimiK3HybridCacheManager"]

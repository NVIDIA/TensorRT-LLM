# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 KDA state cache: the engine's KDA hybrid cache manager with the K3.1 differences.

The engine (``tensorrt_llm._torch.modules.kimi_kda.cache_manager``) owns the KDA conv / recurrent state pools on KV
cache manager V2, the fused-verify replay caches and their lifecycle (pending-draft commit on suspend, seeding of
transferred and resumed requests, slot relocation, disaggregated transfer roles). Kimi K3.1 runs that manager with:

* no per-draft records: the replay caches (k / g / v / beta and the extended conv windows) hold the drafts, and a
  pending-draft commit replays them through the fused verify kernel's commit (``kda_mtp_commit_pending_drafts``);
* the pending drafts committed through the family operations (``kda_mtp_commit_pending_drafts`` then
  ``store_kda_replay_conv_window``); the KDA layers seed the replay windows from the live pool themselves
  (``copy_kda_replay_conv_window``);
* the replay state only: a speculative config with more drafts than the fused verify kernel replays is refused (the
  engine's sequential verify fallback is not part of the Kimi K3.1 layer), and without speculative decoding there is
  no verify state;
* no stochastic rounding of the recurrent state (it applies to an fp16 state only; the K3.1 KDA state is fp32);
* the replay caches of transferred or resumed requests seeded in all local KDA layers at once by the family's
  ``seed_kda_replay_layers`` on views stacked over the layers (the engine's batched seeding launches kernels of
  another family's package), layer by layer only when a buffer kind's views are not evenly spaced (logged once);
* no mid-chunk recurrent-state snapshots (an engine opt-in that needs another family's prefill kernel);
* the pending-draft commit's kernel compiled once the pools are bound, before serving and any CUDA graph capture;
* a disaggregated context server whose DFlash drafter is context-only (``speculative_config.context_kv_only``) keeps
  no replay state, as the engine's context-only cache does under ``TRTLLM_DISAGG_ROLE=context``.
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_draft_pages import (
    dflash2_context_kv_only,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_mtp_commit_pending_drafts import (
    kda_mtp_commit_pending_drafts,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.seed_kda_replay_layers import (
    seed_kda_replay_layers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.store_kda_replay_conv_window import (
    store_kda_replay_conv_window,
)
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAReplayState as _EngineKDAReplayState,
)
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KimiK3HybridCacheManager as _EngineKimiK3HybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import _stack_state_views
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


def _context_kv_only_server(kwargs: dict) -> bool:
    """A disaggregated context server whose DFlash drafter only builds the prompt's drafter context."""
    return bool(kwargs.get("is_disagg", False)) and dflash2_context_kv_only(
        kwargs.get("spec_config")
    )


#: The replay state's per-layer buffers (fields of the per-layer cache it hands the KDA layers): the q / k / v
#: convolution windows, and the draft caches.
_KDA_REPLAY_CONV_ROLES = ("kda_conv_q", "kda_conv_k", "kda_conv_v")
_KDA_REPLAY_DRAFT_ROLES = ("kda_k_cache", "kda_g_cache", "kda_v_cache", "kda_beta_cache")


class _StackedReplaySeed(NamedTuple):
    """Every local KDA layer's seeded buffers, one ``[layers, slots, ...]`` view per buffer kind."""

    conv_pool: torch.Tensor
    caches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    zeroed: tuple[torch.Tensor, ...]


class _StackedReplaySeeding:
    """The engine's batched-seed hook (``seed_transferred_slots`` calls ``seed_kda_replay_layers(plan, slots)``)."""

    @staticmethod
    def seed_kda_replay_layers(plan: _StackedReplaySeed, state_indices: torch.Tensor) -> None:
        seed_kda_replay_layers(plan.conv_pool, *plan.caches, plan.zeroed, state_indices)


def _seed_layer_by_layer(reason: str) -> None:
    """Log, once per reason, that transferred slots are seeded by the per-layer loop; returns the declined plan."""
    logger.warning_once(
        f"KDA replay caches of transferred requests are seeded layer by layer: {reason}. The loop launches "
        "about 11 small kernels per local KDA layer on the executor thread for each request that joins.",
        key=f"kimi_k31_kda_replay_seed:{reason}",
    )
    return None


class K31KDAReplayState(_EngineKDAReplayState):
    """The engine's fused-verify replay state without draft records or stochastic rounding; transferred slots are
    seeded in all local KDA layers at once by the family's ``seed_kda_replay_layers``."""

    #: ``update`` is the engine's ``record_acceptance(destination slots, accepted - 1, dummy rows)`` into the int32
    #: ``prev_num_accepted_tokens``, which the DFlash2 worker runs as one launch together with its KV-length
    #: bookkeeping (``ops/spec/dflash2_step_bookkeeping.k31_dflash2_accept_record``). A subclass that changes
    #: ``update`` or ``record_acceptance`` sets it to False.
    fused_accept_record: bool = True

    @classmethod
    @override
    def estimate_bytes_per_slot(cls, model_config, mapping: Mapping, **kwargs) -> int:
        if _context_kv_only_server(kwargs):
            return 0
        kwargs["mamba_ssm_stochastic_rounding"] = False
        kwargs["kda_draft_records"] = False
        return super().estimate_bytes_per_slot(model_config, mapping, **kwargs)

    @override
    def _plan_batched_seed(self) -> tuple[type[_StackedReplaySeeding], _StackedReplaySeed] | None:
        """Stack each buffer kind ``seed_transferred_slots`` writes over the local KDA layers.

        The V2 manager's per-slot coalesced pools place a role's per-layer views at a constant spacing, so one
        ``[layers, slots, ...]`` view per kind covers every layer and the seeding takes one launch per kind instead
        of one per kind and layer. None, logged once per reason, when the per-layer loop has to run instead.
        """
        conv_states = self._conv_states
        if not conv_states:
            return _seed_layer_by_layer("this rank holds no KDA layers")
        if not conv_states[0].is_cuda:
            return _seed_layer_by_layer("the KDA states are not on a GPU")
        if len(self._conv_section_dims) != 3 or len(set(self._conv_section_dims)) != 1:
            return _seed_layer_by_layer(
                "the convolution state is not three equal q / k / v sections"
            )
        conv_pool = _stack_state_views(list(conv_states))
        if conv_pool is None:
            return _seed_layer_by_layer(
                "the layers' live convolution states are not evenly spaced views of one layout"
            )
        layers = [self._layer_cache_fields(offset) for offset in range(len(conv_states))]
        stacked = {}
        for name in (*_KDA_REPLAY_CONV_ROLES, *_KDA_REPLAY_DRAFT_ROLES):
            if layers[0].get(name) is None:
                continue
            view = _stack_state_views([layer[name] for layer in layers])
            if view is None:
                return _seed_layer_by_layer(
                    f"the layers' {name} buffers are not evenly spaced views of one layout"
                )
            stacked[name] = view
        logger.info(
            f"KDA replay caches of transferred requests are seeded in all {len(conv_states)} local KDA layers "
            "at once (one launch per buffer kind)"
        )
        caches = tuple(stacked.pop(name) for name in _KDA_REPLAY_CONV_ROLES)
        return _StackedReplaySeeding, _StackedReplaySeed(conv_pool, caches, tuple(stacked.values()))

    def _commit_layer(
        self, layer: int, slots: torch.Tensor, num_accepted: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay the pending drafts of ``slots`` in KDA layer ``layer`` onto its recurrent state; returns the
        layer's q / k / v replay conv caches."""
        replay = self._layer_cache_fields(layer)
        assert replay.get("kda_state_tok") is None, "K3.1 keeps no per-draft records"
        conv_states = self._conv_states[layer]
        conv_q, conv_k, conv_v = replay["kda_conv_q"], replay["kda_conv_k"], replay["kda_conv_v"]
        kda_mtp_commit_pending_drafts(
            self._ssm_states[layer],
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
        return conv_q, conv_k, conv_v

    def warm_commit_pending_drafts(self) -> None:
        """Compile the pending-draft commit's kernel variant for this rank's pools before serving and before any
        CUDA graph capture: :meth:`commit_pending_drafts` runs eagerly, one slot per call, when a request is
        suspended with drafts pending, and a variant compiled on that first call stalls the step. Every KDA layer's
        views share one layout, so one call compiles what every layer runs. Slot 0 is committed with nothing
        pending, which keeps its state bit for bit."""
        if not self._ssm_states or self.prev_num_accepted_tokens is None:
            return
        device = self.prev_num_accepted_tokens.device
        none_pending = torch.zeros(1, dtype=torch.int32, device=device)
        self._commit_layer(0, none_pending, none_pending)

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
        for layer, conv_states in enumerate(self._conv_states):
            conv_q, conv_k, conv_v = self._commit_layer(layer, slots, num_accepted)
            store_kda_replay_conv_window(conv_states, conv_q, conv_k, conv_v, slots, num_accepted)
        # zero_() on the element view stays stream-ordered; assigning a Python scalar through the index stages
        # it on the host and waits for the step in flight.
        self.prev_num_accepted_tokens[slot].zero_()


class K31HybridCacheManager(_EngineKimiK3HybridCacheManager):
    """KDA/MLA cache layout and K3-only replay lifecycle handling, with K3.1's replay state."""

    # A disaggregated context server of a context-only DFlash drafter: no replay state.
    _context_kv_only = False

    @override
    def __init__(
        self,
        *args,
        kda_replay_num_spec: int | None = None,
        use_replay_state_update: bool = False,
        **kwargs,
    ) -> None:
        # Stochastic rounding applies to an fp16 recurrent state only; the K3.1
        # KDA state is fp32 (routing.py).
        kwargs.pop("mamba_ssm_stochastic_rounding", None)
        self._context_kv_only = _context_kv_only_server(kwargs)
        super().__init__(
            *args,
            kda_replay_num_spec=kda_replay_num_spec,
            kda_draft_records=False,
            use_replay_state_update=use_replay_state_update,
            **kwargs,
        )

    @override
    def _setup_model_state(self) -> None:
        super()._setup_model_state()
        replay = getattr(self, "_kda_replay", None)
        if isinstance(replay, K31KDAReplayState):
            replay.warm_commit_pending_drafts()

    @override
    def _get_kda_midchunk_snapshot_alignment(self) -> None:
        """Mid-chunk snapshots stay off: snapshot points split context chunks."""
        return None

    @override
    def _initialize_spec_state(self) -> K31KDAReplayState | None:
        """The fused verify's replay state, sized for the speculative config's drafts; None without speculative
        decoding (nothing to verify) and on a context-only server."""
        num_spec = self._requested_num_spec
        if num_spec is None and self.spec_config is not None:
            num_spec = self.spec_config.tokens_per_gen_step - 1
        if num_spec is None:
            self.kda_context_only = False
            return None
        # The verify kernel's draft capacity (imported here: the kernel module needs the CuTe DSL).
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm.kda_mtp_decode_v2 import (
            MAX_NUM_SPEC,
        )

        if num_spec > MAX_NUM_SPEC:
            raise ValueError(
                f"Kimi K3.1 KDA verify replays at most {MAX_NUM_SPEC} draft tokens per step, got {num_spec}: "
                f"set speculative_config.max_draft_len <= {MAX_NUM_SPEC}"
            )
        self.kda_context_only = self.kda_context_only or self._context_kv_only
        if self.kda_context_only:
            return None
        self._kda_replay = K31KDAReplayState(num_spec)
        self._kda_replay.validate(self._state_layout)
        return self._kda_replay

    @classmethod
    @override
    def _extra_state_bytes_per_rank(cls, model_config, mapping: Mapping, **kwargs) -> int:
        return K31KDAReplayState.estimate_bytes_per_slot(model_config, mapping, **kwargs)


__all__ = ["K31KDAReplayState", "K31HybridCacheManager"]

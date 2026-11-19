# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 KDA state cache: the engine's KDA hybrid cache manager with the K3.1 differences.

The engine (``tensorrt_llm._torch.modules.kimi_kda.cache_manager``) owns the KDA conv / recurrent state pools on KV
cache manager V2, the fused-verify replay caches and their lifecycle (replay-slot resets, seeding of transferred and
resumed requests, slot relocation, disaggregated transfer roles). Kimi K3.1 runs that manager with:

* no per-draft records: the replay caches (k / g / v / beta, the extended conv windows and the accepted-draft count)
  hold the drafts of the last verify, and the next verify replays them;
* the replay caches in the recurrent-state slot (``replay_shares_state_slot``): they are buffers of each KDA layer's
  recurrent layer config, so the fused verify / decode kernels address the recurrent state and the replay caches with
  one state index, and the replay caches follow the recurrent state through suspension, offload and slot moves. No
  replay region is declared, so its fixed pool costs nothing; the per-slot estimate carries the replay bytes;
* the replay state only: a speculative config with more drafts than the fused verify kernel replays is refused (the
  engine's sequential verify fallback is not part of the Kimi K3.1 layer), and without speculative decoding there is
  no verify state;
* no stochastic rounding of the recurrent state (it applies to an fp16 state only; the K3.1 KDA state is fp32);
* no mid-chunk recurrent-state snapshots (an engine opt-in that needs another family's prefill kernel);
* a disaggregated context server (``TRTLLM_DISAGG_ROLE=context``), whose DFlash drafter is context-only, keeps no
  replay state: the engine's context-only cache (``kda_context_only``);
* a gate on disaggregated gen-init admission (``prepare_disagg_gen_init``, ``K31GenInitDecodeHeadroom``, 1024
  tokens): on a generation server a gen-init request is admitted only if each pool group keeps decode growth room for
  the resident requests, read from the per-pool-group decode growth and free slots (``token_growth_slots_per_block``,
  ``get_available_primary_slots``);
* no appended draft layers (``accepts_appended_draft_layers``): a drafter's context KV goes to the unified draft cache
  (``UnifiedDraftCacheMixin``), never into the engine's shared target + draft layout.

The KDA layers seed the replay windows from the live pool themselves (``copy_kda_replay_conv_window``).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import replace
from typing import Optional

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_draft_pages import (
    dflash2_draft_window,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.unified_draft_cache import (
    UnifiedDraftCacheMixin,
)
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAReplayState as _EngineKDAReplayState,
)
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KimiK3HybridCacheManager as _EngineKimiK3HybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState, get_draft_token_length
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import GPU_LEVEL, BatchDesc, KVCacheDesc

#: Tokens of decode growth per resident generation request that a gen-init admission leaves free.
_GEN_INIT_DECODE_HEADROOM_TOKENS = 1024
_DISAGG_GENERATION_INIT = LlmRequestState.DISAGG_GENERATION_INIT.value


def _set_draft_window(kwargs: dict) -> None:
    """Keep only the DFlash2 drafter's sliding window of its context in the standalone draft layers.

    The engine appends the drafter's layers to this manager (``draft_*`` keyword arguments) when the
    drafter opts into the unified draft cache; the window is the drafter checkpoint's
    (``dflash2_draft_window``), passed as the manager's ``draft_window_size``.

    Raises:
        ValueError: the drafter declares no sliding window. The window keeps the draft layers out of
            the target's full-attention lifecycle (``UnifiedDraftCacheMixin``).
    """
    if kwargs.get("draft_num_layers") and kwargs.get("draft_window_size") is None:
        spec_config = kwargs.get("spec_config")
        window = dflash2_draft_window(spec_config)
        if window is None:
            drafter = getattr(spec_config, "speculative_model", None)
            raise ValueError(
                f"The unified draft cache keeps the DFlash2 drafter's sliding window of its context, "
                f"but the drafter {drafter!r} declares no sliding_window in its config."
            )
        kwargs["draft_window_size"] = window


class K31KDAReplayState(_EngineKDAReplayState):
    """The engine's fused-verify replay state without draft records or stochastic rounding, in the state slot."""

    #: ``update`` is the engine's ``record_acceptance(destination slots, accepted - 1, dummy rows)`` into the int32
    #: ``prev_num_accepted_tokens``, which the DFlash2 worker runs as one launch together with its KV-length
    #: bookkeeping (``ops/spec/dflash2_step_bookkeeping.k31_dflash2_accept_record``). A subclass that changes
    #: ``update`` or ``record_acceptance`` sets it to False.
    fused_accept_record: bool = True

    @property
    @override
    def replay_shares_state_slot(self) -> bool:
        """The K3.1 verify and decode kernels address the replay caches by the recurrent-state slot."""
        return True

    @classmethod
    @override
    def estimate_bytes_per_slot(cls, model_config, mapping: Mapping, **kwargs) -> int:
        kwargs["mamba_ssm_stochastic_rounding"] = False
        kwargs["kda_draft_records"] = False
        return super().estimate_bytes_per_slot(model_config, mapping, **kwargs)


class _DecodeHeadroomPass:
    """One scheduling pass of :class:`K31GenInitDecodeHeadroom`.

    A disaggregated gen-init request takes its whole prompt's KV in one allocation, while a decoding request grows
    one block at a time. Once admissions have filled the pool, a decoding request whose last block is full cannot
    grow, and its failed allocation stops the scheduling pass, so every request behind it waits as well. A gen-init
    request is therefore admitted only if, after its prompt blocks, each pool group still holds the next
    ``lookahead_tokens`` (capped by the tokens each request may still generate, plus its draft tokens) of every
    resident generation request, its own included.

    Everything is read on this rank: the active requests the executor schedules from and the KV cache manager
    (``token_growth_slots_per_block`` and ``get_available_primary_slots`` of ``K31HybridCacheManager``). The TP
    ranks of a non-ADP engine hold identical copies, so they decide alike.
    """

    _RESIDENT_STATE_VALUES = frozenset(
        state.value
        for state in (
            LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
            LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE,
            LlmRequestState.GENERATION_IN_PROGRESS,
        )
    )

    def __init__(self, kv_cache_manager, requests, lookahead_tokens: int) -> None:
        self._kv = kv_cache_manager
        self._tokens_per_block = kv_cache_manager.tokens_per_block
        self._lookahead_tokens = lookahead_tokens
        self._slots_per_block = kv_cache_manager.token_growth_slots_per_block()
        self._available = list(kv_cache_manager.get_available_primary_slots())
        self.reserved_blocks = 0
        for req in requests:
            if req.state_value not in self._RESIDENT_STATE_VALUES:
                continue
            if not kv_cache_manager.is_request_active(req.py_request_id):
                continue
            cache = kv_cache_manager.kv_cache_map.get(req.py_request_id)
            if cache is not None:
                self.reserved_blocks += self._growth_blocks(req, cache.capacity)

    def _blocks(self, tokens: int) -> int:
        return -(-tokens // self._tokens_per_block)

    def _growth_blocks(self, req, capacity: int) -> int:
        max_new_tokens = getattr(req, "py_max_new_tokens", None)
        lookahead = self._lookahead_tokens
        if max_new_tokens is not None:
            generated = max(0, req.get_num_tokens(0) - req.prompt_len)
            lookahead = min(lookahead, max(0, max_new_tokens - generated))
        tokens = lookahead + get_draft_token_length(req)
        return self._blocks(capacity + tokens) - self._blocks(capacity)

    def _admission_blocks(self, req) -> tuple[int, int]:
        """(prompt blocks still to allocate, growth blocks after them) for ``req``."""
        cache = self._kv.kv_cache_map.get(req.py_request_id)
        held = cache.capacity if cache is not None else 0
        target = max(
            held,
            req.prompt_len + get_draft_token_length(req) + self._kv._kv_alloc_extra_tokens,
        )
        return self._blocks(target) - self._blocks(held), self._growth_blocks(req, target)

    def admits(self, req) -> Optional[tuple[int, int]]:
        """The blocks ``req`` would take, or None when the pool cannot spare them.

        Called before the request's allocation; the result goes to :meth:`commit` once the allocation succeeds.
        """
        prompt_blocks, growth_blocks = self._admission_blocks(req)
        need = prompt_blocks + growth_blocks + self.reserved_blocks
        if all(
            self._available[pool_group] >= slots * need
            for pool_group, slots in self._slots_per_block.items()
        ):
            return prompt_blocks, growth_blocks
        return None

    def commit(self, blocks: tuple[int, int]) -> None:
        """Account for an admitted request: its prompt blocks are taken, its growth reserved."""
        prompt_blocks, growth_blocks = blocks
        for pool_group, slots in self._slots_per_block.items():
            self._available[pool_group] -= slots * prompt_blocks
        self.reserved_blocks += growth_blocks


class K31GenInitDecodeHeadroom:
    """The gen-init admission gate of ``K31HybridCacheManager.prepare_disagg_gen_init``: each pool group keeps
    ``lookahead_tokens`` of decode growth for every resident generation request (:class:`_DecodeHeadroomPass`).

    The manager does not ask it under Helix.
    """

    def __init__(self, lookahead_tokens: int) -> None:
        self.lookahead_tokens = lookahead_tokens

    def begin_pass(self, kv_cache_manager, requests) -> _DecodeHeadroomPass:
        return _DecodeHeadroomPass(kv_cache_manager, requests, self.lookahead_tokens)


class K31HybridCacheManager(UnifiedDraftCacheMixin, _EngineKimiK3HybridCacheManager):
    """KDA/MLA cache layout and K3-only replay lifecycle handling, with K3.1's replay state."""

    #: The gate on disaggregated gen-init admission (``prepare_disagg_gen_init``).
    gen_init_admission = K31GenInitDecodeHeadroom(_GEN_INIT_DECODE_HEADROOM_TOKENS)
    # Whether this manager gates gen-init admission: set by __init__, off under Helix, whose ledger blocks span the
    # CP ranks.
    _gates_gen_init_admission = False
    # The gate's state for the current scheduling pass (``prepare_expect_snapshot_points``); None admits as usual.
    _gen_init_pass: Optional[_DecodeHeadroomPass] = None

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
        _set_draft_window(kwargs)
        super().__init__(
            *args,
            kda_replay_num_spec=kda_replay_num_spec,
            kda_draft_records=False,
            use_replay_state_update=use_replay_state_update,
            **kwargs,
        )
        self._gates_gen_init_admission = not self._has_cp_helix

    @classmethod
    @override
    def accepts_appended_draft_layers(cls) -> bool:
        """A Kimi K3.1 drafter keeps its context KV in the unified draft cache, so the engine never shares this manager
        with appended draft layers."""
        return False

    @override
    def prepare_expect_snapshot_points(self, requests) -> None:
        """Also starts the scheduling pass's gen-init admission gate.

        The executor calls this with its active requests before each scheduling pass, so the gate reads the free
        slots and the resident requests' growth before any allocation of the pass. Without a gen-init request among
        them the pass admits as usual.
        """
        super().prepare_expect_snapshot_points(requests)
        self._gen_init_pass = (
            self.gen_init_admission.begin_pass(self, requests)
            if self._gates_gen_init_admission
            and any(req.state_value == _DISAGG_GENERATION_INIT for req in requests)
            else None
        )

    @override
    def prepare_disagg_gen_init(self, req) -> bool:
        """The engine's preparation, once this pass's gen-init admission gate admits ``req``.

        A request the gate holds back returns False, as a failed preparation does, and is retried in a later pass;
        an admitted request's blocks count against the rest of the pass once its preparation succeeds.
        """
        gate = self._gen_init_pass
        blocks = None
        if gate is not None:
            blocks = gate.admits(req)
            if blocks is None:
                return False
        if not super().prepare_disagg_gen_init(req):
            return False
        if blocks is not None:
            gate.commit(blocks)
        return True

    @override
    def _attention_cache_bytes_per_token(self) -> int:
        """Bytes per token of the model's attention layers (``Role.ALL``) and the standalone draft layers."""
        return self._draft_cache_bytes_per_token() + sum(
            self.get_layer_bytes_per_token(local_layer_idx, Role.ALL)
            for local_layer_idx in range(self.num_local_layers)
            if not self._is_standalone_draft_layer(local_layer_idx)
        )

    def _verify_max_num_spec(self) -> int:
        """The fused verify kernel's draft capacity (imported here: the kernel module needs the CuTe DSL)."""
        from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.ssm.kda_mtp_decode_v2 import (
            MAX_NUM_SPEC,
        )

        return MAX_NUM_SPEC

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
        max_num_spec = self._verify_max_num_spec()
        if num_spec > max_num_spec:
            raise ValueError(
                f"Kimi K3.1 KDA verify replays at most {max_num_spec} draft tokens per step, got {num_spec}: "
                f"set speculative_config.max_draft_len <= {max_num_spec}"
            )
        if self.kda_context_only:
            return None
        self._kda_replay = K31KDAReplayState(num_spec)
        self._kda_replay.validate(self._state_layout)
        return self._kda_replay

    @classmethod
    @override
    def _estimate_fixed_region_bytes(
        cls,
        model_config,
        mapping: Mapping,
        *,
        max_batch_size: int,
        kv_cache_config: KvCacheConfig,
        **kwargs,
    ) -> int:
        """No replay region: the replay caches live in the recurrent-state slots."""
        return 0

    @classmethod
    @override
    def _extra_state_bytes_per_rank(cls, model_config, mapping: Mapping, **kwargs) -> int:
        """Each recurrent-state slot also holds its replay caches."""
        return K31KDAReplayState.estimate_bytes_per_slot(model_config, mapping, **kwargs)

    def token_growth_slots_per_block(self) -> dict[int, int]:
        """GPU pool group -> slots it gives a request for each block of tokens the request grows.

        Every full-attention layer group placed in a pool group takes one slot of it per ``tokens_per_block``
        tokens of capacity. Sliding-window groups release a stale page for each new one once past the window,
        and the KDA state does not grow with tokens, so neither is counted. The layout is fixed at construction.
        """
        growth = getattr(self, "_token_growth_slots_per_block", None)
        if growth is None:
            growth = defaultdict(int)
            for pool_group_id, window_size, kind in self._stats_life_cycle_metadata().values():
                if (
                    kind == "attention"
                    and window_size is not None
                    and window_size >= self.max_seq_len
                ):
                    growth[pool_group_id] += 1
            growth = dict(growth)
            self._token_growth_slots_per_block = growth
        return growth

    def get_available_primary_slots(self) -> list[int]:
        """Available slots of each GPU pool group, indexed by pool group."""
        return [pool.available for pool in self._get_storage_statistics(GPU_LEVEL)]

    # --- The drafter's windowed context (standalone draft layers with a sliding window) ---------
    #
    # The draft layers keep only the window before a request's history length; a context chunk
    # allocates draft pages for all of its positions, and pages that fall out of the window are
    # released when the history length passes them. The recurrent-state cache holds the history
    # length at the last committed snapshot point while a later point is pending, which with a
    # snapshot at the prompt end is the whole prompt, so the draft pages of every chunk since that
    # point would stay locked. These overrides advance the history length to each chunk's end and
    # size the draft pages for the windows of every resident request plus one step of chunks.

    def _windowed_draft_pages(self) -> bool:
        """Whether this manager holds a sliding window of a standalone drafter's context."""
        return (
            bool(getattr(self, "_draft_layer_sizes", None)) and self.draft_window_size is not None
        )

    def _draft_page_floor_tokens(self) -> int:
        """Draft-page tokens that every resident request's window plus one step of chunks need.

        Each request holds the blocks of its retention window plus up to two partial blocks; a
        step's chunks add ``max_num_tokens``. Leases are counted twice over the batch size, for the
        requests that wait on their KV transfer (disaggregated) or on the next step (overlap
        scheduler) while others are scheduled.
        """
        tokens_per_block = self.tokens_per_block
        retention_blocks = -(-self.draft_retention_window_size // tokens_per_block)
        per_lease = (retention_blocks + 2) * tokens_per_block
        leases = 2 * self._max_resident_sequences()
        return (self.max_num_tokens or 0) + leases * per_lease

    @override
    def _build_cache_config(self, config):
        """Add a draft-page floor: the windows of every lease plus one step of chunks.

        One request of that many tokens with no history needs exactly as many draft pages, a few
        tens of thousands of tokens of the other layers' pages and one recurrent state.
        """
        config = super()._build_cache_config(config)
        if not self._windowed_draft_pages():
            return config
        # A request without history: its allocation also covers the drafter's scratch.
        floor_tokens = self._draft_page_floor_tokens() + self._standalone_draft_reserve
        floor = BatchDesc([KVCacheDesc(capacity=floor_tokens, history_length=0)])
        return replace(config, constraints=[*config.constraints, floor])

    def _advance_history(self, request, kv_cache, history_length: int) -> None:
        if not kv_cache.resize(None, history_length):
            raise ValueError(
                f"Failed to advance the KV cache history of request {request.py_request_id} to "
                f"{history_length} tokens"
            )

    @override
    def try_commit_blocks(self, request, kv_cache=None) -> None:
        """Commit at a snapshot point from a history length that already passed the last commit.

        A commit with recurrent snapshots must start or end at the history length; after
        :meth:`update_context_resources` advanced it past the last commit, a commit at the point
        the context reached ends there.
        """
        if (
            self._windowed_draft_pages()
            and self.enable_block_reuse
            and not request.is_dummy_request
        ):
            cache = (
                kv_cache if kv_cache is not None else self.kv_cache_map.get(request.py_request_id)
            )
            position = request.context_current_position
            if (
                cache is not None
                and cache.is_active
                and position in request.expect_snapshot_points
                and cache.num_committed_tokens < cache.history_length < position
            ):
                self._advance_history(request, cache, position)
        super().try_commit_blocks(request, kv_cache)

    @override
    def update_context_resources(self, scheduled_batch) -> None:
        """After the step's commits, advance each context request's history length to its chunk's
        end, which releases the draft pages that fell out of the window."""
        super().update_context_resources(scheduled_batch)
        if not self._windowed_draft_pages():
            return
        for request in scheduled_batch.context_requests:
            cache = self.kv_cache_map.get(request.py_request_id)
            position = request.context_current_position
            if cache is not None and cache.is_active and cache.history_length < position:
                self._advance_history(request, cache, position)


__all__ = ["K31GenInitDecodeHeadroom", "K31KDAReplayState", "K31HybridCacheManager"]

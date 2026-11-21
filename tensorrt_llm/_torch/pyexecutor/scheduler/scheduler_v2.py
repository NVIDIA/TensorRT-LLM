# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import os
import time
from collections import Counter
from typing import Callable, Optional

from tensorrt_llm.llmapi.llm_args import (
    CapacitySchedulerPolicy,
    ContextChunkingPolicy,
    ContextSchedulingPolicy,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import RemoteMemError

from ...disaggregation import diagnostics as disagg_diagnostics
from ..llm_request import (
    LlmRequest,
    LlmRequestState,
    get_draft_token_length,
    rewind_context_after_cache_drop,
)
from . import ctx_decoder_batch
from .prefix_wait_planner import PrefixWaitPlanner
from .scheduler import (
    RemoteTailPhase,
    RequestList,
    RequestScheduler,
    SchedulerOutput,
    _get_forced_context_chunk_size,
    _get_lora_task_id,
    _is_forced_context_chunk_boundary,
    drop_decoder_context_requests_waiting_for_encoder_output,
)

# ---- Out-of-pages context retry gate ----
# With the KV pool full, every pass re-runs the reuse probe and claim (prepare_context) and the
# resize of each waiting first-chunk context request, and each attempt fails again. The gate
# remembers such a failure together with the pools' page counts (available and evictable, every
# pool group, every cache level, target and draft pools) and replays the outcome without the
# prepare/resize work until a count rises, the token budget or chunk cap would give a smaller
# chunk than the failed one, or MAX_SKIP + (request id mod MAX_SKIP) passes have gone by. A
# replayed out-of-pages failure still calls preempt_for_pages exactly as an attempt does, so
# preemption sees the same calls and picks the same victims. The gate stands aside once the
# deadlock detector has counted half its threshold, so it never pushes a pass to a deadlock
# declaration. Scheduling only; the decisions use state identical on every TP rank.
#   default                                       gate on
#   TRTLLM_DISABLE_CTX_OOP_RETRY_GATE=1           every pass attempts (kill switch)
#   TRTLLM_CTX_OOP_RETRY_GATE_SHADOW=1            attempts as without the gate; the gate only counts
#   TRTLLM_CTX_OOP_RETRY_GATE_MAX_SKIP            default 16
_OOP_GATE_DISABLED = os.environ.get("TRTLLM_DISABLE_CTX_OOP_RETRY_GATE", "").strip() == "1"
_OOP_GATE_SHADOW = (
    not _OOP_GATE_DISABLED and os.environ.get("TRTLLM_CTX_OOP_RETRY_GATE_SHADOW", "").strip() == "1"
)
_OOP_GATE_ON = not _OOP_GATE_DISABLED and not _OOP_GATE_SHADOW
_OOP_GATE_MODE = "off" if _OOP_GATE_DISABLED else ("shadow" if _OOP_GATE_SHADOW else "on")
try:
    _OOP_GATE_MAX_SKIP = max(1, int(os.environ.get("TRTLLM_CTX_OOP_RETRY_GATE_MAX_SKIP", "16")))
except ValueError:
    _OOP_GATE_MAX_SKIP = 16
_OOP_GATE_LOG_EVERY_S = 60.0


def _scheduler_rank() -> int:
    try:
        from tensorrt_llm._utils import mpi_rank

        return int(mpi_rank())
    except Exception:  # noqa: BLE001 - the rank only labels log lines
        for key in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "SLURM_PROCID", "RANK"):
            if os.environ.get(key, "").isdigit():
                return int(os.environ[key])
        return -1


class ScheduleAction(enum.Enum):
    """Result of a per-request scheduling attempt."""

    SCHEDULED = "scheduled"  # success — payload contains tokens etc.
    SKIP = "skip"  # skip this request, continue the loop
    STOP = "stop"  # stop the scheduling loop
    DEFER = "defer"  # retry this request after the rest of this iteration's context order


class _RecomputePauseState:
    """Candidate frontier and exact victims for one scheduling iteration."""

    def __init__(self, frontier: int):
        self.frontier = frontier
        self.victim_indices: set[int] = set()


class _CtxPreemptionStats:
    """Context-path KV page pressure, accumulated over one logging window.

    Attributes:
        passes: Scheduling passes with at least one pending context request.
        preempted: Requests released so a context request could allocate.
        preempted_tokens: Prefill progress those requests gave up, in tokens.
        deferred: Preemptions parked until the victim's connector saves retire.
        no_victim: Out-of-pages context requests that found no eligible victim.
        preempt_off: Out-of-pages context requests where preemption does not apply
            (a cache tier below GPU, or recompute pause disabled).
        suspended: Suspend calls on context-state requests.
    """

    __slots__ = (
        "passes",
        "preempted",
        "preempted_tokens",
        "deferred",
        "no_victim",
        "preempt_off",
        "suspended",
    )

    def __init__(self) -> None:
        self.passes = 0
        self.preempted = 0
        self.preempted_tokens = 0
        self.deferred = 0
        self.no_victim = 0
        self.preempt_off = 0
        self.suspended = 0

    def add(self, other: "_CtxPreemptionStats") -> None:
        self.passes += other.passes
        self.preempted += other.preempted
        self.preempted_tokens += other.preempted_tokens
        self.deferred += other.deferred
        self.no_victim += other.no_victim
        self.preempt_off += other.preempt_off
        self.suspended += other.suspended

    def __str__(self) -> str:
        return (
            f"ctx_passes={self.passes} preempted={self.preempted} "
            f"preempted_tokens={self.preempted_tokens} deferred={self.deferred} "
            f"no_victim={self.no_victim} preempt_off={self.preempt_off} "
            f"suspended={self.suspended}"
        )


class BudgetTracker:
    """Tracks token, request, and PEFT budgets for one scheduling iteration.

    Centralizes the budget logic that was previously spread across the main
    scheduling loop.  Designed for future extensibility to multiple memory
    pools — callers interact through ``can_fit_tokens`` /
    ``remaining_tokens`` / ``commit`` / ``peft_pages_needed`` without knowing the underlying pool
    topology.
    """

    def __init__(
        self,
        max_num_tokens: Optional[int],
        max_num_requests: int,
        peft_cache_manager=None,
    ):
        self.max_num_tokens = max_num_tokens
        self.max_num_requests = max_num_requests
        self.num_tokens = 0
        self.num_requests = 0

        # PEFT accounting
        self._peft_cache_manager = peft_cache_manager
        self._max_peft_pages = (
            peft_cache_manager.max_device_pages if peft_cache_manager is not None else 0
        )
        self._claimed_peft_pages = 0
        self._seen_peft_task_ids: set[int] = set()

    # ---- Token / request budget ----

    @property
    def requests_full(self) -> bool:
        return self.num_requests >= self.max_num_requests

    def can_fit_tokens(self, num_tokens: int) -> bool:
        """Check if *num_tokens* fits within the remaining token budget."""
        if self.max_num_tokens is not None and (self.num_tokens + num_tokens > self.max_num_tokens):
            return False
        return True

    @property
    def remaining_tokens(self) -> Optional[int]:
        """Remaining token budget, or ``None`` if unlimited."""
        if self.max_num_tokens is None:
            return None
        return self.max_num_tokens - self.num_tokens

    def commit(self, req: LlmRequest, num_tokens: int, peft_pages: int) -> None:
        """Record a successfully scheduled request's token and PEFT consumption."""
        self.num_tokens += num_tokens
        self.num_requests += 1
        if peft_pages > 0:
            self.commit_peft(req, peft_pages)

    def hold(self, num_tokens: int, num_requests: int) -> None:
        """Set tokens and request slots aside for requests scheduled later in the iteration."""
        self.num_tokens += num_tokens
        self.num_requests += num_requests

    def release(self, num_tokens: int, num_requests: int) -> None:
        """Return what ``hold`` set aside."""
        self.num_tokens -= num_tokens
        self.num_requests -= num_requests

    def commit_peft(self, req: LlmRequest, peft_pages: int) -> None:
        """Record only PEFT consumption without affecting token/request budgets.

        Used by disagg_gen_init requests which need PEFT accounting but don't
        participate in the forward pass.
        """
        lora_task_id = getattr(req, "lora_task_id", None)
        self._claimed_peft_pages += peft_pages
        self._seen_peft_task_ids.add(lora_task_id)

    # ---- PEFT budget ----

    def peft_pages_needed(self, req: LlmRequest) -> Optional[int]:
        """Return PEFT pages needed for *req*, or ``None`` if budget exceeded."""
        if self._peft_cache_manager is None:
            return 0
        lora_task_id = getattr(req, "lora_task_id", None)
        if lora_task_id is None or lora_task_id in self._seen_peft_task_ids:
            return 0
        required = self._peft_cache_manager.determine_num_pages(req)
        if self._claimed_peft_pages + required > self._max_peft_pages:
            return None
        return required

    def pre_claim_peft(self, req: LlmRequest) -> None:
        """Reserve PEFT pages for a non-scheduled request whose adapter is
        still loaded on device (e.g. GENERATION_TO_COMPLETE in the overlap
        executor — not yet terminated, so ensure_batch cannot evict it)."""
        if self._peft_cache_manager is None:
            return
        lora_task_id = getattr(req, "lora_task_id", None)
        if lora_task_id is None or lora_task_id in self._seen_peft_task_ids:
            return
        pages = self._peft_cache_manager.determine_num_pages(req)
        self._claimed_peft_pages += pages
        self._seen_peft_task_ids.add(lora_task_id)


def _env_seconds(name: str, default: float, *aliases: str) -> float:
    """A number of seconds from ``name``, else from the first set of its older spellings ``aliases``."""
    for key in (name, *aliases):
        raw = os.environ.get(key, "").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                raise ValueError(f"{key} must be a number of seconds, got {raw!r}") from None
    return default


def _env_switch(names: tuple, default: bool, kill_names: tuple = ()) -> bool:
    """An on/off knob: the first of ``names`` set to 1 or 0 decides, else any of ``kill_names`` set
    to 1 turns it off, else ``default`` (a model's opt-in, for example)."""
    for name in names:
        raw = os.environ.get(name, "").strip()
        if raw == "1":
            return True
        if raw == "0":
            return False
    if any(os.environ.get(name, "").strip() == "1" for name in kill_names):
        return False
    return default


class _StoreLeaseTouch:
    """Keeps the stored prefix of queued first-chunk context requests leased.

    A store such as Mooncake grants a key a read lease when a lookup finds it,
    and once the pool is full it evicts unleased keys first. A first-chunk
    request waiting behind long prefills can outlive the lease its ranking
    probe granted, and then loses its own prefix and recomputes it. Every
    pass, each waiting request whose last probe is ``interval_s`` old is probed
    again through the connector's side-effect-free probe, at most
    ``max_per_pass`` of them, oldest first; a request never probed is first due
    after ``first_s``. The answers are dropped: ranking, admission and
    allocation stay as they are, only lookups are added.

    Attention DP only: without it the probe is a leader broadcast, and
    wall-clock due times could differ across ranks.
    On where the model opts in (RuntimeHooks.store_lease_touch);
    TLLM_STORE_LEASE_TOUCH=1 / 0 turns it on / off either way.
    TLLM_STORE_LEASE_TOUCH_S, TLLM_STORE_LEASE_TOUCH_FIRST_S and
    TLLM_STORE_LEASE_TOUCH_MAX_PER_PASS set the three limits.
    """

    LOG_INTERVAL_S = 60.0

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.interval_s = max(
            _env_seconds("TLLM_STORE_LEASE_TOUCH_S", 20.0, "TRTLLM_M31_LEASE_TOUCH_S"), 0.1
        )
        self.first_s = min(
            max(
                _env_seconds(
                    "TLLM_STORE_LEASE_TOUCH_FIRST_S", 1.0, "TRTLLM_M31_LEASE_TOUCH_FIRST_S"
                ),
                0.0,
            ),
            self.interval_s,
        )
        self.max_per_pass = max(
            int(
                _env_seconds("TLLM_STORE_LEASE_TOUCH_MAX_PER_PASS", 8, "TRTLLM_M31_LEASE_TOUCH_MAX")
            ),
            1,
        )
        self.clock = clock
        # Clock time of each waiting first-chunk request's last probe.
        self.last: dict[int, float] = {}
        # Counted over one log interval.
        self.touches = 0
        self.shrunk = 0
        self.lost_tokens = 0
        self._last_log = clock()

    @staticmethod
    def engages(connector) -> bool:
        """Whether *connector* can be touched: it loads, and runs under attention DP."""
        return (
            connector is not None
            and connector.capacity_only is not True
            and connector.enable_attention_dp is True
        )

    def note_probe(self, request_id: int) -> None:
        """A ranking probe just looked this request up, so its leases are fresh."""
        self.last[request_id] = self.clock()

    def maybe_log(self, now: float) -> None:
        if now - self._last_log < self.LOG_INTERVAL_S:
            return
        if self.touches:
            logger.info(
                f"KVCacheV2Scheduler store lease touch: touches={self.touches} "
                f"prefixes_shrunk={self.shrunk} lost_tokens={self.lost_tokens} "
                f"in the last {now - self._last_log:.0f} s"
            )
        self.touches = self.shrunk = self.lost_tokens = 0
        self._last_log = now


def _in_other_locality_domain(victim, locality_domain_id) -> bool:
    # Pages are allocated per locality domain, so a victim in another
    # domain frees nothing for this request.
    victim_domain = getattr(victim, "py_locality_domain_id", None)
    return (
        isinstance(locality_domain_id, int)
        and isinstance(victim_domain, int)
        and victim_domain != locality_domain_id
    )


class KVCacheV2Scheduler(RequestScheduler):
    """Interleaved scheduler for KV Cache Manager V2.

    Merges capacity checking (KV cache allocation via resize()) and token budget
    assignment into one unified loop.
    """

    # The RemoteMemError a resume raised because the request's KV pages were on a remote cache level that failed: set
    # by _try_allocate_generation and by the context and disagg generation-init admissions, consumed by
    # _on_lost_kv_cache.
    _lost_kv_error: Optional[RemoteMemError] = None
    # (request, error message) of requests that lost KV pages and cannot recompute them; take_lost_kv_requests.
    _lost_kv_requests: tuple = ()
    # Managers whose remote cache level failing stops this rank (recover_lost_kv_per_rank=False); empty otherwise.
    _lockstep_remote_managers: tuple = ()

    _ctx_decoder_held = False
    # Holds since take_decision_counters (see RequestScheduler.take_decision_counters).
    _num_gen_init_cap_holds = 0
    _num_gen_init_transfer_budget_holds = 0
    _num_ctx_decoder_hold_passes = 0
    _num_ctx_decoder_held_requests = 0
    # Out-of-pages retry gate defaults (set per instance by _oop_gate_init); an instance built
    # without __init__ attempts every pass.
    _oop_gate_track = False
    _oop_gate_iter = 0
    _oop_gate_cnt: Counter = Counter()

    def __init__(
        self,
        max_batch_size: int,
        max_num_tokens: Optional[int],
        kv_cache_manager: object,  # KVCacheManagerV2
        scheduler_policy: CapacitySchedulerPolicy,
        ctx_chunk_config: Optional[tuple] = None,
        peft_cache_manager: object | None = None,
        scheduler_capacity: Optional[int] = None,
        no_schedule_until_state: LlmRequestState = LlmRequestState.CONTEXT_INIT,
        no_schedule_after_state: LlmRequestState = LlmRequestState.GENERATION_TO_COMPLETE,
        draft_kv_cache_manager: object | None = None,  # KVCacheManagerV2 for MTP draft layers
        cross_kv_cache_manager: object | None = None,  # KVCacheManagerV2 for enc-dec cross-attn
        enable_prefix_aware_scheduling: bool = True,
        enable_recompute_pause: bool = True,
        context_scheduling_policy: ContextSchedulingPolicy = ContextSchedulingPolicy.FCFS,
        context_starvation_iterations: Optional[int] = None,
        recover_lost_kv_per_rank: bool = True,
        max_gen_init_admissions_per_pass: Optional[int] = None,
        context_demote_uncached_tokens: Optional[int] = None,
        context_demote_max_iters: Optional[int] = None,
        exact_context_demotion: bool = False,
        enable_context_only_preemption: Optional[bool] = None,
        preempt_least_prefill_progress: Optional[bool] = None,
        store_aware_ranking: Optional[bool] = None,
        store_lease_touch: Optional[bool] = None,
    ) -> None:
        """``recover_lost_kv_per_rank`` False is for ranks that schedule in lock step (TP without attention DP, CP,
        PP): a failed remote cache level then raises RemoteMemError out of ``schedule_request`` instead of re-matching,
        recomputing or failing the affected requests on this rank alone, which would desynchronize the group."""
        self.remote_tail_phase: RemoteTailPhase | None = None
        self._remote_tail_context_ids: frozenset[int] = frozenset()
        self._remote_tail_generation_ids: frozenset[int] = frozenset()
        self.max_num_tokens = max_num_tokens
        self._stalled_schedules = 0
        self.max_num_requests = (
            scheduler_capacity if scheduler_capacity is not None else max_batch_size
        )
        from ..kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, ctx_fill_disabled

        assert isinstance(kv_cache_manager, KVCacheManagerV2), (
            f"KVCacheV2Scheduler requires KVCacheManagerV2, got {type(kv_cache_manager).__name__}"
        )
        self.kv_cache_manager = kv_cache_manager
        self.draft_kv_cache_manager = draft_kv_cache_manager
        self.enable_joint_kv_cache_reuse = kv_cache_manager.enable_joint_kv_cache_reuse
        # The draft pool to keep in lockstep with the target, or None when
        # unpaired. Resolved once so the admission paths cannot disagree.
        self._joint_draft_manager = (
            draft_kv_cache_manager if self.enable_joint_kv_cache_reuse else None
        )
        self.cross_kv_cache_manager = cross_kv_cache_manager
        self.enable_prefix_aware_scheduling = enable_prefix_aware_scheduling
        self.enable_recompute_pause = enable_recompute_pause
        self.recover_lost_kv_per_rank = recover_lost_kv_per_rank
        if not recover_lost_kv_per_rank:
            self._lockstep_remote_managers = tuple(
                manager
                for manager in (kv_cache_manager, draft_kv_cache_manager, cross_kv_cache_manager)
                if manager is not None and manager.has_remote_cache_level
            )
        # Disaggregated generation-init requests admitted per pass (None: all that fit).
        self.max_gen_init_admissions_per_pass = max_gen_init_admissions_per_pass
        # Opt-in (the constructor argument, e.g. a model's RuntimeHooks.ctx_only_preemption;
        # TLLM_V2_CTX_PREEMPT_LEAST_PROGRESS=1 / 0 turns it on / off either way):
        # a context request out of pages preempts the eligible request with the
        # least prefill progress (ties: the newest), so the pass discards the
        # least finished prefill, instead of the newest eligible request. See
        # _try_preempt_for_pages.
        self._ctx_preempt_least_progress = _env_switch(
            ("TLLM_V2_CTX_PREEMPT_LEAST_PROGRESS",),
            bool(preempt_least_prefill_progress),
            ("TRTLLM_M31_DISABLE_CTX_PREEMPT_LEAST_PROGRESS",),
        )
        # Opt-in (the constructor argument; TLLM_V2_CTX_ONLY_PREEMPTION=1 / 0 turns it on / off
        # either way).
        # Disaggregated workers turn recompute pause off because a generation
        # worker cannot replay a prefill it received. A context-only request on a
        # context worker computed every page it holds, so with this opt-in it
        # stays a preemption victim there; without it, a context worker whose
        # pool fills with started prefills can only wait for them or declare a
        # deadlock. Pipeline parallelism keeps a chunk in its microbatch ring for
        # more passes than the in-flight window below covers, so it stays off
        # there.
        enable_context_only_preemption = _env_switch(
            ("TLLM_V2_CTX_ONLY_PREEMPTION",),
            bool(enable_context_only_preemption),
            ("TRTLLM_M31_DISABLE_CTX_ONLY_PREEMPT",),
        )
        pp_size = getattr(getattr(kv_cache_manager, "mapping", None), "pp_size", 1)
        self._ctx_only_preempt = (
            enable_context_only_preemption
            and not enable_recompute_pause
            and not (isinstance(pp_size, int) and pp_size > 1)
        )
        # Context request ids scheduled in the last two passes. The overlap
        # executor schedules a pass while the previous pass's batch may still run
        # on the GPU (a batch that cannot queue under attention DP does not run
        # at all), so such a request may still have a chunk executing or a
        # connector save reading its pages. Two passes leave a margin and expire
        # on their own, so a chunk dropped or shortened after scheduling never
        # pins a request. Context-only preemption never takes one of them.
        self._ctx_scheduled_last: frozenset[int] = frozenset()
        self._ctx_scheduled_prev: frozenset[int] = frozenset()
        # The last batch known to have launched, and the positions of the last
        # pass's requests. The executor applies a batch's chunks
        # (move_to_next_context_chunk) only when it runs it, so a position that
        # moved since the last pass means that pass's batch launched. Under
        # attention DP a launched batch stays unprocessed, and may still run,
        # through passes that launch nothing, which the two-pass window alone
        # would outlast; it is replaced only when another batch launches, after
        # the executor has processed it.
        self._ctx_launched: frozenset[int] = frozenset()
        self._ctx_scheduled_pos: dict[int, int] = {}
        # Context requests of a batch the executor keeps unprocessed (see
        # set_unprocessed_context); never preempted until it is processed.
        self._ctx_unprocessed: frozenset[int] = frozenset()
        # Context page pressure: the current 60 s window and the running totals.
        self._ctx_preempt_stats = _CtxPreemptionStats()
        self._ctx_preempt_totals = _CtxPreemptionStats()
        self._ctx_preempt_last_log = time.monotonic()
        # Every input to the prefix-aware skip is fixed for this scheduler's
        # lifetime: the knob is a constructor argument, and both manager
        # attributes are assigned once in ``KVCacheManagerV2.__init__``. Resolve
        # the gate here rather than re-deriving it on every schedule_request.
        self._prefix_skip_enabled = (
            enable_prefix_aware_scheduling
            and kv_cache_manager.enable_block_reuse
            and self._skip_pays_off_under_reuse_policy()
        )
        # Keeps new context requests queued behind a prefill of a prefix they
        # share, where the cache manager can make that prefix reusable.
        self._prefix_wait_planner = (
            PrefixWaitPlanner(kv_cache_manager)
            if kv_cache_manager.can_share_inflight_prefix
            else None
        )
        if scheduler_policy != CapacitySchedulerPolicy.MAX_UTILIZATION:
            logger.warning(
                "KVCacheV2Scheduler only supports MAX_UTILIZATION for now, "
                f"got {scheduler_policy.name}, setting to MAX_UTILIZATION"
            )
            scheduler_policy = CapacitySchedulerPolicy.MAX_UTILIZATION
        self.policy = scheduler_policy
        self.peft_cache_manager = peft_cache_manager

        # Chunking config.
        self.chunking_enabled = False
        self.chunking_policy: Optional[ContextChunkingPolicy] = None
        self.chunk_unit_size = 0
        self.max_context_length = max_num_tokens
        self.tokens_per_block = kv_cache_manager.tokens_per_block
        self.has_cp_helix = kv_cache_manager._has_cp_helix
        # Every request runs at exactly max_beam_width (checked on admission).
        self._single_beam = kv_cache_manager.max_beam_width == 1
        draft_mgr_name = (
            type(draft_kv_cache_manager).__name__ if draft_kv_cache_manager is not None else "None"
        )
        cross_mgr_name = (
            type(cross_kv_cache_manager).__name__ if cross_kv_cache_manager is not None else "None"
        )
        logger.info(
            f"KVCacheV2Scheduler: tokens_per_block={self.tokens_per_block}, "
            f"max_num_tokens={max_num_tokens}, max_batch_size={max_batch_size}, "
            f"draft_mgr={draft_mgr_name}, cross_mgr={cross_mgr_name}, "
            f"enable_prefix_aware_scheduling={enable_prefix_aware_scheduling}, "
            f"prefix_skip_enabled={self._prefix_skip_enabled}, "
            f"enable_recompute_pause={enable_recompute_pause}, "
            f"max_gen_init_admissions_per_pass={max_gen_init_admissions_per_pass}, "
            f"ctx_only_preempt={self._ctx_only_preempt}, "
            f"preempt_least_prefill_progress={self._ctx_preempt_least_progress}"
        )
        if ctx_chunk_config is not None:
            self.chunking_enabled = True
            self.chunking_policy = ctx_chunk_config[0]
            self.chunk_unit_size = ctx_chunk_config[1]

        # State value caches for fast comparison.
        # Default range [CONTEXT_INIT, GENERATION_TO_COMPLETE) matches C++
        # MicroBatchScheduler. For encoder-decoder models, caller should pass
        # no_schedule_until_state=ENCODER_INIT to widen the range (same as
        # C++ trtEncoderModel which passes kENCODER_INIT).
        self.no_schedule_until_state = no_schedule_until_state
        self.no_schedule_after_state = no_schedule_after_state
        self._no_schedule_until_state_value = no_schedule_until_state.value
        self._no_schedule_after_state_value = no_schedule_after_state.value
        self._context_init_state_value = LlmRequestState.CONTEXT_INIT.value
        # Prefix-aware skip bookkeeping: first-chunk contributors of the previous schedule
        # (block key -> request id), and keys whose deferral did not pay (see
        # ``_deferral_paid``).
        self._prev_first_chunk_contributors: dict = {}
        self._unpaid_prefix_keys: dict = {}
        self._encoder_init_state_value = LlmRequestState.ENCODER_INIT.value
        self._disagg_gen_init_state_value = LlmRequestState.DISAGG_GENERATION_INIT.value
        self._gen_to_complete_state_value = LlmRequestState.GENERATION_TO_COMPLETE.value

        # Opt-in (default off): on the disagg generation server, schedule
        # just-arrived generation-only requests (first token not yet emitted)
        # ahead of older in-flight decodes to reduce TTFT under budget
        # pressure. See _schedule_loop.
        self._prioritize_first_token_gen = (
            os.environ.get("TLLM_DISAGG_GEN_PRIORITIZE_FIRST_TOKEN", "0") == "1"
        )

        # Long-prefill demotion (opt-in). Context scheduling is FCFS, so a
        # multi-hundred-k-token prefill takes the whole token budget for as many
        # iterations as it needs while the short cached turns behind it wait. A
        # request whose uncached prefill the attention-DP router estimates above
        # this many tokens (py_uncached_prefill_tokens) goes to the tail of the
        # iteration's context order from its first chunk on, FCFS within each
        # class, and takes whatever budget the others leave: reordered, never
        # capped. Without an estimate a request is never demoted; mid-prefill
        # the exact remainder releases it once no more than the threshold is
        # left. Gating at the door (not from the second chunk on) keeps long
        # requests from parking mid-prefill with their KV blocks held.
        # The constructor arguments win; unset, the environment variables
        # (TLLM_V2_CTX_DEMOTE_*) apply.
        if context_demote_uncached_tokens is None:
            context_demote_uncached_tokens = int(
                os.environ.get("TLLM_V2_CTX_DEMOTE_UNCACHED_TOKENS", "0")
            )
        demote_uncached = max(0, context_demote_uncached_tokens)
        self._ctx_demote_uncached_tokens = demote_uncached if self.chunking_enabled else 0
        # A request demoted this many iterations is let through for good: it
        # runs in arrival order to the end of its prefill (0 = no bound).
        if context_demote_max_iters is None:
            context_demote_max_iters = int(os.environ.get("TLLM_V2_CTX_DEMOTE_MAX_ITERS", "0"))
        self._ctx_demote_max_iters = max(0, context_demote_max_iters)
        self._ctx_demote_skips: dict[int, int] = {}
        # Requests let through; they keep their estimate for plan_context_pass
        # and the router's least_prefill load.
        self._ctx_demote_released: set[int] = set()
        # Exact classification (RuntimeHooks.exact_context_demotion): by the
        # prefix cache, not by the router's estimate alone. The router only
        # knows the conversation's previous prompt length on this rank, so it
        # calls a forked sub-agent or a new conversation over a cached shared
        # prefix long, and a continuing turn whose prefix was evicted short
        # (which then takes the whole budget for every chunk). An estimate
        # above the threshold is checked once against the radix tree
        # (read-only probe, one walk per request), a request whose cache this
        # pool already holds is judged by its exact remainder, and a request
        # ordered as short whose first prepare shows a long remainder goes
        # behind the rest of the iteration.
        self._ctx_demote_exact = self._ctx_demote_uncached_tokens > 0 and exact_context_demotion
        # First-chunk requests ordered as short on the estimate alone this iteration.
        self._ctx_unverified_short: set[int] = set()
        if self._ctx_demote_uncached_tokens:
            logger.info(
                f"KVCacheV2Scheduler: demoting context requests with more than "
                f"{self._ctx_demote_uncached_tokens} uncached tokens left "
                f"(max iterations demoted: {self._ctx_demote_max_iters or 'unbounded'}, "
                f"exact: {self._ctx_demote_exact})"
            )
        elif demote_uncached:
            logger.warning(
                "TLLM_V2_CTX_DEMOTE_UNCACHED_TOKENS needs chunked prefill; context demotion is off"
            )

        # Shortest-uncached-first: context requests with the fewest uncached prompt
        # tokens go first. With a starvation limit, a request pending for more
        # scheduler iterations than that first takes up to a quarter of
        # max_num_tokens in whole chunks, oldest first (see _starved_ctx_shares).
        self._ctx_shortest_first = (
            context_scheduling_policy == ContextSchedulingPolicy.SHORTEST_UNCACHED_FIRST
        )
        self._ctx_starvation_iterations = context_starvation_iterations
        if context_starvation_iterations is not None and not (
            self._ctx_shortest_first and self.chunking_enabled and max_num_tokens is not None
        ):
            logger.warning(
                "scheduler_config.context_starvation_iterations is ignored: it needs "
                "context_scheduling_policy='shortest_uncached_first', chunked context and a "
                "max_num_tokens budget"
            )
            self._ctx_starvation_iterations = None
        self._starved_ctx_tokens = (
            max_num_tokens // 4 if self._ctx_starvation_iterations is not None else 0
        )
        self._sched_iter = 0
        # Reuse probe results of this scheduling pass, by request id (_uncached_tokens).
        self._reuse_probes: dict[int, Optional[int]] = {}
        # Opt-in (the constructor argument, e.g. a model's RuntimeHooks.ctx_store_aware_ranking;
        # TLLM_CTX_SHORTEST_FIRST_STORE_AWARE, or its older spelling TLLM_CTX_SRPT_STORE_AWARE, set
        # to 1 / 0 turns it on / off either way): shortest-uncached-first also counts the prefix a
        # KV connector can load, so a request its store holds is not ranked as a cold prompt.
        # Otherwise it ranks on the local cache only.
        self._ctx_store_aware = self._ctx_shortest_first and _env_switch(
            ("TLLM_CTX_SHORTEST_FIRST_STORE_AWARE", "TLLM_CTX_SRPT_STORE_AWARE"),
            bool(store_aware_ranking),
        )
        # End of the connector-served prefix of each ranked first-chunk request
        # (None: nothing to load), by request id. Probed once per request.
        self._store_probes: dict[int, Optional[int]] = {}
        # Lease renewal for the stored prefixes of waiting first-chunk requests (opt-in: the
        # constructor argument; TLLM_STORE_LEASE_TOUCH=1 / 0 turns it on / off either way).
        self._lease_touch: Optional[_StoreLeaseTouch] = (
            _StoreLeaseTouch()
            if _env_switch(
                ("TLLM_STORE_LEASE_TOUCH",),
                bool(store_lease_touch),
                ("TLLM_DISABLE_STORE_LEASE_TOUCH", "TRTLLM_M31_DISABLE_LEASE_TOUCH"),
            )
            else None
        )
        # Shortest-uncached-first aging. Age is counted from the pass a request first waits in
        # the context state, and an aged request stays aged until its prefill completes.
        # Counting from the last chunk instead (TLLM_CTX_SRPT_AGE_FROM_ARRIVAL=0) gives a
        # started long prompt one chunk per TLLM_CTX_SRPT_MAX_WAIT_PASSES passes while shorter
        # prompts keep arriving, so a very long cold prompt holds its pages for a very long time.
        age_from_arrival = os.environ.get("TLLM_CTX_SRPT_AGE_FROM_ARRIVAL", "1").strip() != "0"
        # A context request that has waited TLLM_CTX_SRPT_MAX_WAIT_PASSES scheduling passes moves
        # ahead of the shortest-first order, so a long cold prompt cannot be crowded out
        # indefinitely. Aging also keeps unstarted requests from preempting started ones (see
        # preempt_for_pages). 0 disables both. Aged requests take budget from short prompts, so
        # aging from arrival defaults to a longer wait.
        self._ctx_srpt_max_wait_passes = (
            self._resolve_ctx_srpt_max_wait_passes(
                os.environ.get("TLLM_CTX_SRPT_MAX_WAIT_PASSES", ""),
                600 if age_from_arrival else 128,
            )
            if self._ctx_shortest_first
            else 0
        )
        self._ctx_srpt_pass = 0
        self._ctx_srpt_age_from_arrival = self._ctx_srpt_max_wait_passes > 0 and age_from_arrival
        # Pass at which each waiting context request started waiting (from arrival), or last
        # received context tokens (TLLM_CTX_SRPT_AGE_FROM_ARRIVAL=0).
        self._ctx_srpt_waiting_since: dict[int, int] = {}
        # Wall-clock bound on the same age, so it holds when passes slow down: a request
        # TLLM_CTX_SRPT_MAX_WAIT_S seconds past arrival is aged too (0 disables). A clock is
        # read per rank, so the bound is used only where each rank schedules its own requests.
        self._ctx_srpt_max_wait_s = (
            self._resolve_ctx_srpt_max_wait_s(os.environ.get("TLLM_CTX_SRPT_MAX_WAIT_S", ""))
            if self._ctx_srpt_age_from_arrival
            and self._ranks_schedule_independently(kv_cache_manager)
            else 0.0
        )
        self._ctx_srpt_arrival_s: dict[int, float] = {}
        self._ctx_srpt_now = 0.0
        # An admitted request whose prefix a KV connector loads asynchronously leaves the
        # context state until the load lands (DISAGG_GENERATION_TRANS_IN_PROGRESS), then comes
        # back started. Its age is kept across that excursion when aging from arrival;
        # restarting it would make the request wait out the aging bound a second time while its
        # loaded pages stay pinned.
        self._ctx_srpt_keep_state_value = (
            LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS.value
            if self._ctx_srpt_age_from_arrival
            else None
        )
        # How many aged requests go ahead of the shortest-first order, oldest (started first)
        # first. Each takes at most its chunk cap per pass, so the rest of the step still
        # serves short prompts first; aged requests beyond the slots follow the shortest-first
        # requests in arrival order and take what budget is left. 0 puts every aged request
        # first. Used with TLLM_CTX_SRPT_AGE_FROM_ARRIVAL only.
        self._ctx_srpt_aged_slots = (
            self._resolve_ctx_srpt_aged_slots(os.environ.get("TLLM_CTX_SRPT_AGED_SLOTS", ""))
            if self._ctx_srpt_age_from_arrival
            else 0
        )
        # Context fill: a first chunk whose store probe predicts a connector-served prefix is
        # charged only the tokens its forward pass can compute (see _ctx_fill_chunk). Without
        # it an asynchronously loaded store hit leaves the batch after scheduling and the
        # budget it was charged stays unused for that step.
        # TRTLLM_DISABLE_CTX_FILL=1 (or TRTLLM_M31_DISABLE_CTXFILL=1) charges the full chunk again.
        self._ctx_fill = self._ctx_store_aware and not ctx_fill_disabled()
        # Context fill v2 (TRTLLM_DISABLE_CTX_FILL_V2=1 or TRTLLM_M31_DISABLE_CTXFILL_V2=1 keeps v1):
        # * never lose a store hit to the shrink: an asynchronously served first chunk that is
        #   shrunk also reserves the pages of its probed served prefix here, so the connector
        #   hook's resize in prepare_resources needs no new page; when they do not fit, the
        #   request is deferred a pass instead of having its hit discarded and recomputed, and
        #   after TRTLLM_CTX_FILL_MAX_DEFER deferrals it is not shrunk;
        # * shrink only under a backlog: the previous pass left at least
        #   TRTLLM_CTX_FILL_MIN_BACKLOG context requests unscheduled on this rank.
        self._ctx_fill_v2 = self._ctx_fill and not ctx_fill_disabled(v2=True)
        # The manager's half of v2 (the served-prefix candidate) follows this scheduler.
        kv_cache_manager.ctx_fill_v2 = self._ctx_fill_v2
        self._ctx_fill_min_backlog = self._resolve_nonnegative_int(
            "TRTLLM_CTX_FILL_MIN_BACKLOG", 8, "TRTLLM_M31_CTXFILL_MIN_BACKLOG"
        )
        self._ctx_fill_max_defer = self._resolve_nonnegative_int(
            "TRTLLM_CTX_FILL_MAX_DEFER", 4, "TRTLLM_M31_CTXFILL_MAX_DEFER"
        )
        # Context requests the previous pass left unscheduled, and deferrals per request.
        self._ctx_fill_backlog = 0
        self._ctx_fill_defers: dict[int, int] = {}
        # Probed served end whose pages the chunk being sized must reserve (async v2 shrink).
        self._ctx_fill_reserve_end: Optional[int] = None
        # Window counters: async shrinks, sync shrinks, budget tokens returned, shrinks
        # skipped for no backlog, served prefixes reserved, deferrals, shrinks given up after
        # the deferral limit.
        self._ctx_fill_counts = [0] * 7
        self._ctx_fill_totals = [0] * 7
        self._ctx_fill_last_log = time.monotonic()
        if self._ctx_shortest_first:
            logger.info(
                f"KVCacheV2Scheduler: ctx_srpt_max_wait_passes={self._ctx_srpt_max_wait_passes}, "
                f"ctx_srpt_age_from_arrival={self._ctx_srpt_age_from_arrival}, "
                f"ctx_srpt_max_wait_s={self._ctx_srpt_max_wait_s:g}, "
                f"ctx_srpt_aged_slots={self._ctx_srpt_aged_slots}, "
                f"ctx_store_aware={self._ctx_store_aware}, ctx_fill={self._ctx_fill}, "
                f"ctx_fill_v2={self._ctx_fill_v2}, "
                f"ctx_fill_min_backlog={self._ctx_fill_min_backlog}, "
                f"ctx_fill_max_defer={self._ctx_fill_max_defer}"
            )

        # Opt-in (default off): while several context requests are pending, cap each one's
        # chunk at max(TLLM_CTX_REQUEST_CHUNK_CAP, its even share of the step budget), so long
        # prompts interleave with shorter ones instead of taking the whole step. A lone pending
        # context request keeps the full budget. Forced (snapshot) chunking is not capped.
        self._ctx_request_chunk_cap = self._resolve_ctx_request_chunk_cap(
            os.environ.get("TLLM_CTX_REQUEST_CHUNK_CAP", "")
        )
        self._ctx_chunk_cap_this_pass: Optional[int] = None
        # The cap's even share is of the budget still left, divided among the context requests
        # still able to take a chunk this pass rather than every pending one, so budget that
        # held or deferred requests cannot use goes to the others.
        # TLLM_CTX_CHUNK_CAP_ELIGIBLE_SHARE=0 keeps one share of the whole budget per pending
        # request.
        self._ctx_chunk_cap_eligible_share = (
            self._ctx_request_chunk_cap is not None
            and os.environ.get("TLLM_CTX_CHUNK_CAP_ELIGIBLE_SHARE", "1").strip() != "0"
        )
        if self._ctx_request_chunk_cap is not None:
            logger.info(
                "KVCacheV2Scheduler: context chunk cap share over eligible requests "
                f"{self._ctx_chunk_cap_eligible_share} (TLLM_CTX_CHUNK_CAP_ELIGIBLE_SHARE)"
            )
        # In-forward Mamba snapshots (the GDN cache manager's): staging rows
        # per batch, and those left in the batch being scheduled.
        self._inforward_snapshot_rows = kv_cache_manager.inforward_snapshot_rows
        self._inforward_snapshot_rows_left = 0

        # Registered by PyExecutor; see set_async_transfer_manager.
        self._async_transfer_manager = None
        self._retiring_gen_receives_probe: Callable[[], bool] = lambda: False
        # Registered by PyExecutor; see set_disagg_transfer_admission_controller.
        self._disagg_transfer_controller = None

        # Generation requests paused in place (GEN slot preemption).
        self._gen_paused_ids: frozenset[int] = frozenset()
        self._gen_resumed_ids: frozenset[int] = frozenset()
        self._gen_preemption_enabled = False

        # `is True`: a manager without the flag reads as off.
        self._swa_scratch_reuse = (
            getattr(kv_cache_manager, "enable_swa_scratch_reuse", False) is True
        )
        # Decoder window whose final-window chunks this pass holds (ctx_decoder_batch.py); 0: none.
        self._ctx_decoder_hold_window = 0
        self._ctx_decoder_held = False
        self._oop_gate_init()

    def _resolve_ctx_request_chunk_cap(self, raw: str) -> Optional[int]:
        try:
            cap = int(raw) if raw.strip() else 0
        except ValueError:
            raise ValueError(
                f"TLLM_CTX_REQUEST_CHUNK_CAP must be an integer, got {raw!r}"
            ) from None
        if cap <= 0:
            return None
        if not self.chunking_enabled:
            logger.warning("TLLM_CTX_REQUEST_CHUNK_CAP is ignored: chunked prefill is disabled")
            return None
        unit = max(self.chunk_unit_size, 1)
        cap = max(unit, cap // unit * unit)
        logger.info(
            f"KVCacheV2Scheduler: per-request context chunk cap {cap} tokens while other "
            "context requests are pending (TLLM_CTX_REQUEST_CHUNK_CAP)"
        )
        return cap

    @staticmethod
    def _resolve_ctx_srpt_max_wait_passes(raw: str, default: int) -> int:
        try:
            passes = int(raw) if raw.strip() else default
        except ValueError:
            raise ValueError(
                f"TLLM_CTX_SRPT_MAX_WAIT_PASSES must be an integer, got {raw!r}"
            ) from None
        return max(passes, 0)

    @staticmethod
    def _resolve_ctx_srpt_max_wait_s(raw: str) -> float:
        try:
            seconds = float(raw) if raw.strip() else 120.0
        except ValueError:
            raise ValueError(f"TLLM_CTX_SRPT_MAX_WAIT_S must be a number, got {raw!r}") from None
        return seconds if seconds > 0 else 0.0

    @staticmethod
    def _resolve_nonnegative_int(name: str, default: int, *aliases: str) -> int:
        """``name``, else the first set of its older spellings ``aliases``, else ``default``."""
        for key in (name, *aliases):
            raw = os.environ.get(key, "")
            if raw.strip():
                try:
                    return max(int(raw), 0)
                except ValueError:
                    raise ValueError(f"{key} must be an integer, got {raw!r}") from None
        return max(default, 0)

    @staticmethod
    def _resolve_ctx_srpt_aged_slots(raw: str) -> int:
        try:
            slots = int(raw) if raw.strip() else 1
        except ValueError:
            raise ValueError(f"TLLM_CTX_SRPT_AGED_SLOTS must be an integer, got {raw!r}") from None
        return max(slots, 0)

    @staticmethod
    def _ranks_schedule_independently(kv_cache_manager) -> bool:
        """Whether no other rank must reach this rank's scheduling decisions.

        Ranks that share requests (tensor, pipeline or context parallelism without
        attention DP) schedule the same list and must agree on its order, which a
        per-rank clock cannot guarantee.
        """
        mapping = getattr(kv_cache_manager, "mapping", None)
        if mapping is None:
            return True

        def size(name: str) -> int:
            value = getattr(mapping, name, 1)
            return value if isinstance(value, int) else 1

        if size("pp_size") > 1 or size("cp_size") > 1:
            return False
        return getattr(mapping, "enable_attention_dp", False) is True or size("tp_size") <= 1

    def _ctx_chunk_cap_for_pass(self, num_pending_ctx: int) -> Optional[int]:
        """The per-request chunk cap for one scheduling pass, or None when uncapped."""
        if self._ctx_request_chunk_cap is None or num_pending_ctx <= 1:
            return None
        if self.max_num_tokens is None:
            return self._ctx_request_chunk_cap
        unit = max(self.chunk_unit_size, 1)
        share = self.max_num_tokens // num_pending_ctx // unit * unit
        return max(self._ctx_request_chunk_cap, share)

    def _ctx_chunk_cap_for_share(
        self, remaining_budget: Optional[int], num_eligible_left: int
    ) -> Optional[int]:
        """The chunk cap of the next eligible request, given the budget still left."""
        if num_eligible_left <= 1:
            return None
        if remaining_budget is None:
            return self._ctx_request_chunk_cap
        unit = max(self.chunk_unit_size, 1)
        share = remaining_budget // num_eligible_left // unit * unit
        return max(self._ctx_request_chunk_cap, share)

    @staticmethod
    def _may_take_ctx_chunk(
        req: LlmRequest, defer_remote_tail_context: bool, exclusive_remote_mode
    ) -> bool:
        """The pass-invariant phase 2 filters a request must clear to be offered a chunk."""
        hold = getattr(req, "py_kv_gpu_hold", 0)
        if type(hold) is int and hold > 0:
            return False
        remote_mode = getattr(req, "py_csa2_remote_tail_mode", None)
        if defer_remote_tail_context and remote_mode is not None:
            return False
        return exclusive_remote_mode is None or remote_mode == exclusive_remote_mode

    def set_disagg_transfer_admission_controller(self, controller) -> None:
        """Register the executor's KV-transfer window (DisaggTransferAdmissionController).

        Each pass opens a DisaggTransferAdmissionGate on it. Once the gate closes, generation-init
        requests are skipped without KV allocation: the executor's FCFS transfer admission would
        defer them and revert the allocation.
        """
        self._disagg_transfer_controller = controller

    def _ctx_chunk_cap_applies(self, req: LlmRequest) -> bool:
        """Whether the per-request context chunk cap may shorten *req*'s next chunk.

        An atomic destination replay and a pre-grown SWA chunk must run at their
        exact size, so the fairness cap leaves them to the token budget alone.
        """
        if getattr(req, "py_csa2_remote_tail_mode", None) == "destination":
            return False
        return self._pregrown_context_tokens(req) == 0

    def _pregrown_context_tokens(self, req: LlmRequest) -> int:
        """KV capacity a context request holds past its position, left by a skipped forward.

        Under SWA scratch reuse, blocks of the grown chunk outside its end's
        window get no page; a shorter chunk would leave pages missing inside the
        next chunk's window (an invalid page index), so it must replay exactly.
        A suspended cache keeps its capacity, so it counts too.
        """
        if not self._swa_scratch_reuse:
            return 0
        kv_cache = self.kv_cache_manager.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return 0
        grown = (
            kv_cache.capacity
            - self.kv_cache_manager.num_extra_kv_tokens
            - req.context_current_position
        )
        return max(0, min(grown, req.context_remaining_length))

    def set_ctx_decoder_hold(self, window: int) -> None:
        """Hold every holdable final window of this pass (``window`` > 0), or nothing (0)."""
        self._num_ctx_decoder_hold_passes += self._ctx_decoder_held
        self._ctx_decoder_hold_window = int(window)
        self._ctx_decoder_held = False

    def _ctx_decoder_hold_chunk(self, req: LlmRequest, chunk_size: int) -> int:
        """The chunk of a hold pass: unchanged, cut back to the final window, or 0 (held whole)."""
        window = self._ctx_decoder_hold_window
        pos = req.context_current_position
        if ctx_decoder_batch.window_rows(
            pos, chunk_size, req.prompt_len, window
        ) == 0 or not ctx_decoder_batch.holdable(req):
            return chunk_size
        req.py_ctx_decoder_held = True
        self._ctx_decoder_held = True
        self._num_ctx_decoder_held_requests += 1
        return ctx_decoder_batch.hold_chunk(
            pos, chunk_size, req.prompt_len, window, self.chunk_unit_size
        )

    def plan_context_pass(
        self, active_requests: RequestList, inflight_ids, window: int
    ) -> ctx_decoder_batch.ContextPassPlan:
        """An estimate of this rank's context phase, without allocating anything: arrival order
        with long-prefill demotion, token budget, request cap and chunk unit; requests not started
        yet count their uncached-prefill estimate. The shortest-uncached-first and starvation
        orders, SWA scratch pre-growth and the pass's per-request skips are not modeled; a wrong
        estimate costs at most a deferred pass, since a request is held once. Holdable rows count
        only while the queue can refill the held tokens, youngest first, the order the scheduler
        reaches them."""
        candidates = [
            req
            for req in active_requests
            if not req.is_attention_dp_dummy
            and req.state_value == self._context_init_state_value
            and req.request_id not in inflight_ids
            and req.py_csa2_remote_tail_mode is None
        ]
        if self._ctx_demote_uncached_tokens:
            kept, demoted = [], []
            for req in candidates:
                demote = (
                    req.py_request_id not in self._ctx_demote_released
                    and self._is_long_context(req)
                    and not (
                        self._ctx_demote_max_iters
                        and self._ctx_demote_skips.get(req.py_request_id, 0)
                        >= self._ctx_demote_max_iters
                    )
                )
                (demoted if demote else kept).append(req)
            candidates = kept + demoted
        budget = self.max_num_tokens if self.max_num_tokens is not None else 1 << 62
        unit = self.chunk_unit_size
        plan = ctx_decoder_batch.ContextPassPlan()
        items = []  # (rows, holdable, held tokens)
        for req in candidates:
            remaining = int(req.context_remaining_length)
            estimate = req.py_uncached_prefill_tokens
            if req.is_first_context_chunk and isinstance(estimate, int) and estimate > 0:
                remaining = min(remaining, estimate)
            remaining = max(1, remaining)
            if budget < unit or len(items) >= self.max_num_requests:
                plan.backlog += remaining
                continue
            chunk = min(budget, remaining)
            if chunk < remaining:
                chunk = chunk // unit * unit
            if chunk <= 0:
                plan.backlog += remaining
                continue
            prompt_len = int(req.prompt_len)
            pos = prompt_len - remaining
            rows = ctx_decoder_batch.window_rows(pos, chunk, prompt_len, window)
            held_tokens = (
                chunk - ctx_decoder_batch.hold_chunk(pos, chunk, prompt_len, window, unit)
                if rows
                else 0
            )
            items.append((rows, ctx_decoder_batch.holdable(req), held_tokens))
            plan.rows += rows
            budget -= chunk
            plan.backlog += remaining - chunk
        refill = plan.backlog
        for rows, can_hold, held_tokens in reversed(items):
            if rows and can_hold and held_tokens <= refill:
                plan.holdable_rows += rows
                refill -= held_tokens
        return plan

    def set_async_transfer_manager(self, mgr) -> None:
        """Register the AsyncTransferManager the deadlock detector consults.

        A finished disaggregated context sender leaves active_requests once its
        response is emitted, but the transfer manager still owns its pinned KV
        pages until the send lands. Without this reference the detector cannot
        see those pages and reports a false deadlock.
        """
        self._async_transfer_manager = mgr

    def take_decision_counters(self) -> tuple[int, int, int, int]:
        counters = (
            self._num_gen_init_cap_holds,
            self._num_gen_init_transfer_budget_holds,
            self._num_ctx_decoder_hold_passes,
            self._num_ctx_decoder_held_requests,
        )
        self._num_gen_init_cap_holds = 0
        self._num_gen_init_transfer_budget_holds = 0
        self._num_ctx_decoder_hold_passes = 0
        self._num_ctx_decoder_held_requests = 0
        return counters

    def set_retiring_gen_receives_probe(self, probe: Callable[[], bool]) -> None:
        """Expose current rank-local receive ownership to the deadlock detector.

        Args:
            probe: Whether logically cancelled receives still retain local resources.
                Read current ownership, not the previous iteration's fleet-wide
                snapshot: remote retention cannot explain this rank's allocation stall.
                Retirement and its deadline remain the transfer layer's responsibility.
        """
        self._retiring_gen_receives_probe = probe

    @property
    def ctx_demote_uncached_tokens(self) -> int:
        """Effective long-prefill demotion threshold; 0 when off or unchunked."""
        return self._ctx_demote_uncached_tokens

    @property
    def scheduling_state_range(
        self,
    ) -> tuple[LlmRequestState, LlmRequestState]:
        return self.no_schedule_until_state, self.no_schedule_after_state

    def schedule_request(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> SchedulerOutput:
        self._sched_iter += 1
        self._oop_gate_iter += 1
        self._reuse_probes.clear()
        # A level that failed since the last pass (e.g. in an eviction WRITE during prepare_resources) has already
        # dropped pages from this rank's reuse tree only.
        self._raise_if_lockstep_remote_level_failed()
        if self._store_probes:
            self._prune_store_probes(active_requests)
        active_requests = drop_decoder_context_requests_waiting_for_encoder_output(active_requests)
        if self._ctx_srpt_max_wait_passes > 0:
            self._advance_ctx_srpt_clock(active_requests)
        if self._ctx_only_preempt:
            self._track_launched_context(active_requests)
        # Main scheduling loop
        (
            scheduled_encoder,
            scheduled_ctx,
            scheduled_gen,
            evicted,
            recompute_paused,
            disagg_candidates,
            has_chunking,
        ) = self._schedule_loop(active_requests, inflight_request_ids)
        # One that failed in this pass, e.g. in an eviction WRITE that the C++ retried past it, before the batch runs.
        self._raise_if_lockstep_remote_level_failed()
        if self._ctx_only_preempt:
            self._ctx_scheduled_prev = self._ctx_scheduled_last
            self._ctx_scheduled_last = frozenset(r.py_request_id for r in scheduled_ctx)
            self._ctx_scheduled_pos = {
                r.py_request_id: r.context_current_position for r in scheduled_ctx
            }
        ctx_preempt_stats = self._ctx_preempt_stats
        if ctx_preempt_stats.passes or ctx_preempt_stats.suspended:
            self._maybe_log_ctx_preemption_stats()
        if self._lease_touch is not None:
            self._lease_touch_pass(active_requests, scheduled_ctx)
        if self._oop_gate_cnt:
            self._oop_gate_maybe_log()

        # Sort by LoRA task ID
        scheduled_encoder.sort(key=_get_lora_task_id)
        self._sort_requests(scheduled_ctx, scheduled_gen, has_chunking)

        return SchedulerOutput(
            encoder_requests=scheduled_encoder,
            context_requests=scheduled_ctx,
            generation_requests=scheduled_gen,
            paused_requests=evicted,
            recompute_paused_requests=recompute_paused,
            fitting_disagg_gen_init_requests=disagg_candidates,
            num_fitting_requests=(len(scheduled_encoder) + len(scheduled_ctx) + len(scheduled_gen)),
        )

    def set_unprocessed_context(self, request_ids: frozenset[int]) -> None:
        """Record the context requests of the batch the executor keeps unprocessed.

        Under attention DP a rank whose batch cannot queue fleet-wide keeps the
        previous batch for a later pass, which may schedule no context work at
        all; context-only preemption must not free one of its requests before
        the executor processes that batch. The executor reports an empty set
        once no batch is held back.
        """
        self._ctx_unprocessed = request_ids

    def _track_launched_context(self, active_requests: RequestList) -> None:
        """Record which of the last pass's context requests may still run (context-only preemption)."""
        pos = self._ctx_scheduled_pos
        if not pos:
            # The last pass scheduled no context work here: its batch was empty,
            # and the executor processes the previous batch when this rank has
            # nothing, or held only an attention-DP dummy, which an idle rank gets
            # so the batch runs. A batch the executor keeps unprocessed instead
            # is protected through set_unprocessed_context.
            self._ctx_launched = frozenset()
        elif any(
            r.py_request_id in pos and r.context_current_position != pos[r.py_request_id]
            for r in active_requests
        ):
            self._ctx_launched = frozenset(pos)

    def set_remote_tail_phase(
        self,
        phase: RemoteTailPhase,
        context_request_ids: frozenset[int],
        generation_request_ids: frozenset[int],
    ) -> None:
        """Freeze this iteration's forward admission before any KV allocation.

        Transfers and newly fetched requests can progress after the ADP vote,
        but their forward work becomes eligible only in the next iteration.
        """
        self.remote_tail_phase = phase
        self._remote_tail_context_ids = context_request_ids
        self._remote_tail_generation_ids = generation_request_ids

    def set_gen_paused_ids(
        self, request_ids: frozenset[int], resumed_ids: frozenset[int] = frozenset()
    ) -> None:
        """Leave paused requests resident and give resumes priority this pass."""
        self._gen_preemption_enabled = True
        self._gen_paused_ids = request_ids
        self._gen_resumed_ids = resumed_ids

    def _filter_paused_requests(self, requests: RequestList) -> RequestList:
        paused = [req for req in requests if req.py_request_id in self._gen_paused_ids]
        for req in paused:
            # Resume from the host token, not a stale overlap tensor.
            self._clear_request_runtime_state(req)
        return [req for req in requests if req.py_request_id not in self._gen_paused_ids]

    def _filter_remote_tail_phase_requests(self, requests: RequestList) -> RequestList:
        phase = self.remote_tail_phase
        if phase is None:
            return requests
        eligible = []
        for req in requests:
            if req.state_value in (self._context_init_state_value, self._encoder_init_state_value):
                if (
                    phase in (RemoteTailPhase.CONTEXT, RemoteTailPhase.MIXED)
                    and req.request_id in self._remote_tail_context_ids
                ):
                    eligible.append(req)
            elif self.is_request_in_schedulable_state(req):
                if (
                    phase in (RemoteTailPhase.DECODE, RemoteTailPhase.MIXED)
                    and req.request_id in self._remote_tail_generation_ids
                ):
                    eligible.append(req)
                else:
                    # The previous sample is harvested while this request is
                    # deferred. Resume from its host token, not a stale overlap
                    # tensor that now belongs to another phase.
                    self._clear_request_runtime_state(req)
            else:
                # Keep transfer admission and terminal PEFT accounting alive.
                eligible.append(req)
        return eligible

    # ---- Main scheduling loop ----

    def _schedule_loop(
        self,
        active_requests,
        inflight_request_ids,
        *,
        defer_remote_tail_context=False,
        prefix_waits=None,
    ):
        scheduled_ctx: RequestList = []
        scheduled_remote_mode = None
        scheduled_encoder: RequestList = []
        scheduled_gen: RequestList = []
        evicted: RequestList = []
        recompute_paused: RequestList = []
        disagg_candidates: RequestList = []
        scheduled_beam_width = 0
        has_chunking = False

        budget = BudgetTracker(
            self.max_num_tokens,
            self.max_num_requests,
            self.peft_cache_manager,
        )

        # Use indexed iteration (while + req_it_end) so that MAX_UTIL
        # eviction can shrink the range from the tail.
        requests_list = self._filter_remote_tail_phase_requests(list(active_requests))
        if self._gen_paused_ids:
            requests_list = self._filter_paused_requests(requests_list)
        exclusive_remote_mode = None
        if not defer_remote_tail_context and self.remote_tail_phase is not RemoteTailPhase.DECODE:
            # Only context requests can carry an exclusive mode, so test the state
            # first: during decode every request fails it, one cheap compare each.
            context_init_state_value = self._context_init_state_value
            exclusive_remote_mode = next(
                (
                    getattr(req, "py_csa2_remote_tail_mode", None)
                    for req in requests_list
                    if req.state_value == context_init_state_value
                    and getattr(req, "py_csa2_remote_tail_mode", None) == "source"
                    and req.request_id not in inflight_request_ids
                ),
                None,
            )

        # Opt-in: prioritize disagg-gen first-token requests (see __init__).
        # py_decoding_iter == 0 covers DISAGG_GENERATION_INIT /
        # TRANS_IN_PROGRESS / TRANS_COMPLETE. A stable sort keeps FIFO arrival
        # order within each class; no-op without generation-only requests.
        if self._prioritize_first_token_gen:
            requests_list.sort(
                key=lambda req: (
                    0 if (req.is_generation_only_request and req.py_decoding_iter == 0) else 1
                )
            )

        # Resume deadlines take precedence over the first-token preference.
        if self._gen_resumed_ids:
            requests_list.sort(key=lambda req: req.py_request_id not in self._gen_resumed_ids)

        req_it_end = len(requests_list)
        recompute_pause_state = _RecomputePauseState(req_it_end)
        req_it = 0
        disagg_gate = (
            self._disagg_transfer_controller.begin_gate(active_requests)
            if self._disagg_transfer_controller is not None
            else None
        )

        # Context requests are always deferred to a second phase so that
        # generation requests are fully accounted for in the budget before
        # any context request competes for resources.  This mirrors the
        # two-phase approach of the old KVCacheV2DummyScheduler and
        # prevents PEFT adapter eviction failures when gen requests hold
        # adapters that can't be evicted mid-iteration.
        pending_ctx: RequestList = []
        # Index of each pending context request in requests_list.
        pending_ctx_index: dict[int, int] = {}

        # Pre-claim PEFT pages for terminal requests and, with GEN preemption,
        # every live decode, including requests omitted from this batch.
        # In the overlap executor these requests are no longer scheduled
        # (state outside schedulable range) but their adapters haven't
        # been released yet (mark_request_done runs after prepare_resources
        # in the next iteration).  Without this, the budget would appear
        # empty and context requests with a different adapter could be
        # admitted, causing ensure_batch to crash when it can't evict the
        # still-active adapter.
        if self.peft_cache_manager is not None:
            for req in active_requests:
                if req.state_value == self._gen_to_complete_state_value or (
                    self._gen_preemption_enabled
                    and req.state_value == LlmRequestState.GENERATION_IN_PROGRESS.value
                ):
                    # Paused and displaced decode requests still hold their adapters.
                    budget.pre_claim_peft(req)

        # --- Phase 1: generation / disagg only ---
        # Membership costs two binding calls per request; the set is usually empty.
        has_inflight = len(inflight_request_ids) > 0
        while req_it < req_it_end:
            req = requests_list[req_it]

            if req_it in recompute_pause_state.victim_indices:
                req_it += 1
                continue

            # --- Filter ---
            if has_inflight and req.request_id in inflight_request_ids:
                req_it += 1
                continue

            req_state_value = req.state_value

            # Disagg gen init bypasses both state gating and budget.requests_full
            # (same as C++ / V1 scheduler), but the V2 scheduler owns inline KV
            # allocation so we must allocate here. V1 defers allocation to
            # prepare_resources; V2 prepare_resources is a no-op for the primary
            # manager, so allocation must happen in the scheduling loop.
            #
            # Disagg does NOT count toward budget.num_requests because it
            # doesn't participate in the forward pass. Capacity is gated by
            # IndexMapper slot availability: prepare_context returns False when
            # no free slots remain, so the request is skipped and retried next
            # iteration. PEFT budget is still checked and committed.
            if req_state_value == self._disagg_gen_init_state_value:
                if (
                    self.max_gen_init_admissions_per_pass is not None
                    and len(disagg_candidates) >= self.max_gen_init_admissions_per_pass
                ):
                    # Admitted in a later pass, in order.
                    self._num_gen_init_cap_holds += 1
                    req_it += 1
                    continue
                peft_pages = budget.peft_pages_needed(req)
                if peft_pages is None:
                    break

                if disagg_gate is not None and disagg_gate.closed:
                    # The KV-transfer budget of this pass is exhausted: the executor's FCFS admission
                    # defers every candidate after the one that closed the gate and reverts its KV,
                    # so do not allocate it. PEFT is charged as for such a deferred candidate, so the
                    # rest of the pass sees the same budget as without the gate.
                    disagg_gate.num_skipped += 1
                    self._num_gen_init_transfer_budget_holds += 1
                    if peft_pages > 0:
                        budget.commit_peft(req, peft_pages)
                    req_it += 1
                    continue

                action, tokens = self._try_schedule_disagg_gen_init(req, budget)
                if action is ScheduleAction.STOP:
                    break
                if action is ScheduleAction.SKIP:
                    req_it += 1
                    continue
                disagg_candidates.append(req)
                if disagg_gate is not None:
                    disagg_gate.observe_candidate(req)
                # Disagg requests only commit PEFT (not num_requests/num_tokens)
                # because they don't participate in the forward pass. Counting
                # them toward num_requests would steal batch slots from gen/ctx
                # and delay KV transfer initiation. IndexMapper slot availability
                # (via prepare_context) is the real capacity guard.
                if peft_pages > 0:
                    budget.commit_peft(req, peft_pages)
                req_it += 1
                continue

            # Budget check for non-disagg requests (disagg bypasses this
            # because it doesn't participate in the forward pass).
            if budget.requests_full:
                break

            if not (
                req_state_value >= self._no_schedule_until_state_value
                and req_state_value < self._no_schedule_after_state_value
            ):
                req_it += 1
                continue

            # --- Dispatch by request type ---
            if (
                req_state_value == self._encoder_init_state_value
                or req_state_value == self._context_init_state_value
            ):
                # Defer to phase 2 so gen PEFT budget is settled first.
                pending_ctx.append(req)
                pending_ctx_index[req.py_request_id] = req_it
                req_it += 1
                continue

            else:
                if exclusive_remote_mode is not None:
                    # Source-only batches harvest the preceding decode on CPU
                    # without launching its next step. On resume, use the host
                    # token count rather than the one-step-behind overlap path.
                    self._clear_request_runtime_state(req)
                    req_it += 1
                    continue
                peft_pages = budget.peft_pages_needed(req)
                if peft_pages is None:
                    break
                action, tokens, scheduled_beam_width, req_it_end = self._try_schedule_generation(
                    req,
                    budget,
                    requests_list,
                    req_it,
                    req_it_end,
                    recompute_pause_state,
                    evicted,
                    recompute_paused,
                    inflight_request_ids,
                    scheduled_beam_width,
                )
                if action is ScheduleAction.STOP:
                    break
                if action is ScheduleAction.SKIP:
                    req_it += 1
                    continue
                scheduled_gen.append(req)
                budget.commit(req, tokens, peft_pages)

            req_it += 1

        # Requests whose pages were given up during this pass. A victim that is
        # itself a started context request still sits in pending_ctx, so
        # re-admitting it would spend the pages its own preemption released.
        preempted_ids: set[int] = set()

        # Long-prefill demotion (see __init__). Reorder here rather than on
        # requests_list: phase 1 walks that list by index, pending_ctx is this
        # loop's own.
        arrival_ctx = pending_ctx
        demoted_ids: set[int] = set()
        if self._ctx_demote_uncached_tokens:
            pending_ctx, demoted_ids = self._defer_long_context(pending_ctx)

        def preempt_for_pages(req: LlmRequest) -> bool:
            """Free pages for `req` by giving up one started request.

            A success ends the phase 2 loop, reserving the pages for the
            request that paid a re-prefill for them. Letting a later context
            request take them instead would leave `req` preempting again every
            pass without ever being admitted.
            """
            protected = {r.py_request_id for r in scheduled_gen}
            protected.update(r.py_request_id for r in scheduled_ctx)
            protected.add(req.py_request_id)
            # A request this pass already suspended or recompute-paused has
            # given up what it can for now; taking it again would list it
            # twice and report pages that were never released.
            protected.update(r.py_request_id for r in evicted)
            protected.update(r.py_request_id for r in recompute_paused)
            # Demotion reorders compute, not page priority: a demoted request
            # that arrived before `req` would have been scheduled, and so
            # protected, ahead of it in FCFS order.
            if demoted_ids:
                for other in arrival_ctx:
                    if other is req:
                        break
                    if other.py_request_id in demoted_ids:
                        protected.add(other.py_request_id)
            # Shortest-uncached-first moves unstarted requests ahead of started ones; letting
            # them also take a started request's pages would discard its finished chunks.
            if self._ctx_srpt_max_wait_passes > 0 and req.is_first_context_chunk:
                protected.update(self._started_context_ids(requests_list))
            return self._try_preempt_for_pages(
                requests_list,
                protected,
                inflight_request_ids,
                recompute_paused,
                preempted_ids,
                pending_ctx_index[req.py_request_id],
                locality_domain_id=getattr(req, "py_locality_domain_id", None),
            )

        # --- Phase 2: schedule deferred context / encoder requests ---
        # Generation PEFT pages are now fully committed in the budget.
        #
        # Prefix-aware skip: when several first-chunk context requests would
        # contribute the same not-yet-cached block, admitting them together
        # makes every one of them recompute that prefix. Admit one, defer the
        # rest by one iteration and let them reuse the block it commits.
        # `_collect_contributed_blocks` returns the blocks already promised by
        # context requests running this iteration, or None when the skip cannot
        # fire at all and every probe below can be avoided.
        contributed_blocks = self._collect_contributed_blocks(
            requests_list, pending_ctx, inflight_request_ids
        )
        if self._oop_gate_track:
            self._oop_gate_begin_phase2(pending_ctx)
        if pending_ctx:
            self._ctx_preempt_stats.passes += 1
        # A deferral behind an in-flight contributor leaves nothing on any of
        # the scheduled lists, so the deadlock detector below would read the
        # iteration as a stall even though the contributor is running. Deferring
        # is itself the progress in that case.
        deferred_behind_contributor = False
        # Only context-only batches take Mamba snapshots inside the forward.
        self._inforward_snapshot_rows_left = 0 if scheduled_gen else self._inforward_snapshot_rows
        first_chunk_contributors: dict = {}
        context_init_ids = (
            {
                r.py_request_id
                for r in requests_list
                if r.state_value == self._context_init_state_value
            }
            if contributed_blocks is not None and self._prev_first_chunk_contributors
            else frozenset()
        )

        if self._ctx_shortest_first and len(pending_ctx) > 1:
            pending_ctx = self._order_ctx_shortest_first(pending_ctx)
        starved_ctx: RequestList = []
        if self._ctx_starvation_iterations is not None:
            pending_ctx, starved_ctx = self._split_starved_ctx(pending_ctx)
        # After any reordering: pre-grown requests go first in each pass so no
        # other request takes the budget their replay needs (stable sort).
        if self._swa_scratch_reuse:
            pending_ctx, starved_ctx = (
                sorted(
                    reqs,
                    key=lambda r: (
                        not (
                            r.state_value == self._context_init_state_value
                            and self._pregrown_context_tokens(r) > 0
                        )
                    ),
                )
                for reqs in (pending_ctx, starved_ctx)
            )
        # New requests that wait for a prefix another request prefills, planned
        # in scheduling order. Planned once per schedule_request, so a retry
        # below spends no wait budget.
        if prefix_waits is None:
            prefix_waits = (
                self._prefix_wait_planner.plan(pending_ctx + starved_ctx)
                if self._prefix_wait_planner is not None and (pending_ctx or starved_ctx)
                else {}
            )
            if prefix_waits:
                self.kv_cache_manager.count_prefix_waits(len(prefix_waits))
        if self._ctx_demote_exact and prefix_waits:
            # The probe of a waiting request predates the prefix it waits for;
            # it is probed again once it stops waiting.
            for req in pending_ctx + starved_ctx:
                if req.py_request_id in prefix_waits:
                    req.py_uncached_prefill_probed = False
        # Starved requests are scheduled after the others, but their first shares
        # and request slots are held back while the others fill the budget.
        starved_shares = self._starved_ctx_shares(starved_ctx, budget)
        budget.hold(sum(starved_shares), len(starved_ctx))
        held = 0
        ctx_stopped = False
        num_ctx_candidates = len(pending_ctx) + len(starved_ctx)
        self._ctx_chunk_cap_this_pass = self._ctx_chunk_cap_for_pass(num_ctx_candidates)
        # Requests already taken off eligible_left (a DEFER may visit one again). A prefix wait
        # with no owner in this pass waits the whole pass, so it is never counted: an earlier
        # request then sizes its share without it, whatever the order.
        uncounted_ids: set[int] = {
            req_id for req_id, owner_id in prefix_waits.items() if owner_id is None
        }
        eligible_left = (
            sum(
                self._may_take_ctx_chunk(req, defer_remote_tail_context, exclusive_remote_mode)
                for reqs in (pending_ctx, starved_ctx)
                for req in reqs
                if req.py_request_id not in uncounted_ids
            )
            if self._ctx_chunk_cap_eligible_share and self._ctx_chunk_cap_this_pass is not None
            else None
        )
        for starved_pass, ctx_requests in ((False, pending_ctx), (True, starved_ctx)):
            if starved_pass:
                if not starved_ctx:
                    break
                held = sum(starved_shares)
                budget.release(held, len(starved_ctx))
                if ctx_stopped:
                    break
            # A DEFER re-appends its request to the tail of this pass, so the
            # pass iterates over its own copy.
            ctx_requests = list(ctx_requests)
            for idx, req in enumerate(ctx_requests):
                token_cap = None
                if starved_pass:
                    # Its own share plus what the shortest-first fill left,
                    # minus the shares still held for the starved requests after it.
                    held -= starved_shares[idx]
                    token_cap = budget.remaining_tokens - held
                if budget.requests_full:
                    break
                if req.py_request_id in prefix_waits:
                    # A request new in this pass only prefills the prefix if it was
                    # admitted ahead of this one. A prefix wait is not progress; see
                    # the invariant below.
                    owner_id = prefix_waits[req.py_request_id]
                    if owner_id is None or any(r.py_request_id == owner_id for r in scheduled_ctx):
                        # It takes no share of the budget this pass.
                        if (
                            eligible_left is not None
                            and req.py_request_id not in uncounted_ids
                            and self._may_take_ctx_chunk(
                                req, defer_remote_tail_context, exclusive_remote_mode
                            )
                        ):
                            uncounted_ids.add(req.py_request_id)
                            eligible_left -= 1
                        continue
                # A radix probe cannot affect admission once the chunk token budget
                # is exhausted. Keep scanning: an encoder request may still fit.
                if (
                    self.chunking_enabled
                    and req.state_value == self._context_init_state_value
                    and not self._has_context_chunk_budget(budget)
                ):
                    continue
                if scheduled_ctx and self._first_chunk_waits_for_budget(req, budget):
                    continue
                if req.py_request_id in preempted_ids:
                    continue
                # Its host-tier prefix is being copied to the GPU off the
                # execution stream (KVCacheManagerV2.prefetch_to_gpu); admit it
                # when the hold ends.
                hold = getattr(req, "py_kv_gpu_hold", 0)
                if type(hold) is int and hold > 0:
                    continue
                remote_mode = getattr(req, "py_csa2_remote_tail_mode", None)
                if defer_remote_tail_context and remote_mode is not None:
                    continue
                if exclusive_remote_mode is not None and remote_mode != exclusive_remote_mode:
                    continue
                if eligible_left is not None:
                    self._ctx_chunk_cap_this_pass = self._ctx_chunk_cap_for_share(
                        budget.remaining_tokens, eligible_left
                    )
                    uncounted_ids.add(req.py_request_id)
                    eligible_left -= 1
                # Destination replay is atomic; source prefix processing may be chunked.
                if remote_mode == "destination" and not budget.can_fit_tokens(
                    req.context_remaining_length
                ):
                    continue
                if (
                    scheduled_ctx
                    and remote_mode != scheduled_remote_mode
                    and (remote_mode == "source" or scheduled_remote_mode == "source")
                ):
                    continue
                # Probe context requests before peft_pages_needed and before
                # _try_schedule_context so that a deferral costs nothing: KV pages
                # are allocated inline, so a skip decided after prepare_context
                # would pay a create/suspend cycle. Chunk continuations are probed
                # too -- they cannot be deferred, but they do contribute a block, so
                # they register once scheduled.
                first_new_block = None
                if contributed_blocks is not None and (
                    req.state_value == self._context_init_state_value
                ):
                    first_new_block = self.kv_cache_manager.probe_first_new_block_key(req)
                    if (
                        req.is_first_context_chunk
                        and first_new_block is not None
                        and first_new_block in contributed_blocks
                        and self._deferral_paid(first_new_block, context_init_ids)
                    ):
                        logger.debug(
                            f"Deferring context request {req.py_request_id}: its first new "
                            "block is already contributed by a request that runs this iteration"
                        )
                        deferred_behind_contributor = True
                        continue
                peft_pages = budget.peft_pages_needed(req)
                if peft_pages is None:
                    continue
                if req.state_value == self._encoder_init_state_value:
                    action, tokens = self._try_schedule_encoder(req, budget)
                    if action is ScheduleAction.STOP:
                        ctx_stopped = True
                        break
                    scheduled_encoder.append(req)
                    budget.commit(req, tokens, peft_pages)
                else:
                    try:
                        action, tokens, chunking_flag = self._try_schedule_context(
                            req, budget, preempt_for_pages, token_cap
                        )
                    except RemoteMemError as error:
                        # Raised by its cache's resume, before anything was taken for it.
                        self._lost_kv_error = error
                        action = self._on_lost_kv_cache(req, inflight_request_ids, recompute_paused)
                    if action is ScheduleAction.STOP:
                        ctx_stopped = True
                        break
                    if action is ScheduleAction.SKIP:
                        continue
                    if action is ScheduleAction.DEFER:
                        # Ordered as short, prepared, found long: it goes behind
                        # the rest of this pass and counts as demoted for aging.
                        # Starved requests carry a token cap and are never deferred.
                        demoted_ids.add(req.py_request_id)
                        ctx_requests.append(req)
                        continue
                    has_chunking = has_chunking or chunking_flag
                    scheduled_ctx.append(req)
                    if self._prefix_wait_planner is not None:
                        self._prefix_wait_planner.add_owner(req)
                    if len(scheduled_ctx) == 1:
                        scheduled_remote_mode = remote_mode
                    budget.commit(req, tokens, peft_pages)
                    # Register the block only once the request has cleared its
                    # budget and is committed. Registering earlier, inside the check
                    # above, would let a request that then fails to schedule defer
                    # its duplicates anyway -- and this loop `continue`s on every
                    # ScheduleAction.SKIP, so failing to schedule is routine.
                    #
                    # Together with the in-flight-only pre-pass this gives the
                    # invariant that keeps the deadlock detector below sound:
                    # a request is only ever deferred behind a contributor that
                    # actually runs this iteration, so a deferral always leaves
                    # either inflight_request_ids or scheduled_ctx non-empty.
                    #
                    # A prefix wait (PrefixWaitPlanner) is deliberately not
                    # progress. Its owner is either scheduled in this pass or a
                    # prefilling request that is itself a context candidate, and
                    # each request waits at most MAX_WAIT_PASSES (128) passes in
                    # total, well below _DEADLOCK_STALL_ITERS. A
                    # pass that only waits therefore counts as a stall but cannot
                    # trip the detector by itself, while counting it as progress
                    # would let a stalled owner plus new sibling arrivals hide a
                    # real deadlock.
                    if first_new_block is not None:
                        contributed_blocks.add(first_new_block)
                        if req.is_first_context_chunk:
                            first_chunk_contributors[first_new_block] = req.py_request_id

        self._prev_first_chunk_contributors = first_chunk_contributors
        self._ctx_chunk_cap_this_pass = None
        # Context fill v2 backlog gate input for the next pass.
        self._ctx_fill_backlog = num_ctx_candidates - len(scheduled_ctx)
        if self._ctx_srpt_max_wait_passes > 0 and not self._ctx_srpt_age_from_arrival:
            for req in scheduled_ctx:
                self._ctx_srpt_waiting_since[req.py_request_id] = self._ctx_srpt_pass

        made_progress = bool(
            scheduled_gen
            or scheduled_ctx
            or scheduled_encoder
            or disagg_candidates
            or evicted
            or recompute_paused
            or deferred_behind_contributor
            # A decoder-batching hold defers a request by one pass on purpose.
            or self._ctx_decoder_held
        )
        if (
            self.remote_tail_phase is None
            and exclusive_remote_mode is not None
            and not made_progress
        ):
            # A non-ADP replay that cannot acquire KV must let decode release
            # pages. Retry admission without replay; keep its forward isolated.
            # Pass every active request so the retry keeps the PEFT claims of
            # paused decodes, which requests_list has already dropped.
            return self._schedule_loop(
                active_requests,
                inflight_request_ids,
                defer_remote_tail_context=True,
                prefix_waits=prefix_waits,
            )

        if self._ctx_demote_uncached_tokens:
            # Iterations spent demoted. Rebuilt every iteration from the
            # requests demoted in it, so it cannot leak.
            self._ctx_demote_skips = {
                rid: self._ctx_demote_skips.get(rid, 0) + 1 for rid in demoted_ids
            }
            self._ctx_demote_released.intersection_update(
                req.py_request_id for req in active_requests
            )

        # Phase and frozen-membership deferrals are not KV capacity stalls.
        self._detect_deadlock(
            requests_list,
            inflight_request_ids,
            pending_ctx,
            preempted_ids,
            made_progress=made_progress,
        )
        if disagg_gate is not None and disagg_gate.num_skipped:
            logger.debug(
                f"Disagg transfer budget exhausted: skipped KV allocation for "
                f"{disagg_gate.num_skipped} generation-init requests"
            )

        return (
            scheduled_encoder,
            scheduled_ctx,
            scheduled_gen,
            evicted,
            recompute_paused,
            disagg_candidates,
            has_chunking,
        )

    # ---- Long-prefill demotion ----

    def _is_long_context(self, req: LlmRequest) -> bool:
        """Whether the router's uncached-prefill estimate exceeds the threshold
        and, once mid-prefill, more than the threshold is actually left. A
        request without an estimate is never long. No radix probe here unless
        exact demotion is on (then at most one per request): it would walk the
        whole prompt for every request on every iteration."""
        if req.state_value == self._encoder_init_state_value:
            return False
        estimate = req.py_uncached_prefill_tokens
        if estimate is None:
            return False
        threshold = self._ctx_demote_uncached_tokens
        if self._ctx_demote_exact:
            if req.context_current_position > 0 or self._holds_context_cache(req):
                return req.context_remaining_length > threshold
            if estimate > threshold and not req.py_uncached_prefill_probed:
                req.py_uncached_prefill_probed = True
                reused = self.kv_cache_manager.probe_context_reuse(req)
                if reused is not None:
                    estimate = max(req.prompt_len - reused, 0)
                    req.py_uncached_prefill_tokens = estimate
            return estimate > threshold
        if estimate <= threshold:
            return False
        return not (req.context_current_position > 0 and req.context_remaining_length <= threshold)

    def _defer_long_context(self, pending_ctx: RequestList) -> tuple[RequestList, set[int]]:
        """Move long requests to the tail of this iteration's context order,
        and hold requests without an estimate behind started demoted prefills.

        Returns the new order and the ids demoted. With exact demotion also
        records which requests were kept on the estimate alone (see
        _try_schedule_context_chunked).
        """
        keep: RequestList = []
        deferred: RequestList = []
        self._ctx_unverified_short = set()
        for req in pending_ctx:
            rid = req.py_request_id
            if rid not in self._ctx_demote_released and self._is_long_context(req):
                aged_out = (
                    self._ctx_demote_max_iters > 0
                    and self._ctx_demote_skips.get(rid, 0) >= self._ctx_demote_max_iters
                )
                if not aged_out:
                    deferred.append(req)
                    continue
                # Let through for good: a released request keeps its arrival
                # order to the end of its prefill. Demoted again, a started
                # prefill would hold its KV blocks without progress; a chunk
                # that finds no pages is skipped, not suspended, so enough of
                # them fill the pool and no context request can allocate.
                self._ctx_demote_released.add(rid)
            elif (
                self._ctx_demote_exact
                and rid not in self._ctx_demote_released
                and req.py_uncached_prefill_tokens is not None
                and req.context_current_position == 0
                and not self._holds_context_cache(req)
            ):
                self._ctx_unverified_short.add(req.py_request_id)
            keep.append(req)
        demoted = {req.py_request_id for req in deferred}
        # A request without an estimate is of unknown length. Ahead of a
        # demoted prefill that has started, it could start a prefill of any
        # size; once the started one aged out and took the budget back, it
        # would be parked mid-prefill with its pages held, and with a cache
        # tier below the GPU neither can be preempted. So one that has not
        # started yet waits behind the started demoted prefills that arrived
        # before it, at most until they finish; requests known to be short
        # still go first. The demoted requests keep their arrival order and
        # the held ones go right after the last started one.
        started = [i for i, req in enumerate(deferred) if self._started_context(req)]
        if started:
            arrival = {req.py_request_id: i for i, req in enumerate(pending_ctx)}
            first = arrival[deferred[started[0]].py_request_id]
            last = started[-1]
            held = [
                req
                for req in keep
                if req.py_uncached_prefill_tokens is None
                and arrival[req.py_request_id] > first
                and not self._started_context(req)
            ]
            if held:
                held_ids = {req.py_request_id for req in held}
                keep = [req for req in keep if req.py_request_id not in held_ids]
                return keep + deferred[: last + 1] + held + deferred[last + 1 :], demoted
        return keep + deferred, demoted

    def _started_context(self, req: LlmRequest) -> bool:
        """Whether *req* has begun its prefill, so that parking it would hold its pages."""
        return req.context_current_position > 0 or self._holds_context_cache(req)

    # ---- Shortest-uncached-first context order ----

    def _uncached_tokens(self, req: LlmRequest) -> int:
        """Prompt tokens *req* still has to compute, net of reusable cached blocks.

        A first-chunk request has not claimed its reusable prefix yet, so its
        remaining length still counts it; the prefix is probed once per
        scheduling pass, without claiming pages, and capped like the claim
        (``context_reuse_claim_limit``). With store-aware ranking the prefix a
        KV connector can load counts too (see _store_prefix_end).
        """
        remaining = req.context_remaining_length
        if not req.is_first_context_chunk:
            return remaining
        if req.py_request_id not in self._reuse_probes:
            self._reuse_probes[req.py_request_id] = self.kv_cache_manager.probe_context_reuse(req)
        reusable = self._reuse_probes[req.py_request_id]
        if self._ctx_store_aware:
            store_end = self._store_prefix_end(req, reusable)
            if store_end is not None and (reusable is None or store_end > reusable):
                reusable = store_end
        if reusable is None:
            return remaining
        limit = self.kv_cache_manager.context_reuse_claim_limit(req)
        if limit is not None:
            reusable = min(reusable, limit)
        return remaining - min(max(reusable - req.context_current_position, 0), remaining)

    def _store_prefix_end(self, req: LlmRequest, local_reusable: Optional[int]) -> Optional[int]:
        """End of the prompt prefix the KV connector can load for *req*, or None.

        Probed once per request, from the local depth when it is first ranked: the
        query that loads a prefix runs once per allocation and may start the
        transfer, so ranking cannot reuse it. Blocks stored later leave the rank
        conservative, never ahead of what the query will offer.
        """
        request_id = req.py_request_id
        if request_id in self._store_probes:
            return self._store_probes[request_id]
        local_end = max(local_reusable or 0, req.context_current_position)
        offered = self.kv_cache_manager.probe_connector_prefix(req, local_end)
        if self._lease_touch is not None:
            self._lease_touch.note_probe(request_id)
        store_end = min(local_end + offered, req.prompt_len - 1) if offered else None
        self._store_probes[request_id] = store_end
        return store_end

    def _lease_touch_pass(self, active_requests: RequestList, scheduled_ctx: RequestList) -> None:
        """Re-probe the waiting first-chunk context requests whose probe is due.

        See `_StoreLeaseTouch`. Fails open: an error turns the touch off for this
        scheduler and is logged once.
        """
        lease = self._lease_touch
        manager = self.kv_cache_manager
        connector = getattr(manager, "kv_connector_manager", None)
        if not lease.engages(connector):
            lease.last.clear()
            return
        try:
            now = lease.clock()
            admitted = {req.py_request_id for req in scheduled_ctx}
            waiting = []
            seen = set()
            for req in active_requests:
                if (
                    req.state_value != self._context_init_state_value
                    or not req.is_first_context_chunk
                    or req.py_connector_allocation_reported
                ):
                    continue
                request_id = req.py_request_id
                seen.add(request_id)
                if request_id in admitted:
                    # Its own lookup runs at allocation.
                    continue
                last = lease.last.get(request_id)
                if last is None:
                    # First seen unprobed: due once it has waited first_s.
                    last = now - lease.interval_s + lease.first_s
                    lease.last[request_id] = last
                if now - last >= lease.interval_s:
                    waiting.append((last, request_id, req))
            for request_id in [rid for rid in lease.last if rid not in seen]:
                del lease.last[request_id]
            waiting.sort(key=lambda item: item[0])
            for _, request_id, req in waiting[: lease.max_per_pass]:
                lease.last[request_id] = now
                local_end = max(
                    self._reuse_probes.get(request_id) or 0, req.context_current_position
                )
                offered = manager.probe_connector_prefix(req, local_end)
                if offered is None:
                    continue
                lease.touches += 1
                # A stored prefix shorter than the ranking probe saw was lost while queued.
                ranked_end = self._store_probes.get(request_id)
                if ranked_end is not None:
                    now_end = min(local_end + offered, req.prompt_len - 1) if offered else local_end
                    if now_end < ranked_end:
                        lease.shrunk += 1
                        lease.lost_tokens += ranked_end - now_end
                        # Rank and fill from the renewed offer, not the stale one.
                        self._store_probes[request_id] = now_end if offered else None
                        self._ctx_fill_defers.pop(request_id, None)
            lease.maybe_log(now)
        except Exception as error:  # noqa: BLE001 - a lease renewal must never fail scheduling
            logger.warning(
                f"KVCacheV2Scheduler store lease touch off after an error: "
                f"{type(error).__name__}: {error}"
            )
            self._lease_touch = None

    def _prune_store_probes(self, active_requests: RequestList) -> None:
        """Forget the store probe of every request that is no longer ranked."""
        ranked = {
            req.py_request_id
            for req in active_requests
            if req.state_value == self._context_init_state_value and req.is_first_context_chunk
        }
        for request_id in [rid for rid in self._store_probes if rid not in ranked]:
            del self._store_probes[request_id]
        for request_id in [rid for rid in self._ctx_fill_defers if rid not in ranked]:
            del self._ctx_fill_defers[request_id]

    # ---- Context fill: size store-served first chunks by what they compute ----

    def _ctx_fill_chunk(self, req: LlmRequest, chunk_size: int) -> int:
        """Size a first chunk whose ranked store probe predicts a connector-served prefix.

        The connector is queried only after the batch is final, and the prefix it serves
        is never computed: an asynchronous load takes the request out of this step's
        forward pass, and a synchronous one shifts the chunk to start at the served end.
        Either way the forward pass computes at most ``context_chunk_size`` tokens, so a
        smaller chunk keeps the budget charge an upper bound even when the store answers
        less than the probe did. The chunk shrinks only when the probe puts the served end
        at or past the chunk's own end:

        * asynchronous loads: one chunk unit (at least one block), the smallest allocation
          that still lets the shifted window cover a block;
        * synchronous loads: the tokens past the probed end, rounded up to the chunk unit.

        Only the probe cached by the ranking pass is read, so no store query is added.
        Returns ``chunk_size`` unchanged when there is no usable probe or loading mode.
        """
        store_end = self._store_probes.get(req.py_request_id)
        if store_end is None:
            return chunk_size
        position = req.context_current_position
        loads_async = self.kv_cache_manager.connector_loads_prefix_async()
        if loads_async is not True and loads_async is not False:
            return chunk_size
        unit = max(self.chunk_unit_size, 1)
        if loads_async:
            block = max(self.tokens_per_block, 1)
            fill = -(-max(unit, block) // unit) * unit
        else:
            fill = -(-(req.prompt_len - store_end) // unit) * unit
        if fill >= chunk_size or store_end < position + fill:
            return chunk_size
        counts = self._ctx_fill_counts
        if self._ctx_fill_v2:
            if self._ctx_fill_backlog < self._ctx_fill_min_backlog:
                counts[3] += 1
                self._maybe_log_ctx_fill()
                return chunk_size
            if loads_async:
                if self._ctx_fill_defers.get(req.py_request_id, 0) >= self._ctx_fill_max_defer:
                    counts[6] += 1
                    self._maybe_log_ctx_fill()
                    return chunk_size
                self._ctx_fill_reserve_end = store_end
        counts[0 if loads_async else 1] += 1
        counts[2] += chunk_size - fill
        self._maybe_log_ctx_fill()
        return fill

    def _ctx_fill_reserve_served_prefix(self, req: LlmRequest, end: int) -> bool:
        """Reserve the pages of a shrunk chunk's probed served prefix (context fill v2).

        The connector is queried after the batch is final, and its hook then grows the
        allocation to cover the served prefix. A shrunk chunk has no pages of its own to fall
        back on, so a growth that fails there discards the whole hit. Growing here charges
        the prefix while the pass still decides admission; on failure the request is deferred
        (False) and the caller's rollback frees its allocation.
        """
        counts = self._ctx_fill_counts
        if self.kv_cache_manager.reserve_served_prefix_capacity(req, end):
            counts[4] += 1
            self._ctx_fill_defers.pop(req.py_request_id, None)
            return True
        counts[5] += 1
        self._ctx_fill_defers[req.py_request_id] = (
            self._ctx_fill_defers.get(req.py_request_id, 0) + 1
        )
        logger.debug(
            "Context fill: deferring request %s, its served prefix through %d does not fit",
            req.py_request_id,
            end,
        )
        return False

    def _maybe_log_ctx_fill(self) -> None:
        now = time.monotonic()
        if now - self._ctx_fill_last_log < 60.0:
            return
        window_s = now - self._ctx_fill_last_log
        self._ctx_fill_last_log = now
        counts, totals = self._ctx_fill_counts, self._ctx_fill_totals
        for i in range(len(totals)):
            totals[i] += counts[i]
        logger.info(
            f"ctx fill: window_s={window_s:.0f} async_shrunk={counts[0]} "
            f"sync_shrunk={counts[1]} budget_returned={counts[2]} v2={int(self._ctx_fill_v2)} "
            f"no_backlog={counts[3]} prefix_reserved={counts[4]} deferred={counts[5]} "
            f"defer_limit={counts[6]} backlog={self._ctx_fill_backlog}; total async_shrunk="
            f"{totals[0]} sync_shrunk={totals[1]} budget_returned={totals[2]} "
            f"no_backlog={totals[3]} prefix_reserved={totals[4]} deferred={totals[5]} "
            f"defer_limit={totals[6]}"
        )
        self._ctx_fill_counts = [0] * len(totals)

    # ---- Shortest-uncached-first aging ----

    def _advance_ctx_srpt_clock(self, active_requests: RequestList) -> None:
        """Start a scheduling pass of the aging clock.

        A request starts waiting when it is first seen in the context state and
        leaves the table once it is no longer there.
        """
        self._ctx_srpt_pass += 1
        waiting = self._ctx_srpt_waiting_since
        arrival = self._ctx_srpt_arrival_s if self._ctx_srpt_max_wait_s > 0 else None
        if arrival is not None:
            self._ctx_srpt_now = time.monotonic()
        seen = set()
        keep_state = self._ctx_srpt_keep_state_value
        for req in active_requests:
            state = req.state_value
            if state == self._context_init_state_value:
                seen.add(req.py_request_id)
                waiting.setdefault(req.py_request_id, self._ctx_srpt_pass)
                if arrival is not None:
                    arrival.setdefault(req.py_request_id, self._ctx_srpt_now)
            elif keep_state is not None and state == keep_state:
                seen.add(req.py_request_id)
        if len(waiting) > len(seen):
            for request_id in [rid for rid in waiting if rid not in seen]:
                del waiting[request_id]
        if arrival is not None and len(arrival) > len(seen):
            for request_id in [rid for rid in arrival if rid not in seen]:
                del arrival[request_id]

    def _is_ctx_srpt_aged(self, req: LlmRequest) -> bool:
        if self._ctx_srpt_max_wait_passes <= 0:
            return False
        since = self._ctx_srpt_waiting_since.get(req.py_request_id)
        if since is None:
            return False
        if self._ctx_srpt_pass - since >= self._ctx_srpt_max_wait_passes:
            return True
        if self._ctx_srpt_max_wait_s > 0:
            arrived = self._ctx_srpt_arrival_s.get(req.py_request_id)
            return arrived is not None and self._ctx_srpt_now - arrived >= self._ctx_srpt_max_wait_s
        return False

    def _started_context_ids(self, requests: RequestList) -> set[int]:
        """Ids of the context requests in *requests* that have started prefilling."""
        return {
            r.py_request_id
            for r in requests
            if r.state_value == self._context_init_state_value and not r.is_first_context_chunk
        }

    def _order_ctx_shortest_first(self, pending_ctx: RequestList) -> RequestList:
        """Order context requests by uncached prompt tokens, fewest first.

        Aged requests (see _is_ctx_srpt_aged) are not probed. By default age counts
        from arrival and the first TLLM_CTX_SRPT_AGED_SLOTS aged requests go first,
        started ones before unstarted ones, then in arrival order. The other aged
        requests follow every unaged one in the same order, so they take the budget
        the shortest-first requests leave. With TLLM_CTX_SRPT_AGE_FROM_ARRIVAL=0, or
        TLLM_CTX_SRPT_AGED_SLOTS=0, every aged request goes first, in arrival order.
        The sort is stable, so requests with equal counts keep their arrival order.
        A list with encoder requests keeps arrival order.
        """
        if any(req.state_value != self._context_init_state_value for req in pending_ctx):
            return pending_ctx
        slots = self._ctx_srpt_aged_slots
        if slots > 0:
            aged = []
            fresh = []
            for pos, req in enumerate(pending_ctx):
                if self._is_ctx_srpt_aged(req):
                    aged.append((0 if not req.is_first_context_chunk else 1, pos, req))
                else:
                    fresh.append(((self._uncached_tokens(req), pos), req))
            aged.sort(key=lambda item: item[:2])
            fresh.sort(key=lambda item: item[0])
            return (
                [req for _, _, req in aged[:slots]]
                + [req for _, req in fresh]
                + [req for _, _, req in aged[slots:]]
            )
        keyed = sorted(
            (
                (
                    (0, 0) if self._is_ctx_srpt_aged(req) else (1, self._uncached_tokens(req)),
                    pos,
                    req,
                )
                for pos, req in enumerate(pending_ctx)
            ),
            key=lambda item: item[:2],
        )
        return [req for _, _, req in keyed]

    def _split_starved_ctx(self, pending_ctx: RequestList) -> tuple[RequestList, RequestList]:
        """Split off requests pending beyond the starvation limit: ``(others, starved)``, oldest first.

        Stamps each request with the scheduler iteration it was first seen. Nothing is split off
        with a single pending request or with encoder requests in the list.
        """
        now = self._sched_iter
        starved = []
        for req in pending_ctx:
            if req.py_ctx_sched_since_iter is None:
                req.py_ctx_sched_since_iter = now
            elif now - req.py_ctx_sched_since_iter > self._ctx_starvation_iterations:
                starved.append(req)
        if (
            not starved
            or len(pending_ctx) < 2
            or any(req.state_value != self._context_init_state_value for req in pending_ctx)
        ):
            return pending_ctx, []
        starved.sort(key=lambda req: req.py_ctx_sched_since_iter)
        starved_ids = {id(req) for req in starved}
        return [req for req in pending_ctx if id(req) not in starved_ids], starved

    def _starved_ctx_shares(self, starved_ctx: RequestList, budget: BudgetTracker) -> list[int]:
        """First share of each starved request, oldest first, within the budget.

        A share is up to a quarter of max_num_tokens but at least one chunk unit, in whole chunk
        units unless it ends the prompt, so an older request is never left a share too small to
        run a chunk while a younger one runs.
        """
        if not starved_ctx:
            return []
        unit = self.chunk_unit_size
        cap = max(self._starved_ctx_tokens // unit * unit, unit)
        shares = []
        left = budget.remaining_tokens
        for req in starved_ctx:
            uncached = self._uncached_tokens(req)
            share = max(0, min(uncached, cap, left))
            if share < uncached:
                share = share // unit * unit
            shares.append(share)
            left -= share
        return shares

    # ---- Prefix-aware skip ----

    _MAX_UNPAID_PREFIX_KEYS = 4096

    def _deferral_paid(self, key, context_init_ids) -> bool:
        """Whether deferring behind the contributor of ``key`` can still be repaid by a reuse hit.

        The skip assumes the contributor commits ``key`` while it runs, so the deferred duplicate
        matches it on the next pass. When a request contributed ``key`` on the previous schedule,
        has since left the context phase, and ``key`` is still the first new block of a
        duplicate, the block was not committed by that context step (e.g. the block reaches into
        the tokens the request recomputes or reads ahead, so it is only filled during generation)
        and every further deferral is pure delay: with n samples of one prompt this admitted one
        sample per prompt per iteration. Such keys stop deferring.
        """
        if key in self._unpaid_prefix_keys:
            return False
        contributor = self._prev_first_chunk_contributors.get(key)
        if contributor is not None and contributor not in context_init_ids:
            if len(self._unpaid_prefix_keys) >= self._MAX_UNPAID_PREFIX_KEYS:
                self._unpaid_prefix_keys.pop(next(iter(self._unpaid_prefix_keys)))
            self._unpaid_prefix_keys[key] = None
            return False
        return True

    def _is_prefix_skip_candidate(self, req: LlmRequest) -> bool:
        """Whether *req* may be deferred in favour of a duplicate prefix.

        Only first-chunk context requests: a request already mid-prefill cannot
        be skipped, and an encoder request contributes to the cross pool, which
        the skip deliberately leaves alone.
        """
        return req.state_value == self._context_init_state_value and req.is_first_context_chunk

    def _skip_pays_off_under_reuse_policy(self) -> bool:
        """Whether a one-iteration deferral can actually be repaid by a reuse hit.

        The skip trades TTFT for prefill FLOPs on the premise that the contributor
        commits its first new block promptly, so the deferred duplicate matches it
        on the next pass. That premise is a property of the block-reuse policy, not
        of the scheduler.

        Under ``ALL_REUSABLE`` it holds: blocks are committed per block as prefill
        advances (``should_commit = is_all_reusable or ...``,
        ``kv_cache/kv_cache_manager_v2.py:4112``), so the deferral is bounded by
        one iteration and always pays.

        Under every other policy it does not. The case that matters today is
        hybrid/SSM: any mamba layer downgrades ``ALL_REUSABLE`` to ``PER_REQUEST``
        (``kv_cache/mamba_cache_manager.py:3140-3146``), and
        ``MambaHybridCacheManagerV2.update_context_resources``
        (``kv_cache/mamba_cache_manager.py:4217-4218``) then commits only at an SSM
        snapshot boundary or at the end of the whole context. Snapshots are off by
        default (``MambaStateConfig.periodic_snapshot_interval`` defaults to 0,
        ``llm_args.py:3989``), so in the default configuration nothing is committed
        until the contributor's entire prefill finishes -- and the duplicate is
        re-deferred for all of it, because the contributor stays in flight and
        re-registers its key every iteration.

        Worse, a commit tags exactly one block with an SSM page, the ordinal
        holding ``num_committed_tokens - 1`` (``_core/_kv_cache.py:1078``), and
        ``_prune_match`` truncates a match to the last block carrying such a page,
        or to nothing when there is none (``_block_radix_tree.py:755-758``). A
        duplicate that shares a prefix but diverges before the contributor's end
        therefore waits out that whole prefill and then still reuses nothing. That
        is a TTFT regression with no compensating saving, so gate rather than
        defer.

        Keying this on the policy rather than on "is there an SSM life cycle"
        makes the gate self-lifting: configuring periodic snapshots does not by
        itself restore per-block commits, and any future policy that does commit
        eagerly can opt in here explicitly.

        Called once, from ``__init__``, to fold into ``_prefix_skip_enabled``.
        """
        from ..kv_cache.kv_cache_manager_v2 import BlockReusePolicy

        return self.kv_cache_manager.block_reuse_policy == BlockReusePolicy.ALL_REUSABLE

    def _collect_contributed_blocks(
        self, requests_list: RequestList, pending_ctx: RequestList, inflight_request_ids: set[int]
    ) -> Optional[set]:
        """Blocks already promised by context requests executing right now.

        A request that is mid-prefill cannot be skipped, so whatever it is about
        to commit is registered up front and later duplicates defer to it. This
        seeds the set that the phase-2 loop then extends as it schedules.

        Only chunk continuations that are actually in flight qualify. One that
        is merely pending registers through the normal post-``SCHEDULED`` path
        instead: it precedes its duplicates in arrival order, so coverage is the
        same in practice, and nothing is ever deferred behind a contributor that
        might not run. Deferring behind a request that does not run could
        produce an iteration scheduling nothing, which the deadlock detector
        would report as a hang.

        Returns None when the skip provably cannot fire this iteration, which
        lets the caller avoid every probe: nothing else consumes the radix walk,
        so a probe that cannot change a decision is pure overhead.
        """
        if not self._prefix_skip_enabled:
            return None
        num_candidates = sum(1 for req in pending_ctx if self._is_prefix_skip_candidate(req))
        if num_candidates == 0:
            return None
        in_flight = [
            req
            for req in requests_list
            if req.state_value == self._context_init_state_value
            and not req.is_first_context_chunk
            and req.request_id in inflight_request_ids
        ]
        if not in_flight and num_candidates < 2:
            # A lone candidate with nothing running can never be deferred: only
            # requests examined before it can register, and there are none.
            has_pending_continuation = any(
                req.state_value == self._context_init_state_value and not req.is_first_context_chunk
                for req in pending_ctx
            )
            if not has_pending_continuation:
                return None
        contributed = set()
        for req in in_flight:
            key = self.kv_cache_manager.probe_first_new_block_key(req)
            if key is not None:
                contributed.add(key)
        return contributed

    # ---- Per-type scheduling methods ----

    def _try_schedule_encoder(
        self, req: LlmRequest, budget: BudgetTracker
    ) -> tuple[ScheduleAction, int]:
        """Try to schedule an encoder request.

        Encoder admission does not need KV blocks for decoder work. Decoder
        context is the first step that reserves self- and cross-KV blocks and
        writes K/V projections into the cross cache.

        Returns ``(action, tokens)`` where *tokens* is meaningful only
        when *action* is ``SCHEDULED``.
        """
        # Encoder-decoder runtime requires a cross pool for the later decoder
        # context step. If the runtime did not plumb one through, surface this
        # loudly rather than silently routing to the self pool, which would
        # corrupt the dual-pool contract.
        if self.cross_kv_cache_manager is None:
            raise RuntimeError(
                f"Encoder-init request {req.py_request_id} requires a cross_kv_cache_manager."
            )

        req_tokens = req.encoder_output_len
        if not budget.can_fit_tokens(req_tokens):
            return ScheduleAction.STOP, 0
        assert self.max_context_length is None or req_tokens <= self.max_context_length, (
            f"The number of encoder tokens ({req_tokens}) exceeds the limit value ({self.max_context_length})"
        )
        return ScheduleAction.SCHEDULED, req_tokens

    def _try_schedule_disagg_gen_init(
        self, req: LlmRequest, budget: BudgetTracker
    ) -> tuple[ScheduleAction, int]:
        """Try to schedule a disagg generation init request.

        Disagg gen init requests bypass normal state gating but still need
        KV cache allocation inline. Aligned with C++ CapacityScheduler which
        treats disagg_gen_init identically to context_init for block/PEFT/
        maxNumRequests accounting.

        Returns ``(action, tokens)``.  *tokens* is 0 because disagg requests
        don't participate in the forward pass token budget.
        """
        # Only the target manager receives prompt KV. Drafters that need it
        # under disagg keep their layers in that manager (MTP/Eagle fold them
        # in; DFlash uses explicit draft layer specs). A separate draft manager
        # gets no transfer and no paired-reuse path here.
        try:
            prepared = self.kv_cache_manager.prepare_disagg_gen_init(req)
        except RemoteMemError as error:
            # Its prefix match resumed pages a failed remote cache level took with it: drop the match, and the next
            # pass matches again without them.
            self._lost_kv_error = error
            return self._on_lost_kv_cache(req, set(), []), 0
        if not prepared:
            if disagg_diagnostics.DISAGG_TRANSFER_DIAGNOSTICS_ENABLED:
                disagg_diagnostics.emit_kv_admission_result(
                    req, self.kv_cache_manager, admitted=False
                )
            logger.debug("prepare_disagg_gen_init failed for request %s", req.py_request_id)
            return ScheduleAction.SKIP, 0
        return ScheduleAction.SCHEDULED, 0

    def _try_schedule_context(
        self,
        req: LlmRequest,
        budget: BudgetTracker,
        preempt_for_pages: Callable[[LlmRequest], bool],
        token_cap: Optional[int] = None,
    ) -> tuple[ScheduleAction, int, bool]:
        """Try to schedule a context request (chunked or non-chunked).

        Returns ``(action, tokens, chunking_flag)``.  *tokens* and
        *chunking_flag* are meaningful only when *action* is ``SCHEDULED``.
        *token_cap* limits the chunk below the budget left (chunked only).

        Source reservations advance the prefix before budget checks. Destination
        pages are allocated here; loads are accepted after the final batch trims.
        A loading request remains outside the schedulable state range.

        On a disaggregated server without an eviction tier whose connector
        declares `requires_completion_reservation`, or whose
        `kv_cache_config.enable_disagg_completion_reservation` is set, a
        connector first chunk can be deferred for completion headroom; the
        rollback below also drops its bound.
        """
        first_chunk = req.is_first_context_chunk
        gate = False
        if self.chunking_enabled:
            if (
                self._ctx_shortest_first
                and first_chunk
                and not self._holds_context_cache(req)
                and not self._first_chunk_fits_budget(req, budget, token_cap)
            ):
                # Nothing is claimed yet, so there is nothing to release.
                return ScheduleAction.SKIP, 0, False
            gate = self._oop_gate_track
            if gate:
                # Replay a failed first chunk's outcome while nothing it needs changed.
                gated = self._oop_gate_check(req, budget, preempt_for_pages, token_cap)
                if gated is not None:
                    return gated, 0, False
                self._oop_gate_fail_kind = None
            result = self._try_schedule_context_chunked(req, budget, preempt_for_pages, token_cap)
        else:
            result = self._try_schedule_context_full(req, budget, preempt_for_pages)

        if first_chunk and result[0] is not ScheduleAction.SCHEDULED:
            # Failed admission must not retain prefix-reuse holds. Suspension
            # alone cannot release them when the last cache tier is full.
            self.release_context_admission(req)

        if gate:
            # After the release: the recorded page counts include what it gave back.
            self._oop_gate_record(req, first_chunk, result[0])
        return result

    def _holds_context_cache(self, req: LlmRequest) -> bool:
        """Whether *req* holds a cache in any pool, which a failed first admission must release."""
        return any(
            manager is not None and req.py_request_id in manager.kv_cache_map
            for manager in (
                self.kv_cache_manager,
                self.draft_kv_cache_manager,
                self.cross_kv_cache_manager,
            )
        )

    def _first_chunk_fits_budget(
        self, req: LlmRequest, budget: BudgetTracker, token_cap: Optional[int]
    ) -> bool:
        """Whether a first chunk can come out non-empty, decided before preparation claims its prefix.

        No chunk comes out when a starved request's token_cap leaves no budget, or under FORCE_CHUNK
        when a leftover below chunk_unit_size reaches neither the next snapshot point nor the prompt
        end; the claim's end is estimated with the reuse probe (_uncached_tokens).
        """
        remaining = budget.remaining_tokens
        if remaining is None:
            return True
        if token_cap is not None:
            remaining = min(remaining, token_cap)
        if remaining <= 0:
            return False
        if (
            self.chunking_policy != ContextChunkingPolicy.FORCE_CHUNK
            or remaining >= self.chunk_unit_size
        ):
            return True
        pos = (
            req.context_current_position + req.context_remaining_length - self._uncached_tokens(req)
        )
        next_point = min(
            (point for point in req.expect_snapshot_points if point > pos),
            default=req.prompt_len,
        )
        return min(next_point, req.prompt_len) - pos <= remaining

    def release_context_admission(self, req: LlmRequest) -> None:
        """Free a first-chunk context's caches in every pool and rewind its cursor.

        Its next admission then starts from a fresh prefix match.
        """
        for manager in (
            self.kv_cache_manager,
            self.draft_kv_cache_manager,
            self.cross_kv_cache_manager,
        ):
            if manager is not None and req.py_request_id in manager.kv_cache_map:
                manager.free_resources(req)
        rewind_context_after_cache_drop(req, self.tokens_per_block)

    def _try_schedule_context_full(
        self,
        req: LlmRequest,
        budget: BudgetTracker,
        preempt_for_pages: Callable[[LlmRequest], bool],
    ) -> tuple[ScheduleAction, int, bool]:
        """Try to schedule a non-chunked context request.

        Returns ``(action, tokens, chunking_flag)``.
        """
        pre_prepare_context_tokens = req.context_remaining_length
        draft_len = get_draft_token_length(req)
        if not self.enable_prefix_aware_scheduling:
            req_tokens = pre_prepare_context_tokens + draft_len
            if not budget.can_fit_tokens(req_tokens):
                return ScheduleAction.STOP, 0, False
            assert (
                self.max_context_length is None
                or pre_prepare_context_tokens <= self.max_context_length
            ), (
                f"Context tokens ({pre_prepare_context_tokens}) exceeds limit ({self.max_context_length})"
            )

        # Prepare first so block reuse updates context_remaining_length
        # before budget check.
        if not self._prepare_context_pair(req):
            logger.debug(f"prepare_context failed for context request {req.py_request_id}")
            self._preempt_for_resume(req, preempt_for_pages)
            return ScheduleAction.STOP, 0, False

        context_tokens = req.context_remaining_length
        recovery_rows = self.kv_cache_manager.context_replay_tokens(req)
        replay_tokens = recovery_rows or 0
        if self.enable_prefix_aware_scheduling or recovery_rows is not None:
            req_tokens = context_tokens + replay_tokens + draft_len

            if not budget.can_fit_tokens(req_tokens):
                return ScheduleAction.STOP, 0, False

            assert (
                self.max_context_length is None
                or context_tokens + replay_tokens <= self.max_context_length
            ), f"Context tokens ({context_tokens}) exceeds limit ({self.max_context_length})"

        # V2 resizes KV cache directly in the scheduler (no separate
        # prepareResources for main cache), so include draft tokens.
        req.context_chunk_size = context_tokens
        if not self.kv_cache_manager.reserve_context_completion(req):
            # Admitted requests own the completion headroom. Defer this prompt
            # instead of preempting one of them to make room for it.
            logger.debug(
                "Deferring context request %s: admitted requests hold the completion headroom",
                req.py_request_id,
            )
            return ScheduleAction.SKIP, 0, False
        if not self._try_allocate_context(req, context_tokens + draft_len):
            # Out of pages. Give up one started request so this one can
            # proceed, and retry next iteration: a failed resize leaves a
            # first chunk suspended, so the retry has to go back through
            # prepare_context to resume it.
            if preempt_for_pages(req):
                return ScheduleAction.STOP, 0, False
            return ScheduleAction.SKIP, 0, False

        cross_action = self._try_schedule_cross_context(req)
        if cross_action is not ScheduleAction.SCHEDULED:
            self._suspend_request(req)
            return cross_action, 0, False

        return ScheduleAction.SCHEDULED, req_tokens, False

    def _has_context_chunk_budget(self, budget: BudgetTracker) -> bool:
        remaining = budget.remaining_tokens
        return remaining is None or (
            remaining > 0
            and (
                self.chunking_policy == ContextChunkingPolicy.FORCE_CHUNK
                or remaining >= self.chunk_unit_size
            )
        )

    def _first_chunk_waits_for_budget(self, req: LlmRequest, budget: BudgetTracker) -> bool:
        """Whether a first chunk should wait for the next iteration's budget.

        FORCE_CHUNK admits a leftover below chunk_unit_size, because a started
        context can end at a snapshot point or at its end inside it. A first
        chunk learns its reuse only in prepare_context, which matches its prefix
        and resumes its pages, and a failed first chunk releases both again.
        Once another context chunk runs this iteration, the first chunk waits
        for a fresh budget unless its whole remaining prompt fits. Under
        shortest-uncached-first the uncached remainder from the reuse probe
        (_uncached_tokens) must fit instead, and a first chunk that already
        holds a cache does not wait here: only preparation releases that claim.
        """
        remaining = budget.remaining_tokens
        if not (
            self.chunking_policy == ContextChunkingPolicy.FORCE_CHUNK
            and req.state_value == self._context_init_state_value
            and req.is_first_context_chunk
            and remaining is not None
            and remaining < self.chunk_unit_size
        ):
            return False
        if not self._ctx_shortest_first:
            return req.context_remaining_length > remaining
        return self._uncached_tokens(req) > remaining and not self._holds_context_cache(req)

    def _preempt_for_resume(
        self, req: LlmRequest, preempt_for_pages: Callable[[LlmRequest], bool]
    ) -> bool:
        """Free GPU pages for a suspended mid-prefill request that could not resume.

        Under helix, bringing a suspended request's pages back can fail once
        newer requests have taken the GPU pages, for example after its pages
        moved to a tier below GPU. Like a failed resize, give up a newer
        started request so the older one is not left waiting behind it.
        """
        return self.has_cp_helix and not req.is_first_context_chunk and preempt_for_pages(req)

    def _try_schedule_context_chunked(
        self,
        req: LlmRequest,
        budget: BudgetTracker,
        preempt_for_pages: Callable[[LlmRequest], bool],
        token_cap: Optional[int] = None,
    ) -> tuple[ScheduleAction, int, bool]:
        """FCFS interleaved chunking for a single context request.

        Returns ``(action, chunk_tokens, chunking_flag)``.

        Chunking uses implicit skip on failure (not break): a ctx that can't
        be chunked small enough is skipped, allowing subsequent gen requests
        (needing only ~1 token) to still be scheduled.
        """
        remaining_budget = budget.remaining_tokens
        pre_prepare_context_remaining = req.context_remaining_length
        force_chunk = self.chunking_policy == ContextChunkingPolicy.FORCE_CHUNK
        inforward_point = None
        ctx_fill_reserve_end: Optional[int] = None
        if self._inforward_snapshot_rows:
            # A mark from an earlier pass whose batch did not run this request
            # must not outlive that pass.
            req.py_mamba_inforward_snapshot_point = None

        if not self._has_context_chunk_budget(budget):
            return ScheduleAction.SKIP, 0, False

        # Prepare context (create _KVCache, block reuse, resume — no resize)
        if not self._prepare_context_pair(req):
            logger.debug(f"prepare_context failed for chunked context request {req.py_request_id}")
            if self._preempt_for_resume(req, preempt_for_pages):
                return ScheduleAction.STOP, 0, False
            # A first chunk never preempts here (_preempt_for_resume needs a started
            # request), so the retry gate replays this failure as a plain SKIP.
            self._oop_gate_fail_kind = "prepare"
            self._oop_gate_fail_chunk = 0
            return ScheduleAction.SKIP, 0, False

        if req.py_request_id in self._ctx_unverified_short:
            # Ordered as short on the router's estimate alone; the reuse claimed
            # just now makes the remainder exact. Once per request per iteration.
            # A first chunk that is not scheduled releases its admission, so the
            # retry matches its prefix again. A starved request (token_cap set)
            # runs on the share it holds instead: deferring it would not move it,
            # and the next iteration would order it as short and defer it again.
            self._ctx_unverified_short.discard(req.py_request_id)
            if token_cap is None and self._is_long_context(req):
                if (
                    self._ctx_demote_max_iters > 0
                    and self._ctx_demote_skips.get(req.py_request_id, 0)
                    >= self._ctx_demote_max_iters
                ):
                    # Demoted as long as the bound allows: let through for good,
                    # as at the door (see _defer_long_context).
                    self._ctx_demote_released.add(req.py_request_id)
                else:
                    # The exact remainder replaces the estimate, so the next
                    # iteration demotes the request at the door, where the
                    # aging bound applies, without preparing it first.
                    req.py_uncached_prefill_tokens = req.context_remaining_length
                    req.py_uncached_prefill_probed = True
                    return ScheduleAction.DEFER, 0, False

        # Calculate chunk size from remaining budget
        #    (context_remaining_length is now correct after block reuse)
        context_remaining = req.context_remaining_length
        # Recovery is fixed by prepare_context's normal resume, before chunk sizing.
        recovery_rows = self.kv_cache_manager.context_replay_tokens(req)
        has_recovery = recovery_rows is not None
        replay_tokens = recovery_rows or 0
        remaining_budget = budget.remaining_tokens
        if token_cap is not None:
            # The chunk is sized as if only token_cap tokens were left.
            remaining_budget = (
                token_cap if remaining_budget is None else min(remaining_budget, token_cap)
            )
        if replay_tokens and self.max_context_length is not None:
            suffix_capacity = self.max_context_length - replay_tokens
            minimum_suffix = min(context_remaining, self.chunk_unit_size)
            if force_chunk:
                minimum_suffix = min(minimum_suffix, _get_forced_context_chunk_size(req))
            if suffix_capacity < minimum_suffix:
                raise ValueError(
                    "Encoder replay cannot make progress: max_num_tokens must fit "
                    f"{replay_tokens} replay tokens plus at least "
                    f"{minimum_suffix} new tokens for this suffix"
                )
        if remaining_budget is not None:
            remaining_budget -= replay_tokens
            if remaining_budget <= 0:
                return ScheduleAction.SKIP, 0, False
        if force_chunk:
            # Snapshot boundaries can be shorter than chunk_unit_size.
            assert isinstance(req.expect_snapshot_points, list)
            # With no remaining state to snapshot, avoid artificial boundaries.
            chunk_size = _get_forced_context_chunk_size(req)
            budget_context_remaining = context_remaining
            if self._inforward_snapshot_rows_left > 0 and not has_recovery:
                inforward_point = self._inforward_snapshot_point(req, chunk_size, remaining_budget)
                if inforward_point is not None:
                    chunk_size = context_remaining
        else:
            budget_context_remaining = (
                context_remaining
                if self.enable_prefix_aware_scheduling or has_recovery
                else pre_prepare_context_remaining
            )
            chunk_size = (
                min(remaining_budget, budget_context_remaining)
                if remaining_budget is not None
                else budget_context_remaining
            )
            if self._ctx_chunk_cap_this_pass is not None and self._ctx_chunk_cap_applies(req):
                chunk_size = min(chunk_size, self._ctx_chunk_cap_this_pass)
            if self._ctx_fill and req.is_first_context_chunk and not has_recovery:
                self._ctx_fill_reserve_end = None
                chunk_size = self._ctx_fill_chunk(req, chunk_size)
                ctx_fill_reserve_end, self._ctx_fill_reserve_end = self._ctx_fill_reserve_end, None

        if self.max_context_length is not None:
            chunk_size = min(chunk_size, self.max_context_length - replay_tokens)

        chunk_size = min(
            chunk_size, remaining_budget if remaining_budget is not None else chunk_size
        )
        oop_gate_budget_chunk = chunk_size

        # Round down to chunk_unit_size boundary only when not hitting the end
        # or a checkpoint.
        if chunk_size < budget_context_remaining and not (
            force_chunk and _is_forced_context_chunk_boundary(req, chunk_size)
        ):
            chunk_size = (chunk_size // self.chunk_unit_size) * self.chunk_unit_size

        # Run exactly the pre-grown chunk, or wait for a pass with enough budget;
        # a longer chunk would grow the capacity again before the history catches up.
        pregrown = self._pregrown_context_tokens(req)
        if pregrown > 0:
            if chunk_size < pregrown:
                return ScheduleAction.SKIP, 0, False
            chunk_size = pregrown
            # The replayed chunk is not the shrunk one; it reserves nothing extra.
            ctx_fill_reserve_end = None

        if chunk_size <= 0:
            # Out of token budget rather than out of pages, so releasing pages
            # would not help; the next iteration gets a fresh budget. Not
            # suspended either, to avoid pathological suspend/resume cycles.
            return ScheduleAction.SKIP, 0, False

        # A pre-grown chunk was MM-aligned from this same position when it was
        # first sized, and either snap direction would break the exact replay.
        if pregrown == 0:
            chunk_size = self._align_chunk_to_mm_block(
                req, chunk_size, remaining_budget, context_remaining
            )
        # Alignment returns 0 when neither snap-up nor snap-down can preserve
        # bidirectional MM attention this iteration — defer the request rather
        # than schedule a chunk that would corrupt attention. Next iteration
        # gets a fresh budget; under steady-state load this resolves quickly.
        if chunk_size <= 0:
            return ScheduleAction.SKIP, 0, False
        # A pre-grown chunk replays exactly (see above), so it is never cut back.
        if self._ctx_decoder_hold_window and not force_chunk and pregrown == 0:
            chunk_size = self._ctx_decoder_hold_chunk(req, chunk_size)
            if chunk_size <= 0:
                # Held whole for one pass: it keeps its prepared KV and its place in the queue,
                # as on an exhausted budget.
                return ScheduleAction.SKIP, 0, False
        budget_chunk_size = chunk_size
        # Decoder queries are a subset of these physical Encoder rows, including
        # recovery rows, so they fit the same iteration's input-token budget.
        req.context_chunk_size = min(chunk_size, context_remaining)

        # Draft tokens only matter for last chunk (budget + resize)
        chunk_tokens = budget_chunk_size + replay_tokens
        resize_tokens = req.context_chunk_size
        if req.is_last_context_chunk:
            draft_len = get_draft_token_length(req)
            if draft_len > 0:
                chunk_tokens += draft_len
                resize_tokens += draft_len

        if not self.kv_cache_manager.reserve_context_completion(req):
            # As in _try_schedule_context_full: wait for headroom, do not preempt.
            logger.debug(
                "Deferring context request %s: admitted requests hold the completion headroom",
                req.py_request_id,
            )
            return ScheduleAction.SKIP, 0, False
        # V2 resizes KV cache directly in the scheduler, so include
        # draft tokens for last chunk.
        if not self._try_allocate_context(req, resize_tokens):
            # Out of pages, as in _try_schedule_context_full.
            if preempt_for_pages(req):
                return ScheduleAction.STOP, 0, False
            self._oop_gate_fail_kind = "alloc"
            self._oop_gate_fail_chunk = oop_gate_budget_chunk
            self._oop_gate_fail_replay = replay_tokens
            return ScheduleAction.SKIP, 0, False

        if ctx_fill_reserve_end is not None and not self._ctx_fill_reserve_served_prefix(
            req, min(ctx_fill_reserve_end + req.context_chunk_size, req.prompt_len)
        ):
            # Deferred one pass; the first-chunk rollback in _try_schedule_context frees it.
            return ScheduleAction.SKIP, 0, False

        cross_action = self._try_schedule_cross_context(req)
        if cross_action is not ScheduleAction.SCHEDULED:
            self._suspend_request(req)
            return cross_action, 0, False

        chunking_flag = req.context_chunk_size < req.context_remaining_length

        if inforward_point is not None and req.context_chunk_size == context_remaining:
            req.py_mamba_inforward_snapshot_point = inforward_point
            self._inforward_snapshot_rows_left -= 1

        return ScheduleAction.SCHEDULED, chunk_tokens, chunking_flag

    def _inforward_snapshot_point(
        self, req: LlmRequest, chunk_size: int, remaining_budget: Optional[int]
    ) -> Optional[int]:
        """The snapshot point P a forced chunk ends at, if the chunk can run through the prompt end instead.

        The GDN layers then record the state at P inside the forward
        (the GDN cache manager's in-forward snapshots), which saves the extra
        context iteration of the short tail [P, prompt end).
        """
        manager = self.kv_cache_manager
        pos = req.context_current_position
        point = pos + chunk_size
        end = req.prompt_len
        points = req.expect_snapshot_points
        if not points or point != points[-1] or not point < end:
            return None
        if pos + req.context_remaining_length != end:
            return None
        if (
            end - point > manager.inforward_snapshot_max_tail
            or point - pos < manager.inforward_snapshot_min_main
        ):
            return None
        if remaining_budget is not None and end - pos > remaining_budget:
            return None
        if self.max_context_length is not None and end - pos > self.max_context_length:
            return None
        mm_data = getattr(req, "py_multimodal_data", None)
        if isinstance(mm_data, dict) and mm_data.get("mm_bidirectional_blocks", False):
            return None
        return point

    def _reuse_claim_limit(self, req: LlmRequest) -> int | None:
        """Pair the two pools on one depth before either claims.

        Each probe matches through the D-token lookahead and applies its backoff
        in the same tree walk. Their minimum is therefore the common usable depth:
        ``min(max(m_t-D, 0), max(m_d-D, 0)) == max(min(m_t, m_d)-D, 0)``.

        Doing this up front rather than claim-then-reconcile keeps the mismatch
        path (free the draft pages and re-claim shallower) off the common case.

        The target's spec recompute tail (``context_reuse_claim_limit``) caps the
        common depth too: the unpaired path applies it in ``prepare_context``,
        which a paired first chunk does not go through.
        """
        draft_manager = self._joint_draft_manager
        if draft_manager is None:
            return None
        tail_limit = self.kv_cache_manager.context_reuse_claim_limit(req)
        draft_match = draft_manager.probe_context_reuse(req)
        target_match = self.kv_cache_manager.probe_context_reuse(req)
        if draft_match is None or target_match is None:
            return tail_limit
        # Claiming the minimum stops the target's lookup there, hiding where the
        # prompt leaves the tree; the attention-only draft match still reaches it.
        self.kv_cache_manager.note_context_fork(req, draft_match)
        common = min(draft_match, target_match)
        return common if tail_limit is None else min(common, tail_limit)

    def _admit_unpaired_draft(self, req: LlmRequest) -> bool:
        """Create and resume an unpaired draft mirror as part of admitting *req*.

        ``_suspend_request`` suspends both pools whenever a draft manager
        exists, but only the joint pairing admits the draft pool through its
        own calls. An unpaired mirror is therefore suspended by eviction -- or
        simply born suspended, which a fresh ``_KVCache`` always is -- and
        nothing here brings it up, so it reaches the draft manager's
        ``prepare_resources`` still suspended. That method cannot defer: the
        target has already been prepared and the request is already in the
        scheduled batch, so it raises and takes the executor loop with it.
        Admitting here turns a refusal into an ordinary allocation failure the
        caller can roll back and retry.

        Returns True when there is nothing to do: no draft pool at all, or a
        joint one that admits itself.
        """
        draft_manager = self.draft_kv_cache_manager
        if draft_manager is None or self._joint_draft_manager is not None:
            return True
        return draft_manager.admit_mirror(req)

    def _prepare_context_pair(self, req: LlmRequest) -> bool:
        """Prepare target/draft caches with one verified logical reuse depth."""
        from ..kv_cache.kv_cache_manager_v2 import _settle_context_cursor

        draft_manager = self._joint_draft_manager
        if draft_manager is None:
            if not self.kv_cache_manager.prepare_context(req):
                return False
            if self._admit_unpaired_draft(req):
                return True
            # Mirror the joint branch below: a non-first chunk reaches a
            # refused admission with an ACTIVE target cache (prepare_context
            # just resumed it), and leaving it active pins its pages while the
            # request waits on the draft pool -- under pressure, the very
            # requests whose completion would drain that pool can then no
            # longer grow their own target KV. First chunks are rolled back
            # (freed) by _try_schedule_context instead.
            if not req.is_first_context_chunk:
                self._suspend_request(req)
            return False

        if not req.is_first_context_chunk:
            if self.kv_cache_manager.prepare_context(req) and draft_manager.prepare_context(req):
                return True
            self._suspend_request(req)
            return False

        # Both pools claim the depth paired above, so they normally agree first
        # time; the target claim still caps to the draft's in case an eviction
        # landed between the probe and the claim.
        draft_reuse = draft_manager.prepare_context_cache(req, self._reuse_claim_limit(req))
        if draft_reuse is None:
            self._free_kv_caches(req)
            return False
        common_reuse = self.kv_cache_manager.prepare_context_cache(req, draft_reuse)
        if common_reuse is None:
            self._free_kv_caches(req)
            return False

        # A served prefix is held in both pools even when their reuse policies
        # commit different depths. Compare the effective prefixes before freeing it.
        served_prefix = req.py_served_prefix_end
        if max(draft_reuse, served_prefix) != max(common_reuse, served_prefix):
            # The shorter re-claim can itself fall short, so drop straight to
            # no reuse rather than iterate against an eviction we do not own.
            draft_manager.free_resources(req)
            draft_reuse = draft_manager.prepare_context_cache(req, common_reuse)
            if draft_reuse != common_reuse:
                self._free_kv_caches(req)
                common_reuse = self.kv_cache_manager.prepare_context_cache(req, 0)
                draft_reuse = draft_manager.prepare_context_cache(req, 0)
                if common_reuse is None or draft_reuse is None:
                    self._free_kv_caches(req)
                    return False
                # An empty key stream cannot match; anything else means the
                # two page tables describe different prefixes.
                if common_reuse != 0 or draft_reuse != 0:
                    raise RuntimeError(
                        f"Could not establish a common no-reuse fallback for request "
                        f"{req.py_request_id}: target={common_reuse}, draft={draft_reuse}"
                    )

        # No enable_block_reuse guard: reaching here means a paired draft pool
        # exists, and KvCacheCreator only pairs when block reuse is on.
        _settle_context_cursor(req, common_reuse, self.kv_cache_manager.tokens_per_block)
        return True

    def _try_allocate_context(self, req: LlmRequest, num_tokens: int) -> bool:
        """Admit one context chunk in both target and draft KV pools."""
        if not self.kv_cache_manager.resize_context(req, num_tokens):
            if self.enable_joint_kv_cache_reuse:
                self._suspend_request(req)
            return False

        draft_manager = self._joint_draft_manager
        if draft_manager is None or draft_manager.try_allocate_draft_context(req, num_tokens):
            return True

        if not self.kv_cache_manager.revert_allocate_context(req):
            # The target dropped its cache and rewound the shared cursor, so
            # drop the draft pool too or it still describes abandoned progress.
            draft_manager.free_resources(req)
        self._suspend_request(req)
        return False

    def _align_chunk_to_mm_block(
        self,
        req: LlmRequest,
        chunk_size: int,
        remaining_budget: Optional[int],
        context_remaining: int,
    ) -> int:
        """Adjust *chunk_size* so a boundary never lands inside a multimodal
        soft-token run that requires intact bidirectional attention.

        Returns the adjusted chunk size, or ``0`` to signal that the caller
        should SKIP this request this iteration (defer to next iteration with
        a fresh budget).

        Two-pronged strategy:
          1. snap-UP: extend the chunk to swallow the rest of the block
             (preferred — block lands wholly in this iteration).
          2. snap-DOWN fallback: if budget / max_context_length / context_remaining
             can't absorb snap-up, shrink the chunk to end *before* the block
             starts; the block lands wholly in the next iteration.

        Last-resort defer (return 0): if the block starts at the chunk's
        left boundary (snap-down would zero the chunk anyway), defer the
        request. We never let a bidirectional MM block straddle a chunk
        boundary — it would silently break attention correctness.

        Raises ``ValueError`` if the block itself exceeds max_context_length —
        no chunk size can fit it, deferring would livelock; surface the
        config error loudly rather than spin silently.

        Gated on ``mm_bidirectional_blocks`` written by the input processor —
        causal-MM models (Llava, Qwen-VL) skip this entirely.
        """
        # `getattr(..., None) or {}` is not enough: on a Mock req the attribute
        # exists as a Mock (truthy) and `.get` returns another Mock (also
        # truthy). Require an actual dict — that's also what the input
        # processor writes in inputs/registry.py.
        mm_data = getattr(req, "py_multimodal_data", None)
        if not isinstance(mm_data, dict):
            return chunk_size
        if not mm_data.get("mm_bidirectional_blocks", False):
            return chunk_size
        cumsum = mm_data.get("multimodal_embed_mask_cumsum")
        if cumsum is None:
            return chunk_size

        unit_size = self.chunk_unit_size
        lo = req.context_current_position
        end_abs = lo + chunk_size
        prompt_len = lo + context_remaining
        if end_abs >= prompt_len or end_abs <= 0:
            return chunk_size

        # Cumsum is INCLUSIVE [P] (length prompt_len) with
        # cumsum[i] = sum(mask[0..i]). So mask[i] = cumsum[i]-cumsum[i-1] for
        # i>=1, and mask[0] = cumsum[0]. Aligned with inputs/registry.py
        # (``embed_mask.cumsum(0)``) and MultimodalRuntimeData.
        #
        # Three-cumsum boundary-in-block detection: positions end_abs-1 and
        # end_abs are both MM iff mask[end_abs-1]==1 AND mask[end_abs]==1.
        # Gemma4's HF processor wraps every soft-token run with non-MM
        # specials (boi/eoi/boa/eoa, embed_mask=0), so two consecutive
        # embed_mask=1 positions always belong to the same block.
        cs_prev = int(cumsum[end_abs - 1].item())
        cs_cur = int(cumsum[end_abs].item())
        if (cs_cur - cs_prev) != 1:
            return chunk_size  # mask[end_abs] != 1
        if end_abs == 1:
            if cs_prev != 1:
                return chunk_size  # mask[0] != 1
        else:
            if (cs_prev - int(cumsum[end_abs - 2].item())) != 1:
                return chunk_size  # mask[end_abs - 1] != 1

        # --- compute block extent (forward + backward walks from end_abs) ---
        # Forward walk: advance while position block_end_abs is MM
        # (mask[block_end_abs] == cumsum[block_end_abs]-cumsum[block_end_abs-1]).
        # Final block_end_abs is the first non-MM position (exclusive end),
        # or prompt_len if the block extends through EOS.
        # Backward walk uses ``> 0`` (not ``> lo``) so block_size below
        # reflects the TRUE block size — important for the impossibility
        # check, which must not be undercounted if a prior iteration somehow
        # advanced into a block (would otherwise mask a real config error).
        block_end_abs = end_abs
        while block_end_abs < prompt_len:
            if (int(cumsum[block_end_abs].item()) - int(cumsum[block_end_abs - 1].item())) != 1:
                break
            block_end_abs += 1

        # Backward walk: decrement while position (block_start_abs - 1) is MM.
        # Final block_start_abs is the first MM position (inclusive start),
        # or 0 if the block extends to BOS.
        block_start_abs = end_abs
        while block_start_abs > 1:
            if (
                int(cumsum[block_start_abs - 1].item()) - int(cumsum[block_start_abs - 2].item())
            ) != 1:
                break
            block_start_abs -= 1
        # Position 0: mask[0] == cumsum[0]; no cumsum[-1] sentinel.
        if block_start_abs == 1 and int(cumsum[0].item()) == 1:
            block_start_abs = 0

        # --- impossibility check ---
        # If the block itself exceeds max_context_length, no chunk size can
        # ever fit it. Deferring would livelock the request. Raise a clear
        # config error rather than letting the scheduler thrash silently.
        block_size = block_end_abs - block_start_abs
        if self.max_context_length is not None and block_size > self.max_context_length:
            raise ValueError(
                f"req {req.py_request_id}: bidirectional multimodal block of "
                f"{block_size} tokens exceeds max_context_length="
                f"{self.max_context_length}. The block must fit in a single "
                f"chunk to preserve bidirectional attention; deferring would "
                f"livelock. Increase max_num_tokens to at least {block_size}."
            )

        # --- snap-up target ---
        # Round up to unit_size so setPrepopulatedPromptLen's downstream
        # floor() doesn't clip the extension back into the block.
        if unit_size > 0 and block_end_abs < prompt_len:
            up_block_end = ((block_end_abs + unit_size - 1) // unit_size) * unit_size
        else:
            up_block_end = block_end_abs
        up_block_end = min(up_block_end, prompt_len)
        up_chunk_size = up_block_end - lo

        up_fits_budget = remaining_budget is None or up_chunk_size <= remaining_budget
        up_fits_max_ctx = (
            self.max_context_length is None or up_chunk_size <= self.max_context_length
        )
        up_fits_remaining = up_chunk_size <= context_remaining

        if up_fits_budget and up_fits_max_ctx and up_fits_remaining and up_chunk_size > chunk_size:
            return up_chunk_size

        # --- snap-down fallback ---
        # Round down to unit_size so the next chunk starts on a kv-block boundary.
        if unit_size > 0:
            down_block_start = (block_start_abs // unit_size) * unit_size
        else:
            down_block_start = block_start_abs

        if down_block_start <= lo:
            logger.warning(
                "req %s: MM block at chunk left edge "
                "(lo=%s, block=[%s, %s)); snap-up does not fit "
                "(up_size=%s, budget=%s, max_ctx=%s, ctx_rem=%s) and "
                "snap-down would zero the chunk. Deferring request to next "
                "iteration to preserve bidirectional MM attention.",
                req.py_request_id,
                lo,
                block_start_abs,
                block_end_abs,
                up_chunk_size,
                remaining_budget,
                self.max_context_length,
                context_remaining,
            )
            return 0

        return down_block_start - lo

    @staticmethod
    def _get_optional_encoder_output_len(req: LlmRequest) -> Optional[int]:
        get_encoder_output_len = getattr(type(req), "try_get_encoder_output_len", None)
        encoder_output_len = (
            get_encoder_output_len(req)
            if get_encoder_output_len is not None
            else getattr(req, "encoder_output_len", None)
        )
        if encoder_output_len is None:
            return None
        return int(encoder_output_len)

    @classmethod
    def _needs_cross_context_allocation(cls, req: LlmRequest) -> bool:
        """Return whether decoder context must reserve cross-KV for *req*."""
        if cls._get_optional_encoder_output_len(req) is None:
            return False
        skip_projection = getattr(req, "py_skip_cross_kv_projection", False)
        return not (isinstance(skip_projection, bool) and skip_projection)

    def _try_schedule_cross_context(self, req: LlmRequest) -> ScheduleAction:
        """Reserve cross-KV blocks for the first decoder context step."""
        if not self._needs_cross_context_allocation(req):
            return ScheduleAction.SCHEDULED

        if self.cross_kv_cache_manager is None:
            logger.warning(
                "Decoder context request %s requires cross-KV cache but "
                "no cross_kv_cache_manager is configured. Skipping.",
                req.py_request_id,
            )
            return ScheduleAction.STOP

        req_tokens = self._get_optional_encoder_output_len(req)
        if req_tokens is None:
            return ScheduleAction.SCHEDULED
        if not self._try_allocate_cross_context(self.cross_kv_cache_manager, req, req_tokens):
            return ScheduleAction.SKIP
        return ScheduleAction.SCHEDULED

    @staticmethod
    def _try_allocate_cross_context(
        cross_kv_cache_manager, req: LlmRequest, req_tokens: int
    ) -> bool:
        """Reserve V2 cross-KV without mutating decoder context position."""
        kv_cache = cross_kv_cache_manager.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            if not req.is_first_context_chunk:
                logger.debug(
                    "cross KV cache missing for non-first context chunk, request %s",
                    req.py_request_id,
                )
                return False
            input_tokens = (
                req.get_encoder_unique_tokens()
                if cross_kv_cache_manager.enable_block_reuse
                else None
            )
            kv_cache = cross_kv_cache_manager._create_kv_cache(
                req.py_request_id, req.lora_task_id, input_tokens, cache_salt=req.cache_salt
            )
            kv_cache.cuda_stream = cross_kv_cache_manager._stream.cuda_stream

        if not cross_kv_cache_manager.enable_block_reuse:
            kv_cache.stop_committing()

        if not cross_kv_cache_manager._resume_and_restore(req.py_request_id, kv_cache):
            return False

        target_capacity = req_tokens + cross_kv_cache_manager.num_extra_kv_tokens
        if not kv_cache.resize(max(kv_cache.capacity, target_capacity)):
            if req.is_first_context_chunk:
                kv_cache.suspend()
            return False

        return True

    def _try_schedule_generation(
        self,
        req: LlmRequest,
        budget: BudgetTracker,
        requests_list: list,
        req_it: int,
        req_it_end: int,
        recompute_pause_state: _RecomputePauseState,
        evicted: RequestList,
        recompute_paused: RequestList,
        inflight_request_ids: set[int],
        scheduled_beam_width: int,
    ) -> tuple[ScheduleAction, int, int, int]:
        """Try to schedule a generation request.

        Returns ``(action, tokens, scheduled_beam_width, req_it_end)``.
        *tokens* is meaningful only when *action* is ``SCHEDULED``.
        """
        beam_width = (
            1 if self._single_beam else req.get_beam_width_by_iter(for_next_iteration=False)
        )
        req_tokens = beam_width + get_draft_token_length(req)

        if not budget.can_fit_tokens(req_tokens):
            return ScheduleAction.STOP, 0, scheduled_beam_width, req_it_end

        if scheduled_beam_width == 0:
            scheduled_beam_width = beam_width
        elif scheduled_beam_width != beam_width:
            return ScheduleAction.SKIP, 0, scheduled_beam_width, req_it_end

        self._lost_kv_error = None
        success = self._try_allocate_generation(req)
        if self._lost_kv_error is not None:
            lost = self._on_lost_kv_cache(req, inflight_request_ids, recompute_paused)
            return lost, 0, scheduled_beam_width, req_it_end

        if not success and self.kv_cache_manager.flush_deferred_closes():
            # Finished requests whose close is deferred
            # (kv_cache_config.defer_generation_only_release) still hold pages.
            # Closing them now is never earlier than the immediate close they
            # replace, and costs no running request a suspend or a recompute.
            success = self._try_allocate_generation(req)
            if self._lost_kv_error is not None:
                lost = self._on_lost_kv_cache(req, inflight_request_ids, recompute_paused)
                return lost, 0, scheduled_beam_width, req_it_end

        if not success and self.has_cp_helix and not self.enable_recompute_pause:
            # Without recompute pause (e.g. a disaggregated generation server,
            # which cannot replay a prefill) and with the deferred closes above
            # flushed, no page is left to free. Suspending to a lower tier is
            # validated only under aggregated helix, so fail instead of
            # stalling.
            raise RuntimeError(
                f"[V2Scheduler] KV allocation failed for helix request "
                f"{req.py_request_id}; recompute pause is disabled "
                f"(e.g. on a disaggregated generation server, which "
                f"cannot replay the prefill), so the scheduler cannot "
                f"preempt a request. Increase "
                f"kv_cache_config.max_gpu_total_bytes (rank-local bytes) "
                f"or max_tokens (GLOBAL tokens across CP ranks, not "
                f"per-rank), or reduce concurrency."
            )

        if not success:
            # Under helix without a tier below GPU, a suspended request's pages
            # have nowhere to go, so recompute-pause is the only way to make
            # room.
            if not self.has_cp_helix or self.kv_cache_manager.has_cache_tier_below_gpu:
                req_it_end, success = self._try_evict_for_gen(
                    req, requests_list, req_it, req_it_end, evicted, inflight_request_ids
                )

        if not success and self._lost_kv_error is None:
            req_it_end, success = self._try_recompute_pause_for_gen(
                req,
                requests_list,
                req_it,
                req_it_end,
                recompute_pause_state,
                evicted,
                recompute_paused,
                inflight_request_ids,
            )

        if self._lost_kv_error is not None:
            # Lost only once a victim made room for its resume; the ladders stopped there.
            lost = self._on_lost_kv_cache(req, inflight_request_ids, recompute_paused)
            return lost, 0, scheduled_beam_width, req_it_end

        if success:
            return ScheduleAction.SCHEDULED, req_tokens, scheduled_beam_width, req_it_end

        # Self-eviction: suspend this gen request to free its
        # GPU pages so other requests can resume().
        # Skip if already suspended — suspending again is a no-op
        # that frees no pages.
        if self.has_cp_helix and not self.kv_cache_manager.has_cache_tier_below_gpu:
            # Suspending frees nothing under helix without a tier below GPU.
            # The request still sits out this step, so clear its stale batch
            # index.
            logger.debug(
                f"[V2Scheduler] Helix request {req.py_request_id} found no pages or victim; waiting"
            )
            self._clear_request_runtime_state(req)
        elif self.kv_cache_manager.is_request_active(
            req.py_request_id
        ) and not self._has_pending_connector_load(req):
            logger.debug(
                f"[V2Scheduler] Self-evicting request {req.py_request_id} "
                f"(state={req.state.name}) to free GPU pages"
            )
            self._suspend_request(req)
            evicted.append(req)

        return ScheduleAction.STOP, 0, scheduled_beam_width, req_it_end

    def _try_allocate_generation(self, req: LlmRequest) -> bool:
        """Atomically admit one generation step in target and draft pools.

        The draft pool mirrors the target pool's requests and token counts (a
        standalone drafter's pool may use its own ``tokens_per_block``), so a
        draft-side failure just means this request does not fit. Roll the target growth back and report
        a plain allocation failure, letting the caller run the same evict /
        recompute-pause / self-suspend ladder it uses for target-pool pressure.

        An unpaired draft pool sizes itself in ``prepare_resources`` rather than
        here, but its mirror still has to be admitted under this rollback: see
        ``_admit_unpaired_draft``.
        """
        try:
            if not self.kv_cache_manager.try_allocate_generation(req):
                return False
        except RemoteMemError as error:
            # Its resume read pages a failed remote cache level took with it: no room brings them back.
            self._lost_kv_error = error
            return False

        draft_manager = self._joint_draft_manager
        if draft_manager is None:
            if self._admit_unpaired_draft(req):
                return True
        elif draft_manager.try_allocate_generation(req):
            return True

        self.kv_cache_manager.revert_allocate_generation(req)
        return False

    def _on_lost_kv_cache(
        self, req: LlmRequest, inflight_request_ids: set[int], recompute_paused: RequestList
    ) -> ScheduleAction:
        """Handle *req* after its cache's resume raised ``_lost_kv_error``: pages it needs were on a remote cache
        level that failed, which the manager retires and unlinks from the reuse tree.

        A request that computed nothing yet matches its prompt again next pass. A started one re-prefills through
        recompute pause where this executor can, and is failed (``take_lost_kv_requests``) where it cannot.

        Without ``recover_lost_kv_per_rank`` the error is re-raised: lock-step peers did not see it.
        """
        error, self._lost_kv_error = self._lost_kv_error, None
        if not self.recover_lost_kv_per_rank:
            raise RemoteMemError(
                f"Request {req.py_request_id} lost KV pages to a failed remote cache level on this rank, whose "
                f"TP/CP/PP peers schedule in lock step and did not see the failure; stopping the executor"
            ) from error
        if not self._is_started_request(req):
            logger.warning(
                f"[V2Scheduler] Request {req.py_request_id} matched KV pages on a failed remote cache level; "
                f"matching it again: {error}"
            )
            self.release_context_admission(req)
        elif self.enable_recompute_pause and self._is_recompute_pause_candidate(
            req, inflight_request_ids
        ):
            logger.warning(
                f"[V2Scheduler] Request {req.py_request_id} lost KV pages to a failed remote cache level; "
                f"recomputing it: {error}"
            )
            self._preempt_or_defer(req, recompute_paused)
        else:
            logger.error(
                f"[V2Scheduler] Request {req.py_request_id} lost KV pages to a failed remote cache level and "
                f"cannot recompute them here; failing it: {error}"
            )
            self._clear_request_runtime_state(req)
            if all(lost is not req for lost, _ in self._lost_kv_requests):
                message = f"KV cache pages lost with a failed remote cache level: {error}"
                self._lost_kv_requests = (*self._lost_kv_requests, (req, message))
        return ScheduleAction.SKIP

    def _raise_if_lockstep_remote_level_failed(self) -> None:
        """Stop a lock-step rank once any of its remote cache levels failed.

        The failure is rank-local (a donor this rank's chunks sat on, or this rank's transfer timeout), and the level's
        pages leave only this rank's reuse tree, so the group's matches, admissions and collectives would diverge.
        """
        for manager in self._lockstep_remote_managers:
            if manager.any_remote_level_failed:
                raise RemoteMemError(
                    "A remote KV cache level failed on this rank. Its TP/CP/PP peers schedule in lock step and cannot "
                    "drop the same pages or requests, so the executor stops instead of diverging from them."
                )

    def take_lost_kv_requests(self) -> list[tuple[LlmRequest, str]]:
        """Requests the last passes could neither schedule nor recompute after a remote cache level failed, with their
        error messages; the executor fails them."""
        lost, self._lost_kv_requests = self._lost_kv_requests, ()
        return list(lost)

    # ---- Eviction ----

    @staticmethod
    def _is_started_request(req: LlmRequest) -> bool:
        """A request that has begun execution and can be paused.
        Matches V1: (context_init && !first_chunk) || generation_in_progress.
        """
        return (
            req.is_context_init_state and not req.is_first_context_chunk
        ) or req.is_generation_in_progress_state

    def _suspend_request(self, req: LlmRequest) -> None:
        """Suspend a request's KV cache in both main and draft managers.

        TODO: Also release PEFT resources (mark_request_done) for the
        suspended request so the C++ PeftCacheManager can evict its
        adapter pages.  Currently only KV cache is freed; the adapter
        remains "active" on device, which could cause ensure_batch to
        fail if it needs to load a different adapter into a full cache.
        """
        if req.is_context_init_state:
            self._ctx_preempt_stats.suspended += 1
        self._clear_request_runtime_state(req)
        self.kv_cache_manager.suspend_request(req)
        if self.draft_kv_cache_manager is not None:
            self.draft_kv_cache_manager.suspend_request(req)

    def _clear_request_runtime_state(self, req: LlmRequest) -> None:
        req.py_batch_idx = None

    def _try_preempt_for_pages(
        self,
        requests_list: RequestList,
        protected_ids: set[int],
        inflight_request_ids: set[int],
        recompute_paused: RequestList,
        preempted_ids: set[int],
        requester_idx: int,
        locality_domain_id: int | None = None,
    ) -> bool:
        """Release one started request's KV cache so another can allocate.

        The fallback for a pool that suspension cannot drain; see
        `KVCacheManagerV2.preempt_request`. With a cache tier below GPU, this
        waits while an active decode request holds GPU pages; under helix it
        falls back to preemption once none does.

        The victim leaves on `recompute_paused`, the same channel the
        generation side uses, because a re-prefill needs more teardown than the
        KV cache: the executor frees the sequence slot and
        `reset_for_recompute` rewrites the prompt. Pausing the request here
        instead would leave the slot owned by SeqSlotManager while
        `py_seq_slot` is None, which asserts on the next schedule.

        The victim is the newest eligible request, so the requests closest to
        completing keep their pages and the pool drains instead of thrashing.
        With ``preempt_least_prefill_progress`` it is the eligible request with
        the least prefill progress (see `_prefill_progress`), the newest of them
        on a tie, so the pass discards the least finished prefill.

        Every call is counted in the context page-pressure stats
        (`_maybe_log_ctx_preemption_stats`).

        Returns True when pages became available in this iteration.
        """
        stats = self._ctx_preempt_stats
        if self.kv_cache_manager.has_cache_tier_below_gpu:
            # Wait while an active generation request holds GPU pages: they
            # come back when it finishes, or if a failing generation request
            # suspends it; a blocked context request never triggers that
            # suspend. Under helix, once no active generation request is left,
            # every started request is mid-prefill or suspended, and none gives
            # its pages back on its own, so fall through to recompute-pause.
            if not self.has_cp_helix or any(
                r.is_generation_in_progress_state
                and self.kv_cache_manager.is_request_active(r.py_request_id)
                for r in requests_list
            ):
                stats.preempt_off += 1
                return False
        # A disaggregated generation worker received its context KV rather
        # than computing it, so it cannot replay a prefill at all. Without
        # recompute pause only a context-only request still mid-prefill is a
        # victim, and only with context-only preemption (see __init__).
        ctx_only = not self.enable_recompute_pause
        if ctx_only and not self._ctx_only_preempt:
            stats.preempt_off += 1
            return False

        # Newest first, so the requests closest to completing keep their
        # pages and the pool drains instead of thrashing.
        stop = -1
        if self.has_cp_helix:
            # Only requests behind the requester, as the generation ladder
            # does: an older victim would retake the pages on the next pass
            # and preempt the requester in turn.
            stop = requester_idx
        candidates = [
            victim
            for victim in reversed(requests_list[stop + 1 :])
            if victim.py_request_id not in protected_ids
            and not _in_other_locality_domain(victim, locality_domain_id)
            and self._is_recompute_pause_candidate(victim, inflight_request_ids)
            and (not ctx_only or self._is_ctx_only_victim(victim))
        ]
        # An active request holds its pages on GPU, so preempting it always
        # frees GPU memory. A suspended request's pages move to the lower tier
        # whenever that tier has room, so it is worth preempting only once the
        # tier is full and its pages are stuck on GPU; take one only when no
        # active request is left to give.
        active = [
            victim
            for victim in candidates
            if self.kv_cache_manager.is_request_active(victim.py_request_id)
        ]
        if active or not self.kv_cache_manager.has_cache_tier_below_gpu:
            candidates = active
        if not candidates:
            stats.no_victim += 1
            return False
        victim = candidates[0]
        if self._ctx_preempt_least_progress:
            # min() keeps the first of equal values, the newest.
            victim = min(candidates, key=self._prefill_progress)
        if not self._preempt_or_defer(victim, recompute_paused):
            # A connector save parks the victim; it may still be on this pass's
            # pending context list, and scheduling it there would size a chunk
            # for a request outside the context phase.
            preempted_ids.add(victim.py_request_id)
            stats.deferred += 1
            return False
        victim_progress = self._prefill_progress(victim)
        logger.debug(
            f"[V2Scheduler] Preempting request {victim.py_request_id} "
            f"(state={victim.state.name}, prefill_progress={victim_progress})"
        )
        stats.preempted += 1
        stats.preempted_tokens += victim_progress
        preempted_ids.add(victim.py_request_id)
        return True

    def _is_ctx_only_victim(self, req: LlmRequest) -> bool:
        """A context-only request still in its prefill and idle.

        Its own worker computed the pages it holds and can compute them again,
        none of them is being sent yet (a remote tail request may already be
        exchanging its prefix, so it is left alone), and no chunk of it is still
        on the GPU (not scheduled in the last two passes, not in the last batch
        known to have launched).
        """
        rid = req.py_request_id
        return (
            getattr(req, "is_context_only_request", False) is True
            and req.is_context_init_state
            and getattr(req, "py_csa2_remote_tail_mode", None) is None
            and rid not in self._ctx_scheduled_last
            and rid not in self._ctx_scheduled_prev
            and rid not in self._ctx_launched
            and rid not in self._ctx_unprocessed
        )

    @staticmethod
    def _prefill_progress(req: LlmRequest) -> int:
        """Prompt tokens *req* has computed past the prefix it reused or loaded.

        This is the prefill work a preemption discards; the reused prefix can be
        matched again on re-admission.
        """
        return max(req.context_current_position - req.prepopulated_prompt_len, 0)

    def _maybe_log_ctx_preemption_stats(self) -> None:
        """Log the context page-pressure counters once per 60 s window, then reset them.

        Called on passes that saw a pending context request or a context-state
        suspend, so a server without context requests stays silent.
        """
        now = time.monotonic()
        elapsed = now - self._ctx_preempt_last_log
        if elapsed < 60.0:
            return
        self._ctx_preempt_last_log = now
        window = self._ctx_preempt_stats
        totals = self._ctx_preempt_totals
        totals.add(window)
        victim_order = "least_progress" if self._ctx_preempt_least_progress else "newest"
        logger.debug(
            f"ctx preemption: window_s={elapsed:.0f} {window}; "
            f"total preempted={totals.preempted} preempted_tokens={totals.preempted_tokens} "
            f"suspended={totals.suspended}; victim_order={victim_order} "
            f"ctx_only_preempt={self._ctx_only_preempt}"
        )
        self._ctx_preempt_stats = _CtxPreemptionStats()

    # Consecutive scheduling passes that reclaimed nothing before this counts
    # as a deadlock. A stalled pass costs ~2ms, so it trips within seconds,
    # while transient one-iteration deferrals (multimodal chunk alignment,
    # PEFT budget, IndexMapper slots) clear long before.
    _DEADLOCK_STALL_ITERS = 1000

    # States in which an in-flight KV transfer still owns pages it is about to
    # release: a context-only request sending its cache after prefill, one
    # whose send landed and which the executor has yet to reap, and a
    # generation request receiving a cache.
    _TRANSFER_HOLDING_STATE_VALUES = frozenset(
        {
            LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS.value,
            LlmRequestState.DISAGG_CONTEXT_COMPLETE.value,
            LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS.value,
        }
    )

    def _detect_deadlock(
        self,
        active_requests: RequestList,
        inflight_request_ids: set[int],
        pending_ctx: RequestList,
        preempted_ids: set[int],
        made_progress: bool,
    ) -> None:
        """Fail loudly when no request can be scheduled or reclaimed.

        Unchecked, the executor spins at full speed while scheduling nothing,
        which looks healthy to the hang detector and to `/health`. Context
        candidates count alongside generation ones because a disaggregated
        prefill server has no generation requests at all.
        """
        if made_progress:
            if self._stalled_schedules:
                logger.debug(
                    f"[V2Scheduler] Stall cleared after {self._stalled_schedules} pass(es); "
                    "scheduling resumed."
                )
            self._stalled_schedules = 0
            return

        num_gen_candidates = sum(
            1
            for r in active_requests
            if r.is_generation_in_progress_state
            and not r.is_generation_to_complete_state
            and r.request_id not in inflight_request_ids
            and not self._has_pending_connector_load(r)
        )
        num_ctx_candidates = sum(
            1
            for r in pending_ctx
            if r.py_request_id not in preempted_ids and r.request_id not in inflight_request_ids
        )
        if num_gen_candidates == 0 and num_ctx_candidates == 0:
            # Legitimately idle: nothing to schedule.
            if self._stalled_schedules:
                logger.debug(
                    f"[V2Scheduler] Stall cleared after {self._stalled_schedules} pass(es); "
                    "no candidates remain."
                )
            self._stalled_schedules = 0
            return

        # A connector load in flight releases its pages when it lands, so a
        # pass that reclaims nothing while one is outstanding is not a stall.
        # A preemption awaiting the connector's saves is the same case from
        # the other direction: those pages are already given up, but the
        # victim keeps its state until they land, so nothing above sees them.
        if (
            any(self._has_pending_connector_load(r) for r in active_requests)
            or self.kv_cache_manager.has_pending_preemption()
        ):
            self._stalled_schedules = 0
            return

        # Waiting on a transfer is not a deadlock: the pages come back when it
        # lands. None of those states are schedulable, so a context server
        # whose pool is full of pending sends shows no progress here at all. A
        # send that never lands is the transfer layer's timeout to report.
        #
        # The active_requests scan covers senders still on the list and the
        # generation-side receiver. The transfer manager covers finished
        # senders that have already left active_requests (see
        # set_async_transfer_manager). Logically cancelled receivers also leave
        # that list while the coordinator retains their physical ownership.
        transfer_holding = (
            any(req.state_value in self._TRANSFER_HOLDING_STATE_VALUES for req in active_requests)
            or (
                self._async_transfer_manager is not None
                and self._async_transfer_manager.has_any_inflight_requests()
            )
            or self._retiring_gen_receives_probe()
        )
        if transfer_holding:
            if self._stalled_schedules:
                logger.debug(
                    f"[V2Scheduler] {num_gen_candidates} generation and "
                    f"{num_ctx_candidates} context request(s) cannot allocate, but KV "
                    "transfers in flight will release pages; stall cleared."
                )
            self._stalled_schedules = 0
            return

        self._stalled_schedules += 1
        if self._stalled_schedules < self._DEADLOCK_STALL_ITERS:
            # Warn as the stall builds so the raise is not a surprise. A quarter
            # of the threshold keeps this to a handful of lines beforehand.
            if self._stalled_schedules == 1:
                logger.debug(
                    "[V2Scheduler] Scheduling stall started: "
                    f"{self._summarize_stall(active_requests, pending_ctx, inflight_request_ids, preempted_ids)}"
                )
            elif self._stalled_schedules % max(1, self._DEADLOCK_STALL_ITERS // 4) == 0:
                logger.warning(
                    f"[V2Scheduler] Stalled {self._stalled_schedules}/"
                    f"{self._DEADLOCK_STALL_ITERS} consecutive passes: "
                    f"{self._summarize_stall(active_requests, pending_ctx, inflight_request_ids, preempted_ids)}"
                )
            return

        if self.has_cp_helix:
            # Under helix, max_gpu_total_bytes is per rank but max_tokens
            # counts tokens across all CP ranks.
            gpu_remedy = (
                "increase kv_cache_config.max_gpu_total_bytes (rank-local bytes), "
                "max_tokens (GLOBAL tokens across CP ranks) or "
                "free_gpu_memory_fraction, or lower max_batch_size."
            )
            # Only aggregated helix (the side that can recompute-pause) can
            # suspend to a host tier.
            if not self.enable_recompute_pause:
                remedy = gpu_remedy[0].upper() + gpu_remedy[1:]
            elif self.kv_cache_manager.has_cache_tier_below_gpu:
                remedy = (
                    "The host tier may be full: increase "
                    f"kv_cache_config.host_cache_size, or {gpu_remedy}"
                )
            else:
                remedy = f"Configure kv_cache_config.host_cache_size, or {gpu_remedy}"
        elif getattr(self.kv_cache_manager, "kv_connector_manager", None) is not None:
            # A connector rejects every tier below GPU at bring-up
            # (`PyExecutor._reject_non_gpu_cache_tiers`), so host_cache_size is
            # not an option here.
            remedy = (
                "A KV connector is attached, which requires a GPU-only cache, "
                "so no secondary tier can be configured. Increase "
                "kv_cache_config.max_tokens or "
                "kv_cache_config.free_gpu_memory_fraction, or lower "
                "max_num_tokens to hand memory back to the KV pool."
            )
        else:
            remedy = (
                "Configure kv_cache_config.host_cache_size, increase "
                "kv_cache_config.max_tokens, or lower max_batch_size."
            )
        # Logged before the raise so the diagnostic survives even if the
        # exception is caught or truncated upstream.
        logger.warning(
            f"[V2Scheduler] Declaring deadlock after {self._stalled_schedules} passes: "
            f"{self._summarize_stall(active_requests, pending_ctx, inflight_request_ids, preempted_ids)} "
            f"{remedy}"
        )
        raise RuntimeError(
            f"V2 scheduler deadlock: {num_gen_candidates} generation and "
            f"{num_ctx_candidates} context request(s) active but none could "
            f"be scheduled, suspended or preempted in "
            f"{self._stalled_schedules} consecutive attempts. The KV cache "
            f"pool is likely exhausted. {remedy}"
        )

    def _summarize_stall(
        self,
        active_requests: RequestList,
        pending_ctx: RequestList,
        inflight_request_ids: set[int],
        preempted_ids: set[int],
    ) -> str:
        """One-line breakdown of a stalled pass for the deadlock logs.

        Re-derives the blocked candidates so the per-pass detector stays
        cheap, this running only on the rare logging branches.
        """
        gen = [
            r
            for r in active_requests
            if r.is_generation_in_progress_state
            and not r.is_generation_to_complete_state
            and r.request_id not in inflight_request_ids
        ]
        ctx = [
            r
            for r in pending_ctx
            if r.py_request_id not in preempted_ids and r.request_id not in inflight_request_ids
        ]
        states = Counter(LlmRequestState(r.state_value).name for r in gen + ctx)
        sample_ids = [r.py_request_id for r in (gen + ctx)[:5]]
        has_connector = getattr(self.kv_cache_manager, "kv_connector_manager", None) is not None
        transfers_in_flight = (
            self._async_transfer_manager is not None
            and self._async_transfer_manager.has_any_inflight_requests()
        )
        return (
            f"{len(gen)} generation and {len(ctx)} context request(s) blocked; "
            f"states={dict(states)}; sample_ids={sample_ids}; "
            f"kv_connector={has_connector}; "
            f"cache_tier_below_gpu={self.kv_cache_manager.has_cache_tier_below_gpu}; "
            f"transfers_in_flight={transfers_in_flight}; "
            f"local_retiring_gen_receives={self._retiring_gen_receives_probe()}"
        )

    def _free_kv_caches(self, req: LlmRequest) -> None:
        self.kv_cache_manager.free_resources(req)
        if self.draft_kv_cache_manager is not None:
            self.draft_kv_cache_manager.free_resources(req)

    def _has_pending_connector_load(self, req: LlmRequest) -> bool:
        connector = self.kv_cache_manager.kv_connector_manager
        return connector is not None and connector.has_pending_load(req)

    def _is_evictable(self, req: LlmRequest, inflight_request_ids: set[int]) -> bool:
        """A started request whose KV cache is still active on GPU.

        Already-suspended requests are not useful eviction victims
        because suspending them again is a no-op that frees no pages.
        """
        if req.request_id in inflight_request_ids:
            return False
        if not self._is_started_request(req):
            return False
        if self._has_pending_connector_load(req):
            return False
        return self.kv_cache_manager.is_request_active(req.py_request_id)

    def _is_recompute_pause_candidate(
        self, req: LlmRequest, inflight_request_ids: set[int]
    ) -> bool:
        if req.request_id in inflight_request_ids:
            return False
        if self._has_pending_connector_load(req):
            return False
        # is_generation_in_progress_state also includes GENERATION_TO_COMPLETE,
        # which is outside the schedulable range and may still be finalizing.
        if req.state_value == self._gen_to_complete_state_value:
            return False
        # Completed multimodal prefill deliberately releases the inputs and
        # embedding needed for replay, leaving an empty or MRoPE-only dict as
        # the durable marker. Partial-context requests still retain replay data.
        if req.is_generation_in_progress_state and req.py_multimodal_data is not None:
            return False
        return self._is_started_request(req)

    def _preempt_or_defer(self, req: LlmRequest, recompute_paused: RequestList) -> bool:
        """Give up *req*'s pages for another request, if they are free to give.

        Every destructive release goes through here, because a page a connector
        save is still reading cannot be freed: a later request would allocate it
        and overwrite the bytes mid-transfer, and the store would publish one
        request's KV under another's hash. `KVCacheManagerV2.preempt_request`
        answers that question and parks the victim until the executor completes
        the release; see `PyExecutor._resume_preempted_request`.

        On success *req* lands on `recompute_paused`, the channel a re-prefill
        needs: the executor frees the sequence slot and `reset_for_recompute`
        rewrites the prompt. False means the pages are unavailable this
        iteration, and the caller stops looking rather than taking a second
        victim, so one drains at a time and the next pass finds those pages
        instead of a second request having given up its cache for nothing.
        """
        if not self.kv_cache_manager.preempt_request(req):
            logger.debug(
                f"[V2Scheduler] Preemption of request {req.py_request_id} "
                "deferred until its connector saves retire"
            )
            return False
        self._clear_request_runtime_state(req)
        if self.draft_kv_cache_manager is not None:
            self.draft_kv_cache_manager.free_resources(req)
        recompute_paused.append(req)
        return True

    def _try_evict_for_gen(
        self, req, requests_list, req_it, req_it_end, evicted, inflight_request_ids
    ):
        """Evict started requests from active_requests tail to make room.

        Search backwards from req_it_end
        for evictable requests (started AND KV cache active on GPU),
        suspend them to free pages, then retry allocation.

        Victims are always at indices >= req_it (not yet processed by the
        main loop), so they are never in scheduled_ctx/scheduled_gen and
        no token budget reclaim is needed.

        Returns (new_req_it_end, success) tuple. new_req_it_end is always
        updated to reflect evicted victims (even on failure) so the caller
        can skip already-evicted requests.
        """
        while req_it_end > req_it:
            victim_idx = None
            for i in range(req_it_end - 1, req_it, -1):
                if self._is_evictable(requests_list[i], inflight_request_ids):
                    victim_idx = i
                    break

            if victim_idx is None:
                break

            victim = requests_list[victim_idx]
            logger.debug(
                f"[V2Scheduler] Evicting request {victim.py_request_id} "
                f"(state={victim.state.name}) to free pages for request {req.py_request_id}"
            )
            self._suspend_request(victim)
            evicted.append(victim)
            req_it_end = victim_idx

            if self._try_allocate_generation(req):
                return req_it_end, True
            if self._lost_kv_error is not None:
                break  # no victim brings its pages back

        return req_it_end, False

    def _try_recompute_pause_for_gen(
        self,
        req: LlmRequest,
        requests_list: RequestList,
        req_it: int,
        req_it_end: int,
        recompute_pause_state: _RecomputePauseState,
        evicted: RequestList,
        recompute_paused: RequestList,
        inflight_request_ids: set[int],
    ) -> tuple[int, bool]:
        """Use destructive recompute pause when ordinary suspend is insufficient.

        The recompute frontier is independent from the ordinary-eviction frontier.
        This keeps previously-suspended started requests visible even when ordinary
        eviction skips over them and shrinks its frontier to an earlier victim.
        Recompute pause does not shrink the ordinary scheduling frontier; exact
        victim indices prevent destructively-freed requests from being revisited.
        """
        if not self.enable_recompute_pause:
            return req_it_end, False

        while True:
            victim_idx = None
            for i in range(recompute_pause_state.frontier - 1, req_it, -1):
                if i in recompute_pause_state.victim_indices:
                    continue
                candidate = requests_list[i]
                was_evicted = any(evicted_req is candidate for evicted_req in evicted)
                if (
                    self.kv_cache_manager.can_evict or not was_evicted
                ) and self._is_recompute_pause_candidate(candidate, inflight_request_ids):
                    victim_idx = i
                    break

            if victim_idx is None and self.kv_cache_manager.can_evict:
                victim = next(
                    (
                        candidate
                        for candidate in evicted
                        if self._is_recompute_pause_candidate(candidate, inflight_request_ids)
                    ),
                    None,
                )
                if victim is None:
                    break
                victim_idx = next(
                    i for i, candidate in enumerate(requests_list) if candidate is victim
                )
            elif victim_idx is not None:
                victim = requests_list[victim_idx]
            else:
                break

            evicted_victim_pos = next(
                (i for i, candidate in enumerate(evicted) if candidate is victim), None
            )
            if evicted_victim_pos is not None:
                evicted.pop(evicted_victim_pos)
            logger.debug(
                f"[V2Scheduler] Recompute-pausing request {victim.py_request_id} "
                f"to free pages for request {req.py_request_id}"
            )
            if not self._preempt_or_defer(victim, recompute_paused):
                # Off `evicted` above and onto no list here: a parked victim
                # keeps its pages, and the `pause` an evicted request gets
                # would overwrite the state that holds it there.
                return req_it_end, False
            recompute_pause_state.victim_indices.add(victim_idx)
            recompute_pause_state.frontier = min(recompute_pause_state.frontier, victim_idx)

            # Retry immediately: full teardown can make the allocation fit even
            # for an already-suspended victim. If it still fails, use any
            # secondary-tier capacity just released to ordinary-suspend another
            # active victim.
            success = self._try_allocate_generation(req)
            if self._lost_kv_error is not None:
                return req_it_end, False
            if not success and self.kv_cache_manager.can_evict:
                req_it_end, success = self._try_evict_for_gen(
                    req, requests_list, req_it, req_it_end, evicted, inflight_request_ids
                )

            if success:
                return req_it_end, True

        return req_it_end, False

    # ---- Sorting ----

    @staticmethod
    def _lora_key(req: LlmRequest):
        return _get_lora_task_id(req)

    def _sort_requests(self, context_requests, generation_requests, has_chunks):
        """Sort by LoRA task ID. Non-last chunks before last chunks."""
        if has_chunks:
            not_last = [r for r in context_requests if not r.is_last_context_chunk]
            last = [r for r in context_requests if r.is_last_context_chunk]
            not_last.sort(key=self._lora_key)
            last.sort(key=self._lora_key)
            context_requests.clear()
            context_requests.extend(not_last)
            context_requests.extend(last)
        else:
            context_requests.sort(key=self._lora_key)
        # Requests without a LoRA task share one key, which the stable sort keeps in place.
        if any(getattr(req, "lora_task_id", None) is not None for req in generation_requests):
            generation_requests.sort(key=self._lora_key)

    # ---- can_schedule (PP dry-run) ----

    def can_schedule(self, requests: RequestList) -> bool:
        """Conservative heuristic for PP retry loop. Does NOT allocate.

        TODO: V2's try-and-see model lacks a free-blocks query API.
        Implementing this requires exposing storage statistics from the
        V2 runtime (e.g., free GPU slots per pool group). For now,
        always returns True — PP is not yet supported (asserted in
        KVCacheManagerV2 constructor via kv_connector_manager=None).
        """
        return True

    # ---- Out-of-pages context retry gate (see the module header) ----

    def _oop_gate_init(self) -> None:
        # The gate engages only on the chunked context path (_try_schedule_context).
        self._oop_gate_track = not _OOP_GATE_DISABLED and self.chunking_enabled
        self._oop_gate_iter = 0
        # py_request_id -> (page counts after the failure, budget chunk of the failed attempt
        # (0: prepare failure), replay tokens of it, "alloc" | "prepare", pass of the attempt)
        self._oop_gate_fail: dict = {}
        self._oop_gate_sig_cache = None
        self._oop_gate_fail_kind = None
        self._oop_gate_fail_chunk = 0
        self._oop_gate_fail_replay = 0
        self._oop_gate_retry_reason: dict = {}
        self._oop_gate_shadow_ids: set = set()
        self._oop_gate_cnt = Counter()
        self._oop_gate_tot = Counter()
        self._oop_gate_last_log = time.monotonic()
        self._oop_gate_rank = _scheduler_rank()
        sig_len = -1
        if self._oop_gate_track:
            try:
                sig_len = len(self._oop_gate_signature())
                why = None if sig_len > 0 else "no pool reports page counts"
            except Exception as exc:  # noqa: BLE001 - any failure leaves the gate off
                why = f"{type(exc).__name__}: {exc}"
            if why is not None:
                self._oop_gate_track = False
                logger.warning(
                    f"Context out-of-pages retry gate disabled on rank {self._oop_gate_rank}: "
                    f"page counts unavailable ({why})"
                )
            self._oop_gate_sig_cache = None
        logger.debug(
            f"Context out-of-pages retry gate: rank={self._oop_gate_rank} mode={_OOP_GATE_MODE} "
            f"active={int(self._oop_gate_track)} max_skip={_OOP_GATE_MAX_SKIP} "
            f"chunked={int(self.chunking_enabled)} "
            f"ctx_only_preempt={int(self._ctx_only_preempt)} "
            f"recompute_pause={int(self.enable_recompute_pause)} "
            f"tier_below_gpu={int(bool(self.kv_cache_manager.has_cache_tier_below_gpu))} "
            f"draft={int(self.draft_kv_cache_manager is not None)} counts={sig_len} "
            f"stall_bypass={self._DEADLOCK_STALL_ITERS // 2}"
        )

    def _oop_gate_signature(self) -> tuple:
        """(available, evictable) page counts of every pool group at every cache level, target
        then draft pool. Both, because a finished request's pages can turn into reusable cached
        pages (evictable) rather than free ones; a rise in either ends the gate for everyone."""
        sig = self._oop_gate_sig_cache
        if sig is None:
            vals = []
            for mgr in (self.kv_cache_manager, self.draft_kv_cache_manager):
                impl = getattr(mgr, "impl", None)
                if impl is None:
                    continue
                for level in range(len(impl.cache_tier_list)):
                    for st in impl.get_storage_statistics(level):
                        vals.append(int(st.available))
                        vals.append(int(st.evictable))
            sig = self._oop_gate_sig_cache = tuple(vals)
        return sig

    def _oop_gate_begin_phase2(self, pending_ctx: RequestList) -> None:
        self._oop_gate_sig_cache = None  # phase 1 grew the generation requests
        self._oop_gate_retry_reason.clear()
        self._oop_gate_shadow_ids.clear()
        if self._oop_gate_fail:
            live = {r.py_request_id for r in pending_ctx}
            for rid in [rid for rid in self._oop_gate_fail if rid not in live]:
                del self._oop_gate_fail[rid]
        if pending_ctx:
            self._oop_gate_cnt["phase2_waiting"] += 1
            self._oop_gate_cnt["pending_sum"] += len(pending_ctx)

    def _oop_gate_check(
        self,
        req: LlmRequest,
        budget: BudgetTracker,
        preempt_for_pages: Callable[[LlmRequest], bool],
        token_cap: Optional[int] = None,
    ) -> Optional[ScheduleAction]:
        """The outcome to replay without the prepare/resize work, or None to attempt."""
        rid = req.py_request_id
        ent = self._oop_gate_fail.get(rid)
        if ent is None:
            return None
        sig_rec, chunk_rec, replay_rec, kind_rec, it_rec = ent
        reason = None
        if not req.is_first_context_chunk or self._holds_context_cache(req):
            reason = "state"
        elif self._stalled_schedules >= self._DEADLOCK_STALL_ITERS // 2:
            # Near a deadlock declaration every pass is a real attempt.
            reason = "stall"
        elif self._oop_gate_iter - it_rec >= _OOP_GATE_MAX_SKIP + (rid % _OOP_GATE_MAX_SKIP):
            reason = "cap"
        else:
            if chunk_rec:
                limit = budget.remaining_tokens
                if token_cap is not None:
                    limit = token_cap if limit is None else min(limit, token_cap)
                if limit is not None:
                    limit -= replay_rec
                cap = self._ctx_chunk_cap_this_pass
                if cap is not None and self._ctx_chunk_cap_applies(req):
                    limit = cap if limit is None else min(limit, cap)
                if limit is not None and limit < chunk_rec:
                    reason = "chunk"
            if reason is None:
                try:
                    sig = self._oop_gate_signature()
                except Exception:  # noqa: BLE001 - unknown counts mean attempt
                    sig = None
                if (
                    sig is None
                    or len(sig) != len(sig_rec)
                    or any(a > b for a, b in zip(sig, sig_rec))
                ):
                    reason = "rise"
        if reason is not None:
            self._oop_gate_cnt["retry_" + reason] += 1
            self._oop_gate_retry_reason[rid] = reason
            return None
        if not _OOP_GATE_ON:
            self._oop_gate_cnt["shadow_gated"] += 1
            self._oop_gate_shadow_ids.add(rid)
            return None
        self._oop_gate_cnt["gated"] += 1
        self._oop_gate_cnt["gated_" + kind_rec] += 1
        if kind_rec == "alloc" and preempt_for_pages(req):
            # An attempt reaches the out-of-pages branch and preempts the same victim; it ends
            # the pass and the next pass attempts for real.
            self._oop_gate_sig_cache = None
            self._oop_gate_cnt["gated_preempt"] += 1
            del self._oop_gate_fail[rid]
            return ScheduleAction.STOP
        return ScheduleAction.SKIP

    def _oop_gate_record(self, req: LlmRequest, first_chunk: bool, action: ScheduleAction) -> None:
        rid = req.py_request_id
        self._oop_gate_sig_cache = None  # the attempt may have changed the pool
        kind, self._oop_gate_fail_kind = self._oop_gate_fail_kind, None
        cnt = self._oop_gate_cnt
        cnt["attempts"] += 1
        shadow_hit = rid in self._oop_gate_shadow_ids
        reason = self._oop_gate_retry_reason.pop(rid, None)
        old = self._oop_gate_fail.get(rid)
        if action is ScheduleAction.SCHEDULED:
            if old is not None:
                del self._oop_gate_fail[rid]
                cnt["retry_ok"] += 1
                cnt["retry_ok_" + (reason or ("shadow" if shadow_hit else "none"))] += 1
            if shadow_hit:
                cnt["shadow_miss"] += 1
            return
        if not first_chunk or kind is None:
            self._oop_gate_fail.pop(rid, None)
            return
        cnt["fails"] += 1
        cnt["fail_" + kind] += 1
        if shadow_hit and old is not None:
            return  # shadow: the gate would not have attempted; keep its last real record
        try:
            sig = self._oop_gate_signature()
        except Exception:  # noqa: BLE001 - unknown counts mean no record
            self._oop_gate_fail.pop(rid, None)
            return
        self._oop_gate_fail[rid] = (
            sig,
            self._oop_gate_fail_chunk if kind == "alloc" else 0,
            self._oop_gate_fail_replay if kind == "alloc" else 0,
            kind,
            self._oop_gate_iter,
        )

    def _oop_gate_maybe_log(self) -> None:
        now = time.monotonic()
        elapsed = now - self._oop_gate_last_log
        first = (
            not self._oop_gate_tot
            and self._oop_gate_cnt["gated"] + self._oop_gate_cnt["shadow_gated"] > 0
        )
        if elapsed < _OOP_GATE_LOG_EVERY_S and not first:
            return
        self._oop_gate_last_log = now
        w = self._oop_gate_cnt
        self._oop_gate_tot.update(w)
        t = self._oop_gate_tot
        keys = (
            "phase2_waiting",
            "attempts",
            "fails",
            "fail_alloc",
            "fail_prepare",
            "gated",
            "gated_alloc",
            "gated_prepare",
            "gated_preempt",
            "shadow_gated",
            "shadow_miss",
            "retry_rise",
            "retry_chunk",
            "retry_cap",
            "retry_stall",
            "retry_state",
            "retry_ok",
            "retry_ok_rise",
            "retry_ok_chunk",
            "retry_ok_cap",
            "retry_ok_stall",
            "retry_ok_state",
        )
        pw = max(1, w["phase2_waiting"])
        logger.debug(
            f"Context out-of-pages retry gate: rank={self._oop_gate_rank} mode={_OOP_GATE_MODE} "
            f"it={self._oop_gate_iter} window_s={elapsed:.0f} "
            f"pending_mean={w['pending_sum'] / pw:.2f} tracked={len(self._oop_gate_fail)} "
            + " ".join(f"{k}={w[k]}" for k in keys)
            + " | total "
            + " ".join(
                f"{k}={t[k]}"
                for k in ("attempts", "fails", "gated", "shadow_gated", "shadow_miss", "retry_ok")
            )
        )
        self._oop_gate_cnt = Counter()

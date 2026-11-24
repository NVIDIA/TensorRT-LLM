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
from bisect import bisect_left, insort
from collections import Counter
from typing import Callable, Optional

from tensorrt_llm.llmapi.llm_args import (
    CapacitySchedulerPolicy,
    ContextChunkCapConfig,
    ContextChunkingPolicy,
)
from tensorrt_llm.logger import logger

from ..llm_request import (
    LlmRequest,
    LlmRequestState,
    get_draft_token_length,
    rewind_context_after_cache_drop,
)
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

# "1": an unscheduled first context chunk keeps its cache claim unless admission
# ran out of KV pages (see KVCacheV2Scheduler.__init__).
KEEP_UNSCHEDULED_FIRST_CHUNK_CLAIM_ENV = "TLLM_V2_KEEP_UNSCHEDULED_FIRST_CHUNK_CLAIM"


class ScheduleAction(enum.Enum):
    """Result of a per-request scheduling attempt."""

    SCHEDULED = "scheduled"  # success — payload contains tokens etc.
    SKIP = "skip"  # skip this request, continue the loop
    STOP = "stop"  # stop the scheduling loop
    DEFER = "defer"  # retry this request after the rest of this iteration's order
    # A generation request found no pages; the loop stops unless
    # TLLM_V2_GEN_SKIP_BLOCKED is set (see KVCacheV2Scheduler.__init__).
    BLOCKED = "blocked"


class _RecomputePauseState:
    """Candidate frontier and exact victims for one scheduling iteration."""

    def __init__(self, frontier: int):
        self.frontier = frontier
        self.victim_indices: set[int] = set()


class _EqualProgress:
    """The equal-progress split of one scheduling pass's context token budget.

    Every context request the pass may still offer a chunk counts with the prompt tokens it
    needs (exact once it holds its cache, else an estimate). A request is offered the water
    level of the remaining budget over itself and the requests after it: the largest L such
    that each of them can take min(need, L).
    """

    def __init__(self, needs: dict[int, int]) -> None:
        self._needs = needs
        self._sorted = sorted(needs.values())

    def offered(self, request_id: int) -> None:
        """Take a request out of the ones still to be offered (its own need is exact later)."""
        need = self._needs.pop(request_id, None)
        if need is not None:
            del self._sorted[bisect_left(self._sorted, need)]

    def level(self, budget: int, own_need: int) -> int:
        """The water level of ``budget`` over this request (``own_need``) and the rest."""
        needs = list(self._sorted)
        insort(needs, own_need)
        left = budget
        for i, need in enumerate(needs):
            rest = len(needs) - i
            if need * rest > left:
                return left // rest
            left -= need
        return own_need


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


# Opt-in decode headroom for disagg gen-init admission: the number of tokens of
# future growth, per resident generation request, that admission keeps room for.
GEN_INIT_DECODE_HEADROOM_ENV = "TRTLLM_DISAGG_GEN_INIT_DECODE_HEADROOM_TOKENS"

# Opt-in extra receive slots on a disaggregated generation server: how many
# generation-only requests the executor may hold beyond max_batch_size while
# their KV cache is still arriving. 0 (the default) is off.
GEN_EXTRA_SLOTS_ENV = "TRTLLM_DISAGG_GEN_EXTRA_SLOTS"


def disagg_gen_extra_slots() -> int:
    """Return the extra receive slots set by ``TRTLLM_DISAGG_GEN_EXTRA_SLOTS`` (0 = off)."""
    raw = os.environ.get(GEN_EXTRA_SLOTS_ENV, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{GEN_EXTRA_SLOTS_ENV} must be a non-negative integer, got {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError(f"{GEN_EXTRA_SLOTS_ENV} must be a non-negative integer, got {value}")
    return value


class _GenInitDecodeHeadroom:
    """Keeps room for the decode growth of the requests already on a generation engine.

    A disagg gen-init request takes its whole prompt's KV in one allocation,
    while a decoding request grows one block at a time. Once admissions have
    filled the pool, a decoding request whose last block is full cannot grow,
    and its failed allocation stops the scheduling pass, so every request
    behind it waits as well. A gen-init request is therefore admitted only if,
    after its prompt blocks, each pool group still holds the next
    ``lookahead_tokens`` (capped by the tokens each request may still generate,
    plus its draft tokens) of every resident generation request, its own
    included.

    Everything is read from this rank's scheduler inputs: the request states
    and the KV cache manager. The TP ranks of a non-ADP engine hold identical
    copies, so they decide alike; ``align_gen_admission_across_tp`` drops any
    request on which they still differ.
    """

    _RESIDENT_STATE_VALUES = frozenset(
        state.value
        for state in (
            LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
            LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE,
            LlmRequestState.GENERATION_IN_PROGRESS,
        )
    )

    def __init__(self, kv_cache_manager, requests: RequestList, lookahead_tokens: int) -> None:
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

    def _growth_blocks(self, req: LlmRequest, capacity: int) -> int:
        max_new_tokens = getattr(req, "py_max_new_tokens", None)
        lookahead = self._lookahead_tokens
        if max_new_tokens is not None:
            generated = max(0, req.get_num_tokens(0) - req.prompt_len)
            lookahead = min(lookahead, max(0, max_new_tokens - generated))
        tokens = lookahead + get_draft_token_length(req)
        return self._blocks(capacity + tokens) - self._blocks(capacity)

    def _admission_blocks(self, req: LlmRequest) -> tuple[int, int]:
        """(prompt blocks still to allocate, growth blocks after them) for *req*."""
        cache = self._kv.kv_cache_map.get(req.py_request_id)
        held = cache.capacity if cache is not None else 0
        target = max(
            held, req.prompt_len + get_draft_token_length(req) + self._kv.num_extra_kv_tokens
        )
        return self._blocks(target) - self._blocks(held), self._growth_blocks(req, target)

    def admits(self, req: LlmRequest) -> Optional[tuple[int, int]]:
        """The blocks *req* would take, or None when the pool cannot spare them.

        Called before the request's allocation; pass the result to ``commit``
        once the allocation succeeds.
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


#: "1" turns on ShortestPrefillFirstOrder (default off: arrival order).
SHORTEST_PREFILL_FIRST_ENV = "TRTLLM_SCHED_SHORTEST_PREFILL_FIRST"
#: ShortestPrefillFirstOrder's aging factor; read only with shortest prefill first on.
_AGING_FACTOR_ENV = "TRTLLM_SCHED_SHORTEST_PREFILL_AGING_FACTOR"
#: With long-prefill demotion on too: what a request demotion lets through for good
#: (TLLM_V2_CTX_DEMOTE_MAX_ITERS) is ranked by. Default: its clock reading alone, so no
#: later arrival is tried ahead of it. "1": its shortest-first key.
SPF_RELEASE_KEEPS_KEY_ENV = "TRTLLM_SCHED_SHORTEST_PREFILL_RELEASE_KEEPS_KEY"
#: "1": one INFO line per context request when its last chunk is scheduled
#: (see _ContextAdmissionLog). Default off.
CTX_ADMIT_LOG_ENV = "TRTLLM_SCHED_CTX_ADMIT_LOG"


def _read_non_negative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as err:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from err
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return value


class ShortestPrefillFirstOrder:
    """Order pending context requests shortest prefill first, aged by admitted context work.

    A request's work is the part of its prompt that is not cached yet: its prompt length
    minus the reusable prefix the KV cache manager reports, probed once (or the exact
    remainder long-prefill demotion already probed, TLLM_V2_CTX_DEMOTE_EXACT); a request
    first seen after it started prefilling counts its remaining context instead. Its key is
    the context tokens the scheduler had admitted when the request reached the context state,
    plus the aging factor F times that work. Pending context requests are tried in key order,
    ties in arrival order.

    Requests that arrive between the same two context admissions share the clock reading, so
    the shortest goes first. An earlier request goes ahead of a later one once the context
    tokens admitted between their arrivals exceed F times the difference in their work. A long
    cold prompt that shorter ones keep crowding out therefore reaches the front once the
    scheduler has admitted F times as many context tokens as it needs itself: from then on no
    later arrival is tried ahead of it. The bound scales with each request's own work; F is the
    order's only setting (``TRTLLM_SCHED_SHORTEST_PREFILL_AGING_FACTOR``, default
    ``DEFAULT_AGING_FACTOR``). F = 0 tries pending context in arrival order and skips the
    prefix probe; a larger F stays closer to pure shortest first and lets a long prompt wait
    longer.

    With long-prefill demotion on as well (TLLM_V2_CTX_DEMOTE_UNCACHED_TOKENS), demotion
    sets the class and this order the position within it: requests demoted this pass are
    tried after all the others, each class in key order. A request demotion lets through for
    good (TLLM_V2_CTX_DEMOTE_MAX_ITERS) leaves the demoted class; with ``release_aged`` (the
    default) it also counts as fully aged: its key drops the work term, so no later arrival
    is tried ahead of it, as no later arrival is in arrival order. With
    ``TRTLLM_SCHED_SHORTEST_PREFILL_RELEASE_KEEPS_KEY=1`` it keeps its shortest-first key,
    and the aging factor alone bounds its wait. A request ordered as short and found long once prepared
    still goes behind the rest of the pass (ScheduleAction.DEFER).

    The clock counts the context tokens of the requests the scheduler returned, a function
    of its decisions, so ranks that schedule the same requests order them alike. Under
    attention DP every rank schedules its own requests, so each rank keeps its own clock
    and orders only the requests routed to it.

    The order may try a request that has not started prefilling ahead of one that has. The
    scheduler therefore never lets the former preempt the latter for KV pages: arrival order
    would have scheduled, and protected, the started request first.

    The prefix probe is a radix walk over the prompt, so it runs at most once per request
    (again after a pass spent waiting for an in-flight prefix, see ``forget_work``) and only
    when there is something to reorder.
    """

    #: Default F: how many times its own prefill work the scheduler admits for other requests
    #: before a pending request stops being overtaken by later arrivals. 1 is the tightest
    #: nonzero bound; in a closed-loop replay of a multi-turn agentic trace (aggregated,
    #: interval 12) it raised p90 TTFT several-fold, because under load it degrades to arrival
    #: order. 16 and 32 were unstable across replay variants; 64 kept shortest-first latencies.
    DEFAULT_AGING_FACTOR = 64

    def __init__(self, aging_factor: int = DEFAULT_AGING_FACTOR, release_aged: bool = True):
        if aging_factor < 0:
            raise ValueError(f"aging_factor must be non-negative, got {aging_factor}")
        self._aging_factor = aging_factor
        self._release_aged = release_aged
        self._work = 0
        self._arrival_work: dict[int, int] = {}
        self._key: dict[int, int] = {}
        # Requests long-prefill demotion let through for good.
        self._released: set[int] = set()

    @property
    def aging_factor(self) -> int:
        """F: the multiple of its own uncached work after which a request is no longer overtaken."""
        return self._aging_factor

    @property
    def work(self) -> int:
        """Context tokens the scheduler has admitted so far: the order's clock."""
        return self._work

    def describe(self) -> str:
        """The order, for the scheduler's startup log."""
        return (
            f"shortest prefill first, aged by admitted context tokens (factor {self._aging_factor})"
        )

    def describe_release(self) -> str:
        """How requests demotion lets through for good are ranked, for the startup log."""
        if self._release_aged:
            return "released requests fully aged"
        return "released requests keep their key"

    def _uncached_tokens(self, req: LlmRequest, kv_cache_manager) -> int:
        if not req.is_first_context_chunk:
            return req.context_remaining_length
        # `is True` / isinstance: the demotion fields of a request it never probed.
        if req.py_uncached_prefill_probed is True and isinstance(
            req.py_uncached_prefill_tokens, int
        ):
            return max(req.py_uncached_prefill_tokens, 1)
        reusable = kv_cache_manager.probe_context_reuse(req)
        if reusable is None:
            return req.context_remaining_length
        return max(req.prompt_len - reusable, 1)

    def _key_of(self, req: LlmRequest, kv_cache_manager) -> int:
        request_id = req.py_request_id
        key = self._key.get(request_id)
        if key is None:
            key = self._arrival_work[request_id]
            if self._aging_factor and request_id not in self._released:
                key += self._aging_factor * self._uncached_tokens(req, kv_cache_manager)
            self._key[request_id] = key
        return key

    def uncached_estimate(self, request_id: int) -> Optional[int]:
        """The work this order ranked *request_id* by, None before it was keyed."""
        key = self._key.get(request_id)
        if key is None or not self._aging_factor or request_id in self._released:
            return None
        return (key - self._arrival_work[request_id]) // self._aging_factor

    @property
    def release_aged(self) -> bool:
        """Whether a request demotion lets through for good counts as fully aged."""
        return self._release_aged

    def release(self, request_id: int) -> None:
        """Count a request long-prefill demotion let through for good as fully aged."""
        if not self._release_aged:
            return
        self._released.add(request_id)
        self._key.pop(request_id, None)

    def forget_work(self, request_ids) -> None:
        """Probe these requests again on their next pass, keeping their clock reading.

        For requests that wait for a prefix another request is prefilling: their
        probe predates that prefix.
        """
        for request_id in request_ids:
            self._key.pop(request_id, None)

    def order(
        self,
        pending_ctx: RequestList,
        active_requests: RequestList,
        context_init_state_value: int,
        kv_cache_manager,
        demoted_ids: frozenset = frozenset(),
    ) -> RequestList:
        """Account one scheduling pass and return *pending_ctx* in admission order.

        *pending_ctx* holds the requests this pass may try; *active_requests*
        all active requests, whose context requests take their clock reading on
        the first pass they are seen, even when this pass does not reach them.
        Requests in other states (encoder init) keep their relative order ahead
        of the context requests. Requests in *demoted_ids* go after the others.
        """
        context_ids = {
            r.py_request_id for r in active_requests if r.state_value == context_init_state_value
        }
        for request_id in self._arrival_work.keys() - context_ids:
            del self._arrival_work[request_id]
            self._key.pop(request_id, None)
            self._released.discard(request_id)
        for request_id in context_ids - self._arrival_work.keys():
            self._arrival_work[request_id] = self._work
        context_reqs = [r for r in pending_ctx if r.state_value == context_init_state_value]
        if len(context_reqs) < 2:
            return pending_ctx
        # sorted() is stable, so equal keys keep arrival order.
        ordered = sorted(
            context_reqs,
            key=lambda req: (
                req.py_request_id in demoted_ids,
                self._key_of(req, kv_cache_manager),
            ),
        )
        others = [r for r in pending_ctx if r.state_value != context_init_state_value]
        return others + ordered

    def record_admitted(self, context_tokens: int) -> None:
        """Advance the clock by the context tokens of the requests this pass returned."""
        self._work += context_tokens


class _ContextAdmissionLog:
    """Opt-in (CTX_ADMIT_LOG_ENV) per-request context admission line, for splitting TTFT.

    One INFO line per context request, in the pass that schedules its last chunk:

        [V2Scheduler] ctx_admit id=<py_request_id> uncached=<tokens> bin=<class>
        wait_s=<s> prefill_s=<s> chunks=<n> demoted=<passes> released=<0|1> sjf_work=<tokens|->

    ``uncached`` is the request's remaining context when its first chunk was
    scheduled, after its prefix was matched: the tokens it computes. ``wait_s``
    runs from the first pass that saw the request in the context state on this
    rank to the pass that scheduled its first chunk; ``prefill_s`` from there to
    the pass that scheduled its last chunk (that chunk's forward is not
    included). Time spent before the request reached this rank's active
    requests (the attention-DP waiting queue) is in neither. ``demoted`` counts
    the passes long-prefill demotion held it back; ``sjf_work`` is the work
    ShortestPrefillFirstOrder ranked it by. ``bin`` is the uncached-token size
    class from ``_BINS``.
    """

    _BINS = (
        (2048, "lt2k"),
        (8192, "2k-8k"),
        (16384, "8k-16k"),
        (32768, "16k-32k"),
        (131072, "32k-131k"),
    )

    def __init__(self):
        self._seen: dict[int, float] = {}
        # request id -> [first chunk time, uncached tokens, chunks]
        self._started: dict[int, list] = {}
        self._demoted: Counter = Counter()
        # Requests long-prefill demotion let through for good.
        self._released: set[int] = set()

    @classmethod
    def _bin(cls, tokens: int) -> str:
        for bound, name in cls._BINS:
            if tokens < bound:
                return name
        return "ge131k"

    def record_pass(
        self,
        requests_list: RequestList,
        context_init_state_value: int,
        scheduled_ctx: RequestList,
        demoted_ids: set,
        released_ids: set,
        order: Optional[ShortestPrefillFirstOrder],
    ) -> None:
        now = time.monotonic()
        context_ids = set()
        for req in requests_list:
            if req.state_value == context_init_state_value:
                context_ids.add(req.py_request_id)
                self._seen.setdefault(req.py_request_id, now)
        for request_id in demoted_ids:
            self._demoted[request_id] += 1
        self._released.update(released_ids)
        for req in scheduled_ctx:
            request_id = req.py_request_id
            started = self._started.get(request_id)
            if started is None:
                started = [now, req.context_remaining_length, 0]
                self._started[request_id] = started
            started[2] += 1
            if not req.is_last_context_chunk:
                continue
            seen = self._seen.get(request_id, started[0])
            work = order.uncached_estimate(request_id) if order is not None else None
            logger.info(
                f"[V2Scheduler] ctx_admit id={request_id} uncached={started[1]} "
                f"bin={self._bin(started[1])} wait_s={started[0] - seen:.3f} "
                f"prefill_s={now - started[0]:.3f} chunks={started[2]} "
                f"demoted={self._demoted.get(request_id, 0)} "
                f"released={int(request_id in self._released)} "
                f"sjf_work={'-' if work is None else work}"
            )
            self._seen.pop(request_id, None)
            del self._started[request_id]
            self._demoted.pop(request_id, None)
            self._released.discard(request_id)
        # Drop requests that left the context state some other way (cancelled, paused).
        for book in (self._seen, self._started, self._demoted):
            for request_id in [rid for rid in book if rid not in context_ids]:
                del book[request_id]
        self._released &= context_ids


class KVCacheV2Scheduler(RequestScheduler):
    """Interleaved scheduler for KV Cache Manager V2.

    Merges capacity checking (KV cache allocation via resize()) and token budget
    assignment into one unified loop.
    """

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
        context_chunk_cap: Optional[ContextChunkCapConfig] = None,
        equal_progress: bool = False,
    ) -> None:
        self.remote_tail_phase: RemoteTailPhase | None = None
        self._remote_tail_context_ids: frozenset[int] = frozenset()
        self._remote_tail_generation_ids: frozenset[int] = frozenset()
        self.max_num_tokens = max_num_tokens
        self._stalled_schedules = 0
        self.max_num_requests = (
            scheduler_capacity if scheduler_capacity is not None else max_batch_size
        )
        from ..kv_cache import kv_cache_manager_v2
        from ..kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

        assert isinstance(kv_cache_manager, KVCacheManagerV2), (
            f"KVCacheV2Scheduler requires KVCacheManagerV2, got {type(kv_cache_manager).__name__}"
        )
        self.kv_cache_manager = kv_cache_manager
        from ..kv_cache.dkv import Dkv

        # The manager stores KV once per DKV group (KVCacheManagerV2.dkv), so every rank of the
        # group must take the same scheduling decisions; fixed at the manager's construction.
        self._dkv = isinstance(getattr(kv_cache_manager, "dkv", None), Dkv)
        # TRTLLM_KV_V2_WAIT_FOR_INFLIGHT_PREFIX: ask the manager which new
        # context requests wait for a prefix another request computes.
        self._wait_for_inflight_prefix = kv_cache_manager_v2._WAIT_FOR_INFLIGHT_PREFIX
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
        # Every input to the prefix-aware skip is fixed for this scheduler's
        # lifetime: the knob is a constructor argument, and both manager
        # attributes are assigned once in ``KVCacheManagerV2.__init__``. Resolve
        # the gate here rather than re-deriving it on every schedule_request.
        # Not under DKV: the skip keys on this rank's radix tree, which can differ between the
        # ranks of the group, and every rank must defer the same requests.
        self._prefix_skip_enabled = (
            enable_prefix_aware_scheduling
            and kv_cache_manager.enable_block_reuse
            and self._skip_pays_off_under_reuse_policy()
            and not self._dkv
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
            f"enable_recompute_pause={enable_recompute_pause}"
        )
        if ctx_chunk_config is not None:
            self.chunking_enabled = True
            self.chunking_policy = ctx_chunk_config[0]
            self.chunk_unit_size = ctx_chunk_config[1]

        # Per-request chunk cap (scheduler_config.context_chunk_cap): set per pass, on while at
        # least the configured number of context requests wait (see _schedule_loop).
        self._ctx_req_chunk_cap = (
            context_chunk_cap.max_tokens_per_request
            if context_chunk_cap is not None and self.chunking_enabled
            else 0
        )
        self._ctx_req_chunk_cap_min_waiting = (
            context_chunk_cap.min_waiting_requests if context_chunk_cap is not None else 0
        )
        self._ctx_req_chunk_cap_active = False
        if self._ctx_req_chunk_cap:
            logger.info(
                f"KVCacheV2Scheduler: context chunks capped at {self._ctx_req_chunk_cap} tokens "
                f"per request while {self._ctx_req_chunk_cap_min_waiting} or more context "
                f"requests wait"
            )
        # EQUAL_PROGRESS splits each pass's context budget by water level (_EqualProgress).
        # ``equal_progress`` keeps it when a forced chunk policy replaced the configured one.
        self._equal_progress = self.chunking_enabled and (
            equal_progress or self.chunking_policy == ContextChunkingPolicy.EQUAL_PROGRESS
        )
        self._ctx_equal_progress: Optional[_EqualProgress] = None
        if self._equal_progress:
            logger.info(
                f"KVCacheV2Scheduler: context chunks split the step budget by equal progress "
                f"(chunk policy {self.chunking_policy})"
            )

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
        self._encoder_init_state_value = LlmRequestState.ENCODER_INIT.value
        self._disagg_gen_init_state_value = LlmRequestState.DISAGG_GENERATION_INIT.value
        self._disagg_gen_trans_complete_state_value = (
            LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE.value
        )
        self._gen_to_complete_state_value = LlmRequestState.GENERATION_TO_COMPLETE.value

        # Opt-in (default off): extra receive slots on a disagg generation
        # server (TRTLLM_DISAGG_GEN_EXTRA_SLOTS). The executor may then hold
        # more active requests than max_num_requests, the surplus still
        # receiving KV. The decode batch stays within max_num_requests: a
        # request whose KV has landed takes a decode seat only while fewer
        # than max_num_requests active requests outside the receive phase hold
        # one, so it never displaces a decoding request and the sequence-slot
        # pool is never exceeded (see _schedule_loop).
        self.gen_extra_slots = disagg_gen_extra_slots()
        self._receive_state_values = frozenset(
            state.value
            for state in (
                LlmRequestState.DISAGG_GENERATION_INIT,
                LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
                LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE,
            )
        )
        if self.gen_extra_slots:
            logger.info(
                f"KVCacheV2Scheduler: disagg generation extra receive slots "
                f"{self.gen_extra_slots}; decode seats {self.max_num_requests}"
            )

        # Opt-in (default off): admit a disagg gen-init request only while the
        # pool keeps this many tokens of decode growth for every resident
        # generation request (see _GenInitDecodeHeadroom). Off under helix,
        # whose ledger blocks span the CP ranks.
        self._gen_init_decode_headroom_tokens = (
            0
            if self.has_cp_helix
            else max(0, int(os.environ.get(GEN_INIT_DECODE_HEADROOM_ENV, "0") or 0))
        )
        if self._gen_init_decode_headroom_tokens:
            logger.info(
                f"KVCacheV2Scheduler: disagg gen-init admission keeps "
                f"{self._gen_init_decode_headroom_tokens} tokens of decode headroom "
                f"per resident generation request"
            )

        # Opt-in (default off): on the disagg generation server, schedule
        # just-arrived generation-only requests (first token not yet emitted)
        # ahead of older in-flight decodes to reduce TTFT under budget
        # pressure. See _schedule_loop.
        self._prioritize_first_token_gen = (
            os.environ.get("TLLM_DISAGG_GEN_PRIORITIZE_FIRST_TOKEN", "0") == "1"
        )

        # Opt-in (default off): a generation request that cannot get pages no
        # longer ends the pass. The pass goes on, without eviction, for the
        # generation requests behind it, which may still have room in their
        # last block; context and disagg gen-init requests behind it wait, as
        # they would after the stop. Without a cache tier below GPU a suspended
        # page stays where it is, so the blocked request skips the eviction
        # ladder too: suspending the requests behind it would free nothing and
        # only drop them from the pass.
        self._skip_blocked_generation = os.environ.get("TLLM_V2_GEN_SKIP_BLOCKED", "0") == "1"

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
        demote_uncached = int(os.environ.get("TLLM_V2_CTX_DEMOTE_UNCACHED_TOKENS", "0"))
        self._ctx_demote_uncached_tokens = max(0, demote_uncached) if self.chunking_enabled else 0
        # A request demoted this many iterations is let through for good: it
        # runs in arrival order to the end of its prefill (0 = no bound).
        self._ctx_demote_max_iters = max(
            0, int(os.environ.get("TLLM_V2_CTX_DEMOTE_MAX_ITERS", "0"))
        )
        self._ctx_demote_skips: dict[int, int] = {}
        # Opt-in (default off): classify by the prefix cache, not by the router's
        # estimate alone. The router only knows the conversation's previous prompt
        # length on this rank, so it calls a forked sub-agent or a new conversation
        # over a cached shared prefix long, and a continuing turn whose prefix was
        # evicted short (which then takes the whole budget for every chunk). With
        # this on, an estimate above the threshold is checked once against the
        # radix tree (read-only probe, one walk per request), a request whose
        # cache this pool already holds is judged by its exact remainder, and a
        # request ordered as short whose first prepare shows a long remainder
        # goes behind the rest of the iteration.
        self._ctx_demote_exact = (
            self._ctx_demote_uncached_tokens > 0
            and os.environ.get("TLLM_V2_CTX_DEMOTE_EXACT", "0") == "1"
        )
        # First-chunk requests ordered as short on the estimate alone this iteration.
        self._ctx_unverified_short: set[int] = set()
        # Opt-in (default off): a first context chunk that is not scheduled keeps
        # its cache claim unless admission ran out of KV pages. By default every
        # unscheduled first chunk releases its claim (release_context_admission),
        # so a request that waits for token budget, or is deferred to the tail of
        # the iteration, matches its prefix again on its next pass.
        self._keep_unscheduled_first_chunk_claim = (
            os.environ.get(KEEP_UNSCHEDULED_FIRST_CHUNK_CLAIM_ENV, "0") == "1"
        )
        # Set by the context admission paths when the request found no KV pages.
        self._admission_out_of_pages = False
        # Unscheduled first chunks that held a cache claim, keyed by
        # (schedule action, out of pages, released); logged at INFO.
        self._unscheduled_first_chunk_claims: Counter = Counter()
        self._unscheduled_first_chunk_claims_logged_at = 0.0
        if self._keep_unscheduled_first_chunk_claim:
            logger.info(
                "KVCacheV2Scheduler: unscheduled first context chunks keep their cache "
                "claim unless admission ran out of KV pages"
            )
        if self._ctx_demote_uncached_tokens:
            logger.info(
                f"KVCacheV2Scheduler: demoting context requests with more than "
                f"{self._ctx_demote_uncached_tokens} uncached tokens left "
                f"(max iterations demoted: {self._ctx_demote_max_iters or 'unbounded'}, "
                f"exact: {self._ctx_demote_exact})"
            )

        # Opt-in (default off): try pending context shortest prefill first; with
        # long-prefill demotion on too, within each demotion class (see
        # ShortestPrefillFirstOrder). Unset, context is tried in arrival order.
        self._shortest_prefill_first = None
        if os.environ.get(SHORTEST_PREFILL_FIRST_ENV, "0") == "1":
            self._shortest_prefill_first = ShortestPrefillFirstOrder(
                _read_non_negative_int_env(
                    _AGING_FACTOR_ENV, ShortestPrefillFirstOrder.DEFAULT_AGING_FACTOR
                ),
                release_aged=os.environ.get(SPF_RELEASE_KEEPS_KEY_ENV, "0") != "1",
            )
            logger.info(
                "KVCacheV2Scheduler context admission: "
                f"{self._shortest_prefill_first.describe()}, "
                + (
                    "within each long-prefill demotion class, "
                    f"{self._shortest_prefill_first.describe_release()}"
                    if self._ctx_demote_uncached_tokens
                    else "no long-prefill demotion"
                )
            )
        else:
            for name in (_AGING_FACTOR_ENV, SPF_RELEASE_KEEPS_KEY_ENV):
                if os.environ.get(name, "").strip():
                    logger.warning(
                        f"{name} is ignored: it applies only with {SHORTEST_PREFILL_FIRST_ENV}=1"
                    )
        # Opt-in (default off): one log line per context request (see _ContextAdmissionLog).
        self._ctx_admission_log = (
            _ContextAdmissionLog() if os.environ.get(CTX_ADMIT_LOG_ENV, "0") == "1" else None
        )
        if self._ctx_admission_log is not None:
            logger.info("KVCacheV2Scheduler: per-request context admission log on")
        # Requests long-prefill demotion let through for good in this pass.
        self._ctx_demote_released: set[int] = set()

        # Registered by PyExecutor; see set_async_transfer_manager.
        self._async_transfer_manager = None

        # `is True`: a manager without the flag reads as off.
        self._swa_scratch_reuse = (
            getattr(kv_cache_manager, "enable_swa_scratch_reuse", False) is True
        )

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

    def _estimated_context_tokens(self, req: LlmRequest) -> int:
        """Prompt tokens a context request still computes, for the equal-progress split.

        Exact once the request holds its cache, up to its next forced chunk end. A first chunk
        without one subtracts the reuse it will claim where that is known without a radix walk
        (the DKV group's agreed depth), else takes the router's uncached estimate.
        """
        remaining = req.context_remaining_length
        if self.chunking_policy == ContextChunkingPolicy.FORCE_CHUNK:
            # This step's demand ends at the next snapshot point.
            remaining = min(remaining, max(1, _get_forced_context_chunk_size(req)))
        if not req.is_first_context_chunk or self._holds_context_cache(req):
            return remaining
        reused = self.kv_cache_manager.probe_context_reuse(req) if self._dkv else None
        if reused is not None:
            return max(1, min(remaining, req.prompt_len - reused))
        estimate = req.py_uncached_prefill_tokens
        return remaining if estimate is None else max(1, min(remaining, estimate))

    def _context_chunk_limit(self, budget: Optional[int], need: int) -> int | None:
        """The most prompt tokens a context request may compute this step, besides the token
        budget: the per-request cap while it is on and the equal-progress share of ``budget`` for
        a request that could take ``need`` tokens. At least one chunk unit, so a request always
        progresses; None when neither applies."""
        limit = None
        if self._ctx_req_chunk_cap_active:
            limit = self._ctx_req_chunk_cap
        if self._ctx_equal_progress is not None and budget is not None:
            share = self._ctx_equal_progress.level(budget, need)
            limit = share if limit is None else min(limit, share)
        if limit is None:
            return None
        return max(limit, self.chunk_unit_size)

    def _dkv_unpin(self, req: LlmRequest) -> bool:
        """Under DKV, give back the step's reuse pins of the other requests when ``req`` found no
        pages (KVCacheManagerV2.unpin_dkv_context_reuse); True when the caller retries once."""
        return self._dkv and self.kv_cache_manager.unpin_dkv_context_reuse(req.py_request_id)

    def set_async_transfer_manager(self, mgr) -> None:
        """Register the AsyncTransferManager the deadlock detector consults.

        A finished disaggregated context sender leaves active_requests once its
        response is emitted, but the transfer manager still owns its pinned KV
        pages until the send lands. Without this reference the detector cannot
        see those pages and reports a false deadlock.
        """
        self._async_transfer_manager = mgr

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
        active_requests = drop_decoder_context_requests_waiting_for_encoder_output(active_requests)
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

    def _filter_remote_tail_phase_requests(self, requests: RequestList) -> RequestList:
        phase = self.remote_tail_phase
        if phase is None:
            return requests
        eligible = []
        for req in requests:
            if req.state_value in (self._context_init_state_value, self._encoder_init_state_value):
                if (
                    phase is RemoteTailPhase.CONTEXT
                    and req.request_id in self._remote_tail_context_ids
                ):
                    eligible.append(req)
            elif self.is_request_in_schedulable_state(req):
                if (
                    phase is RemoteTailPhase.DECODE
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
        self, active_requests, inflight_request_ids, *, defer_remote_tail_context=False
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
        exclusive_remote_mode = next(
            (
                getattr(req, "py_csa2_remote_tail_mode", None)
                for req in requests_list
                if not defer_remote_tail_context
                and self.remote_tail_phase is not RemoteTailPhase.DECODE
                and req.request_id not in inflight_request_ids
                and req.state_value == self._context_init_state_value
                and getattr(req, "py_csa2_remote_tail_mode", None) is not None
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

        req_it_end = len(requests_list)
        recompute_pause_state = _RecomputePauseState(req_it_end)
        req_it = 0

        # Extra receive slots (see __init__): the decode seats left for
        # requests whose KV has landed. Every active request outside the
        # receive phase holds a seat, including one that finished but still
        # holds its sequence slot. None when the option is off.
        seats_free = None
        if self.gen_extra_slots:
            seats_free = self.max_num_requests - sum(
                1 for r in active_requests if r.state_value not in self._receive_state_values
            )
        # Received requests left waiting for a decode seat in this pass.
        seat_waiting_ids: set[int] = set()

        # Set once a generation request could not get pages (opt-in
        # TLLM_V2_GEN_SKIP_BLOCKED): from then on only generation requests that
        # need no eviction are scheduled in this pass.
        pages_exhausted = False

        # Context requests are always deferred to a second phase so that
        # generation requests are fully accounted for in the budget before
        # any context request competes for resources.  This mirrors the
        # two-phase approach of the old KVCacheV2DummyScheduler and
        # prevents PEFT adapter eviction failures when gen requests hold
        # adapters that can't be evicted mid-iteration.
        pending_ctx: RequestList = []

        decode_headroom = (
            _GenInitDecodeHeadroom(
                self.kv_cache_manager, requests_list, self._gen_init_decode_headroom_tokens
            )
            if self._gen_init_decode_headroom_tokens
            and any(req.state_value == self._disagg_gen_init_state_value for req in requests_list)
            else None
        )

        # Pre-claim PEFT pages for GENERATION_TO_COMPLETE requests.
        # In the overlap executor these requests are no longer scheduled
        # (state outside schedulable range) but their adapters haven't
        # been released yet (mark_request_done runs after prepare_resources
        # in the next iteration).  Without this, the budget would appear
        # empty and context requests with a different adapter could be
        # admitted, causing ensure_batch to crash when it can't evict the
        # still-active adapter.
        for req in requests_list:
            if req.state_value == self._gen_to_complete_state_value:
                budget.pre_claim_peft(req)

        # --- Phase 1: generation / disagg only ---
        while req_it < req_it_end:
            req = requests_list[req_it]

            if req_it in recompute_pause_state.victim_indices:
                req_it += 1
                continue

            # --- Filter ---
            if req.request_id in inflight_request_ids:
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
                if pages_exhausted:
                    req_it += 1
                    continue
                if seats_free is not None and len(disagg_candidates) >= self.max_num_requests:
                    # The executor prepares a pass's gen-init requests as one
                    # batch, whose per-request state rows are sized like a
                    # forward batch: with extra receive slots, more of them
                    # can be waiting, so the rest start in the next pass.
                    req_it += 1
                    continue
                peft_pages = budget.peft_pages_needed(req)
                if peft_pages is None:
                    break

                headroom_blocks = None
                if decode_headroom is not None:
                    headroom_blocks = decode_headroom.admits(req)
                    if headroom_blocks is None:
                        logger.debug(
                            f"[V2Scheduler] Deferring disagg gen-init request "
                            f"{req.py_request_id}: no decode headroom "
                            f"({decode_headroom.reserved_blocks} blocks reserved)"
                        )
                        req_it += 1
                        continue
                action, tokens = self._try_schedule_disagg_gen_init(req, budget)
                if action is ScheduleAction.STOP:
                    break
                if action is ScheduleAction.SKIP:
                    req_it += 1
                    continue
                if headroom_blocks is not None:
                    decode_headroom.commit(headroom_blocks)
                disagg_candidates.append(req)
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
                if seats_free is None:
                    break
                # With extra receive slots the list can be longer than the
                # decode batch: keep walking so that gen-init requests behind
                # a full batch still get their KV cache and start receiving.
                req_it += 1
                continue

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
                if not pages_exhausted:
                    pending_ctx.append(req)
                req_it += 1
                continue

            else:
                if exclusive_remote_mode is not None:
                    # Replay-only batches harvest the preceding decode on CPU
                    # without launching its next step. On resume, use the host
                    # token count rather than the one-step-behind overlap path.
                    self._clear_request_runtime_state(req)
                    req_it += 1
                    continue
                takes_seat = (
                    seats_free is not None
                    and req_state_value == self._disagg_gen_trans_complete_state_value
                )
                if takes_seat and seats_free <= 0:
                    # Its KV is in, but every decode seat is held: it waits,
                    # without allocating, until a seat holder is released.
                    seat_waiting_ids.add(req.request_id)
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
                    allow_eviction=not pages_exhausted,
                )
                if action is ScheduleAction.BLOCKED:
                    if not self._skip_blocked_generation:
                        break
                    pages_exhausted = True
                    req_it += 1
                    continue
                if action is ScheduleAction.STOP:
                    break
                if action is ScheduleAction.SKIP:
                    req_it += 1
                    continue
                scheduled_gen.append(req)
                budget.commit(req, tokens, peft_pages)
                if takes_seat:
                    seats_free -= 1

            req_it += 1

        # Requests whose pages were given up during this pass. A victim that is
        # itself a started context request still sits in pending_ctx, so
        # re-admitting it would spend the pages its own preemption released.
        preempted_ids: set[int] = set()

        # Under DKV the group agrees on the reuse depth of every new context request at once
        # (one collective) before anything reads it: demotion, prefix waits and admission all
        # see the agreed depth (probe_context_reuse), the same on every rank.
        new_ctx = [req for req in pending_ctx if req.state_value == self._context_init_state_value]
        if self._dkv:
            self.kv_cache_manager.agree_dkv_context_reuse(new_ctx, self._joint_draft_manager)
        # Per-request chunk cap (scheduler_config.context_chunk_cap): on while enough context
        # requests wait to fill the rest of the budget, so a lone request keeps the whole step.
        self._ctx_req_chunk_cap_active = (
            self._ctx_req_chunk_cap > 0 and len(pending_ctx) >= self._ctx_req_chunk_cap_min_waiting
        )

        # Long-prefill demotion (see __init__). Reorder here rather than on
        # requests_list: phase 1 walks that list by index, pending_ctx is this
        # loop's own.
        arrival_ctx = pending_ctx
        demoted_ids: set[int] = set()
        if self._ctx_demote_uncached_tokens:
            pending_ctx, demoted_ids = self._defer_long_context(pending_ctx)
        spf = self._shortest_prefill_first
        if spf is not None:
            pending_ctx = spf.order(
                pending_ctx,
                requests_list,
                self._context_init_state_value,
                self.kv_cache_manager,
                demoted_ids,
            )

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
            # Demotion reorders compute, not page priority: a demoted request
            # that arrived before `req` would have been scheduled, and so
            # protected, ahead of it in FCFS order.
            if demoted_ids:
                for other in arrival_ctx:
                    if other is req:
                        break
                    if other.py_request_id in demoted_ids:
                        protected.add(other.py_request_id)
            # Shortest prefill first reorders compute the same way: every request
            # that arrived before `req`, and every started prefill when `req` has
            # not started, would have been tried, and protected, ahead of it.
            if spf is not None:
                for other in arrival_ctx:
                    if other is req:
                        break
                    protected.add(other.py_request_id)
                if req.is_first_context_chunk:
                    protected.update(self._started_context_ids(requests_list))
            return self._try_preempt_for_pages(
                requests_list, protected, inflight_request_ids, recompute_paused, preempted_ids
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
        # A deferral behind an in-flight contributor leaves nothing on any of
        # the scheduled lists, so the deadlock detector below would read the
        # iteration as a stall even though the contributor is running. Deferring
        # is itself the progress in that case.
        deferred_behind_contributor = False

        # After any reordering: pre-grown requests go first so no other request
        # takes the budget their replay needs (stable sort).
        if self._swa_scratch_reuse:
            pending_ctx = sorted(
                pending_ctx,
                key=lambda r: not (
                    r.state_value == self._context_init_state_value
                    and self._pregrown_context_tokens(r) > 0
                ),
            )

        # New context requests that wait this pass for a prefix another request
        # is prefilling (see KVCacheManagerV2.plan_context_waits).
        waiting_ids = (
            self.kv_cache_manager.plan_context_waits(pending_ctx)
            if self._wait_for_inflight_prefix
            else ()
        )
        if self._ctx_demote_exact and waiting_ids:
            # The probe of a waiting request predates the prefix it waits for;
            # it is probed again once it stops waiting.
            for req in pending_ctx:
                if req.py_request_id in waiting_ids:
                    req.py_uncached_prefill_probed = False
        if spf is not None and waiting_ids:
            spf.forget_work(waiting_ids)

        # Under DKV, hold the agreed prefixes of the new requests this step can admit (the first
        # free request slots, waiters left out) until their claims; the others claim next step.
        if self._dkv:
            free_slots = max(0, budget.max_num_requests - budget.num_requests)
            self.kv_cache_manager.pin_dkv_context_reuse(
                [
                    req
                    for req in pending_ctx
                    if req.state_value == self._context_init_state_value
                    and req.py_request_id not in waiting_ids
                ][:free_slots],
                self._joint_draft_manager,
            )
        self._ctx_equal_progress = (
            _EqualProgress(
                {
                    req.py_request_id: self._estimated_context_tokens(req)
                    for req in pending_ctx
                    if req.state_value == self._context_init_state_value
                    and req.py_request_id not in waiting_ids
                }
            )
            if self._equal_progress
            else None
        )
        # Indexed over a copy: a DEFER re-appends its request to the tail.
        ctx_order = list(pending_ctx)
        ctx_idx = 0
        while ctx_idx < len(ctx_order):
            req = ctx_order[ctx_idx]
            ctx_idx += 1
            if self._ctx_equal_progress is not None:
                self._ctx_equal_progress.offered(req.py_request_id)
            if budget.requests_full:
                break
            if req.py_request_id in waiting_ids:
                continue
            if self._dkv and not self.kv_cache_manager.dkv_claim_ready(req):
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
            remote_mode = getattr(req, "py_csa2_remote_tail_mode", None)
            if defer_remote_tail_context and remote_mode is not None:
                continue
            if exclusive_remote_mode is not None and remote_mode != exclusive_remote_mode:
                continue
            # Destination replay is atomic; source prefix processing may be chunked.
            if remote_mode == "destination" and not budget.can_fit_tokens(
                req.context_remaining_length
            ):
                continue
            if scheduled_ctx and remote_mode != scheduled_remote_mode:
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
                    break
                scheduled_encoder.append(req)
                budget.commit(req, tokens, peft_pages)
            else:
                action, tokens, chunking_flag = self._try_schedule_context(
                    req, budget, preempt_for_pages
                )
                if action is ScheduleAction.STOP:
                    break
                if action is ScheduleAction.SKIP:
                    continue
                if action is ScheduleAction.DEFER:
                    # Ordered as short, prepared, found long: it goes behind the
                    # rest of this iteration and counts as demoted for aging.
                    demoted_ids.add(req.py_request_id)
                    ctx_order.append(req)
                    continue
                has_chunking = has_chunking or chunking_flag
                scheduled_ctx.append(req)
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
                if first_new_block is not None:
                    contributed_blocks.add(first_new_block)
        if self._dkv:
            self.kv_cache_manager.release_dkv_reuse_pins()
        self._ctx_equal_progress = None

        made_progress = bool(
            scheduled_gen
            or scheduled_ctx
            or scheduled_encoder
            or disagg_candidates
            or evicted
            or recompute_paused
            or deferred_behind_contributor
        )
        if (
            self.remote_tail_phase is None
            and exclusive_remote_mode is not None
            and not made_progress
        ):
            # A non-ADP replay that cannot acquire KV must let decode release
            # pages. Retry admission without replay; keep its forward isolated.
            return self._schedule_loop(
                requests_list, inflight_request_ids, defer_remote_tail_context=True
            )

        if self._ctx_demote_uncached_tokens:
            # Iterations spent demoted. Rebuilt every iteration from the
            # requests demoted in it, so it cannot leak.
            self._ctx_demote_skips = {
                rid: self._ctx_demote_skips.get(rid, 0) + 1 for rid in demoted_ids
            }
        if spf is not None:
            spf.record_admitted(
                sum(
                    req.context_chunk_size
                    if self.chunking_enabled
                    else req.context_remaining_length
                    for req in scheduled_ctx
                )
            )
        if self._ctx_admission_log is not None:
            self._ctx_admission_log.record_pass(
                requests_list,
                self._context_init_state_value,
                scheduled_ctx,
                demoted_ids,
                self._ctx_demote_released,
                spf,
            )
            self._ctx_demote_released = set()

        # Phase and frozen-membership deferrals are not KV capacity stalls,
        # nor is a received request waiting for a decode seat.
        self._detect_deadlock(
            requests_list,
            inflight_request_ids,
            pending_ctx,
            preempted_ids,
            made_progress=made_progress,
            seat_waiting_ids=seat_waiting_ids,
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
        request without an estimate (none published, or dropped by the aging
        bound) is never long. No radix probe here unless
        TLLM_V2_CTX_DEMOTE_EXACT is on (then at most one per request): it would
        walk the whole prompt for every request on every iteration."""
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

    def _started_context_ids(self, requests: RequestList) -> set[int]:
        """Ids of the context requests in *requests* that have started prefilling."""
        return {
            r.py_request_id
            for r in requests
            if r.state_value == self._context_init_state_value and not r.is_first_context_chunk
        }

    def _holds_context_cache(self, req: LlmRequest) -> bool:
        """Whether prepare_context already ran for this request: its reuse is claimed,
        so context_remaining_length is the exact uncached remainder."""
        return req.py_request_id in self.kv_cache_manager.kv_cache_map

    def _defer_long_context(self, pending_ctx: RequestList) -> tuple[RequestList, set[int]]:
        """Move long requests to the tail of this iteration's context order.

        Returns the new order and the ids demoted. In exact mode also records
        which requests were kept on the estimate alone (see
        _try_schedule_context_chunked).
        """
        keep: RequestList = []
        deferred: RequestList = []
        self._ctx_unverified_short = set()
        for req in pending_ctx:
            if self._is_long_context(req):
                aged_out = (
                    self._ctx_demote_max_iters > 0
                    and self._ctx_demote_skips.get(req.py_request_id, 0)
                    >= self._ctx_demote_max_iters
                )
                if not aged_out:
                    deferred.append(req)
                    continue
                # Let through for good: with no estimate the request is never
                # long again, so it keeps its arrival order to the end of its
                # prefill. Demoted again, a started prefill would hold its KV
                # blocks without progress; a chunk that finds no pages is
                # skipped, not suspended, so enough of them fill the pool and
                # no context request can allocate.
                req.py_uncached_prefill_tokens = None
                if self._ctx_admission_log is not None:
                    self._ctx_demote_released.add(req.py_request_id)
                if self._shortest_prefill_first is not None:
                    self._shortest_prefill_first.release(req.py_request_id)
            elif (
                self._ctx_demote_exact
                and req.py_uncached_prefill_tokens is not None
                and req.context_current_position == 0
                and not self._holds_context_cache(req)
            ):
                self._ctx_unverified_short.add(req.py_request_id)
            keep.append(req)
        return keep + deferred, {req.py_request_id for req in deferred}

    # ---- Prefix-aware skip ----

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
        # Cache-transceiver mode disables the separate one-model draft manager,
        # so disagg generation init has no paired-reuse path. Supporting one
        # would also require draft KV transfer and history_length=prompt_len.
        if not self.kv_cache_manager.prepare_disagg_gen_init(req):
            logger.debug("prepare_disagg_gen_init failed for request %s", req.py_request_id)
            return ScheduleAction.SKIP, 0
        return ScheduleAction.SCHEDULED, 0

    def _try_schedule_context(
        self,
        req: LlmRequest,
        budget: BudgetTracker,
        preempt_for_pages: Callable[[LlmRequest], bool],
    ) -> tuple[ScheduleAction, int, bool]:
        """Try to schedule a context request (chunked or non-chunked).

        Returns ``(action, tokens, chunking_flag)``.  *tokens* and
        *chunking_flag* are meaningful only when *action* is ``SCHEDULED``.

        Source reservations advance the prefix before budget checks. Destination
        pages are allocated here; loads are accepted after the final batch trims.
        A loading request remains outside the schedulable state range.
        """
        first_chunk = req.is_first_context_chunk
        self._admission_out_of_pages = False
        if self.chunking_enabled:
            result = self._try_schedule_context_chunked(req, budget, preempt_for_pages)
        else:
            result = self._try_schedule_context_full(req, budget, preempt_for_pages)

        if first_chunk and result[0] is not ScheduleAction.SCHEDULED:
            release = self._admission_out_of_pages or not self._keep_unscheduled_first_chunk_claim
            if self._holds_context_cache(req):
                self._count_unscheduled_first_chunk_claim(result[0], release)
            if release:
                # Failed admission must not retain prefix-reuse holds. Suspension
                # alone cannot release them when the last cache tier is full.
                self.release_context_admission(req)

        return result

    def _count_unscheduled_first_chunk_claim(self, action: ScheduleAction, released: bool) -> None:
        """Count a first chunk that held a cache claim and was not scheduled.

        Logged at INFO at most once a minute: each released claim is matched
        against the radix tree again on the request's next pass.
        """
        key = (action.value, self._admission_out_of_pages, released)
        self._unscheduled_first_chunk_claims[key] += 1
        now = time.monotonic()
        if now - self._unscheduled_first_chunk_claims_logged_at < 60.0:
            return
        self._unscheduled_first_chunk_claims_logged_at = now
        counts = " ".join(
            f"{action_value}/{'pages' if out_of_pages else 'other'}/"
            f"{'released' if was_released else 'kept'}={count}"
            for (action_value, out_of_pages, was_released), count in sorted(
                self._unscheduled_first_chunk_claims.items()
            )
        )
        logger.info(f"KVCacheV2Scheduler: unscheduled first context chunks with a claim {counts}")

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
        if not self._prepare_context_pair(req) and not (
            self._dkv_unpin(req) and self._prepare_context_pair(req)
        ):
            logger.debug(f"prepare_context failed for context request {req.py_request_id}")
            self._admission_out_of_pages = True
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
        if not self._try_allocate_context(req, context_tokens + draft_len) and not (
            self._dkv_unpin(req)
            and self._prepare_context_pair(req)
            and self._try_allocate_context(req, context_tokens + draft_len)
        ):
            # Out of pages. Give up one started request so this one can
            # proceed, and retry next iteration: a failed resize leaves a
            # first chunk suspended, so the retry has to go back through
            # prepare_context to resume it.
            self._admission_out_of_pages = True
            if preempt_for_pages(req):
                return ScheduleAction.STOP, 0, False
            return ScheduleAction.SKIP, 0, False

        cross_action = self._try_schedule_cross_context(req)
        if cross_action is not ScheduleAction.SCHEDULED:
            self._admission_out_of_pages = True
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
        for a fresh budget unless its whole remaining prompt fits.
        """
        remaining = budget.remaining_tokens
        return (
            self.chunking_policy == ContextChunkingPolicy.FORCE_CHUNK
            and req.state_value == self._context_init_state_value
            and req.is_first_context_chunk
            and remaining is not None
            and remaining < self.chunk_unit_size
            and req.context_remaining_length > remaining
        )

    def _try_schedule_context_chunked(
        self,
        req: LlmRequest,
        budget: BudgetTracker,
        preempt_for_pages: Callable[[LlmRequest], bool],
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

        if not self._has_context_chunk_budget(budget):
            return ScheduleAction.SKIP, 0, False

        # Prepare context (create _KVCache, block reuse, resume — no resize)
        if not self._prepare_context_pair(req) and not (
            self._dkv_unpin(req) and self._prepare_context_pair(req)
        ):
            logger.debug(f"prepare_context failed for chunked context request {req.py_request_id}")
            self._admission_out_of_pages = True
            return ScheduleAction.SKIP, 0, False

        if req.py_request_id in self._ctx_unverified_short:
            # Ordered as short on the router's estimate alone; the reuse claimed
            # just now makes the remainder exact. Once per request per iteration.
            # Unless it keeps its claim (KEEP_UNSCHEDULED_FIRST_CHUNK_CLAIM_ENV),
            # a first chunk that is not scheduled releases its admission, so the
            # retry matches its prefix again.
            self._ctx_unverified_short.discard(req.py_request_id)
            if self._is_long_context(req):
                # Keep the exact remainder (probed): with the claim released, the router's
                # estimate would order it as short again and defer it every iteration without
                # aging it out.
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

        if self.max_context_length is not None:
            chunk_size = min(chunk_size, self.max_context_length - replay_tokens)

        chunk_size = min(
            chunk_size, remaining_budget if remaining_budget is not None else chunk_size
        )

        # Run exactly the pre-grown chunk, or wait for a pass with enough budget;
        # a longer chunk would grow the capacity again before the history catches up.
        pregrown = self._pregrown_context_tokens(req)
        # The per-request cap and the equal-progress share shorten a chunk like the budget does;
        # an exact pre-grown replay is left to the budget alone.
        limit = self._context_chunk_limit(remaining_budget, chunk_size)
        if limit is not None and pregrown == 0:
            chunk_size = min(chunk_size, limit)

        # Round down to chunk_unit_size boundary only when not hitting the end
        # or a checkpoint.
        if chunk_size < budget_context_remaining and not (
            force_chunk and _is_forced_context_chunk_boundary(req, chunk_size)
        ):
            chunk_size = (chunk_size // self.chunk_unit_size) * self.chunk_unit_size
        if pregrown > 0:
            if chunk_size < pregrown:
                return ScheduleAction.SKIP, 0, False
            chunk_size = pregrown

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

        # V2 resizes KV cache directly in the scheduler, so include
        # draft tokens for last chunk.
        if not self._try_allocate_context(req, resize_tokens) and not (
            self._dkv_unpin(req)
            and self._prepare_context_pair(req)
            and self._try_allocate_context(req, resize_tokens)
        ):
            # Out of pages, as in _try_schedule_context_full.
            self._admission_out_of_pages = True
            if preempt_for_pages(req):
                return ScheduleAction.STOP, 0, False
            return ScheduleAction.SKIP, 0, False

        cross_action = self._try_schedule_cross_context(req)
        if cross_action is not ScheduleAction.SCHEDULED:
            self._admission_out_of_pages = True
            self._suspend_request(req)
            return cross_action, 0, False

        chunking_flag = req.context_chunk_size < req.context_remaining_length

        return ScheduleAction.SCHEDULED, chunk_tokens, chunking_flag

    def _reuse_claim_limit(self, req: LlmRequest) -> int | None:
        """Pair the two pools on one depth before either claims.

        Each probe matches through the D-token lookahead and applies its backoff
        in the same tree walk. Their minimum is therefore the common usable depth:
        ``min(max(m_t-D, 0), max(m_d-D, 0)) == max(min(m_t, m_d)-D, 0)``.

        Doing this up front rather than claim-then-reconcile keeps the mismatch
        path (free the draft pages and re-claim shallower) off the common case.
        """
        draft_manager = self._joint_draft_manager
        if draft_manager is None:
            return None
        draft_match = draft_manager.probe_context_reuse(req)
        target_match = self.kv_cache_manager.probe_context_reuse(req)
        if draft_match is None or target_match is None:
            return None
        return min(draft_match, target_match)

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
        allow_eviction: bool = True,
    ) -> tuple[ScheduleAction, int, int, int]:
        """Try to schedule a generation request.

        Returns ``(action, tokens, scheduled_beam_width, req_it_end)``.
        *tokens* is meaningful only when *action* is ``SCHEDULED``. With
        ``allow_eviction`` False a request that finds no pages is skipped; with
        TLLM_V2_GEN_SKIP_BLOCKED and no cache tier to evict to it is ``BLOCKED``.
        """
        beam_width = req.get_beam_width_by_iter(for_next_iteration=False)
        req_tokens = beam_width + get_draft_token_length(req)

        if not budget.can_fit_tokens(req_tokens):
            return ScheduleAction.STOP, 0, scheduled_beam_width, req_it_end

        if scheduled_beam_width == 0:
            scheduled_beam_width = beam_width
        elif scheduled_beam_width != beam_width:
            return ScheduleAction.SKIP, 0, scheduled_beam_width, req_it_end

        success = self._try_allocate_generation(req)

        if not success and not allow_eviction:
            return ScheduleAction.SKIP, 0, scheduled_beam_width, req_it_end
        if not success and self._skip_blocked_generation and not self.kv_cache_manager.can_evict:
            return ScheduleAction.BLOCKED, 0, scheduled_beam_width, req_it_end

        if not success:
            if self.has_cp_helix:
                # No-evict stance: every validated helix run used
                # GUARANTEED_NO_EVICT semantics; eviction stays off under
                # helix until a KV-pressure e2e validates it.
                raise RuntimeError(
                    f"[V2Scheduler] KV allocation failed for helix request "
                    f"{req.py_request_id}; eviction is disabled under helix "
                    f"CP pending end-to-end validation. Increase "
                    f"kv_cache_config.max_gpu_total_bytes (rank-local bytes) "
                    f"or max_tokens (GLOBAL tokens across CP ranks, not "
                    f"per-rank), or reduce concurrency."
                )
            req_it_end, success = self._try_evict_for_gen(
                req, requests_list, req_it, req_it_end, evicted, inflight_request_ids
            )

        if not success:
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

        if success:
            return ScheduleAction.SCHEDULED, req_tokens, scheduled_beam_width, req_it_end

        # Self-eviction: suspend this gen request to free its
        # GPU pages so other requests can resume().
        # Skip if already suspended — suspending again is a no-op
        # that frees no pages.
        if self.kv_cache_manager.is_request_active(
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
        if not self.kv_cache_manager.try_allocate_generation(req):
            return False

        draft_manager = self._joint_draft_manager
        if draft_manager is None:
            if self._admit_unpaired_draft(req):
                return True
        elif draft_manager.try_allocate_generation(req):
            return True

        self.kv_cache_manager.revert_allocate_generation(req)
        return False

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
    ) -> bool:
        """Release one started request's KV cache so another can allocate.

        The fallback for a pool that suspension cannot drain; see
        `KVCacheManagerV2.preempt_request`. With a cache tier below GPU,
        suspension is cheaper and keeps the pages, so that path is left alone.

        The victim leaves on `recompute_paused`, the same channel the
        generation side uses, because a re-prefill needs more teardown than the
        KV cache: the executor frees the sequence slot and
        `reset_for_recompute` rewrites the prompt. Pausing the request here
        instead would leave the slot owned by SeqSlotManager while
        `py_seq_slot` is None, which asserts on the next schedule.

        Returns True when pages became available in this iteration.
        """
        if self.kv_cache_manager.has_cache_tier_below_gpu:
            return False
        # A disaggregated generation worker received its context KV rather
        # than computing it, so it cannot replay a prefill at all.
        if not self.enable_recompute_pause:
            return False

        # Newest first, so the requests closest to completing keep their
        # pages and the pool drains instead of thrashing.
        for i in range(len(requests_list) - 1, -1, -1):
            victim = requests_list[i]
            if victim.py_request_id in protected_ids:
                continue
            if not self._is_recompute_pause_candidate(victim, inflight_request_ids):
                continue
            if not self.kv_cache_manager.is_request_active(victim.py_request_id):
                continue

            if not self._preempt_or_defer(victim, recompute_paused):
                return False
            logger.debug(
                f"[V2Scheduler] Preempting request {victim.py_request_id} "
                f"(state={victim.state.name})"
            )
            preempted_ids.add(victim.py_request_id)
            return True

        return False

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
        seat_waiting_ids: set[int] = frozenset(),
    ) -> None:
        """Fail loudly when no request can be scheduled or reclaimed.

        Unchecked, the executor spins at full speed while scheduling nothing,
        which looks healthy to the hang detector and to `/health`. Context
        candidates count alongside generation ones because a disaggregated
        prefill server has no generation requests at all. A received request
        in ``seat_waiting_ids`` waits for a decode seat, which its holder
        releases when it finishes, so it is not a candidate.
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
            and r.request_id not in seat_waiting_ids
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
        # set_async_transfer_manager).
        transfer_holding = any(
            req.state_value in self._TRANSFER_HOLDING_STATE_VALUES for req in active_requests
        ) or (
            self._async_transfer_manager is not None
            and self._async_transfer_manager.has_any_inflight_requests()
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

        # A connector rejects every tier below GPU at bring-up
        # (`PyExecutor._reject_non_gpu_cache_tiers`), so offering host_cache_size
        # there is advice the user cannot act on.
        if getattr(self.kv_cache_manager, "kv_connector_manager", None) is not None:
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
            f"transfers_in_flight={transfers_in_flight}"
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

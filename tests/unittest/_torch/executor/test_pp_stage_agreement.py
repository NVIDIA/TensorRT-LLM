# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline stages and the prefix each one reuses for a context chunk.

The first PP rank schedules a micro-batch; every other rank runs its own
``KVCacheV2Scheduler`` pass, whose reuse claim matches that stage's own KV pool,
and then adopts the first rank's chunks (``PyExecutor._pp_adopt_leader_schedule``).
When the stages matched different prefixes
of the same prompt, adoption alone checks positions only when chunk sizes differ, so an equal-size chunk
runs at another position on the follower: the KV of every later stage is wrong
and nothing fails. The PP stage agreement (``pp_stage_agree.py``), which a
disaggregated pipeline-parallel server runs, makes the stages agree on one
depth that every stage can claim exactly, holds that claim on every stage
until the chunk runs, and admits the chunk there.

The scheduler, the reuse claim in ``KVCacheManagerV2`` (``prepare_context``,
``prepare_context_cache``, ``probe_context_reuse``), the schedule serialization
and the adoption are the real code. Two stand-in KV cores sit under them:
``_StagePool``, a fixed reusable prefix per stage; and ``_SwaCore``, a port of
the C++ core's match pruning (``BlockRadixTree::pruneMatch``) with sliding-window
and full-attention life cycles, page holds and eviction, which reproduces the
sliding-window case the fixed-prefix pool cannot: a claim capped at an interior
depth comes back shorter.
"""

import enum
import math
import pickle
import types
from typing import Dict, List, Optional, Sequence, Set, Tuple
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_v2_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BlockReusePolicy,
    KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.pp_stage_agree import (
    DEFAULT_MAX_RESERVED,
    SPF_MAX_WAIT_ITERATIONS,
    PPStageAgreement,
    pp_stage_agree_layout,
    set_reserved_depth,
    spf_aged_slots,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests, SerializableSchedulerOutput
from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2 as scheduler_v2_module
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy, ContextChunkingPolicy

pytestmark = pytest.mark.cpu_only

TOKENS_PER_BLOCK = 32
MAX_NUM_TOKENS = 8192
REQUEST_ID = 7
PROMPT_LEN = 40000
# The local layers of a hybrid model: a 2048-token sliding window.
SWA_WINDOW = 2048


class _State(enum.IntEnum):
    """Request states in the order the scheduler compares them."""

    UNKNOWN = 0
    ENCODER_INIT = 1
    DISAGG_GENERATION_INIT = 2
    CONTEXT_INIT = 3
    GENERATION_IN_PROGRESS = 4
    GENERATION_TO_COMPLETE = 5
    GENERATION_COMPLETE = 6


@pytest.fixture(autouse=True)
def _request_states(monkeypatch):
    # Pin the scheduler to plain integer states so the stand-in requests below
    # compare like the real ones with or without the compiled bindings.
    monkeypatch.setattr(scheduler_v2_module, "LlmRequestState", _State)
    monkeypatch.setattr(kv_v2_module, "ReuseScope", lambda **kwargs: kwargs)


class _Request:
    """The ``LlmRequest`` surface the V2 scheduler, the reuse claim and the adoption read."""

    def __init__(self, request_id: int, prompt_len: int):
        self.request_id = request_id
        self.py_request_id = request_id
        self.parent_request_id = request_id
        self.prompt_len = prompt_len
        self.context_current_position = 0
        self.context_chunk_size = prompt_len
        self.prepopulated_prompt_len = 0
        self.state = _State.CONTEXT_INIT
        self.py_draft_tokens = None
        self.is_child = False
        self.is_generation_only_request = False
        self.py_decoding_iter = 0
        self.is_dummy_request = False
        self.is_dummy = False
        self.is_attention_dp_dummy = False
        self.lora_task_id = None
        self.cache_salt = None
        self.return_perf_metrics = False
        self.estimated_reusable_tokens = 0
        self.py_ctx_pre_resize_cap = None
        self.py_connector_served_position = 0
        self.py_batch_idx = None
        self.is_disagg_generation_init_state = False

    @property
    def state_value(self) -> int:
        return int(self.state)

    @property
    def is_context_init_state(self) -> bool:
        return self.state == _State.CONTEXT_INIT

    @property
    def is_encoder_init_state(self) -> bool:
        return False

    @property
    def is_generation_in_progress_state(self) -> bool:
        return False

    @property
    def is_generation_to_complete_state(self) -> bool:
        return False

    @property
    def is_first_context_chunk(self) -> bool:
        return self.context_current_position == self.prepopulated_prompt_len

    @property
    def is_last_context_chunk(self) -> bool:
        return self.context_current_position + self.context_chunk_size >= self.prompt_len

    @property
    def context_remaining_length(self) -> int:
        return self.prompt_len - self.context_current_position

    def set_prepopulated_prompt_len(self, length: int, tokens_per_block: int) -> None:
        # LlmRequest.setPrepopulatedPromptLen: records the prefix and moves the
        # cursor forward to it, never back.
        self.prepopulated_prompt_len = length
        self.context_current_position = max(self.context_current_position, length)


# --------------------------------------------------------------------------- KV pools


class _PoolBase(KVCacheManagerV2):
    """Fields of ``KVCacheManagerV2`` the claim, scheduler and adoption paths read."""

    def _init_common(self):
        self.kv_cache_map: Dict[int, object] = {}
        self.tokens_per_block = TOKENS_PER_BLOCK
        self.enable_block_reuse = True
        self.block_reuse_policy = BlockReusePolicy.ALL_REUSABLE
        self.enable_joint_kv_cache_reuse = False
        self.kv_connector_manager = None
        self.conversation_manager = None
        self.is_draft = False
        self.is_estimating_kv_cache = False
        self._has_cp_helix = False
        self.reuse_match_backoff = 0
        self.resizes: List[tuple] = []
        self.fail_resizes = 0

    def probe_first_new_block_key(self, req):
        return None

    def resize_context(self, req, num_tokens):
        if self.fail_resizes:
            self.fail_resizes -= 1
            return False
        self.resizes.append((req.py_request_id, req.context_current_position, num_tokens))
        return True

    def has_active_cache(self, req):
        return req.py_request_id in self.kv_cache_map

    def suspend_request(self, req):
        cache = self.kv_cache_map.get(req.py_request_id)
        if cache is not None:
            cache.is_active = False

    def revert_allocate_context(self, req):
        return True

    def has_pending_preemption(self):
        return False

    def has_cache_tier_below_gpu(self):
        return False

    # Pages never run out: no page statistics (no pin log), always room.
    available_pages_per_level = None

    def has_room_for_context_tokens(self, num_tokens):
        return True


class _Cache:
    """A ``_KVCache``: created suspended, holding its matched pages until closed."""

    def __init__(self, num_committed_tokens: int, holds: Set[Tuple[int, int]]):
        self.num_committed_tokens = num_committed_tokens
        self.holds = holds
        self.is_active = False
        self.cuda_stream = None
        self.resumes = 0

    def resume(self, stream):
        self.is_active = True
        self.resumes += 1
        return True

    def stop_committing(self):
        pass


class _StagePool(_PoolBase):
    """A stage's KV pool: a stand-in tree that reuses a fixed prefix of every prompt.

    A claim capped below ``reusable`` matches the cap; pages never run out.
    """

    def __init__(self, reusable: int):
        self._init_common()
        self.reusable = reusable
        self.claims: List[Optional[int]] = []

    def probe_context_reuse(self, req, include_claimed: bool = False):
        if req.py_request_id in self.kv_cache_map and not include_claimed:
            return getattr(req, "py_pp_reserved_depth", None)
        return self.reusable

    def prepare_context_cache(self, req, reuse_limit=None, resume=True):
        self.claims.append(reuse_limit)
        if req.py_request_id not in self.kv_cache_map:
            matched = self.reusable if reuse_limit is None else min(self.reusable, reuse_limit)
            self.kv_cache_map[req.py_request_id] = _Cache(matched, set())
        cache = self.kv_cache_map[req.py_request_id]
        if resume:
            cache.resume(None)
        return cache.num_committed_tokens

    def free_resources(self, req, pin_on_release=False):
        self.kv_cache_map.pop(req.py_request_id, None)


class _SwaCore:
    """The C++ KV core's reuse match over one prompt's committed blocks.

    Ports ``BlockRadixTree::match`` / ``pruneMatch``
    (``cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.cpp``)
    and ``AttnLifeCycle::getStaleRange`` (``lifeCycleRegistry.h``), for the
    no-backoff, no-SSM case, with partial reuse on or off:

    * ``matchTokenPath``: one entry per evidence block whose key is in the tree,
      i.e. a block with the same tokens: a full evidence block matches a full
      node, a partial one only a partial node of the same length. At the first
      miss, with partial reuse (``findBestPartialMatchInNextNodes``), the
      child node sharing the longest leading run of tokens matches that many:
      here the next node of the one chain, ``min(evidence tokens in the block,
      node tokens)``.
    * ``pruneMatch``: at the candidate endpoint, every life cycle needs a page
      covering each matched block outside its stale range: all blocks for full
      attention; the sink blocks and the blocks from
      ``(endpoint + 1 - window) / tokens_per_block`` on for a sliding window.
      The first block that lacks one cuts the match there (or to the page's
      coverage), and the check repeats at the shorter endpoint.
    * ``KvCache::_setupForReuse``: a created cache holds the page of every
      non-stale matched block of every life cycle; a held page is never evicted.

    Args:
        num_blocks: committed blocks (tree nodes) of the prompt in this stage's tree.
        life_cycles: ``(window or None, sink blocks)`` per attention life cycle.
        coverage: ``coverage[lc][block]`` = tokens the block's page covers (0: no page).
        node_tokens: tokens per node (default: every node a full block; the
            last node of a chain that ended mid-block is shorter).
        partial: partial reuse on (``enable_partial_reuse``).
    """

    def __init__(self, num_blocks: int, life_cycles, coverage, node_tokens=None, partial=False):
        self.num_blocks = num_blocks
        self.life_cycles = life_cycles
        self.coverage = coverage
        self.node_tokens = node_tokens or [TOKENS_PER_BLOCK] * num_blocks
        self.partial = partial
        self.held: Dict[Tuple[int, int], int] = {}

    def stale_range(self, lc: int, history: int) -> Tuple[int, int]:
        window, sink_blocks = self.life_cycles[lc]
        start = min(math.ceil(history / TOKENS_PER_BLOCK), sink_blocks)
        if window is None:
            return start, start
        # C++ divides with truncation; max() with start >= 0 makes it the same.
        window_start = int((history + 1 - window) / TOKENS_PER_BLOCK)
        return start, max(start, window_start)

    def match(self, num_evidence_tokens: int) -> int:
        matched = []
        while len(matched) < self.num_blocks:
            in_block = min(TOKENS_PER_BLOCK, num_evidence_tokens - TOKENS_PER_BLOCK * len(matched))
            if in_block <= 0:
                break
            node = self.node_tokens[len(matched)]
            if node == in_block:
                matched.append(in_block)
                if in_block < TOKENS_PER_BLOCK:
                    break
                continue
            if self.partial and min(in_block, node) > 0:
                matched.append(min(in_block, node))
            break
        while matched:
            num_tokens = TOKENS_PER_BLOCK * (len(matched) - 1) + matched[-1]
            shortened = False
            for lc in range(len(self.life_cycles)):
                stale_beg, stale_end = self.stale_range(lc, num_tokens)
                for i in range(len(matched)):
                    if stale_beg <= i < stale_end:
                        continue
                    coverage = self.coverage[lc][i]
                    if coverage >= matched[i]:
                        continue
                    if coverage > 0:
                        matched = matched[: i + 1]
                        matched[-1] = coverage
                    else:
                        matched = matched[:i]
                    shortened = True
                    break
                if shortened:
                    break
            if not shortened:
                break
        if not matched:
            return 0
        return TOKENS_PER_BLOCK * (len(matched) - 1) + matched[-1]

    def hold(self, num_tokens: int) -> Set[Tuple[int, int]]:
        holds = set()
        num_blocks = math.ceil(num_tokens / TOKENS_PER_BLOCK)
        for lc in range(len(self.life_cycles)):
            stale_beg, stale_end = self.stale_range(lc, num_tokens)
            for i in range(num_blocks):
                if not stale_beg <= i < stale_end:
                    holds.add((lc, i))
        for page in holds:
            self.held[page] = self.held.get(page, 0) + 1
        return holds

    def release(self, holds: Set[Tuple[int, int]]) -> None:
        for page in holds:
            self.held[page] -= 1
            if not self.held[page]:
                del self.held[page]

    def evict_all_unheld(self) -> int:
        """Evict every page no cache holds (a follower allocating under pressure)."""
        evicted = 0
        for lc, pages in enumerate(self.coverage):
            for i, coverage in enumerate(pages):
                if coverage and (lc, i) not in self.held:
                    pages[i] = 0
                    evicted += 1
        return evicted


def _hybrid_window_core(num_blocks: int, window_ends: Sequence[int]) -> _SwaCore:
    """A stage holding 2048-token sliding-window and full-attention layers.

    Every full-attention page of the committed prompt is there; the sliding
    window keeps only the trailing window of each committed endpoint in
    ``window_ends`` (a turn's end), as the drop plan and scratch reuse leave it.
    """
    full = [TOKENS_PER_BLOCK] * num_blocks
    swa = [0] * num_blocks
    for end in window_ends:
        for i in range(int((end + 1 - SWA_WINDOW) / TOKENS_PER_BLOCK), end // TOKENS_PER_BLOCK):
            swa[i] = TOKENS_PER_BLOCK
    return _SwaCore(num_blocks, [(SWA_WINDOW, 0), (None, 0)], [swa, full])


def _chain_core(chain_end: int, window_ends: Sequence[int], partial: bool) -> _SwaCore:
    """A hybrid sliding-window stage whose committed chain ends at ``chain_end`` (mid-block
    when not a multiple of the block size), keeping the trailing 2048-token
    window of each end in ``window_ends``. A page covers the tokens of its block
    that were committed when its window end was reached."""
    num_blocks = math.ceil(chain_end / TOKENS_PER_BLOCK)
    nodes = [TOKENS_PER_BLOCK] * num_blocks
    if chain_end % TOKENS_PER_BLOCK:
        nodes[-1] = chain_end % TOKENS_PER_BLOCK
    full = list(nodes)
    swa = [0] * num_blocks
    for end in window_ends:
        last = math.ceil(end / TOKENS_PER_BLOCK)
        for i in range(int((end + 1 - SWA_WINDOW) / TOKENS_PER_BLOCK), last):
            covered = min(TOKENS_PER_BLOCK, end - i * TOKENS_PER_BLOCK)
            swa[i] = max(swa[i], covered)
    return _SwaCore(num_blocks, [(SWA_WINDOW, 0), (None, 0)], [swa, full], nodes, partial)


class _SwaStagePool(_PoolBase):
    """A stage's KV pool: the real ``KVCacheManagerV2`` claim and probe over ``_SwaCore``."""

    def __init__(self, core: _SwaCore):
        self._init_common()
        self.core = core
        self._stream = types.SimpleNamespace(cuda_stream=None)
        self.impl = types.SimpleNamespace(
            probe_reuse=lambda scope, tokens: self.core.match(len(tokens))
        )
        self.claims: List[Tuple[Optional[int], int]] = []

    def _reuse_token_source(self, req):
        return list(range(req.prompt_len))

    def _augment_tokens_for_block_reuse(self, tokens, req, start=0, end=None):
        return tokens[start:end]

    def _create_kv_cache(self, request_id, lora_task_id, input_tokens, **kwargs):
        assert request_id not in self.kv_cache_map
        depth = self.core.match(len(input_tokens)) if input_tokens else 0
        cache = _Cache(depth, self.core.hold(depth))
        self.kv_cache_map[request_id] = cache
        self.claims.append((len(input_tokens) if input_tokens is not None else None, depth))
        return cache

    def _restore_page_index_bufs(self, request_id, kv_cache):
        pass

    def free_resources(self, req, pin_on_release=False):
        cache = self.kv_cache_map.pop(req.py_request_id, None)
        if cache is not None:
            self.core.release(cache.holds)


def _scheduler(pool) -> KVCacheV2Scheduler:
    return KVCacheV2Scheduler(
        max_batch_size=8,
        max_num_tokens=MAX_NUM_TOKENS,
        kv_cache_manager=pool,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        ctx_chunk_config=(ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, TOKENS_PER_BLOCK),
        no_schedule_until_state=_State.CONTEXT_INIT,
        no_schedule_after_state=_State.GENERATION_TO_COMPLETE,
    )


# --------------------------------------------------------------------------- ranks


class _Mailbox:
    """Point-to-point object messages between the stand-in ranks (eager MPI)."""

    def __init__(self):
        self.queues: Dict[tuple, list] = {}

    def send(self, obj, src, dst, tag):
        self.queues.setdefault((src, dst, int(tag)), []).append(pickle.dumps(obj))
        return types.SimpleNamespace(wait=lambda: None)

    def recv(self, src, dst, tag):
        return pickle.loads(self.queues[(src, dst, int(tag))].pop(0))


def _rank(pp_rank: int, pp_size: int, pool, prompt_len: int, mailbox: _Mailbox):
    """A PP rank's executor state around the real scheduler, reuse claim and adoption."""
    executor = object.__new__(PyExecutor)
    executor.dist = types.SimpleNamespace(
        rank=pp_rank,
        pp_rank=pp_rank,
        pp_size=pp_size,
        tp_size=1,
        cp_size=1,
        recv_object=lambda src, tag: mailbox.recv(src, pp_rank, tag),
        isend_object=lambda obj, dest, tag: mailbox.send(obj, pp_rank, dest, tag),
    )
    executor.kv_cache_manager = pool
    executor.scheduler = _scheduler(pool)
    executor.active_requests = [_Request(REQUEST_ID, prompt_len)]
    executor._is_kv_manager_v2 = True
    executor.enable_attention_dp = False
    executor.enable_kv_cache_reuse = True
    executor.enable_joint_kv_cache_reuse = False
    executor.kv_connector_manager = None
    # A disaggregated server: the precondition of the stage agreement.
    executor._pp_disaggregated = True
    executor.num_micro_batches = pp_size
    executor.pp_async_broadcast_sample_state = False
    executor._revert_ctx_alloc = lambda requests: None
    return executor


def _make_ranks(pools, prompt_len: int, stage_agree: bool, monkeypatch, spf_order: bool = False):
    """The stand-in ranks; with ``stage_agree`` each builds the agreement as the
    executor does. The executor's first rank reserves shortest-uncached-first;
    ``spf_order`` False keeps the agreement's arrival order instead, which the
    tests below were written against."""
    monkeypatch.setenv("TRTLLM_SCHED_SHORTEST_PREFILL_FIRST", "1")
    mailbox = _Mailbox()
    ranks = [_rank(r, len(pools), pool, prompt_len, mailbox) for r, pool in enumerate(pools)]
    if stage_agree:
        for executor in ranks:
            executor._pp_stage_agree = executor._create_pp_stage_agreement()
            assert executor._pp_stage_agree is not None
            if not spf_order:
                executor._pp_stage_agree.uncached_estimate = None
    return ranks


def _adopt_only_pp_iteration(ranks) -> Optional[SerializableSchedulerOutput]:
    """One PP loop iteration's scheduling with the leader-batch adoption and
    without the stage agreement: the first rank's pass, the schedule with its
    chunks on the wire, each follower's pass and adoption."""
    leader, followers = ranks[0], ranks[1:]
    output = leader.scheduler.schedule_request(leader.active_requests, set())
    if not output.context_requests:
        return None
    scheduled = ScheduledRequests()
    scheduled.reset_context_requests(list(output.context_requests))
    wire = pickle.loads(
        pickle.dumps(
            SerializableSchedulerOutput.from_scheduler_result(
                scheduled, [], 1, record_context_chunks=True
            )
        )
    )
    for executor in followers:
        batch, _, _ = wire.to_scheduler_result(executor.active_requests)
        local = executor.scheduler.schedule_request(executor.active_requests, set())
        PyExecutor._pp_adopt_leader_schedule(executor, batch, local)
    return wire


def _pp_iteration(ranks, between=None) -> SerializableSchedulerOutput:
    """The same iteration with the stage agreement, in the loop's order.

    First rank: the agreement step (reports due, rounds, reservations, holds),
    its pass without the held requests, the admitted-depth check, the schedule
    with the first chunks and the proposals. Each follower: its pass without the
    waiting first chunks, the adoption, then its answers to the proposals; with
    the early release (``_pp_follower_early_release``) it first applies the
    released reservations of the message, as the loop does. The first rank
    sends its allocation order where ``_pp_adopt_in_admission_order`` is set.
    ``between(ranks)`` runs after the first rank scheduled, before the followers.
    """
    leader, followers = ranks[0], ranks[1:]
    leader._pp_stage_agree_exchange()
    output = leader.scheduler.schedule_request(leader._schedulable_active_requests(), set())
    scheduled = ScheduledRequests()
    scheduled.reset_context_requests(list(output.context_requests))
    leader._pp_stage_agree_check_admitted(scheduled)
    wire = pickle.loads(
        pickle.dumps(
            SerializableSchedulerOutput.from_scheduler_result(
                scheduled,
                [],
                1,
                record_first_chunks=True,
                reuse_proposals=leader._pp_stage_agree.take_proposals(),
                context_admission_order=PyExecutor._pp_leader_admission_order(leader),
            )
        )
    )
    if between is not None:
        between(ranks)
    for executor in followers:
        executor._pp_stage_agree_exchange()
        batch, _, _ = wire.to_scheduler_result(executor.active_requests)
        if executor._pp_follower_early_release:
            executor._pp_stage_agree_early_release(batch)
        local = executor.scheduler.schedule_request(
            executor._pp_stage_agree_local_requests(batch), set()
        )
        PyExecutor._pp_adopt_leader_schedule(executor, batch, local)
        executor._pp_stage_agree_answer(batch)
    return wire


def _run_until_admitted(ranks, max_iterations: int = 32, between=None):
    """Iterations until the first rank admits the request; returns (wire, iterations)."""
    for iteration in range(max_iterations):
        wire = _pp_iteration(ranks, between)
        if wire.context_chunks:
            return wire, iteration
    raise AssertionError("the first chunk was never admitted")


def _chunks(ranks):
    return [
        (r.active_requests[0].context_current_position, r.active_requests[0].context_chunk_size)
        for r in ranks
    ]


# --------------------------------------------------------------------------- the adoption-only bug

# Prompt 40000 tokens; stage 0 matched 16384, the follower another depth.
CASES = {
    "equal_reuse": [16384, 16384],
    "follower_matched_less": [16384, 8192],
    "follower_matched_more": [8192, 16384],
    "pp4_middle_stage_matched_less": [24576, 24576, 8192, 16384],
}


@pytest.mark.parametrize("reusable", list(CASES.values()), ids=list(CASES))
def test_adoption_alone_runs_another_prefix_on_the_follower_silently(reusable, monkeypatch):
    """The bug, with the leader-batch adoption but without the stage
    agreement: equal chunk sizes skip the position check."""
    ranks = _make_ranks(
        [_StagePool(d) for d in reusable], PROMPT_LEN, stage_agree=False, monkeypatch=monkeypatch
    )

    wire = _adopt_only_pp_iteration(ranks)

    leader_chunk = wire.context_chunks[REQUEST_ID]
    assert leader_chunk == (reusable[0], MAX_NUM_TOKENS)
    # Every stage runs an 8192-token chunk from the prefix it matched itself.
    assert _chunks(ranks) == [(depth, MAX_NUM_TOKENS) for depth in reusable]


# --------------------------------------------------------------------------- the agreement


@pytest.mark.parametrize("reusable", list(CASES.values()), ids=list(CASES))
def test_stage_agreement_runs_the_smallest_common_prefix_on_every_stage(reusable, monkeypatch):
    ranks = _make_ranks(
        [_StagePool(d) for d in reusable], PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch
    )
    lag = ranks[0]._pp_stage_agree.lag
    common = min(reusable)
    # One round when the first rank's prefix is the smallest, two otherwise.
    rounds = 1 if reusable[0] == common else 2

    wire, iterations = _run_until_admitted(ranks)

    assert iterations == rounds * lag
    assert wire.context_chunks[REQUEST_ID] == (common, MAX_NUM_TOKENS)
    assert wire.context_first_chunks == [REQUEST_ID]
    assert _chunks(ranks) == [(common, MAX_NUM_TOKENS)] * len(ranks)
    for executor in ranks:
        pool = executor.kv_cache_manager
        # Each stage resumed its reservation at the common prefix and sized one
        # chunk after it; the request no longer counts as reserved.
        assert executor.active_requests[0].py_pp_reserved_depth is None
        assert pool.kv_cache_map[REQUEST_ID].num_committed_tokens == common
        assert pool.kv_cache_map[REQUEST_ID].is_active
        assert pool.resizes[-1] == (REQUEST_ID, common, MAX_NUM_TOKENS)


# Sliding-window stages (MuseSpark Open: a 2048-token window on every stage). The committed
# chain and the window ends each stage still holds pages for.
SWA_CASES = {
    # Stage 1 reuses deeper (24576) but holds no window at 16384: a claim capped
    # there prunes to its window at 8192, which stage 0 holds too.
    "follower_window_below_cap": ([(512, [8192, 16384]), (768, [8192, 24576])], 8192, 2),
    # Stage 0 holds no window at 8192: its own claim at the follower's 8192
    # prunes further, to 4096, before it proposes (still two rounds).
    "first_rank_prunes_further": (
        [(512, [4096, 16384]), (768, [4096, 8192, 24576])],
        4096,
        2,
    ),
    # PP3: stage 1 lowers 16384 to 8192, then stage 2 lowers 8192 to 4096.
    "pp3_two_lowerings": (
        [(512, [4096, 8192, 16384]), (768, [4096, 8192, 24576]), (768, [4096, 16384, 24576])],
        4096,
        3,
    ),
    # A PP4 middle stage holds only its own end: nothing common but 0.
    "pp4_no_common_window": (
        [(512, [8192, 16384]), (768, [8192, 24576]), (640, [20480]), (512, [8192, 16384])],
        0,
        2,
    ),
    # The follower holds a window at the first rank's depth: one round.
    "follower_holds_the_window": ([(512, [16384]), (768, [16384, 24576])], 16384, 1),
}


def _swa_pools(stages):
    return [_SwaStagePool(_hybrid_window_core(n, ends)) for n, ends in stages]


def test_swa_core_prunes_a_capped_claim_below_the_cap():
    """On a sliding-window stage a claim capped at an interior depth comes
    back shorter, at the next endpoint whose trailing window still has pages."""
    core = _hybrid_window_core(768, [8192, 24576])
    assert core.match(PROMPT_LEN - 1) == 24576
    assert core.match(24576) == 24576
    assert core.match(16384) == 8192
    assert core.match(8192) == 8192
    # One token short of that endpoint: the window then starts in block 191,
    # which has no page, and no earlier endpoint has a window either.
    assert core.match(8191) == 0
    # Full attention alone: every depth is claimable.
    full = _SwaCore(768, [(None, 0)], [[TOKENS_PER_BLOCK] * 768])
    assert full.match(16384) == 16384
    # The live window is [h + 1 - W, h]: at an exact block boundary with a
    # one-token window every earlier block is stale (lifeCycleRegistry.h).
    assert _SwaCore(4, [(1, 0)], [[0, 0, 0, 0]]).stale_range(0, 4 * TOKENS_PER_BLOCK) == (0, 4)


def test_capped_adoption_raises_on_a_sliding_window_stage(monkeypatch):
    """With capped claims only: the first rank admits at the smallest prefix the
    followers reported (16384 here, the first rank's own); the follower's claim
    capped there comes back at 8192 and the adoption raises, stopping the server."""
    pools = _swa_pools(SWA_CASES["follower_window_below_cap"][0])
    ranks = _make_ranks(pools, PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch)
    leader, follower = ranks
    reported = [pool.probe_context_reuse(r.active_requests[0]) for pool, r in zip(pools, ranks)]
    assert reported == [16384, 24576]
    leader_request = leader.active_requests[0]
    leader_request.context_current_position = min(reported)
    leader_request.prepopulated_prompt_len = min(reported)
    leader_request.context_chunk_size = MAX_NUM_TOKENS
    scheduled = ScheduledRequests()
    scheduled.reset_context_requests([leader_request])
    wire = SerializableSchedulerOutput.from_scheduler_result(
        scheduled, [], 1, record_first_chunks=True
    )
    batch, _, _ = pickle.loads(pickle.dumps(wire)).to_scheduler_result(follower.active_requests)
    local = types.SimpleNamespace(context_requests=[], generation_requests=[])

    with pytest.raises(RuntimeError, match="at position 8192 only"):
        PyExecutor._pp_adopt_leader_schedule(follower, batch, local)
    assert pools[1].claims[-1] == (16384, 8192)


@pytest.mark.parametrize("stages, common, rounds", list(SWA_CASES.values()), ids=list(SWA_CASES))
def test_stage_agreement_admits_at_a_depth_every_swa_stage_claims_exactly(
    stages, common, rounds, monkeypatch
):
    """The stages converge on the largest depth valid on all of them."""
    pools = _swa_pools(stages)
    ranks = _make_ranks(pools, PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch)
    lag = ranks[0]._pp_stage_agree.lag

    wire, iterations = _run_until_admitted(ranks)

    assert iterations == rounds * lag
    assert wire.context_chunks[REQUEST_ID] == (common, MAX_NUM_TOKENS)
    assert _chunks(ranks) == [(common, MAX_NUM_TOKENS)] * len(ranks)
    for pool in pools:
        assert pool.kv_cache_map[REQUEST_ID].num_committed_tokens == common
        # No stage needed a fresh claim at admission: its reservation was exact.
        assert pool.kv_cache_map[REQUEST_ID].resumes == 1
    agreement = ranks[0]._pp_stage_agree
    first = stages[0][1][-1]
    assert agreement.num_admitted == 1
    assert agreement.num_lowered == int(common < first)
    assert agreement.num_extra_rounds == rounds - 1


# --------------------------------------------------------------------------- held prefixes


@pytest.mark.parametrize("stages", [SWA_CASES["follower_window_below_cap"][0]])
def test_reservations_survive_eviction_between_the_answer_and_the_chunk(stages, monkeypatch):
    """Every stage holds the agreed prefix from its answer until the chunk
    runs, so a follower that evicts everything it can in the meantime still
    claims it exactly (with capped claims only, the prefix was gone and the server
    stopped)."""
    pools = _swa_pools(stages)
    ranks = _make_ranks(pools, PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch)
    evicted = []

    def evict_on_followers(ranks):
        # From the follower's first answer on, it evicts every page it does not hold.
        for executor in ranks[1:]:
            if REQUEST_ID in executor.kv_cache_manager.kv_cache_map:
                evicted.append(executor.kv_cache_manager.core.evict_all_unheld())

    wire, _ = _run_until_admitted(ranks, between=evict_on_followers)

    assert sum(evicted) > 0
    assert wire.context_chunks[REQUEST_ID] == (8192, MAX_NUM_TOKENS)
    assert _chunks(ranks) == [(8192, MAX_NUM_TOKENS)] * 2
    # What the follower kept is exactly what its reservation held.
    core = pools[1].core
    assert core.match(8192) == 8192
    assert core.match(PROMPT_LEN - 1) == 8192


def test_an_unheld_prefix_evicted_before_the_chunk_would_fail_the_claim(monkeypatch):
    """The same eviction without the follower's hold: the claim comes back short,
    which the reservation prevents (control for the test above)."""
    pools = _swa_pools(SWA_CASES["follower_window_below_cap"][0])
    core = pools[1].core
    assert core.match(8192) == 8192
    core.evict_all_unheld()
    assert core.match(8192) == 0


def test_first_rank_that_loses_its_reservation_agrees_again_without_raising(monkeypatch):
    """On the first rank: a confirmed first chunk whose admission fails (no
    pages for the chunk) keeps its reservation, suspended, and is admitted at
    the agreed depth in a later pass."""
    pools = [_StagePool(16384), _StagePool(8192)]
    ranks = _make_ranks(pools, PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch)
    lag = ranks[0]._pp_stage_agree.lag
    for _ in range(2 * lag):
        wire = _pp_iteration(ranks)
        assert not wire.context_chunks
    pools[0].fail_resizes = 1

    wire = _pp_iteration(ranks)
    assert not wire.context_chunks
    leader_request = ranks[0].active_requests[0]
    assert REQUEST_ID in pools[0].kv_cache_map
    assert not pools[0].kv_cache_map[REQUEST_ID].is_active
    assert leader_request.context_current_position == 0

    wire = _pp_iteration(ranks)
    assert wire.context_chunks[REQUEST_ID] == (8192, MAX_NUM_TOKENS)
    assert _chunks(ranks) == [(8192, MAX_NUM_TOKENS)] * 2


def test_first_rank_takes_an_off_depth_admission_back_out_of_the_batch(monkeypatch):
    """A first chunk the first rank's pass admitted at another depth than every
    stage confirmed leaves the batch and is agreed again, instead of reaching a
    follower that cannot claim it."""
    pools = [_StagePool(16384), _StagePool(8192)]
    ranks = _make_ranks(pools, PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch)
    leader, follower = ranks
    lag = leader._pp_stage_agree.lag
    for _ in range(2 * lag):
        _pp_iteration(ranks)
    leader._pp_stage_agree_exchange()
    assert leader._pp_stage_agree.confirmed_depth(REQUEST_ID) == 8192
    request = leader.active_requests[0]
    # The pass claimed afresh (its reservation gone) and found another depth.
    pools[0].free_resources(request)
    request.set_prepopulated_prompt_len(16384, TOKENS_PER_BLOCK)
    request.context_chunk_size = MAX_NUM_TOKENS
    pools[0].kv_cache_map[REQUEST_ID] = _Cache(16384, set())
    scheduled = ScheduledRequests()
    scheduled.reset_context_requests([request])

    leader._pp_stage_agree_check_admitted(scheduled)

    assert scheduled.context_requests == []
    assert REQUEST_ID not in pools[0].kv_cache_map
    assert request.context_current_position == 0
    assert leader._pp_stage_agree.confirmed_depth(REQUEST_ID) is None
    assert leader._pp_stage_agree.num_undone == 1
    # Finish the iteration on the follower: nothing to run, and the schedule
    # tells it to drop its reservation too, so no follower keeps a hold the
    # first rank no longer counts.
    proposals = leader._pp_stage_agree.take_proposals()
    assert proposals == {REQUEST_ID: None}
    wire = SerializableSchedulerOutput.from_scheduler_result(
        scheduled,
        [],
        1,
        record_first_chunks=True,
        reuse_proposals=proposals,
    )
    batch, _, _ = wire.to_scheduler_result(follower.active_requests)
    local = follower.scheduler.schedule_request(
        follower._pp_stage_agree_local_requests(batch), set()
    )
    PyExecutor._pp_adopt_leader_schedule(follower, batch, local)
    follower._pp_stage_agree_answer(batch)
    assert REQUEST_ID not in pools[1].kv_cache_map
    # The next rounds agree again and admit at the common depth.
    wire, _ = _run_until_admitted(ranks)
    assert wire.context_chunks[REQUEST_ID] == (8192, MAX_NUM_TOKENS)
    assert _chunks(ranks) == [(8192, MAX_NUM_TOKENS)] * 2


# --------------------------------------------------------------------------- other paths


def test_stage_agreement_refuses_a_mid_prompt_chunk_at_another_position(monkeypatch):
    """A non-first chunk at another position raises with the agreement even
    though the sizes agree (adoption alone checks only when sizes differ)."""
    ranks = _make_ranks(
        [_StagePool(0), _StagePool(0)], PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch
    )
    leader, follower = ranks
    leader_request = leader.active_requests[0]
    follower_request = follower.active_requests[0]
    leader_request.context_current_position = 16384
    leader_request.context_chunk_size = MAX_NUM_TOKENS
    follower_request.context_current_position = 8192
    follower_request.context_chunk_size = MAX_NUM_TOKENS
    for executor, depth in ((leader, 16384), (follower, 8192)):
        executor.kv_cache_manager.kv_cache_map[REQUEST_ID] = _Cache(depth, set())
    scheduled = ScheduledRequests()
    scheduled.reset_context_requests([leader_request])
    wire = SerializableSchedulerOutput.from_scheduler_result(
        scheduled, [], 1, record_first_chunks=True
    )
    assert wire.context_first_chunks == []
    batch, _, _ = pickle.loads(pickle.dumps(wire)).to_scheduler_result([follower_request])
    local = types.SimpleNamespace(context_requests=[follower_request], generation_requests=[])

    with pytest.raises(RuntimeError, match="is at position 8192"):
        PyExecutor._pp_adopt_leader_schedule(follower, batch, local)


def test_stage_agreement_keeps_context_requests_out_of_the_follower_pass(monkeypatch):
    """The follower's own pass sees no context request at all -- no waiting first
    chunk, admitted or not, and no started prompt, scheduled by the first rank
    or not: every context chunk is adopted at the first rank's size. Other work
    (here a generation request) stays in the pass."""
    ranks = _make_ranks(
        [_StagePool(0), _StagePool(0)], PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch
    )
    follower = ranks[1]
    other = _Request(9, 1000)
    scheduled_mid = _Request(11, 1000)
    unscheduled_mid = _Request(12, 1000)
    for req in (scheduled_mid, unscheduled_mid):
        req.context_current_position = 512
    generating = _Request(13, 1000)
    generating.state = _State.GENERATION_IN_PROGRESS
    follower.active_requests += [other, scheduled_mid, unscheduled_mid, generating]
    batch = ScheduledRequests()
    batch.leader_context_chunks = {REQUEST_ID: (0, MAX_NUM_TOKENS), 11: (512, 488)}
    batch.leader_first_context_chunks = {REQUEST_ID}

    local = follower._pp_stage_agree_local_requests(batch)

    assert [r.request_id for r in local] == [13]


def test_agreement_is_skipped_on_configurations_it_does_not_cover():
    """Where the agreement does not apply the executor runs the stock PP loop
    (None) instead of refusing the deployment."""
    executor = _rank(1, 2, _StagePool(0), 100, _Mailbox())
    assert executor._create_pp_stage_agreement() is not None
    executor._is_kv_manager_v2 = False
    assert executor._create_pp_stage_agreement() is None
    executor._is_kv_manager_v2 = True
    executor.dist.tp_size = 2
    assert executor._create_pp_stage_agreement() is None
    executor.dist.tp_size = 1
    executor.enable_attention_dp = True
    assert executor._create_pp_stage_agreement() is None
    executor.enable_attention_dp = False
    executor.kv_connector_manager = object()
    assert executor._create_pp_stage_agreement() is None
    executor.kv_connector_manager = None
    executor._pp_disaggregated = False
    assert executor._create_pp_stage_agreement() is None
    executor._pp_disaggregated = True
    executor.dist.pp_size = 1
    assert executor._create_pp_stage_agreement() is None


@pytest.mark.parametrize(
    "layout, expected",
    [
        (
            dict(pp_size=4, tp_size=1, cp_size=1, enable_attention_dp=False, disaggregated=True),
            True,
        ),
        (
            dict(pp_size=1, tp_size=1, cp_size=1, enable_attention_dp=False, disaggregated=True),
            False,
        ),
        (
            dict(pp_size=4, tp_size=1, cp_size=1, enable_attention_dp=False, disaggregated=False),
            False,
        ),
        (
            dict(pp_size=4, tp_size=2, cp_size=1, enable_attention_dp=False, disaggregated=True),
            False,
        ),
        (
            dict(pp_size=4, tp_size=1, cp_size=2, enable_attention_dp=False, disaggregated=True),
            False,
        ),
        (
            dict(pp_size=4, tp_size=1, cp_size=1, enable_attention_dp=True, disaggregated=True),
            False,
        ),
    ],
)
def test_stage_agreement_layout(layout, expected):
    """Only a disaggregated server at pp_size > 1 with tp == cp == 1 and no
    attention DP has the agreement's layout; pp_size 1 never has it."""
    assert pp_stage_agree_layout(**layout) is expected


def test_max_reserved_default():
    assert DEFAULT_MAX_RESERVED == 16


def test_first_rank_reserves_at_most_max_reserved_first_chunks(monkeypatch):
    monkeypatch.setattr(py_executor_module, "DEFAULT_MAX_RESERVED", 2)
    ranks = _make_ranks(
        [_StagePool(0), _StagePool(0)], PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch
    )
    for executor in ranks:
        executor.active_requests += [_Request(rid, 1000) for rid in (20, 21, 22)]

    wire = _pp_iteration(ranks)

    assert sorted(wire.reuse_proposals) == [REQUEST_ID, 20]
    for executor in ranks:
        assert sorted(executor.kv_cache_manager.kv_cache_map) == [REQUEST_ID, 20]
    assert ranks[0]._pp_stage_agree_held == {REQUEST_ID, 20, 21, 22}


@pytest.mark.parametrize("pp_size, disaggregated", [(2, False), (1, True), (1, False)])
def test_without_the_precondition_the_schedule_message_is_stock(pp_size, disaggregated):
    executor = _rank(0, pp_size, _StagePool(0), 100, _Mailbox())
    executor._pp_disaggregated = disaggregated
    assert executor._create_pp_stage_agreement() is None
    request = _Request(REQUEST_ID, 100)
    scheduled = ScheduledRequests()
    scheduled.reset_context_requests([request])
    wire = SerializableSchedulerOutput.from_scheduler_result(scheduled, [], 1)
    assert wire.context_first_chunks is None
    assert wire.reuse_proposals is None
    assert wire.context_chunks is None
    batch, _, _ = wire.to_scheduler_result([request])
    assert batch.leader_first_context_chunks is None
    assert batch.leader_reuse_proposals is None
    assert batch.leader_context_chunks is None
    # Nothing is held and no request carries a reuse cap or a reservation.
    assert executor._schedulable_active_requests() is executor.active_requests
    assert not hasattr(request, "py_pp_reuse_cap")
    assert not hasattr(request, "py_pp_reserved_depth")


@pytest.mark.parametrize("cap, expected", [(None, None), (4096, 4096)])
def test_prepare_context_caps_the_claim_at_the_stage_agreement(cap, expected):
    pool = _StagePool(reusable=16384)
    request = _Request(REQUEST_ID, PROMPT_LEN)
    if cap is not None:
        request.py_pp_reuse_cap = cap

    assert KVCacheManagerV2.prepare_context(pool, request)

    assert pool.claims == [expected]
    assert request.context_current_position == (16384 if cap is None else cap)


def test_real_claim_without_resume_holds_the_pages_and_leaves_the_cursor():
    """The real ``prepare_context_cache`` / ``prepare_context`` with ``resume=False``:
    the cache exists, suspended, at the capped depth; the cursor has not moved;
    admitting later resumes it at that depth."""
    pool = _SwaStagePool(_hybrid_window_core(768, [8192, 24576]))
    request = _Request(REQUEST_ID, PROMPT_LEN)
    request.py_pp_reuse_cap = 16384

    assert KVCacheManagerV2.prepare_context(pool, request, resume=False)

    cache = pool.kv_cache_map[REQUEST_ID]
    assert (cache.num_committed_tokens, cache.is_active) == (8192, False)
    assert request.context_current_position == 0
    assert pool.claims == [(16384, 8192)]
    assert KVCacheManagerV2.prepare_context(pool, request)
    assert cache.is_active and request.context_current_position == 8192
    assert len(pool.claims) == 1


def test_joint_target_and_draft_pools_reserve_and_adopt_one_depth():
    """A paired draft pool (joint reuse) reserves with the target, without
    resuming either, at their common exact depth; adoption resumes both there."""
    target = _SwaStagePool(_hybrid_window_core(768, [8192, 24576]))
    draft = _SwaStagePool(_hybrid_window_core(768, [8192, 16384, 24576]))
    target.enable_joint_kv_cache_reuse = True
    draft.is_draft = True
    draft.try_allocate_draft_context = lambda req, num_tokens: True
    scheduler = KVCacheV2Scheduler(
        max_batch_size=8,
        max_num_tokens=MAX_NUM_TOKENS,
        kv_cache_manager=target,
        draft_kv_cache_manager=draft,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        ctx_chunk_config=(ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, TOKENS_PER_BLOCK),
        no_schedule_until_state=_State.CONTEXT_INIT,
        no_schedule_after_state=_State.GENERATION_TO_COMPLETE,
    )
    request = _Request(REQUEST_ID, PROMPT_LEN)

    # Capped at 16384, the draft holds that window but the target only 8192.
    assert scheduler.reserve_first_context_chunk(request, 16384) == 8192
    for pool in (target, draft):
        cache = pool.kv_cache_map[REQUEST_ID]
        assert (cache.num_committed_tokens, cache.is_active) == (8192, False)
    assert request.context_current_position == 0
    assert request.py_pp_reserved_depth == 8192

    assert scheduler.adopt_first_context_chunk(request, 8192, MAX_NUM_TOKENS)
    for pool in (target, draft):
        cache = pool.kv_cache_map[REQUEST_ID]
        assert (cache.num_committed_tokens, cache.is_active, cache.resumes) == (8192, True, 1)
    assert (request.context_current_position, request.context_chunk_size) == (
        8192,
        MAX_NUM_TOKENS,
    )


def test_probe_reports_the_reserved_depth_of_a_held_request(monkeypatch):
    pool = object.__new__(KVCacheManagerV2)
    pool.enable_block_reuse = True
    pool.kv_cache_map = {REQUEST_ID: object()}
    probes = []
    pool.impl = types.SimpleNamespace(
        probe_reuse=lambda scope, tokens: probes.append(tokens) or 512
    )
    pool._context_reuse_tokens = lambda req: [1, 2, 3]
    pool._derive_reuse_salt = lambda salt: None
    request = types.SimpleNamespace(py_request_id=REQUEST_ID, lora_task_id=None, cache_salt=None)

    # Stock: a request holding a cache has nothing to probe.
    assert pool.probe_context_reuse(request) is None
    assert pool.probe_context_reuse(request, include_claimed=True) == 512
    request.py_pp_reserved_depth = 256
    assert pool.probe_context_reuse(request) == 256
    assert probes == [[1, 2, 3]]


def test_commit_stops_at_the_handled_chunk_not_the_live_cursor():
    """Under PP the cursor is already past chunks still queued on the stage."""
    committed = []

    class _CommitCache:
        is_active = True
        num_committed_tokens = 0

        def commit(self, tokens):
            committed.append(len(tokens))
            self.num_committed_tokens += len(tokens)

        def stop_committing(self):
            committed.append("stop")

        def resize(self, capacity, history=None):
            return True

    pool = _StagePool(reusable=0)
    pool._can_publish_block_reuse = True
    pool._reuse_token_source = lambda req: list(range(req.prompt_len))
    pool._augment_tokens_for_block_reuse = lambda tokens, req, start, end: tokens[start:end]
    request = _Request(REQUEST_ID, 3 * MAX_NUM_TOKENS)
    # Chunks 0 and 1 launched; the batch being handled ran chunk 0 only.
    request.context_current_position = 2 * MAX_NUM_TOKENS
    pool.kv_cache_map[REQUEST_ID] = _CommitCache()
    batch = ScheduledRequests()
    batch.reset_context_requests([request])

    KVCacheManagerV2.update_context_resources(
        pool, batch, context_ends={REQUEST_ID: MAX_NUM_TOKENS}
    )
    assert committed == [MAX_NUM_TOKENS]

    request.context_current_position = 3 * MAX_NUM_TOKENS
    KVCacheManagerV2.update_context_resources(
        pool, batch, context_ends={REQUEST_ID: 3 * MAX_NUM_TOKENS}
    )
    assert committed == [MAX_NUM_TOKENS, 2 * MAX_NUM_TOKENS, "stop"]

    # Without chunk ends (stock) the live cursor is the frontier.
    committed.clear()
    pool.kv_cache_map[REQUEST_ID] = _CommitCache()
    request.context_current_position = 2 * MAX_NUM_TOKENS
    KVCacheManagerV2.update_context_resources(pool, batch)
    assert committed == [2 * MAX_NUM_TOKENS]


def test_follower_that_cannot_allocate_raises_without_reaping_transfers(monkeypatch):
    """The follower does not loop on reap_context_sends, a PP-group collective."""
    ranks = _make_ranks(
        [_StagePool(0), _StagePool(0)], PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch
    )
    for _ in range(ranks[0]._pp_stage_agree.lag):
        _pp_iteration(ranks)
    follower = ranks[1]
    follower.kv_cache_manager.fail_resizes = 1
    follower._disagg_coordinator = types.SimpleNamespace(
        reap_context_sends=lambda *a: pytest.fail("reaped on one rank"),
        check_transfer_timeouts=lambda *a: pytest.fail("timeouts checked on one rank"),
    )

    with pytest.raises(RuntimeError, match="does not retry by reaping"):
        _pp_iteration(ranks)


@pytest.mark.parametrize(
    "reusable, followers_disagreed, leader_below, lowered",
    [
        (CASES["equal_reuse"], 0, 0, 0),
        (CASES["follower_matched_more"], 0, 1, 0),
        (CASES["follower_matched_less"], 0, 0, 1),
        (CASES["pp4_middle_stage_matched_less"], 1, 0, 1),
    ],
    ids=["equal_reuse", "follower_matched_more", "follower_matched_less", "pp4_middle_less"],
)
def test_first_rank_accounts_the_disagreements_the_agreement_resolved(
    reusable, followers_disagreed, leader_below, lowered, monkeypatch
):
    """The first rank logs and counts first chunks on which the stages disagreed,
    so a GPU run shows whether the agreement changed any chunk (a run that never
    disagreed has not tested the fix)."""
    ranks = _make_ranks(
        [_StagePool(d) for d in reusable], PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch
    )
    lines = []
    monkeypatch.setattr(py_executor_module.logger, "info", lines.append)

    _run_until_admitted(ranks)

    agreement = ranks[0]._pp_stage_agree
    assert agreement.num_admitted == 1
    assert agreement.num_followers_disagreed == followers_disagreed
    assert agreement.num_leader_below == leader_below
    assert agreement.num_lowered == lowered
    assert agreement.max_follower_spread == max(reusable[1:]) - min(reusable[1:])
    if followers_disagreed or leader_below or lowered:
        assert lines == [
            f"PP stage agreement: request {REQUEST_ID} admitted at position "
            f"{min(reusable)}; the first rank's own prefix was {reusable[0]}, follower "
            f"stages reported {min(reusable[1:])}..{max(reusable[1:])} reusable tokens "
            f"({lowered} extra proposal round(s))."
        ]
    else:
        assert lines == []
    summary = agreement.summary()
    assert summary.startswith("PP stage agreement: 1 first chunk(s) admitted; ")
    assert f"follower stages disagreed on {followers_disagreed} (largest spread" in summary
    assert f"reused less than every follower on {leader_below};" in summary
    assert f"admitted below the first rank's own prefix on {lowered} " in summary


def test_answers_to_an_abandoned_proposal_are_dropped():
    agreement = PPStageAgreement(0, 2, 1, True)
    request = _Request(REQUEST_ID, 100)
    depths = iter([64, 32])
    held = agreement.step([request], lambda req, cap: next(depths), lambda req: True)
    assert held == {REQUEST_ID} and agreement.take_proposals() == {REQUEST_ID: 64}
    agreement.advance()
    # The answer to 64 says 32: the first rank claims at most 32 and proposes it.
    agreement.accept(1, {"iteration": 0, "answers": {REQUEST_ID: (64, 32, 96)}}, 0)
    agreement.step([request], lambda req, cap: next(depths), lambda req: True)
    assert agreement.take_proposals() == {REQUEST_ID: 32}
    agreement.advance()
    # An answer to 64 does not confirm 32.
    agreement.accept(1, {"iteration": 1, "answers": {REQUEST_ID: (64, 64, 96)}}, 1)
    assert agreement.step([request], lambda req, cap: 0, lambda req: True) == {REQUEST_ID}
    agreement.advance()
    # Nor does an answer to 32 from another iteration than the one 32 went out in.
    agreement.accept(1, {"iteration": 2, "answers": {REQUEST_ID: (32, 32, 96)}}, 2)
    assert agreement.step([request], lambda req, cap: 0, lambda req: True) == {REQUEST_ID}
    assert agreement.confirmed_depth(REQUEST_ID) is None


# --------------------------------------------------------------------------- several requests, capacity

# A token id encodes its request, so one stage pool can keep one prompt chain per
# request (``_CapacityPool``).
_ID_STRIDE = 10**7


class _CapacityPool(_SwaStagePool):
    """``_SwaStagePool`` with one ``_SwaCore`` per request, a page capacity and index slots.

    Pages a cache holds (its reuse match) or allocated for its chunks count
    against ``capacity_pages``; pages no cache holds are evictable on demand, so
    they do not. ``resize_context`` grows a cache to cover
    ``context_current_position + num_tokens`` as the real one does (its capacity
    is a maximum: a second resize of the same chunk is not charged again).
    ``_create_kv_cache`` fails like the real one when no index slot is free
    (``index_slots``), or ``create_failures`` more times.
    """

    def __init__(self, cores: Dict[int, _SwaCore], capacity_pages: int, index_slots: int = 64):
        super().__init__(next(iter(cores.values())))
        self.cores = cores
        self.capacity_pages = capacity_pages
        self.index_slots = index_slots
        self.create_failures = 0
        self.allocated: Dict[int, int] = {}
        self.target: Dict[int, int] = {}
        self.impl = types.SimpleNamespace(probe_reuse=self._probe)

    def _core_of(self, tokens):
        return self.cores[tokens[0] // _ID_STRIDE]

    def _probe(self, scope, tokens):
        return self._core_of(tokens).match(len(tokens)) if tokens else 0

    def _reuse_token_source(self, req):
        return [req.py_request_id * _ID_STRIDE + i for i in range(req.prompt_len)]

    def used_pages(self) -> int:
        held = sum(len(core.held) for core in self.cores.values())
        return held + sum(self.allocated.values())

    def available_pages_per_level(self):
        return [[self.capacity_pages - self.used_pages()]]

    def gpu_utilization_per_pool_group(self):
        return [round(self.used_pages() / self.capacity_pages, 4)]

    def has_room_for_context_tokens(self, num_tokens):
        # Two life cycles, one page per block each, plus a partial block.
        return self.capacity_pages - self.used_pages() >= 2 * (
            math.ceil(num_tokens / TOKENS_PER_BLOCK) + 1
        )

    def _create_kv_cache(self, request_id, lora_task_id, input_tokens, **kwargs):
        assert request_id not in self.kv_cache_map
        if self.create_failures or len(self.kv_cache_map) >= self.index_slots:
            self.create_failures = max(self.create_failures - 1, 0)
            return None
        core = self.cores[request_id]
        depth = core.match(len(input_tokens)) if input_tokens else 0
        cache = _Cache(depth, core.hold(depth))
        cache.core = core
        self.kv_cache_map[request_id] = cache
        self.claims.append((len(input_tokens) if input_tokens is not None else None, depth))
        return cache

    def resize_context(self, req, num_tokens):
        # One page per block per life cycle (two life cycles in these pools),
        # from the committed prefix (held) up to the cache's capacity.
        rid = req.py_request_id
        start = self.kv_cache_map[rid].num_committed_tokens
        target = max(self.target.get(rid, 0), req.context_current_position + num_tokens)
        pages = 2 * (math.ceil(target / TOKENS_PER_BLOCK) - math.ceil(start / TOKENS_PER_BLOCK))
        if self.used_pages() - self.allocated.get(rid, 0) + pages > self.capacity_pages:
            return False
        previous = self.target.get(rid, 0)
        req.py_ctx_pre_resize_cap = previous if target > previous else None
        self.target[rid] = target
        self.allocated[rid] = pages
        self.resizes.append((rid, req.context_current_position, num_tokens))
        return True

    def revert_allocate_context(self, req):
        """Undo this iteration's growth (``py_ctx_pre_resize_cap``) and suspend."""
        previous, req.py_ctx_pre_resize_cap = req.py_ctx_pre_resize_cap, None
        if previous is None:
            return True
        rid = req.py_request_id
        start = self.kv_cache_map[rid].num_committed_tokens
        self.target[rid] = previous
        self.allocated[rid] = 2 * max(
            0, math.ceil(previous / TOKENS_PER_BLOCK) - math.ceil(start / TOKENS_PER_BLOCK)
        )
        self.kv_cache_map[rid].is_active = False
        return True

    def free_resources(self, req, pin_on_release=False):
        if self.conversation_manager is not None:
            self.conversation_manager.finish_request(req)
        cache = self.kv_cache_map.pop(req.py_request_id, None)
        self.allocated.pop(req.py_request_id, None)
        self.target.pop(req.py_request_id, None)
        if cache is not None:
            cache.core.release(cache.holds)


def _full_attention_core(num_blocks: int) -> _SwaCore:
    return _SwaCore(
        num_blocks, [(None, 0), (None, 0)], [[TOKENS_PER_BLOCK] * num_blocks for _ in range(2)]
    )


def _ranks_with_requests(pools, prompt_lens: Dict[int, int], stage_agree: bool, monkeypatch):
    ranks = _make_ranks(pools, PROMPT_LEN, stage_agree, monkeypatch)
    for executor in ranks:
        executor.active_requests = [_Request(rid, n) for rid, n in prompt_lens.items()]
    return ranks


def _advance_launched_chunks(ranks, wire) -> None:
    """``_update_request_states`` in the same PP iteration: each scheduled context
    request's cursor moves past the chunk it launched, so it is no longer a first
    chunk when the next iteration's agreement step runs."""
    for executor in ranks:
        for req in executor.active_requests:
            chunk = wire.context_chunks.get(req.py_request_id)
            if chunk is not None:
                assert (req.context_current_position, req.context_chunk_size) == chunk
                req.context_current_position += req.context_chunk_size
                req.context_chunk_size = 0


def _finish_done(ranks) -> None:
    """Free every request whose prompt ran to its end, on every stage."""
    for executor in ranks:
        for req in list(executor.active_requests):
            if req.context_current_position >= req.prompt_len:
                executor.kv_cache_manager.free_resources(req)
                executor.active_requests.remove(req)


def _reserved_ids(executor) -> Set[int]:
    return {
        req.py_request_id
        for req in executor.active_requests
        if getattr(req, "py_pp_reserved_depth", None) is not None
        and req.py_request_id in executor.kv_cache_manager.kv_cache_map
    }


def _drive(ranks, max_iterations: int, arrivals=None, before=None) -> Dict[int, int]:
    """Run PP iterations as the loop does until every prompt has run.

    Cursors advance with the launched chunks and finished prompts are freed on
    every stage. Checks, every iteration, that no follower holds a reservation
    the first rank had during its pass but gave back -- the followers' set is
    within the first rank's set of that pass -- and that every follower and the
    first rank start each first chunk from their own reservation at exactly the
    admitted depth (no fresh claim at admission). ``arrivals`` maps an iteration
    to requests that arrive at its start on every stage; ``before(ranks)`` runs
    at the start of every iteration. Returns each request's first-chunk position.
    """
    leader = ranks[0]
    leader_sets = {}
    original_exchange = leader._pp_stage_agree_exchange

    def exchange():
        original_exchange()
        agreement = leader._pp_stage_agree
        leader_sets[agreement.iteration - 1] = set(agreement._proposed) | set(agreement._confirmed)

    leader._pp_stage_agree_exchange = exchange
    for executor in ranks[1:]:
        original = executor.scheduler.adopt_first_context_chunk

        def adopt(req, position, chunk_size, executor=executor, original=original):
            pool = executor.kv_cache_manager
            iteration = executor._pp_stage_agree.iteration
            assert _reserved_ids(executor) <= leader_sets[iteration]
            cache = pool.kv_cache_map[req.py_request_id]
            assert req.py_pp_reserved_depth == position == cache.num_committed_tokens
            num_claims = len(pool.claims)
            adopted = original(req, position, chunk_size)
            assert len(pool.claims) == num_claims, "a fresh claim at admission"
            return adopted

        executor.scheduler.adopt_first_context_chunk = adopt
    first = {}
    for iteration in range(max_iterations):
        for rid, prompt_len in (arrivals or {}).get(iteration, {}).items():
            for executor in ranks:
                executor.active_requests.append(_Request(rid, prompt_len))
        if before is not None:
            before(ranks)
        wire = _pp_iteration(ranks)
        for executor in ranks[1:]:
            assert _reserved_ids(executor) <= leader_sets[iteration]
        for rid, (position, _) in wire.context_chunks.items():
            if rid in first:
                continue
            first[rid] = position
            # Followers resumed their reservation once, at admission; the first
            # rank may have resumed and suspended its own on refused passes.
            assert all(r.kv_cache_manager.kv_cache_map[rid].resumes == 1 for r in ranks[1:])
            assert leader.kv_cache_manager.kv_cache_map[rid].num_committed_tokens == position
        _advance_launched_chunks(ranks, wire)
        _finish_done(ranks)
        if not any(executor.active_requests for executor in ranks) and iteration >= max(
            arrivals or {0: None}
        ):
            return first
    raise AssertionError(f"prompts left after {max_iterations} iterations: {first}")


# Per request, per PP3 stage: (committed blocks, window ends) and the depth every
# stage can claim exactly.
SWA_MULTI = {
    7: ([(512, [8192, 16384]), (768, [8192, 24576]), (768, [4096, 8192, 24576])], 8192),
    8: ([(768, [4096, 24576]), (768, [4096, 12288, 24576]), (640, [4096, 20480])], 4096),
    9: ([(900, [28800]), (900, [28800]), (900, [28800])], 28800),
}


def test_several_swa_requests_each_converge_at_their_own_common_depth(monkeypatch):
    """Three requests agreed at once on PP3 (one round, one lowering on a
    middle stage, one lowering on the last), with cursors advancing as the loop
    does; each is admitted once, at its own greatest common exact depth."""
    pools = [
        _CapacityPool(
            {rid: _hybrid_window_core(*stages[stage]) for rid, (stages, _) in SWA_MULTI.items()},
            capacity_pages=10**9,
        )
        for stage in range(3)
    ]
    ranks = _ranks_with_requests(pools, {rid: PROMPT_LEN for rid in SWA_MULTI}, True, monkeypatch)
    first = {}
    for _ in range(64):
        wire = _pp_iteration(ranks)
        for rid, (position, _) in wire.context_chunks.items():
            assert first.setdefault(rid, position) == position or position > first[rid]
        _advance_launched_chunks(ranks, wire)
        if len(first) == len(SWA_MULTI):
            break
    assert first == {rid: common for rid, (_, common) in SWA_MULTI.items()}
    for pool in pools:
        for rid, (_, common) in SWA_MULTI.items():
            cache = pool.kv_cache_map[rid]
            assert (cache.num_committed_tokens, cache.resumes) == (common, 1)
    agreement = ranks[0]._pp_stage_agree
    assert (agreement.num_admitted, agreement.num_undone, agreement.num_extra_rounds) == (3, 0, 2)


def test_first_rank_reclaim_failure_in_a_lowering_round_agrees_again(monkeypatch):
    """The first rank's own re-claim in a lowering round finds no index
    slot. If its fresh claim in the same step succeeds, the agreement restarts
    from the first rank's own depth; if that fails too, the followers are told
    to drop their reservation (None). Either way the request is admitted later
    at the common depth, with no raise."""
    for failures, proposal in ((1, 16384), (2, None)):
        pools = [
            _CapacityPool({REQUEST_ID: _hybrid_window_core(*stage)}, capacity_pages=10**9)
            for stage in SWA_CASES["follower_window_below_cap"][0]
        ]
        ranks = _ranks_with_requests(pools, {REQUEST_ID: PROMPT_LEN}, True, monkeypatch)
        for _ in range(ranks[0]._pp_stage_agree.lag):
            _pp_iteration(ranks)
        pools[0].create_failures = failures

        wire = _pp_iteration(ranks)

        assert wire.reuse_proposals == {REQUEST_ID: proposal}
        if proposal is None:
            assert REQUEST_ID not in pools[1].kv_cache_map
        wire, _ = _run_until_admitted(ranks)
        assert wire.context_chunks[REQUEST_ID] == (8192, MAX_NUM_TOKENS)
        assert _chunks(ranks) == [(8192, MAX_NUM_TOKENS)] * 2


def test_follower_without_a_free_index_slot_is_asked_again_until_it_has_one(monkeypatch):
    """A follower that cannot create the reservation answers None; the
    first rank proposes the same depth again every lag iterations (no raise, the
    request waits) and admits once the follower has a slot."""
    pools = [
        _CapacityPool({REQUEST_ID: _full_attention_core(768)}, capacity_pages=10**9),
        _CapacityPool({REQUEST_ID: _full_attention_core(768)}, capacity_pages=10**9, index_slots=0),
    ]
    ranks = _ranks_with_requests(pools, {REQUEST_ID: PROMPT_LEN}, True, monkeypatch)
    for _ in range(20):
        assert not _pp_iteration(ranks).context_chunks
    agreement = ranks[0]._pp_stage_agree
    assert agreement.num_retries >= 20 // agreement.lag - 2
    pools[1].index_slots = 64

    wire, _ = _run_until_admitted(ranks)

    assert wire.context_chunks[REQUEST_ID] == (24576, MAX_NUM_TOKENS)


# Two conversations, a 24576-token prefix each (2 pages per block: 1536 pages).
# One prefix plus an 8192-token chunk fits (1536 + 512); two held prefixes plus a
# chunk do not (3072 + 512 > 2600).
_PINNED_PROMPTS = {7: PROMPT_LEN, 8: PROMPT_LEN}


def _pinned_pools(num_stages: int):
    return [
        _CapacityPool({rid: _full_attention_core(768) for rid in _PINNED_PROMPTS}, 2600)
        for _ in range(num_stages)
    ]


def test_adoption_alone_admits_one_of_two_prefixes_that_do_not_fit_together(monkeypatch):
    """Control for the test below: without the stage agreement only the admitted request
    holds its prefix; the other one's pages stay evictable."""
    ranks = _ranks_with_requests(_pinned_pools(2), _PINNED_PROMPTS, False, monkeypatch)
    wire = _adopt_only_pp_iteration(ranks)
    assert wire is not None and len(wire.context_chunks) == 1
    # No agreement: the refused admission was released as before, nothing was
    # recorded for the stage agreement, and the deadlock check counts as stock.
    for executor in ranks:
        assert all(not hasattr(req, "py_pp_page_refused") for req in executor.active_requests)
        assert executor.scheduler.stage_held_context_candidates == 0
    assert [rid for rid in ranks[0].kv_cache_manager.kv_cache_map] == list(wire.context_chunks)


@pytest.mark.parametrize("pp_size", [2, 4])
def test_reservations_that_fill_the_last_tier_still_admit_a_first_chunk(pp_size, monkeypatch):
    """Reservations of unadmitted first chunks held their
    prefixes in the last cache tier; once they left no room for a chunk every
    confirmed admission failed, kept its holds, and the V2 scheduler's deadlock
    check stopped the server. The first rank now gives reservations back when
    its pass stalls on pages, and every follower drops them with it: both
    prompts run to the end, one after the other, each from its reservation."""
    pools = _pinned_pools(pp_size)
    ranks = _ranks_with_requests(pools, _PINNED_PROMPTS, True, monkeypatch)

    first = _drive(ranks, scheduler_v2_module.KVCacheV2Scheduler._DEADLOCK_STALL_ITERS + 64)

    assert first == {7: 24576, 8: 24576}
    agreement = ranks[0]._pp_stage_agree
    assert agreement.num_admitted == 2 and agreement.num_undone == 0
    assert agreement.num_given_back >= 1
    assert all(not pool.kv_cache_map and not pool.allocated for pool in pools)
    assert all(not core.held for pool in pools for core in pool.cores.values())


def test_adoption_alone_runs_both_prompts_that_do_not_fit_together(monkeypatch):
    """Control: without the agreement the same pools run both prompts too."""
    ranks = _ranks_with_requests(_pinned_pools(2), _PINNED_PROMPTS, False, monkeypatch)
    for _ in range(16):
        wire = _adopt_only_pp_iteration(ranks)
        if wire is None:
            break
        _advance_launched_chunks(ranks, wire)
        _finish_done(ranks)
    assert all(not executor.active_requests for executor in ranks)


class _Mapping:
    def __init__(self, pp_size, tp_size=1, cp_size=1, enable_attention_dp=False):
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.enable_attention_dp = enable_attention_dp


@pytest.mark.parametrize(
    "mapping, is_disagg, local_max_tokens, keeps",
    [
        (_Mapping(4), True, float("inf"), True),
        (_Mapping(4), False, float("inf"), False),
        (_Mapping(4), True, 123456, False),
        (_Mapping(4, tp_size=2), True, float("inf"), False),
        (_Mapping(4, enable_attention_dp=True), True, float("inf"), False),
        (_Mapping(1), True, float("inf"), False),
        (_Mapping(1), False, float("inf"), False),
    ],
)
def test_all_swa_stage_keeps_its_quota_only_on_the_agreement_layout(
    mapping, is_disagg, local_max_tokens, keeps
):
    """An all-SWA stage keeps its quota where the followers adopt the first rank's
    batch (the agreement's layout), and is clamped everywhere else."""
    assert kv_v2_module._keeps_all_swa_stage_quota(mapping, is_disagg, local_max_tokens) is keeps


def test_executor_class_default_sends_no_chunks():
    """An executor built without ``__init__`` schedules as without the agreement."""
    assert object.__new__(PyExecutor)._pp_adopt_leader_chunks is False


def test_stage_agreement_records_the_chunks_it_builds_on():
    """``record_first_chunks`` (stage agreement) implies the context chunks."""
    request = _Request(REQUEST_ID, 100)
    scheduled = ScheduledRequests()
    scheduled.reset_context_requests([request])
    wire = SerializableSchedulerOutput.from_scheduler_result(
        scheduled, [], 1, record_first_chunks=True
    )
    assert wire.context_chunks == {
        REQUEST_ID: (request.context_current_position, request.context_chunk_size)
    }


# --------------------------------------------------------------------------- give-back protocol


def _unit_agreement(pp_size=2, lag=1, max_reserved=16):
    return PPStageAgreement(0, pp_size, lag, True, max_reserved=max_reserved)


def test_relieve_halves_newest_unconfirmed_first_and_keeps_the_oldest():
    """Under pressure the first rank keeps the oldest reservation, gives back
    unconfirmed ones before confirmed ones and the newest first, and lowers the
    bound to what it kept; passes without pressure raise the bound by one."""
    agreement = _unit_agreement(lag=1)
    requests = [_Request(rid, 100) for rid in (1, 2, 3, 4, 5, 6)]
    agreement.step(requests, lambda req, cap: 0, lambda req: True)
    agreement.take_proposals()
    agreement.advance()
    # Followers confirmed 1, 2 and 5 (answers to the proposals of iteration 0).
    agreement.accept(1, {"iteration": 0, "answers": {r: (0, 0, 0) for r in (1, 2, 5)}}, 0)
    agreement.step(requests, lambda req, cap: 0, lambda req: True)
    assert sorted(agreement._confirmed) == [1, 2, 5]
    released = []

    given_back = agreement.relieve(True, True, set(), False, released.append)

    # 6 held -> keep 3: the oldest (1) plus the two that rank highest to keep.
    assert agreement.budget == 3
    assert given_back == released == [6, 4, 3]
    assert agreement.take_proposals() == {3: None, 4: None, 6: None}
    assert sorted(agreement._confirmed) == [1, 2, 5] and not agreement._proposed
    for _ in range(2):
        assert agreement.relieve(False, False, set(), False, released.append) == []
    assert agreement.budget == 5
    # No pressure, no give-back, and the bound never exceeds max_reserved.
    for _ in range(20):
        agreement.relieve(False, False, set(), False, released.append)
    assert agreement.budget == agreement.max_reserved and released == [6, 4, 3]


def test_relieve_keeps_the_oldest_even_when_it_is_the_one_unconfirmed():
    agreement = _unit_agreement(lag=1)
    requests = [_Request(rid, 100) for rid in (1, 2, 3)]
    agreement.step(requests, lambda req, cap: 0, lambda req: True)
    agreement.advance()
    agreement.accept(1, {"iteration": 0, "answers": {r: (0, 0, 0) for r in (2, 3)}}, 0)
    agreement.step(requests, lambda req, cap: 0, lambda req: True)
    assert sorted(agreement._confirmed) == [2, 3] and list(agreement._proposed) == [1]

    assert agreement.relieve(True, True, set(), False, lambda rid: None) == [3, 2]
    assert list(agreement._proposed) == [1] and not agreement._confirmed


def test_a_lost_reservation_tells_the_followers_to_drop_theirs():
    """The first rank lost a confirmed reservation (its failed admission released
    it) and cannot claim the request again now: the followers are told to drop
    theirs, so none keeps an orphan hold the first rank no longer counts."""
    agreement = _unit_agreement(lag=1)
    request = _Request(1, 100)
    agreement.step([request], lambda req, cap: 0, lambda req: True)
    agreement.take_proposals()
    agreement.advance()
    agreement.accept(1, {"iteration": 0, "answers": {1: (0, 0, 0)}}, 0)
    agreement.step([request], lambda req, cap: 0, lambda req: True)
    assert agreement.confirmed_depth(1) == 0
    agreement.advance()

    agreement.step([request], lambda req, cap: None, lambda req: False)

    assert agreement.take_proposals() == {1: None}
    assert agreement.confirmed_depth(1) is None and agreement.num_undone == 1


def test_an_admission_clears_the_rotation():
    """A request refused alone goes to the back of the line until any first
    chunk is admitted: the pool changed, so the original order returns."""
    agreement = _unit_agreement(lag=1, max_reserved=1)
    requests = [_Request(rid, 100) for rid in (1, 2, 3)]
    agreement.step(requests, lambda req, cap: 0, lambda req: True)
    agreement.advance()
    agreement.accept(1, {"iteration": 0, "answers": {1: (0, 0, 0)}}, 0)
    agreement.step(requests, lambda req, cap: 0, lambda req: True)
    assert agreement.relieve(True, True, {1}, False, lambda rid: None) == [1]
    agreement.step(requests, lambda req, cap: 0, lambda req: True)  # one step without
    agreement.relieve(False, False, set(), False, lambda rid: None)
    agreement.step(requests, lambda req, cap: 0, lambda req: True)
    assert agreement._order == [2, 3, 1] and list(agreement._proposed) == [2]
    agreement.advance()
    agreement.accept(1, {"iteration": agreement._proposed_at[2], "answers": {2: (0, 0, 0)}}, 1)
    agreement.step(requests, lambda req, cap: 0, lambda req: True)
    assert agreement.check_admitted(2, 0) == (True, None)
    remaining = [requests[0], requests[2]]
    agreement.step(remaining, lambda req, cap: 0, lambda req: True)
    assert agreement._order == [1, 3]


def test_a_pass_that_schedules_work_is_not_pressure(monkeypatch):
    """Refusals are routine on a busy context server; a pass that still scheduled
    work does not give reservations back (only a stalled pass does)."""
    ranks = _make_ranks(
        [_StagePool(0), _StagePool(0)], PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch
    )
    leader = ranks[0]
    leader.active_requests += [_Request(rid, 1000) for rid in (20, 21, 22)]
    leader._pp_stage_agree_exchange()
    agreement = leader._pp_stage_agree
    assert len(agreement._proposed) == 4
    leader.scheduler._ctx_page_refused = True
    busy = ScheduledRequests()
    busy.reset_context_requests([_Request(99, 1000)])

    leader._pp_stage_agree_relieve(busy)
    assert agreement.num_given_back == 0 and agreement.num_pressure_passes == 0

    leader._pp_stage_agree_relieve(ScheduledRequests())
    assert agreement.num_given_back == 2 and agreement.budget == 2
    assert sorted(agreement.take_proposals()) == [7, 20, 21, 22]


@pytest.mark.parametrize(
    "stalled, refused, others, waiting, given_back",
    [
        (True, {1}, False, 1, []),  # the only request at all: kept, as stock retries it
        (True, {1}, True, 1, [1]),  # other work blocked by its pins: one pass without
        (True, {1}, False, 2, [1]),  # another waiting first chunk to try
        (True, set(), True, 2, []),  # not refused itself (unconfirmed): kept
        (False, {1}, True, 2, []),  # preemption only: the last one is never given back
    ],
)
def test_the_last_reservation_is_given_back_only_when_that_can_help(
    stalled, refused, others, waiting, given_back
):
    agreement = _unit_agreement(max_reserved=1)
    requests = [_Request(rid, 100) for rid in range(1, waiting + 1)]
    agreement.step(requests, lambda req, cap: 0, lambda req: True)
    assert agreement.relieve(True, stalled, refused, others, lambda rid: None) == given_back
    if given_back:
        # One step with no reservation at all, then the others before it.
        assert agreement.step(requests, lambda req, cap: 0, lambda req: True) == {
            r.py_request_id for r in requests
        }
        assert not agreement._proposed
        agreement.relieve(False, False, set(), others, lambda rid: None)
        agreement.step(requests, lambda req, cap: 0, lambda req: True)
        expected = [2] if waiting == 2 else [1]
        assert sorted(agreement._proposed) == expected


def test_an_answer_to_a_proposal_given_back_does_not_confirm_the_same_depth_again():
    """A follower's answer to a proposal the first rank gave back (and then
    proposed again at the same depth) is still in flight when the new proposal
    is out; it must not count, or the first rank would admit at a depth the
    follower, which released and re-claimed in between, may no longer hold."""
    agreement = _unit_agreement(lag=4)
    a, b = _Request(1, 100), _Request(2, 100)
    agreement.step([a, b], lambda req, cap: 64, lambda req: True)
    assert agreement.take_proposals() == {1: 64, 2: 64}
    agreement.advance()
    agreement.step([a, b], lambda req, cap: 64, lambda req: True)
    assert agreement.relieve(True, True, set(), False, lambda rid: None) == [2]
    assert agreement.take_proposals() == {2: None}
    agreement.advance()
    agreement.step([a, b], lambda req, cap: 64, lambda req: True)
    agreement.relieve(False, False, set(), False, lambda rid: None)
    agreement.advance()
    # Iteration 3: the bound allows two again; 2 is proposed at 64 once more.
    agreement.step([a, b], lambda req, cap: 64, lambda req: True)
    assert agreement.take_proposals() == {2: 64}
    agreement.advance()
    # Iteration 4: the report of iteration 0 arrives, answering the old 64.
    agreement.accept(1, {"iteration": 0, "answers": {1: (64, 64, 64), 2: (64, 64, 64)}}, 0)
    agreement.step([a, b], lambda req, cap: 64, lambda req: True)
    assert agreement.confirmed_depth(1) == 64
    assert agreement.confirmed_depth(2) is None
    agreement.advance()
    for due in (1, 2):
        agreement.accept(1, {"iteration": due, "answers": {}}, due)
        agreement.step([a, b], lambda req, cap: 64, lambda req: True)
        agreement.advance()
    # Iteration 7: the follower's answer to the new proposal: it got only 48.
    agreement.accept(1, {"iteration": 3, "answers": {2: (64, 48, 48)}}, 3)
    agreement.step([a, b], lambda req, cap: 48, lambda req: True)
    assert agreement.confirmed_depth(2) is None
    assert agreement.take_proposals() == {2: 48}


# Request 7 holds a prefix too long for itself and a chunk to fit next to anything;
# request 8 alone fits. Pages: 7 holds 1536, 8 holds 128, a chunk needs up to 512.
_ROTATION_PROMPTS = {7: PROMPT_LEN, 8: 2048 + 4000}


def _rotation_pools(num_stages: int, capacity: int = 1700):
    return [
        _CapacityPool({7: _full_attention_core(768), 8: _full_attention_core(64)}, capacity)
        for _ in range(num_stages)
    ]


def test_a_reservation_refused_alone_is_tried_after_the_others(monkeypatch):
    """The oldest first chunk does not fit even alone; the stock pass skips it
    and admits the next. The first rank gives the refused one back and tries
    the other, which runs; the one that never fits stays waiting."""
    stock = _ranks_with_requests(_rotation_pools(2), _ROTATION_PROMPTS, False, monkeypatch)
    wire = _adopt_only_pp_iteration(stock)
    assert list(wire.context_chunks) == [8]

    pools = _rotation_pools(2)
    ranks = _ranks_with_requests(pools, _ROTATION_PROMPTS, True, monkeypatch)
    for _ in range(40):
        wire = _pp_iteration(ranks)
        _advance_launched_chunks(ranks, wire)
        _finish_done(ranks)
        if all(r.py_request_id == 7 for r in ranks[0].active_requests):
            break
    assert [r.py_request_id for r in ranks[0].active_requests] == [7]
    agreement = ranks[0]._pp_stage_agree
    assert agreement.num_admitted == 1 and agreement.num_refused_alone >= 1


def test_a_pool_that_admits_nothing_still_trips_the_deadlock_check(monkeypatch):
    """Request 7 alone never fits. Stock, the V2 scheduler's deadlock check stops
    the server; with the agreement it does too (the reservation stays: giving back
    the only candidate cannot help), after the same number of stalled passes."""
    monkeypatch.setattr(scheduler_v2_module.KVCacheV2Scheduler, "_DEADLOCK_STALL_ITERS", 40)
    prompts = {7: PROMPT_LEN}
    stock = _ranks_with_requests(_rotation_pools(2), prompts, False, monkeypatch)
    with pytest.raises(RuntimeError, match="V2 scheduler deadlock: 0 generation and 1 context"):
        for _ in range(60):
            _adopt_only_pp_iteration(stock)
    ranks = _ranks_with_requests(_rotation_pools(2), prompts, True, monkeypatch)
    with pytest.raises(RuntimeError, match="V2 scheduler deadlock: 0 generation and 1 context"):
        for _ in range(80):
            _pp_iteration(ranks)
    assert ranks[0]._pp_stage_agree.num_given_back == 0


def test_a_sweep_with_nothing_admissible_alone_counts_the_held_chunks(monkeypatch):
    """Two first chunks, neither fits even alone. Each is refused alone in turn
    and the reservations rotate; held passes would reset the deadlock check
    forever, so after a full sweep the held first chunks count as candidates
    and the check fires as it does stock."""
    monkeypatch.setattr(scheduler_v2_module.KVCacheV2Scheduler, "_DEADLOCK_STALL_ITERS", 60)
    prompts = {7: PROMPT_LEN, 8: PROMPT_LEN}

    def pools():
        return [
            _CapacityPool({rid: _full_attention_core(768) for rid in prompts}, 1700)
            for _ in range(2)
        ]

    stock = _ranks_with_requests(pools(), prompts, False, monkeypatch)
    with pytest.raises(RuntimeError, match="V2 scheduler deadlock: 0 generation and 2 context"):
        for _ in range(100):
            _adopt_only_pp_iteration(stock)
    ranks = _ranks_with_requests(pools(), prompts, True, monkeypatch)
    with pytest.raises(RuntimeError, match="V2 scheduler deadlock: 0 generation and 2 context"):
        for _ in range(400):
            _pp_iteration(ranks)
    agreement = ranks[0]._pp_stage_agree
    assert agreement.num_sweeps >= 1 and agreement.exhausted
    assert ranks[0].scheduler.stage_held_context_candidates >= 1


def test_a_started_prompt_blocked_by_reserved_prefixes_gets_a_pass_without_them(monkeypatch):
    """Request 9 is in the middle of its prompt; two later first chunks reserve
    prefixes that leave its next chunk no pages (stock: they stay evictable and
    9 runs). The first rank halves the reservations, then gives the last one
    back once it is itself refused, so 9 runs in a pass with no reservation; all
    three prompts finish, each first chunk from its reservation."""
    prompts = {9: 2 * MAX_NUM_TOKENS}
    later = {7: 24576 + MAX_NUM_TOKENS, 8: 24576 + MAX_NUM_TOKENS}
    cores = {rid: _full_attention_core(768) for rid in later}
    cores[9] = _full_attention_core(0)
    pools = [_CapacityPool(dict(cores), 2100) for _ in range(2)]
    ranks = _ranks_with_requests(pools, prompts, True, monkeypatch)
    lag = ranks[0]._pp_stage_agree.lag

    first = _drive(ranks, 200, arrivals={lag + 1: later})

    assert first == {9: 0, 7: 24576, 8: 24576}
    agreement = ranks[0]._pp_stage_agree
    assert agreement.num_given_back >= 2 and agreement.num_refused_alone >= 1


@pytest.mark.parametrize("pp_size", [2, 3])
def test_give_back_with_sliding_window_stages_keeps_every_admission_exact(pp_size, monkeypatch):
    """The give-back on SWA stages that also lower the depth: three requests
    agreed at once; the first rank's pool first has room for one held prefix
    and a chunk only, then for everything. The ones given back are agreed
    again (with their lowering rounds) and every request still runs from its
    own reservation at its greatest common exact depth."""
    pools = [
        _CapacityPool(
            {rid: _hybrid_window_core(*stages[stage]) for rid, (stages, _) in SWA_MULTI.items()},
            capacity_pages=10**9,
        )
        for stage in range(pp_size)
    ]
    pools[0].capacity_pages = 1000
    agreement = None

    def widen_after_the_first_give_back(ranks):
        if agreement.num_given_back:
            pools[0].capacity_pages = 10**9

    ranks = _ranks_with_requests(pools, {rid: PROMPT_LEN for rid in SWA_MULTI}, True, monkeypatch)
    agreement = ranks[0]._pp_stage_agree

    first = _drive(ranks, 200, before=widen_after_the_first_give_back)

    expected = {rid: common for rid, (_, common) in SWA_MULTI.items()}
    if pp_size == 2:
        # The first two stages only: request 8's last-stage lowering is gone.
        expected[8] = 24576
    assert first == expected
    assert agreement.num_given_back >= 1
    assert all(not core.held for pool in pools for core in pool.cores.values())


# --------------------------------------------------------------------------- partial reuse


def test_swa_core_partial_reuse_matches_mid_block_endpoints():
    """The port with partial reuse: a chain committed to 16400 (mid-block) is
    reused to 16400 only with partial reuse; a claim capped mid-block is
    matched mid-block, and a page that covers less of its block than the match
    cuts the match to its coverage (``pruneMatch``)."""
    on = _chain_core(16400, [8192, 16400], partial=True)
    off = _chain_core(16400, [8192, 16400], partial=False)
    assert on.match(PROMPT_LEN - 1) == 16400
    assert off.match(PROMPT_LEN - 1) == 16384
    # A cap between window ends falls to the window at 8192 either way.
    assert on.match(12300) == off.match(12300) == 8192
    # A longer chain with a window ending mid-block at 12300: the SWA page of
    # block 384 covers 12 tokens, so a claim at 14336 is cut to 12300 even
    # without partial reuse (a coverage cut), and a capped claim at 12300 is
    # matched there only with partial reuse.
    long_on = _chain_core(24576, [12300, 24576], partial=True)
    long_off = _chain_core(24576, [12300, 24576], partial=False)
    assert long_on.match(14336) == long_off.match(14336) == 12300
    assert long_on.match(12300) == 12300
    assert long_off.match(12300) == 12288
    # Termination of the rounds needs only that a claim never exceeds its cap.
    # With partial reuse a claim also never shrinks when the cap grows and a
    # claim capped at a returned depth returns it again, so the rounds end at
    # the greatest common depth. Without it neither holds at mid-block points:
    # a cap at a partial node's end matches the node by key (16400) but one a
    # few tokens deeper misses it (16384); a coverage cut (12300) is not
    # reproduced by a claim capped there (12288). Those cost a round or a few
    # tokens of reuse, never an inexact claim.
    caps = sorted({*range(0, PROMPT_LEN, 13), 8192, 12288, 12300, 14336, 16384, 16400, 24576})
    for core in (on, off, long_on, long_off):
        depths = [core.match(cap) for cap in caps]
        assert all(d <= cap for d, cap in zip(depths, caps))
        if core.partial:
            assert depths == sorted(depths)
            assert all(core.match(d) == d for d in set(depths))
    assert off.match(16400) == 16400 and off.match(16410) == 16384


# Per stage: (chain end, window ends); partial reuse on every stage or on none.
PARTIAL_CASES = {
    # The first rank's chain ends mid-block; a follower whose window at that
    # end has pages confirms it in one round, mid-block, only with partial reuse.
    "mid_block_end_held": ([(16400, [8192, 16400]), (24576, [8192, 16400, 24576])], 16400, 16384),
    # The follower lacks that window: both fall back to 8192.
    "mid_block_end_not_held": ([(16400, [8192, 16400]), (24576, [8192, 24576])], 8192, 8192),
    # PP3: a coverage cut lands mid-block on a follower (12300); with partial
    # reuse every stage claims 12300 exactly, without it the first rank's own
    # claim there falls to 12288 and the stages agree on that.
    "pp3_coverage_cut": (
        [(16400, [12300, 16400]), (24576, [12300, 24576]), (24576, [12300, 20480, 24576])],
        12300,
        12288,
    ),
}


@pytest.mark.parametrize("partial", [True, False], ids=["partial", "no_partial"])
@pytest.mark.parametrize(
    "stages, common_partial, common_full", list(PARTIAL_CASES.values()), ids=list(PARTIAL_CASES)
)
def test_rounds_converge_with_partial_reuse(
    stages, common_partial, common_full, partial, monkeypatch
):
    """With partial reuse the depths are token-granular, not block-granular,
    and a coverage cut can land mid-block. The proposal rounds still end at
    the greatest depth every stage claims exactly, and every stage runs the
    chunk from its reservation there."""
    pools = [_SwaStagePool(_chain_core(end, ends, partial)) for end, ends in stages]
    ranks = _make_ranks(pools, PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch)
    common = common_partial if partial else common_full

    wire, iterations = _run_until_admitted(ranks)

    assert wire.context_chunks[REQUEST_ID][0] == common
    assert all(chunk[0] == common for chunk in _chunks(ranks))
    for pool in pools:
        cache = pool.kv_cache_map[REQUEST_ID]
        assert (cache.num_committed_tokens, cache.resumes) == (common, 1)
        # Greatest: no deeper endpoint up to the first rank's own claim is exact
        # on every stage (block ends, window ends and mid-block cuts checked).
    first = pools[0].core.match(PROMPT_LEN - 1)
    candidates = {e for _, ends in stages for e in ends} | set(
        range(0, first + 1, TOKENS_PER_BLOCK)
    )
    deeper = [e for e in candidates if common < e <= first]
    assert not [e for e in deeper if all(p.core.match(e) == e for p in pools)]
    assert iterations % ranks[0]._pp_stage_agree.lag == 0


# --------------------------------------------------------------------------- refused and supported configurations


def test_agreement_is_skipped_with_a_mamba_hybrid_kv_cache_manager():
    """MambaHybridCacheManagerV2.update_context_resources has no context_ends, so
    the stage agreement's commit would raise TypeError in the PP loop; such a
    server runs without the agreement instead."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
    )

    assert MambaHybridCacheManagerV2._supports_pp_context_ends is False
    executor = _rank(1, 2, _StagePool(0), 100, _Mailbox())
    executor.kv_cache_manager = object.__new__(MambaHybridCacheManagerV2)
    assert executor._create_pp_stage_agreement() is None


def test_agreement_is_skipped_with_a_manager_that_cannot_stop_the_commit():
    """Any other V2 manager whose update_context_resources takes no context_ends
    (e.g. a sparse-attention manager overriding it), target or paired draft."""

    class _NoContextEnds(_StagePool):
        _supports_pp_context_ends = False

        def update_context_resources(self, scheduled_batch):
            pass

    executor = _rank(1, 2, _NoContextEnds(0), 100, _Mailbox())
    assert executor._create_pp_stage_agreement() is None
    executor.kv_cache_manager = _StagePool(0)
    executor.enable_joint_kv_cache_reuse = True
    executor.draft_kv_cache_manager = _NoContextEnds(0)
    assert executor._create_pp_stage_agreement() is None
    executor.draft_kv_cache_manager = _StagePool(0)
    assert executor._create_pp_stage_agreement() is not None


def test_the_v2_manager_declares_context_ends_and_its_overriders_do_not():
    """The capability check reads a class flag, not the method signature: the
    base V2 manager's update_context_resources takes context_ends, and the
    built-in subclasses that override it without the argument opt out."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
    )

    assert KVCacheManagerV2._supports_pp_context_ends is True
    assert MambaHybridCacheManagerV2._supports_pp_context_ends is False
    try:
        from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
            DeepseekV4CacheManager,
        )
    except ImportError:  # pragma: no cover - optional sparse-attention deps
        return
    assert DeepseekV4CacheManager._supports_pp_context_ends is False


def test_try_commit_blocks_stops_on_the_old_test_without_commit_end():
    """Without commit_end the commit stops exactly when context_remaining_length
    is 0, as before the stage agreement; with commit_end (the stage-agreement
    path) it stops once the handled chunk reaches the prompt end."""

    class _Committed:
        def __init__(self, committed):
            self.num_committed_tokens = committed
            self.commits = []
            self.stopped = False

        def commit(self, tokens):
            self.commits.append(tokens)

        def stop_committing(self):
            self.stopped = True

    def run(position, remaining, committed, commit_end=None):
        manager = object.__new__(KVCacheManagerV2)
        manager.enable_block_reuse = True
        manager._can_publish_block_reuse = True
        cache = _Committed(committed)
        manager.kv_cache_map = {5: cache}
        manager._reuse_token_source = lambda req: "tokens"
        manager._augment_tokens_for_block_reuse = lambda src, req, start, end: (start, end)
        req = types.SimpleNamespace(
            py_request_id=5,
            is_dummy_request=False,
            context_current_position=position,
            context_remaining_length=remaining,
        )
        manager.try_commit_blocks(req, commit_end=commit_end)
        return cache

    # unset path (stock): commit to the live cursor, stop iff remaining == 0
    cache = run(position=96, remaining=0, committed=64)
    assert cache.commits == [(64, 96)] and cache.stopped
    cache = run(position=64, remaining=32, committed=64)
    assert cache.commits == [] and not cache.stopped
    # a cursor past the prompt end (remaining < 0) does not stop, as before
    cache = run(position=96, remaining=-8, committed=64)
    assert cache.commits == [(64, 96)] and not cache.stopped
    # stage-agreement path: commit to the handled chunk, stop at the prompt end
    cache = run(position=128, remaining=0, committed=64, commit_end=96)
    assert cache.commits == [(64, 96)] and not cache.stopped
    cache = run(position=128, remaining=0, committed=96, commit_end=128)
    assert cache.commits == [(96, 128)] and cache.stopped


def test_per_conversation_reuse_registers_the_admitted_turn_on_every_stage(monkeypatch):
    """PER_CONVERSATION is supported under the agreement. Two turns of one
    conversation wait at once and their held prefixes do not fit together, so
    one is given back. A reservation registers its turn as the conversation's
    current request (``ConversationManager.prepare_request``), a release clears
    it, and the admission re-registers it; so whichever turn runs is the current
    one on every stage when its first chunk runs, which is what its drop plan at
    context end needs. (Drop plans never drop a held page:
    ``PlannedDropHandle::drop`` only unlists pages that are droppable.)"""
    pools = _pinned_pools(2)
    for pool in pools:
        pool.block_reuse_policy = BlockReusePolicy.PER_CONVERSATION
        pool.conversation_manager = kv_v2_module.ConversationManager(2)
    ranks = _ranks_with_requests(pools, _PINNED_PROMPTS, True, monkeypatch)
    for executor in ranks:
        for req in executor.active_requests:
            req.py_conversation_params = types.SimpleNamespace(conversation_id="conversation")
    assert ranks[0]._create_pp_stage_agreement() is not None
    admitted = []
    for _ in range(64):
        wire = _pp_iteration(ranks)
        for rid, (position, _) in wire.context_chunks.items():
            if position == 24576:
                admitted.append(rid)
                states = [
                    pool.conversation_manager._conversation_states["conversation"] for pool in pools
                ]
                assert [state.current_request_id for state in states] == [rid] * len(pools)
        _advance_launched_chunks(ranks, wire)
        _finish_done(ranks)
        if not ranks[0].active_requests:
            break
    assert admitted == [7, 8]
    assert ranks[0]._pp_stage_agree.num_given_back >= 1
    assert all(not pool.conversation_manager._conversation_states for pool in pools)


# --------------------------------------------------------------------------- follower page pressure


def _follower_callbacks(rooms: Dict[int, bool]):
    """Reserve / release / probe / room callbacks of a follower's answer step."""
    released = []

    def reserve(req, cap):
        set_reserved_depth(req, cap)
        return cap

    def release(req):
        released.append(req.py_request_id)
        set_reserved_depth(req, None)

    return (
        reserve,
        release,
        (lambda req: 0),
        (lambda req: rooms.get(req.py_request_id, True)),
        released,
    )


def test_a_follower_declines_a_further_reservation_that_leaves_no_room():
    """A follower that already holds a reservation releases a further one the
    pools have no room for and answers that it claimed nothing (the report lists
    it as declined); it never declines its only reservation. Drops are applied
    before the proposals of the same message."""
    follower = PPStageAgreement(1, 2, 1, True)
    requests = {rid: _Request(rid, 100) for rid in (1, 2, 3)}
    reserve, release, probe, has_room, released = _follower_callbacks({1: False, 2: False, 3: True})

    follower.answer_proposals({1: 0, 2: 0, 3: 0}, requests, reserve, release, probe, has_room)
    report = follower.build_report()

    # 1 is the only reservation when it is made, so it is kept without room.
    assert report["answers"] == {1: (0, 0, 0), 2: (0, None, 0), 3: (0, 0, 0)}
    assert report["declined"] == [2] and released == [2] and follower.num_declined == 1
    assert requests[2].py_pp_reserved_depth is None
    follower.advance()

    # The first rank gave 1 back and proposed 2 again in the same message: the
    # drop comes first, but 3 is still held, so 2 is declined again.
    follower.answer_proposals({2: 0, 1: None}, requests, reserve, release, probe, has_room)
    report = follower.build_report()
    assert report["answers"] == {2: (0, None, 0)} and report["declined"] == [2]
    follower.advance()

    follower.answer_proposals({2: 0, 3: None}, requests, reserve, release, probe, has_room)
    report = follower.build_report()
    assert report["answers"] == {2: (0, 0, 0)} and "declined" not in report


def test_a_follower_decline_is_page_pressure_on_the_first_rank():
    """A decline answers the current proposal with nothing claimed, so the first
    rank cannot confirm it and asks again; it also counts as page pressure: the
    reservations are halved, the declined ones given back first (the oldest is
    kept). A decline answering an abandoned proposal is ignored."""
    agreement = _unit_agreement(pp_size=3, lag=1)
    requests = [_Request(rid, 100) for rid in (1, 2, 3, 4)]
    agreement.step(requests, lambda req, cap: 0, lambda req: True)
    agreement.take_proposals()
    agreement.advance()
    # 1 confirmed, 2 declined; the answers for 3 and 4 are not complete yet
    # (the second follower has not answered them), so both stay unconfirmed.
    agreement.accept(
        1,
        {"iteration": 0, "answers": {1: (0, 0, 0), 2: (0, None, 0)}, "declined": [2]},
        0,
    )
    agreement.accept(2, {"iteration": 0, "answers": {1: (0, 0, 0), 2: (0, 0, 0)}}, 0)
    agreement.step(requests, lambda req, cap: 0, lambda req: True)
    assert sorted(agreement._confirmed) == [1] and sorted(agreement._proposed) == [2, 3, 4]
    assert agreement.num_retries == 1 and agreement.num_declines_received == 1

    given_back = agreement.relieve(False, False, set(), False, lambda rid: None)

    # Of the unconfirmed ones the declined goes first, then the newest.
    assert agreement.num_pressure_passes == 1 and agreement.budget == 2
    assert given_back == [2, 4]
    assert agreement.take_proposals() == {2: None, 4: None}
    # Consumed: the next pass without pressure raises the bound again.
    assert agreement.relieve(False, False, set(), False, lambda rid: None) == []
    assert agreement.budget == 3
    # A decline of an answer to an abandoned proposal does not count.
    agreement.advance()
    agreement.accept(1, {"iteration": 1, "answers": {2: (0, None, 0)}, "declined": [2]}, 1)
    assert agreement.num_declines_received == 1
    assert agreement.relieve(False, False, set(), False, lambda rid: None) == []


def test_a_pass_with_kv_transfers_in_flight_is_not_stalled(monkeypatch):
    """Pages held by a KV transfer come back when it lands, so a pass that
    scheduled nothing while the pool refused pages is not a stalled one then
    (as for the V2 deadlock check); without the transfer it is."""
    ranks = _make_ranks([_StagePool(0), _StagePool(0)], PROMPT_LEN, True, monkeypatch)
    leader = ranks[0]
    leader._pp_stage_agree_exchange()
    seen = []
    monkeypatch.setattr(
        leader._pp_stage_agree,
        "relieve",
        lambda pressure, stalled, refused, others, release: seen.append((pressure, stalled)) or [],
    )
    inflight = types.SimpleNamespace(has_any_inflight_requests=lambda: True)
    for manager, expected in ((inflight, (False, False)), (None, (True, True))):
        leader.scheduler._async_transfer_manager = manager
        leader.scheduler._ctx_page_refused = True
        leader._pp_stage_agree_relieve(ScheduledRequests())
        assert seen.pop() == expected


@pytest.mark.parametrize("reserved_first", [False, True])
def test_the_pass_refusal_flag_keeps_every_refusal_of_the_pass(reserved_first, monkeypatch):
    """``context_pages_refused_last_pass`` (and the stock ContextPageBackoff that
    reads the same flag) is the OR of every page refusal in the pass, whether a
    request holding a stage reservation was tried before or after the refused
    one, and whichever of the two was refused."""
    pool = _StagePool(0)
    scheduler = _scheduler(pool)
    reserved, plain = _Request(1, 1000), _Request(2, 1000)
    set_reserved_depth(reserved, 0)
    budget = scheduler_v2_module.BudgetTracker(MAX_NUM_TOKENS, 8)
    for refused in (plain, reserved):
        for req in (reserved, plain):
            pool.kv_cache_map.pop(req.py_request_id, None)
            req.context_current_position = 0
        set_reserved_depth(reserved, 0)
        scheduler._ctx_page_refused = False
        order = [reserved, plain] if reserved_first else [plain, reserved]
        for req in order:
            pool.fail_resizes = 1 if req is refused else 0
            scheduler._try_schedule_context(req, budget, lambda req: False)
        assert scheduler.context_pages_refused_last_pass(), (refused.py_request_id, order)
        assert getattr(reserved, "py_pp_page_refused", False) == (refused is reserved)


@pytest.mark.parametrize("capacity", [1550, 1620])
def test_adoption_alone_undoes_a_larger_local_chunk_before_adopting(capacity, monkeypatch):
    """The leader-batch adoption alone: the follower's own pass sized
    started request 9's chunk with the whole budget (8192) where the first rank
    gave it 7192 next to first chunk 7. The adoption only grows a cache, so the
    follower undoes its local growth before it adopts any chunk: 9 ends at the
    first rank's 15384 tokens and 7 fits next to it (1550 pages hold the first
    rank's batch, not the follower's 8192-token chunk plus 7). Started request
    5, whose local chunk matches, keeps its allocation."""
    pool = _CapacityPool({rid: _full_attention_core(0) for rid in (9, 7, 5)}, capacity)
    follower = _rank(1, 2, pool, PROMPT_LEN, _Mailbox())
    follower._revert_ctx_alloc = types.MethodType(PyExecutor._revert_ctx_alloc, follower)
    follower._pp_adopt_leader_chunks = True
    request_9, request_7 = _Request(9, 3 * MAX_NUM_TOKENS), _Request(7, 1000)
    request_5 = _Request(5, 2 * MAX_NUM_TOKENS)
    follower.active_requests = [request_9, request_7, request_5]
    # 9's and 5's first chunks ran (8192 tokens allocated each), then the local
    # pass of this iteration sized 9's second chunk at 8192 and 5's at 100.
    for req, local_chunk in ((request_9, MAX_NUM_TOKENS), (request_5, 100)):
        assert follower.scheduler.adopt_context_chunk(req, MAX_NUM_TOKENS)
        req.context_current_position = MAX_NUM_TOKENS
        req.py_ctx_pre_resize_cap = None
        assert pool.resize_context(req, local_chunk)
        req.context_chunk_size = local_chunk
    local = types.SimpleNamespace(context_requests=[request_9, request_5], generation_requests=[])
    batch = ScheduledRequests()
    batch.reset_context_requests([request_7, request_9, request_5])
    batch.leader_context_chunks = {
        7: (0, 1000),
        9: (MAX_NUM_TOKENS, MAX_NUM_TOKENS - 1000),
        5: (MAX_NUM_TOKENS, 100),
    }

    PyExecutor._pp_adopt_leader_schedule(follower, batch, local)

    assert pool.target == {9: 2 * MAX_NUM_TOKENS - 1000, 7: 1000, 5: MAX_NUM_TOKENS + 100}
    assert (request_9.context_current_position, request_9.context_chunk_size) == (
        MAX_NUM_TOKENS,
        MAX_NUM_TOKENS - 1000,
    )
    assert pool.used_pages() == 2 * (
        math.ceil((2 * MAX_NUM_TOKENS - 1000) / TOKENS_PER_BLOCK)
        + math.ceil(1000 / TOKENS_PER_BLOCK)
        + math.ceil((MAX_NUM_TOKENS + 100) / TOKENS_PER_BLOCK)
    )


@pytest.mark.parametrize(
    "available, room",
    [
        # GPU only: full-attention group 0, sliding-window group 1; 8192 tokens
        # need 257 blocks in each.
        ([[257, 257]], True),
        ([[257, 256]], False),
        # A host tier below GPU (one cold group serving both life cycles) needs
        # the room too: the pages the chunk evicts move down.
        ([[300, 300], [514]], True),
        ([[300, 300], [513]], False),
    ],
)
def test_room_for_a_chunk_counts_every_attention_life_cycle_on_every_level(
    available, room, monkeypatch
):
    # The compiled module's cache levels are plain ints, GPU first.
    monkeypatch.setattr(kv_v2_module, "CacheLevel", int)
    monkeypatch.setattr(kv_v2_module, "GPU_LEVEL", 0)
    manager = object.__new__(KVCacheManagerV2)
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager.impl = types.SimpleNamespace(
        cache_tier_list=[None] * len(available),
        get_storage_statistics=lambda level: [
            types.SimpleNamespace(available=n) for n in available[level]
        ],
    )
    manager._stats_life_cycle_metadata = lambda: {
        0: (0, None, "attention"),
        1: (1, SWA_WINDOW, "attention"),
    }
    manager._cold_pool_group_membership = lambda: ((0, frozenset({0, 1})),)

    assert manager.available_pages_per_level() == available
    assert manager.has_room_for_context_tokens(MAX_NUM_TOKENS) is room


def test_connector_reaches_the_agreement_when_the_executor_is_built(monkeypatch):
    """``PyExecutor.__init__`` creates the stage agreement before it assigns
    ``self.kv_connector_manager``; the connector it was given is what the
    agreement's check sees (built through the real constructor, in its real
    order, up to the agreement), and that check keeps a server with a
    connector off the agreement."""

    class _Stop(Exception):
        pass

    seen = []

    def spy(self, kv_connector_manager=None):
        seen.append((kv_connector_manager, self._pp_disaggregated))
        raise _Stop

    create_agreement = PyExecutor._create_pp_stage_agreement
    monkeypatch.setattr(PyExecutor, "_create_pp_stage_agreement", spy)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "Stream", MagicMock)
    monkeypatch.setattr(torch.cuda, "Event", MagicMock)
    dist = MagicMock(rank=0, pp_size=2, pp_rank=0, tp_size=1, cp_size=1)
    engine = MagicMock(enable_attention_dp=False)
    # The model declares no step-loop hooks (a bare MagicMock attribute is not a RuntimeHooks).
    engine.model.runtime_hooks = None
    engine.get_max_num_sequences.return_value = 8
    engine.llm_args = MagicMock(
        attention_dp_config=None,
        batch_wait_timeout_iters=0,
        batch_wait_max_tokens_ratio=0.0,
        max_num_tokens=MAX_NUM_TOKENS,
        batch_wait_timeout_ms=0,
    )
    pool = _StagePool(0)
    pool.event_buffer_max_size = 0
    pool.enable_partial_reuse = False
    resource_manager = MagicMock()
    resource_manager.resource_managers = {ResourceManagerType.KV_CACHE_MANAGER: pool}
    connector = MagicMock()

    with pytest.raises(_Stop):
        PyExecutor(
            resource_manager,
            MagicMock(),
            engine,
            MagicMock(),
            dist,
            8,
            kv_connector_manager=connector,
            start_worker=False,
        )
    assert seen == [(connector, False)]
    executor = _rank(1, 2, _StagePool(0), 100, _Mailbox())
    assert create_agreement(executor) is not None
    assert create_agreement(executor, connector) is None


# --------------------------------------------------------------------------- follower early release


def test_follower_early_release_follows_the_agreement_and_block_reuse():
    """A follower releases early exactly where the agreement runs with block
    reuse (``PyExecutor.__init__``): the early-release flag is the agreement's
    ``reuse_enabled``."""
    assert PPStageAgreement(1, 4, 1, True).reuse_enabled is True
    assert PPStageAgreement(1, 4, 1, False).reuse_enabled is False


class _Recorder:
    """Reserve / release / probe callbacks of a follower that record every call.

    ``claims`` maps a request id to the depth its claims land at (a cap above
    it is pruned to it; None: the claim fails).
    """

    def __init__(self, claims=None, rooms=None):
        self.claims = claims or {}
        self.rooms = rooms or {}
        self.calls = []

    def reserve(self, req, cap):
        self.calls.append(("reserve", req.py_request_id, cap))
        rid = req.py_request_id
        top = self.claims.get(rid, cap)
        depth = None if top is None else (top if cap is None else min(top, cap))
        set_reserved_depth(req, depth)
        return depth

    def release(self, req):
        self.calls.append(("release", req.py_request_id))
        set_reserved_depth(req, None)

    def probe(self, req):
        return 1000 + req.py_request_id

    def has_room(self, req):
        return self.rooms.get(req.py_request_id, True)


def _follower_requests(held: Dict[int, Optional[int]]):
    requests = {}
    for rid, depth in held.items():
        req = _Request(rid, 4 * MAX_NUM_TOKENS)
        set_reserved_depth(req, depth)
        requests[rid] = req
    return requests


def _answer_both_ways(held, proposals, *, claims=None, rooms=None, in_batch=(), gone=()):
    """The follower's report and reservations for one message, without and with
    ``pre_release`` before the answer; ``gone`` requests leave the active set (or
    their first chunk) between the two steps."""
    results = []
    for early in (False, True):
        follower = PPStageAgreement(1, 2, 1, True)
        requests = _follower_requests(held)
        recorder = _Recorder(claims, rooms)
        applied = {}
        if early:
            applied = follower.pre_release(
                proposals, requests, set(in_batch), recorder.reserve, recorder.release
            )
        for rid in gone:
            requests.pop(rid)
        follower.answer_proposals(
            proposals,
            requests,
            recorder.reserve,
            recorder.release,
            recorder.probe,
            recorder.has_room,
        )
        report = follower.build_report()
        depths = {rid: req.py_pp_reserved_depth for rid, req in requests.items()}
        results.append((report, depths, recorder.calls, applied, follower))
    return results


def test_early_release_then_answer_reports_like_the_answer_alone():
    """A drop and a lowering are applied before the adoption, and the answer of
    the same message reports what it would have reported alone; a raise, an
    equal depth, a new claim and a request missing or no longer waiting are
    left to the answer."""
    held = {1: 900, 2: 900, 3: 500, 4: 500, 5: None, 6: 300}
    proposals = {1: None, 2: 400, 3: 700, 4: 500, 5: 200, 6: 100, 99: 50}
    (plain, plain_depths, plain_calls, _, _), (early, depths, calls, applied, follower) = (
        _answer_both_ways(held, proposals, gone=(6,))
    )
    assert early == plain
    assert depths == plain_depths
    assert applied == {1: (900, None), 2: (900, 400), 6: (300, 100)}
    # The pre-applied ones come first and are not claimed or released again.
    assert calls[:3] == [("release", 1), ("reserve", 2, 400), ("reserve", 6, 100)]
    assert ("release", 1) not in calls[3:] and ("reserve", 2, 400) not in calls[3:]
    assert sorted(calls[3:]) == sorted(
        c for c in plain_calls if c[:2] not in {("release", 1), ("reserve", 2)}
    )
    assert follower.num_early_releases == 1 and follower.num_early_lowerings == 2
    assert follower.max_early_lowered_by == 500
    assert follower._pre_applied == {}
    assert "1 early release(s), 2 early lowering(s) (largest lowered by 500 tokens)" in (
        follower.follower_summary()
    )


def test_early_release_skips_requests_of_the_batch():
    held = {1: 900, 2: 900}
    (plain, plain_depths, plain_calls, _, _), (early, depths, calls, applied, follower) = (
        _answer_both_ways(held, {1: None, 2: 400}, in_batch=(1, 2))
    )
    assert applied == {} and follower.num_early_skipped_in_batch == 2
    assert calls == plain_calls and early == plain and depths == plain_depths


def test_a_failed_early_lowering_is_claimed_once_more_by_the_answer():
    held = {2: 900}
    (plain, _, plain_calls, _, _), (early, depths, calls, applied, _) = _answer_both_ways(
        held, {2: 400}, claims={2: None}
    )
    assert applied == {2: (900, None)}
    assert calls == [("reserve", 2, 400), ("reserve", 2, 400)]
    assert plain_calls == [("reserve", 2, 400)]
    assert early == plain == {"iteration": 0, "answers": {2: (400, None, 1002)}}


def test_a_decline_after_an_early_lowering_still_declines_and_releases():
    held = {1: 300, 2: 900}
    (plain, plain_depths, _, _, _), (early, depths, calls, _, _) = _answer_both_ways(
        held, {2: 400}, rooms={2: False}
    )
    assert early == plain
    assert early["declined"] == [2] and early["answers"][2] == (400, None, 1002)
    assert depths == plain_depths == {1: 300, 2: None}
    assert calls == [("reserve", 2, 400), ("release", 2)]


def test_answer_alone_calls_back_in_the_same_order_as_before():
    """Knob off (no ``pre_release``): the answer's reserve / release calls are the
    ones it always made, drops first."""
    follower = PPStageAgreement(1, 2, 1, True)
    requests = _follower_requests({1: 900, 2: 900, 3: None})
    recorder = _Recorder()
    follower.answer_proposals(
        {2: 400, 1: None, 3: 0},
        requests,
        recorder.reserve,
        recorder.release,
        recorder.probe,
        recorder.has_room,
    )
    assert recorder.calls == [("release", 1), ("reserve", 2, 400), ("reserve", 3, 0)]
    assert follower.build_report()["answers"] == {2: (400, 400, 1002), 3: (0, 0, 1003)}


def test_executor_class_default_does_not_release_early():
    assert PyExecutor._pp_follower_early_release is False
    assert PyExecutor._pp_early_released is None


# ------------------------------------------- shortest-uncached-first reservations


class _Reserver:
    """Records reservation calls; claims ``claims.get(request id, 0)`` tokens."""

    def __init__(self, claims=None):
        self.claims = claims or {}
        self.calls: List[Tuple[int, Optional[int]]] = []

    def __call__(self, req, cap):
        self.calls.append((req.py_request_id, cap))
        claimed = self.claims.get(req.py_request_id, 0)
        return claimed if cap is None else min(claimed, cap)


def _spf_agreement(uncached, max_reserved=2, max_wait=512, aged_slots=None, calls=None):
    """A first-rank agreement whose estimate reads ``uncached[request id]``."""

    def estimate(req):
        if calls is not None:
            calls.append(req.py_request_id)
        return uncached[req.py_request_id]

    return PPStageAgreement(
        0,
        2,
        1,
        True,
        max_reserved=max_reserved,
        uncached_estimate=estimate,
        max_wait_iterations=max_wait,
        aged_slots=aged_slots,
    )


_SPF_PROMPTS = {1: 100_000, 2: 50, 3: 80_000, 4: 10}


def test_spf_order_off_reserves_in_arrival_order():
    """Knob off: the order and the reservation calls are the arrival order, as
    before, and the summary has no reservation-order part."""
    agreement = _unit_agreement(max_reserved=2)
    assert not agreement.spf_order
    requests = [_Request(rid, n) for rid, n in _SPF_PROMPTS.items()]
    reserve = _Reserver()
    agreement.step(requests, reserve, lambda req: True)
    assert agreement._order == [1, 2, 3, 4]
    assert reserve.calls == [(1, None), (2, None)]
    assert agreement.take_proposals() == {1: 0, 2: 0}
    assert "Shortest-uncached-first" not in agreement.summary()
    assert not agreement._first_seen and not agreement._uncached


def test_spf_order_reserves_the_fewest_uncached_tokens_first():
    agreement = _spf_agreement(_SPF_PROMPTS, max_reserved=2)
    requests = [_Request(rid, n) for rid, n in _SPF_PROMPTS.items()]
    reserve = _Reserver()
    held = agreement.step(requests, reserve, lambda req: True)
    assert agreement._order == [4, 2, 3, 1]
    assert reserve.calls == [(4, None), (2, None)]
    assert agreement.take_proposals() == {4: 0, 2: 0}
    # Both out of arrival order: an arrival-order pass would have reserved 1 first.
    assert agreement.num_out_of_order == 2 and agreement.num_aged_reserved == 0
    assert held == {1, 2, 3, 4}
    assert "2 made out of arrival order" in agreement.summary()


def test_spf_order_in_arrival_order_counts_nothing_out_of_order():
    agreement = _spf_agreement({1: 10, 2: 20, 3: 30}, max_reserved=3)
    agreement.step([_Request(r, 1000) for r in (1, 2, 3)], _Reserver(), lambda req: True)
    assert agreement._order == [1, 2, 3] and agreement.num_out_of_order == 0


def test_spf_order_ties_keep_arrival_order():
    agreement = _spf_agreement({1: 5, 2: 3, 3: 5, 4: 3}, max_reserved=4)
    agreement.step([_Request(r, 1000) for r in (1, 2, 3, 4)], _Reserver(), lambda req: True)
    assert agreement._order == [2, 4, 1, 3]


def test_spf_order_gives_aged_first_chunks_a_bounded_share():
    """A first chunk waiting ``max_wait`` iterations goes ahead of fresher warm
    ones, in arrival order, but at most ``aged_slots`` aged ones hold a
    reservation: the other aged ones compete by uncached tokens."""
    uncached = {1: 100_000, 2: 90_000, 3: 10, 4: 20, 5: 5, 6: 6, 7: 7}
    agreement = _spf_agreement(uncached, max_reserved=2, max_wait=2, aged_slots=1)
    cold = [_Request(1, 100_000), _Request(2, 90_000)]
    reserve = _Reserver()
    agreement.step(cold + [_Request(3, 10), _Request(4, 20)], reserve, lambda req: True)
    assert reserve.calls == [(3, None), (4, None)]
    agreement.advance()
    # 3 and 4 leave (admitted elsewhere); 1 and 2 keep waiting, not aged yet.
    agreement.step(cold, reserve, lambda req: True)
    assert [c[0] for c in reserve.calls[2:]] == [2, 1]
    agreement.take_proposals()
    # Give both back (pressure, keeping the head 2), then let 2 leave.
    agreement.relieve(True, True, set(), False, lambda rid: None)
    assert list(agreement._proposed) == [2]
    agreement.relieve(False, False, set(), False, lambda rid: None)
    agreement.advance()
    warm = [_Request(5, 5), _Request(6, 6)]
    reserve.calls.clear()
    # Iteration 2: 1 has waited 2 iterations (aged); 5 and 6 are fresh and warm.
    agreement.step([cold[0]] + warm, reserve, lambda req: True)
    assert agreement._order[0] == 1
    assert reserve.calls == [(1, None), (5, None)]
    assert agreement.num_aged_reserved == 1
    agreement.advance()
    # Iteration 3: 5 left; 1 (aged) still holds the one aged slot, so the newly
    # aged request 7 does not get a second one ahead of warm 6.
    agreement._first_seen[7] = 0
    reserve.calls.clear()
    agreement.step([cold[0], warm[1], _Request(7, 100_000)], reserve, lambda req: True)
    assert agreement._order == [1, 6, 7]
    assert reserve.calls == [(6, None)]
    assert agreement.num_aged_reserved == 1


def test_spf_order_probes_each_request_once():
    calls = []
    agreement = _spf_agreement(_SPF_PROMPTS, max_reserved=1, calls=calls)
    requests = [_Request(rid, n) for rid, n in _SPF_PROMPTS.items()]
    for _ in range(5):
        agreement.step(requests, _Reserver(), lambda req: True)
        agreement.advance()
    assert sorted(calls) == [1, 2, 3, 4] and agreement.num_probes == 4
    # A request that leaves the waiting set is forgotten, and probed again if it
    # came back (a new request id in practice).
    agreement.step(requests[1:], _Reserver(), lambda req: True)
    assert 1 not in agreement._uncached and 1 not in agreement._first_seen
    agreement.step(requests, _Reserver(), lambda req: True)
    assert sorted(calls) == [1, 1, 2, 3, 4]


def test_spf_order_refreshes_the_estimate_from_the_claim():
    """The probe saw 990 reusable tokens of request 1, but the prefix was evicted
    since: the claim finds none, and the estimate becomes the whole prompt."""
    agreement = _spf_agreement({1: 10, 2: 500}, max_reserved=1)
    requests = [_Request(1, 1000), _Request(2, 1000)]
    reserve = _Reserver(claims={1: 0, 2: 500})
    agreement.step(requests, reserve, lambda req: True)
    assert reserve.calls == [(1, None)]
    assert agreement._uncached == {1: 1000, 2: 500}
    agreement.advance()
    agreement.step(requests, reserve, lambda req: True)
    assert agreement._order == [2, 1]
    # A deeper claim than probed lowers it (at least 1 token stays uncached).
    agreement2 = _spf_agreement({3: 900}, max_reserved=1)
    agreement2.step([_Request(3, 1000)], _Reserver(claims={3: 1000}), lambda req: True)
    assert agreement2._uncached == {3: 1}


def test_spf_order_relieve_keeps_the_first_of_the_order_and_gives_back_the_newest():
    agreement = _spf_agreement({1: 40, 2: 30, 3: 20, 4: 10}, max_reserved=4)
    requests = [_Request(r, 1000) for r in (1, 2, 3, 4)]
    agreement.step(requests, _Reserver(), lambda req: True)
    assert agreement._order == [4, 3, 2, 1]
    assert list(agreement._proposed) == [4, 3, 2, 1]
    released = []
    given_back = agreement.relieve(True, True, set(), False, released.append)
    # 4 is the head (first of the order); 1 and 2 were started last.
    assert given_back == released == [1, 2]
    assert sorted(agreement._proposed) == [3, 4] and agreement.budget == 2


def test_spf_order_defaults():
    assert SPF_MAX_WAIT_ITERATIONS == 512
    assert spf_aged_slots(16) == 4
    assert spf_aged_slots(2) == 1
    assert spf_aged_slots(1) == 1
    with pytest.raises(ValueError, match="aged_slots"):
        PPStageAgreement(0, 2, 1, True, aged_slots=0)
    with pytest.raises(ValueError, match="max_wait_iterations"):
        PPStageAgreement(0, 2, 1, True, max_wait_iterations=-1)


@pytest.mark.parametrize("spf_order", [False, True])
def test_executor_reserves_shortest_uncached_first(spf_order, monkeypatch):
    """The real executor path: the first rank's estimate is its own probe
    (prompt length minus the reusable prefix); followers never order. Without
    the estimate the agreement reserves in arrival order."""
    monkeypatch.setattr(py_executor_module, "DEFAULT_MAX_RESERVED", 2)
    ranks = _make_ranks(
        [_StagePool(0), _StagePool(0)],
        PROMPT_LEN,
        stage_agree=True,
        monkeypatch=monkeypatch,
        spf_order=spf_order,
    )
    for executor in ranks:
        executor.active_requests += [
            _Request(rid, n) for rid, n in ((20, 5000), (21, 500), (22, 2000))
        ]

    wire = _pp_iteration(ranks)

    leader, follower = ranks[0]._pp_stage_agree, ranks[1]._pp_stage_agree
    assert leader.spf_order is spf_order and not follower.spf_order
    expected = [21, 22] if spf_order else [REQUEST_ID, 20]
    assert sorted(wire.reuse_proposals) == expected
    for executor in ranks:
        assert sorted(executor.kv_cache_manager.kv_cache_map) == expected
    if spf_order:
        assert leader._uncached == {REQUEST_ID: PROMPT_LEN, 20: 5000, 21: 500, 22: 2000}
        assert leader.max_wait_iterations == 512 and leader.aged_slots == 1


# --------------------------------------------------------------------------- started prefills first


def test_warm_first_chunks_go_ahead_of_a_started_prompt(monkeypatch):
    """A long cold prompt is mid-prefill when warm prompts arrive whose uncached
    tail fills a whole chunk. Shortest prefill first puts each warm first chunk
    ahead of the long prompt, which then gets no tokens while they run. Every
    first chunk starts at its reserved depth."""
    long_id, long_len = 9, 16 * MAX_NUM_TOKENS
    warm = {rid: 24576 + MAX_NUM_TOKENS for rid in (11, 12, 13)}
    cores = {rid: _full_attention_core(768) for rid in warm}
    cores[long_id] = _full_attention_core(0)
    pools = [_CapacityPool(dict(cores), 10**9) for _ in range(2)]
    ranks = _ranks_with_requests(pools, {long_id: long_len}, True, monkeypatch)

    chunks = []
    arrived = False
    for _ in range(200):
        wire = _pp_iteration(ranks)
        chunks.append(dict(wire.context_chunks))
        _advance_launched_chunks(ranks, wire)
        _finish_done(ranks)
        if not arrived and long_id in chunks[-1]:
            # The long prompt has started: the warm prompts arrive on every stage.
            arrived = True
            for executor in ranks:
                executor.active_requests += [_Request(rid, n) for rid, n in warm.items()]
        if arrived and not any(executor.active_requests for executor in ranks):
            break
    else:
        raise AssertionError("prompts left after 200 iterations")

    starts = {}
    for iteration_chunks in chunks:
        for rid, (position, _) in iteration_chunks.items():
            starts.setdefault(rid, position)
    assert starts == {long_id: 0, **{rid: 24576 for rid in warm}}
    long_iterations = [i for i, c in enumerate(chunks) if long_id in c]
    assert len(long_iterations) == 16
    back_to_back = long_iterations == list(range(long_iterations[0], long_iterations[-1] + 1))
    assert not back_to_back
    warm_iterations = [i for i, c in enumerate(chunks) if set(c) & set(warm)]
    assert min(warm_iterations) < long_iterations[-1]

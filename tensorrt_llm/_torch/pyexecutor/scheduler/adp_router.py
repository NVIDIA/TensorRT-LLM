# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Attention Data Parallelism (ADP) abstractions.

Provides RankState and ADPRouter interface for distributing
new requests across ADP ranks.

Protocol:
    1. Each rank builds its local RankState
    2. All ranks exchange RankState via allgather
    3. ADPRouter.route_requests() distributes new requests

Includes:
    - DefaultADPRouter: load-balanced min-heap routing
    - KVCacheAwareADPRouter: cache-aware routing that factors in
      prefix match length from the KV cache radix tree
"""

from __future__ import annotations

import heapq
import math
import os
import random
from abc import ABC, abstractmethod
from collections import OrderedDict, namedtuple
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from tensorrt_llm.logger import logger

from ..llm_request import LlmRequestState
from .rank_state import RankIterStatsPayload, RankState

if TYPE_CHECKING:
    from tensorrt_llm._torch.distributed.communicator import Distributed
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

    from ..executor_request_queue import RequestQueueItem

HeapVal = namedtuple("HeapVal", ["num_tokens", "num_requests", "rank", "request_list"])


def _num_input_tokens(request) -> int:
    """Token count via the cheap num_input_tokens accessor (avoids materializing
    input_token_ids); 0 when request is None."""
    if request is None:
        return 0
    num_input_tokens = getattr(request, "num_input_tokens", None)
    if isinstance(num_input_tokens, int):
        return num_input_tokens
    return len(getattr(request, "input_token_ids", []))


def is_retiring_request(request) -> bool:
    """True if ``request`` has produced its final token and is being torn down."""
    return request.state == LlmRequestState.GENERATION_TO_COMPLETE


def build_active_requests_for_overlap(active_requests):
    """Return ``active_requests`` without the requests that are already retiring."""
    return [req for req in active_requests if not is_retiring_request(req)]


def count_retiring_requests(active_requests) -> int:
    """Count the retiring requests in ``active_requests``."""
    return sum(1 for req in active_requests if is_retiring_request(req))


class ADPRouter(ABC):
    """Abstract interface for distributing new requests across ADP ranks.

    This is an **instance-level** router: it distributes requests across the
    DP ranks within a single instance (e.g., one mpirun job controlling 8 GPUs
    that together form a single logical server).

    In disaggregated serving architectures, a separate higher-level router
    orchestrates traffic across multiple instances (e.g., routing between
    prefill and decode instances). That cross-instance routing is outside
    the scope of this class.

    Interface:
        Input:  list[RankState], list[Request]
        Output: dict[rank, list[Request]]
    """

    needs_prefix_matches: bool = False

    def __init__(self, dist: Distributed, has_seq_slot_headroom: bool = False):
        self.dist = dist
        self.exclude_retiring_requests = has_seq_slot_headroom

    @classmethod
    def create(
        cls,
        dist: "Distributed",
        has_seq_slot_headroom: bool,
        kv_cache_manager=None,
        attention_dp_config=None,
        async_transfer_manager=None,
    ) -> "ADPRouter":
        """Factory method to create the appropriate ADP router.

        Args:
            dist: Distributed communicator.
            has_seq_slot_headroom: Whether the executor's sequence-slot pool was
                sized with the extra overlap headroom.
            kv_cache_manager: KV cache manager instance (may be None).
            attention_dp_config: AttentionDpConfig instance (may be None).
            async_transfer_manager: PyExecutor's AsyncTransferManager, used by
                KVCacheAwareADPRouter to account for KV-transfer-in-progress
                requests in per-rank load.  Ignored by DefaultADPRouter.

        Returns:
            A KVCacheAwareADPRouter if config requests it and the
            kv_cache_manager has block reuse enabled; DefaultADPRouter
            otherwise.
        """
        # Env modifier for the conversation router (default off): place new conversations by pinned prompt
        # tokens instead of round-robin / least-queued. Read once here, at router construction.
        token_aware = os.environ.get(ConversationAwareADPRouter.TOKEN_AWARE_ENV, "0") == "1"

        if (
            attention_dp_config is not None
            and attention_dp_config.kv_cache_routing_conversation_affinity
        ):
            # Explicit conversation_id -> rank affinity. Independent of the KV
            # cache manager (works with or without block reuse, though it is
            # most beneficial with reuse on), so it is checked before the
            # KV-cache-aware path and takes precedence when both are enabled.
            return ConversationAwareADPRouter(
                dist=dist,
                has_seq_slot_headroom=has_seq_slot_headroom,
                max_sessions=attention_dp_config.kv_cache_routing_max_sessions,
                fair_share_multiplier=attention_dp_config.kv_cache_routing_fair_share_multiplier,
                new_conv_placement=attention_dp_config.kv_cache_routing_new_conv_placement,
                token_aware_placement=token_aware,
                kv_cache_manager=kv_cache_manager,
            )

        if token_aware:
            # Make a misconfigured A/B visible: the modifier only applies to ConversationAwareADPRouter.
            logger.warning_once(
                f"{ConversationAwareADPRouter.TOKEN_AWARE_ENV}=1 is set but "
                "kv_cache_routing_conversation_affinity is off; the token-aware placement modifier "
                "only applies to ConversationAwareADPRouter and is ignored",
                key="adp_token_aware_ignored",
            )

        if (
            attention_dp_config is not None
            and attention_dp_config.enable_kv_cache_aware_routing
            and kv_cache_manager is not None
            and kv_cache_manager.enable_block_reuse
        ):
            return KVCacheAwareADPRouter(
                dist=dist,
                has_seq_slot_headroom=has_seq_slot_headroom,
                kv_cache_manager=kv_cache_manager,
                load_balance_weight=attention_dp_config.kv_cache_routing_load_balance_weight,
                match_rate_threshold=attention_dp_config.kv_cache_routing_match_rate_threshold,
                fair_share_multiplier=attention_dp_config.kv_cache_routing_fair_share_multiplier,
                cold_start_warmup=attention_dp_config.kv_cache_routing_cold_start_warmup,
                async_transfer_manager=async_transfer_manager,
                account_for_in_transfer=attention_dp_config.kv_cache_routing_account_for_in_transfer,
            )

        return DefaultADPRouter(dist=dist, has_seq_slot_headroom=has_seq_slot_headroom)

    @abstractmethod
    def create_rank_state(
        self,
        active_requests: list[LlmRequest],
        new_requests: list[RequestQueueItem],
    ) -> RankState:
        """Create local RankState from current rank's active and new requests.

        Args:
            active_requests: Currently active LlmRequests on this rank.
            new_requests: New requests popped from the waiting queue.

        Returns:
            RankState for this rank, to be serialized and allgathered.
        """
        raise NotImplementedError

    def gather_all_rank_states(
        self,
        active_requests: list[LlmRequest],
        new_requests: list[RequestQueueItem] | None = None,
        iter_stats_payload: RankIterStatsPayload | None = None,
        forward_work: tuple[int, int] | None = None,
        include_kv_cache_load: bool = False,
    ) -> list[RankState]:
        """Build local RankState, allgather across DP ranks, return all states.

        Args:
            active_requests: Currently active LlmRequests on this rank.
            new_requests: New requests popped from the waiting queue.
                Currently unused; reserved for future routers that need
                new-request info (e.g. KV-cache-aware routing).
            iter_stats_payload: Completed previous-iteration stats payload to
                piggyback on this allgather, if one is pending.
            forward_work: Context and generation work eligible before this
                iteration polls transfers, used by conditional attention DP.
        """
        if self.exclude_retiring_requests:
            active_requests_for_overlap = build_active_requests_for_overlap(active_requests)
            num_retiring_requests = len(active_requests) - len(active_requests_for_overlap)
        else:
            active_requests_for_overlap = active_requests
            num_retiring_requests = 0
        local_state = self.create_rank_state(active_requests_for_overlap, new_requests or [])
        local_state.num_retiring_requests = num_retiring_requests
        local_state.copy_iter_stats_from(iter_stats_payload)
        if forward_work is not None:
            local_state.num_context_requests, local_state.num_generation_requests = forward_work
        responses = self.dist.tp_allgather(local_state.serialize(include_kv_cache_load))
        return [RankState.deserialize(data=resp) for resp in responses]

    @staticmethod
    def _assign_explicit_dp_ranks(
        requests: List["RequestQueueItem"],
        all_ranks_new_requests: Dict[int, List["RequestQueueItem"]],
        all_ranks_num_active_requests: List[int],
        max_num_active_requests: int,
    ) -> List["RequestQueueItem"]:
        """Place requests carrying an explicit ``attention_dp_rank`` on that rank
        (while it is under ``max_num_active_requests``) and return the rest for
        load-balanced assignment. Mutates the ``all_ranks_*`` accumulators in
        place. Shared by Default and ConversationAware routers.
        """
        remaining: List["RequestQueueItem"] = []
        for req_item in requests:
            scheduling_params = getattr(req_item.request, "py_scheduling_params", None)
            target_dp_rank = (
                scheduling_params.attention_dp_rank if scheduling_params is not None else None
            )
            if (
                target_dp_rank is not None
                and all_ranks_num_active_requests[target_dp_rank] < max_num_active_requests
            ):
                all_ranks_num_active_requests[target_dp_rank] += 1
                all_ranks_new_requests[target_dp_rank].append(req_item)
            else:
                remaining.append(req_item)
        return remaining

    @staticmethod
    def _expected_num_active_requests(
        all_ranks_num_active_requests: List[int],
        num_new_requests: int,
        tp_size: int,
        *,
        multiplier: float = 1.0,
        hard_cap: Optional[int] = None,
    ) -> int:
        """Loose per-rank target = ``multiplier * ceil(fair-share)``, floored at the
        busiest rank's current load and optionally capped at ``hard_cap``
        (``max_num_active_requests``). It is the soft cap that bounds per-rank
        assignment so no rank exceeds the returned value -- the ADP padding
        invariant asserted in py_executor._pad_attention_dp_dummy_request.
        """
        fair_share = (
            sum(all_ranks_num_active_requests) + num_new_requests + tp_size - 1
        ) // tp_size
        expected = max(
            math.ceil(multiplier * fair_share),
            max(all_ranks_num_active_requests),
        )
        return min(expected, hard_cap) if hard_cap is not None else expected

    @abstractmethod
    def route_requests(
        self,
        all_rank_states: list[RankState],
        new_requests: list[RequestQueueItem],
        max_num_active_requests: int,
    ) -> Tuple[Dict[int, List[RequestQueueItem]], int]:
        """Assign new requests to ranks based on gathered rank states.

        Args:
            all_rank_states: State of all ranks (from allgather).
            new_requests: New RequestQueueItems to distribute.
            max_num_active_requests: Maximum active requests per rank.

        Returns:
            Tuple of:
                - dict mapping rank -> list of assigned requests
                - expected_num_active_requests per rank after assignment
        """
        raise NotImplementedError


class DefaultADPRouter(ADPRouter):
    """Default heap-based request router.

    Distributes requests across tensor parallel ranks for attention DP.
    It first tries to assign requests to their target dp_rank (if specified
    and has capacity), then balances the remaining requests across all ranks.

    Algorithm:
        1. Sort requests so non-relaxed (strict dp_rank) requests come first.
        2. For each request with a target_dp_rank, assign it to that rank if
           the rank has not reached max_num_active_requests.
        3. Balance remaining unscheduled requests across ranks using a min-heap
           keyed on (num_active_tokens, num_active_requests), so ranks with
           fewer tokens get more requests. Requests are sorted by token count
           (descending) for better load balancing.
    """

    def create_rank_state(
        self,
        active_requests: list[LlmRequest],
        new_requests: list[RequestQueueItem],
    ) -> RankState:
        if self.dist.has_cp_helix:
            num_active_tokens = sum(req.total_input_len_cp for req in active_requests)
        else:
            num_active_tokens = sum(req.py_orig_prompt_len for req in active_requests)
        return RankState(
            rank=self.dist.tp_rank,
            num_active_requests=len(active_requests),
            num_active_tokens=num_active_tokens,
        )

    def route_requests(
        self,
        all_rank_states: list[RankState],
        new_requests: list[RequestQueueItem],
        max_num_active_requests: int,
    ) -> Tuple[Dict[int, List[RequestQueueItem]], int]:
        tp_size = len(all_rank_states)
        all_ranks_new_requests: Dict[int, List[RequestQueueItem]] = {
            s.rank: [] for s in all_rank_states
        }
        all_ranks_num_active_requests = [s.num_active_requests for s in all_rank_states]
        all_ranks_num_active_tokens = [s.num_active_tokens for s in all_rank_states]

        def get_relax_value(req_item):
            scheduling_params = getattr(req_item.request, "py_scheduling_params", None)
            if scheduling_params is None or scheduling_params.attention_dp_rank is None:
                return True
            val = scheduling_params.attention_dp_relax
            return True if val is None else val

        sorted_requests = sorted(new_requests, key=get_relax_value)

        remaining_unscheduled = self._assign_explicit_dp_ranks(
            sorted_requests,
            all_ranks_new_requests,
            all_ranks_num_active_requests,
            max_num_active_requests,
        )

        num_new_requests_all_ranks = len(remaining_unscheduled)
        # Cap at max_num_active_requests so the per-rank target never
        # exceeds what the rank can physically schedule.
        expected_num_active_requests = self._expected_num_active_requests(
            all_ranks_num_active_requests,
            num_new_requests_all_ranks,
            tp_size,
            hard_cap=max_num_active_requests,
        )

        all_ranks_new_requests = self._balance_requests_across_ranks(
            remaining_unscheduled,
            all_ranks_new_requests,
            all_ranks_num_active_requests,
            all_ranks_num_active_tokens,
            expected_num_active_requests,
        )

        return all_ranks_new_requests, expected_num_active_requests

    def _balance_requests_across_ranks(
        self,
        new_requests: List,
        all_ranks_new_requests: Dict[int, List],
        all_ranks_num_active_requests: List[int],
        all_ranks_num_active_tokens: List[int],
        expected_num_active_requests: int,
    ) -> Dict[int, List]:
        """Balance requests across ranks for attention DP.

        Uses a heap-based algorithm to distribute requests evenly across ranks,
        prioritizing ranks with fewer tokens for better load balancing.

        Args:
            new_requests: List of new requests to distribute.
            all_ranks_new_requests: Dict mapping rank to list of already assigned
                requests (will be extended in-place).
            all_ranks_num_active_requests: Number of active requests per rank.
            all_ranks_num_active_tokens: Number of active tokens per rank.
            expected_num_active_requests: Target number of active requests per rank.

        Returns:
            Updated all_ranks_new_requests dict with new requests distributed.
        """
        if not new_requests:
            return all_ranks_new_requests

        all_ranks_new_requests_heap = [
            HeapVal(all_ranks_num_active_tokens[tp_rank], val, tp_rank, [])
            for tp_rank, val in enumerate(all_ranks_num_active_requests)
        ]

        all_ranks_new_requests_heap = [
            val
            for val in all_ranks_new_requests_heap
            if val.num_requests < expected_num_active_requests
        ]

        all_ranks_new_scheduled_requests = {
            val.rank: val.request_list for val in all_ranks_new_requests_heap
        }

        heapq.heapify(all_ranks_new_requests_heap)

        counted = [(_num_input_tokens(item.request), item) for item in new_requests]
        counted.sort(key=lambda item: item[0], reverse=True)

        for token_count, req_item in counted:
            val = heapq.heappop(all_ranks_new_requests_heap)
            val = val._replace(
                num_tokens=val.num_tokens + token_count,
                num_requests=val.num_requests + 1,
            )

            val.request_list.append(req_item)
            if val.num_requests < expected_num_active_requests:
                heapq.heappush(all_ranks_new_requests_heap, val)

        for rank, reqs in all_ranks_new_scheduled_requests.items():
            all_ranks_new_requests[rank].extend(reqs)

        return all_ranks_new_requests


class KVCacheAwareADPRouter(ADPRouter):
    """KV cache-aware request router for attention data parallelism.

    Routes requests considering both load balance and KV cache prefix match
    length on each rank. When a request's prefix is already cached on a rank,
    that rank is preferred to avoid redundant prefill computation.

    Scoring: score(rank, request) = effective_tokens + β * normalized_load
    where:
        effective_tokens  = input_tokens - prefix_match_length
        normalized_load   = rank_active_tokens / max(total_active_tokens, req_tokens) * req_tokens
    Lower score = better rank.

    The load term is normalized by the total active tokens across eligible
    ranks (floored at req_tokens) so that both terms remain on the same
    scale regardless of absolute load levels.

    Requires a KV cache manager with enable_block_reuse=True.
    Falls back to load-based routing when no cache hits exist.
    """

    needs_prefix_matches: bool = True

    def __init__(
        self,
        dist: "Distributed",
        kv_cache_manager,
        has_seq_slot_headroom: bool = False,
        load_balance_weight: float = 1.0,
        match_rate_threshold: float = 0.1,
        fair_share_multiplier: float = 2.0,
        cold_start_warmup: bool = False,
        async_transfer_manager=None,
        account_for_in_transfer: bool = False,
    ):
        super().__init__(dist, has_seq_slot_headroom=has_seq_slot_headroom)
        self.kv_cache_manager = kv_cache_manager
        self.load_balance_weight = load_balance_weight
        self.match_rate_threshold = match_rate_threshold
        self.fair_share_multiplier = fair_share_multiplier
        self._all_ranks_prefix_matches: List[Dict[int, int]] = []
        # Cold-start warmup: ranks not yet targeted through this router.
        # Empty set disables warmup; see ``route_requests``.
        self._pending_warmup_ranks: Set[int] = (
            set(range(self.dist.mapping.dp_size)) if cold_start_warmup else set()
        )
        # Requests still sending KV to GEN are invisible in active_requests;
        # when ``account_for_in_transfer`` is True, fold them back in via the
        # transfer manager (see create_rank_state). Disabled by default
        # because counting them inflates num_active_requests reported to the
        # inference loop, which then skips the idle-fetch wait and yields
        # empty fetch cycles.
        self.account_for_in_transfer = account_for_in_transfer
        self.async_transfer_manager = async_transfer_manager

    def create_rank_state(
        self,
        active_requests: list[LlmRequest],
        new_requests: list[RequestQueueItem],
    ) -> RankState:
        # Remaining-to-compute tokens for a live request (net of KV cache
        # hits), matching the (req_tokens - match_len) scale used by
        # route_requests scoring.
        def _active_tokens(req) -> int:
            if self.dist.has_cp_helix:
                return max(req.total_input_len_cp - req.cached_tokens, 0)
            return max(req.py_orig_prompt_len - req.cached_tokens, 0)

        num_active_tokens = sum(_active_tokens(req) for req in active_requests)
        n_active_for_state = len(active_requests)

        # Fold in requests mid KV-transfer to GEN; they are removed from
        # active_requests but still counted as per-rank load. Gated behind
        # ``account_for_in_transfer`` because the inflated active count
        # makes the inference loop's idle-fetch wait expire even when no
        # request is actually runnable, causing empty fetch cycles.
        if self.account_for_in_transfer and self.async_transfer_manager is not None:
            in_transfer = self.async_transfer_manager.requests_in_transfer()
            num_active_tokens += sum(_active_tokens(r) for r in in_transfer.values())
            n_active_for_state += len(in_transfer)

        return RankState(
            rank=self.dist.tp_rank,
            num_active_requests=n_active_for_state,
            num_active_tokens=num_active_tokens,
        )

    def gather_prefix_matches(
        self,
        new_requests: list[RequestQueueItem],
    ) -> None:
        """Probe local radix tree for each new request, allgather across ranks.

        Populates self._all_ranks_prefix_matches for use by route_requests.
        Must be called after new_requests are available and before route_requests.
        """
        local_matches: list[int] = []
        for req_item in new_requests:
            req = req_item.request
            if req is None:
                local_matches.extend([req_item.id, 0])
                continue
            input_tokens = getattr(req, "input_token_ids", None) or []
            probe_tokens = input_tokens[:-1] if len(input_tokens) > 1 else []
            lora_config = getattr(req, "lora_config", None)
            lora_task_id = lora_config.task_id if lora_config is not None else None
            # cache_salt scopes block reuse on the backend; passing None for
            # non-salted requests is a no-op.
            cache_salt = getattr(req, "cache_salt", None)
            match_len = self.kv_cache_manager.probe_prefix_match_length(
                probe_tokens,
                lora_task_id,
                cache_salt=cache_salt,
            )
            local_matches.extend([req_item.id, match_len])

        all_data = self.dist.tp_allgather(local_matches)

        self._all_ranks_prefix_matches = []
        for rank_data in all_data:
            matches: Dict[int, int] = {}
            for i in range(0, len(rank_data), 2):
                req_id = rank_data[i]
                matches[req_id] = rank_data[i + 1]
            self._all_ranks_prefix_matches.append(matches)

    def _score_rank(
        self,
        req_tokens: int,
        match_len: int,
        rank_active_tokens: float,
        load_denom: float,
    ) -> float:
        """Score a candidate rank for a request (lower is better).

        Args:
            req_tokens: Total input tokens of the request.
            match_len: Prefix match length on this rank's radix tree.
            rank_active_tokens: Active tokens currently on this rank.
            load_denom: Normalization denominator for the load term.

        Returns:
            Score combining cache miss cost and load penalty.
        """
        effective = req_tokens - match_len
        normalized_load = rank_active_tokens / load_denom * req_tokens
        return effective + self.load_balance_weight * normalized_load

    @staticmethod
    def _prefix_fingerprint(token_ids, num_tokens: int = 64) -> tuple:
        """Return a hashable fingerprint from the first num_tokens tokens.

        Requests sharing the same fingerprint likely belong to the same
        conversation / prefix group and benefit from being routed to the
        same rank.
        """
        if not token_ids:
            return ()
        return tuple(token_ids[:num_tokens])

    def _match_len(self, rank: int, req_id: int) -> int:
        matches = self._all_ranks_prefix_matches
        return matches[rank].get(req_id, 0) if rank < len(matches) else 0

    def route_requests(
        self,
        all_rank_states: list[RankState],
        new_requests: list[RequestQueueItem],
        max_num_active_requests: int,
    ) -> Tuple[Dict[int, List[RequestQueueItem]], int]:
        tp_size = len(all_rank_states)
        all_ranks_new_requests: Dict[int, List[RequestQueueItem]] = {
            s.rank: [] for s in all_rank_states
        }
        all_ranks_num_active_requests = [s.num_active_requests for s in all_rank_states]
        all_ranks_num_active_tokens = [float(s.num_active_tokens) for s in all_rank_states]

        def get_relax_value(req_item):
            scheduling_params = getattr(req_item.request, "py_scheduling_params", None)
            if scheduling_params is None or scheduling_params.attention_dp_rank is None:
                return True
            val = scheduling_params.attention_dp_relax
            return True if val is None else val

        sorted_requests = sorted(new_requests, key=get_relax_value)

        # Strict ``attention_dp_rank`` is honoured first; relaxed requests
        # fall back to the smallest unwarmed rank until every rank has been
        # targeted, seeding the shared system prompt before scoring would
        # otherwise pin all traffic to the first warm rank.
        remaining_unscheduled = []
        for req_item in sorted_requests:
            scheduled = False
            target_dp_rank = None
            scheduling_params = getattr(req_item.request, "py_scheduling_params", None)
            if scheduling_params is not None:
                target_dp_rank = scheduling_params.attention_dp_rank
            if target_dp_rank is None and self._pending_warmup_ranks:
                # Smallest-first: deterministic across DP ranks.
                target_dp_rank = min(self._pending_warmup_ranks)
            if (
                target_dp_rank is not None
                and all_ranks_num_active_requests[target_dp_rank] < max_num_active_requests
            ):
                all_ranks_num_active_requests[target_dp_rank] += 1
                # Keep token tally in sync with the soft-balancing phase.
                effective = max(
                    _num_input_tokens(req_item.request)
                    - self._match_len(target_dp_rank, req_item.id),
                    0,
                )
                all_ranks_num_active_tokens[target_dp_rank] += effective
                scheduled = True
                all_ranks_new_requests[target_dp_rank].append(req_item)
                # Only mark a rank as warmed once a request actually landed
                # there; saturated picks stay pending for a later call.
                self._pending_warmup_ranks.discard(target_dp_rank)

            if not scheduled:
                remaining_unscheduled.append(req_item)

        # Group requests by prefix fingerprint so related turns of the same
        # conversation see each other's load-tracker updates; longer ISL
        # first within a group.
        def _sort_key(req_item):
            tokens = getattr(req_item.request, "input_token_ids", []) if req_item.request else []
            return (self._prefix_fingerprint(tokens), -len(tokens))

        remaining_unscheduled = sorted(remaining_unscheduled, key=_sort_key)

        # Loose cap at fair_share_multiplier * ceil(total / tp_size): safety
        # net against runaway concentration, still loose enough that cache
        # affinity wins in the common case.  Hard-cap additionally at
        # max_num_active_requests so the per-rank target never exceeds what
        # the rank can physically schedule (matches DefaultADPRouter).
        num_new_requests_all_ranks = len(remaining_unscheduled)
        expected_num_active_requests = self._expected_num_active_requests(
            all_ranks_num_active_requests,
            num_new_requests_all_ranks,
            tp_size,
            multiplier=self.fair_share_multiplier,
            hard_cap=max_num_active_requests,
        )
        eligible_ranks = [
            rank
            for rank in range(tp_size)
            if all_ranks_num_active_requests[rank] < expected_num_active_requests
        ]

        for req_item in remaining_unscheduled:
            if not eligible_ranks:
                break

            req_tokens = _num_input_tokens(req_item.request)
            req_id = req_item.id

            # Shuffle eligible ranks per decision so score ties fall to a
            # uniform random pick instead of "lowest index wins" (which
            # starves high-index ranks during cold start).  Seed with
            # req_id so every TP rank produces the same permutation --
            # route_requests runs locally on every rank with no broadcast,
            # so divergence would deadlock the distributed protocol.
            iter_ranks = list(eligible_ranks)
            random.Random(req_id).shuffle(iter_ranks)
            best_rank = iter_ranks[0]
            best_score = float("inf")

            # Normalize per-rank active_tokens by the total load across
            # eligible ranks (floored to req_tokens to avoid ~0 division),
            # so the load penalty is scale-invariant.
            total_load = sum(all_ranks_num_active_tokens[r] for r in eligible_ranks)
            load_denom = max(total_load, float(req_tokens))

            # Per-rank prefix match lengths, reused by gate + scoring +
            # post-decision bookkeeping.
            match_lens = {r: self._match_len(r, req_id) for r in eligible_ranks}

            # Cache-affinity gate: below the threshold, zero out match_len
            # so routing is driven purely by load.
            max_match_for_req = max(match_lens.values(), default=0)
            cache_affinity_active = (
                max_match_for_req / max(req_tokens, 1)
            ) > self.match_rate_threshold

            for rank in iter_ranks:
                match_len = match_lens[rank] if cache_affinity_active else 0
                score = self._score_rank(
                    req_tokens, match_len, all_ranks_num_active_tokens[rank], load_denom
                )
                # Tie-break on active_tokens to spread traffic when scores
                # collide; cache-affinity wins are unaffected (lower score).
                if (score, all_ranks_num_active_tokens[rank]) < (
                    best_score,
                    all_ranks_num_active_tokens[best_rank],
                ):
                    best_score = score
                    best_rank = rank

            all_ranks_new_requests[best_rank].append(req_item)
            all_ranks_num_active_requests[best_rank] += 1

            effective_added = max(req_tokens - match_lens[best_rank], 0)
            all_ranks_num_active_tokens[best_rank] += effective_added

            # Progressive eviction: rank leaves eligibility once it hits
            # the cap for the rest of this batch.
            if all_ranks_num_active_requests[best_rank] >= expected_num_active_requests:
                eligible_ranks.remove(best_rank)

        logger.debug(
            f"[adp_router] new_reqs_per_rank="
            f"{[len(all_ranks_new_requests[r]) for r in range(tp_size)]}"
        )

        return all_ranks_new_requests, expected_num_active_requests


class ConversationAwareADPRouter(ADPRouter):
    """Pins each conversation to a single attention-DP rank: the first request
    of a conversation is placed by ``new_conv_placement`` (``round_robin`` by
    default, ``least_queued``, or ``least_tokens``), and every later request with
    the same ``conversation_id`` returns to that rank, keeping the conversation's
    KV-cache prefix on one rank. Requests without a ``conversation_id`` fall
    back to the same ``new_conv_placement`` policy.

    Env modifier ``TLLM_ADP_TOKEN_AWARE_PLACEMENT=1`` (default off, read by
    ``ADPRouter.create``): the ``rank is None`` branch (new conversation,
    sticky overflow, no ``conversation_id``) picks the rank with the least
    *pinned prompt tokens* instead of round-robin / least-queued. Pinned load
    per rank is the sum over its live conversations of the prompt length of
    their most recent turn (a turn replaces, not adds); a conversation stops
    counting once its rank has written more than ``window_tokens`` tokens
    since the conversation's last turn (the two-tier KV capacity in tokens by
    default; ``TLLM_ADP_TOKEN_AWARE_WINDOW_TOKENS`` overrides, 0 = never) or
    when it falls out of the LRU session map. Ties go round-robin. The state
    is a deterministic function of the broadcast inputs, so every DP rank
    computes the same assignment; the variable must be set identically on all
    DP ranks of an instance.
    """

    # Default LRU cap on the conversation->rank map (entries are ~tens of
    # bytes each). Bounds memory on long-running servers as conversations churn.
    DEFAULT_MAX_SESSIONS = 1 << 16

    # Env modifier: token-aware placement of new conversations (see class docstring).
    TOKEN_AWARE_ENV = "TLLM_ADP_TOKEN_AWARE_PLACEMENT"
    TOKEN_AWARE_WINDOW_ENV = "TLLM_ADP_TOKEN_AWARE_WINDOW_TOKENS"

    def __init__(
        self,
        dist: "Distributed",
        has_seq_slot_headroom: bool = False,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        fair_share_multiplier: float = 2.0,
        new_conv_placement: str = "round_robin",
        token_aware_placement: bool = False,
        kv_cache_manager=None,
        token_window_tokens: Optional[int] = None,
    ):
        super().__init__(dist, has_seq_slot_headroom=has_seq_slot_headroom)
        self._conv_to_rank: "OrderedDict[str, int]" = OrderedDict()
        self._max_sessions = max(1, int(max_sessions))
        self._fair_share_multiplier = max(1.0, float(fair_share_multiplier))
        self._new_conv_placement = (
            new_conv_placement
            if new_conv_placement in ("round_robin", "least_queued", "least_tokens")
            else "round_robin"
        )
        self._round_robin_cursor = 0
        self._token_aware = bool(token_aware_placement)
        if self._token_aware:
            # Per rank: pinned load (sum of the last-seen prompt length of every live conversation pinned
            # there), a monotone estimate of KV tokens written, and the live conversations ordered by
            # their last write mark (front = oldest). Per live conversation: its last-seen prompt length.
            self._pinned_tokens: List[int] = []
            self._written_tokens: List[int] = []
            self._live: List["OrderedDict[str, int]"] = []
            self._last_isl: Dict[str, int] = {}
            self._window_tokens = self._resolve_window_tokens(kv_cache_manager, token_window_tokens)
            logger.info(
                f"[adp_router][conv] token-aware new-conversation placement enabled "
                f"({self.TOKEN_AWARE_ENV}=1); window_tokens={self._window_tokens}; "
                f"overrides new_conv_placement={self._new_conv_placement}"
            )

    def create_rank_state(
        self,
        active_requests: list[LlmRequest],
        new_requests: list[RequestQueueItem],
    ) -> RankState:
        if self.dist.has_cp_helix:
            num_active_tokens = sum(req.total_input_len_cp for req in active_requests)
        else:
            num_active_tokens = sum(req.py_orig_prompt_len for req in active_requests)
        return RankState(
            rank=self.dist.tp_rank,
            num_active_requests=len(active_requests),
            num_active_tokens=num_active_tokens,
        )

    @staticmethod
    def _conversation_id(req_item) -> "str | None":
        # Read routing affinity before ExecutorRequest is merged into LlmRequest.
        req = req_item.request
        if req is None:
            return None
        scheduling_params = getattr(req, "py_scheduling_params", None)
        affinity_id = getattr(scheduling_params, "subagent_affinity_id", None)
        if affinity_id:
            return affinity_id
        conversation_params = req.py_conversation_params
        if conversation_params is None:
            return None
        return conversation_params.conversation_id

    def _record_target_rank(self, conv_id: str, rank: int) -> None:
        """Bind/refresh the rank a conversation is pinned to (LRU-touch + evict)."""
        self._conv_to_rank[conv_id] = rank
        self._conv_to_rank.move_to_end(conv_id)
        while len(self._conv_to_rank) > self._max_sessions:
            evicted_conv, evicted_rank = self._conv_to_rank.popitem(last=False)
            if self._token_aware:
                self._forget_conv(evicted_conv, evicted_rank)

    # ---- token-aware placement (TLLM_ADP_TOKEN_AWARE_PLACEMENT=1) -------------------------------------
    # Every DP rank runs route_requests on identical inputs (allgathered RankState, broadcast requests
    # incl. num_input_tokens and conversation params) and keeps its own copy of the state below, with no
    # broadcast of the result. The assignment must therefore be a deterministic function of those inputs
    # only: never feed rank-local data (local KV statistics, wall-clock time, randomness) into this state,
    # or ranks disagree on the assignment and the next collective hangs.

    def _resolve_window_tokens(self, kv_cache_manager, override: Optional[int]) -> int:
        """Write-volume window W after which a conversation stops counting as pinned load.

        W approximates the two-tier (GPU + host) KV capacity in tokens: the host tier is a FIFO ring in
        tokens written, so once a rank has written ~W tokens after a conversation's last turn, that
        conversation's blocks have most likely been recycled. 0 disables expiry (only the LRU session
        cap then drops load).
        """
        if override is not None:
            return max(0, int(override))
        env = os.environ.get(self.TOKEN_AWARE_WINDOW_ENV)
        if env is not None:
            try:
                return max(0, int(env))
            except ValueError:
                logger.warning(
                    f"[adp_router][conv] token-aware: ignoring non-integer "
                    f"{self.TOKEN_AWARE_WINDOW_ENV}={env!r}"
                )
        # KVCacheManagerV2 has no blocks_in_secondary_pool (its host tiers are cache levels, not a
        # second pool); it reports the summed GPU + host capacity in tokens directly.
        capacity = getattr(kv_cache_manager, "kv_capacity_tokens_all_tiers", None)
        if isinstance(capacity, int) and not isinstance(capacity, bool):
            return max(0, capacity)
        primary = getattr(kv_cache_manager, "blocks_in_primary_pool", None)
        secondary = getattr(kv_cache_manager, "blocks_in_secondary_pool", None)
        tokens_per_block = getattr(kv_cache_manager, "tokens_per_block", None)
        if all(
            isinstance(v, int) and not isinstance(v, bool)
            for v in (primary, secondary, tokens_per_block)
        ):
            return max(0, (primary + secondary) * tokens_per_block)
        logger.warning(
            "[adp_router][conv] token-aware: KV pool sizes unavailable; write-volume expiry "
            f"disabled (set {self.TOKEN_AWARE_WINDOW_ENV})"
        )
        return 0

    def _ensure_rank_arrays(self, tp_size: int) -> None:
        while len(self._pinned_tokens) < tp_size:
            self._pinned_tokens.append(0)
            self._written_tokens.append(0)
            self._live.append(OrderedDict())

    def _expire_stale(self, tp_size: int) -> None:
        """Drop conversations whose rank has written more than W tokens since their last turn.

        The conversation keeps its rank binding (stickiness survives); only its load is forgotten.
        """
        if self._window_tokens <= 0:
            return
        for r in range(tp_size):
            live = self._live[r]
            while live:
                conv_id, mark = next(iter(live.items()))
                if self._written_tokens[r] - mark <= self._window_tokens:
                    break
                live.popitem(last=False)
                self._pinned_tokens[r] -= self._last_isl.pop(conv_id, 0)

    def _note_turn(self, conv_id: str, rank: int, isl: int) -> None:
        """Account one turn of ``conv_id`` on its home ``rank``: the last-seen prompt length replaces the
        previous one in the pinned load; only growth counts as newly written KV."""
        prev = self._last_isl.get(conv_id, 0)
        delta = isl - prev
        self._pinned_tokens[rank] += delta
        self._written_tokens[rank] += max(delta, 0)
        self._last_isl[conv_id] = isl
        live = self._live[rank]
        live[conv_id] = self._written_tokens[rank]
        live.move_to_end(conv_id)

    def _note_anonymous(self, rank: int, isl: int) -> None:
        """A request that is not pinned to ``rank`` (no conversation_id, or sticky overflow placed away
        from home) only adds to the rank's write estimate."""
        self._written_tokens[rank] += isl

    def _forget_conv(self, conv_id: str, rank: int) -> None:
        isl = self._last_isl.pop(conv_id, None)
        if isl is not None:
            self._pinned_tokens[rank] -= isl
            self._live[rank].pop(conv_id, None)

    def _least_pinned_rank(self, tp_size: int, active: List[int], soft_cap: int) -> int:
        """Rank with the least pinned tokens among those under ``soft_cap`` (all ranks if none is);
        ties go round-robin from the shared cursor, never to the lowest index."""
        cands = [r for r in range(tp_size) if active[r] < soft_cap] or list(range(tp_size))
        best = min(self._pinned_tokens[r] for r in cands)
        cand_set = set(cands)
        for k in range(tp_size):
            r = (self._round_robin_cursor + k) % tp_size
            if r in cand_set and self._pinned_tokens[r] == best:
                self._round_robin_cursor = (r + 1) % tp_size
                return r
        return cands[0]  # not reachable: some candidate holds the minimum

    def pinned_tokens_per_rank(self) -> List[int]:
        """Read-only view of the pinned load per rank (empty when the modifier is off)."""
        return list(self._pinned_tokens) if self._token_aware else []

    def _assign_new_conversation_explicit_dp_ranks(
        self,
        requests: List["RequestQueueItem"],
        all_ranks_new_requests: Dict[int, List["RequestQueueItem"]],
        all_ranks_num_active_requests: List[int],
        max_num_active_requests: int,
    ) -> List["RequestQueueItem"]:
        """Place explicit first turns and establish their affinity binding.

        Once a conversation is bound, its recorded rank takes precedence over
        later explicit rank hints. Requests whose explicit target is full remain
        eligible for the normal affinity/load-balanced path below.
        """
        remaining: List["RequestQueueItem"] = []
        for req_item in requests:
            conv_id = self._conversation_id(req_item)
            if conv_id is not None and conv_id in self._conv_to_rank:
                remaining.append(req_item)
                continue

            scheduling_params = getattr(req_item.request, "py_scheduling_params", None)
            target_dp_rank = (
                scheduling_params.attention_dp_rank if scheduling_params is not None else None
            )
            if (
                target_dp_rank is not None
                and all_ranks_num_active_requests[target_dp_rank] < max_num_active_requests
            ):
                all_ranks_num_active_requests[target_dp_rank] += 1
                all_ranks_new_requests[target_dp_rank].append(req_item)
                if conv_id is not None:
                    self._record_target_rank(conv_id, target_dp_rank)
                if self._token_aware:
                    isl = _num_input_tokens(req_item.request)
                    if conv_id is not None:
                        self._note_turn(conv_id, target_dp_rank, isl)
                    else:
                        self._note_anonymous(target_dp_rank, isl)
            else:
                remaining.append(req_item)
        return remaining

    def route_requests(
        self,
        all_rank_states: list[RankState],
        new_requests: list[RequestQueueItem],
        max_num_active_requests: int,
    ) -> Tuple[Dict[int, List[RequestQueueItem]], int]:
        tp_size = len(all_rank_states)
        all_ranks_new_requests: Dict[int, List[RequestQueueItem]] = {
            s.rank: [] for s in all_rank_states
        }
        all_ranks_num_active_requests = [s.num_active_requests for s in all_rank_states]
        if self._token_aware:
            self._ensure_rank_arrays(tp_size)
            self._expire_stale(tp_size)

        def get_relax_value(req_item):
            scheduling_params = getattr(req_item.request, "py_scheduling_params", None)
            if scheduling_params is None:
                return True
            return scheduling_params.attention_dp_relax

        sorted_requests = sorted(new_requests, key=get_relax_value)

        # 1) Honour an explicit attention_dp_rank for a new conversation and
        #    record that placement. Existing conversations keep their binding.
        remaining_unscheduled = self._assign_new_conversation_explicit_dp_ranks(
            sorted_requests,
            all_ranks_new_requests,
            all_ranks_num_active_requests,
            max_num_active_requests,
        )

        # Only allgathered state and broadcast requests enter the estimate;
        # local KV statistics would make ranks disagree on placement.
        token_load: list[int] = []
        if self._new_conv_placement == "least_tokens":
            token_load = [0] * tp_size
            for state in all_rank_states:
                token_load[state.rank] = state.num_active_tokens + sum(
                    _num_input_tokens(item.request) for item in all_ranks_new_requests[state.rank]
                )

        # 2) Soft cap for spreading new conversations across ranks, clamped to
        #    per-rank slot capacity so no placement path can overfill a rank.
        #    Sticky returns gate on the hard cap and may still exceed this cap,
        #    so the returned value is re-bumped after the loop.
        expected_num_active_requests = self._expected_num_active_requests(
            all_ranks_num_active_requests,
            len(remaining_unscheduled),
            tp_size,
            multiplier=self._fair_share_multiplier,
            hard_cap=max_num_active_requests,
        )

        def _least_loaded(soft_cap: int) -> int:
            """Lowest-active rank under soft_cap (deterministic), else global min."""
            cands = [r for r in range(tp_size) if all_ranks_num_active_requests[r] < soft_cap]
            if not cands:
                cands = list(range(tp_size))
            return min(cands, key=lambda r: (all_ranks_num_active_requests[r], r))

        def _next_rr(soft_cap: int) -> int:
            """Round-robin to the next rank under soft_cap; advance the cursor."""
            for _ in range(tp_size):
                r = self._round_robin_cursor % tp_size
                self._round_robin_cursor = (self._round_robin_cursor + 1) % tp_size
                if all_ranks_num_active_requests[r] < soft_cap:
                    return r
            return _least_loaded(soft_cap)

        def _least_active_tokens(soft_cap: int) -> int:
            """Shared active prompt load plus this call's admissions."""
            cands = [r for r in range(tp_size) if all_ranks_num_active_requests[r] < soft_cap]
            rank = min(
                cands or range(tp_size),
                key=lambda r: (
                    token_load[r],
                    all_ranks_num_active_requests[r],
                    (r - self._round_robin_cursor) % tp_size,
                ),
            )
            self._round_robin_cursor = (rank + 1) % tp_size
            return rank

        def _least_pinned(soft_cap: int) -> int:
            """Token-aware modifier: least pinned prompt tokens under soft_cap, ties round-robin."""
            return self._least_pinned_rank(tp_size, all_ranks_num_active_requests, soft_cap)

        for req_item in remaining_unscheduled:
            conv_id = self._conversation_id(req_item)
            rank = None
            kind = None  # sticky | new | overflow | noid (token-aware accounting only)

            if conv_id is not None and conv_id in self._conv_to_rank:
                target_rank = self._conv_to_rank[conv_id]
                # Sticky: return the conversation to its target rank while that
                # rank is under the HARD cap (max_num_active_requests), NOT the
                # soft fair-share `expected`. Stickiness wins over balance so a
                # conversation's KV prefix stays resident on one rank instead of
                # migrating when the rank gets moderately busy.
                if all_ranks_num_active_requests[target_rank] < max_num_active_requests:
                    rank = target_rank
                    self._record_target_rank(conv_id, target_rank)  # LRU touch
                    kind = "sticky"
                # else: target saturated this batch -> fall through to overflow
                # WITHOUT rebinding, so the conversation returns next batch.

            if rank is None:
                # First turns, sticky overflow and anonymous requests use the
                # selected placement policy within the existing soft cap.
                if self._token_aware:
                    rank = _least_pinned(expected_num_active_requests)
                elif self._new_conv_placement == "least_tokens":
                    rank = _least_active_tokens(expected_num_active_requests)
                elif self._new_conv_placement == "least_queued":
                    rank = _least_loaded(expected_num_active_requests)
                else:
                    rank = _next_rr(expected_num_active_requests)
                if conv_id is not None and conv_id not in self._conv_to_rank:
                    # Bind this new conversation to its first-turn rank.
                    self._record_target_rank(conv_id, rank)
                    kind = "new"
                elif conv_id is not None:
                    kind = "overflow"
                else:
                    kind = "noid"

            if self._token_aware:
                isl = _num_input_tokens(req_item.request)
                if kind in ("sticky", "new"):
                    self._note_turn(conv_id, rank, isl)
                else:
                    self._note_anonymous(rank, isl)
                logger.debug(
                    f"[adp_router][conv][token_aware] req={req_item.id} conv={conv_id} "
                    f"kind={kind} rank={rank} isl={isl}"
                )

            all_ranks_new_requests[rank].append(req_item)
            all_ranks_num_active_requests[rank] += 1
            if self._new_conv_placement == "least_tokens":
                token_load[rank] += _num_input_tokens(req_item.request)

        # Sticky returns use the hard cap, so a rank may now exceed the pre-loop
        # soft `expected`. Re-bump so the returned value covers the actual
        # per-rank max -- _pad_attention_dp_dummy_request compares `expected`
        # against each rank's routable active count.
        expected_num_active_requests = max(
            expected_num_active_requests, max(all_ranks_num_active_requests)
        )

        if self._token_aware:
            logger.debug(
                f"[adp_router][conv][token_aware] "
                f"pinned_tokens_per_rank={self._pinned_tokens[:tp_size]} "
                f"written_tokens_per_rank={self._written_tokens[:tp_size]} "
                f"live_convs_per_rank={[len(self._live[r]) for r in range(tp_size)]} "
                f"window={self._window_tokens}"
            )

        logger.debug(
            f"[adp_router][conv] new_reqs_per_rank="
            f"{[len(all_ranks_new_requests[r]) for r in range(tp_size)]} "
            f"tracked_convs={len(self._conv_to_rank)}"
        )

        return all_ranks_new_requests, expected_num_active_requests

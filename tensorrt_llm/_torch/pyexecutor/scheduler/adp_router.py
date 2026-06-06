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
import random
from abc import ABC, abstractmethod
from collections import OrderedDict, namedtuple
from dataclasses import MISSING, astuple, dataclass, field, fields, replace
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.distributed.communicator import Distributed
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

    from ..executor_request_queue import RequestQueueItem

HeapVal = namedtuple("HeapVal", ["num_tokens", "num_requests", "rank", "request_list"])


@dataclass
class RankState:
    """Per-rank state information shared via allgather before request assignment.

    Each rank populates its local RankState, then all ranks exchange states
    via allgather. The collected list[RankState] is the input to
    ADPRouter.
    """

    rank: int
    num_active_requests: int = 0
    num_active_tokens: int = 0

    def serialize(self) -> list[int]:
        """Serialize to a flat list for allgather transport."""
        return list(astuple(self))

    @classmethod
    def deserialize(cls, data: list[int]) -> RankState:
        """Deserialize from a flat list received via allgather."""
        return cls(*data)


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

    def __init__(self, dist: Distributed):
        self.dist = dist

    @classmethod
    def create(
        cls,
        dist: "Distributed",
        kv_cache_manager=None,
        attention_dp_config=None,
        async_transfer_manager=None,
    ) -> "ADPRouter":
        """Factory method to create the appropriate ADP router.

        Args:
            dist: Distributed communicator.
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
        if (attention_dp_config is not None
                and attention_dp_config.kv_cache_routing_conversation_affinity):
            # Explicit conversation_id -> rank affinity. Independent of the KV
            # cache manager (works with or without block reuse, though it is
            # most beneficial with reuse on), so it is checked before the
            # KV-cache-aware path and takes precedence when both are enabled.
            return ConversationAwareADPRouter(
                dist=dist,
                max_sessions=attention_dp_config.kv_cache_routing_max_sessions,
                fair_share_multiplier=attention_dp_config.kv_cache_routing_fair_share_multiplier,
            )

        if (
            attention_dp_config is not None
            and attention_dp_config.enable_kv_cache_aware_routing
            and kv_cache_manager is not None
            and kv_cache_manager.enable_block_reuse
        ):
            return KVCacheAwareADPRouter(
                dist=dist,
                kv_cache_manager=kv_cache_manager,
                load_balance_weight=attention_dp_config.kv_cache_routing_load_balance_weight,
                match_rate_threshold=attention_dp_config.kv_cache_routing_match_rate_threshold,
                fair_share_multiplier=attention_dp_config.kv_cache_routing_fair_share_multiplier,
                cold_start_warmup=attention_dp_config.kv_cache_routing_cold_start_warmup,
                async_transfer_manager=async_transfer_manager,
                account_for_in_transfer=attention_dp_config.kv_cache_routing_account_for_in_transfer,
            )

        return DefaultADPRouter(dist=dist)

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
    ) -> list[RankState]:
        """Build local RankState, allgather across DP ranks, return all states.

        Args:
            active_requests: Currently active LlmRequests on this rank.
            new_requests: New requests popped from the waiting queue.
                Currently unused; reserved for future routers that need
                new-request info (e.g. KV-cache-aware routing).
        """
        local_state = self.create_rank_state(active_requests, new_requests or [])
        responses = self.dist.tp_allgather(local_state.serialize())
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
            if scheduling_params is None:
                return True
            return scheduling_params.attention_dp_relax

        sorted_requests = sorted(new_requests, key=get_relax_value)

        remaining_unscheduled = self._assign_explicit_dp_ranks(
            sorted_requests, all_ranks_new_requests,
            all_ranks_num_active_requests, max_num_active_requests,
        )

        num_new_requests_all_ranks = len(remaining_unscheduled)
        # Cap at max_num_active_requests so the per-rank target never
        # exceeds what the rank can physically schedule.
        expected_num_active_requests = self._expected_num_active_requests(
            all_ranks_num_active_requests, num_new_requests_all_ranks, tp_size,
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

        new_requests = sorted(
            new_requests,
            key=lambda x: len(getattr(x.request, "input_token_ids", [])) if x.request else 0,
            reverse=True,
        )

        for req_item in new_requests:
            val = heapq.heappop(all_ranks_new_requests_heap)
            token_count = (
                len(getattr(req_item.request, "input_token_ids", [])) if req_item.request else 0
            )
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
        load_balance_weight: float = 1.0,
        match_rate_threshold: float = 0.1,
        fair_share_multiplier: float = 2.0,
        cold_start_warmup: bool = False,
        async_transfer_manager=None,
        account_for_in_transfer: bool = False,
    ):
        super().__init__(dist)
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
            # cache_salt_id scopes block reuse on both v1 and v2 backends;
            # passing None for non-salted requests is a no-op.
            cache_salt_id = getattr(req, "cache_salt_id", None)
            match_len = self.kv_cache_manager.probe_prefix_match_length(
                probe_tokens,
                lora_task_id,
                cache_salt_id=cache_salt_id,
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

    @staticmethod
    def _req_tokens(req_item) -> int:
        if req_item.request is None:
            return 0
        return len(getattr(req_item.request, "input_token_ids", []))

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
            if scheduling_params is None:
                return True
            return scheduling_params.attention_dp_relax

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
                    self._req_tokens(req_item) - self._match_len(target_dp_rank, req_item.id),
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
        # affinity wins in the common case.
        num_new_requests_all_ranks = len(remaining_unscheduled)
        expected_num_active_requests = self._expected_num_active_requests(
            all_ranks_num_active_requests, num_new_requests_all_ranks, tp_size,
            multiplier=self.fair_share_multiplier, hard_cap=max_num_active_requests,
        )
        eligible_ranks = [
            rank
            for rank in range(tp_size)
            if all_ranks_num_active_requests[rank] < expected_num_active_requests
        ]

        for req_item in remaining_unscheduled:
            if not eligible_ranks:
                break

            req_tokens = self._req_tokens(req_item)
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
    of a conversation is round-robined, and every later request with the same
    ``conversation_id`` returns to that rank, keeping the conversation's
    KV-cache prefix on one rank. Falls back to load-balanced round-robin when no
    ``conversation_id`` is present.
    """

    # Default LRU cap on the conversation->rank map (entries are ~tens of
    # bytes each). Bounds memory on long-running servers as conversations churn.
    DEFAULT_MAX_SESSIONS = 1 << 16

    def __init__(self, dist: "Distributed", max_sessions: int = DEFAULT_MAX_SESSIONS,
                 fair_share_multiplier: float = 2.0):
        super().__init__(dist)
        self._conv_to_rank: "OrderedDict[str, int]" = OrderedDict()
        # LRU cap on the conversation->rank map; oldest entries evicted beyond this.
        self._max_sessions = max(1, int(max_sessions))
        self._fair_share_multiplier = max(1.0, float(fair_share_multiplier))
        self._round_robin_cursor = 0

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
        # NOTE: req_item.request may be a raw tensorrt_llm.bindings.executor.Request
        # (e.g. warmup/dummy requests) that lacks the Python-side
        # py_disaggregated_params attribute -- getattr (not direct access) so those
        # fall through to the load-balanced path instead of raising AttributeError.
        req = getattr(req_item, "request", None)
        if req is None:
            return None
        disagg = getattr(req, "py_disaggregated_params", None)
        if disagg is None:
            return None
        conv_id = getattr(disagg, "conversation_id", None)
        return conv_id if conv_id else None

    def _record_target_rank(self, conv_id: str, rank: int) -> None:
        """Bind/refresh the rank a conversation is pinned to (LRU-touch + evict)."""
        self._conv_to_rank[conv_id] = rank
        self._conv_to_rank.move_to_end(conv_id)
        while len(self._conv_to_rank) > self._max_sessions:
            self._conv_to_rank.popitem(last=False)

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

        def get_relax_value(req_item):
            scheduling_params = getattr(req_item.request, "py_scheduling_params", None)
            if scheduling_params is None:
                return True
            return scheduling_params.attention_dp_relax

        sorted_requests = sorted(new_requests, key=get_relax_value)

        # 1) Honour an explicit attention_dp_rank first (strict placement).
        remaining_unscheduled = self._assign_explicit_dp_ranks(
            sorted_requests, all_ranks_new_requests,
            all_ranks_num_active_requests, max_num_active_requests,
        )

        # 2) Loose per-rank cap = fair_share_multiplier * ceil(fair-share). BOTH
        #    new-conversation spreading and sticky returns are capped at this, so
        #    no rank exceeds expected_num_active_requests -- exceeding it breaks
        #    the ADP padding invariant (py_executor._pad_attention_dp_dummy_request).
        expected_num_active_requests = self._expected_num_active_requests(
            all_ranks_num_active_requests, len(remaining_unscheduled), tp_size,
            multiplier=self._fair_share_multiplier,
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

        for req_item in remaining_unscheduled:
            conv_id = self._conversation_id(req_item)
            rank = None

            if conv_id is not None and conv_id in self._conv_to_rank:
                target_rank = self._conv_to_rank[conv_id]
                # Sticky: return the conversation to its target rank while that
                # rank is under the HARD cap (max_num_active_requests), NOT the
                # soft fair-share `expected`. Stickiness intentionally wins over
                # balance so a conversation's KV prefix stays resident on one
                # rank -- capping sticky at the soft `expected` (~fair_share)
                # displaces popular conversations as soon as their rank is
                # moderately busy, causing rank migration -> prefix-cache miss ->
                # worse TTFT (measured: cache hit drops below kv-aware). The hard
                # cap can push a rank above the pre-loop `expected`; the returned
                # value is re-bumped after the loop to keep the ADP padding
                # invariant (_pad_attention_dp_dummy_request) satisfied.
                if all_ranks_num_active_requests[target_rank] < max_num_active_requests:
                    rank = target_rank
                    self._record_target_rank(conv_id, target_rank)  # LRU touch
                # else: target saturated this batch -> fall through to overflow
                # WITHOUT rebinding, so the conversation returns next batch.

            if rank is None:
                # First turn of a new conversation, sticky-overflow, or no
                # conversation_id -> round-robin spread under the soft cap.
                rank = _next_rr(expected_num_active_requests)
                if conv_id is not None and conv_id not in self._conv_to_rank:
                    # Bind this new conversation to its first-turn rank.
                    self._record_target_rank(conv_id, rank)

            all_ranks_new_requests[rank].append(req_item)
            all_ranks_num_active_requests[rank] += 1

        # Sticky returns use the hard cap, so a rank may now exceed the pre-loop
        # soft `expected`. Re-bump so the returned value covers the actual
        # per-rank max -- _pad_attention_dp_dummy_request asserts
        # expected >= len(active_requests) on every rank.
        expected_num_active_requests = max(
            expected_num_active_requests, max(all_ranks_num_active_requests)
        )

        logger.debug(
            f"[adp_router][conv] new_reqs_per_rank="
            f"{[len(all_ranks_new_requests[r]) for r in range(tp_size)]} "
            f"tracked_convs={len(self._conv_to_rank)}"
        )

        return all_ranks_new_requests, expected_num_active_requests

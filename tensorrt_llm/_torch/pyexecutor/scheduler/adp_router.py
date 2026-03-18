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
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import astuple, dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

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
        cls, dist: "Distributed", kv_cache_manager=None, attention_dp_config=None
    ) -> "ADPRouter":
        """Factory method to create the appropriate ADP router.

        Args:
            dist: Distributed communicator.
            kv_cache_manager: KV cache manager instance (may be None).
            attention_dp_config: AttentionDpConfig instance (may be None).

        Returns:
            A KVCacheAwareADPRouter if config requests it and the
            kv_cache_manager has block reuse enabled; DefaultADPRouter
            otherwise.
        """
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

        remaining_unscheduled = []
        for req_item in sorted_requests:
            scheduled = False
            scheduling_params = getattr(req_item.request, "py_scheduling_params", None)
            if scheduling_params is not None:
                target_dp_rank = scheduling_params.attention_dp_rank
                if (
                    target_dp_rank is not None
                    and all_ranks_num_active_requests[target_dp_rank] < max_num_active_requests
                ):
                    all_ranks_num_active_requests[target_dp_rank] += 1
                    scheduled = True
                    all_ranks_new_requests[target_dp_rank].append(req_item)

            if not scheduled:
                remaining_unscheduled.append(req_item)

        num_new_requests_all_ranks = len(remaining_unscheduled)
        total_num_active_requests = sum(all_ranks_num_active_requests)
        expected_num_active_requests = max(
            (total_num_active_requests + num_new_requests_all_ranks + tp_size - 1) // tp_size,
            max(all_ranks_num_active_requests),
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

    def __init__(self, dist: "Distributed", kv_cache_manager, load_balance_weight: float = 1.0):
        super().__init__(dist)
        self.kv_cache_manager = kv_cache_manager
        self.load_balance_weight = load_balance_weight
        self._all_ranks_prefix_matches: List[Dict[int, int]] = []

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
            match_len = self.kv_cache_manager.probe_prefix_match_length(probe_tokens, lora_task_id)
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

        remaining_unscheduled = []
        for req_item in sorted_requests:
            scheduled = False
            scheduling_params = getattr(req_item.request, "py_scheduling_params", None)
            if scheduling_params is not None:
                target_dp_rank = scheduling_params.attention_dp_rank
                if (
                    target_dp_rank is not None
                    and all_ranks_num_active_requests[target_dp_rank] < max_num_active_requests
                ):
                    all_ranks_num_active_requests[target_dp_rank] += 1
                    scheduled = True
                    all_ranks_new_requests[target_dp_rank].append(req_item)

            if not scheduled:
                remaining_unscheduled.append(req_item)

        num_new_requests_all_ranks = len(remaining_unscheduled)
        total_num_active_requests = sum(all_ranks_num_active_requests)
        expected_num_active_requests = max(
            (total_num_active_requests + num_new_requests_all_ranks + tp_size - 1) // tp_size,
            max(all_ranks_num_active_requests),
        )

        # --- Prefix-affinity sorting ---
        # Sort by prefix fingerprint first (group related requests together),
        # then by ISL descending within each group.  This ensures that when
        # request A (conv X, turn 5) is routed to rank R, request B (conv X,
        # turn 3) is processed next and the load tracker still favours rank R.
        def _sort_key(req_item):
            tokens = getattr(req_item.request, "input_token_ids", []) if req_item.request else []
            fp = self._prefix_fingerprint(tokens)
            # Negate length so longer requests come first within each group
            return (fp, -len(tokens))

        remaining_unscheduled = sorted(remaining_unscheduled, key=_sort_key)

        eligible_ranks = [
            rank
            for rank in range(tp_size)
            if all_ranks_num_active_requests[rank] < expected_num_active_requests
        ]

        prefix_matches = self._all_ranks_prefix_matches

        for req_item in remaining_unscheduled:
            if not eligible_ranks:
                break

            req_tokens = (
                len(getattr(req_item.request, "input_token_ids", [])) if req_item.request else 0
            )
            req_id = req_item.id

            best_rank = eligible_ranks[0]
            best_score = float("inf")

            # --- Normalize load term ---
            # Normalize each rank's active_tokens by the total load across all
            # eligible ranks (floored to req_tokens to avoid dividing by ~0).
            # This makes the load term scale-invariant: a rank carrying 2x the
            # average load gets a normalized penalty of ~req_tokens regardless
            # of whether total load is 100 tokens or 100 000 tokens.
            # Using load_range (max-min) instead caused the penalty to blow up
            # when all ranks were near-idle (small range, large relative fraction).
            total_load = sum(all_ranks_num_active_tokens[r] for r in eligible_ranks)
            load_denom = max(total_load, float(req_tokens))

            for rank in eligible_ranks:
                match_len = prefix_matches[rank].get(req_id, 0) if rank < len(prefix_matches) else 0
                score = self._score_rank(
                    req_tokens, match_len, all_ranks_num_active_tokens[rank], load_denom
                )
                if score < best_score:
                    best_score = score
                    best_rank = rank

            all_ranks_new_requests[best_rank].append(req_item)
            all_ranks_num_active_requests[best_rank] += 1

            match_len = (
                prefix_matches[best_rank].get(req_id, 0) if best_rank < len(prefix_matches) else 0
            )
            effective_added = req_tokens - match_len
            all_ranks_num_active_tokens[best_rank] += effective_added

            if all_ranks_num_active_requests[best_rank] >= expected_num_active_requests:
                eligible_ranks.remove(best_rank)

        return all_ranks_new_requests, expected_num_active_requests

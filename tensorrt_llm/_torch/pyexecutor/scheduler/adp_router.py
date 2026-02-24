"""
Attention Data Parallelism (ADP) abstractions.

Provides RankState and ADPRouter interface for distributing
new requests across ADP ranks.

Protocol:
    1. Each rank builds its local RankState
    2. All ranks exchange RankState via allgather
    3. ADPRouter.route_requests() distributes new requests
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import astuple, dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

    from ..executor_request_queue import RequestQueueItem

HeapVal = namedtuple("HeapVal",
                     ["num_tokens", "num_requests", "rank", "request_list"])


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

    Interface:
        Input:  list[RankState], list[Request]
        Output: dict[rank, list[Request]]
    """

    @abstractmethod
    def create_rank_state(
        self,
        rank: int,
        active_requests: list[LlmRequest],
        new_requests: list[RequestQueueItem],
    ) -> RankState:
        """Create local RankState from current rank's active and new requests.

        Args:
            rank: The current rank index.
            active_requests: Currently active LlmRequests on this rank.
            new_requests: New requests popped from the waiting queue.
                Currently unused; reserved for future routers that need
                new-request info (e.g. KV-cache-aware routing).

        Returns:
            RankState for this rank, to be serialized and allgathered.
        """
        raise NotImplementedError

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

    def __init__(self, has_cp_helix: bool = False):
        self.has_cp_helix = has_cp_helix

    def create_rank_state(
        self,
        rank: int,
        active_requests: list[LlmRequest],
        new_requests: list[RequestQueueItem],
    ) -> RankState:
        if self.has_cp_helix:
            num_active_tokens = sum(req.total_input_len_cp for req in active_requests)
        else:
            num_active_tokens = sum(req.py_orig_prompt_len for req in active_requests)
        return RankState(
            rank=rank,
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

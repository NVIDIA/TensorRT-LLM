import dataclasses
import datetime
import heapq
import queue
import threading
import time
from collections import deque, namedtuple
from itertools import repeat
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from tensorrt_llm._utils import mpi_disabled, nvtx_range
from tensorrt_llm.mapping import CpType

from ..distributed import Distributed
from .llm_request import (ExecutorRequest, LlmRequest,
                          executor_request_to_llm_request)

SHUTDOWN_REQUEST_ID = -1


@dataclasses.dataclass
class RequestQueueItem:
    id: int
    request: Optional[ExecutorRequest] = None
    _ = dataclasses.KW_ONLY
    child_req_ids: Optional[list] = None
    is_canceled_request: bool = False
    query: Optional[list] = None  # only used in `StarAttention`

    @property
    def is_shutdown_request(self):
        return self.id == SHUTDOWN_REQUEST_ID

    @property
    def is_normal_request(self):
        return not (self.is_shutdown_request or self.is_canceled_request)


class ExecutorRequestQueue:
    """Handles fetching and processing of new requests from the request queue."""

    def __init__(self, dist: Distributed, enable_attention_dp: bool,
                 max_batch_size: int, max_beam_width: int,
                 max_num_active_requests: int, enable_iter_perf_stats: bool,
                 batch_wait_timeout_ms: float, is_disaggregated: bool):
        self.dist = dist
        self.request_queue: queue.Queue[RequestQueueItem] = queue.Queue()
        self.waiting_queue: deque[RequestQueueItem] = deque()
        self.canceled_req_ids = []
        self.enable_attention_dp = enable_attention_dp
        self.max_batch_size = max_batch_size
        self.max_beam_width = max_beam_width
        self.max_num_active_requests = max_num_active_requests
        self.is_disaggregated = is_disaggregated
        self.enqueue_lock = threading.Lock()
        self.next_request_id = max_batch_size
        self.enable_iter_perf_stats = enable_iter_perf_stats
        self.start_times = {}
        self.active = True
        self.batch_wait_timeout_ms = batch_wait_timeout_ms

        # State tracking
        self.num_fetch_requests = 0
        self.num_fetch_requests_cur_rank = 0
        self.expected_num_active_requests = 0
        self.new_active_requests_queue_latency_ms = 0
        self.is_shutdown = False
        self.should_exclude_last_generation_logits = False

        self._disable_mpi = mpi_disabled()

    def _get_from_request_queue(
            self,
            timeout: Optional[datetime.timedelta]) -> List[RequestQueueItem]:

        items = []
        timeout_secs = timeout.total_seconds() if timeout is not None else None

        try:
            if self.request_queue.empty() and (timeout_secs is None
                                               or timeout_secs > 0):
                # if queue is empty and want to wait, wait
                items.append(self.request_queue.get(timeout=timeout_secs))
            else:
                # if not empty or don't want to wait, just return all items in queue
                while True:
                    queue_item = self.request_queue.get_nowait()
                    items.append(queue_item)
        except queue.Empty:
            pass

        if self.batch_wait_timeout_ms == 0:
            return items

        if len(items) >= self.max_batch_size:
            return items

        deadline = time.monotonic() + self.batch_wait_timeout_ms / 1000.0
        while len(items) < self.max_batch_size:
            remaining_timeout = deadline - time.monotonic()

            if remaining_timeout <= 0:
                break

            try:
                item = self.request_queue.get(timeout=remaining_timeout)
                items.append(item)
            except queue.Empty:
                break

        return items

    @staticmethod
    def _get_num_child_requests(request: ExecutorRequest) -> int:
        sampling_config = request.sampling_config
        return 0 if sampling_config.beam_width > 1 else (
            sampling_config.num_return_sequences or 1) - 1

    def _get_from_waiting_queue(
        self,
        waiting_queue: deque[RequestQueueItem],
        max_req_count: int,
        enable_attention_dp: bool,
        all_ranks_num_active_requests: Optional[List[int]] = None,
    ) -> List[RequestQueueItem]:
        """
        Args:
            waiting_queue: The queue to pop items from.
            max_req_count: Maximum items to retrieve. Returns empty list if <=0.
            enable_attention_dp: Whether to enable attention DP scheduling.
            all_ranks_num_active_requests: Number of active requests for each rank.
        Returns:
            List of requests that can be processed.
        """

        if max_req_count <= 0:
            return []

        req_count = 0
        items = []
        pending_requests = []

        # Track the request with strict requirements
        scheduling_all_ranks_num_active_requests = all_ranks_num_active_requests.copy(
        ) if enable_attention_dp else None
        while req_count < max_req_count and waiting_queue:
            req_item = waiting_queue[0]
            num_children = len(
                req_item.child_req_ids) if req_item.child_req_ids else 0
            if (req_count + 1 + num_children) > max_req_count:
                break
            req_item = waiting_queue.popleft()
            can_process = self._can_process_attention_dp_request(
                req_item, scheduling_all_ranks_num_active_requests
            ) if enable_attention_dp else True

            if can_process:
                items.append(req_item)
                req_count += 1 + num_children
            else:
                pending_requests.append(req_item)

        # Put the pending requests back to the waiting queue
        # All ranks should have the same waiting queue
        waiting_queue.extendleft(reversed(pending_requests))

        return items

    def _can_process_attention_dp_request(
            self, req_item: RequestQueueItem,
            all_ranks_num_active_requests: List[int]) -> bool:
        """Return True if the request can be processed immediately, else False."""

        scheduling_params = getattr(req_item.request, 'py_scheduling_params',
                                    None)
        if scheduling_params is None:
            return True

        target_dp_rank = scheduling_params.attention_dp_rank
        if target_dp_rank is None or scheduling_params.attention_dp_relax:
            return True

        if all_ranks_num_active_requests[
                target_dp_rank] < self.max_num_active_requests:
            all_ranks_num_active_requests[target_dp_rank] += 1
            return True

        return False

    def _get_request_id(self):
        # (next_request_id + 1) % UINT64_MAX
        current_id = self.next_request_id
        self.next_request_id = (self.next_request_id + 1) & ((1 << 64) - 1)
        return current_id

    def _generate_child_request_ids(
            self, request: ExecutorRequest) -> List[int] | None:
        """ Generate child request IDs if needed. """
        child_req_ids = None
        num_children = self._get_num_child_requests(request)
        if num_children > 0:
            child_req_ids = []
            for _ in range(num_children):
                child_req_id = self._get_request_id()
                if self.enable_iter_perf_stats:
                    self.start_times[child_req_id] = time.time()
                child_req_ids.append(child_req_id)

        return child_req_ids

    def _enqueue_impl(
        self, requests_and_queries: Iterable[Tuple[ExecutorRequest,
                                                   Optional[List]]]
    ) -> List[int]:
        req_ids = []
        with self.enqueue_lock:
            assert self.active, "PyExecutor has already been shutdown."
            start_time = time.time()
            for request, query in requests_and_queries:
                req_id = self._get_request_id()
                if self.enable_iter_perf_stats:
                    self.start_times[req_id] = start_time
                child_req_ids = self._generate_child_request_ids(request)

                self.request_queue.put(
                    RequestQueueItem(req_id,
                                     request,
                                     child_req_ids=child_req_ids,
                                     query=query))
                req_ids.append(req_id)
        return req_ids

    def enqueue_requests(self, requests: List[ExecutorRequest]) -> List[int]:
        """
        Enqueue new requests
        """
        return self._enqueue_impl(zip(requests, repeat(None)))

    def enqueue_request(self,
                        request: ExecutorRequest,
                        query: Optional[List] = None) -> int:
        """
        Enqueue a new request, query is only used in `StarAttention`.
        """
        return self._enqueue_impl([(request, query)])[0]

    def enqueue_cancel_request(self, req_id: int):
        with self.enqueue_lock:
            self.request_queue.put(
                RequestQueueItem(req_id, is_canceled_request=True))

    def enqueue_shutdown_request(self):
        with self.enqueue_lock:
            self.request_queue.put(RequestQueueItem(SHUTDOWN_REQUEST_ID))
            self.active = False

    def can_enqueue_request(self) -> bool:
        with self.enqueue_lock:
            return self.active and self.dist.rank == 0

    def _fetch_and_process_requests(
        self,
        total_num_active_requests: int,
        total_max_num_active_requests: int,
        enable_attention_dp: bool,
        all_ranks_num_active_requests: Optional[List[int]] = None
    ) -> List[RequestQueueItem]:
        """Common logic for fetching and processing requests from the queue."""
        # Calculate timeout
        idle = (total_num_active_requests == 0) and len(self.waiting_queue) == 0
        if idle:
            # In Ray path (TLLM_DISABLE_MPI=1), use a periodic heartbeat timeout so rank 0
            # reaches the broadcast path regularly to prevent trtllm-serve timeout when idle.
            timeout = datetime.timedelta(
                seconds=1200) if self._disable_mpi else None
        else:
            timeout = datetime.timedelta(0)

        # Fetch requests from rank 0
        new_requests = []
        if self.dist.rank == 0:
            new_requests = self._get_from_request_queue(timeout)

        # Broadcast requests and handle Python objects
        new_requests, py_request_objects = self._handle_request_broadcasting(
            new_requests)

        # Validate and filter requests
        new_requests = self._validate_and_filter_requests(new_requests)

        # Attach Python objects to requests
        if py_request_objects and (self.dist.tp_size > 1
                                   or self.dist.has_pp) and self.dist.rank > 0:
            self._attach_py_objects_to_requests(new_requests,
                                                py_request_objects)

        self.waiting_queue.extend(new_requests)

        new_requests = self._get_from_waiting_queue(
            self.waiting_queue,
            total_max_num_active_requests - total_num_active_requests,
            enable_attention_dp, all_ranks_num_active_requests)

        # Update performance metrics
        if self.enable_iter_perf_stats and self.dist.rank == 0:
            self._update_new_active_requests_queue_latency(new_requests)

        return new_requests

    @nvtx_range("_fetch_new_requests")
    def fetch_new_requests(
            self, activate_requests: List[LlmRequest]) -> List[LlmRequest]:

        if self.enable_attention_dp:
            return self._fetch_new_requests_attention_dp(activate_requests)
        else:
            return self._fetch_new_requests_attention_tp(len(activate_requests))

    def _fetch_new_requests_attention_tp(
            self, num_active_requests: int) -> List[LlmRequest]:
        """Handle standard (non-attention DP) request fetching."""
        total_num_active_requests = num_active_requests
        total_max_num_active_requests = self.max_num_active_requests

        # fetch and process requests into waiting queue
        new_requests = self._fetch_and_process_requests(
            total_num_active_requests,
            total_max_num_active_requests,
            enable_attention_dp=False)

        # Merge requests and add to active list
        merged_requests = self._merge_requests(new_requests)
        return merged_requests

    def _fetch_new_requests_attention_dp(
            self, activate_requests: List[LlmRequest]) -> List[LlmRequest]:
        """Handle attention DP request fetching with load balancing."""
        # Get active request counts across all ranks
        all_ranks_num_active_requests = []
        all_ranks_num_active_tokens = []
        num_active_tokens = sum(
            [req.py_orig_prompt_len for req in activate_requests])
        responses_list = self.dist.tp_allgather(
            [len(activate_requests), num_active_tokens])
        for num_active_requests, num_active_tokens in responses_list:
            all_ranks_num_active_requests.append(num_active_requests)
            all_ranks_num_active_tokens.append(num_active_tokens)

        total_num_active_requests = sum(all_ranks_num_active_requests)
        total_max_num_active_requests = self.dist.tp_size * self.max_num_active_requests

        # fetch and process requests into waiting queue
        new_requests = self._fetch_and_process_requests(
            total_num_active_requests,
            total_max_num_active_requests,
            enable_attention_dp=True,
            all_ranks_num_active_requests=all_ranks_num_active_requests)

        # Schedule attention dp requests
        all_ranks_new_requests = self._schedule_attention_dp_requests(
            new_requests, all_ranks_num_active_requests,
            all_ranks_num_active_tokens)
        new_requests_cur_rank = all_ranks_new_requests[self.dist.tp_rank]

        # Update performance metrics
        if self.enable_iter_perf_stats and self.start_times:
            self._update_new_active_requests_queue_latency(
                new_requests_cur_rank)

        # Update counters
        self.num_fetch_requests += len(new_requests)
        self.num_fetch_requests_cur_rank += len(new_requests_cur_rank)

        # Merge requests and add to active list
        new_requests_cur_rank = self._merge_requests(new_requests_cur_rank)
        return new_requests_cur_rank

    def _schedule_attention_dp_requests(
            self, new_requests: List[RequestQueueItem],
            all_ranks_num_active_requests: List[int],
            all_ranks_num_active_tokens: List[int]) -> List[RequestQueueItem]:
        """Schedule attention dp requests."""

        # Map from ranks to new requests
        all_ranks_new_requests = {
            tp_rank: []
            for tp_rank in range(self.dist.tp_size)
        }

        # Prioritize the requests that are not in relax mode
        def get_relax_value(req_item):
            scheduling_params = getattr(req_item.request,
                                        'py_scheduling_params', None)
            if scheduling_params is None:
                return True
            return scheduling_params.attention_dp_relax

        new_requests = sorted(new_requests, key=get_relax_value, reverse=True)

        # Try to put the requests to the target dp rank until the max_num_active_requests is reached
        remaining_unscheduled = []
        for req_item in new_requests:
            scheduled = False
            scheduling_params = getattr(req_item.request,
                                        'py_scheduling_params', None)
            if scheduling_params is not None:
                target_dp_rank = scheduling_params.attention_dp_rank
                if target_dp_rank is not None and all_ranks_num_active_requests[
                        target_dp_rank] < self.max_num_active_requests:
                    all_ranks_num_active_requests[target_dp_rank] += 1
                    scheduled = True
                    all_ranks_new_requests[target_dp_rank].append(req_item)

            if not scheduled:
                remaining_unscheduled.append(req_item)

        # Balance the remaining unscheduled requests across ranks
        num_new_requests_all_ranks = len(remaining_unscheduled)
        total_num_active_requests = sum(all_ranks_num_active_requests)
        self.expected_num_active_requests = max(
            (total_num_active_requests + num_new_requests_all_ranks +
             self.dist.tp_size - 1) // self.dist.tp_size,
            max(all_ranks_num_active_requests),
        )

        all_ranks_new_requests = self._balance_requests_across_ranks(
            remaining_unscheduled, all_ranks_new_requests,
            all_ranks_num_active_requests, all_ranks_num_active_tokens)

        return all_ranks_new_requests

    def _handle_request_broadcasting(self,
                                     new_requests: List[RequestQueueItem]):
        """Handle broadcasting of requests and Python objects across ranks."""
        if self.dist.rank == 0:
            py_logits_post_processors = self._collect_py_objects_from_requests(
                new_requests, "py_logits_post_processors")
            py_multimodal_data = self._collect_py_objects_from_requests(
                new_requests, "py_multimodal_data")
            py_scheduling_params = self._collect_py_objects_from_requests(
                new_requests, "py_scheduling_params")
            py_request_objects = tuple(
                filter(None, [
                    py_logits_post_processors, py_multimodal_data,
                    py_scheduling_params
                ]))
        else:
            py_request_objects = None

        if self.dist.rank == 0:
            # Preserve original `new_requests` on rank 0
            _ = self._broadcast_new_requests(new_requests, py_request_objects)
        else:
            new_requests, py_request_objects = self._broadcast_new_requests(
                new_requests, py_request_objects)

        return new_requests, py_request_objects

    def _validate_and_filter_requests(
            self,
            new_requests: List[RequestQueueItem]) -> List[RequestQueueItem]:
        """Validate and filter requests, handling shutdown signals."""
        valid_new_requests = []
        for req_item in new_requests:
            if req_item.is_shutdown_request:
                self.is_shutdown = True
                break
            elif req_item.is_canceled_request:
                self.canceled_req_ids.append(req_item.id)
            else:
                valid_new_requests.append(req_item)

        # Check beam width validation
        for req_item in valid_new_requests:
            if req_item.request and hasattr(req_item.request,
                                            'sampling_config'):
                assert req_item.request.sampling_config.beam_width == self.max_beam_width, \
                    f"Request beam width {req_item.request.sampling_config.beam_width} " \
                    f"is not equal to max_beam_width {self.max_beam_width}. This is not supported!"

        return valid_new_requests

    def _balance_requests_across_ranks(
            self, new_requests: List[RequestQueueItem],
            all_ranks_new_requests: Dict[int, List[RequestQueueItem]],
            all_ranks_num_active_requests: List[int],
            all_ranks_num_active_tokens: List[int]) -> List[RequestQueueItem]:
        """Balance requests across ranks for attention DP."""
        if new_requests:
            # Balance context tokens across ranks using heap
            HeapVal = namedtuple(
                'HeapVal',
                ['num_tokens', 'num_requests', 'rank', 'request_list'])

            all_ranks_new_requests_heap = [
                HeapVal(all_ranks_num_active_tokens[tp_rank], val, tp_rank, [])
                for tp_rank, val in enumerate(all_ranks_num_active_requests)
            ]

            all_ranks_new_requests_heap = [
                val for val in all_ranks_new_requests_heap
                if val.num_requests < self.expected_num_active_requests
            ]

            all_ranks_new_scheduled_requests = {
                val.rank: val.request_list
                for val in all_ranks_new_requests_heap
            }

            heapq.heapify(all_ranks_new_requests_heap)

            # Sort by token count (descending) for better load balancing
            new_requests = sorted(
                new_requests,
                key=lambda x: len(getattr(x.request, 'input_token_ids', []))
                if x.request else 0,
                reverse=True)

            # Distribute requests across ranks
            for req_item in new_requests:

                val = heapq.heappop(all_ranks_new_requests_heap)
                token_count = len(
                    getattr(req_item.request, 'input_token_ids',
                            [])) if req_item.request else 0
                # Update the heap value with the new request
                val = val._replace(
                    num_tokens=val.num_tokens + token_count,
                    num_requests=val.num_requests + 1,
                )

                val.request_list.append(req_item)
                # If rank still has room for new requests, push back into heap
                if val.num_requests < self.expected_num_active_requests:
                    heapq.heappush(all_ranks_new_requests_heap, val)

            # Extend all_ranks_new_requests with the new requests that have been scheduled
            for rank, reqs in all_ranks_new_scheduled_requests.items():
                all_ranks_new_requests[rank].extend(reqs)

        return all_ranks_new_requests

    def _collect_py_objects_from_requests(
            self, requests: List[RequestQueueItem],
            attribute_name: str) -> Optional[Tuple[str, Dict]]:
        """Collect Python-only objects from requests."""
        req_id_to_obj = {}
        for item in requests:
            if not item.is_normal_request:
                continue
            if item.request:
                obj = getattr(item.request, attribute_name, None)
                if obj is not None:
                    req_id_to_obj[item.id] = obj
        return None if not req_id_to_obj else (attribute_name, req_id_to_obj)

    def _broadcast_new_requests(
            self, new_requests: List[RequestQueueItem], py_request_objects
    ) -> Tuple[List[RequestQueueItem], Optional[Dict]]:
        """Broadcast new_requests and optional Python-only metadata across pipeline stages."""
        payloads = (new_requests, py_request_objects)

        if not self.dist.has_pp:
            return self.dist.broadcast(payloads, root=0)

        # Broadcast within first tp group before send/recv chain to other tp groups
        if self.dist.tp_size > 1 and self.dist.is_first_pp_rank:
            payloads = self.dist.tp_broadcast(payloads, root=0)

        # Tag for communication
        tag = self.dist.pp_size  # Use pp_size as tag to avoid conflicts

        # Send payloads
        if not self.dist.is_first_pp_rank:
            payloads = self.dist.recv_object(self.dist.prev_pp_rank, tag)

        if not self.dist.is_last_pp_rank:
            if self._disable_mpi:
                isend_payload = self.dist.isend_object(payloads,
                                                       self.dist.next_pp_rank,
                                                       tag)
                isend_payload.wait()
            else:
                self.dist.send_object(payloads, self.dist.next_pp_rank, tag)

        return payloads

    def _attach_py_objects_to_requests(self, requests: List[RequestQueueItem],
                                       py_request_objects) -> None:
        """Attach Python-only objects to each request."""
        for attr_name, req_obj_dict in py_request_objects:
            for item in requests:
                if item.request:
                    py_obj = req_obj_dict.get(item.id)
                    if py_obj is not None:
                        setattr(item.request, attr_name, py_obj)

    def _update_new_active_requests_queue_latency(
            self, new_requests: List[RequestQueueItem]):
        """Update queue latency metrics for new requests."""
        now = time.time()
        for req_item in new_requests:
            if req_item.id in self.start_times:
                self.new_active_requests_queue_latency_ms += now - self.start_times.pop(
                    req_item.id)
            if req_item.child_req_ids:
                for child_id in req_item.child_req_ids:
                    self.new_active_requests_queue_latency_ms += now - self.start_times.pop(
                        child_id)

    # Note: Helix parallelism is a decode-only feature run with disaggregated serving. This function gets called on gen server
    # during initialization of a new request.
    def _merge_helix_requests(self, new_requests: list[RequestQueueItem],
                              tokens_per_block: int):
        req_with_children = []
        num_cp_ranks = self.dist.cp_size
        curr_cp_rank = self.dist.cp_rank

        # For each request, partition the input_token_ids into blocks and then partition blocks across CP ranks.
        # Currently, the partitioning is such that contiguous blocks are assigned to the same CP rank (as opposed
        # to round-robin).
        for req_item in new_requests:
            all_input_ids = torch.tensor(req_item.request.input_token_ids,
                                         dtype=torch.int64).unsqueeze(0)
            input_len = all_input_ids.shape[-1]

            num_total_blocks = (input_len + tokens_per_block -
                                1) // tokens_per_block
            if num_total_blocks < num_cp_ranks:
                raise ValueError(
                    f"There aren't enough tokens to get at least one block per CP rank. num_total_blocks {num_total_blocks} < num_cp_ranks {num_cp_ranks}. Please use smaller tokens_per_block for KV cache or reduce the number of CP ranks."
                )

            # Padding to ensure torch.stack used with torch.tensor_split works properly.
            padding_len = 0
            if input_len % tokens_per_block != 0:
                padding_len = tokens_per_block - (input_len % tokens_per_block)
                padding_ids = torch.zeros([1, padding_len], dtype=torch.int64)
                all_input_ids = torch.cat((all_input_ids, padding_ids), dim=-1)
            all_position_ids = torch.arange(0,
                                            input_len + padding_len,
                                            dtype=torch.int64).unsqueeze(0)

            input_id_blocks_per_rank = torch.tensor_split(
                torch.stack(all_input_ids.split(tokens_per_block, dim=-1)),
                num_cp_ranks)
            position_id_blocks_per_rank = torch.tensor_split(
                torch.stack(all_position_ids.split(tokens_per_block, dim=-1)),
                num_cp_ranks)

            # Get the input_ids and position_ids for this rank.
            input_ids_this_rank = input_id_blocks_per_rank[
                curr_cp_rank].flatten().tolist()
            position_ids_this_rank = position_id_blocks_per_rank[
                curr_cp_rank].flatten().tolist()

            # Undo the padding. Only last rank's last block will be padded right now
            # given contiguous block assignment.
            if curr_cp_rank == num_cp_ranks - 1 and padding_len > 0:
                input_ids_this_rank = input_ids_this_rank[:-padding_len]
                position_ids_this_rank = position_ids_this_rank[:-padding_len]

            req = executor_request_to_llm_request(
                req_id=req_item.id,
                executor_request=req_item.request,
                child_req_ids=req_item.child_req_ids,
                exclude_last_generation_logits=self.
                _should_exclude_last_generation_logits(),
                input_token_ids=input_ids_this_rank,
                position_ids=position_ids_this_rank,
            )
            req_with_children.append(req)
            if req.child_requests:
                req_with_children.extend(req.child_requests)
        return req_with_children

    @nvtx_range("_merge_requests")
    def _merge_requests(
            self, new_requests: list[RequestQueueItem]) -> List[LlmRequest]:
        cp_config = self.dist.cp_config
        if 'cp_type' in cp_config:
            cp_type = cp_config['cp_type']
            if cp_type == CpType.STAR:
                return self._merge_star_attention_requests(new_requests)
            elif cp_type == CpType.HELIX:
                # Take the usual route below.
                return self._merge_helix_requests(
                    new_requests,
                    tokens_per_block=cp_config['tokens_per_block'])
            else:
                raise NotImplementedError(
                    f'Unsupported cp type {cp_type.name}.')

        req_with_children = []
        for req_item in new_requests:
            req = executor_request_to_llm_request(
                req_item.id, req_item.request, req_item.child_req_ids,
                self._should_exclude_last_generation_logits())
            req_with_children.append(req)
            if req.child_requests:
                req_with_children.extend(req.child_requests)
        return req_with_children

    def _merge_star_attention_requests(
            self, new_requests: list[RequestQueueItem]) -> List[LlmRequest]:
        result = []
        for req_item in new_requests:
            req_id, exe_req, query_token_ids = req_item.id, req_item.request, req_item.query
            ctx_len0 = len(exe_req.input_token_ids)
            ctx_blocks, position_blocks, last_block_padding_num = [
                exe_req.input_token_ids
            ], [[i for i in range(ctx_len0)]], 0
            ctx_blocks, position_blocks, last_block_padding_num = self._partition_context(
                exe_req.input_token_ids)
            if self.dist.cp_rank == self.dist.cp_size - 1 and last_block_padding_num > 0:
                ctx_blocks[-1] = ctx_blocks[-1][:-last_block_padding_num]
                position_blocks[-1] = position_blocks[
                    -1][:-last_block_padding_num]
            #if has query
            if query_token_ids:
                ctx_blocks.append(query_token_ids)
                position_blocks.append([
                    i for i in range(ctx_len0, ctx_len0 + len(query_token_ids))
                ])

            # insert the dummy block to align the number of ctx iterations of each rank
            block_size = self.dist.cp_config['block_size']
            total_blocks = (ctx_len0 + block_size - 1) // block_size
            num_blocks_per_rank = (
                total_blocks + self.dist.cp_size -
                1) // self.dist.cp_size + 1  # 1 for query block
            if len(ctx_blocks) == num_blocks_per_rank:
                ctx_blocks.insert(1, [])
                position_blocks.insert(1, [])
            elif len(ctx_blocks) == num_blocks_per_rank + 1:
                # anchor + ctx_blocks + qry_block
                pass
            else:
                print(
                    f'rank = {self.dist.cp_rank}, len(ctx_blocks)  = {len(ctx_blocks) }, num_blocks_per_rank = {num_blocks_per_rank}'
                )
                assert False, f'invalid context partition'

            # fake data for scheduler
            ctx_blocks_list = [0] * (block_size +
                                     self.dist.cp_config['cp_anchor_size'])

            req = executor_request_to_llm_request(
                req_id, exe_req, self._should_exclude_last_generation_logits(),
                ctx_blocks_list)
            req.gen_iters = 0
            req.ctx_iters = 0
            req.ctx_blocks = ctx_blocks
            req.ctx_position_blocks = position_blocks
            req.query_id = query_token_ids

            result.append(req)

        return result

    def _partition_context(self, ctx_ids_list):
        ctx_ids = torch.tensor(ctx_ids_list).unsqueeze(0)
        ctx_len = ctx_ids.shape[-1]
        block_size = self.dist.cp_config['block_size']
        if block_size is None:
            block_size = ctx_len // self.dist.cp_size
        anchor_block_size = self.dist.cp_config['cp_anchor_size']
        if anchor_block_size is None:
            anchor_block_size = block_size

        assert anchor_block_size <= block_size, f'cp_anchor_size {anchor_block_size} should be smaller than block_size {block_size}'
        padding = 0
        if ctx_len % block_size != 0:
            padding = block_size - (ctx_len % block_size)
            assert padding <= ctx_len, f'block size is too large for context, please set it smaller'
            ctx_ids = torch.cat(
                (ctx_ids, torch.zeros_like(ctx_ids)[:, :padding]), dim=-1)
        position_ids = torch.arange(0, ctx_ids.shape[-1]).unsqueeze(0)

        ctx_ids_blocks = torch.tensor_split(
            torch.stack(ctx_ids.split(block_size, dim=-1)), self.dist.cp_size)
        position_ids_blocks = torch.tensor_split(
            torch.stack(position_ids.split(block_size, dim=-1)),
            self.dist.cp_size)
        if self.dist.cp_rank != 0:
            ctx_blocks, position_blocks = [
                ctx_ids_blocks[0][0].tolist()[0][:anchor_block_size]
            ], [position_ids_blocks[0][0].tolist()[0][:anchor_block_size]]
        else:
            ctx_blocks, position_blocks = [], []

        for idx in range(len(ctx_ids_blocks[self.dist.cp_rank])):
            ctx_block = ctx_ids_blocks[self.dist.cp_rank][idx]
            position_block = position_ids_blocks[self.dist.cp_rank][idx]
            ctx_blocks.append(ctx_block.tolist()[0])
            position_blocks.append(position_block.tolist()[0])
        return ctx_blocks, position_blocks, padding

    def set_exclude_last_generation_logits(self,
                                           disable_overlap_scheduler: bool,
                                           pp_size: int) -> None:
        # When overlap scheduler is enabled then when starting to handle a new prompt,
        # sample_async is called twice before the first call to update_requests:
        # - 1st time as a context request that handles on the 1st generated token
        # - 2nd time as a generation request that handles on the 2nd generated token.
        # and only after these two calls the sampler's update_request method is called.
        # So in a sampler that works by the expected flow of handling the logits in
        # sample_async, every update_request doesn't handle the newest token, but one
        # before it. Since all these calls work on the same request object, then its
        # logits storage contains the logits of both the token update_requests should work
        # on, and also its next token. Thus, excluding the last generation logits from any
        # getter is required.
        self.should_exclude_last_generation_logits = not disable_overlap_scheduler and pp_size == 1

    def _should_exclude_last_generation_logits(self) -> bool:
        return self.should_exclude_last_generation_logits

    def get_new_active_requests_queue_latency(self) -> float:
        return self.new_active_requests_queue_latency_ms

    def get_expected_num_active_requests(self) -> int:
        return self.expected_num_active_requests

    def get_request_queue_size(self) -> int:
        return self.request_queue.qsize()

    def get_request_queue(self) -> queue.Queue[RequestQueueItem]:
        return self.request_queue

    def get_waiting_queue(self) -> deque[RequestQueueItem]:
        return self.waiting_queue

    def update_waiting_queue(self):
        # Remove cancel request in the waiting queue
        self.waiting_queue = deque(req for req in self.waiting_queue
                                   if req.id not in self.canceled_req_ids)

    def get_waiting_queue_size(self) -> int:
        return len(self.waiting_queue)

    def get_canceled_req_ids_size(self) -> int:
        return len(self.canceled_req_ids)

    def get_canceled_req_ids(self) -> List[int]:
        return self.canceled_req_ids

    def clear_canceled_req_ids(self):
        self.canceled_req_ids.clear()

"""Utility functions for request processing."""

import heapq
import os
from collections import deque, namedtuple
from typing import Any, Dict, List, Optional, Tuple

import torch

from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.mapping import CpType

from ..distributed import Distributed
from .hang_detector import HangDetector
from .llm_request import ExecutorRequest, LlmRequest, executor_request_to_llm_request

# Type alias for request queue items (to avoid circular import)
# The actual RequestQueueItem class is defined in executor_request_queue.py


def get_num_child_requests(request: ExecutorRequest) -> int:
    """Get the number of child requests for a given request.

    Args:
        request: The executor request to check.

    Returns:
        Number of child requests (0 if beam search, otherwise num_return_sequences - 1).
    """
    sampling_config = request.sampling_config
    return 0 if sampling_config.beam_width > 1 else (sampling_config.num_return_sequences or 1) - 1


def collect_py_objects_from_requests(
    requests: List, attribute_name: str
) -> Optional[Tuple[str, Dict]]:
    """Collect Python-only objects from requests.

    Args:
        requests: List of RequestQueueItem objects.
        attribute_name: Name of the attribute to collect.

    Returns:
        Tuple of (attribute_name, dict mapping request_id to object) or None if empty.
    """
    req_id_to_obj = {}
    for item in requests:
        if not item.is_normal_request:
            continue
        if item.request:
            obj = getattr(item.request, attribute_name, None)
            if obj is not None:
                req_id_to_obj[item.id] = obj
    return None if not req_id_to_obj else (attribute_name, req_id_to_obj)


def attach_py_objects_to_requests(requests: List, py_request_objects: Tuple) -> None:
    """Attach Python-only objects to each request.

    Args:
        requests: List of RequestQueueItem objects.
        py_request_objects: Tuple of (attribute_name, dict) pairs.
    """
    for attr_name, req_obj_dict in py_request_objects:
        for item in requests:
            if item.request:
                py_obj = req_obj_dict.get(item.id)
                if py_obj is not None:
                    setattr(item.request, attr_name, py_obj)


def schedule_attention_dp_requests(
    new_requests: List[Any],
    all_ranks_num_active_requests: List[int],
    all_ranks_num_active_tokens: List[int],
    tp_size: int,
    max_num_active_requests: int,
) -> Tuple[Dict[int, List[Any]], int]:
    """Schedule attention DP requests across ranks.

    This function distributes requests across tensor parallel ranks for attention DP.
    It first tries to assign requests to their target dp_rank (if specified and has capacity),
    then balances the remaining requests across all ranks.

    Args:
        new_requests: List of RequestQueueItem to schedule.
        all_ranks_num_active_requests: Number of active requests per rank (will be modified).
        all_ranks_num_active_tokens: Number of active tokens per rank.
        tp_size: Number of tensor parallel ranks.
        max_num_active_requests: Maximum number of active requests per rank.

    Returns:
        Tuple of:
            - all_ranks_new_requests: Dict mapping rank to list of assigned requests.
            - expected_num_active_requests: Expected number of active requests per rank.
    """
    # Map from ranks to new requests
    all_ranks_new_requests = {tp_rank: [] for tp_rank in range(tp_size)}

    # Prioritize the requests that are not in relax mode
    def get_relax_value(req_item):
        scheduling_params = getattr(req_item.request, "py_scheduling_params", None)
        if scheduling_params is None:
            return True
        return scheduling_params.attention_dp_relax

    new_requests = sorted(new_requests, key=get_relax_value)

    # Try to put the requests to the target dp rank until the max_num_active_requests is reached
    remaining_unscheduled = []
    for req_item in new_requests:
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

    # Balance the remaining unscheduled requests across ranks
    num_new_requests_all_ranks = len(remaining_unscheduled)
    total_num_active_requests = sum(all_ranks_num_active_requests)
    expected_num_active_requests = max(
        (total_num_active_requests + num_new_requests_all_ranks + tp_size - 1) // tp_size,
        max(all_ranks_num_active_requests),
    )

    all_ranks_new_requests = balance_requests_across_ranks(
        remaining_unscheduled,
        all_ranks_new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        expected_num_active_requests,
    )

    return all_ranks_new_requests, expected_num_active_requests


def balance_requests_across_ranks(
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
        all_ranks_new_requests: Dict mapping rank to list of already assigned requests.
        all_ranks_num_active_requests: Number of active requests per rank.
        all_ranks_num_active_tokens: Number of active tokens per rank.
        expected_num_active_requests: Target number of active requests per rank.

    Returns:
        Updated all_ranks_new_requests dict with new requests distributed.
    """
    if new_requests:
        # Balance context tokens across ranks using heap
        HeapVal = namedtuple("HeapVal", ["num_tokens", "num_requests", "rank", "request_list"])

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

        # Sort by token count (descending) for better load balancing
        new_requests = sorted(
            new_requests,
            key=lambda x: len(getattr(x.request, "input_token_ids", [])) if x.request else 0,
            reverse=True,
        )

        # Distribute requests across ranks
        for req_item in new_requests:
            val = heapq.heappop(all_ranks_new_requests_heap)
            token_count = (
                len(getattr(req_item.request, "input_token_ids", [])) if req_item.request else 0
            )
            # Update the heap value with the new request
            val = val._replace(
                num_tokens=val.num_tokens + token_count,
                num_requests=val.num_requests + 1,
            )

            val.request_list.append(req_item)
            # If rank still has room for new requests, push back into heap
            if val.num_requests < expected_num_active_requests:
                heapq.heappush(all_ranks_new_requests_heap, val)

        # Extend all_ranks_new_requests with the new requests that have been scheduled
        for rank, reqs in all_ranks_new_scheduled_requests.items():
            all_ranks_new_requests[rank].extend(reqs)

    return all_ranks_new_requests


def can_process_attention_dp_request(
    req_item, all_ranks_num_active_requests: List[int], max_num_active_requests: int
) -> bool:
    """Check if a request can be processed immediately for attention DP.

    Args:
        req_item: The request queue item to check.
        all_ranks_num_active_requests: Number of active requests for each rank.
        max_num_active_requests: Maximum number of active requests per rank.

    Returns:
        True if the request can be processed, False otherwise.
    """
    scheduling_params = getattr(req_item.request, "py_scheduling_params", None)
    if scheduling_params is None:
        return True

    target_dp_rank = scheduling_params.attention_dp_rank
    if target_dp_rank is None or scheduling_params.attention_dp_relax:
        return True

    if all_ranks_num_active_requests[target_dp_rank] < max_num_active_requests:
        all_ranks_num_active_requests[target_dp_rank] += 1
        return True

    return False


def get_from_waiting_queue(
    waiting_queue: deque,
    max_req_count: int,
    enable_attention_dp: bool,
    max_num_active_requests: int,
    all_ranks_num_active_requests: Optional[List[int]] = None,
) -> List:
    """Get requests from the waiting queue.

    Args:
        waiting_queue: The queue to pop items from.
        max_req_count: Maximum items to retrieve. Returns empty list if <=0.
        enable_attention_dp: Whether to enable attention DP scheduling.
        max_num_active_requests: Maximum number of active requests per rank.
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
    scheduling_all_ranks_num_active_requests = (
        all_ranks_num_active_requests.copy() if enable_attention_dp else None
    )

    while req_count < max_req_count and waiting_queue:
        req_item = waiting_queue[0]
        num_children = len(req_item.child_req_ids) if req_item.child_req_ids else 0
        if (req_count + 1 + num_children) > max_req_count:
            break
        req_item = waiting_queue.popleft()

        can_process = (
            can_process_attention_dp_request(
                req_item, scheduling_all_ranks_num_active_requests, max_num_active_requests
            )
            if enable_attention_dp
            else True
        )

        if can_process:
            items.append(req_item)
            req_count += 1 + num_children
        else:
            pending_requests.append(req_item)

    # Put the pending requests back to the waiting queue
    # All ranks should have the same waiting queue
    waiting_queue.extendleft(reversed(pending_requests))

    return items


def partition_context_for_star_attention(
    ctx_ids_list: List[int], cp_rank: int, cp_size: int, block_size: int, anchor_block_size: int
) -> Tuple[List[List[int]], List[List[int]], int]:
    """Partition context for Star Attention CP.

    Args:
        ctx_ids_list: List of context token IDs.
        cp_rank: Current CP rank.
        cp_size: Total number of CP ranks.
        block_size: Size of each block.
        anchor_block_size: Size of anchor block.

    Returns:
        Tuple of (ctx_blocks, position_blocks, padding).
    """
    ctx_ids = torch.tensor(ctx_ids_list).unsqueeze(0)
    ctx_len = ctx_ids.shape[-1]

    if block_size is None:
        block_size = ctx_len // cp_size
    if anchor_block_size is None:
        anchor_block_size = block_size

    assert anchor_block_size <= block_size, (
        f"cp_anchor_size {anchor_block_size} should be smaller than block_size {block_size}"
    )

    padding = 0
    if ctx_len % block_size != 0:
        padding = block_size - (ctx_len % block_size)
        assert padding <= ctx_len, "block size is too large for context, please set it smaller"
        ctx_ids = torch.cat((ctx_ids, torch.zeros_like(ctx_ids)[:, :padding]), dim=-1)
    position_ids = torch.arange(0, ctx_ids.shape[-1]).unsqueeze(0)

    ctx_ids_blocks = torch.tensor_split(torch.stack(ctx_ids.split(block_size, dim=-1)), cp_size)
    position_ids_blocks = torch.tensor_split(
        torch.stack(position_ids.split(block_size, dim=-1)), cp_size
    )

    if cp_rank != 0:
        ctx_blocks = [ctx_ids_blocks[0][0].tolist()[0][:anchor_block_size]]
        position_blocks = [position_ids_blocks[0][0].tolist()[0][:anchor_block_size]]
    else:
        ctx_blocks, position_blocks = [], []

    for idx in range(len(ctx_ids_blocks[cp_rank])):
        ctx_block = ctx_ids_blocks[cp_rank][idx]
        position_block = position_ids_blocks[cp_rank][idx]
        ctx_blocks.append(ctx_block.tolist()[0])
        position_blocks.append(position_block.tolist()[0])

    return ctx_blocks, position_blocks, padding


def partition_context_for_helix(
    input_token_ids: List[int], cp_rank: int, cp_size: int, tokens_per_block: int
) -> Tuple[List[int], List[int], int, int]:
    """Partition context for Helix CP.

    Args:
        input_token_ids: List of input token IDs.
        cp_rank: Current CP rank.
        cp_size: Total number of CP ranks.
        tokens_per_block: Number of tokens per block.

    Returns:
        Tuple of (input_ids_this_rank, position_ids_this_rank, input_len, padding_len).

    Raises:
        ValueError: If there aren't enough tokens for at least one block per CP rank.
    """
    all_input_ids = torch.tensor(input_token_ids, dtype=torch.int64).unsqueeze(0)
    input_len = all_input_ids.shape[-1]

    num_total_blocks = (input_len + tokens_per_block - 1) // tokens_per_block
    if num_total_blocks < cp_size:
        raise ValueError(
            f"There aren't enough tokens to get at least one block per CP rank. "
            f"num_total_blocks {num_total_blocks} < num_cp_ranks {cp_size}. "
            f"Please use smaller tokens_per_block for KV cache or reduce the number of CP ranks."
        )

    # Padding to ensure torch.stack used with torch.tensor_split works properly.
    padding_len = 0
    if input_len % tokens_per_block != 0:
        padding_len = tokens_per_block - (input_len % tokens_per_block)
        padding_ids = torch.zeros([1, padding_len], dtype=torch.int64)
        all_input_ids = torch.cat((all_input_ids, padding_ids), dim=-1)
    all_position_ids = torch.arange(0, input_len + padding_len, dtype=torch.int64).unsqueeze(0)

    input_id_blocks_per_rank = torch.tensor_split(
        torch.stack(all_input_ids.split(tokens_per_block, dim=-1)), cp_size
    )
    position_id_blocks_per_rank = torch.tensor_split(
        torch.stack(all_position_ids.split(tokens_per_block, dim=-1)), cp_size
    )

    # Get the input_ids and position_ids for this rank.
    input_ids_this_rank = input_id_blocks_per_rank[cp_rank].flatten().tolist()
    position_ids_this_rank = position_id_blocks_per_rank[cp_rank].flatten().tolist()

    # Undo the padding. Only last rank's last block will be padded right now
    # given contiguous block assignment.
    if cp_rank == cp_size - 1 and padding_len > 0:
        input_ids_this_rank = input_ids_this_rank[:-padding_len]
        position_ids_this_rank = position_ids_this_rank[:-padding_len]

    return input_ids_this_rank, position_ids_this_rank, input_len, padding_len


def merge_requests_to_llm_requests(
    new_requests: List, exclude_last_generation_logits: bool
) -> List[LlmRequest]:
    """Merge RequestQueueItems to LlmRequests (basic case without CP).

    Args:
        new_requests: List of RequestQueueItem objects.
        exclude_last_generation_logits: Whether to exclude last generation logits.

    Returns:
        List of LlmRequest objects including child requests.
    """
    req_with_children = []
    for req_item in new_requests:
        req = executor_request_to_llm_request(
            req_item.id, req_item.request, req_item.child_req_ids, exclude_last_generation_logits
        )
        req_with_children.append(req)
        if req.child_requests:
            req_with_children.extend(req.child_requests)
    return req_with_children


def merge_helix_requests(
    new_requests: List,
    cp_rank: int,
    cp_size: int,
    tokens_per_block: int,
    exclude_last_generation_logits: bool,
) -> List[LlmRequest]:
    """Merge requests for Helix CP.

    Note: Helix parallelism is a decode-only feature run with disaggregated serving.
    This function gets called on gen server during initialization of a new request.

    Args:
        new_requests: List of RequestQueueItem objects.
        cp_rank: Current CP rank.
        cp_size: Total number of CP ranks.
        tokens_per_block: Number of tokens per block.
        exclude_last_generation_logits: Whether to exclude last generation logits.

    Returns:
        List of LlmRequest objects including child requests.
    """
    req_with_children = []

    for req_item in new_requests:
        input_ids_this_rank, position_ids_this_rank, input_len, _ = partition_context_for_helix(
            req_item.request.input_token_ids, cp_rank, cp_size, tokens_per_block
        )

        req = executor_request_to_llm_request(
            req_id=req_item.id,
            executor_request=req_item.request,
            child_req_ids=req_item.child_req_ids,
            exclude_last_generation_logits=exclude_last_generation_logits,
            input_token_ids=input_ids_this_rank,
            position_ids=position_ids_this_rank,
        )
        req.total_input_len_cp = input_len
        req.seqlen_this_rank_cp = len(input_ids_this_rank)
        req_with_children.append(req)
        if req.child_requests:
            req_with_children.extend(req.child_requests)

    return req_with_children


def merge_star_attention_requests(
    new_requests: List,
    cp_rank: int,
    cp_size: int,
    cp_config: dict,
    exclude_last_generation_logits: bool,
) -> List[LlmRequest]:
    """Merge requests for Star Attention CP.

    Args:
        new_requests: List of RequestQueueItem objects.
        cp_rank: Current CP rank.
        cp_size: Total number of CP ranks.
        cp_config: CP configuration dict containing 'block_size' and 'cp_anchor_size'.
        exclude_last_generation_logits: Whether to exclude last generation logits.

    Returns:
        List of LlmRequest objects.
    """
    result = []
    block_size = cp_config["block_size"]
    anchor_block_size = cp_config["cp_anchor_size"]

    for req_item in new_requests:
        req_id, exe_req, query_token_ids = req_item.id, req_item.request, req_item.query
        ctx_len0 = len(exe_req.input_token_ids)

        ctx_blocks, position_blocks, last_block_padding_num = partition_context_for_star_attention(
            exe_req.input_token_ids, cp_rank, cp_size, block_size, anchor_block_size
        )

        if cp_rank == cp_size - 1 and last_block_padding_num > 0:
            ctx_blocks[-1] = ctx_blocks[-1][:-last_block_padding_num]
            position_blocks[-1] = position_blocks[-1][:-last_block_padding_num]

        # if has query
        if query_token_ids:
            ctx_blocks.append(query_token_ids)
            position_blocks.append([i for i in range(ctx_len0, ctx_len0 + len(query_token_ids))])

        # insert the dummy block to align the number of ctx iterations of each rank
        total_blocks = (ctx_len0 + block_size - 1) // block_size
        num_blocks_per_rank = (total_blocks + cp_size - 1) // cp_size + 1  # 1 for query block
        if len(ctx_blocks) == num_blocks_per_rank:
            ctx_blocks.insert(1, [])
            position_blocks.insert(1, [])
        elif len(ctx_blocks) == num_blocks_per_rank + 1:
            # anchor + ctx_blocks + qry_block
            pass
        else:
            raise ValueError(
                f"Invalid context partition: rank = {cp_rank}, "
                f"len(ctx_blocks) = {len(ctx_blocks)}, "
                f"num_blocks_per_rank = {num_blocks_per_rank}"
            )

        # fake data for scheduler
        ctx_blocks_list = [0] * (block_size + anchor_block_size)

        req = executor_request_to_llm_request(
            req_id, exe_req, exclude_last_generation_logits, ctx_blocks_list
        )
        req.gen_iters = 0
        req.ctx_iters = 0
        req.ctx_blocks = ctx_blocks
        req.ctx_position_blocks = position_blocks
        req.query_id = query_token_ids

        result.append(req)

    return result


@nvtx_range("merge_requests")
def merge_requests(
    new_requests: List,
    cp_config: dict,
    cp_rank: int,
    cp_size: int,
    exclude_last_generation_logits: bool,
) -> List[LlmRequest]:
    """Merge RequestQueueItems to LlmRequests based on CP configuration.

    This is a router function that dispatches to the appropriate merge function
    based on the CP (Context Parallelism) configuration.

    Args:
        new_requests: List of RequestQueueItem objects.
        cp_config: CP configuration dict. May contain 'cp_type', 'tokens_per_block',
            'block_size', 'cp_anchor_size'.
        cp_rank: Current CP rank.
        cp_size: Total number of CP ranks.
        exclude_last_generation_logits: Whether to exclude last generation logits.

    Returns:
        List of LlmRequest objects.

    Raises:
        NotImplementedError: If cp_type is not supported.
    """
    if "cp_type" in cp_config:
        cp_type = cp_config["cp_type"]
        if cp_type == CpType.STAR:
            return merge_star_attention_requests(
                new_requests,
                cp_rank=cp_rank,
                cp_size=cp_size,
                cp_config=cp_config,
                exclude_last_generation_logits=exclude_last_generation_logits,
            )
        elif cp_type == CpType.HELIX:
            return merge_helix_requests(
                new_requests,
                cp_rank=cp_rank,
                cp_size=cp_size,
                tokens_per_block=cp_config["tokens_per_block"],
                exclude_last_generation_logits=exclude_last_generation_logits,
            )
        else:
            raise NotImplementedError(f"Unsupported cp type {cp_type.name}.")

    return merge_requests_to_llm_requests(new_requests, exclude_last_generation_logits)


class RequestBroadcaster:
    """Broadcasts requests across distributed ranks (TP, PP, CP)."""

    def __init__(self, dist: Distributed, hang_detector: HangDetector):
        self.dist = dist
        self.hang_detector = hang_detector
        self.send_requests_handler = None

    def broadcast(self, new_requests: List) -> Tuple[List, Optional[Tuple]]:
        """Broadcast requests and Python objects across ranks."""
        if self.dist.rank == 0:
            py_request_objects = self._collect_py_objects(new_requests)
        else:
            py_request_objects = None

        if self.dist.rank == 0:
            # Preserve original `new_requests` on rank 0
            _ = self._broadcast_requests(new_requests, py_request_objects)
        else:
            with self.hang_detector.pause():
                new_requests, py_request_objects = self._broadcast_requests(
                    new_requests, py_request_objects
                )

        return new_requests, py_request_objects

    def _collect_py_objects(self, new_requests: List) -> Tuple:
        """Collect Python-only objects from requests."""
        py_logits_post_processors = collect_py_objects_from_requests(
            new_requests, "py_logits_post_processors"
        )
        py_multimodal_data = collect_py_objects_from_requests(new_requests, "py_multimodal_data")
        py_scheduling_params = collect_py_objects_from_requests(
            new_requests, "py_scheduling_params"
        )
        py_num_logprobs = collect_py_objects_from_requests(new_requests, "py_num_logprobs")
        py_disaggregated_params = collect_py_objects_from_requests(
            new_requests, "py_disaggregated_params"
        )

        return tuple(
            filter(
                None,
                [
                    py_logits_post_processors,
                    py_multimodal_data,
                    py_scheduling_params,
                    py_num_logprobs,
                    py_disaggregated_params,
                ],
            )
        )

    @nvtx_range("broadcast_requests")
    def _broadcast_requests(
        self, new_requests: List, py_request_objects
    ) -> Tuple[List, Optional[Dict]]:
        """Broadcast requests across pipeline stages."""
        payloads = (new_requests, py_request_objects)

        if not self.dist.has_pp:
            return self.dist.broadcast(payloads, root=0)

        # Broadcast within first PP stage before send/recv chain to other PP stages.
        # This needs to cover both TP and CP ranks within the first PP stage.
        if self.dist.is_first_pp_rank:
            payloads = self.dist.tp_cp_broadcast(payloads, root=0)

        # Tag for communication
        tag = self.dist.pp_size  # Use pp_size as tag to avoid conflicts

        # Send payloads
        if not self.dist.is_first_pp_rank:
            with nvtx_range("recv_requests_from_prev_pp"):
                payloads = self.dist.recv_object(self.dist.prev_pp_rank, tag)

        # isend new requests may cause deadlock, when CUDA_LAUNCH_BLOCKING=1
        # or PP microbatches can't overlap, the deadlock will happen:
        # 1. rank1 will wait on nccl.send(rank2), without invoking mpi.wait(isend-handle)
        # 2. rank2 will wait on mpi.recv(rank1) but never receive the new requests.
        # 3. rank1 will hang on nccl.send because rank2 will never reach nccl.recv(rank1).
        pp_send_func = (
            self.dist.isend_object
            if os.environ.get("TRTLLM_PP_REQ_SEND_ASYNC", "0") == "1"
            else self.dist.send_object
        )

        if not self.dist.is_last_pp_rank:
            if self.send_requests_handler is not None:
                with nvtx_range("wait_prev_send_requests_handler"):
                    self.send_requests_handler.wait()
            with nvtx_range("send_requests_to_next_pp"):
                self.send_requests_handler = pp_send_func(payloads, self.dist.next_pp_rank, tag)

        return payloads

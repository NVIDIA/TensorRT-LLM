# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Request Utilities - Pure stateless functions for request operations.

This module provides utility functions that can be used by any component:
- Broadcasting requests across ranks (TP/PP/CP)
- Merging RequestQueueItem to LlmRequest
- Context partitioning for CP modes (Helix, StarAttention)

Design:
- All functions are pure: input → output, no side effects
- No state management - just data transformation and I/O
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

from tensorrt_llm._utils import nvtx_range

from tensorrt_llm._torch.pyexecutor.request_queue import RequestQueueItem

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.distributed import Distributed


# ========== Broadcasting Utilities ==========

class RequestBroadcaster:
    """
    Handles broadcasting requests across distributed ranks.
    
    Manages the PP send handler for async communication.
    Separated from Fetcher to keep broadcast logic isolated.
    """
    
    def __init__(self, dist: 'Distributed'):
        """
        Initialize broadcaster.
        
        Args:
            dist: Distributed communication handler
        """
        self.dist = dist
        self._send_requests_handler = None
    
    @nvtx_range("_broadcast_new_requests")
    def broadcast(
        self,
        new_requests: List[RequestQueueItem],
        py_request_objects: Optional[tuple],
    ) -> Tuple[List[RequestQueueItem], Optional[tuple]]:
        """
        Broadcast new_requests and Python objects across pipeline stages.
        
        Corresponds to ExecutorRequestQueue._broadcast_new_requests().
        
        Args:
            new_requests: Requests to broadcast (only valid on rank 0)
            py_request_objects: Python-only objects to broadcast
        
        Returns:
            Tuple of (requests, py_objects) after broadcast
        """
        payloads = (new_requests, py_request_objects)
        
        if not self.dist.has_pp:
            return self.dist.broadcast(payloads, root=0)
        
        # Broadcast within first PP stage before send/recv chain to other PP stages.
        # This needs to cover both TP and CP ranks within the first PP stage.
        if self.dist.is_first_pp_rank:
            payloads = self.dist.tp_cp_broadcast(payloads, root=0)
        
        # Tag for communication
        tag = self.dist.pp_size
        
        # Send payloads
        if not self.dist.is_first_pp_rank:
            with nvtx_range("recv_requests_from_prev_pp"):
                payloads = self.dist.recv_object(self.dist.prev_pp_rank, tag)
        
        if not self.dist.is_last_pp_rank:
            if self._send_requests_handler is not None:
                with nvtx_range("wait_prev_send_requests_handler"):
                    self._send_requests_handler.wait()
            with nvtx_range("send_requests_to_next_pp"):
                self._send_requests_handler = self.dist.isend_object(
                    payloads, self.dist.next_pp_rank, tag)
        
        return payloads


def collect_py_objects(
    requests: List[RequestQueueItem],
) -> Optional[tuple]:
    """
    Collect Python-only objects from requests for broadcasting.
    
    These objects cannot be pickled normally and need special handling.
    
    Args:
        requests: Requests to collect objects from
    
    Returns:
        Tuple of (attr_name, dict) pairs, or None if empty
    """
    def collect_attr(attribute_name: str) -> Optional[Tuple[str, Dict]]:
        req_id_to_obj = {}
        for item in requests:
            if not item.is_normal_request:
                continue
            if item.request:
                obj = getattr(item.request, attribute_name, None)
                if obj is not None:
                    req_id_to_obj[item.id] = obj
        return None if not req_id_to_obj else (attribute_name, req_id_to_obj)
    
    py_logits_post_processors = collect_attr("py_logits_post_processors")
    py_multimodal_data = collect_attr("py_multimodal_data")
    py_scheduling_params = collect_attr("py_scheduling_params")
    py_num_logprobs = collect_attr("py_num_logprobs")
    
    return tuple(filter(None, [
        py_logits_post_processors, py_multimodal_data,
        py_scheduling_params, py_num_logprobs
    ])) or None


def attach_py_objects(
    requests: List[RequestQueueItem],
    py_request_objects: tuple,
) -> None:
    """
    Attach Python-only objects to requests (on non-rank-0).
    
    Args:
        requests: Requests to attach objects to
        py_request_objects: Objects to attach
    """
    for attr_name, req_obj_dict in py_request_objects:
        for item in requests:
            if item.request:
                py_obj = req_obj_dict.get(item.id)
                if py_obj is not None:
                    setattr(item.request, attr_name, py_obj)


# ========== Request Merging Utilities ==========

@nvtx_range("_merge_requests")
def merge_requests(
    new_requests: List[RequestQueueItem],
    dist: 'Distributed',
    exclude_last_generation_logits: bool = False,
) -> List['LlmRequest']:
    """
    Convert RequestQueueItem to LlmRequest.
    
    Handles special CP modes (Helix, StarAttention) that require
    context partitioning.
    
    Corresponds to ExecutorRequestQueue._merge_requests().
    
    Args:
        new_requests: Items to convert
        dist: Distributed handler for CP info
        exclude_last_generation_logits: Whether to exclude last gen logits
    
    Returns:
        List of LlmRequests
    """
    from ..llm_request import executor_request_to_llm_request
    from tensorrt_llm.mapping import CpType
    
    cp_config = dist.cp_config
    if 'cp_type' in cp_config:
        cp_type = cp_config['cp_type']
        if cp_type == CpType.STAR:
            return _merge_star_attention_requests(
                new_requests, dist, exclude_last_generation_logits
            )
        elif cp_type == CpType.HELIX:
            return _merge_helix_requests(
                new_requests, dist,
                tokens_per_block=cp_config['tokens_per_block'],
                exclude_last_generation_logits=exclude_last_generation_logits,
            )
        else:
            raise NotImplementedError(f'Unsupported cp type {cp_type.name}.')
    
    req_with_children = []
    for req_item in new_requests:
        req = executor_request_to_llm_request(
            req_item.id, req_item.request, req_item.child_req_ids,
            exclude_last_generation_logits,
        )
        req_with_children.append(req)
        if req.child_requests:
            req_with_children.extend(req.child_requests)
    return req_with_children


def _merge_helix_requests(
    new_requests: List[RequestQueueItem],
    dist: 'Distributed',
    tokens_per_block: int,
    exclude_last_generation_logits: bool,
) -> List['LlmRequest']:
    """
    Merge requests for Helix CP mode.
    
    Helix partitions tokens into blocks and distributes across CP ranks
    in an interleaved fashion.
    
    Args:
        new_requests: Requests to merge
        dist: Distributed handler
        tokens_per_block: Block size for partitioning
        exclude_last_generation_logits: Whether to exclude last gen logits
    
    Returns:
        List of LlmRequests with partitioned tokens
    """
    from ..llm_request import executor_request_to_llm_request
    
    req_with_children = []
    num_cp_ranks = dist.cp_size
    curr_cp_rank = dist.cp_rank
    
    for req_item in new_requests:
        all_input_ids = torch.tensor(req_item.request.input_token_ids, dtype=torch.int64).unsqueeze(0)
        input_len = all_input_ids.shape[-1]
        
        num_total_blocks = (input_len + tokens_per_block - 1) // tokens_per_block
        if num_total_blocks < num_cp_ranks:
            raise ValueError(
                f"There aren't enough tokens to get at least one block per CP rank. "
                f"num_total_blocks {num_total_blocks} < num_cp_ranks {num_cp_ranks}."
            )
        
        padding_len = 0
        if input_len % tokens_per_block != 0:
            padding_len = tokens_per_block - (input_len % tokens_per_block)
            padding_ids = torch.zeros([1, padding_len], dtype=torch.int64)
            all_input_ids = torch.cat((all_input_ids, padding_ids), dim=-1)
        all_position_ids = torch.arange(0, input_len + padding_len, dtype=torch.int64).unsqueeze(0)
        
        input_id_blocks_per_rank = torch.tensor_split(
            torch.stack(all_input_ids.split(tokens_per_block, dim=-1)), num_cp_ranks)
        position_id_blocks_per_rank = torch.tensor_split(
            torch.stack(all_position_ids.split(tokens_per_block, dim=-1)), num_cp_ranks)
        
        input_ids_this_rank = input_id_blocks_per_rank[curr_cp_rank].flatten().tolist()
        position_ids_this_rank = position_id_blocks_per_rank[curr_cp_rank].flatten().tolist()
        
        if curr_cp_rank == num_cp_ranks - 1 and padding_len > 0:
            input_ids_this_rank = input_ids_this_rank[:-padding_len]
            position_ids_this_rank = position_ids_this_rank[:-padding_len]
        
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


def _merge_star_attention_requests(
    new_requests: List[RequestQueueItem],
    dist: 'Distributed',
    exclude_last_generation_logits: bool,
) -> List['LlmRequest']:
    """
    Merge requests for Star Attention CP mode.
    
    Star Attention uses an anchor block that is shared across all ranks,
    with the remaining context distributed.
    
    Args:
        new_requests: Requests to merge
        dist: Distributed handler
        exclude_last_generation_logits: Whether to exclude last gen logits
    
    Returns:
        List of LlmRequests with Star Attention partitioning
    """
    from ..llm_request import executor_request_to_llm_request
    
    result = []
    for req_item in new_requests:
        req_id, exe_req, query_token_ids = req_item.id, req_item.request, req_item.query
        ctx_len0 = len(exe_req.input_token_ids)
        ctx_blocks, position_blocks, last_block_padding_num = (
            _partition_context(exe_req.input_token_ids, dist)
        )
        
        if dist.cp_rank == dist.cp_size - 1 and last_block_padding_num > 0:
            ctx_blocks[-1] = ctx_blocks[-1][:-last_block_padding_num]
            position_blocks[-1] = position_blocks[-1][:-last_block_padding_num]
        
        if query_token_ids:
            ctx_blocks.append(query_token_ids)
            position_blocks.append([i for i in range(ctx_len0, ctx_len0 + len(query_token_ids))])
        
        block_size = dist.cp_config['block_size']
        total_blocks = (ctx_len0 + block_size - 1) // block_size
        num_blocks_per_rank = (total_blocks + dist.cp_size - 1) // dist.cp_size + 1
        
        if len(ctx_blocks) == num_blocks_per_rank:
            ctx_blocks.insert(1, [])
            position_blocks.insert(1, [])
        elif len(ctx_blocks) != num_blocks_per_rank + 1:
            assert False, f'invalid context partition'
        
        ctx_blocks_list = [0] * (block_size + dist.cp_config['cp_anchor_size'])
        
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


def _partition_context(ctx_ids_list: List[int], dist: 'Distributed'):
    """
    Partition context for Star Attention.
    
    Creates anchor block (shared) + distributed blocks per rank.
    
    Args:
        ctx_ids_list: Full context token IDs
        dist: Distributed handler
    
    Returns:
        Tuple of (ctx_blocks, position_blocks, padding)
    """
    ctx_ids = torch.tensor(ctx_ids_list).unsqueeze(0)
    ctx_len = ctx_ids.shape[-1]
    block_size = dist.cp_config['block_size']
    if block_size is None:
        block_size = ctx_len // dist.cp_size
    anchor_block_size = dist.cp_config['cp_anchor_size']
    if anchor_block_size is None:
        anchor_block_size = block_size
    
    assert anchor_block_size <= block_size
    padding = 0
    if ctx_len % block_size != 0:
        padding = block_size - (ctx_len % block_size)
        assert padding <= ctx_len
        ctx_ids = torch.cat((ctx_ids, torch.zeros_like(ctx_ids)[:, :padding]), dim=-1)
    position_ids = torch.arange(0, ctx_ids.shape[-1]).unsqueeze(0)
    
    ctx_ids_blocks = torch.tensor_split(
        torch.stack(ctx_ids.split(block_size, dim=-1)), dist.cp_size
    )
    position_ids_blocks = torch.tensor_split(
        torch.stack(position_ids.split(block_size, dim=-1)), dist.cp_size
    )
    
    if dist.cp_rank != 0:
        ctx_blocks = [ctx_ids_blocks[0][0].tolist()[0][:anchor_block_size]]
        position_blocks = [position_ids_blocks[0][0].tolist()[0][:anchor_block_size]]
    else:
        ctx_blocks, position_blocks = [], []
    
    for idx in range(len(ctx_ids_blocks[dist.cp_rank])):
        ctx_block = ctx_ids_blocks[dist.cp_rank][idx]
        position_block = position_ids_blocks[dist.cp_rank][idx]
        ctx_blocks.append(ctx_block.tolist()[0])
        position_blocks.append(position_block.tolist()[0])
    
    return ctx_blocks, position_blocks, padding

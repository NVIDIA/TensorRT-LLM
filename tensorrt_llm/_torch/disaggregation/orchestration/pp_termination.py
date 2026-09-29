# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.pp_utils import PPCommTag
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger


class DisaggPPTerminationHandler:
    """Handles termination synchronization across pipeline parallel ranks under disaggregated serving.

    We require synchronization when terminating requests in disaggregated PP when
    KV cache reuse is enabled. All PP ranks need to reach consensus before freeing
    resources to avoid a NCCL hang.
    """

    def __init__(self, dist, terminator_func: Callable[[LlmRequest], None]):
        self._dist = dist
        self._terminator_func = terminator_func
        self._pending_termination = {}
        self._terminating_iteration = 0
        self._send_handle = None
        self._comm_tag = PPCommTag.TERMINATION

    def terminate(self, request: LlmRequest):
        self._pending_termination[request.py_request_id] = request

    @nvtx_range("_disagg_pp_termination_handler_sync")
    def terminate_pending_requests(self):
        """
        Ring-style communicating to decide which requests to be terminated and avoid bubbles.
        This ensures that one request is terminated from rank_0 to rank_(pp_size-1) in order.
        """
        terminate_req_ids = []
        term_state = None
        if self._send_handle:
            self._send_handle.wait()

        if not (self._dist.is_first_pp_rank and self._terminating_iteration == 0):
            term_state = self._dist.recv_object(src=self._dist.prev_pp_rank, tag=self._comm_tag)

        ready_req_map = (
            term_state["ready"] if term_state else {}
        )  # {req_id: num_ranks} ranks vote in the ready dict
        terminate_req_ids = (
            term_state["term"] if term_state else []
        )  # request ids to be terminated in the current iteration

        reqs_to_terminate = {
            req_id: self._pending_termination.pop(req_id, None)
            for req_id in terminate_req_ids
            if req_id in self._pending_termination
        }

        if self._dist.is_first_pp_rank:
            # rank0 proposes the requests to be terminated
            ready_req_map = {req_id: 1 for req_id in self._pending_termination}
        else:
            # if a rank agrees to terminate a request, increase the vote count for the request id
            for req_id in ready_req_map.keys():
                if req_id in self._pending_termination:
                    ready_req_map[req_id] += 1

        if self._dist.is_last_pp_rank:
            new_terminate_req_ids = [
                req_id
                for req_id, num_ranks in ready_req_map.items()
                if num_ranks == self._dist.pp_size
            ]
            # by determining the terminate ids in the last rank, we can save the overhead of
            # sending the ready dict back to rank0
            new_term_state = {"ready": {}, "term": new_terminate_req_ids}
        else:
            # other pp ranks pass the updated ready dict and terminate request ids to the next rank, and the
            # terminate_req_ids will not change in a given iteration, so we can terminate the requests synchronously
            new_term_state = {"ready": ready_req_map, "term": terminate_req_ids}

        self._send_handle = self._dist.isend_object(
            new_term_state, dest=self._dist.next_pp_rank, tag=self._comm_tag
        )

        if reqs_to_terminate:
            logger.debug(
                f"rank {self._dist.pp_rank} terminates {list(reqs_to_terminate.keys())} "
                f"in iter {self._terminating_iteration}"
            )
        for req_id, req in reqs_to_terminate.items():
            if req:
                self._terminator_func(req)
        self._terminating_iteration += 1

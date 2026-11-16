# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Iterable, List, Optional

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


@dataclasses.dataclass
class DisaggTransferAdmissionResult:
    admitted_requests: List[LlmRequest]
    active_transfer_blocks: int = 0
    admitted_transfer_blocks: int = 0
    deferred_request_count: int = 0
    limited_by_budget: bool = False

    def is_blocked_by_active_transfers(self) -> bool:
        return (
            self.limited_by_budget
            and not self.admitted_requests
            and self.active_transfer_blocks > 0
        )


class DisaggTransferAdmissionController:
    """FCFS admission gate for disaggregated generation KV transfers.

    Args:
        max_tokens_in_buffer: The transfer budget in tokens; None or 0 disables the window.
        tokens_per_block: KV cache tokens per block.
        budget_is_explicit: The user set max_tokens_in_buffer (it was not defaulted), so the
            window also applies where the coordinator would bypass it
            (transfer_window_bypass_eligible).
    """

    def __init__(
        self,
        max_tokens_in_buffer: Optional[int],
        tokens_per_block: Optional[int],
        budget_is_explicit: bool = False,
    ) -> None:
        self.max_transfer_blocks = self._to_block_budget(max_tokens_in_buffer, tokens_per_block)
        self.tokens_per_block = tokens_per_block or 0
        self.budget_is_explicit = budget_is_explicit

    def enabled(self) -> bool:
        return self.max_transfer_blocks is not None

    @staticmethod
    def _to_block_budget(
        max_tokens_in_buffer: Optional[int], tokens_per_block: Optional[int]
    ) -> Optional[int]:
        if (
            max_tokens_in_buffer is None
            or max_tokens_in_buffer == 0
            or tokens_per_block is None
            or tokens_per_block <= 0
        ):
            return None
        return (max_tokens_in_buffer + tokens_per_block - 1) // tokens_per_block

    @staticmethod
    def _to_nonnegative_int(value) -> Optional[int]:
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return None

    def _get_request_transfer_token_count(self, request: LlmRequest) -> int:
        for attr_name in ("total_input_len_cp", "py_prompt_len", "prompt_len"):
            token_count = self._to_nonnegative_int(getattr(request, attr_name, None))
            if token_count is not None:
                return token_count
        return 0

    def _estimate_request_blocks(self, request: LlmRequest) -> int:
        if self.tokens_per_block <= 0:
            return 0
        prompt_len = self._get_request_transfer_token_count(request)
        return (prompt_len + self.tokens_per_block - 1) // self.tokens_per_block

    def _estimate_requests_blocks(self, requests: Iterable[LlmRequest]) -> int:
        return sum(self._estimate_request_blocks(request) for request in requests)

    def _estimate_active_transfer_blocks(self, active_requests: Iterable[LlmRequest]) -> int:
        return sum(
            self._estimate_request_blocks(request)
            for request in active_requests
            if request.is_disagg_generation_transmission_in_progress
        )

    def select(
        self, active_requests: Iterable[LlmRequest], candidates: List[LlmRequest]
    ) -> DisaggTransferAdmissionResult:
        if not self.enabled():
            return DisaggTransferAdmissionResult(
                admitted_requests=list(candidates),
                active_transfer_blocks=self._estimate_active_transfer_blocks(active_requests),
                admitted_transfer_blocks=self._estimate_requests_blocks(candidates),
            )

        result = DisaggTransferAdmissionResult(admitted_requests=[])
        result.active_transfer_blocks = self._estimate_active_transfer_blocks(active_requests)

        used_blocks = result.active_transfer_blocks
        max_transfer_blocks = self.max_transfer_blocks
        assert max_transfer_blocks is not None
        for request in candidates:
            request_blocks = self._estimate_request_blocks(request)
            fits_budget = used_blocks + request_blocks <= max_transfer_blocks
            admit_oversized_head = (
                not result.admitted_requests
                and result.active_transfer_blocks == 0
                and request_blocks > max_transfer_blocks
            )
            if not fits_budget and not admit_oversized_head:
                result.limited_by_budget = True
                break

            result.admitted_requests.append(request)
            used_blocks += request_blocks
            result.admitted_transfer_blocks += request_blocks

        result.deferred_request_count = len(candidates) - len(result.admitted_requests)
        return result

    def begin_gate(
        self, active_requests: Iterable[LlmRequest]
    ) -> Optional["DisaggTransferAdmissionGate"]:
        """A pre-allocation gate for one scheduling pass, or None when the budget is disabled
        (select() then admits every candidate)."""
        if not self.enabled():
            return None
        return DisaggTransferAdmissionGate(self, active_requests)


class DisaggTransferAdmissionGate:
    """Pre-allocation view of DisaggTransferAdmissionController.select() for one scheduling pass.

    KVCacheV2Scheduler allocates KV for every DISAGG_GENERATION_INIT request it visits. select()
    then admits the longest FCFS prefix of those candidates that fits the transfer budget, and the
    coordinator reverts every later candidate's allocation. With many waiting requests, most of
    each iteration's host time goes to allocating and reverting KV for requests that cannot be
    admitted.

    select() stops at the first candidate that misses the budget and defers every candidate after
    it. The scheduler reports each candidate here after its allocation succeeds. Once one misses
    the budget the gate closes, and the scheduler skips the remaining generation-init requests of
    the pass without allocating them. The candidate that closed the gate is still passed on, so
    select() sees the same prefix, admits the same requests, reports the same limited_by_budget,
    and reverts only that one allocation.
    """

    def __init__(
        self, controller: DisaggTransferAdmissionController, active_requests: Iterable[LlmRequest]
    ) -> None:
        self._controller = controller
        self.active_transfer_blocks = controller._estimate_active_transfer_blocks(active_requests)
        self._used_blocks = self.active_transfer_blocks
        self._num_fitting = 0
        self.closed = False
        # Generation-init requests skipped without allocation in this pass.
        self.num_skipped = 0

    def observe_candidate(self, request: LlmRequest) -> None:
        """Account a candidate whose KV allocation succeeded, with the rule of select(): FCFS,
        plus an oversized head when nothing is in flight."""
        if self.closed:
            return
        max_transfer_blocks = self._controller.max_transfer_blocks
        request_blocks = self._controller._estimate_request_blocks(request)
        fits_budget = self._used_blocks + request_blocks <= max_transfer_blocks
        admit_oversized_head = (
            self._num_fitting == 0
            and self.active_transfer_blocks == 0
            and request_blocks > max_transfer_blocks
        )
        if fits_budget or admit_oversized_head:
            self._num_fitting += 1
            self._used_blocks += request_blocks
        else:
            self.closed = True

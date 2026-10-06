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
    """FCFS admission gate for disaggregated generation KV transfers."""

    def __init__(
        self, max_tokens_in_buffer: Optional[int], tokens_per_block: Optional[int]
    ) -> None:
        self.max_transfer_blocks = self._to_block_budget(max_tokens_in_buffer, tokens_per_block)
        self.tokens_per_block = tokens_per_block or 0
        self._scheduling_result: DisaggTransferAdmissionResult | None = None

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

    def _estimate_active_transfer_blocks(self, active_requests: Iterable[LlmRequest]) -> int:
        return sum(
            self._estimate_request_blocks(request)
            for request in active_requests
            if request.is_disagg_generation_transmission_in_progress
        )

    def start_scheduling(
        self, active_requests: Iterable[LlmRequest]
    ) -> DisaggTransferAdmissionResult:
        """Start a fresh allocation pass, including its active-transfer backpressure."""
        self._scheduling_result = self._new_result(active_requests)
        return self._scheduling_result

    @property
    def scheduling_blocked_by_active_transfers(self) -> bool:
        """Whether the current allocation pass is waiting for transfers, not KV space."""
        return (
            self._scheduling_result is not None
            and self._scheduling_result.is_blocked_by_active_transfers()
        )

    def _new_result(self, active_requests: Iterable[LlmRequest]) -> DisaggTransferAdmissionResult:
        return DisaggTransferAdmissionResult(
            admitted_requests=[],
            active_transfer_blocks=self._estimate_active_transfer_blocks(active_requests),
        )

    def can_admit(self, result: DisaggTransferAdmissionResult, request: LlmRequest) -> bool:
        """Check the next FCFS candidate; a budget rejection closes this pass."""
        if not self.enabled():
            return True
        if result.limited_by_budget:
            return False
        request_blocks = self._estimate_request_blocks(request)
        max_transfer_blocks = self.max_transfer_blocks
        assert max_transfer_blocks is not None
        fits_budget = (
            result.active_transfer_blocks + result.admitted_transfer_blocks + request_blocks
            <= max_transfer_blocks
        )
        admit_oversized_head = (
            not result.admitted_requests
            and result.active_transfer_blocks == 0
            and request_blocks > max_transfer_blocks
        )
        result.limited_by_budget = not (fits_budget or admit_oversized_head)
        return not result.limited_by_budget

    def commit(self, result: DisaggTransferAdmissionResult, request: LlmRequest) -> None:
        """Charge an accepted candidate; allocation callers commit only on success."""
        result.admitted_requests.append(request)
        result.admitted_transfer_blocks += self._estimate_request_blocks(request)

    def select(
        self, active_requests: Iterable[LlmRequest], candidates: List[LlmRequest]
    ) -> DisaggTransferAdmissionResult:
        result = self._new_result(active_requests)
        for request in candidates:
            if not self.can_admit(result, request):
                break
            self.commit(result, request)

        result.deferred_request_count = len(candidates) - len(result.admitted_requests)
        return result

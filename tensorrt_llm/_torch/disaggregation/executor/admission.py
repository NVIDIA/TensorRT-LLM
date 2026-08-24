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

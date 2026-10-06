# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import TYPE_CHECKING, Iterable, List, Optional

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2


@dataclasses.dataclass
class DisaggTransferAdmissionResult:
    admitted_requests: List[LlmRequest]
    active_transfer_cost: int = 0
    admitted_transfer_cost: int = 0
    deferred_request_count: int = 0
    limited_by_budget: bool = False

    def is_blocked_by_active_transfers(self) -> bool:
        return (
            self.limited_by_budget and not self.admitted_requests and self.active_transfer_cost > 0
        )


class DisaggTransferAdmissionController:
    """FCFS admission gate for disaggregated generation KV transfers."""

    def __init__(
        self,
        max_tokens_in_buffer: Optional[int],
        tokens_per_block: Optional[int],
        *,
        max_transfer_bytes: Optional[int] = None,
        python_transceiver: Optional["KvCacheTransceiverV2"] = None,
    ) -> None:
        if (max_transfer_bytes is None) != (python_transceiver is None):
            raise ValueError("A byte budget requires both capacity and a Python transceiver")
        if max_transfer_bytes is not None and max_transfer_bytes <= 0:
            raise ValueError("The byte budget must be positive")
        self.max_transfer_cost = (
            max_transfer_bytes
            if max_transfer_bytes is not None
            else self._to_block_budget(max_tokens_in_buffer, tokens_per_block)
        )
        self.cost_unit = "bytes" if max_transfer_bytes is not None else "blocks"
        self._python_transceiver = python_transceiver
        self.tokens_per_block = tokens_per_block or 0
        # Refreshed by the V2 scheduler before coordinator admission each pass.
        self.early_admission_blocked = False

    def enabled(self) -> bool:
        return self.max_transfer_cost is not None

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

    def estimate_request_cost(self, request: LlmRequest) -> int:
        if self._python_transceiver is not None:
            return self._python_transceiver.get_receive_admission_request_bytes(request)
        if self.tokens_per_block <= 0:
            return 0
        prompt_len = self._get_request_transfer_token_count(request)
        return (prompt_len + self.tokens_per_block - 1) // self.tokens_per_block

    def estimate_active_transfer_cost(self, active_requests: Iterable[LlmRequest]) -> int:
        """Snapshot the total cost of transfers in progress for this scheduling pass."""
        return sum(
            self.estimate_request_cost(request)
            for request in active_requests
            if request.is_disagg_generation_transmission_in_progress
        )

    def allows(self, request_cost: int, used_cost: int, *, has_admitted: bool) -> bool:
        """Only an empty window may admit an oversized head request."""
        return (
            self.max_transfer_cost is None
            or used_cost + request_cost <= self.max_transfer_cost
            or (used_cost == 0 and not has_admitted)
        )

    def select(
        self, active_requests: Iterable[LlmRequest], candidates: List[LlmRequest]
    ) -> DisaggTransferAdmissionResult:
        active_transfer_cost = self.estimate_active_transfer_cost(active_requests)
        result = DisaggTransferAdmissionResult(
            admitted_requests=[], active_transfer_cost=active_transfer_cost
        )
        for request in candidates:
            request_cost = self.estimate_request_cost(request)
            used_cost = active_transfer_cost + result.admitted_transfer_cost
            if not self.allows(
                request_cost, used_cost, has_admitted=bool(result.admitted_requests)
            ):
                result.limited_by_budget = True
                break
            result.admitted_requests.append(request)
            result.admitted_transfer_cost += request_cost

        result.deferred_request_count = len(candidates) - len(result.admitted_requests)
        return result

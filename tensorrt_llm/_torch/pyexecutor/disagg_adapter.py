# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyExecutor implementations of the disagg orchestration interfaces."""

from typing import TYPE_CHECKING, Collection, List, Optional, Sequence

from ..disaggregation.orchestration.interfaces import ActiveRequestRegistry, ExecutorEffects
from .llm_request import LlmRequest, LlmResponse

if TYPE_CHECKING:
    from .py_executor import PyExecutor


class PyExecutorEffects(ExecutorEffects):
    def __init__(self, executor: "PyExecutor") -> None:
        self._executor = executor

    def terminate_request(self, request: LlmRequest) -> None:
        self._executor._terminate_request(request)

    def stage_transfer_response(
        self, request_id: int, response: LlmResponse, terminate_after_publish: Optional[LlmRequest]
    ) -> None:
        self._executor._pending_transfer_responses.append((request_id, response))
        if terminate_after_publish is not None:
            self._executor._pending_response_terminations.append(terminate_after_publish)

    def fail_requests(
        self, error_msg: str, requests: List[LlmRequest], *, charge_budget: bool
    ) -> None:
        self._executor._handle_errors(
            error_msg=error_msg, requests=requests, charge_budget=charge_budget
        )

    def fail_fatal(self, error_msg: str) -> None:
        executor = self._executor
        executor._fatal_error = RuntimeError(f"Fatal error: {error_msg}")
        executor.is_shutdown = True
        executor._handle_errors(
            error_msg, requests=None, charge_budget=False, fatal_is_collective_aligned=True
        )


class PyExecutorRequestRegistry(ActiveRequestRegistry):
    def __init__(self, executor: "PyExecutor") -> None:
        self._executor = executor

    def active_requests(self) -> Sequence[LlmRequest]:
        return self._executor.active_requests

    def contains(self, request: LlmRequest) -> bool:
        return request in self._executor.active_requests

    def remove(self, request: LlmRequest) -> None:
        self._executor.active_requests.remove(request)

    def canceled_request_ids(self) -> Collection[int]:
        return self._executor.canceled_req_ids

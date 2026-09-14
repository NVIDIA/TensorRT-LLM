# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateful fake of the executor side of the coordinator's interfaces.

Records what the coordinator asked the executor to do, in order, so tests can
assert on outcomes (which requests were terminated, which responses were
staged, which failed) rather than on call shapes.
"""

from typing import Collection, List, Optional, Sequence, Tuple

from tensorrt_llm._torch.disaggregation.orchestration.interfaces import (
    ActiveRequestRegistry,
    ExecutorEffects,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmResponse


class FakeExecutorEffects(ExecutorEffects):
    def __init__(self) -> None:
        self.terminated: List[LlmRequest] = []
        # (request_id, response, request to terminate after publish or None)
        self.staged_responses: List[Tuple[int, LlmResponse, Optional[LlmRequest]]] = []
        # (error_msg, requests, charge_budget)
        self.failed: List[Tuple[str, List[LlmRequest], bool]] = []
        # Interleaved history of every effect, for relative-order assertions.
        self.history: List[Tuple[str, object]] = []
        # Optional exception raised from fail_requests, to model a fatal error.
        self.fail_raises: Optional[BaseException] = None

    def terminate_request(self, request: LlmRequest) -> None:
        self.terminated.append(request)
        self.history.append(("terminate", request))

    def stage_transfer_response(
        self,
        request_id: int,
        response: LlmResponse,
        terminate_after_publish: Optional[LlmRequest],
    ) -> None:
        self.staged_responses.append((request_id, response, terminate_after_publish))
        self.history.append(("stage", request_id))

    def fail_requests(
        self, error_msg: str, requests: List[LlmRequest], *, charge_budget: bool
    ) -> None:
        self.failed.append((error_msg, list(requests), charge_budget))
        self.history.append(("fail", error_msg))
        if self.fail_raises is not None:
            raise self.fail_raises


class FakeRequestRegistry(ActiveRequestRegistry):
    """Registry over a caller-owned active list, read live on every call."""

    def __init__(self, active_requests: List[LlmRequest], canceled_request_ids=()) -> None:
        self.active = active_requests
        self.canceled = list(canceled_request_ids)
        self.removed: List[LlmRequest] = []

    def active_requests(self) -> Sequence[LlmRequest]:
        return tuple(self.active)

    def contains(self, request: LlmRequest) -> bool:
        return request in self.active

    def remove(self, request: LlmRequest) -> None:
        self.active.remove(request)
        self.removed.append(request)

    def canceled_request_ids(self) -> Collection[int]:
        return self.canceled

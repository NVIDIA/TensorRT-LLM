# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor-side abstractions the disagg coordinator depends on.

The coordinator never holds the executor. It reaches executor-owned
behavior only through these Protocols; ``PyExecutor`` implements them with
thin adapters (see ``pyexecutor/disagg_adapter.py``).
"""

from typing import Collection, List, Optional, Protocol, Sequence

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmResponse


class ExecutorEffects(Protocol):
    """Executor-owned side effects the coordinator may trigger.

    These three are the complete set; adding one is a design decision, not
    a convenience.
    """

    def terminate_request(self, request: LlmRequest) -> None:
        """Release the request's resources and routing; the executor decides
        whether PP termination consensus applies."""
        ...

    def stage_transfer_response(
        self,
        request_id: int,
        response: LlmResponse,
        terminate_after_publish: Optional[LlmRequest],
    ) -> None:
        """Queue a response for the executor's rank-synchronized flush.

        ``terminate_after_publish`` names a request that must be terminated
        only after that flush has published the response.
        """
        ...

    def fail_requests(
        self,
        error_msg: str,
        requests: List[LlmRequest],
        *,
        charge_budget: bool,
    ) -> None:
        """Fail requests through the executor's error path."""
        ...


class ActiveRequestRegistry(Protocol):
    """The executor's request bookkeeping, as the coordinator may touch it.

    ``remove`` is the only mutation. Callers must re-read on every use: the
    executor may rebind its list.
    """

    def active_requests(self) -> Sequence[LlmRequest]:
        """Live read-only view of the active requests, for scans."""
        ...

    def contains(self, request: LlmRequest) -> bool: ...

    def remove(self, request: LlmRequest) -> None:
        """Drop a request whose transfer released it; it is no longer active."""
        ...

    def canceled_request_ids(self) -> Collection[int]: ...

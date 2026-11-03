# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Types shared by the disaggregated client, service and server."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Union

from tensorrt_llm.serve.openai_protocol import (UCompletionRequest,
                                                UCompletionResponse)


class ResponseHooks(ABC):
    """
    Hooks for response processing and (disagg) service perf observability.
    """

    @abstractmethod
    def on_req_begin(self, request: UCompletionRequest):
        pass

    def on_disagg_request_id(self, disagg_request_id: int):
        """Receive the request ID immediately after the service allocates it."""

    def on_ctx_dispatch(self, request: UCompletionRequest):
        """Record when the disaggregated service starts context placement.

        Arrival to this point measures the pre-context wait in the orchestrator
        or fleet. The default is a no-op for non-instrumented implementations.
        """

    def on_perf_metrics(self, server: str, role: str, metrics: dict):
        """Receive request-local metrics carried by an upstream response."""

    @abstractmethod
    def on_ctx_resp(self, ctx_server: str, response: UCompletionResponse):
        pass

    @abstractmethod
    def on_first_token(self,
                       gen_server: str,
                       request: UCompletionRequest,
                       response: UCompletionResponse = None):
        pass

    @abstractmethod
    def on_resp_done(self,
                     gen_server: str,
                     request: UCompletionRequest,
                     response: UCompletionResponse = None):
        pass


async def done_generator() -> AsyncGenerator[bytes, None]:
    yield "data: [DONE]\n\n".encode('utf-8')


UCompletionResponseOrGenerator = Union[UCompletionResponse,
                                       AsyncGenerator[Any, None]]

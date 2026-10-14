# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared doubles for the OpenEngine adapter tests.

One gRPC context fake for all three modules. Three copies had already diverged
-- `add_done_callback` recorded callbacks in one and was a no-op in another --
so the files disagreed about whether the servicer's RPC-done callback fires,
and a change to the context contract would be caught by whichever copy happened
to model it.
"""

from types import SimpleNamespace
from typing import Any, Sequence

import grpc


class AbortError(Exception):
    """Raised by FakeServicerContext.abort, as grpc.aio's abort does."""

    def __init__(self, code: grpc.StatusCode | None = None, details: str | None = None) -> None:
        super().__init__(details)
        self.code = code
        self.details = details


class FakeServicerContext:
    """Stands in for grpc.aio.ServicerContext.

    Records the status an RPC aborted with and the done-callbacks the servicer
    registered; `cancelled` is settable so the RPC-cancellation path is
    reachable.
    """

    def __init__(self, metadata: Sequence[tuple[str, str]] = ()) -> None:
        self._metadata = [SimpleNamespace(key=key, value=value) for key, value in metadata]
        self.abort_code: grpc.StatusCode | None = None
        self.abort_details: str | None = None
        self.done_callbacks: list[Any] = []
        self.is_cancelled = False

    def invocation_metadata(self) -> list[SimpleNamespace]:
        return self._metadata

    def cancelled(self) -> bool:
        return self.is_cancelled

    def add_done_callback(self, callback: Any) -> None:
        self.done_callbacks.append(callback)

    async def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.abort_code = code
        self.abort_details = details
        raise AbortError(code, details)

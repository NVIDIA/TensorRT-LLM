# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Errors raised by the OpenEngine client."""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc

if TYPE_CHECKING:
    from tensorrt_llm.grpc.openengine.bindings import generation_pb2


class OpenEngineClientError(Exception):
    """Base class for the OpenEngine client's own errors.

    Args:
        message: Human-readable description.
        target: The ``host:port`` of the engine the client talks to.
        request_id: The Generate request the error belongs to, if any.
    """

    def __init__(self, message: str, *, target: str, request_id: str | None = None) -> None:
        super().__init__(message)
        self.target = target
        self.request_id = request_id


class OpenEngineRpcError(OpenEngineClientError):
    """An RPC failed, identified by a gRPC status code.

    Usually the non-OK status the RPC ended with, from the engine or from gRPC
    itself. The client also reports ``UNAVAILABLE`` when the engine does not
    become ready within ``connect()``'s ``ready_timeout``, and ``CANCELLED``
    when a Generate call is cancelled other than through its stream.

    Args:
        code: The gRPC status code.
        details: The status details.
        rpc: The RPC name, such as ``"Generate"`` or ``"GetServerInfo"``.
        target: See :class:`OpenEngineClientError`.
        request_id: See :class:`OpenEngineClientError`.
    """

    def __init__(
        self,
        code: grpc.StatusCode,
        details: str,
        *,
        rpc: str,
        target: str,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            f"OpenEngine {rpc} on {target} failed with {code.name}: {details}",
            target=target,
            request_id=request_id,
        )
        self.code = code
        self.details = details
        self.rpc = rpc


class OpenEngineEngineError(OpenEngineClientError):
    """The engine ended an accepted Generate request with an ``EngineError`` event.

    Args:
        code: The ``openengine.v1.ErrorCode`` value.
        engine_message: The message the engine attached to the event.
        retryable: Whether the engine says the same request may succeed on retry.
        usage: The cumulative usage carried by the terminal event, if any.
        target: See :class:`OpenEngineClientError`.
        request_id: See :class:`OpenEngineClientError`.
    """

    def __init__(
        self,
        code: int,
        engine_message: str,
        *,
        retryable: bool,
        usage: generation_pb2.Usage | None,
        target: str,
        request_id: str,
    ) -> None:
        super().__init__(
            f"OpenEngine request {request_id} on {target} failed with engine error "
            f"{code}: {engine_message}",
            target=target,
            request_id=request_id,
        )
        self.code = code
        self.engine_message = engine_message
        self.retryable = retryable
        self.usage = usage


class OpenEngineProtocolError(OpenEngineClientError):
    """The engine violated the OpenEngine protocol, for example a malformed response stream."""


class OpenEngineIncompatibleServerError(OpenEngineClientError):
    """The engine's ``ServerInfo`` reports a schema revision this client cannot use."""


__all__ = [
    "OpenEngineClientError",
    "OpenEngineEngineError",
    "OpenEngineIncompatibleServerError",
    "OpenEngineProtocolError",
    "OpenEngineRpcError",
]

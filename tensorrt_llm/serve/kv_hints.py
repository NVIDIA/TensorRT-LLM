# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-controlled hint trust policy applied before request model validation."""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.routing import APIRoute

from tensorrt_llm.kv_hints.metrics import count_ignored_kv_hints


class KvHintsRequest(Request):
    """Request that strips untrusted optional hints before nested validation."""

    def filter_kv_hints(self, body: object) -> object:
        """Strip and count hints unless this server process trusts its ingress.

        Args:
            body: Decoded request body; non-object bodies keep normal validation.

        Returns:
            The decoded body with untrusted hints removed. Required P/D handoff
            parameters and their independent authentication remain unchanged.
        """
        if (
            isinstance(body, dict)
            and "kv_hints" in body
            and not getattr(self.app.state, "trust_kv_hints", False)
        ):
            if body.pop("kv_hints") is not None:
                count_ignored_kv_hints("untrusted")
        return body

    async def json(self) -> object:
        """Decode JSON and apply the server's trust policy before validation.

        Returns:
            The filtered JSON body, cached by the underlying request.
        """
        return self.filter_kv_hints(await super().json())


class KvHintsRoute(APIRoute):
    """Apply hint ingress policy to typed FastAPI endpoints."""

    request_type = KvHintsRequest

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        """Wrap FastAPI validation with the configured request decoder.

        Returns:
            An ASGI request handler that filters hints before model validation.
        """
        original_route_handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            return await original_route_handler(self.request_type(request.scope, request.receive))

        return route_handler

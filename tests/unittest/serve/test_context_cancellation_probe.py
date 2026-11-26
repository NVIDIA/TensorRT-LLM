# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only authentication of the frontend-to-CTX control path."""

import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from test_common.serve.app_factory import AggConfig, ServeTestEnv

from tensorrt_llm.executor.rpc import RPCTimeout
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.disagg_auth import (
    CONTEXT_CANCELLATION_AUTH_HEADER,
    CONTEXT_PROBE_HEADER,
    build_context_cancellation_headers,
    build_context_probe_headers,
    context_cancellation_reservation_payload,
)
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.router import (
    ContextCancellationCapabilityError,
    RoundRobinRouter,
    ServerPreparationPending,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def probe_worker(serve_env: ServeTestEnv) -> OpenAIServer:
    """Use the real HTTP handlers with a CTX-capable fake executor."""
    worker = serve_env.build_openai_server(AggConfig(role=ServerRole.CONTEXT))
    worker._internal_disagg_auth_key = "secret"
    worker.generator.args.cache_transceiver_config = SimpleNamespace(
        backend="DEFAULT", kv_transfer_timeout_ms=None
    )
    worker.generator._disaggregated_params = {}
    worker.generator._startup_metrics = {}
    executor = worker.generator._executor
    executor.get_kv_cache_capacity = Mock(return_value={})
    executor.supports_logical_context_cancellation = Mock(return_value=True)
    executor.reserve_context_cancellation = Mock()
    executor.cancel_context_cancellation = Mock()
    return worker


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "key_mismatch",
        "no_worker_key",
        "generation",
        "no_transport",
        "unavailable",
        "transport_unavailable",
        "legacy_worker",
        "stale_reply",
        "redirect",
        "ordinary",
    ],
)
async def test_frontend_proves_worker_control_path_without_reserving(
    probe_worker: OpenAIServer, mode: str
) -> None:
    """Run preparation through HTTP, including legacy, redirected, and unavailable responses."""
    executor = probe_worker.generator._executor
    if mode == "no_worker_key":
        probe_worker._internal_disagg_auth_key = None
    elif mode == "generation":
        probe_worker.server_role = ServerRole.GENERATION
    elif mode == "no_transport":
        probe_worker.generator.args.cache_transceiver_config = None
    elif mode == "unavailable":
        executor.supports_logical_context_cancellation.side_effect = RPCTimeout("unavailable")

    received = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=probe_worker.app), base_url="http://worker"
    ) as asgi:

        async def serve(request: web.Request) -> web.Response:
            """Pass the router request through ASGI, optionally presenting a legacy reply."""
            received.append((request.path, request.headers.get(CONTEXT_PROBE_HEADER)))
            if mode == "redirect":
                return web.Response(status=307, headers={"Location": "/redirected"})
            if mode == "transport_unavailable":
                return web.Response(status=503)
            if mode == "legacy_worker":
                return web.json_response({"supports_logical_context_cancellation": True})
            headers = (
                build_context_probe_headers("secret") if mode == "stale_reply" else request.headers
            )
            response = await asgi.get("/server_info", headers=dict(headers))
            proof = response.headers.get(CONTEXT_CANCELLATION_AUTH_HEADER)
            return web.json_response(
                response.json(),
                status=response.status_code,
                headers={CONTEXT_CANCELLATION_AUTH_HEADER: proof} if proof is not None else None,
            )

        app = web.Application()
        app.router.add_get("/{path}", serve)
        async with TestServer(app) as http:
            server = str(http.make_url("")).rstrip("/")
            router = RoundRobinRouter(
                ServerRole.CONTEXT,
                [server],
                require_context_cancellation=mode != "ordinary",
                context_auth_key="wrong" if mode == "key_mismatch" else "secret",
            )
            try:
                if mode in ("valid", "ordinary"):
                    await router.prepare_servers()
                    assert router.prepared_servers == {server}
                else:
                    error = (
                        ServerPreparationPending
                        if mode in ("unavailable", "transport_unavailable")
                        else ContextCancellationCapabilityError
                    )
                    with pytest.raises(error):
                        await router.prepare_servers()
                    assert not router.prepared_servers
            finally:
                await router.close()

    assert len(received) == 1 and received[0][0] == "/server_info"
    assert (received[0][1] is None) == (mode == "ordinary")
    executor.reserve_context_cancellation.assert_not_called()
    executor.cancel_context_cancellation.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("clock_offset_s", [-61, 61])
async def test_worker_rejects_clock_skew_without_signing(
    clock_offset_s: int, probe_worker: OpenAIServer
) -> None:
    """A correct key does not authorize an expired or too-far-future challenge."""
    now_ns = time.time_ns()
    with patch.object(time, "time_ns", return_value=now_ns + clock_offset_s * 10**9):
        headers = build_context_probe_headers("secret")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=probe_worker.app), base_url="http://worker"
    ) as client:
        with patch.object(time, "time_ns", return_value=now_ns):
            response = await client.get("/server_info", headers=headers)
    assert response.status_code == 403 and CONTEXT_CANCELLATION_AUTH_HEADER not in response.headers
    probe_worker.generator._executor.supports_logical_context_cancellation.assert_not_called()
    probe_worker.generator._executor.reserve_context_cancellation.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["reserve", "cancel"])
async def test_worker_without_control_operation_returns_conflict(
    probe_worker: OpenAIServer, action: str
) -> None:
    """A valid signature on an unsupported executor is not an internal server error."""
    executor = probe_worker.generator._executor
    delattr(executor, f"{action}_context_cancellation")
    payload = (
        context_cancellation_reservation_payload(42) if action == "reserve" else {"token": "opaque"}
    )
    headers = build_context_cancellation_headers("secret", action, payload)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=probe_worker.app), base_url="http://worker"
    ) as client:
        response = await client.post(
            f"/internal/context-cancellation/{action}", json=payload, headers=headers
        )
    assert response.status_code == 409

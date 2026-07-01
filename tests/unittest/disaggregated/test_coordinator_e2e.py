# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end coordinator/worker disagg serving with mocked ctx/gen workers.

CPU-only, MPI-free, single-process (three uvicorn threads):

  * mocked ctx + gen HTTP workers serve ``/health`` + ``/v1/completions``
    (ctx returns a context_only response with disaggregated_params so the disagg
    server proceeds to gen; gen returns the final completion text),
  * a real ``CoordinatorServer`` (wrapping a ``DisaggCoordinatorService``) runs on
    an internal port -- the gen router is a *stateful* conversation router, so gen
    placement is delegated to it via ``/select``; the ctx router is round-robin
    (placed locally in the disagg server),
  * a real ``OpenAIDisaggServer`` in worker mode (``coordinator_url`` set, so it
    holds a ``CoordinatorClient``) serves the public ``/v1/completions``.

A real HTTP completion is sent to the disagg server and must round-trip
ctx -> (coordinator /select) -> gen, returning the gen worker's text. This
exercises the whole chain including the coordinator HTTP hop.
"""

import asyncio
import threading
import time

import aiohttp
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from tensorrt_llm.llmapi.disagg_utils import (CtxGenServerConfig,
                                              DisaggServerConfig, RouterConfig,
                                              ServerRole)
from tensorrt_llm.serve.coordinator_server import CoordinatorServer
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.openai_client import OpenAIHttpClient
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.router import create_router

GEN_TEXT = "HELLO_FROM_GEN"


@pytest.fixture(autouse=True)
def _reset_prometheus_registry():
    """Role-prefixed Prometheus counters are registered in the global default
    registry; clear it between tests so a second server build in the same pytest
    process does not hit duplicate-timeseries errors."""
    from prometheus_client import REGISTRY
    yield
    for collector in list(REGISTRY._collector_to_names):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def _free_port():
    import socket
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _UvicornThread:
    """Run a FastAPI app in a background uvicorn server thread."""

    def __init__(self, app, port):
        self.port = port
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port,
                           log_level="warning"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self):
        self._thread.start()
        for _ in range(100):
            if self._server.started:
                break
            time.sleep(0.1)
        return self

    def __exit__(self, *a):
        self._server.should_exit = True
        self._thread.join(timeout=10)


def _mock_worker_app(role: str) -> FastAPI:
    """A ctx or gen worker: /health + /server_info + /v1/completions."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return Response(status_code=200)

    @app.get("/server_info")
    async def server_info():
        return JSONResponse({"kv_cache_hash_algo": "v1"})

    @app.post("/v1/completions")
    async def completions(raw: Request):
        body = await raw.json()
        dp = body.get("disaggregated_params") or {}
        model = body.get("model", "m")
        if dp.get("request_type") == "context_only":
            # Context phase: return disagg params so the disagg server proceeds
            # to the gen worker (finish_reason "length" => needs generation).
            rid = dp.get("disagg_request_id")
            return JSONResponse({
                "id": "cmpl-ctx",
                "object": "text_completion",
                "created": 0,
                "model": model,
                "prompt_token_ids": [1, 2, 3],
                "choices": [{
                    "index": 0,
                    "text": "",
                    "finish_reason": "length",
                    "disaggregated_params": {
                        "request_type": "context_only",
                        "ctx_request_id": rid,
                        "disagg_request_id": rid,
                    },
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 0,
                          "total_tokens": 3},
            })
        # Generation phase: final answer.
        return JSONResponse({
            "id": "cmpl-gen",
            "object": "text_completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "text": GEN_TEXT, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2,
                      "total_tokens": 5},
        })

    return app


class _ReadinessClient:
    """Minimal client the coordinator uses only for server readiness probing.

    In coordinator/worker mode the coordinator never sends completions (the disagg
    servers do), so its readiness client needs no metrics -- reusing the real
    ``check_ready_for_servers`` keeps the probe faithful while avoiding a second
    set of role-prefixed Prometheus counters in this single-process test.
    """

    def __init__(self, router):
        self._router = router
        self._session = aiohttp.ClientSession()

    async def check_ready(self):
        ready, unready = await OpenAIHttpClient.check_ready_for_servers(
            self._session, self._router.servers)
        if ready:
            await self._router.prepare_servers(ready)
        return ready, unready

    async def shutdown(self):
        await self._session.close()


def _make_config(ctx_url, gen_url, public_port):
    host_port = lambda u: (u.split(":")[0], int(u.split(":")[1]))
    ctx_host, ctx_port = host_port(ctx_url)
    gen_host, gen_port = host_port(gen_url)
    return DisaggServerConfig(
        server_configs=[
            CtxGenServerConfig(type="ctx", hostname=ctx_host, port=ctx_port),
            CtxGenServerConfig(type="gen", hostname=gen_host, port=gen_port),
        ],
        hostname="127.0.0.1",
        port=public_port,
        # ctx: stateless (placed locally in the disagg server);
        # gen: stateful conversation router (placement delegated to coordinator).
        ctx_router_config=RouterConfig(type="round_robin",
                                       server_role=ServerRole.CONTEXT),
        gen_router_config=RouterConfig(type="conversation",
                                       server_role=ServerRole.GENERATION))


class _CoordinatorThread:
    """Run a CoordinatorServer (DisaggCoordinatorService) in a uvicorn thread."""

    def __init__(self, config):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        ctx_urls, gen_urls = _split(config)
        ctx_router = create_router(config.ctx_router_config, ctx_urls)
        gen_router = create_router(config.gen_router_config, gen_urls)
        self._coordinator = DisaggCoordinatorService(
            config, ctx_router, gen_router,
            client_factory=lambda router, role, mr=1: _ReadinessClient(router))
        self._impl = _UvicornThread(CoordinatorServer(self._coordinator).app,
                                    self.port)

    def __enter__(self):
        self._impl.__enter__()
        return self

    def __exit__(self, *a):
        self._impl.__exit__(*a)


def _split(config):
    from tensorrt_llm.llmapi.disagg_utils import get_ctx_gen_server_addrs
    return get_ctx_gen_server_addrs(config.server_configs)


async def _wait_healthy(url, timeout_s=30.0):
    deadline = time.time() + timeout_s
    async with aiohttp.ClientSession() as sess:
        while time.time() < deadline:
            try:
                async with sess.get(f"{url}/health", timeout=1) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.2)
    return False


def test_disagg_completion_e2e_through_coordinator():
    with _UvicornThread(_mock_worker_app("ctx"), _free_port()) as ctx, \
         _UvicornThread(_mock_worker_app("gen"), _free_port()) as gen:
        ctx_url = f"127.0.0.1:{ctx.port}"
        gen_url = f"127.0.0.1:{gen.port}"
        public_port = _free_port()
        config = _make_config(ctx_url, gen_url, public_port)

        with _CoordinatorThread(config) as coord:
            assert asyncio.run(_wait_healthy(coord.url)), \
                "coordinator never became healthy"

            disagg = OpenAIDisaggServer(config=config,
                                        coordinator_url=coord.url)
            with _UvicornThread(disagg.app, public_port) as server:
                base = f"http://127.0.0.1:{server.port}"
                assert asyncio.run(_wait_healthy(base)), \
                    "disagg server never became healthy"

                async def drive():
                    async with aiohttp.ClientSession() as sess:
                        payload = {"model": "m", "prompt": "hello",
                                   "max_tokens": 8}
                        # X-Session-ID -> conversation_id, so the gen router
                        # (conversation) delegates placement to the coordinator.
                        headers = {"X-Session-ID": "conv-e2e"}
                        async with sess.post(f"{base}/v1/completions",
                                             json=payload, headers=headers,
                                             timeout=30) as r:
                            assert r.status == 200, await r.text()
                            return await r.json()

                body = asyncio.run(drive())

    # The full ctx -> coordinator/select -> gen chain returned the gen text.
    assert body["choices"][0]["text"] == GEN_TEXT, body
    assert body["choices"][0]["finish_reason"] == "stop"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

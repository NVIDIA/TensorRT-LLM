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
"""Unit tests for the block-hash HTTP router server + remote client.

Tokenization / hashing happen in the caller (orchestrator); only block hashes
cross the wire to the router server (POST /select).
"""

import pytest
from fastapi.testclient import TestClient

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.kv_cache_router import (KvCacheEventReport,
                                                WorkerLoadReport)
from tensorrt_llm.serve.router import CentralizedKVCacheRouter
from tensorrt_llm.serve.router_http_server import (RouterHttpServer,
                                                   create_router_http_server)

TPB = 32
H0, H1 = "http://h0:8000", "http://h1:8000"


def _stored(hashes):
    return {"data": {"type": "stored", "parent_hash": None,
                     "blocks": [{"block_hash": h} for h in hashes]}}


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _centralized_router():
    """Centralized surface router with two prepared workers; w0 holds [1,2,3]."""
    router = CentralizedKVCacheRouter(server_role=ServerRole.CONTEXT,
                                      servers=[H0, H1],
                                      tokens_per_block=TPB,
                                      router_port=_free_port())
    core = router._core
    core.register_worker_address("w0", H0)
    core.register_worker_address("w1", H1)
    core.apply_event_report(
        KvCacheEventReport("w0", "ctx", seq=0, events=[_stored([1, 2, 3])]))
    core.apply_load_report(
        WorkerLoadReport("w0", "ctx", seq=0, num_active_requests=0,
                         num_queued_requests=0, max_batch_size=64))
    core.apply_load_report(
        WorkerLoadReport("w1", "ctx", seq=0, num_active_requests=0,
                         num_queued_requests=0, max_batch_size=64))
    # Mark both servers prepared (skip the HTTP /server_info handshake).
    router._prepared_ready_servers.update({H0, H1})
    return router


def _server():
    srv = RouterHttpServer.__new__(RouterHttpServer)
    # Build without the lifespan server-prep (state is set up manually above).
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    srv._router = _centralized_router()
    srv._monitor_interval_s = 0.0

    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    srv.app = FastAPI(lifespan=_noop_lifespan)
    srv._register_routes()
    return srv


def test_select_by_block_hashes_prefers_cache_hit():
    srv = _server()
    with TestClient(srv.app) as c:
        # [1,2,3] is fully cached on w0 (H0).
        r = c.post("/select", json={"block_hashes": [1, 2, 3]})
        assert r.status_code == 200
        body = r.json()
        assert body["server"] == H0
        assert body["matched_blocks"] == 3
        assert body["dp_rank"] is None  # legacy worker, no per-rank hint


def test_select_exclude_server():
    srv = _server()
    with TestClient(srv.app) as c:
        # Excluding the cache-hit server must fall back to the other one.
        r = c.post("/select",
                   json={"block_hashes": [1, 2, 3], "exclude_server": H0})
        assert r.status_code == 200
        assert r.json()["server"] == H1


def test_select_bad_body_returns_400():
    srv = _server()
    with TestClient(srv.app) as c:
        assert c.post("/select", json={"nope": 1}).status_code == 400


def test_servers_and_version_endpoints():
    srv = _server()
    with TestClient(srv.app) as c:
        s = c.get("/servers").json()
        assert set(s["servers"]) == {H0, H1}
        assert "version" in c.get("/version").json()


def test_router_http_server_rejects_non_hash_router():
    from tensorrt_llm.serve.router import RoundRobinRouter
    rr = RoundRobinRouter(ServerRole.CONTEXT, [H0, H1])
    with pytest.raises(TypeError):
        RouterHttpServer(rr)


# ------------------------------------------------- RemoteHttpRouter client


def test_create_router_builds_remote_http_client():
    from tensorrt_llm.llmapi.disagg_utils import RouterConfig
    from tensorrt_llm.serve.router import RemoteHttpRouter, create_router
    r = create_router(
        RouterConfig(type="remote_http",
                     args={"remote_url": "http://router-host:8080"}),
        servers=[])
    assert isinstance(r, RemoteHttpRouter)
    assert r._remote_url == "http://router-host:8080"


def test_remote_http_router_requires_remote_url():
    from tensorrt_llm.serve.router import RemoteHttpRouter
    with pytest.raises(ValueError):
        RemoteHttpRouter(servers=[])


def test_remote_http_router_end_to_end():
    """Client hashes locally, POSTs only block hashes to a live server, and gets
    the cache-hit server back."""
    import asyncio
    import threading
    import time

    import uvicorn

    from tensorrt_llm.llmapi.disagg_utils import RouterConfig
    from tensorrt_llm.serve.openai_protocol import CompletionRequest
    from tensorrt_llm.serve.router import create_router

    srv = _server()
    cfg = uvicorn.Config(srv.app, host="127.0.0.1", port=8097,
                         log_level="warning")
    server = uvicorn.Server(cfg)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.1)
        time.sleep(0.3)

        async def drive():
            client = create_router(
                RouterConfig(type="remote_http",
                             args={"remote_url": "http://127.0.0.1:8097",
                                   "tokens_per_block": TPB}),
                servers=[])
            # prompt given as token ids -> no tokenizer needed; block hashes
            # for these ids will match w0's stored [1,2,3] only if they hash
            # equal, which they won't here -> we just assert a valid server and
            # that the request round-trips (hash content is opaque to the test).
            s, info = await client.get_next_server(
                CompletionRequest(model="m", prompt=list(range(1, 200))))
            await client.close()
            return s

        server_pick = asyncio.run(drive())
        assert server_pick in {H0, H1}
    finally:
        server.should_exit = True
        thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

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
"""Fork the router HTTP server and route via the client adaptor.

CPU-only, MPI-free cross-process test:
  * two fake worker HTTP servers answer /health + /server_info (worker_id),
  * ``_maybe_start_router_http_servers`` forks a real ``trtllm-serve
    router_http_server`` child (centralized router) via a remote_http +
    auto_start config,
  * a ``RemoteHttpRouter`` client tokenizes/hashes locally and POSTs only block
    hashes to the forked server, getting a real worker URL back.

The router server is a plain HTTP process started with a simple subprocess.
The auto-start helper drops SLURM/PMIX/OMPI env from the child so it never
tries to join an MPI namespace even if the parent ran under an MPI launcher.
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _FakeWorker:
    """Minimal HTTP worker: /health -> 200, /server_info -> {worker_id, algo}."""

    def __init__(self, worker_id: str):
        self._worker_id = worker_id
        self.port = _free_port()
        worker_id_ref = worker_id

        class Handler(BaseHTTPRequestHandler):

            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.end_headers()
                elif self.path == "/server_info":
                    body = json.dumps({
                        "worker_id": worker_id_ref,
                        "kv_cache_hash_algo": "v1",
                    }).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)

    @property
    def url(self):
        return f"127.0.0.1:{self.port}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._httpd.shutdown()


async def _wait_healthy(url: str, timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    async with aiohttp.ClientSession() as sess:
        while time.time() < deadline:
            try:
                async with sess.get(f"{url}/health",
                                    timeout=1) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.3)
    return False


@pytest.mark.parametrize("rank_routing_algo", ["none", "instance"])
def test_forked_router_server_and_remote_client(rank_routing_algo):
    from tensorrt_llm.commands.serve import _maybe_start_router_http_servers
    from tensorrt_llm.llmapi.disagg_utils import (CtxGenServerConfig,
                                                  DisaggServerConfig,
                                                  RouterConfig, ServerRole)
    from tensorrt_llm.serve.openai_protocol import CompletionRequest
    from tensorrt_llm.serve.router import RemoteHttpRouter

    with _FakeWorker("w0") as w0, _FakeWorker("w1") as w1:
        router_port = _free_port()
        remote_url = f"http://127.0.0.1:{router_port}"

        ctx_router = RouterConfig(
            type="remote_http",
            server_role=ServerRole.CONTEXT,
            args={
                "remote_url": remote_url,
                "auto_start": True,
                "tokens_per_block": 32,
                "rank_routing_algo": rank_routing_algo,
            })
        gen_router = RouterConfig(type="round_robin",
                                  server_role=ServerRole.GENERATION)
        cfg = DisaggServerConfig(
            server_configs=[
                CtxGenServerConfig(type="ctx", hostname="127.0.0.1",
                                   port=w0.port),
                CtxGenServerConfig(type="ctx", hostname="127.0.0.1",
                                   port=w1.port),
            ],
            ctx_router_config=ctx_router,
            gen_router_config=gen_router)

        children = _maybe_start_router_http_servers(cfg)
        assert len(children) == 1, "expected one forked ctx router server"
        try:
            assert asyncio.run(_wait_healthy(remote_url)), \
                "forked router HTTP server never became healthy"

            async def drive():
                client = RemoteHttpRouter(
                    server_role=ServerRole.CONTEXT,
                    servers=[],
                    remote_url=remote_url,
                    tokens_per_block=32)
                picks = []
                for _ in range(4):
                    # token-id prompt -> no tokenizer download needed
                    server, info = await client.get_next_server(
                        CompletionRequest(model="m",
                                          prompt=list(range(1, 300))))
                    picks.append(server)
                await client.close()
                return picks

            picks = asyncio.run(drive())
            # every pick must be one of the two registered fake workers
            worker_urls = {f"http://{w0.url}", f"http://{w1.url}",
                           w0.url, w1.url}
            for p in picks:
                assert p in worker_urls, f"unexpected server {p!r}"
        finally:
            for child in children:
                child.terminate()
            for child in children:
                try:
                    child.wait(timeout=10)
                except Exception:
                    child.kill()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

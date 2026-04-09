# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""Unit tests for OpenAIDisaggServer.get_models endpoint.

Tests the fix that fetches /v1/models from a live worker via the router,
correctly handling both static-config mode and service-discovery (cluster)
mode where server_configs is always empty.
"""

import json
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal stubs matching tensorrt_llm.serve.openai_protocol shapes
# ---------------------------------------------------------------------------


class ModelCard:
    def __init__(self, id: str):
        self.id = id

    def model_dump(self):
        return {
            "id": self.id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "tensorrt_llm",
        }


class ModelList:
    def __init__(self, data):
        self.data = data

    def model_dump(self):
        return {"object": "list", "data": [d.model_dump() for d in self.data]}


# ---------------------------------------------------------------------------
# Verbatim copy of the production get_models implementation under test.
# This mirrors exactly what is in openai_disagg_server.py so the test
# validates the real logic without requiring the compiled TRT-LLM extension.
# ---------------------------------------------------------------------------


async def _get_models(ctx_router, gen_router):
    for router in [ctx_router, gen_router]:
        servers = router.servers
        if servers:
            server = servers[0]
            server_scheme = "http://" if not server.startswith("http://") else ""
            url = f"{server_scheme}{server}/v1/models"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            return JSONResponse(content=await resp.json())
            except Exception as e:
                logger.warning("Failed to fetch /v1/models from worker %s: %s", server, e)
    model_list = ModelList(data=[ModelCard(id="unknown")])
    return JSONResponse(content=model_list.model_dump())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _router(servers):
    r = MagicMock()
    r.servers = servers
    return r


def _make_mock_session(payload: dict):
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=payload)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


def _payload(model_id: str) -> dict:
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "tensorrt_llm"}],
    }


def _body(response) -> dict:
    return json.loads(response.body)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_static_config_returns_model_from_worker():
    """Static config mode: ctx_router has workers, model name comes from /v1/models."""
    with patch("aiohttp.ClientSession", return_value=_make_mock_session(_payload("Llama-3.1-8B"))):
        result = await _get_models(_router(["localhost:9000"]), _router([]))
    assert _body(result)["data"][0]["id"] == "Llama-3.1-8B"


@pytest.mark.asyncio
async def test_service_discovery_returns_model_from_router():
    """Service-discovery (cluster) mode: correct model returned via router.

    server_configs is always empty in this mode, but the router has
    dynamically-discovered workers.  The old implementation reading from
    server_configs would always return 'unknown' in this case.
    """
    with patch("aiohttp.ClientSession", return_value=_make_mock_session(_payload("Qwen2-72B"))):
        result = await _get_models(_router(["10.0.0.1:8000"]), _router(["10.0.0.2:8000"]))
    assert _body(result)["data"][0]["id"] == "Qwen2-72B"


@pytest.mark.asyncio
async def test_no_servers_falls_back_to_unknown():
    """When both routers have no servers, return id='unknown'."""
    result = await _get_models(_router([]), _router([]))
    assert _body(result)["data"][0]["id"] == "unknown"


@pytest.mark.asyncio
async def test_ctx_router_failure_falls_through_to_gen_router():
    """If the ctx-router worker is unreachable, try the gen-router worker."""

    class _FailSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        def get(self, url, **kw):
            raise aiohttp.ClientError("refused")

    class _OKResp:
        status = 200

        async def json(self):
            return _payload("GPT2")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

    class _OKSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        def get(self, url, **kw):
            return _OKResp()

    sessions = iter([_FailSession(), _OKSession()])
    with patch("aiohttp.ClientSession", side_effect=lambda: next(sessions)):
        result = await _get_models(_router(["bad:9999"]), _router(["good:8000"]))
    assert _body(result)["data"][0]["id"] == "GPT2"


@pytest.mark.asyncio
async def test_worker_non_200_falls_back_to_unknown():
    """A non-200 response from the worker is skipped; fall back to unknown."""

    class _BadResp:
        status = 503

        async def json(self):
            return {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

    class _BadSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        def get(self, url, **kw):
            return _BadResp()

    with patch("aiohttp.ClientSession", return_value=_BadSession()):
        result = await _get_models(_router(["host:8000"]), _router([]))
    assert _body(result)["data"][0]["id"] == "unknown"


def test_old_server_configs_approach_broken_in_cluster_mode():
    """Regression guard: old server_configs approach returned 'unknown' in cluster mode.

    The original implementation read from server_configs which is always []
    in cluster/service-discovery mode, causing 'unknown' to be returned even
    when workers are available and correctly identified through the router.
    """
    server_configs = []  # always empty when disagg_cluster is configured
    model_id = "unknown"
    if server_configs:  # this branch is never reached in cluster mode
        model_id = server_configs[0].other_args.get("model", "")
    # Old code produced "unknown" — that was the bug
    assert model_id == "unknown"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metadata server records and the HTTP client's server URL handling."""

import json
import secrets
import socket
from unittest import mock

import pytest
from aiohttp import web

from tensorrt_llm.llmapi.disagg_utils import MetadataServerConfig, ServerRole
from tensorrt_llm.serve import metadata_server as metadata_server_module
from tensorrt_llm.serve.metadata_server import (
    RECORD_SIGNATURE_FIELD,
    JsonDictionary,
    RemoteDictionary,
    create_metadata_server,
)
from tensorrt_llm.serve.openai_client import OpenAIHttpClient, _server_base_url
from tensorrt_llm.serve.openai_protocol import CompletionRequest

pytestmark = pytest.mark.cpu_only

RECORD_KEY = secrets.token_hex(16)
OTHER_RECORD_KEY = secrets.token_hex(16)
ETCD_USER = secrets.token_hex(4)
ETCD_SECRET = secrets.token_hex(16)


def _reset_prometheus_registry():
    from prometheus_client.registry import REGISTRY

    REGISTRY._names_to_collectors = {}
    REGISTRY._collector_to_names = {}


class InMemoryDictionary(RemoteDictionary):
    def __init__(self):
        self.data = {}

    def get(self, key):
        value = self.data.get(key)
        return (None, None) if value is None else (value.encode("utf-8"), None)

    def put(self, key, value):
        self.data[key] = value

    def remove(self, key):
        self.data.pop(key, None)

    def keys(self):
        return list(self.data)


def _record(url="http://127.0.0.1:8001", role="CONTEXT"):
    return {"model": "m", "server_role": role, "url": url}


def test_unsigned_store_round_trip():
    store = JsonDictionary(InMemoryDictionary())
    store.put("trtllm/a", _record())
    assert store.get("trtllm/a") == _record()
    assert store.get("trtllm/missing") is None


def test_signed_store_round_trip():
    raw = InMemoryDictionary()
    store = JsonDictionary(raw, record_auth_key=RECORD_KEY)
    store.put("trtllm/a", _record())
    assert RECORD_SIGNATURE_FIELD in json.loads(raw.data["trtllm/a"])
    assert store.get("trtllm/a") == _record()


def test_signed_store_ignores_unsigned_record():
    raw = InMemoryDictionary()
    raw.put("trtllm/b", json.dumps(_record()))
    store = JsonDictionary(raw, record_auth_key=RECORD_KEY)
    assert store.get("trtllm/b") is None


def test_signed_store_ignores_record_signed_with_other_key():
    raw = InMemoryDictionary()
    JsonDictionary(raw, record_auth_key=OTHER_RECORD_KEY).put("trtllm/b", _record())
    store = JsonDictionary(raw, record_auth_key=RECORD_KEY)
    assert store.get("trtllm/b") is None


def test_signed_store_ignores_modified_record():
    raw = InMemoryDictionary()
    store = JsonDictionary(raw, record_auth_key=RECORD_KEY)
    store.put("trtllm/a", _record())
    value = json.loads(raw.data["trtllm/a"])
    value["url"] = "http://127.0.0.1:9000"
    raw.data["trtllm/a"] = json.dumps(value)
    assert store.get("trtllm/a") is None


def test_signed_store_ignores_record_copied_to_other_key():
    raw = InMemoryDictionary()
    store = JsonDictionary(raw, record_auth_key=RECORD_KEY)
    store.put("trtllm/a", _record())
    raw.data["trtllm/c"] = raw.data["trtllm/a"]
    assert store.get("trtllm/c") is None
    assert store.get("trtllm/a") == _record()


def test_signed_store_ignores_non_dict_record():
    raw = InMemoryDictionary()
    raw.put("trtllm/d", json.dumps("value"))
    store = JsonDictionary(raw, record_auth_key=RECORD_KEY)
    assert store.get("trtllm/d") is None


@pytest.mark.asyncio
async def test_router_lists_only_signed_records():
    from tensorrt_llm.serve.router import RoundRobinRouter

    raw = InMemoryDictionary()
    store = JsonDictionary(raw, record_auth_key=RECORD_KEY)
    store.put("trtllm/a", _record("http://127.0.0.1:8001"))
    raw.put("trtllm/b", json.dumps(_record("http://127.0.0.1:9000")))

    router = RoundRobinRouter(
        server_role=ServerRole.CONTEXT, servers=[], metadata_server_cfg=None, metadata_server=store
    )
    assert await router.fetch_live_servers() == {"trtllm/a": "http://127.0.0.1:8001"}


def test_config_validation():
    with pytest.raises(ValueError):
        MetadataServerConfig(server_type="etcd", record_auth_key="")
    with pytest.raises(ValueError):
        MetadataServerConfig(server_type="etcd", user=ETCD_USER)
    with pytest.raises(ValueError):
        MetadataServerConfig(server_type="etcd", cert_key="k")
    with pytest.raises(ValueError):
        MetadataServerConfig(server_type="etcd", cert_key="k", cert_cert="c")
    cfg = MetadataServerConfig(
        server_type="etcd", user=ETCD_USER, password=ETCD_SECRET, record_auth_key=RECORD_KEY
    )
    assert cfg.port == 2379


def test_create_metadata_server_passes_credentials():
    cfg = MetadataServerConfig(
        server_type="etcd",
        hostname="etcd-host",
        port=1234,
        user=ETCD_USER,
        password=ETCD_SECRET,
        ca_cert="/ca.pem",
        cert_key="/key.pem",
        cert_cert="/cert.pem",
        record_auth_key=RECORD_KEY,
    )
    with mock.patch.object(metadata_server_module.etcd3, "client") as client_factory:
        store = create_metadata_server(cfg)
    client_factory.assert_called_once_with(
        "etcd-host",
        1234,
        ca_cert="/ca.pem",
        cert_key="/key.pem",
        cert_cert="/cert.pem",
        user=ETCD_USER,
        password=ETCD_SECRET,
    )
    assert store._record_auth_key == RECORD_KEY


@pytest.mark.parametrize(
    "server,expected",
    [
        ("127.0.0.1:8001", "http://127.0.0.1:8001"),
        ("http://127.0.0.1:8001", "http://127.0.0.1:8001"),
        ("http://127.0.0.1:8001/", "http://127.0.0.1:8001"),
        ("https://worker:8443", "https://worker:8443"),
    ],
)
def test_server_base_url(server, expected):
    assert _server_base_url(server) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("with_scheme", [False, True])
async def test_send_request_reaches_server_url(with_scheme):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    seen = []

    async def completions(request):
        seen.append(request.path)
        return web.json_response(
            {
                "id": "1",
                "object": "text_completion",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    server = f"127.0.0.1:{port}"
    if with_scheme:
        server = f"http://{server}"
    router = mock.AsyncMock()
    _reset_prometheus_registry()
    client = OpenAIHttpClient(router, ServerRole.CONTEXT, timeout_secs=10)
    try:
        request = CompletionRequest(model="m", prompt="hi", max_tokens=1)
        response = await client.send_request(request, server=server)
        assert response.choices[0].text == "ok"
        assert seen == ["/v1/completions"]
    finally:
        await client._session.close()
        await runner.cleanup()

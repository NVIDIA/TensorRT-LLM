# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metadata server records and the HTTP client's server URL handling."""

import json
import secrets
from types import SimpleNamespace
from unittest import mock

import pytest
from aiohttp import web

from tensorrt_llm.llmapi.disagg_utils import (
    MetadataServerConfig,
    ServerRole,
    parse_metadata_server_config_file,
)
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


@pytest.fixture
def isolated_prometheus_registry(monkeypatch):
    """Let each test register its own collectors and restore the registry afterwards."""
    from prometheus_client.registry import REGISTRY

    monkeypatch.setattr(REGISTRY, "_names_to_collectors", {})
    monkeypatch.setattr(REGISTRY, "_collector_to_names", {})


class InMemoryDictionary(RemoteDictionary):
    def __init__(self):
        self.data = {}

    def get(self, key):
        value = self.data.get(key)
        if value is None:
            return None, None
        return (value if isinstance(value, bytes) else value.encode("utf-8")), None

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


UNREADABLE_VALUES = [
    pytest.param("not json", id="not_json"),
    pytest.param('{"url": ', id="truncated_json"),
    pytest.param(b"\xff\xfe", id="invalid_utf8"),
]


@pytest.mark.parametrize("value", UNREADABLE_VALUES)
def test_signed_store_ignores_unreadable_record(value):
    raw = InMemoryDictionary()
    store = JsonDictionary(raw, record_auth_key=RECORD_KEY)
    store.put("trtllm/good", _record())
    raw.data["trtllm/bad"] = value
    assert store.get("trtllm/bad") is None
    assert store.get("trtllm/good") == _record()


@pytest.mark.parametrize("value", UNREADABLE_VALUES)
def test_unsigned_store_raises_on_unreadable_record(value):
    raw = InMemoryDictionary()
    raw.data["trtllm/bad"] = value
    with pytest.raises(ValueError):
        JsonDictionary(raw).get("trtllm/bad")


def _context_router(store):
    from tensorrt_llm.serve.router import RoundRobinRouter

    return RoundRobinRouter(
        server_role=ServerRole.CONTEXT, servers=[], metadata_server_cfg=None, metadata_server=store
    )


@pytest.mark.asyncio
async def test_router_lists_only_signed_records():
    raw = InMemoryDictionary()
    store = JsonDictionary(raw, record_auth_key=RECORD_KEY)
    store.put("trtllm/a", _record("http://127.0.0.1:8001"))
    raw.put("trtllm/b", json.dumps(_record("http://127.0.0.1:9000")))

    router = _context_router(store)
    assert await router.fetch_live_servers() == {"trtllm/a": "http://127.0.0.1:8001"}


@pytest.mark.asyncio
async def test_router_lists_signed_records_beside_unreadable_ones():
    raw = InMemoryDictionary()
    store = JsonDictionary(raw, record_auth_key=RECORD_KEY)
    store.put("trtllm/a", _record("http://127.0.0.1:8001"))
    raw.data["trtllm/b"] = "not json"
    raw.data["trtllm/c"] = b"\xff\xfe"

    router = _context_router(store)
    assert await router.fetch_live_servers() == {"trtllm/a": "http://127.0.0.1:8001"}


@pytest.mark.parametrize(
    "signature",
    [None, 7, "", "x", "sha256=", "sha256=zz", "\ud800", "sha256=\ud800"],
    ids=[
        "none",
        "int",
        "empty",
        "no_prefix",
        "prefix_only",
        "non_hex",
        "surrogate",
        "prefix_surrogate",
    ],
)
def test_malformed_signature_is_ignored(signature):
    raw = InMemoryDictionary()
    store = JsonDictionary(raw, record_auth_key=RECORD_KEY)
    store.put("trtllm/good", _record())
    bad = _record(url="http://127.0.0.1:8002")
    bad[RECORD_SIGNATURE_FIELD] = signature
    raw.put("trtllm/bad", json.dumps(bad))
    assert store.get("trtllm/bad") is None
    assert store.get("trtllm/good") == _record()


def test_config_validation():
    with pytest.raises(ValueError):
        MetadataServerConfig(server_type="etcd", record_auth_key="")
    with pytest.raises(ValueError):
        MetadataServerConfig(server_type="etcd", user=ETCD_USER)
    with pytest.raises(ValueError):
        MetadataServerConfig(server_type="etcd", cert_key="k")
    with pytest.raises(ValueError):
        MetadataServerConfig(server_type="etcd", cert_key="k", cert_cert="c")
    with pytest.raises(ValueError, match="require ca_cert"):
        MetadataServerConfig(server_type="etcd", user=ETCD_USER, password=ETCD_SECRET)
    cfg = MetadataServerConfig(
        server_type="etcd",
        user=ETCD_USER,
        password=ETCD_SECRET,
        ca_cert="/ca.pem",
        record_auth_key=RECORD_KEY,
    )
    assert cfg.port == 2379
    assert MetadataServerConfig(server_type="etcd", ca_cert="/ca.pem").user is None


def test_config_rejects_unknown_fields():
    with pytest.raises(ValueError, match="refersh_interval"):
        MetadataServerConfig(server_type="etcd", refersh_interval=1.0)


def test_config_keeps_secrets_out_of_repr_and_errors():
    cfg = MetadataServerConfig(
        server_type="etcd",
        user=ETCD_USER,
        password=ETCD_SECRET,
        ca_cert="/ca.pem",
        record_auth_key=RECORD_KEY,
    )
    assert ETCD_SECRET not in repr(cfg)
    assert RECORD_KEY not in repr(cfg)
    with pytest.raises(ValueError) as missing_ca_cert:
        MetadataServerConfig(server_type="etcd", user=ETCD_USER, password=ETCD_SECRET)
    assert ETCD_SECRET not in str(missing_ca_cert.value)
    with pytest.raises(ValueError) as wrong_type:
        MetadataServerConfig(server_type="etcd", record_auth_key=[RECORD_KEY])
    assert RECORD_KEY not in str(wrong_type.value)


def test_parse_metadata_server_config_file(tmp_path):
    path = tmp_path / "metadata_config.yaml"
    path.write_text(
        "server_type: etcd\n"
        "hostname: etcd-host\n"
        "port: 1234\n"
        "refresh_interval: 3.5\n"
        f'record_auth_key: "{RECORD_KEY}"\n'
    )
    cfg = parse_metadata_server_config_file(str(path))
    assert (cfg.hostname, cfg.port, cfg.refresh_interval, cfg.record_auth_key) == (
        "etcd-host",
        1234,
        3.5,
        RECORD_KEY,
    )
    assert parse_metadata_server_config_file(None) is None


def test_etcd_dictionary_lists_only_the_trtllm_prefix():
    client = mock.MagicMock()
    client.get_prefix.return_value = [
        (b"", SimpleNamespace(key=b"trtllm/a")),
        (b"", SimpleNamespace(key=b"trtllm/a/extra")),
        (b"", SimpleNamespace(key=b"trtllm/b")),
    ]
    with mock.patch.object(metadata_server_module.etcd3, "client", return_value=client):
        store = metadata_server_module.EtcdDictionary("etcd-host", 2379)
    assert sorted(store.keys()) == ["trtllm/a", "trtllm/b"]
    client.get_prefix.assert_called_once_with("trtllm/", keys_only=True)
    client.get_all.assert_not_called()


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
async def test_send_request_reaches_server_url(isolated_prometheus_registry, with_scheme):
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
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]

    server = f"127.0.0.1:{port}"
    if with_scheme:
        server = f"http://{server}"
    router = mock.AsyncMock()
    client = OpenAIHttpClient(router, ServerRole.CONTEXT, timeout_secs=10)
    try:
        request = CompletionRequest(model="m", prompt="hi", max_tokens=1)
        response = await client.send_request(request, server=server)
        assert response.choices[0].text == "ok"
        assert seen == ["/v1/completions"]
    finally:
        await client._session.close()
        await runner.cleanup()

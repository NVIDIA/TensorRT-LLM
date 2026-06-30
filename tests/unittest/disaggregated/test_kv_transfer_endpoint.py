# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Unit tests for the /v1/kv_transfer endpoint request handling.

Exercises the HTTP-layer logic of ``OpenAIServer.kv_transfer`` /
``kv_transfer_stats`` -- validation, dedup, the in-flight cap, the source
allowlist, and the stats counters -- with the background transfer
(``_do_kv_transfer``) stubbed out, so no engine or GPU is required.
"""

import asyncio
import json
import types

import pytest

from tensorrt_llm.serve.openai_server import OpenAIServer


class _FakeRequest:
    """Minimal stand-in for starlette Request; .json() returns the body."""

    def __init__(self, body, bad=False):
        self._body = body
        self._bad = bad

    async def json(self):
        if self._bad:
            raise ValueError("bad json")
        return self._body


async def _noop_transfer(*args, **kwargs):
    return None


def _make_server(max_inflight=4, dedup_ttl=300.0, allow=frozenset({"http://a:8001"}), inflight=0):
    """A bare object carrying just the state kv_transfer touches.

    `allow` defaults to a single-source allowlist so the happy-path tests pass the
    fail-closed gate; pass allow=None to exercise the disabled-by-default behavior.
    """
    s = types.SimpleNamespace()
    s.model = "test-model"
    s.port = 8000
    s._kvt_stats = {
        "fired": 0,
        "transferred": 0,
        "failed": 0,
        "deduped": 0,
        "skipped_busy": 0,
    }
    s._kvt_seen = {}
    s._kvt_inflight = inflight
    s._kvt_max = max_inflight
    s._kvt_dedup_ttl = dedup_ttl
    s._kvt_allow = allow
    # Never run the real two-step transfer in a unit test.
    s._do_kv_transfer = _noop_transfer
    return s


def _call(server, body=None, bad=False):
    req = _FakeRequest(body, bad=bad)
    resp = asyncio.run(OpenAIServer.kv_transfer(server, req))
    return resp.status_code, json.loads(resp.body)


def _stats(server):
    resp = asyncio.run(OpenAIServer.kv_transfer_stats(server, _FakeRequest({})))
    return json.loads(resp.body)


def test_bad_json_returns_400():
    s = _make_server()
    code, body = _call(s, bad=True)
    assert code == 400
    assert "error" in body


@pytest.mark.parametrize(
    "body",
    [
        {"messages": [{"role": "user", "content": "hi"}]},  # no source
        {"source": "http://a:8001"},  # no messages/prompt
        {},  # neither
    ],
)
def test_missing_required_fields_returns_400(body):
    s = _make_server()
    code, _ = _call(s, body)
    assert code == 400


def test_valid_request_is_accepted_and_counted():
    s = _make_server()
    code, body = _call(s, {"source": "http://a:8001", "prompt": "hello"})
    assert code == 202
    assert body["status"] == "accepted"
    assert "request_hash" in body
    assert s._kvt_stats["fired"] == 1


def test_duplicate_within_ttl_is_deduped():
    s = _make_server()
    req = {"source": "http://a:8001", "prompt": "same"}
    code1, _ = _call(s, req)
    code2, body2 = _call(s, req)
    assert code1 == 202 and code2 == 202
    assert body2["status"] == "deduped"
    assert s._kvt_stats["deduped"] == 1
    assert s._kvt_stats["fired"] == 1  # only the first one fired


def test_inflight_cap_sheds_as_skipped_busy():
    s = _make_server(max_inflight=2, inflight=2)
    code, body = _call(s, {"source": "http://a:8001", "prompt": "x"})
    assert code == 202
    assert body["status"] == "skipped_busy"
    assert s._kvt_stats["skipped_busy"] == 1
    assert s._kvt_stats["fired"] == 0


def test_source_allowlist_blocks_and_allows():
    blocked = _make_server(allow={"http://allowed:8001"})
    code, _ = _call(blocked, {"source": "http://evil:8001", "prompt": "x"})
    assert code == 403

    ok = _make_server(allow={"http://allowed:8001"})
    code, body = _call(ok, {"source": "http://allowed:8001", "prompt": "x"})
    assert code == 202 and body["status"] == "accepted"


def test_no_allowlist_is_disabled_fail_closed():
    # With no allowlist configured the endpoint fails closed (SSRF guard).
    s = _make_server(allow=None)
    code, _ = _call(s, {"source": "http://a:8001", "prompt": "x"})
    assert code == 403


def test_stats_endpoint_reports_counters():
    s = _make_server(max_inflight=7)
    _call(s, {"source": "http://a:8001", "prompt": "p"})
    st = _stats(s)
    for k in (
        "fired",
        "transferred",
        "failed",
        "deduped",
        "skipped_busy",
        "inflight",
        "seen",
        "max_inflight",
    ):
        assert k in st
    assert st["fired"] == 1
    assert st["max_inflight"] == 7
    assert st["seen"] == 1

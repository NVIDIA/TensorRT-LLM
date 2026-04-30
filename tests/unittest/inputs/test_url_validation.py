"""Regression tests for SSRF/DoS protections in tensorrt_llm.inputs.utils.

Covers NVBugs 5911304: user-supplied multimodal URLs were fetched without
validation, allowing SSRF to private/loopback/IMDS addresses, unbounded
response sizes, and unrestricted redirects.
"""
import asyncio
import socket
from unittest.mock import patch

import pytest

from tensorrt_llm.inputs.utils import (_MAX_RESPONSE_BYTES, _safe_aiohttp_get,
                                       _validate_public_url)


def _dns(ip: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]


PUBLIC_DNS = _dns("93.184.216.34")  # example.com


class TestValidatePublicUrl:

    def test_rejects_non_http_scheme(self):
        with pytest.raises(RuntimeError, match="Only http/https"):
            _validate_public_url("file:///etc/passwd")

    def test_rejects_missing_hostname(self):
        with pytest.raises(RuntimeError, match="no hostname"):
            _validate_public_url("http:///path")

    @pytest.mark.parametrize("ip", [
        "127.0.0.1",
        "::1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # AWS/Azure/GCP IMDS
    ])
    def test_rejects_non_public_addresses(self, ip):
        with patch("tensorrt_llm.inputs.utils.socket.getaddrinfo",
                   return_value=_dns(ip)):
            with pytest.raises(RuntimeError, match="non-public"):
                _validate_public_url("http://target.example/")

    def test_rejects_unresolvable_hostname(self):
        with patch("tensorrt_llm.inputs.utils.socket.getaddrinfo",
                   side_effect=socket.gaierror("nope")):
            with pytest.raises(RuntimeError, match="Could not resolve"):
                _validate_public_url("http://this.does.not.exist.invalid/")

    def test_accepts_public_address(self):
        with patch("tensorrt_llm.inputs.utils.socket.getaddrinfo",
                   return_value=PUBLIC_DNS):
            _validate_public_url("https://example.com/image.jpg")  # no raise


class _FakeContent:

    def __init__(self, data: bytes, chunk_size: int = 1 << 20):
        self._data = data
        self._chunk = chunk_size

    async def iter_chunked(self, size):
        for i in range(0, len(self._data), self._chunk):
            yield self._data[i:i + self._chunk]


class _FakeResponse:

    def __init__(self, status=200, headers=None, body=b""):
        self.status = status
        self.headers = headers or {}
        self.content = _FakeContent(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"http {self.status}")


class _FakeSession:

    def __init__(self, responses):
        self._responses = list(responses)

    def get(self, url, **kwargs):
        return self._responses.pop(0)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestSafeAiohttpGet:

    def test_validates_before_request(self):
        session = _FakeSession([])  # no responses needed; should fail validation
        with patch("tensorrt_llm.inputs.utils.socket.getaddrinfo",
                   return_value=_dns("169.254.169.254")):
            with pytest.raises(RuntimeError, match="non-public"):
                _run(_safe_aiohttp_get("http://target/", session))

    def test_rejects_redirect_to_private(self):
        responses = [
            _FakeResponse(status=302,
                          headers={"Location": "http://internal.corp/secret"}),
        ]
        session = _FakeSession(responses)

        def dns(host, *a, **kw):
            return _dns("10.0.0.1") if "internal" in host else PUBLIC_DNS

        with patch("tensorrt_llm.inputs.utils.socket.getaddrinfo",
                   side_effect=dns):
            with pytest.raises(RuntimeError, match="non-public"):
                _run(_safe_aiohttp_get("http://example.com/", session))

    def test_rejects_oversized_response(self):
        big = b"x" * (_MAX_RESPONSE_BYTES + 1)
        responses = [_FakeResponse(status=200, body=big)]
        session = _FakeSession(responses)
        with patch("tensorrt_llm.inputs.utils.socket.getaddrinfo",
                   return_value=PUBLIC_DNS):
            with pytest.raises(RuntimeError, match="maximum allowed size"):
                _run(_safe_aiohttp_get("http://example.com/", session))

    def test_rejects_too_many_redirects(self):
        # 7 redirects -> exceeds _MAX_REDIRECTS (5)
        responses = [
            _FakeResponse(status=302,
                          headers={"Location": "http://example.com/next"})
            for _ in range(7)
        ]
        session = _FakeSession(responses)
        with patch("tensorrt_llm.inputs.utils.socket.getaddrinfo",
                   return_value=PUBLIC_DNS):
            with pytest.raises(RuntimeError, match="Too many redirects"):
                _run(_safe_aiohttp_get("http://example.com/", session))

    def test_returns_body_on_success(self):
        responses = [_FakeResponse(status=200, body=b"hello")]
        session = _FakeSession(responses)
        with patch("tensorrt_llm.inputs.utils.socket.getaddrinfo",
                   return_value=PUBLIC_DNS):
            assert _run(_safe_aiohttp_get("http://example.com/",
                                          session)) == b"hello"

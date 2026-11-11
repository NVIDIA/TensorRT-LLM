# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for SSRF-prevention URL validation helpers in inputs/media_io.py.

Tests cover _validate_url(), _safe_request_get(), and _safe_aiohttp_get()
without making real network connections.
"""

import asyncio
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from tensorrt_llm.inputs import media_io as media_io_module
from tensorrt_llm.inputs.media_io import (
    _MAX_REDIRECTS,
    MediaFetchError,
    _safe_aiohttp_get,
    _safe_request_get,
    _validate_url,
    _ValidatingResolver,
)

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dns(ip: str):
    """Return a minimal getaddrinfo result that resolves to *ip*."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]


PUBLIC_DNS = _dns("93.184.216.34")  # example.com


def _requests_response(status_code=200, headers=None, chunks=(b"body",)):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.iter_content.side_effect = lambda chunk_size: iter(chunks)
    return resp


class _AsyncContent:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def read(self, _):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class _AiohttpErrorResponse:
    headers = {}

    def __init__(self, status):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    def raise_for_status(self):
        raise aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=self.status,
            message="test status",
        )


class _AiohttpSession:
    def __init__(self, response):
        self._response = response

    def get(self, url, **kwargs):
        return self._response


# ---------------------------------------------------------------------------
# _validate_url
# ---------------------------------------------------------------------------


class TestValidateUrl:
    def test_rejects_file_scheme(self):
        with pytest.raises(RuntimeError, match="Only http"):
            _validate_url("file:///etc/passwd")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(RuntimeError, match="Only http"):
            _validate_url("ftp://example.com/file")

    def test_rejects_missing_hostname(self):
        with pytest.raises(RuntimeError, match="no hostname"):
            _validate_url("http:///path")

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=_dns("127.0.0.1"))
    def test_rejects_loopback_ipv4(self, _):
        with pytest.raises(RuntimeError, match="non-public"):
            _validate_url("http://localhost/")

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=_dns("::1"))
    def test_rejects_loopback_ipv6(self, _):
        with pytest.raises(RuntimeError, match="non-public"):
            _validate_url("http://ip6-localhost/")

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=_dns("10.0.0.1"))
    def test_rejects_rfc1918_10(self, _):
        with pytest.raises(RuntimeError, match="non-public"):
            _validate_url("http://internal.corp/")

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=_dns("172.16.0.1"))
    def test_rejects_rfc1918_172(self, _):
        with pytest.raises(RuntimeError, match="non-public"):
            _validate_url("http://vpn.example.com/")

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=_dns("192.168.1.100"))
    def test_rejects_rfc1918_192(self, _):
        with pytest.raises(RuntimeError, match="non-public"):
            _validate_url("http://router.local/")

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=_dns("169.254.169.254"))
    def test_rejects_cloud_metadata_imds(self, _):
        """AWS / Azure / GCP instance metadata service must be blocked."""
        with pytest.raises(RuntimeError, match="non-public"):
            _validate_url("http://169.254.169.254/latest/meta-data/")

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=_dns("100.64.0.1"))
    def test_rejects_carrier_grade_nat(self, _):
        with pytest.raises(RuntimeError, match="non-public"):
            _validate_url("http://shared-address.example/")

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=_dns("224.0.0.1"))
    def test_rejects_multicast(self, _):
        with pytest.raises(RuntimeError, match="multicast"):
            _validate_url("http://multicast.example/")

    @patch(
        "tensorrt_llm.inputs.media_io.socket.getaddrinfo",
        side_effect=socket.gaierror(socket.EAI_NONAME, "Name or service not known"),
    )
    def test_rejects_unresolvable_hostname(self, _):
        with pytest.raises(RuntimeError, match="Could not resolve") as error:
            _validate_url("http://this.does.not.exist.invalid/")
        assert not isinstance(error.value, MediaFetchError)

    @patch(
        "tensorrt_llm.inputs.media_io.socket.getaddrinfo",
        side_effect=socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"),
    )
    def test_maps_temporary_resolution_failure_to_fetch_error(self, _):
        with pytest.raises(MediaFetchError, match="Could not resolve"):
            _validate_url("http://temporarily-unavailable.example/")

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    def test_accepts_public_hostname(self, _):
        _validate_url("http://example.com/image.jpg")  # must not raise

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    def test_accepts_https(self, _):
        _validate_url("https://example.com/image.jpg")  # must not raise


class TestAllowPrivateUrlsOptIn:
    """TRTLLM_MEDIA_ALLOW_PRIVATE_URLS=1 relaxes the is_global requirement.

    Default (unset / not "1") keeps the SSRF posture unchanged; the opt-in
    admits private, loopback, link-local, and CGNAT addresses but never
    multicast, non-http(s) schemes, or hostname-less URLs.
    """

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "::1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.100",
            "169.254.169.254",
            "100.64.0.1",
        ],
    )
    def test_env_set_allows_non_global_addresses(self, monkeypatch, ip):
        monkeypatch.setenv("TRTLLM_MEDIA_ALLOW_PRIVATE_URLS", "1")
        with patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=_dns(ip)):
            _validate_url("http://internal.host/media.jpg")  # must not raise

    @pytest.mark.parametrize("env_value", [None, "0", "true", "yes", ""])
    def test_default_and_non_1_values_still_reject(self, monkeypatch, env_value):
        if env_value is None:
            monkeypatch.delenv("TRTLLM_MEDIA_ALLOW_PRIVATE_URLS", raising=False)
        else:
            monkeypatch.setenv("TRTLLM_MEDIA_ALLOW_PRIVATE_URLS", env_value)
        with patch(
            "tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=_dns("10.0.0.1")
        ):
            with pytest.raises(RuntimeError, match="non-public"):
                _validate_url("http://internal.corp/media.jpg")

    def test_env_set_still_rejects_multicast(self, monkeypatch):
        monkeypatch.setenv("TRTLLM_MEDIA_ALLOW_PRIVATE_URLS", "1")
        with patch(
            "tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=_dns("224.0.0.1")
        ):
            with pytest.raises(RuntimeError, match="multicast"):
                _validate_url("http://multicast.example/")

    def test_env_set_still_rejects_non_http_scheme(self, monkeypatch):
        monkeypatch.setenv("TRTLLM_MEDIA_ALLOW_PRIVATE_URLS", "1")
        with pytest.raises(RuntimeError, match="Only http"):
            _validate_url("ftp://internal.host/file")

    def test_env_set_allows_private_redirect_target(self, monkeypatch):
        """The opt-in applies to redirect hops too (same validator)."""
        monkeypatch.setenv("TRTLLM_MEDIA_ALLOW_PRIVATE_URLS", "1")

        def dns_side_effect(host, *args, **kwargs):
            return _dns("192.168.1.1") if "internal" in str(host) else PUBLIC_DNS

        redirect = _requests_response(
            status_code=302,
            headers={"Location": "http://internal.host/media.jpg"},
            chunks=(),
        )
        final = _requests_response(chunks=(b"media",))
        with (
            patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", side_effect=dns_side_effect),
            patch("tensorrt_llm.inputs.media_io.requests.get", side_effect=[redirect, final]),
        ):
            result = _safe_request_get("http://example.com/media.jpg")
        assert result._content == b"media"

    def test_default_rejects_private_url_in_safe_request_get(self, monkeypatch):
        monkeypatch.delenv("TRTLLM_MEDIA_ALLOW_PRIVATE_URLS", raising=False)
        with (
            patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=_dns("10.0.0.1")),
            patch("tensorrt_llm.inputs.media_io.requests.get") as mock_get,
        ):
            with pytest.raises(RuntimeError, match="non-public"):
                _safe_request_get("http://internal.corp/image.jpg")
            mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# _safe_request_get
# ---------------------------------------------------------------------------


class TestSafeRequestGet:
    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    @patch("tensorrt_llm.inputs.media_io.requests.get")
    def test_returns_response_on_success(self, mock_get, _):
        resp = _requests_response(chunks=(b"fake-image",))
        mock_get.return_value = resp
        result = _safe_request_get("http://example.com/image.jpg")
        assert result.status_code == 200
        assert result._content == b"fake-image"
        assert result.raw.read() == b"fake-image"

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    @patch("tensorrt_llm.inputs.media_io.requests.get")
    def test_follows_valid_redirect(self, mock_get, _):
        redirect = _requests_response(
            status_code=302,
            headers={"Location": "http://example.com/final.jpg"},
            chunks=(),
        )
        final = _requests_response(chunks=(b"image-data",))
        mock_get.side_effect = [redirect, final]
        result = _safe_request_get("http://example.com/image.jpg")
        assert result.status_code == 200
        assert mock_get.call_count == 2
        assert result._content == b"image-data"

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    @patch("tensorrt_llm.inputs.media_io.requests.get")
    def test_follows_relative_redirect(self, mock_get, _):
        redirect = _requests_response(
            status_code=302,
            headers={"Location": "/final.jpg"},
            chunks=(),
        )
        final = _requests_response(chunks=(b"image-data",))
        mock_get.side_effect = [redirect, final]

        result = _safe_request_get("http://example.com/image.jpg")

        assert result.status_code == 200
        assert mock_get.call_args_list[1].args[0] == "http://example.com/final.jpg"

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo")
    @patch("tensorrt_llm.inputs.media_io.requests.get")
    def test_raises_on_redirect_to_private_ip(self, mock_get, mock_dns):
        def dns_side_effect(host, *args, **kwargs):
            return _dns("192.168.1.1") if "evil" in str(host) else PUBLIC_DNS

        mock_dns.side_effect = dns_side_effect

        redirect = _requests_response(
            status_code=301,
            headers={"Location": "http://evil.internal/secret"},
            chunks=(),
        )
        mock_get.return_value = redirect

        with pytest.raises(RuntimeError, match="non-public"):
            _safe_request_get("http://example.com/image.jpg")
        redirect.close.assert_called_once()

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    @patch("tensorrt_llm.inputs.media_io.requests.get")
    def test_raises_after_too_many_redirects(self, mock_get, _):
        redirect = _requests_response(
            status_code=302,
            headers={"Location": "http://example.com/image.jpg"},
            chunks=(),
        )
        mock_get.return_value = redirect  # always redirect

        with pytest.raises(RuntimeError, match="Too many redirects"):
            _safe_request_get("http://example.com/image.jpg")

        # Initial request + _MAX_REDIRECTS follow-ups before giving up
        assert mock_get.call_count == _MAX_REDIRECTS + 1

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    @patch("tensorrt_llm.inputs.media_io.requests.get")
    def test_raises_on_oversized_response(self, mock_get, _):
        resp = _requests_response(chunks=(b"xx", b"xx"))
        mock_get.return_value = resp

        with patch("tensorrt_llm.inputs.media_io._MAX_RESPONSE_BYTES", 3):
            with pytest.raises(RuntimeError, match="maximum allowed size"):
                _safe_request_get("http://example.com/huge.bin", stream=False)
        resp.close.assert_called_once()

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    @patch("tensorrt_llm.inputs.media_io.requests.get")
    def test_raises_on_oversized_stream_response(self, mock_get, _):
        resp = _requests_response(chunks=(b"xx", b"xx"))
        mock_get.return_value = resp

        with patch("tensorrt_llm.inputs.media_io._MAX_RESPONSE_BYTES", 3):
            with pytest.raises(RuntimeError, match="maximum allowed size"):
                _safe_request_get("http://example.com/huge.bin", stream=True)
        resp.close.assert_called_once()

    def test_rejects_private_url_before_request(self):
        """_validate_url() must fire before any requests.get() call."""
        with (
            patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=_dns("10.0.0.1")),
            patch("tensorrt_llm.inputs.media_io.requests.get") as mock_get,
        ):
            with pytest.raises(RuntimeError, match="non-public"):
                _safe_request_get("http://internal.corp/image.jpg")
            mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# _safe_aiohttp_get
# ---------------------------------------------------------------------------


class TestSafeAiohttpGet:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_shared_session_uses_validating_resolver(self):
        async def check():
            session = await media_io_module._get_aiohttp_session()
            try:
                assert isinstance(session.connector._resolver, _ValidatingResolver)
            finally:
                await media_io_module._close_aiohttp_session()

        self._run(check())

    def test_each_event_loop_closes_its_own_session(self):
        async def get_session():
            session = await media_io_module._get_aiohttp_session()
            await media_io_module._close_aiohttp_session()
            return session

        first = self._run(get_session())
        second = self._run(get_session())

        assert second is not first
        assert first.closed
        assert second.closed

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    def test_maps_timeout_to_fetch_error(self, _):
        class _FakeSession:
            def get(self, url, **kwargs):
                raise asyncio.TimeoutError("timed out")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

        with patch(
            "tensorrt_llm.inputs.media_io._get_aiohttp_session",
            new=AsyncMock(return_value=_FakeSession()),
        ):
            with pytest.raises(MediaFetchError, match="fetch failed"):
                self._run(_safe_aiohttp_get("http://example.com/file.bin"))

    @pytest.mark.parametrize(
        "error",
        [
            aiohttp.InvalidURL("not a URL"),
            aiohttp.ClientSSLError(MagicMock(), OSError("TLS failed")),
        ],
    )
    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    def test_preserves_permanent_client_error(self, _, error):
        class _FakeSession:
            def get(self, url, **kwargs):
                raise error

        with patch(
            "tensorrt_llm.inputs.media_io._get_aiohttp_session",
            new=AsyncMock(return_value=_FakeSession()),
        ):
            with pytest.raises(type(error)) as raised:
                self._run(_safe_aiohttp_get("http://example.com/file.bin"))
        assert raised.value is error

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    def test_preserves_non_retryable_http_error(self, _):
        with patch(
            "tensorrt_llm.inputs.media_io._get_aiohttp_session",
            new=AsyncMock(return_value=_AiohttpSession(_AiohttpErrorResponse(404))),
        ):
            with pytest.raises(aiohttp.ClientResponseError) as error:
                self._run(_safe_aiohttp_get("http://example.com/missing.jpg"))
        assert error.value.status == 404

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    def test_maps_retryable_http_error_to_fetch_error(self, _):
        with patch(
            "tensorrt_llm.inputs.media_io._get_aiohttp_session",
            new=AsyncMock(return_value=_AiohttpSession(_AiohttpErrorResponse(503))),
        ):
            with pytest.raises(MediaFetchError, match="HTTP 503"):
                self._run(_safe_aiohttp_get("http://example.com/unavailable.jpg"))

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    def test_reads_response_chunks_to_eof(self, _):
        class _FakeResponse:
            status = 200
            headers = {}
            content = _AsyncContent([b"ab", b"cd"])

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

            async def raise_for_status(self):
                pass

        class _FakeSession:
            def get(self, url, **kwargs):
                return _FakeResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

        with patch(
            "tensorrt_llm.inputs.media_io._get_aiohttp_session",
            new=AsyncMock(return_value=_FakeSession()),
        ):
            assert self._run(_safe_aiohttp_get("http://example.com/file.bin")) == b"abcd"

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    def test_raises_after_too_many_redirects(self, _):
        class _FakeResponse:
            status = 302
            headers = {"Location": "http://example.com/image.jpg"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

            async def raise_for_status(self):
                pass

        class _FakeSession:
            def get(self, url, **kwargs):
                return _FakeResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

        with patch(
            "tensorrt_llm.inputs.media_io._get_aiohttp_session",
            new=AsyncMock(return_value=_FakeSession()),
        ):
            with pytest.raises(RuntimeError, match="Too many redirects"):
                self._run(_safe_aiohttp_get("http://example.com/image.jpg"))

    @patch("tensorrt_llm.inputs.media_io.socket.getaddrinfo", return_value=PUBLIC_DNS)
    def test_raises_on_oversized_response(self, _):
        class _FakeResponse:
            status = 200
            headers = {}
            content = _AsyncContent([b"xx", b"xx"])

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

            async def raise_for_status(self):
                pass

        class _FakeSession:
            def get(self, url, **kwargs):
                return _FakeResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

        with (
            patch(
                "tensorrt_llm.inputs.media_io._get_aiohttp_session",
                new=AsyncMock(return_value=_FakeSession()),
            ),
            patch("tensorrt_llm.inputs.media_io._MAX_RESPONSE_BYTES", 3),
        ):
            with pytest.raises(RuntimeError, match="maximum allowed size"):
                self._run(_safe_aiohttp_get("http://example.com/huge.bin"))

    def test_rejects_private_url_before_request(self):
        """_validate_url() must fire before any aiohttp call."""
        with (
            patch(
                "tensorrt_llm.inputs.media_io.socket.getaddrinfo",
                return_value=_dns("169.254.169.254"),
            ),
            patch("tensorrt_llm.inputs.media_io._get_aiohttp_session") as mock_session,
        ):
            with pytest.raises(RuntimeError, match="non-public"):
                self._run(_safe_aiohttp_get("http://169.254.169.254/latest/"))
            mock_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_redirect_to_private_literal_before_second_request(self):
        class _RedirectResponse:
            status = 302
            headers = {"Location": "http://127.0.0.1/private"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

        class _FakeSession:
            def __init__(self):
                self.urls = []

            def get(self, url, **kwargs):
                self.urls.append(url)
                return _RedirectResponse()

        def resolve(host, *_args, **_kwargs):
            return _dns("127.0.0.1") if host == "127.0.0.1" else PUBLIC_DNS

        session = _FakeSession()
        with (
            patch(
                "tensorrt_llm.inputs.media_io.socket.getaddrinfo",
                side_effect=resolve,
            ),
            patch(
                "tensorrt_llm.inputs.media_io._get_aiohttp_session",
                new=AsyncMock(return_value=session),
            ),
        ):
            with pytest.raises(RuntimeError, match="non-public"):
                await _safe_aiohttp_get("http://example.com/image.jpg")

        assert session.urls == ["http://example.com/image.jpg"]

    def test_rejects_private_address_from_connection_resolver(self):
        async def resolve():
            resolver = _ValidatingResolver()
            resolver._resolver.resolve = AsyncMock(return_value=[{"host": "169.254.169.254"}])
            try:
                await resolver.resolve("example.com", 80)
            finally:
                await resolver.close()

        with pytest.raises(RuntimeError, match="non-public"):
            self._run(resolve())

    @pytest.mark.parametrize(
        ("error", "expected_type"),
        [
            (
                socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"),
                MediaFetchError,
            ),
            (
                socket.gaierror(socket.EAI_NONAME, "Name or service not known"),
                RuntimeError,
            ),
        ],
    )
    def test_connection_resolver_classifies_dns_failures(self, error, expected_type):
        async def resolve():
            resolver = _ValidatingResolver()
            resolver._resolver.resolve = AsyncMock(side_effect=error)
            try:
                await resolver.resolve("example.com", 80)
            finally:
                await resolver.close()

        with pytest.raises(expected_type, match="Could not resolve") as raised:
            self._run(resolve())
        if error.errno != socket.EAI_AGAIN:
            assert not isinstance(raised.value, MediaFetchError)

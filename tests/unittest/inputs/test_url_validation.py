"""Unit tests for SSRF-prevention URL validation helpers in inputs/utils.py.

Tests cover _validate_url(), _safe_request_get(), and _safe_aiohttp_get()
without making real network connections.
"""

import asyncio
import socket
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm.inputs.utils import (
    _MAX_REDIRECTS,
    _MAX_RESPONSE_BYTES,
    _safe_aiohttp_get,
    _safe_request_get,
    _validate_url,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dns(ip: str):
    """Return a minimal getaddrinfo result that resolves to *ip*."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]


PUBLIC_DNS = _dns("93.184.216.34")  # example.com


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

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=_dns("127.0.0.1"))
    def test_rejects_loopback_ipv4(self, _):
        with pytest.raises(RuntimeError, match="non-public"):
            _validate_url("http://localhost/")

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=_dns("::1"))
    def test_rejects_loopback_ipv6(self, _):
        with pytest.raises(RuntimeError, match="non-public"):
            _validate_url("http://ip6-localhost/")

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=_dns("10.0.0.1"))
    def test_rejects_rfc1918_10(self, _):
        with pytest.raises(RuntimeError, match="non-public"):
            _validate_url("http://internal.corp/")

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=_dns("172.16.0.1"))
    def test_rejects_rfc1918_172(self, _):
        with pytest.raises(RuntimeError, match="non-public"):
            _validate_url("http://vpn.example.com/")

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=_dns("192.168.1.100"))
    def test_rejects_rfc1918_192(self, _):
        with pytest.raises(RuntimeError, match="non-public"):
            _validate_url("http://router.local/")

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=_dns("169.254.169.254"))
    def test_rejects_cloud_metadata_imds(self, _):
        """AWS / Azure / GCP instance metadata service must be blocked."""
        with pytest.raises(RuntimeError, match="non-public"):
            _validate_url("http://169.254.169.254/latest/meta-data/")

    @patch(
        "tensorrt_llm.inputs.utils.socket.getaddrinfo",
        side_effect=socket.gaierror("Name or service not known"),
    )
    def test_rejects_unresolvable_hostname(self, _):
        with pytest.raises(RuntimeError, match="Could not resolve"):
            _validate_url("http://this.does.not.exist.invalid/")

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=PUBLIC_DNS)
    def test_accepts_public_hostname(self, _):
        _validate_url("http://example.com/image.jpg")  # must not raise

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=PUBLIC_DNS)
    def test_accepts_https(self, _):
        _validate_url("https://example.com/image.jpg")  # must not raise


# ---------------------------------------------------------------------------
# _safe_request_get
# ---------------------------------------------------------------------------


class TestSafeRequestGet:
    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=PUBLIC_DNS)
    @patch("tensorrt_llm.inputs.utils.requests.get")
    def test_returns_response_on_success(self, mock_get, _):
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"fake-image"
        mock_get.return_value = resp
        result = _safe_request_get("http://example.com/image.jpg")
        assert result.status_code == 200

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=PUBLIC_DNS)
    @patch("tensorrt_llm.inputs.utils.requests.get")
    def test_follows_valid_redirect(self, mock_get, _):
        redirect = MagicMock()
        redirect.status_code = 302
        redirect.headers = {"Location": "http://example.com/final.jpg"}
        final = MagicMock()
        final.status_code = 200
        final.content = b"image-data"
        mock_get.side_effect = [redirect, final]
        result = _safe_request_get("http://example.com/image.jpg")
        assert result.status_code == 200
        assert mock_get.call_count == 2

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo")
    @patch("tensorrt_llm.inputs.utils.requests.get")
    def test_raises_on_redirect_to_private_ip(self, mock_get, mock_dns):
        def dns_side_effect(host, *args, **kwargs):
            return _dns("192.168.1.1") if "evil" in str(host) else PUBLIC_DNS

        mock_dns.side_effect = dns_side_effect

        redirect = MagicMock()
        redirect.status_code = 301
        redirect.headers = {"Location": "http://evil.internal/secret"}
        mock_get.return_value = redirect

        with pytest.raises(RuntimeError, match="non-public"):
            _safe_request_get("http://example.com/image.jpg")

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=PUBLIC_DNS)
    @patch("tensorrt_llm.inputs.utils.requests.get")
    def test_raises_after_too_many_redirects(self, mock_get, _):
        redirect = MagicMock()
        redirect.status_code = 302
        redirect.headers = {"Location": "http://example.com/image.jpg"}
        mock_get.return_value = redirect  # always redirect

        with pytest.raises(RuntimeError, match="Too many redirects"):
            _safe_request_get("http://example.com/image.jpg")

        # Initial request + _MAX_REDIRECTS follow-ups before giving up
        assert mock_get.call_count == _MAX_REDIRECTS + 1

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=PUBLIC_DNS)
    @patch("tensorrt_llm.inputs.utils.requests.get")
    def test_raises_on_oversized_response(self, mock_get, _):
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"x" * (_MAX_RESPONSE_BYTES + 1)
        mock_get.return_value = resp

        with pytest.raises(RuntimeError, match="maximum allowed size"):
            _safe_request_get("http://example.com/huge.bin", stream=False)

    def test_rejects_private_url_before_request(self):
        """_validate_url() must fire before any requests.get() call."""
        with (
            patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=_dns("10.0.0.1")),
            patch("tensorrt_llm.inputs.utils.requests.get") as mock_get,
        ):
            with pytest.raises(RuntimeError, match="non-public"):
                _safe_request_get("http://internal.corp/image.jpg")
            mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# _safe_aiohttp_get
# ---------------------------------------------------------------------------


class TestSafeAiohttpGet:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=PUBLIC_DNS)
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

        with patch("tensorrt_llm.inputs.utils.aiohttp.ClientSession", return_value=_FakeSession()):
            with pytest.raises(RuntimeError, match="Too many redirects"):
                self._run(_safe_aiohttp_get("http://example.com/image.jpg"))

    @patch("tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=PUBLIC_DNS)
    def test_raises_on_oversized_response(self, _):
        oversized = b"x" * (_MAX_RESPONSE_BYTES + 1)

        class _FakeResponse:
            status = 200
            headers = {}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

            async def raise_for_status(self):
                pass

            class content:
                @staticmethod
                async def read(n):
                    return oversized[:n]

        class _FakeSession:
            def get(self, url, **kwargs):
                return _FakeResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

        with patch("tensorrt_llm.inputs.utils.aiohttp.ClientSession", return_value=_FakeSession()):
            with pytest.raises(RuntimeError, match="maximum allowed size"):
                self._run(_safe_aiohttp_get("http://example.com/huge.bin"))

    def test_rejects_private_url_before_request(self):
        """_validate_url() must fire before any aiohttp call."""
        with (
            patch(
                "tensorrt_llm.inputs.utils.socket.getaddrinfo", return_value=_dns("169.254.169.254")
            ),
            patch("tensorrt_llm.inputs.utils.aiohttp.ClientSession") as mock_session,
        ):
            with pytest.raises(RuntimeError, match="non-public"):
                self._run(_safe_aiohttp_get("http://169.254.169.254/latest/"))
            mock_session.assert_not_called()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for HTTP transport (_send_to_gxt) and live staging endpoint."""

import json
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm.usage import schema, usage_lib

# ---------------------------------------------------------------------------
# HTTP transport tests
# ---------------------------------------------------------------------------


class TestSendToGxt:
    def test_send_fail_silent(self):
        """_send_to_gxt never raises on network error."""
        with patch(
            "tensorrt_llm.usage.usage_lib._get_stats_server",
            return_value="http://192.0.2.1/nonexistent",
        ):
            usage_lib._send_to_gxt({"test": "data"})  # Must not raise

    def test_send_uses_json_content_type(self):
        captured_req = {}

        class MockOpener:
            def open(self, req, timeout=None):
                captured_req["headers"] = dict(req.headers)
                captured_req["data"] = req.data
                captured_req["method"] = req.method
                return MagicMock()

        with patch("urllib.request.build_opener", return_value=MockOpener()):
            usage_lib._send_to_gxt({"key": "value"})

        assert captured_req["headers"]["Content-type"] == "application/json"
        assert captured_req["headers"]["Accept"] == "application/json"
        assert captured_req["method"] == "POST"
        assert json.loads(captured_req["data"]) == {"key": "value"}

    def test_send_to_gxt_does_not_follow_redirects(self):
        """Custom opener excludes HTTPRedirectHandler (SSRF protection)."""
        captured_handlers = []

        def mock_build_opener(*handlers):
            captured_handlers.extend(handlers)
            return MagicMock()

        with patch("urllib.request.build_opener", side_effect=mock_build_opener):
            usage_lib._send_to_gxt({"test": True})

        handler_types = set(captured_handlers)
        assert urllib.request.HTTPRedirectHandler not in handler_types
        assert usage_lib._NoRedirectHandler in handler_types

    def test_no_redirect_handler_blocks_redirects(self):
        """_NoRedirectHandler rejects all redirect responses."""
        handler = usage_lib._NoRedirectHandler()
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            handler.redirect_request(
                MagicMock(full_url="http://example.com"),
                None,
                302,
                "Found",
                {},
                "http://evil.com",
            )
        assert exc_info.value.code == 302

    def test_real_opener_lacks_default_redirect_handler(self):
        """Verify the real opener built by build_opener has no HTTPRedirectHandler."""
        opener = urllib.request.build_opener(
            urllib.request.HTTPHandler,
            urllib.request.HTTPSHandler,
            usage_lib._NoRedirectHandler,
        )
        handler_names = [h.__class__.__name__ for h in opener.handlers]
        assert "HTTPRedirectHandler" not in handler_names
        assert "_NoRedirectHandler" in handler_names

    def test_send_to_gxt_catches_url_error(self, monkeypatch):
        """_send_to_gxt silently handles URLError."""
        monkeypatch.setenv("TRTLLM_USAGE_STATS_SERVER", "http://localhost:1")
        usage_lib._send_to_gxt({"test": True})  # should not raise


# ---------------------------------------------------------------------------
# HTTPS handler tests
# ---------------------------------------------------------------------------


class TestHttpsHandler:
    def test_opener_has_https_handler(self):
        """Opener includes HTTPSHandler for HTTPS endpoints."""
        captured_handlers = []

        def mock_build_opener(*handlers):
            captured_handlers.extend(handlers)
            return MagicMock()

        with patch("urllib.request.build_opener", side_effect=mock_build_opener):
            usage_lib._send_to_gxt({"test": True})

        handler_set = set(captured_handlers)
        has_http = urllib.request.HTTPHandler in handler_set or any(
            isinstance(h, urllib.request.HTTPHandler) for h in captured_handlers
        )
        has_https = urllib.request.HTTPSHandler in handler_set or any(
            isinstance(h, urllib.request.HTTPSHandler) for h in captured_handlers
        )
        assert has_http, f"HTTPHandler not found in {captured_handlers}"
        assert has_https, f"HTTPSHandler not found in {captured_handlers}"


# ---------------------------------------------------------------------------
# Malformed server URL tests
# ---------------------------------------------------------------------------


class TestMalformedServerUrl:
    """Verify _send_to_gxt handles malformed server URLs without crashing."""

    def test_empty_server_url_fail_silent(self, monkeypatch):
        """Empty TRTLLM_USAGE_STATS_SERVER doesn't crash."""
        monkeypatch.setenv("TRTLLM_USAGE_STATS_SERVER", "")
        usage_lib._send_to_gxt({"test": True})  # should not raise

    def test_non_http_scheme_fail_silent(self, monkeypatch):
        """Non-HTTP scheme doesn't crash."""
        monkeypatch.setenv("TRTLLM_USAGE_STATS_SERVER", "ftp://bad.example.com")
        usage_lib._send_to_gxt({"test": True})

    def test_garbage_url_fail_silent(self, monkeypatch):
        """Completely invalid URL doesn't crash."""
        monkeypatch.setenv("TRTLLM_USAGE_STATS_SERVER", "not-a-url")
        usage_lib._send_to_gxt({"test": True})


# ---------------------------------------------------------------------------
# Live staging endpoint tests (opt-in via --run-staging or -m staging)
# ---------------------------------------------------------------------------

_STAGING_ENDPOINT = "https://events.gfestage.nvidia.com/v1.1/events/json"


def _post_to_staging(payload: dict, timeout: float = 10.0) -> int:
    """POST payload to GXT staging and return HTTP status code."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _STAGING_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp.status


@pytest.mark.skipif(
    "not config.getoption('--run-staging', default=False)",
    reason="Live staging tests require --run-staging flag",
)
@pytest.mark.staging
class TestStagingEndpoint:
    """Live tests that send payloads to the GXT staging endpoint.

    These are opt-in only -- they require network access to
    events.gfestage.nvidia.com and are gated behind ``--run-staging``.
    """

    def test_initial_report_accepted(self):
        """GXT staging accepts a well-formed trtllm_initial_report envelope (HTTP 200)."""
        import os
        import platform as plat
        import uuid

        report = schema.TrtllmInitialReport(
            trtllmVersion="0.0.0-test",
            platform=plat.platform(),
            pythonVersion=plat.python_version(),
            cpuArchitecture=plat.machine(),
            cpuCount=os.cpu_count() or 0,
            gpuCount=0,
            gpuName="",
            gpuMemoryMB=0,
            cudaVersion="",
            architectureClassName="TestModel",
            backend="pytorch",
            tensorParallelSize=1,
            pipelineParallelSize=1,
            contextParallelSize=1,
            moeExpertParallelSize=0,
            moeTensorParallelSize=0,
            dtype="float16",
            quantizationAlgo="",
            kvCacheDtype="",
            ingressPoint="cli_serve",
            featuresJson='{"lora":false,"speculative_decoding":false,"prefix_caching":false,"cuda_graphs":false,"chunked_context":false,"data_parallel_size":1,"checkpoint_format":"HF","load_format":"AUTO"}',
            disaggRole="",
            deploymentId="",
        )
        payload = schema.build_gxt_payload(
            event=report,
            session_id=uuid.uuid4().hex,
            trtllm_version="0.0.0-test",
        )

        status = _post_to_staging(payload)
        assert status == 200, f"Expected HTTP 200, got {status}"

    def test_heartbeat_accepted(self):
        """GXT staging accepts a well-formed trtllm_heartbeat envelope (HTTP 200)."""
        import uuid

        heartbeat = schema.TrtllmHeartbeat(seq=0)
        payload = schema.build_gxt_payload(
            event=heartbeat,
            session_id=uuid.uuid4().hex,
            trtllm_version="0.0.0-test",
        )

        status = _post_to_staging(payload)
        assert status == 200, f"Expected HTTP 200, got {status}"

    def test_ingress_point_in_accepted_payload(self):
        """Staging accepts payloads containing ingressPoint without envelope rejection."""
        import os
        import platform as plat
        import uuid

        for context_value in ("cli_serve", "cli_bench", "cli_eval", "llm_class", "unknown", ""):
            report = schema.TrtllmInitialReport(
                trtllmVersion="0.0.0-test",
                platform=plat.platform(),
                pythonVersion=plat.python_version(),
                cpuArchitecture=plat.machine(),
                cpuCount=os.cpu_count() or 0,
                gpuCount=0,
                gpuName="",
                gpuMemoryMB=0,
                cudaVersion="",
                architectureClassName="TestModel",
                backend="pytorch",
                tensorParallelSize=1,
                pipelineParallelSize=1,
                contextParallelSize=1,
                moeExpertParallelSize=0,
                moeTensorParallelSize=0,
                dtype="float16",
                quantizationAlgo="",
                kvCacheDtype="",
                ingressPoint=context_value,
                featuresJson="{}",
                disaggRole="",
                deploymentId="",
            )
            payload = schema.build_gxt_payload(
                event=report,
                session_id=uuid.uuid4().hex,
                trtllm_version="0.0.0-test",
            )

            status = _post_to_staging(payload)
            assert status == 200, f"Staging rejected ingressPoint={context_value!r}: HTTP {status}"

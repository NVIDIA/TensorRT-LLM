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
"""Tests for OpenSearchDB.queryFromOpenSearchDB resilience.

All HTTP requests are mocked — no real network calls are made.

Run:
    cd jenkins/scripts && python3 -m pytest test_open_search_db.py -v
"""

import json
from unittest.mock import call, patch

import open_search_db
import pytest
from open_search_db import DEFAULT_RETRY_COUNT, OpenSearchDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class MockResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code, json_body=None, text=None):
        self.status_code = status_code
        self._json_body = json_body
        self.text = (
            text if text is not None else (json.dumps(json_body) if json_body is not None else "")
        )

    def json(self):
        return self._json_body


FAKE_URL = "https://fake-opensearch.example.com"
VALID_PROJECT = open_search_db.JOB_PROJECT_NAME  # in READ_ACCESS list
SANDBOX_PROJECT = "sandbox-test-project"

# Log tag used in structured log output
_TAG = "[OpenSearchDB]"


@pytest.fixture(autouse=True)
def _patch_env():
    """Ensure every test has a fake base URL and no real sleep."""
    with (
        patch.object(open_search_db, "OPEN_SEARCH_DB_BASE_URL", FAKE_URL),
        patch("open_search_db.time.sleep"),
    ):
        yield


# ---------------------------------------------------------------------------
# 1-5: Pre-validation (no HTTP request)
# ---------------------------------------------------------------------------
class TestPreValidation:
    @patch.object(open_search_db, "OPEN_SEARCH_DB_BASE_URL", "")
    def test_empty_base_url_returns_none(self):
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is None

    def test_project_not_in_read_list_returns_none(self):
        result = OpenSearchDB.queryFromOpenSearchDB("{}", "not-a-valid-project")
        assert result is None

    @patch("open_search_db.requests.get")
    def test_sandbox_project_skips_permission_check(self, mock_get):
        mock_get.return_value = MockResponse(200, {"hits": {"hits": []}})
        result = OpenSearchDB.queryFromOpenSearchDB("{}", SANDBOX_PROJECT)
        assert result is not None
        mock_get.assert_called_once()

    @patch("open_search_db.requests.get")
    def test_dict_input_auto_serialized(self, mock_get):
        mock_get.return_value = MockResponse(200, {"hits": {"hits": []}})
        body = {"query": {"match_all": {}}}
        OpenSearchDB.queryFromOpenSearchDB(body, VALID_PROJECT)
        sent_data = mock_get.call_args[1].get("data") or mock_get.call_args[0][0]
        # Should be a JSON string, not a dict
        assert isinstance(sent_data, str)
        assert "match_all" in sent_data

    @patch("open_search_db.requests.get")
    def test_str_input_used_directly(self, mock_get):
        mock_get.return_value = MockResponse(200, {"hits": {"hits": []}})
        raw = '{"query":{}}'
        OpenSearchDB.queryFromOpenSearchDB(raw, VALID_PROJECT)
        sent_data = mock_get.call_args[1].get("data") or mock_get.call_args[0][0]
        assert sent_data == raw


# ---------------------------------------------------------------------------
# 6-7: HTTP success
# ---------------------------------------------------------------------------
class TestHTTPSuccess:
    @patch("open_search_db.requests.get")
    def test_200_returns_response(self, mock_get):
        resp = MockResponse(200, {"hits": {"hits": [{"_id": "1"}]}})
        mock_get.return_value = resp
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is resp

    @patch("open_search_db.requests.get")
    def test_201_returns_response(self, mock_get):
        resp = MockResponse(201, {"hits": {"hits": []}})
        mock_get.return_value = resp
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is resp


# ---------------------------------------------------------------------------
# 8-10: Non-retryable HTTP errors (immediate failure)
# ---------------------------------------------------------------------------
class TestNonRetryableErrors:
    @patch("open_search_db.requests.get")
    def test_400_no_retry(self, mock_get):
        mock_get.return_value = MockResponse(400, text="Bad Request")
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is None
        assert mock_get.call_count == 1  # no retry

    @patch("open_search_db.requests.get")
    def test_403_no_retry(self, mock_get):
        mock_get.return_value = MockResponse(403, text="Forbidden")
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is None
        assert mock_get.call_count == 1

    @patch("open_search_db.requests.get")
    def test_404_no_retry(self, mock_get):
        mock_get.return_value = MockResponse(404, text="Not Found")
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is None
        assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# 11-13: Retryable HTTP errors
# ---------------------------------------------------------------------------
class TestRetryableErrors:
    @patch("open_search_db.requests.get")
    def test_500_retries_then_none(self, mock_get):
        mock_get.return_value = MockResponse(500, text="Internal Server Error")
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is None
        assert mock_get.call_count == DEFAULT_RETRY_COUNT

    @patch("open_search_db.requests.get")
    def test_429_retries(self, mock_get):
        mock_get.return_value = MockResponse(429, text="Too Many Requests")
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is None
        assert mock_get.call_count == DEFAULT_RETRY_COUNT

    @patch("open_search_db.requests.get")
    def test_503_then_200_succeeds(self, mock_get):
        success_resp = MockResponse(200, {"hits": {"hits": [{"_id": "1"}]}})
        mock_get.side_effect = [
            MockResponse(503, text="Service Unavailable"),
            success_resp,
        ]
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is success_resp
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# 14-16: Network exceptions
# ---------------------------------------------------------------------------
class TestNetworkExceptions:
    @patch("open_search_db.requests.get")
    def test_connection_error_retries_then_none(self, mock_get):
        import requests as req

        mock_get.side_effect = req.exceptions.ConnectionError("Connection refused")
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is None
        assert mock_get.call_count == DEFAULT_RETRY_COUNT

    @patch("open_search_db.requests.get")
    def test_timeout_then_success(self, mock_get):
        import requests as req

        success_resp = MockResponse(200, {"hits": {"hits": []}})
        mock_get.side_effect = [
            req.exceptions.Timeout("Read timed out"),
            success_resp,
        ]
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is success_resp
        assert mock_get.call_count == 2

    @patch("open_search_db.requests.get")
    def test_read_timeout_retries(self, mock_get):
        import requests as req

        mock_get.side_effect = req.exceptions.ReadTimeout("Read timed out")
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is None
        assert mock_get.call_count == DEFAULT_RETRY_COUNT


# ---------------------------------------------------------------------------
# 17: Backoff timing
# ---------------------------------------------------------------------------
class TestBackoff:
    @patch("open_search_db.requests.get")
    @patch("open_search_db.time.sleep")
    def test_backoff_increases_linearly(self, mock_sleep, mock_get):
        mock_get.return_value = MockResponse(500, text="error")
        OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        # 5 attempts → 4 sleeps: min(2,10), min(4,10), min(6,10), min(8,10)
        expected = [call(2), call(4), call(6), call(8)]
        assert mock_sleep.call_args_list == expected


# ---------------------------------------------------------------------------
# 18-20: Logging
# ---------------------------------------------------------------------------
class TestLogging:
    @patch("open_search_db.requests.get")
    def test_non_retryable_logs_tag_and_status(self, mock_get, caplog):
        mock_get.return_value = MockResponse(400, text="Bad Request")
        import logging

        with caplog.at_level(logging.INFO):
            OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        log_output = caplog.text
        assert f"{_TAG}[ERROR]" in log_output
        assert "non_retryable" in log_output
        assert "status=400" in log_output
        assert f"db={VALID_PROJECT}" in log_output

    @patch("open_search_db.requests.get")
    def test_network_exception_logs_tag_and_type(self, mock_get, caplog):
        import logging

        import requests as req

        mock_get.side_effect = req.exceptions.ConnectionError("refused")

        with caplog.at_level(logging.INFO):
            OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        log_output = caplog.text
        assert f"{_TAG}[WARN]" in log_output
        assert "cat=network" in log_output
        assert "ConnectionError" in log_output
        assert f"db={VALID_PROJECT}" in log_output

    @patch("open_search_db.requests.get")
    def test_final_failure_logs_tag_and_count(self, mock_get, caplog):
        mock_get.return_value = MockResponse(500, text="error")
        import logging

        with caplog.at_level(logging.INFO):
            OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        log_output = caplog.text
        assert f"{_TAG}[ERROR]" in log_output
        assert "cat=exhausted" in log_output
        assert f"after {DEFAULT_RETRY_COUNT} attempts" in log_output
        assert f"db={VALID_PROJECT}" in log_output

    @patch("open_search_db.requests.get")
    def test_newlines_in_error_are_sanitized(self, mock_get, caplog):
        """Multiline error text (e.g. HTML error page) becomes single-line."""
        multiline_body = "<html>\n<body>\n<h1>503 Service Unavailable</h1>\n</body>\n</html>"
        mock_get.return_value = MockResponse(400, text=multiline_body)
        import logging

        with caplog.at_level(logging.INFO):
            OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        error_records = [r for r in caplog.records if f"{_TAG}[ERROR]" in r.getMessage()]
        assert len(error_records) >= 1, f"Expected at least one {_TAG}[ERROR] log record"
        for record in error_records:
            msg = record.getMessage()
            assert "\n" not in msg, "Log message should not contain raw newlines"
            assert " | " in msg, "Newlines in error text should be replaced with ' | '"

    @patch("open_search_db.requests.get")
    def test_transient_retry_logs_warn_level(self, mock_get, caplog):
        """Intermediate retries use WARN, not ERROR."""
        mock_get.side_effect = [
            MockResponse(503, text="unavailable"),
            MockResponse(200, {"hits": {"hits": []}}),
        ]
        import logging

        with caplog.at_level(logging.WARNING):
            OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        log_output = caplog.text
        assert f"{_TAG}[WARN]" in log_output
        assert "cat=transient" in log_output
        # Should NOT have an ERROR since it recovered
        assert f"{_TAG}[ERROR]" not in log_output


# ---------------------------------------------------------------------------
# 21-25: Additional edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    @patch("open_search_db.requests", None)
    def test_requests_module_none_returns_none(self):
        """When requests is not installed, query should not crash."""
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is None

    def test_unserializable_dict_returns_none(self):
        """Dict that json.dumps cannot serialize returns None, no crash."""
        import datetime

        body = {"timestamp": datetime.datetime.now()}  # not JSON serializable
        result = OpenSearchDB.queryFromOpenSearchDB(body, VALID_PROJECT)
        assert result is None

    @patch("open_search_db.requests.get")
    def test_long_error_text_truncated(self, mock_get, caplog):
        """Very long response text is truncated in error log, no crash."""
        long_text = "x" * 10000
        mock_get.return_value = MockResponse(400, text=long_text)
        import logging

        with caplog.at_level(logging.INFO):
            result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is None
        # The error log should contain truncated text, not all 10000 chars
        assert "x" * 200 in caplog.text
        assert "x" * 10000 not in caplog.text

    @patch("open_search_db.requests.get")
    def test_unknown_4xx_status_code_non_retryable(self, mock_get):
        """Any 4xx (except 429) is treated as a client error and fails immediately."""
        mock_get.return_value = MockResponse(418, text="I'm a teapot")
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is None
        assert mock_get.call_count == 1  # no retry for client errors

    @patch("open_search_db.requests.get")
    def test_202_returns_response(self, mock_get):
        """202 Accepted is treated as success."""
        resp = MockResponse(202, {"hits": {"hits": []}})
        mock_get.return_value = resp
        result = OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert result is resp

    def test_sanitize_log_replaces_newlines(self):
        result = OpenSearchDB._sanitize_log("line1\nline2\r\nline3")
        assert "\n" not in result
        assert "\r" not in result
        assert "line1 | line2" in result

    def test_sanitize_log_caps_length(self):
        result = OpenSearchDB._sanitize_log("x" * 1000, max_len=100)
        assert len(result) == 100

    def test_serialize_failure_logs_error_with_tag(self, caplog):
        """json.dumps failure logs ERROR with structured tag."""
        import datetime
        import logging

        with caplog.at_level(logging.INFO):
            OpenSearchDB.queryFromOpenSearchDB({"bad": datetime.datetime.now()}, VALID_PROJECT)
        assert f"{_TAG}[ERROR]" in caplog.text
        assert "cat=serialize" in caplog.text
        assert f"db={VALID_PROJECT}" in caplog.text

    @patch.object(open_search_db, "OPEN_SEARCH_DB_BASE_URL", "")
    def test_url_not_set_logs_error(self, caplog):
        """Missing URL is ERROR since caller expected DB access."""
        import logging

        with caplog.at_level(logging.INFO):
            OpenSearchDB.queryFromOpenSearchDB("{}", VALID_PROJECT)
        assert f"{_TAG}[ERROR]" in caplog.text
        assert "cat=config" in caplog.text

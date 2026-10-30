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
"""Tests for OpenSearchDB query and post resilience.

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
VALID_POST_PROJECT = open_search_db.PERF_SANITY_PROJECT_NAME
SANDBOX_PROJECT = "sandbox-test-project"

# Log tag used in structured log output
_TAG = "[OpenSearchDB]"


@pytest.fixture(autouse=True)
def _patch_env():
    """Ensure every test has a fake base URL and no real sleep."""
    with (
        patch.object(open_search_db, "OPEN_SEARCH_DB_BASE_URL", FAKE_URL),
        patch.object(open_search_db, "OPEN_SEARCH_DB_USERNAME", "test-user"),
        patch.object(open_search_db, "OPEN_SEARCH_DB_PASSWORD", "test-password"),
        patch("open_search_db.time.sleep"),
    ):
        yield


# ---------------------------------------------------------------------------
# POST resilience
# ---------------------------------------------------------------------------
class TestPostPreValidation:
    @patch("open_search_db.requests.post")
    def test_local_bypass_precedes_all_validation(self, mock_post):
        unserializable = {"flat_payload": {"bad": object()}}
        with (
            patch.object(open_search_db, "DISABLE_OPEN_SEARCH_DB_FOR_LOCAL_TEST", True),
            patch.object(open_search_db, "OPEN_SEARCH_DB_BASE_URL", ""),
            patch.object(open_search_db, "OPEN_SEARCH_DB_USERNAME", ""),
            patch.object(open_search_db, "OPEN_SEARCH_DB_PASSWORD", ""),
        ):
            result = OpenSearchDB.postToOpenSearchDB(unserializable, "not-writeable")
        assert result is True
        mock_post.assert_not_called()

    @patch("open_search_db.requests.post")
    @patch.object(open_search_db, "OPEN_SEARCH_DB_BASE_URL", "")
    def test_empty_base_url_returns_false(self, mock_post):
        result = OpenSearchDB.postToOpenSearchDB({"s_name": "test"}, VALID_POST_PROJECT)
        assert result is False
        mock_post.assert_not_called()

    @patch("open_search_db.requests.post")
    @patch.object(open_search_db, "OPEN_SEARCH_DB_USERNAME", "")
    def test_missing_credentials_returns_false(self, mock_post):
        result = OpenSearchDB.postToOpenSearchDB({"s_name": "test"}, VALID_POST_PROJECT)
        assert result is False
        mock_post.assert_not_called()

    @patch("open_search_db.requests.post")
    def test_project_without_write_access_returns_false(self, mock_post):
        result = OpenSearchDB.postToOpenSearchDB({"s_name": "test"}, VALID_PROJECT)
        assert result is False
        mock_post.assert_not_called()

    @patch("open_search_db.requests.post")
    def test_type_check_failure_returns_false(self, mock_post):
        result = OpenSearchDB.postToOpenSearchDB({"unknown_key": "test"}, VALID_POST_PROJECT)
        assert result is False
        mock_post.assert_not_called()


class TestPostSerializationAndSuccess:
    @patch("open_search_db.requests.post")
    def test_serialization_failure_returns_false(self, mock_post, caplog):
        import logging

        body = {"flat_payload": {"bad": object()}}
        with caplog.at_level(logging.INFO):
            result = OpenSearchDB.postToOpenSearchDB(body, VALID_POST_PROJECT)
        assert result is False
        mock_post.assert_not_called()
        assert f"{_TAG}[ERROR]" in caplog.text
        assert "cat=serialize" in caplog.text

    @pytest.mark.parametrize("status_code", [200, 201, 202])
    @patch("open_search_db.requests.post")
    def test_success_serializes_payload_and_returns_true(self, mock_post, status_code):
        mock_post.return_value = MockResponse(status_code)
        body = {"s_name": "test", "l_count": 1}
        result = OpenSearchDB.postToOpenSearchDB(body, VALID_POST_PROJECT)
        assert result is True
        mock_post.assert_called_once()
        request = mock_post.call_args.kwargs
        assert request["url"] == f"{FAKE_URL}/dataflow2/{VALID_POST_PROJECT}/posting"
        assert json.loads(request["data"]) == body
        assert request["headers"]["Content-Type"] == "application/json"
        assert request["timeout"] == open_search_db.POST_TIMEOUT_SECONDS
        assert "auth" in request


class TestPostRetryBehavior:
    @pytest.mark.parametrize("status_code", [400, 418])
    @patch("open_search_db.requests.post")
    def test_non_retryable_status_fails_immediately(self, mock_post, status_code, caplog):
        import logging

        mock_post.return_value = MockResponse(status_code, text="client error")
        with caplog.at_level(logging.INFO):
            result = OpenSearchDB.postToOpenSearchDB({"s_name": "test"}, VALID_POST_PROJECT)
        assert result is False
        assert mock_post.call_count == 1
        open_search_db.time.sleep.assert_not_called()
        assert "cat=non_retryable" in caplog.text
        assert f"status={status_code}" in caplog.text

    @patch("open_search_db.requests.post")
    def test_retryable_status_then_success(self, mock_post, caplog):
        import logging

        mock_post.side_effect = [
            MockResponse(503, text="unavailable"),
            MockResponse(200),
        ]
        with caplog.at_level(logging.INFO):
            result = OpenSearchDB.postToOpenSearchDB({"s_name": "test"}, VALID_POST_PROJECT)
        assert result is True
        assert mock_post.call_count == 2
        open_search_db.time.sleep.assert_called_once_with(2)
        assert "cat=transient" in caplog.text

    @patch("open_search_db.requests.post")
    def test_network_exception_then_success(self, mock_post, caplog):
        import logging

        import requests as req

        mock_post.side_effect = [
            req.exceptions.ConnectionError("connection refused"),
            MockResponse(200),
        ]
        with caplog.at_level(logging.INFO):
            result = OpenSearchDB.postToOpenSearchDB({"s_name": "test"}, VALID_POST_PROJECT)
        assert result is True
        assert mock_post.call_count == 2
        open_search_db.time.sleep.assert_called_once_with(2)
        assert "cat=network" in caplog.text
        assert "ConnectionError" in caplog.text

    @patch("open_search_db.requests.post")
    def test_retryable_status_exhaustion_returns_false(self, mock_post, caplog):
        import logging

        mock_post.return_value = MockResponse(503, text="unavailable")
        with caplog.at_level(logging.INFO):
            result = OpenSearchDB.postToOpenSearchDB({"s_name": "test"}, VALID_POST_PROJECT)
        assert result is False
        assert mock_post.call_count == DEFAULT_RETRY_COUNT
        assert open_search_db.time.sleep.call_args_list == [
            call(2),
            call(4),
            call(6),
            call(8),
        ]
        transient_records = [
            record for record in caplog.records if "cat=transient" in record.getMessage()
        ]
        assert len(transient_records) == DEFAULT_RETRY_COUNT - 1
        assert "cat=exhausted" in caplog.text

    @patch("open_search_db.requests.post")
    def test_network_exception_exhaustion_returns_false(self, mock_post, caplog):
        import logging

        import requests as req

        mock_post.side_effect = req.exceptions.Timeout("timed out")
        with caplog.at_level(logging.INFO):
            result = OpenSearchDB.postToOpenSearchDB({"s_name": "test"}, VALID_POST_PROJECT)
        assert result is False
        assert mock_post.call_count == DEFAULT_RETRY_COUNT
        assert "cat=network" in caplog.text
        assert "cat=exhausted" in caplog.text


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


# ---------------------------------------------------------------------------
# 32-38: _log() helper unit tests
# ---------------------------------------------------------------------------
class TestLogHelper:
    def test_basic_format(self, caplog):
        """_log produces the expected structured tag and body."""
        import logging

        with caplog.at_level(logging.INFO):
            OpenSearchDB._log("info", "query", "my-project", "hello world")
        assert f"{_TAG}[INFO](op=query, db=my-project) hello world" in caplog.text

    def test_all_optional_fields_in_order(self, caplog):
        """Optional fields appear after op, before db, in documented order."""
        import logging

        with caplog.at_level(logging.WARNING):
            OpenSearchDB._log(
                "warning",
                "post",
                "my-project",
                "will retry",
                cat="transient",
                status=503,
                attempt=2,
                total=5,
            )
        msg = caplog.records[0].getMessage()
        assert "cat=transient" in msg
        assert "status=503" in msg
        assert "db=my-project" in msg
        assert "attempt=2/5" in msg
        assert msg.index("cat=") < msg.index("status=") < msg.index("db=") < msg.index("attempt=")

    def test_sanitizes_db_newline(self, caplog):
        """Newlines in db name are collapsed — no raw newline in the log record."""
        import logging

        with caplog.at_level(logging.ERROR):
            OpenSearchDB._log("error", "query", "proj\ninjected", "msg")
        for record in caplog.records:
            assert "\n" not in record.getMessage()

    def test_sanitizes_url(self, caplog):
        """Newlines in url are collapsed."""
        import logging

        with caplog.at_level(logging.WARNING):
            OpenSearchDB._log("warning", "query", "proj", "msg", url="http://x.com\nX-Header: evil")
        for record in caplog.records:
            assert "\n" not in record.getMessage()
        assert "url: http://x.com | X-Header: evil" in caplog.text

    def test_invalid_level_logs_error_no_exception(self, caplog):
        """Invalid level emits an error record and never raises."""
        import logging

        with caplog.at_level(logging.ERROR):
            OpenSearchDB._log("trace", "query", "proj", "msg")  # 'trace' is not standard
        assert len(caplog.records) >= 1
        assert any("invalid level" in r.getMessage() for r in caplog.records)

    def test_attempt_without_total_logs_error_no_exception(self, caplog):
        """Passing attempt without total emits an error record and never raises."""
        import logging

        with caplog.at_level(logging.ERROR):
            OpenSearchDB._log("warning", "query", "proj", "msg", attempt=1)
        assert any(
            "attempt" in r.getMessage() and "total" in r.getMessage() for r in caplog.records
        )
        # attempt field must not appear in any structured log line
        assert "attempt=1/" not in caplog.text

    def test_warning_emits_warn_tag_not_warning(self, caplog):
        """'warning' level uses [WARN] tag (not [WARNING]) for grep compatibility."""
        import logging

        with caplog.at_level(logging.WARNING):
            OpenSearchDB._log("warning", "query", "proj", "msg")
        assert f"{_TAG}[WARN]" in caplog.text
        assert f"{_TAG}[WARNING]" not in caplog.text

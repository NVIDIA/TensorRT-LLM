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
"""Tests for report_usage(), background reporter, thread lifecycle, and heartbeat."""

import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tensorrt_llm.usage import usage_lib

# ---------------------------------------------------------------------------
# Console notification tests
# ---------------------------------------------------------------------------


class TestNotification:
    def test_usage_notification_shown(self, monkeypatch, caplog, enable_telemetry):
        """Notification is logged when telemetry is enabled."""
        usage_lib._NOTIFICATION_SHOWN.clear()
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)

        mock_thread = MagicMock()
        with patch("tensorrt_llm.usage.usage_lib.threading.Thread", return_value=mock_thread):
            with caplog.at_level(logging.INFO, logger="tensorrt_llm"):
                usage_lib.report_usage()

        assert "anonymous usage data" in caplog.text

    def test_usage_notification_not_shown_when_disabled(self, monkeypatch, caplog):
        """Notification is NOT shown when telemetry is disabled."""
        usage_lib._NOTIFICATION_SHOWN.clear()
        monkeypatch.setenv("TRTLLM_NO_USAGE_STATS", "1")

        with caplog.at_level(logging.INFO, logger="tensorrt_llm"):
            usage_lib.report_usage()

        assert "anonymous usage data" not in caplog.text


# ---------------------------------------------------------------------------
# Thread lifecycle tests
# ---------------------------------------------------------------------------


class TestReportUsage:
    def test_spawns_daemon_thread(self, monkeypatch, enable_telemetry):
        """report_usage() spawns a daemon thread named 'trtllm-usage-stats'."""
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)
        usage_lib._NOTIFICATION_SHOWN.set()

        mock_thread = MagicMock()
        with patch(
            "tensorrt_llm.usage.usage_lib.threading.Thread", return_value=mock_thread
        ) as thread_cls:
            usage_lib.report_usage()
            thread_cls.assert_called_once()
            call_kwargs = thread_cls.call_args
            assert call_kwargs.kwargs["daemon"] is True
            assert call_kwargs.kwargs["name"] == "trtllm-usage-stats"
            mock_thread.start.assert_called_once()

    def test_noop_when_disabled(self, monkeypatch):
        """report_usage() does nothing when telemetry is disabled."""
        monkeypatch.setenv("TRTLLM_NO_USAGE_STATS", "1")
        with patch("tensorrt_llm.usage.usage_lib.threading.Thread") as thread_cls:
            usage_lib.report_usage()
            thread_cls.assert_not_called()

    def test_fail_silent(self, monkeypatch, enable_telemetry):
        """report_usage() never raises, even if thread creation fails."""
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)
        usage_lib._NOTIFICATION_SHOWN.set()

        with patch(
            "tensorrt_llm.usage.usage_lib.threading.Thread", side_effect=RuntimeError("boom")
        ):
            usage_lib.report_usage()  # Must not raise

    def test_report_usage_passes_args(self, monkeypatch, enable_telemetry):
        """report_usage() passes llm_args and pretrained_config to thread."""
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)
        usage_lib._NOTIFICATION_SHOWN.set()

        mock_args = MagicMock()
        mock_config = MagicMock()
        mock_thread = MagicMock()

        with patch(
            "tensorrt_llm.usage.usage_lib.threading.Thread", return_value=mock_thread
        ) as thread_cls:
            usage_lib.report_usage(
                llm_args=mock_args,
                pretrained_config=mock_config,
            )
            call_args = thread_cls.call_args
            assert call_args.kwargs["target"].__name__ == "_background_reporter"
            assert call_args.kwargs["args"] == (mock_args, mock_config, "")

    def test_report_usage_telemetry_disabled_no_thread(self, monkeypatch):
        """report_usage with TelemetryConfig(disabled=True) should not start a thread."""
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)

        telemetry_config = SimpleNamespace(disabled=True)

        initial_count = threading.active_count()
        usage_lib.report_usage(telemetry_config=telemetry_config)
        assert threading.active_count() == initial_count

    def test_get_trtllm_version_returns_string(self):
        """_get_trtllm_version returns a string."""
        result = usage_lib._get_trtllm_version()
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Duplicate reporter guard tests
# ---------------------------------------------------------------------------


class TestDuplicateReporterGuard:
    def test_second_call_is_noop(self, monkeypatch, enable_telemetry):
        """Calling report_usage() twice only spawns one thread."""
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)
        usage_lib._NOTIFICATION_SHOWN.set()

        mock_thread = MagicMock()
        with patch(
            "tensorrt_llm.usage.usage_lib.threading.Thread", return_value=mock_thread
        ) as thread_cls:
            usage_lib.report_usage()
            usage_lib.report_usage()  # second call should be a no-op
            assert thread_cls.call_count == 1

    def test_guard_resets_on_thread_failure(self, monkeypatch, enable_telemetry):
        """_REPORTER_STARTED resets if thread creation fails, allowing retry."""
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)
        usage_lib._NOTIFICATION_SHOWN.set()

        # First call: thread creation fails
        with patch(
            "tensorrt_llm.usage.usage_lib.threading.Thread",
            side_effect=RuntimeError("too many threads"),
        ):
            usage_lib.report_usage()  # should not raise

        assert not usage_lib._REPORTER_STARTED

        # Second call: thread creation succeeds
        mock_thread = MagicMock()
        with patch(
            "tensorrt_llm.usage.usage_lib.threading.Thread", return_value=mock_thread
        ) as thread_cls:
            usage_lib.report_usage()
            thread_cls.assert_called_once()
            mock_thread.start.assert_called_once()


# ---------------------------------------------------------------------------
# Heartbeat interval tests
# ---------------------------------------------------------------------------


class TestHeartbeatInterval:
    def test_default_value(self, monkeypatch):
        """Default heartbeat interval is 600."""
        monkeypatch.delenv("TRTLLM_USAGE_HEARTBEAT_INTERVAL", raising=False)
        assert usage_lib._get_heartbeat_interval() == 600

    def test_custom_value(self, monkeypatch):
        """Custom heartbeat interval is parsed correctly."""
        monkeypatch.setenv("TRTLLM_USAGE_HEARTBEAT_INTERVAL", "120")
        assert usage_lib._get_heartbeat_interval() == 120

    def test_invalid_value_falls_back(self, monkeypatch):
        """Invalid env var falls back to 600 instead of crashing."""
        monkeypatch.setenv("TRTLLM_USAGE_HEARTBEAT_INTERVAL", "abc")
        assert usage_lib._get_heartbeat_interval() == 600

    def test_empty_value_falls_back(self, monkeypatch):
        """Empty env var falls back to 600."""
        monkeypatch.setenv("TRTLLM_USAGE_HEARTBEAT_INTERVAL", "")
        assert usage_lib._get_heartbeat_interval() == 600


# ---------------------------------------------------------------------------
# Env vars read at call time tests
# ---------------------------------------------------------------------------


class TestEnvVarCallTime:
    def test_stats_server_reads_at_call_time(self, monkeypatch):
        """Stats server URL is read at call time, not import time."""
        monkeypatch.setenv(
            "TRTLLM_USAGE_STATS_SERVER", "https://events.gfestage.nvidia.com/v1.1/events/json"
        )
        assert (
            usage_lib._get_stats_server() == "https://events.gfestage.nvidia.com/v1.1/events/json"
        )

    def test_stats_server_default(self, monkeypatch):
        """Default stats server is the GXT endpoint."""
        monkeypatch.delenv("TRTLLM_USAGE_STATS_SERVER", raising=False)
        assert usage_lib._get_stats_server() == "https://events.gfe.nvidia.com/v1.1/events/json"

    def test_stats_server_rejects_non_nvidia_domain(self, monkeypatch):
        """Non-nvidia.com domains fall back to the default endpoint."""
        monkeypatch.setenv("TRTLLM_USAGE_STATS_SERVER", "https://evil.example.com/capture")
        assert usage_lib._get_stats_server() == usage_lib._DEFAULT_ENDPOINT

    def test_stats_server_rejects_http(self, monkeypatch):
        """HTTP (non-TLS) endpoints fall back to the default."""
        monkeypatch.setenv(
            "TRTLLM_USAGE_STATS_SERVER", "http://events.gfe.nvidia.com/v1.1/events/json"
        )
        assert usage_lib._get_stats_server() == usage_lib._DEFAULT_ENDPOINT

    def test_stats_server_rejects_garbage(self, monkeypatch):
        """Garbage URLs fall back to the default."""
        monkeypatch.setenv("TRTLLM_USAGE_STATS_SERVER", "not-a-url")
        assert usage_lib._get_stats_server() == usage_lib._DEFAULT_ENDPOINT

    def test_stats_server_accepts_nvidia_subdomain(self, monkeypatch):
        """Any *.nvidia.com HTTPS URL is accepted."""
        monkeypatch.setenv(
            "TRTLLM_USAGE_STATS_SERVER", "https://telemetry.internal.nvidia.com/v2/events"
        )
        assert usage_lib._get_stats_server() == "https://telemetry.internal.nvidia.com/v2/events"


# ---------------------------------------------------------------------------
# Notice text accuracy tests
# ---------------------------------------------------------------------------


class TestNoticeText:
    def test_notice_does_not_claim_no_model_names(self):
        """Notice no longer claims 'no model names' since arch class is collected."""
        assert "No model names" not in usage_lib._USAGE_NOTICE
        assert "No user-identifying information" in usage_lib._USAGE_NOTICE


# ---------------------------------------------------------------------------
# Ingress point reporter tests
# ---------------------------------------------------------------------------


class TestIngressPointReporter:
    """Tests for usage_context flowing through report_usage()."""

    def test_report_usage_passes_usage_context_to_thread(self, monkeypatch, enable_telemetry):
        """report_usage() passes usage_context string to background thread."""
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)
        usage_lib._NOTIFICATION_SHOWN.set()

        mock_thread = MagicMock()
        mock_config = MagicMock()
        mock_config.disabled = False
        mock_config.usage_context = MagicMock()
        mock_config.usage_context.value = "cli_serve"

        with patch(
            "tensorrt_llm.usage.usage_lib.threading.Thread",
            return_value=mock_thread,
        ) as thread_cls:
            usage_lib.report_usage(telemetry_config=mock_config)
            call_args = thread_cls.call_args
            assert call_args.kwargs["args"][2] == "cli_serve"

    def test_report_usage_none_config_sends_empty_context(self, monkeypatch, enable_telemetry):
        """report_usage(telemetry_config=None) sends empty usage_context."""
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)
        usage_lib._NOTIFICATION_SHOWN.set()

        mock_thread = MagicMock()
        with patch(
            "tensorrt_llm.usage.usage_lib.threading.Thread",
            return_value=mock_thread,
        ) as thread_cls:
            usage_lib.report_usage(telemetry_config=None)
            call_args = thread_cls.call_args
            assert call_args.kwargs["args"][2] == ""

    def test_report_usage_context_without_value_falls_back_to_str(
        self, monkeypatch, enable_telemetry
    ):
        """usage_context without .value attribute falls back to str()."""
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)
        usage_lib._NOTIFICATION_SHOWN.set()

        mock_thread = MagicMock()
        mock_config = SimpleNamespace(disabled=False, usage_context="plain_string")

        with patch(
            "tensorrt_llm.usage.usage_lib.threading.Thread",
            return_value=mock_thread,
        ) as thread_cls:
            usage_lib.report_usage(telemetry_config=mock_config)
            call_args = thread_cls.call_args
            assert call_args.kwargs["args"][2] == "plain_string"

    def test_report_usage_disabled_via_telemetry_config(self, monkeypatch):
        """report_usage with TelemetryConfig(disabled=True) is a no-op."""
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)

        mock_config = MagicMock()
        mock_config.disabled = True

        with patch("tensorrt_llm.usage.usage_lib.threading.Thread") as thread_cls:
            usage_lib.report_usage(telemetry_config=mock_config)
            thread_cls.assert_not_called()


# ---------------------------------------------------------------------------
# _clamp_str integration tests
# ---------------------------------------------------------------------------


class TestClampStrIntegration:
    """Verify _background_reporter() clamps long strings to schema limits."""

    def test_background_reporter_clamps_long_platform_string(self):
        """Long platform string does not cause ValidationError; len <= 256."""
        long_platform = "x" * 300

        captured = {}

        def fake_send(payload):
            captured.update(payload)

        stop_event = threading.Event()
        stop_event.set()

        with (
            patch.object(
                usage_lib,
                "_collect_system_info",
                return_value={
                    "platform": long_platform,
                    "python_version": "3.12.0",
                    "cpu_architecture": "x86_64",
                    "cpu_count": 8,
                },
            ),
            patch.object(usage_lib, "_send_to_gxt", side_effect=fake_send),
            patch.object(usage_lib, "_REPORTER_STOP", stop_event),
        ):
            usage_lib._background_reporter(None, None, "")

        assert captured, "No payload was captured"
        params = captured["events"][0]["parameters"]
        assert len(params["platform"]) <= 256


# ---------------------------------------------------------------------------
# Disaggregated serving metadata tests
# ---------------------------------------------------------------------------


class TestDisaggMetadata:
    """Verify _background_reporter() reads disagg env vars into initial report."""

    def test_disagg_env_vars_appear_in_payload(self, monkeypatch):
        """Disagg env vars appear as disaggRole and deploymentId in payload."""
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
        monkeypatch.setenv("TRTLLM_DISAGG_DEPLOYMENT_ID", "abc123")

        captured = {}

        def fake_send(payload):
            captured.update(payload)

        stop_event = threading.Event()
        stop_event.set()

        with (
            patch.object(usage_lib, "_send_to_gxt", side_effect=fake_send),
            patch.object(usage_lib, "_REPORTER_STOP", stop_event),
        ):
            usage_lib._background_reporter(None, None, "")

        assert captured, "No payload was captured"
        params = captured["events"][0]["parameters"]
        assert params["disaggRole"] == "context"
        assert params["deploymentId"] == "abc123"


class TestDisaggMetadataEmpty:
    """Verify empty defaults when disagg env vars are unset (non-disagg mode)."""

    def test_disagg_fields_empty_when_unset(self, monkeypatch):
        """Without disagg env vars, disaggRole and deploymentId are empty strings."""
        monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
        monkeypatch.delenv("TRTLLM_DISAGG_DEPLOYMENT_ID", raising=False)

        captured = {}

        def fake_send(payload):
            captured.update(payload)

        stop_event = threading.Event()
        stop_event.set()

        with (
            patch.object(usage_lib, "_send_to_gxt", side_effect=fake_send),
            patch.object(usage_lib, "_REPORTER_STOP", stop_event),
        ):
            usage_lib._background_reporter(None, None, "")

        assert captured, "No payload was captured"
        params = captured["events"][0]["parameters"]
        assert params["disaggRole"] == ""
        assert params["deploymentId"] == ""


# ---------------------------------------------------------------------------
# Rank-0 guard tests
# ---------------------------------------------------------------------------


class TestRankGuard:
    """Verify report_usage() skips reporting for non-zero MPI ranks."""

    def _setup_reporter(self, monkeypatch):
        """Reset reporter state so report_usage() can proceed."""
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)
        usage_lib._NOTIFICATION_SHOWN.set()

    def test_rank_nonzero_no_thread(self, monkeypatch, enable_telemetry):
        """report_usage() is a no-op when mpi_rank() != 0."""
        self._setup_reporter(monkeypatch)

        with patch("tensorrt_llm.usage.usage_lib.threading.Thread") as thread_cls:
            with patch("tensorrt_llm._utils.mpi_rank", return_value=1):
                usage_lib.report_usage()
            thread_cls.assert_not_called()

    def test_rank_zero_proceeds(self, monkeypatch, enable_telemetry):
        """report_usage() proceeds normally when mpi_rank() == 0."""
        self._setup_reporter(monkeypatch)

        mock_thread = MagicMock()
        with patch(
            "tensorrt_llm.usage.usage_lib.threading.Thread",
            return_value=mock_thread,
        ) as thread_cls:
            with patch("tensorrt_llm._utils.mpi_rank", return_value=0):
                usage_lib.report_usage()
            thread_cls.assert_called_once()
            mock_thread.start.assert_called_once()

    def test_rank_import_fails_proceeds(self, monkeypatch, enable_telemetry):
        """report_usage() proceeds (fail-open) when mpi_rank import fails."""
        self._setup_reporter(monkeypatch)

        mock_thread = MagicMock()
        with patch(
            "tensorrt_llm.usage.usage_lib.threading.Thread",
            return_value=mock_thread,
        ) as thread_cls:
            with patch.dict(
                "sys.modules",
                {"tensorrt_llm._utils": None},
            ):
                usage_lib.report_usage()
            thread_cls.assert_called_once()
            mock_thread.start.assert_called_once()


# ---------------------------------------------------------------------------
# Reporter shutdown tests
# ---------------------------------------------------------------------------


class TestReporterShutdown:
    """Verify _REPORTER_STOP event exits the heartbeat loop."""

    def test_reporter_stop_event_exits_heartbeat_loop(self):
        """Setting _REPORTER_STOP causes the heartbeat loop to exit."""
        send_count = {"n": 0}

        def counting_send(payload):
            send_count["n"] += 1

        stop_event = threading.Event()
        threading.Timer(0.1, stop_event.set).start()

        with (
            patch.object(usage_lib, "_send_to_gxt", side_effect=counting_send),
            patch.object(usage_lib, "_REPORTER_STOP", stop_event),
            patch.object(usage_lib, "_get_heartbeat_interval", return_value=3600),
        ):
            usage_lib._background_reporter(None, None, "")

        assert send_count["n"] == 1


# ---------------------------------------------------------------------------
# Heartbeat fail-silent continuation test
# ---------------------------------------------------------------------------


class TestHeartbeatFailSilent:
    """Verify transient heartbeat failure doesn't kill the loop."""

    def test_heartbeat_continues_after_transient_failure(self):
        """OSError on one heartbeat doesn't prevent subsequent heartbeats."""
        calls = []

        def tracking_send(payload):
            calls.append(payload)
            if len(calls) == 2:  # first heartbeat (seq=0)
                raise OSError("transient network failure")

        stop = threading.Event()
        timer = threading.Timer(0.05, stop.set)
        timer.start()

        with (
            patch.object(usage_lib, "_send_to_gxt", side_effect=tracking_send),
            patch.object(usage_lib, "_REPORTER_STOP", stop),
            patch.object(usage_lib, "_get_heartbeat_interval", return_value=0),
        ):
            usage_lib._background_reporter(None, None, "")

        timer.join(timeout=1)

        # call 1 = initial report, call 2 = heartbeat (failed), call 3+ = more heartbeats
        assert len(calls) >= 3, (
            f"Expected >=3 _send_to_gxt calls (loop should continue after failure), got {len(calls)}"
        )


# ---------------------------------------------------------------------------
# Concurrent reporter start test
# ---------------------------------------------------------------------------


class TestConcurrentReporterStart:
    """Verify _REPORTER_LOCK works under real thread contention."""

    def test_concurrent_calls_spawn_single_thread(self, monkeypatch, enable_telemetry):
        """10 concurrent report_usage() calls produce exactly 1 reporter thread."""
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)
        usage_lib._NOTIFICATION_SHOWN.set()

        call_count_lock = threading.Lock()
        threads_started = {"count": 0}

        mock_thread = MagicMock()

        def counting_thread(*args, **kwargs):
            with call_count_lock:
                threads_started["count"] += 1
            return mock_thread

        with (
            patch.object(
                usage_lib,
                "threading",
                wraps=threading,
            ) as mock_threading_mod,
            patch("tensorrt_llm._utils.mpi_rank", return_value=0),
        ):
            mock_threading_mod.Thread = MagicMock(side_effect=counting_thread)
            mock_threading_mod.Lock = threading.Lock
            mock_threading_mod.Event = threading.Event

            barrier = threading.Barrier(10)

            def call_report():
                barrier.wait()
                usage_lib.report_usage()

            pool = [threading.Thread(target=call_report) for _ in range(10)]
            for t in pool:
                t.start()
            for t in pool:
                t.join(timeout=5)

        assert threads_started["count"] == 1

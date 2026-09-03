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

import json
import logging
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from tensorrt_llm.usage import usage_lib
from tensorrt_llm.usage.config import UsageContext


@pytest.fixture(autouse=True)
def _reset_process_telemetry_state():
    """Keep process-scoped telemetry state isolated between unit tests."""
    usage_lib._SESSION = None
    usage_lib._SESSION_DISABLED = False
    usage_lib._SESSION_LOCK = threading.Lock()
    usage_lib._REPORTER_STARTED = False
    usage_lib._REPORTER_ACTIVE = False
    usage_lib._REPORTER_LOCK = threading.Lock()
    usage_lib._REPORTER_STOP = threading.Event()
    usage_lib._PENDING_TERMINAL = None
    usage_lib._PROCESS_PID = os.getpid()
    yield
    usage_lib._REPORTER_STOP.set()
    usage_lib._SESSION = None
    usage_lib._SESSION_DISABLED = False
    usage_lib._REPORTER_STARTED = False
    usage_lib._REPORTER_ACTIVE = False
    usage_lib._PENDING_TERMINAL = None


pytestmark = pytest.mark.cpu_only


@pytest.fixture
def reporter_session(enable_telemetry):
    """Create the session required by the background reporter."""
    assert usage_lib.apply_usage_session_config()


def _visual_gen_args():
    """Return the small validated-config surface consumed by the reporter."""
    parallel = SimpleNamespace(
        cfg_size=1,
        ulysses_size=1,
        async_ulysses=False,
        ring_size=1,
        attn2d_size=(1, 1),
        tp_size=1,
        parallel_vae_size=1,
        parallel_vae_split_dim="width",
    )
    attention = SimpleNamespace(
        backend="VANILLA",
        sparse_attention_config=None,
        quant_attention_config=None,
    )
    return SimpleNamespace(
        parallel_config=parallel,
        attention_config=attention,
        cache_config=None,
        cuda_graph_config=SimpleNamespace(enable=False),
        torch_compile_config=SimpleNamespace(enable=False),
    )


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

    def test_report_visual_gen_usage_spawns_visual_gen_reporter(
        self, monkeypatch, enable_telemetry
    ):
        """VisualGen starts its dedicated initial/heartbeat reporter."""
        usage_lib._NOTIFICATION_SHOWN.set()
        mock_thread = MagicMock()

        with patch.object(usage_lib.threading, "Thread", return_value=mock_thread) as thread_cls:
            usage_lib.report_visual_gen_usage(_visual_gen_args())

        assert thread_cls.call_args.kwargs["target"] is usage_lib._visual_gen_background_reporter
        assert thread_cls.call_args.kwargs["name"] == "trtllm-visual-gen-usage-stats"
        assert thread_cls.call_args.kwargs["daemon"] is True
        mock_thread.start.assert_called_once()

    def test_visual_gen_background_reporter_sends_bounded_initial_event(self, reporter_session):
        """Resolved worker metadata appears in the VisualGen initial event."""
        usage_lib._REPORTER_STOP.set()
        sent = []
        metadata = {
            "model_id": "nvidia/test-model",
            "pipeline_class_name": "TestPipeline",
            "resolved_pipeline_class": "ResolvedPipeline",
            "modality": "image",
            "launch_mode": "local_spawn",
            "node_count": 1,
            "n_workers": 2,
            "quantization_algo": "NVFP4",
            "dynamic_weight_quant": True,
            "quantized_components": ["transformer"],
        }

        with (
            patch.object(usage_lib, "_collect_system_info", return_value={}),
            patch.object(usage_lib, "_collect_gpu_info", return_value={"gpu_count": 2}),
            patch.object(
                usage_lib,
                "_collect_visual_gen_config_payloads",
                return_value=("{}", "{}"),
            ),
            patch.object(usage_lib, "_send_to_gxt", side_effect=sent.append),
        ):
            usage_lib._visual_gen_background_reporter(
                _visual_gen_args(), metadata, "visual_gen_class"
            )

        parameters = sent[0]["events"][0]["parameters"]
        assert sent[0]["events"][0]["name"] == "trtllm_visual_gen_initial_report"
        assert parameters["modelId"] == "nvidia/test-model"
        assert parameters["modality"] == "image"
        assert parameters["nWorkers"] == 2
        assert parameters["quantizedComponentsJson"] == '["transformer"]'

    def test_visual_gen_heartbeat_reports_runtime_and_current_counters(self, monkeypatch):
        """VisualGen heartbeat carries topology and the latest lifecycle snapshot."""

        class _OneHeartbeat:
            def __init__(self):
                self.wait_count = 0

            def wait(self, timeout):
                del timeout
                self.wait_count += 1
                return self.wait_count > 1

        monkeypatch.setenv("TRTLLM_USAGE_FORCE_ENABLED", "1")
        assert usage_lib.record_visual_gen_initialization_attempt()
        assert usage_lib.record_visual_gen_initialized()
        sent = []

        with (
            patch.object(usage_lib, "_collect_system_info", return_value={}),
            patch.object(usage_lib, "_collect_gpu_info", return_value={"gpu_count": 4}),
            patch.object(
                usage_lib,
                "_collect_visual_gen_config_payloads",
                return_value=("{}", "{}"),
            ),
            patch.object(usage_lib, "_send_to_gxt", side_effect=sent.append),
            patch.object(usage_lib, "_REPORTER_STOP", _OneHeartbeat()),
        ):
            usage_lib._visual_gen_background_reporter(
                _visual_gen_args(), {"n_workers": 3}, "visual_gen_class"
            )

        assert [payload["events"][0]["name"] for payload in sent] == [
            "trtllm_visual_gen_initial_report",
            "trtllm_visual_gen_heartbeat",
        ]
        heartbeat = sent[1]["events"][0]["parameters"]
        assert heartbeat["seq"] == 0
        assert heartbeat["runtimeKind"] == "visual_gen"
        assert heartbeat["ingressPoint"] == "visual_gen_class"
        assert heartbeat["nWorkers"] == 3
        assert heartbeat["gpuCount"] == 4
        assert heartbeat["visualGenInitializationAttempts"] == 1
        assert heartbeat["visualGenInstancesCreated"] == 1
        assert heartbeat["activeVisualGenInstances"] == 1
        assert heartbeat["visualGenInitializationFailures"] == 0


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
        mock_config.usage_context = UsageContext.CLI_SERVE

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

    @pytest.mark.parametrize(
        "invalid_context",
        ["plain_string", SimpleNamespace(value="cli_serve"), object()],
    )
    def test_arbitrary_usage_context_falls_back(self, invalid_context):
        """Only enum-backed categorical ingress values reach telemetry."""
        disabled, usage_context = usage_lib._telemetry_settings(
            SimpleNamespace(disabled=False, usage_context=invalid_context),
            default_usage_context=UsageContext.LLM_CLASS.value,
        )

        assert disabled is False
        assert usage_context == UsageContext.LLM_CLASS.value

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

    def test_background_reporter_clamps_long_platform_string(self, reporter_session):
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

    def test_disagg_env_vars_appear_in_payload(self, monkeypatch, reporter_session):
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

    def test_disagg_payload_includes_llm_api_config_json(self, monkeypatch, reporter_session):
        """Disagg payloads retain sanitized LLM API config JSON fields."""

        class _DisaggTelemetryArgs(BaseModel):
            tensor_parallel_size: int = Field(
                default=2, json_schema_extra={"telemetry": {"kind": "value"}}
            )

        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "generation")
        monkeypatch.setenv("TRTLLM_DISAGG_DEPLOYMENT_ID", "deploy123")

        captured = {}

        def fake_send(payload):
            captured.update(payload)

        stop_event = threading.Event()
        stop_event.set()

        with (
            patch.object(usage_lib, "_send_to_gxt", side_effect=fake_send),
            patch.object(usage_lib, "_REPORTER_STOP", stop_event),
        ):
            usage_lib._background_reporter(_DisaggTelemetryArgs(), None, "cli_serve")

        assert captured, "No payload was captured"
        params = captured["events"][0]["parameters"]
        assert params["disaggRole"] == "generation"
        assert params["deploymentId"] == "deploy123"
        assert json.loads(params["llmApiConfigJson"]) == {"tensor_parallel_size": 2}
        meta = json.loads(params["llmApiConfigMetaJson"])
        assert meta["capture_succeeded"] is True
        assert meta["args_class"] == "_DisaggTelemetryArgs"


class TestDisaggMetadataEmpty:
    """Verify empty defaults when disagg env vars are unset (non-disagg mode)."""

    def test_disagg_fields_empty_when_unset(self, monkeypatch, reporter_session):
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

        with (
            patch("tensorrt_llm.usage.usage_lib.threading.Thread") as thread_cls,
            patch.object(usage_lib, "_is_reporting_rank", return_value=False),
        ):
            usage_lib.report_usage()
        thread_cls.assert_not_called()

    def test_rank_zero_proceeds(self, monkeypatch, enable_telemetry):
        """report_usage() proceeds normally when mpi_rank() == 0."""
        self._setup_reporter(monkeypatch)

        mock_thread = MagicMock()
        with (
            patch(
                "tensorrt_llm.usage.usage_lib.threading.Thread",
                return_value=mock_thread,
            ) as thread_cls,
            patch.object(usage_lib, "_is_reporting_rank", return_value=True),
        ):
            usage_lib.report_usage()
            thread_cls.assert_called_once()
            mock_thread.start.assert_called_once()

    def test_pre_split_session_reports_after_becoming_subgroup_rank_zero(
        self, monkeypatch, enable_telemetry
    ):
        """Session setup before a communicator split does not fix its rank."""
        self._setup_reporter(monkeypatch)

        with patch.object(usage_lib, "_is_reporting_rank", return_value=False):
            assert not usage_lib._is_reporting_rank()
            assert usage_lib.apply_usage_session_config()

        mock_thread = MagicMock()
        with (
            patch.object(usage_lib, "_is_reporting_rank", return_value=True),
            patch.object(usage_lib.threading, "Thread", return_value=mock_thread) as thread_cls,
        ):
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

    def test_rank_import_fails_skips_known_distributed_run(self, monkeypatch, enable_telemetry):
        """An unknown rank cannot make every process in a known job report."""
        self._setup_reporter(monkeypatch)
        monkeypatch.setenv("WORLD_SIZE", "8")

        with (
            patch("tensorrt_llm.usage.usage_lib.threading.Thread") as thread_cls,
            patch.dict("sys.modules", {"tensorrt_llm._utils": None}),
        ):
            usage_lib.report_usage()

        thread_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Reporter shutdown tests
# ---------------------------------------------------------------------------


class TestReporterShutdown:
    """Verify _REPORTER_STOP event exits the heartbeat loop."""

    def test_reporter_stop_event_exits_heartbeat_loop(self, reporter_session):
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

    def test_heartbeat_reads_latest_counter_snapshot(self, enable_telemetry):
        """A heartbeat reflects lifecycle changes made after session startup."""

        class _OneHeartbeat:
            def __init__(self):
                self.wait_count = 0

            def wait(self, timeout):
                del timeout
                self.wait_count += 1
                return self.wait_count > 1

        assert usage_lib.record_llm_initialization_attempt()
        assert usage_lib.record_llm_initialized()
        sent = []

        with (
            patch.object(usage_lib, "_send_to_gxt", side_effect=sent.append),
            patch.object(usage_lib, "_REPORTER_STOP", _OneHeartbeat()),
        ):
            usage_lib._background_reporter(None, None, "llm_class")

        assert [payload["events"][0]["name"] for payload in sent] == [
            "trtllm_initial_report",
            "trtllm_heartbeat",
        ]
        heartbeat = sent[1]["events"][0]["parameters"]
        assert heartbeat["llmInstancesCreated"] == 1
        assert heartbeat["activeLlmInstances"] == 1


# ---------------------------------------------------------------------------
# Heartbeat fail-silent continuation test
# ---------------------------------------------------------------------------


class TestHeartbeatFailSilent:
    """Verify transient heartbeat failure doesn't kill the loop."""

    def test_heartbeat_continues_after_transient_failure(self, reporter_session):
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


class TestBackgroundReporterOptOut:
    def test_late_opt_out_prevents_initial_event(self, enable_telemetry):
        """A reporter waking after session deactivation sends nothing."""
        assert usage_lib.apply_usage_session_config()
        usage_lib._deactivate_usage_session()

        with patch.object(usage_lib, "_send_to_gxt") as send:
            usage_lib._background_reporter(None, None, "")

        send.assert_not_called()

    def test_opt_out_after_initial_claim_cancels_delivery(self, monkeypatch, enable_telemetry):
        """Opt-out between claiming and sending the initial event wins."""
        assert usage_lib.apply_usage_session_config()
        session = usage_lib._SESSION
        claimed = threading.Event()
        resume = threading.Event()
        original_claim = session.claim_initial

        def claim_then_pause():
            result = original_claim()
            claimed.set()
            resume.wait(timeout=5)
            return result

        monkeypatch.setattr(session, "claim_initial", claim_then_pause)
        with patch.object(usage_lib, "_send_to_gxt") as send:
            reporter = threading.Thread(
                target=usage_lib._background_reporter,
                args=(None, None, ""),
            )
            reporter.start()
            assert claimed.wait(timeout=5)
            usage_lib._deactivate_usage_session()
            resume.set()
            reporter.join(timeout=5)

        assert not reporter.is_alive()
        send.assert_not_called()

    def test_opt_out_before_heartbeat_cancels_delivery(self, enable_telemetry):
        """A heartbeat prepared before late opt-out is not delivered afterward."""
        assert usage_lib.apply_usage_session_config()

        class _OptOutBeforeHeartbeat:
            def __init__(self):
                self.wait_count = 0

            def wait(self, timeout):
                del timeout
                self.wait_count += 1
                if self.wait_count == 1:
                    usage_lib._deactivate_usage_session()
                    return False
                return True

            def set(self):
                pass

        sent = []
        with (
            patch.object(usage_lib, "_send_to_gxt", side_effect=sent.append),
            patch.object(usage_lib, "_REPORTER_STOP", _OptOutBeforeHeartbeat()),
        ):
            usage_lib._background_reporter(None, None, "")

        assert [payload["events"][0]["name"] for payload in sent] == ["trtllm_initial_report"]


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
            patch.object(usage_lib, "_is_reporting_rank", return_value=True),
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


class TestProcessTelemetrySession:
    """Verify local session creation and terminal-event behavior."""

    @staticmethod
    def _reset_session(monkeypatch):
        monkeypatch.setattr(usage_lib, "_SESSION", None)
        monkeypatch.setattr(usage_lib, "_SESSION_DISABLED", False)
        monkeypatch.setattr(usage_lib, "_REPORTER_STOP", threading.Event())
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)
        monkeypatch.setattr(usage_lib, "_REPORTER_ACTIVE", False)
        monkeypatch.setattr(usage_lib, "_PENDING_TERMINAL", None)
        usage_lib._NOTIFICATION_SHOWN.set()

    def test_apply_session_config_is_local_only(self, monkeypatch, enable_telemetry):
        """Session creation generates identity without threads or network I/O."""
        self._reset_session(monkeypatch)
        telemetry_config = SimpleNamespace(disabled=False, usage_context="cli_serve")

        with (
            patch.object(usage_lib.threading, "Thread") as thread_cls,
            patch.object(usage_lib, "_send_to_gxt") as send,
        ):
            assert usage_lib.apply_usage_session_config(telemetry_config)

        assert usage_lib._SESSION is not None
        assert len(usage_lib._SESSION.session_id) == 32
        assert usage_lib._SESSION.usage_context == "cli_serve"
        thread_cls.assert_not_called()
        send.assert_not_called()

    def test_apply_session_config_does_not_publish_partial_session(
        self, monkeypatch, enable_telemetry
    ):
        """A failed exit-hook registration cannot expose a partial session."""
        monkeypatch.setattr(usage_lib, "_PROCESS_EXIT_HOOK_REGISTERED", False)

        with patch.object(
            usage_lib.atexit,
            "register",
            side_effect=RuntimeError("registration failed"),
        ):
            assert not usage_lib.apply_usage_session_config()

        assert usage_lib._SESSION is None
        assert not usage_lib._PROCESS_EXIT_HOOK_REGISTERED

    def test_initial_report_slot_can_be_claimed_once(self, enable_telemetry):
        """Concurrent reporter starts cannot create two initial events."""
        assert usage_lib.apply_usage_session_config()

        assert usage_lib._SESSION.claim_initial()
        assert not usage_lib._SESSION.claim_initial()

    def test_counter_transitions_for_multiple_llms(self, enable_telemetry):
        """Sequential lifecycle calls expose overlap and final active state."""
        assert usage_lib.record_llm_initialization_attempt()
        assert usage_lib.record_llm_initialized()
        assert usage_lib.record_llm_initialization_attempt()
        assert usage_lib.record_llm_initialized()

        snapshot = usage_lib._SESSION.snapshot()
        assert snapshot["llmInitializationAttempts"] == 2
        assert snapshot["llmInstancesCreated"] == 2
        assert snapshot["activeLlmInstances"] == 2
        assert snapshot["maxConcurrentLlmInstances"] == 2

        usage_lib.record_llm_shutdown()
        usage_lib.record_llm_shutdown()
        assert usage_lib._SESSION.snapshot()["activeLlmInstances"] == 0

    def test_initialization_failure_updates_only_failure_counter(self, enable_telemetry):
        """A failed constructor attempt does not create an active instance."""
        assert usage_lib.record_llm_initialization_attempt()
        usage_lib.record_llm_initialization_failure()

        snapshot = usage_lib._SESSION.snapshot()
        assert snapshot["llmInitializationAttempts"] == 1
        assert snapshot["llmInitializationFailures"] == 1
        assert snapshot["llmInstancesCreated"] == 0
        assert snapshot["activeLlmInstances"] == 0

    def test_visual_gen_lifecycle_updates_separate_counters(self, enable_telemetry):
        """VisualGen lifecycle state is independent from LLM lifecycle state."""
        assert usage_lib.record_visual_gen_initialization_attempt()
        assert usage_lib.record_visual_gen_initialized()

        snapshot = usage_lib._SESSION.snapshot()
        assert snapshot["runtimeKind"] == "visual_gen"
        assert snapshot["visualGenInitializationAttempts"] == 1
        assert snapshot["visualGenInstancesCreated"] == 1
        assert snapshot["activeVisualGenInstances"] == 1
        assert snapshot["llmInitializationAttempts"] == 0

        usage_lib.record_visual_gen_shutdown()
        assert usage_lib._SESSION.snapshot()["activeVisualGenInstances"] == 0

    def test_mixed_session_exit_contains_both_runtime_counters(self, enable_telemetry):
        """A shared process exit snapshot identifies mixed LLM/VisualGen use."""
        assert usage_lib.record_llm_initialization_attempt()
        assert usage_lib.record_llm_initialized()
        assert usage_lib.record_visual_gen_initialization_attempt()
        assert usage_lib.record_visual_gen_initialized()
        sent = []

        with patch.object(usage_lib, "_send_to_gxt", side_effect=sent.append):
            assert usage_lib.report_exit(
                usage_lib.TerminalOutcome(
                    termination_kind="clean",
                    component="server",
                    exit_code_known=True,
                    exit_code=0,
                )
            )

        parameters = sent[0]["events"][0]["parameters"]
        assert parameters["runtimeKind"] == "mixed"
        assert parameters["llmInstancesCreated"] == 1
        assert parameters["visualGenInstancesCreated"] == 1

    def test_monotonic_counters_saturate_at_uint32(self, enable_telemetry):
        """Cumulative counters never exceed the SMS PositiveInt bound."""
        assert usage_lib.apply_usage_session_config()
        usage_lib._SESSION.llm_initialization_attempts = usage_lib.schema._UINT32_MAX

        assert usage_lib._SESSION.record_llm_initialization_attempt()
        snapshot = usage_lib._SESSION.snapshot()
        assert snapshot["llmInitializationAttempts"] == usage_lib.schema._UINT32_MAX

    def test_session_resets_when_process_id_changes(self, enable_telemetry):
        """A forked child cannot reuse its parent's session identity or locks."""
        assert usage_lib.apply_usage_session_config()
        parent_session_id = usage_lib._SESSION.session_id
        usage_lib._PROCESS_PID -= 1

        assert usage_lib.apply_usage_session_config()
        assert usage_lib._SESSION.session_id != parent_session_id
        assert usage_lib._SESSION.owner_pid == os.getpid()

    def test_invalid_disabled_value_fails_closed(self, monkeypatch, enable_telemetry):
        """An unvalidated opt-out value cannot accidentally enable telemetry."""
        self._reset_session(monkeypatch)

        assert not usage_lib.apply_usage_session_config({"disabled": "false"})
        assert usage_lib._SESSION is None

    def test_unvalidated_enabled_dict_fails_closed(self, enable_telemetry):
        """Raw config dictionaries cannot opt in to early failure reporting."""
        assert not usage_lib.apply_usage_session_config({"disabled": False})
        assert usage_lib._SESSION is None

    def test_late_config_opt_out_deactivates_early_session(self, enable_telemetry):
        """A parsed config opt-out overrides an earlier CLI-only decision."""
        assert usage_lib.apply_usage_session_config(default_usage_context="cli_serve")
        stale_session = usage_lib._SESSION

        assert not usage_lib.apply_usage_session_config({"disabled": True})
        assert usage_lib._SESSION is None
        assert usage_lib._REPORTER_STOP.is_set()
        assert not stale_session.claim_initial()
        assert (
            stale_session.claim_terminal(usage_lib.TerminalOutcome(termination_kind="unknown"))
            is None
        )

    def test_explicit_opt_out_remains_sticky_after_fork_reset(self, enable_telemetry):
        """Resetting inherited locks cannot discard an explicit user opt-out."""
        assert usage_lib.apply_usage_session_config()
        assert not usage_lib.apply_usage_session_config({"disabled": True})
        usage_lib._PROCESS_PID -= 1

        assert not usage_lib.apply_usage_session_config()
        assert usage_lib._SESSION is None

    def test_rank_transition_keeps_local_session_but_skips_emission(self, enable_telemetry):
        """Early global rank selection cannot suppress a later subgroup rank 0."""
        with patch.object(usage_lib, "_is_reporting_rank", return_value=True):
            assert usage_lib.apply_usage_session_config()

        with (
            patch.object(usage_lib, "_is_reporting_rank", return_value=False),
            patch("tensorrt_llm.usage.usage_lib.threading.Thread") as thread_cls,
        ):
            assert usage_lib.apply_usage_session_config()
            usage_lib.report_usage()

        assert usage_lib._SESSION is not None
        assert not usage_lib._REPORTER_STOP.is_set()
        thread_cls.assert_not_called()

    def test_nonreporting_rank_does_not_claim_terminal_slot(self, enable_telemetry):
        """A later subgroup rank 0 can still emit the process terminal event."""
        assert usage_lib.apply_usage_session_config()

        with patch.object(usage_lib, "_is_reporting_rank", return_value=False):
            assert not usage_lib.report_exit(
                usage_lib.TerminalOutcome(
                    termination_kind="exception",
                    exit_code_known=True,
                    exit_code=1,
                )
            )

        sent = []
        with (
            patch.object(usage_lib, "_is_reporting_rank", return_value=True),
            patch.object(usage_lib, "_send_to_gxt", side_effect=sent.append),
        ):
            assert usage_lib.report_exit(
                usage_lib.TerminalOutcome(
                    termination_kind="exception",
                    exit_code_known=True,
                    exit_code=1,
                )
            )

        assert len(sent) == 1

    def test_first_termination_observation_is_preserved(self, enable_telemetry):
        """A later shutdown symptom cannot replace the original failure cause."""
        assert usage_lib.apply_usage_session_config()
        usage_lib.record_termination_observation(
            usage_lib.TerminalOutcome(
                termination_kind="worker_failure",
                component="engine_worker",
                reporting_source="executor_proxy",
                exit_code_known=True,
                exit_code=137,
                signal_number=9,
            ),
            lifecycle_phase="serving",
        )
        usage_lib.record_termination_observation(
            usage_lib.TerminalOutcome(
                termination_kind="signal",
                component="server",
            ),
            lifecycle_phase="unknown",
        )

        session = usage_lib._get_session()
        assert session is not None
        terminal = session.claim_terminal(
            usage_lib.TerminalOutcome(
                termination_kind="signal",
                component="server",
                exit_code_known=True,
                exit_code=130,
                signal_number=2,
            )
        )
        assert terminal is not None
        snapshot, outcome = terminal
        assert snapshot["lifecyclePhase"] == "serving"
        assert outcome == usage_lib.TerminalOutcome(
            termination_kind="worker_failure",
            component="engine_worker",
            reporting_source="executor_proxy",
            exit_code_known=True,
            exit_code=137,
            signal_number=9,
        )

    def test_terminal_observer_does_not_create_a_missing_session(self, enable_telemetry):
        """Late observers cannot bypass an earlier opt-out or rank decision."""
        assert not usage_lib.report_exit(
            usage_lib.TerminalOutcome(
                termination_kind="exception",
                component="server",
                exit_code_known=True,
                exit_code=1,
            ),
            lifecycle_phase="serving",
        )
        assert usage_lib._SESSION is None

    def test_terminal_event_uses_shared_session_id(self, monkeypatch, enable_telemetry):
        """Initial and terminal payloads can be joined by sessionId."""
        self._reset_session(monkeypatch)
        telemetry_config = SimpleNamespace(disabled=False, usage_context="cli_serve")
        assert usage_lib.apply_usage_session_config(telemetry_config)
        session_id = usage_lib._SESSION.session_id
        monkeypatch.setattr(usage_lib, "_TERMINAL_FLUSH_TIMEOUT", 0)

        mock_thread = MagicMock()
        with patch.object(usage_lib.threading, "Thread", return_value=mock_thread) as thread_cls:
            assert usage_lib.report_exit(
                usage_lib.TerminalOutcome(
                    termination_kind="signal",
                    component="server",
                    exit_code_known=True,
                    exit_code=130,
                    signal_number=2,
                ),
                lifecycle_phase="serving",
                telemetry_config=telemetry_config,
            )

        _, payload, _ = thread_cls.call_args.kwargs["args"]
        event = payload["events"][0]
        assert payload["sessionId"] == session_id
        assert event["name"] == "trtllm_exit_report"
        assert event["parameters"]["ingressPoint"] == "cli_serve"
        assert event["parameters"]["exitCode"] == 130

    def test_terminal_event_contains_correlation_and_counter_snapshot(
        self, monkeypatch, enable_telemetry
    ):
        """Terminal-only sessions carry enough state for independent analysis."""
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "ctx0")
        monkeypatch.setenv("TRTLLM_DISAGG_DEPLOYMENT_ID", "deployment")
        assert usage_lib.record_llm_initialization_attempt(default_usage_context="cli_serve")
        usage_lib.record_llm_initialization_failure()

        sent = []
        with patch.object(usage_lib, "_send_to_gxt", side_effect=sent.append):
            assert usage_lib.report_exit(
                usage_lib.TerminalOutcome(
                    termination_kind="exception",
                    component="disagg_worker",
                    exit_code_known=True,
                    exit_code=1,
                ),
                lifecycle_phase="model_initialization",
            )

        params = sent[0]["events"][0]["parameters"]
        assert params["deploymentId"] == "deployment"
        assert params["disaggRole"] == "ctx0"
        assert params["llmInitializationAttempts"] == 1
        assert params["llmInitializationFailures"] == 1
        assert params["llmInstancesCreated"] == 0

    def test_unknown_exit_code_uses_zero_sentinel(self, enable_telemetry):
        """Unknown exit status cannot leak a guessed or stale numeric code."""
        assert usage_lib.apply_usage_session_config()
        sent = []

        with patch.object(usage_lib, "_send_to_gxt", side_effect=sent.append):
            assert usage_lib.report_exit(
                usage_lib.TerminalOutcome(
                    termination_kind="unknown",
                    exit_code_known=False,
                    exit_code=137,
                )
            )

        params = sent[0]["events"][0]["parameters"]
        assert params["exitCodeKnown"] is False
        assert params["exitCode"] == 0

    def test_repeated_terminal_calls_send_once(self, monkeypatch, enable_telemetry):
        """The first terminal caller wins and later calls are no-ops."""
        self._reset_session(monkeypatch)
        telemetry_config = SimpleNamespace(disabled=False, usage_context="llm_class")
        assert usage_lib.apply_usage_session_config(telemetry_config)
        sent = []

        with patch.object(usage_lib, "_send_to_gxt", side_effect=sent.append):
            first = usage_lib.report_exit(
                usage_lib.TerminalOutcome(
                    termination_kind="clean",
                    component="llm",
                    exit_code_known=False,
                ),
                lifecycle_phase="serving",
                telemetry_config=telemetry_config,
            )
            second = usage_lib.report_exit(
                usage_lib.TerminalOutcome(
                    termination_kind="exception",
                    component="llm",
                    exit_code_known=False,
                ),
                lifecycle_phase="serving",
                telemetry_config=telemetry_config,
            )

        assert first is True
        assert second is False
        assert len(sent) == 1
        assert sent[0]["events"][0]["parameters"]["terminationKind"] == "clean"

    def test_terminal_claim_survives_sender_thread_failure(self, enable_telemetry):
        """Delivery setup failure cannot reopen the terminal-event slot."""
        assert usage_lib.apply_usage_session_config()

        with patch.object(usage_lib.threading, "Thread", side_effect=RuntimeError("no threads")):
            assert usage_lib.report_exit(
                usage_lib.TerminalOutcome(
                    termination_kind="exception",
                    exit_code_known=True,
                    exit_code=1,
                )
            )

        assert not usage_lib.report_exit(
            usage_lib.TerminalOutcome(
                termination_kind="clean",
                exit_code_known=True,
            )
        )

    def test_terminal_wait_is_bounded(self, monkeypatch, enable_telemetry):
        """A blocked network send cannot hold shutdown past the configured bound."""
        self._reset_session(monkeypatch)
        monkeypatch.setattr(usage_lib, "_TERMINAL_FLUSH_TIMEOUT", 0.01)
        assert usage_lib.apply_usage_session_config(default_usage_context="llm_class")
        release_send = threading.Event()
        terminal_threads = []
        real_thread = threading.Thread

        def blocking_send(payload):
            del payload
            release_send.wait(timeout=1)

        def track_thread(*args, **kwargs):
            thread = real_thread(*args, **kwargs)
            terminal_threads.append(thread)
            return thread

        started = time.monotonic()
        with (
            patch.object(usage_lib, "_send_to_gxt", side_effect=blocking_send),
            patch.object(usage_lib.threading, "Thread", side_effect=track_thread),
        ):
            assert usage_lib.report_exit(
                usage_lib.TerminalOutcome(
                    termination_kind="exception",
                    component="llm",
                    exit_code_known=True,
                    exit_code=1,
                ),
                lifecycle_phase="model_initialization",
                default_usage_context="llm_class",
            )
        elapsed = time.monotonic() - started
        release_send.set()
        assert len(terminal_threads) == 1
        terminal_threads[0].join(timeout=1)

        assert elapsed < 0.2
        assert not terminal_threads[0].is_alive()

    def test_terminal_reuses_active_background_reporter(self, monkeypatch, enable_telemetry):
        """Process-exit fallback need not create a thread during Python atexit."""
        self._reset_session(monkeypatch)
        monkeypatch.setattr(usage_lib, "_TERMINAL_FLUSH_TIMEOUT", 0)
        assert usage_lib.apply_usage_session_config()
        usage_lib._REPORTER_ACTIVE = True
        sent = []

        with (
            patch.object(usage_lib.threading, "Thread") as thread_cls,
            patch.object(usage_lib, "_send_to_gxt", side_effect=sent.append),
        ):
            assert usage_lib.report_exit(
                usage_lib.TerminalOutcome(
                    termination_kind="unknown",
                    exit_code_known=False,
                )
            )
            thread_cls.assert_not_called()
            assert usage_lib._PENDING_TERMINAL is not None
            usage_lib._finish_background_reporter()

        assert len(sent) == 1
        assert usage_lib._PENDING_TERMINAL is None
        assert usage_lib._REPORTER_ACTIVE is False

    def test_opt_out_cancels_queued_terminal_and_releases_waiter(
        self, monkeypatch, enable_telemetry
    ):
        """Late opt-out clears queued terminal work and wakes its waiter."""
        self._reset_session(monkeypatch)
        monkeypatch.setattr(usage_lib, "_TERMINAL_FLUSH_TIMEOUT", 0)
        assert usage_lib.apply_usage_session_config()
        usage_lib._REPORTER_ACTIVE = True

        with patch.object(usage_lib, "_send_to_gxt") as send:
            assert usage_lib.report_exit(
                usage_lib.TerminalOutcome(
                    termination_kind="exception",
                    exit_code_known=True,
                    exit_code=1,
                )
            )
            pending = usage_lib._PENDING_TERMINAL
            assert pending is not None
            assert not pending.completion.is_set()

            usage_lib._deactivate_usage_session()
            assert pending.completion.is_set()
            assert usage_lib._PENDING_TERMINAL is None
            usage_lib._finish_background_reporter()

        send.assert_not_called()

    def test_concurrent_terminal_calls_send_once(self, monkeypatch, enable_telemetry):
        """Racing shutdown paths still produce one terminal event."""
        self._reset_session(monkeypatch)
        assert usage_lib.apply_usage_session_config(default_usage_context="cli_serve")
        sent = []
        sent_lock = threading.Lock()
        barrier = threading.Barrier(10)
        results = []

        def capture_send(payload):
            with sent_lock:
                sent.append(payload)

        def call_report_exit():
            barrier.wait()
            result = usage_lib.report_exit(
                usage_lib.TerminalOutcome(
                    termination_kind="exception",
                    component="server",
                    exit_code_known=True,
                    exit_code=1,
                ),
                lifecycle_phase="serving",
                default_usage_context="cli_serve",
            )
            with sent_lock:
                results.append(result)

        with patch.object(usage_lib, "_send_to_gxt", side_effect=capture_send):
            callers = [threading.Thread(target=call_report_exit) for _ in range(10)]
            for caller in callers:
                caller.start()
            for caller in callers:
                caller.join(timeout=2)

        assert results.count(True) == 1
        assert len(sent) == 1

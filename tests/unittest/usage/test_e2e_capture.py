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
"""End-to-end telemetry capture test.

Verifies the full data flow: LLM.__init__() → report_usage() →
_background_reporter() → _send_to_gxt() → HTTP POST with valid JSON.

Uses a local HTTP capture server to intercept the telemetry payload without
hitting any external endpoint.

Requirements:
    - GPU (loads TinyLlama via PyTorch backend)
    - LLM_MODELS_ROOT set (or /home/scratch.trt_llm_data_ci accessible)
    - Must be run with TRTLLM_USAGE_FORCE_ENABLED=1 to bypass pytest
      auto-detection (conftest or env)

Usage:
    TRTLLM_USAGE_FORCE_ENABLED=1 LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models \
        python -m pytest tests/unittest/usage/test_e2e_capture.py -v -s
"""

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Model path resolution (same pattern as test_llm_telemetry.py)
# ---------------------------------------------------------------------------

MODEL_NAME = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


def _get_model_path():
    """Resolve TinyLlama model path from LLM_MODELS_ROOT."""
    root = os.environ.get("LLM_MODELS_ROOT")
    if root is None:
        # Fallback to standard scratch path
        fallback = Path("/home/scratch.trt_llm_data_ci/llm-models")
        if fallback.exists():
            root = str(fallback)
    if root is None:
        pytest.skip("LLM_MODELS_ROOT not set and fallback path not available")
    model_path = Path(root) / MODEL_NAME
    if not model_path.exists():
        pytest.skip(f"Model not found at {model_path}")
    return str(model_path)


# ---------------------------------------------------------------------------
# Local HTTP capture server
# ---------------------------------------------------------------------------


class CaptureHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures POST bodies."""

    captured_payloads = []
    capture_event = threading.Event()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"_raw": body.decode("utf-8", errors="replace")}

        CaptureHandler.captured_payloads.append(payload)
        CaptureHandler.capture_event.set()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def log_message(self, format, *args):
        """Suppress request logging to keep test output clean."""
        pass


@pytest.fixture
def capture_server():
    """Start a local HTTP server on a free port and yield its URL."""
    # Reset state from any previous test
    CaptureHandler.captured_payloads = []
    CaptureHandler.capture_event = threading.Event()

    server = HTTPServer(("127.0.0.1", 0), CaptureHandler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/events"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield url

    server.shutdown()


def _assert_llm_api_config_capture(params):
    """Assert that the public LLM entrypoint populated config telemetry."""
    assert "llmApiConfigJson" in params
    assert "llmApiConfigMetaJson" in params

    config = json.loads(params["llmApiConfigJson"])
    meta = json.loads(params["llmApiConfigMetaJson"])

    assert config["tensor_parallel_size"] == 1
    assert config["pipeline_parallel_size"] == 1
    # Sensitive identifiers must never be captured.
    assert "model" not in config
    assert "tokenizer" not in config
    assert meta["args_class"] == "TorchLlmArgs"
    assert meta["capture_succeeded"] is True
    assert meta["captured_field_count"] > 0


def _wait_for_event(
    event_name: str,
    timeout: float = 30,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Wait until the local server captures a matching event."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for payload in CaptureHandler.captured_payloads:
            event = payload.get("events", [{}])[0]
            parameters = event.get("parameters", {})
            if event.get("name") == event_name and (predicate is None or predicate(parameters)):
                return payload
        time.sleep(0.05)
    pytest.fail(f"Timed out waiting for {event_name}")


def test_wait_for_event_applies_parameter_predicate(monkeypatch):
    """An earlier heartbeat does not mask a later matching snapshot."""
    active_heartbeat = {
        "events": [{"name": "trtllm_heartbeat", "parameters": {"activeLlmInstances": 1}}]
    }
    shutdown_heartbeat = {
        "events": [{"name": "trtllm_heartbeat", "parameters": {"activeLlmInstances": 0}}]
    }
    monkeypatch.setattr(
        CaptureHandler,
        "captured_payloads",
        [active_heartbeat, shutdown_heartbeat],
    )

    payload = _wait_for_event(
        "trtllm_heartbeat",
        predicate=lambda parameters: parameters.get("activeLlmInstances") == 0,
    )

    assert payload is shutdown_heartbeat


def _assert_lifecycle_snapshot(
    params,
    *,
    active_instances,
    usage_context,
):
    """Verify the process-scoped lifecycle counters emitted by a real LLM."""
    assert params["ingressPoint"] == usage_context
    assert params["llmInitializationAttempts"] == 1
    assert params["llmInstancesCreated"] == 1
    assert params["activeLlmInstances"] == active_instances
    assert params["maxConcurrentLlmInstances"] == 1
    assert params["llmInitializationFailures"] == 0


def _assert_event_matches_sms_schema(event):
    """Validate captured event parameters against the committed SMS schema."""
    import jsonschema

    from tensorrt_llm.usage import schemas

    sms_schema = json.loads(schemas.SMS_SCHEMA_PATH.read_text())
    event_schema = sms_schema["definitions"]["events"][event["name"]].copy()
    event_schema["definitions"] = sms_schema["definitions"]
    jsonschema.validate(instance=event["parameters"], schema=event_schema)


@pytest.fixture(autouse=True)
def reset_usage_state():
    """Isolate process-scoped telemetry and stop its daemon between GPU tests."""
    import tensorrt_llm.usage.usage_lib as usage_lib

    def reset():
        usage_lib._REPORTER_STOP.set()
        deadline = time.monotonic() + 2
        while usage_lib._REPORTER_ACTIVE and time.monotonic() < deadline:
            time.sleep(0.01)
        usage_lib._SESSION = None
        usage_lib._SESSION_DISABLED = False
        usage_lib._SESSION_LOCK = threading.Lock()
        usage_lib._REPORTER_STARTED = False
        usage_lib._REPORTER_ACTIVE = False
        usage_lib._REPORTER_LOCK = threading.Lock()
        usage_lib._REPORTER_STOP = threading.Event()
        usage_lib._PENDING_TERMINAL = None
        usage_lib._PROCESS_PID = os.getpid()

    reset()
    yield
    reset()


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.mark.skipif(
    not os.environ.get("TRTLLM_USAGE_FORCE_ENABLED"),
    reason="Set TRTLLM_USAGE_FORCE_ENABLED=1 to run e2e telemetry tests",
)
class TestE2ECapture:
    """End-to-end telemetry capture: real model → real HTTP POST → validate JSON."""

    def test_initial_report_captured(self, capture_server, monkeypatch):
        """Verify real-model initial telemetry and the post-shutdown heartbeat."""
        import tensorrt_llm.usage.usage_lib as usage_lib

        # Bypass endpoint validation for local capture server
        monkeypatch.setattr(usage_lib, "_get_stats_server", lambda: capture_server)
        monkeypatch.setenv("TRTLLM_USAGE_FORCE_ENABLED", "1")
        monkeypatch.setenv("TRTLLM_USAGE_HEARTBEAT_INTERVAL", "1")
        # The parent conftest (tests/unittest/conftest.py) sets
        # TRTLLM_NO_USAGE_STATS=1 to prevent telemetry during normal tests.
        # We must clear it for e2e verification.
        monkeypatch.delenv("TRTLLM_NO_USAGE_STATS", raising=False)

        # Reset the global reporter guard so we can trigger a fresh report
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)

        model_path = _get_model_path()

        from tensorrt_llm import LLM as LLM_torch
        from tensorrt_llm.llmapi import KvCacheConfig

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)

        with LLM_torch(model=model_path, kv_cache_config=kv_cache_config) as _:
            # Wait for the background thread to POST the initial report
            received = CaptureHandler.capture_event.wait(timeout=30)
            assert received, (
                "Timed out waiting for telemetry POST. The background reporter may not have fired."
            )

        heartbeat_payload = _wait_for_event(
            "trtllm_heartbeat",
            timeout=5,
            predicate=lambda parameters: parameters.get("activeLlmInstances") == 0,
        )

        # --- Validate the captured payload ---
        assert len(CaptureHandler.captured_payloads) >= 1, "Expected at least 1 captured payload"
        payload = CaptureHandler.captured_payloads[0]

        # GXT envelope fields
        assert "clientId" in payload
        assert "eventProtocol" in payload
        assert payload["eventProtocol"] == "1.6"
        assert "sessionId" in payload
        assert "sentTs" in payload
        assert "events" in payload
        assert len(payload["events"]) == 1

        event = payload["events"][0]
        assert event["name"] == "trtllm_initial_report"
        assert "ts" in event
        assert "parameters" in event

        params = event["parameters"]

        # TRT-LLM version
        assert "trtllmVersion" in params
        assert isinstance(params["trtllmVersion"], str)

        # System info
        assert "platform" in params
        assert "pythonVersion" in params
        assert "cpuArchitecture" in params
        assert "cpuCount" in params
        assert params["cpuCount"] > 0

        # GPU info (we require a GPU for this test)
        assert "gpuCount" in params
        assert params["gpuCount"] > 0
        assert "gpuName" in params
        assert len(params["gpuName"]) > 0
        assert "gpuMemoryMB" in params
        assert params["gpuMemoryMB"] > 0
        assert "cudaVersion" in params

        # Model architecture
        assert params["architectureClassName"] == "LlamaForCausalLM"
        assert params["architectureClassHash"] == ""

        # Backend
        assert params["backend"] == "pytorch"

        # Parallelism defaults for single-GPU
        assert params["tensorParallelSize"] == 1
        assert params["pipelineParallelSize"] == 1

        # Process lifecycle after successful model initialization.
        _assert_lifecycle_snapshot(
            params,
            active_instances=1,
            usage_context="llm_class",
        )

        # Features JSON
        assert "featuresJson" in params
        features = json.loads(params["featuresJson"])
        expected_keys = {
            "lora",
            "speculative_decoding",
            "prefix_caching",
            "cuda_graphs",
            "chunked_context",
            "data_parallel_size",
            "checkpoint_format",
            "load_format",
        }
        assert set(features.keys()) == expected_keys

        # Schema version
        assert payload["eventSchemaVer"] == "0.7"

        # Disagg fields present (may be empty strings)
        assert "disaggRole" in params
        assert "deploymentId" in params

        _assert_llm_api_config_capture(params)
        _assert_event_matches_sms_schema(event)

        # LLM.shutdown() decrements the active instance count. The next real
        # heartbeat must contain the current counter snapshot for the same
        # process-scoped telemetry session.
        assert heartbeat_payload["sessionId"] == payload["sessionId"]
        assert heartbeat_payload["eventSchemaVer"] == "0.7"
        heartbeat = heartbeat_payload["events"][0]
        assert heartbeat["name"] == "trtllm_heartbeat"
        _assert_lifecycle_snapshot(
            heartbeat["parameters"],
            active_instances=0,
            usage_context="llm_class",
        )
        _assert_event_matches_sms_schema(heartbeat)

    def test_cli_serve_context_e2e(self, capture_server, monkeypatch):
        """Verify CLI_SERVE context flows through to the captured payload."""
        import tensorrt_llm.usage.usage_lib as usage_lib

        # Bypass endpoint validation for local capture server
        monkeypatch.setattr(usage_lib, "_get_stats_server", lambda: capture_server)
        monkeypatch.setenv("TRTLLM_USAGE_FORCE_ENABLED", "1")
        monkeypatch.delenv("TRTLLM_NO_USAGE_STATS", raising=False)
        monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)

        model_path = _get_model_path()

        from tensorrt_llm import LLM as LLM_torch
        from tensorrt_llm.llmapi import KvCacheConfig
        from tensorrt_llm.usage import TelemetryConfig, UsageContext

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)

        with LLM_torch(
            model=model_path,
            kv_cache_config=kv_cache_config,
            telemetry_config=TelemetryConfig(usage_context=UsageContext.CLI_SERVE),
        ) as _:
            received = CaptureHandler.capture_event.wait(timeout=30)
            assert received, "Timed out waiting for telemetry POST"

        payload = CaptureHandler.captured_payloads[0]
        event = payload["events"][0]
        params = event["parameters"]
        assert payload["eventSchemaVer"] == "0.7"
        _assert_lifecycle_snapshot(
            params,
            active_instances=1,
            usage_context="cli_serve",
        )
        _assert_llm_api_config_capture(params)
        _assert_event_matches_sms_schema(event)

    def test_direct_llm_process_exit_e2e(self, capture_server, monkeypatch):
        """A real LLM child emits one correlated atexit terminal report."""
        monkeypatch.delenv("TRTLLM_NO_USAGE_STATS", raising=False)

        child_code = """
import os
import time

import tensorrt_llm.usage.usage_lib as usage_lib
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig

usage_lib._get_stats_server = lambda: os.environ["CAPTURE_SERVER_URL"]

with LLM(
    model=os.environ["MODEL_PATH"],
    kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        session = usage_lib._get_session()
        if session is not None and session.initial_reported:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("initial telemetry was not prepared")
"""
        env = os.environ.copy()
        env["CAPTURE_SERVER_URL"] = capture_server
        env["MODEL_PATH"] = _get_model_path()
        env["TRTLLM_USAGE_FORCE_ENABLED"] = "1"
        env.pop("TRTLLM_NO_USAGE_STATS", None)
        env.pop("DO_NOT_TRACK", None)
        env.pop("TELEMETRY_DISABLED", None)

        completed = subprocess.run(
            [sys.executable, "-c", child_code],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, (
            f"child failed with {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

        initial_payload = _wait_for_event("trtllm_initial_report", timeout=5)
        exit_payload = _wait_for_event("trtllm_exit_report", timeout=5)
        assert exit_payload["sessionId"] == initial_payload["sessionId"]
        assert exit_payload["eventSchemaVer"] == "0.7"

        terminal_events = [
            payload
            for payload in CaptureHandler.captured_payloads
            if payload.get("events", [{}])[0].get("name") == "trtllm_exit_report"
        ]
        assert len(terminal_events) == 1

        event = exit_payload["events"][0]
        params = event["parameters"]
        assert params["exitCodeKnown"] is False
        assert params["exitCode"] == 0
        assert params["signalNumber"] == 0
        assert params["terminationKind"] == "unknown"
        assert params["lifecyclePhase"] == "serving"
        assert params["component"] == "llm"
        assert params["reportingSource"] == "self"
        _assert_lifecycle_snapshot(
            params,
            active_instances=0,
            usage_context="llm_class",
        )
        _assert_event_matches_sms_schema(event)

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
    - LLM_MODELS_ROOT set (or /home/scratch.trt_llm_data accessible)
    - Must be run with TRTLLM_USAGE_FORCE_ENABLED=1 to bypass pytest
      auto-detection (conftest or env)

Usage:
    TRTLLM_USAGE_FORCE_ENABLED=1 LLM_MODELS_ROOT=/home/scratch.trt_llm_data/llm-models \
        python -m pytest tests/unittest/usage/test_e2e_capture.py -v -s
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

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
        fallback = Path("/home/scratch.trt_llm_data/llm-models")
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
        """Load TinyLlama and verify the initial telemetry report arrives."""
        import tensorrt_llm.usage.usage_lib as usage_lib

        # Bypass endpoint validation for local capture server
        monkeypatch.setattr(usage_lib, "_get_stats_server", lambda: capture_server)
        monkeypatch.setenv("TRTLLM_USAGE_FORCE_ENABLED", "1")
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

        # Backend
        assert params["backend"] == "pytorch"

        # Parallelism defaults for single-GPU
        assert params["tensorParallelSize"] == 1
        assert params["pipelineParallelSize"] == 1

        # Ingress point (default Python API → promoted to llm_class)
        assert params["ingressPoint"] == "llm_class"

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
        }
        assert set(features.keys()) == expected_keys

        # Schema version
        assert payload["eventSchemaVer"] == "0.1"

        # Disagg fields present (may be empty strings)
        assert "disaggRole" in params
        assert "deploymentId" in params

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
        params = payload["events"][0]["parameters"]
        assert params["ingressPoint"] == "cli_serve"

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
"""End-to-end payload verification: real model → real JSON → ground truth check.

Loads a real model on real GPUs, captures the report_usage kwargs via a spy,
then calls _background_reporter() with those real args and verifies every
JSON parameter against ground truth values from torch.cuda, platform, etc.
"""

import json
import os
import platform
import sys
import threading
from unittest.mock import patch

import pytest

from tensorrt_llm import LLM as LLM_torch
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.usage import schemas

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.llm_data import llm_models_root  # noqa: E402

pytestmark = pytest.mark.threadleak(enabled=False)

MODEL_NAME = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
_kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)


def _get_model_path():
    root = llm_models_root()
    assert root is not None, (
        "LLM_MODELS_ROOT must be set or /home/scratch.trt_llm_data must be "
        "accessible to run payload verification tests"
    )
    return str(root / MODEL_NAME)


def _make_spy():
    captured = {}

    def spy_report_usage(**kwargs):
        captured.update(kwargs)

    return captured, spy_report_usage


class TestPayloadVerification:
    """Verify actual JSON payload parameters against ground truth on real GPUs."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.model_path = _get_model_path()

    def test_payload_parameters_match_ground_truth(self):
        """Load real model, build payload, verify every parameter is accurate."""
        import torch

        import tensorrt_llm.usage.usage_lib as usage_lib

        # Step 1: Load real model, capture report_usage kwargs
        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM_torch(model=self.model_path, kv_cache_config=_kv_cache_config) as _:
                pass

        llm_args = captured.get("llm_args")
        pretrained_config = captured.get("pretrained_config")
        telemetry_config = captured.get("telemetry_config")

        assert llm_args is not None, "report_usage was not called with llm_args"
        assert pretrained_config is not None, "report_usage was not called with pretrained_config"

        # Extract usage_context the same way report_usage does
        usage_context = ""
        if telemetry_config is not None:
            ctx = getattr(telemetry_config, "usage_context", None)
            if ctx is not None:
                usage_context = ctx.value if hasattr(ctx, "value") else str(ctx)

        # Step 2: Call _background_reporter with real args, capture payload
        captured_payloads = []

        def capture_send(payload):
            captured_payloads.append(json.loads(json.dumps(payload)))

        stop = threading.Event()
        stop.set()

        with (
            patch.object(usage_lib, "_send_to_gxt", side_effect=capture_send),
            patch.object(usage_lib, "_REPORTER_STOP", stop),
        ):
            usage_lib._background_reporter(llm_args, pretrained_config, usage_context)

        assert captured_payloads, "No payloads captured from _background_reporter"

        payload = captured_payloads[0]
        params = payload["events"][0]["parameters"]
        event_name = payload["events"][0]["name"]

        assert event_name == "trtllm_initial_report"

        # Step 3: Spot-check against ground truth
        assert params["gpuName"] == torch.cuda.get_device_name(0)
        assert params["gpuCount"] == torch.cuda.device_count()
        assert params["gpuMemoryMB"] == torch.cuda.get_device_properties(0).total_memory // (
            1024 * 1024
        )
        assert params["platform"] == platform.platform()
        assert params["pythonVersion"] == platform.python_version()
        assert params["cpuArchitecture"] == platform.machine()
        assert params["cpuCount"] == os.cpu_count()
        assert params["cudaVersion"] == torch.version.cuda
        assert params["architectureClassName"] == "LlamaForCausalLM"
        assert params["backend"] == "pytorch"

        # Step 4: String length checks (ShortString<=128, LongString<=256)
        short_fields = [
            "trtllmVersion",
            "pythonVersion",
            "cpuArchitecture",
            "cudaVersion",
            "cloudProvider",
            "backend",
            "dtype",
            "quantizationAlgo",
            "kvCacheDtype",
            "ingressPoint",
            "disaggRole",
            "deploymentId",
        ]
        long_fields = ["platform", "gpuName", "architectureClassName"]

        for f in short_fields:
            v = params.get(f, "")
            assert len(v) <= 128, f"{f} len={len(v)} exceeds ShortString max 128"

        for f in long_fields:
            v = params.get(f, "")
            assert len(v) <= 256, f"{f} len={len(v)} exceeds LongString max 256"

        # Step 5: Integer range checks (0 <= x <= 4294967295)
        int_fields = [
            "cpuCount",
            "gpuCount",
            "gpuMemoryMB",
            "tensorParallelSize",
            "pipelineParallelSize",
            "contextParallelSize",
            "moeExpertParallelSize",
            "moeTensorParallelSize",
        ]
        for f in int_fields:
            v = params.get(f)
            assert isinstance(v, int), f"{f} is not int: {type(v)}"
            assert 0 <= v <= 4294967295, f"{f}={v} out of PositiveInt range"

        # Step 6: featuresJson check
        fj = params.get("featuresJson", "")
        features = json.loads(fj)
        expected_keys = {
            "lora",
            "speculative_decoding",
            "prefix_caching",
            "cuda_graphs",
            "chunked_context",
            "data_parallel_size",
        }
        assert set(features.keys()) == expected_keys

        # Step 7: Full jsonschema validation
        import jsonschema

        schema = json.loads(schemas.SMS_SCHEMA_PATH.read_text())
        initial_schema = schema["definitions"]["events"]["trtllm_initial_report"].copy()
        initial_schema["definitions"] = schema["definitions"]
        jsonschema.validate(instance=params, schema=initial_schema)

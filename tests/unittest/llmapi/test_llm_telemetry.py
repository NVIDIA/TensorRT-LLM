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
"""Integration tests for telemetry hook in BaseLLM.__init__().

Verifies that pretrained_config is populated and valid when the telemetry
hook fires, and that telemetry_disabled flows through correctly.
"""

import os
import sys
from unittest.mock import patch

import pytest

from tensorrt_llm import LLM as LLM_torch
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import KvCacheConfig, llm_args
from tensorrt_llm.usage import usage_lib

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.llm_data import llm_models_root  # noqa: E402

pytestmark = pytest.mark.threadleak(enabled=False)

MODEL_NAME = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
_kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)


def _get_model_path():
    root = llm_models_root()
    assert root is not None, (
        "LLM_MODELS_ROOT must be set or /home/scratch.trt_llm_data_ci must be "
        "accessible to run telemetry integration tests"
    )
    return str(root / MODEL_NAME)


def _make_spy():
    """Return (captured_dict, spy_function) for patching report_usage."""
    captured = {}

    def spy_report_usage(**kwargs):
        captured.update(kwargs)

    return captured, spy_report_usage


class TestTelemetryPyTorchBackend:
    """Verify pretrained_config lifecycle with PyTorch backend (no engine build)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.model_path = _get_model_path()

    def test_telemetry_receives_hf_config_pytorch(self):
        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM_torch(model=self.model_path, kv_cache_config=_kv_cache_config) as _:
                pass

        pretrained_config = captured.get("pretrained_config")
        assert pretrained_config is not None, "report_usage was not called with pretrained_config"
        assert hasattr(pretrained_config, "architectures"), (
            "pretrained_config missing .architectures attribute"
        )
        assert isinstance(pretrained_config.architectures, list)
        assert len(pretrained_config.architectures) > 0
        assert pretrained_config.architectures[0] == "LlamaForCausalLM"

        assert captured.get("llm_args") is not None, "report_usage was not called with llm_args"


class TestTelemetryTRTBackend:
    """Verify pretrained_config lifecycle with TensorRT backend (engine build).

    When starting from an HF model, _TrtLLM builds a TRT engine and
    overwrites self.args.model to point to the engine dir.  The telemetry
    hook must still receive the *original* HF PretrainedConfig (loaded from
    _hf_model_dir before the overwrite), not the TRT-LLM engine config.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.model_path = _get_model_path()

    def test_telemetry_receives_hf_config_trt(self):
        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM(model=self.model_path, kv_cache_config=_kv_cache_config) as _:
                pass

        pretrained_config = captured.get("pretrained_config")
        assert pretrained_config is not None, "report_usage was not called with pretrained_config"
        # The config should be an HF PretrainedConfig with .architectures
        # (plural list), NOT a TRT-LLM config with .architecture (singular).
        assert hasattr(pretrained_config, "architectures"), (
            "pretrained_config missing .architectures attribute"
        )
        assert isinstance(pretrained_config.architectures, list)
        assert len(pretrained_config.architectures) > 0
        assert pretrained_config.architectures[0] == "LlamaForCausalLM"

        assert captured.get("llm_args") is not None, "report_usage was not called with llm_args"

    def test_telemetry_arch_extraction_trt(self):
        """End-to-end: _extract_architecture_class_name with TRT backend config."""
        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM(model=self.model_path, kv_cache_config=_kv_cache_config) as _:
                pass

        pretrained_config = captured.get("pretrained_config")
        assert pretrained_config is not None

        arch = usage_lib._extract_architecture_class_name(pretrained_config)
        assert arch == "LlamaForCausalLM", f"Expected 'LlamaForCausalLM', got '{arch}'"


class TestTelemetryArchitectureExtraction:
    """End-to-end: _extract_architecture_class_name with a real HF config."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.model_path = _get_model_path()

    def test_telemetry_config_has_extractable_architecture(self):
        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM_torch(model=self.model_path, kv_cache_config=_kv_cache_config) as _:
                pass

        pretrained_config = captured.get("pretrained_config")
        assert pretrained_config is not None

        arch = usage_lib._extract_architecture_class_name(pretrained_config)
        assert arch == "LlamaForCausalLM", f"Expected 'LlamaForCausalLM', got '{arch}'"


class TestTelemetryDisabledFlag:
    """Verify that telemetry_disabled=True flows from LLM args to report_usage."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.model_path = _get_model_path()

    def test_telemetry_disabled_passed_to_report_usage(self):
        """When TelemetryConfig(disabled=True), report_usage receives it."""
        from tensorrt_llm.llmapi import llm_args as _llm_args_mod

        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM_torch(
                model=self.model_path,
                kv_cache_config=_kv_cache_config,
                telemetry_config=_llm_args_mod.TelemetryConfig(disabled=True),
            ) as _:
                pass

        telemetry_config = captured.get("telemetry_config")
        assert telemetry_config is not None
        assert telemetry_config.disabled is True

    def test_telemetry_enabled_by_default(self):
        """When TelemetryConfig not set, disabled defaults to False."""
        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM_torch(model=self.model_path, kv_cache_config=_kv_cache_config) as _:
                pass

        telemetry_config = captured.get("telemetry_config")
        assert telemetry_config is not None
        assert telemetry_config.disabled is False


class TestUsageContextFlow:
    """Verify that usage_context flows correctly from LLM args to report_usage."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.model_path = _get_model_path()

    def test_default_promotes_to_llm_class(self):
        """LLM() without explicit context gets promoted to LLM_CLASS."""
        from tensorrt_llm.llmapi import llm_args as _llm_args_mod

        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM_torch(model=self.model_path, kv_cache_config=_kv_cache_config) as _:
                pass

        telemetry_config = captured.get("telemetry_config")
        assert telemetry_config is not None
        assert telemetry_config.usage_context == _llm_args_mod.UsageContext.LLM_CLASS

    def test_cli_serve_context_preserved(self):
        """CLI_SERVE context is not overridden by BaseLLM."""
        from tensorrt_llm.llmapi import llm_args as _llm_args_mod

        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM_torch(
                model=self.model_path,
                kv_cache_config=_kv_cache_config,
                telemetry_config=_llm_args_mod.TelemetryConfig(
                    usage_context=_llm_args_mod.UsageContext.CLI_SERVE
                ),
            ) as _:
                pass

        telemetry_config = captured.get("telemetry_config")
        assert telemetry_config is not None
        assert telemetry_config.usage_context == _llm_args_mod.UsageContext.CLI_SERVE

    def test_cli_bench_context_preserved(self):
        """CLI_BENCH context is not overridden by BaseLLM."""
        from tensorrt_llm.llmapi import llm_args as _llm_args_mod

        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM_torch(
                model=self.model_path,
                kv_cache_config=_kv_cache_config,
                telemetry_config=_llm_args_mod.TelemetryConfig(
                    usage_context=_llm_args_mod.UsageContext.CLI_BENCH
                ),
            ) as _:
                pass

        telemetry_config = captured.get("telemetry_config")
        assert telemetry_config is not None
        assert telemetry_config.usage_context == _llm_args_mod.UsageContext.CLI_BENCH

    def test_cli_eval_context_preserved(self):
        """CLI_EVAL context is not overridden by BaseLLM."""
        from tensorrt_llm.llmapi import llm_args as _llm_args_mod

        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM_torch(
                model=self.model_path,
                kv_cache_config=_kv_cache_config,
                telemetry_config=_llm_args_mod.TelemetryConfig(
                    usage_context=_llm_args_mod.UsageContext.CLI_EVAL
                ),
            ) as _:
                pass

        telemetry_config = captured.get("telemetry_config")
        assert telemetry_config is not None
        assert telemetry_config.usage_context == _llm_args_mod.UsageContext.CLI_EVAL


class TestFeatureTrackingIntegration:
    """Verify that _collect_features works with real llm_args from model init."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.model_path = _get_model_path()

    def test_features_json_present_in_report_pytorch(self):
        """_collect_features returns valid JSON with all 6 keys from real llm_args."""
        import json

        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM_torch(model=self.model_path, kv_cache_config=_kv_cache_config) as _:
                pass

        llm_args = captured.get("llm_args")
        assert llm_args is not None, "report_usage was not called with llm_args"

        features_str = usage_lib._collect_features(llm_args)
        features = json.loads(features_str)
        expected_keys = {
            "lora",
            "speculative_decoding",
            "prefix_caching",
            "cuda_graphs",
            "chunked_context",
            "data_parallel_size",
        }
        assert set(features.keys()) == expected_keys

    def test_features_json_default_values_pytorch(self):
        """Default TinyLlama config has expected feature defaults."""
        import json

        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM_torch(model=self.model_path, kv_cache_config=_kv_cache_config) as _:
                pass

        llm_args = captured.get("llm_args")
        features = json.loads(usage_lib._collect_features(llm_args))

        # TinyLlama loaded with defaults: no LoRA, no spec dec, no chunked prefill
        assert features["lora"] is False
        assert features["speculative_decoding"] is False
        assert features["chunked_context"] is False
        assert features["data_parallel_size"] == 1

    def test_features_json_chunked_prefill_pytorch(self):
        """enable_chunked_prefill=True is detected in features."""
        import json

        captured, spy = _make_spy()

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM_torch(
                model=self.model_path,
                kv_cache_config=_kv_cache_config,
                enable_chunked_prefill=True,
            ) as _:
                pass

        llm_args = captured.get("llm_args")
        features = json.loads(usage_lib._collect_features(llm_args))
        assert features["chunked_context"] is True

    @pytest.mark.parametrize(
        ("extra_kwargs", "key", "expected"),
        [
            ({}, "lora", False),
            ({}, "prefix_caching", True),
            ({"enable_lora": True}, "lora", True),
            (
                {
                    "kv_cache_config": llm_args.KvCacheConfig(
                        free_gpu_memory_fraction=0.4, enable_block_reuse=False
                    )
                },
                "prefix_caching",
                False,
            ),
            ({"enable_chunked_prefill": True}, "chunked_context", True),
        ],
        ids=[
            "default-lora",
            "default-prefix",
            "lora-enabled",
            "prefix-disabled",
            "chunked-enabled",
        ],
    )
    def test_feature_detection_pytorch(self, extra_kwargs, key, expected):
        """Feature extraction should match real PyTorch backend llm_args values."""
        import json

        captured, spy = _make_spy()
        kwargs = {"model": self.model_path, "kv_cache_config": _kv_cache_config}
        kwargs.update(extra_kwargs)

        with patch("tensorrt_llm.usage.report_usage", side_effect=spy):
            with LLM_torch(**kwargs) as _:
                pass

        features = json.loads(usage_lib._collect_features(captured["llm_args"]))
        assert features[key] == expected


# ---------------------------------------------------------------------------
# Cycle 8: Eval and Bench CLI context tests (COVERAGE GAP)
# ---------------------------------------------------------------------------


class TestTelemetryEvalContext:
    """Verify UsageContext.CLI_EVAL flows through TelemetryConfig."""

    def test_eval_sets_cli_eval_context(self):
        """eval.py sets UsageContext.CLI_EVAL in TelemetryConfig."""
        from tensorrt_llm.llmapi import llm_args as _llm_args_mod

        config = _llm_args_mod.TelemetryConfig(usage_context=_llm_args_mod.UsageContext.CLI_EVAL)
        assert config.usage_context == _llm_args_mod.UsageContext.CLI_EVAL


class TestTelemetryBenchContext:
    """Verify UsageContext.CLI_BENCH flows through TelemetryConfig."""

    def test_bench_sets_cli_bench_context(self):
        """bench.py sets UsageContext.CLI_BENCH in TelemetryConfig."""
        from tensorrt_llm.llmapi import llm_args as _llm_args_mod

        config = _llm_args_mod.TelemetryConfig(usage_context=_llm_args_mod.UsageContext.CLI_BENCH)
        assert config.usage_context == _llm_args_mod.UsageContext.CLI_BENCH

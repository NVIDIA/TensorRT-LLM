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
"""Tests for data collection functions: system info, GPU, model, config, features."""

import json
from unittest.mock import MagicMock, patch

from tensorrt_llm.usage import schema, usage_lib

# ---------------------------------------------------------------------------
# System info tests
# ---------------------------------------------------------------------------


class TestSystemInfo:
    def test_collect_system_info(self):
        """System info returns expected keys with valid types."""
        info = usage_lib._collect_system_info()
        assert "platform" in info
        assert isinstance(info["platform"], str)
        assert "python_version" in info
        assert isinstance(info["python_version"], str)
        assert "cpu_architecture" in info
        assert isinstance(info["cpu_architecture"], str)
        assert "cpu_count" in info
        assert isinstance(info["cpu_count"], int)
        assert info["cpu_count"] > 0

    def test_collect_system_info_handles_none_cpu_count(self):
        """cpu_count can be None (e.g. some container/embedded envs)."""
        with patch("os.cpu_count", return_value=None):
            info = usage_lib._collect_system_info()
            assert info["cpu_count"] is None

    def test_collect_gpu_info_no_torch(self):
        """GPU info returns empty dict when torch is unavailable."""
        with patch.dict("sys.modules", {"torch": None}):
            result = usage_lib._collect_gpu_info()
            assert result == {}

    def test_collect_gpu_info_no_cuda(self):
        """GPU info returns empty dict when CUDA is unavailable."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch, "torch.cuda": mock_torch.cuda}):
            result = usage_lib._collect_gpu_info()
            assert result == {}

    def test_collect_gpu_info_with_cuda(self):
        """GPU info returns populated dict when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 8
        mock_torch.cuda.get_device_name.return_value = "NVIDIA H100"
        mock_props = MagicMock()
        mock_props.total_memory = 80 * 1024 * 1024 * 1024  # 80 GB in bytes
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.version.cuda = "12.4"
        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "torch.cuda": mock_torch.cuda,
                "torch.version": mock_torch.version,
            },
        ):
            result = usage_lib._collect_gpu_info()
            assert result["gpu_count"] == 8
            assert result["gpu_name"] == "NVIDIA H100"
            assert result["gpu_memory_mb"] == 80 * 1024  # 80 GB in MB
            assert result["cuda_version"] == "12.4"

    def test_collect_gpu_info_catches_import_error(self):
        """_collect_gpu_info returns {} when torch is not installed."""
        with patch.dict("sys.modules", {"torch": None}):
            assert usage_lib._collect_gpu_info() == {}

    def test_collect_gpu_info_catches_runtime_error(self):
        """_collect_gpu_info returns {} when CUDA is broken."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA error")
        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "torch.cuda": mock_torch.cuda},
        ):
            assert usage_lib._collect_gpu_info() == {}


# ---------------------------------------------------------------------------
# Model info tests
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_extract_architecture_class_name(self):
        """Extracts first architecture from config.architectures list."""
        mock = MagicMock()
        mock.architectures = ["LlamaForCausalLM"]
        assert usage_lib._extract_architecture_class_name(mock) == "LlamaForCausalLM"

    def test_extract_architecture_multiple(self):
        """Extracts first architecture when list has multiple entries."""
        mock = MagicMock()
        mock.architectures = ["Qwen2ForCausalLM", "GPT2LMHeadModel"]
        assert usage_lib._extract_architecture_class_name(mock) == "Qwen2ForCausalLM"

    def test_extract_architecture_none_config(self):
        """Returns None when config is None."""
        assert usage_lib._extract_architecture_class_name(None) is None

    def test_extract_architecture_empty_list(self):
        """Falls back to class name when architectures list is empty."""
        mock = MagicMock(spec=[])  # No attributes by default
        mock.architectures = []
        result = usage_lib._extract_architecture_class_name(mock)
        assert result is not None  # Should return the class name

    def test_extract_architecture_no_attr(self):
        """Falls back to class name when architecture attr is missing."""

        class FakeConfig:
            pass

        config = FakeConfig()
        result = usage_lib._extract_architecture_class_name(config)
        assert result == "FakeConfig"

    def test_extract_architecture_trtllm_singular(self):
        """TRT-LLM PretrainedConfig uses .architecture (singular string)."""
        mock = MagicMock(spec=[])
        mock.architecture = "LlamaForCausalLM"
        assert usage_lib._extract_architecture_class_name(mock) == "LlamaForCausalLM"

    def test_extract_architecture_engine_config_nested(self):
        """HF from_pretrained on engine dir produces nested pretrained_config dict."""

        class FakeEngineConfig:
            pretrained_config = {
                "architecture": "LlamaForCausalLM",
                "dtype": "float16",
                "hidden_size": 2048,
            }
            build_config = {"max_batch_size": 8}

        config = FakeEngineConfig()
        assert usage_lib._extract_architecture_class_name(config) == "LlamaForCausalLM"

    def test_extract_architecture_hf_takes_priority(self):
        """HF .architectures (plural) takes priority over TRT-LLM .architecture."""
        mock = MagicMock(spec=[])
        mock.architectures = ["LlamaForCausalLM"]
        mock.architecture = "ShouldNotBeUsed"
        assert usage_lib._extract_architecture_class_name(mock) == "LlamaForCausalLM"

    def test_extract_architecture_singular_over_nested(self):
        """Direct .architecture (singular) takes priority over nested dict."""

        class FakeConfig:
            architecture = "MixtralForCausalLM"
            pretrained_config = {
                "architecture": "ShouldNotBeUsed",
            }

        assert usage_lib._extract_architecture_class_name(FakeConfig()) == "MixtralForCausalLM"

    def test_no_raw_config_fields(self):
        """Ensure PR #11299's raw config fields are NOT in the schema."""
        fields = schema.TrtllmInitialReport.model_fields
        assert "num_layers" not in fields
        assert "hidden_size" not in fields
        assert "num_attention_heads" not in fields
        assert "model_type" not in fields

    def test_extract_arch_catches_attribute_error(self):
        """_extract_architecture_class_name handles configs without expected attrs."""
        result = usage_lib._extract_architecture_class_name(42)  # int has no .architectures
        assert result is not None  # falls through to type(42).__name__ == "int"


# ---------------------------------------------------------------------------
# TRT-LLM config extraction tests
# ---------------------------------------------------------------------------


class TestConfigExtraction:
    def test_extract_config(self):
        """Extracts all config fields from a fully-populated mock."""
        mock = MagicMock()
        mock.backend = "pytorch"
        mock.parallel_config.tp_size = 4
        mock.parallel_config.pp_size = 2
        mock.parallel_config.cp_size = 1
        mock.parallel_config.moe_ep_size = 8
        mock.parallel_config.moe_tp_size = 2
        mock.dtype = "float16"
        mock.quant_config.quant_algo = "fp8"
        mock.kv_cache_config.dtype = "auto"

        result = usage_lib._extract_trtllm_config(mock)
        assert result["backend"] == "pytorch"
        assert result["tensor_parallel_size"] == 4
        assert result["pipeline_parallel_size"] == 2
        assert result["context_parallel_size"] == 1
        assert result["moe_expert_parallel_size"] == 8
        assert result["moe_tensor_parallel_size"] == 2
        assert result["dtype"] == "float16"
        assert result["quantization_algo"] == "fp8"
        assert result["kv_cache_dtype"] == "auto"

    def test_extract_config_moe_sentinel_mapped_to_zero(self):
        """MoE parallel sizes of -1 (auto) are mapped to 0 for telemetry."""
        mock = MagicMock()
        mock.backend = "pytorch"
        mock.parallel_config.tp_size = 4
        mock.parallel_config.pp_size = 1
        mock.parallel_config.cp_size = 1
        mock.parallel_config.moe_ep_size = -1
        mock.parallel_config.moe_tp_size = -1
        mock.dtype = "float16"
        mock.quant_config.quant_algo = None
        mock.kv_cache_config.dtype = None

        result = usage_lib._extract_trtllm_config(mock)
        assert result["moe_expert_parallel_size"] == 0
        assert result["moe_tensor_parallel_size"] == 0

    def test_extract_config_none(self):
        """Returns empty dict when llm_args is None."""
        assert usage_lib._extract_trtllm_config(None) == {}

    def test_extract_config_partial(self):
        """Extracts available fields without crashing on missing ones."""

        class PartialArgs:
            backend = "tensorrt"

        result = usage_lib._extract_trtllm_config(PartialArgs())
        assert result["backend"] == "tensorrt"

    def test_extract_config_defaults_for_missing(self):
        """Missing optional configs are omitted from the result dict."""
        mock = MagicMock()
        mock.backend = "pytorch"
        mock.parallel_config.tp_size = 4
        mock.parallel_config.pp_size = 1
        mock.parallel_config.cp_size = 1
        mock.parallel_config.moe_ep_size = None
        mock.parallel_config.moe_tp_size = None
        mock.dtype = None
        mock.quant_config.quant_algo = None
        mock.kv_cache_config.dtype = None

        result = usage_lib._extract_trtllm_config(mock)
        assert "moe_expert_parallel_size" not in result
        assert "moe_tensor_parallel_size" not in result
        assert "dtype" not in result
        assert "quantization_algo" not in result
        assert "kv_cache_dtype" not in result

    def test_extract_config_infers_backend_from_class_name(self):
        """Backend inferred as 'tensorrt' when backend missing and class name contains 'TrtLlm'."""

        class TrtLlmArgsLike:
            pass  # no backend attr -> triggers cls_name inference

        result = usage_lib._extract_trtllm_config(TrtLlmArgsLike())
        assert result.get("backend") == "tensorrt"

    def test_extract_config_no_backend_no_trtllm_in_name(self):
        """Backend omitted when class name does not contain 'TrtLlm'."""

        class GenericArgs:
            pass

        result = usage_lib._extract_trtllm_config(GenericArgs())
        assert "backend" not in result


# ---------------------------------------------------------------------------
# _clamp_str truncation helper tests
# ---------------------------------------------------------------------------


class TestClampStr:
    """Tests for _clamp_str truncation helper."""

    def test_truncates_long_value(self):
        """Strings exceeding max_len are truncated."""
        assert usage_lib._clamp_str("a" * 200, 128) == "a" * 128

    def test_preserves_short_value(self):
        """Strings shorter than max_len are unchanged."""
        assert usage_lib._clamp_str("short", 128) == "short"

    def test_exact_boundary_not_truncated(self):
        """String exactly at max_len is not truncated."""
        assert usage_lib._clamp_str("a" * 128, 128) == "a" * 128

    def test_one_over_boundary_truncated(self):
        """String one char over max_len is truncated."""
        assert usage_lib._clamp_str("a" * 129, 128) == "a" * 128


# ---------------------------------------------------------------------------
# Feature flag extraction tests
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    """Tests for _collect_features() -- extracting feature flags from llm_args."""

    # --- All keys always present ---

    def test_all_keys_always_present(self):
        """Every expected key is present regardless of config values."""
        mock = MagicMock()
        result = json.loads(usage_lib._collect_features(mock))
        expected_keys = set(usage_lib._FEATURES_DEFAULTS.keys())
        assert set(result.keys()) == expected_keys

    # --- Default / None scenarios ---

    def test_all_features_disabled_defaults(self):
        """All features default to false/1 when llm_args has no feature configs."""
        mock = MagicMock(spec=[])  # No attributes
        result = json.loads(usage_lib._collect_features(mock))
        assert result == {
            "lora": False,
            "speculative_decoding": False,
            "prefix_caching": False,
            "cuda_graphs": False,
            "chunked_context": False,
            "data_parallel_size": 1,
            "checkpoint_format": "HF",
            "load_format": "AUTO",
        }

    def test_none_llm_args_returns_defaults(self):
        """None llm_args returns all defaults."""
        result = json.loads(usage_lib._collect_features(None))
        assert result == dict(usage_lib._FEATURES_DEFAULTS)

    def test_exception_in_extraction_returns_partial(self):
        """If extraction raises, returns whatever was collected (fail-silent)."""
        mock = MagicMock()
        mock.enable_lora = True
        mock.lora_config = None
        # Make speculative_config raise AttributeError (caught by narrowed handler)
        type(mock).speculative_config = property(
            lambda self: (_ for _ in ()).throw(AttributeError("boom"))
        )
        result = json.loads(usage_lib._collect_features(mock))
        # lora should be collected before the exception
        assert result["lora"] is True
        # All keys must still be present
        assert set(result.keys()) == set(usage_lib._FEATURES_DEFAULTS.keys())

    # --- LoRA ---

    def test_lora_enabled_via_lora_config(self):
        """LoRA detected when lora_config is not None."""
        mock = MagicMock()
        mock.enable_lora = False
        mock.lora_config = MagicMock()  # non-None
        result = json.loads(usage_lib._collect_features(mock))
        assert result["lora"] is True

    def test_lora_enabled_via_enable_lora_flag(self):
        """LoRA detected when enable_lora is True."""
        mock = MagicMock()
        mock.enable_lora = True
        mock.lora_config = None
        result = json.loads(usage_lib._collect_features(mock))
        assert result["lora"] is True

    def test_lora_both_signals(self):
        """LoRA detected when both enable_lora and lora_config are set."""
        mock = MagicMock()
        mock.enable_lora = True
        mock.lora_config = MagicMock()
        result = json.loads(usage_lib._collect_features(mock))
        assert result["lora"] is True

    def test_lora_disabled(self):
        """LoRA is false when neither signal is set."""
        mock = MagicMock()
        mock.enable_lora = False
        mock.lora_config = None
        result = json.loads(usage_lib._collect_features(mock))
        assert result["lora"] is False

    # --- Speculative decoding ---

    def test_speculative_decoding_enabled(self):
        """Speculative decoding detected when speculative_config is not None."""
        mock = MagicMock()
        mock.speculative_config = MagicMock()
        result = json.loads(usage_lib._collect_features(mock))
        assert result["speculative_decoding"] is True

    def test_speculative_decoding_none(self):
        """Speculative decoding is false when speculative_config is None."""
        mock = MagicMock()
        mock.speculative_config = None
        result = json.loads(usage_lib._collect_features(mock))
        assert result["speculative_decoding"] is False

    # --- Prefix caching ---

    def test_prefix_caching_enabled(self):
        """Prefix caching detected when enable_block_reuse is True."""
        mock = MagicMock()
        mock.kv_cache_config.enable_block_reuse = True
        result = json.loads(usage_lib._collect_features(mock))
        assert result["prefix_caching"] is True

    def test_prefix_caching_disabled(self):
        """Prefix caching is false when enable_block_reuse is False."""
        mock = MagicMock()
        mock.kv_cache_config.enable_block_reuse = False
        result = json.loads(usage_lib._collect_features(mock))
        assert result["prefix_caching"] is False

    def test_prefix_caching_no_kv_config(self):
        """Prefix caching defaults to false when kv_cache_config is None."""
        mock = MagicMock()
        mock.kv_cache_config = None
        result = json.loads(usage_lib._collect_features(mock))
        assert result["prefix_caching"] is False

    # --- CUDA graphs ---

    def test_cuda_graphs_pytorch_backend(self):
        """CUDA graphs detected via cuda_graph_config (PyTorch backend)."""
        mock = MagicMock()
        mock.cuda_graph_config = MagicMock()  # non-None = enabled
        mock.extended_runtime_perf_knob_config = None
        result = json.loads(usage_lib._collect_features(mock))
        assert result["cuda_graphs"] is True

    def test_cuda_graphs_pytorch_disabled(self):
        """CUDA graphs false when cuda_graph_config is None (PyTorch)."""
        mock = MagicMock()
        mock.cuda_graph_config = None
        mock.extended_runtime_perf_knob_config = None
        result = json.loads(usage_lib._collect_features(mock))
        assert result["cuda_graphs"] is False

    def test_cuda_graphs_trt_backend(self):
        """CUDA graphs detected via extended_runtime_perf_knob_config (TRT)."""
        mock = MagicMock()
        mock.cuda_graph_config = None
        mock.extended_runtime_perf_knob_config.cuda_graph_mode = True
        result = json.loads(usage_lib._collect_features(mock))
        assert result["cuda_graphs"] is True

    def test_cuda_graphs_trt_disabled(self):
        """CUDA graphs false when cuda_graph_mode is False (TRT)."""
        mock = MagicMock()
        mock.cuda_graph_config = None
        mock.extended_runtime_perf_knob_config.cuda_graph_mode = False
        result = json.loads(usage_lib._collect_features(mock))
        assert result["cuda_graphs"] is False

    def test_cuda_graphs_no_config_either_backend(self):
        """CUDA graphs false when neither backend config is present."""
        mock = MagicMock()
        mock.cuda_graph_config = None
        mock.extended_runtime_perf_knob_config = None
        result = json.loads(usage_lib._collect_features(mock))
        assert result["cuda_graphs"] is False

    # --- Chunked context ---

    def test_chunked_context_enabled(self):
        """Chunked context detected when enable_chunked_prefill is True."""
        mock = MagicMock()
        mock.enable_chunked_prefill = True
        result = json.loads(usage_lib._collect_features(mock))
        assert result["chunked_context"] is True

    def test_chunked_context_disabled(self):
        """Chunked context is false when enable_chunked_prefill is False."""
        mock = MagicMock()
        mock.enable_chunked_prefill = False
        result = json.loads(usage_lib._collect_features(mock))
        assert result["chunked_context"] is False

    # --- Data parallel size ---

    def test_data_parallel_size_with_attention_dp(self):
        """dp_size = tp_size when enable_attention_dp is True."""
        mock = MagicMock()
        mock.parallel_config.enable_attention_dp = True
        mock.parallel_config.tp_size = 4
        result = json.loads(usage_lib._collect_features(mock))
        assert result["data_parallel_size"] == 4

    def test_data_parallel_size_without_attention_dp(self):
        """dp_size = 1 when enable_attention_dp is False."""
        mock = MagicMock()
        mock.parallel_config.enable_attention_dp = False
        mock.parallel_config.tp_size = 4
        result = json.loads(usage_lib._collect_features(mock))
        assert result["data_parallel_size"] == 1

    def test_data_parallel_size_no_parallel_config(self):
        """dp_size defaults to 1 when parallel_config is None."""
        mock = MagicMock()
        mock.parallel_config = None
        result = json.loads(usage_lib._collect_features(mock))
        assert result["data_parallel_size"] == 1

    # --- All features enabled ---

    def test_all_features_enabled(self):
        """All features active simultaneously."""
        mock = MagicMock()
        mock.enable_lora = True
        mock.lora_config = MagicMock()
        mock.speculative_config = MagicMock()
        mock.kv_cache_config.enable_block_reuse = True
        mock.cuda_graph_config = MagicMock()
        mock.extended_runtime_perf_knob_config = None
        mock.enable_chunked_prefill = True
        mock.parallel_config.enable_attention_dp = True
        mock.parallel_config.tp_size = 8
        result = json.loads(usage_lib._collect_features(mock))
        assert result == {
            "lora": True,
            "speculative_decoding": True,
            "prefix_caching": True,
            "cuda_graphs": True,
            "chunked_context": True,
            "data_parallel_size": 8,
            "checkpoint_format": "HF",
            "load_format": "AUTO",
        }

    # --- Checkpoint/load axes ---

    def test_checkpoint_and_load_format_defaults(self):
        """Checkpoint/load formats default to effective HF/AUTO baseline."""
        mock = MagicMock(spec=[])
        result = json.loads(usage_lib._collect_features(mock))
        assert result["checkpoint_format"] == "HF"
        assert result["load_format"] == "AUTO"

    def test_checkpoint_and_load_format_explicit_strings(self):
        """Checkpoint/load format string values are captured directly."""
        mock = MagicMock()
        mock.checkpoint_format = "MX"
        mock.load_format = "GMS"
        result = json.loads(usage_lib._collect_features(mock))
        assert result["checkpoint_format"] == "MX"
        assert result["load_format"] == "GMS"

    def test_checkpoint_and_load_format_enums(self):
        """Enum-like objects are captured via their ``name`` attribute."""
        mock = MagicMock()
        mock.checkpoint_format = None
        mock.load_format.name = "AUTO"
        result = json.loads(usage_lib._collect_features(mock))
        assert result["checkpoint_format"] == "HF"
        assert result["load_format"] == "AUTO"

    # --- Output format ---

    def test_output_is_valid_json(self):
        """Output is valid JSON."""
        mock = MagicMock()
        result_str = usage_lib._collect_features(mock)
        parsed = json.loads(result_str)
        assert isinstance(parsed, dict)

    def test_output_uses_compact_separators(self):
        """Output uses compact separators (no spaces after : or ,)."""
        mock = MagicMock()
        result_str = usage_lib._collect_features(mock)
        assert ": " not in result_str
        assert ", " not in result_str

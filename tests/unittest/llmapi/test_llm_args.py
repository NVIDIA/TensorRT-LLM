import math
import tempfile
from collections import defaultdict
from dataclasses import is_dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, get_args, get_origin

import pydantic_core
import pytest
import torch
import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError
from utils.llm_data import llm_models_root
from utils.util import force_ampere

import tensorrt_llm.bindings.executor as tle
import tensorrt_llm.llmapi.llm_args as llm_args_mod
from tensorrt_llm import LLM as TorchLLM
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm._torch.auto_deploy.llm_args import \
    LlmArgs as AutoDeployLlmArgs
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_llama import LlamaForCausalLM
from tensorrt_llm._torch.virtual_memory import RestoreMode
from tensorrt_llm.builder import LoraConfig
from tensorrt_llm.commands.serve import get_llm_args, is_non_default_or_required
from tensorrt_llm.llmapi import (BuildConfig, CapacitySchedulerPolicy,
                                 SchedulerConfig)
# fmt: off
from tensorrt_llm.llmapi.llm_args import (BaseLlmArgs, CacheTransceiverConfig,
                                          CalibConfig, ContextChunkingPolicy,
                                          CudaGraphConfig,
                                          DecodeCudaGraphConfig,
                                          DecodingBaseConfig,
                                          DeepSeekV4SparseAttentionConfig,
                                          DynamicBatchConfig,
                                          Eagle3DecodingConfig,
                                          EagleDecodingConfig,
                                          EncodeCudaGraphConfig,
                                          ExecutorMemoryType,
                                          ExtendedRuntimePerfKnobConfig,
                                          KvCacheConfig,
                                          LookaheadDecodingConfig, MoeConfig,
                                          MTPDecodingConfig, MultimodalConfig,
                                          MultimodalEncoderCudaGraphConfig,
                                          PeftCacheConfig, PybindMirror,
                                          RayPlacementConfig,
                                          SkipSoftmaxAttentionConfig,
                                          SleepConfig, SpeculativeConfig,
                                          StrictBaseModel, TorchCompileConfig,
                                          TorchLlmArgs, TrtLlmArgs,
                                          UserProvidedDecodingConfig,
                                          update_llm_args_with_extra_dict)
# fmt: on
from tensorrt_llm.llmapi.llm_utils import (_resolve_kv_cache_manager_v2_auto,
                                           apply_model_defaults_to_llm_args)
from tensorrt_llm.llmapi.mm_encoder import MultimodalEncoder
from tensorrt_llm.llmapi.utils import print_traceback_on_error
from tensorrt_llm.models.modeling_utils import LayerQuantConfig, QuantConfig
from tensorrt_llm.plugin import PluginConfig

from .test_llm import llama_model_path


def test_LookaheadDecodingConfig():
    # from constructor
    config = LookaheadDecodingConfig(max_window_size=4,
                                     max_ngram_size=3,
                                     max_verification_set_size=4)
    assert config.max_window_size == 4
    assert config.max_ngram_size == 3
    assert config.max_verification_set_size == 4

    # from dict
    config = LookaheadDecodingConfig(**{
        "max_window_size": 4,
        "max_ngram_size": 3,
        "max_verification_set_size": 4
    })
    assert config.max_window_size == 4
    assert config.max_ngram_size == 3
    assert config.max_verification_set_size == 4

    # to pybind
    pybind_config = config._to_pybind()
    assert isinstance(pybind_config, tle.LookaheadDecodingConfig)
    assert pybind_config.max_window_size == 4
    assert pybind_config.max_ngram_size == 3
    assert pybind_config.max_verification_set_size == 4


def test_MTPDecodingConfig_default_draft_len_is_not_user_set():
    config = MTPDecodingConfig()

    # Unset max_draft_len stays None (the "use the model's
    # num_nextn_predict_layers" sentinel) until resolved at model load.
    assert config.max_draft_len is None
    assert config.max_total_draft_tokens is None
    assert "max_draft_len" not in config.model_fields_set

    explicit_config = MTPDecodingConfig(max_draft_len=1)
    assert explicit_config.max_draft_len == 1
    assert explicit_config.max_total_draft_tokens == 1
    assert "max_draft_len" in explicit_config.model_fields_set


class TestYaml:

    def _yaml_to_dict(self, yaml_content: str) -> dict:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(yaml_content.encode('utf-8'))
            f.flush()
            f.seek(0)
            dict_content = yaml.safe_load(f)
        return dict_content

    def test_llm_args_yaml_with_speculative_config(self):
        yaml_content = """
speculative_config:
    decoding_type: Lookahead
    max_window_size: 4
    max_ngram_size: 3
    """
        llm_args_dict = self._yaml_to_dict(yaml_content)

        llm_args = TrtLlmArgs(model=llama_model_path, **llm_args_dict)
        assert llm_args.speculative_config.max_window_size == 4
        assert llm_args.speculative_config.max_ngram_size == 3
        assert llm_args.speculative_config.max_verification_set_size == 4

    def test_llm_args_with_invalid_yaml(self):
        yaml_content = """
pytorch_backend_config: # this is deprecated
    max_num_tokens: 1
    max_seq_len: 1
"""
        llm_args_dict = self._yaml_to_dict(yaml_content)

        with pytest.raises(ValueError):
            llm_args = TrtLlmArgs(model=llama_model_path, **llm_args_dict)

    def test_llm_args_with_build_config(self):
        yaml_content = """
build_config:
    max_beam_width: 4
    max_batch_size: 8
    max_num_tokens: 256
    """
        llm_args_dict = self._yaml_to_dict(yaml_content)

        llm_args = TrtLlmArgs(model=llama_model_path, **llm_args_dict)
        assert llm_args.build_config.max_beam_width == 4
        assert llm_args.build_config.max_batch_size == 8
        assert llm_args.build_config.max_num_tokens == 256

    def test_llm_args_with_kvcache_config(self):
        yaml_content = """
kv_cache_config:
    enable_block_reuse: True
    max_tokens: 1024
    max_attention_window: [1024, 1024, 1024]
    """
        llm_args_dict = self._yaml_to_dict(yaml_content)

        llm_args = TrtLlmArgs(model=llama_model_path, **llm_args_dict)
        assert llm_args.kv_cache_config.enable_block_reuse == True
        assert llm_args.kv_cache_config.max_tokens == 1024
        assert llm_args.kv_cache_config.max_attention_window == [
            1024, 1024, 1024
        ]

    def test_llm_args_with_pydantic_options(self):
        yaml_content = """
max_batch_size: 16
max_num_tokens: 256
max_seq_len: 128
    """
        llm_args_dict = self._yaml_to_dict(yaml_content)

        llm_args = TrtLlmArgs(model=llama_model_path, **llm_args_dict)
        assert llm_args.max_batch_size == 16
        assert llm_args.max_num_tokens == 256
        assert llm_args.max_seq_len == 128

    @pytest.mark.parametrize("llm_args_cls", [TrtLlmArgs, TorchLlmArgs])
    def test_llm_args_with_model_kwargs(self, llm_args_cls):
        yaml_content = """
model_kwargs:
    num_hidden_layers: 2
    """
        llm_args_dict = self._yaml_to_dict(yaml_content)
        llm_args = llm_args_cls(model=llama_model_path, **llm_args_dict)
        assert llm_args.model_kwargs['num_hidden_layers'] == 2


@pytest.mark.parametrize("llm_args_cls", [TorchLlmArgs])
class TestEncoderRuntimeSizes:
    """Cover encoder runtime size fields and fallback to LLM limits.

    `encoder_max_batch_size` / `encoder_max_num_tokens` are user-facing
    knobs that size multimodal encoder AttentionMetadata; when unset they
    fall back to the LLM-side `max_batch_size` / `max_num_tokens`. They are
    PyTorch-backend only (the multimodal encoder profiling path), so they
    live on `TorchLlmArgs` rather than the shared `BaseLlmArgs`.
    """

    def test_defaults_are_none(self, llm_args_cls):
        llm_args = llm_args_cls(model=llama_model_path)
        assert llm_args.encoder_max_batch_size is None
        assert llm_args.encoder_max_num_tokens is None

    @pytest.mark.parametrize(
        "kwargs, expected_runtime_sizes",
        [
            # Neither encoder knob set -- falls back to LLM limits.
            (dict(max_batch_size=64, max_num_tokens=2048), (64, 2048)),
            # Only encoder_max_batch_size overrides.
            (dict(max_batch_size=64,
                  max_num_tokens=2048,
                  encoder_max_batch_size=512), (512, 2048)),
            # Only encoder_max_num_tokens overrides.
            (dict(max_batch_size=64,
                  max_num_tokens=2048,
                  encoder_max_num_tokens=32768), (64, 32768)),
            # Both encoder knobs override.
            (dict(max_batch_size=64,
                  max_num_tokens=2048,
                  encoder_max_batch_size=512,
                  encoder_max_num_tokens=32768), (512, 32768)),
        ],
        ids=["fallback", "only_batch", "only_tokens", "both"],
    )
    def test_get_encoder_runtime_sizes(self, llm_args_cls, kwargs,
                                       expected_runtime_sizes):
        llm_args = llm_args_cls(model=llama_model_path, **kwargs)
        assert llm_args.get_encoder_runtime_sizes() == expected_runtime_sizes

    @pytest.mark.parametrize(
        "field_name, invalid_value",
        [
            ("encoder_max_batch_size", 0),
            ("encoder_max_batch_size", -1),
            ("encoder_max_num_tokens", 0),
            ("encoder_max_num_tokens", -1),
        ],
    )
    def test_rejects_non_positive(self, llm_args_cls, field_name,
                                  invalid_value):
        with pytest.raises(ValidationError):
            llm_args_cls(model=llama_model_path, **{field_name: invalid_value})


def test_decoding_type_eagle3_parses_to_eagle3_decoding_config():
    adapter = TypeAdapter(SpeculativeConfig)
    spec_cfg = adapter.validate_python(
        dict(decoding_type="Eagle3",
             max_draft_len=3,
             speculative_model="/path/to/draft/model"))
    assert isinstance(spec_cfg, Eagle3DecodingConfig)


def test_decoding_type_eagle_warns_on_pytorch_backend(monkeypatch):
    warnings_seen: list[str] = []

    def _capture_warning(msg, *args, **kwargs):
        warnings_seen.append(str(msg))

    monkeypatch.setattr(llm_args_mod.logger, "warning", _capture_warning)

    spec_cfg = EagleDecodingConfig(decoding_type="Eagle",
                                   max_draft_len=3,
                                   speculative_model="/path/to/draft/model")

    TorchLlmArgs(model=llama_model_path, speculative_config=spec_cfg)

    assert any(
        "EAGLE (v1/v2) draft checkpoints are incompatible with Eagle3" in m
        for m in warnings_seen)


def test_decoding_type_eagle3_errors_on_tensorrt_backend():
    spec_cfg = Eagle3DecodingConfig(decoding_type="Eagle3",
                                    max_draft_len=3,
                                    speculative_model="/path/to/draft/model")
    with pytest.raises(ValueError,
                       match="only supported on the PyTorch backend"):
        TrtLlmArgs(model=llama_model_path, speculative_config=spec_cfg)


def test_post_processor_hook_rejected_with_skip_tokenizer_init():
    """post_processor_hook + skip_tokenizer_init must fail fast.

    The hook is a text-based guardrail; pairing it with skip_tokenizer_init (no
    detokenized text) must be rejected rather than silently disabling it.
    """
    with pytest.raises(ValidationError, match="skip_tokenizer_init"):
        TorchLlmArgs(model="/tmp/dummy_model",
                     skip_tokenizer_init=True,
                     post_processor_hook="my_pkg.guardrail.Hook")
    # skip_tokenizer_init alone (no hook) is still fine.
    TorchLlmArgs(model="/tmp/dummy_model", skip_tokenizer_init=True)


class TestModelDefaults:
    """Test suite for model-specific default overrides functionality."""

    def test_compute_applied_llm_defaults_simple_field(self):
        model_defaults = {"enable_chunked_prefill": False}
        llm_args = TorchLlmArgs(model="/tmp/dummy_model")
        applied = apply_model_defaults_to_llm_args(llm_args, model_defaults)
        assert applied == model_defaults

    @pytest.mark.parametrize("explicit_auto", [False, True])
    def test_kv_cache_manager_v2_auto_uses_model_default(self, explicit_auto):
        kv_cache_config = (KvCacheConfig(use_kv_cache_manager_v2="auto")
                           if explicit_auto else KvCacheConfig())
        llm_args = TorchLlmArgs(model="/tmp/dummy_model",
                                kv_cache_config=kv_cache_config)
        model_defaults = {"kv_cache_config": {"use_kv_cache_manager_v2": True}}

        apply_model_defaults_to_llm_args(llm_args, model_defaults)
        _resolve_kv_cache_manager_v2_auto(llm_args, model_defaults)

        assert llm_args.kv_cache_config.use_kv_cache_manager_v2 is True

    def test_kv_cache_manager_v2_auto_falls_back_to_false(self):
        llm_args = TorchLlmArgs(model="/tmp/dummy_model")

        _resolve_kv_cache_manager_v2_auto(llm_args, {})

        assert llm_args.kv_cache_config.use_kv_cache_manager_v2 is False

    @pytest.mark.parametrize("user_setting", [False, True])
    def test_kv_cache_manager_v2_explicit_value_overrides_model_default(
            self, user_setting):
        llm_args = TorchLlmArgs(
            model="/tmp/dummy_model",
            kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=user_setting))
        model_defaults = {
            "kv_cache_config": {
                "use_kv_cache_manager_v2": not user_setting
            }
        }

        apply_model_defaults_to_llm_args(llm_args, model_defaults)
        _resolve_kv_cache_manager_v2_auto(llm_args, model_defaults)

        assert llm_args.kv_cache_config.use_kv_cache_manager_v2 is user_setting

    @pytest.mark.parametrize(
        "defaults_dict,should_raise,error_contains",
        [
            # Invalid field name - this will definitely fail
            ({
                "invalid_field_that_does_not_exist": True
            }, True, "invalid_field_that_does_not_exist"),
            # Another invalid field name
            ({
                "non_existent_parameter": 123
            }, True, "non_existent_parameter"),
            # Invalid nested config field type (should be bool, not string)
            ({
                "kv_cache_config": {
                    "enable_block_reuse": "not_a_boolean"
                }
            }, True, "enable_block_reuse"),
            # Invalid nested config with extra fields (extra="forbid")
            ({
                "kv_cache_config": {
                    "enable_block_reuse": True,
                    "made_up_field": 999
                }
            }, True, "made_up_field"),
            # Valid simple field
            ({
                "enable_chunked_prefill": False
            }, False, None),
            # Valid nested config
            ({
                "kv_cache_config": {
                    "enable_block_reuse": False
                }
            }, False, None),
            # Valid with type coercion (Pydantic will convert string to bool)
            (
                {
                    "enable_chunked_prefill": "false"
                },  # String will be coerced to bool
                False,
                None),
        ])
    def test_model_defaults_validation(self, defaults_dict, should_raise,
                                       error_contains):
        # Use a dummy model path for testing (doesn't need to exist for validation)
        llm_args = TorchLlmArgs(model="/tmp/dummy_model_for_validation_test")

        if should_raise:
            # Should raise error with expected message when applying invalid defaults
            with pytest.raises(
                (ValueError, AttributeError, ValidationError)) as exc_info:
                apply_model_defaults_to_llm_args(llm_args, defaults_dict)
            assert error_contains in str(exc_info.value)
        else:
            # Should pass validation and apply successfully
            applied = apply_model_defaults_to_llm_args(llm_args, defaults_dict)

            # Check that the applied defaults match what we requested
            # Note: We check 'applied' dict, not llm_args directly, because
            # some fields may have validators that change values
            for key in defaults_dict:
                if key in applied:
                    # The field was applied (not overridden by user)
                    if isinstance(applied[key], dict):
                        # For nested configs, check they're in the applied dict
                        assert key in applied
                    else:
                        # For simple fields, check they're in the applied dict
                        assert key in applied

    def test_mock_model_with_invalid_defaults(self):
        """Test that a model class returning invalid defaults fails during application."""

        # Simulate a model that returns invalid defaults
        class ModelWithBadDefaults:

            @classmethod
            def get_model_defaults(cls, llm_args):
                return {
                    "kv_cache_config": {
                        "enable_block_reuse": "this_should_be_boolean",
                        "max_tokens": "not_a_number"
                    }
                }

        llm_args = TorchLlmArgs(model="/tmp/test")
        bad_defaults = ModelWithBadDefaults.get_model_defaults(llm_args)

        # This should raise ValidationError when trying to apply
        with pytest.raises(ValidationError) as exc_info:
            apply_model_defaults_to_llm_args(llm_args, bad_defaults)

        # Check that the error message is helpful
        error_str = str(exc_info.value)
        assert "enable_block_reuse" in error_str or "max_tokens" in error_str


def test_KvCacheConfig_declaration():
    assert KvCacheConfig().kv_cache_event_hash_algo == "auto"
    assert KvCacheConfig().block_reuse_policy == "all_reusable"
    assert KvCacheConfig().enable_swa_scratch_reuse is False
    assert KvCacheConfig().use_kv_cache_manager_v2 == "auto"
    assert KvCacheConfig(
        use_kv_cache_manager_v2=True).use_kv_cache_manager_v2 is True
    assert KvCacheConfig(
        use_kv_cache_manager_v2=False).use_kv_cache_manager_v2 is False
    with pytest.raises(ValidationError, match="use_kv_cache_manager_v2"):
        KvCacheConfig(use_kv_cache_manager_v2="invalid")

    config = KvCacheConfig(enable_block_reuse=True,
                           max_tokens=1024,
                           max_attention_window=[1024, 1024, 1024],
                           free_gpu_memory_fraction=0.5,
                           host_cache_size=1024,
                           disk_cache_size=2048,
                           disk_cache_path="/tmp",
                           cross_kv_cache_fraction=0.5,
                           secondary_offload_min_priority=1,
                           event_buffer_max_size=0,
                           kv_cache_event_hash_algo="v2_sha256_64",
                           enable_swa_scratch_reuse=True,
                           enable_partial_reuse=True,
                           copy_on_partial_reuse=True,
                           pool_ratio=[0.25, 0.75],
                           avg_seq_len=2048,
                           block_reuse_policy="per_request",
                           attention_dp_events_gather_period_ms=10)

    pybind_config = config._to_pybind()
    assert pybind_config.enable_block_reuse == True
    assert pybind_config.max_tokens == 1024
    assert pybind_config.max_attention_window == [1024, 1024, 1024]
    assert pybind_config.free_gpu_memory_fraction == 0.5
    assert pybind_config.host_cache_size == 1024
    assert config.disk_cache_size == 2048
    assert config.disk_cache_path == "/tmp"
    assert config.enable_swa_scratch_reuse is True
    assert KvCacheConfig().enable_swa_scratch_reuse is False
    assert pybind_config.cross_kv_cache_fraction == 0.5
    assert pybind_config.secondary_offload_min_priority == 1
    assert pybind_config.event_buffer_max_size == 0
    assert config.kv_cache_event_hash_algo == "v2_sha256_64"
    assert config.pool_ratio == [0.25, 0.75]
    assert config.avg_seq_len == 2048
    assert config.block_reuse_policy == "per_request"
    assert not hasattr(pybind_config, "pool_ratio")
    assert not hasattr(pybind_config, "avg_seq_len")
    assert not hasattr(pybind_config, "block_reuse_policy")
    assert not hasattr(pybind_config, "enable_swa_scratch_reuse")
    assert KvCacheConfig(
        kv_cache_event_hash_algo="auto").kv_cache_event_hash_algo == "auto"
    assert KvCacheConfig(kv_cache_event_hash_algo="v1_block_key"
                         ).kv_cache_event_hash_algo == "v1_block_key"
    assert pybind_config.enable_partial_reuse == True
    assert pybind_config.copy_on_partial_reuse == True
    assert pybind_config.attention_dp_events_gather_period_ms == 10
    with pytest.raises(ValidationError):
        KvCacheConfig(block_reuse_policy="invalid")


def test_KvCacheConfig_disk_cache_validation(tmp_path):
    config = KvCacheConfig(disk_cache_size=2048, disk_cache_path=str(tmp_path))

    assert config.disk_cache_size == 2048
    assert config.disk_cache_path == str(tmp_path)

    with pytest.raises(ValidationError) as exc_info:
        KvCacheConfig(disk_cache_size=2048)
    assert "disk_cache_path" in str(exc_info.value)


class TestMultimodalEncoderCudaGraphConfig:

    def test_minimal_required_fields(self):
        config = MultimodalEncoderCudaGraphConfig(buckets=[(1035, 1)])
        assert config.buckets == [(1035, 1)]
        assert config.enable_padding is True
        assert config.warmup_steps == 2
        assert config.enable_replay_stats is False

        config = MultimodalEncoderCudaGraphConfig(buckets=[(1035, 1)],
                                                  enable_replay_stats=True)
        assert config.enable_replay_stats is True

    def test_explicit_buckets_deduped_and_sorted(self):
        config = MultimodalEncoderCudaGraphConfig(buckets=[(2069,
                                                            2), (1035,
                                                                 1), (2069, 2)])
        assert config.buckets == [(1035, 1), (2069, 2)]

    def test_explicit_buckets_accept_yaml_style_lists(self):
        config = MultimodalEncoderCudaGraphConfig(
            buckets=[[2069, 2], [1035, 1]])
        assert config.buckets == [(1035, 1), (2069, 2)]

    @pytest.mark.parametrize("kwargs", [{
        "buckets": []
    }, {}],
                             ids=["empty", "missing"])
    def test_rejects_absent_buckets(self, kwargs):
        with pytest.raises(ValidationError):
            MultimodalEncoderCudaGraphConfig(**kwargs)

    @pytest.mark.parametrize("buckets", [[(0, 1)], [(1, -2)], [(3, 0)]])
    def test_rejects_non_positive_buckets(self, buckets):
        with pytest.raises(ValidationError):
            MultimodalEncoderCudaGraphConfig(buckets=buckets)

    def test_rejects_buckets_with_too_few_tokens(self):
        with pytest.raises(ValidationError):
            MultimodalEncoderCudaGraphConfig(buckets=[(1, 2)])


class TestMultimodalConfig:

    def test_default_encoder_cuda_graph_is_none(self):
        assert MultimodalConfig().encoder_cuda_graph is None
        assert MultimodalConfig().encoder_side_stream_max_ahead == 0
        assert MultimodalConfig().video_pruning_rate is None

    def test_torch_llm_args_default_multimodal_config(self):
        args = TorchLlmArgs(model=llama_model_path)
        assert isinstance(args.multimodal_config, MultimodalConfig)
        assert args.multimodal_config.encoder_cuda_graph is None
        assert args.multimodal_config.encoder_side_stream_max_ahead == 0
        assert args.multimodal_config.video_pruning_rate is None

    def test_torch_llm_args_with_encoder_side_stream_max_ahead(self):
        args = TorchLlmArgs(
            model=llama_model_path,
            multimodal_config=MultimodalConfig(encoder_side_stream_max_ahead=2))
        assert args.multimodal_config.encoder_side_stream_max_ahead == 2

    def test_torch_llm_args_with_multimodal_video_pruning_rate(self):
        args = TorchLlmArgs(
            model=llama_model_path,
            multimodal_config=MultimodalConfig(video_pruning_rate=0.5))
        assert args.multimodal_config.video_pruning_rate == 0.5

    def test_torch_llm_args_with_encoder_cuda_graph_buckets(self):
        args = TorchLlmArgs(
            model=llama_model_path,
            multimodal_config=MultimodalConfig(
                encoder_cuda_graph={
                    "vision":
                    MultimodalEncoderCudaGraphConfig(buckets=[(1035,
                                                               1), (2069, 2)])
                }))
        encoder_graph = args.multimodal_config.encoder_cuda_graph
        assert encoder_graph["vision"].buckets == [(1035, 1), (2069, 2)]

    def test_torch_llm_args_with_encoder_cuda_graph_buckets_yaml(self):
        args = TorchLlmArgs(
            model=llama_model_path,
            multimodal_config={
                "encoder_cuda_graph": {
                    "vision": {
                        "buckets": [[2069, 2], [1035, 1]]
                    }
                }
            },
        )
        encoder_graph = args.multimodal_config.encoder_cuda_graph
        assert encoder_graph["vision"].buckets == [(1035, 1), (2069, 2)]

    def test_torch_llm_args_with_encoder_side_stream_max_ahead_yaml(self):
        args = TorchLlmArgs(
            model=llama_model_path,
            multimodal_config={
                "encoder_side_stream_max_ahead": 2,
                "video_pruning_rate": 0.5,
            },
        )
        assert args.multimodal_config.encoder_side_stream_max_ahead == 2
        assert args.multimodal_config.video_pruning_rate == 0.5

    def test_encoder_cuda_graph_and_side_stream_max_ahead_are_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            TorchLlmArgs(
                model=llama_model_path,
                multimodal_config={
                    "encoder_cuda_graph": {
                        "vision": {
                            "buckets": [[1035, 1]]
                        }
                    },
                    "encoder_side_stream_max_ahead": 1,
                },
            )


@pytest.mark.parametrize("kwargs", [
    {
        "pool_ratio": []
    },
    {
        "pool_ratio": [0.0, 1.0]
    },
    {
        "pool_ratio": [0.25, 0.25]
    },
    {
        "avg_seq_len": 0
    },
])
def test_KvCacheConfig_pool_ratio_avg_seq_len_validation(kwargs):
    with pytest.raises(ValidationError):
        KvCacheConfig(**kwargs)


def test_CapacitySchedulerPolicy():
    val = CapacitySchedulerPolicy.MAX_UTILIZATION
    assert PybindMirror.maybe_to_pybind(
        val) == tle.CapacitySchedulerPolicy.MAX_UTILIZATION


def test_ContextChunkingPolicy():
    val = ContextChunkingPolicy.EQUAL_PROGRESS
    assert PybindMirror.maybe_to_pybind(
        val) == tle.ContextChunkingPolicy.EQUAL_PROGRESS


def test_SleepConfig_restore_modes_normalized_from_dict():
    sleep_config = SleepConfig(
        restore_modes={
            ExecutorMemoryType.KV_CACHE.value: "NONE",
            ExecutorMemoryType.MODEL_WEIGHTS_MAIN.value: "CPU",
        })

    assert sleep_config.restore_modes[
        ExecutorMemoryType.KV_CACHE] == RestoreMode.NONE
    assert sleep_config.restore_modes[
        ExecutorMemoryType.MODEL_WEIGHTS_MAIN] == RestoreMode.CPU
    assert isinstance(sleep_config.restore_modes[ExecutorMemoryType.SAMPLER],
                      RestoreMode)


def test_SleepConfig_restore_modes_normalized_from_defaultdict():
    sleep_config = SleepConfig(restore_modes=defaultdict(
        lambda: RestoreMode.CPU, {
            ExecutorMemoryType.KV_CACHE: RestoreMode.NONE,
            ExecutorMemoryType.MODEL_WEIGHTS_MAIN: "PINNED",
        }))

    assert sleep_config.restore_modes[
        ExecutorMemoryType.KV_CACHE] == RestoreMode.NONE
    assert sleep_config.restore_modes[
        ExecutorMemoryType.MODEL_WEIGHTS_MAIN] == RestoreMode.PINNED
    assert sleep_config.restore_modes[
        ExecutorMemoryType.SAMPLER] == RestoreMode.CPU


@force_ampere
def test_SleepConfig_is_picklable():
    """SleepConfig with default construction must survive a pickle round-trip.

    MPI worker initialisation serialises llm_args (including SleepConfig) via
    pickle to distribute configuration to each rank.  The defaultdict inside
    restore_modes previously used a closure lambda as its default_factory, which
    is not picklable.  This test catches any regression to that pattern.
    """
    import pickle

    cfg_default = SleepConfig()
    rt = pickle.loads(pickle.dumps(cfg_default))  # noqa: S301
    assert rt.restore_modes == cfg_default.restore_modes


@force_ampere
def test_SleepConfig_pickle_custom_restore_modes_roundtrip():
    """SleepConfig with explicit per-key overrides must survive a pickle round-trip."""
    import pickle

    cfg_custom = SleepConfig(
        restore_modes={
            ExecutorMemoryType.KV_CACHE.value: "NONE",
            ExecutorMemoryType.MODEL_WEIGHTS_MAIN.value: "CPU",
        })
    rt_custom = pickle.loads(pickle.dumps(cfg_custom))  # noqa: S301
    assert rt_custom.restore_modes[
        ExecutorMemoryType.KV_CACHE] == RestoreMode.NONE
    assert rt_custom.restore_modes[
        ExecutorMemoryType.MODEL_WEIGHTS_MAIN] == RestoreMode.CPU


@force_ampere
def test_SleepConfig_pickle_defaultfactory_survives_roundtrip():
    """The defaultdict default_factory must remain functional after pickle.

    Missing keys should return a valid RestoreMode rather than raising
    KeyError, proving the factory (not just the already-present entries)
    was serialised correctly.
    """
    import pickle

    cfg_default = SleepConfig()
    rt = pickle.loads(pickle.dumps(cfg_default))  # noqa: S301

    missing_key = ExecutorMemoryType.SAMPLER
    assert isinstance(rt.restore_modes[missing_key], RestoreMode)
    assert rt.restore_modes[missing_key] == cfg_default.restore_modes[
        missing_key]


def test_DynamicBatchConfig_declaration():
    config = DynamicBatchConfig(enable_batch_size_tuning=True,
                                enable_max_num_tokens_tuning=True,
                                dynamic_batch_moving_average_window=10)

    pybind_config = PybindMirror.maybe_to_pybind(config)

    assert pybind_config.enable_batch_size_tuning == True
    assert pybind_config.enable_max_num_tokens_tuning == True
    assert pybind_config.dynamic_batch_moving_average_window == 10


def test_SchedulerConfig_declaration() -> None:
    default_config = SchedulerConfig()
    default_pybind_config = PybindMirror.maybe_to_pybind(default_config)
    assert default_config.enable_prefix_aware_scheduling is True
    assert default_pybind_config.enable_prefix_aware_scheduling is True

    config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        context_chunking_policy=ContextChunkingPolicy.EQUAL_PROGRESS,
        dynamic_batch_config=DynamicBatchConfig(
            enable_batch_size_tuning=True,
            enable_max_num_tokens_tuning=True,
            dynamic_batch_moving_average_window=10),
        enable_prefix_aware_scheduling=False)

    pybind_config = PybindMirror.maybe_to_pybind(config)
    assert pybind_config.capacity_scheduler_policy == tle.CapacitySchedulerPolicy.MAX_UTILIZATION
    assert pybind_config.context_chunking_policy == tle.ContextChunkingPolicy.EQUAL_PROGRESS
    assert PybindMirror.pybind_equals(pybind_config.dynamic_batch_config,
                                      config.dynamic_batch_config._to_pybind())
    assert pybind_config.enable_prefix_aware_scheduling is False


def test_PeftCacheConfig_declaration():
    config = PeftCacheConfig(num_host_module_layer=1,
                             num_device_module_layer=1,
                             optimal_adapter_size=64,
                             max_adapter_size=128,
                             num_put_workers=1,
                             num_ensure_workers=1,
                             num_copy_streams=1,
                             max_pages_per_block_host=24,
                             max_pages_per_block_device=8,
                             device_cache_percent=0.5,
                             host_cache_size=1024,
                             lora_prefetch_dir=".")

    pybind_config = PybindMirror.maybe_to_pybind(config)
    assert pybind_config.num_host_module_layer == 1
    assert pybind_config.num_device_module_layer == 1
    assert pybind_config.optimal_adapter_size == 64
    assert pybind_config.max_adapter_size == 128
    assert pybind_config.num_put_workers == 1
    assert pybind_config.num_ensure_workers == 1
    assert pybind_config.num_copy_streams == 1
    assert pybind_config.max_pages_per_block_host == 24
    assert pybind_config.max_pages_per_block_device == 8
    assert pybind_config.device_cache_percent == 0.5
    assert pybind_config.host_cache_size == 1024
    assert pybind_config.lora_prefetch_dir == "."


def test_PeftCacheConfig_from_pybind():
    pybind_config = tle.PeftCacheConfig(num_host_module_layer=1,
                                        num_device_module_layer=1,
                                        optimal_adapter_size=64,
                                        max_adapter_size=128,
                                        num_put_workers=1,
                                        num_ensure_workers=1,
                                        num_copy_streams=1,
                                        max_pages_per_block_host=24,
                                        max_pages_per_block_device=8,
                                        device_cache_percent=0.5,
                                        host_cache_size=1024,
                                        lora_prefetch_dir=".")

    config = PeftCacheConfig.from_pybind(pybind_config)
    assert config.num_host_module_layer == 1
    assert config.num_device_module_layer == 1
    assert config.optimal_adapter_size == 64
    assert config.max_adapter_size == 128
    assert config.num_put_workers == 1
    assert config.num_ensure_workers == 1
    assert config.num_copy_streams == 1
    assert config.max_pages_per_block_host == 24
    assert config.max_pages_per_block_device == 8
    assert config.device_cache_percent == 0.5
    assert config.host_cache_size == 1024
    assert config.lora_prefetch_dir == "."


def test_PeftCacheConfig_from_pybind_gets_python_only_default_values_when_none(
):
    pybind_config = tle.PeftCacheConfig(num_host_module_layer=1,
                                        num_device_module_layer=1,
                                        optimal_adapter_size=64,
                                        max_adapter_size=128,
                                        num_put_workers=1,
                                        num_ensure_workers=1,
                                        num_copy_streams=1,
                                        max_pages_per_block_host=24,
                                        max_pages_per_block_device=8,
                                        device_cache_percent=None,
                                        host_cache_size=None,
                                        lora_prefetch_dir=".")

    config = PeftCacheConfig.from_pybind(pybind_config)
    assert config.num_host_module_layer == 1
    assert config.num_device_module_layer == 1
    assert config.optimal_adapter_size == 64
    assert config.max_adapter_size == 128
    assert config.num_put_workers == 1
    assert config.num_ensure_workers == 1
    assert config.num_copy_streams == 1
    assert config.max_pages_per_block_host == 24
    assert config.max_pages_per_block_device == 8
    assert config.device_cache_percent == PeftCacheConfig.model_fields[
        "device_cache_percent"].default
    assert config.host_cache_size == PeftCacheConfig.model_fields[
        "host_cache_size"].default
    assert config.lora_prefetch_dir == "."


def test_update_llm_args_with_extra_dict_with_nested_dict():
    llm_api_args_dict = {
        "model":
        "dummy-model",
        "build_config":
        None,  # Will override later.
        "extended_runtime_perf_knob_config":
        ExtendedRuntimePerfKnobConfig(multi_block_mode=True),
        "kv_cache_config":
        KvCacheConfig(enable_block_reuse=False),
        "peft_cache_config":
        PeftCacheConfig(num_host_module_layer=0),
        "scheduler_config":
        SchedulerConfig(capacity_scheduler_policy=CapacitySchedulerPolicy.
                        GUARANTEED_NO_EVICT)
    }
    plugin_config = PluginConfig(dtype='float16', nccl_plugin=None)
    build_config = BuildConfig(max_input_len=1024,
                               lora_config=LoraConfig(lora_ckpt_source='hf'),
                               plugin_config=plugin_config)
    extra_llm_args_dict = {
        "build_config": build_config.model_dump(mode="json"),
    }

    llm_api_args_dict = update_llm_args_with_extra_dict(llm_api_args_dict,
                                                        extra_llm_args_dict,
                                                        "build_config")
    initialized_llm_args = TrtLlmArgs(**llm_api_args_dict)

    def check_nested_dict_equality(dict1, dict2, path=""):
        if not isinstance(dict1, dict) or not isinstance(dict2, dict):
            if dict1 != dict2:
                raise ValueError(f"Mismatch at {path}: {dict1} != {dict2}")
            return True
        if dict1.keys() != dict2.keys():
            raise ValueError(f"Different keys at {path}:")
        for key in dict1:
            new_path = f"{path}.{key}" if path else key
            if not check_nested_dict_equality(dict1[key], dict2[key], new_path):
                raise ValueError(f"Mismatch at {path}: {dict1} != {dict2}")
        return True

    build_config_dict1 = build_config.model_dump(mode="json")
    build_config_dict2 = initialized_llm_args.build_config.model_dump(
        mode="json")
    check_nested_dict_equality(build_config_dict1, build_config_dict2)


class TestTelemetryConfigPrecedence:
    """Telemetry-config precedence in the merge helper.

    Two modes are exercised:
    - `explicit_cli_keys is None` (legacy / programmatic): YAML wins on
      conflicts; `usage_context` carve-out still applies.
    - `explicit_cli_keys` provided (CLI mode): explicit keys win on
      conflicts; see `TestExplicitCliKeysPrecedence` for that path.
    """

    def test_default_telemetry_config_preserved_when_no_yaml(self):
        """Default telemetry_config survives YAML merge when YAML has none."""
        from tensorrt_llm.usage.config import TelemetryConfig, UsageContext
        base = {
            "model":
            "dummy",
            "telemetry_config":
            TelemetryConfig(disabled=False,
                            usage_context=UsageContext.CLI_SERVE),
        }
        yaml_dict = {"max_batch_size": 8}
        merged = update_llm_args_with_extra_dict(base, yaml_dict)
        tc = merged["telemetry_config"]
        assert isinstance(tc, TelemetryConfig)
        assert tc.disabled is False
        assert tc.usage_context == UsageContext.CLI_SERVE

    def test_yaml_can_override_disabled(self):
        """YAML telemetry_config.disabled overrides the default."""
        from tensorrt_llm.usage.config import TelemetryConfig, UsageContext
        base = {
            "model":
            "dummy",
            "telemetry_config":
            TelemetryConfig(disabled=False,
                            usage_context=UsageContext.CLI_SERVE),
        }
        yaml_dict = {"telemetry_config": {"disabled": True}}
        merged = update_llm_args_with_extra_dict(base, yaml_dict)
        tc = merged["telemetry_config"]
        assert isinstance(tc, TelemetryConfig)
        assert tc.disabled is True

    def test_yaml_cannot_override_usage_context(self):
        """usage_context is coupled to the CLI entry point.

        The CLI entry point (serve, eval, etc.) that first creates the
        TelemetryConfig sets usage_context, so YAML must not override it.
        """
        from tensorrt_llm.usage.config import TelemetryConfig, UsageContext
        base = {
            "model":
            "dummy",
            "telemetry_config":
            TelemetryConfig(disabled=False,
                            usage_context=UsageContext.CLI_SERVE),
        }
        yaml_dict = {
            "telemetry_config": {
                "disabled": True,
                "usage_context": "cli_eval",
            }
        }
        merged = update_llm_args_with_extra_dict(base, yaml_dict)
        tc = merged["telemetry_config"]
        assert isinstance(tc, TelemetryConfig)
        assert tc.usage_context == UsageContext.CLI_SERVE
        assert tc.disabled is True

    def test_cli_disabled_overrides_yaml_enabled_legacy_fixup(self):
        """Legacy post-merge fixup pattern (preserved for back-compat).

        This exercises the pre-`explicit_cli_keys` flow where the CLI entry
        point overrode `disabled` after the merge by hand. The CLI tools no
        longer use this pattern (they pass `explicit_cli_keys={"telemetry"}`
        instead — see `TestExplicitCliKeysPrecedence`), but third-party
        callers may still build the merge this way.
        """
        from tensorrt_llm.usage.config import TelemetryConfig, UsageContext
        base = {
            "model":
            "dummy",
            "telemetry_config":
            TelemetryConfig(disabled=False,
                            usage_context=UsageContext.CLI_EVAL),
        }
        yaml_dict = {"telemetry_config": {"disabled": False}}
        merged = update_llm_args_with_extra_dict(base, yaml_dict)
        telemetry = False
        if not telemetry:
            merged["telemetry_config"] = merged["telemetry_config"].model_copy(
                update={"disabled": True})
        tc = merged["telemetry_config"]
        assert tc.disabled is True
        assert tc.usage_context == UsageContext.CLI_EVAL

    def test_yaml_disabled_respected_when_cli_not_set(self):
        """YAML disabled=true is honored when explicit_cli_keys is None."""
        from tensorrt_llm.usage.config import TelemetryConfig, UsageContext
        base = {
            "model":
            "dummy",
            "telemetry_config":
            TelemetryConfig(disabled=False,
                            usage_context=UsageContext.CLI_SERVE),
        }
        yaml_dict = {"telemetry_config": {"disabled": True}}
        merged = update_llm_args_with_extra_dict(base, yaml_dict)
        tc = merged["telemetry_config"]
        assert tc.disabled is True
        assert tc.usage_context == UsageContext.CLI_SERVE

    @pytest.mark.parametrize("yaml_value", [None, False, "invalid", 0])
    def test_yaml_null_telemetry_config_preserves_default(self, yaml_value):
        """YAML telemetry_config: null/false/invalid preserves the CLI default."""
        from tensorrt_llm.usage.config import TelemetryConfig, UsageContext
        base = {
            "model":
            "dummy",
            "telemetry_config":
            TelemetryConfig(disabled=False,
                            usage_context=UsageContext.CLI_SERVE),
        }
        yaml_dict = {"telemetry_config": yaml_value}
        merged = update_llm_args_with_extra_dict(base, yaml_dict)
        tc = merged["telemetry_config"]
        assert isinstance(tc, TelemetryConfig)
        assert tc.usage_context == UsageContext.CLI_SERVE
        assert tc.disabled is False


class TestExplicitCliKeysPrecedence:
    """`explicit_cli_keys` makes the CLI side win over YAML on conflicts."""

    def test_explicit_cli_key_wins_over_yaml_scalar(self):
        base = {"model": "dummy", "tensor_parallel_size": 4}
        yaml_dict = {"tensor_parallel_size": 8}
        merged = update_llm_args_with_extra_dict(
            base, yaml_dict, explicit_cli_keys={"tensor_parallel_size"})
        assert merged["tensor_parallel_size"] == 4

    def test_non_explicit_value_loses_to_yaml_scalar(self):
        # Backward-compat: when explicit_cli_keys is None, today's "YAML wins"
        # behavior is preserved.
        base = {"model": "dummy", "tensor_parallel_size": 4}
        yaml_dict = {"tensor_parallel_size": 8}
        merged = update_llm_args_with_extra_dict(base, yaml_dict)
        assert merged["tensor_parallel_size"] == 8

    def test_kv_cache_config_explicit_field_wins_yaml_siblings_preserved(self):
        # CLI builds a KvCacheConfig from --free_gpu_memory_fraction; YAML
        # provides a partial kv_cache_config with sibling fields that should
        # survive the merge.
        base = {
            "model": "dummy",
            "kv_cache_config": KvCacheConfig(free_gpu_memory_fraction=0.85),
        }
        yaml_dict = {
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.5,
                "enable_block_reuse": False,
            }
        }
        merged = update_llm_args_with_extra_dict(
            base, yaml_dict, explicit_cli_keys={"free_gpu_memory_fraction"})
        kv = merged["kv_cache_config"]
        assert kv.free_gpu_memory_fraction == 0.85
        assert kv.enable_block_reuse is False

    def test_multimodal_config_yaml_siblings_preserved(self):
        base = {
            "model": "dummy",
            "multimodal_config":
            MultimodalConfig(encoder_side_stream_max_ahead=2),
        }
        yaml_dict = {
            "multimodal_config": {
                "video_pruning_rate": 0.5,
            }
        }
        merged = update_llm_args_with_extra_dict(base, yaml_dict)
        multimodal_config = merged["multimodal_config"]
        assert multimodal_config.encoder_side_stream_max_ahead == 2
        assert multimodal_config.video_pruning_rate == 0.5

    def test_multimodal_config_null_yaml_preserves_base_config(self):
        base = {
            "model": "dummy",
            "multimodal_config":
            MultimodalConfig(encoder_side_stream_max_ahead=2),
        }
        yaml_dict = {"multimodal_config": None}
        merged = update_llm_args_with_extra_dict(base, yaml_dict)
        multimodal_config = merged["multimodal_config"]
        assert multimodal_config.encoder_side_stream_max_ahead == 2

    def test_build_config_tier_cli_wins(self):
        # Tier 1: explicit CLI scalar wins over both top-level YAML and nested.
        base = {
            "model": "dummy",
            "max_batch_size": 64,
            "build_config": BuildConfig(max_batch_size=64),
        }
        yaml_dict = {
            "max_batch_size": 256,
            "build_config": {
                "max_batch_size": 300
            },
        }
        merged = update_llm_args_with_extra_dict(
            base, yaml_dict, explicit_cli_keys={"max_batch_size"})
        assert merged["max_batch_size"] == 64
        assert merged["build_config"].max_batch_size == 64

    def test_build_config_tier_yaml_top_level_wins(self):
        # Tier 2: no explicit CLI, but YAML top-level scalar -> propagate to
        # build_config (legacy behavior).
        base = {
            "model": "dummy",
            "build_config": BuildConfig(max_batch_size=8),
        }
        yaml_dict = {"max_batch_size": 256}
        merged = update_llm_args_with_extra_dict(base, yaml_dict)
        assert merged["max_batch_size"] == 256
        assert merged["build_config"].max_batch_size == 256

    def test_build_config_tier_yaml_nested_only_leaves_alone(self):
        # Tier 3: no explicit CLI, no top-level YAML scalar; nested YAML
        # build_config is imported by the outer merge.
        base = {
            "model": "dummy",
            "build_config": BuildConfig(max_batch_size=8),
        }
        yaml_dict = {"build_config": {"max_batch_size": 256}}
        merged = update_llm_args_with_extra_dict(base, yaml_dict)
        assert merged["build_config"].max_batch_size == 256

    def test_telemetry_explicit_disabled_wins_over_yaml(self):
        from tensorrt_llm.usage.config import TelemetryConfig, UsageContext
        base = {
            "model":
            "dummy",
            "telemetry_config":
            TelemetryConfig(disabled=True,
                            usage_context=UsageContext.CLI_SERVE),
        }
        yaml_dict = {"telemetry_config": {"disabled": False}}
        merged = update_llm_args_with_extra_dict(
            base, yaml_dict, explicit_cli_keys={"telemetry"})
        assert merged["telemetry_config"].disabled is True

    def test_kv_cache_dtype_explicit_wins_over_yaml(self):
        # Mirrors the kv_cache_config tier-2 path for the second mapped CLI
        # scalar (`--kv_cache_dtype` -> `kv_cache_config.dtype`).
        base = {
            "model": "dummy",
            "kv_cache_config": KvCacheConfig(dtype="fp8"),
        }
        yaml_dict = {"kv_cache_config": {"dtype": "auto"}}
        merged = update_llm_args_with_extra_dict(
            base, yaml_dict, explicit_cli_keys={"kv_cache_dtype"})
        assert merged["kv_cache_config"].dtype == "fp8"

    def test_enable_block_reuse_explicit_wins_over_yaml(self):
        # Mirrors the kv_cache_config tier-2 path for `--disable_kv_cache_reuse`,
        # which translates to `enable_block_reuse` in explicit_cli_keys.
        base = {
            "model": "dummy",
            "kv_cache_config": KvCacheConfig(enable_block_reuse=False),
        }
        yaml_dict = {"kv_cache_config": {"enable_block_reuse": True}}
        merged = update_llm_args_with_extra_dict(
            base, yaml_dict, explicit_cli_keys={"enable_block_reuse"})
        assert merged["kv_cache_config"].enable_block_reuse is False


class TestEvalTranslationMap:
    """eval's _CLICK_TO_LLM_ARG via the shared helper."""

    def _collect(self, click_param_names):
        """Simulate a Click ctx with the given params explicitly set."""
        import click as _click

        from tensorrt_llm.commands import eval as eval_mod
        from tensorrt_llm.commands.utils import collect_explicit_cli_keys

        class _FakeCtx:
            params = {name: object() for name in click_param_names}

            @staticmethod
            def get_parameter_source(name):
                from click.core import ParameterSource
                return ParameterSource.COMMANDLINE

        original = _click.get_current_context
        _click.get_current_context = lambda: _FakeCtx
        try:
            return collect_explicit_cli_keys(
                exclude=("extra_llm_api_options", "config"),
                translate=eval_mod._CLICK_TO_LLM_ARG)
        finally:
            _click.get_current_context = original

    @pytest.mark.parametrize(
        "click_name,expected",
        [
            ("tp_size", "tensor_parallel_size"),
            ("pp_size", "pipeline_parallel_size"),
            ("ep_size", "moe_expert_parallel_size"),
            ("kv_cache_free_gpu_memory_fraction", "free_gpu_memory_fraction"),
            ("disable_kv_cache_reuse", "enable_block_reuse"),
            ("max_batch_size", "max_batch_size"),  # unmapped: identity
        ])
    def test_translation(self, click_name, expected):
        assert expected in self._collect({click_name})

    def test_meta_flags_excluded(self):
        assert self._collect({"extra_llm_api_options", "config"}) == set()


class TestBenchTranslationMap:
    """`collect_explicit_cli_keys` in bench.benchmark rewrites Click param names."""

    def _collect(self, click_param_names):
        import click as _click

        from tensorrt_llm.bench import benchmark as bench_mod

        class _FakeCtx:
            params = {name: object() for name in click_param_names}

            @staticmethod
            def get_parameter_source(name):
                from click.core import ParameterSource
                return ParameterSource.COMMANDLINE

        # `bench_mod.collect_explicit_cli_keys()` calls `click.get_current_context()`.
        original = _click.get_current_context
        _click.get_current_context = lambda: _FakeCtx
        try:
            return bench_mod.collect_explicit_cli_keys()
        finally:
            _click.get_current_context = original

    @pytest.mark.parametrize(
        "click_name,expected",
        [
            ("tp", "tensor_parallel_size"),
            ("pp", "pipeline_parallel_size"),
            ("ep", "moe_expert_parallel_size"),
            ("cluster_size", "moe_cluster_parallel_size"),
            ("kv_cache_free_gpu_mem_fraction", "free_gpu_memory_fraction"),
            ("enable_chunked_context", "enable_chunked_prefill"),
            ("max_batch_size", "max_batch_size"),  # unmapped: identity
        ])
    def test_translation(self, click_name, expected):
        assert expected in self._collect({click_name})

    def test_beam_width_does_not_participate(self):
        # `--beam_width` is a SamplingParams flag, not an llm_args field, so
        # it must be left out of the translation map. Otherwise an explicit
        # `--beam_width N` would silently drop YAML's `max_beam_width`
        # without anything in llm_args to replace it.
        explicit = self._collect({"beam_width"})
        assert "max_beam_width" not in explicit
        assert "beam_width" in explicit

    def test_meta_flags_excluded(self):
        assert self._collect({"extra_llm_api_options", "config"}) == set()


class TestDisaggLauncherKwargsPreservation:
    """Regression tests for `_build_llm_args_from_disagg_server_cfg`.

    The disagg launcher takes a single `server_cfg.other_args` dict and
    must produce an llm_args dict that contains every user-set field —
    including kwargs that fell through `get_llm_args`'s named signature
    into its `**llm_args_extra_dict` catch-all (e.g. `quant_config`,
    `lora_config`, `pytorch_backend_config`).
    """

    def test_extra_kwargs_survive(self):
        from tensorrt_llm.commands.serve import \
            _build_llm_args_from_disagg_server_cfg

        other_args = {
            "model": llama_model_path,
            "backend": "pytorch",
            "tensor_parallel_size": 1,
            "gpus_per_node": 1,
            # These do not match get_llm_args's named params; they go into
            # **llm_args_extra_dict and must reach the LLM constructor.
            "quant_config": {
                "quant_algo": "FP8"
            },
            "lora_config": {
                "lora_dir": ["/tmp/lora-test"]
            },
        }

        final = _build_llm_args_from_disagg_server_cfg(other_args)

        assert "quant_config" in final
        assert "lora_config" in final
        assert isinstance(final["quant_config"], QuantConfig)
        assert isinstance(final["lora_config"], LoraConfig)

    def test_default_valued_named_params_survive(self):
        """A disagg-YAML field equal to its LlmArgs class default survives."""
        from tensorrt_llm.commands.serve import \
            _build_llm_args_from_disagg_server_cfg

        other_args = {
            "model": llama_model_path,
            "backend": "pytorch",
            "tensor_parallel_size": 1,  # equals LlmArgs class default
            "gpus_per_node": 1,
        }
        final = _build_llm_args_from_disagg_server_cfg(other_args)
        assert final.get("tensor_parallel_size") == 1


class TestTorchLlmArgsCudaGraphSettings:

    def test_cuda_graph_batch_sizes_case_0(self):
        # set both cuda_graph_batch_sizes and cuda_graph_config.max_batch_size, and
        # cuda_graph_batch_sizes is not equal to generated
        with pytest.raises(ValueError):
            TorchLlmArgs(
                model=llama_model_path,
                cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 3],
                                                  max_batch_size=128),
            )

    def test_cuda_graph_batch_sizes_case_0_1(self):
        # set both cuda_graph_batch_sizes and cuda_graph_config.max_batch_size, and
        # cuda_graph_batch_sizes is equal to generated
        args = TorchLlmArgs(
            model=llama_model_path,
            cuda_graph_config=CudaGraphConfig(
                batch_sizes=CudaGraphConfig._generate_cuda_graph_batch_sizes(
                    128, True),
                enable_padding=True,
                max_batch_size=128))
        assert args.cuda_graph_config.batch_sizes == CudaGraphConfig._generate_cuda_graph_batch_sizes(
            128, True)
        assert args.cuda_graph_config.max_batch_size == 128

    def test_cuda_graph_batch_sizes_case_1(self):
        # set cuda_graph_batch_sizes only
        args = TorchLlmArgs(model=llama_model_path,
                            cuda_graph_config=CudaGraphConfig(
                                batch_sizes=[1, 2, 4], enable_padding=True))
        assert args.cuda_graph_config.batch_sizes == [1, 2, 4]

    def test_cuda_graph_batch_sizes_case_2(self):
        # set cuda_graph_config.max_batch_size only
        args = TorchLlmArgs(model=llama_model_path,
                            cuda_graph_config=CudaGraphConfig(
                                max_batch_size=128, enable_padding=True))
        assert args.cuda_graph_config.batch_sizes == CudaGraphConfig._generate_cuda_graph_batch_sizes(
            128, True)
        assert args.cuda_graph_config.max_batch_size == 128

    def test_cuda_graph_config_legacy_alias_uses_decode_config(self):
        config = CudaGraphConfig(batch_sizes=[1, 2, 4], enable_padding=True)

        assert isinstance(config, DecodeCudaGraphConfig)
        assert config.mode == "decode"
        assert config.batch_sizes == [1, 2, 4]

    def test_cuda_graph_config_accepts_encoder_config(self):
        args = TorchLlmArgs(model=llama_model_path,
                            cuda_graph_config=EncodeCudaGraphConfig(
                                batch_sizes=[1, 4],
                                num_tokens=[16, 64],
                                seq_lens=[8, 32],
                                enable_padding=True,
                            ))

        assert isinstance(args.cuda_graph_config, EncodeCudaGraphConfig)
        assert args.cuda_graph_config.mode == "encode"
        assert args.cuda_graph_config.num_tokens == [16, 64]
        assert args.cuda_graph_config.max_num_token == 64
        assert args.cuda_graph_config.seq_lens == [8, 32]
        assert args.cuda_graph_config.max_seq_len == 32

    def test_cuda_graph_config_infers_encode_mode_from_raw_dict(self):
        args = TorchLlmArgs(
            model=llama_model_path,
            cuda_graph_config={
                "batch_sizes": [1, 4],
                "num_tokens": [16, 64],
                "seq_lens": [8, 32],
                "enable_padding": True,
            },
        )

        assert isinstance(args.cuda_graph_config, EncodeCudaGraphConfig)
        assert args.cuda_graph_config.mode == "encode"

    def test_cuda_graph_config_infers_decode_mode_from_raw_dict(self):
        args = TorchLlmArgs(
            model=llama_model_path,
            cuda_graph_config={
                "batch_sizes": [1, 4],
                "enable_padding": True,
            },
        )

        assert isinstance(args.cuda_graph_config, DecodeCudaGraphConfig)
        assert args.cuda_graph_config.mode == "decode"

    @pytest.mark.parametrize("max_batch_size", [64, 129, 320])
    def test_generate_cuda_graph_batch_sizes_padding_edge_cases(
            self, max_batch_size):
        # All sizes must be <= max_batch_size, sorted, and include max_batch_size
        batch_sizes = CudaGraphConfig._generate_cuda_graph_batch_sizes(
            max_batch_size, enable_padding=True)
        assert all(s <= max_batch_size for s in batch_sizes)
        assert batch_sizes == sorted(batch_sizes)
        assert max_batch_size in batch_sizes


class TestPiecewiseCudaGraphCaptureDefaults:
    """Piecewise CUDA graph capture-set defaults and reachable-ceiling filter.

    Three invariants are exercised:

    1. `TorchCompileConfig.capture_num_tokens` defaults to a fixed
       powers-of-2 + 256-stride list when `enable_piecewise_cuda_graph`
       is True (and stays `None` otherwise). The fixed list keeps the
       capture set small to bound startup time and CUDA graph memory;
       the model-engine filter (invariants 2 and 3) ensures the largest
       reachable size is always captured even when it is not in this
       default list.
    2. `_filter_piecewise_capture_num_tokens` caps the candidate list at
       `max_batch_size * (max_seq_len - 1 - num_extra_decoding_steps)` --
       the largest forward-pass `num_tokens` the warmup builder can
       construct, since every in-flight request must leave room for at
       least one decode token.
    3. The reachable ceiling itself is always present in the returned
       capture set (when positive), so runtime ISLs in the gap between
       the next-largest candidate and the ceiling get a graph rather
       than falling back to eager.
    """

    _EXPECTED_DEFAULT_CAPTURE_NUM_TOKENS = [2**i for i in range(8)] + list(
        range(256, 3073, 256))

    def test_torch_compile_config_capture_num_tokens_default_when_piecewise_enabled(
            self):
        """Default capture set is the powers-of-2 + 256-stride list.

        Keeps the capture set bounded (~20 entries) so server startup
        time and CUDA graph memory stay predictable. The model engine
        further filters and appends the reachable ceiling, so
        out-of-range entries (e.g. > max_seq_len-1) are never recorded
        and gap ISLs still get a graph.
        """
        config = TorchCompileConfig(enable_piecewise_cuda_graph=True)
        assert config.capture_num_tokens == self._EXPECTED_DEFAULT_CAPTURE_NUM_TOKENS

    def test_torch_compile_config_capture_num_tokens_stays_none_when_piecewise_disabled(
            self):
        """No default is populated when piecewise is off.

        The capture set is irrelevant in this case; populating it would
        be misleading in serialized configs.
        """
        config = TorchCompileConfig(enable_piecewise_cuda_graph=False)
        assert config.capture_num_tokens is None

    def test_torch_compile_config_capture_num_tokens_user_override_preserved(
            self):
        """User-supplied `capture_num_tokens` is not overwritten by the default."""
        user_list = [4, 8, 16]
        config = TorchCompileConfig(enable_piecewise_cuda_graph=True,
                                    capture_num_tokens=user_list)
        # `validate_capture_num_tokens` dedupes and reverse-sorts.
        assert config.capture_num_tokens == sorted(set(user_list), reverse=True)

    def test_torch_llm_args_capture_num_tokens_default_when_piecewise_enabled(
            self):
        """Same default applies when reached through `TorchLlmArgs` construction.

        This is the path real users hit via `trtllm-serve` YAML.
        """
        args = TorchLlmArgs(
            model=llama_model_path,
            max_batch_size=1,
            max_seq_len=128,
            max_beam_width=10,
            enable_chunked_prefill=True,
            cuda_graph_config=CudaGraphConfig(max_batch_size=128,
                                              enable_padding=True),
            torch_compile_config=TorchCompileConfig(
                enable_piecewise_cuda_graph=True),
        )
        assert args.torch_compile_config.capture_num_tokens == self._EXPECTED_DEFAULT_CAPTURE_NUM_TOKENS

    def test_piecewise_filter_drops_entries_above_reachable_ceiling(self):
        """Drop candidates above `max_batch_size * (max_seq_len - 1)`.

        Without the cap, the warmup loop would silently skip these entries
        and the outer padding logic would pad to a target with no captured
        graph. They must be removed from `kept` and surfaced in
        `unrecordable` so the warning fires. The ceiling itself is then
        appended so ISLs in the gap still get a graph.
        """
        from tensorrt_llm._torch.pyexecutor.model_engine import \
            _filter_piecewise_capture_num_tokens

        candidates = CudaGraphConfig._generate_cuda_graph_batch_sizes(
            128, enable_padding=True)
        max_capturable = 1 * (128 - 1)
        # Precondition: candidate list contains at least one entry above
        # the reachable ceiling, otherwise the assertions below are vacuous.
        assert any(i > max_capturable for i in candidates), (
            "Test precondition no longer holds: cuda_graph_batch_sizes "
            f"for max_batch_size=128 no longer contains entries above "
            f"{max_capturable}. Update this test if CudaGraphConfig "
            "behavior changed.")

        kept, unrecordable = _filter_piecewise_capture_num_tokens(
            candidates,
            max_num_tokens=128,
            max_batch_size=1,
            max_seq_len=128,
        )

        assert kept[-1] == max_capturable  # ceiling appended
        assert 120 in kept  # densely-packed entries below ceiling preserved
        assert 128 not in kept
        assert unrecordable == [128]

    def test_piecewise_filter_keeps_all_entries_when_within_ceiling(self):
        """Keep all candidates when the largest fits within the ceiling.

        Symmetric case: when `max_batch_size * (max_seq_len - 1)` is at
        least as large as the biggest candidate, nothing is dropped and
        `unrecordable` is empty. The ceiling (128 here) coincides with the
        largest candidate so no extra entry is appended.
        """
        from tensorrt_llm._torch.pyexecutor.model_engine import \
            _filter_piecewise_capture_num_tokens

        candidates = CudaGraphConfig._generate_cuda_graph_batch_sizes(
            128, enable_padding=True)
        kept, unrecordable = _filter_piecewise_capture_num_tokens(
            candidates,
            max_num_tokens=129,
            max_batch_size=1,
            max_seq_len=129,
        )
        assert kept[-1] == 128
        assert 128 in kept
        # Ceiling (128) was already in candidates, must not be duplicated.
        assert kept.count(128) == 1
        assert unrecordable == []

    def test_piecewise_filter_subtracts_extra_decoding_steps(self):
        """Subtract `num_extra_decoding_steps` from the ceiling.

        Drafting loops consume extra decode steps; the filter must mirror
        the `max_seq_len - 1 - num_extra_decoding_steps` constraint
        applied when warmup requests are built. The ceiling is appended
        whenever it is strictly greater than the largest surviving
        candidate.
        """
        from tensorrt_llm._torch.pyexecutor.model_engine import \
            _filter_piecewise_capture_num_tokens

        candidates = [1, 2, 4, 8, 16, 32, 64, 100, 120]
        # max_seq_len=128, batch=1, 5 extra decoding steps -> ceiling 122.
        kept, unrecordable = _filter_piecewise_capture_num_tokens(
            candidates,
            max_num_tokens=128,
            max_batch_size=1,
            max_seq_len=128,
            num_extra_decoding_steps=5,
        )
        assert kept[-1] == 122
        assert 120 in kept
        assert unrecordable == []
        # Same setup with 9 extra decoding steps -> ceiling 118; 120 drops.
        kept, unrecordable = _filter_piecewise_capture_num_tokens(
            candidates,
            max_num_tokens=128,
            max_batch_size=1,
            max_seq_len=128,
            num_extra_decoding_steps=9,
        )
        assert kept[-1] == 118
        assert 100 in kept
        assert 120 not in kept
        assert unrecordable == [120]

    def test_piecewise_filter_does_not_double_append_ceiling(self):
        """Ceiling already present in candidates -> not duplicated."""
        from tensorrt_llm._torch.pyexecutor.model_engine import \
            _filter_piecewise_capture_num_tokens

        kept, _ = _filter_piecewise_capture_num_tokens(
            [1, 64, 128],
            max_num_tokens=129,
            max_batch_size=1,
            max_seq_len=129,
        )
        assert kept == [1, 64, 128]

    def test_piecewise_filter_returns_empty_when_ceiling_is_zero(self):
        """`max_seq_len=1` -> ceiling 0 -> nothing captured.

        With ceiling 0 every positive candidate is unrecordable and the
        ceiling itself is not appended, so the warning "exceeds reachable
        ceiling 0; raise max_seq_len" fires for the full candidate list.
        """
        from tensorrt_llm._torch.pyexecutor.model_engine import \
            _filter_piecewise_capture_num_tokens

        kept, unrecordable = _filter_piecewise_capture_num_tokens(
            [1, 2, 4],
            max_num_tokens=8,
            max_batch_size=1,
            max_seq_len=1,
        )
        assert kept == []
        assert unrecordable == [1, 2, 4]

    def test_piecewise_filter_appends_ceiling_when_only_smaller_candidates(
            self):
        """No candidate near the ceiling -> ceiling still appended."""
        from tensorrt_llm._torch.pyexecutor.model_engine import \
            _filter_piecewise_capture_num_tokens

        kept, _ = _filter_piecewise_capture_num_tokens(
            [1, 2, 4, 8],
            max_num_tokens=1024,
            max_batch_size=8,
            max_seq_len=128,
        )
        # Ceiling: 8 * (128 - 1) = 1016.
        assert kept == [1, 2, 4, 8, 1016]


class TestTrtLlmArgs:

    def test_dynamic_setattr(self):
        with pytest.raises(pydantic_core._pydantic_core.ValidationError):
            args = TrtLlmArgs(model=llama_model_path, invalid_arg=1)

        with pytest.raises(ValueError):
            args = TrtLlmArgs(model=llama_model_path)
            args.invalid_arg = 1


class TestTorchLlmArgs:

    @print_traceback_on_error
    def test_runtime_sizes(self):
        with TorchLLM(llama_model_path,
                      max_beam_width=1,
                      max_num_tokens=256,
                      max_seq_len=128,
                      max_batch_size=8) as llm:
            assert llm.args.max_beam_width == 1
            assert llm.args.max_num_tokens == 256
            assert llm.args.max_seq_len == 128
            assert llm.args.max_batch_size == 8

            (
                max_beam_width,
                max_num_tokens,
                max_seq_len,
                max_batch_size,
            ) = llm.args.get_runtime_sizes()
            assert max_beam_width == 1
            assert max_num_tokens == 256
            assert max_seq_len == 128
            assert max_batch_size == 8

    def test_dynamic_setattr(self):
        with pytest.raises(pydantic_core._pydantic_core.ValidationError):
            args = TorchLlmArgs(model=llama_model_path, invalid_arg=1)

        with pytest.raises(ValueError):
            args = TorchLlmArgs(model=llama_model_path)
            args.invalid_arg = 1

    def test_speculative_model_alias(self):
        spec_config = EagleDecodingConfig(
            max_draft_len=3,
            speculative_model_dir="/path/to/model",
            eagle3_one_model=False,
        )

        args = TorchLlmArgs(model=llama_model_path,
                            speculative_config=spec_config)
        assert args.speculative_model == "/path/to/model"

    @print_traceback_on_error
    def test_model_kwargs_with_num_hidden_layers(self):
        config_no_kwargs = ModelConfig.from_pretrained(
            llama_model_path).pretrained_config
        model_kwargs = {'num_hidden_layers': 2}
        config_with_kwargs = ModelConfig.from_pretrained(
            llama_model_path, model_kwargs=model_kwargs).pretrained_config
        assert config_no_kwargs.num_hidden_layers != config_with_kwargs.num_hidden_layers
        assert config_with_kwargs.num_hidden_layers == 2


class TestTrtLlmArgs:

    def test_build_config_default(self):
        args = TrtLlmArgs(model=llama_model_path)
        # It will create a default build_config
        assert args.build_config
        assert args.build_config.max_beam_width == 1

    def test_build_config_change(self):
        build_config = BuildConfig(
            max_beam_width=4,
            max_batch_size=8,
            max_num_tokens=256,
        )
        args = TrtLlmArgs(model=llama_model_path, build_config=build_config)
        assert args.build_config.max_beam_width == build_config.max_beam_width
        assert args.build_config.max_batch_size == build_config.max_batch_size
        assert args.build_config.max_num_tokens == build_config.max_num_tokens

    def test_LLM_with_build_config(self):
        build_config = BuildConfig(
            max_beam_width=4,
            max_batch_size=8,
            max_num_tokens=256,
        )
        args = TrtLlmArgs(model=llama_model_path, build_config=build_config)

        assert args.build_config.max_beam_width == build_config.max_beam_width
        assert args.build_config.max_batch_size == build_config.max_batch_size
        assert args.build_config.max_num_tokens == build_config.max_num_tokens

        assert args.max_beam_width == build_config.max_beam_width

    def test_to_dict_and_from_dict(self):
        build_config = BuildConfig(
            max_beam_width=4,
            max_batch_size=8,
            max_num_tokens=256,
        )
        args = TrtLlmArgs(model=llama_model_path, build_config=build_config)
        args_dict = args.model_dump()

        new_args = TrtLlmArgs(**args_dict)

        assert new_args.model_dump() == args_dict

    def test_build_config_from_engine(self):
        build_config = BuildConfig(max_batch_size=8, max_num_tokens=256)
        tmp_dir = tempfile.mkdtemp()
        with LLM(model=llama_model_path, build_config=build_config) as llm:
            llm.save(tmp_dir)

        args = TrtLlmArgs(
            model=tmp_dir,
            # runtime values
            max_num_tokens=16,
            max_batch_size=4,
        )
        assert args.build_config.max_batch_size == build_config.max_batch_size
        assert args.build_config.max_num_tokens == build_config.max_num_tokens

        assert args.max_num_tokens == 16
        assert args.max_batch_size == 4

    def test_model_dump_does_not_mutate_original(self):
        """Test that model_dump() and update_llm_args_with_extra_dict don't mutate the original."""
        # Create args with specific build_config values
        build_config = BuildConfig(
            max_batch_size=8,
            max_num_tokens=256,
        )
        args = TrtLlmArgs(model=llama_model_path, build_config=build_config)

        # Store original values
        original_max_batch_size = args.build_config.max_batch_size
        original_max_num_tokens = args.build_config.max_num_tokens

        # Convert to dict and pass through update_llm_args_with_extra_dict with overrides
        args_dict = args.model_dump()
        extra_dict = {
            "max_batch_size": 128,
            "max_num_tokens": 1024,
        }
        updated_dict = update_llm_args_with_extra_dict(args_dict, extra_dict)

        # Verify original args was NOT mutated
        assert args.build_config.max_batch_size == original_max_batch_size
        assert args.build_config.max_num_tokens == original_max_num_tokens

        # Verify updated dict has new values
        new_args = TrtLlmArgs(**updated_dict)
        assert new_args.build_config.max_batch_size == 128
        assert new_args.build_config.max_num_tokens == 1024


class TestStrictBaseModelArbitraryArgs:
    """Test that StrictBaseModel prevents arbitrary arguments from being accepted."""

    def test_cuda_graph_config_arbitrary_args(self):
        """Test that CudaGraphConfig rejects arbitrary arguments."""
        # Valid arguments should work
        config = CudaGraphConfig(batch_sizes=[1, 2, 4], max_batch_size=4)
        assert config.batch_sizes == [1, 2, 4]
        assert config.max_batch_size == 4

        # Arbitrary arguments should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            CudaGraphConfig(batch_sizes=[1, 2, 4], invalid_arg="should_fail")
        assert "invalid_arg" in str(exc_info.value)

    def test_moe_config_arbitrary_args(self):
        """Test that MoeConfig rejects arbitrary arguments."""
        # Valid arguments should work
        config = MoeConfig(backend="CUTLASS", max_num_tokens=1024)
        assert config.backend == "CUTLASS"
        assert config.max_num_tokens == 1024

        # Arbitrary arguments should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            MoeConfig(backend="CUTLASS", unknown_field="should_fail")
        assert "unknown_field" in str(exc_info.value)

    def test_calib_config_arbitrary_args(self):
        """Test that CalibConfig rejects arbitrary arguments."""
        # Valid arguments should work
        config = CalibConfig(device="cuda", calib_batches=512)
        assert config.device == "cuda"
        assert config.calib_batches == 512

        # Arbitrary arguments should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            CalibConfig(device="cuda", extra_field="should_fail")
        assert "extra_field" in str(exc_info.value)

    def test_decoding_base_config_arbitrary_args(self):
        """Test that DecodingBaseConfig rejects arbitrary arguments."""
        # Valid arguments should work
        config = DecodingBaseConfig(max_draft_len=10)
        assert config.max_draft_len == 10

        # Arbitrary arguments should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            DecodingBaseConfig(max_draft_len=10, random_field="should_fail")
        assert "random_field" in str(exc_info.value)

    def test_dynamic_batch_config_arbitrary_args(self):
        """Test that DynamicBatchConfig rejects arbitrary arguments."""
        # Valid arguments should work
        config = DynamicBatchConfig(enable_batch_size_tuning=True,
                                    enable_max_num_tokens_tuning=True,
                                    dynamic_batch_moving_average_window=8)
        assert config.enable_batch_size_tuning == True

        # Arbitrary arguments should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            DynamicBatchConfig(enable_batch_size_tuning=True,
                               enable_max_num_tokens_tuning=True,
                               dynamic_batch_moving_average_window=8,
                               fake_param="should_fail")
        assert "fake_param" in str(exc_info.value)

    def test_scheduler_config_arbitrary_args(self):
        """Test that SchedulerConfig rejects arbitrary arguments."""
        # Valid arguments should work
        config = SchedulerConfig(
            capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION)
        assert config.capacity_scheduler_policy == CapacitySchedulerPolicy.MAX_UTILIZATION

        # Arbitrary arguments should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            SchedulerConfig(capacity_scheduler_policy=CapacitySchedulerPolicy.
                            MAX_UTILIZATION,
                            invalid_option="should_fail")
        assert "invalid_option" in str(exc_info.value)

    def test_peft_cache_config_arbitrary_args(self):
        """Test that PeftCacheConfig rejects arbitrary arguments."""
        # Valid arguments should work
        config = PeftCacheConfig(num_host_module_layer=1,
                                 num_device_module_layer=1)
        assert config.num_host_module_layer == 1
        assert config.num_device_module_layer == 1

        # Arbitrary arguments should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            PeftCacheConfig(num_host_module_layer=1,
                            unexpected_field="should_fail")
        assert "unexpected_field" in str(exc_info.value)

    def test_kv_cache_config_arbitrary_args(self):
        """Test that KvCacheConfig rejects arbitrary arguments."""
        # Valid arguments should work
        config = KvCacheConfig(enable_block_reuse=True, max_tokens=1024)
        assert config.enable_block_reuse == True
        assert config.max_tokens == 1024

        # Arbitrary arguments should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            KvCacheConfig(enable_block_reuse=True,
                          non_existent_field="should_fail")
        assert "non_existent_field" in str(exc_info.value)

    def test_extended_runtime_perf_knob_config_arbitrary_args(self):
        """Test that ExtendedRuntimePerfKnobConfig rejects arbitrary arguments."""
        # Valid arguments should work
        config = ExtendedRuntimePerfKnobConfig(multi_block_mode=True,
                                               cuda_graph_mode=False)
        assert config.multi_block_mode == True
        assert config.cuda_graph_mode == False

        # Arbitrary arguments should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            ExtendedRuntimePerfKnobConfig(multi_block_mode=True,
                                          bogus_setting="should_fail")
        assert "bogus_setting" in str(exc_info.value)

    def test_cache_transceiver_config_arbitrary_args(self):
        """Test that CacheTransceiverConfig rejects arbitrary arguments."""
        # Valid arguments should work
        config = CacheTransceiverConfig(backend="UCX",
                                        max_tokens_in_buffer=1024)
        assert config.backend == "UCX"
        assert config.max_tokens_in_buffer == 1024
        assert config.kv_transfer_poll_interval_ms == 5000

        # The bounce on/off switch defaults to off (0), accepts a positive size, and rejects
        # negatives at the Pydantic boundary (ge=0). It is a Python-only field consumed directly by
        # the v2 transceiver, so it is intentionally not part of _to_pybind().
        assert config.kv_cache_bounce_size_mb == 0
        assert CacheTransceiverConfig(
            kv_cache_bounce_size_mb=384).kv_cache_bounce_size_mb == 384
        with pytest.raises(pydantic_core._pydantic_core.ValidationError):
            CacheTransceiverConfig(kv_cache_bounce_size_mb=-1)

        # Arbitrary arguments should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            CacheTransceiverConfig(backend="UCX", invalid_config="should_fail")
        assert "invalid_config" in str(exc_info.value)

    def test_torch_compile_config_arbitrary_args(self):
        """Test that TorchCompileConfig rejects arbitrary arguments."""
        # Valid arguments should work
        config = TorchCompileConfig(enable_fullgraph=True,
                                    enable_inductor=False)
        assert config.enable_fullgraph == True
        assert config.enable_inductor == False

        # Arbitrary arguments should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            TorchCompileConfig(enable_fullgraph=True,
                               invalid_flag="should_fail")
        assert "invalid_flag" in str(exc_info.value)

    def test_encode_only_rejects_piecewise_cuda_graph(self):
        """Test that encode_only rejects unsupported piecewise CUDA graphs."""
        with pytest.raises(
                ValueError,
                match="encode_only does not support piecewise CUDA graph"):
            TorchLlmArgs(
                model=llama_model_path,
                encode_only=True,
                torch_compile_config=TorchCompileConfig(
                    enable_piecewise_cuda_graph=True),
            )

    def test_encode_only_rejects_mm_encoder_only(self):
        """Test that encode_only and mm_encoder_only cannot both be enabled."""
        with pytest.raises(
                ValueError,
                match="encode_only and mm_encoder_only are mutually exclusive"):
            TorchLlmArgs(
                model=llama_model_path,
                encode_only=True,
                mm_encoder_only=True,
            )

    def test_multimodal_encoder_rejects_encode_only(self):
        """Test that MultimodalEncoder owns mm_encoder_only mode internally."""
        encoder = object.__new__(MultimodalEncoder)
        with pytest.raises(
                ValueError,
                match="MultimodalEncoder does not support encode_only"):
            encoder._validate_mm_args_for_torch_backend({"encode_only": True})

    def test_trt_llm_args_arbitrary_args(self):
        """Test that TrtLlmArgs rejects arbitrary arguments."""
        # Valid arguments should work
        args = TrtLlmArgs(model=llama_model_path, max_batch_size=8)
        assert args.model == llama_model_path
        assert args.max_batch_size == 8

        # Arbitrary arguments should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            TrtLlmArgs(model=llama_model_path, invalid_setting="should_fail")
        assert "invalid_setting" in str(exc_info.value)

    def test_torch_llm_args_arbitrary_args(self):
        """Test that TorchLlmArgs rejects arbitrary arguments."""
        # Valid arguments should work
        args = TorchLlmArgs(model=llama_model_path, max_batch_size=8)
        assert args.model == llama_model_path
        assert args.max_batch_size == 8

        # Arbitrary arguments should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            TorchLlmArgs(model=llama_model_path,
                         unsupported_option="should_fail")
        assert "unsupported_option" in str(exc_info.value)

    def test_nested_config_arbitrary_args(self):
        """Test that nested configurations also reject arbitrary arguments."""
        # Test with nested KvCacheConfig
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            KvCacheConfig(enable_block_reuse=True,
                          max_tokens=1024,
                          invalid_nested_field="should_fail")
        assert "invalid_nested_field" in str(exc_info.value)

        # Test with nested SchedulerConfig
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            SchedulerConfig(capacity_scheduler_policy=CapacitySchedulerPolicy.
                            MAX_UTILIZATION,
                            nested_invalid_field="should_fail")
        assert "nested_invalid_field" in str(exc_info.value)

    def test_strict_base_model_inheritance(self):
        """Test that StrictBaseModel properly forbids extra fields."""
        # Verify that StrictBaseModel is properly configured
        assert StrictBaseModel.model_config.get("extra") == "forbid"

        # Test that a simple StrictBaseModel instance rejects arbitrary fields
        class TestConfig(StrictBaseModel):
            field1: str = "default"
            field2: int = 42

        # Valid configuration should work
        config = TestConfig(field1="test", field2=100)
        assert config.field1 == "test"
        assert config.field2 == 100

        # Arbitrary field should be rejected
        with pytest.raises(
                pydantic_core._pydantic_core.ValidationError) as exc_info:
            TestConfig(field1="test", field2=100, extra_field="should_fail")
        assert "extra_field" in str(exc_info.value)


class TestServeDefaults:

    def test_serve_get_llm_args_preserves_model_defaults(self):
        # No explicit CLI flags: only required params and serve-side defaults
        # reach the constructor; everything else is left for YAML / model
        # defaults to provide.
        llm_args, _ = get_llm_args(
            model=llama_model_path,
            backend="pytorch",
        )

        assert "model" in llm_args
        assert "backend" in llm_args
        assert "postprocess_tokenizer_dir" in llm_args

        # PyTorch backend: build_config / scheduler_config stay None and are
        # filtered out.
        assert "build_config" not in llm_args
        assert "scheduler_config" not in llm_args

        # Explicit CLI flags survive the filter.
        llm_args_with_values, _ = get_llm_args(
            model=llama_model_path,
            backend="pytorch",
            max_batch_size=128,
            tensor_parallel_size=4,
            explicit_cli_keys={"max_batch_size", "tensor_parallel_size"},
        )
        assert llm_args_with_values.get("max_batch_size") == 128
        assert llm_args_with_values.get("tensor_parallel_size") == 4

    def test_serve_filters_default_values(self):
        # All defaults, no explicit CLI flags.
        llm_args, _ = get_llm_args(model=llama_model_path, backend="pytorch")

        assert "model" in llm_args
        assert "backend" in llm_args
        assert "postprocess_tokenizer_dir" in llm_args

        assert "build_config" not in llm_args
        assert "scheduler_config" not in llm_args

        # Custom values survive only when listed in explicit_cli_keys.
        llm_args, _ = get_llm_args(
            model=llama_model_path,
            backend="pytorch",
            max_batch_size=128,
            tensor_parallel_size=4,
            explicit_cli_keys={"max_batch_size", "tensor_parallel_size"},
        )

        assert llm_args.get("max_batch_size") == 128
        assert llm_args.get("tensor_parallel_size") == 4

    def test_serve_video_pruning_rate_maps_to_multimodal_config(self):
        llm_args, _ = get_llm_args(
            model=llama_model_path,
            backend="pytorch",
            video_pruning_rate=0.5,
            explicit_cli_keys={"video_pruning_rate"},
        )

        assert "video_pruning_rate" not in llm_args
        assert llm_args["multimodal_config"].video_pruning_rate == 0.5

    def test_serve_backend_specific_configs(self):
        # PyTorch backend: build_config / scheduler_config stay None and are
        # filtered out.
        llm_args_pytorch, _ = get_llm_args(model=llama_model_path,
                                           backend="pytorch")
        assert "build_config" not in llm_args_pytorch
        assert "scheduler_config" not in llm_args_pytorch

        # TensorRT backend: both are non-None and differ from the LlmArgs
        # class default, so the value-based filter keeps them.
        llm_args_trt, _ = get_llm_args(model=llama_model_path,
                                       backend="tensorrt")
        assert "build_config" in llm_args_trt
        assert "scheduler_config" in llm_args_trt

    def test_serve_explicit_cli_default_value_wins_over_yaml(self):
        """Typing --tensor_parallel_size 1 (the default) must beat YAML."""
        llm_args, _ = get_llm_args(
            model=llama_model_path,
            backend="pytorch",
            tensor_parallel_size=1,
            explicit_cli_keys={"tensor_parallel_size"},
        )
        # The CLI value lands in llm_args because it is explicit.
        assert llm_args["tensor_parallel_size"] == 1
        merged = update_llm_args_with_extra_dict(
            llm_args,
            {"tensor_parallel_size": 8},
            explicit_cli_keys={"tensor_parallel_size"},
        )
        assert merged["tensor_parallel_size"] == 1

    def test_serve_is_non_default_or_required_helper(self):
        # Test always_include parameters
        assert is_non_default_or_required("model", "test-model", "pytorch",
                                          set())
        assert is_non_default_or_required("backend", "pytorch", "pytorch",
                                          set())
        assert is_non_default_or_required("tokenizer", "test-tokenizer",
                                          "pytorch", set())

        # Test None values
        assert not is_non_default_or_required("max_batch_size", None, "pytorch",
                                              set())

        # Test default values (should return False)
        assert not is_non_default_or_required("tensor_parallel_size", 1,
                                              "pytorch", set())
        assert not is_non_default_or_required("pipeline_parallel_size", 1,
                                              "pytorch", set())

        # Test non-default values (should return True)
        assert is_non_default_or_required("tensor_parallel_size", 4, "pytorch",
                                          set())
        assert is_non_default_or_required("max_batch_size", 128, "pytorch",
                                          set())

        # Test explicit CLI source overrides the default-equals-value check
        assert is_non_default_or_required("tensor_parallel_size", 1, "pytorch",
                                          {"tensor_parallel_size"})
        # Test CLI-derived field (--free_gpu_memory_fraction -> kv_cache_config)
        assert is_non_default_or_required("kv_cache_config", KvCacheConfig(),
                                          "pytorch",
                                          {"free_gpu_memory_fraction"})


class TestPyTorchBackendModelDefaults:

    def get_tinyllama_path(self):
        # Use local model path if available, otherwise use HuggingFace ID
        model_root = llm_models_root()
        if model_root:
            local_path = model_root / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
            if local_path.exists():
                return str(local_path)

        # Fallback to HuggingFace model ID
        return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch, tmp_path):
        self.get_model_defaults_called = False
        self.tmp_path = tmp_path

        def mock_get_model_defaults(cls, llm_args):
            self.get_model_defaults_called = True
            return {
                "enable_chunked_prefill": True,
                "max_batch_size": 999,
                "max_input_len": 12345,
                "kv_cache_config": {
                    "enable_block_reuse": False,
                    "free_gpu_memory_fraction": 0.75,
                }
            }

        self.original_get_model_defaults = getattr(LlamaForCausalLM,
                                                   'get_model_defaults', None)
        setattr(LlamaForCausalLM, 'get_model_defaults',
                classmethod(mock_get_model_defaults))

        yield

        if self.original_get_model_defaults is None:
            delattr(LlamaForCausalLM, 'get_model_defaults')
        else:
            setattr(LlamaForCausalLM, 'get_model_defaults',
                    self.original_get_model_defaults)

    @pytest.mark.part0
    def test_model_defaults_application(self):
        self.get_model_defaults_called = False

        with TorchLLM(
                model=self.get_tinyllama_path(),
                backend='pytorch',
                skip_tokenizer_init=True,
                env_overrides={"TLLM_WORKER_USE_SINGLE_PROCESS": "1"},
        ) as llm:
            assert self.get_model_defaults_called

            modified_args = llm._executor.engine.model_engine.llm_args
            assert modified_args.enable_chunked_prefill == True
            assert modified_args.kv_cache_config.enable_block_reuse == False
            assert modified_args.kv_cache_config.free_gpu_memory_fraction == 0.75

    @pytest.mark.part0
    def test_user_overrides_respected(self):
        self.get_model_defaults_called = False

        with TorchLLM(
                model=self.get_tinyllama_path(),
                backend='pytorch',
                enable_chunked_prefill=False,
                max_batch_size=42,
                max_input_len=256,
                kv_cache_config=KvCacheConfig(enable_block_reuse=True),
                skip_tokenizer_init=True,
                env_overrides={"TLLM_WORKER_USE_SINGLE_PROCESS": "1"},
        ) as llm:
            assert self.get_model_defaults_called

            modified_args = llm._executor.engine.model_engine.llm_args
            assert modified_args.enable_chunked_prefill == False
            assert modified_args.max_batch_size == 42
            assert modified_args.max_input_len == 256
            assert modified_args.kv_cache_config.enable_block_reuse == True

    @pytest.mark.part0
    def test_partial_user_override(self):
        self.get_model_defaults_called = False

        with TorchLLM(
                model=self.get_tinyllama_path(),
                backend='pytorch',
                max_batch_size=42,
                skip_tokenizer_init=True,
                env_overrides={"TLLM_WORKER_USE_SINGLE_PROCESS": "1"},
        ) as llm:
            assert self.get_model_defaults_called

            modified_args = llm._executor.engine.model_engine.llm_args
            assert modified_args.max_batch_size == 42
            assert modified_args.enable_chunked_prefill == True
            assert modified_args.kv_cache_config.enable_block_reuse == False

    @pytest.mark.part0
    def test_empty_nested_config_preserves_defaults(self):
        """Passing an empty nested config should not block model defaults.

        This covers the pattern used by tests that conditionally build a
        KvCacheConfig: ``kv_cache_config=KvCacheConfig(...) if cond else
        KvCacheConfig()``.  The else-branch must not shadow model defaults
        like ``enable_block_reuse=False``.
        """
        self.get_model_defaults_called = False

        with TorchLLM(
                model=self.get_tinyllama_path(),
                backend='pytorch',
                kv_cache_config=KvCacheConfig(),
                skip_tokenizer_init=True,
                env_overrides={"TLLM_WORKER_USE_SINGLE_PROCESS": "1"},
        ) as llm:
            assert self.get_model_defaults_called

            modified_args = llm._executor.engine.model_engine.llm_args
            # Model defaults set enable_block_reuse=False and
            # free_gpu_memory_fraction=0.75.  An empty KvCacheConfig()
            # should not prevent these from being applied.
            assert modified_args.kv_cache_config.enable_block_reuse == False
            assert modified_args.kv_cache_config.free_gpu_memory_fraction == 0.75


def test_executor_config_consistency():
    """Verify that BaseLlmArgs exposes all ExecutorConfig options."""
    # max_beam_width is not included since vague behavior due to lacking the support for dynamic beam width during
    # generation
    black_list = set(["max_beam_width"])
    executor_config_attrs = set(attr for attr in dir(tle.ExecutorConfig)
                                if not attr.startswith('_')
                                and callable(getattr(tle.ExecutorConfig, attr)))
    executor_config_attrs -= black_list
    llm_args_attr = set(BaseLlmArgs.model_fields.keys())
    # NOTE: When cpp ExecutorConfig add new options, please add the new options into `LlmArgs` with docs as well
    # ASK chunweiy for help if you are not sure about the new options.
    assert executor_config_attrs.issubset(llm_args_attr), \
        f"New options found in underlying ExecutorConfig: {executor_config_attrs - llm_args_attr}"


def _get_all_llm_args_classes():
    """Get all subclasses of BaseLlmArgs by traversing the class hierarchy."""
    subclasses = []
    to_visit = [BaseLlmArgs]
    while to_visit:
        cls = to_visit.pop()
        for subclass in cls.__subclasses__():
            subclasses.append(subclass)
            to_visit.append(subclass)
    return subclasses


def _get_all_pydantic_models_from_llm_args():
    """Get all Pydantic models referenced by BaseLlmArgs and its subclasses."""
    visited = set()
    models = []

    def extract_types_from_annotation(annotation):
        """Extract all concrete types from an annotation (handles Union, Optional, List, etc.)."""
        args = get_args(annotation)
        if args:
            # Generic type - recurse into all type arguments
            return [
                t for arg in args if arg is not type(None)
                for t in extract_types_from_annotation(arg)
            ]
        elif isinstance(annotation, type):
            return [annotation]
        return []

    def visit_model(model_cls):
        if model_cls in visited:
            return
        if not isinstance(model_cls, type) or not issubclass(
                model_cls, BaseModel):
            return

        visited.add(model_cls)
        models.append(model_cls)

        # Visit all field types
        for field_name, field_info in model_cls.model_fields.items():
            annotation = field_info.annotation
            if annotation is None:
                continue

            # Extract all types from the annotation
            types_in_annotation = extract_types_from_annotation(annotation)
            for type_cls in types_in_annotation:
                if isinstance(type_cls, type) and issubclass(
                        type_cls, BaseModel):
                    visit_model(type_cls)

    # Start with BaseLlmArgs and all its subclasses
    for cls in [BaseLlmArgs] + _get_all_llm_args_classes():
        visit_model(cls)

    return models


def _get_qualified_name(cls: type) -> str:
    """Return the fully qualified name of a class."""
    return f"{cls.__module__}.{cls.__qualname__}"


class TestPydanticBestPractices:
    """Ensure that the user-facing LlmArgs and its subfields follow Pydantic best practices.
    """

    # Fields exempt from Pydantic compatibility checks due to typing limitations or other edge cases.
    # Avoid adding to this list unless absolutely necessary, especially if a field is user-facing.
    # Keys are Pydantic model classes, values are lists of field names.
    _COMPATIBILITY_EXEMPT_FIELDS: dict[type, list[str]] = {
        BaseLlmArgs: [
            "batched_logits_processor",  # abstract base class type
            "decoding_config",  # deprecated field, typed as object
            "model_kwargs",  # typed as Dict[str, Any] for flexibility
            "mpi_session",  # abstract base class type
            "tokenizer",  # uses PreTrainedTokenizerBase
        ],
        TorchLlmArgs: [
            "checkpoint_loader",  # abstract base class type
        ],
        AutoDeployLlmArgs: [
            "transforms",  # typed as Dict[str, Dict[str, Any]] for flexibility
            "model_kwargs",  # typed as Dict[str, Any] for flexibility
            "speculative_model_kwargs",  # typed as Dict[str, Any] for flexibility (overrides draft model HF config)
            "tokenizer_kwargs",  # typed as Dict[str, Any] for flexibility
        ],
        UserProvidedDecodingConfig: [
            "drafter",  # abstract base class type
            "resource_manager",  # abstract base class type
        ],
        MoeConfig: ["load_balancer"],  # allows multiple types including dict
        RayPlacementConfig: ["placement_groups"],  # contains Ray-specific types
    }

    def _is_allowed_type(self, annotation, model_cls: type,
                         field_name: str) -> tuple[bool, str]:
        """Check if a type annotation is allowed for user-facing config fields.

        Allowed:
        - Pydantic models (must inherit from StrictBaseModel)
        - Enums, primitives (str, int, float, bool, bytes, Path, type(None))
        - Union/Optional/List/Dict of allowed types
        Not allowed:
        - object, Any, bare containers (dict, list without type parameters)
        - dataclasses, non-Pydantic classes, Pydantic models not inheriting from StrictBaseModel

        Returns (is_allowed, reason) tuple.
        """
        # Check if this field is exempt (check class and all parent classes)
        for cls in model_cls.__mro__:
            if field_name in self._COMPATIBILITY_EXEMPT_FIELDS.get(cls, []):
                return True, "exempt"

        # Check explicitly blocked types
        if annotation is object:
            return False, "bare 'object' type (use a specific type or Pydantic model)"
        if annotation is Any:
            return False, "'Any' type (use a specific type or Pydantic model)"

        # Check for bare container types (missing type parameters)
        if annotation in (dict, list, tuple, set, frozenset):
            name = annotation.__name__
            return False, f"bare '{name}' type (use {name}[...] with specific type parameters)"

        # Check for dataclasses (should use Pydantic models instead)
        if isinstance(annotation, type) and is_dataclass(annotation):
            return False, f"class '{annotation.__name__}' is a dataclass (convert to a Pydantic StrictBaseModel instead)"

        # Check for regular classes that aren't Pydantic models, enums, or primitives
        _ALLOWED_CLASS_BASES = (BaseModel, Enum, str, int, float, bool, bytes,
                                Path, type(None))
        if isinstance(annotation, type) and not issubclass(
                annotation, _ALLOWED_CLASS_BASES):
            return False, f"class '{annotation.__name__}' is not a Pydantic model (convert to StrictBaseModel)"

        # Require user-facing Pydantic models to forbid extra fields. StrictBaseModel
        # enforces this; usage-local models (e.g. TelemetryConfig) may instead set
        # model_config extra="forbid" directly so they stay importable without the
        # heavy llmapi.utils dependency chain (which would otherwise pull torch/HF and
        # create a circular import via llm_args).
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            is_strict = issubclass(annotation, StrictBaseModel)
            forbids_extra = annotation.model_config.get("extra") == "forbid"
            if not (is_strict or forbids_extra):
                return False, f"Pydantic model '{annotation.__name__}' does not forbid extra fields (inherit StrictBaseModel or set model_config extra='forbid')"

        # Recursively check generic type arguments for disallowed types
        origin = get_origin(annotation)
        if origin is not None:
            # Skip Literal types - their args are values, not types
            if origin is Literal:
                return True, "Literal type"

            # For Annotated types, only check the first arg (the actual type)
            # The rest are metadata (validators, Field info, etc.)
            if origin is Annotated:
                args = get_args(annotation)
                return self._is_allowed_type(args[0], model_cls, field_name)

            # For other generic types (Union, List, Dict, etc.), check all type args
            container_name = getattr(origin, "__name__", str(origin))
            for arg in get_args(annotation):
                if arg is type(None):
                    continue
                # Explicitly check for Any (it's a special form, not a type)
                if arg is Any:
                    return False, f"in {container_name}: 'Any' type (use a specific type)"
                # Skip non-type args (e.g., Callable parameter specs)
                if not isinstance(arg, type) and get_origin(arg) is None:
                    continue
                ok, reason = self._is_allowed_type(arg, model_cls, field_name)
                if not ok:
                    return False, f"in {container_name}: {reason}"

        return True, "allowed"

    def test_all_fields_have_descriptions(self):
        """Test that all fields in LlmArgs classes (including subfields) have descriptions."""
        violations = []

        for cls in _get_all_pydantic_models_from_llm_args():
            for field_name, field_info in cls.model_fields.items():
                if field_name.startswith("_"):
                    # Skip private / non user-facing fields
                    continue
                # Skip discriminator fields (single-value Literals)
                annotation = field_info.annotation
                if get_origin(annotation) is Literal and len(
                        get_args(annotation)) == 1:
                    continue
                if field_info.description is None or field_info.description.strip(
                ) == "":
                    violations.append(
                        f"{_get_qualified_name(cls)}.{field_name}: missing description"
                    )

        if violations:
            pytest.fail(
                "The following fields are missing descriptions:\n" +
                "\n".join(violations) +
                "\n\nPlease add a description to each by using Field(description=\"...\")."
            )

    def test_all_fields_have_allowed_types(self):
        """Test that all fields in LlmArgs classes have allowed types.

        Checks that fields (including subfields) have Pydantic-compatible
        types according to the logic in _is_allowed_type.
        """
        violations = []

        for cls in _get_all_pydantic_models_from_llm_args():
            for field_name, field_info in cls.model_fields.items():
                if field_name.startswith("_"):
                    # Skip private / non user-facing fields
                    continue
                is_compatible, reason = self._is_allowed_type(
                    field_info.annotation, cls, field_name)
                if not is_compatible:
                    violations.append(
                        f"{_get_qualified_name(cls)}.{field_name}: {reason}")

        if violations:
            pytest.fail(
                "The following user-facing fields have types that are not allowed:\n"
                + "\n".join(violations) +
                "\n\nPlease use Pydantic-compatible types (primitives, Pydantic models that inherit from StrictBaseModel, "
                "or other compatible types). If this is intentional, add the field "
                "to _COMPATIBILITY_EXEMPT_FIELDS.")

    # Methods that shouldn't be manually defined on Pydantic models
    # Keys are method names, values are suggestions for replacement
    _FORBIDDEN_METHODS = {
        "from_dict":
        "Construct the class directly from the dict instead, i.e. MyModel(**my_dict).",
        "from_kwargs":
        "Construct the class directly from the kwargs instead, i.e. MyModel(**kwargs).",
        "to_dict": "Use Pydantic's model_dump() instead.",
        "validate":
        "Use Pydantic's @field_validator or @model_validator instead.",
        "is_valid":
        "Use Pydantic's @field_validator or @model_validator instead.",
    }

    # Classes exempt from specific forbidden methods. Avoid adding to this list unless absolutely
    # necessary, e.g. if needed to preserve backward compatibility with external libraries
    # Keys are method names, values are sets of class names
    _FORBIDDEN_METHODS_EXEMPT_CLASSES = {
        "from_dict": {QuantConfig, LayerQuantConfig},
    }

    def test_no_manual_serialization_or_validation_methods(self):
        """Test that LlmArgs and all nested Pydantic models do not define manual serialization/validation methods."""
        violations = []

        for cls in _get_all_pydantic_models_from_llm_args():
            for method_name, suggestion in self._FORBIDDEN_METHODS.items():
                if method_name in cls.__dict__:
                    if cls in self._FORBIDDEN_METHODS_EXEMPT_CLASSES.get(
                            method_name, set()):
                        continue
                    violations.append(
                        f"{_get_qualified_name(cls)}.{method_name}(): {suggestion}"
                    )

        if violations:
            pytest.fail(
                "The following models define forbidden methods:\n" +
                "\n".join(violations) +
                "\n\nPydantic models should follow the recommendations above instead."
            )

    def test_no_mutable_default_values(self):
        """Test that LlmArgs and all nested Pydantic models do not use mutable default values directly."""
        violations = []

        for cls in _get_all_pydantic_models_from_llm_args():
            for field_name, field_info in cls.model_fields.items():
                if field_name.startswith("_"):
                    continue
                # Check if the default is a mutable type instance
                default = field_info.default
                if isinstance(default, (list, dict, set)):
                    type_name = type(default).__name__
                    violations.append(
                        f"{_get_qualified_name(cls)}.{field_name}: uses mutable default {type_name}. "
                        f"Use Field(default_factory={type_name}) instead.")

        if violations:
            pytest.fail(
                "The following fields use mutable default values:\n" +
                "\n".join(violations) +
                "\n\nMutable defaults are shared across instances and cause bugs. "
                "Use Field(default_factory=...) instead.")

    def test_no_custom_init_methods(self):
        """Test that LlmArgs and all nested Pydantic models do not define custom __init__ methods.

        Pydantic models should not override __init__ because:
        - It bypasses Pydantic's validation and type coercion
        - It can cause subtle bugs with model inheritance

        There are better alternatives for all common use cases:
        - For validation: use @field_validator or @model_validator
        - For post-validation initialization: use model_post_init()
        - For custom construction patterns: use classmethods (e.g. from_yaml())

        See: https://docs.pydantic.dev/latest/concepts/models/#defining-a-custom-__init__
        """
        violations = []

        for cls in _get_all_pydantic_models_from_llm_args():
            if "__init__" in cls.__dict__:
                violations.append(_get_qualified_name(cls))

        if violations:
            pytest.fail(
                "The following Pydantic models define custom __init__ methods:\n"
                + "\n".join(violations) +
                "\n\nThese should be replaced with alternatives like validators, model_post_init, or classmethods. See this test's docstring for more details."
            )


class TestSkipSoftmaxAttentionConfig:
    """Test LLM Skip Softmax Attention config behavior."""

    CHECKPOINT_SPARSE_ATTENTION_CONFIG: ClassVar[dict] = {
        "config_groups": {
            "group_0": {
                "algorithm": "skip_softmax",
                "threshold_scale_factor": {
                    "formula": "a * exp(b * target_sparsity)",
                    "prefill": {
                        "a": 100.0,
                        "b": 5.0
                    },
                    "decode": {
                        "a": 0.05,
                        "b": 10.0
                    },
                },
                "target_sparsity": {
                    "prefill": 0.5,
                    "decode": 0.3
                },
            },
        },
    }
    CHECKPOINT_CONFIG: ClassVar[dict] = {
        "sparse_attention_config": CHECKPOINT_SPARSE_ATTENTION_CONFIG,
    }

    @classmethod
    def _checkpoint_config(cls) -> dict:
        import copy

        return copy.deepcopy(cls.CHECKPOINT_CONFIG)

    @staticmethod
    def _kernel_params(config: SkipSoftmaxAttentionConfig, **kwargs):
        sparse_params = config.to_sparse_params(**kwargs)
        return sparse_params.scheduler.get_kernel_params()

    def test_python_api_parses_skip_softmax_config(self):
        args = TorchLlmArgs(
            model="/tmp/dummy_model",
            sparse_attention_config={
                "algorithm": "skip_softmax",
                "threshold_scale_factor": {
                    "prefill": 1000.0,
                    "decode": 500.0,
                },
            },
        )

        config = args.sparse_attention_config

        assert isinstance(config, SkipSoftmaxAttentionConfig)
        assert config.threshold_scale_factor == {
            "prefill": 1000.0,
            "decode": 500.0,
        }

    def test_yaml_api_parses_skip_softmax_config(self):
        config_dict = yaml.safe_load("""
sparse_attention_config:
  algorithm: skip_softmax
  target_sparsity:
    prefill: 0.5
    decode: 0.3
""")

        args = TorchLlmArgs(model="/tmp/dummy_model", **config_dict)

        config = args.sparse_attention_config
        assert isinstance(config, SkipSoftmaxAttentionConfig)
        assert config.target_sparsity == {
            "prefill": 0.5,
            "decode": 0.3,
        }

    @pytest.mark.parametrize("target_sparsity", [-0.1, 1.1])
    def test_target_sparsity_scalar_must_be_in_unit_interval(
            self, target_sparsity):
        with pytest.raises(ValidationError, match="target_sparsity"):
            SkipSoftmaxAttentionConfig(target_sparsity=target_sparsity)

    @pytest.mark.parametrize("target_sparsity", [
        {
            "prefill": -0.1,
            "decode": 0.3,
        },
        {
            "prefill": 0.5,
            "decode": 1.1,
        },
    ])
    def test_target_sparsity_phase_values_must_be_in_unit_interval(
            self, target_sparsity):
        with pytest.raises(ValidationError, match="target_sparsity"):
            SkipSoftmaxAttentionConfig(target_sparsity=target_sparsity)

    def test_direct_threshold_scale_factor_scalar_does_not_need_checkpoint(
            self):
        config = SkipSoftmaxAttentionConfig(threshold_scale_factor=1000.0)

        params = self._kernel_params(config)

        assert params.threshold_scale_factor_prefill == pytest.approx(1000.0)
        assert params.threshold_scale_factor_decode == pytest.approx(1000.0)

    def test_direct_threshold_scale_factor_can_be_configured_per_phase(self):
        config = SkipSoftmaxAttentionConfig(threshold_scale_factor={
            "prefill": 1000.0,
            "decode": 500.0,
        })

        params = self._kernel_params(config)

        assert params.threshold_scale_factor_prefill == pytest.approx(1000.0)
        assert params.threshold_scale_factor_decode == pytest.approx(500.0)

    def test_target_sparsity_scalar_uses_checkpoint_formula_for_each_phase(
            self):
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.3)

        params = self._kernel_params(
            config,
            checkpoint_config=self._checkpoint_config(),
        )

        assert params.threshold_scale_factor_prefill == pytest.approx(
            100.0 * math.exp(5.0 * 0.3))
        assert params.threshold_scale_factor_decode == pytest.approx(
            0.05 * math.exp(10.0 * 0.3))

    def test_target_sparsity_can_be_configured_per_phase(self):
        config = SkipSoftmaxAttentionConfig(target_sparsity={
            "prefill": 0.5,
            "decode": 0.3,
        })

        params = self._kernel_params(
            config,
            checkpoint_config=self._checkpoint_config(),
        )

        assert params.threshold_scale_factor_prefill == pytest.approx(
            100.0 * math.exp(5.0 * 0.5))
        assert params.threshold_scale_factor_decode == pytest.approx(
            0.05 * math.exp(10.0 * 0.3))

    def test_checkpoint_target_sparsity_default_is_used_when_user_omits_it(
            self):
        config = SkipSoftmaxAttentionConfig()

        params = self._kernel_params(
            config,
            checkpoint_config=self._checkpoint_config(),
        )

        assert params.threshold_scale_factor_prefill == pytest.approx(
            100.0 * math.exp(5.0 * 0.5))
        assert params.threshold_scale_factor_decode == pytest.approx(
            0.05 * math.exp(10.0 * 0.3))

    def test_checkpoint_formula_can_be_shared_by_prefill_and_decode(self):
        checkpoint_config = {
            "sparse_attention_config": {
                "config_groups": {
                    "group_0": {
                        "algorithm": "skip_softmax",
                        "threshold_scale_factor": {
                            "formula": "sqrt(a + target_sparsity)",
                            "a": 0.75,
                        },
                        "target_sparsity": 0.25,
                    },
                },
            },
        }
        config = SkipSoftmaxAttentionConfig()

        params = self._kernel_params(config,
                                     checkpoint_config=checkpoint_config)

        assert params.threshold_scale_factor_prefill == pytest.approx(1.0)
        assert params.threshold_scale_factor_decode == pytest.approx(1.0)

    def test_user_target_sparsity_overrides_checkpoint_default(self):
        config = SkipSoftmaxAttentionConfig(target_sparsity={
            "prefill": 0.2,
            "decode": 0.4,
        })

        params = self._kernel_params(
            config,
            checkpoint_config=self._checkpoint_config(),
        )

        assert params.threshold_scale_factor_prefill == pytest.approx(
            100.0 * math.exp(5.0 * 0.2))
        assert params.threshold_scale_factor_decode == pytest.approx(
            0.05 * math.exp(10.0 * 0.4))

    def test_threshold_scale_factor_wins_over_target_sparsity_without_checkpoint(
            self):
        config = SkipSoftmaxAttentionConfig(
            threshold_scale_factor={
                "prefill": 1000.0,
                "decode": 500.0,
            },
            target_sparsity=0.9,
        )

        params = self._kernel_params(config)

        assert params.threshold_scale_factor_prefill == pytest.approx(1000.0)
        assert params.threshold_scale_factor_decode == pytest.approx(500.0)

    def test_target_sparsity_requires_checkpoint_formula(self):
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.5)

        with pytest.raises(ValueError, match="formula"):
            config.to_sparse_params(
                checkpoint_config={
                    "sparse_attention_config": {
                        "config_groups": {
                            "group_0": {
                                "algorithm": "skip_softmax",
                                "target_sparsity": 0.5,
                            },
                        },
                    },
                })

    def test_other_checkpoint_groups_do_not_affect_skip_softmax_group_selection(
            self):
        checkpoint_config = self._checkpoint_config()
        checkpoint_config["sparse_attention_config"]["config_groups"][
            "group_1"] = {
                "algorithm": "rocket",
                "prompt_budget": 2048,
            }
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.5)

        params = self._kernel_params(config,
                                     checkpoint_config=checkpoint_config)

        assert params.threshold_scale_factor_prefill == pytest.approx(
            100.0 * math.exp(5.0 * 0.5))

    def test_checkpoint_ignore_patterns_disable_matching_module_name(self):
        checkpoint_config = self._checkpoint_config()
        group = checkpoint_config["sparse_attention_config"]["config_groups"][
            "group_0"]
        group["ignore"] = [
            "model.layers.0.self_attn",
            "model.layers.1.*",
        ]
        config = SkipSoftmaxAttentionConfig(threshold_scale_factor=1000.0)

        assert (config.to_sparse_params(
            module_name="model.layers.0.self_attn",
            checkpoint_config=checkpoint_config,
        ) is None)
        assert (config.to_sparse_params(
            module_name="model.layers.1.self_attn",
            checkpoint_config=checkpoint_config,
        ) is None)
        params = self._kernel_params(
            config,
            module_name="model.layers.2.self_attn",
            checkpoint_config=checkpoint_config,
        )
        assert params.threshold_scale_factor_prefill == pytest.approx(1000.0)

    def test_checkpoint_ignore_patterns_match_layer_idx_aliases(self):
        checkpoint_config = self._checkpoint_config()
        group = checkpoint_config["sparse_attention_config"]["config_groups"][
            "group_0"]
        group["ignore"] = ["model.layers.0.self_attn"]
        config = SkipSoftmaxAttentionConfig(threshold_scale_factor=1000.0)

        assert (config.to_sparse_params(
            layer_idx=0,
            checkpoint_config=checkpoint_config,
        ) is None)
        params = self._kernel_params(
            config,
            layer_idx=1,
            checkpoint_config=checkpoint_config,
        )
        assert params.threshold_scale_factor_prefill == pytest.approx(1000.0)

    def test_multiple_skip_softmax_checkpoint_groups_are_invalid(self):
        checkpoint_config = self._checkpoint_config()
        groups = checkpoint_config["sparse_attention_config"]["config_groups"]
        groups["group_1"] = {
            "algorithm": "rocket",
            "prompt_budget": 2048,
        }
        groups["group_2"] = dict(groups["group_0"])
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.5)

        with pytest.raises(ValueError, match="multiple skip-softmax"):
            config.to_sparse_params(checkpoint_config=checkpoint_config)

    def test_ckpt_sparse_attention_config_can_be_passed_directly(self):
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.5)

        params = self._kernel_params(
            config,
            ckpt_sparse_attention_config=self.
            CHECKPOINT_SPARSE_ATTENTION_CONFIG,
        )

        assert params.threshold_scale_factor_prefill == pytest.approx(
            100.0 * math.exp(5.0 * 0.5))


class TestDeepSeekV4SparseAttentionConfig:

    def test_zero_compress_ratios_are_normalized(self):
        config = DeepSeekV4SparseAttentionConfig(compress_ratios=[0, 4, 128])

        assert config.compress_ratios == [1, 4, 128]

    def test_default_fp4_falls_back_to_fp8_before_blackwell(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr("tensorrt_llm._utils.get_sm_version", lambda: 90)

        config = DeepSeekV4SparseAttentionConfig()

        assert config.indexer_k_dtype == "fp8"

    def test_explicit_fp4_is_rejected_before_blackwell(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr("tensorrt_llm._utils.get_sm_version", lambda: 90)

        with pytest.raises(ValidationError, match="requires SM>=100"):
            DeepSeekV4SparseAttentionConfig(indexer_k_dtype="fp4")

    def test_lowers_to_deepseek_v4_sparse_params(self):
        config = DeepSeekV4SparseAttentionConfig(compress_ratios=[0, 4, 128])

        sparse_params = config.to_sparse_params()
        sparse_metadata_params = config.to_sparse_metadata_params()

        assert sparse_params.algorithm == "deepseek_v4"
        assert sparse_params.compress_ratios == [1, 4, 128]
        assert sparse_metadata_params.compress_ratios == [1, 4, 128]
        assert sparse_metadata_params.window_size == 128

    @pytest.mark.parametrize("compress_ratios", [[], [-1, 4, 128]])
    def test_invalid_compress_ratios_raise(self, compress_ratios):
        with pytest.raises(ValidationError, match="compress_ratios"):
            DeepSeekV4SparseAttentionConfig(compress_ratios=compress_ratios)


class TestEnableLowLatencyHostDispatch:

    def test_default_is_false(self):
        args = TorchLlmArgs(model="gpt2")
        assert args.enable_low_latency_host_dispatch is False

    @pytest.mark.parametrize("value", [True, False])
    def test_accepts_bool(self, value):
        args = TorchLlmArgs(model="gpt2",
                            enable_low_latency_host_dispatch=value)
        assert args.enable_low_latency_host_dispatch is value

    def test_yaml_round_trip(self):
        args = TorchLlmArgs(model="gpt2", enable_low_latency_host_dispatch=True)
        data = args.model_dump()
        assert data["enable_low_latency_host_dispatch"] is True
        restored = TorchLlmArgs(**data)
        assert restored.enable_low_latency_host_dispatch is True

    def test_rejects_non_bool(self):
        with pytest.raises(ValidationError):
            TorchLlmArgs(model="gpt2",
                         enable_low_latency_host_dispatch={"val": 1})

import tempfile
from dataclasses import is_dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, get_args, get_origin

import pydantic_core
import pytest
import yaml
from pydantic import BaseModel, TypeAdapter

import tensorrt_llm.bindings.executor as tle
from tensorrt_llm import LLM as TorchLLM
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm._torch.auto_deploy.llm_args import \
    LlmArgs as AutoDeployLlmArgs
from tensorrt_llm.llmapi import (BuildConfig, CapacitySchedulerPolicy,
                                 SchedulerConfig)
from tensorrt_llm.llmapi.llm_args import *
from tensorrt_llm.llmapi.utils import print_traceback_on_error
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


def test_decoding_type_eagle3_parses_to_eagle3_decoding_config():
    adapter = TypeAdapter(SpeculativeConfig)
    spec_cfg = adapter.validate_python(
        dict(decoding_type="Eagle3",
             max_draft_len=3,
             speculative_model="/path/to/draft/model"))
    assert isinstance(spec_cfg, Eagle3DecodingConfig)


def test_decoding_type_eagle_warns_on_pytorch_backend(monkeypatch):
    import tensorrt_llm.llmapi.llm_args as llm_args_mod

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


def check_defaults(py_config_cls, pybind_config_cls):
    py_config = py_config_cls()
    pybind_config = pybind_config_cls()
    # get member variables from pybinding_config
    for member in PybindMirror.get_pybind_variable_fields(pybind_config_cls):
        py_value = getattr(py_config, member)
        pybind_value = getattr(pybind_config, member)
        assert py_value == pybind_value, f"{member} default value is not equal"


def test_KvCacheConfig_declaration():
    config = KvCacheConfig(enable_block_reuse=True,
                           max_tokens=1024,
                           max_attention_window=[1024, 1024, 1024],
                           sink_token_length=32,
                           free_gpu_memory_fraction=0.5,
                           host_cache_size=1024,
                           onboard_blocks=True,
                           cross_kv_cache_fraction=0.5,
                           secondary_offload_min_priority=1,
                           event_buffer_max_size=0,
                           enable_partial_reuse=True,
                           copy_on_partial_reuse=True,
                           attention_dp_events_gather_period_ms=10)

    pybind_config = config._to_pybind()
    assert pybind_config.enable_block_reuse == True
    assert pybind_config.max_tokens == 1024
    assert pybind_config.max_attention_window == [1024, 1024, 1024]
    assert pybind_config.sink_token_length == 32
    assert pybind_config.free_gpu_memory_fraction == 0.5
    assert pybind_config.host_cache_size == 1024
    assert pybind_config.onboard_blocks == True
    assert pybind_config.cross_kv_cache_fraction == 0.5
    assert pybind_config.secondary_offload_min_priority == 1
    assert pybind_config.event_buffer_max_size == 0
    assert pybind_config.enable_partial_reuse == True
    assert pybind_config.copy_on_partial_reuse == True
    assert pybind_config.attention_dp_events_gather_period_ms == 10


def test_CapacitySchedulerPolicy():
    val = CapacitySchedulerPolicy.MAX_UTILIZATION
    assert PybindMirror.maybe_to_pybind(
        val) == tle.CapacitySchedulerPolicy.MAX_UTILIZATION


def test_ContextChunkingPolicy():
    val = ContextChunkingPolicy.EQUAL_PROGRESS
    assert PybindMirror.maybe_to_pybind(
        val) == tle.ContextChunkingPolicy.EQUAL_PROGRESS


def test_DynamicBatchConfig_declaration():
    config = DynamicBatchConfig(enable_batch_size_tuning=True,
                                enable_max_num_tokens_tuning=True,
                                dynamic_batch_moving_average_window=10)

    pybind_config = PybindMirror.maybe_to_pybind(config)

    assert pybind_config.enable_batch_size_tuning == True
    assert pybind_config.enable_max_num_tokens_tuning == True
    assert pybind_config.dynamic_batch_moving_average_window == 10


def test_SchedulerConfig_declaration():
    config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        context_chunking_policy=ContextChunkingPolicy.EQUAL_PROGRESS,
        dynamic_batch_config=DynamicBatchConfig(
            enable_batch_size_tuning=True,
            enable_max_num_tokens_tuning=True,
            dynamic_batch_moving_average_window=10))

    pybind_config = PybindMirror.maybe_to_pybind(config)
    assert pybind_config.capacity_scheduler_policy == tle.CapacitySchedulerPolicy.MAX_UTILIZATION
    assert pybind_config.context_chunking_policy == tle.ContextChunkingPolicy.EQUAL_PROGRESS
    assert PybindMirror.pybind_equals(pybind_config.dynamic_batch_config,
                                      config.dynamic_batch_config._to_pybind())


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
        """Test that speculative_model_dir is accepted as an alias for speculative_model."""

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
        """Test that model_kwargs can override num_hidden_layers."""
        from tensorrt_llm._torch.model_config import ModelConfig
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
    """
    Get all Pydantic models referenced by BaseLlmArgs and its subclasses,
    including nested models.
    """

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


# Fields exempt from Pydantic compatibility checks due to typing limitations or other edge cases.
# Avoid adding to this list unless absolutely necessary, especially if a field is user-facing.
# Keys are Pydantic model classes, values are lists of field names.
EXEMPT_FIELDS: dict[type, list[str]] = {
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
        "draft_checkpoint_loader",  # typed as object due to circular import
        "transforms",  # typed as Dict[str, Dict[str, Any]] for flexibility
        "model_kwargs",  # typed as Dict[str, Any] for flexibility
        "tokenizer_kwargs",  # typed as Dict[str, Any] for flexibility
    ],
    UserProvidedDecodingConfig: [
        "drafter",  # abstract base class type
        "resource_manager",  # abstract base class type
    ],
    MoeConfig: ["load_balancer"],  # allows multiple types including dict
    RayPlacementConfig: ["placement_groups"],  # contains Ray-specific types
}


def _get_qualified_name(cls: type) -> str:
    """Return the fully qualified name of a class."""
    return f"{cls.__module__}.{cls.__qualname__}"


class TestPydanticBestPractices:
    """
    Ensure that the user-facing LlmArgs and its subfields follow Pydantic best practices.
    """

    def _is_allowed_type(self, annotation, model_cls: type,
                         field_name: str) -> tuple[bool, str]:
        """
        Check if a type annotation is allowed for user-facing config fields.

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
            if field_name in EXEMPT_FIELDS.get(cls, []):
                return True, "exempt"

        # Check explicitly blocked types
        if annotation is object:
            return False, "bare 'object' type (use a specific type or Pydantic model)"
        if annotation is Any:
            return False, "'Any' type (use a specific type)"

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

        # Require user-facing Pydantic models to inherit from StrictBaseModel
        if isinstance(annotation, type) and issubclass(
                annotation,
                BaseModel) and not issubclass(annotation, StrictBaseModel):
            return False, f"Pydantic model '{annotation.__name__}' is not a StrictBaseModel (convert to StrictBaseModel)"

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
                f"The following fields are missing descriptions:\n" +
                "\n".join(violations) +
                "\n\nPlease add a description to each by using Field(description=\"...\")."
            )

    def test_all_fields_have_allowed_types(self):
        """Test that all fields in LlmArgs classes (including subfields) have types that are allowed
        (i.e. are Pydantic-compatible) according to the logic in _is_allowed_type."""
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
                f"The following user-facing fields have types that are not allowed:\n"
                + "\n".join(violations) +
                "\n\nPlease use Pydantic-compatible types (primitives, Pydantic models that inherit from StrictBaseModel, "
                "or other compatible types). If this is intentional, add the field "
                "to EXEMPT_FIELDS.")

    # Methods that shouldn't be manually defined on Pydantic models
    _FORBIDDEN_METHODS = {
        "from_dict":
        "Construct the class directly from the dict instead, i.e. MyModel(**my_dict).",
        "to_dict": "Use Pydantic's model_dump() instead.",
        "validate":
        "Use Pydantic's @field_validator or @model_validator instead.",
        "is_valid":
        "Use Pydantic's @field_validator or @model_validator instead.",
    }

    def test_no_manual_serialization_or_validation_methods(self):
        """Test that LlmArgs and all nested Pydantic models do not define manual serialization/validation methods."""
        violations = []

        for cls in _get_all_pydantic_models_from_llm_args():
            for method_name, suggestion in self._FORBIDDEN_METHODS.items():
                if method_name in cls.__dict__:
                    violations.append(
                        f"{_get_qualified_name(cls)}.{method_name}(): {suggestion}"
                    )

        if violations:
            pytest.fail(
                f"The following models define forbidden methods:\n" +
                "\n".join(violations) +
                "\n\nPydantic models should use built-in methods instead. "
                "See: https://docs.pydantic.dev/latest/concepts/serialization/")

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
                f"The following fields use mutable default values:\n" +
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
                f"The following Pydantic models define custom __init__ methods:\n"
                + "\n".join(violations) +
                "\n\nThese should be replaced with alternatives like validators, model_post_init, or classmethods. See this test's docstring for more details."
            )

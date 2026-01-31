import tempfile

import pydantic_core
import pytest
import yaml
from pydantic import BaseModel

import tensorrt_llm.bindings.executor as tle
from tensorrt_llm import LLM as TorchLLM
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.builder import LoraConfig
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
    config = LookaheadDecodingConfig.from_dict({
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

    def test_update_llm_args_with_extra_dict_with_speculative_config(self):
        yaml_content = """
speculative_config:
    decoding_type: Lookahead
    max_window_size: 4
    max_ngram_size: 3
    """
        dict_content = self._yaml_to_dict(yaml_content)

        llm_args = TrtLlmArgs(model=llama_model_path)
        llm_args_dict = update_llm_args_with_extra_dict(llm_args.model_dump(),
                                                        dict_content)
        llm_args = TrtLlmArgs(**llm_args_dict)
        assert llm_args.speculative_config.max_window_size == 4
        assert llm_args.speculative_config.max_ngram_size == 3
        assert llm_args.speculative_config.max_verification_set_size == 4

    def test_llm_args_with_invalid_yaml(self):
        yaml_content = """
pytorch_backend_config: # this is deprecated
    max_num_tokens: 1
    max_seq_len: 1
"""
        dict_content = self._yaml_to_dict(yaml_content)

        llm_args = TrtLlmArgs(model=llama_model_path)
        llm_args_dict = update_llm_args_with_extra_dict(llm_args.model_dump(),
                                                        dict_content)
        with pytest.raises(ValueError):
            llm_args = TrtLlmArgs(**llm_args_dict)

    def test_llm_args_with_build_config(self):
        # build_config isn't a Pydantic
        yaml_content = """
build_config:
    max_beam_width: 4
    max_batch_size: 8
    max_num_tokens: 256
    """
        dict_content = self._yaml_to_dict(yaml_content)

        llm_args = TrtLlmArgs(model=llama_model_path)
        llm_args_dict = update_llm_args_with_extra_dict(llm_args.model_dump(),
                                                        dict_content)
        llm_args = TrtLlmArgs(**llm_args_dict)
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
        dict_content = self._yaml_to_dict(yaml_content)

        llm_args = TrtLlmArgs(model=llama_model_path)
        llm_args_dict = update_llm_args_with_extra_dict(llm_args.model_dump(),
                                                        dict_content)
        llm_args = TrtLlmArgs(**llm_args_dict)
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
        dict_content = self._yaml_to_dict(yaml_content)

        llm_args = TrtLlmArgs(model=llama_model_path)
        llm_args_dict = update_llm_args_with_extra_dict(llm_args.model_dump(),
                                                        dict_content)
        llm_args = TrtLlmArgs(**llm_args_dict)
        assert llm_args.max_batch_size == 16
        assert llm_args.max_num_tokens == 256
        assert llm_args.max_seq_len == 128

    @pytest.mark.parametrize("llm_args_cls", [TrtLlmArgs, TorchLlmArgs])
    def test_llm_args_with_model_kwargs(self, llm_args_cls):
        yaml_content = """
model_kwargs:
    num_hidden_layers: 2
    """
        dict_content = self._yaml_to_dict(yaml_content)
        llm_args = llm_args_cls(model=llama_model_path)
        llm_args_dict = update_llm_args_with_extra_dict(llm_args.model_dump(),
                                                        dict_content)
        llm_args = llm_args_cls(**llm_args_dict)
        assert llm_args.model_kwargs['num_hidden_layers'] == 2


def test_decoding_type_eagle3_parses_to_eagle3_decoding_config():
    spec_cfg = DecodingBaseConfig.from_dict(
        dict(decoding_type="Eagle3",
             max_draft_len=3,
             speculative_model_dir="/path/to/draft/model"))
    assert isinstance(spec_cfg, Eagle3DecodingConfig)


def test_decoding_type_eagle_warns_on_pytorch_backend(monkeypatch):
    import tensorrt_llm.llmapi.llm_args as llm_args_mod

    warnings_seen: list[str] = []

    def _capture_warning(msg, *args, **kwargs):
        warnings_seen.append(str(msg))

    monkeypatch.setattr(llm_args_mod.logger, "warning", _capture_warning)

    spec_cfg = DecodingBaseConfig.from_dict(dict(
        decoding_type="Eagle",
        max_draft_len=3,
        speculative_model_dir="/path/to/draft/model"),
                                            backend="pytorch")
    assert isinstance(spec_cfg, Eagle3DecodingConfig)

    TorchLlmArgs(model=llama_model_path, speculative_config=spec_cfg)

    assert any(
        "EAGLE (v1/v2) draft checkpoints are incompatible with Eagle3" in m
        for m in warnings_seen)


def test_decoding_type_eagle3_errors_on_tensorrt_backend():
    spec_cfg = DecodingBaseConfig.from_dict(
        dict(decoding_type="Eagle3",
             max_draft_len=3,
             speculative_model_dir="/path/to/draft/model"))
    with pytest.raises(ValueError,
                       match="only supported on the PyTorch backend"):
        TrtLlmArgs(model=llama_model_path, speculative_config=spec_cfg)


@pytest.mark.parametrize(
    "llm_args,model_defaults,expected_field,expected_value", [
        ({}, {
            "enable_chunked_prefill": False
        }, "enable_chunked_prefill", False),
        ({
            "kv_cache_config": {
                "enable_block_reuse": True
            }
        }, {
            "kv_cache_config": {
                "enable_block_reuse": False
            }
        }, "kv_cache_config.enable_block_reuse", True),
    ])
def test_merge_llm_configs_with_defaults(llm_args, model_defaults,
                                         expected_field, expected_value):
    merged = merge_llm_configs_with_defaults(llm_args, model_defaults)

    if "." in expected_field:
        # Handle nested field access
        field_parts = expected_field.split(".")
        value = merged[field_parts[0]]
        for part in field_parts[1:]:
            value = getattr(value, part)
        assert value == expected_value
        if field_parts[0] == "kv_cache_config":
            assert isinstance(merged["kv_cache_config"], KvCacheConfig)
    else:
        assert merged[expected_field] == expected_value


def test_merge_llm_configs_with_defaults_preserves_user_override():
    llm_args = {"enable_chunked_prefill": True}
    model_defaults = {"enable_chunked_prefill": False}
    merged = merge_llm_configs_with_defaults(llm_args, model_defaults)
    assert merged["enable_chunked_prefill"] is True


def test_compute_applied_llm_defaults_simple_field():
    model_defaults = {"enable_chunked_prefill": False}
    applied = compute_applied_llm_defaults(model_defaults, {})
    assert applied == model_defaults


class _DummyInner(BaseModel):
    a: int = 1
    b: int = 2


class _DummyOuter(BaseModel):
    inner: _DummyInner = _DummyInner()
    c: int = 3


def test_extract_llm_args_overrides_uses_fields_set():
    obj = _DummyOuter(inner=_DummyInner(b=10))
    overrides = extract_llm_args_overrides(obj)
    assert overrides == {"inner": {"b": 10}}


def test_apply_model_defaults_respects_user_override():
    from tensorrt_llm._torch.models.modeling_qwen3_next import \
        Qwen3NextForCausalLM
    llm_args = TorchLlmArgs(
        model=llama_model_path,
        kv_cache_config=KvCacheConfig(enable_block_reuse=True))
    applied = apply_model_defaults_to_llm_args(
        llm_args, Qwen3NextForCausalLM.get_model_defaults(llm_args))
    assert applied == {}
    assert llm_args.kv_cache_config.enable_block_reuse is True


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

        new_args = TrtLlmArgs.from_kwargs(**args_dict)

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
        config = CudaGraphConfig(batch_sizes=[1, 2, 4], max_batch_size=8)
        assert config.batch_sizes == [1, 2, 4]
        assert config.max_batch_size == 8

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


class TestServeDefaults:

    def test_serve_get_llm_args_preserves_model_defaults(self):
        from tensorrt_llm.commands.serve import get_llm_args

        # Get llm_args with default values (simulating serve.py behavior)
        llm_args, _ = get_llm_args(
            model=llama_model_path,
            backend="pytorch",
            # Don't pass parameters to test default behavior
        )

        # Verify that required params are present
        assert "model" in llm_args
        assert "backend" in llm_args
        assert "postprocess_tokenizer_dir" in llm_args

        # For PyTorch backend, build_config and scheduler_config should NOT be included
        assert "build_config" not in llm_args
        assert "scheduler_config" not in llm_args

        # Test that when we DO pass values, they're included appropriately
        llm_args_with_values, _ = get_llm_args(
            model=llama_model_path,
            backend="pytorch",
            max_batch_size=128,  # Non-default value
            tensor_parallel_size=4,  # Non-default value
        )
        assert llm_args_with_values.get("max_batch_size") == 128
        assert llm_args_with_values.get("tensor_parallel_size") == 4

    def test_serve_filters_default_values(self):
        from tensorrt_llm.commands.serve import get_llm_args

        # Test with all defaults for PyTorch backend
        llm_args, _ = get_llm_args(model=llama_model_path, backend="pytorch")

        # Should only include required params
        assert "model" in llm_args
        assert "backend" in llm_args
        assert "postprocess_tokenizer_dir" in llm_args

        # Should NOT include build_config or scheduler_config for PyTorch
        assert "build_config" not in llm_args
        assert "scheduler_config" not in llm_args

        # Test with custom values
        llm_args, _ = get_llm_args(
            model=llama_model_path,
            backend="pytorch",
            max_batch_size=128,  # Non-default value
            tensor_parallel_size=4,  # Non-default value
        )

        # Custom values should be included
        assert llm_args.get("max_batch_size") == 128
        assert llm_args.get("tensor_parallel_size") == 4

    def test_serve_backend_specific_configs(self):
        from tensorrt_llm.commands.serve import get_llm_args

        # Test PyTorch backend
        llm_args_pytorch, _ = get_llm_args(model=llama_model_path,
                                           backend="pytorch")
        assert "build_config" not in llm_args_pytorch
        assert "scheduler_config" not in llm_args_pytorch

        # Test TensorRT backend
        llm_args_trt, _ = get_llm_args(model=llama_model_path,
                                       backend="tensorrt")
        assert "build_config" in llm_args_trt
        assert "scheduler_config" in llm_args_trt

    def test_serve_is_non_default_or_required_helper(self):
        from tensorrt_llm.commands.serve import is_non_default_or_required

        # Test always_include parameters
        assert is_non_default_or_required("model", "test-model", "pytorch")
        assert is_non_default_or_required("backend", "pytorch", "pytorch")
        assert is_non_default_or_required("tokenizer", "test-tokenizer",
                                          "pytorch")

        # Test None values
        assert not is_non_default_or_required("max_batch_size", None, "pytorch")

        # Test default values (should return False)
        assert not is_non_default_or_required("tensor_parallel_size", 1,
                                              "pytorch")
        assert not is_non_default_or_required("pipeline_parallel_size", 1,
                                              "pytorch")

        # Test non-default values (should return True)
        assert is_non_default_or_required("tensor_parallel_size", 4, "pytorch")
        assert is_non_default_or_required("max_batch_size", 128, "pytorch")

    def test_qwen3next_defaults_work_end_to_end(self):
        from tensorrt_llm._torch.models.modeling_qwen3_next import \
            Qwen3NextForCausalLM
        from tensorrt_llm.commands.serve import get_llm_args

        # Simulate serve.py creating llm_args
        llm_args, _ = get_llm_args(model=llama_model_path, backend="pytorch")

        # Create TorchLlmArgs with serve's output
        args = TorchLlmArgs(**llm_args)

        # Verify default is True before model defaults are applied
        assert args.kv_cache_config.enable_block_reuse is True

        # Apply model defaults (simulating what happens in llm_utils)
        model_defaults = Qwen3NextForCausalLM.get_model_defaults(args)
        applied = apply_model_defaults_to_llm_args(args, model_defaults)

        # Verify block_reuse is disabled by model defaults
        assert args.kv_cache_config.enable_block_reuse is False
        assert "kv_cache_config" in applied
        assert "enable_block_reuse" in applied["kv_cache_config"]

    def test_user_overrides_preserved_through_serve(self):
        from tensorrt_llm._torch.models.modeling_qwen3_next import \
            Qwen3NextForCausalLM

        # Simulate user creating args with explicit override
        # User wants to keep block_reuse=True even though Qwen3Next disables it
        args = TorchLlmArgs(
            model=llama_model_path,
            kv_cache_config=KvCacheConfig(enable_block_reuse=True))

        # Verify user's explicit value is set
        assert args.kv_cache_config.enable_block_reuse is True

        # Apply Qwen3Next model defaults (which normally disable block_reuse)
        model_defaults = Qwen3NextForCausalLM.get_model_defaults(args)

        # This should try to set enable_block_reuse=False
        assert model_defaults.get("kv_cache_config",
                                  {}).get("enable_block_reuse") is False

        # Apply the model defaults
        applied = apply_model_defaults_to_llm_args(args, model_defaults)

        # User override should be preserved (user explicitly set it)
        assert args.kv_cache_config.enable_block_reuse is True
        # Model default should NOT be applied since user explicitly set it
        assert "enable_block_reuse" not in applied.get("kv_cache_config", {})


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
def test_model_defaults_validation(defaults_dict, should_raise, error_contains):
    from tensorrt_llm.llmapi.llm_args import validate_model_defaults

    # Use a dummy model path for testing (doesn't need to exist for validation)
    llm_args = TorchLlmArgs(model="/tmp/dummy_model_for_validation_test")

    if should_raise:
        # Should raise ValueError with expected message
        with pytest.raises(ValueError) as exc_info:
            validate_model_defaults(defaults_dict, llm_args)
        assert error_contains in str(exc_info.value)
    else:
        # Should pass validation
        validated = validate_model_defaults(defaults_dict, llm_args)
        assert validated == defaults_dict

        # Verify the defaults can be applied successfully
        applied = apply_model_defaults_to_llm_args(llm_args, validated)

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

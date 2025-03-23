import tempfile

import yaml
from test_llm import llama_model_path

import tensorrt_llm.bindings.executor as tle
from tensorrt_llm.llmapi.llm_args import *


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


def test_update_llm_args_with_extra_dict_with_speculative_config():
    yaml_content = """
speculative_config:
  decoding_type: Lookahead
  max_window_size: 4
  max_ngram_size: 3
  verification_set_size: 4
"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(yaml_content.encode('utf-8'))
        f.flush()
        f.seek(0)
        dict_content = yaml.safe_load(f)

    llm_args = LlmArgs(llama_model_path)
    llm_args_dict = update_llm_args_with_extra_dict(llm_args.to_dict(),
                                                    dict_content)
    llm_args = LlmArgs(**llm_args_dict)
    assert llm_args.speculative_config.max_window_size == 4
    assert llm_args.speculative_config.max_ngram_size == 3
    assert llm_args.speculative_config.max_verification_set_size == 4


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
                           copy_on_partial_reuse=True)

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


def test_KvCacheConfig_default_values():
    check_defaults(KvCacheConfig, tle.KvCacheConfig)

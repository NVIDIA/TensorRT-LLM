import random

import pytest
import tensorrt as _trt

import tensorrt_llm.plugin as _tlp


def test_load_library():
    """Test loading the TensorRT-LLM plugin library."""
    runtime = _trt.Runtime(_trt.Logger(_trt.Logger.WARNING))
    registry = runtime.get_plugin_registry()
    handle = registry.load_library(_tlp.plugin_lib_path())
    creators = registry.plugin_creator_list
    assert len(creators) >= 10
    for creator in creators:
        assert creator.plugin_namespace == _tlp.TRT_LLM_PLUGIN_NAMESPACE

    registry.deregister_library(handle)
    assert len(registry.plugin_creator_list) == 0


@pytest.mark.parametrize('dtype', ['float16', 'bfloat16', 'float32'])
def test_plugin_config(dtype):
    plugin_config = _tlp.PluginConfig.from_dict({'dtype': dtype})
    assert plugin_config.dtype == dtype

    assert plugin_config._gpt_attention_plugin == 'auto'
    assert plugin_config.gpt_attention_plugin == dtype
    assert plugin_config._nccl_plugin == 'auto'
    assert plugin_config.nccl_plugin == dtype
    assert plugin_config._gemm_plugin is None
    assert plugin_config.gemm_plugin is None

    new_dtype_options = ['float16', 'bfloat16', 'float32']
    new_dtype_options.remove(dtype)
    new_dtype = random.choice(new_dtype_options)
    assert new_dtype != dtype
    plugin_config.dtype = new_dtype
    plugin_config.gpt_attention_plugin = dtype
    plugin_config.gemm_plugin = 'auto'

    assert plugin_config._gpt_attention_plugin == dtype
    assert plugin_config.gpt_attention_plugin == dtype
    assert plugin_config._nccl_plugin == 'auto'
    assert plugin_config.nccl_plugin == new_dtype
    assert plugin_config._gemm_plugin == 'auto'
    assert plugin_config.gemm_plugin == new_dtype

    with pytest.raises(Exception):
        plugin_config.dtype = None
    with pytest.raises(Exception):
        plugin_config.dtype = 'auto'
    with pytest.raises(Exception):
        plugin_config.dtype = 'xyz'
    with pytest.raises(Exception):
        plugin_config.gpt_attention_plugin = 'abc'
    with pytest.raises(Exception):
        plugin_config.nccl_plugin = 123
    with pytest.raises(Exception):
        plugin_config.a_new_xxx_plugin = 'float16'

    config_dict = plugin_config.to_dict()
    new_plugin_config = _tlp.PluginConfig.from_dict(config_dict)
    assert config_dict == new_plugin_config.to_dict()

    plugin_config.to_legacy_setting()

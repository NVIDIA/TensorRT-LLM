import random

import pytest
import tensorrt as _trt

import tensorrt_llm.plugin as _tlp


def test_load_library():
    """Test loading the TensorRT LLM plugin library."""
    runtime = _trt.Runtime(_trt.Logger(_trt.Logger.WARNING))
    _trt.init_libnvinfer_plugins(runtime.logger,
                                 namespace=_tlp.TRT_LLM_PLUGIN_NAMESPACE)
    registry = runtime.get_plugin_registry()
    handle = registry.load_library(_tlp.plugin_lib_path())
    creators = registry.all_creators
    # This will give all plugins statically registered in getCreators (only V3 plugins for now)
    assert len(creators) > 0
    for creator in creators:
        assert creator.plugin_namespace == _tlp.TRT_LLM_PLUGIN_NAMESPACE

    registry.deregister_library(handle)
    assert len(registry.plugin_creator_list) == 0


@pytest.mark.parametrize('dtype', ['float16', 'bfloat16', 'float32'])
def test_plugin_config(dtype):
    plugin_config = _tlp.PluginConfig(**{'dtype': dtype})
    assert plugin_config.dtype == dtype

    assert plugin_config.gpt_attention_plugin == dtype
    assert plugin_config.nccl_plugin == dtype
    assert plugin_config.gemm_plugin is None

    new_dtype_options = ['float16', 'bfloat16', 'float32']
    new_dtype_options.remove(dtype)
    new_dtype = random.choice(new_dtype_options)
    assert new_dtype != dtype
    plugin_config.dtype = new_dtype
    plugin_config.gpt_attention_plugin = dtype
    plugin_config.gemm_plugin = 'auto'

    assert plugin_config.gpt_attention_plugin == dtype
    assert plugin_config.nccl_plugin == new_dtype
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

    config_dict = plugin_config.model_dump(mode="python")
    new_plugin_config = _tlp.PluginConfig(**config_dict)
    assert config_dict == new_plugin_config.model_dump(mode="python")

    plugin_config.to_legacy_setting()

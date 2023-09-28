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

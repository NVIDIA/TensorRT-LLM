{% if add_header %}
# //header begin//

import ctypes
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt

from tensorrt_llm._common import default_trtnet
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.functional import Tensor, _create_tensor
from tensorrt_llm.module import Module

TRT_LLM_PLUGIN_NAMESPACE = 'tensorrt_llm'

def _load_triton_plugin_lib():
    triton_plugin_dir = Path(__file__).parent.absolute()
    plugin_lib = "[[ plugin_lib_path ]]"
    handle = ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
    if handle is None:
        raise ImportError('TensorRT LLM Triton Plugin is unavailable')
    handle.initLibNvInferPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.initLibNvInferPlugins.restype = ctypes.c_bool
    assert handle.initLibNvInferPlugins(
        None, TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))

_load_triton_plugin_lib()

# //header end//
{% endif %}

def [[ kernel_name ]]([[ arg_list ]]):
    '''
    Inputs:
    {% for arg in metadata.get_params() -%}
    - [[arg.name]]: [[arg.dtype.dtype.to('np')]]
    {% endfor %}
    {% for arg in metadata.get_inputs() -%}
    - [[arg.name]]: {% if arg.is_tensor %}tensor<{%endif%}[[arg.dtype.dtype.to('np')]]>
    {% endfor %}
    Outputs:
    {% for arg in metadata.get_outputs() -%}
    - [[arg.name]]: {% if arg.is_tensor %}tensor<{%endif%}[[arg.dtype.dtype.to('np')]]>
    {% endfor -%}
    '''
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        '[[ plugin_name ]]', '[[ kernel_version ]]', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    pfc = trt.PluginFieldCollection([
        {% for arg in params -%}
        {# input is a dict[name, np_type, trt_type ] #}
        trt.PluginField("[[arg.name]]", np.array([ [[ arg.name ]] ], np.[[ arg.dtype.dtype.to('np') ]]),
                        trt.PluginFieldType.[[ arg.dtype.dtype.to('trt_plugin_py') ]]),
        {% endfor %}
    ])

    plugin = plg_creator.create_plugin("[[ plugin_name ]]", pfc)

    plug_inputs = [ [[ input_list ]] ]
    layer = default_trtnet().add_plugin_v2(plug_inputs, plugin)

    return [
        {% for id in range(num_outputs) %}
        _create_tensor(layer.get_output([[ id ]]), layer),
        {% endfor %}
    ]

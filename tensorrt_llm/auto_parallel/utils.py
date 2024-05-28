import contextlib
import threading

try:
    from types import NoneType
except ImportError:
    NoneType = type(None)
from typing import ByteString, Iterable, MutableMapping

import tensorrt as trt
import torch

from tensorrt_llm._utils import get_extra_attr, np_dtype_to_trt, set_extra_attr
from tensorrt_llm.logger import logger
from tensorrt_llm.network import PluginInfo, get_plugin_info

LAYER_TYPE_2_CLASS = {
    trt.LayerType.ACTIVATION: trt.IActivationLayer,
    trt.LayerType.CONCATENATION: trt.IConcatenationLayer,
    trt.LayerType.CONSTANT: trt.IConstantLayer,
    trt.LayerType.ELEMENTWISE: trt.IElementWiseLayer,
    trt.LayerType.FILL: trt.IFillLayer,
    trt.LayerType.GATHER: trt.IGatherLayer,
    trt.LayerType.MATRIX_MULTIPLY: trt.IMatrixMultiplyLayer,
    trt.LayerType.REDUCE: trt.IReduceLayer,
    trt.LayerType.SELECT: trt.ISelectLayer,
    trt.LayerType.SHUFFLE: trt.IShuffleLayer,
    trt.LayerType.SLICE: trt.ISliceLayer,
    trt.LayerType.SOFTMAX: trt.ISoftMaxLayer,
    trt.LayerType.UNARY: trt.IUnaryLayer,
    trt.LayerType.SHAPE: trt.IShapeLayer,
    trt.LayerType.ASSERTION: trt.IAssertionLayer,
    trt.LayerType.CAST: trt.ICastLayer,
    trt.LayerType.NORMALIZATION: trt.INormalizationLayer,
    trt.LayerType.IDENTITY: trt.IIdentityLayer,
    trt.LayerType.PLUGIN_V2: trt.IPluginV2Layer,
}


def to_subclass_layer(trt_layer):
    trt_layer.__class__ = LAYER_TYPE_2_CLASS[trt_layer.type]


def to_base_class_layer(trt_layer):
    trt_layer.__class__ = trt.ILayer


def to_trt_weights(ndarray):
    weight = trt.Weights(
        np_dtype_to_trt(ndarray.dtype),
        ndarray.ctypes.data,
        ndarray.size,
    )
    # Prevent numpy array from going out of weight's lifetime scope
    set_extra_attr(weight, "numpy", ndarray)
    return weight


@contextlib.contextmanager
def silent_trt_logger():
    min_severity = logger.trt_logger.min_severity
    logger.trt_logger.min_severity = trt.Logger.ERROR
    yield
    logger.trt_logger.min_severity = min_severity


def compare_tensor(trt_tensor, new_trt_tensor):
    assert trt_tensor.name == new_trt_tensor.name
    assert trt_tensor.dtype == new_trt_tensor.dtype
    assert tuple(trt_tensor.shape) == tuple(new_trt_tensor.shape)
    assert trt_tensor.broadcast_across_batch == new_trt_tensor.broadcast_across_batch
    assert trt_tensor.location == new_trt_tensor.location
    assert trt_tensor.is_network_input == new_trt_tensor.is_network_input
    assert trt_tensor.is_network_output == new_trt_tensor.is_network_output
    assert trt_tensor.dynamic_range == new_trt_tensor.dynamic_range
    assert trt_tensor.is_shape_tensor == new_trt_tensor.is_shape_tensor
    assert trt_tensor.is_execution_tensor == new_trt_tensor.is_execution_tensor
    assert trt_tensor.allowed_formats == new_trt_tensor.allowed_formats


def compare_network(trt_network, new_trt_network):
    assert trt_network.num_inputs == new_trt_network.num_inputs
    for i in range(trt_network.num_inputs):
        input = trt_network.get_input(i)
        new_input = new_trt_network.get_input(i)
        compare_tensor(input, new_input)
    assert trt_network.num_outputs == new_trt_network.num_outputs
    for i in range(trt_network.num_outputs):
        output = trt_network.get_output(i)
        new_output = new_trt_network.get_output(i)
        compare_tensor(output, new_output)
    assert trt_network.num_layers == new_trt_network.num_layers
    for index, new_index in zip(get_sorted_layer_ids(trt_network),
                                get_sorted_layer_ids(new_trt_network)):
        layer = trt_network.get_layer(index)
        new_layer = new_trt_network.get_layer(new_index)
        assert layer.name == new_layer.name
        assert layer.type == new_layer.type
        assert layer.precision_is_set == new_layer.precision_is_set
        assert layer.precision == new_layer.precision
        assert layer.num_inputs == new_layer.num_inputs
        for j in range(layer.num_inputs):
            input = layer.get_input(j)
            new_input = new_layer.get_input(j)
            if input is None:
                assert new_input is None
            else:
                assert new_input is not None
                compare_tensor(input, new_input)
        assert layer.num_outputs == new_layer.num_outputs
        for j in range(layer.num_outputs):
            output = layer.get_output(j)
            new_output = new_layer.get_output(j)
            compare_tensor(output, new_output)
            assert layer.output_type_is_set(j) == new_layer.output_type_is_set(
                j)
            if layer.output_type_is_set(j):
                assert layer.get_output_type(j) == new_layer.get_output_type(j)


def get_sorted_layer_ids(trt_network):
    inputs = set()
    for i in range(trt_network.num_inputs):
        inputs.add(trt_network.get_input(i).name)
    layer_ids = [*range(trt_network.num_layers)]
    sorted_layer_ids = []
    walked_tensors = set(inputs)
    while len(layer_ids) > 0:
        layer_id = layer_ids.pop(0)
        layer = trt_network.get_layer(layer_id)
        no_dependencies = True
        for j in range(layer.num_inputs):
            input = layer.get_input(j)
            if input is None:
                continue
            if input.name in walked_tensors:
                continue
            else:
                no_dependencies = False
                break
        if no_dependencies:
            sorted_layer_ids.append(layer_id)
            for j in range(layer.num_outputs):
                output = layer.get_output(j)
                if output is None:
                    continue
                walked_tensors.add(output.name)
        else:
            layer_ids.append(layer_id)
    assert len(sorted_layer_ids) == trt_network.num_layers
    return sorted_layer_ids


def to_tuple(values):
    if isinstance(values, (int, float, str, bool, NoneType, ByteString)):
        return values
    elif isinstance(values, (trt.Dims, trt.Permutation)):
        if values.__len__() < 0:
            return None
        else:
            return tuple(values)
    elif isinstance(values, Iterable):
        return tuple(to_tuple(v) for v in values)
    elif isinstance(values, MutableMapping):
        return tuple((k, to_tuple(v)) for k, v in values.items())
    else:
        return values


_base_layer_attr_names = set(dir(trt.ILayer))


def get_cache_key(layer, shapes, values, dtypes=None, updated_attrs=None):
    updated_attrs = updated_attrs or {}
    layer_type = layer.type
    to_subclass_layer(layer)
    attr_names = set(dir(layer)) - _base_layer_attr_names
    if layer_type == trt.LayerType.CONSTANT:
        attr_names.remove("weights")
    elif layer_type == trt.LayerType.SHUFFLE:
        if layer.num_inputs >= 2:
            attr_names.remove("reshape_dims")
    elif layer_type == trt.LayerType.SLICE:
        if layer.num_inputs >= 2 and layer.get_input(1) is not None:
            attr_names.remove("start")
        if layer.num_inputs >= 3 and layer.get_input(2) is not None:
            attr_names.remove("shape")
        if layer.num_inputs >= 4 and layer.get_input(3) is not None:
            attr_names.remove("stride")
    elif layer_type == trt.LayerType.FILL:
        attr_names.remove("is_alpha_beta_int64")
        if layer.num_inputs >= 1 and layer.get_input(0) is not None:
            attr_names.remove("shape")
        if layer.num_inputs >= 2 and layer.get_input(1) is not None:
            attr_names.remove("alpha")
        if layer.num_inputs >= 3 and layer.get_input(2) is not None:
            attr_names.remove("beta")
    if layer_type != trt.LayerType.PLUGIN_V2:
        attr_key = tuple(
            (name, to_tuple(updated_attrs.get(name) or getattr(layer, name)))
            for name in sorted(attr_names))
    else:
        network = get_trt_network(layer)
        plugin_info = get_plugin_info(network, layer.name)
        assert plugin_info is not None, f"layer {layer.name} does not register plugin info"
        attr_key = tuple(
            (name, tuple(updated_attrs.get(name) or data))
            for name, data in sorted(plugin_info.pfc_as_list.items()))
    to_base_class_layer(layer)
    shape_key = ()
    value_key = ()
    dtype_key = ()
    for i in range(layer.num_inputs):
        input = layer.get_input(i)
        if input is not None:
            shape_key += (tuple(shapes[input.name]), )
            if input.name in values:
                value = values[input.name]
                # All torch tensors are derived from input shapes and pfc,
                # thus we ignore them in cache key
                if isinstance(value, torch.Tensor):
                    value = None
                else:
                    value = tuple(value)
                value_key += (value, )
            else:
                value_key += (None, )
            if dtypes is not None:
                dtype_key += (dtypes[input.name], )
        else:
            shape_key += (None, )
            value_key += (None, )
            dtype_key += (None, )
    if dtypes is not None:
        for i in range(layer.num_outputs):
            output = layer.get_output(i)
            dtype_key += (dtypes[output.name], )
    cache_key = (layer.type, attr_key, shape_key, value_key)
    if dtypes is not None:
        cache_key += (dtype_key, )
    return cache_key


def get_trt_network(layer: trt.ILayer):
    network = get_extra_attr(layer, "network")
    assert network is not None
    return network


def set_trt_network(layer: trt.ILayer, network: trt.INetworkDefinition):
    set_extra_attr(layer, "network", network)


def get_updated_plugin(plugin_info: PluginInfo, updated_attrs):
    fields = []
    for field in plugin_info.pfc:
        name = field.name
        if name in updated_attrs:
            field = trt.PluginField(name, updated_attrs[name], field.type)
        else:
            field = trt.PluginField(name, plugin_info.pfc_as_ndarray[name],
                                    field.type)
        fields.append(field)
    pfc = trt.PluginFieldCollection(fields)
    plugin = plugin_info.plugin_creator.create_plugin(plugin_info.plugin_name,
                                                      pfc)
    new_plugin_info = PluginInfo(plugin_info.plugin_creator,
                                 plugin_info.plugin_name, pfc)
    return plugin, new_plugin_info


_builder_flags = threading.local()
_strongly_typed = threading.local()


def get_builder_flags():
    return getattr(_builder_flags, 'value', 0)


def get_strongly_typed():
    return getattr(_strongly_typed, 'value', False)


@contextlib.contextmanager
def current_flags(builder_flags, strongly_typed):
    previous_builder_flags = get_builder_flags()
    _builder_flags.value = builder_flags
    previous_strongly_typed = get_strongly_typed()
    _strongly_typed.value = strongly_typed
    yield
    _builder_flags.value = previous_builder_flags
    _strongly_typed.value = previous_strongly_typed


def get_engine_information(engine_file) -> str:
    with open(engine_file, "rb") as f:
        engine_buffer = f.read()
    runtime = trt.Runtime(logger.trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_buffer)
    inspector = engine.create_engine_inspector()
    return inspector.get_engine_information(trt.LayerInformationFormat.JSON)


def print_engine_info(engine_file) -> dict:
    with open(engine_file, "rb") as f:
        engine_buffer = f.read()
    from tensorrt_llm.runtime.session import Session
    Session.from_serialized_engine(engine_buffer)._print_engine_info()

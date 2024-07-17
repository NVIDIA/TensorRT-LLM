from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set

import numpy as np
import tensorrt as trt
import torch

from tensorrt_llm._common import _is_building
from tensorrt_llm._utils import (trt_dtype_to_np, trt_dtype_to_str,
                                 trt_dtype_to_torch)
from tensorrt_llm.logger import logger

from .pipeline_graph import PipelineGraph
from .utils import (get_builder_flags, get_cache_key, get_sorted_layer_ids,
                    set_trt_network, to_base_class_layer, to_subclass_layer,
                    to_trt_weights)


class ShapeType(Enum):
    MIN = 0
    OPT = 1
    MAX = 2


_trt_to_type_dict = {
    trt.int64: int,
    trt.bool: bool,
}


def get_shape_layers(trt_network):
    shape_layers = set()
    for i in range(trt_network.num_layers):
        layer = trt_network.get_layer(i)
        if (layer.num_inputs > 0 and np.all([
                layer.get_input(j).is_shape_tensor
                for j in range(layer.num_inputs)
                if layer.get_input(j) is not None
        ])) or (layer.num_outputs > 0 and np.all([
                layer.get_output(j).is_shape_tensor
                for j in range(layer.num_outputs)
        ])):
            shape_layers.add(layer.name)
    return shape_layers


def get_layers_in_shape_network(trt_network, shape_layers, sorted_layer_ids):
    layers = set()
    shape_tensors = set()
    for layer_id in reversed(sorted_layer_ids):
        layer = trt_network.get_layer(layer_id)
        in_shape_network = False
        if layer.name in shape_layers:
            in_shape_network = True
        else:
            for j in range(layer.num_outputs):
                output = layer.get_output(j)
                if output.name in shape_tensors:
                    in_shape_network = True
                    break
        if in_shape_network:
            layers.add(layer.name)
            for j in range(layer.num_inputs):
                input = layer.get_input(j)
                if input is not None:
                    shape_tensors.add(input.name)
    return layers


def get_shape_network(trt_network,
                      shapes,
                      values,
                      sorted_layer_ids,
                      profile=None,
                      shape_type: ShapeType = ShapeType.OPT):
    shape_layers = get_shape_layers(trt_network)
    layers_in_shape_network = get_layers_in_shape_network(
        trt_network, shape_layers, sorted_layer_ids)
    shape_graph = PipelineGraph.create_graph()
    shape_network = shape_graph.as_trt()
    shape_builder = shape_network.builder
    shape_profile = shape_builder.create_optimization_profile()
    for i in range(trt_network.num_inputs):
        input = trt_network.get_input(i)
        shapes[input.name] = input.shape
        new_input = shape_graph.add_input(input)
        if profile is not None:
            if -1 in input.shape:
                shape = profile.get_shape(input.name)
                shape = shape[shape_type.value]
                shapes[input.name] = shape
                new_input.raw_shape = shape
            if input.is_shape_tensor:
                shape_values = profile.get_shape_input(input.name)
                value = shape_values[shape_type.value]
                values[input.name] = value
                shape_profile.set_shape_input(input.name, value, value, value)
    output_mapping = {}
    for layer_id in sorted_layer_ids:
        layer = trt_network.get_layer(layer_id)
        if layer.name in shape_layers:
            new_layer = shape_graph.add_layer(layer)
            for i in range(layer.num_outputs):
                output = layer.get_output(i)
                if output.dtype == trt.DataType.BOOL:
                    proxy_layer = shape_network.add_cast(
                        new_layer.as_trt().get_output(i),
                        trt.DataType.INT32,
                    )
                    proxy_output = proxy_layer.get_output(0)
                    shape_graph.register_layer(proxy_layer)
                    shape_graph.add_output_shape(proxy_output)
                    output_mapping[proxy_output.name] = (output.name,
                                                         output.dtype)
                else:
                    shape_graph.add_output_shape(output)
        elif layer.name in layers_in_shape_network:
            if layer.type == trt.LayerType.CONSTANT:
                shape_graph.add_input(layer.get_output(0))
            else:
                shape_graph.add_layer(layer)
    return shape_network, shape_profile, shape_layers, output_mapping


def get_per_layer_graph(
    layer,
    shapes,
    values,
    updated_attrs=None,
    is_shape_io: bool = None,
):
    graph = PipelineGraph.create_graph()
    network = graph.as_trt()
    is_shape_layer = layer.num_inputs != 0
    for i in range(layer.num_inputs):
        input = layer.get_input(i)
        if input is not None:
            shape = shapes[input.name]
            if (values.get(input.name) is not None
                    and not isinstance(values[input.name], torch.Tensor)):
                value = values[input.name]
                weights = np.asarray(value, dtype=trt_dtype_to_np(input.dtype))
                weights = to_trt_weights(weights)
                input_layer = network.add_constant(shape, weights)
                new_input = input_layer.get_output(0)
                new_input.name = input.name
                graph.register_layer(input_layer)
            elif graph.get_input(input.name) is None:
                new_input = graph.add_input(input)
                new_input.raw_shape = shapes[input.name]
                is_shape_layer = False
    new_layer = graph.add_layer(
        layer,
        updated_attrs=updated_attrs,
    )
    output_mapping = {}
    if layer.type == trt.LayerType.SHAPE:
        is_shape_layer = True
    if layer.num_inputs == 0:
        is_shape_layer = False
    if is_shape_io is not None:
        is_shape_layer = is_shape_io
    for i in range(layer.num_outputs):
        output = layer.get_output(i)
        value = values.get(output.name)
        if value is not None and isinstance(value, torch.Tensor):
            is_output_shape = False
        elif is_shape_layer:
            is_output_shape = True
        else:
            is_output_shape = False
        if is_output_shape:
            if output.dtype == trt.DataType.BOOL:
                proxy_layer = network.add_cast(
                    new_layer.as_trt().get_output(i),
                    trt.DataType.INT32,
                )
                proxy_output = proxy_layer.get_output(0)
                graph.register_layer(proxy_layer)
                output_mapping[proxy_output.name] = (output.name, output.dtype)
                output = proxy_output
            graph.add_output_shape(output)
        else:
            graph.add_output(output)
    return graph, output_mapping


@_is_building
def infer_shapes(network, shapes, values, profile=None):
    if network.num_outputs == 0:
        return
    builder = network.builder
    config = builder.create_builder_config()
    config.builder_optimization_level = 0
    config.flags = get_builder_flags()
    profile = profile or builder.create_optimization_profile()
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(
            'Engine building failed when inferring shapes, please check the error log.'
        )
    runtime = trt.Runtime(logger.trt_logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    for i in range(network.num_inputs):
        input = network.get_input(i)
        if input.is_shape_tensor:
            value = values[input.name]
            context.set_shape_input(engine[input.name], value)
    for i in range(network.num_outputs):
        output = network.get_output(i)
        shape = context.get_tensor_shape(output.name)
        shapes[output.name] = shape
        if output.is_shape_tensor:
            if shape == [0]:
                values[output.name] = []
            else:
                if shape == []:
                    shape = [1]
                value = torch.empty(
                    list(shape),
                    dtype=trt_dtype_to_torch(output.dtype),
                    device="cpu",
                )
                values[output.name] = value
                context.set_tensor_address(output.name, value.data_ptr())
    context.infer_shapes()
    assert context.all_binding_shapes_specified
    for i in range(network.num_outputs):
        output = network.get_output(i)
        if isinstance(values.get(output.name), torch.Tensor):
            values[output.name] = values[output.name].tolist()


@dataclass
class ShapeInfo:
    shapes: Dict[str, trt.Dims]
    values: Dict[str, List[int]]
    shape_layers: Set[str]
    max_shapes: Dict[str, trt.Dims] = None


def set_constant_value(layer, values):
    to_subclass_layer(layer)
    output_name = layer.get_output(0).name
    weights = layer.weights
    if isinstance(weights, trt.Weights):
        weights = weights.numpy()
    values[output_name] = list(weights)
    to_base_class_layer(layer)


def infer_per_layer_shapes(
    layer: trt.ILayer,
    shapes,
    values,
    cache=None,
    is_shape_io=False,
):
    if layer.type == trt.LayerType.CONSTANT:
        to_subclass_layer(layer)
        output_name = layer.get_output(0).name
        shape = layer.shape
        shapes[output_name] = shape
        if is_shape_io:
            set_constant_value(layer, values)
        to_base_class_layer(layer)
        return
    elif layer.type == trt.LayerType.SHAPE:
        input_name = layer.get_input(0).name
        output_name = layer.get_output(0).name
        shape = [*shapes[input_name]]
        shapes[output_name] = trt.Dims([len(shape)])
        values[output_name] = shape
        return
    if cache is not None:
        cache_key = get_cache_key(layer, shapes, values)
        if cache_key in cache:
            output_shapes, output_values = cache[cache_key]
            for i in range(layer.num_outputs):
                output = layer.get_output(i)
                shapes[output.name] = output_shapes[i]
                if output_values[i] is not None:
                    values[output.name] = output_values[i]
            return
    graph, output_mapping = get_per_layer_graph(layer, shapes, values)
    dtypes = [
        trt_dtype_to_str(layer.get_input(i).dtype)
        for i in range(layer.num_inputs)
    ]
    layer_info = (f"type={cache_key[0]}, "
                  f"attrs={dict(cache_key[1])}, "
                  f"dtypes={dtypes}, "
                  f"shapes={list(cache_key[2])}, "
                  f"values={list(cache_key[3])}")
    logger.debug(f"infer shapes for layer {layer.name} ({layer_info})")
    try:
        infer_shapes(graph.as_trt(), shapes, values)
    except RuntimeError as e:
        raise RuntimeError(
            f"infer shapes failed for layer {layer.name} ({layer_info})") from e
    for proxy_output, (output, dtype) in output_mapping.items():
        shapes[output] = shapes[proxy_output]
        del shapes[proxy_output]
        if proxy_output in values:
            values[output] = [
                *map(_trt_to_type_dict[dtype], values[proxy_output])
            ]
            del values[proxy_output]
    if cache is not None:
        logger.debug(
            f"shape inference cache miss, layer: {layer.name}, cache key: {cache_key}"
        )
        output_shapes = []
        output_values = []
        for i in range(layer.num_outputs):
            output = layer.get_output(i)
            output_shapes.append(shapes[output.name])
            output_values.append(values.get(output.name))
        cache[cache_key] = (output_shapes, output_values)


def get_shape_info(trt_network, profile, shape_type: ShapeType = ShapeType.OPT):
    shapes = {}
    values = {}
    sorted_layer_ids = get_sorted_layer_ids(trt_network)
    infer_shape_layers = False

    shape_network, shape_profile, shape_layers, output_mapping = get_shape_network(
        trt_network,
        shapes,
        values,
        sorted_layer_ids,
        profile=profile,
        shape_type=shape_type)
    try:
        infer_shapes(shape_network, shapes, values, shape_profile)
        for proxy_output, (output, dtype) in output_mapping.items():
            shapes[output] = shapes[proxy_output]
            values[output] = [
                *map(_trt_to_type_dict[dtype], values[proxy_output])
            ]
            del shapes[proxy_output]
            del values[proxy_output]
    except RuntimeError:
        infer_shape_layers = True

    cache = {}
    for layer_id in sorted_layer_ids:
        layer = trt_network.get_layer(layer_id)
        is_shape_io = layer.name in shape_layers
        if is_shape_io and not infer_shape_layers:
            continue
        set_trt_network(layer, trt_network)
        infer_per_layer_shapes(layer,
                               shapes,
                               values,
                               cache,
                               is_shape_io=is_shape_io)
    return ShapeInfo(shapes, values, shape_layers)

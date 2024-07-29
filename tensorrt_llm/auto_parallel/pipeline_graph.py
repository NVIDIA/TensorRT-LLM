from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import tensorrt as trt
import torch

from tensorrt_llm._utils import trt_dtype_to_str, trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.network import Network, get_plugin_info, set_plugin_info
from tensorrt_llm.plugin.plugin import PluginConfig
from tensorrt_llm.runtime.session import Session

from .utils import (current_flags, get_builder_flags, get_sorted_layer_ids,
                    get_strongly_typed, get_trt_network, set_trt_network,
                    to_base_class_layer, to_subclass_layer)


class Tensor:

    def __init__(self, graph: "PipelineGraph"):
        self._graph = graph
        self._trt = None
        self._shape = None
        self._max_shape = None
        self._value = None
        self.producer: Layer = None
        self.output_index = None
        self.consumers = []
        self.graph_input_index = -1
        self.graph_output_index = -1
        self.attrs = {}

    @staticmethod
    def from_trt(graph: "PipelineGraph", trt_tensor: trt.ITensor):
        tensor = Tensor(graph)
        tensor._trt = trt_tensor
        return tensor

    def as_trt(self) -> trt.ITensor:
        return self._trt

    def copy(self) -> "Tensor":
        tensor = Tensor(self._graph)
        tensor._trt = self._trt
        tensor._shape = self._shape
        tensor._max_shape = self._max_shape
        tensor._value = self._value
        tensor.producer = self.producer
        tensor.output_index = self.output_index
        tensor.consumers = [*self.consumers]
        tensor.graph_input_index = self.graph_input_index
        tensor.graph_output_index = self.graph_output_index
        tensor.attrs = self.attrs.copy()
        return tensor

    @property
    def graph(self) -> "PipelineGraph":
        return self._graph

    @property
    def name(self) -> str:
        return self._trt.name

    @name.setter
    def name(self, name: str):
        old_name = self._trt.name
        if name != old_name:
            self._trt.name = name
            self.graph._tensors[name] = self
            del self.graph._tensors[old_name]
            if self.is_graph_input:
                self.graph._inputs[name] = self
                del self.graph._inputs[old_name]
            elif self.is_graph_output:
                self.graph._outputs[name] = self
                del self.graph._outputs[old_name]

    @property
    def shape(self):
        return self._shape

    @property
    def max_shape(self):
        return self._max_shape

    @property
    def raw_shape(self):
        assert isinstance(self._trt, trt.ITensor)
        return self._trt.shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @max_shape.setter
    def max_shape(self, max_shape):
        self._max_shape = max_shape

    @raw_shape.setter
    def raw_shape(self, raw_shape):
        assert isinstance(self._trt, trt.ITensor)
        self._trt.shape = raw_shape

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def dtype(self):
        return self._trt.dtype

    @property
    def broadcast_across_batch(self):
        return self._trt.broadcast_across_batch

    @property
    def dtype_size(self):
        return self.dtype.itemsize

    @property
    def dtype_str(self):
        return trt_dtype_to_str(self.dtype)

    @property
    def dtype_str_size(self):
        return [trt_dtype_to_str(self.dtype), self.dtype.itemsize]

    @property
    def is_graph_input(self) -> bool:
        return self.graph_input_index != -1

    @property
    def is_graph_output(self) -> bool:
        return self.graph_output_index != -1

    @property
    def is_graph_io(self) -> bool:
        return self.is_graph_input or self.is_graph_output


class Layer:

    def __init__(self, graph):
        self._graph = graph
        self._trt = None
        self._index = None
        self._inputs = []
        self._outputs = []
        self._is_shape_io = False
        self.attrs = {}

    @staticmethod
    def from_trt(graph, trt_layer, index):
        layer = Layer(graph)
        layer._trt = trt_layer
        layer._index = index
        for i in range(trt_layer.num_inputs):
            input = trt_layer.get_input(i)
            if input is not None:
                layer._inputs.append(graph.get_tensor(input.name))
                layer._inputs[i].consumers.append((layer, i))
            else:
                layer._inputs.append(None)
        for i in range(trt_layer.num_outputs):
            output = trt_layer.get_output(i)
            layer._outputs.append(graph.get_tensor(output.name))
            layer._outputs[i].producer = layer
            layer._outputs[i].output_index = i
        set_trt_network(trt_layer, graph.as_trt())
        return layer

    def as_trt(self) -> trt.ILayer:
        return self._trt

    @property
    def graph(self) -> "PipelineGraph":
        return self._graph

    @property
    def name(self) -> str:
        return self._trt.name

    @name.setter
    def name(self, name: str):
        old_name = self._trt.name
        if name != old_name:
            self._trt.name = name
            self.graph._layers[name] = self
            del self.graph._layers[old_name]

    @property
    def type(self) -> trt.LayerType:
        return self._trt.type

    @property
    def index(self) -> int:
        return self._index

    @property
    def inputs(self) -> List[Tensor]:
        return self._inputs

    @property
    def outputs(self) -> List[Tensor]:
        return self._outputs

    def get_input(self, index: int) -> Tensor:
        return self._inputs[index]

    def get_output(self, index: int) -> Tensor:
        return self._outputs[index]

    @property
    def num_inputs(self) -> int:
        return self._trt.num_inputs

    @property
    def num_outputs(self) -> int:
        return self._trt.num_outputs

    @property
    def is_shape_io(self) -> bool:
        return self._is_shape_io

    def to_subclass(self):
        to_subclass_layer(self._trt)

    def to_base_class(self):
        to_base_class_layer(self._trt)

    def assign_shapes(self, shapes, values):
        for output in self.outputs:
            output.shape = shapes[output.name]
            output.value = values.get(output.name)


@dataclass
class GraphRunner:
    session: Session
    inputs: Dict[str, torch.Tensor]
    outputs: Dict[str, torch.Tensor]
    stream: torch.Stream

    def run(self):
        cuda_stream = self.stream.cuda_stream
        assert self.session.run(self.inputs, self.outputs, cuda_stream)
        self.stream.synchronize()
        return self.outputs


class PipelineGraph:

    def __init__(self):
        self._trt = None
        self._inputs: Dict[str, Tensor] = {}
        self._outputs: Dict[str, Tensor] = {}
        self._layers: Dict[str, Layer] = {}
        self._tensors: Dict[str, Tensor] = {}
        self._io_buffer_mapping = {}
        self._unfilled_weights = {}
        self._auto_parallel_config = None
        self._plugin_config: PluginConfig = None

    @staticmethod
    def create_graph():
        graph = PipelineGraph()
        trt_builder = trt.Builder(logger.trt_logger)
        explicit_batch_flag = 0
        # Explicit batch flag will be deprecated in TRT 10
        if "EXPLICIT_BATCH" in trt.NetworkDefinitionCreationFlag.__members__.keys(
        ):
            explicit_batch_flag = 1 << int(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        if get_strongly_typed():
            network = trt_builder.create_network(
                explicit_batch_flag
                | (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)))
        else:
            network = trt_builder.create_network(explicit_batch_flag)
        graph._trt = network
        return graph

    def _register_unfilled_weights(self, layer_name, weights, values):
        self._unfilled_weights[layer_name] = (weights, values)

    def _add_tensor(self, tensor, old_tensor, prefix):
        if prefix is not None:
            tensor.name = prefix + old_tensor.name
        else:
            tensor.name = old_tensor.name
        tensor.location = old_tensor.location
        if old_tensor.dynamic_range is not None:
            tensor.dynamic_range = old_tensor.dynamic_range
        if tensor.is_network_input:
            tensor.shape = old_tensor.shape
            for i in range(len(old_tensor.shape)):
                name = old_tensor.get_dimension_name(i)
                if name is not None:
                    tensor.set_dimension_name(i, name)
        return self._register_tensor(tensor)

    def _register_tensor(self, tensor):
        wrapped_tensor = Tensor.from_trt(self, tensor)
        assert tensor.name not in self._tensors
        self._tensors[tensor.name] = wrapped_tensor
        return wrapped_tensor

    def add_input(self, tensor, prefix=None):
        tensor_name = tensor.name
        if prefix is not None:
            tensor_name = prefix + tensor_name
        input = self._trt.add_input(tensor_name, tensor.dtype, tensor.shape)
        new_tensor = self._add_tensor(input, tensor, prefix)
        new_tensor.graph_input_index = len(self._inputs)
        self._inputs[tensor_name] = new_tensor
        return new_tensor

    def register_input(self, tensor, index=None):
        if index is None:
            index = self.num_inputs - 1
        assert self._trt.get_input(index).name == tensor.name
        wrapped_input = self._register_tensor(tensor)
        wrapped_input.graph_input_index = index
        self._inputs[tensor.name] = wrapped_input
        return wrapped_input

    def add_output(self, tensor, prefix=None):
        tensor_name = tensor.name
        if prefix is not None:
            tensor_name = prefix + tensor_name
        output = self.get_tensor(tensor_name)
        output.graph_output_index = len(self._outputs)
        trt_output = output.as_trt()
        self._trt.mark_output(trt_output)
        trt_output.dtype = tensor.dtype
        self._outputs[tensor_name] = output
        return output

    def add_output_shape(self, tensor, prefix=None):
        tensor_name = tensor.name
        if prefix is not None:
            tensor_name = prefix + tensor_name
        output = self.get_tensor(tensor_name)
        trt_output = output.as_trt()
        self._trt.mark_output_for_shapes(trt_output)
        trt_output.dtype = tensor.dtype
        self._outputs[tensor_name] = output
        return output

    def add_layer(
        self,
        layer,
        input_mapping=None,
        prefix=None,
        updated_attrs=None,
    ) -> Layer:

        def get_input(i):
            name = layer.get_input(i).name
            if prefix is not None:
                name = prefix + name
            if input_mapping is not None and name in input_mapping:
                name = input_mapping[name]
            return self.get_tensor(name).as_trt()

        network = self._trt
        layer_type = layer.type
        to_subclass_layer(layer)
        if layer_type == trt.LayerType.ACTIVATION:
            trt_input = get_input(0)
            new_layer = network.add_activation(trt_input, layer.type)
            new_layer.alpha = layer.alpha
            new_layer.beta = layer.beta
        elif layer_type == trt.LayerType.CONCATENATION:
            trt_inputs = [get_input(i) for i in range(layer.num_inputs)]
            new_layer = network.add_concatenation(trt_inputs)
            new_layer.axis = layer.axis
        elif layer_type == trt.LayerType.CONSTANT:
            new_layer = network.add_constant(layer.shape, layer.weights)
        elif layer_type == trt.LayerType.ELEMENTWISE:
            new_layer = network.add_elementwise(get_input(0), get_input(1),
                                                layer.op)
        elif layer_type == trt.LayerType.FILL:
            if layer.num_inputs >= 1 and layer.get_input(0) is not None:
                shape_input = get_input(0)
                shape = [1]
            else:
                shape_input = None
                shape = layer.shape
            new_layer = network.add_fill(shape, layer.operation, layer.to_type)
            if shape_input is not None:
                new_layer.set_input(0, shape_input)
            if layer.num_inputs >= 1 and layer.get_input(0) is not None:
                new_layer.set_input(0, get_input(0))
            if layer.num_inputs >= 2 and layer.get_input(1) is not None:
                new_layer.set_input(1, get_input(1))
            else:
                new_layer.alpha = layer.alpha
            if layer.num_inputs >= 3 and layer.get_input(2) is not None:
                new_layer.set_input(2, get_input(2))
            else:
                new_layer.beta = layer.beta
        elif layer_type == trt.LayerType.GATHER:
            trt_input = get_input(0)
            trt_indices = get_input(1)
            new_layer = network.add_gather_v2(trt_input, trt_indices,
                                              layer.mode)
            new_layer.axis = layer.axis
            new_layer.num_elementwise_dims = layer.num_elementwise_dims
            new_layer.mode = layer.mode
        elif layer_type == trt.LayerType.MATRIX_MULTIPLY:
            new_layer = network.add_matrix_multiply(get_input(0), layer.op0,
                                                    get_input(1), layer.op1)
        elif layer_type == trt.LayerType.REDUCE:
            new_layer = network.add_reduce(get_input(0), layer.op, layer.axes,
                                           layer.keep_dims)
        elif layer_type == trt.LayerType.SELECT:
            trt_condition = get_input(0)
            trt_then = get_input(1)
            trt_else = get_input(2)
            new_layer = network.add_select(trt_condition, trt_then, trt_else)
        elif layer_type == trt.LayerType.SHUFFLE:
            new_layer = network.add_shuffle(get_input(0))
            new_layer.first_transpose = layer.first_transpose
            new_layer.second_transpose = layer.second_transpose
            new_layer.zero_is_placeholder = layer.zero_is_placeholder
            if layer.num_inputs >= 2:
                trt_reshape_dims_tensor = get_input(1)
                new_layer.set_input(1, trt_reshape_dims_tensor)
            else:
                new_layer.reshape_dims = layer.reshape_dims
        elif layer_type == trt.LayerType.SLICE:
            if layer.num_inputs >= 2 and layer.get_input(1) is not None:
                trt_start = get_input(1)
                start = []
            else:
                trt_start = None
                start = layer.start
            if layer.num_inputs >= 3 and layer.get_input(2) is not None:
                trt_shape = get_input(2)
                shape = []
            else:
                trt_shape = None
                shape = layer.shape
            if layer.num_inputs >= 4 and layer.get_input(3) is not None:
                trt_stride = get_input(3)
                stride = []
            else:
                trt_stride = None
                stride = layer.stride
            new_layer = network.add_slice(get_input(0), start, shape, stride)
            new_layer.mode = layer.mode
            if trt_start is not None:
                new_layer.set_input(1, trt_start)
            if trt_shape is not None:
                new_layer.set_input(2, trt_shape)
            if trt_stride is not None:
                new_layer.set_input(3, trt_stride)
        elif layer_type == trt.LayerType.SOFTMAX:
            new_layer = network.add_softmax(get_input(0))
            new_layer.axes = layer.axes
        elif layer_type == trt.LayerType.UNARY:
            new_layer = network.add_unary(get_input(0), layer.op)
        elif layer_type == trt.LayerType.SHAPE:
            new_layer = network.add_shape(get_input(0))
        elif layer_type == trt.LayerType.ASSERTION:
            new_layer = network.add_assertion(get_input(0), layer.message)
        elif layer_type == trt.LayerType.CAST:
            new_layer = network.add_cast(get_input(0), layer.to_type)
        elif layer_type == trt.LayerType.NORMALIZATION:
            trt_input = get_input(0)
            trt_scale = get_input(1)
            trt_bias = get_input(2)
            new_layer = network.add_normalization(trt_input, trt_scale,
                                                  trt_bias, layer.axes)
            new_layer.epsilon = layer.epsilon
            new_layer.num_groups = layer.num_groups
            new_layer.compute_precision = layer.compute_precision
        elif layer_type == trt.LayerType.IDENTITY:
            new_layer = network.add_identity(get_input(0))
        elif layer_type == trt.LayerType.PLUGIN_V2:
            plugin = layer.plugin
            updated = False
            if (updated_attrs is not None
                    and updated_attrs.get("plugin") is not None):
                plugin = updated_attrs["plugin"]
                updated = True
            updated_attrs = None
            new_layer = network.add_plugin_v2(
                [get_input(i) for i in range(layer.num_inputs)],
                plugin,
            )
        else:
            raise NotImplementedError(
                "Unsupported layer type: {}".format(layer_type))

        if updated_attrs is not None:
            for attr_name, attr_value in updated_attrs.items():
                setattr(new_layer, attr_name, attr_value)

        to_base_class_layer(layer)
        to_base_class_layer(new_layer)
        layer_index = network.num_layers - 1
        layer_name = layer.name
        if prefix is not None:
            layer_name = prefix + layer_name
        new_layer.name = layer_name
        new_layer.metadata = new_layer.name
        if layer.precision_is_set:
            new_layer.precision = layer.precision
        for i in range(layer.num_outputs):
            if layer.output_type_is_set(i):
                new_layer.set_output_type(i, layer.get_output_type(i))
            output = new_layer.get_output(i)
            self._add_tensor(output, layer.get_output(i), prefix)
        wrapped_layer = Layer.from_trt(self, new_layer, layer_index)
        assert layer_name not in self._layers
        self._layers[layer_name] = wrapped_layer
        if layer_type == trt.LayerType.PLUGIN_V2:
            if not updated:
                plugin_info = get_plugin_info(get_trt_network(layer),
                                              layer.name)
                set_plugin_info(self.as_trt(), new_layer.name, plugin_info)
        return wrapped_layer

    def register_layer(self, layer, index=None):
        if index is None:
            index = self.num_layers - 1
        assert self._trt.get_layer(index).name == layer.name
        to_base_class_layer(layer)
        for i in range(layer.num_outputs):
            output = layer.get_output(i)
            self._register_tensor(output)
        wrapped_layer = Layer.from_trt(self, layer, index)
        assert layer.name not in self._layers
        self._layers[layer.name] = wrapped_layer
        to_subclass_layer(layer)
        return wrapped_layer

    def get_runner(
        self,
        shapes=None,
        values=None,
        profile=None,
        timing_cache=None,
        opt_level=None,
    ) -> GraphRunner:
        shapes = shapes or {}
        values = values or {}
        inputs = {}
        outputs = {}
        for input in self.inputs:
            if input is not None:
                value = values.get(input.name)
                if value is None:
                    value = input.value
                if value is not None:
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor(
                            value,
                            dtype=trt_dtype_to_torch(input.dtype),
                            device='cpu',
                        )
                    inputs[input.name] = value
                else:
                    shape = shapes.get(input.name)
                    if shape is None:
                        shape = input.shape
                    assert shape is not None
                    inputs[input.name] = torch.empty(
                        tuple(shape),
                        dtype=trt_dtype_to_torch(input.dtype),
                        device=torch.cuda.current_device(),
                    )
                    if torch.is_floating_point(inputs[input.name]):
                        inputs[input.name].normal_()
                    # inputs[input.name][:] = random.choice([2, 3, 5, 7])
        for output in self.outputs:
            if output.as_trt().is_shape_tensor:
                continue
            if output.name in self._io_buffer_mapping:
                input_name = self._io_buffer_mapping[output.name]
                if input_name in inputs:
                    outputs[output.name] = inputs[input_name]
                    continue
            value = values.get(output.name)
            if value is not None and isinstance(value, torch.Tensor):
                outputs[output.name] = value
            else:
                shape = shapes.get(output.name)
                if shape is None:
                    shape = output.shape
                assert shape is not None
                outputs[output.name] = torch.empty(
                    tuple(shape),
                    dtype=trt_dtype_to_torch(output.dtype),
                    device=torch.cuda.current_device(),
                )
        network = self.as_trt()
        config = network.builder.create_builder_config()
        if opt_level is not None:
            config.builder_optimization_level = opt_level
        config.flags = get_builder_flags()
        profile = profile or network.builder.create_optimization_profile()
        profile_index = config.add_optimization_profile(profile)
        if timing_cache is not None:
            config.set_timing_cache(timing_cache, ignore_mismatch=False)
        plan = network.builder.build_serialized_network(network, config)
        if plan is None:
            logger.error('Engine building failed, please check the error log.')
        session = Session.from_serialized_engine(plan)
        stream = torch.cuda.current_stream()
        cuda_stream = stream.cuda_stream
        context = session.context
        context.set_optimization_profile_async(profile_index, cuda_stream)
        runner = GraphRunner(session, inputs, outputs, stream)
        return runner

    def run(
        self,
        shapes=None,
        values=None,
        profile=None,
        timing_cache=None,
        opt_level=None,
    ):
        return self.get_runner(
            shapes,
            values,
            profile,
            timing_cache,
            opt_level,
        ).run()

    def duplicate_graph(self):
        graph = PipelineGraph.create_graph()
        network = self.as_trt()
        for i in range(network.num_inputs):
            input = network.get_input(i)
            graph.add_input(input)
        sorted_layer_ids = get_sorted_layer_ids(network)
        for i in sorted_layer_ids:
            layer = network.get_layer(i)
            graph.add_layer(layer)
        for i in range(network.num_outputs):
            output = network.get_output(i)
            if output.is_shape_tensor:
                graph.add_output_shape(output)
            else:
                graph.add_output(output)
        return graph

    @staticmethod
    def from_trt(trt_network):
        graph = PipelineGraph()
        graph._trt = trt_network

        # construct inputs and tensors
        for i in range(trt_network.num_inputs):
            trt_input = trt_network.get_input(i)
            tensor = Tensor.from_trt(graph, trt_input)
            tensor.graph_input_index = i
            graph._tensors[tensor.name] = tensor
            graph._inputs[tensor.name] = tensor
        for i in range(trt_network.num_layers):
            trt_layer = trt_network.get_layer(i)
            for i in range(trt_layer.num_outputs):
                trt_output = trt_layer.get_output(i)
                tensor = Tensor.from_trt(graph, trt_output)
                graph._tensors[tensor.name] = tensor

        # construct layers and outputs
        for i in range(trt_network.num_layers):
            layer = Layer.from_trt(graph, trt_network.get_layer(i), i)
            graph._layers[layer.name] = layer
        for i in range(trt_network.num_outputs):
            tensor_name = trt_network.get_output(i).name
            output_tensor = graph._tensors[tensor_name]
            output_tensor.graph_output_index = i
            graph._outputs[tensor_name] = output_tensor

        return graph

    @staticmethod
    def from_network(network: Network, builder_config):
        builder_flags = builder_config.trt_builder_config.flags
        with current_flags(builder_flags, network.strongly_typed):
            graph = PipelineGraph.from_trt(network.trt_network)
            graph.infer_shapes(network._generate_optimization_profiles()[-1])
            return graph

    def assign_shapes(self, shape_info=None, is_partial=False):
        if shape_info is None:
            for tensor in self.tensors:
                tensor.shape = tensor.raw_shape
            return
        for tensor in self.tensors:
            if tensor.name in shape_info.shapes:
                tensor.shape = shape_info.shapes[tensor.name]
            elif not is_partial:
                raise ValueError(f"Cannot find shape for tensor: {tensor.name}")
            if shape_info.max_shapes is not None:
                if tensor.name in shape_info.max_shapes:
                    tensor.max_shape = shape_info.max_shapes[tensor.name]
                elif not is_partial:
                    raise ValueError(
                        f"Cannot find max shape for tensor: {tensor.name}")
            if tensor.name in shape_info.values:
                tensor.value = shape_info.values[tensor.name]
        for layer in self.layers:
            if layer.name in shape_info.shape_layers:
                layer._is_shape_io = True

    def infer_shapes(self, profile=None):
        from .shape_info import get_shape_info

        shape_info = get_shape_info(self._trt, profile)
        self.assign_shapes(shape_info)

    def as_trt(self) -> trt.INetworkDefinition:
        return self._trt

    def get_input(self, name: str) -> Tensor:
        return self._inputs.get(name)

    def is_input(self, name: str) -> bool:
        return name in self._inputs

    @property
    def inputs(self) -> List[Tensor]:
        return [*self._inputs.values()]

    @property
    def num_inputs(self) -> int:
        return self._trt.num_inputs

    def get_output(self, name: str) -> Tensor:
        return self._outputs.get(name)

    def is_output(self, name: str) -> bool:
        return name in self._outputs

    @property
    def outputs(self) -> List[Tensor]:
        return [*self._outputs.values()]

    @property
    def num_outputs(self) -> int:
        return self._trt.num_outputs

    def get_tensor(self, name: str) -> Tensor:
        return self._tensors.get(name)

    @property
    def tensors(self) -> List[Tensor]:
        return [*self._tensors.values()]

    def get_layer(self, name: str) -> Layer:
        return self._layers.get(name)

    @property
    def layers(self) -> List[Layer]:
        return [*self._layers.values()]

    @property
    def sorted_layers(self) -> List[Layer]:
        sorted_layer_ids = get_sorted_layer_ids(self.as_trt())
        return [
            self.get_layer(self.as_trt().get_layer(layer_id).name)
            for layer_id in sorted_layer_ids
        ]

    @property
    def num_layers(self) -> int:
        return self._trt.num_layers

    def to_dot(self,
               path=None,
               per_device=False,
               per_block=False,
               ignore_shape_io=False,
               no_style=False,
               extra_attrs=None) -> Optional[str]:
        '''
        Get a graphviz representation of the graph.

        Parameters:
            path: the path to save the graphviz file, if not provided, will return the graphviz source code
        '''
        try:
            import graphviz
        except ImportError:
            logger.error(
                "Failed to import graphviz, please install graphviz to enable PipelineGraph.to_dot()"
            )
            return

        extra_attrs = extra_attrs or []

        graph = graphviz.Digraph()
        input_block_graph = graphviz.Digraph(name='cluster_inputs')
        output_block_graph = graphviz.Digraph(name='cluster_outputs')
        device_graphs = {}
        block_graphs = {}
        block_graph_mapping = []
        tensor_names = set()
        layer_names = set()

        common_style = dict(fontname='Arial', )
        node_style = dict(
            **common_style,
            style='rounded,filled,bold',
        )
        tensor_style = dict(
            **node_style,
            shape='ellipse',
            fillcolor='white',
        )
        input_tensor_style = {**tensor_style, 'fillcolor': 'green'}
        output_tensor_style = {**tensor_style, 'fillcolor': 'lightgreen'}
        layer_style = dict(
            **node_style,
            shape='box',
            fillcolor='white',
        )
        shape_layer_style = {**layer_style, 'fillcolor': 'grey'}
        helper_layer_style = {**layer_style, 'fillcolor': 'lightgrey'}
        graph_style = dict(
            **common_style,
            style='rounded',
            penwidth='5',
            fontsize='28',
        )
        device_graph_style = dict(
            **graph_style,
            color='cornflowerblue',
        )
        block_graph_style = dict(
            **graph_style,
            color='darkcyan',
        )
        input_block_style = dict(
            **graph_style,
            color='green',
        )
        output_block_style = dict(
            **graph_style,
            color='lightgreen',
        )
        if no_style:
            device_graph_style = {}
            block_graph_style = {}
            input_block_style = {}
            output_block_style = {}
        input_block_graph.attr(label='inputs', **input_block_style)
        output_block_graph.attr(label='outputs', **output_block_style)

        def get_tensor_labels(tensor):
            labels = []
            if tensor.value is not None:
                labels.append(f"value={tensor.value}")
            else:
                labels.append(f"dtype={tensor.dtype.name}{tensor.shape}")
            for attr_name in extra_attrs:
                if attr_name in tensor.attrs:
                    labels.append(f"{attr_name}={tensor.attrs[attr_name]}")
            return labels

        def get_device_graph(name):
            if per_device and name.startswith('device'):
                device_name = name.split('_')[0]
                if device_name not in device_graphs:
                    device_graph = graphviz.Digraph(name='cluster_' +
                                                    device_name)
                    device_graph.attr(label=device_name, **device_graph_style)
                    device_graphs[device_name] = device_graph
                return device_graphs[device_name]
            return None

        def get_block_graph(layer, current_graph):
            if per_block and 'block_id' in layer.attrs:
                block_label = f"block{layer.attrs['block_id']}"
                if current_graph.name is not None:
                    graph_label = current_graph.name[len('cluster_'):]
                else:
                    graph_label = ''
                block_name = f"{graph_label}{block_label}"
                if block_name not in block_graphs:
                    block_graph = graphviz.Digraph(name='cluster_' + block_name)
                    block_graph.attr(label=block_label, **block_graph_style)
                    block_graphs[block_name] = block_graph
                    block_graph_mapping.append((current_graph, block_graph))
                return block_graphs[block_name]
            return current_graph

        for name, tensor in self._tensors.items():
            style = tensor_style
            if tensor.is_graph_input:
                style = input_tensor_style
                current_graph = input_block_graph
            elif tensor.is_graph_output:
                style = output_tensor_style
                current_graph = output_block_graph
            elif tensor.producer.num_outputs == 1:
                continue
            else:
                current_graph = get_device_graph(name) or graph
                current_graph = get_block_graph(tensor.producer, current_graph)
            if no_style:
                style = {}
            labels = [name, *get_tensor_labels(tensor)]
            content = "\n".join(labels)
            current_graph.node(name, content, **style)
            tensor_names.add(name)

        for layer in self.sorted_layers:
            name = layer.name

            style = layer_style
            if layer.is_shape_io:
                if ignore_shape_io:
                    continue
                style = shape_layer_style
            elif layer.attrs.get("role", None) == "helper":
                style = helper_layer_style
            fillcolor = None
            plugin_type = None
            if layer.type == trt.LayerType.PLUGIN_V2:
                fillcolor = 'yellow'
                layer.to_subclass()
                plugin_type = layer.as_trt().plugin.plugin_type
                layer.to_base_class()
            if layer.type == trt.LayerType.MATRIX_MULTIPLY or plugin_type == 'Gemm':
                fillcolor = 'orange'
            if fillcolor is not None:
                style = {**style, 'fillcolor': fillcolor}
            if no_style:
                style = {}

            layer_attrs = {}
            layer_type = layer.type
            layer.to_subclass()
            if layer_type == trt.LayerType.CONSTANT:
                if not layer.is_shape_io:
                    if trt.volume(layer.get_output(0).shape) <= 8:
                        weights = layer.as_trt().weights
                        if isinstance(weights, trt.Weights):
                            weights = weights.numpy()
                        value = np.array2string(
                            weights,
                            formatter={'float_kind': lambda x: f"{x:.2e}"})
                        layer_attrs['value'] = value
            elif layer_type == trt.LayerType.SHUFFLE:
                for attr_name in ['first_transpose', 'second_transpose']:
                    attr_value = getattr(layer.as_trt(), attr_name)
                    if tuple(attr_value) != (0, 1, 2, 3, 4, 5, 6, 7):
                        tensor = layer.get_input(
                            0
                        ) if attr_name == 'first_transpose' else layer.get_output(
                            0)
                        layer_attrs[attr_name] = tuple(
                            attr_value)[:len(tensor.shape)]
                if layer.num_inputs < 2:
                    attr_value = layer.as_trt().reshape_dims
                    layer_attrs['reshape_dims'] = attr_value
            elif layer_type == trt.LayerType.SLICE:
                if layer.num_inputs < 2 or layer.get_input(1) is None:
                    layer_attrs['start'] = layer.as_trt().start
                if layer.num_inputs < 4 or layer.get_input(3) is None:
                    attr_value = layer.as_trt().stride
                    if attr_value != tuple(
                        [1] * len(layer.get_output(0).shape)):
                        layer_attrs['stride'] = attr_value
            layer.to_base_class()

            if layer.is_shape_io:
                labels = [layer.type.name]
            else:
                labels = [name, layer.type.name]
            for key, value in layer_attrs.items():
                labels.append(f"{key}={value}")
            for attr_name in extra_attrs:
                if attr_name in layer.attrs:
                    labels.append(f"{attr_name}={layer.attrs[attr_name]}")
            if layer.num_outputs == 1:
                output = layer.get_output(0)
                if output.name != f'{layer.name}_output_0':
                    labels.append(f"output={output.name}")
                labels.extend(get_tensor_labels(output))
            content = "\n".join(labels)

            current_graph = get_device_graph(name) or graph
            current_graph = get_block_graph(layer, current_graph)
            current_graph.node(name, content, **style)
            layer_names.add(name)

            for index, input in enumerate(layer.inputs):
                if input is not None:
                    if input.is_graph_input or input.producer.num_outputs > 1:
                        if input.name in tensor_names:
                            graph.edge(input.name, name, str(index))
                    else:
                        if input.producer.name in layer_names:
                            graph.edge(input.producer.name, name, str(index))
            if layer.num_outputs > 1 or (layer.num_outputs == 1 and
                                         layer.get_output(0).is_graph_output):
                for index, output in enumerate(layer.outputs):
                    graph.edge(name, output.name, str(index))

        graph.subgraph(input_block_graph)
        graph.subgraph(output_block_graph)
        for parent_graph, block_graph in block_graph_mapping:
            parent_graph.subgraph(block_graph)
        for device_graph in device_graphs.values():
            graph.subgraph(device_graph)

        if not path:
            return graph.source
        graph.save(path)

    @staticmethod
    def trt_to_dot(trt_network, path=None):
        graph = PipelineGraph.from_trt(trt_network)
        graph.assign_shapes()
        dot = graph.to_dot(no_style=True)
        if path is not None:
            with open(path, "w") as f:
                f.write(dot)
        else:
            return dot

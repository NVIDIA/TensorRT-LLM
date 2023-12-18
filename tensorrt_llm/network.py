# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import contextlib
import hashlib
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, OrderedDict, Set

import numpy as np
import tensorrt as trt

from ._common import set_network
from .logger import logger
from .plugin import PluginConfig


class _UniqueNameGenerator(object):

    def __init__(self, prefix=''):
        self.ids = collections.defaultdict(int)
        self.prefix = prefix

    def __call__(self, key, module_name=''):
        if module_name != '':
            module_name = module_name.replace(".", "/")
            key = module_name + '/' + key
        tmp = self.ids[key]
        self.ids[key] += 1
        return f"{self.prefix}{key}_{tmp}"


class Network(object):

    def __init__(self, **kwargs):
        # intentionally use **kwargs, user should never call this ctor directly
        # use Builder.create_network() instead

        # Holds the removed layers and disable them in graph rewritings and other phases.
        # This is a hacky way since INetwork python API doesn't provide a way to remove a layer.
        # TODO: remove this when TensorRT provides a better way to remove a layer
        self._removed_layers: Set[str] = set()

        self.is_graph_altered = False

        from .graph_rewriting import FLayerInfoMemo
        self.flayer_memo = FLayerInfoMemo()  # holds the functional metadata

    def _init(self, trt_network):
        self._trt_network = trt_network
        self._inputs = {}
        self._named_parameters = None
        # layer precision of a given scope, this is used together with precision(dtype) context manager
        self._dtype = None
        self._name_generator = _UniqueNameGenerator()
        self._plugin_config = PluginConfig()
        self._module_call_stack = _TrtLlmModuleCallStack()
        self._registered_ndarrays = []
        self._strongly_typed = trt.INetworkDefinition.get_flag(
            self._trt_network, trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

        return self

    @property
    def dtype(self) -> trt.DataType:
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: trt.DataType):
        assert isinstance(dtype, trt.DataType) or dtype is None
        self._dtype = dtype

    @property
    def trt_network(self) -> trt.INetworkDefinition:
        return self._trt_network

    @property
    def plugin_config(self) -> PluginConfig:
        return self._plugin_config

    @property
    def strongly_typed(self) -> bool:
        return self._strongly_typed

    def _add_input(self,
                   tensor,
                   name,
                   dtype,
                   shape,
                   dim_range: OrderedDict = None):
        assert isinstance(dtype, trt.DataType)
        tensor.trt_tensor = self.trt_network.add_input(
            name=name,
            shape=shape,
            dtype=dtype,
        )
        if dim_range is not None:
            logger.debug(
                f'Add input: {name}, shape: {shape}, dtype: {dtype}, dimension names:{list(dim_range.keys())}'
            )
            for i, dim_name in enumerate(dim_range.keys()):
                tensor.trt_tensor.set_dimension_name(i, str(dim_name))
        else:
            logger.debug(f'Add input: {name}, shape: {shape}, dtype: {dtype}')
        self._inputs[name] = tensor

    def _mark_output(self, tensor, name, dtype):
        from .functional import cast

        if self.strongly_typed:
            if tensor.trt_tensor.dtype != dtype:
                # If stronglyTyped mode is enabled and inferred output dtype does not match desired dtype, add a cast.
                cast_output = cast(tensor, dtype)
                self.trt_network.mark_output(cast_output.trt_tensor)
                cast_output.trt_tensor.name = name
            else:
                # Otherwise, mark the tensor as network output. We should not set tensor dtype in stronglyTyped mode.
                self.trt_network.mark_output(tensor.trt_tensor)
                tensor.trt_tensor.name = name
        else:
            self.trt_network.mark_output(tensor.trt_tensor)
            tensor.trt_tensor.name = name
            tensor.trt_tensor.dtype = dtype
        logger.debug(f'Mark output: {name}, dtype: {dtype}')

    def set_named_parameters(self, named_parameters):
        self._named_parameters = named_parameters

    @property
    def named_parameters(self):
        return self._named_parameters

    def _set_layer_name(self, layer):
        layer_name = str(layer.type).split('.')[-1]
        current_module = self._module_call_stack.get_current_module()

        if layer.type == trt.LayerType.PLUGIN_V2:
            layer_name = '_'.join(
                [layer_name,
                 str(layer.plugin.plugin_type).split('.')[-1]])
        elif layer.type in [
                trt.LayerType.UNARY, trt.LayerType.REDUCE,
                trt.LayerType.ELEMENTWISE
        ]:
            layer_name = '_'.join([layer_name, str(layer.op).split('.')[-1]])

        layer.name = self._name_generator(layer_name, current_module)
        for idx in range(layer.num_outputs):
            # TRT initializes tensor names from the initial layer's name when the layer is created,
            # and does not update tensor names when layer name changed by application, needs to
            # change the tensor name to align with the new layer name for better debugging
            layer.get_output(idx).name = f"{layer.name}_output_{idx}"

    def register_ndarray(self, ndarray: np.ndarray) -> None:
        ''' When the functional APIs need to create local numpy array and use as weights for constant or other layers,
            they need to register the ndarray objects to the TRT-LLM Network to prolong the lifetime of the ndarray, such that weights are
            still valid when functional API returned.
            All the weights referenced by the trt Network are weak referenced, it's TRT-LLM's responsibility to keep the weights alive
            during the TRT network construction and TRT engine building process.
        '''
        self._registered_ndarrays.append(ndarray)

    def get_inputs(self):
        '''
        Get the inputs of the network.

        Returns:
            Iterable[Tensor]
        '''
        return self._inputs.values()

    def get_outputs(self):
        '''
        Get the outputs of the network.

        Returns:
            Iterable[Tensor]
        '''
        from .functional import Tensor
        for i in range(self._trt_network.num_outputs):
            tensor = self._trt_network.get_output(i)
            yield Tensor(trt_tensor=tensor,
                         network=self,
                         is_network_input=False)

    def is_input(self, tensor) -> bool:
        '''
        Tell if a tensor is a input of the network.

        Parameters:
            tensor: Union[Tensor, str, trt.ITensor]
        '''
        from .functional import Tensor

        if isinstance(tensor, str):
            tensor_name = tensor
        elif isinstance(tensor, (trt.ITensor, Tensor)):
            tensor_name = tensor.name
        else:
            raise ValueError(
                f"tensor should be Tensor, str or ITensor, got {tensor}")

        return self._inputs.get(tensor_name, False)

    def is_output(self, tensor) -> bool:
        '''
        Tell if a tensor is a output of the network.

        Parameters:
            tensor: Tensor
        '''
        for i in range(self._trt_network.num_outputs):
            if tensor.trt_tensor is self._trt_network.get_output(i):
                return True
        return False

    def get_layers(self) -> Iterable["Layer"]:
        '''
        Get all the layers of network.

        Returns:
            Iterable[Layer]
        '''
        from .graph_rewriting import Layer
        for i in range(self._trt_network.num_layers):
            layer = Layer(network=self,
                          trt_layer=self._trt_network.get_layer(i))
            yield layer

    def get_layer_by_name(self, name: str) -> Optional["Layer"]:
        state = self._get_graph()
        return state.name_to_layer.get(name, None)

    def get_tensor_users(self, tensor) -> Iterable["Layer"]:
        '''
        Get the layers those consumes this tensor.
        '''
        state = self._get_graph()
        for layer in state.tensor_to_consumers[tensor]:
            yield layer

    def get_tensor_parent(self, tensor) -> Optional["Layer"]:
        '''
        Get the layer that produces this tensor.
        '''
        state = self._get_graph()
        return state.tensor_to_producer.get(tensor, None)

    def mark_removed_layer(self, layer: "Layer"):
        from .graph_rewriting import FLayerInfoMemo
        self._removed_layers.add(layer.name)

        # Try to delete the layer if it is a Plugin
        FLayerInfoMemo.instance().remove(layer.name)

    def is_removed_layer(self, layer: "Layer") -> bool:
        return layer.name in self._removed_layers

    @property
    def removed_layers(self) -> Iterable["Layer"]:
        for layer_name in self._removed_layers:
            layer = self.get_layer_by_name(layer_name)
            assert layer, "Invalid layer name"
            yield layer

    def to_dot(self, path=None) -> Optional[str]:
        '''
        Get a graphviz representation of the network.

        NOTE, the graph might be redundancy since TRT's INetwork won't clean the unused inputs and layers
        automatically.
        TODO: add an flag to hide all the removed layers and their output tensors
        TODO: replace this when TensorRT provides a better way to get the graph of INetworkDefinition
        TODO: a little feature, add blocks in the figure to highlight the subgraphes of Modules

        Parameters:
            path: the path to save the graphviz file, if not provided, will return the graphviz source code
        '''
        format = 'text' if not path else path.split('.')[-1]

        try:
            import graphviz
        except ImportError:
            logger.error(
                "Failed to import graphviz, please install graphviz to enable Network.to_dot()"
            )
            return

        dot = graphviz.Digraph(comment='TensorRT Graph',
                               format=format if format != 'text' else None)

        inputs_names = set([x.name for x in self.get_inputs()])
        output_names = set([x.name for x in self.get_outputs()])

        node_style = dict(
            shape='box',
            style='rounded,filled,bold',
            fontname='Arial',
            fillcolor='#ffffff',
            color='#303A3A',
            width='1.3',
            height='0.84',
        )

        hl_node_style = dict(
            shape='box',
            style='rounded,filled,bold',
            fontname='Arial',
            fillcolor='lightblue',
            color='#303A3A',
            width='1.3',
            height='0.84',
        )

        state = self._get_graph()
        nodes = set()
        tensor_to_alias = {}
        tensor_id = [0]

        def get_alias(tensor, tensor_id):
            if tensor not in tensor_to_alias:
                if (not tensor in inputs_names) and (not tensor
                                                     in output_names):
                    tensor_to_alias[tensor] = f"t{tensor_id[0]}"
                    tensor_id[0] += 1
                else:
                    tensor_to_alias[tensor] = tensor

            return tensor_to_alias[tensor]

        def create_tensor_node(tensor: str):
            tensor_alias = get_alias(tensor, tensor_id)
            if tensor_alias not in nodes:
                dot.node(tensor_alias, tensor_alias, **node_style)
                nodes.add(tensor_alias)
            return tensor_alias

        def create_layer_node(layer: str):
            if layer not in nodes:
                dot.node(layer, layer, **hl_node_style)
                nodes.add(layer)

        for tensor, layer in state.tensor_to_producer.items():
            tensor_alias = create_tensor_node(tensor.name)
            create_layer_node(layer.name)
            dot.edge(layer.name, tensor_alias)
        for tensor, layers in state.tensor_to_consumers.items():
            tensor_alias = create_tensor_node(tensor.name)
            for layer in layers:
                create_layer_node(layer.name)
                dot.edge(tensor_alias, layer.name)

        if format == "text":
            return dot.source
        dot.render(path)

    def _get_graph(self) -> "Network._GraphState":
        '''
        Get the graph of the network.

        Returns:
            Network._GraphState
        '''
        return self._get_graph_impl(self._get_network_hash())

    #TODO: tali, using one LRU cache here can cause the Network object to be leaked, need a way to speed this function w/o using global lru cache.
    def _get_graph_impl(self, network_hash: bytes) -> "Network._GraphState":
        graph = Network._GraphState()
        graph.build(self)
        return graph

    @dataclass
    class _GraphState:
        # Tensor to Layers
        tensor_to_consumers: Dict[Any, List["Layer"]] = field(
            default_factory=lambda: defaultdict(list))
        # Tensor to Layer
        tensor_to_producer: Dict[Any, "Layer"] = field(default_factory=dict)
        inputs: Dict[str, Any] = field(default_factory=OrderedDict)
        outputs: Dict[str, Any] = field(default_factory=OrderedDict)
        name_to_layer: Dict[str, "Layer"] = field(default_factory=dict)

        def build(self, network: "Network") -> None:
            from .graph_rewriting import Layer
            self.inputs = network.get_inputs()
            self.outputs = network.get_outputs()

            for layer in network.get_layers():
                self.name_to_layer[layer.name] = Layer(
                    network=network, trt_layer=layer.trt_layer)
                for i in range(layer.num_inputs):
                    input_tensor = layer.get_inputs(i)[0]
                    if input_tensor.is_trt_wrapper():
                        self.tensor_to_consumers[input_tensor].append(layer)
                for i in range(layer.num_outputs):
                    output_tensor = layer.get_outputs(i)[0]
                    if output_tensor.is_trt_wrapper():
                        self.tensor_to_producer[output_tensor] = layer

    def _get_network_hash(self, lightweight=True) -> bytes:
        # TODO: Ask TensorRT team to add a hash function for INetworkDefinition instead of using this hacky way
        num_layers = self.trt_network.num_layers

        # Some special layers, such as slice, may be associated with tensors that do not have the `trt_tensor` member.
        get_tensor_tag = lambda tensor: tensor.trt_tensor.name if tensor.is_trt_wrapper(
        ) else 'None'

        if lightweight and not self.is_graph_altered:
            return num_layers
        self.is_graph_altered = False

        data = hashlib.sha256()
        # network layer count
        data.update(str(num_layers).encode())
        # network inputs
        data.update(','.join(
            [get_tensor_tag(tensor) for tensor in self.get_inputs()]).encode())
        # network outputs
        data.update(','.join(
            [get_tensor_tag(tensor) for tensor in self.get_outputs()]).encode())
        # layer names
        data.update(','.join(
            [layer.trt_layer.name for layer in self.get_layers()]).encode())

        # layer -> output
        data.update(','.join([
            f'{layer.trt_layer.name}->{get_tensor_tag(tensor)}'
            for layer in self.get_layers() for tensor in layer.get_outputs()
        ]).encode())

        # input -> layer
        data.update(','.join([
            f'{get_tensor_tag(tensor)}->{layer.trt_layer.name}'
            for layer in self.get_layers() for tensor in layer.get_inputs()
        ]).encode())

        return data.hexdigest()


@contextlib.contextmanager
def net_guard(network):
    from ._common import net
    assert isinstance(
        network, Network
    ), f"Invalid network, can only guard Network instance, got: {network}"

    old_net = net
    set_network(network)
    yield
    set_network(old_net)


class _TrtLlmModuleCallStack(object):
    call_stack = []
    module_name_map = weakref.WeakKeyDictionary()

    def __init__(self):
        super().__init__()
        self.mod_names_set = False

    def module_names_set(self):
        return self.mod_names_set

    def set_module_names(self, top_level_module):
        assert top_level_module, "Expected a top level module"
        for name, mod in top_level_module.named_modules(
                prefix=top_level_module._get_name()):
            if mod not in self.module_name_map:
                self.module_name_map[mod] = name
        self.mod_names_set = True
        return

    def get_current_module(self):
        mod_name = ''
        if len(self.call_stack):
            mod_name = self.call_stack[-1]
        return mod_name

    def get_mod_name(self, mod_obj):
        name = ''
        if mod_obj in self.module_name_map:
            name = self.module_name_map[mod_obj]
        return name

    def get_stack(self):
        return self.call_stack

    @contextlib.contextmanager
    def call_stack_mgr(self):
        call_stack = self.get_stack()
        try:
            yield call_stack
        finally:
            call_stack.pop()

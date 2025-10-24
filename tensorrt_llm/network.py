# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import inspect
import weakref
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Optional, OrderedDict, Set,
                    Tuple, Union)

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt

from tensorrt_llm.module import Module

from ._common import set_network
from ._utils import get_extra_attr, has_extra_attr, set_extra_attr
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


class PluginInfo:
    plugin_creator: trt.IPluginCreator
    plugin_name: str
    pfc: trt.PluginFieldCollection

    def __init__(self, plugin_creator: trt.IPluginCreator, plugin_name: str,
                 pfc: trt.PluginFieldCollection):
        self.plugin_creator = plugin_creator
        self.plugin_name = plugin_name
        self.pfc = pfc
        self._parse_pfc(pfc)

    def _parse_pfc(self, pfc: trt.PluginFieldCollection):
        self.pfc_as_ndarray = {}
        self.pfc_as_list = {}
        for i in range(len(pfc)):
            name, data = pfc[i].name, pfc[i].data
            array_data = data
            self.pfc_as_ndarray[name] = array_data.copy()
            list_data = array_data.tolist()
            self.pfc_as_list[name] = list_data


def get_plugin_info(trt_network: trt.INetworkDefinition,
                    layer_name: str) -> PluginInfo:
    if not has_extra_attr(trt_network, "plugin_infos"):
        return None
    plugin_infos = get_extra_attr(trt_network, "plugin_infos")
    if layer_name not in plugin_infos:
        return None
    return plugin_infos[layer_name]


def set_plugin_info(trt_network: trt.INetworkDefinition, layer_name: str,
                    plugin_info: PluginInfo):
    if not has_extra_attr(trt_network, "plugin_infos"):
        set_extra_attr(trt_network, "plugin_infos", {})
    plugin_infos = get_extra_attr(trt_network, "plugin_infos")
    plugin_infos[layer_name] = plugin_info


def delete_plugin_info(trt_network: trt.INetworkDefinition, layer_name: str):
    if not has_extra_attr(trt_network, "plugin_infos"):
        return
    plugin_infos = get_extra_attr(trt_network, "plugin_infos")
    if layer_name not in plugin_infos:
        return
    del plugin_infos[layer_name]


# TODO: remove this WAR after https://nvbugs/4359151 fixed.
def get_np_weight(trt_network: trt.INetworkDefinition,
                  layer_name: str) -> np.array:
    if not has_extra_attr(trt_network, "np_weights"):
        return None
    np_weights = get_extra_attr(trt_network, "np_weights")
    if layer_name not in np_weights:
        return None
    return np_weights[layer_name]


# TODO: remove this WAR after https://nvbugs/4359151 fixed.
def set_np_weight(trt_network: trt.INetworkDefinition, layer_name: str,
                  np_weight: np.array):
    if not has_extra_attr(trt_network, "np_weights"):
        set_extra_attr(trt_network, "np_weights", {})
    np_weights = get_extra_attr(trt_network, "np_weights")
    np_weights[layer_name] = np_weight


class Network(object):

    def __init__(self, **kwargs):
        # intentionally use **kwargs, user should never call this ctor directly
        # use Builder.create_network() instead

        # Holds the removed layers and disable them in graph rewriting and other phases.
        # This is a hacky way since INetwork python API doesn't provide a way to remove a layer.
        # TODO: remove this when TensorRT provides a better way to remove a layer
        self._removed_layers: Set[str] = set()

        self.is_graph_altered = False

        from .graph_rewriting import FLayerInfoMemo
        self.flayer_memo = FLayerInfoMemo()  # holds the functional metadata
        self._parameter_tensors = {}  # holds the parameter tensors

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
        self._unfilled_weights: Dict[str, Tuple[np.array, np.array]] = {}

        return self

    def _register_unfilled_weights(self, layer_name: str, weights: np.array,
                                   values: np.array):
        self._unfilled_weights[layer_name] = (weights, values)

    def _fill_weights(self):
        from tensorrt_llm.parameter import Parameter

        for layer_name in list(self._unfilled_weights.keys()):
            weights, values = self._unfilled_weights.pop(layer_name)
            self.register_ndarray(weights)
            if values is not None:
                np.copyto(weights, values, casting='no')
            else:
                Parameter.xavier_init(weights)

    @property
    def parameter_tensors(self):
        return self._parameter_tensors

    def get_parameter_tensor(self, param):
        return self.parameter_tensors.get(param, None)

    def set_parameter_tensor(self, param, tensor):
        assert param not in self.parameter_tensors
        self.parameter_tensors[param] = tensor

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

    @plugin_config.setter
    def plugin_config(self, cfg: PluginConfig):
        assert isinstance(
            cfg,
            PluginConfig), f"Expecting a PluginConfig object, got {type(cfg)}"
        self._plugin_config = cfg

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
        assert tensor.trt_tensor is not None, f"Couldn't create TRT tensor for {name} {dtype} {shape}"
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

        # In strongly_typed, if tensor output is not the same, add a cast
        if dtype is not None and self.strongly_typed:
            tensor = cast(tensor, dtype)
        self.trt_network.mark_output(tensor.trt_tensor)
        tensor.trt_tensor.name = name
        if not self.strongly_typed:
            tensor.trt_tensor.dtype = dtype or tensor.trt_tensor.dtype
        logger.debug(f'Mark output: {name}, dtype: {dtype}')

    def set_named_parameters(self, named_parameters):
        self._named_parameters = named_parameters

    @property
    def named_parameters(self):
        return self._named_parameters

    def _set_layer_name(self, layer):
        original_layer_name = layer.name
        layer_name = str(layer.type).split('.')[-1]
        current_module = self._module_call_stack.get_current_module()

        func_stack = []
        frame = inspect.currentframe().f_back.f_back
        while frame:
            func_name = frame.f_code.co_name
            line_num = frame.f_lineno
            if func_name == "forward":
                break
            func_stack.insert(0, f"{func_name}_L{line_num}")
            if len(func_stack) >= 10:
                # NOTE: TRT error messages has a character limit.
                #       Limiting to only 10 levels helps retain
                #       the true error message from TRT.
                break
            frame = frame.f_back
        current_module = f"{current_module}.{'.'.join(func_stack)}"

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
        if original_layer_name != layer.name:
            if layer.type == trt.LayerType.PLUGIN_V2:
                plugin_info = get_plugin_info(self.trt_network,
                                              original_layer_name)
                if plugin_info is not None:
                    set_plugin_info(self.trt_network, layer.name, plugin_info)
                    delete_plugin_info(self.trt_network, original_layer_name)

        # Set layer metadata to the same as the layer name so that it can show up in NVTX.
        layer.metadata = layer.name

    def register_ndarray(self, ndarray: np.ndarray) -> None:
        ''' When the functional APIs need to create local numpy array and use as weights for constant or other layers,
            they need to register the ndarray objects to the TRT-LLM Network to prolong the lifetime of the ndarray, such that weights are
            still valid when functional API returned.
            All the weights referenced by the trt Network are weak referenced, it's TRT-LLM's responsibility to keep the weights alive
            during the TRT network construction and TRT engine building process.
        '''
        self._registered_ndarrays.append(ndarray)

    def _generate_optimization_profiles(self) -> List[trt.IOptimizationProfile]:
        input_tensors = self._inputs
        if len(input_tensors) == 0:
            return []
        num_profiles = len(list(input_tensors.values())[0].profiles)
        profiles = []
        for i in range(num_profiles):
            logger.debug(f'Adding optimization profile {i+1}/{num_profiles}')
            profile = self._trt_network.builder.create_optimization_profile()
            for input_name, input_tensor in input_tensors.items():
                shape_profile = input_tensor.profiles[i]
                min_shape = list(shape_profile.min)
                opt_shape = list(shape_profile.opt)
                max_shape = list(shape_profile.max)
                if input_tensor.trt_tensor.is_shape_tensor:
                    profile.set_shape_input(input_name, min_shape, opt_shape,
                                            max_shape)
                else:
                    profile.set_shape(input_name, min_shape, opt_shape,
                                      max_shape)
                logger.debug(
                    f'{input_name}, min: {min_shape}, opt: {opt_shape}, max: {max_shape}'
                )
            profiles.append(profile)
        return profiles

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

    def to_dot(self, path: Union[str, Path] = None) -> Optional[str]:
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

        dot = graphviz.Digraph(
            comment=
            f'TensorRT Graph of {self._get_network_hash(lightweight=False)}',
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

        def create_tensor_node(tensor: str, dtype=None, shape=None):
            tensor_alias = get_alias(tensor, tensor_id)
            if tensor_alias not in nodes:
                dot.node(tensor_alias,
                         str(dtype) + "\n" + tensor_alias + "\n" + str(shape),
                         **node_style)
                nodes.add(tensor_alias)
            return tensor_alias

        def create_layer_node(layer: str):
            if layer not in nodes:
                dot.node(layer, layer, **hl_node_style)
                nodes.add(layer)

        for tensor, layer in state.tensor_to_producer.items():
            tensor_alias = create_tensor_node(tensor.name, tensor.dtype,
                                              tensor.shape)
            create_layer_node(layer.name)
            dot.edge(layer.name, tensor_alias)
        for tensor, layers in state.tensor_to_consumers.items():
            tensor_alias = create_tensor_node(tensor.name, tensor.dtype,
                                              tensor.shape)
            for layer in layers:
                create_layer_node(layer.name)
                dot.edge(tensor_alias, layer.name)

        if format == "text":
            return dot.source
        dot.save(path)

    def to_onnx(self, path: str = None) -> None:
        '''
        Export the network into a "ONNX-like" file for visualization.

        Parameters:
            path: the path to the output file
        '''
        if path is None:
            return

        network = self.trt_network
        b_onnx_type = True  # Use ONNX style operator names

        def layer_type_to_class(layer: trt.ILayer = None) -> trt.ILayer:
            '''
            Convert trt.ILayer into a exact TensorRT layer
            '''
            layer_type_name = str(layer.type)[10:]
            # Some special cases
            if layer_type_name == "ELEMENTWISE":
                return trt.IElementWiseLayer
            if layer_type_name == "LRN":
                return trt.ILRNLayer
            if layer_type_name == "NMS":
                return trt.INMSLayer
            if layer_type_name == "PARAMETRIC_RELU":
                return trt.IParametricReLULayer
            if layer_type_name == "PLUGIN":
                return None  # IPluginLayer is not supported any more
            if layer_type_name == "RAGGED_SOFTMAX":
                return trt.IRaggedSoftMaxLayer
            if layer_type_name == "SOFTMAX":
                return trt.ISoftMaxLayer
            if layer_type_name == "TOPK":
                return trt.ITopKLayer

            # general cases, e.g. MATRIX_MULTIPLY -> MatrixMultiply
            name = "".join(name[0] + name[1:].lower()
                           for name in layer_type_name.split("_"))
            return trt.__builtins__["getattr"](trt, f"I{name}Layer")

        def convert_type_to_onnx(node_type: str = "",
                                 attribution: OrderedDict = OrderedDict()):
            '''
            Convert layer names of TensorRT style into ONNX style
            '''
            if node_type == "ACTIVATION":
                convert_list = {
                    "RELU": "Relu",
                    "SIGMOID": "Sigmoid",
                    "TANH": "Tanh",
                    "LEAKY_RELU": "LeakyRelu",
                    "ELU": "Elu",
                    "SELU": "Selu",
                    "SOFTSIGN": "Softsign",
                    "SOFTPLUS": "Softplus",
                    "CLIP": "Clip",
                    "HARD_SIGMOID": "HardSigmoid",
                    "SCALED_TANH": "ScaledTanh",
                    "THRESHOLDED_RELU": "ThresholdedRelu",
                }
                # No corresponding operator for GELU_ERF, GELU_TANH
                if "algo-type" in attribution.keys() and attribution[
                        "algo-type"].split(".")[-1] in convert_list.keys():
                    return convert_list[attribution["algo-type"].split(".")[-1]]
                return node_type
            if node_type == "CAST":
                return "Cast"
            if node_type == "CONCATENATION":
                return "Concat"
            if node_type == "CONSTANT":
                return "Constant"
            if node_type == "CONVOLUTION":
                return "Conv"
            if node_type == "DECONVOLUTION":
                return "Deconv"
            if node_type == "ELEMENTWISE":
                convert_list = {
                    "SUM": "Add",
                    "PROD": "Mul",
                    "MAX": "Max",
                    "MIN": "Min",
                    "SUB": "Sub",
                    "DIV": "Div",
                    "POW": "Pow",
                    "AND": "And",
                    "OR": "Or",
                    "XOR": "Xor",
                    "EQUAL": "Equal",
                    "GREATER": "Greater",
                    "LESS": "Less",
                }
                if "op" in attribution.keys() and attribution["op"].split(
                        ".")[-1] in convert_list.keys():
                    return convert_list[attribution["op"].split(".")[-1]]
                return node_type
            if node_type == "GATHER":
                return "Gather"
            if node_type == "LOOP":
                return "Loop"
            if node_type == "MATRIX_MULTIPLY":
                return "Gemm"
            if node_type == "POOLING":
                convert_list = {"MAX": "MaxPool", "AVERAGE": "AveragePool"}
                if "algo-type" in attribution.keys() and attribution[
                        "algo-type"].split(".")[-1] in convert_list.keys():
                    return convert_list[attribution["algo-type"].split(".")[-1]]
                return node_type
            if node_type == "REDUCE":
                convert_list = {
                    "SUM": "ReduceSum",
                    "PROD": "ReduceProd",
                    "MAX": "ReduceMax",
                    "MIN": "ReduceMin",
                    "AVG": "ReduceMean",
                }
                if "op" in attribution.keys() and attribution["op"].split(
                        ".")[-1] in convert_list.keys():
                    return convert_list[attribution["op"].split(".")[-1]]
                return node_type
            if node_type == "SELECT":
                return "Where"
            if node_type == "SHUFFLE":
                return "Reshape"
            if node_type == "SHAPE":
                return "Shape"
            if node_type == "SLICE":
                return "Slice"
            if node_type == "SOFTMAX":
                return "Softmax"
            if node_type == "TOPK":
                return "TopK"
            if node_type == "UNARY":
                convert_list = {"SQRT": "Sqrt", "NOT": "Not"}
                if "op" in attribution.keys() and attribution["op"].split(
                        ".")[-1] in convert_list.keys():
                    return convert_list[attribution["op"].split(".")[-1]]
                return node_type
            return node_type

        def add_node_for_trt_network(
            graph: gs.Graph = None,
            node_name: str = "",
            node_type: str = "",
            input_list: List[gs.Variable] = [],
            attribution: OrderedDict = OrderedDict(),
            name_list: Union[str, List[str]] = "",
            datatype_list: Union[np.dtype, List[np.dtype]] = [],
            shape_list: Union[list, List[list]] = [],
            number: int = 0,
            b_onnx_type: bool = False,
        ) -> Tuple[gs.Variable, int]:
            '''
            Simplify version of function `add_node`, and we do some beautify to it.
            '''
            if isinstance(name_list, list) or isinstance(
                    datatype_list, list) or isinstance(
                        shape_list, list):  # Case of multi-output
                assert len(name_list) == len(datatype_list)
                assert len(name_list) == len(shape_list)
            else:  # Case of single-output
                name_list = [name_list]
                datatype_list = [datatype_list]
                shape_list = [shape_list]

            n_output = len(name_list)
            output_list = []
            for i in range(n_output):
                tensor = gs.Variable(name_list[i], datatype_list[i],
                                     shape_list[i])
                output_list.append(tensor)

            if b_onnx_type:
                node_type = convert_type_to_onnx(node_type, attribution)

            node = gs.Node(node_type,
                           node_name,
                           inputs=input_list,
                           outputs=output_list,
                           attrs=attribution)
            graph.nodes.append(node)  # Update graph inside `add_node`

            if len(output_list) == 1:  # Case of single-output
                output_list = output_list[0]
            return output_list, number + 1

        graph = gs.Graph(nodes=[], inputs=[], outputs=[])
        graph.name = "" if network.name == "Unnamed Network 0" else network.name
        n = 0

        # mapping from TRT tensor (trt.ITensor) to GS tensor (gs.Variable)
        global_tensor_map = {}

        for i in range(network.num_inputs):
            trt_tensor = network.get_input(i)
            gs_tensor = gs.Variable(trt_tensor.name,
                                    trt.nptype(trt_tensor.dtype),
                                    trt_tensor.shape)
            global_tensor_map[trt_tensor] = gs_tensor
            if gs_tensor not in graph.inputs:
                graph.inputs.append(gs_tensor)

        for i in range(network.num_layers):
            layer = network.get_layer(i)

            input_tensor_list = []
            for j in range(layer.num_inputs):
                trt_tensor = layer.get_input(j)
                if trt_tensor is None:  # Useful for constant layer
                    gs_tensor = None
                elif trt_tensor in global_tensor_map.keys(
                ):  # already in the map
                    gs_tensor = global_tensor_map[trt_tensor]
                else:
                    logger.debug(
                        f"[ExportONNX]Layer input tensor not in global_tensor_map: {trt_tensor.name}"
                    )  # ■
                    gs_tensor = gs.Variable(trt_tensor.name,
                                            trt.nptype(trt_tensor.dtype),
                                            trt_tensor.shape)
                    global_tensor_map[trt_tensor] = gs_tensor
                input_tensor_list.append(gs_tensor)

            output_name_list = []
            output_datatype_list = []
            output_shape_list = []
            for i in range(layer.num_outputs):
                trt_tensor = layer.get_output(i)
                # Don't do this check because we need this trt_tensor to overwrite the placeholder tensor in ■
                # if trt_tensor in global_tensor_map.keys():
                #     gs_tensor = global_tensor_map[trt_tensor]
                output_name_list.append(trt_tensor.name)
                output_datatype_list.append(trt.nptype(trt_tensor.dtype))
                output_shape_list.append(trt_tensor.shape)

            attr = OrderedDict()
            # Set attribution of ILayer
            for key in dir(layer):
                if not (key.startswith("_")
                        or callable(layer.__getattribute__(key))):
                    attr[key] = str(layer.__getattribute__(key))
            # Set attribution of exact layer type
            layer.__class__ = layer_type_to_class(layer)
            for key in dir(layer):
                if key in dir(trt.ILayer) and key != "type":
                    continue
                if key == "type" and not isinstance(layer.type, trt.LayerType):
                    attr["algo-type"] = str(layer.type)
                    continue
                value = layer.__getattribute__(key)
                if isinstance(value, np.ndarray):
                    # Convert all attributions into string besides weights
                    value = value.astype(np.float32)  # In case of overflow
                    ss = f"shape={value.shape}, SumAbs={np.sum(abs(value)):.5e}, Var={np.var(value):.5f}, "
                    ss += f"Max={np.max(value):.5f}, Min={np.min(value):.5f}, SAD={np.sum(np.abs(np.diff(value.reshape(-1)))):.5f}, "
                    ss += f"[:5]={value.reshape(-1)[:5]}, [-5:]={value.reshape(-1)[-5:]}"
                    attr[key] = ss
                else:
                    attr[key] = str(value)

            output_tensor_list, n = add_node_for_trt_network(graph, layer.name, attr["type"][10:], input_tensor_list, attr, \
                output_name_list, output_datatype_list, output_shape_list, n, b_onnx_type)

            if layer.num_outputs == 1:
                global_tensor_map[layer.get_output(0)] = output_tensor_list
            else:
                for i in range(layer.num_outputs):
                    global_tensor_map[layer.get_output(
                        i)] = output_tensor_list[i]

        for i in range(network.num_outputs):
            gs_tensor = global_tensor_map[network.get_output(i)]
            if gs_tensor not in graph.outputs:
                graph.outputs.append(gs_tensor)

        onnx.save(gs.export_onnx(graph),
                  path + "/network.onnx",
                  save_as_external_data=False)
        return

    def _get_graph(self) -> "Network._GraphState":
        '''
        Get the graph of the network.

        Returns:
            Network._GraphState
        '''
        return self._get_graph_impl(self._get_network_hash())

    #TODO: using one LRU cache here can cause the Network object to be leaked, need a way to speed this function w/o using global lru cache.
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

    def __init__(self):
        super().__init__()
        self.call_stack = []
        self.module_name_map = weakref.WeakKeyDictionary()
        self.module_to_layer_range_map: Dict[str, range] = {}
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

    def set_layer_range(self, mod_obj: Module, layer_range: range):
        if mod_obj in self.module_name_map:
            name = self.module_name_map[mod_obj]
            self.module_to_layer_range_map[name] = layer_range

    def get_stack(self):
        return self.call_stack

    @contextlib.contextmanager
    def call_stack_mgr(self):
        call_stack = self.get_stack()
        try:
            yield call_stack
        finally:
            call_stack.pop()

import contextlib
import copy
import itertools
import pickle  # nosec B403
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
from filelock import FileLock

from tensorrt_llm._utils import (str_dtype_to_trt, trt_dtype_to_np,
                                 trt_dtype_to_torch, trt_gte_10)
from tensorrt_llm.functional import (AllReduceConfig, AllReduceFusionParams,
                                     AllReduceStrategy, create_allreduce_plugin)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.network import (PluginInfo, delete_plugin_info, get_np_weight,
                                  get_plugin_info, set_plugin_info)
from tensorrt_llm.plugin import TRT_LLM_PLUGIN_NAMESPACE, init_all_reduce_helper
from tensorrt_llm.plugin.plugin import (CustomAllReduceHelper,
                                        current_all_reduce_helper)
from tensorrt_llm.version import __version__

from .config import AutoParallelConfig
from .device_mesh import LogicalDeviceMesh
from .pipeline_graph import Layer, PipelineGraph, Tensor
from .shape_info import (ShapeInfo, get_per_layer_graph, get_shape_layers,
                         infer_per_layer_shapes)
from .simplifier import GraphConfig, GraphMapping, Simplifier, StageType
from .tensor_parallel.comm_spec import CommSpec
from .tensor_parallel.plugin_nodes.gpt_attention_node import (
    GPTAttentionPlugin, IdxEntry, IdxEntryParser)
from .tensor_parallel.sharding_spec import ShardingSpec, get_sharding_sequence
from .tensor_parallel.sharding_strategy import ShardingStrategy
from .utils import (get_updated_plugin, to_base_class_layer, to_subclass_layer,
                    to_trt_weights)

default_int_dtype = trt.int64 if trt_gte_10() else trt.int32


@dataclass
class ParallelConfig:
    VERSION: ClassVar[str] = __version__

    version: str = VERSION
    network_hash: str = None
    auto_parallel_config: AutoParallelConfig = None
    graph_config: GraphConfig = None
    lmesh: LogicalDeviceMesh = None
    cost: float = None
    graph_strategy: Dict[str, ShardingStrategy] = None
    stage_type: StageType = None

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def from_file(filename) -> "ParallelConfig":
        with open(filename, "rb") as file:
            return pickle.load(file)  # nosec B301

    def print_graph_strategy(self, file=None):
        for index, (node_name,
                    strategy) in enumerate(self.graph_strategy.items()):
            print(f'\n[{index}]: node_name = {node_name}', file=file)
            strategy.print_strategy(best_resharding_cost_only=True, file=file)


def desimplify_strategy(
    graph: PipelineGraph,
    graph_strategy: Dict[str, ShardingStrategy],
    graph_mapping: GraphMapping,
):
    for strategy in graph_strategy.values():
        for name, commspec in list(strategy.communication_actions.items()):
            strategy.communication_actions[name] = [commspec]
            strategy.sharding_specs[
                f"{name}_after_comm"] = strategy.sharding_specs[name]

    # insert same spec layers' communication actions after
    # its producer's communication actions
    same_spec_layer_mapping = graph_mapping.same_spec_layer_mapping
    for same_spec_layer_name in same_spec_layer_mapping.keys():
        same_spec_strategy = graph_strategy[same_spec_layer_name]
        same_spec_commspecs = same_spec_strategy.best_resharding_cost[0][0][1]
        if len(same_spec_commspecs) == 0:
            continue
        output_name = same_spec_layer_name[:-len("_same_spec")]
        output = graph.get_tensor(output_name)
        layer_name = output.producer.name
        output_index = output.output_index
        strategy = graph_strategy[layer_name]
        commspecs = strategy.communication_actions.get(f"output{output_index}",
                                                       [])
        commspecs.extend(same_spec_commspecs)
        strategy.communication_actions[f"output{output_index}"] = commspecs
        strategy.sharding_specs[
            f"output{output_index}_after_comm"] = same_spec_strategy.sharding_specs[
                "output0"]

    layer_mapping = graph_mapping.layer_mapping
    for removed_layer_name, layer_name in layer_mapping.items():
        if layer_name in graph_strategy:
            strategy = copy.copy(graph_strategy[layer_name])
            layer = graph.get_layer(removed_layer_name)
            if layer is not None:
                strategy.node_names = strategy.node_names.copy()
                for index, name in list(strategy.node_names.items()):
                    input = layer.get_input(index)
                    node_name = input.name if input.producer is None else input.producer.name
                    strategy.node_names[index] = node_name
            graph_strategy[removed_layer_name] = strategy


@dataclass
class SplitInfo:
    input_dim: Union[int, trt.ITensor]
    partition: int

    def __deepcopy__(self, memo) -> "SplitInfo":
        return SplitInfo(self.input_dim, self.partition)


@dataclass
class TensorInfo:
    name: str = None
    split_infos: Dict[int, SplitInfo] = field(default_factory=dict)

    def set_split_info(self, dim, split_info):
        self.split_infos[dim] = split_info

    def __deepcopy__(self, memo) -> "TensorInfo":
        return TensorInfo(self.name, copy.deepcopy(self.split_infos))


@dataclass
class TensorContext:
    info_by_device: Dict[int, TensorInfo] = field(default_factory=dict)
    device_dims_for_shape: Set[int] = field(default_factory=set)

    def update_name_mapping(self, device_id, new_name):
        if device_id not in self.info_by_device:
            self.info_by_device[device_id] = TensorInfo()
        self.info_by_device[device_id].name = new_name

    def set_split_info(self, device_id, dim, split_info):
        if device_id not in self.info_by_device:
            self.info_by_device[device_id] = TensorInfo()
        self.info_by_device[device_id].set_split_info(dim, split_info)

    def set_split_infos(self, device_id, split_infos: Dict[int, SplitInfo]):
        if device_id not in self.info_by_device:
            self.info_by_device[device_id] = TensorInfo()
        self.info_by_device[device_id].split_infos = split_infos

    def __deepcopy__(self, memo) -> "TensorContext":
        return TensorContext(copy.deepcopy(self.info_by_device),
                             set(self.device_dims_for_shape))


@dataclass
class LayerUpdate:
    updated_attrs: Dict[str, Any] = field(default_factory=dict)
    updated_inputs: Dict[int, trt.ITensor] = field(default_factory=dict)
    split_info_updated: bool = False

    @staticmethod
    def none() -> "LayerUpdate":
        return LayerUpdate()


@dataclass
class GraphContext:
    tensor_contexts: Dict[str, TensorContext] = field(default_factory=dict)

    def get_name(self, tensor_name, device_id):
        if tensor_name not in self.tensor_contexts:
            return None
        if device_id not in self.tensor_contexts[tensor_name].info_by_device:
            return None
        return self.tensor_contexts[tensor_name].info_by_device[device_id].name

    def update_name_mapping(self, tensor_name, device_id, new_name):
        if tensor_name not in self.tensor_contexts:
            self.tensor_contexts[tensor_name] = TensorContext()
        self.tensor_contexts[tensor_name].update_name_mapping(
            device_id, new_name)

    def get_name_mapping(self, device_id, prefix: str) -> Dict[str, str]:
        name_mapping = {}
        for tensor_name in self.tensor_contexts.keys():
            new_name = self.get_name(tensor_name, device_id)
            if new_name is not None:
                name_mapping[f"{prefix}{tensor_name}"] = new_name
        return name_mapping

    def add_device_dims_for_shape(self, tensor_name: str,
                                  device_dims: Sequence[int]):
        if tensor_name not in self.tensor_contexts:
            self.tensor_contexts[tensor_name] = TensorContext()
        self.tensor_contexts[tensor_name].device_dims_for_shape.update(
            device_dims)

    def get_device_dims_for_shape(self, tensor_name: str):
        if tensor_name not in self.tensor_contexts:
            return set()
        return self.tensor_contexts[tensor_name].device_dims_for_shape

    def get_split_infos(self, tensor_name, device_id):
        if tensor_name not in self.tensor_contexts:
            return None
        if device_id not in self.tensor_contexts[tensor_name].info_by_device:
            return None
        return self.tensor_contexts[tensor_name].info_by_device[
            device_id].split_infos

    def set_split_info(self, tensor_name, device_id, dim, split_info):
        if tensor_name not in self.tensor_contexts:
            self.tensor_contexts[tensor_name] = TensorContext()
        self.tensor_contexts[tensor_name].set_split_info(
            device_id, dim, split_info)

    def set_split_infos(self, tensor_name, device_id,
                        split_infos: Dict[int, SplitInfo]):
        if tensor_name not in self.tensor_contexts:
            self.tensor_contexts[tensor_name] = TensorContext()
        self.tensor_contexts[tensor_name].set_split_infos(
            device_id, split_infos)

    def update_layer_context(self, wrapped_layer: Layer,
                             layer_update: LayerUpdate,
                             local_context: "GraphContext", device_id: int,
                             device_ids: np.ndarray,
                             sharding_specs: Dict[str, ShardingSpec]):
        layer = wrapped_layer.as_trt()
        for i in range(layer.num_outputs):
            output = layer.get_output(i)
            new_name = local_context.get_name(output.name, device_id)
            if new_name is not None:
                self.update_name_mapping(output.name, device_id, new_name)
        if layer_update.split_info_updated:
            for i in range(layer.num_outputs):
                output = layer.get_output(i)
                split_infos = local_context.get_split_infos(
                    output.name, device_id)
                if split_infos is not None:
                    self.set_split_infos(output.name, device_id, split_infos)
            return
        split_info_by_device_dim = {}
        for i in range(layer.num_inputs):
            input = layer.get_input(i)
            if input is None:
                continue
            sharding_spec = sharding_specs[f"input{i}"]
            split_infos = local_context.get_split_infos(input.name, device_id)
            if split_infos is None:
                continue
            for dim, split_info in split_infos.items():
                device_dim = tuple(sharding_spec.dim_partition_dict[dim])
                split_info_by_device_dim[device_dim] = split_info
        for i in range(layer.num_outputs):
            output = layer.get_output(i)
            sharding_spec = sharding_specs[f"output{i}"]
            for dim, device_dim in sharding_spec.dim_partition_dict.items():
                split_info = split_info_by_device_dim.get(tuple(device_dim))
                if split_info is None:
                    if device_dim == [0, 1] or device_dim == [1, 0]:
                        if (0, ) in split_info_by_device_dim and (
                                1, ) in split_info_by_device_dim:
                            split_info = SplitInfo(
                                split_info_by_device_dim[(0, )].input_dim *
                                split_info_by_device_dim[(1, )].input_dim,
                                split_info_by_device_dim[(0, )].partition *
                                split_info_by_device_dim[(1, )].partition,
                            )
                assert split_info is not None
                partition = get_partition(device_dim, device_ids)
                if split_info.input_dim != output.shape[dim]:
                    assert output.shape[
                        dim] > 0 and output.shape[dim] % partition == 0
                output_split_info = SplitInfo(output.shape[dim], partition)
                self.set_split_info(output.name, device_id, dim,
                                    output_split_info)

    def get_local_context(self, layer: trt.ILayer) -> "GraphContext":
        local_context = GraphContext()
        for i in range(layer.num_inputs):
            input = layer.get_input(i)
            if input is None:
                continue
            local_context.tensor_contexts[input.name] = copy.deepcopy(
                self.tensor_contexts[input.name])
        return local_context

    def get_local_context_for_output(self,
                                     output: trt.ITensor) -> "GraphContext":
        local_context = GraphContext()
        local_context.tensor_contexts[output.name] = copy.deepcopy(
            self.tensor_contexts[output.name])
        return local_context

    def merge_context(self, context: "GraphContext"):
        self.tensor_contexts.update(context.tensor_contexts)


@dataclass
class ShardContext:
    graph_context: GraphContext
    layer: Layer
    nditer: np.nditer
    device_ids: np.ndarray
    strategy: ShardingStrategy


def get_partition(device_dim, device_ids):
    if device_dim == [0]:
        partition = device_ids.shape[0]
    elif device_dim == [1]:
        partition = device_ids.shape[1]
    else:
        assert device_dim == [0, 1] or device_dim == [1, 0]
        partition = device_ids.size
    return partition


def get_index(device_dim, iter):
    if device_dim == [0]:
        index = iter.multi_index[0]
    elif device_dim == [1]:
        index = iter.multi_index[1]
    else:
        assert device_dim == [0, 1] or device_dim == [1, 0]
        index = iter.iterindex
    return index


def get_full_sharding_spec(sharding_spec):
    return ShardingSpec(sharding_spec.device_mesh,
                        sharding_spec.data_type_size,
                        sharding_spec.entire_shape,
                        sharding_spec.max_entire_shape,
                        sharding_spec.raw_shape,
                        dim_partition_dict={})


def get_comm_action_sequence(from_sharding_sepc, to_sharding_sepc):
    comm_action_sequence = from_sharding_sepc.device_mesh.shape_consistency_manager.shape_consistency(
        from_sharding_sepc, to_sharding_sepc)[1]
    # TODO: should merged by shape_consistency
    if len(comm_action_sequence) == 2:
        if comm_action_sequence[0].comm_pattern == comm_action_sequence[
                1].comm_pattern == "all_gather":
            if comm_action_sequence[0].gather_dim == comm_action_sequence[
                    1].gather_dim:
                comm_action_sequence = [
                    CommSpec(
                        comm_action_sequence[0].comm_pattern,
                        comm_action_sequence[0].sharding_spec,
                        comm_action_sequence[0].gather_dim,
                        comm_action_sequence[0].shard_dim, [[
                            *comm_action_sequence[0].logical_process_axis[0],
                            *comm_action_sequence[1].logical_process_axis[0]
                        ]], comm_action_sequence[0].mix_gather,
                        comm_action_sequence[0].forward_only)
                ]
                assert len(comm_action_sequence[0].logical_process_axis[0]) <= 2
    assert len(comm_action_sequence) <= 1
    return comm_action_sequence


class GraphGroup(ABC):

    @staticmethod
    def from_graph(
        graph: PipelineGraph,
        config: ParallelConfig,
        auto_parallel_config: AutoParallelConfig,
    ) -> "GraphGroup":
        if auto_parallel_config.debug_mode:
            return PrefixedGraphGroup(graph, config, auto_parallel_config)
        else:
            return DistributedGraphGroup(graph, config, auto_parallel_config)

    @property
    @abstractmethod
    def auto_parallel_config(self) -> AutoParallelConfig:
        ...

    @abstractmethod
    def add_input(self, tensor, device_ids, strategy: ShardingStrategy):
        ...

    @abstractmethod
    def add_layer(self, layer, device_ids, strategy: ShardingStrategy):
        ...

    @abstractmethod
    def add_output(self, tensor, device_ids, sharding_spec: ShardingSpec):
        ...

    @abstractmethod
    def get_network(self, device_id) -> trt.INetworkDefinition:
        ...

    @abstractmethod
    def get_graph(self, device_id) -> PipelineGraph:
        ...

    @property
    @abstractmethod
    def full_graph(self) -> PipelineGraph:
        ...

    @abstractmethod
    def get_prefix(self, device_id) -> str:
        ...

    @abstractmethod
    def get_shapes(self, device_id) -> Dict[str, Tuple[int, ...]]:
        ...

    @abstractmethod
    def get_values(self, device_id) -> Dict[str, List[int]]:
        ...

    @abstractmethod
    def add_all_reduce_layer(self, context: GraphContext, input_name,
                             output_name, device_ids, to_reduce_tensors):
        ...

    @abstractmethod
    def add_all_gather_layer(self, context: GraphContext, input_name,
                             output_name, device_ids, to_gather_tensors):
        ...

    @abstractmethod
    def register_layer(self,
                       layer,
                       base_name,
                       input_name,
                       output_name=None,
                       device_id=None,
                       keep_tensor_name=False) -> Layer:
        ...

    def get_tensor(self, context: GraphContext, tensor_name: str,
                   device_id: int) -> Tensor:
        name = context.get_name(tensor_name, device_id)
        return self.get_graph(device_id).get_tensor(name)

    def add_comm(self,
                 context: GraphContext,
                 input_name,
                 device_ids,
                 commspec,
                 output_name=None,
                 is_singleton=False):
        remove_index = []
        for i, device_dim in enumerate(commspec.logical_process_axis):
            partition = get_partition(device_dim, device_ids)
            if partition == 1:
                remove_index.append(i)
        if len(remove_index) > 0:
            if commspec.comm_pattern in ["all_gather", "all_to_all"]:
                commspec.gather_dim = [
                    dim for i, dim in enumerate(commspec.gather_dim)
                    if i not in remove_index
                ]
            if commspec.comm_pattern in [
                    "split", "reduce_scatter", "all_to_all"
            ]:
                commspec.shard_dim = [
                    dim for i, dim in enumerate(commspec.shard_dim)
                    if i not in remove_index
                ]
            commspec.logical_process_axis = [
                dim for i, dim in enumerate(commspec.logical_process_axis)
                if i not in remove_index
            ]
        flatten_device_dim = list(
            itertools.chain.from_iterable(commspec.logical_process_axis))
        if flatten_device_dim == []:
            return
        if flatten_device_dim == [0, 1] or flatten_device_dim == [1, 0]:
            self._add_comm(context, input_name, device_ids, commspec,
                           output_name, is_singleton)
        elif flatten_device_dim == [0]:
            for i in range(device_ids.shape[1]):
                self._add_comm(context, input_name, device_ids[:, i:i + 1],
                               commspec, output_name, is_singleton)
        elif flatten_device_dim == [1]:
            for i in range(device_ids.shape[0]):
                self._add_comm(context, input_name, device_ids[i:i + 1, :],
                               commspec, output_name, is_singleton)
        else:
            raise RuntimeError(
                f"Invalid flatten device_dim: {flatten_device_dim}")

    def _add_comm(self,
                  context: GraphContext,
                  input_name,
                  device_ids,
                  commspec,
                  output_name=None,
                  is_singleton=False):
        input_tensors = [
            self.get_tensor(context, input_name, device_id.item())
            for device_id in np.nditer(device_ids)
        ]
        comm_pattern = commspec.comm_pattern
        if comm_pattern == "split":
            self.add_split(context, input_name, output_name, device_ids,
                           commspec.shard_dim, commspec.logical_process_axis)
        elif comm_pattern == "all_gather":
            self.add_all_gather(context, input_name, output_name, device_ids,
                                commspec.gather_dim,
                                commspec.logical_process_axis, is_singleton)
        elif comm_pattern == "all_reduce":
            self.add_all_reduce(context, input_name, output_name, device_ids)
        elif comm_pattern == "reduce_scatter":
            self.add_reduce_scatter(context, input_name, output_name,
                                    device_ids, commspec.shard_dim,
                                    commspec.logical_process_axis)
        elif comm_pattern == "all_to_all":
            self.add_all_to_all(context, input_name, output_name, device_ids,
                                commspec.gather_dim, commspec.shard_dim,
                                commspec.logical_process_axis)
        else:
            raise NotImplementedError
        output_tensors = [
            self.get_tensor(context, input_name, device_id.item())
            for device_id in np.nditer(device_ids)
        ]
        for input_tensor, output_tensor in zip(input_tensors, output_tensors):
            if input_tensor.dtype != output_tensor.dtype:
                raise ValueError(
                    f"Input tensor and output tensor should have the same dtype for communication layers, "
                    f"input dtype is {input_tensor.dtype} for {input_tensor.name}, "
                    f"but output dtype is {output_tensor.dtype} for {output_tensor.name}"
                )

    def add_all_reduce(self, context: GraphContext, input_name, output_name,
                       device_ids):
        dtype = str_dtype_to_trt(self.full_graph._plugin_config.dtype)
        to_reduce_tensors = []
        for device_id in np.nditer(device_ids):
            device_id = device_id.item()
            layer_info = (input_name, output_name, device_id)
            network = self.get_network(device_id)
            input_tensor = self.get_tensor(context, input_name,
                                           device_id).as_trt()
            input_dtype = input_tensor.dtype
            if input_dtype != dtype:
                to_reduce_tensor = self.cast(
                    network,
                    input_tensor,
                    dtype,
                    layer_info,
                )
            else:
                to_reduce_tensor = input_tensor
            to_reduce_tensors.append(to_reduce_tensor)
        self.add_all_reduce_layer(context, input_name, output_name, device_ids,
                                  to_reduce_tensors)
        if input_dtype != dtype:
            for device_id in np.nditer(device_ids):
                device_id = device_id.item()
                layer_info = (input_name, output_name, device_id)
                network = self.get_network(device_id)
                input_tensor = self.get_tensor(
                    context,
                    input_name,
                    device_id,
                ).as_trt()
                output_tensor = self.cast(
                    network,
                    input_tensor,
                    input_dtype,
                    layer_info,
                )
                context.update_name_mapping(
                    input_name,
                    device_id,
                    output_tensor.name,
                )

    def add_reduce_scatter(self, context: GraphContext, input_name, output_name,
                           device_ids, shard_dims, device_dims):
        self.add_all_reduce(context, input_name, output_name, device_ids)
        self.add_split(context, input_name, output_name, device_ids, shard_dims,
                       device_dims)

    # TODO: use native all_to_all operation
    def add_all_to_all(self, context: GraphContext, input_name, output_name,
                       device_ids, gather_dims, shard_dims, device_dims):
        self.add_all_gather(context, input_name, output_name, device_ids,
                            gather_dims, device_dims)
        self.add_split(context, input_name, output_name, device_ids, shard_dims,
                       device_dims)

    def get_item(self, network, tensor, index, layer_info):
        get_item_layer = network.add_slice(tensor, [index], [1], [1])
        self.register_layer(get_item_layer, f"get_item{index}", *layer_info)
        return get_item_layer.get_output(0)

    def get_shape(self, network, tensor, layer_info):
        shape_layer = network.add_shape(tensor)
        self.register_layer(shape_layer, "shape", *layer_info)
        return shape_layer.get_output(0)

    def concat(self, network, tensors, layer_info):
        concat_layer = network.add_concatenation(tensors)
        self.register_layer(concat_layer, "concat", *layer_info)
        return concat_layer.get_output(0)

    def flatten(self, network, tensor, layer_info):
        shuffle_layer = network.add_shuffle(tensor)
        shuffle_layer.reshape_dims = [-1]
        shuffle_layer.zero_is_placeholder = False
        self.register_layer(shuffle_layer, "flatten", *layer_info)
        return shuffle_layer.get_output(0)

    def reshape(self, network, tensor, reshape_dims, layer_info):
        reshape_layer = network.add_shuffle(tensor)
        reshape_layer.set_input(1, reshape_dims)
        reshape_layer.zero_is_placeholder = False
        self.register_layer(reshape_layer, "reshape", *layer_info)
        return reshape_layer.get_output(0)

    def cast(self, network, tensor, dtype, layer_info):
        if tensor.dtype == dtype:
            return tensor
        cast_layer = network.add_cast(tensor, dtype)
        self.register_layer(cast_layer, "cast", *layer_info)
        return cast_layer.get_output(0)

    def const_int(self, network, name, value, layer_info):
        const_layer = network.add_constant(
            [1], np.array([value], dtype=trt_dtype_to_np(default_int_dtype)))
        self.register_layer(const_layer, name, *layer_info)
        return const_layer.get_output(0)

    def get_dim_size(self, network, tensor, dim, layer_info, shape_tensor=None):
        raw_shape = tensor.shape
        dim_size = raw_shape[dim]
        if dim_size != -1:
            return dim_size
        else:
            if shape_tensor is None:
                shape_tensor = self.get_shape(network, tensor, layer_info)
            return self.get_item(network, shape_tensor, dim, layer_info)

    def add_split(self, context: GraphContext, input_name, output_name,
                  device_ids, shard_dims, device_dims):
        it = np.nditer(device_ids, flags=['multi_index'])
        for device_id in it:
            device_id = device_id.item()
            layer_info = (input_name, output_name, device_id)
            network = self.get_network(device_id)
            input_tensor = self.get_tensor(context, input_name,
                                           device_id).as_trt()
            raw_input_shape = input_tensor.shape
            start = []
            output_dims = []
            stride = []
            input_shape_tensor = self.get_shape(network, input_tensor,
                                                layer_info)
            for dim in range(len(raw_input_shape)):
                stride.append(1)
                if dim not in shard_dims:
                    start.append(0)
                    output_dims.append(
                        self.get_item(network, input_shape_tensor, dim,
                                      layer_info))
                else:
                    start.append(None)
                    output_dims.append(None)

            for dim, device_dim in zip(shard_dims, device_dims):
                partition = get_partition(device_dim, device_ids)
                index = get_index(device_dim, it)
                input_dim = raw_input_shape[dim]
                assert input_dim != -1
                assert input_dim % partition == 0
                quotient = input_dim // partition
                start[dim] = index * quotient
                output_dims[dim] = self.const_int(network, f"output_dim{dim}",
                                                  quotient, layer_info)
                context.set_split_info(input_name, device_id, dim,
                                       SplitInfo(input_dim, partition))
            output_dims_tensor = self.concat(network, output_dims, layer_info)
            split_layer = network.add_slice(input_tensor, start, [], stride)
            split_layer.set_input(2, output_dims_tensor)
            wrapped_layer = self.register_layer(split_layer, "split",
                                                *layer_info)
            wrapped_layer.attrs["strategy"] = get_sharding_sequence(
                len(raw_input_shape),
                shard_dims,
                device_dims,
            )

            output_tensor = split_layer.get_output(0)
            context.update_name_mapping(input_name, device_id,
                                        output_tensor.name)

    def add_all_gather(self,
                       context: GraphContext,
                       input_name,
                       output_name,
                       device_ids,
                       gather_dims,
                       device_dims,
                       is_singleton=False):
        to_gather_tensors = []
        for device_id in np.nditer(device_ids):
            device_id = device_id.item()
            layer_info = (input_name, output_name, device_id)
            network = self.get_network(device_id)
            input_tensor = self.get_tensor(context, input_name,
                                           device_id).as_trt()
            to_gather_tensor = self.flatten(network, input_tensor, layer_info)
            to_gather_tensors.append(to_gather_tensor)

        all_gather_layers = self.add_all_gather_layer(
            context,
            input_name,
            output_name,
            device_ids,
            to_gather_tensors,
        )

        if len(device_dims) == 1:
            gather_indices = [0]
        elif len(device_dims) == 2 and device_dims[0] == [1]:
            gather_indices = [1, 0]
        else:
            gather_indices = [0, 1]

        for device_id, all_gather_layer in zip(np.nditer(device_ids),
                                               all_gather_layers):
            device_id = device_id.item()
            layer_info = (input_name, output_name, device_id)
            network = self.get_network(device_id)
            input_tensor = self.get_tensor(context, input_name,
                                           device_id).as_trt()
            permutation = []
            gathered_dims = []
            output_dims = []
            partitions = []
            raw_input_shape = input_tensor.shape

            wrapped_layer = self.get_graph(device_id).get_layer(
                all_gather_layer.name)
            wrapped_layer.attrs["strategy"] = get_sharding_sequence(
                len(raw_input_shape),
                gather_dims,
                device_dims,
            )

            input_shape_layer = network.add_shape(input_tensor)
            self.register_layer(input_shape_layer, "input_shape", *layer_info)
            input_shape_tensor = input_shape_layer.get_output(0)
            split_infos = context.get_split_infos(input_name, device_id)
            for index in gather_indices:
                gather_dim = gather_dims[index]
                device_dim = device_dims[index]
                partition = get_partition(device_dim, device_ids)
                assert partition == split_infos[gather_dim].partition
                partitions.append(
                    self.const_int(network, f"partition_num{gather_dim}",
                                   partition, layer_info))
            for dim in range(len(raw_input_shape)):
                if dim in gather_dims:
                    gather_index = gather_dims.index(dim)
                    device_dim = device_dims[gather_index]
                    permutation.append(gather_indices.index(gather_index))
                permutation.append(dim + len(gather_dims))
                if dim not in split_infos:
                    output_dim_layer = network.add_slice(
                        input_shape_tensor, [dim], [1], [1])
                    self.register_layer(output_dim_layer, f"output_dim{dim}",
                                        *layer_info)
                    dim_tensor = output_dim_layer.get_output(0)
                    output_dims.append(dim_tensor)
                    gathered_dims.append(dim_tensor)
                else:
                    input_dim = split_infos[dim].input_dim
                    partition = split_infos[dim].partition
                    assert input_dim != -1
                    assert input_dim % partition == 0
                    quotient = input_dim // partition
                    output_dims.append(
                        self.const_int(network, f"output_dim{dim}", quotient,
                                       layer_info))
                    if dim in gather_dims:
                        gathered_dims.append(
                            self.const_int(network, f"gathered_dim{dim}",
                                           quotient * partition, layer_info))
                        del split_infos[dim]
                    else:
                        gathered_dims.append(output_dim_layer.get_output(0))

            reshape_dims_for_transpose_layer = network.add_concatenation(
                [*partitions, *output_dims])
            self.register_layer(reshape_dims_for_transpose_layer,
                                "reshape_dims_for_transpose", *layer_info)
            reshape_dims_tensor = reshape_dims_for_transpose_layer.get_output(0)
            transpose_layer = network.add_shuffle(
                all_gather_layer.get_output(0))
            transpose_layer.set_input(1, reshape_dims_tensor)
            transpose_layer.second_transpose = permutation
            transpose_layer.zero_is_placeholder = False
            self.register_layer(transpose_layer, "transpose", *layer_info)

            reshape_dims_for_reshape_layer = network.add_concatenation(
                gathered_dims)
            self.register_layer(reshape_dims_for_reshape_layer,
                                "reshape_dims_for_reshape", *layer_info)
            reshape_dims_tensor = reshape_dims_for_reshape_layer.get_output(0)
            output_tensor = self.reshape(
                network,
                transpose_layer.get_output(0),
                reshape_dims_tensor,
                layer_info,
            )
            context.update_name_mapping(input_name, device_id,
                                        output_tensor.name)
            if is_singleton:
                break

    def register_unfilled_weights(self, graph, layer):
        if (layer.name in self.full_graph._unfilled_weights
                and layer.name not in graph._unfilled_weights):
            weights, values = self.full_graph._unfilled_weights[layer.name]
            graph._register_unfilled_weights(
                layer.name,
                weights,
                values,
            )

    def shard_constant(self, context: ShardContext):
        sharding_spec = context.strategy.sharding_specs["output0"]
        shard_dims = sharding_spec.dim_partition_dict
        device_id = context.nditer.value.item()
        device_ids = context.device_ids
        layer = context.layer.as_trt()
        graph = self.get_graph(device_id)
        if len(shard_dims) == 0:
            self.register_unfilled_weights(graph, layer)
            return LayerUpdate(split_info_updated=True)
        flatten_device_dim = list(
            itertools.chain.from_iterable(shard_dims.values()))
        output_name = layer.get_output(0).name
        output_dtype = layer.get_output(0).dtype
        output_shape = layer.shape
        output_dims = []
        weight_index = []
        for dim in range(len(output_shape)):
            output_dim = output_shape[dim]
            if dim in shard_dims:
                device_dim = shard_dims[dim]
                partition = get_partition(device_dim, device_ids)
                index = get_index(device_dim, context.nditer)
                assert output_dim % partition == 0
                quotient = output_dim // partition
                output_dims.append(quotient)
                weight_index.append(
                    slice(index * quotient, (index + 1) * quotient))
                context.graph_context.set_split_info(
                    output_name, device_id, dim,
                    SplitInfo(output_dim, partition))
            else:
                output_dims.append(output_dim)
                weight_index.append(slice(None))
        if layer.name in self.full_graph._unfilled_weights:
            values = self.full_graph._unfilled_weights[layer.name][1]
        else:
            values = layer.weights
            if isinstance(values, trt.Weights):
                values = values.numpy()
                # TODO: remove this WAR after https://nvbugs/4359151 fixed.
                if isinstance(values, trt.Weights):
                    network = context.layer.graph.as_trt()
                    values = get_np_weight(network, layer.name)
        if values is not None:
            values = values.reshape(layer.shape)
            assert values.size == np.prod(layer.shape)
            sharded_values = values[tuple(weight_index)]
            assert sharded_values.size * get_partition(
                flatten_device_dim, device_ids) == np.prod(layer.shape)
        else:
            sharded_values = None
        dtype = trt_dtype_to_np(output_dtype)
        sharded_weights = np.empty(tuple(output_dims), dtype)
        graph._register_unfilled_weights(
            f"device{device_id}_{layer.name}",
            sharded_weights,
            sharded_values,
        )
        sharded_weights = to_trt_weights(sharded_weights)
        return LayerUpdate(
            updated_attrs=dict(
                shape=trt.Dims(output_dims),
                weights=sharded_weights,
            ),
            split_info_updated=True,
        )

    def shard_fill(self, context: ShardContext):
        sharding_spec = context.strategy.sharding_specs["output0"]
        shard_dims = sharding_spec.dim_partition_dict
        if len(shard_dims) == 0:
            return LayerUpdate(split_info_updated=True)
        device_id = context.nditer.value.item()
        device_ids = context.device_ids
        layer = context.layer.as_trt()
        output_name = layer.get_output(0).name
        output_shape = layer.shape
        output_dims = []
        for dim in range(len(output_shape)):
            output_dim = output_shape[dim]
            if dim in shard_dims:
                device_dim = shard_dims[dim]
                partition = get_partition(device_dim, device_ids)
                assert output_dim % partition == 0
                quotient = output_dim // partition
                output_dims.append(quotient)
                context.graph_context.set_split_info(
                    output_name, device_id, dim,
                    SplitInfo(output_dim, partition))
            else:
                output_dims.append(output_dim)
        return LayerUpdate(
            updated_attrs=dict(shape=trt.Dims(output_dims), ),
            split_info_updated=True,
        )

    def update_shape(self, context: ShardContext):
        if not context.layer.is_shape_io:
            return
        layer = context.layer.as_trt()
        input_name = layer.get_input(0).name
        output_name = layer.get_output(0).name
        device_id = context.nditer.value.item()
        layer_info = (output_name, None, device_id)
        split_infos = context.graph_context.get_split_infos(
            input_name, device_id)
        if len(split_infos) == 0:
            return
        network = self.get_network(device_id)
        shape_tensor = self.get_tensor(context.graph_context, output_name,
                                       device_id).as_trt()
        output_dims = []
        for dim in range(len(context.layer.get_input(0).shape)):
            if dim not in split_infos:
                output_dim_layer = network.add_slice(shape_tensor, [dim], [1],
                                                     [1])
            else:
                input_dim = split_infos[dim].input_dim
                output_dim_layer = network.add_constant(
                    [1], np.array([input_dim], dtype=default_int_dtype))
            self.register_layer(output_dim_layer, f"output_dim{dim}",
                                *layer_info)
            output_dims.append(output_dim_layer.get_output(0))
        new_shape_layer = network.add_concatenation(output_dims)
        self.register_layer(new_shape_layer, "new_shape", *layer_info)
        new_shape_tensor = new_shape_layer.get_output(0)
        context.graph_context.update_name_mapping(output_name, device_id,
                                                  new_shape_tensor.name)

    def shard_slice(self, context: ShardContext):
        sharding_spec = context.strategy.sharding_specs["output0"]
        shard_dims = sharding_spec.dim_partition_dict
        if len(shard_dims) == 0:
            return LayerUpdate.none()
        device_id = context.nditer.value.item()
        network = self.get_network(device_id)
        device_ids = context.device_ids
        layer = context.layer.as_trt()
        output_dims = []
        updated_attrs = {}
        updated_inputs = {}

        if layer.num_inputs >= 3:
            raw_output_shape = layer.get_output(0).shape
            input_name = layer.get_input(2).name
            layer_info = (input_name, layer.name, device_id)
            shape_tensor = self.get_tensor(context.graph_context, input_name,
                                           device_id).as_trt()
            for dim in range(len(raw_output_shape)):
                output_dim_layer = network.add_slice(shape_tensor, [dim], [1],
                                                     [1])
                self.register_layer(output_dim_layer, f"output_dim{dim}",
                                    *layer_info)
                if dim in shard_dims:
                    device_dim = shard_dims[dim]
                    partition = get_partition(device_dim, device_ids)
                    partition_num_tensor = self.const_int(
                        network, f"partition_num{dim}", partition, layer_info)
                    quotient_layer = network.add_elementwise(
                        output_dim_layer.get_output(0), partition_num_tensor,
                        trt.ElementWiseOperation.FLOOR_DIV)
                    self.register_layer(quotient_layer, f"quotient{dim}",
                                        *layer_info)
                    output_dim = self.cast(network,
                                           quotient_layer.get_output(0),
                                           default_int_dtype, layer_info)
                    output_dims.append(output_dim)
                else:
                    output_dims.append(output_dim_layer.get_output(0))
            output_dims_layer = network.add_concatenation(output_dims)
            self.register_layer(output_dims_layer, "output_dims", *layer_info)
            updated_inputs[2] = output_dims_layer.get_output(0)
        else:
            output_shape = layer.shape
            for dim in range(len(output_shape)):
                output_dim = output_shape[dim]
                assert output_dim != -1
                if dim in shard_dims:
                    device_dim = shard_dims[dim]
                    partition = get_partition(device_dim, device_ids)
                    assert output_dim % partition == 0
                    quotient = output_dim // partition
                    output_dims.append(quotient)
                else:
                    output_dims.append(output_dim)
            updated_attrs["shape"] = trt.Dims(output_dims)
        return LayerUpdate(updated_attrs, updated_inputs)

    def shard_shuffle(self, context: ShardContext):
        sharding_spec = context.strategy.sharding_specs["output0"]
        shard_dims = sharding_spec.dim_partition_dict
        if len(shard_dims) == 0:
            return LayerUpdate.none()
        device_id = context.nditer.value.item()
        network = self.get_network(device_id)
        device_ids = context.device_ids
        layer = context.layer.as_trt()
        updated_attrs = {}
        updated_inputs = {}
        updated_reshape_dims = {}
        second_transpose = layer.second_transpose

        if layer.num_inputs >= 2:
            raw_output_shape = layer.get_output(0).shape
            input_name = layer.get_input(1).name
            layer_info = (input_name, layer.name, device_id)
            reshape_dims_tensor = self.get_tensor(context.graph_context,
                                                  input_name, device_id)
            reshape_dims = context.layer.get_input(1).value
            reshape_dims_tensor = reshape_dims_tensor.as_trt()
            for dim in range(len(raw_output_shape)):
                if second_transpose is not None:
                    reshape_dim = second_transpose[dim]
                else:
                    reshape_dim = dim
                output_dim_layer = network.add_slice(reshape_dims_tensor,
                                                     [reshape_dim], [1], [1])
                self.register_layer(output_dim_layer, f"output_dim{dim}",
                                    *layer_info)
                output_dim = reshape_dims[reshape_dim]
                if dim in shard_dims and output_dim != -1:
                    device_dim = shard_dims[dim]
                    partition = get_partition(device_dim, device_ids)
                    partition_num_tensor = self.const_int(
                        network, f"partition_num{dim}", partition, layer_info)
                    quotient_layer = network.add_elementwise(
                        output_dim_layer.get_output(0), partition_num_tensor,
                        trt.ElementWiseOperation.FLOOR_DIV)
                    self.register_layer(quotient_layer, f"quotient{dim}",
                                        *layer_info)
                    updated_reshape_dims[reshape_dim] = self.cast(
                        network,
                        quotient_layer.get_output(0),
                        default_int_dtype,
                        layer_info,
                    )
                else:
                    updated_reshape_dims[
                        reshape_dim] = output_dim_layer.get_output(0)
            updated_reshape_dims = list(
                map(lambda x: x[1], sorted(updated_reshape_dims.items())))
            reshape_dims_layer = network.add_concatenation(updated_reshape_dims)
            self.register_layer(reshape_dims_layer, "reshape_dims", *layer_info)
            updated_inputs[1] = reshape_dims_layer.get_output(0)
        else:
            reshape_dims = layer.reshape_dims
            if reshape_dims.__len__() < 0:
                return LayerUpdate.none()
            for dim in range(len(reshape_dims)):
                if second_transpose is not None:
                    reshape_dim = second_transpose[dim]
                else:
                    reshape_dim = dim
                output_dim = reshape_dims[reshape_dim]
                if dim in shard_dims and output_dim != -1:
                    device_dim = shard_dims[dim]
                    partition = get_partition(device_dim, device_ids)
                    quotient = output_dim // partition
                    updated_reshape_dims[reshape_dim] = quotient
                else:
                    updated_reshape_dims[reshape_dim] = output_dim
            updated_reshape_dims = list(
                map(lambda x: x[1], sorted(updated_reshape_dims.items())))
            updated_attrs["reshape_dims"] = trt.Dims(updated_reshape_dims)
        return LayerUpdate(updated_attrs, updated_inputs)

    def shard_gpt_attention(self, context: ShardContext):
        layer = context.layer.as_trt()
        plugin_info = get_plugin_info(
            self.full_graph.as_trt(),
            layer.name,
        )
        parser = IdxEntryParser(plugin_info)
        head_dim = 1 if parser.remove_input_padding else 2
        sharding_spec = context.strategy.sharding_specs[
            f"input{parser.get_index(IdxEntry.QKV_TENSOR)}"]
        shard_dims = sharding_spec.dim_partition_dict
        if head_dim not in shard_dims:
            return LayerUpdate.none()
        device_id = context.nditer.value.item()
        network = self.get_network(device_id)
        device_ids = context.device_ids
        updated_attrs = {}
        updated_inputs = {}
        device_dim = shard_dims[head_dim]
        partition = get_partition(device_dim, device_ids)
        index = get_index(device_dim, context.nditer)
        if parser.is_entry_used(IdxEntry.K_TENSOR):
            kv_sharding_spec = context.strategy.sharding_specs[
                f"input{parser.get_index(IdxEntry.K_TENSOR)}"]
            kv_shard_dims = kv_sharding_spec.dim_partition_dict
            if head_dim in kv_shard_dims:
                kv_device_dim = kv_shard_dims[head_dim]
                kv_partition = get_partition(kv_device_dim, device_ids)
            else:
                kv_partition = 1
        else:
            kv_partition = 1
        num_heads = plugin_info.pfc_as_ndarray["num_heads"].copy()
        num_kv_heads = plugin_info.pfc_as_ndarray["num_kv_heads"].copy()
        tp_size = plugin_info.pfc_as_ndarray["tp_size"].copy()
        tp_rank = plugin_info.pfc_as_ndarray["tp_rank"].copy()
        num_kv_heads = np.maximum(num_kv_heads // kv_partition, 1)
        num_heads = np.maximum(num_heads // partition, 1)
        tp_size[0] = partition
        tp_rank[0] = index

        new_plugin, new_plugin_info = get_updated_plugin(
            plugin_info,
            dict(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                tp_size=tp_size,
                tp_rank=tp_rank,
            ))
        prefix = self.get_prefix(device_id)
        new_layer_name = f"{prefix}{layer.name}"
        set_plugin_info(network, new_layer_name, new_plugin_info)
        updated_attrs["plugin"] = new_plugin
        return LayerUpdate(updated_attrs, updated_inputs)

    def shard_lookup(self, context: ShardContext):
        sharding_spec = context.strategy.sharding_specs["input1"]
        shard_dims = sharding_spec.dim_partition_dict
        if 0 not in shard_dims:
            return LayerUpdate.none()
        layer = context.layer.as_trt()
        plugin_info = get_plugin_info(
            self.full_graph.as_trt(),
            layer.name,
        )
        device_id = context.nditer.value.item()
        network = self.get_network(device_id)
        updated_attrs = {}
        device_dim = shard_dims[0]
        index = get_index(device_dim, context.nditer)
        rank = plugin_info.pfc_as_ndarray["rank"].copy()
        rank[0] = index

        new_plugin, new_plugin_info = get_updated_plugin(
            plugin_info, dict(rank=rank, ))
        prefix = self.get_prefix(device_id)
        new_layer_name = f"{prefix}{layer.name}"
        set_plugin_info(network, new_layer_name, new_plugin_info)
        updated_attrs["plugin"] = new_plugin
        return LayerUpdate(updated_attrs)


class GraphGroupBase(GraphGroup):

    def __init__(
        self,
        full_graph: PipelineGraph,
        config: ParallelConfig,
        auto_parallel_config: AutoParallelConfig,
    ) -> None:
        self._full_graph = full_graph
        self.config = config
        self._auto_parallel_config = auto_parallel_config
        self.infer_shape = auto_parallel_config.infer_shape
        self.global_context = GraphContext()
        self.shape_cache = {}
        self.suffix = 0
        self.current_block_id = -1

    @property
    def auto_parallel_config(self) -> AutoParallelConfig:
        return self._auto_parallel_config

    @property
    def full_graph(self) -> PipelineGraph:
        return self._full_graph

    def register_layer(self,
                       layer,
                       base_name,
                       input_name,
                       output_name=None,
                       device_id=None,
                       keep_tensor_name=False) -> Layer:
        layer_name = f"{base_name}_{input_name}"
        if device_id is not None:
            layer_name = f"{self.get_prefix(device_id)}{layer_name}"
        if output_name is not None:
            layer_name = f"{layer_name}_to_{output_name}"
        suffix = self.suffix
        self.suffix += 1
        layer_name = f"{layer_name}_{suffix}"
        if layer.type == trt.LayerType.PLUGIN_V2:
            network = self.get_network(device_id)
            plugin_info = get_plugin_info(network, layer.name)
            if plugin_info is not None:
                set_plugin_info(network, layer_name, plugin_info)
                delete_plugin_info(network, layer.name)
        layer.name = layer_name
        layer.metadata = layer.name
        if not keep_tensor_name:
            for i in range(layer.num_outputs):
                output_tensor = layer.get_output(i)
                assert output_tensor.shape.__len__() >= 0
                output_tensor.name = f"{layer.name}_output_{i}"
        wrapped_layer = self.get_graph(device_id).register_layer(layer)
        if self.current_block_id != -1:
            wrapped_layer.attrs["block_id"] = self.current_block_id
        wrapped_layer.attrs["role"] = "helper"
        if self.infer_shape:
            infer_per_layer_shapes(
                layer,
                self.get_shapes(device_id),
                self.get_values(device_id),
                self.shape_cache,
                is_shape_io=True,
            )
            wrapped_layer.assign_shapes(
                self.get_shapes(device_id),
                self.get_values(device_id),
            )
        return wrapped_layer

    def add_layer(self, wrapped_layer: Layer, device_ids,
                  strategy: ShardingStrategy):
        layer = wrapped_layer.as_trt()
        local_context = self.global_context.get_local_context(layer)
        self.current_block_id = wrapped_layer.attrs["block_id"]

        for i, input in enumerate(wrapped_layer.inputs):
            if input is None:
                continue
            if i not in strategy.best_resharding_cost:
                continue
            comm_action_sequence = strategy.best_resharding_cost[i][0][1]
            for commspec in comm_action_sequence:
                self.add_comm(local_context,
                              input.name,
                              device_ids,
                              commspec,
                              output_name=layer.name)

        it = np.nditer(device_ids, flags=['multi_index'])
        for device_id in it:
            device_id = device_id.item()

            layer_type = layer.type
            to_subclass_layer(layer)
            shard_context = ShardContext(
                local_context,
                wrapped_layer,
                it,
                device_ids,
                strategy,
            )
            if layer_type == trt.LayerType.CONSTANT:
                layer_update = self.shard_constant(shard_context)
            elif layer_type == trt.LayerType.FILL:
                layer_update = self.shard_fill(shard_context)
            elif layer_type == trt.LayerType.SLICE:
                layer_update = self.shard_slice(shard_context)
            elif layer_type == trt.LayerType.SHUFFLE:
                layer_update = self.shard_shuffle(shard_context)
            elif layer_type == trt.LayerType.PLUGIN_V2:
                if layer.plugin.plugin_type == "GPTAttention":
                    layer_update = self.shard_gpt_attention(shard_context)
                elif layer.plugin.plugin_type == "Lookup":
                    layer_update = self.shard_lookup(shard_context)
                else:
                    layer_update = LayerUpdate.none()
            else:
                layer_update = LayerUpdate.none()
            to_base_class_layer(layer)

            for i, updated_input in layer_update.updated_inputs.items():
                input_name = layer.get_input(i).name
                local_context.update_name_mapping(input_name, device_id,
                                                  updated_input.name)
                if layer.get_input(i).dtype != updated_input.dtype:
                    raise ValueError(
                        f"Input dtype mismatch for {layer.name}, "
                        f"expect {layer.get_input(i).dtype} for {input_name}, "
                        f"get {updated_input.dtype} for {updated_input.name}")

            prefix = self.get_prefix(device_id)
            new_wrapped_layer = self.get_graph(device_id).add_layer(
                layer,
                prefix=prefix,
                input_mapping=local_context.get_name_mapping(device_id,
                                                             prefix=prefix),
                updated_attrs=layer_update.updated_attrs,
            )
            new_wrapped_layer.attrs["strategy"] = strategy.name
            new_wrapped_layer.attrs["block_id"] = self.current_block_id
            new_layer = new_wrapped_layer.as_trt()

            if self.infer_shape:
                infer_per_layer_shapes(
                    new_layer,
                    self.get_shapes(device_id),
                    self.get_values(device_id),
                    self.shape_cache,
                    is_shape_io=wrapped_layer.is_shape_io,
                )
                new_wrapped_layer.assign_shapes(
                    self.get_shapes(device_id),
                    self.get_values(device_id),
                )

            for i in range(layer.num_outputs):
                output_tensor = new_layer.get_output(i)
                assert output_tensor.shape.__len__() >= 0
                local_context.update_name_mapping(
                    layer.get_output(i).name, device_id, output_tensor.name)

            if layer.type == trt.LayerType.SHAPE:
                self.update_shape(shard_context)

            self.global_context.update_layer_context(
                wrapped_layer,
                layer_update,
                local_context,
                device_id,
                device_ids,
                strategy.sharding_specs,
            )

        for i in range(layer.num_outputs):
            commspecs = strategy.communication_actions.get(f"output{i}")
            if commspecs is None:
                continue
            output = layer.get_output(i)
            for commspec in commspecs:
                self.add_comm(
                    self.global_context,
                    output.name,
                    device_ids,
                    commspec,
                )

        self.current_block_id = -1


class DistributedGraphGroup(GraphGroupBase):

    def __init__(
        self,
        full_graph: PipelineGraph,
        config: ParallelConfig,
        auto_parallel_config: AutoParallelConfig,
    ) -> None:
        super().__init__(full_graph, config, auto_parallel_config)
        self.graphs = {}
        self.io_tensor_shards = {}
        self.shapes_by_device = {}
        self.values_by_device = {}
        self.use_custom_all_reduce = False
        phy_mesh = config.graph_config.phy_mesh
        device_ids = phy_mesh.phy_devices_id
        for device_id in np.nditer(device_ids):
            device_id = device_id.item()
            graph = PipelineGraph.create_graph()
            graph._auto_parallel_config = {
                "io_shards": {},
                "mapping":
                Mapping(
                    world_size=device_ids.size,
                    rank=device_id,
                    gpus_per_node=device_ids.shape[1],
                    tp_size=device_ids.size // config.graph_config.num_stages,
                    pp_size=config.graph_config.num_stages,
                ),
            }
            self.graphs[device_id] = graph
            self.shapes_by_device[device_id] = {}
            self.values_by_device[device_id] = {}

    @contextlib.contextmanager
    def disable_infer_shape(self):
        infer_shape = self.infer_shape
        self.infer_shape = False
        yield
        self.infer_shape = infer_shape

    def get_network(self, device_id) -> trt.INetworkDefinition:
        return self.graphs[device_id].as_trt()

    def get_graph(self, device_id) -> PipelineGraph:
        return self.graphs[device_id]

    def get_prefix(self, device_id) -> str:
        return ""

    def get_shapes(self, device_id) -> Dict[str, Tuple[int, ...]]:
        return self.shapes_by_device[device_id]

    def get_values(self, device_id) -> Dict[str, List[int]]:
        return self.values_by_device[device_id]

    def add_reduce_scatter(self, context: GraphContext, input_name, output_name,
                           device_ids, shard_dims, device_dims):
        dtype = str_dtype_to_trt(self.full_graph._plugin_config.dtype)
        it = np.nditer(device_ids, flags=['multi_index'])
        for device_id in it:
            device_id = device_id.item()
            layer_info = (input_name, output_name, device_id)
            network = self.get_network(device_id)
            input_tensor = self.get_tensor(context, input_name,
                                           device_id).as_trt()
            raw_input_shape = input_tensor.shape
            input_shape_tensor = self.get_shape(network, input_tensor,
                                                layer_info)
            if shard_dims != [0]:
                permutation = list(range(len(raw_input_shape)))
                for dim in shard_dims:
                    permutation.remove(dim)
                permutation = shard_dims + permutation
                transpose_layer = network.add_shuffle(input_tensor)
                transpose_layer.second_transpose = permutation
                self.register_layer(transpose_layer, "input_transpose",
                                    *layer_info)
                input_tensor = transpose_layer.get_output(0)
            flatten_tensor = self.flatten(network, input_tensor, layer_info)
            input_dtype = flatten_tensor.dtype
            if input_dtype != dtype:
                to_reduce_tensor = self.cast(
                    network,
                    flatten_tensor,
                    dtype,
                    layer_info,
                )
            else:
                to_reduce_tensor = flatten_tensor

            reduce_scatter_plg_creator = trt.get_plugin_registry(
            ).get_plugin_creator('ReduceScatter', '1', TRT_LLM_PLUGIN_NAMESPACE)
            assert reduce_scatter_plg_creator is not None

            group = trt.PluginField(
                "group",
                np.ascontiguousarray(device_ids.reshape(-1).astype(np.int32)),
                trt.PluginFieldType.INT32)
            pf_type = trt.PluginField(
                "type_id", np.array([int(to_reduce_tensor.dtype)], np.int32),
                trt.PluginFieldType.INT32)

            pfc = trt.PluginFieldCollection([group, pf_type])
            rs_plug = reduce_scatter_plg_creator.create_plugin(
                "reduce_scatter", pfc)

            reduce_scatter_layer = network.add_plugin_v2([to_reduce_tensor],
                                                         rs_plug)
            plugin_info = PluginInfo(reduce_scatter_plg_creator,
                                     "reduce_scatter", pfc)
            set_plugin_info(network, reduce_scatter_layer.name, plugin_info)
            with self.disable_infer_shape():
                wrapped_tensor = self.register_layer(
                    reduce_scatter_layer,
                    "reduce_scatter",
                    *layer_info,
                ).get_output(0)
            reduce_scatter_tensor = reduce_scatter_layer.get_output(0)
            if self.infer_shape:
                shape = self.shapes_by_device[device_id][to_reduce_tensor.name]
                assert len(shape) == 1
                output_shape = (shape[0] // device_ids.size, )
                self.shapes_by_device[device_id][
                    reduce_scatter_tensor.name] = output_shape
                wrapped_tensor.shape = output_shape
            if input_dtype != dtype:
                reduce_scatter_tensor = self.cast(
                    network,
                    reduce_scatter_tensor,
                    input_dtype,
                    layer_info,
                )

            start = []
            output_dims = []
            stride = []
            for dim in range(len(raw_input_shape)):
                stride.append(1)
                if dim not in shard_dims:
                    start.append(0)
                    output_dims.append(
                        self.get_item(network, input_shape_tensor, dim,
                                      layer_info))
                else:
                    start.append(None)
                    output_dims.append(None)

            for dim, device_dim in zip(shard_dims, device_dims):
                partition = get_partition(device_dim, device_ids)
                index = get_index(device_dim, it)
                input_dim = raw_input_shape[dim]
                assert input_dim != -1
                assert input_dim % partition == 0
                quotient = input_dim // partition
                start[dim] = index * quotient
                output_dims[dim] = self.const_int(network, f"output_dim{dim}",
                                                  quotient, layer_info)
                context.set_split_info(input_name, device_id, dim,
                                       SplitInfo(input_dim, partition))
            if shard_dims != [0]:
                output_dims = [
                    output_dims[permutation[i]] for i in range(len(output_dims))
                ]
            output_dims_tensor = self.concat(network, output_dims, layer_info)
            output_tensor = self.reshape(
                network,
                reduce_scatter_tensor,
                output_dims_tensor,
                layer_info,
            )
            if shard_dims != [0]:
                transpose_layer = network.add_shuffle(output_tensor)
                transpose_layer.second_transpose = permutation
                self.register_layer(transpose_layer, "output_transpose",
                                    *layer_info)
                output_tensor = transpose_layer.get_output(0)
            context.update_name_mapping(input_name, device_id,
                                        output_tensor.name)

    def add_all_reduce_layer(self, context: GraphContext, input_name,
                             output_name, device_ids, to_reduce_tensors):
        counter = 0
        if self.use_custom_all_reduce:
            counter = current_all_reduce_helper().gen_id()
        for device_id, to_reduce_tensor in zip(np.nditer(device_ids),
                                               to_reduce_tensors):
            device_id = device_id.item()
            layer_info = (input_name, output_name, device_id)
            network = self.get_network(device_id)
            graph = self.get_graph(device_id)
            if self.use_custom_all_reduce:
                strategy = AllReduceStrategy.AUTO
                workspace = graph.get_input("all_reduce_workspace").as_trt()
            else:
                strategy = AllReduceStrategy.NCCL
                workspace = None

            all_reduce_layer, allreduce_plg_creator, pfc = create_allreduce_plugin(
                network=network,
                tensor=to_reduce_tensor,
                workspace=workspace,
                group=np.ascontiguousarray(
                    device_ids.reshape(-1).astype(np.int32)),
                strategy=strategy,
                dtype=to_reduce_tensor.dtype,
                config=AllReduceConfig(0),
                counter=counter,
                reduce_fusion_params=AllReduceFusionParams(),
            )
            plugin_info = PluginInfo(allreduce_plg_creator, "allreduce", pfc)
            set_plugin_info(network, all_reduce_layer.name, plugin_info)
            with self.disable_infer_shape():
                wrapped_tensor = self.register_layer(
                    all_reduce_layer,
                    "all_reduce",
                    *layer_info,
                ).get_output(0)
            output_tensor = all_reduce_layer.get_output(0)
            if self.infer_shape:
                shape = self.shapes_by_device[device_id][to_reduce_tensor.name]
                self.shapes_by_device[device_id][output_tensor.name] = shape
                wrapped_tensor.shape = shape
            context.update_name_mapping(input_name, device_id,
                                        output_tensor.name)

    def add_all_gather_layer(self, context: GraphContext, input_name,
                             output_name, device_ids, to_gather_tensors):
        all_gather_layers = []
        for device_id, to_gather_tensor in zip(np.nditer(device_ids),
                                               to_gather_tensors):
            device_id = device_id.item()
            layer_info = (input_name, output_name, device_id)
            network = self.get_network(device_id)

            allgather_plg_creator = trt.get_plugin_registry(
            ).get_plugin_creator('AllGather', '1', TRT_LLM_PLUGIN_NAMESPACE)
            assert allgather_plg_creator is not None

            group = trt.PluginField(
                "group",
                np.ascontiguousarray(device_ids.reshape(-1).astype(np.int32)),
                trt.PluginFieldType.INT32)
            pf_type = trt.PluginField(
                "type_id", np.array([int(to_gather_tensor.dtype)], np.int32),
                trt.PluginFieldType.INT32)
            pfc = trt.PluginFieldCollection([group, pf_type])
            allgather = allgather_plg_creator.create_plugin("allgather", pfc)

            all_gather_layer = network.add_plugin_v2([to_gather_tensor],
                                                     allgather)
            plugin_info = PluginInfo(allgather_plg_creator, "allgather", pfc)
            set_plugin_info(network, all_gather_layer.name, plugin_info)
            with self.disable_infer_shape():
                wrapped_tensor = self.register_layer(
                    all_gather_layer,
                    "all_gather",
                    *layer_info,
                ).get_output(0)
            if self.infer_shape:
                output_tensor = all_gather_layer.get_output(0)
                shape = self.shapes_by_device[device_id][to_gather_tensor.name]
                assert len(shape) == 1
                output_shape = (shape[0] * device_ids.size, )
                self.shapes_by_device[device_id][
                    output_tensor.name] = output_shape
                wrapped_tensor.shape = output_shape
            all_gather_layers.append(all_gather_layer)
        return all_gather_layers

    def set_shard_num(self, tensor_name, dim, shard_num):
        for graph in self.graphs.values():
            io_shards = graph._auto_parallel_config["io_shards"]
            if tensor_name not in io_shards:
                io_shards[tensor_name] = {}
            io_shards[tensor_name][dim] = shard_num

    def add_input(self, tensor: Tensor, device_ids, strategy: ShardingStrategy):
        context = self.global_context
        sharding_spec = strategy.sharding_specs["output0"]
        shard_dims = sharding_spec.dim_partition_dict
        for dim, device_dim in shard_dims.items():
            partition = get_partition(device_dim, device_ids)
            self.set_shard_num(tensor.name, dim, partition)
        for device_id in np.nditer(device_ids):
            device_id = device_id.item()
            graph = self.get_graph(device_id)
            new_input = graph.add_input(tensor.as_trt())
            shape = [*tensor.shape]
            if len(shard_dims) != 0:
                output_shape = [*tensor.raw_shape]
                for dim, device_dim in shard_dims.items():
                    partition = get_partition(device_dim, device_ids)
                    output_dim = output_shape[dim]
                    assert output_dim != -1
                    assert output_dim % partition == 0
                    quotient = output_dim // partition
                    output_shape[dim] = quotient
                    shape[dim] = quotient
                    assert tensor.value is None
                    context.set_split_info(tensor.name, device_id, dim,
                                           SplitInfo(output_dim, partition))
                new_input.raw_shape = output_shape
            context.update_name_mapping(tensor.name, device_id, tensor.name)
            if self.infer_shape:
                self.shapes_by_device[device_id][tensor.name] = tuple(shape)
                new_input.shape = tuple(shape)
                if tensor.value is not None:
                    self.values_by_device[device_id][tensor.name] = tensor.value
                    new_input.value = tensor.value

    def add_output(self, tensor: Tensor, device_ids,
                   strategy: ShardingStrategy):
        comm_action_sequence = strategy.best_resharding_cost[0][0][1]
        for commspec in comm_action_sequence:
            self.add_comm(self.global_context, tensor.name, device_ids,
                          commspec)
        for device_id in np.nditer(device_ids):
            device_id = device_id.item()
            graph = self.get_graph(device_id)
            output_name = tensor.name
            new_output_name = self.global_context.get_name(
                output_name, device_id)
            if new_output_name != output_name:
                suffix = self.suffix
                self.suffix += 1
                original_name = f"original_{output_name}_{suffix}"
                original_tensor = graph.get_tensor(output_name)
                original_tensor.as_trt().name = original_name
                output_tensor = graph.get_tensor(new_output_name)
                output_tensor.as_trt().name = output_name
                graph._tensors[original_name] = original_tensor
                graph._tensors[output_name] = output_tensor
                del graph._tensors[new_output_name]
            else:
                output_tensor = graph.get_tensor(output_name)
            trt_output = output_tensor.as_trt()
            if trt_output.is_shape_tensor:
                graph.add_output_shape(trt_output)
            else:
                graph.add_output(trt_output)
            trt_output.dtype = tensor.dtype
            if tensor.dtype != output_tensor.dtype:
                raise ValueError(
                    f"Output dtype mismatch, "
                    f"expect {tensor.dtype} for {tensor.name}, "
                    f"get {output_tensor.dtype} for {output_tensor.name}")

        shard_dims = strategy.sharding_specs["input0"].dim_partition_dict
        for dim, device_dim in shard_dims.items():
            partition = get_partition(device_dim, device_ids)
            self.set_shard_num(tensor.name, dim, partition)


class PrefixedGraphGroup(GraphGroupBase):

    def __init__(
        self,
        full_graph: PipelineGraph = None,
        config: ParallelConfig = None,
        auto_parallel_config: AutoParallelConfig = None,
    ) -> None:
        auto_parallel_config = auto_parallel_config or dict(
            infer_shape=False,
            validation_mode=False,
        )
        super().__init__(full_graph, config, auto_parallel_config)
        self.validation_mode = auto_parallel_config.validation_mode
        if not self.infer_shape:
            self.validation_mode = False
        self.prefixed_graph = PipelineGraph.create_graph()
        if self.validation_mode:
            self.layer_mapping = config.graph_config.graph_mapping.layer_mapping
            self.graph_strategy = config.graph_strategy
        self.shapes = {}
        self.values = {}
        self.timing_cache = None

    def get_network(self, device_id) -> trt.INetworkDefinition:
        return self.prefixed_graph.as_trt()

    def get_graph(self, device_id) -> PipelineGraph:
        return self.prefixed_graph

    def get_prefix(self, device_id) -> str:
        return f"device{device_id}_"

    def get_shapes(self, device_id) -> Dict[str, Tuple[int, ...]]:
        return self.shapes

    def get_values(self, device_id) -> Dict[str, List[int]]:
        return self.values

    def add_all_reduce_layer(self, context: GraphContext, input_name,
                             output_name, device_ids, to_reduce_tensors):
        reshaped_tensors = []
        for device_id, to_reduce_tensor in zip(np.nditer(device_ids),
                                               to_reduce_tensors):
            device_id = device_id.item()
            layer_info = (input_name, output_name, device_id)
            network = self.get_network(device_id)
            reshape_dims_tensor = self.concat(
                network,
                [
                    self.get_shape(network, to_reduce_tensor, layer_info),
                    self.const_int(network, "expanded_dim", 1, layer_info)
                ],
                layer_info,
            )
            reshaped_tensor = self.reshape(
                network,
                to_reduce_tensor,
                reshape_dims_tensor,
                layer_info,
            )
            reshaped_tensors.append(reshaped_tensor)

        for device_id in np.nditer(device_ids):
            device_id = device_id.item()
            layer_info = (input_name, output_name, device_id)
            input_tensor = self.get_tensor(context, input_name, 0).as_trt()
            num_dims = len(input_tensor.shape)
            network = self.get_network(device_id)
            concat_layer = network.add_concatenation(reshaped_tensors)
            concat_layer.axis = num_dims
            self.register_layer(concat_layer, "concat", *layer_info)
            reduce_layer = network.add_reduce(concat_layer.get_output(0),
                                              trt.ReduceOperation.SUM,
                                              axes=1 << num_dims,
                                              keep_dims=False)
            dtype = to_reduce_tensors[0].dtype
            reduce_layer.precision = dtype
            reduce_layer.set_output_type(0, dtype)
            self.register_layer(reduce_layer, "reduce", *layer_info)
            output_tensor = reduce_layer.get_output(0)

            context.update_name_mapping(input_name, device_id,
                                        output_tensor.name)

    def add_all_gather_layer(self, context: GraphContext, input_name,
                             output_name, device_ids, to_gather_tensors):
        all_gather_layers = []
        for device_id in np.nditer(device_ids):
            device_id = device_id.item()
            layer_info = (input_name, output_name, device_id)
            network = self.get_network(device_id)
            all_gather_layer = network.add_concatenation(to_gather_tensors)
            all_gather_layer.axis = 0
            self.register_layer(all_gather_layer, "all_gather", *layer_info)
            all_gather_layers.append(all_gather_layer)
        return all_gather_layers

    def add_input(self, tensor: Tensor, device_ids, strategy: ShardingStrategy):

        def add_identity():
            identity_layer = network.add_identity(input.as_trt())
            return identity_layer

        input = self.prefixed_graph.add_input(tensor.as_trt())
        if self.infer_shape:
            self.shapes[tensor.name] = tensor.shape
            input.shape = tensor.shape
            if tensor.value is not None:
                self.values[tensor.name] = tensor.value
                input.value = tensor.value
        network = self.get_network(None)
        if self.validation_mode:
            identity_layer = add_identity()
            identity_layer.get_output(0).name = f"ref_{tensor.name}"
            layer_info = (tensor.name, None, None)
            self.register_layer(identity_layer,
                                "identity",
                                *layer_info,
                                keep_tensor_name=True)
        input.attrs["strategy"] = strategy.name
        sharding_spec = strategy.sharding_specs["output0"]
        pre_sharding_sepc = get_full_sharding_spec(sharding_spec)
        comm_action_sequence = get_comm_action_sequence(pre_sharding_sepc,
                                                        sharding_spec)
        context = self.global_context
        for device_id in np.nditer(device_ids):
            device_id = device_id.item()
            layer_info = (tensor.name, None, device_id)
            context.update_name_mapping(tensor.name, device_id, tensor.name)
            if len(comm_action_sequence
                   ) == 0 and not tensor.as_trt().is_shape_tensor:
                identity_layer = add_identity()
                self.register_layer(identity_layer, "identity", *layer_info)
                context.update_name_mapping(
                    tensor.name,
                    device_id,
                    identity_layer.get_output(0).name,
                )
        for commspec in comm_action_sequence:
            self.add_comm(context, tensor.name, device_ids, commspec)

    def get_graph_in_range(self, graph_group, src_layer, layer_range,
                           device_ids, shapes, values):
        src_network = self.prefixed_graph.as_trt()
        graph = graph_group.prefixed_graph
        network = graph.as_trt()
        input_mapping = {}
        for device_id in np.nditer(device_ids):
            device_id = device_id.item()
            for i in range(src_layer.num_inputs):
                src_input = src_layer.get_input(i)
                if src_input is not None:
                    input = self.get_tensor(
                        self.global_context,
                        src_input.name,
                        device_id,
                    ).as_trt()
                    if graph.get_input(src_input.name) is not None:
                        new_input = graph_group.get_tensor(
                            graph_group.global_context,
                            src_input.name,
                            device_id,
                        ).as_trt()
                        input_mapping[input.name] = new_input.name
                        continue
                    if graph.get_tensor(input.name) is not None:
                        continue
                    shape = shapes[input.name]
                    assert input.name in values
                    value = values[input.name]
                    weights = np.asarray(value,
                                         dtype=trt_dtype_to_np(input.dtype))
                    weights = to_trt_weights(weights)
                    input_layer = network.add_constant(shape, weights)
                    new_input = input_layer.get_output(0)
                    new_input.name = input.name
                    graph.register_layer(input_layer)
        for i in layer_range:
            layer = src_network.get_layer(i)
            graph.add_layer(layer, input_mapping=input_mapping)

    def add_layer_singleton(self, output, device_ids, sharding_spec):
        assert self.prefixed_graph.get_tensor(output.name) is None
        network = self.prefixed_graph.as_trt()
        full_sharding_sepc = get_full_sharding_spec(sharding_spec)
        comm_action_sequence = get_comm_action_sequence(sharding_spec,
                                                        full_sharding_sepc)
        output_context = self.global_context.get_local_context_for_output(
            output)
        if len(comm_action_sequence) != 0:
            for commspec in comm_action_sequence[:-1]:
                self.add_comm(output_context, output.name, device_ids, commspec)
            self.add_comm(
                output_context,
                output.name,
                device_ids,
                comm_action_sequence[-1],
                is_singleton=True,
            )
        device_id = next(np.nditer(device_ids)).item()
        layer_info = (output.name, None, device_id)
        output_tensor = self.get_tensor(output_context, output.name,
                                        device_id).as_trt()
        singleton_layer = network.add_identity(output_tensor)
        singleton_layer.get_output(0).name = output.name
        self.register_layer(singleton_layer,
                            "singleton",
                            *layer_info,
                            keep_tensor_name=True)

    def add_layer(self, wrapped_layer: Layer, device_ids,
                  strategy: ShardingStrategy):
        graph = self.prefixed_graph
        network = graph.as_trt()
        start_layer_id = network.num_layers

        super().add_layer(wrapped_layer, device_ids, strategy)

        layer = wrapped_layer.as_trt()

        if self.validation_mode:
            is_shape = (wrapped_layer.is_shape_io
                        or layer.type == trt.LayerType.SHAPE)

            if not is_shape:
                self.current_block_id = wrapped_layer.attrs["block_id"]
                for i, wrapped_output in enumerate(wrapped_layer.outputs):
                    if wrapped_output.is_graph_output:
                        continue
                    output = wrapped_output.as_trt()
                    output_name = f"output{i}"
                    if strategy.communication_actions.get(
                            output_name) is not None:
                        output_name += "_after_comm"
                    sharding_spec = strategy.sharding_specs[output_name]
                    self.add_layer_singleton(output, device_ids, sharding_spec)
                self.current_block_id = -1
            end_layer_id = network.num_layers

            is_skip = (is_shape or layer.type == trt.LayerType.CONSTANT
                       or layer.name in self.layer_mapping)
            sharded = False
            for sharding_spec in strategy.sharding_specs.values():
                if len(sharding_spec.dim_partition_dict) > 0:
                    sharded = True
                    break
            if not sharded:
                is_skip = True

            ref_layer = graph.add_layer(layer, prefix="ref_")
            ref_layer.attrs["strategy"] = strategy.name
            ref_layer.attrs["block_id"] = wrapped_layer.attrs["block_id"]
            if layer.type == trt.LayerType.CONSTANT:
                self.register_unfilled_weights(graph, layer)

            if is_skip:
                return

            logger.debug(f"validating layer {layer.name}")

            layer_type = layer.type
            generated_input_values = {}
            to_subclass_layer(layer)
            if layer_type == trt.LayerType.PLUGIN_V2:
                if layer.plugin.plugin_type == "GPTAttention":
                    sharding_specs = {}
                    for name, sharding_spec in strategy.sharding_specs.items():
                        sharding_specs[name] = get_full_sharding_spec(
                            sharding_spec)
                    plugin_info = get_plugin_info(
                        self.full_graph.as_trt(),
                        layer.name,
                    )
                    generated_input_values = GPTAttentionPlugin.parameter_generator(
                        sharding_specs, plugin_info)
            to_base_class_layer(layer)

            validation_graph_group = PrefixedGraphGroup()
            validation_graph = validation_graph_group.prefixed_graph
            validation_graph._io_buffer_mapping = self.full_graph._io_buffer_mapping
            extra_input_values = {}
            validation_shapes = {}
            for i, wrapped_input in enumerate(wrapped_layer.inputs):
                if wrapped_input is None:
                    continue
                input = wrapped_input.as_trt()
                validation_shapes[input.name] = wrapped_input.shape
                if wrapped_input.value is None:
                    if i in generated_input_values:
                        extra_input_value = generated_input_values[i]
                    else:
                        extra_input_value = torch.empty(
                            tuple(wrapped_input.shape),
                            dtype=trt_dtype_to_torch(input.dtype),
                            device=torch.cuda.current_device(),
                        )
                        if torch.is_floating_point(extra_input_value):
                            extra_input_value.normal_()
                        # extra_input_value[:] = random.choice([2, 3, 5, 7])
                    extra_input_values[input.name] = extra_input_value
                    self.values[input.name] = extra_input_value
                    if wrapped_input.producer is not None:
                        node_name = wrapped_input.producer.name
                        output_index = wrapped_input.output_index
                    else:
                        node_name = wrapped_input.name
                        output_index = 0
                    sharding_spec = self.graph_strategy[
                        node_name].sharding_specs[f"output{output_index}"]
                    validation_graph_group.add_input(
                        wrapped_input,
                        device_ids,
                        ShardingStrategy(
                            sharding_specs={"output0": sharding_spec}),
                    )
                    validation_graph.get_input(
                        input.name).raw_shape = wrapped_input.shape

            self.get_graph_in_range(
                validation_graph_group,
                layer,
                range(start_layer_id, end_layer_id),
                device_ids,
                self.shapes,
                self.values,
            )

            for i, wrapped_output in enumerate(wrapped_layer.outputs):
                output = wrapped_output.as_trt()
                if wrapped_output.is_graph_output:
                    output_name = f"output{i}"
                    if strategy.communication_actions.get(
                            output_name) is not None:
                        output_name += "_after_comm"
                    sharding_spec = strategy.sharding_specs[output_name]
                    validation_graph_group.global_context.merge_context(
                        self.global_context.get_local_context_for_output(
                            output))
                    validation_graph_group.add_layer_singleton(
                        output, device_ids, sharding_spec)
                validation_graph.add_output(output)
                validation_shapes[output.name] = wrapped_output.shape
            if not self.timing_cache:
                self.timing_cache = network.builder.create_builder_config(
                ).create_timing_cache(b"")
            logger.debug(f"run validation graph for layer {layer.name}")
            validation_runner = validation_graph.get_runner(
                validation_shapes,
                self.values,
                timing_cache=self.timing_cache,
                opt_level=0,
            )
            values = validation_runner.run()
            refer_input_values = {}
            for wrapped_input in wrapped_layer.inputs:
                if wrapped_input is None:
                    continue
                if wrapped_input.value is not None:
                    refer_input_values[wrapped_input.name] = wrapped_input.value
            refer_graph, output_mapping = get_per_layer_graph(
                layer,
                validation_shapes,
                refer_input_values,
                is_shape_io=False,
            )
            refer_graph._io_buffer_mapping = self.full_graph._io_buffer_mapping
            for proxy_output, output in output_mapping.items():
                validation_shapes[proxy_output] = validation_shapes[output]
            logger.debug(f"run refer graph for layer {layer.name}")
            refer_runner = refer_graph.get_runner(
                validation_shapes,
                self.values,
                timing_cache=self.timing_cache,
                opt_level=0,
            )
            refer_outputs = refer_runner.run()
            for name, refer_output in refer_outputs.items():
                if name in output_mapping:
                    refer_output = refer_output.bool()
                output = values[name]
                # ∣output−refer_output∣ <= atol+rtol*∣refer_output∣
                atol = 1e-02
                rtol = 1e-02
                if not torch.allclose(
                        output,
                        refer_output,
                        rtol=rtol,
                        atol=atol,
                        equal_nan=True,
                ):
                    size = output.nelement()
                    diff = (output - refer_output).abs()
                    diff_index = (~torch.isnan(diff)) & (
                        diff > (atol + rtol * refer_output.abs()))
                    diff_output = diff[diff_index]
                    diff_size = diff_output.nelement()
                    logger.warning(
                        f"output {name} of {layer.name} is not accurate after parallelization. "
                        f"{diff_size} out of {size} elements ({diff_size / size * 100:.2f}%) are not close. "
                        f"max: {diff_output.max():.5f}, mean: {diff_output.float().mean():.5f}, std: {diff_output.float().std():.5f}. "
                        f"mean of reference: {refer_output.float().mean():.5f}, mean of output: {output.float().mean():.5f}."
                    )
            for name in extra_input_values.keys():
                del self.values[name]

    def add_output(self, tensor: Tensor, device_ids,
                   strategy: ShardingStrategy):
        trt_output = tensor.as_trt()
        comm_action_sequence = strategy.best_resharding_cost[0][0][1]
        for commspec in comm_action_sequence:
            self.add_comm(self.global_context, tensor.name, device_ids,
                          commspec)
        self.add_layer_singleton(trt_output, device_ids,
                                 strategy.sharding_specs["input0"])
        if trt_output.is_shape_tensor:
            output = self.prefixed_graph.add_output_shape(trt_output)
        else:
            output = self.prefixed_graph.add_output(trt_output)
        trt_output.dtype = tensor.dtype
        output.attrs["strategy"] = strategy.name

    def assign_shapes(self, shape_info: ShapeInfo):
        if self.validation_mode:
            shapes = {
                f"ref_{name}": shape
                for name, shape in shape_info.shapes.items()
            }
            values = {
                f"ref_{name}": value
                for name, value in shape_info.values.items()
            }
            self.shapes.update(shapes)
            self.values.update(values)
        shape_layers = get_shape_layers(self.prefixed_graph.as_trt())
        shape_info = ShapeInfo(self.shapes, self.values, shape_layers)
        self.prefixed_graph.assign_shapes(shape_info)


def parallelize(
    simplifier: Simplifier,
    config: ParallelConfig,
):
    auto_parallel_config = simplifier.config
    debug_mode = auto_parallel_config.debug_mode
    dump_path = auto_parallel_config.dump_path
    debug_outputs = auto_parallel_config.debug_outputs

    simplifier.infer_shapes(config.graph_config.num_micro_batches)
    network = simplifier.network
    graph = simplifier.graph
    phy_mesh = config.graph_config.phy_mesh
    # TODO: test device_ids = [[0]]
    device_ids = phy_mesh.phy_devices_id
    stage_phy_meshes = config.graph_config.stage_phy_meshes
    block_to_stage = config.graph_config.graph_mapping.block_to_stage
    graph_strategy = config.graph_strategy
    desimplify_strategy(
        graph,
        graph_strategy,
        config.graph_config.graph_mapping,
    )
    graph._plugin_config = simplifier.llm_network.plugin_config
    graph_group = GraphGroup.from_graph(graph, config, auto_parallel_config)

    use_custom_all_reduce = graph._plugin_config.use_custom_all_reduce
    if use_custom_all_reduce and not debug_mode:
        graph_group.use_custom_all_reduce = True
        init_all_reduce_helper()
        tp_size = phy_mesh.size // config.graph_config.num_stages
        shape = (CustomAllReduceHelper.POINTERS_PER_RANK * tp_size, )
        workspace = graph.as_trt().add_input(
            name="all_reduce_workspace",
            dtype=trt.int64,
            shape=shape,
        )
        tensor = graph.register_input(workspace)
        tensor.shape = shape
        graph_strategy["all_reduce_workspace"] = ShardingStrategy(
            sharding_specs={
                "output0":
                ShardingSpec(
                    device_mesh=phy_mesh.as_logical_mesh(),
                    data_type_size=tensor.dtype_str_size,
                    data_shape=shape,
                    max_data_shape=shape,
                    raw_data_shape=shape,
                    dim_partition_dict={},
                )
            })

    if dump_path is not None:
        lock = FileLock(f"{dump_path}/path.lock", thread_local=False)
        with lock:
            with open(f'{dump_path}/sharded_graph.log', 'w+') as file:
                config.print_graph_strategy(file)

    for input in graph.inputs:
        graph_group.add_input(input, device_ids, graph_strategy[input.name])
    for block in simplifier.blocks:
        stage_id = block_to_stage[block.block_id]
        stage_phy_mesh = stage_phy_meshes[stage_id]
        stage_device_ids = stage_phy_mesh.phy_devices_id.reshape(
            config.lmesh.mesh_shape)
        for i in block.sorted_layer_ids:
            layer = graph.get_layer(network.get_layer(i).name)
            layer.attrs["block_id"] = block.block_id
            graph_group.add_layer(
                layer,
                stage_device_ids,
                graph_strategy[layer.name],
            )
    for output in graph.outputs:
        graph_group.add_output(output, device_ids, graph_strategy[output.name])

    if debug_mode:
        new_graph = graph_group.prefixed_graph
        debug_outputs = debug_outputs or []
        if isinstance(debug_outputs, str):
            if debug_outputs == 'validation':
                debug_outputs = []
                for tensor in new_graph.tensors:
                    if tensor.name.startswith('ref_'):
                        original_name = tensor.name[4:]
                        original_tensor = new_graph.get_tensor(original_name)
                        if original_tensor is not None:
                            if not original_tensor.is_graph_io:
                                debug_outputs.append(tensor.name)
                                debug_outputs.append(original_name)
                            if original_tensor.is_graph_output:
                                debug_outputs.append(tensor.name)
            else:
                pattern = debug_outputs
                debug_outputs = []
                for tensor in new_graph.tensors:
                    if tensor.as_trt().is_shape_tensor:
                        continue
                    if tensor.producer is not None:
                        layer = tensor.producer
                        if layer.type == trt.LayerType.SHAPE:
                            continue
                    if re.match(pattern, tensor.name):
                        debug_outputs.append(tensor.name)
        for output_name in debug_outputs:
            trt_output = new_graph.get_tensor(output_name).as_trt()
            if trt_output.is_shape_tensor:
                output = new_graph.add_output_shape(trt_output)
            else:
                output = new_graph.add_output(trt_output)
        graph_group.assign_shapes(simplifier.shape_info)
        if dump_path is not None:
            with lock:
                new_graph.to_dot(
                    f'{dump_path}/sharded_graph.dot',
                    per_device=True,
                    per_block=True,
                    # ignore_shape_io=True,
                    extra_attrs=['strategy'],
                )
        return [new_graph]
    else:
        graphs = []
        for device_id in np.nditer(device_ids):
            device_id = device_id.item()
            graph = graph_group.graphs[device_id]
            graphs.append(graph)
        return graphs

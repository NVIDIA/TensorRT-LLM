import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

from tensorrt_llm.network import Network

from .config import AutoParallelConfig
from .device_mesh import PhysicalDeviceMesh
from .pipeline_graph import PipelineGraph
from .shape_info import ShapeInfo, ShapeType, get_shape_info
from .tensor_parallel.p2p_node import P2PType
from .utils import get_cache_key, get_sorted_layer_ids, silent_trt_logger


class StageType(Enum):
    START = 0
    BLOCK = 1
    END = 2


class BuildingBlock:

    def __init__(self, graph, layer_range) -> None:
        self.graph = graph
        self.layer_range = layer_range
        self.network = graph.as_trt()
        self.owned_inputs = {}
        self.is_edges_collected = False
        self.intra_edges = []
        self.src_inter_edges = []
        self.dst_inter_edges = []
        self.relative_src_inter_edges = []
        self.relative_dst_inter_edges = []
        self.relative_inter_edges = set()
        self.edge_hash = None
        self.outputs = None
        self.type_id = -1
        self.block_id = -1
        self.p2p_type = None
        self.is_superset = False
        self.is_subset = False
        self.sorted_layer_ids = []

    def collect_edges(self):
        if self.is_edges_collected:
            return
        for layer_index in self.layer_range:
            trt_layer = self.network.get_layer(layer_index)
            layer = self.graph.get_layer(trt_layer.name)
            layer_offset = layer.index - self.layer_range.start
            for input_index, input in enumerate(layer.inputs):
                if input is not None:
                    if input.is_graph_input:
                        is_owned = input.graph_input_index in self.owned_inputs
                        if not is_owned and np.all([
                                layer.index in self.layer_range or np.all([
                                    output.as_trt().is_shape_tensor
                                    for output in layer.outputs
                                ]) for layer, _ in input.consumers
                        ]):
                            self.owned_inputs[input.graph_input_index] = len(
                                self.owned_inputs)
                            is_owned = True
                        if is_owned:
                            self.intra_edges.append(
                                (-1, self.owned_inputs[input.graph_input_index],
                                 layer_offset, input_index))
                        else:
                            self.dst_inter_edges.append(
                                (-1, input.graph_input_index, layer_offset,
                                 input_index))
                    else:
                        src_layer_index = input.producer.index
                        if src_layer_index < self.layer_range.start or src_layer_index >= self.layer_range.stop:
                            self.dst_inter_edges.append(
                                (src_layer_index, input.output_index,
                                 layer_offset, input_index))
                        else:
                            src_layer_offset = src_layer_index - self.layer_range.start
                            self.intra_edges.append(
                                (src_layer_offset, input.output_index,
                                 layer_offset, input_index))
            for output_index, output in enumerate(layer.outputs):
                for dst_layer, dst_input_index in output.consumers:
                    dst_layer_index = dst_layer.index
                    if dst_layer_index < self.layer_range.start or dst_layer_index >= self.layer_range.stop:
                        self.src_inter_edges.append(
                            (layer_offset, output_index, dst_layer_index,
                             dst_input_index))
        self.edge_hash = tuple(self.intra_edges)
        self.outputs = sorted(
            set((edge[0], edge[1]) for edge in self.src_inter_edges))
        self.is_edges_collected = True

    def collect_relative_inter_edges(self, layer_to_block):
        self.collect_edges()
        for src_layer_index, src_output_index, dst_layer_index, dst_input_index in self.dst_inter_edges:
            if src_layer_index in layer_to_block:
                src_block = layer_to_block[src_layer_index]
                src_layer_offset = src_layer_index - src_block.layer_range.start
                dst = (self.type_id, dst_layer_index, dst_input_index)
                self.relative_dst_inter_edges.append(
                    (src_block.type_id, src_layer_offset, src_output_index,
                     *dst))
            else:
                self.relative_dst_inter_edges.append(
                    (-1, src_layer_index, src_output_index, self.type_id,
                     dst_layer_index, dst_input_index))
        self.relative_inter_edges = set(self.relative_dst_inter_edges +
                                        self.outputs)

    def get_input_names(self):
        self.collect_edges()
        input_tensor_names = []
        for edge in self.dst_inter_edges:
            layer_index = edge[0]
            output_index = edge[1]
            if layer_index == -1:
                tensor_name = self.network.get_input(output_index).name
            else:
                tensor_name = self.network.get_layer(layer_index).get_output(
                    output_index).name
            input_tensor_names.append(tensor_name)
        return input_tensor_names

    def get_input_mapping(self, last_blocks):
        input_mapping = {}
        for tensor_name, relative_edge in zip(self.get_input_names(),
                                              self.relative_dst_inter_edges):
            type_id = relative_edge[0]
            output_index = relative_edge[2]
            if type_id >= 0:
                last_block = last_blocks[type_id]
                layer_offset = relative_edge[1]
                mapped_layer_index = last_block.layer_range.start + layer_offset
                mapped_tensor_name = self.network.get_layer(
                    mapped_layer_index).get_output(output_index).name
                input_mapping[tensor_name] = mapped_tensor_name
            else:
                input_mapping[tensor_name] = tensor_name
        return input_mapping


@dataclass
class GraphMapping:
    layer_mapping: Dict[int, int] = None
    block_mapping: Dict[int, int] = None
    p2p_types: Dict[int, P2PType] = None
    p2p_tensors: Dict[int, List[str]] = None
    block_to_stage: Dict[int, int] = None
    same_spec_layer_mapping: Dict[str, str] = None


@dataclass
class GraphConfig:
    num_micro_batches: int = 1
    num_blocks: int = 1
    num_stages: int = 1
    has_cross_device: bool = False
    has_cross_host: bool = False
    graph_mapping: GraphMapping = None
    phy_mesh: PhysicalDeviceMesh = None
    stage_phy_meshes: List[PhysicalDeviceMesh] = None


class Simplifier:

    def __init__(self, network: Network, config: AutoParallelConfig):
        self.config = config
        self.sharded_io_allowlist = config.sharded_io_allowlist
        self.same_buffer_io = config.same_buffer_io
        self.same_spec_io = config.same_spec_io.copy()
        for key, value in self.same_buffer_io.items():
            if key not in self.same_spec_io:
                self.same_spec_io[key] = value

        self.llm_network = network
        self.network = network.trt_network
        self.module_to_layer_range_map = network._module_call_stack.module_to_layer_range_map
        self.graph = self.get_graph()
        self.init_layer_hash()

        module_tree = self.get_module_tree()
        building_blocks = self.collect_building_blocks(module_tree)
        blocks_by_module_hash = self.get_blocks_by_module_hash(building_blocks)
        self.blocks_by_edge_hash = self.get_blocks_by_edge_hash(
            blocks_by_module_hash)
        self.layer_to_block = self.get_layer_to_block()
        self.blocks = self.get_all_blocks()
        self.backbone_blocks = self.get_backbone_blocks()
        self.graph_mapping_for_shape = self.get_graph_mapping_for_shape()
        self.graph_for_shape = self.create_simplified_graph_for_shape()
        self.shape_info = None
        self.num_micro_batches = None

    def infer_shapes(self, num_micro_batches):
        if self.num_micro_batches == num_micro_batches:
            return
        with silent_trt_logger():
            self.shape_info = self.get_full_shape_info(num_micro_batches)
            self.graph.assign_shapes(self.shape_info)
            self.num_micro_batches = num_micro_batches

    def list_all_num_micro_batches(self):
        opt_batch_size = self.get_opt_batch_size()
        candidates = []
        for num_micro_batches in range(1, self.get_opt_batch_size() + 1):
            if opt_batch_size % num_micro_batches == 0:
                candidates.append(num_micro_batches)
        return candidates

    def get_graph(self):
        graph = PipelineGraph.from_trt(self.network)
        graph._unfilled_weights = self.llm_network._unfilled_weights.copy()
        graph._io_buffer_mapping
        for input in graph.inputs:
            input_name = input.name
            for pattern, repl in self.same_buffer_io.items():
                if re.match(pattern, input_name):
                    output_name = re.sub(pattern, repl, input_name)
                    output = graph.get_output(output_name)
                    if output is not None:
                        graph._io_buffer_mapping[output_name] = input_name
        return graph

    def get_opt_batch_size(self):
        input_tensors = self.llm_network._inputs
        num_profiles = len(list(input_tensors.values())[0].profiles)
        opt_batch_sizes = []
        for i in range(num_profiles):
            for input_tensor in input_tensors.values():
                shape_profile = input_tensor.profiles[i]
                opt_shape = shape_profile.opt
                for j in range(len(input_tensor.shape)):
                    name = input_tensor.trt_tensor.get_dimension_name(j)
                    if name == 'batch_size':
                        opt_batch_sizes.append(opt_shape[j])
        return min(opt_batch_sizes)

    def get_module_hash(self, layer_range):
        module_hash = ()
        for i in layer_range:
            assert i < self.network.num_layers, f"layer index {i} in {layer_range} out of range of {self.network.num_layers}"
            layer_name = self.network.get_layer(i).name
            layer = self.graph.get_layer(layer_name)
            module_hash += (layer.attrs["hash"], )
        return module_hash

    def get_network_hash(self) -> str:
        return str(self.get_module_hash(range(self.network.num_layers)))

    def collect_building_blocks(self, module_tree):
        building_blocks = {}
        queue = []
        for tree in module_tree["children"].values():
            queue.append(tree)
        while len(queue) > 0:
            while len(queue) > 0:
                tree = queue.pop(0)
                module_name = tree["name"]
                if module_name is None:
                    for child in tree["children"].values():
                        queue.append(child)
                    continue
                layer_range = self.module_to_layer_range_map[module_name]
                module_hash = self.get_module_hash(layer_range)
                if module_hash in building_blocks:
                    building_blocks[module_hash].append(tree)
                else:
                    building_blocks[module_hash] = [tree]
            for module_hash in [*building_blocks.keys()]:
                if len(building_blocks[module_hash]) == 1:
                    tree = building_blocks[module_hash][0]
                    for child in tree["children"].values():
                        queue.append(child)
                    del building_blocks[module_hash]
        blocks_by_module_hash = {
            module_hash: [
                BuildingBlock(self.graph,
                              self.module_to_layer_range_map[tree["name"]])
                for tree in trees
            ]
            for module_hash, trees in building_blocks.items()
        }
        building_blocks = []
        for block_list in blocks_by_module_hash.values():
            for block in block_list:
                building_blocks.append(block)
        building_blocks = sorted(building_blocks,
                                 key=lambda x: x.layer_range.start)
        if len(building_blocks) >= 2:
            for block, next_block in zip(building_blocks[:-1],
                                         building_blocks[1:]):
                block.layer_range = range(block.layer_range.start,
                                          next_block.layer_range.start)
        return building_blocks

    def get_all_blocks(self):
        building_blocks = []
        for block_list in self.blocks_by_edge_hash.values():
            for block in block_list:
                building_blocks.append(block)
        building_blocks = sorted(building_blocks,
                                 key=lambda x: x.layer_range.start)
        all_blocks = []
        current_layer_index = 0
        block_id = 0
        for block in building_blocks:
            assert current_layer_index <= block.layer_range.start
            if current_layer_index < block.layer_range.start:
                new_block = BuildingBlock(
                    self.graph,
                    range(current_layer_index, block.layer_range.start))
                new_block.block_id = block_id
                block_id += 1
                all_blocks.append(new_block)
            block.block_id = block_id
            block_id += 1
            all_blocks.append(block)
            current_layer_index = block.layer_range.stop
        if current_layer_index < self.graph.num_layers:
            new_block = BuildingBlock(
                self.graph, range(current_layer_index, self.graph.num_layers))
            new_block.block_id = block_id
            all_blocks.append(new_block)
        sorted_layer_ids = get_sorted_layer_ids(self.network)
        for block in all_blocks:
            block.collect_relative_inter_edges(self.layer_to_block)
            for layer_id in sorted_layer_ids:
                if layer_id in block.layer_range:
                    block.sorted_layer_ids.append(layer_id)
        return all_blocks

    def get_backbone_blocks(self):
        sorted_blocks = sorted(
            self.blocks_by_edge_hash.values(),
            key=lambda blocks: (len(blocks), len(blocks[0].layer_range)),
        )
        if len(sorted_blocks) == 0:
            return []
        else:
            return sorted_blocks[-1]

    def get_blocks_by_module_hash(self, blocks):
        blocks_by_module_hash = {}
        for block in blocks:
            module_hash = self.get_module_hash(block.layer_range)
            if module_hash not in blocks_by_module_hash:
                blocks_by_module_hash[module_hash] = []
            blocks_by_module_hash[module_hash].append(block)
        for module_hash in [*blocks_by_module_hash.keys()]:
            if len(blocks_by_module_hash[module_hash]) == 1:
                del blocks_by_module_hash[module_hash]
        return blocks_by_module_hash

    def get_module_tree(self):
        module_tree = {"children": {}, "name": None}
        for module_name in self.module_to_layer_range_map.keys():
            full_name = module_name.split('.')
            current_tree = module_tree["children"]
            for depth, name in enumerate(full_name):
                if name not in current_tree:
                    current_tree[name] = {"children": {}, "name": None}
                if depth == len(full_name) - 1:
                    current_tree[name]["name"] = module_name
                else:
                    current_tree = current_tree[name]["children"]
        return module_tree

    def get_blocks_by_edge_hash(self, blocks_by_module_hash):
        blocks_by_edge_hash = {}
        for block_list in blocks_by_module_hash.values():
            for block in block_list:
                block.collect_edges()
                edge_hash = block.edge_hash
                if edge_hash not in blocks_by_edge_hash:
                    blocks_by_edge_hash[edge_hash] = []
                blocks_by_edge_hash[edge_hash].append(block)
        for edge_hash in [*blocks_by_edge_hash.keys()]:
            if len(blocks_by_edge_hash[edge_hash]) == 1:
                del blocks_by_edge_hash[edge_hash]
            else:
                block_list = blocks_by_edge_hash[edge_hash]
                blocks_by_edge_hash[edge_hash] = sorted(
                    block_list, key=lambda x: x.layer_range.start)
        for type_id, block_list in enumerate(blocks_by_edge_hash.values()):
            for block in block_list:
                block.type_id = type_id
        return blocks_by_edge_hash

    def get_layer_to_block(self):
        layer_to_block = {}
        for block_list in self.blocks_by_edge_hash.values():
            for block in block_list:
                for layer_index in block.layer_range:
                    layer_to_block[layer_index] = block
        return layer_to_block

    def clean_blocks(self):
        for block in self.blocks:
            block.p2p_type = None
            block.is_superset = False
            block.is_subset = False

    def mark_p2p_type(self, phy_mesh, stage_phy_meshes,
                      graph_config: GraphConfig):
        if len(self.backbone_blocks) == 0 or len(stage_phy_meshes) == 1:
            return
        assert len(self.backbone_blocks) % len(stage_phy_meshes) == 0
        block_per_stage = len(self.backbone_blocks) // len(stage_phy_meshes)

        for block in self.backbone_blocks:
            block.p2p_type = None
        for stage_index, stage_phy_mesh in enumerate(stage_phy_meshes[:-1]):
            next_stage_phy_mesh = stage_phy_meshes[stage_index + 1]
            last_device_id = stage_phy_mesh.phy_devices_id.flatten()[-1]
            next_first_device_id = next_stage_phy_mesh.phy_devices_id.flatten(
            )[0]
            num_devices_per_host = phy_mesh.num_devices_per_host
            next_block = self.backbone_blocks[(stage_index + 1) *
                                              block_per_stage]
            if last_device_id // num_devices_per_host != next_first_device_id // num_devices_per_host:
                next_block.p2p_type = P2PType.CROSS_HOST
                graph_config.has_cross_host = True
            else:
                next_block.p2p_type = P2PType.CROSS_DEVICE
                graph_config.has_cross_device = True

    def get_graph_mapping(self):
        layer_mapping = {}
        block_mapping = {}
        p2p_types = {}
        p2p_tensors = {}
        for block_list in self.blocks_by_edge_hash.values():
            superset_blocks = []
            superset_block_index = {}
            for block in block_list:
                block_added = False
                for index, superset_block in enumerate(list(superset_blocks)):
                    if block.p2p_type == superset_block.p2p_type:
                        if block.relative_inter_edges.issubset(
                                superset_block.relative_inter_edges):
                            block.is_subset = True
                            block.is_superset = False
                            superset_block_index[id(block)] = index
                            block_added = True
                            break
                        elif superset_block.relative_inter_edges.issubset(
                                block.relative_inter_edges):
                            superset_block.is_subset = True
                            superset_block.is_superset = False
                            block.is_subset = False
                            block.is_superset = True
                            superset_blocks[index] = block
                            superset_block_index[id(block)] = index
                            block_added = True
                            break
                if not block_added:
                    block.is_subset = False
                    block.is_superset = True
                    superset_blocks.append(block)
                    superset_block_index[id(block)] = len(superset_blocks) - 1
            for block in block_list:
                assert not (block.is_subset and block.is_superset)
                if block.is_subset:
                    superset_block = superset_blocks[superset_block_index[id(
                        block)]]
                    block_mapping[block.block_id] = superset_block.block_id
                    owned_inputs = map(
                        lambda x: x[0],
                        sorted(block.owned_inputs.items(), key=lambda x: x[1]))
                    superset_owned_inputs = map(
                        lambda x: x[0],
                        sorted(superset_block.owned_inputs.items(),
                               key=lambda x: x[1]))
                    for from_input_id, to_input_id in zip(
                            owned_inputs, superset_owned_inputs):
                        from_input_name = self.network.get_input(
                            from_input_id).name
                        to_input_name = self.network.get_input(to_input_id).name
                        layer_mapping[from_input_name] = to_input_name
                    for from_layer_id, to_layer_id in zip(
                            block.layer_range, superset_block.layer_range):
                        from_layer = self.network.get_layer(from_layer_id)
                        to_layer = self.network.get_layer(to_layer_id)
                        layer_mapping[from_layer.name] = to_layer.name
                        for i in range(from_layer.num_outputs):
                            from_output = from_layer.get_output(i)
                            if from_output.is_network_output:
                                to_output = to_layer.get_output(i)
                                layer_mapping[from_output.name] = to_output.name
                    if block.p2p_type is not None:
                        p2p_types[block.block_id] = block.p2p_type
                        p2p_tensors[block.block_id] = [
                            *set(block.get_input_names())
                        ]
                        for from_name, to_name in zip(
                                block.get_input_names(),
                                superset_block.get_input_names()):
                            layer_mapping[
                                f"p2p_block{block.block_id}_{from_name}"] = f"p2p_block{superset_block.block_id}_{to_name}"
        stage_id = 0
        block_to_stage = {}
        for block in self.blocks:
            if block.p2p_type is not None:
                stage_id += 1
            block_to_stage[block.block_id] = stage_id
        return GraphMapping(
            layer_mapping,
            block_mapping,
            p2p_types,
            p2p_tensors,
            block_to_stage,
        )

    def create_simplified_graph(self, graph_config: GraphConfig):
        new_graph = PipelineGraph.create_graph()
        new_graph._io_buffer_mapping = self.graph._io_buffer_mapping
        layer_mapping = graph_config.graph_mapping.layer_mapping

        for i in range(self.network.num_inputs):
            trt_input = self.network.get_input(i)
            if trt_input.name not in layer_mapping:
                new_graph.add_input(trt_input)

        last_blocks = {}
        same_spec_mapping = {}
        same_spec_layer_mapping = {}
        shape_mapping = {}
        building_block_id = 0
        same_spec_ids = {}
        same_spec_count = 0
        for block in self.blocks:
            if not block.is_subset:
                stage_type = None
                if not block.is_superset:
                    if block.block_id == 0:
                        stage_type = StageType.START
                    elif block.block_id == len(self.blocks) - 1:
                        stage_type = StageType.END
                input_mapping = block.get_input_mapping(last_blocks)
                for from_name, to_name in [*input_mapping.items()]:
                    if to_name in same_spec_mapping:
                        input_mapping[from_name] = same_spec_mapping[to_name]
                    if to_name in layer_mapping:
                        input_mapping[from_name] = layer_mapping[to_name]
                if block.is_superset and block.p2p_type is not None:
                    for from_name, to_name in [*input_mapping.items()]:
                        output_tensor = new_graph.get_tensor(to_name)
                        p2p_layer = new_graph.as_trt().add_identity(
                            output_tensor.as_trt())
                        p2p_layer.name = f"p2p_block{block.block_id}_{from_name}"
                        p2p_layer.metadata = p2p_layer.name
                        p2p_tensor = p2p_layer.get_output(0)
                        p2p_tensor.name = f"{p2p_layer.name}_output"
                        wrapped_layer = new_graph.register_layer(p2p_layer)
                        wrapped_layer.attrs[
                            "building_block_id"] = building_block_id
                        wrapped_layer.attrs["p2p_type"] = block.p2p_type
                        input_mapping[from_name] = p2p_tensor.name
                        shape_mapping[p2p_tensor.name] = from_name
                    building_block_id += 1
                for i in block.sorted_layer_ids:
                    layer = self.network.get_layer(i)
                    wrapped_layer = new_graph.add_layer(
                        layer,
                        input_mapping=input_mapping,
                    )
                    wrapped_layer.attrs["building_block_id"] = building_block_id
                    wrapped_layer.attrs["stage_type"] = stage_type
                if block.is_superset:
                    last_blocks[block.type_id] = block

                    if block.type_id in same_spec_ids:
                        same_spec_id = same_spec_ids[block.type_id]
                        update_same_spec_count = False
                    else:
                        same_spec_id = same_spec_count
                        same_spec_ids[block.type_id] = same_spec_id
                        update_same_spec_count = True
                    count = same_spec_id
                    for i, (layer_offset,
                            output_index) in enumerate(block.outputs):
                        layer = self.network.get_layer(block.layer_range.start +
                                                       layer_offset)
                        tensor_name = layer.get_output(output_index).name
                        output_tensor = new_graph.get_tensor(tensor_name)
                        same_spec_layer = new_graph.as_trt().add_identity(
                            output_tensor.as_trt())
                        same_spec_layer.name = f"{tensor_name}_same_spec"
                        same_spec_layer.metadata = same_spec_layer.name
                        same_spec_tensor = same_spec_layer.get_output(0)
                        same_spec_tensor.name = f"{same_spec_layer.name}_output"
                        wrapped_layer = new_graph.register_layer(
                            same_spec_layer)
                        wrapped_layer.attrs[
                            "building_block_id"] = building_block_id
                        wrapped_layer.attrs["same_spec_id"] = count
                        count += 1
                        same_spec_mapping[tensor_name] = same_spec_tensor.name
                        same_spec_layer_mapping[
                            same_spec_layer.name] = layer.name
                        shape_mapping[same_spec_tensor.name] = tensor_name
                    for i, graph_input_index in enumerate(
                            block.owned_inputs.keys()):
                        input_name = self.network.get_input(
                            graph_input_index).name
                        input_tensor = new_graph.get_input(input_name)
                        input_tensor.attrs["same_spec_id"] = count
                        count += 1
                    if update_same_spec_count:
                        same_spec_count = count
                building_block_id += 1
        graph_config.graph_mapping.same_spec_layer_mapping = same_spec_layer_mapping

        if len(self.backbone_blocks) >= 2:
            start_block = self.backbone_blocks[0]
            if start_block.is_subset:
                start_block = self.blocks[graph_config.graph_mapping.
                                          block_mapping[start_block.block_id]]
            for i in start_block.layer_range:
                layer_name = self.network.get_layer(i).name
                layer = new_graph.get_layer(layer_name)
                layer.attrs["in_start_block"] = True
            end_block = self.backbone_blocks[-1]
            if end_block.is_subset:
                end_block = self.blocks[graph_config.graph_mapping.
                                        block_mapping[end_block.block_id]]
            for i in end_block.layer_range:
                layer_name = self.network.get_layer(i).name
                layer = new_graph.get_layer(layer_name)
                layer.attrs["in_end_block"] = True
        slowest_p2p_type = None
        if graph_config.has_cross_host:
            slowest_p2p_type = P2PType.CROSS_HOST
        elif graph_config.has_cross_device:
            slowest_p2p_type = P2PType.CROSS_DEVICE
        if slowest_p2p_type is not None:
            for block in self.blocks:
                if block.is_superset and block.p2p_type == slowest_p2p_type:
                    for i in block.layer_range:
                        layer_name = self.network.get_layer(i).name
                        layer = new_graph.get_layer(layer_name)
                        layer.attrs["in_slowest_block"] = True

        for i in range(self.network.num_outputs):
            trt_output = self.network.get_output(i)
            output = self.graph.get_output(trt_output.name)
            if output.producer is not None and output.producer.index in self.layer_to_block and self.layer_to_block[
                    output.producer.index].is_subset:
                continue
            if trt_output.is_shape_tensor:
                new_output = new_graph.add_output_shape(trt_output)
            else:
                new_output = new_graph.add_output(trt_output)
            sharded_io = False
            for pattern in self.sharded_io_allowlist:
                if re.match(pattern, new_output.name):
                    sharded_io = True
                    break
            if not sharded_io:
                new_output.producer.attrs["is_replicated"] = True

        for input in new_graph.inputs:
            input_name = input.name
            sharded_io = False
            for pattern in self.sharded_io_allowlist:
                if re.match(pattern, input_name):
                    sharded_io = True
                    break
            if not sharded_io:
                input.attrs["is_replicated"] = True
            for pattern, repl in self.same_spec_io.items():
                if re.match(pattern, input_name):
                    output_name = re.sub(pattern, repl, input_name)
                    output = new_graph.get_output(output_name)
                    if output is not None:
                        if "same_spec_id" in input.attrs:
                            same_spec_id = input.attrs["same_spec_id"]
                        else:
                            same_spec_id = same_spec_count
                            same_spec_count += 1
                            input.attrs["same_spec_id"] = same_spec_id
                        output.attrs["same_spec_id"] = same_spec_id
                        if math.prod(self.graph.get_input(
                                input_name).shape) < math.prod(
                                    self.graph.get_output(output_name).shape):
                            input.attrs["no_memory_footprint"] = True
                        else:
                            output.attrs["no_memory_footprint"] = True

        return new_graph, shape_mapping

    def enrich_shape_info(self, shape_mapping):
        shapes = self.shape_info.shapes.copy()
        max_shapes = self.shape_info.max_shapes.copy()
        values = self.shape_info.values.copy()
        shape_layers = self.shape_info.shape_layers
        for from_name, to_name in shape_mapping.items():
            if to_name in shapes:
                shapes[from_name] = shapes[to_name]
            if to_name in max_shapes:
                max_shapes[from_name] = max_shapes[to_name]
            if to_name in values:
                values[from_name] = values[to_name]
        shape_info = ShapeInfo(shapes, values, shape_layers, max_shapes)
        return shape_info

    def simplify_graph(
            self, phy_mesh: PhysicalDeviceMesh, num_stages: int,
            num_devices_per_stage: int) -> Tuple[PipelineGraph, GraphConfig]:
        num_blocks = len(self.backbone_blocks)
        if num_blocks % num_stages != 0:
            return None, None
        graph_config = GraphConfig()
        graph_config.num_micro_batches = self.num_micro_batches
        graph_config.num_blocks = num_blocks
        graph_config.num_stages = num_stages
        graph_config.phy_mesh = phy_mesh
        stage_phy_meshes = phy_mesh.split_pipeline_meshes(
            num_stages, num_devices_per_stage)
        graph_config.stage_phy_meshes = stage_phy_meshes
        with silent_trt_logger():
            self.clean_blocks()
            self.mark_p2p_type(phy_mesh, stage_phy_meshes, graph_config)
            graph_config.graph_mapping = self.get_graph_mapping()
            new_graph, shape_mapping = self.create_simplified_graph(
                graph_config)
            shape_info = self.enrich_shape_info(shape_mapping)
            new_graph.assign_shapes(shape_info)
            return new_graph, graph_config

    def get_graph_mapping_for_shape(self):
        layer_mapping = {}
        tensor_mapping = {}
        for block_list in self.blocks_by_edge_hash.values():
            head_block = block_list[0]
            for block in block_list[1:]:
                for from_layer_id, to_layer_id in zip(block.layer_range,
                                                      head_block.layer_range):
                    from_layer = self.network.get_layer(from_layer_id)
                    to_layer = self.network.get_layer(to_layer_id)
                    layer_mapping[from_layer.name] = to_layer.name
                    for i in range(from_layer.num_outputs):
                        tensor_mapping[from_layer.get_output(
                            i).name] = to_layer.get_output(i).name
        return layer_mapping, tensor_mapping

    def create_simplified_graph_for_shape(self):
        new_graph = PipelineGraph.create_graph()

        for i in range(self.network.num_inputs):
            trt_input = self.network.get_input(i)
            new_graph.add_input(trt_input)

        head_blocks = {}
        removed_blocks = set()
        removed_layers = set()
        for block_list in self.blocks_by_edge_hash.values():
            head_block = block_list[0]
            head_blocks[head_block.type_id] = head_block
            for block in block_list[1:]:
                removed_blocks.add(id(block))
                for layer_index in block.layer_range:
                    removed_layers.add(layer_index)

        for block in self.blocks:
            if not id(block) in removed_blocks:
                input_mapping = block.get_input_mapping(head_blocks)
                for i in block.sorted_layer_ids:
                    layer = self.network.get_layer(i)
                    new_graph.add_layer(
                        layer,
                        input_mapping=input_mapping,
                    )

        for i in range(self.network.num_outputs):
            trt_output = self.network.get_output(i)
            output = self.graph.get_output(trt_output.name)
            if output.producer is not None and output.producer.index in removed_layers:
                continue
            if trt_output.is_shape_tensor:
                new_graph.add_output_shape(trt_output)
            else:
                new_graph.add_output(trt_output)

        return new_graph

    def get_full_shape_info(self, num_micro_batches):
        layer_mapping, tensor_mapping = self.graph_mapping_for_shape
        optimization_profiles = self.llm_network._generate_optimization_profiles(
        )
        if len(optimization_profiles) > 0:
            optimization_profile = optimization_profiles[-1]
        else:
            optimization_profile = None
        shape_info = get_shape_info(self.graph_for_shape.as_trt(),
                                    optimization_profile)
        max_shape_info = get_shape_info(self.graph_for_shape.as_trt(),
                                        optimization_profile,
                                        shape_type=ShapeType.MAX)
        shape_info.max_shapes = max_shape_info.shapes
        for removed_tensor_name, tensor_name in tensor_mapping.items():
            shape_info.shapes[removed_tensor_name] = shape_info.shapes[
                tensor_name]
            shape_info.max_shapes[removed_tensor_name] = shape_info.max_shapes[
                tensor_name]
            if tensor_name in shape_info.values:
                shape_info.values[removed_tensor_name] = shape_info.values[
                    tensor_name]
        for removed_layer_name, layer_name in layer_mapping.items():
            if layer_name in shape_info.shape_layers:
                shape_info.shape_layers.add(removed_layer_name)
        return shape_info

    def init_layer_hash(self):
        with silent_trt_logger():
            optimization_profiles = self.llm_network._generate_optimization_profiles(
            )
            if len(optimization_profiles) > 0:
                optimization_profile = optimization_profiles[-1]
            else:
                optimization_profile = None
            shape_info = get_shape_info(self.network, optimization_profile)
        dtypes = {tensor.name: tensor.dtype for tensor in self.graph.tensors}
        for layer in self.graph.layers:
            layer_hash = get_cache_key(
                layer.as_trt(),
                shape_info.shapes,
                shape_info.values,
                dtypes,
            )
            layer.attrs["hash"] = layer_hash

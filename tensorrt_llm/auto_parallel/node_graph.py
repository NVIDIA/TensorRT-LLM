from typing import List

import pandas as pd
import tensorrt as trt

from .pipeline_graph import PipelineGraph
from .runtime_profiling import RuntimeProfiler
from .simplifier import GraphConfig, StageType
from .solver import CostGraph, Solver
from .tensor_parallel.activation_node import Activation
from .tensor_parallel.assertion_node import Assertion
from .tensor_parallel.cast_node import Cast
from .tensor_parallel.concatenation_node import Concatenation
from .tensor_parallel.constant_node import Constant
from .tensor_parallel.elementwise_node import ElementWise
from .tensor_parallel.fill_node import Fill
from .tensor_parallel.gather_node import Gather
from .tensor_parallel.identity_node import Identity
from .tensor_parallel.input_node import InputNode
from .tensor_parallel.matmul_node import MatrixMultiply
from .tensor_parallel.node import Node
from .tensor_parallel.normalization_node import Normalization
from .tensor_parallel.output_node import OuputNode
from .tensor_parallel.p2p_node import P2PNode, P2PType
from .tensor_parallel.plugin_node import PluginNode
from .tensor_parallel.plugin_nodes.gemm_node import GemmPlugin
from .tensor_parallel.plugin_nodes.gpt_attention_node import GPTAttentionPlugin
from .tensor_parallel.plugin_nodes.identity_node import IdentityPlugin
from .tensor_parallel.plugin_nodes.look_up_node import LookupPlugin
from .tensor_parallel.plugin_nodes.normalization_node import (LayernormPlugin,
                                                              RMSnormPlugin)
from .tensor_parallel.reduce_node import Reduce
from .tensor_parallel.select_node import Select
from .tensor_parallel.shape_node import Shape
from .tensor_parallel.shuffle_node import Shuffle
from .tensor_parallel.slice_node import Slice
from .tensor_parallel.softmax_node import SoftMax
from .tensor_parallel.unary_node import Unary

LAYER_TYPE_2_NODE_TYPE = {
    trt.LayerType.ACTIVATION: Activation,
    trt.LayerType.ASSERTION: Assertion,
    trt.LayerType.CAST: Cast,
    trt.LayerType.CONCATENATION: Concatenation,
    trt.LayerType.CONSTANT: Constant,
    trt.LayerType.ELEMENTWISE: ElementWise,
    trt.LayerType.FILL: Fill,
    trt.LayerType.GATHER: Gather,
    trt.LayerType.IDENTITY: Identity,
    trt.LayerType.MATRIX_MULTIPLY: MatrixMultiply,
    trt.LayerType.NORMALIZATION: Normalization,
    trt.LayerType.PLUGIN_V2: PluginNode,
    trt.LayerType.REDUCE: Reduce,
    trt.LayerType.SELECT: Select,
    trt.LayerType.SHAPE: Shape,
    trt.LayerType.SHUFFLE: Shuffle,
    trt.LayerType.SLICE: Slice,
    trt.LayerType.SOFTMAX: SoftMax,
    trt.LayerType.UNARY: Unary,
}
# TODO: BertAttention/All Quant plugins
PLUGIN_LAYER_TYPE_2_NODE_TYPE = {
    'GPTAttention': GPTAttentionPlugin,
    'Gemm': GemmPlugin,
    'Layernorm': LayernormPlugin,
    'Rmsnorm': RMSnormPlugin,
    'Lookup': LookupPlugin,
    'Identity': IdentityPlugin,
}


class NodeGraph:

    def __init__(self, graph: PipelineGraph):
        self._nodes = {}

        # construct nodes
        for input in graph.inputs:
            self._nodes[input.name] = InputNode(input)
        for layer in graph.layers:
            layer.to_base_class()
            if "p2p_type" in layer.attrs:
                self._nodes[layer.name] = P2PNode(layer)
            elif layer.type == trt.LayerType.PLUGIN_V2:
                layer.to_subclass()
                plugin_type = layer.as_trt().plugin.plugin_type
                layer.to_base_class()
                if plugin_type in PLUGIN_LAYER_TYPE_2_NODE_TYPE:
                    node = PLUGIN_LAYER_TYPE_2_NODE_TYPE[plugin_type](layer)
                else:
                    node = LAYER_TYPE_2_NODE_TYPE[layer.type](layer)
                self._nodes[layer.name] = node
            else:
                node = LAYER_TYPE_2_NODE_TYPE[layer.type](layer)
                self._nodes[layer.name] = node
        for output in graph.outputs:
            self._nodes[output.name] = OuputNode(output)
        for node in self.nodes:
            node.post_init(self)
            node.node_runtime_profiler = RuntimeProfiler()

    def get_node(self, name):
        return self._nodes[name]

    @property
    def nodes(self) -> List[Node]:
        return [*self._nodes.values()]

    def assign_cost_weights(self, graph_config: GraphConfig):
        layer_mapping = graph_config.graph_mapping.layer_mapping
        for layer_name in layer_mapping.values():
            node = self.get_node(layer_name)
            node.sharding_weight += 1
            node.resharding_weight += 1
        same_spec_layer_mapping = graph_config.graph_mapping.same_spec_layer_mapping
        for same_spec_layer_name, layer_name in same_spec_layer_mapping.items():
            node = self.get_node(layer_name)
            same_spec_node = self.get_node(same_spec_layer_name)
            same_spec_node.sharding_weight = node.sharding_weight
            same_spec_node.resharding_weight = node.resharding_weight

    def set_slowest_stage(self, stage_type: StageType,
                          graph_config: GraphConfig):
        num_micro_batches = graph_config.num_micro_batches
        block_per_stage = graph_config.num_blocks // graph_config.num_stages
        block_pipeline_weight = block_per_stage * (num_micro_batches - 1)
        for node in self.nodes:
            node.pipeline_weight = 0
            node.cost_level = -1
            if node.stage_type == StageType.START:
                if stage_type == StageType.START:
                    node.pipeline_weight = num_micro_batches - 1
                    node.cost_level = 1
                else:
                    node.cost_level = 0
            if stage_type == StageType.START and node.in_start_block:
                node.pipeline_weight = block_pipeline_weight
            if node.stage_type == StageType.END:
                if stage_type == StageType.END:
                    node.pipeline_weight = num_micro_batches - 1
                    node.cost_level = 1
                else:
                    node.cost_level = 0
            if stage_type == StageType.END and node.in_end_block:
                node.pipeline_weight = block_pipeline_weight
            if isinstance(node, P2PNode):
                if (graph_config.has_cross_host
                        and node.p2p_type == P2PType.CROSS_HOST) or (
                            not graph_config.has_cross_host
                            and node.p2p_type == P2PType.CROSS_DEVICE):
                    if stage_type == StageType.BLOCK:
                        node.pipeline_weight += num_micro_batches - 1
                        node.cost_level = 1
                    else:
                        node.cost_level = 0
                elif (graph_config.has_cross_device
                      and node.p2p_type == P2PType.CROSS_DEVICE) or (
                          not graph_config.has_cross_device
                          and node.p2p_type == P2PType.CROSS_HOST):
                    node.pipeline_weight += num_micro_batches - 1
            if stage_type == StageType.BLOCK and node.in_slowest_block:
                node.pipeline_weight = block_pipeline_weight

    def get_cost_graph(self, lmesh):
        leaf_strategies = []
        for node in self.nodes:
            if node.is_replicated:
                node.set_strategy(None, lmesh)
            else:
                node.collect_strategies(lmesh)
        for node in self.nodes:
            strategies_vector = node.update_resharding_cost()
            if len(strategies_vector) != 0:
                leaf_strategies.append(strategies_vector)
        cost_graph = CostGraph(leaf_strategies)
        return cost_graph

    def find_solution(self, cost_graph, memory_budget):
        solver = Solver(cost_graph, memory_budget=memory_budget)
        solution = solver.find_solution()[1]

        graph_strategy = solution.node_best_strategy
        for node_name, strategy in graph_strategy.items():
            node = self._nodes[node_name]
            for idx, pre_node in enumerate(node.predecessor_nodes):
                if pre_node is None:
                    continue
                if pre_node.node_name not in strategy.best_resharding_cost:
                    continue
                strategy.best_resharding_cost[
                    idx] = strategy.best_resharding_cost[pre_node.node_name]
                strategy.node_names[idx] = pre_node.node_name
            for key in list(strategy.best_resharding_cost.keys()):
                if isinstance(key, str):
                    del strategy.best_resharding_cost[key]

        return solution

    def visualize(self, name='pp_graph'):
        with open(name + '.dot', 'w') as f:
            f.write("digraph {\n")
            '''
            f.write("    // Value Nodes\n")
            for name, tensor in self._tensors.items():
                f.write("    \"{}\" [fillcolor = \"green\", label = \"{}\", shape = \"box\", style = \"filled\"];\n".format(name, tensor.shape))
            '''
            f.write("    // Operation Nodes\n")
            for name, node in self._nodes.items():
                fillcolor = 'white'
                if 'MATRIX_MULTIPLY' in name:
                    fillcolor = 'green'
                label = name
                if len(node.outputs) > 0:
                    label = name + '\\n' + str(node.outputs[0].shape)
                f.write(
                    "    \"{}\" [fillcolor = \"{}\", label = \"{}\", shape = \"box\", style = \"filled\"];\n"
                    .format(name, fillcolor, label))
            f.write("    // Edges\n")
            for name, node in self._nodes.items():
                for successor_node in node.successor_nodes:
                    if successor_node:
                        f.write("    \"{}\" ->\"{}\";\n".format(
                            name, successor_node.node_name))
            f.write("    }\n")

    def visualize_solution(self,
                           solution,
                           fname='pp_graph_solution',
                           ignore_shape_io=True):
        with open(fname + '.dot', 'w') as f:
            names, costs, block_ids = [], [], []
            f.write("digraph {\n")
            f.write("    // Operation Nodes\n")
            for name, node in self._nodes.items():
                if ignore_shape_io and node.layer is not None and node.layer.is_shape_io:
                    continue
                cost = 0.0
                fillcolor = 'white'
                if 'MATRIX_MULTIPLY' in name or 'PLUGIN_V2_Gemm' in name:
                    fillcolor = 'orange'
                elif '_same_spec' in name:
                    fillcolor = 'gray'
                elif 'p2p_block' in name:
                    fillcolor = 'blue'
                elif 'PLUGIN' in name:
                    fillcolor = 'yellow'

                shape = 'box'
                if 'output_node' == node.node_type or 'input_node' == node.node_type:
                    shape = 'ellipse'
                    fillcolor = 'green'

                label = name + f'_block{node.building_block_id}_weight{node.sharding_weight}'
                if len(node.inputs) > 0:
                    for idx, input in enumerate(node.inputs):
                        if not input:
                            continue
                        label = label + f'\\ninput{idx}_' + str(
                            input.shape) + f'_{input.dtype_str_size[0]}_'
                        if node.node_name in solution.node_best_strategy:
                            best_strategy = solution.node_best_strategy[
                                node.node_name]
                            shard_seq = str(
                                best_strategy.sharding_specs[f'input{idx}'].
                                sharding_sequence)
                            label = label + shard_seq
                            if idx not in best_strategy.best_resharding_cost:
                                continue
                            rcosts = best_strategy.best_resharding_cost[idx][0]
                            comm_action_sequence, resharding_cost = rcosts[
                                1], rcosts[2]
                            if len(comm_action_sequence) > 0:
                                label = label + '|'
                            for commspec in comm_action_sequence:
                                comm = [
                                    commspec.comm_pattern, commspec.gather_dim,
                                    commspec.shard_dim,
                                    commspec.logical_process_axis
                                ]
                                label = label + '->' + str(comm)
                            if resharding_cost > 0:
                                label = label + '_rcost{:.2}'.format(
                                    resharding_cost)
                            cost = cost + resharding_cost
                if len(node.outputs) > 0:
                    best_strategy = None
                    for idx, output in enumerate(node.outputs):
                        label = label + f'\\noutput{idx}_' + str(
                            output.shape) + f'_{output.dtype_str_size[0]}'
                        if node.node_name in solution.node_best_strategy:
                            best_strategy = solution.node_best_strategy[
                                node.node_name]
                            shard_seq = str(
                                best_strategy.sharding_specs[f'output{idx}'].
                                sharding_sequence)
                            comm = None
                            if f'output{idx}' in best_strategy.communication_actions:
                                commspec = best_strategy.communication_actions[
                                    f'output{idx}']
                                comm = [
                                    commspec.comm_pattern, commspec.gather_dim,
                                    commspec.shard_dim,
                                    commspec.logical_process_axis
                                ]
                            label = label + '_' + shard_seq
                            if comm:
                                label = label + f' | {comm}'
                    if best_strategy:
                        cost = cost + best_strategy.sharding_cost + best_strategy.communication_cost
                        label = label + '| scost{:.2}'.format(
                            best_strategy.sharding_cost)
                        if best_strategy.communication_cost > 0:
                            label = label + ' | ccost{:.2}'.format(
                                best_strategy.communication_cost)
                names.append(name)
                costs.append(cost)
                block_ids.append([
                    node.building_block_id, node.cost_level,
                    node.sharding_weight + node.pipeline_weight,
                    node.same_spec_id
                ])
                f.write(
                    "    \"{}\" [fillcolor = \"{}\", label = \"{}\", shape = \"{}\", style = \"filled\"];\n"
                    .format(name, fillcolor, label, shape))
            f.write("    // Edges\n")
            for name, node in self._nodes.items():
                if ignore_shape_io and node.layer is not None and node.layer.is_shape_io:
                    continue
                for successor_node in node.successor_nodes:
                    if successor_node:
                        if ignore_shape_io and successor_node.layer is not None and successor_node.layer.is_shape_io:
                            continue
                        f.write("    \"{}\" ->\"{}\";\n".format(
                            name, successor_node.node_name))
            f.write("    }\n")
            df = pd.DataFrame.from_dict({
                'node':
                names,
                'cost':
                costs,
                'block_id': [block[0] for block in block_ids],
                'cost_level': [block[1] for block in block_ids],
                'sharding_weight': [block[2] for block in block_ids],
                'same_spec_id': [block[3] for block in block_ids]
            })
            df['weight_cost'] = df['sharding_weight'] * df['cost']
            df.to_csv(fname + '.csv')

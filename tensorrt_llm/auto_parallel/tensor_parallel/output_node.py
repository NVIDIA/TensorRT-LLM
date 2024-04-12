from .node import Node
from .sharding_strategy import StrategiesVector


class OuputNode(Node):

    def _update_memory_cost(self, strategies):
        for strategy in strategies:
            if not self.no_memory_footprint:
                strategy.const_memory_footprint = strategy.sharding_specs[
                    'input0'].get_max_sharded_size_per_device()

    def __init__(self, tensor):
        self._layer = None
        self.is_shape_io = False
        self._inputs = []
        self._outputs = []
        self.predecessor_nodes = []
        self.predecessor_nodes_out_index = {}
        self.successor_nodes = []
        self.op_data = {}
        self.global_to_local_op_name = {}
        self.is_replicated = tensor.attrs.get("is_replicated", False)
        self.same_spec_id = tensor.attrs.get("same_spec_id", -1)
        self.no_memory_footprint = tensor.attrs.get("no_memory_footprint",
                                                    False)
        self.building_block_id = -1
        self.cost_level = -1
        self.stage_type = None
        self.in_start_block = None
        self.in_end_block = None
        self.in_slowest_block = None
        input = tensor.copy()
        self._inputs.append(input)
        self.op_data['input0'] = input
        self.global_to_local_op_name[input.name] = 'input0'

        self.sharding_weight = 1.0
        self.resharding_weight = 1.0
        self.pipeline_weight = 0
        self.node_name = tensor.name
        self.node_type = 'output_node'
        self.num_inputs = 0
        self.num_outputs = 1
        self.dtype = tensor.dtype
        self.strategies_vector = []
        self.node_runtime_profiler = None

    def _collect_strategies(self, device_mesh):
        dim_partition_list = []
        dim_size = len(self.op_data['input0'].shape)
        dim_partition_list.append({})
        dim_partition_list.extend(
            self._enumerate_all_possible_1d_sharding([0], dim_size))
        dim_partition_list.extend(
            self._enumerate_all_possible_1d_sharding([1], dim_size))
        dim_partition_list.extend(
            self._enumerate_all_possible_1d_sharding([0, 1], dim_size))
        dim_partition_list.extend(
            self._enumerate_all_possible_2d_sharding([0], [1], dim_size))

        strategies_vector = StrategiesVector(self)
        for dim_partition_dict in dim_partition_list:
            dim_partition_dict_mapping = {'input0': dim_partition_dict}
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            sharding_seq = sharding_spec_mapping['input0'].sharding_sequence
            sharding_strategy = self._get_sharding_strategy(
                name=f'output-op {sharding_seq}',
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)

        return strategies_vector

    def _profile_sharding_cost(self, strategy, device_mesh):
        return 0.0

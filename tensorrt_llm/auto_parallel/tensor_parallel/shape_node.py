import copy

from .node import Node
from .sharding_strategy import StrategiesVector


class Shape(Node):

    def _update_memory_cost(self, strategies):
        pass

    def _collect_strategies(self, device_mesh):
        # one input for softmax node
        predecessor = self.predecessor_nodes[0]
        strategies_vector = StrategiesVector(self)
        for idx, strategy in enumerate(predecessor.strategies_vector):
            # current node's local name input0 -> global name xxx
            global_input_name = self.op_data['input0'].name
            # global name xxx -> pre node local output name
            prenode_local_name = predecessor.global_to_local_op_name[
                global_input_name]
            dim_partition_dict = copy.deepcopy(
                strategy.sharding_specs[prenode_local_name].dim_partition_dict)
            in0_partition_dict = dim_partition_dict
            dim_partition_dict_mapping = {
                "input0": in0_partition_dict,
                "output0": {},
            }
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                return strategies_vector
            name = '{} = <shape> {}'.format(
                sharding_spec_mapping['output0'].sharding_sequence,
                sharding_spec_mapping['input0'].sharding_sequence)
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)
        return strategies_vector

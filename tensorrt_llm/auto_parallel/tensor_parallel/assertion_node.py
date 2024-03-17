import copy

from .node import Node
from .sharding_strategy import StrategiesVector


class Assertion(Node):

    def _collect_strategies(self, device_mesh):
        predecessor = self.predecessor_nodes[0]  # one input for softmax node
        strategies_vector = StrategiesVector(self)
        for idx, strategy in enumerate(predecessor.strategies_vector):
            global_input_name = self.op_data[
                'input0'].name  # current node's local name input0 -> global name xxx
            prenode_local_name = predecessor.global_to_local_op_name[
                global_input_name]  # global name xxx -> pre node local output name
            dim_partition_dict = copy.deepcopy(
                strategy.sharding_specs[prenode_local_name].dim_partition_dict)
            in0_partition_dict = dim_partition_dict
            dim_partition_dict_mapping = {
                "input0": in0_partition_dict,
            }
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                return strategies_vector
            name = '<assertion> {}'.format(
                sharding_spec_mapping['input0'].sharding_sequence)
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)
        return strategies_vector

import copy

from .node import Node
from .sharding_strategy import StrategiesVector


class Cast(Node):

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
            in0_partition_dict = dim_partition_dict
            out_partition_dict = copy.deepcopy(dim_partition_dict)
            dim_partition_dict_mapping = {
                "input0": in0_partition_dict,
                "output0": out_partition_dict,
            }
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            name = '{} = <cast op> {}'.format(
                sharding_spec_mapping['output0'].sharding_sequence,
                sharding_spec_mapping['input0'].sharding_sequence)
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)

        return strategies_vector

    def _update_memory_cost(self, strategies):
        pass

from .node import Node
from .sharding_strategy import StrategiesVector


class Constant(Node):

    def _update_memory_cost(self, strategies):
        super()._update_memory_cost(strategies)
        for strategy in strategies:
            strategy.inout_memory_footprint = 0.0
            strategy.peak_memory_footprint = 0.0
            strategy.const_memory_footprint = strategy.sharding_specs[
                'output0'].get_max_sharded_size_per_device()

    def _collect_strategies(self, device_mesh):
        dim_partition_list = []
        dim_size = len(self.op_data['output0'].shape)
        dim_partition_list.append({})
        dim_partition_list.extend(
            self._enumerate_all_possible_1d_sharding([0, 1], dim_size))
        dim_partition_list.extend(
            self._enumerate_all_possible_2d_sharding([0], [1], dim_size))
        dim_partition_list.extend(
            self._enumerate_all_possible_1d_sharding([0], dim_size))
        dim_partition_list.extend(
            self._enumerate_all_possible_1d_sharding([1], dim_size))

        strategies_vector = StrategiesVector(self)
        for dim_partition_dict in dim_partition_list:
            dim_partition_dict_mapping = {'output0': dim_partition_dict}
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            sharding_seq = sharding_spec_mapping['output0'].sharding_sequence
            sharding_strategy = self._get_sharding_strategy(
                name=f'constant-op {sharding_seq}',
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)

        return strategies_vector

    def _profile_sharding_cost(self, strategy, device_mesh):
        return 0.0

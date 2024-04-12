import tensorrt as trt

from .node import Node
from .sharding_strategy import StrategiesVector


class Fill(Node):

    def __init__(self, layer):
        super().__init__(layer)
        layer.to_subclass()
        self.operation = layer.as_trt().operation
        layer.to_base_class()

    def _collect_strategies(self, device_mesh):
        dim_partition_list = []
        dim_size = len(self.op_data['output0'].shape)
        dim_partition_list.append({})
        if self.num_inputs == 0 and self.operation != trt.FillOperation.LINSPACE:
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
            dim_partition_dict_mapping = {'output0': dim_partition_dict}
            for i in range(self.num_inputs):
                dim_partition_dict_mapping[f'input{i}'] = {}
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            sharding_seq = sharding_spec_mapping['output0'].sharding_sequence
            sharding_strategy = self._get_sharding_strategy(
                name=f'fill-op {sharding_seq}',
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)

        return strategies_vector

    def _profile_sharding_cost(self, strategy, device_mesh):
        updated_layer_attrs = {}
        updated_input_values = {}
        shape = strategy.sharding_specs['output0'].get_sharded_shape_per_device(
        )
        if self.layer.num_inputs >= 1:
            updated_input_values[0] = shape
        else:
            updated_layer_attrs['shape'] = shape
        elapsed_time = self.node_runtime_profiler.runtime_profile(
            self.layer, updated_layer_attrs, updated_input_values, strategy,
            device_mesh)
        return elapsed_time

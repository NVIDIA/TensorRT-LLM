from .node import Node
from .sharding_strategy import StrategiesVector


class Normalization(Node):

    def __init__(self, layer):
        super().__init__(layer)
        layer.to_subclass()
        self.axes = layer.as_trt().axes
        self.weight_bias_dim_base = 0
        layer.to_base_class()

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
            shard_reduction_axes = False
            for dim in range(len(self.get_input(0).shape)):
                if (self.axes & (1 << dim)) and dim in dim_partition_dict:
                    shard_reduction_axes = True
                    break
            if shard_reduction_axes:
                continue
            dim_partition_dict_mapping = {
                "input0": dim_partition_dict,
                "output0": dim_partition_dict,
            }
            if self.num_inputs >= 2:
                dim_partition_dict_mapping['input1'] = {}
            if self.num_inputs >= 3:
                dim_partition_dict_mapping['input2'] = {}
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            name = '{} = {} <normalization op> scale {}, bias {}'.format(
                sharding_spec_mapping['output0'].sharding_sequence,
                sharding_spec_mapping['input0'].sharding_sequence,
                sharding_spec_mapping['input1'].sharding_sequence
                if self.num_inputs >= 2 else 'None',
                sharding_spec_mapping['input2'].sharding_sequence
                if self.num_inputs >= 3 else 'None',
            )
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)
        return strategies_vector

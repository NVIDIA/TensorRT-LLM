import copy

from .node import Node
from .sharding_strategy import StrategiesVector


class Concatenation(Node):

    def __init__(self, layer):
        super().__init__(layer)
        layer.to_subclass()
        batch_dims = [i for i in range(len(self.get_output(0).shape))]
        self.axis = layer.as_trt().axis
        batch_dims.remove(self.axis)
        self._generate_bcast_dims(batch_dims, self.get_output(0).shape)
        layer.to_base_class()

    def _collect_strategies(self, device_mesh):
        dim_partition_list = []
        dim_size = len(self.op_data['output0'].shape)
        dim_partition_list.append({})
        dim_partition_list.extend(
            self._enumerate_all_possible_1d_sharding([0], dim_size))
        dim_partition_list.extend(
            self._enumerate_all_possible_1d_sharding([1], dim_size))
        dim_partition_list.extend(
            self._enumerate_all_possible_1d_sharding([0, 1], dim_size))
        dim_partition_list.extend(
            self._enumerate_all_possible_2d_sharding([0], [1], dim_size))

        dim_partition_dict_mapping = {}
        strategies_vector = StrategiesVector(self)
        for dim_partition_dict in dim_partition_list:
            if self.axis in dim_partition_dict:
                dim_partition_dict.pop(self.axis)
            for idx in range(self.num_inputs):
                in_partition_dict = copy.deepcopy(dim_partition_dict)
                dim_partition_dict_mapping[f'input{idx}'] = in_partition_dict
            out_partition_dict = dim_partition_dict
            dim_partition_dict_mapping['output0'] = out_partition_dict

            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            name = '{} = <concate along dim {}> {}'.format(
                sharding_spec_mapping['output0'].sharding_sequence, self.axis, [
                    sharding_spec_mapping[f'input{idx}'].sharding_sequence
                    for idx in range(self.num_inputs)
                ])
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)
        return strategies_vector

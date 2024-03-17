from .node import Node
from .sharding_strategy import StrategiesVector


class Select(Node):

    def __init__(self, layer):
        super().__init__(layer)
        batch_dims = [i for i in range(len(self.get_output(0).shape))]
        self._generate_bcast_dims(batch_dims, self.get_output(0).shape)

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

        strategies_vector = StrategiesVector(self)
        for dim_partition_dict in dim_partition_list:
            # the three inputs are condition, true tensor and false tensor
            in0_partition_dict = self._recover_bcast_partition_dict(
                dim_partition_dict, self.op_data['input0'])
            in1_partition_dict = self._recover_bcast_partition_dict(
                dim_partition_dict, self.op_data['input1'])
            in2_partition_dict = self._recover_bcast_partition_dict(
                dim_partition_dict, self.op_data['input2'])
            out_partition_dict = dim_partition_dict
            dim_partition_dict_mapping = {
                "input0": in0_partition_dict,
                "input1": in1_partition_dict,
                "input2": in2_partition_dict,
                "output0": out_partition_dict,
            }
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            name = '{} = <select op {}> {} {}'.format(
                sharding_spec_mapping['output0'].sharding_sequence,
                sharding_spec_mapping['input0'].sharding_sequence,
                sharding_spec_mapping['input1'].sharding_sequence,
                sharding_spec_mapping['input2'].sharding_sequence)

            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)
        return strategies_vector

from tensorrt_llm._utils import trt_axes_to_dim

from .node import Node
from .sharding_strategy import StrategiesVector


class Reduce(Node):

    def __init__(self, layer):
        super().__init__(layer)
        layer.to_subclass()
        self.reduce_dims = trt_axes_to_dim(layer.as_trt().axes)
        self.sum_mapping_dict = {}
        num_input_dims = len(self.get_input(0).shape)
        if layer.as_trt().keep_dims:
            for i in range(num_input_dims):
                self.sum_mapping_dict[i] = i
        else:
            output_index = 0
            for i in range(num_input_dims):
                if i not in self.reduce_dims:
                    self.sum_mapping_dict[i] = output_index
                    output_index += 1
            assert output_index == len(self.get_output(0).shape)
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
            recover_dims = []
            out_partition_dict = {}
            for dim in dim_partition_dict.keys():
                if dim in self.reduce_dims:
                    recover_dims.append(dim)
                elif dim in self.sum_mapping_dict:
                    out_partition_dict[
                        self.sum_mapping_dict[dim]] = dim_partition_dict[dim]
                else:
                    assert 0, f'dim {dim} is not in sum_dims or sum_mapping_dict'

            for dim in recover_dims:
                dim_partition_dict.pop(dim)

            in0_parition_dict = dim_partition_dict
            dim_partition_dict_mapping = {
                "input0": in0_parition_dict,
                "output0": out_partition_dict,
            }
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            name = '{} = <reduce along dim {}> {}'.format(
                sharding_spec_mapping['output0'].sharding_sequence,
                self.reduce_dims,
                sharding_spec_mapping['input0'].sharding_sequence)
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)
        return strategies_vector

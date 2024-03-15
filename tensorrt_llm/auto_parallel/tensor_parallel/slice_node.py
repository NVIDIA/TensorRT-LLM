import copy

from .node import Node
from .sharding_strategy import StrategiesVector


class Slice(Node):

    def __init__(self, layer):
        super().__init__(layer)
        layer.to_subclass()
        input_shape = self.get_input(0).shape
        output_shape = self.get_output(0).shape
        assert len(input_shape) == len(
            output_shape
        ), f'dims of input shape {input_shape} != dims of output shape {output_shape}'
        if layer.num_inputs >= 2 and layer.get_input(1) is not None:
            start = layer.get_input(1).value
        else:
            start = layer.as_trt().start
        if layer.num_inputs >= 4 and layer.get_input(3) is not None:
            stride = layer.get_input(3).value
        else:
            stride = layer.as_trt().stride
        self.keep_partition_dims = [(input_shape[i] == output_shape[i]
                                     and start[i] == 0 and stride[i] == 1)
                                    for i in range(len(input_shape))]
        layer.to_base_class()

    def _update_memory_cost(self, strategies):
        for strategy in strategies:
            # for slice node, it input0's read = output0's write
            inout_memory_footprint = strategy.sharding_specs[
                'output0'].get_sharded_size_per_device() * 2
            strategy.inout_memory_footprint = inout_memory_footprint
            strategy.peak_memory_footprint = (
                strategy.sharding_specs['input0'].
                get_max_sharded_size_per_device() + strategy.
                sharding_specs['output0'].get_max_sharded_size_per_device())

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
        # dim_partition_dict can be the same as previous node if solver's time is a problem
        for dim_partition_dict in dim_partition_list:
            for dim in range(len(self.keep_partition_dims)):
                if (not self.keep_partition_dims[dim]
                    ) and dim in dim_partition_dict:
                    dim_partition_dict.pop(dim)

            in0_partition_dict = dim_partition_dict
            out_partition_dict = copy.deepcopy(dim_partition_dict)
            dim_partition_dict_mapping = {
                "input0": in0_partition_dict,
                "output0": out_partition_dict,
            }
            for i in range(1, self.num_inputs):
                if self.predecessor_nodes[i]:
                    dim_partition_dict_mapping[f"input{i}"] = {}
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            name = '{} = {} <slice op> '.format(
                sharding_spec_mapping['output0'].sharding_sequence,
                sharding_spec_mapping['input0'].sharding_sequence)
            for i in range(1, self.num_inputs):
                if self.predecessor_nodes[i]:
                    name = name + str(
                        sharding_spec_mapping[f'input{i}'].sharding_sequence)
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)
        return strategies_vector

    def _profile_sharding_cost(self, strategy, device_mesh):
        updated_layer_attrs = {}
        updated_input_values = {}
        shape = strategy.sharding_specs['output0'].get_sharded_shape_per_device(
        )
        if self.layer.num_inputs >= 3 and self.layer.get_input(2) is not None:
            updated_input_values[2] = shape
        else:
            updated_layer_attrs['shape'] = shape
        elapsed_time = self.node_runtime_profiler.runtime_profile(
            self.layer, updated_layer_attrs, updated_input_values, strategy,
            device_mesh)
        return elapsed_time

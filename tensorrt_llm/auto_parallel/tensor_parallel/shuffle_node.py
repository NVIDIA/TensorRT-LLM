import copy

from .node import Node
from .sharding_strategy import StrategiesVector


class Shuffle(Node):

    def __init__(self, layer):
        super().__init__(layer)
        layer.to_subclass()
        self.first_tanspose_dims = layer.as_trt().first_transpose
        self.second_transpose_dims = layer.as_trt().second_transpose
        self.zero_is_placeholder = layer.as_trt().zero_is_placeholder
        self.is_first_transepose_identity = (sorted(
            self.first_tanspose_dims) == list(self.first_tanspose_dims))
        self.input_shape = self.get_input(0).shape
        self.is_second_transepose_identity = (sorted(
            self.second_transpose_dims) == list(self.second_transpose_dims))

        output_shape = list(self.get_output(0).shape)
        self.reshape_dims = copy.deepcopy(output_shape)
        if not self.is_second_transepose_identity:
            for i in self.second_transpose_dims:
                if self.second_transpose_dims[i] != i:
                    self.reshape_dims[
                        self.second_transpose_dims[i]] = output_shape[i]
        self.is_reshape_identity = (list(self.reshape_dims) == list(
            self.input_shape))
        layer.to_base_class()

    def _collect_transpose_strategies(self, device_mesh, transpose_dims):
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
            in0_partition_dict = dim_partition_dict
            out_partition_dict = {}
            for split_dim, mesh_dim in in0_partition_dict.items():
                trans_dim = transpose_dims[split_dim]
                out_partition_dict[trans_dim] = mesh_dim

            dim_partition_dict_mapping = {
                "input0": in0_partition_dict,
                "output0": out_partition_dict,
            }
            if self.num_inputs == 2:
                dim_partition_dict_mapping["input1"] = {}

            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            name = '{} = <shuffle_transpose_only op> {}'.format(
                sharding_spec_mapping['output0'].sharding_sequence,
                sharding_spec_mapping['input0'].sharding_sequence)
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)
        return strategies_vector

    def _find_reshape_partitions(self, input_shape, output_shape,
                                 input_partition_dict):
        len_input_shape, len_output_shape = len(input_shape), len(output_shape)
        output_partition_dict = {}
        i, j = 0, 0
        while i < len_input_shape or j < len_output_shape:
            if i < len_input_shape and input_shape[i] == 1:
                i = i + 1
                continue
            if j < len_output_shape and output_shape[j] == 1:
                j = j + 1
                continue

            if input_shape[i] == output_shape[j]:
                if i in input_partition_dict:
                    output_partition_dict[j] = input_partition_dict[i]
                # it keep the dimension, so need to keep the partition dims
                i, j = i + 1, j + 1

            elif input_shape[i] < output_shape[j]:
                # we detect if the input dims are merged in the reshape dim
                value = input_shape[i]
                for ii in range(i + 1, len_input_shape):
                    value = value * input_shape[ii]
                    if value == output_shape[j]:
                        # it is merged, we set the output's merged dim partition as all inputs' dims
                        mesh_dim = []
                        for in_dim in range(i, ii + 1):
                            if in_dim in input_partition_dict:
                                mesh_dim = mesh_dim + input_partition_dict[
                                    in_dim]
                        if len(mesh_dim) > 0:
                            output_partition_dict[j] = sorted(mesh_dim)
                        i, j = ii + 1, j + 1
                        break
                else:
                    # we don't find the merged dimensions, the difference may from random reshape, we don't support it now
                    return {}, {}
            else:
                # we detect if the input dim is split into reshape dims
                value = output_shape[j]
                for jj in range(j + 1, len_output_shape):
                    value = value * output_shape[jj]
                    if value == input_shape[i]:
                        # it is split pattern
                        if i in input_partition_dict:
                            output_partition_dict[j] = input_partition_dict[i]
                        i, j = i + 1, jj + 1
                        break
                else:
                    # we don't find the split dimensions, the difference may from random reshape
                    return {}, {}
        return input_partition_dict, output_partition_dict

    def _collect_reshape_strategies(self, device_mesh):
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
            in0_partition_dict = dim_partition_dict
            in0_partition_dict, out_partition_dict = self._find_reshape_partitions(
                self.input_shape, self.reshape_dims, in0_partition_dict)
            dim_partition_dict_mapping = {
                "input0": in0_partition_dict,
                "output0": out_partition_dict,
            }
            if self.num_inputs == 2:
                dim_partition_dict_mapping["input1"] = {}
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            name = '{} = <shuffle_reshape op> {}'.format(
                sharding_spec_mapping['output0'].sharding_sequence,
                sharding_spec_mapping['input0'].sharding_sequence)
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)
        return strategies_vector

    def _collect_identity_strategies(self, device_mesh):
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
            in0_partition_dict = dim_partition_dict
            out_partition_dict = copy.deepcopy(dim_partition_dict)
            dim_partition_dict_mapping = {
                "input0": in0_partition_dict,
                "output0": out_partition_dict,
            }
            if self.num_inputs == 2:
                dim_partition_dict_mapping["input1"] = {}
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            name = '{} = <shuffle_identity op> {}'.format(
                sharding_spec_mapping['output0'].sharding_sequence,
                sharding_spec_mapping['input0'].sharding_sequence)
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)
        return strategies_vector

    def _collect_strategies(self, device_mesh):
        is_identify_list = (self.is_first_transepose_identity,
                            self.is_reshape_identity,
                            self.is_second_transepose_identity)
        if is_identify_list == (True, True, True):
            return self._collect_identity_strategies(device_mesh)
        elif is_identify_list == (True, True, False):
            return self._collect_transpose_strategies(
                device_mesh, self.second_transpose_dims)
        elif is_identify_list == (False, True, True):
            return self._collect_transpose_strategies(device_mesh,
                                                      self.first_transpose_dims)
        elif is_identify_list == (True, False, True):
            return self._collect_reshape_strategies(device_mesh)
        else:
            assert False, f"Unsupported shuffle pattern now {is_identify_list}"

    def _profile_sharding_cost(self, strategy, device_mesh):
        updated_layer_attrs = {}
        updated_input_values = {}
        output_shape = strategy.sharding_specs[
            'output0'].get_sharded_shape_per_device()
        self.layer.to_subclass()
        second_transpose = self.layer.as_trt().second_transpose
        self.layer.to_base_class()
        reshape_dims = [*output_shape]
        for i in range(len(output_shape)):
            reshape_dims[second_transpose[i]] = output_shape[i]
        if self.layer.num_inputs >= 2:
            updated_input_values[1] = reshape_dims
        else:
            updated_layer_attrs['reshape_dims'] = reshape_dims
        elapsed_time = self.node_runtime_profiler.runtime_profile(
            self.layer, updated_layer_attrs, updated_input_values, strategy,
            device_mesh)
        return elapsed_time

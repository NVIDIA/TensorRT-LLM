import copy

import tensorrt as trt

from .comm_spec import CommSpec
from .node import Node
from .sharding_spec import DimSpec
from .sharding_strategy import StrategiesVector


class Gather(Node):

    def __init__(self, layer):
        super().__init__(layer)
        layer.to_subclass()
        self.mode = layer.as_trt().mode
        self.axis = layer.as_trt().axis
        self.num_elementwise_dims = layer.as_trt().num_elementwise_dims
        self.input_id = 0
        self.indice_id = 1
        self.support_vocab_tp = False
        layer.to_base_class()

    def _update_memory_cost(self, strategies):
        for strategy in strategies:
            # for gather node, it input0's read = output0's write
            inout_memory_footprint = (
                strategy.sharding_specs['output0'].get_sharded_size_per_device(
                ) * 2 +
                strategy.sharding_specs['input1'].get_sharded_size_per_device())
            strategy.inout_memory_footprint = inout_memory_footprint
            strategy.peak_memory_footprint = (
                strategy.sharding_specs['output0'].
                get_max_sharded_size_per_device() + strategy.
                sharding_specs['input0'].get_max_sharded_size_per_device() +
                strategy.sharding_specs['input1'].
                get_max_sharded_size_per_device())

    def _collect_strategies(self, device_mesh):
        if self.mode == trt.GatherMode.DEFAULT:
            return self._default_gather_strategies(device_mesh)
        elif self.mode == trt.GatherMode.ELEMENT:
            return self._element_gather_strategies(device_mesh)
        elif self.mode == trt.GatherMode.ND:
            assert 0, 'unsupport gatherND'
        else:
            assert 0, f'unsupport gather mode {self.mode}'

    def _element_gather_strategies(self, device_mesh):
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
            if self.axis in dim_partition_dict:
                dim_partition_dict.pop(self.axis)

            dim_partition_dict_mapping = {
                'input0': dim_partition_dict,
                'input1': copy.deepcopy(dim_partition_dict),
                'output0': copy.deepcopy(dim_partition_dict),
            }

            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            name = '{} = {} <element gather op axis {}> {}'.format(
                sharding_spec_mapping['output0'].sharding_sequence,
                sharding_spec_mapping['input0'].sharding_sequence, self.axis,
                sharding_spec_mapping['input1'].sharding_sequence)
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)

        return strategies_vector

    # for plugin, indice is input0, and weight is input1, which is different from gather node
    def _default_gather_strategies(self, device_mesh):

        def add_sharding_strategy(dim_partition_dict_mapping,
                                  vocab_tp_dim=None):
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if len(sharding_spec_mapping) > 0:
                name = '{} = {} <default gather op axis {}, num_elementwise_dims {}> {}'.format(
                    sharding_spec_mapping['output0'].sharding_sequence,
                    sharding_spec_mapping['input0'].sharding_sequence,
                    self.axis, self.num_elementwise_dims,
                    sharding_spec_mapping['input1'].sharding_sequence)
                communication_action_mapping = {}
                if vocab_tp_dim is not None:
                    name += f'_allreduce{DimSpec(vocab_tp_dim)}'
                    output0_comm_action = CommSpec(
                        comm_pattern='all_reduce',
                        sharding_spec=sharding_spec_mapping['output0'],
                        logical_process_axis=[vocab_tp_dim],
                    )
                    communication_action_mapping[
                        'output0'] = output0_comm_action
                sharding_strategy = self._get_sharding_strategy(
                    name=name,
                    sharding_spec_mapping=sharding_spec_mapping,
                    communication_action_mapping=communication_action_mapping)
                strategies_vector.append(sharding_strategy)

        input_id, indice_id = self.input_id, self.indice_id
        strategies_vector = StrategiesVector(self)
        input_size = len(self.op_data[f'input{input_id}'].shape)
        indice_size = len(self.op_data[f'input{indice_id}'].shape)
        output_dim = input_size + indice_size - 1 - self.num_elementwise_dims
        for strategy in self.predecessor_nodes[input_id].strategies_vector:
            # current node's local name input0 -> global name xxx
            global_input_name = self.op_data[f'input{input_id}'].name
            # global name xxx -> pre node local output name
            prenode_local_name = self.predecessor_nodes[
                input_id].global_to_local_op_name[global_input_name]
            input_dim_partition_dict = copy.deepcopy(
                strategy.sharding_specs[prenode_local_name].dim_partition_dict)

            vocab_tp_dim = input_dim_partition_dict.pop(self.axis, None)

            input_mesh_dims = []
            for dim, mesh_dims in input_dim_partition_dict.items():
                input_mesh_dims += mesh_dims
            input_mesh_dims = set(input_mesh_dims)

            for idx_strategy in self.predecessor_nodes[
                    indice_id].strategies_vector:
                # current node's local name input0 -> global name xxx
                global_indice_name = self.op_data[f'input{indice_id}'].name
                # global name xxx -> pre node local output name
                prenode_local_name = self.predecessor_nodes[
                    indice_id].global_to_local_op_name[global_indice_name]
                indice_dim_partition_dict = copy.deepcopy(
                    idx_strategy.sharding_specs[prenode_local_name].
                    dim_partition_dict)

                for dim, indice_mesh_dims in idx_strategy.sharding_specs[
                        prenode_local_name].dim_partition_dict.items():
                    for indice_mesh_dim in indice_mesh_dims:
                        if indice_mesh_dim in input_mesh_dims:
                            indice_dim_partition_dict.pop(dim)
                            break

                out_partition_dict = {}

                for dim in range(output_dim):
                    if dim < self.axis:
                        if dim in input_dim_partition_dict:
                            out_partition_dict[dim] = \
                                input_dim_partition_dict[dim]
                    elif dim >= self.axis and dim < self.axis + indice_size - self.num_elementwise_dims:
                        indice_dim = dim - self.axis + self.num_elementwise_dims
                        if indice_dim in indice_dim_partition_dict:
                            out_partition_dict[dim] = \
                                indice_dim_partition_dict[indice_dim]
                    else:
                        input_dim = dim - (indice_size -
                                           self.num_elementwise_dims) + 1
                        if input_dim in input_dim_partition_dict:
                            out_partition_dict[dim] = \
                                input_dim_partition_dict[input_dim]

                dim_partition_dict_mapping = {
                    f"input{input_id}": input_dim_partition_dict,
                    f"input{indice_id}": indice_dim_partition_dict,
                    "output0": out_partition_dict,
                }
                add_sharding_strategy(dim_partition_dict_mapping)

                if self.support_vocab_tp and vocab_tp_dim is not None:
                    vocab_tp_dim_partition_dict = {
                        **input_dim_partition_dict,
                        self.axis: vocab_tp_dim,
                    }
                    dim_partition_dict_mapping = {
                        f"input{input_id}": vocab_tp_dim_partition_dict,
                        f"input{indice_id}": indice_dim_partition_dict,
                        "output0": out_partition_dict,
                    }
                    add_sharding_strategy(dim_partition_dict_mapping,
                                          vocab_tp_dim)

        return strategies_vector

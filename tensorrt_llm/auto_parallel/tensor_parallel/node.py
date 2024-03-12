from abc import ABC

from ..config import CostModel
from ..device_mesh import LogicalDeviceMesh
from .comm_spec import CommSpec
from .sharding_spec import ShardingSpec
from .sharding_strategy import ShardingStrategy, StrategiesVector


class Node(ABC):

    def __init__(self, layer):
        self._layer = layer
        self.is_shape_io = self._layer.is_shape_io
        self._inputs = []
        self._outputs = []
        self.predecessor_nodes = []
        self.predecessor_nodes_out_index = {}
        self.successor_nodes = []
        self.op_data = {}
        self.global_to_local_op_name = {}
        self.num_inputs = 0
        self.is_replicated = layer.attrs.get("is_replicated", False)
        self.same_spec_id = layer.attrs.get("same_spec_id", -1)
        self.is_fake = self.same_spec_id != -1
        self.building_block_id = layer.attrs.get("building_block_id", -1)
        self.cost_level = -1
        self.stage_type = layer.attrs.get("stage_type", None)
        self.in_start_block = layer.attrs.get("in_start_block", False)
        self.in_end_block = layer.attrs.get("in_end_block", False)
        self.in_slowest_block = layer.attrs.get("in_slowest_block", False)
        for i, input in enumerate(layer.inputs):
            if input is None:
                self._inputs.append(None)
                self.op_data[f'input{i}'] = None
                continue
            input = input.copy()
            input.attrs["broadcast_dims"] = []
            self._inputs.append(input)
            self.op_data[f'input{i}'] = input
            self.global_to_local_op_name[input.name] = f'input{i}'

        for i, output in enumerate(layer.outputs):
            output = output.copy()
            output.attrs["broadcast_dims"] = []
            self._outputs.append(output)
            self.op_data[f'output{i}'] = output
            self.global_to_local_op_name[output.name] = f'output{i}'

        self.sharding_weight = 1.0
        self.resharding_weight = 1.0
        self.pipeline_weight = 0
        self.node_name = layer.name
        self.node_type = 'normal_node'
        self.num_inputs = layer.num_inputs
        self.num_outputs = layer.num_outputs
        self.dtype = layer.as_trt().precision
        self.strategies_vector = []
        self.node_runtime_profiler = None

    def post_init(self, graph):
        for input in self.inputs:
            if input is None:
                self.predecessor_nodes.append(None)
                continue
            if input.producer is None:
                predecessor_node = graph.get_node(input.name)
                self.predecessor_nodes.append(predecessor_node)
                self.predecessor_nodes_out_index[predecessor_node] = 0
                predecessor_node.successor_nodes.append(self)
            else:
                predecessor_node = graph.get_node(input.producer.name)
                self.predecessor_nodes.append(predecessor_node)
                self.predecessor_nodes_out_index[
                    predecessor_node] = input.output_index
                predecessor_node.successor_nodes.append(self)

    @property
    def layer(self):
        return self._layer

    def get_input(self, index):
        return self._inputs[index]

    @property
    def inputs(self):
        return self._inputs

    def get_output(self, index):
        return self._outputs[index]

    @property
    def outputs(self):
        return self._outputs

    def collect_strategies(self, device_mesh):
        strategies_vector = self._collect_strategies(device_mesh)
        strategies_vector = self._post_process(strategies_vector)
        self._update_sharding_cost(strategies_vector, device_mesh)
        self.strategies_vector = strategies_vector
        return self.strategies_vector

    def _set_strategy(self, strategy, device_mesh):
        strategies_vector = StrategiesVector(self)
        if strategy is None:
            dim_partition_dict_mapping = {}
            for i in range(self.num_inputs):
                dim_partition_dict_mapping[f'input{i}'] = {}
            for i in range(self.num_outputs):
                dim_partition_dict_mapping[f'output{i}'] = {}

            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            assert 0 != len(
                sharding_spec_mapping
            ), f'failed to set default(all Replicate) strategy for node {self.node_name}'
            name = 'RRs'
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)

        else:
            sharding_specs_map = strategy.sharding_specs
            comm_specs_map = strategy.communication_actions
            dim_partition_dict_mapping = {}
            for op_name, sharding_spec in sharding_specs_map.items():
                dim_partition_dict_mapping[
                    op_name] = sharding_spec.dim_partition_dict
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            assert 0 != len(
                sharding_spec_mapping
            ), f'failed to set strategy for node {self.node_name}'
            comm_specs_mapping = {}
            if len(comm_specs_map) > 0:
                for op_name, comm_spec in comm_specs_map.items():
                    comm_specs_mapping[op_name] = CommSpec(
                        comm_pattern=comm_spec.comm_pattern,
                        sharding_spec=sharding_spec_mapping[op_name],
                        logical_process_axis=comm_spec.logical_process_axis,
                    )
            strategies_vector.append(
                self._get_sharding_strategy(
                    name=strategy.name,
                    sharding_spec_mapping=sharding_spec_mapping,
                    communication_action_mapping=comm_specs_mapping))
        return strategies_vector

    def set_strategy(self, strategy, device_mesh):
        strategies_vector = self._set_strategy(strategy, device_mesh)
        strategies_vector = self._post_process(strategies_vector)
        self._update_sharding_cost(strategies_vector, device_mesh)
        self.strategies_vector = strategies_vector
        return self.strategies_vector

    def update_resharding_cost(self):
        self._update_resharding_cost(self.strategies_vector)
        return self.strategies_vector

    def _to_sharding_spec_mapping(self, dim_partition_dict_mapping,
                                  device_mesh):
        results = {}
        for op_data_name, dim_partition_dict in dim_partition_dict_mapping.items(
        ):
            if op_data_name in self.op_data:
                op_data = self.op_data[op_data_name]

            def _to_sharding_spec(op_data, dim_partition_dict):
                sharding_spec = ShardingSpec(
                    device_mesh,
                    op_data.dtype_str_size, [*op_data.shape],
                    [*op_data.max_shape], [*op_data.raw_shape],
                    dim_partition_dict=dim_partition_dict)
                if sharding_spec.sanity_check():
                    return sharding_spec
                else:
                    return None

            sharding_spec = _to_sharding_spec(op_data, dim_partition_dict)
            if sharding_spec:
                results[op_data_name] = sharding_spec
            else:
                return {}
        return results

    def _get_sharding_strategy(self, name, sharding_spec_mapping,
                               communication_action_mapping):
        return ShardingStrategy(
            name=name,
            sharding_specs=sharding_spec_mapping,
            communication_actions=communication_action_mapping,
        )

    def _remove_duplicated_strategy(self, strategies_vector):
        name_checklist = []
        remove_list = []
        for strategy in strategies_vector:
            if strategy.name not in name_checklist:
                name_checklist.append(strategy.name)
            else:
                remove_list.append(strategy)
        for strategy in remove_list:
            strategies_vector.remove(strategy)

    def _post_process(self, strategies_vector):
        # TODO:[KDuan] deal with transpose and dimension 1 problem in ClossalAI, which have been processed before
        for i in range(len(strategies_vector) - 1, -1, -1):
            if strategies_vector[i] is None:
                strategies_vector.pop(i)

        self._remove_duplicated_strategy(strategies_vector)
        return strategies_vector

    def _profile_sharding_cost(self, strategy, device_mesh: LogicalDeviceMesh):
        elapsed_time = self.node_runtime_profiler.runtime_profile(
            self.layer, {}, {}, strategy, device_mesh)
        return elapsed_time

    def _model_sharding_cost_from_s_curve(self, strategy,
                                          device_mesh: LogicalDeviceMesh):
        '''
        [ToDo][KDuan] preprofile the s_curve
        '''
        sharding_cost = 0.0
        return sharding_cost

    # this method might be overwritten by some Ops
    def _get_math_time(self, strategy, device_mesh: LogicalDeviceMesh):
        return 0.0

    # this method might be overwritten by some Ops
    def _get_memory_time(self, strategy, device_mesh: LogicalDeviceMesh):
        memory_time = (strategy.inout_memory_footprint /
                       device_mesh.cluster_info.memory_bw * 1e-3 *
                       device_mesh.cluster_info.memory_efficiency)
        return memory_time

    def _model_sharding_cost_from_alpha_beta(self, strategy,
                                             device_mesh: LogicalDeviceMesh):
        math_time = self._get_math_time(strategy, device_mesh)
        mem_time = self._get_memory_time(strategy, device_mesh)
        return max(math_time, mem_time)

    def _get_communication_cost(self, strategy):
        total_comm_cost = 0.0
        for op_data_name, comm_spec in strategy.communication_actions.items():
            comm_cost = comm_spec.get_comm_cost()
            total_comm_cost = total_comm_cost + comm_cost
        return total_comm_cost

    def _update_sharding_cost(self, strategies, device_mesh: LogicalDeviceMesh):
        self._update_memory_cost(strategies)

        if device_mesh.config.sharding_cost_model == CostModel.ALPHA_BETA:
            for strategy in strategies:
                strategy.sharding_cost = self._model_sharding_cost_from_alpha_beta(
                    strategy, device_mesh)
        elif device_mesh.config.sharding_cost_model == CostModel.S_CURVE:
            for strategy in strategies:
                strategy.sharding_cost = self._model_sharding_cost_from_s_curve(
                    strategy, device_mesh)
        elif device_mesh.config.sharding_cost_model == CostModel.PROFILE:
            for strategy in strategies:
                strategy.alpha_beta_cost = self._model_sharding_cost_from_alpha_beta(
                    strategy, device_mesh)
                if self.is_shape_io:
                    strategy.sharding_cost = strategy.alpha_beta_cost
                else:
                    strategy.sharding_cost = self._profile_sharding_cost(
                        strategy, device_mesh)
        elif device_mesh.config.sharding_cost_model == CostModel.ZERO:
            for strategy in strategies:
                strategy.sharding_cost = 0.0
        else:
            assert False, 'unsupport sharding cost model option: {}'.format(
                device_mesh.config.sharding_cost_model)

        for strategy in strategies:
            strategy.communication_cost = self._get_communication_cost(strategy)

    def _compute_resharding_cost(self, pre_sharding_sepc, cur_sharding_spec,
                                 op_data):
        transform_path, comm_action_sequence, resharding_cost = cur_sharding_spec.device_mesh.shape_consistency_manager.shape_consistency(
            pre_sharding_sepc, cur_sharding_spec)
        return (transform_path, comm_action_sequence, resharding_cost)

    def _update_resharding_cost(self, strategies):
        for strategy in strategies:
            resharding_costs = {}
            for pre_node, out_index in self.predecessor_nodes_out_index.items():
                if pre_node is None:
                    continue
                pre_node_out_data_name = pre_node.get_output(out_index).name
                pre_node_out_data_lname = pre_node.global_to_local_op_name[
                    pre_node_out_data_name]
                if pre_node_out_data_name not in self.global_to_local_op_name:
                    print(f"pre_node_out_data_name = {pre_node_out_data_name}")
                    continue
                cur_node_inp_data_lname = self.global_to_local_op_name[
                    pre_node_out_data_name]
                cur_sharding_spec = strategy.sharding_specs[
                    cur_node_inp_data_lname]

                pre_node_out_sharding_specs = []
                for pre_strategy in pre_node.strategies_vector:
                    pre_node_out_sharding_specs.append(
                        pre_strategy.sharding_specs[pre_node_out_data_lname])

                if pre_node not in resharding_costs:
                    resharding_costs[pre_node.node_name] = []
                for prev_sharding_spec in pre_node_out_sharding_specs:
                    resharding_cost = self._compute_resharding_cost(
                        prev_sharding_spec, cur_sharding_spec,
                        self.op_data[cur_node_inp_data_lname])
                    resharding_costs[pre_node.node_name].append(resharding_cost)
            strategy.resharding_costs = resharding_costs

    def _enumerate_all_possible_1d_sharding(self, mesh_dim, dim_size):
        dim_partition_list = []
        for i in range(dim_size):
            dim_partition_list.append({i: mesh_dim})
        return dim_partition_list

    def _enumerate_all_possible_2d_sharding(self, mesh_dim0, mesh_dim1,
                                            dim_size):
        dim_partition_list = []
        for i in range(dim_size):
            for j in range(dim_size):
                if i != j:
                    dim_partition_list.append({i: mesh_dim0, j: mesh_dim1})
        return dim_partition_list

    def _update_memory_cost(self, strategies):
        for strategy in strategies:
            inout_memory_footprint, max_inout_memory_footprint = 0.0, 0.0
            for spec in strategy.sharding_specs.values():
                inout_memory_footprint += spec.get_sharded_size_per_device()
                max_inout_memory_footprint += spec.get_max_sharded_size_per_device(
                )

            # the communication happens
            comm_buffer_footprint, max_comm_buffer_footprint = 0.0, 0.0
            for comm_spec in strategy.communication_actions.values():
                comm_buffer_footprint += comm_spec.get_mem_cost()
                max_comm_buffer_footprint += comm_spec.get_max_mem_cost()

            # when doing the output0 comm action, the input buffer should be released, the buffer is used to estimate the memory time
            # rather than memory usage
            strategy.inout_memory_footprint = inout_memory_footprint

            strategy.comm_buff_memory_footprint = comm_buffer_footprint
            strategy.peak_memory_footprint = max(max_inout_memory_footprint,
                                                 max_comm_buffer_footprint)

            # The const memory (weight) is recorded in constant layers and should be accumulated
            strategy.const_memory_footprint = 0.0

    def _generate_bcast_dims(self, batch_dims, out_data_shape):
        for output in self.outputs:
            if output.broadcast_across_batch:
                for bs in batch_dims:
                    if output.shape[
                            bs] == 1 and output.shape[bs] != out_data_shape[bs]:
                        output.attrs["broadcast_dims"].append(bs)

    def _recover_bcast_partition_dict(self, partition_dict, op_data):
        ret = {}
        for data_dim, mesh_dim in partition_dict.items():
            if data_dim not in op_data.attrs[
                    "broadcast_dims"] and data_dim + len(
                        op_data.shape) not in op_data.attrs[
                            "broadcast_dims"] and op_data.shape[data_dim] != 1:
                ret[data_dim] = mesh_dim
        return ret

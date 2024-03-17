import copy
import operator
from functools import reduce

import tensorrt as trt

from ..device_mesh import LogicalDeviceMesh
from ..utils import get_builder_flags
from .comm_spec import CommSpec
from .node import Node
from .sharding_spec import DimSpec
from .sharding_strategy import StrategiesVector


class MatrixMultiply(Node):

    def __init__(self, layer):
        super().__init__(layer)
        layer.to_subclass()
        batch_dims = [i for i in range(len(self.get_output(0).shape))][:-2]
        self._generate_bcast_dims(batch_dims, self.get_output(0).shape)
        self.op0_transpose = layer.as_trt().op0 == trt.MatrixOperation.TRANSPOSE
        self.op1_transpose = layer.as_trt().op1 == trt.MatrixOperation.TRANSPOSE
        self.num_out_dims = len(self.get_output(0).shape)
        dtypes_str = [
            self.get_input(0).dtype_str,
            self.get_input(1).dtype_str,
            self.get_output(0).dtype_str
        ]
        dtypes_size = [
            self.get_input(0).dtype_size,
            self.get_input(1).dtype_size,
            self.get_output(0).dtype_size
        ]
        min_idx = dtypes_size.index(min(dtypes_size))
        self.dtype = dtypes_str[min_idx]
        layer.to_base_class()

    def _split_lhs_space_rhs_space(self, mesh_dim_0, mesh_dim_1, device_mesh):
        in0_split_dim = -1 if self.op0_transpose else -2
        in1_split_dim = -2 if self.op1_transpose else -1
        name = (f'{DimSpec(mesh_dim_0)}{DimSpec(mesh_dim_1)} = '
                f'{DimSpec(mesh_dim_0)}R x R{DimSpec(mesh_dim_1)}')
        dim_partition_dict_mapping = {
            "input0": {
                in0_split_dim: mesh_dim_0
            },
            "input1": {
                in1_split_dim: mesh_dim_1
            },
            "output0": {
                -2: mesh_dim_0,
                -1: mesh_dim_1
            },
        }
        sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(sharding_spec_mapping) == 0:
            return None
        strategy = self._get_sharding_strategy(name = name, \
            sharding_spec_mapping = sharding_spec_mapping, \
            communication_action_mapping = {})
        return strategy

    def _split_lhs_space_both_contract(self, mesh_dim_0, mesh_dim_1,
                                       device_mesh):
        # handle the case SR = SS x SR
        name = (
            f'{DimSpec(mesh_dim_0)}R = '
            f'{DimSpec(mesh_dim_0)}{DimSpec(mesh_dim_1)} x {DimSpec(mesh_dim_1)}R'
            f'_allreduce{DimSpec(mesh_dim_1)}')
        in0_split_dim = [-1, -2] if self.op0_transpose else [-2, -1]
        in1_split_dim = -1 if self.op1_transpose else -2
        # get sharding spec mapping
        dim_partition_dict_mapping = {
            "input0": {
                in0_split_dim[0]: mesh_dim_0,
                in0_split_dim[1]: mesh_dim_1
            },
            "input1": {
                in1_split_dim: mesh_dim_1
            },
            "output0": {
                -2: mesh_dim_0
            },
        }

        sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(sharding_spec_mapping) == 0:
            return None
        # get communication action mapping
        communication_action_mapping = {}
        output0_comm_action = CommSpec(
            comm_pattern='all_reduce',
            sharding_spec=sharding_spec_mapping['output0'],
            logical_process_axis=[mesh_dim_1],
        )
        communication_action_mapping['output0'] = output0_comm_action
        return self._get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping)

    def _split_both_contract_rs(self, name, rs_dim, rs_mesh_dim, src_spec,
                                dim_partition_dict_mapping, device_mesh):
        output0_comm_action = CommSpec(
            comm_pattern='reduce_scatter',
            sharding_spec=src_spec,
            shard_dim=[rs_dim],
            logical_process_axis=[rs_mesh_dim],
        )
        rs_out_partition_dict_mapping = copy.deepcopy(
            dim_partition_dict_mapping)
        rs_out_partition_dict_mapping["output0"][rs_dim] = rs_mesh_dim
        rs_out_sharding_spec_mapping = self._to_sharding_spec_mapping(
            rs_out_partition_dict_mapping, device_mesh)
        if len(rs_out_sharding_spec_mapping) == 0:
            return None

        communication_action_mapping = {}
        communication_action_mapping['output0'] = output0_comm_action
        return self._get_sharding_strategy(
            name=name,
            sharding_spec_mapping=rs_out_sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping)

    def _split_lhs_space_both_contract_rs(self, mesh_dim_0, mesh_dim_1,
                                          device_mesh):
        # handle the case SS = SS x SR -> reduce_scatter
        in0_split_dim = [-1, -2] if self.op0_transpose else [-2, -1]
        in1_split_dim = -1 if self.op1_transpose else -2
        # get sharding spec mapping
        dim_partition_dict_mapping = {
            "input0": {
                in0_split_dim[0]: mesh_dim_0,
                in0_split_dim[1]: mesh_dim_1
            },
            "input1": {
                in1_split_dim: mesh_dim_1
            },
            "output0": {
                -2: mesh_dim_0,
            },
        }
        mm_out_sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(mm_out_sharding_spec_mapping) == 0:
            return []
        strategies = []
        for rs_dim in range(self.num_out_dims):
            if rs_dim != self.num_out_dims - 2:
                name_in0, name_in1, name_out0 = ['R'] * self.num_out_dims, [
                    'R'
                ] * self.num_out_dims, ['R'] * self.num_out_dims
                name_in0[-2], name_in0[-1] = str(DimSpec(mesh_dim_0)), str(
                    DimSpec(mesh_dim_1))
                name_in1[-2] = str(DimSpec(mesh_dim_1))
                name_out0[-2], name_out0[rs_dim] = str(
                    DimSpec(mesh_dim_0)), str(DimSpec(mesh_dim_1))
                name_in0, name_in1, name_out0 = ', '.join(name_in0), ', '.join(
                    name_in1), ', '.join(name_out0)
                name = (f'[{name_out0}] = [{name_in0}] x [{name_in1}]'
                        f'_reducescatter{(rs_dim, DimSpec(mesh_dim_1))}')
                ret = self._split_both_contract_rs(
                    name, rs_dim, mesh_dim_1,
                    mm_out_sharding_spec_mapping['output0'],
                    dim_partition_dict_mapping, device_mesh)
                if ret:
                    strategies.append(ret)
        return strategies

    def _split_rhs_space_both_contract(self, mesh_dim_0, mesh_dim_1,
                                       device_mesh):
        name = (
            f'R{DimSpec(mesh_dim_1)} = '
            f'R{DimSpec(mesh_dim_0)} x {DimSpec(mesh_dim_0)}{DimSpec(mesh_dim_1)}'
            f'_allreduce{DimSpec(mesh_dim_0)}')
        in0_split_dim = -2 if self.op0_transpose else -1
        in1_split_dim = [-1, -2] if self.op1_transpose else [-2, -1]
        # get sharding specs
        dim_partition_dict_mapping = {
            "input0": {
                in0_split_dim: mesh_dim_0
            },
            "input1": {
                in1_split_dim[0]: mesh_dim_0,
                in1_split_dim[1]: mesh_dim_1
            },
            "output0": {
                -1: mesh_dim_1
            },
        }
        sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(sharding_spec_mapping) == 0:
            return None
        # get communication actions
        communication_action_mapping = {}
        output0_comm_action = CommSpec(
            comm_pattern='all_reduce',
            sharding_spec=sharding_spec_mapping['output0'],
            logical_process_axis=[mesh_dim_0],
        )
        communication_action_mapping['output0'] = output0_comm_action
        return self._get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping)

    def _split_rhs_space_both_contract_rs(self, mesh_dim_0, mesh_dim_1,
                                          device_mesh):
        in0_split_dim = -2 if self.op0_transpose else -1
        in1_split_dim = [-1, -2] if self.op1_transpose else [-2, -1]
        # get sharding specs
        dim_partition_dict_mapping = {
            "input0": {
                in0_split_dim: mesh_dim_0
            },
            "input1": {
                in1_split_dim[0]: mesh_dim_0,
                in1_split_dim[1]: mesh_dim_1
            },
            "output0": {
                -1: mesh_dim_1
            },
        }
        mm_out_sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(mm_out_sharding_spec_mapping) == 0:
            return []
        strategies = []
        for rs_dim in range(self.num_out_dims):
            if rs_dim != self.num_out_dims - 1:
                name_in0, name_in1, name_out0 = ['R'] * self.num_out_dims, [
                    'R'
                ] * self.num_out_dims, ['R'] * self.num_out_dims
                name_in1[-2], name_in1[-1] = str(DimSpec(mesh_dim_0)), str(
                    DimSpec(mesh_dim_1))
                name_in0[-1] = str(DimSpec(mesh_dim_0))
                name_out0[-1], name_out0[rs_dim] = str(
                    DimSpec(mesh_dim_1)), str(DimSpec(mesh_dim_0))
                name_in0, name_in1, name_out0 = ', '.join(name_in0), ', '.join(
                    name_in1), ', '.join(name_out0)
                name = (f'[{name_out0}] = [{name_in0}] x [{name_in1}]'
                        f'_reducescatter{(rs_dim, DimSpec(mesh_dim_0))}')
                ret = self._split_both_contract_rs(
                    name, rs_dim, mesh_dim_0,
                    mm_out_sharding_spec_mapping['output0'],
                    dim_partition_dict_mapping, device_mesh)
                if ret:
                    strategies.append(ret)
        return strategies

    def _recompute_split_both_contract(self, mesh_dim, device_mesh):
        name = (f'RR = R{DimSpec(mesh_dim)} x {DimSpec(mesh_dim)}R'
                f'_allreduce{DimSpec(mesh_dim)}')
        in0_split_dim = -2 if self.op0_transpose else -1
        in1_split_dim = -1 if self.op1_transpose else -2
        dim_partition_dict_mapping = {
            "input0": {
                in0_split_dim: mesh_dim
            },
            "input1": {
                in1_split_dim: mesh_dim
            },
            "output0": {},
        }
        sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(sharding_spec_mapping) == 0:
            return None

        # get communication action
        communication_action_mapping = {}
        output0_comm_action = CommSpec(
            comm_pattern='all_reduce',
            sharding_spec=sharding_spec_mapping['output0'],
            logical_process_axis=[mesh_dim],
        )
        communication_action_mapping['output0'] = output0_comm_action
        return self._get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping)

    def _recompute_split_both_contract_rs(self, mesh_dim, device_mesh):
        name = (f'{DimSpec(mesh_dim)}R = '
                f'R{DimSpec(mesh_dim)} x {DimSpec(mesh_dim)}R'
                f'_reducescatter0_{DimSpec(mesh_dim)}')
        in0_split_dim = -2 if self.op0_transpose else -1
        in1_split_dim = -1 if self.op1_transpose else -2
        dim_partition_dict_mapping = {
            "input0": {
                in0_split_dim: mesh_dim
            },
            "input1": {
                in1_split_dim: mesh_dim
            },
            "output0": {},
        }
        mm_out_sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(mm_out_sharding_spec_mapping) == 0:
            return []

        strategies = []
        for rs_dim in range(self.num_out_dims):
            name_in0, name_in1, name_out0 = ['R'] * self.num_out_dims, [
                'R'
            ] * self.num_out_dims, ['R'] * self.num_out_dims
            name_in0[-1], name_in1[-2], name_out0[rs_dim] = str(
                DimSpec(mesh_dim)), str(DimSpec(mesh_dim)), str(
                    DimSpec(mesh_dim))
            name_in0, name_in1, name_out0 = ', '.join(name_in0), ', '.join(
                name_in1), ', '.join(name_out0)
            name = f'[{name_out0}] = [{name_in0}] x [{name_in1}]_reducescatter{(rs_dim, DimSpec(mesh_dim))}'
            ret = self._split_both_contract_rs(
                name, rs_dim, mesh_dim, mm_out_sharding_spec_mapping['output0'],
                dim_partition_dict_mapping, device_mesh)
            if ret:
                strategies.append(ret)
        return strategies

    def _split_rhs_space_only(self, mesh_dim, device_mesh):
        name = f'R{DimSpec(mesh_dim)} = RR x R{DimSpec(mesh_dim)}'
        in1_split_dim = -2 if self.op1_transpose else -1
        # get sharding spec
        dim_partition_dict_mapping = {
            "input0": {},
            "input1": {
                in1_split_dim: mesh_dim
            },
            "output0": {
                -1: mesh_dim
            },
        }
        # We don't have to do anything special for bias here, because
        # the bias is already the same sharding spec as the output0.
        sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(sharding_spec_mapping) == 0:
            return None
        return self._get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping={})

    def _split_lhs_space_only(self, mesh_dim, device_mesh):
        name = f'{DimSpec(mesh_dim)}R = {DimSpec(mesh_dim)}R x RR'
        in0_split_dim = -1 if self.op0_transpose else -2
        # get sharding spec
        dim_partition_dict_mapping = {
            "input0": {
                in0_split_dim: mesh_dim
            },
            "input1": {},
            "output0": {
                -2: mesh_dim
            },
        }
        sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(sharding_spec_mapping) == 0:
            return None
        return self._get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping={})

    def _non_split(self, device_mesh):
        name = 'RR = RR x RR'
        # get sharding spec
        dim_partition_dict_mapping = {
            "input0": {},
            "input1": {},
            "output0": {},
        }
        sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(sharding_spec_mapping) == 0:
            return None
        return self._get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping={})

    def _split_one_batch_dim(self, batch_dim, mesh_dim, device_mesh):
        name = (
            f'{DimSpec(mesh_dim)}b{batch_dim}RR = '
            f'{DimSpec(mesh_dim)}b{batch_dim}RR x {DimSpec(mesh_dim)}b{batch_dim}RR'
        )
        in0_data = self.op_data['input0']
        in1_data = self.op_data['input1']

        batch_partition_dict = {batch_dim: mesh_dim}
        in0_parition_dict = self._recover_bcast_partition_dict(
            batch_partition_dict, in0_data)
        in1_parition_dict = self._recover_bcast_partition_dict(
            batch_partition_dict, in1_data)
        out_partition_dict = {batch_dim: mesh_dim}
        # TODO:[KDuan] Double check if MatrixMultiplication's output has bcast in dim
        dim_partition_dict_mapping = {
            "input0": in0_parition_dict,
            "input1": in1_parition_dict,
            "output0": out_partition_dict,
        }
        sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(sharding_spec_mapping) == 0:
            return None
        return self._get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping={})

    def _split_two_batch_dims(self, batch_dim0, batch_dim1, mesh_dim0,
                              mesh_dim1, device_mesh):
        name = (
            f'{DimSpec(mesh_dim0)}b{batch_dim0}{DimSpec(mesh_dim1)}b{batch_dim1}RR = '
            f'{DimSpec(mesh_dim0)}b{batch_dim0}RR x {DimSpec(mesh_dim1)}b{batch_dim1}RR'
        )
        in0_data = self.op_data['input0']
        in1_data = self.op_data['input1']

        in0_parition_dict = {}
        if batch_dim0 not in in0_data.attrs["broadcast_dims"]:
            in0_parition_dict[batch_dim0] = mesh_dim0
        if batch_dim1 not in in0_data.attrs["broadcast_dims"]:
            in0_parition_dict[batch_dim1] = mesh_dim1

        in1_parition_dict = {}
        if batch_dim0 not in in1_data.attrs["broadcast_dims"]:
            in1_parition_dict[batch_dim0] = mesh_dim0
        if batch_dim1 not in in1_data.attrs["broadcast_dims"]:
            in1_parition_dict[batch_dim1] = mesh_dim1

        batch_partition_dict = {batch_dim0: mesh_dim0, batch_dim1: mesh_dim1}
        in0_parition_dict = self._recover_bcast_partition_dict(
            batch_partition_dict, in0_data)
        in1_parition_dict = self._recover_bcast_partition_dict(
            batch_partition_dict, in1_data)
        out_partition_dict = {batch_dim0: mesh_dim0, batch_dim1: mesh_dim1}
        dim_partition_dict_mapping = {
            "input0": in0_parition_dict,
            "input1": in1_parition_dict,
            "output0": out_partition_dict,
        }
        sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(sharding_spec_mapping) == 0:
            return None
        return self._get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping={})

    def _split_batch_dim_lhs_space(self, batch_dim, mesh_dim0, mesh_dim1,
                                   device_mesh):

        name = (
            f'{DimSpec(mesh_dim0)}b{batch_dim}{DimSpec(mesh_dim1)}R = '
            f'{DimSpec(mesh_dim0)}b{batch_dim}{DimSpec(mesh_dim1)}R x {DimSpec(mesh_dim0)}b{batch_dim}RR'
        )
        in0_data = self.op_data['input0']
        in1_data = self.op_data['input1']
        in0_parition_dict = {batch_dim: mesh_dim0}
        in1_parition_dict = {batch_dim: mesh_dim0}
        in0_lhs_split_dim = -1 if self.op0_transpose else -2
        in0_parition_dict[in0_lhs_split_dim] = mesh_dim1

        in0_parition_dict = self._recover_bcast_partition_dict(
            in0_parition_dict, in0_data)
        in1_parition_dict = self._recover_bcast_partition_dict(
            in1_parition_dict, in1_data)
        out_partition_dict = {batch_dim: mesh_dim0, -2: mesh_dim1}

        dim_partition_dict_mapping = {
            "input0": in0_parition_dict,
            "input1": in1_parition_dict,
            "output0": out_partition_dict,
        }
        sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(sharding_spec_mapping) == 0:
            return None
        return self._get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping={})

    def _split_batch_dim_rhs_space(self, batch_dim, mesh_dim0, mesh_dim1,
                                   device_mesh):

        name = (
            f'{DimSpec(mesh_dim0)}b{batch_dim}R{DimSpec(mesh_dim1)} = '
            f'{DimSpec(mesh_dim0)}b{batch_dim}RR x {DimSpec(mesh_dim0)}b{batch_dim}R{DimSpec(mesh_dim1)}'
        )
        in0_data = self.op_data['input0']
        in1_data = self.op_data['input1']
        in0_parition_dict = {batch_dim: mesh_dim0}
        in1_parition_dict = {batch_dim: mesh_dim0}

        in1_rhs_split_dim = -2 if self.op1_transpose else -1
        in1_parition_dict[in1_rhs_split_dim] = mesh_dim1

        in0_parition_dict = self._recover_bcast_partition_dict(
            in0_parition_dict, in0_data)
        in1_parition_dict = self._recover_bcast_partition_dict(
            in1_parition_dict, in1_data)
        out_partition_dict = {batch_dim: mesh_dim0, -1: mesh_dim1}
        dim_partition_dict_mapping = {
            "input0": in0_parition_dict,
            "input1": in1_parition_dict,
            "output0": out_partition_dict,
        }
        sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(sharding_spec_mapping) == 0:
            return None
        return self._get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping={})

    def _split_batch_dim_both_contract(self, batch_dim, mesh_dim0, mesh_dim1,
                                       device_mesh):

        name = (
            f'{DimSpec(mesh_dim0)}b{batch_dim}RR = '
            f'{DimSpec(mesh_dim0)}b{batch_dim}R{DimSpec(mesh_dim1)} x '
            f'{DimSpec(mesh_dim0)}b{batch_dim}{DimSpec(mesh_dim1)}R_AR{mesh_dim1}'
        )
        in0_data = self.op_data['input0']
        in1_data = self.op_data['input1']
        in0_parition_dict = {batch_dim: mesh_dim0}
        in1_parition_dict = {batch_dim: mesh_dim0}

        in0_contract_dim = -2 if self.op0_transpose else -1
        in1_contract_dim = -1 if self.op1_transpose else -2
        in0_parition_dict[in0_contract_dim] = mesh_dim1
        in1_parition_dict[in1_contract_dim] = mesh_dim1

        in0_parition_dict = self._recover_bcast_partition_dict(
            in0_parition_dict, in0_data)
        in1_parition_dict = self._recover_bcast_partition_dict(
            in1_parition_dict, in1_data)
        out_partition_dict = {batch_dim: mesh_dim0}
        dim_partition_dict_mapping = {
            "input0": in0_parition_dict,
            "input1": in1_parition_dict,
            "output0": out_partition_dict,
        }
        sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(sharding_spec_mapping) == 0:
            return None

        # get communication actions
        communication_action_mapping = {}
        output0_comm_action = CommSpec(
            comm_pattern='all_reduce',
            sharding_spec=sharding_spec_mapping['output0'],
            logical_process_axis=[mesh_dim1],
        )
        communication_action_mapping['output0'] = output0_comm_action
        return self._get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping)

    def _split_batch_dim_both_contract_rs(self, batch_dim, mesh_dim0, mesh_dim1,
                                          device_mesh):

        name = (
            f'{DimSpec(mesh_dim0)}b{batch_dim}RR = '
            f'{DimSpec(mesh_dim0)}b{batch_dim}R{DimSpec(mesh_dim1)} x '
            f'{DimSpec(mesh_dim0)}b{batch_dim}{DimSpec(mesh_dim1)}R_AR{mesh_dim1}'
        )
        in0_data = self.op_data['input0']
        in1_data = self.op_data['input1']
        in0_parition_dict = {batch_dim: mesh_dim0}
        in1_parition_dict = {batch_dim: mesh_dim0}

        in0_contract_dim = -2 if self.op0_transpose else -1
        in1_contract_dim = -1 if self.op1_transpose else -2
        in0_parition_dict[in0_contract_dim] = mesh_dim1
        in1_parition_dict[in1_contract_dim] = mesh_dim1

        in0_parition_dict = self._recover_bcast_partition_dict(
            in0_parition_dict, in0_data)
        in1_parition_dict = self._recover_bcast_partition_dict(
            in1_parition_dict, in1_data)
        out_partition_dict = {batch_dim: mesh_dim0}
        dim_partition_dict_mapping = {
            "input0": in0_parition_dict,
            "input1": in1_parition_dict,
            "output0": out_partition_dict,
        }
        mm_out_sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if len(mm_out_sharding_spec_mapping) == 0:
            return []

        strategies = []
        for rs_dim in range(self.num_out_dims):
            if rs_dim != batch_dim:
                name_in0, name_in1, name_out0 = ['R'] * self.num_out_dims, [
                    'R'
                ] * self.num_out_dims, ['R'] * self.num_out_dims
                name_in0[batch_dim], name_in0[-1] = str(
                    DimSpec(mesh_dim0)), str(DimSpec(mesh_dim1))
                name_in1[batch_dim], name_in1[-2] = str(
                    DimSpec(mesh_dim0)), str(DimSpec(mesh_dim1))
                name_in1[batch_dim], name_out0[rs_dim] = str(
                    DimSpec(mesh_dim0)), str(DimSpec(mesh_dim1))
                name_in0, name_in1, name_out0 = ', '.join(name_in0), ', '.join(
                    name_in1), ', '.join(name_out0)
                name = f'[{name_out0}] = [{name_in0}] x [{name_in1}]_reducescatter{(rs_dim, DimSpec(mesh_dim1))}'
                ret = self._split_both_contract_rs(
                    name, rs_dim, mesh_dim1,
                    mm_out_sharding_spec_mapping['output0'],
                    dim_partition_dict_mapping, device_mesh)
                if ret:
                    strategies.append(ret)
        return strategies

    def _dp_strategies(self, device_mesh):
        strategies = []
        # S0R  = S0R x RR
        strategies.append(self._split_lhs_space_only([0], device_mesh))
        # S1R  = S1R x RR
        strategies.append(self._split_lhs_space_only([1], device_mesh))
        # S01R = S01R x RR
        strategies.append(self._split_lhs_space_only([0, 1], device_mesh))
        return strategies

    def _tp_strategies(self, device_mesh: LogicalDeviceMesh):
        strategies = []
        # RR = RS x SR _ AR
        strategies.append(self._recompute_split_both_contract([0], device_mesh))
        strategies.append(self._recompute_split_both_contract([1], device_mesh))
        strategies.append(
            self._recompute_split_both_contract([0, 1], device_mesh))

        if device_mesh.config.enable_reduce_scatter:
            #  RS x SR _ reduce scatter
            strategies.extend(
                self._recompute_split_both_contract_rs([0], device_mesh))
            strategies.extend(
                self._recompute_split_both_contract_rs([1], device_mesh))
            strategies.extend(
                self._recompute_split_both_contract_rs([0, 1], device_mesh))

        # RS = RR x RS
        strategies.append(self._split_rhs_space_only([0], device_mesh))
        strategies.append(self._split_rhs_space_only([1], device_mesh))
        strategies.append(self._split_rhs_space_only([0, 1], device_mesh))

        # RS = RS x SS _ AR
        strategies.append(
            self._split_rhs_space_both_contract([0], [1], device_mesh))
        strategies.append(
            self._split_rhs_space_both_contract([1], [0], device_mesh))

        if device_mesh.config.enable_reduce_scatter:
            # RS x SS _ reduce scatter
            strategies.extend(
                self._split_rhs_space_both_contract_rs([0], [1], device_mesh))
            strategies.extend(
                self._split_rhs_space_both_contract_rs([1], [0], device_mesh))

        return strategies

    def _mix_strategies(self, device_mesh):
        strategies = []

        # SR = SS x SR_AR
        strategies.append(
            self._split_lhs_space_both_contract([0], [1], device_mesh))
        strategies.append(
            self._split_lhs_space_both_contract([1], [0], device_mesh))
        if device_mesh.config.enable_reduce_scatter:
            # RS x SS _ reduce scatter
            strategies.extend(
                self._split_lhs_space_both_contract_rs([0], [1], device_mesh))
            strategies.extend(
                self._split_lhs_space_both_contract_rs([1], [0], device_mesh))
        # SS = SR x RS
        strategies.append(self._split_lhs_space_rhs_space([0], [1],
                                                          device_mesh))
        strategies.append(self._split_lhs_space_rhs_space([0], [1],
                                                          device_mesh))

        # RR = RR x RR
        strategies.append(self._non_split(device_mesh))
        return strategies

    def _bmm_strategies(self, device_mesh: LogicalDeviceMesh):
        strategies = []
        bmm_dim = len(self.op_data['output0'].shape)
        if bmm_dim >= 3:
            for batch_dim in range(0, bmm_dim - 2):
                strategies.append(
                    self._split_one_batch_dim(batch_dim, [0], device_mesh))
                strategies.append(
                    self._split_one_batch_dim(batch_dim, [1], device_mesh))
                strategies.append(
                    self._split_one_batch_dim(batch_dim, [0, 1], device_mesh))

                strategies.append(
                    self._split_batch_dim_lhs_space(batch_dim, [0], [1],
                                                    device_mesh))
                strategies.append(
                    self._split_batch_dim_lhs_space(batch_dim, [1], [0],
                                                    device_mesh))

                strategies.append(
                    self._split_batch_dim_rhs_space(batch_dim, [0], [1],
                                                    device_mesh))
                strategies.append(
                    self._split_batch_dim_rhs_space(batch_dim, [1], [0],
                                                    device_mesh))

                strategies.append(
                    self._split_batch_dim_both_contract(batch_dim, [0], [1],
                                                        device_mesh))
                strategies.append(
                    self._split_batch_dim_both_contract(batch_dim, [1], [0],
                                                        device_mesh))
                if device_mesh.config.enable_reduce_scatter:
                    strategies.extend(
                        self._split_batch_dim_both_contract_rs(
                            batch_dim, [0], [1], device_mesh))
                    strategies.extend(
                        self._split_batch_dim_both_contract_rs(
                            batch_dim, [1], [0], device_mesh))
            if bmm_dim >= 4:
                for batch_dim0 in range(0, bmm_dim - 2):
                    for batch_dim1 in range(0, bmm_dim - 2):
                        if batch_dim0 != batch_dim1:
                            strategies.append(
                                self._split_two_batch_dims(
                                    batch_dim0, batch_dim1, [0], [1],
                                    device_mesh))

        return strategies

    def _collect_strategies(self, device_mesh):
        strategies_vector = StrategiesVector(self)
        dp_strategies = self._dp_strategies(device_mesh)
        tp_strategies = self._tp_strategies(device_mesh)
        mix_strategies = self._mix_strategies(device_mesh)
        bmm_strategies = self._bmm_strategies(device_mesh)
        strategies_vector.extend(dp_strategies)
        strategies_vector.extend(tp_strategies)
        strategies_vector.extend(mix_strategies)
        strategies_vector.extend(bmm_strategies)
        return strategies_vector

    def is_fp16(self):
        builder_flags = get_builder_flags()
        return builder_flags & (1 << int(trt.BuilderFlag.FP16)) != 0

    def _get_math_time(self, strategy, device_mesh):
        shape_in0 = strategy.sharding_specs[
            'input0'].get_sharded_shape_per_device()
        shape_out = strategy.sharding_specs[
            'output0'].get_sharded_shape_per_device()
        m, n = shape_out[-2], shape_out[-1]
        batches = shape_out[:-2]
        k = shape_in0[-2] if self.op0_transpose else shape_in0[-1]
        macs_shape = batches + [m, n, k]
        macs = reduce(operator.mul, macs_shape, 1) * 2
        config = device_mesh.config
        cluster_info = device_mesh.cluster_info
        dtype = self.dtype
        # For fp16 matmul ops that use_fp32_acc=True.
        # They are mistaken for fp32 ops since all of their IO tensors use fp32 dtype.
        if self.is_fp16() and self.dtype == "float32":
            dtype = "float16"
        math_throughput_tflops = getattr(cluster_info.math_throughput, dtype)
        assert math_throughput_tflops != 0, \
            "Undefined {} math throughput of cluster {}".format(dtype, config.cluster_key)
        math_time = macs / math_throughput_tflops * 1e-6 * cluster_info.math_efficiency
        return math_time

    def _update_memory_cost(self, strategies):
        super()._update_memory_cost(strategies)
        # For fp16 matmul ops that use_fp32_acc=True.
        # Their memory footprints are calculated based on fp32 IO tensors.
        # Actually they will use fp16 IO tensors after fused.
        # So we divide all the memory footprints by 2.
        if self.is_fp16() and self.dtype == "float32":
            for strategy in strategies:
                strategy.inout_memory_footprint /= 2
                strategy.peak_memory_footprint /= 2
                strategy.comm_buff_memory_footprint /= 2

import math
import operator
from functools import reduce
from typing import List, Tuple

import pandas as pd

from .comm_spec import CommSpec
from .sharding_spec import ShardingSpec


class ShapeConsistencyManager(object):

    def __init__(self):
        self.forward_only = True
        self.cached_spec_pairs_transform_path = {}
        self.cache_hit = 0
        self.cache_miss = 0

    def all_gather_simulator(self, target_pair):
        _, shard_list = target_pair
        new_shard_list = []
        return new_shard_list

    def all_to_all_simulator(self, f_target_pair, b_target_pair):
        '''
        Simulating all-to-all operation, analyze the communication cost
        and simulate the influence of the DimSpec.

        We BANNED all representations which shard_list in decreasing order,
        such as S10, so all-to-all(S0, S1) -> RS01 is NOT allowed.
        Therefore, if the behind shard_list is not None, we just extend it to the front shard_list.
        Argument:
            target_pair(Tuple[int, List[int]]): The first element is the dimension of tensor to be sharded,
            and the second element describes which logical axis will be sharded in that dimension.
        e.g.:
            all-to-all(S0, S1) -> [S01, R]
            all-to-all(S0, R) -> [R, S0]
        Otherwise, we extend the front shard_list to behind.
        e.g.:
            all-to-all(R, S1) -> [S1, R]

        Argument:
            target_pair(Tuple[int, List[int]]): The first element is the dimension of tensor to be sharded,
            and the second element describes which logical axis will be sharded in that dimension.
        '''
        _, f_shard_list = f_target_pair
        _, b_shard_list = b_target_pair
        if not len(b_shard_list):
            b_shard_list.extend(f_shard_list)
            f_shard_list = []
        else:
            f_shard_list.extend(b_shard_list)
            b_shard_list = []

        return f_shard_list, b_shard_list

    def shard_simulator(self, target_pair, legal_sharding_dims):
        '''
        Simulating shard operation, analyze the communication cost(always ZERO)
        and simulate the influence of the DimSpec.

        We don't allow uncontiguous layout, such as shard(S0)->S02 is NOT allowed.
        In addition, We BANNED all representations which shard_list in decreasing order,
        such as S10, so shard(S0) -> S10 is NOT allowed.
        Therefore, for the R dimension, we could just append any legal sharding dim on it.
        e.g.:
            shard(R) -> S0
        For the S dimension, we need to make sure the shard_list after sharding still keep rising order.
        e.g:
            shard(S0) -> S01

        Argument:
            target_pair(Tuple[int, List[int]]): The first element is the dimension of tensor to be sharded,
            and the second element describes which logical axis will be sharded in that dimension.
        '''
        _, shard_list = target_pair
        shard_list_list, logical_process_axis = [], []
        for dim in legal_sharding_dims:
            if len(shard_list) != 0 and dim <= shard_list[-1]:
                continue
            new_shard_list = shard_list + [dim]
            shard_list_list.append(new_shard_list)
            logical_process_axis.append([dim])

        # we support sorted 2D mesh here
        if len(legal_sharding_dims) == 2 and len(shard_list) == 0:
            shard_list_list.append(legal_sharding_dims)
            logical_process_axis.append(legal_sharding_dims)
        return shard_list_list, logical_process_axis

    def mix_gather_simulator(self, f_target_pair, b_target_pair):
        '''
        Assume index of f and b target pairs are 'f' and 'b'
        S0S1 => Input: (f, [0]), (b, [1]) Output: [f, b], [[0], [1]]
        S1S0 => Input: (f, [1]), (b, [0]) Output: [f, b], [[1], [0]]
        S01R => Input: (f, [0, 1]), (b, []) Output: [f], [[0, 1]]
        RS01 => Input: (f, []), (b, [0, 1]) Output: [b], [[0, 1]]
        '''
        if f_target_pair[1] and b_target_pair[1]:
            return [f_target_pair[0],
                    b_target_pair[0]], [f_target_pair[1], b_target_pair[1]]
        if f_target_pair[1]:
            return [f_target_pair[0]], [f_target_pair[1]]
        if b_target_pair[1]:
            return [b_target_pair[0]], [b_target_pair[1]]

    def get_all_all_gather_spec(self, source_spec, orig_cost):
        '''
        Get all valid sharding specs from source_spec with single all-gather operation, and
        accumulate communication cost on origin cost which will finally be used in auto sharding solver.
        For the all-gather operation, we just care about the S dimension.

        Argument:
            source_spec(ShardingSpec): the ShardingSpec of the source_spec.
            orig_cost(Dict[str, float]): the original communication cost before this operation.

        Return:
            valid_spec_dict(Dict[ShardingSpec, float]): all valid sharding specs from source_spec with single all-gather operation.

        Example:
            dim_partition_dict = {0: [0], 1: [1]}
            # DistSpec:
            #     shard_sequence: S0,S1,R
            #     device_mesh_shape: (4, 4)
            sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)
            shape_consistency_manager = ShapeConsistencyManager()
            rst_dict = shape_consistency_manager.get_all_all_gather_spec(sharding_spec, {'forward': 0, 'backward': 0, 'total': 0})
            print(rst_dict)

        Output:
            {DistSpec:
            shard_sequence: R,S1,R
            device_mesh_shape: (4, 4): 0, DistSpec:
            shard_sequence: S0,R,R
            device_mesh_shape: (4, 4): 0}
        '''
        valid_spec_dict = {}
        comm_pattern = 'all_gather'
        for target_pair in source_spec.dim_partition_dict.items():
            shard_list = self.all_gather_simulator(target_pair)
            index = target_pair[0]
            new_dim_partition_dict = source_spec.dim_partition_dict.copy()

            # We won't add empty list into dim_partition_dict
            # The key will be popped if the related shard_list is empty
            if shard_list:
                new_dim_partition_dict[index] = shard_list
            else:
                new_dim_partition_dict.pop(index)

            # generate the CommSpec to record the action of source_sharding_spec->new_sharding_spec
            gather_dim = index
            logical_process_axis = target_pair[1]
            comm_spec = CommSpec(comm_pattern,
                                 sharding_spec=source_spec,
                                 gather_dim=[gather_dim],
                                 logical_process_axis=[logical_process_axis],
                                 forward_only=self.forward_only)

            # compute the communication cost with CommSpec

            # generate new sharding spec
            new_sharding_spec = ShardingSpec(
                source_spec.device_mesh,
                source_spec.data_type_size,
                source_spec.entire_shape,
                source_spec.max_entire_shape,
                source_spec.raw_shape,
                dim_partition_dict=new_dim_partition_dict)

            if not new_sharding_spec.sanity_check():
                continue
            cost = comm_spec.get_comm_cost()
            valid_spec_dict[new_sharding_spec] = (comm_spec, orig_cost + cost)
        return valid_spec_dict

    def get_all_all_to_all_spec(self, source_spec, orig_cost):
        '''
        Get all valid sharding specs from source_spec with single all-to-all operation, and
        accumulate communication cost on origin cost which will finally be used in auto sharding solver.
        For the all-to-all operation, we just care about the pairs containing S dimension.

        Argument:
            source_spec(ShardingSpec): the ShardingSpec of the source_spec.
            orig_cost(Dict[str, float]): the original communication cost before this operation.

        Return:
            valid_spec_dict(Dict[ShardingSpec, float]): all valid sharding specs from source_spec with single all-to-all operation.

        Example:
            dim_partition_dict = {0: [0], 1: [1]}
            # DistSpec:
            #     shard_sequence: S0,S1,R
            #     device_mesh_shape: (4, 4)
            sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)
            shape_consistency_manager = ShapeConsistencyManager()
            rst_dict = shape_consistency_manager.get_all_all_to_all_spec(sharding_spec, {'forward': 0, 'backward': 0, 'total': 0})
            print(rst_dict)

        Output:
            {DistSpec:
            shard_sequence: S01,R,R
            device_mesh_shape: (4, 4): 0, DistSpec:
            shard_sequence: R,S1,S0
            device_mesh_shape: (4, 4): 0, DistSpec:
            shard_sequence: S0,R,S1
            device_mesh_shape: (4, 4): 0}
        '''
        valid_spec_dict = {}
        comm_pattern = 'all_to_all'
        tensor_dims = len(source_spec.entire_shape)
        for f_index in range(tensor_dims - 1):
            for b_index in range(f_index + 1, tensor_dims):
                # skip (R, R) cases
                if f_index not in source_spec.dim_partition_dict and b_index not in source_spec.dim_partition_dict:
                    continue
                else:
                    if f_index in source_spec.dim_partition_dict:
                        '''
                        # skip (S01, R) -> (R, S01) is NOT allowed
                        if len(source_spec.dim_partition_dict[f_index]) >= 2:
                            continue
                        '''
                        f_target_pair = (f_index, [
                            *source_spec.dim_partition_dict[f_index]
                        ])
                    else:
                        f_target_pair = (f_index, [])
                    if b_index in source_spec.dim_partition_dict:
                        '''
                        # skip (R, S01) -> (S01, R) is NOT allowed
                        if len(source_spec.dim_partition_dict[b_index]) >= 2:
                            continue
                        '''
                        b_target_pair = (b_index, [
                            *source_spec.dim_partition_dict[b_index]
                        ])
                    else:
                        b_target_pair = (b_index, [])

                # skip (S1, S0) -> S10
                if f_target_pair[1] and b_target_pair[
                        1] and f_target_pair[1][0] >= b_target_pair[1][0]:
                    continue
                f_shard_list, b_shard_list = self.all_to_all_simulator(
                    f_target_pair, b_target_pair)
                f_index = f_target_pair[0]
                b_index = b_target_pair[0]

                # generate the CommSpec to record the action of source_sharding_spec->new_sharding_spec
                if len(f_shard_list) < len(f_target_pair[1]):
                    gather_dim = f_index
                    shard_dim = b_index
                    logical_process_axis = f_target_pair[1]
                else:
                    gather_dim = b_index
                    shard_dim = f_index
                    logical_process_axis = b_target_pair[1]
                comm_spec = CommSpec(
                    comm_pattern,
                    sharding_spec=source_spec,
                    gather_dim=[gather_dim],
                    shard_dim=[shard_dim],
                    logical_process_axis=[logical_process_axis],
                    forward_only=self.forward_only)

                # compute the communication cost with CommSpec

                new_dim_partition_dict = source_spec.dim_partition_dict.copy()

                # We won't add empty list into dim_partition_dict
                # The key will be popped if the related shard_list is empty
                if f_shard_list:
                    new_dim_partition_dict[f_index] = f_shard_list
                else:
                    new_dim_partition_dict.pop(f_index)
                if b_shard_list:
                    new_dim_partition_dict[b_index] = b_shard_list
                else:
                    new_dim_partition_dict.pop(b_index)

                # generate new sharding spec

                new_sharding_spec = ShardingSpec(
                    source_spec.device_mesh,
                    source_spec.data_type_size,
                    source_spec.entire_shape,
                    source_spec.max_entire_shape,
                    source_spec.raw_shape,
                    dim_partition_dict=new_dim_partition_dict)
                if not new_sharding_spec.sanity_check():
                    continue
                cost = comm_spec.get_comm_cost()
                valid_spec_dict[new_sharding_spec] = (comm_spec,
                                                      cost + orig_cost)

        return valid_spec_dict

    def get_all_shard_spec(self, source_spec, orig_cost):
        '''
        Get all valid sharding specs from source_spec with single shard operation, and
        accumulate commucation cost on origin cost which will finally be used in auto sharding solver.
        For the sharding operation, we just care about legal sharding dimensions.

        Argument:
            source_spec(ShardingSpec): the ShardingSpec of the source_spec.
            orig_cost(float): the original communication cost before this operation.

        Return:
            valid_spec_dict(Dict[ShardingSpec, float]): all valid sharding specs from source_spec with single all-to-all operation.

        Example:
            dim_partition_dict = {0: [0]}
            # DistSpec:
            #     shard_sequence: S0,R,R
            #     device_mesh_shape: (4, 4)
            sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)
            shape_consistency_manager = ShapeConsistencyManager()
            rst_dict = shape_consistency_manager.get_all_shard_spec(sharding_spec, {'forward': 0, 'backward': 0, 'total': 0})
            print(rst_dict)

        Output:
            {DistSpec:
            shard_sequence: S01,R,R
            device_mesh_shape: (4, 4): 0, DistSpec:
            shard_sequence: S0,S1,R
            device_mesh_shape: (4, 4): 0, DistSpec:
            shard_sequence: S0,R,S1
            device_mesh_shape: (4, 4): 0}
        '''
        valid_spec_dict = {}
        comm_pattern = 'split'

        # legal sharding dims means the mesh_id is still available to use.
        legal_sharding_dims = [
            i for i in range(len(source_spec.device_mesh.mesh_shape))
        ]
        for dim, shard_list in source_spec.dim_partition_dict.items():
            for element in shard_list:
                legal_sharding_dims.remove(element)
        if len(legal_sharding_dims) == 0:
            return valid_spec_dict

        tensor_dims = len(source_spec.entire_shape)

        for index in range(tensor_dims):
            if index not in source_spec.dim_partition_dict:
                shard_list_list, logical_process_axes = self.shard_simulator(
                    (index, []), legal_sharding_dims)
            else:
                shard_list_list, logical_process_axes = self.shard_simulator(
                    (index, source_spec.dim_partition_dict[index]),
                    legal_sharding_dims)
            if not shard_list_list:
                continue
            for shard_list, logical_process_axis in zip(shard_list_list,
                                                        logical_process_axes):
                new_dim_partition_dict = source_spec.dim_partition_dict.copy()
                new_dim_partition_dict[index] = shard_list

                # generate the CommSpec to record the action of source_sharding_spec->new_sharding_spec
                comm_spec = CommSpec(
                    comm_pattern,
                    sharding_spec=source_spec,
                    shard_dim=[index],
                    logical_process_axis=[logical_process_axis],
                    forward_only=self.forward_only)

                # generate new sharding spec
                new_sharding_spec = ShardingSpec(
                    source_spec.device_mesh,
                    source_spec.data_type_size,
                    source_spec.entire_shape,
                    source_spec.max_entire_shape,
                    source_spec.raw_shape,
                    dim_partition_dict=new_dim_partition_dict)
                if not new_sharding_spec.sanity_check():
                    continue
                # compute the communication cost with CommSpec
                cost = comm_spec.get_comm_cost()
                valid_spec_dict[new_sharding_spec] = (comm_spec,
                                                      cost + orig_cost)

        return valid_spec_dict

    def get_all_mixed_shard_spec(self, source_spec, orig_cost):
        '''
        Get all valid sharding specs from source_spec with single shard operation, and
        accumulate commucation cost on origin cost which will finally be used in auto sharding solver.
        For the sharding operation, we just care about legal sharding dimensions.
        '''
        valid_spec_dict = {}
        comm_pattern = 'split'

        # legal sharding dims means the mesh_id is still available to use.
        legal_sharding_dims = [
            i for i in range(len(source_spec.device_mesh.mesh_shape))
        ]
        for dim, shard_list in source_spec.dim_partition_dict.items():
            for element in shard_list:
                legal_sharding_dims.remove(element)
        if len(legal_sharding_dims) != 2:
            return valid_spec_dict

        tensor_dims = len(source_spec.entire_shape)
        for f_index in range(tensor_dims):
            for b_index in range(tensor_dims):
                if f_index != b_index:
                    shard_dims = [f_index, b_index]
                    logical_process_axes = [[legal_sharding_dims[0]],
                                            [legal_sharding_dims[1]]]
                    new_dim_partition_dict = source_spec.dim_partition_dict.copy(
                    )
                    new_dim_partition_dict[f_index] = [legal_sharding_dims[0]]
                    new_dim_partition_dict[b_index] = [legal_sharding_dims[1]]
                    comm_spec = CommSpec(
                        comm_pattern,
                        sharding_spec=source_spec,
                        shard_dim=shard_dims,
                        logical_process_axis=logical_process_axes,
                        forward_only=self.forward_only)

                    # generate new sharding spec
                    new_sharding_spec = ShardingSpec(
                        source_spec.device_mesh,
                        source_spec.data_type_size,
                        source_spec.entire_shape,
                        source_spec.max_entire_shape,
                        source_spec.raw_shape,
                        dim_partition_dict=new_dim_partition_dict)
                    if not new_sharding_spec.sanity_check():
                        continue
                    cost = comm_spec.get_comm_cost()
                    valid_spec_dict[new_sharding_spec] = (comm_spec,
                                                          cost + orig_cost)
        return valid_spec_dict

    def get_all_mix_gather_spec(self, source_spec, orig_cost):
        '''
        S0S1 -> RR
        S1S0 -> RR
        S01R -> RR
        RS01 -> RR
        '''
        valid_spec_dict = {}
        comm_pathern = 'all_gather'
        tensor_dims = len(source_spec.entire_shape)
        for f_index in range(tensor_dims - 1):
            for b_index in range(f_index + 1, tensor_dims):
                if (f_index not in source_spec.dim_partition_dict) and (
                        b_index not in source_spec.dim_partition_dict):
                    continue
                else:
                    if f_index in source_spec.dim_partition_dict:
                        # skip (S10, R) -> (R, R)
                        '''
                        if len(
                                f_target_pair[1]
                        ) == 2 and f_target_pair[1][0] >= f_target_pair[1][1]:
                            continue
                        '''
                        f_target_pair = (f_index, [
                            *source_spec.dim_partition_dict[f_index]
                        ])
                    else:
                        f_target_pair = (f_index, [])
                    if b_index in source_spec.dim_partition_dict:
                        # skip (R, S10) -> (R, R)
                        '''
                        if len(
                                b_target_pair[1]
                        ) == 2 and b_target_pair[1][0] >= b_target_pair[1][1]:
                            continue
                        '''
                        b_target_pair = (b_index, [
                            *source_spec.dim_partition_dict[b_index]
                        ])
                    else:
                        b_target_pair = (b_index, [])
                if len(f_target_pair[1]) + len(b_target_pair[1]) != 2:
                    continue
                gather_dim, logical_process_axes = self.mix_gather_simulator(
                    f_target_pair, b_target_pair)
                comm_spec = CommSpec(comm_pathern,
                                     sharding_spec=source_spec,
                                     gather_dim=gather_dim,
                                     logical_process_axis=logical_process_axes,
                                     forward_only=self.forward_only,
                                     mix_gather=True)

                new_dim_partition_dict = {}
                # generate new sharding spec
                new_sharding_spec = ShardingSpec(
                    source_spec.device_mesh,
                    source_spec.data_type_size,
                    source_spec.entire_shape,
                    source_spec.max_entire_shape,
                    source_spec.raw_shape,
                    dim_partition_dict=new_dim_partition_dict)
                if not new_sharding_spec.sanity_check():
                    continue
                cost = comm_spec.get_comm_cost()
                valid_spec_dict[new_sharding_spec] = (comm_spec,
                                                      cost + orig_cost)

        return valid_spec_dict

    def get_all_one_step_transform_spec(self, source_spec, orig_cost):
        '''
        Get all valid sharding specs from source_spec with one step transform, and
        accumulate commucation cost on origin cost which will finally be used in auto sharding solver.
        Note:
            all-gather will eliminate a sharding dimension, all-to-all will keep sharding dimension same as before,
            and shard will add a sharding dimension. Therefore, the result of above operations are mutual exclusive,
            we could safely put them together.

        Argument:
            source_spec(ShardingSpec): the ShardingSpec of the source_spec.
            orig_cost(float): the original communication cost before this operation.

        Return:
            valid_spec_dict(Dict[ShardingSpec, float]): all valid sharding specs from source_spec with single all-to-all operation.
        '''
        valid_spec_dict = {}
        valid_spec_dict.update(
            self.get_all_all_gather_spec(source_spec, orig_cost))
        valid_spec_dict.update(
            self.get_all_all_to_all_spec(source_spec, orig_cost))
        valid_spec_dict.update(
            self.get_all_mix_gather_spec(source_spec, orig_cost))
        valid_spec_dict.update(
            self.get_all_mixed_shard_spec(source_spec, orig_cost))
        valid_spec_dict.update(self.get_all_shard_spec(source_spec, orig_cost))
        return valid_spec_dict

    def mem_cost(self, comm_action_sequence: List[CommSpec], mem_pattern='opt'):
        """memory cost of the communication action sequence

        Args:
            comm_action_sequence (List[CommSpec]): list of communication actions

        Returns:
            TrainCycleItem: memory (numel) cost of such comm_action_sequence
        """

        def compute_shape(sharding_spec: ShardingSpec):
            if 'opt' == mem_pattern:
                return sharding_spec.get_sharded_shape_per_device()
            elif 'max' == mem_pattern:
                return sharding_spec.get_max_sharded_shape_per_device()
            else:
                return 0.0

        def gather_analysis(comm_spec, peak_mem):
            """analyze all_gather memory footprint
            all_gather will allocate memory for the output tensor, and there will be temp memory for
            all_gather operation, which is twice the size of output tensor

            Args:
                comm_spec (CommSpec): input CommSpec
            """
            input_shape = compute_shape(comm_spec.sharding_spec)
            input_numel = reduce(operator.mul, input_shape, 1)
            for axes in comm_spec.logical_process_axis:
                for axis in axes:
                    output_numel = input_numel * comm_spec.device_mesh.mesh_shape[
                        axis]
            alloc_mem = (input_numel +
                         output_numel * 2) * comm_spec.sharding_spec.dtype_size
            peak_mem = max(peak_mem, alloc_mem)
            return peak_mem

        def reduce_scatter_analysis(comm_spec, peak_mem):

            input_shape = compute_shape(comm_spec.sharding_spec)
            input_numel = reduce(operator.mul, input_shape, 1)
            output_numel = input_numel
            for axes in comm_spec.logical_process_axis:
                for axis in axes:
                    output_numel = output_numel / comm_spec.device_mesh.mesh_shape[
                        axis]
            alloc_mem = (input_numel +
                         output_numel * 2) * comm_spec.sharding_spec.dtype_size
            peak_mem = max(peak_mem, alloc_mem)

            return peak_mem

        def split_analysis(comm_spec: CommSpec, peak_mem: int):
            """analyze split memory footprint
            split will allocate memory for the output tensor if we don't apply shard on the first dimension of
            the input tensor. If we apply shard on the first dimension, the `torch.tensor.contiguous()` will not
            generate new tensor in this case, so no memory will be allocated.

            Args:
                comm_spec (CommSpec): input CommSpec
                discard_input (bool): whether to discard the input tensor
                alloc_numel (int): current allocated numel
                peak_numel (int): current peak numel
            """
            shard_dim = comm_spec.shard_dim
            if shard_dim != 0:
                # if we don't shard the tensor on the first dimension, the split action will
                # generate a new tensor
                input_shape = compute_shape(comm_spec.sharding_spec)
                input_numel = reduce(operator.mul, input_shape, 1)
                output_numel = input_numel
                for axes in comm_spec.logical_process_axis:
                    for axis in axes:
                        output_numel = output_numel / comm_spec.device_mesh.mesh_shape[
                            axis]
                alloc_mem = (input_numel +
                             output_numel) * comm_spec.sharding_spec.dtype_size
                peak_mem = max(peak_mem, alloc_mem)
            else:
                # if we shard the tensor on the first dimension, the split action will not generate
                # a new tensor, and as it will preserve a reference to the input tensor, we could
                # override the discard_input option here
                # NOTE: this special case might fail in some weird cases, e.g. if we have three split
                # actions in the comm actions sequence, the first split action operate on the second dimension,
                # the second split action operate on the first dimension, and the third split action operate, again,
                # on the second dimension. Therefore, after the first two actions in the sequence, we will allocate
                # memory the same size as the output of first split action. However, the third split action will discard
                # the input tensor, and it actually should discard the tensor generated by the first split action, so in
                # the current memory estimation framework, we will overestimate the memory usage. But the above case is
                # kind of weird, and I think we could ignore it for now.
                pass
            return peak_mem

        def reduce_analysis(comm_spec: CommSpec, peak_mem: int):
            input_shape = compute_shape(comm_spec.sharding_spec)
            input_numel = reduce(operator.mul, input_shape, 1)
            output_numel = input_numel
            alloc_mem = (input_numel +
                         output_numel) * comm_spec.sharding_spec.dtype_size
            peak_mem = max(peak_mem, alloc_mem)
            return peak_mem

        def all2all_analysis(comm_spec: CommSpec, peak_mem: int):
            input_shape = compute_shape(comm_spec.sharding_spec)
            input_numel = reduce(operator.mul, input_shape, 1)
            output_numel = input_numel
            comm_spec.shard_dim
            alloc_mem = (input_numel +
                         output_numel * 3) * comm_spec.sharding_spec.dtype_size
            peak_mem = max(peak_mem, alloc_mem)
            return peak_mem

        def peer_to_peer_analysis(comm_spec: CommSpec, peak_mem: int):
            input_shape = compute_shape(comm_spec.sharding_spec)
            input_numel = reduce(operator.mul, input_shape, 1)
            alloc_mem = (input_numel) * comm_spec.sharding_spec.dtype_size
            peak_mem = max(peak_mem, alloc_mem)
            return peak_mem

        pattern_to_func_dict = {
            'all_gather': gather_analysis,
            'all_to_all': all2all_analysis,
            'split': split_analysis,
            'all_reduce': reduce_analysis,
            'reduce_scatter': reduce_scatter_analysis,
            'peer_to_peer': peer_to_peer_analysis
        }

        fwd_actions = []
        # construct forward and backward comm actions sequence
        for comm_spec in comm_action_sequence:
            fwd_action = pattern_to_func_dict[comm_spec.comm_pattern]
            fwd_actions.append(fwd_action)

        # analyze memory footprint of forward comm actions sequence
        fwd_peak_numel = 0
        for idx, action_spec_pair in enumerate(
                zip(fwd_actions, comm_action_sequence)):
            # the first forward comm action will not discard input
            fwd_action, comm_spec = action_spec_pair
            fwd_peak_numel = fwd_action(comm_spec, fwd_peak_numel)

        return fwd_peak_numel

    def print_shape_consistency_result(self,
                                       transform_path,
                                       comm_action_sequence,
                                       resharding_cost,
                                       file=None):
        for idx, tpath in enumerate(transform_path):
            print(
                f'sharding_info = [op_shape:{tpath.entire_shape}, sharding_spec:{tpath.sharding_sequence}, sharded_shape:{tpath.get_sharded_shape_per_device()}]',
                end=" ",
                file=file)
            print('->', end=" ", file=file)
            try:
                commspec = comm_action_sequence[idx]
                comm = [
                    commspec.comm_pattern, commspec.gather_dim,
                    commspec.shard_dim, commspec.logical_process_axis
                ]
            except:
                comm = ''
            print(f'comm_info = {comm}', end=" ", file=file)
            print('->', end=" ", file=file)
        print(f'total_cost = {resharding_cost}', file=file)

    def construct_transform_path_from_cache(self, src_spec, target_spec,
                                            old_transform_path,
                                            old_comm_action_sequence,
                                            orig_cost):
        new_transform_path = [src_spec]
        new_comm_action_sequence = []
        new_cost = orig_cost
        new_src_spec = src_spec
        for idx, old_comm_spec in enumerate(old_comm_action_sequence):
            new_comm_spec = CommSpec(
                old_comm_spec.comm_pattern,
                sharding_spec=new_src_spec,
                gather_dim=old_comm_spec.gather_dim,
                shard_dim=old_comm_spec.shard_dim,
                logical_process_axis=old_comm_spec.logical_process_axis,
                forward_only=old_comm_spec.forward_only,
                mix_gather=old_comm_spec.mix_gather)
            new_comm_action_sequence.append(new_comm_spec)
            new_cost += new_comm_spec.get_comm_cost()
            old_target_spec = old_transform_path[idx + 1]
            new_target_spec = ShardingSpec(new_src_spec.device_mesh,
                                           new_src_spec.data_type_size,
                                           new_src_spec.entire_shape,
                                           new_src_spec.max_entire_shape,
                                           new_src_spec.raw_shape,
                                           old_target_spec.dim_partition_dict)
            new_transform_path.append(new_target_spec)
            new_src_spec = new_target_spec
        assert new_transform_path[-1].get_sharded_shape_per_device(
        ) == target_spec.get_sharded_shape_per_device(
        ), 'failed to insert the cache transform path'
        return new_transform_path, new_comm_action_sequence, new_cost

    def shape_consistency(
        self, source_spec: ShardingSpec, target_spec: ShardingSpec
    ) -> Tuple[List[ShardingSpec], List[CommSpec], float]:
        '''
        This method will find a path to transform source_spec to target_spec with
        a greedy algorithm.
        The basic idea is:
        Step1:
            Generate all one-step transform sequences from source_spec.
        Step2:
            Pick the 'best' sharding spec following the heuristic function.
        Step3:
            Repeat above steps until the source spec transform to target spec.
        '''
        MAX_TRANSFORM_STEPS = 20
        total_cost = 0.0
        total_steps = 0
        transform_path = []
        comm_action_sequence = []
        # We do nothing if the sharding spec is all the same.
        if source_spec.sharding_sequence_difference(target_spec) == 0:
            return (transform_path, comm_action_sequence, total_cost)

        spec_pairs = (str(source_spec.sharding_sequence),
                      str(target_spec.sharding_sequence))

        if spec_pairs in self.cached_spec_pairs_transform_path:
            transform_path, comm_action_sequence = self.cached_spec_pairs_transform_path[
                spec_pairs]
            new_transform_path, new_comm_action_sequence, new_total_cost = self.construct_transform_path_from_cache(
                source_spec, target_spec, transform_path, comm_action_sequence,
                total_cost)
            self.cache_hit += 1
            return (new_transform_path, new_comm_action_sequence,
                    new_total_cost)

        else:
            self.cache_miss += 1

        temp_sharding_spec = source_spec
        transform_path.append(temp_sharding_spec)
        # To avoid dead loop, the loop will break after MAX_TRANSFORM_STEPS transforms
        while total_steps <= MAX_TRANSFORM_STEPS:
            valid_transform_spec_dict = self.get_all_one_step_transform_spec(
                temp_sharding_spec, total_cost)
            best_difference_score = math.inf

            for sharding_spec, info_pairs in valid_transform_spec_dict.items():
                comm_spec, cost = info_pairs
                spec_difference = sharding_spec.sharding_sequence_difference(
                    target_spec)

                if spec_difference == 0:
                    total_cost = cost
                    transform_path.append(sharding_spec)
                    comm_action_sequence.append(comm_spec)
                    self.cached_spec_pairs_transform_path[spec_pairs] = (
                        transform_path, comm_action_sequence)
                    return (transform_path, comm_action_sequence, total_cost)

                if spec_difference < best_difference_score:
                    temp_sharding_spec = sharding_spec
                    temp_cost = cost
                    temp_comm_spec = comm_spec
                    best_difference_score = spec_difference

            transform_path.append(temp_sharding_spec)
            comm_action_sequence.append(temp_comm_spec)
            total_cost = temp_cost
            total_steps += 1

        raise RuntimeError(
            f"Could not find a valid transform path with in {MAX_TRANSFORM_STEPS} steps."
        )

    def dum_transform_path_from_cache(self):
        src_specs, tgt_specs, path_strs = [], [], []
        for spec_pairs, trans_comm_path in self.cached_spec_pairs_transform_path.items(
        ):
            src_specs.append(spec_pairs[0])
            tgt_specs.append(spec_pairs[1])
            trans_paths, comm_specs = trans_comm_path[0], trans_comm_path[1]
            path_str = f'{spec_pairs[0]}->'
            for idx in range(1, len(trans_paths)):
                comm_spec = comm_specs[idx - 1]
                comm_str = f'{comm_spec.comm_pattern}: gather_dim{comm_spec.gather_dim}, shard_dim{comm_spec.shard_dim}, mesh_axis{comm_spec.logical_process_axis}->'
                path_str += comm_str
                path_str += f'{trans_paths[idx].sharding_sequence}->'
            path_strs.append(path_str)
        ret_dict = {
            'src_spec': src_specs,
            'dst_specs': tgt_specs,
            'trans_path': path_strs
        }
        ret_df = pd.DataFrame.from_dict(ret_dict)
        return ret_df

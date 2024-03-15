import operator
from functools import reduce

import tensorrt as trt

from tensorrt_llm.logger import logger

ALLGATHER_COST = 20
SHARD_COST = 5
STEP_PENALTY = 6
NAN = 'nan'


def _convert_str_to_shard_list(str_spec):
    '''
    Convert str_spec into shard_list.

    Argument:
        str_spec(str): dim spec in str type.
    '''

    if str_spec == 'R':
        return []
    if str_spec == 'S0':
        return [0]
    if str_spec == 'S1':
        return [1]
    if str_spec == 'S01':
        return [0, 1]


def _build_difference_2d_dict():
    '''
    Build a difference mapping for 2D device mesh case. It will be used to
    compute the difference between DimSpec pairs.
    '''

    source_spec_list = ['R', 'S0', 'S1', 'S01']
    target_spec_list = ['R', 'S0', 'S1', 'S01']
    difference_dict = {}
    for source_spec in source_spec_list:
        for target_spec in target_spec_list:
            spec_pair = (source_spec, target_spec)
            source_shard_list = _convert_str_to_shard_list(source_spec)
            target_shard_list = _convert_str_to_shard_list(target_spec)

            # source same as target
            if source_shard_list == target_shard_list:
                difference = 0

            # all_gather(source) -> target
            elif len(source_shard_list) == len(
                    target_shard_list
            ) + 1 and source_shard_list[:-1] == target_shard_list:
                difference = ALLGATHER_COST

            # shard(source) -> target
            elif len(source_shard_list) == len(
                    target_shard_list
            ) - 1 and source_shard_list == target_shard_list[:-1] and target_shard_list[
                    -1] not in source_shard_list:
                difference = SHARD_COST

            # S1 -> S0 or S0 -> S1
            elif len(source_shard_list) == len(target_shard_list):
                # source -> R -> target
                difference = ALLGATHER_COST + STEP_PENALTY + SHARD_COST

            # R -> S01
            elif len(source_shard_list) == len(target_shard_list) - 2:
                difference = SHARD_COST + STEP_PENALTY + SHARD_COST

            # S01 -> R
            elif len(source_shard_list) == len(target_shard_list) + 2:
                difference = ALLGATHER_COST + STEP_PENALTY + ALLGATHER_COST

            # S1 -> S01
            elif len(source_shard_list) == len(target_shard_list) - 1:
                difference = ALLGATHER_COST + STEP_PENALTY + SHARD_COST + STEP_PENALTY + SHARD_COST

            # S01 -> S1
            elif len(source_shard_list) == len(target_shard_list) + 1:
                difference = ALLGATHER_COST + STEP_PENALTY + ALLGATHER_COST + STEP_PENALTY + SHARD_COST

            else:
                difference = NAN
            difference_dict[spec_pair] = difference

    return difference_dict


_difference_dict = _build_difference_2d_dict()


class DimSpec:
    '''
    Sharding spec for single dimension of the sharded tensor describe the sharding dimension of
    logical device mesh and give a method to compute the difference between them.

    Argument:
        shard_list(List[int]): if shard_list is empty, the dim spec will be 'R' type.
            Otherwise, the element in shard_list means the data will be sharded in that dimension.
    '''

    def __init__(self, shard_list):
        self.is_replica = len(shard_list) == 0
        self.shard_list = shard_list

    def __eq__(self, other):
        return str(self) == str(other)

    def __repr__(self):
        if self.is_replica:
            return 'R'
        target = 'S'
        for dim in self.shard_list:
            target += str(dim)
        return target

    def difference(self, other):
        '''
        The difference between two DimSpec.

        Argument:
            other(DimSpec): the dim spec to compare with.

        Return:
            difference(int): the difference between two DimSpec.

        Example:
            dim_spec = DimSpec([0])
            other_dim_spec = DimSpec([0, 1])
            print(dim_spec.difference(other_dim_spec))

        Output:
            5
        '''
        difference = _difference_dict[(str(self), str(other))]
        return difference


def get_sharding_sequence(num_dims, dims, device_dims):
    sharding_sequence = [DimSpec([])] * num_dims
    for dim, shard_list in zip(dims, device_dims):
        sharding_sequence[dim] = DimSpec(shard_list)
    return sharding_sequence


class ShardingSpec:
    '''
    Sharding spec for a tensor, it contains info of the logical device mesh this tensor belong
    to, the entire shape of the tensor before sharded, and the sharding sequence looks like
    [R, R, S0, S1].

    Argument:
        device_mesh: A logical view of a physical mesh.
        entire_shape: The entire shape of tensor before sharded.
        dim_partition_dict: The key is the dimension of tensor to be sharded,
            and the value of the key describe which logical axis will be sharded in that dimension.
        sharding_sequence(List[DimSpec], optional): A straight view of ShardingSpec looks like [R, R, S0, S1].
    '''

    def __init__(self,
                 device_mesh,
                 data_type_size,
                 data_shape,
                 max_data_shape,
                 raw_data_shape,
                 dim_partition_dict=None,
                 sharding_sequence=None):
        self.device_mesh = device_mesh
        self.data_type_size = data_type_size
        self.dtype = data_type_size[0]
        self.dtype_size = data_type_size[1]
        self.entire_shape = data_shape
        self.max_entire_shape = max_data_shape
        self.raw_shape = raw_data_shape
        self.dim_partition_dict = dim_partition_dict
        self.sharding_sequence = sharding_sequence
        self.enable_shard_unbalanced_shape = device_mesh.config.enable_shard_unbalanced_shape
        self.enable_shard_dynamic_shape = device_mesh.config.enable_shard_dynamic_shape
        if self.sharding_sequence is None:
            self.dim_partition_dict = self._merge_same_dim_mesh_list(
                len(self.entire_shape), self.dim_partition_dict)
            self.dim_partition_dict = self._convert_dim_partition_dict(
                len(self.entire_shape), self.dim_partition_dict)
            assert self.dim_partition_dict is not None, f'dim_partition_dict should not be None, if sharding_sequence is NoneType object.'
            self.convert_dict_to_shard_sequence()
        elif self.dim_partition_dict is None:
            assert self.sharding_sequence is not None, f'sharding_sequence should not be None, if dim_partition_dict is NoneType object.'
            self.convert_shard_sequence_to_dict()
            self.dim_partition_dict = self._merge_same_dim_mesh_list(
                len(self.entire_shape), self.dim_partition_dict)
            self.dim_partition_dict = self._convert_dim_partition_dict(
                len(self.entire_shape), self.dim_partition_dict)

        self.sharded_shape, self.max_sharded_shape = [*self.entire_shape], [
            *self.max_entire_shape
        ]
        for dim, shard_list in self.dim_partition_dict.items():
            mesh_list = [
                self.device_mesh.mesh_shape[mesh_dim] for mesh_dim in shard_list
            ]
            shard_partitions = reduce(operator.mul, mesh_list, 1)
            self.sharded_shape[dim] = (self.sharded_shape[dim] +
                                       shard_partitions - 1) // shard_partitions
            self.max_sharded_shape[dim] = (self.max_sharded_shape[dim] +
                                           shard_partitions -
                                           1) // shard_partitions

    def print_spec(self, file=None):
        print(
            f"sharding_sequence = {self.sharding_sequence}, shape = {self.get_sharded_shape_per_device()}",
            file=file,
        )

    def _merge_same_dim_mesh_list(self, dim_size, dim_partition_dict):
        '''
        This method is used to merge the different key value which points to same physical position.

        For example:
            dim_partition_dict: {1 :[0], -1: [1]} or {1: [0], 1: [1]} for a 2d tensor, the dim 1 and -1 point same physical position.
            In this method, above dim_partition_dict will be converted to {1: [0, 1]}
        '''
        converted_dim_partition_dict = {}
        for dim, mesh_list in dim_partition_dict.items():
            if dim < 0:
                dim = dim_size + dim
            if dim not in converted_dim_partition_dict:
                converted_dim_partition_dict[dim] = mesh_list
            else:
                converted_dim_partition_dict[dim].extend(mesh_list)
                converted_dim_partition_dict[dim].sort()
        return converted_dim_partition_dict

    def _convert_dim_partition_dict(self, dim_size, dim_partition_dict):
        dims_to_convert = []
        for dim, mesh_list in dim_partition_dict.items():
            if dim < 0:
                dims_to_convert.append(dim)
        for dim in dims_to_convert:
            dim_partition_dict.pop(dim)
            dim_partition_dict[dim_size + dim] = mesh_list
        return dim_partition_dict

    def _remove_mesh_dim_one(self, dim_partition_dict):
        dims_to_remove = []
        for dim, mesh_list in dim_partition_dict.items():
            new_mesh_list = []
            for mesh_dim in mesh_list:
                if self.device_mesh.mesh_shape[mesh_dim] != 1:
                    new_mesh_list.append(mesh_dim)
            if 0 != len(new_mesh_list):
                dim_partition_dict[dim] = new_mesh_list
            else:
                dims_to_remove.append(dim)
        for dim in dims_to_remove:
            dim_partition_dict.pop(dim)
        return dim_partition_dict

    def __repr__(self):
        res = "DistSpec("
        res += f"shard_sequence={self.sharding_sequence},"
        res += f"shape={self.device_mesh.mesh_shape}"
        res += ")"
        return res

    def sanity_check(self):
        # make sure all axes in logical device mesh only be used once
        dim_check_list = [*range(len(self.device_mesh.mesh_shape))]
        for dim, shard_list in self.dim_partition_dict.items():
            for element in shard_list:
                if element in dim_check_list:
                    dim_check_list.remove(element)
                else:
                    logger.warning(
                        f"find an invalid sharding axis {element} in dim_partition_dict in tensor dimension {dim}. dim_partition_dict={self.dim_partition_dict}"
                    )
                    return False

        # make sure that the dimension is not out of index
        for dim in self.dim_partition_dict.keys():
            # we have tried to convert the negative value to positive value, if it is larger than the dim_size or negative still, it is out of index
            if dim >= len(self.entire_shape) or dim < 0:
                print(
                    f"The dim_partition_dict specifies to shard dimension {dim} but the entire_shape only has {len(self.entire_shape)} dimensions"
                )
                return False

        if not self.enable_shard_dynamic_shape:
            # make sure to not to shard on dynamic shape
            for dim, shard_list in self.dim_partition_dict.items():
                if len(shard_list) == 0:
                    continue
                if len(self.raw_shape) == 0:
                    continue
                if -1 == self.raw_shape[dim]:
                    return False

        # make sure that the sharding for a dimension is divisible by the number of devices
        for dim, shard_list in self.dim_partition_dict.items():
            if len(shard_list) == 0:
                continue
            tensor_dim_size = self.entire_shape[dim]
            num_devices = 1

            for element in shard_list:
                num_devices *= self.device_mesh.mesh_shape[element]
            if num_devices == 1:
                # we only support RR when the device is 1
                return False

            if not self.enable_shard_unbalanced_shape:
                if tensor_dim_size % num_devices != 0 or tensor_dim_size == 1:
                    '''
                    print(
                        f'The size of static dimension at index {dim} is {tensor_dim_size}, it cannot be sharded over {num_devices} devices.'
                    )
                    '''
                    return False
            else:
                if tensor_dim_size == 1:
                    return False
        '''
        if self.get_sharded_size_per_device() > (2**31 - 1):
            print(
                f'memory footprint per device {self.get_sharded_size_per_device()} is larger than 2**31 - 1'
            )
            return False
        '''
        return True

    def convert_dict_to_shard_sequence(self):
        '''
        Convert dim_partition_dict into list of DimSpec, and assign it to sharding_sequence.
        '''
        sharding_sequence = [DimSpec([])] * len(self.entire_shape)
        for dim, shard_list in self.dim_partition_dict.items():
            sharding_sequence[dim] = DimSpec(shard_list)
        self.sharding_sequence = sharding_sequence

    def convert_shard_sequence_to_dict(self):
        '''
        Convert sharding_sequence into dim_partition_dict.
        '''
        new_dim_partition_dict = {}
        for index, dim_spec in enumerate(self.sharding_sequence):
            if not dim_spec.is_replica:
                if index not in new_dim_partition_dict:
                    new_dim_partition_dict[index] = []
                new_dim_partition_dict[index].extend(dim_spec.shard_list)
        self.dim_partition_dict = new_dim_partition_dict

    def sharding_sequence_difference(self, other):
        '''
        This function is a naive version of difference computation. It just simply accumulates difference every dimension between the
        pair of sharding sequence.

        Example:
            dim_partition_dict = {0: [0, 1]}
            # DistSpec:
            #     shard_sequence: S01,R,R
            #     device_mesh_shape: (4, 4)
            sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)
            dim_partition_dict_to_compare = {0: [0], 1: [1]}
            # DistSpec:
            #     shard_sequence: S0,S1,R
            #     device_mesh_shape: (4, 4)
            sharding_spec_to_compare = ShardingSpec(device_mesh, entire_shape, dim_partition_dict_to_compare)
            print(sharding_spec.sharding_sequence_difference(sharding_spec_to_compare))

        Output:
            25

        Argument:
            other(ShardingSpec): The ShardingSpec to compared with.

        Return:
            difference(int): Difference between two ShardingSpec.
        '''
        assert len(self.sharding_sequence) == len(
            other.sharding_sequence
        ), f'Cannot compare difference for two sharding specs with different length.'
        difference = 0
        for orig_dim_spec, other_dim_spec in zip(self.sharding_sequence,
                                                 other.sharding_sequence):
            difference += orig_dim_spec.difference(other_dim_spec)
        return difference

    def get_sharded_shape_per_device(self, ):
        return self.sharded_shape

    def get_sharded_element_per_device(self, ):
        sharded_shape = self.get_sharded_shape_per_device()
        if len(sharded_shape) == 0:
            num_elements = 1
        else:
            num_elements = trt.volume(sharded_shape)
        return num_elements

    def get_sharded_size_per_device(self, ):
        num_elements = self.get_sharded_element_per_device()
        return num_elements * self.dtype_size

    def get_max_sharded_shape_per_device(self, ):
        return self.max_sharded_shape

    def get_max_sharded_element_per_device(self, ):
        max_sharded_shape = self.get_max_sharded_shape_per_device()
        if len(max_sharded_shape) == 0:
            num_elements = 1
        else:
            num_elements = trt.volume(max_sharded_shape)
        return num_elements

    def get_max_sharded_size_per_device(self, ):
        num_elements = self.get_max_sharded_element_per_device()
        return num_elements * self.dtype_size

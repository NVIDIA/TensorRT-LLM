__all__ = [
    'CommSpec',
]


class CommSpec:

    def __init__(self,
                 comm_pattern,
                 sharding_spec,
                 gather_dim=None,
                 shard_dim=None,
                 logical_process_axis=None,
                 mix_gather=False,
                 forward_only=True):
        self.comm_pattern = comm_pattern
        self.sharding_spec = sharding_spec
        self.gather_dim = gather_dim
        self.shard_dim = shard_dim
        self.logical_process_axis = logical_process_axis
        self.device_mesh = self.sharding_spec.device_mesh
        self.mix_gather = mix_gather
        self.forward_only = forward_only
        if self.gather_dim:
            assert len(self.gather_dim) == len(
                self.logical_process_axis
            ), f'unmatched gather dim {self.gather_dim} and logical process axis {self.logical_process_axis}'
        if self.shard_dim:
            assert len(self.shard_dim) == len(
                self.logical_process_axis
            ), f'unmatched shard dim {self.shard_dim} and logical process axis {self.logical_process_axis}'
        if self.gather_dim and self.shard_dim:
            assert len(self.shard_dim) == len(
                self.gather_dim
            ), f'unmatched gather dim {self.gather_dim} and shard dim {self.shard_dim}'

    def get_comm_cost(self):
        '''
        For all_gather, all2all, and all_reduce operation, the formula provided in DeviceMesh with alpha-beta model is used to
        compute the communication cost.
        For shard operation, it is an on-chip operation, so the communication cost is zero.
        '''
        comm_size = self.sharding_spec.get_sharded_size_per_device()
        dtype = self.sharding_spec.dtype

        # reduce list_of_list to list
        comm_dims = sum(self.logical_process_axis, [])
        comm_cost = self.device_mesh.estimate_comm_cost(self.comm_pattern,
                                                        comm_dims, comm_size,
                                                        dtype)
        return comm_cost

    def get_mem_cost(self):
        return self.device_mesh.shape_consistency_manager.mem_cost([self])

    def get_max_mem_cost(self):
        return self.device_mesh.shape_consistency_manager.mem_cost(
            [self], mem_pattern='max')

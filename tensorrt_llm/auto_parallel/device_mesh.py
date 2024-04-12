import os
import re
from abc import ABC, abstractmethod
from typing import List

import h5py
import numpy as np
from filelock import FileLock

from .config import AutoParallelConfig, CostModel
from .tensor_parallel.shape_consistency import ShapeConsistencyManager


class ProfileDB(ABC):
    """A database that stores profiling results for multiple device mesh
    shapes."""

    @abstractmethod
    def query(self, cluster_key, data_key):
        ...

    @abstractmethod
    def update(self, cluster_key, data_key, mesh_result):
        ...

    def close(self):
        pass


class MemDB(ProfileDB):

    def __init__(self):
        self.data = {}

    def query(self, cluster_key, data_key):
        key = (cluster_key, data_key)
        mesh_result = self.data.get(key, None)
        if mesh_result is None:
            return None
        else:
            return mesh_result[0]

    def update(self, cluster_key, data_key, mesh_result):
        key = (cluster_key, data_key)
        self.data[key] = mesh_result


class Hdf5DB(ProfileDB):

    def __init__(self, name):
        self.name = name
        lock_name = self.name + ".lock"
        self.lock = FileLock(lock_name, thread_local=False)

    def query(self, cluster_key, data_key):
        file_name = f"{self.name}.hdf5"
        key = str((cluster_key, data_key))
        self.lock.acquire()
        mesh_result = None
        with h5py.File(file_name, 'a') as f:
            if key in f:
                self.lock.release()
                mesh_result = f[key]
                return mesh_result[0]
            else:
                return None

    def update(self, cluster_key, data_key, mesh_result):
        key = str((cluster_key, data_key))
        file_name = f"{self.name}.hdf5"
        with h5py.File(file_name, 'a') as f:
            f[key] = mesh_result

    def close(self):
        self.lock.release(force=True)


class LogicalDeviceMesh(object):

    def __init__(self,
                 phy_mesh_shape,
                 mesh_shape,
                 phy_ids,
                 config: AutoParallelConfig,
                 alpha,
                 beta,
                 sharp,
                 prof_database=None,
                 shape_consistency_manager=None,
                 host_ips=None):
        self.phy_mesh_shape = phy_mesh_shape
        self.mesh_shape = mesh_shape
        self.phy_ids = phy_ids
        self.host_ips = host_ips
        self.cluster_key = config.cluster_key + '_mesh_shape{}'.format('_'.join(
            [str(i) for i in mesh_shape]))
        self.prof_min_max_size = [1, 2**34]
        self.prof_comm_dtypes = [
            "int8", "uint8", "int32", "uint32", "int64", "uint64", "float16",
            "float32", "float64", "bfloat16"
        ]
        self.devices_group = {
            (0, ): [self.phy_ids.transpose(), self.mesh_shape[1] - 1],
            (1, ): [self.phy_ids, self.mesh_shape[1]],
            (0, 1): [self.phy_ids.reshape([1, self.phy_ids.size]), 0]
        }
        self.prof_database = prof_database
        self.shape_consistency_manager = shape_consistency_manager
        self.config = config
        self.cluster_info = config.get_cluster_info()
        self.hw_alpha = alpha
        self.hw_beta = beta
        self.hw_sharp = sharp
        self.algo_alpha_beta = self._estimate_algo_alpha_beta()
        self.comm_op_to_nccl_test_func_name = {
            'all_reduce': 'all_reduce_perf_mpi',
            'all_gather': 'all_gather_perf_mpi',
            'all_to_all': 'alltoall_perf_mpi',
            'reduce_scatter': 'reduce_scatter_perf_mpi',
            'split': 'split',
        }

    @property
    def size(self) -> int:
        return self.phy_ids.size

    def _estimate_algo_alpha_beta(self):
        ret = {}
        ar_alpha, ar_beta = {}, {}
        ag_alpha, ag_beta = {}, {}
        rs_alpha, rs_beta = {}, {}
        a2a_alpha, a2a_beta = {}, {}
        phy_num_hosts, phy_num_devices_per_host = self.phy_mesh_shape
        if phy_num_hosts == 1 or phy_num_devices_per_host == 1:
            for dims in [(0, ), (1, ), (0, 1), (1, 0)]:
                num_devices = 1
                for dim in dims:
                    num_devices = self.mesh_shape[dim] * num_devices
                if num_devices != 1:
                    ar_alpha[dims] = self.hw_alpha[0] if self.hw_sharp[
                        0] else self.hw_alpha[0] * num_devices / 2 / (
                            num_devices - 1)
                    ar_beta[dims] = self.hw_beta[0]
                    ag_alpha[dims] = self.hw_alpha[0] * num_devices / (
                        num_devices - 1)
                    ag_beta[dims] = self.hw_beta[0]
                    rs_alpha[dims] = self.hw_alpha[0] * num_devices / (
                        num_devices - 1)
                    rs_beta[dims] = self.hw_beta[0]
                    a2a_alpha[dims] = self.hw_alpha[0] * num_devices / (
                        num_devices - 1)
                    a2a_beta[dims] = self.hw_beta[0]
        # phy and logical have the same mesh shape if num_hosts > 1 and num_devices_per_host > 1
        else:
            for dims in [(0, ), (1, ), (0, 1), (1, 0)]:
                num_devices = 1
                for dim in dims:
                    num_devices = self.mesh_shape[dim] * num_devices
                if num_devices != 1:
                    if len(dims) == 1:
                        dim = dims[0]
                        ar_alpha[dims] = self.hw_alpha[dim] if self.hw_sharp[
                            dim] else self.hw_alpha[dim] * num_devices / 2 / (
                                num_devices - 1)
                        ar_beta[dims] = self.hw_beta[dim]
                        ag_alpha[dims] = self.hw_alpha[dim] * num_devices / (
                            num_devices - 1)
                        ag_beta[dims] = self.hw_beta[dim]
                        rs_alpha[dims] = self.hw_alpha[dim] * num_devices / (
                            num_devices - 1)
                        rs_beta[dims] = self.hw_beta[dim]
                        a2a_alpha[dims] = self.hw_alpha[dim] * num_devices / (
                            num_devices - 1)
                        a2a_beta[dims] = self.hw_beta[dim]
                    elif len(dims) == 2:  # two level communication
                        num_hosts, num_devices_per_host = phy_num_hosts, phy_num_devices_per_host
                        inter_node_col_alpha = self.hw_alpha[
                            0] * num_devices_per_host
                        inter_node_ar_alpha = inter_node_col_alpha if self.hw_sharp[
                            0] else inter_node_col_alpha * num_hosts / 2 / (
                                num_hosts - 1)
                        intra_node_ar_alpha = self.hw_alpha[1]
                        intra_node_ar_alpha = intra_node_ar_alpha if self.hw_sharp[
                            1] else intra_node_ar_alpha * num_devices_per_host / 2 / (
                                num_devices_per_host - 1)
                        ar_alpha[dims] = min(inter_node_ar_alpha,
                                             intra_node_ar_alpha)
                        ar_beta[dims] = max(self.hw_beta)
                        ag_alpha[dims] = min(
                            inter_node_col_alpha * num_hosts / (num_hosts - 1),
                            self.hw_alpha[1] * num_devices_per_host /
                            (num_devices_per_host - 1))
                        ag_beta[dims] = max(self.hw_beta)
                        rs_alpha[dims] = ag_alpha[dims]
                        rs_beta[dims] = ag_beta[dims]
                        a2a_alpha[dims] = min(
                            num_hosts * self.hw_alpha[0] / (num_hosts - 1),
                            self.hw_alpha[1] * num_hosts)
                        a2a_beta[dims] = max(self.hw_beta)
                    else:
                        pass
        ret['all_to_all'] = [a2a_alpha, a2a_beta]
        ret['all_reduce'] = [ar_alpha, ar_beta]
        ret['all_gather'] = [ag_alpha, ag_beta]
        ret['reduce_scatter'] = [rs_alpha, rs_beta]
        ret['p2p_cross_device'] = [
            self.cluster_info.intra_node_bw_per_device,
            self.cluster_info.intra_node_latency
        ]
        ret['p2p_cross_host'] = [
            self.cluster_info.inter_node_bw_per_device,
            self.cluster_info.inter_node_latency
        ]
        return ret

    #[ToDo][KDuan] stub functions here
    def _profile_split(self, min_max_comm_size):
        comm_size, elapsed_time = [], []
        size = min_max_comm_size[0]
        while size <= min_max_comm_size[1]:
            time = size * 2 / self.cluster_info.memory_bw
            comm_size.append(size)
            elapsed_time.append(time)
            size = size * 2
        return np.array([comm_size, elapsed_time])

    def _prase_nccl_test_results(self, f_nccl_test_out_log):
        '''[ToDo][KDuan] There is some dtye that may not been supported by nccl test, using default dtype (float)'''
        start_parse = False
        comm_size, elapsed_time = [], []
        try:
            with open(f_nccl_test_out_log, 'r') as lines:
                for line in lines:
                    if start_parse:
                        prof_data = re.split(r"[ ]+", line.strip())
                        if len(prof_data) != 13:
                            continue
                        comm_size.append(float(prof_data[0]))
                        elapsed_time.append(float(prof_data[5]))
                    if 'GB/s' in line and 'us' in line:
                        start_parse = True
        except Exception:
            print(f'failed to parse {f_nccl_test_out_log}')
        return comm_size, elapsed_time

    def _profile_with_nccl_test(self, min_max_comm_size, dtype, device_group,
                                func_name, step, workload_key):

        if func_name == 'split':
            if 2 == step:
                return self._profile_split(min_max_comm_size)
            else:
                return None
        workspace_dir = self.config['profiling_workspace'] + f'/{workload_key}'
        os.makedirs(workspace_dir, exist_ok=True)
        outfile, errfile = workspace_dir + '/profile.out', workspace_dir + '/profile.err'
        if 1 == step:
            num_nodes = len(self.host_ips)
            num_gpus = self.mesh_shape[0] * self.mesh_shape[1]
            ntasks_per_node = num_gpus // num_nodes
            nccl_test_command = '"export NCCL_TESTS_SPLIT_MASK={} && export NCCL_COLLNET_ENABLE=1 && {} -b {} -e {} -g 1 -d {} -f {}"'.format(
                device_group[1], func_name, min_max_comm_size[0],
                min_max_comm_size[1], dtype, 2)
            sbatch_command = '#!/bin/bash\n'
            sbatch_command += '#SBATCH -p {}\n'.format(self.config['partition'])
            sbatch_command += '#SBATCH -A {}\n'.format(self.config['account'])
            sbatch_command += '#SBATCH -J {}\n'.format(self.config['jobname'])
            sbatch_command += '#SBATCH -N {}\n'.format(num_nodes)
            sbatch_command += '#SBATCH -t {}\n'.format(self.config['time'])
            sbatch_command += '#SBATCH --ntasks-per-node={}\n'.format(
                ntasks_per_node)
            sbatch_command += '#SBATCH --exclusive\n'
            sbatch_command += '#SBATCH --mem=0\n'
            sbatch_command += '#SBATCH --network=sharp\n'
            sbatch_command += '#SBATCH --mail-type=FAIL\n'
            srun_command = 'srun --nodes={} --mpi=pmix --ntasks-per-node={} --network=sharp -o {} -e {} --container-image={} bash -c '.format(
                num_nodes, ntasks_per_node, outfile, errfile,
                self.config['container'])
            command = sbatch_command + srun_command + nccl_test_command
            with open(workspace_dir + '/workload.sub', 'w') as f:
                f.write(command)
            with open('./preprofiling_step1.sh', 'a') as f:
                f.write(f'sbatch {workspace_dir}/workload.sub\n')
            return None

        else:
            comm_size, elapsed_time = self._prase_nccl_test_results(outfile)
            if len(comm_size) < 2:
                assert 0, 'the profiling for {} was failed at step1, please try again'.format(
                    workload_key)
            else:
                print(workload_key, comm_size, elapsed_time)
            return np.array([comm_size, elapsed_time])

    def _profile_single_comm_perf(self, device_group, comm_op, step, data_key):
        results = {}
        func_name = self.comm_op_to_nccl_test_func_name[comm_op]
        for dtype in self.prof_comm_dtypes:
            size_time = self._profile_with_nccl_test(
                self.prof_min_max_size, dtype, device_group, func_name, step,
                data_key + f'_dtype{dtype}')
            results[dtype] = size_time
        return results

    def profile_all_comms_perf(self, step):
        if self.mesh_shape == (1, 1):
            return None
        mesh_results = self.prof_database.query(self.cluster_key,
                                                self.mesh_shape)
        if mesh_results:
            return mesh_results

        mesh_results = {}
        data_key = self.cluster_key + f'_mesh_shape{self.mesh_shape[0]}x{self.mesh_shape[1]}'
        for comm_op in [
                'all_reduce', 'all_to_all', 'all_gather', 'reduce_scatter',
                'split'
        ]:
            comm_perf = {}
            for dim, device_group in self.devices_group.items():
                # don't need to profile for mesh dim == 1
                if len(dim) == 1 and self.mesh_shape[dim[0]] == 1:
                    continue

                comm_perf[dim] = self._profile_single_comm_perf(
                    device_group, comm_op, step, data_key +
                    '_comm_op{}_dim{}'.format(comm_op, ''.join(map(str, dim))))
            mesh_results[comm_op] = comm_perf
        if 2 == step:
            self.prof_database.update(self.cluster_key, self.mesh_shape,
                                      mesh_results)

        return mesh_results

    def _model_comm_cost_from_s_curve(self, size_time_array, realsize):
        assert size_time_array[0][0] <= realsize <= size_time_array[0][-1],\
            'the comm_size: {} is not in the profile range: [{}{}]'\
                .format(realsize, size_time_array[0][0], size_time_array[0][-1])
        return np.interp(realsize, size_time_array[0], size_time_array[1])

    def _model_comm_cost_from_alpha_beta(self, comm_op, dim_key, size_in_bytes):
        elapsed_time = 0.0
        if 'split' == comm_op:
            elapsed_time = size_in_bytes * 2 / (
                self.cluster_info.memory_bw *
                self.cluster_info.memory_efficiency) * 1e-3
        else:
            dict_alpha, dict_beta = self.algo_alpha_beta[comm_op]
            alpha, beta = dict_alpha[dim_key], dict_beta[dim_key]
            elapsed_time = (size_in_bytes /
                            (alpha * self.cluster_info.communication_efficiency)
                            * 1e-3) + beta
        return elapsed_time

    def _input_size_to_comm_size(self, comm_op, dims, input_size):
        ret = input_size
        if 'all_gather' == comm_op:
            for dim in dims:
                ret = ret * self.mesh_shape[dim]
        return ret

    def estimate_comm_cost(self, comm_op, dim, input_size, dtype):

        size = self._input_size_to_comm_size(comm_op, dim, input_size)
        if self.config.comm_cost_model == CostModel.S_CURVE:
            mesh_perf = self.prof_database.query(self.cluster_key,
                                                 self.mesh_shape)
            assert mesh_perf is not None, 'the mesh is not profiled, mesh_shape = {}'.format(
                self.mesh_shape)
            comm_op_perf = mesh_perf.get(comm_op, None)
            assert comm_op_perf is not None, '{} is not profiled'.format(
                comm_op)
            elapsed_time = self._model_comm_cost_from_s_curve(
                comm_op_perf[tuple(dim)][dtype], size)
            return elapsed_time
        elif self.config.comm_cost_model == CostModel.ALPHA_BETA:
            elapsed_time = self._model_comm_cost_from_alpha_beta(
                comm_op, tuple(dim), size)
        elif self.config.comm_cost_model == CostModel.PROFILE:
            assert False, 'Unsupported profile based communication cost model now'
        elif self.config.comm_cost_model == CostModel.ZERO:
            elapsed_time = 0.0

        return elapsed_time  # us


class PhysicalDeviceMesh(object):

    def __init__(self,
                 phy_devices_id,
                 config: AutoParallelConfig,
                 prof_database=None,
                 shape_consistency_manager=None,
                 host_ips=None):
        self.phy_devices_id = np.array(phy_devices_id)
        self.num_hosts, self.num_devices_per_host = self.phy_devices_id.shape
        self.host_ips = host_ips
        if host_ips is None:
            self.host_ips = [''] * self.num_hosts
        self.config = config
        self.cluster_info = config.get_cluster_info()
        self.prof_database: ProfileDB = prof_database
        self.shape_consistency_manager = shape_consistency_manager
        if self.config.comm_cost_model not in CostModel:
            raise ValueError(
                f'unsupported communication cost model: {self.config.comm_cost_model}'
            )
        if self.config.sharding_cost_model not in CostModel:
            raise ValueError(
                f'unsupported sharding cost model: {self.config.sharding_cost_model}'
            )
        if self.config.comm_cost_model == CostModel.S_CURVE or self.config.sharding_cost_model == CostModel.PROFILE:
            if self.prof_database is None:
                profile_cache = config.profile_cache
                if profile_cache is None:
                    self.prof_database = MemDB()
                else:
                    self.prof_database = Hdf5DB(profile_cache)
        elif self.config.comm_cost_model == CostModel.ALPHA_BETA:
            assert self.cluster_info.intra_node_bw_per_device > 0, 'intra_node_bw_per_device is needed for alpha_beta method'
            assert self.cluster_info.inter_node_bw_per_device > 0, 'inter_node_bw_per_device is needed for alpha_beta method'
        if self.config.sharding_cost_model == CostModel.ALPHA_BETA:
            assert self.cluster_info.memory_bw > 0, 'memory_bw is needed for alpha_beta method'

        if not shape_consistency_manager:
            self.shape_consistency_manager = ShapeConsistencyManager()

    @property
    def size(self) -> int:
        return self.phy_devices_id.size

    def close(self):
        if self.prof_database is not None:
            self.prof_database.close()

    def split_pipeline_meshes(
            self, num_stages,
            num_devices_per_stage) -> List["PhysicalDeviceMesh"]:
        sub_meshes = []
        if num_devices_per_stage <= self.num_devices_per_host:
            assert self.num_devices_per_host % num_devices_per_stage == 0, \
                "num_devices_per_host ({}) % num_devices_per_stage ({}) != 0"\
                    .format(self.num_devices_per_host, num_devices_per_stage)
            num_clusters_per_host = self.num_devices_per_host // num_devices_per_stage
            num_clusters = self.num_hosts * num_clusters_per_host
            assert num_stages % num_clusters == 0, \
                "num_stages({}) % num_clusters({}) !=0".format(num_stages, num_clusters)
            for mesh_id in range(num_stages):
                cluster_id = mesh_id % num_clusters
                cluster_col = cluster_id % num_clusters_per_host
                cluster_row = cluster_id // num_clusters_per_host
                sub_devices_id = [
                    self.phy_devices_id[cluster_row][cluster_col *
                                                     num_devices_per_stage:(
                                                         (cluster_col + 1) *
                                                         num_devices_per_stage)]
                ]
                sub_meshes.append(
                    PhysicalDeviceMesh(sub_devices_id, self.config,
                                       self.prof_database,
                                       self.shape_consistency_manager,
                                       [self.host_ips[cluster_row]]))
        else:
            assert num_devices_per_stage % self.num_devices_per_host == 0, \
                "num_devices_per_stage ({}) %  num_devices_per_host ({}) != 0"\
                    .format(num_devices_per_stage, self.num_devices_per_host)
            num_host_per_cluster = num_devices_per_stage // self.num_devices_per_host
            assert self.num_hosts % num_host_per_cluster == 0, \
                "num_hosts ({}) % num_host_per_cluster({}) != 0".format(self.num_hosts, num_host_per_cluster)
            num_clusters = self.num_hosts // num_host_per_cluster
            for mesh_id in range(num_stages):
                cluster_id = mesh_id % num_clusters
                cluster_row = cluster_id * num_host_per_cluster
                sub_devices_id = self.phy_devices_id[cluster_row:(
                    cluster_row + num_host_per_cluster)]
                host_ips = self.host_ips[cluster_row:(cluster_row +
                                                      num_host_per_cluster)]
                sub_meshes.append(
                    PhysicalDeviceMesh(sub_devices_id, self.config,
                                       self.prof_database,
                                       self.shape_consistency_manager,
                                       host_ips))
        return sub_meshes

    def _profile_logical_meshes(self, logical_meshes, step):
        for lmesh in logical_meshes:
            lmesh.profile_all_comms_perf(step)

    def as_logical_mesh(self) -> LogicalDeviceMesh:
        alpha = [
            self.cluster_info.inter_node_bw_per_device,
            self.cluster_info.intra_node_bw_per_device
        ]
        beta = [
            self.cluster_info.inter_node_latency,
            self.cluster_info.intra_node_latency
        ]
        sharp = [
            self.cluster_info.inter_node_sharp,
            self.cluster_info.intra_node_sharp
        ]
        return LogicalDeviceMesh(
            self.phy_devices_id.shape,
            self.phy_devices_id.shape,
            self.phy_devices_id,
            self.config,
            alpha,
            beta,
            sharp,
            self.prof_database,
            self.shape_consistency_manager,
            self.host_ips,
        )

    def get_logical_meshes(self):
        logical_meshes = []
        # (1, 2) -> (1, 2)
        # (1, 4) -> (2, 2)
        # (1, 8) -> (2, 4)
        # (1, 16) -> (2, 8), (4, 4)
        # (1, 32) -> (2, 16), (4, 8)
        # (1, 48) -> (2, 24), (3, 16), (4, 12), (6, 8)
        # (1, 64) -> (2, 32), (4, 16), (8, 8)
        # we will traverse logical shape's axis in sharding spec, thus (2, 8) contains (8, 2)
        # we will merge logical shapes' axis, thus (2, 8) contains (1, 16) and (16, 1)
        if self.num_hosts == 1:
            alpha = [self.cluster_info.intra_node_bw_per_device]
            beta = [self.cluster_info.intra_node_latency]
            sharp = [self.cluster_info.intra_node_sharp]
            for i in range(2, self.num_devices_per_host):
                if self.num_devices_per_host % i == 0 and i * i <= self.num_devices_per_host:
                    lmesh_shape = (i, self.num_devices_per_host // i)
                    lmesh_phy_ids = self.phy_devices_id.reshape(lmesh_shape)
                    logical_meshes.append(
                        LogicalDeviceMesh(self.phy_devices_id.shape,
                                          lmesh_shape, lmesh_phy_ids,
                                          self.config, alpha, beta, sharp,
                                          self.prof_database,
                                          self.shape_consistency_manager,
                                          self.host_ips))
        # (8, 1) -> (2, 4)
        # (16, 1) -> (2, 8), (4, 4)
        elif self.num_devices_per_host == 1:
            alpha = [self.cluster_info.inter_node_bw_per_device]
            beta = [self.cluster_info.inter_node_latency]
            sharp = [self.cluster_info.inter_node_sharp]
            for i in range(2, self.num_hosts):
                if self.num_hosts % i == 0 and i * i <= self.num_hosts:
                    lmesh_shape = (i, self.num_hosts // i)
                    lmesh_phy_ids = self.phy_devices_id.reshape(lmesh_shape)
                    logical_meshes.append(
                        LogicalDeviceMesh(self.phy_devices_id.shape,
                                          lmesh_phy_ids, self.config, alpha,
                                          beta, sharp, self.prof_database,
                                          self.shape_consistency_manager,
                                          self.host_ips))
        # (2, 1) -> (2, 1)
        # (2, 8) -> (2, 8)
        # (1, 2) -> (1, 2)
        # (1, 3) -> (1, 3)
        # (1, 5) -> (1, 5)
        if 0 == len(logical_meshes):
            logical_meshes.append(self.as_logical_mesh())
        return logical_meshes

    '''
    we assume we can evenly split the pipeline and deviceMesh
    '''

    def _list_all_sub_meshes(self):
        sub_meshes = []
        for num_devices_per_stage in range(1, self.num_devices_per_host + 1):
            if self.num_devices_per_host % num_devices_per_stage == 0:
                num_stages = self.num_hosts * self.num_devices_per_host // num_devices_per_stage
                sub_meshes.append(
                    self.split_pipeline_meshes(num_stages,
                                               num_devices_per_stage)[0])
        for num_hosts_per_stage in range(2, self.num_hosts + 1):
            if self.num_hosts % num_hosts_per_stage == 0:
                num_stages = self.num_hosts // num_hosts_per_stage
                sub_meshes.append(
                    self.split_pipeline_meshes(
                        num_stages,
                        num_hosts_per_stage * self.num_devices_per_host)[0])
        return sub_meshes

    def list_all_pipeline_configs(self):
        configs = []
        for num_devices_per_stage in range(1, self.num_devices_per_host + 1):
            if self.num_devices_per_host % num_devices_per_stage == 0:
                num_stages = self.num_hosts * self.num_devices_per_host // num_devices_per_stage
                configs.append((num_stages, num_devices_per_stage))
        for num_hosts_per_stage in range(2, self.num_hosts + 1):
            if self.num_hosts % num_hosts_per_stage == 0:
                num_stages = self.num_hosts // num_hosts_per_stage
                configs.append(
                    (num_stages,
                     num_hosts_per_stage * self.num_devices_per_host))
        return configs

    def profile_s_curve(self, step):
        sub_phy_device_meshes = self._list_all_sub_meshes()
        for phy_mesh in sub_phy_device_meshes:
            lmeshes = phy_mesh.get_logical_meshes()
            self._profile_logical_meshes(lmeshes, step)
        if 2 == step:
            self.save_profile_database()

    def profile_alpha_beta(self):
        alpha = [250, 25]
        beta = [100, 100]
        return alpha, beta

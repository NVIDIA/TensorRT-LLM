# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device-local occupancy probes shared by PrimTS attention planners."""

import functools


@functools.cache
def _cluster_limits(device_index: int) -> tuple[int, int]:
    """Return single-CTA occupancy and the largest launchable CTA cluster."""
    import torch
    import cutlass.utils as utils
    from cuda.bindings import driver as cuda_drv

    with torch.cuda.device(device_index):
        # HardwareInfo must first see a live primary context. Do this only on
        # cache misses; new plan shapes reuse the device's occupancy result.
        torch.empty(0, device="cuda")
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        info = utils.HardwareInfo(device_index)
        single_cta = info.get_max_active_clusters(1, stream)
        config = cuda_drv.CUlaunchConfig()
        config.blockDimX, config.blockDimY, config.blockDimZ = 128, 1, 1
        config.gridDimX, config.gridDimY, config.gridDimZ = 1, 1, 1
        config.hStream = stream
        status, limit = cuda_drv.cuOccupancyMaxPotentialClusterSize(info.kernel, config)
        if status != cuda_drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cannot query maximum CUDA cluster size: {status}")
        return int(single_cta), int(limit)


@functools.cache
def _query_max_active_clusters(device_index: int, cluster_size: int) -> int:
    import torch
    import cutlass.utils as utils
    from cuda.bindings import driver as cuda_drv

    single_cta, max_cluster_size = _cluster_limits(device_index)
    if cluster_size == 1:
        return single_cta
    if cluster_size > max_cluster_size:
        return 0
    with torch.cuda.device(device_index):
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        return int(
            utils.HardwareInfo(device_index).get_max_active_clusters(
                cluster_size, stream
            )
        )


def get_max_active_clusters(cluster_size: int) -> int:
    import torch

    return _query_max_active_clusters(torch.cuda.current_device(), cluster_size)

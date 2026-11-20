# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
def _query_max_active_clusters(device_index: int, cluster_size: int) -> int:
    import torch
    import cutlass.utils as utils
    from cuda.bindings import driver as cuda_drv

    with torch.cuda.device(device_index):
        # HardwareInfo must first see a live primary context. Do this only on
        # cache misses; new plan shapes reuse the device's occupancy result.
        torch.empty(0, device="cuda")
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        return int(
            utils.HardwareInfo(device_index).get_max_active_clusters(
                cluster_size, stream
            )
        )


def get_max_active_clusters(cluster_size: int) -> int:
    import torch

    return _query_max_active_clusters(torch.cuda.current_device(), cluster_size)


def is_sm107_compile_target() -> bool:
    """Whether generated code may use resources exclusive to native SM107."""
    from cutlass.base_dsl.arch import Arch
    from cutlass.cutlass_dsl import BaseDSL
    from flashinfer.cute_dsl.availability import is_rubin_cute_dsl_available

    arch = BaseDSL._get_dsl().get_arch_enum()
    return is_rubin_cute_dsl_available() and Arch.sm_107 <= arch <= Arch.sm_107f

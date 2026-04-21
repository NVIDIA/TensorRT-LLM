# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom ops required for implementing tensor parallelism.

This module defines atomic distributed ops - each op uses a specific backend
(torch.distributed or TRT-LLM) without internal dispatch logic.
"""

from typing import List, Optional

import torch

from ....distributed.symm_mem_allgather import SymmetricMemoryAllGather
from ...distributed import common as dist

# Cache SymmetricMemoryAllGather module for CUDA Graph safety
_symm_mem_allgather_cache = {}

# ============================================================================
# PyTorch Distributed Backend Ops (demollm mode)
# ============================================================================


@torch.library.custom_op("auto_deploy::torch_dist_all_gather", mutates_args=(), device_types="cuda")
def torch_dist_all_gather(
    tensor: torch.Tensor, dim: int = 0, sizes: Optional[List[int]] = None
) -> torch.Tensor:
    """All gather using PyTorch distributed backend.

    This op always uses torch.distributed.all_gather and is used in demollm mode.
    """
    tl = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tl, tensor)
    return torch.cat(tl, dim=dim)


@torch_dist_all_gather.register_fake
def torch_dist_all_gather_fake(tensor, dim=0, sizes=None):
    return torch.cat([torch.empty_like(tensor) for _ in range(dist.get_world_size())], dim=dim)


def _get_symm_mem_allgather_torch():
    """Get or create a cached SymmetricMemoryAllGather instance for torch.distributed mode."""
    from .....mapping import Mapping

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cache_key = (rank, world_size)
    if cache_key not in _symm_mem_allgather_cache:
        p_config = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        _symm_mem_allgather_cache[cache_key] = SymmetricMemoryAllGather(
            mapping=p_config, dtype=torch.bfloat16
        )
    return _symm_mem_allgather_cache[cache_key]


@torch.library.custom_op(
    "auto_deploy::symm_mem_all_gather_torch", mutates_args=(), device_types="cuda"
)
def symm_mem_all_gather_torch(
    tensor: torch.Tensor, dim: int = 0, sizes: Optional[List[int]] = None
) -> torch.Tensor:
    """AllGather using symmetric memory for demollm (torch.distributed) mode.

    Falls back to NCCL for unsupported cases.
    """
    if sizes is None:
        ag_module = _get_symm_mem_allgather_torch()
        result = ag_module(tensor, dim=dim)
        if result is not None:
            return result

    # Fallback: standard torch.distributed all_gather
    tl = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tl, tensor)
    return torch.cat(tl, dim=dim)


@symm_mem_all_gather_torch.register_fake
def symm_mem_all_gather_torch_fake(tensor, dim=0, sizes=None):
    return torch.cat([torch.empty_like(tensor) for _ in range(dist.get_world_size())], dim=dim)


@torch.library.custom_op("auto_deploy::torch_dist_all_reduce", mutates_args=(), device_types="cuda")
def torch_dist_all_reduce(t: torch.Tensor, strategy: str) -> torch.Tensor:
    """All_reduce using PyTorch distributed backend. Reduction op is SUM.

    This op always uses torch.distributed.all_reduce and is used in demollm mode.

    NOTE: this op requires an extra memory copy and should ONLY be used for debugging + testing. For
    efficient all_reduce ops one should write/replace it with a fused op.
    """
    t_res = t.clone()
    dist.all_reduce(t_res, op=dist.ReduceOp.SUM)
    return t_res


@torch_dist_all_reduce.register_fake
def torch_dist_all_reduce_fake(tensor, strategy):
    return torch.empty_like(tensor)

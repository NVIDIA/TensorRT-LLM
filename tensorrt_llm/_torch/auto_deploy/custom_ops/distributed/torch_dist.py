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

# SymmetricMemoryAllGather instances keyed on (rank, world_size, workspace_id).
# See trtllm_dist.py for the workspace_id contract.
_symm_mem_allgather_cache = {}

# ============================================================================
# PyTorch Distributed Backend Ops (demollm mode)
# ============================================================================


def _torch_allgather_fallback(tensor, dim, sizes=None):
    """Plain torch.distributed all_gather with concat along *dim*."""
    tl = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tl, tensor)
    return torch.cat(tl, dim=dim)


def _get_symm_mem_allgather_torch(workspace_id: int):
    """Get or create a cached SymmetricMemoryAllGather instance for *workspace_id*."""
    import torch.distributed as torch_dist

    from .....mapping import Mapping

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cache_key = (rank, world_size, workspace_id)
    if cache_key not in _symm_mem_allgather_cache:
        p_config = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        if workspace_id == 0:
            group = None  # constructor will create the default symm-mem group
        else:
            group = torch_dist.new_group(p_config.tp_group)
        _symm_mem_allgather_cache[cache_key] = SymmetricMemoryAllGather(
            mapping=p_config, dtype=torch.bfloat16, group=group
        )
    return _symm_mem_allgather_cache[cache_key]


def _torch_symm_mem_allgather_impl(tensor, dim, sizes, workspace_id):
    """Symm-mem allgather with torch.distributed fallback."""
    if sizes is None:
        ag_module = _get_symm_mem_allgather_torch(workspace_id)
        result = ag_module(tensor, dim=dim)
        if result is not None:
            return result
    return _torch_allgather_fallback(tensor, dim, sizes=sizes)


@torch.library.custom_op("auto_deploy::torch_dist_all_gather", mutates_args=(), device_types="cuda")
def torch_dist_all_gather(
    tensor: torch.Tensor,
    dim: int = 0,
    sizes: Optional[List[int]] = None,
    strategy: str = "AUTO",
    workspace_id: int = 0,
) -> torch.Tensor:
    """AllGather via torch.distributed backend (demollm mode).

    Strategy:
        AUTO     — torch.distributed.all_gather.
        SYMM_MEM — symmetric memory (multimem_all_gather_out) with
                   torch.distributed fallback.

    See trtllm_dist_all_gather for the workspace_id contract.
    """
    if strategy == "SYMM_MEM":
        return _torch_symm_mem_allgather_impl(tensor, dim, sizes, workspace_id)
    return _torch_allgather_fallback(tensor, dim, sizes=sizes)


@torch_dist_all_gather.register_fake
def torch_dist_all_gather_fake(tensor, dim=0, sizes=None, strategy="AUTO", workspace_id=0):
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

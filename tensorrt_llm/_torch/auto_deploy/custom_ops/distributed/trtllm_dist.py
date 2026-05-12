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

"""TRT-LLM distributed operations and fused kernels.

This module defines atomic TRT-LLM-specific ops that use optimized kernels.
The torch fallback variants are defined separately to enable multi-pattern matching.
"""

from typing import List, Optional

import torch

from tensorrt_llm._torch.distributed import AllReduce, allgather
from tensorrt_llm._torch.distributed.symm_mem_allgather import SymmetricMemoryAllGather
from tensorrt_llm._torch.modules.linear import AllReduceFusionOp, AllReduceParams, AllReduceStrategy
from tensorrt_llm.mapping import Mapping

from ...distributed.common import ReduceOp, get_rank_world_size, get_world_size, is_ompi

# Cache AllReduce modules to avoid recreating on every call
# This is critical for CUDA graph compatibility - recreating modules during
# warmup causes hangs due to workspace allocation with CPU synchronization
_allreduce_cache = {}

# SymmetricMemoryAllGather instances keyed on (rank, world_size, workspace_id).
# workspace_id == 0 uses the default TP process group. Higher workspace_ids
# allocate fresh process groups (via dist.new_group), each with its own
# symm_mem workspace buffer; this lets concurrent symm-mem allgathers on
# different streams (e.g. multi-stream MLA) avoid workspace conflicts.
_symm_mem_allgather_cache = {}


def trtllm_allgather(tensor, dim, sizes=None):
    rank, world_size = get_rank_world_size()
    p_config = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
    return allgather(tensor, p_config, dim=dim, sizes=sizes)


def _get_symm_mem_allgather(workspace_id: int):
    """Get or create a cached SymmetricMemoryAllGather instance for *workspace_id*."""
    import torch.distributed as dist

    rank, world_size = get_rank_world_size()
    cache_key = (rank, world_size, workspace_id)
    if cache_key not in _symm_mem_allgather_cache:
        p_config = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        if workspace_id == 0:
            group = None  # use default TP group
        else:
            group = dist.new_group(p_config.tp_group)
        _symm_mem_allgather_cache[cache_key] = SymmetricMemoryAllGather(
            mapping=p_config, dtype=torch.bfloat16, group=group
        )
    return _symm_mem_allgather_cache[cache_key]


def trtllm_symm_mem_allgather_impl(tensor, dim, sizes, workspace_id):
    """Symm-mem allgather with TRT-LLM NCCL fallback."""
    # Uneven per-rank sizes (allgatherv) aren't supported by multimem_all_gather_out; fall back to NCCL.
    if sizes is not None:
        return trtllm_allgather(tensor, dim=dim, sizes=sizes)

    ag_module = _get_symm_mem_allgather(workspace_id)
    result = ag_module(tensor, dim=dim)
    if result is not None:
        return result

    return trtllm_allgather(tensor, dim=dim, sizes=sizes)


def trtllm_allreduce(tensor, op, strategy: str, all_reduce_params=None):
    rank, world_size = get_rank_world_size()
    assert op == ReduceOp.SUM, "TRT-LLM all reduce only supports SUM op."

    # Convert string strategy to enum
    try:
        strategy_enum = getattr(AllReduceStrategy, strategy)
    except AttributeError:
        raise ValueError(
            f"Invalid allreduce strategy: {strategy}. "
            f"Valid options: AUTO, NCCL, ONESHOT, TWOSHOT, MIN_LATENCY, "
            f"LOWPRECISION, UB, MNNVL, NCCL_SYMMETRIC"
        )

    # Cache key includes rank, world_size, dtype, and strategy to handle different configurations
    cache_key = (rank, world_size, tensor.dtype, strategy_enum)
    if cache_key not in _allreduce_cache:
        p_config = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        _allreduce_cache[cache_key] = AllReduce(
            mapping=p_config, strategy=strategy_enum, dtype=tensor.dtype
        )

    torch_op = _allreduce_cache[cache_key]
    return torch_op(tensor, all_reduce_params=all_reduce_params)


# ============================================================================
# TRT-LLM Backend Ops (MPI mode)
# ============================================================================


@torch.library.custom_op(
    "auto_deploy::trtllm_dist_all_gather", mutates_args=(), device_types="cuda"
)
def trtllm_dist_all_gather(
    tensor: torch.Tensor,
    strategy: str,
    dim: int = 0,
    sizes: Optional[List[int]] = None,
    # Picks the symm_mem workspace; use distinct ids for concurrent multi-stream allgathers to avoid buffer races.
    workspace_id: int = 0,
) -> torch.Tensor:
    """AllGather via TRT-LLM optimized backend.

    Strategy (required, no default — callers must pick the strategy explicitly
    from the AD config to avoid silently using a default when emitting this
    op into the graph):
        AUTO     — TRT-LLM NCCL allgather.
        SYMM_MEM — symmetric memory (multimem_all_gather_out) with NCCL fallback.

    workspace_id picks the symm_mem ProcessGroup/workspace and is only
    relevant when strategy == "SYMM_MEM" (ignored otherwise). Use distinct
    workspace_ids for symm-mem allgathers running concurrently on different
    streams to avoid workspace buffer conflicts.
    """
    if strategy == "SYMM_MEM":
        return trtllm_symm_mem_allgather_impl(tensor, dim, sizes, workspace_id)
    return trtllm_allgather(tensor, dim=dim, sizes=sizes)


@trtllm_dist_all_gather.register_fake
def trtllm_dist_all_gather_fake(tensor, strategy, dim=0, sizes=None, workspace_id=0):
    return torch.cat([torch.empty_like(tensor) for _ in range(get_world_size())], dim=dim)


@torch.library.custom_op(
    "auto_deploy::trtllm_dist_all_reduce", mutates_args=(), device_types="cuda"
)
def trtllm_dist_all_reduce(t: torch.Tensor, strategy: str) -> torch.Tensor:
    """All_reduce using TRT-LLM optimized backend. Reduction op is SUM.

    This op always uses TRT-LLM's optimized allreduce and is used in MPI mode.
    """
    return trtllm_allreduce(t, op=ReduceOp.SUM, strategy=strategy)


@trtllm_dist_all_reduce.register_fake
def trtllm_dist_all_reduce_fake(tensor, strategy):
    return torch.empty_like(tensor)


# TRT-LLM fused op (atomic - always uses TRT-LLM backend)
@torch.library.custom_op(
    "dist::trtllm_fused_allreduce_residual_rmsnorm", mutates_args=(), device_types="cuda"
)
def trtllm_fused_allreduce_residual_rmsnorm(
    tensor: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused allreduce + residual + rmsnorm using TRT-LLM optimized kernel.

    This op always uses TRT-LLM's fused kernel and is used in MPI mode.
    """
    all_reduce_params = AllReduceParams(
        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
        bias=None,
        residual=residual,
        norm_weight=norm_weight,
        eps=eps,
    )
    return trtllm_allreduce(
        tensor, ReduceOp.SUM, strategy=strategy, all_reduce_params=all_reduce_params
    )


@trtllm_fused_allreduce_residual_rmsnorm.register_fake
def trtllm_fused_allreduce_residual_rmsnorm_fake(
    tensor: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(tensor), torch.empty_like(tensor)


def is_trtllm_op_available():
    """Check if TRT-LLM ops are available and running with MPI."""
    return is_ompi()

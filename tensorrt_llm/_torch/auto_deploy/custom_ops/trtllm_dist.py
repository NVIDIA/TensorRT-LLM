"""TRT-LLM distributed operations and fused kernels.

This module defines atomic TRT-LLM-specific ops that use optimized kernels.
The torch fallback variants are defined separately to enable multi-pattern matching.
"""

from typing import List, Optional

import torch

# use trtllm distributed ops to improve TP performance if possible
from ....mapping import Mapping
from ...distributed import AllReduce, allgather
from ...modules.linear import AllReduceFusionOp, AllReduceParams, AllReduceStrategy
from ..distributed.common import ReduceOp, get_rank_world_size, get_world_size, is_ompi

# Cache AllReduce modules to avoid recreating on every call
# This is critical for CUDA graph compatibility - recreating modules during
# warmup causes hangs due to workspace allocation with CPU synchronization
_allreduce_cache = {}


def trtllm_allgather(tensor, dim, sizes=None):
    rank, world_size = get_rank_world_size()
    p_config = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
    return allgather(tensor, p_config, dim=dim, sizes=sizes)


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
    tensor: torch.Tensor, dim: int = 0, sizes: Optional[List[int]] = None
) -> torch.Tensor:
    """All gather using TRT-LLM optimized backend.

    This op always uses TRT-LLM's optimized allgather and is used in MPI mode.
    """
    return trtllm_allgather(tensor, dim=dim, sizes=sizes)


@trtllm_dist_all_gather.register_fake
def trtllm_dist_all_gather_fake(tensor, dim=0, sizes=None):
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

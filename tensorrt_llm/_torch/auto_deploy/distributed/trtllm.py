"""TRT-LLM distributed operations and fused kernels.

This module defines atomic TRT-LLM-specific ops that use optimized kernels.
The torch fallback variants are defined separately to enable multi-pattern matching.
"""

import torch

from .common import ReduceOp, get_rank_world_size, is_ompi

# use trtllm distributed ops to improve TP performance if possible
try:
    from ....mapping import Mapping
    from ...distributed import AllReduce, allgather
    from ...modules.linear import AllReduceFusionOp, AllReduceParams, AllReduceStrategy

    # Cache AllReduce modules to avoid recreating on every call
    # This is critical for CUDA graph compatibility - recreating modules during
    # warmup causes hangs due to workspace allocation with CPU synchronization
    _allreduce_cache = {}

    def trtllm_allgather(tensor, dim, sizes=None):
        rank, world_size = get_rank_world_size()
        p_config = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        return allgather(tensor, p_config, dim=dim, sizes=sizes)

    def trtllm_allreduce(tensor, op, all_reduce_params=None):
        rank, world_size = get_rank_world_size()
        assert op == ReduceOp.SUM, "TRT-LLM all reduce only supports SUM op."

        # Cache key includes rank, world_size, and dtype to handle different configurations
        cache_key = (rank, world_size, tensor.dtype)
        if cache_key not in _allreduce_cache:
            p_config = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
            # Use Strategy.AUTO for optimal performance
            _allreduce_cache[cache_key] = AllReduce(
                mapping=p_config, strategy=AllReduceStrategy.NCCL, dtype=tensor.dtype
            )

        torch_op = _allreduce_cache[cache_key]
        return torch_op(tensor, all_reduce_params=all_reduce_params)

    # TRT-LLM fused op (atomic - always uses TRT-LLM backend)
    @torch.library.custom_op(
        "dist::trtllm_fused_allreduce_residual_rmsnorm", mutates_args=(), device_types="cuda"
    )
    def trtllm_fused_allreduce_residual_rmsnorm(
        tensor: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float
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
        return trtllm_allreduce(tensor, ReduceOp.SUM, all_reduce_params=all_reduce_params)

    @trtllm_fused_allreduce_residual_rmsnorm.register_fake
    def trtllm_fused_allreduce_residual_rmsnorm_fake(
        tensor: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(tensor), torch.empty_like(tensor)

    TRTLLM_OP_AVAILABLE = True
except ImportError:

    def trtllm_allgather(tensor, dim, sizes=None):
        raise ImportError("TRT-LLM is not available.")

    def trtllm_allreduce(tensor, op, all_reduce_params=None):
        raise ImportError("TRT-LLM is not available.")

    TRTLLM_OP_AVAILABLE = False


def is_trtllm_op_available():
    """Check if TRT-LLM ops are available and running with MPI."""
    return TRTLLM_OP_AVAILABLE and is_ompi()

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

    # Global allreduce strategy configuration
    # Can be set via set_allreduce_strategy() to override the default AUTO strategy
    _global_allreduce_strategy = AllReduceStrategy.AUTO

    def set_allreduce_strategy(strategy: AllReduceStrategy):
        """Set the global allreduce strategy for distributed operations.

        Args:
            strategy: AllReduceStrategy enum value (AUTO, NCCL, ONESHOT, TWOSHOT, etc.)
        """
        global _global_allreduce_strategy
        _global_allreduce_strategy = strategy
        # Clear cache when strategy changes to force recreation with new strategy
        _allreduce_cache.clear()

    def trtllm_allgather(tensor, dim, sizes=None):
        rank, world_size = get_rank_world_size()
        p_config = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        return allgather(tensor, p_config, dim=dim, sizes=sizes)

    def trtllm_allreduce(tensor, op, all_reduce_params=None):
        rank, world_size = get_rank_world_size()
        assert op == ReduceOp.SUM, "TRT-LLM all reduce only supports SUM op."

        # Cache key includes rank, world_size, dtype, and strategy to handle different configurations
        cache_key = (rank, world_size, tensor.dtype, _global_allreduce_strategy)
        if cache_key not in _allreduce_cache:
            p_config = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
            # Use the configured global strategy
            _allreduce_cache[cache_key] = AllReduce(
                mapping=p_config, strategy=_global_allreduce_strategy, dtype=tensor.dtype
            )

        torch_op = _allreduce_cache[cache_key]
        return torch_op(tensor, all_reduce_params=all_reduce_params)

    @torch.library.custom_op(
        "dist::fused_allreduce_residual_rmsnorm", mutates_args=(), device_types="cuda"
    )
    def fused_allreduce_residual_rmsnorm(
        tensor: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fusing allreduce, residual (add), and hf_rms_norm together."""
        all_reduce_params = AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            bias=None,
            residual=residual,
            norm_weight=norm_weight,
            eps=eps,
        )
        return trtllm_allreduce(tensor, ReduceOp.SUM, all_reduce_params=all_reduce_params)

    @fused_allreduce_residual_rmsnorm.register_fake
    def fused_allreduce_residual_rmsnorm_fake(
        tensor: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(tensor), torch.empty_like(tensor)

    TRTLLM_OP_AVAILABLE = True
except ImportError:

    def set_allreduce_strategy(strategy):
        raise ImportError("TRT-LLM is not available.")

    def trtllm_allgather(tensor, dim, sizes=None):
        raise ImportError("TRT-LLM is not available.")

    def trtllm_allreduce(tensor, op):
        raise ImportError("TRT-LLM is not available.")

    TRTLLM_OP_AVAILABLE = False


def is_trtllm_op_available():
    # TRT-LLM only work with MPI
    return TRTLLM_OP_AVAILABLE and is_ompi()

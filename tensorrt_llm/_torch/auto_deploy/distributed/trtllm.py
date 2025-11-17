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

    @torch.library.custom_op(
        "dist::fused_allreduce_residual_rmsnorm", mutates_args=(), device_types="cuda"
    )
    def fused_allreduce_residual_rmsnorm(
        tensor: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fusing allreduce, residual (add), and hf_rms_norm together.

        When TRT-LLM ops are available (MPI mode), uses the fused kernel.
        Otherwise, falls back to separate operations using torch distributed.
        """
        # Only use TRT-LLM fused op when running with MPI
        if is_trtllm_op_available():
            all_reduce_params = AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                bias=None,
                residual=residual,
                norm_weight=norm_weight,
                eps=eps,
            )
            return trtllm_allreduce(tensor, ReduceOp.SUM, all_reduce_params=all_reduce_params)
        else:
            # Fallback: unfused implementation using torch distributed
            # This is used in demollm mode without MPI
            from .common import all_reduce as torch_all_reduce

            # 1. All-reduce the tensor
            tensor_reduced = tensor.clone()
            torch_all_reduce(tensor_reduced, op=ReduceOp.SUM)

            # 2. Add residual
            tensor_with_residual = tensor_reduced + residual

            # 3. Apply RMSNorm using PyTorch's built-in function
            norm_out = torch.nn.functional.rms_norm(
                tensor_with_residual,
                normalized_shape=(tensor_with_residual.size(-1),),
                weight=norm_weight,
                eps=eps,
            )

            return norm_out, tensor_with_residual

    @fused_allreduce_residual_rmsnorm.register_fake
    def fused_allreduce_residual_rmsnorm_fake(
        tensor: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(tensor), torch.empty_like(tensor)

    TRTLLM_OP_AVAILABLE = True
except ImportError:

    def trtllm_allgather(tensor, dim, sizes=None):
        raise ImportError("TRT-LLM is not available.")

    def trtllm_allreduce(tensor, op):
        raise ImportError("TRT-LLM is not available.")

    TRTLLM_OP_AVAILABLE = False


def is_trtllm_op_available():
    # TRT-LLM only work with MPI
    return TRTLLM_OP_AVAILABLE and is_ompi()

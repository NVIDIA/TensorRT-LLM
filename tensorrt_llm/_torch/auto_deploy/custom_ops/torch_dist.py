"""Custom ops required for implementing tensor parallelism.

This module defines atomic distributed ops - each op uses a specific backend
(torch.distributed or TRT-LLM) without internal dispatch logic.
"""

from typing import List, Optional

import torch

from ..distributed import common as dist

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
    # Get the process group (use default if not set)
    group = dist.DistGroup.get()
    dist.all_gather(tl, tensor, group=group)
    return torch.cat(tl, dim=dim)


@torch_dist_all_gather.register_fake
def torch_dist_all_gather_fake(tensor, dim=0, sizes=None):
    return torch.cat([torch.empty_like(tensor) for _ in range(dist.get_world_size())], dim=dim)


@torch.library.custom_op("auto_deploy::torch_dist_all_reduce", mutates_args=(), device_types="cuda")
def torch_dist_all_reduce(t: torch.Tensor, strategy: str) -> torch.Tensor:
    """All_reduce using PyTorch distributed backend. Reduction op is SUM.

    This op always uses torch.distributed.all_reduce and is used in demollm mode.

    NOTE: this op requires an extra memory copy and should ONLY be used for debugging + testing. For
    efficient all_reduce ops one should write/replace it with a fused op.
    """
    t_res = t.clone()
    # Get the process group (use default if not set)
    group = dist.DistGroup.get()
    dist.all_reduce(t_res, op=dist.ReduceOp.SUM, group=group)
    return t_res


@torch_dist_all_reduce.register_fake
def torch_dist_all_reduce_fake(tensor, strategy):
    return torch.empty_like(tensor)


@torch.library.custom_op(
    "auto_deploy::torch_dist_reduce_scatter", mutates_args=(), device_types="cuda"
)
def torch_dist_reduce_scatter(
    tensor: torch.Tensor, dim: int = 0, sizes: Optional[List[int]] = None
) -> torch.Tensor:
    """Reduce-scatter using PyTorch distributed backend.

    This op performs a reduce-scatter: reduces across ranks and scatters the result.
    Each rank receives 1/world_size of the reduced tensor.

    Args:
        tensor: Input tensor to reduce and scatter
        dim: Dimension along which to split the result (default: 0)
        sizes: Optional per-rank sizes. If None, splits evenly.

    Returns:
        Tensor of shape tensor.shape with dim reduced by world_size (or according to sizes)
    """
    import torch.distributed as dist_backend

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Get the process group (use default if not set)
    group = dist.DistGroup.get()

    if sizes is None:
        # Equal split - use optimized reduce_scatter_tensor
        assert tensor.shape[dim] % world_size == 0, (
            f"Tensor dim {dim} size {tensor.shape[dim]} must be divisible by world_size {world_size}"
        )
        output_size = tensor.shape[dim] // world_size
        output_shape = list(tensor.shape)
        output_shape[dim] = output_size
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)

        # Use reduce_scatter_tensor for equal splits
        dist_backend.reduce_scatter_tensor(
            output, tensor, op=dist_backend.ReduceOp.SUM, group=group
        )
    else:
        # Variable sizes - need to do reduce + slice
        # First reduce (all_reduce)
        reduced = tensor.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
        # Then slice out this rank's portion
        start = sum(sizes[:rank])
        output = reduced.narrow(dim, start, sizes[rank]).contiguous()

    return output


@torch_dist_reduce_scatter.register_fake
def torch_dist_reduce_scatter_fake(tensor, dim=0, sizes=None):
    world_size = dist.get_world_size()
    if sizes is None:
        output_size = tensor.shape[dim] // world_size
    else:
        rank = dist.get_rank()
        output_size = sizes[rank]
    output_shape = list(tensor.shape)
    output_shape[dim] = output_size
    return torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)

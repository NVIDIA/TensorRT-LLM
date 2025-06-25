"""Custom ops required for implementing tensor parallelism."""

from typing import List, Optional

import torch

from ..distributed import common as dist
from ..distributed import trtllm as trtllm_dist


@torch.library.custom_op("auto_deploy::torch_dist_all_gather", mutates_args=(), device_types="cuda")
def all_gather(
    tensor: torch.Tensor, dim: int = 0, sizes: Optional[List[int]] = None
) -> torch.Tensor:
    """All gather followed by concat in dim = 0. This is the default nccl behavior."""
    if trtllm_dist.is_trtllm_op_available():
        return trtllm_dist.trtllm_allgather(tensor, dim=dim, sizes=sizes)
    tl = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tl, tensor)
    return torch.cat(tl, dim=dim)


@all_gather.register_fake
def all_gather_fake(tensor, dim=0):
    return torch.cat([torch.empty_like(tensor) for _ in range(dist.get_world_size())], dim=dim)


@torch.library.custom_op("auto_deploy::torch_dist_all_reduce", mutates_args=(), device_types="cuda")
def all_reduce(t: torch.Tensor) -> torch.Tensor:
    """All_reduce across the ranks. Reduction op is SUM.

    NOTE: this op requires an extra memory copy and should ONLY be used for debugging + testing. For
    efficient all_reduce ops one should write/replace it with a fused op.
    """
    if trtllm_dist.is_trtllm_op_available():
        return trtllm_dist.trtllm_allreduce(t, op=dist.ReduceOp.SUM)
    t_res = t.clone()
    dist.all_reduce(t_res, op=dist.ReduceOp.SUM)
    return t_res


@all_reduce.register_fake
def all_reduce_fake(tensor):
    return torch.empty_like(tensor)

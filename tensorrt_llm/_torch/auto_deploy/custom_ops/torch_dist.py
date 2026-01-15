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
    dist.all_gather(tl, tensor)
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
    dist.all_reduce(t_res, op=dist.ReduceOp.SUM)
    return t_res


@torch_dist_all_reduce.register_fake
def torch_dist_all_reduce_fake(tensor, strategy):
    return torch.empty_like(tensor)

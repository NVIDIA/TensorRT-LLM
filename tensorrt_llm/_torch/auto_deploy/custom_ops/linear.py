"""Custom ops for linear layers."""

from typing import Optional

import torch

from ..distributed import common as dist
from ..distributed import trtllm as trtllm_dist


@torch.library.custom_op("auto_deploy::torch_linear_simple", mutates_args=())
def simple(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """A wrapper for the linear functional to control how it is exposed.

    By default F.linear (used in linear layers) will be represented as a call to
    torch.ops.aten.linear.default wrapped with two view ops to flatten/unflatten multiple batch
    dimensions into one batch dimension.

    This wrapper avoids exposing this view op during the export graph.
    """
    return torch.ops.aten.linear(input, weight, bias)


@simple.register_fake
def simple_fake(input, weight, bias):
    """Fake implementation of simple_linear."""
    # return torch.empty(
    # input.shape[:-1] + (weight.shape[-1],), dtype=input.dtype, device=input.device
    # )
    return torch.ops.aten.linear(input, weight, bias)


@torch.library.custom_op(
    "auto_deploy::trtllm_dist_fused_linear_all_reduce", mutates_args=(), device_types="cuda"
)
def fused_linear_all_reduce(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """Fused linear followed by all_reduce on the output."""
    output = torch.ops.aten.linear(input, weight, bias)
    if trtllm_dist.is_trtllm_op_available():
        return trtllm_dist.trtllm_allreduce(output, op=dist.ReduceOp.SUM)
    dist.all_reduce(output, op=dist.ReduceOp.SUM)
    return output


@fused_linear_all_reduce.register_fake
def fused_linear_all_reduce_fake(input, weight, bias):
    return torch.ops.aten.linear(input, weight, bias)

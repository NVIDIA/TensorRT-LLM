"""Tests for basic fusion of the allreduce, residual, and rmsnorm."""

import pytest
import torch
from _dist_test_utils import get_device_counts

from tensorrt_llm._torch.auto_deploy.distributed import common as dist
from tensorrt_llm._torch.auto_deploy.distributed.trtllm import is_trtllm_op_available
from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export, torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transformations.library.collectives import (
    fuse_allreduce_residual_rmsnorm,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm.llmapi.mpi_session import MpiPoolSession


class RMSNorm(torch.nn.Module):
    """Implementation of LlamaRMSNorm."""

    def __init__(self, hidden_size, eps=1e-6, dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size).to(dtype).cuda())
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class AllreduceResidualNorm(torch.nn.Module):
    def __init__(self, hidden_size, dtype):
        super().__init__()
        self.norm = RMSNorm(hidden_size, 1e-5, dtype)

    def forward(self, x, residual):
        x = torch.ops.dist.all_reduce(x)
        y = x + residual
        normed = self.norm(y)
        return normed, y


def _test_allreduce_fusion(port: int):
    if not is_trtllm_op_available():
        pytest.skip("Require trtllm ops to run test_allreduce_fusion.")

    _, _ = dist.initialize_or_skip(port=port)

    # Testing tensors
    dtype = torch.float16
    x = torch.randn(16, 16).to(dtype).cuda()
    residual = torch.randn(16, 16).to(dtype).cuda()

    # Trace the original model
    model = AllreduceResidualNorm(16, dtype)
    args = (
        x,
        residual,
    )
    gm = torch_export_to_gm(model, args=args, clone=True)
    # Run the original
    original_outputs, residual_original = gm(x, residual)

    # Fuse ops
    gm_fused = fuse_allreduce_residual_rmsnorm(gm)

    # Run the fused graph
    fused_outputs, residual_fused = gm_fused(x, residual)

    # Check if fused node in the graph
    has_fused_node = False
    for node in gm_fused.graph.nodes:
        if is_op(node, torch.ops.dist.fused_allreduce_residual_rmsnorm):
            has_fused_node = True
    assert has_fused_node, "Fused node not found."

    # Verify outputs are consistent
    assert torch.allclose(residual_original, residual_fused, atol=1e-5), (
        "Outputs differ between original and fused models."
    )
    assert torch.allclose(original_outputs, fused_outputs, atol=1e-5), (
        "Outputs differ between original and fused models."
    )

    # check if we can still export the model as expected
    torch_export(gm_fused, args=args)
    torch_export_to_gm(gm_fused, args=args)


@pytest.mark.parametrize("device_count", get_device_counts())
def test_allreduce_fusion(device_count):
    if device_count <= 1:
        pytest.skip("Require multi GPUs to run test_allreduce_fusion.")
    port = dist.get_free_port()

    n_workers = device_count
    mpi_pool = MpiPoolSession(n_workers=n_workers)
    mpi_pool.submit_sync(_test_allreduce_fusion, port=port)

from functools import partial

import pytest
import torch
from _graph_test_helpers import run_test
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.custom_ops.rms_norm import *  # noqa
from tensorrt_llm._torch.auto_deploy.transformations.library.rms_norm import fuse_rmsnorm
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, device="cuda"))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class TestModel(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.linear1 = torch.nn.Linear(1024, 1024, device="cuda", dtype=torch.float16)
        self.rms_norm = RMSNorm(1024, eps).to(torch.float16)
        self.linear2 = torch.nn.Linear(1024, 1024, device="cuda", dtype=torch.float16)

    def forward(self, x):
        x = self.linear1(x)
        x = self.rms_norm(x)
        x = self.linear2(x)
        return x


@pytest.mark.parametrize("eps", [1e-2, 1e-6])
@pytest.mark.parametrize(
    "variant, op",
    [
        ("flashinfer", torch.ops.auto_deploy.flashinfer_rms_norm),
        ("triton", torch.ops.auto_deploy.triton_rms_norm),
        ("torch", torch.ops.auto_deploy.torch_rmsnorm),
    ],
)
def test_rmsnorm_fusion(eps, variant, op):
    def checker(gm):
        return any(is_op(n, op) for n in gm.graph.nodes)

    model = TestModel(eps)
    gm_transformed = run_test(
        model,
        torch.randn(2, 1024, device="cuda", dtype=torch.float16),
        partial(fuse_rmsnorm, backend=variant),
        checker,
        lambda num_p_og: num_p_og,
        dynamic_shapes={0: Dim("batch_size", max=8)},
    )
    print(gm_transformed.graph)
    new_input = torch.randn(4, 1024, device="cuda", dtype=torch.float16)
    y_transformed = gm_transformed(new_input)
    y_model = model(new_input)
    torch.testing.assert_close(y_transformed, y_model, atol=1e-3, rtol=1e-3)

import pytest
import torch
from _graph_test_helpers import run_test_transformed_gm
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.custom_ops.l2norm import *  # noqa
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class L2Norm(torch.nn.Module):
    """L2 normalization module that normalizes along the last dimension."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        sum_sq = (x * x).sum(dim=-1, keepdim=True)
        x = x * torch.rsqrt(sum_sq + self.eps)
        return x.to(input_dtype)


class L2NormNoCast(torch.nn.Module):
    """L2 normalization module without dtype casting (for float32 inputs)."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        sum_sq = (x * x).sum(dim=-1, keepdim=True)
        return x * torch.rsqrt(sum_sq + self.eps)


class TestModel(torch.nn.Module):
    def __init__(self, eps: float = 1e-6, use_no_cast: bool = False):
        super().__init__()
        self.linear1 = torch.nn.Linear(1024, 1024, device="cuda", dtype=torch.float16)
        if use_no_cast:
            self.l2_norm = L2NormNoCast(eps)
        else:
            self.l2_norm = L2Norm(eps)
        self.linear2 = torch.nn.Linear(1024, 1024, device="cuda", dtype=torch.float16)

    def forward(self, x):
        x = self.linear1(x)
        x = self.l2_norm(x)
        x = self.linear2(x)
        return x


def _run_test(model, op, variant):
    def checker(gm):
        return any(is_op(n, op) for n in gm.graph.nodes)

    x = torch.randn(2, 1024, device="cuda", dtype=torch.float16)
    dynamic_shapes = {0: Dim.DYNAMIC}
    gm = torch_export_to_gm(model, args=(x,), dynamic_shapes=(dynamic_shapes,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "match_l2norm_pattern": {
                "stage": "pattern_matcher",
            },
            "fuse_l2norm": {
                "stage": "post_load_fusion",
                "backend": variant,
            },
        },
    )(None, gm)

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        checker,
        lambda num_p_og: num_p_og,
        dynamic_shapes=dynamic_shapes,
    )

    new_input = torch.randn(4, 1024, device="cuda", dtype=torch.float16)
    y_transformed = gm_transformed(new_input)
    y_model = model(new_input)
    torch.testing.assert_close(y_transformed, y_model, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("eps", [1e-2, 1e-6])
@pytest.mark.parametrize(
    "variant, op",
    [
        ("fla", torch.ops.auto_deploy.fla_l2norm.default),
        ("torch", torch.ops.auto_deploy.torch_l2norm.default),
    ],
)
def test_l2norm_fusion(eps, variant, op):
    model = TestModel(eps)
    _run_test(model, op, variant)

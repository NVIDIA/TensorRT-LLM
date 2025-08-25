"""Tests for basic fusion of the collective."""

from functools import partial
from typing import Type

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _dist_test_utils import get_device_counts
from _graph_test_helpers import run_test_transformed_gm
from _torch_test_utils import fp8_compatible

import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
from tensorrt_llm._torch.auto_deploy.custom_ops.quant import FP8Linear
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class MLPAllReduce(nn.Module):
    def __init__(self, in_features, out_features, bias, cls):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear1 = cls(in_features, 4 * in_features, bias=bias)
        self.linear2 = cls(4 * in_features, out_features, bias=bias)

    def forward(self, x):
        y = F.relu(torch.ops.auto_deploy.torch_dist_all_reduce(self.linear1(x)))
        return torch.ops.auto_deploy.torch_dist_all_reduce(self.linear2(y))


def _run_job(
    model_cls: Type[nn.Module],
    linear_cls: Type[nn.Module],
    fused_op_expected: str,
    rank: int,
    world_size: int,
) -> None:
    # init model and input
    batch_size = 4
    num_features = 16

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)

    model = model_cls(num_features, num_features, bias=True, cls=linear_cls).to(device="cuda")
    x = torch.randn(batch_size, num_features, device="cuda")

    torch.set_default_dtype(default_dtype)

    op_expected = torch.ops
    for op_name in fused_op_expected.split("."):
        op_expected = getattr(op_expected, op_name)

    def _get_expected_num_params(num_p_og: int) -> int:
        return num_p_og

    def check_transformed_graph(gm):
        return any(is_op(n, op_expected) for n in gm.graph.nodes) and not any(
            is_op(n, torch.ops.auto_deploy.torch_dist_all_reduce) for n in gm.graph.nodes
        )

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_collectives": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    # now run the test
    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        check_transformed_graph=check_transformed_graph,
        _get_expected_num_params=_get_expected_num_params,
        test_load_hook=False,
    )


@pytest.mark.parametrize("device_count", get_device_counts())
@pytest.mark.parametrize(
    "linear_cls, dist_op_expected",
    (
        (nn.Linear, "auto_deploy.trtllm_dist_fused_linear_all_reduce"),
        pytest.param(
            FP8Linear,
            "auto_deploy.torch_quant_fused_fp8_linear_all_reduce",
            marks=pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support"),
        ),
    ),
)
@pytest.mark.parametrize("model_cls", (MLPAllReduce,))
def test_collective_fusion(
    model_cls: Type[nn.Module],
    linear_cls: Type[nn.Module],
    dist_op_expected: str,
    device_count: int,
):
    dist_common.spawn_multiprocess_job(
        job=partial(_run_job, model_cls, linear_cls, dist_op_expected),
        size=device_count,
    )

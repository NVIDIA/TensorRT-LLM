"""
Tests for basic graph sharding.
"""

import pytest
import torch
from _graph_test_helpers import run_test
from _model_test_utils import MLP
from _torch_test_utils import fp4_compatible, fp8_compatible

from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export, torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transformations.library import quantize
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


def check_quantized(gm):
    op_expected = {torch.ops.quant.fp8_linear, torch.ops.quant.fp4_linear}
    return any(is_op(n, op_expected) for n in gm.graph.nodes)


@pytest.mark.parametrize(
    "quantization,atol,rtol",
    [
        pytest.param(
            "FP8",
            0.05,
            0.01,
            marks=pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support"),
        ),
        pytest.param(
            "NVFP4",
            0.16,
            0.016,
            marks=pytest.mark.skipif(not fp4_compatible(), reason="Requires fp4 support"),
        ),
    ],
)
def test_quantization(quantization, atol, rtol):
    model = MLP(32, 64, 32).to(torch.float16).to("cuda")
    x = torch.randn(3, 32, dtype=torch.float16).to("cuda")

    gm_transformed = run_test(
        model,
        x,
        quantize,
        check_quantized,
        lambda num_p_og: num_p_og,
        atol,
        rtol,
        True,  # test_load_hook
        False,  # strict_loading
        quantization,
    )

    # check there's quantization error during transformation
    assert not torch.allclose(model(x), gm_transformed(x))
    # check if we can still export the model as expected
    torch_export(gm_transformed, args=(x,))
    torch_export_to_gm(gm_transformed, args=(x,))

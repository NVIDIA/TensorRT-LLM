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
    op_expected = {
        torch.ops.auto_deploy.torch_quant_fp8_linear,
        torch.ops.auto_deploy.torch_quant_fp4_linear,
    }
    return any(is_op(n, op_expected) for n in gm.graph.nodes)


@pytest.mark.parametrize(
    "quant_config,atol,rtol,num_p_og",
    [
        pytest.param(
            {"quant_algo": "FP8"},
            0.05,
            0.01,
            lambda num_p_og: num_p_og,
            marks=pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support"),
        ),
        pytest.param(
            {"quant_algo": "NVFP4"},
            0.16,
            0.016,
            lambda num_p_og: num_p_og // 2,
            marks=pytest.mark.skipif(not fp4_compatible(), reason="Requires fp4 support"),
        ),
    ],
)
def test_quantization(quant_config, atol, rtol, num_p_og):
    pytest.skip("https://nvbugspro.nvidia.com/bug/5170222")
    model = MLP(32, 64, 32).to(torch.float16).to("cuda")
    x = torch.randn(3, 32, dtype=torch.float16).to("cuda")

    # register input_scale that is required by NVFP4 op
    if quant_config.get("quant_algo") == "NVFP4":
        model.linear1.register_buffer(
            "input_scale", torch.tensor([1.0], device=model.linear1.weight.device)
        )
        model.linear2.register_buffer(
            "input_scale", torch.tensor([1.0], device=model.linear2.weight.device)
        )

    gm_transformed = run_test(
        model,
        x,
        quantize,
        check_quantized,
        num_p_og,
        atol,
        rtol,
        True,  # test_load_hook
        False,  # strict_loading
        None,  # dynamic_shapes
        None,  # check_num_matches
        quant_config,
    )

    # check there's quantization error during transformation
    assert not torch.allclose(model(x), gm_transformed(x))
    # check if we can still export the model as expected
    torch_export(gm_transformed, args=(x,))
    torch_export_to_gm(gm_transformed, args=(x,))

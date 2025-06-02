"""
Tests for basic graph sharding.
"""

import pytest
import torch
from _graph_test_helpers import run_test
from _model_test_utils import MLP, BMMDynamicModel, BMMModel
from _torch_test_utils import fp4_compatible, fp8_compatible

from tensorrt_llm._torch.auto_deploy.custom_ops.quant import QUANT_OPS
from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export, torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transformations.library import quantize
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp8_scale


def check_quantized(gm):
    return any(is_op(n, QUANT_OPS) for n in gm.graph.nodes)


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
        quant_config,
    )

    # check there's quantization error during transformation
    assert not torch.allclose(model(x), gm_transformed(x))
    # check if we can still export the model as expected
    torch_export(gm_transformed, args=(x,))
    torch_export_to_gm(gm_transformed, args=(x,))


@pytest.mark.parametrize(
    "quant_config,atol,rtol,num_p_og,model_class",
    [
        pytest.param(
            {"quant_algo": "FP8"},
            5e-1,
            5e-1,
            lambda num_p_og: num_p_og,
            BMMModel,
            marks=pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support"),
        ),
        pytest.param(
            {"quant_algo": "FP8"},
            5e-1,
            5e-1,
            lambda num_p_og: num_p_og,
            BMMDynamicModel,
            marks=pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support"),
        ),
    ],
)
def test_bmm_quantization(quant_config, atol, rtol, num_p_og, model_class):
    batch_size, seq_len, hidden_dim, num_experts = 2, 2, 16, 1

    # Create model based on class
    if model_class == BMMModel:
        model = model_class(hidden_dim, batch_size, num_experts).to(torch.float16).to("cuda")
    else:  # BMMDynamicModel
        model = model_class(hidden_dim, batch_size).to(torch.float16).to("cuda")

    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16).to("cuda")

    # Register fp8 scales based on model type
    if quant_config["quant_algo"] == "FP8":
        if model_class == BMMModel:
            # Parameter case - register scales in the expert module
            model.experts[0].register_buffer("weight1_input_scale", fp8_scale(x))
            model.experts[0].register_buffer(
                "weight1_weight_scale", fp8_scale(model.experts[0].weight1)
            )
        else:  # BMMDynamicModel
            # Dynamic case - register scales in the root model
            dummy_weight_shape = (batch_size, hidden_dim, hidden_dim)
            dummy_weight = torch.randn(dummy_weight_shape, dtype=torch.float16, device="cuda")

            model.register_buffer("bmm_dynamic_input_scale", fp8_scale(x))
            model.register_buffer("bmm_dynamic_weight_scale", fp8_scale(dummy_weight))

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
        quant_config,
    )

    # check there's quantization error during transformation
    assert not torch.allclose(model(x), gm_transformed(x))
    # check if we can still export the model as expected
    torch_export(gm_transformed, args=(x,))
    torch_export_to_gm(gm_transformed, args=(x,))

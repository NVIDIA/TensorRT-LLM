"""
Tests for basic graph sharding.
"""

from typing import List

import pytest
import torch
import torch.nn as nn
from _graph_test_helpers import run_test_transformed_gm
from _model_test_utils import MLP, BMMDynamicModel, BMMModel
from _torch_test_utils import fp4_compatible, fp8_compatible

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.factory import (
    FullModelExportInfo,
    ModelFactory,
    SubModuleExportInfo,
)
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp8_scale, pack_int4_in_uint8


class DummyFactory(ModelFactory):
    """Dummy factory to pass quant_config for testing."""

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def _build_model(self, device: str):
        return

    def _load_checkpoint(self, model, device):
        return

    def get_quant_config(self):
        return self.quant_config

    def get_export_infos(self, model: nn.Module) -> List[SubModuleExportInfo]:
        return [FullModelExportInfo()]


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
        QUANT_OP = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear
    elif quant_config.get("quant_algo") == "FP8":
        QUANT_OP = torch.ops.auto_deploy.torch_fake_quant_fp8_linear
    # set up sequence+cache objects
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        DummyFactory(quant_config),
        {
            "quantize_fp8_linear_from_config": {
                "stage": "pattern_matcher",
            },
            "quantize_nvfp4_linear_from_config": {
                "stage": "pattern_matcher",
            },
        },
    )(None, gm)
    gm_transformed.to("cuda")

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        lambda gm: any(is_op(n, QUANT_OP) for n in gm.graph.nodes),
        num_p_og,
        atol,
        rtol,
        True,  # test_load_hook
        False,  # strict_loading
        None,  # dynamic_shapes
        None,  # check_num_matches
        False,  # skip_output_assert
        quant_config,
    )

    # check there's quantization error during transformation
    assert not torch.allclose(model(x), gm_transformed(x))
    # check if we can still export the model as expected
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

    # set up sequence+cache objects
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        DummyFactory(quant_config),
        {
            "quantize_fp8_bmm_from_config": {
                "stage": "pattern_matcher",
            },
        },
    )(None, gm)
    gm_transformed.to("cuda")
    QUANT_OP = torch.ops.auto_deploy.torch_quant_fp8_bmm

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        lambda gm: any(is_op(n, QUANT_OP) for n in gm.graph.nodes),
        num_p_og,
        atol,
        rtol,
        True,  # test_load_hook
        False,  # strict_loading
        None,  # dynamic_shapes
        None,  # check_num_matches
        False,  # skip_output_assert
        quant_config,
    )

    # check there's quantization error during transformation
    assert not torch.allclose(model(x), gm_transformed(x))
    # check if we can still export the model as expected
    torch_export_to_gm(gm_transformed, args=(x,))


def _per_block_amax(W: torch.Tensor, block: int = 128) -> torch.Tensor:
    N, K = W.shape
    return W.abs().view(N, K // block, block).amax(dim=-1).to(torch.float32)


def test_int4awq_transform_graph_and_load_hook():
    """INT4 AWQ transform with FP model's state_dict rewritten via hook to packed+scales."""
    device = "cuda"
    torch.manual_seed(0)
    quant_config = {"quant_algo": "W4A16_AWQ"}
    BLOCK = 128
    QUANT_OP = torch.ops.auto_deploy.torch_fake_quant_int4_linear

    # FP model (K divisible by 128, out_dims even)
    model = MLP(256, 128, 256).to(torch.float16).to(device)
    x = torch.randn(3, 256, dtype=torch.float16, device=device)

    def int4awq_state_dict_hook(module: nn.Module, state_dict: dict, prefix: str, local_meta: dict):
        for name, m in module.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            key_w = f"{prefix}{name}.weight"
            if key_w not in state_dict:
                continue
            W = state_dict[key_w].detach().to(torch.float32).to(device)  # (N, K) fp
            N, K = W.shape
            assert N % 2 == 0 and K % BLOCK == 0
            amax = _per_block_amax(W, BLOCK)  # (N, K//128)
            weights_scaling_factor = (amax / 7.0).to(torch.float32)
            W_packed = pack_int4_in_uint8(W, weights_scaling_factor)  # (N//2, K) uint8
            state_dict[key_w] = W_packed.to(torch.uint8)
            state_dict[f"{prefix}{name}.pre_quant_scale"] = torch.ones(
                K, dtype=torch.float32, device=W.device
            )
            state_dict[f"{prefix}{name}.weight_scale"] = weights_scaling_factor

    model._register_state_dict_hook(int4awq_state_dict_hook)

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        DummyFactory(quant_config),
        {"quantize_int4_linear_from_config": {"stage": "pattern_matcher"}},
    )(None, gm).to(device)

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        lambda gm_: any(is_op(n, QUANT_OP) for n in gm_.graph.nodes),
        lambda num_p_og: num_p_og // 2,  # stored params halved by packing
        0.5,  # atol
        0.5,  # rtol
        True,  # test_load_hook
        False,  # strict_loading
        None,  # dynamic_shapes
        None,  # check_num_matches
        False,  # skip_output_assert
        quant_config,
    )

    # Still exportable
    torch_export_to_gm(gm_transformed, args=(x,))

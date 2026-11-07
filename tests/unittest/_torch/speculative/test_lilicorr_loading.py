# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_lilicorr import LiLiCorrForCausalLM, _load_linear
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 loading requires CUDA")
def test_fp8_projection_loads_real_weights_and_scales() -> None:
    weight = torch.linspace(-1, 1, 64 * 32, device="cuda").reshape(32, 64)
    weights = {
        "weight": weight.to(torch.float8_e4m3fn),
        "weight_scale": torch.tensor(0.25, device="cuda"),
        "input_scale": torch.tensor(0.5, device="cuda"),
    }
    layer = _load_linear(
        weights,
        64,
        32,
        torch.bfloat16,
        bias=False,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
        device=torch.device("cuda"),
    )
    assert isinstance(layer, Linear)
    assert layer.weight.shape == (32, 64)
    torch.testing.assert_close(layer.weight.float(), weights["weight"].float())
    torch.testing.assert_close(layer.weight_scale, weights["weight_scale"])
    torch.testing.assert_close(layer.input_scale, weights["input_scale"])
    torch.testing.assert_close(layer.inv_input_scale, 1 / weights["input_scale"])
    inputs = torch.linspace(-1, 1, 5 * 64, device="cuda", dtype=torch.bfloat16).reshape(5, 64)
    reference_input = (inputs.float() / weights["input_scale"]).to(torch.float8_e4m3fn).float()
    reference_input *= weights["input_scale"]
    reference_weight = weights["weight"].float() * weights["weight_scale"]
    layer.post_load_weights()
    with torch.inference_mode():
        output = layer(inputs)
    torch.testing.assert_close(
        output.float(), reference_input @ reference_weight.T, rtol=2e-2, atol=2e-2
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="NVFP4 projection GEMMs require SM100 or SM103",
)
@pytest.mark.parametrize(
    "algo, group_size",
    [(QuantAlgo.NVFP4, 16), (QuantAlgo.W4A16_NVFP4, 16), (QuantAlgo.W4A16_NVFP4, 32)],
)
@pytest.mark.parametrize("projection", ["fc", "head"])
def test_nvfp4_projection_forward_matches_dequantized_checkpoint(
    algo: QuantAlgo, group_size: int, projection: str
) -> None:
    torch.manual_seed(0)
    in_features, out_features = 256, 128
    packed = torch.randint(
        0, 256, (out_features, in_features // 2), device="cuda", dtype=torch.uint8
    )
    scale = torch.tensor([0.5, 1.0, 2.0, 4.0], device="cuda").repeat(
        out_features, in_features // group_size // 4
    )
    weights = {
        "weight": packed,
        "weight_scale": scale.to(torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor(0.125, device="cuda"),
    }
    if algo == QuantAlgo.NVFP4:
        weights["input_scale"] = torch.tensor(0.25, device="cuda")
    quant_config = QuantConfig(quant_algo=algo, group_size=group_size)
    if projection == "fc":
        model = LiLiCorrForCausalLM.__new__(LiLiCorrForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=out_features, torch_dtype=torch.bfloat16)
        model.model_config = ModelConfig(quant_config=quant_config)
        model.model = nn.Module()
        model.model.norm = nn.LayerNorm(out_features, device="cuda")
        checkpoint = {"fc." + name: value for name, value in weights.items()}
        model._load_target_projection(checkpoint)
        assert not checkpoint
        layer = model.fc
    else:
        weights["bias"] = torch.linspace(
            -0.5, 0.5, out_features, device="cuda", dtype=torch.bfloat16
        )
        layer = _load_linear(
            weights,
            in_features,
            out_features,
            torch.bfloat16,
            bias=True,
            quant_config=quant_config,
            device=torch.device("cuda"),
        )
    assert layer.in_features == in_features
    torch.testing.assert_close(layer.weight_scale_2.reshape(()), weights["weight_scale_2"])
    if algo == QuantAlgo.NVFP4:
        torch.testing.assert_close(layer.input_scale.reshape(()), 1 / weights["input_scale"])
        torch.testing.assert_close(
            layer.alpha.reshape(()), weights["input_scale"] * weights["weight_scale_2"]
        )
    else:
        assert layer.input_scale is None

    codebook = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long().flatten(-2)
    reference_weight = codebook[codes] * scale.repeat_interleave(group_size, dim=-1)
    reference_weight *= weights["weight_scale_2"]
    # Exact E2M1 inputs isolate loading errors from activation quantization loss.
    inputs = codebook[(torch.arange(5 * in_features, device="cuda") % 16).reshape(5, in_features)]
    inputs = (inputs * 0.25).to(torch.bfloat16)
    expected = inputs.float() @ reference_weight.T
    if projection == "head":
        expected += weights["bias"].float()
    layer.post_load_weights()
    with torch.inference_mode():
        output = layer(inputs)
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=3e-2)

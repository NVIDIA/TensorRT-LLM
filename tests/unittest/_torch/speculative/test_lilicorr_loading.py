# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.models.modeling_lilicorr import _load_linear
from tensorrt_llm._torch.modules.linear import Linear
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

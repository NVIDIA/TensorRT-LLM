# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FP8 block-scaled checkpoint tensors that feed an unquantized Linear (the DSA
indexer ``wk`` projection of DeepSeek-V3.2 / GLM-5 checkpoints) must be
dequantized at load time instead of being cast as raw FP8 codes."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_deepseekv3 import (
    maybe_dequantize_fp8_block_scaled_weight,
    weight_dequant,
)
from tensorrt_llm._torch.modules.linear import Linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def _fp8_block_scaled(out_features: int, in_features: int, block: int = 128):
    """Build a random FP8 weight with a 128x128 block scale like the DeepSeek
    recipe (``weight_scale_inv`` is a multiplier)."""
    torch.manual_seed(0)
    reference = torch.randn(out_features, in_features) * 0.05
    scale = torch.empty(out_features // block, in_features // block)
    codes = torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn)
    for i in range(out_features // block):
        for j in range(in_features // block):
            tile = reference[i * block : (i + 1) * block, j * block : (j + 1) * block]
            s = tile.abs().max() / 448.0
            scale[i, j] = s
            codes[i * block : (i + 1) * block, j * block : (j + 1) * block] = (tile / s).to(
                torch.float8_e4m3fn
            )
    return codes, scale, reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unquantized_linear_gets_dequantized_weight(dtype):
    out_features, in_features = 128, 512
    codes, scale, reference = _fp8_block_scaled(out_features, in_features)

    linear = Linear(
        in_features,
        out_features,
        bias=False,
        dtype=dtype,
        quant_config=None,
        skip_create_weights_in_init=True,
    )
    linear.create_weights()
    linear = linear.cuda()

    module_weights = {"weight": codes, "weight_scale_inv": scale}
    prepared = maybe_dequantize_fp8_block_scaled_weight(linear, module_weights)

    assert "weight_scale_inv" not in prepared
    assert prepared["weight"].dtype == dtype
    linear.load_weights(weights=[prepared])

    loaded = linear.weight.data.float().cpu()
    expected = weight_dequant(codes.cuda(), scale.cuda()).float().cpu()
    torch.testing.assert_close(loaded, expected, rtol=1e-2, atol=1e-2)
    # The block scale is applied: raw codes are ~1/scale larger than the weight.
    assert torch.allclose(loaded, reference, rtol=0.1, atol=0.02)
    assert not torch.allclose(loaded, codes.float(), rtol=0.1, atol=0.1)


def test_quantized_or_scaleless_weights_are_untouched():
    linear = Linear(
        512,
        128,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=None,
        skip_create_weights_in_init=True,
    )
    linear.create_weights()

    plain = {"weight": torch.randn(128, 512, dtype=torch.bfloat16)}
    assert maybe_dequantize_fp8_block_scaled_weight(linear, plain) is plain

    scaleless_fp8 = {"weight": torch.randn(128, 512).to(torch.float8_e4m3fn)}
    assert maybe_dequantize_fp8_block_scaled_weight(linear, scaleless_fp8) is scaleless_fp8

    not_a_linear = torch.nn.LayerNorm(128)
    weights = {
        "weight": torch.randn(128, 512).to(torch.float8_e4m3fn),
        "weight_scale_inv": torch.ones(1, 4),
    }
    assert maybe_dequantize_fp8_block_scaled_weight(not_a_linear, weights) is weights

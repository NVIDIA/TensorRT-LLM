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
"""UE8M0 block scales that the fused shared FC12 kernel reads on SM100 and SM103."""

import pytest
import torch

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.megamoe_shared_mlp import _unpack_r128c4
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.quantization.utils.fp8_utils import transform_k128_scales_to_cutedsl_mxfp8_layout

pytestmark = pytest.mark.skipif(
    not is_sm_100f() or get_sm_version() == 107,
    reason="SM107 converts these scales to the CuTe DSL MXFP8 layout instead",
)


def _block_scaled_weights(out_features: int, in_features: int) -> dict[str, torch.Tensor]:
    weight = torch.randn(out_features, in_features) / in_features**0.5
    blocks = weight.reshape(out_features // 128, 128, in_features // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)) / torch.finfo(torch.float8_e4m3fn).max
    quantized = (blocks / scale[:, None, :, None]).reshape(out_features, in_features)
    return {"weight": quantized.to(torch.float8_e4m3fn), "weight_scale": scale.float()}


def _dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    expanded = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return weight.float() * expanded


@torch.inference_mode()
def test_linear_rounds_block_scales_to_ue8m0_when_asked():
    out_features, in_features = 256, 512
    weights = _block_scaled_weights(out_features, in_features)
    linear = Linear(
        in_features,
        out_features,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        use_cute_dsl_blockscaling_mm=True,
    )
    linear.round_block_scales_to_ue8m0 = True
    linear.load_weights([weights])
    linear.post_load_weights()
    linear.cuda()

    scale = linear.weight_scale
    assert scale.dtype == torch.float32
    assert tuple(scale.shape) == (out_features // 128, in_features // 128)
    assert linear.weight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(scale, torch.exp2(torch.round(torch.log2(scale))), rtol=0, atol=0)
    # Rounding a block's scale up to a power of two rescales its FP8 values, so the
    # dequantized weight changes by at most FP8 rounding.
    torch.testing.assert_close(
        _dequantize(linear.weight, scale),
        _dequantize(weights["weight"].cuda(), weights["weight_scale"].cuda()),
        rtol=0.07,
        atol=1e-3,
    )
    # The fused kernel reads these scales through the CuTe DSL MXFP8 layout.
    packed = transform_k128_scales_to_cutedsl_mxfp8_layout(scale, mn=out_features, k=in_features)
    unpacked = _unpack_r128c4(packed, out_features, in_features // 32)
    expected = scale.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)
    torch.testing.assert_close(
        unpacked, expected.to(torch.float8_e8m0fnu).view(torch.uint8), rtol=0, atol=0
    )


@torch.inference_mode()
def test_linear_keeps_block_scales_without_the_request():
    out_features, in_features = 256, 512
    weights = _block_scaled_weights(out_features, in_features)
    linear = Linear(
        in_features,
        out_features,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        use_cute_dsl_blockscaling_mm=True,
    )
    linear.load_weights([weights])
    linear.post_load_weights()
    linear.cuda()

    torch.testing.assert_close(linear.weight_scale.cpu(), weights["weight_scale"], rtol=0, atol=0)

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
"""NVFP4_PLUS format helpers: UE5M3 codes, E8M0 conversion and reference quantization."""

import math

import pytest
import torch

from tensorrt_llm.quantization.mode import QuantAlgo, QuantMode
from tensorrt_llm.quantization.utils.fp4_utils import (
    UE5M3_MAX,
    e8m0_to_ue5m3,
    float_to_ue5m3,
    nvfp4_plus_dequantize_ref,
    nvfp4_plus_quantize_ref,
    ue5m3_to_float,
)


def _ue5m3_value(code: int) -> float:
    if code == 0xFF:
        return float("nan")
    exponent, mantissa = code >> 3, code & 7
    if exponent == 0:
        return mantissa * 2.0**-17
    return (1 + mantissa / 8) * 2.0 ** (exponent - 15)


def test_nvfp4_plus_quant_mode():
    mode = QuantMode.from_quant_algo(QuantAlgo.NVFP4_PLUS)
    assert mode.has_nvfp4_plus()
    assert mode.has_any_quant()
    # The scale format differs, so NVFP4 code paths must not pick it up.
    assert not mode.has_nvfp4()
    assert not mode.has_mxfp4()


def test_ue5m3_to_float_covers_every_code():
    decoded = ue5m3_to_float(torch.arange(256, dtype=torch.uint8)).tolist()
    for code, value in enumerate(decoded):
        expected = _ue5m3_value(code)
        assert (math.isnan(value) and math.isnan(expected)) or value == expected, hex(code)
    assert decoded[0xFE] == UE5M3_MAX


def test_float_to_ue5m3_rounds_up_and_saturates():
    exact = ue5m3_to_float(torch.arange(255, dtype=torch.uint8))
    assert torch.equal(float_to_ue5m3(exact), torch.arange(255, dtype=torch.uint8))
    # Just above a representable value rounds to the next code.
    above = float_to_ue5m3(exact[1:-1] * (1 + 2.0**-20))
    assert torch.equal(above, torch.arange(2, 255, dtype=torch.uint8))
    special = float_to_ue5m3(torch.tensor([0.0, 1e-30, 2e5, float("inf"), float("nan")]))
    assert special.tolist() == [0x00, 0x01, 0xFE, 0xFE, 0xFF]


def test_e8m0_to_ue5m3_is_exact_in_range():
    e8m0 = torch.arange(127 - 17, 127 + 17, dtype=torch.uint8)
    converted = ue5m3_to_float(e8m0_to_ue5m3(e8m0)).double()
    assert torch.equal(converted, torch.exp2(e8m0.double() - 127))
    # Below 2**-17 the block is negligible and becomes zero.
    assert e8m0_to_ue5m3(torch.tensor([0, 100], dtype=torch.uint8)).tolist() == [0, 0]
    with pytest.raises(ValueError):
        e8m0_to_ue5m3(torch.tensor([127 + 17], dtype=torch.uint8))


def test_reference_quantization_round_trip():
    torch.manual_seed(0)
    block_magnitudes = torch.exp2(torch.randint(-8, 8, (16, 8, 1)).float())
    x = (torch.randn(16, 8, 32) * block_magnitudes).view(16, 256)
    fp4, sf = nvfp4_plus_quantize_ref(x)
    assert fp4.shape == (16, 128) and sf.shape == (16, 8)
    blocks = x.view(16, 8, 32)
    scale = ue5m3_to_float(sf)
    # The rounded-up scale never clips: |x| / scale stays within E2M1's 6.
    assert bool(((blocks.abs().amax(-1) / scale) <= 6.0).all())
    error = (nvfp4_plus_dequantize_ref(fp4, sf).view(16, 8, 32) - blocks).abs().amax(-1)
    # E2M1 steps are at most 2 in units of the scale, so rounding is within 1.
    assert bool((error <= scale).all())

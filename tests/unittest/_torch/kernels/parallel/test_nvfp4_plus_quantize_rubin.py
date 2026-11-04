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
"""SM107 NVFP4_PLUS: the UE5M3 conversion and the activation quantize kernel."""

import math

import pytest
import torch
from utils.util import getSMVersion

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm.quantization.utils.fp4_utils import (
    UE5M3_MAX,
    float_to_ue5m3,
    nvfp4_plus_quantize_ref,
    ue5m3_to_float,
)

pytestmark = pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="NVFP4_PLUS runs on SM107 with the CuTe DSL Rubin support",
)


def _activations(m: int, k: int, dtype: torch.dtype) -> torch.Tensor:
    """Blocks of 32 spanning UE5M3's range, plus zero, exact, clamped and tiny blocks."""
    g = torch.Generator(device="cuda").manual_seed(m * 1000 + k)
    blocks = torch.randn((m, k // 32, 32), generator=g, device="cuda")
    blocks *= torch.exp2(
        torch.randint(-22, 19, (m, k // 32, 1), generator=g, device="cuda").float()
    )
    flat = blocks.view(-1, 32)
    flat[0] = 0.0
    if flat.shape[0] > 3:
        flat[1] = 0.0
        flat[1, 5] = 6.0 * 1.125 * 8.0  # amax / 6 is exactly a UE5M3 value
        flat[2] *= 1e6 / flat[2].abs().max()  # above 6 * UE5M3_MAX: the scale saturates
        flat[3] *= 1e-7 / flat[3].abs().max()  # below the smallest UE5M3 subnormal
    return blocks.view(m, k).to(dtype)


def test_hardware_conversion_matches_reference() -> None:
    import cutlass
    import cutlass.torch as cutlass_torch

    g = torch.Generator().manual_seed(0)
    values = torch.cat(
        [
            torch.exp2(torch.empty(1 << 18).uniform_(-24.0, math.log2(UE5M3_MAX), generator=g)),
            ue5m3_to_float(torch.arange(255, dtype=torch.uint8)),
            ue5m3_to_float(torch.arange(254, dtype=torch.uint8)) * (1 + 2.0**-23),
            torch.tensor([0.0, 1e-45, UE5M3_MAX]),
        ]
    ).float()
    values = values[: values.numel() // 4 * 4]
    _, hardware = cutlass_torch.cute_tensor_like(
        values.cuda(), cutlass.FloatNV8E5M3FNU, is_dynamic_layout=True, assumed_align=16
    )
    assert torch.equal(hardware.cpu(), float_to_ue5m3(values))


@pytest.mark.parametrize("m, k", [(1, 32), (7, 2880), (300, 7168)])
def test_quantize_kernel_matches_reference(m: int, k: int) -> None:
    """The row-major scale layout that the NVFP4_PLUS MoE quantizes its input to."""
    from tensorrt_llm._torch.kernels.rubin.nvfp4_plus_quantize import nvfp4_plus_quantize

    x = _activations(m, k, torch.bfloat16)
    ref_fp4, ref_sf = nvfp4_plus_quantize_ref(x)
    fp4, sf = nvfp4_plus_quantize(x, sf_swizzled=False)
    assert torch.equal(fp4, ref_fp4)
    assert torch.equal(sf, ref_sf)

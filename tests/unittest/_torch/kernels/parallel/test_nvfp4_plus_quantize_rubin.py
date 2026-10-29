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
"""SM107 NVFP4_PLUS: the UE5M3 conversion, the activation quantize kernel and the GEMM."""

import math

import pytest
import torch
from utils.util import getSMVersion

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm.quantization.utils.fp4_utils import (
    UE5M3_MAX,
    float_to_ue5m3,
    nvfp4_plus_dequantize_ref,
    nvfp4_plus_quantize_ref,
    pad_up,
    ue5m3_to_float,
)

pytestmark = pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="NVFP4_PLUS runs on SM107 with the CuTe DSL Rubin support",
)


def _unswizzle_sf(sf: torch.Tensor, rows: int, sf_cols: int) -> torch.Tensor:
    """Undo the 128x4 swizzle: [rows, sf_cols] plus the zero-padded remainder."""
    padded_rows, padded_cols = pad_up(rows, 128), pad_up(sf_cols, 4)
    full = (
        sf.view(padded_rows // 128, padded_cols // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(padded_rows, padded_cols)
    )
    return full


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


def test_hardware_conversion_matches_reference():
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m, k", [(1, 32), (3, 96), (7, 2880), (130, 4096), (300, 7168)])
@pytest.mark.parametrize("sf_swizzled", [True, False])
def test_quantize_kernel_matches_reference(m, k, dtype, sf_swizzled):
    from tensorrt_llm._torch.kernels.rubin.nvfp4_plus_quantize import nvfp4_plus_quantize

    x = _activations(m, k, dtype)
    ref_fp4, ref_sf = nvfp4_plus_quantize_ref(x)
    fp4, sf = nvfp4_plus_quantize(x, sf_swizzled=sf_swizzled)
    assert torch.equal(fp4, ref_fp4)
    if sf_swizzled:
        full = _unswizzle_sf(sf, m, k // 32)
        assert torch.equal(full[:m, : k // 32], ref_sf)
        full[:m, : k // 32] = 0
        assert not full.any(), "the swizzled padding must be zero"
    else:
        assert torch.equal(sf, ref_sf)


def test_quantized_gemm_matches_dequantized_reference():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack, make_ptr

    from tensorrt_llm._torch.kernels.rubin import dense_blockscaled_gemm_persistent as rubin_gemm
    from tensorrt_llm._torch.kernels.rubin.nvfp4_plus_quantize import nvfp4_plus_quantize

    m, n, k = 300, 256, 4096
    g = torch.Generator(device="cuda").manual_seed(7)
    a = (torch.randn((m, k), generator=g, device="cuda") * 3).bfloat16()
    b = (torch.randn((n, k), generator=g, device="cuda") * 0.02).bfloat16()
    a_fp4, a_sf = nvfp4_plus_quantize(a)
    b_fp4, b_sf = nvfp4_plus_quantize(b)

    gemm = rubin_gemm.Sm107BlockScaledPersistentDenseGemmKernel(
        32, (128, 128, 128), (128, 128, 256), (1, 1)
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    alpha = torch.ones(1, dtype=torch.float32, device="cuda")
    alpha_tensor = from_dlpack(alpha, assumed_align=16)
    compiled = rubin_gemm.scaled_mm(
        gemm,
        cutlass.Float4E2M1FN,
        cutlass.Float4E2M1FN,
        cutlass.BFloat16,
        cutlass.FloatNV8E5M3FNU,
        "k",
        "k",
        "n",
        utils.HardwareInfo().get_max_active_clusters(1),
        stream,
        alpha_tensor=alpha_tensor,
        batch_size=1,
    )
    c = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    gmem = cute.AddressSpace.gmem
    compiled(
        m,
        n,
        k,
        m,
        n,
        k // 32,
        make_ptr(cutlass.Float4E2M1FN, a_fp4.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, b_fp4.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.FloatNV8E5M3FNU, a_sf.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.FloatNV8E5M3FNU, b_sf.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.BFloat16, c.data_ptr(), gmem, assumed_align=16),
        alpha_tensor,
        0,
        stream,
    )
    ref = nvfp4_plus_dequantize_ref(*nvfp4_plus_quantize_ref(a)) @ (
        nvfp4_plus_dequantize_ref(*nvfp4_plus_quantize_ref(b)).T
    )
    torch.testing.assert_close(c.float(), ref, rtol=1e-2, atol=1e-2 * ref.abs().max().item())

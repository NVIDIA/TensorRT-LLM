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
"""Validates how to slice a padded NVFP4 activation by num_tokens so it can
feed the NVFP4 GEMM -- the enabling step for folding the layer-boundary
add+RMSNorm+NVFP4-quant on the **non-DSA** MLA path (e.g. Kimi-K2.5).

On that path, MLA.forward_impl slices the next layer's input by num_tokens
(``hidden_states = hidden_states[:num_tokens]``) to drop CUDA-graph padding,
BEFORE feeding kv_a_proj_with_mqa. To let the boundary fusion pre-quantize that
input (producing an Fp4QuantizedTensor sized for the padded batch), the slice
must happen on the FP4 form.

Key fact this test pins down: NVFP4 quantization is per-row independent, and
the swizzled scale-factor (computeSwizzledLayoutSFSize = padUp(rows,128) x
padUp(cols,4)) is laid out tile-by-tile in 128-row tiles. Therefore the slice
is simply:

  * fp4_tensor[:num_tokens]                                 (row-major, trivial)
  * scaling_factor.view(-1)[:padUp(num_tokens,128)*padUp(cols,4)]
        -- the LEADING swizzled bytes, which are byte-identical to freshly
           swizzling only num_tokens rows because each 128-row tile's layout
           is independent of the total row count.

This test feeds that sliced FP4 input through the NVFP4 GEMM and asserts it
matches the unpadded reference (quantize the real tokens, then GEMM). If it
holds, forward_impl can slice the Fp4QuantizedTensor directly and the is_dsa
gate on the boundary fold can be relaxed to cover non-DSA models.
"""

import pytest
import torch

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tests.unittest.utils.util import skip_pre_blackwell

SF_VEC = 16


def _gemm(x_fp4, w_fp4, x_sf, w_sf, alpha, dtype):
    return torch.ops.trtllm.nvfp4_gemm_cutlass(x_fp4, w_fp4, x_sf, w_sf, alpha, dtype)


def _leading_sf_len(num_tokens, k):
    return fp4_utils.pad_up(num_tokens, 128) * fp4_utils.pad_up(k // SF_VEC, 4)


@skip_pre_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
# (padded_m, num_tokens, n, k): padded_m = CUDA-graph batch, num_tokens = real
# count. Includes num_tokens that straddle a 128-row swizzle-tile boundary
# (130 with padded_m 256) so we exercise multi-tile slicing.
@pytest.mark.parametrize(
    "shape",
    [
        (128, 40, 512, 1536),
        (128, 100, 2048, 7168),
        (256, 130, 512, 1536),
        (32, 7, 512, 1536),
    ],
)
def test_fp4_num_tokens_slice_feeds_gemm(dtype, shape):
    padded_m, num_tokens, n, k = shape
    assert num_tokens <= padded_m and k % SF_VEC == 0
    torch.manual_seed(0)
    dev = torch.device("cuda")

    # Weight (static NVFP4), shared across paths.
    w = torch.randn((n, k), dtype=dtype, device=dev)
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, w_sf_global, SF_VEC, False)

    # Padded activation: rows [num_tokens:padded_m] are graph padding (garbage).
    # The per-tensor input scale is calibrated on the real tokens only, matching
    # the static-NVFP4 Linear (input_scale comes from calibration, not runtime).
    x_pad = torch.randn((padded_m, k), dtype=dtype, device=dev)
    x_sf_global = (448 * 6) / x_pad[:num_tokens].abs().max().float()
    x_pad_fp4, x_pad_sf = torch.ops.trtllm.fp4_quantize(x_pad, x_sf_global, SF_VEC, False)
    alpha = (1.0 / (w_sf_global * x_sf_global)).to(torch.float32).view(1)

    # ---- Reference: quantize ONLY the real tokens, then GEMM (what the unfused
    # path produces today: slice BF16 first, then quantize). ----
    x_real = x_pad[:num_tokens].contiguous()
    xr_fp4, xr_sf = torch.ops.trtllm.fp4_quantize(x_real, x_sf_global, SF_VEC, False)
    out_ref = _gemm(xr_fp4, w_fp4, xr_sf, w_sf, alpha, dtype)
    assert out_ref.shape[0] == num_tokens

    # ---- Sliced: fp4_tensor[:num_tokens] + leading swizzled SF bytes. This is
    # the slice forward_impl would apply to a pre-quantized Fp4QuantizedTensor. ----
    sliced_fp4 = x_pad_fp4[:num_tokens].contiguous()
    sliced_sf = x_pad_sf.view(-1)[: _leading_sf_len(num_tokens, k)].contiguous()
    out_sliced = _gemm(sliced_fp4, w_fp4, sliced_sf, w_sf, alpha, dtype)

    torch.testing.assert_close(out_sliced.float(), out_ref.float(), rtol=2e-2, atol=0.2)

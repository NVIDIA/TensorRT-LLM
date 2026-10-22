# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
from warnings import warn

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from utils.util import getSMVersion


@pytest.mark.skipif(getSMVersion() != 103,
                    reason="the exact-m cuBLASLt configs are tuned for SM 10.3")
@pytest.mark.parametrize("m", [224, 448, 504, 672, 896])
def test_fp8_scaled_mm_exact_m_configs(m):
    """Shapes with an exact-m entry in cublas_lut::sm103_fp8_exact_m_algo_list run their config and match torch."""
    k, n = 4096, 2560
    torch.random.manual_seed(0)
    x = torch.rand((m, k), device="cuda").to(torch.float8_e4m3fn)
    w = torch.rand((n, k), device="cuda").to(torch.float8_e4m3fn)
    scale_x = torch.rand(1, device="cuda")
    scale_w = torch.rand(1, device="cuda")
    output = torch.ops.trtllm.cublas_scaled_mm(x,
                                               w.t(),
                                               scale_x,
                                               scale_w,
                                               bias=None,
                                               out_dtype=torch.bfloat16)
    ref = torch._scaled_mm(x,
                           w.t(),
                           out_dtype=torch.bfloat16,
                           scale_a=scale_x,
                           scale_b=scale_w,
                           use_fast_accum=True)
    np.testing.assert_allclose(ref.float().cpu(),
                               output.float().cpu(),
                               atol=0.01,
                               rtol=0.01)


@pytest.mark.skipif(
    getSMVersion() < 90 or getSMVersion() >= 120,
    reason="custom scaled_mm is only supported in SM90",
)  # Skip tests that are not supported in SM90
@pytest.mark.parametrize(
    "k_n",
    [(8192, 10240), (8192, 8192), (8192, 57344), (28672, 8192)],
)
@pytest.mark.parametrize(
    "m",
    [2048, 8, 228],
)
@pytest.mark.parametrize(
    "output_dtype",
    [torch.float16, torch.float32, torch.bfloat16],
)
def test_fp8_scaled_mm(output_dtype, m, k_n):
    k, n = k_n
    torch.random.manual_seed(0)
    shape_x = (m, k)
    shape_w = (n, k)
    x = torch.rand(shape_x, device="cuda").to(torch.float8_e4m3fn)
    w = torch.rand(shape_w, device="cuda").to(torch.float8_e4m3fn)
    scale_x = torch.rand(1, device="cuda")
    scale_w = torch.rand(1, device="cuda")
    output = torch.ops.trtllm.cublas_scaled_mm(
        x,
        w.t(),
        scale_x,
        scale_w,
        bias=None,
        out_dtype=output_dtype,
    )
    # Set pytorch's cublas workspace size to 32MB to be aligned with trtllm.
    # If anywhere else calls torch's cublas op, the static workspace size will
    # be fixed to 1MB. If not aligned, will cause cause pytorch not using splitK
    # algo, while trtllm may use.
    old_env = os.environ.get("CUBLASLT_WORKSPACE_SIZE", "")
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = f"{32*1024}"
    ref = torch._scaled_mm(
        x,
        w.t(),
        out_dtype=output_dtype,
        scale_a=scale_x,
        scale_b=scale_w,
        use_fast_accum=True,
    )
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = old_env
    np.testing.assert_allclose(ref.float().cpu(),
                               output.float().cpu(),
                               atol=0.01,
                               rtol=0.01)

    if getSMVersion() == 90:
        cutlass_output = torch.ops.trtllm.cutlass_scaled_mm(
            x,
            w.t(),
            scale_x,
            scale_w,
            bias=None,
            out_dtype=output_dtype,
        )
        # TODO(zhenhuan): cutlass kernel has acc issue on some shapes
        try:
            np.testing.assert_allclose(ref.float().cpu(),
                                       cutlass_output.float().cpu(),
                                       atol=1,
                                       rtol=0.01)
        except Exception as e:
            warn(RuntimeWarning("cutlass result is not correct: " + repr(e)))


if __name__ == '__main__':
    test_fp8_scaled_mm(torch.float16, 12, (8192, 10240))

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
_SM89_PLUS = pytest.mark.skipif(getSMVersion() < 89,
                                reason="FP8 GEMMs need SM89+")


def _epilogue_operands(m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, device="cuda", generator=g)
    w = torch.randn(n, k, device="cuda", generator=g) * 0.05
    scale_a = (a.abs().amax() / E4M3_MAX).reshape(1)
    scale_w = (w.abs().amax() / E4M3_MAX).reshape(1)
    a8 = (a / scale_a).to(E4M3)
    w8 = (w / scale_w).to(E4M3)
    bias = (torch.randn(n, device="cuda", generator=g) * 0.1).bfloat16()
    ref = (a8.float() * scale_a) @ (w8.float() * scale_w).t()
    return a8, w8, scale_a, scale_w, bias, ref


@_SM89_PLUS
@pytest.mark.parametrize("m,n,k", [(4097, 13824, 5120), (33, 3072, 3072)])
# The two combinations callers use: bias into a BF16 output, and bias + GELU into an e4m3 output.
@pytest.mark.parametrize("gelu,out_fp8", [(False, False), (True, True)])
def test_fp8_scaled_mm_epilogue(m, n, k, gelu, out_fp8):
    """cublas_scaled_mm bias / GELU epilogues and e4m3 output against torch."""
    a8, w8, scale_a, scale_w, bias, ref = _epilogue_operands(m,
                                                             n,
                                                             k,
                                                             seed=m + n)
    ref = ref + bias.float()
    if gelu:
        ref = F.gelu(ref, approximate="tanh")
    d_scale = (E4M3_MAX / ref.abs().amax()).reshape(1) if out_fp8 else None
    out = torch.ops.trtllm.cublas_scaled_mm(
        a8,
        w8.t(),
        scale_a,
        scale_w,
        bias,
        E4M3 if out_fp8 else torch.bfloat16,
        epilogue=1 if gelu else 0,
        d_scale=d_scale,
    )
    assert out.shape == (m, n) and out.dtype == (E4M3
                                                 if out_fp8 else torch.bfloat16)
    result = out.float() / d_scale if out_fp8 else out.float()
    cosine = F.cosine_similarity(result.flatten(), ref.flatten(), dim=0).item()
    assert cosine > (0.998 if out_fp8 else 0.9999)


@_SM89_PLUS
def test_fp8_scaled_mm_epilogue_rejects_bad_arguments():
    a8, w8, scale_a, scale_w, bias, _ = _epilogue_operands(64, 256, 128, seed=4)
    with pytest.raises(RuntimeError, match="epilogue"):
        torch.ops.trtllm.cublas_scaled_mm(a8,
                                          w8.t(),
                                          scale_a,
                                          scale_w,
                                          bias,
                                          torch.bfloat16,
                                          epilogue=2)
    with pytest.raises(RuntimeError, match="d_scale"):
        torch.ops.trtllm.cublas_scaled_mm(a8,
                                          w8.t(),
                                          scale_a,
                                          scale_w,
                                          bias,
                                          E4M3,
                                          epilogue=1)

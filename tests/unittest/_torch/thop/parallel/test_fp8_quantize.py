# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math

import pytest
import torch
from parameterized import parameterized
from utils.util import (getSMVersion, isSM100Family,
                        skip_pre_blackwell_unittest, unittest_name_func)

from tensorrt_llm.quantization.utils.fp8_utils import \
    per_token_quant_and_transform


def _dequant_fp8(input, scale, transpose_scale, block_m, block_n):
    input = input.to(torch.float)
    scale = scale.to(torch.float)
    if transpose_scale:
        scale = scale.t()
    output = torch.zeros_like(input)
    m, n = input.shape
    m_tile = 128 if block_m else 1
    n_tile = 128 if block_n else 1

    if m_tile == 1:
        assert n % 16 == 0, "n must be divisible by 16"
        total_blocks = math.ceil(n / 128)
        for block in range(total_blocks):
            # Calculate start position in 2D array
            start_col = block * 128
            end_col = min(start_col + 128, n)
            output[:, start_col:
                   end_col] = input[:, start_col:end_col] * scale[:,
                                                                  block].view(
                                                                      -1, 1)

    elif n_tile == 1:
        assert m % 16 == 0, "m must be divisible by 16"
        total_blocks = math.ceil(m / 128)
        for block in range(total_blocks):
            # Calculate start position in 2D array
            start_row = block * 128
            end_row = min(start_row + 128, m)
            output[start_row:end_row, :] = input[start_row:end_row, :] * scale[
                block, :]
    else:
        assert n % 16 == 0, "n must be divisible by 16"
        assert m % 16 == 0, "m must be divisible by 16"
        n_blocks = math.ceil(n / 128)
        m_blocks = math.ceil(m / 128)
        for i in range(n_blocks):
            for j in range(m_blocks):
                start_row = j * 128
                end_row = min(start_row + 128, m)
                start_col = i * 128
                end_col = min(start_col + 128, n)
                output[start_row:end_row,
                       start_col:end_col] = input[start_row:end_row,
                                                  start_col:end_col] * scale[j,
                                                                             i]
    return output


@pytest.mark.skipif(
    getSMVersion() != 100 and getSMVersion() != 90,
    reason="Only test on Blackwell and Hopper",
)
@pytest.mark.parametrize("k", [576, 256, 32])
@pytest.mark.parametrize(
    "m",
    [4, 16, 256],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
def test_fp8_quantize_blackwell(dtype, m, k):
    torch.random.manual_seed(0)
    # TODO: make sure there is no padding for now
    assert m % 4 == 0, "Disable padding for now"
    a = torch.randn((m, k), device='cuda', dtype=dtype)
    fp8_a, fp8_a_scale = torch.ops.trtllm.fp8_quantize_1x128(a)
    fp8_a_scale = fp8_a_scale.view(-1,
                                   fp8_a.shape[0])  # transpose the scale view
    a_dequant = _dequant_fp8(fp8_a, fp8_a_scale, True, False, True)

    torch.testing.assert_close(a_dequant.cpu().to(torch.float32),
                               a.cpu().to(torch.float32),
                               atol=1e-1,
                               rtol=1e-1)


def mxfp8_quantize_check_accuracy(a, b, atol, rtol, percent):
    if torch.any(torch.isnan(a)):
        raise Exception("NaN in a")
    if torch.any(torch.isnan(b)):
        raise Exception("NaN in b")
    assert a.shape == b.shape
    left = torch.abs(a - b)
    right = atol + rtol * torch.abs(b)
    count = torch.sum(left > right)
    mismatch_percent = count / a.numel()
    if mismatch_percent > 1 - percent:
        raise Exception("Mismatch percentage is %f for rtol %f" %
                        (mismatch_percent, rtol))


@parameterized.expand(list([[1, 1024, torch.half, True],
                            [2, 512, torch.bfloat16, True],
                            [16, 512, torch.half, True],
                            [16, 512, torch.half, False],
                            [1024, 512, torch.half, False],
                            [1024, 512, torch.half, False]]),
                      name_func=unittest_name_func)
@skip_pre_blackwell_unittest
def test_mxfp8_quantize_torch_host(m, k, dtype, is_sf_swizzled_layout):
    torch.random.manual_seed(0)
    a = (torch.randn([m, k], dtype=torch.float) * 16).cpu().contiguous()

    a_fp8, a_sf = torch.ops.tensorrt_llm.quantize_mxe4m3_host(
        a, is_sf_swizzled_layout)

    a_pt = torch.ops.tensorrt_llm.dequantize_mxe4m3_host(
        a_fp8.view(torch.uint8), a_sf.view(torch.uint8), is_sf_swizzled_layout)

    torch.cuda.synchronize()

    mxfp8_quantize_check_accuracy(a_pt, a, 8, 0, 0.999)


@parameterized.expand(list([[1, 1024, torch.half, True],
                            [2, 512, torch.bfloat16, True],
                            [16, 512, torch.half, True],
                            [16, 512, torch.half, False],
                            [1024, 512, torch.half, False],
                            [1024, 512, torch.half, False]]),
                      name_func=unittest_name_func)
@skip_pre_blackwell_unittest
def test_mxfp8_quantize_torch_device(m, k, dtype, is_sf_swizzled_layout):
    torch.random.manual_seed(0)
    a = (torch.randn([m, k], dtype=torch.float) *
         16).to(dtype).cuda().contiguous()

    # Quantize it on device.
    a_fp8, a_sf = torch.ops.trtllm.mxfp8_quantize(a, is_sf_swizzled_layout, 32)

    # Dequantize it on host.
    a_pt = torch.ops.tensorrt_llm.dequantize_mxe4m3_host(
        a_fp8.cpu().view(torch.uint8),
        a_sf.cpu().view(torch.uint8), is_sf_swizzled_layout)

    torch.cuda.synchronize()

    mxfp8_quantize_check_accuracy(a_pt.cpu().to(torch.float32),
                                  a.cpu().to(torch.float32), 8, 0, 0.999)


@pytest.mark.parametrize("m", [1, 2, 16, 1024])
@pytest.mark.parametrize("k", [1568])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("alignment", [64, 128])
@skip_pre_blackwell_unittest
def test_mxfp8_quantize_alignment_torch_device(m, k, dtype,
                                               is_sf_swizzled_layout,
                                               alignment):
    torch.random.manual_seed(0)
    a = (torch.randn([m, k], dtype=torch.float) *
         16).to(dtype).cuda().contiguous()
    padded_k = ((k + alignment - 1) // alignment) * alignment

    # Quantize it on device.
    a_fp8, a_sf = torch.ops.trtllm.mxfp8_quantize(a, is_sf_swizzled_layout,
                                                  alignment)
    assert a_fp8.shape[1] == padded_k

    # Dequantize it on host.
    a_pt = torch.ops.tensorrt_llm.dequantize_mxe4m3_host(
        a_fp8.cpu().view(torch.uint8),
        a_sf.cpu().view(torch.uint8), is_sf_swizzled_layout)

    # Check if the bits of paddings are zero.
    paddings = a_fp8.view(torch.int8)[:, k:]
    assert torch.all(paddings == 0), "Paddings should be zero"

    torch.cuda.synchronize()

    mxfp8_quantize_check_accuracy(a_pt[:, :k].cpu().to(torch.float32),
                                  a.cpu().to(torch.float32), 8, 0, 0.999)


def _run_megamoe_prepare(hidden_states, token_selected_experts,
                         token_final_scales):
    m, k = hidden_states.shape
    top_k = token_selected_experts.shape[1]

    ref_x, ref_x_sf_u8 = torch.ops.trtllm.mxfp8_quantize(
        hidden_states, False, 32)
    ref_x_sf = ref_x_sf_u8.view(m, k // 32).view(torch.int32)

    buf_rows = m + 3
    x_out = torch.empty((buf_rows, k), dtype=torch.float8_e4m3fn, device="cuda")
    x_sf_out = torch.empty((buf_rows, k // 128),
                           dtype=torch.int32,
                           device="cuda")
    topk_idx_out = torch.empty((buf_rows, top_k),
                               dtype=torch.int64,
                               device="cuda")
    topk_weights_out = torch.empty((buf_rows, top_k),
                                   dtype=torch.float32,
                                   device="cuda")

    torch.ops.trtllm.megamoe_prepare(hidden_states, token_selected_experts,
                                     token_final_scales, x_out, x_sf_out,
                                     topk_idx_out, topk_weights_out)
    torch.cuda.synchronize()

    assert torch.equal(x_out[:m].view(torch.uint8), ref_x.view(torch.uint8))
    assert torch.equal(x_sf_out[:m], ref_x_sf)
    assert torch.equal(topk_idx_out[:m], token_selected_experts.to(torch.int64))
    assert torch.equal(topk_weights_out[:m],
                       token_final_scales.to(torch.float32))


@pytest.mark.parametrize("m", [1, 7, 128])
@pytest.mark.parametrize("k", [128, 512])
@pytest.mark.parametrize("top_k", [1, 6])
@skip_pre_blackwell_unittest
def test_megamoe_prepare_matches_mxfp8_quantize(m, k, top_k):
    torch.random.manual_seed(123)
    hidden_states = (torch.randn([m, k], dtype=torch.float) * 16).to(
        torch.bfloat16).cuda().contiguous()
    token_selected_experts = torch.randint(0,
                                           384, (m, top_k),
                                           dtype=torch.int32,
                                           device="cuda")
    token_final_scales = torch.randn((m, top_k),
                                     dtype=torch.float32,
                                     device="cuda")

    _run_megamoe_prepare(hidden_states, token_selected_experts,
                         token_final_scales)


@pytest.mark.parametrize("expert_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("scale_dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
@skip_pre_blackwell_unittest
def test_megamoe_prepare_accepts_supported_topk_dtypes(expert_dtype,
                                                       scale_dtype):
    torch.random.manual_seed(123)
    m, k, top_k = 5, 256, 3
    hidden_states = (torch.randn([m, k], dtype=torch.float) * 16).to(
        torch.bfloat16).cuda().contiguous()
    token_selected_experts = torch.randint(0,
                                           384, (m, top_k),
                                           dtype=expert_dtype,
                                           device="cuda")
    token_final_scales = torch.randn((m, top_k),
                                     dtype=scale_dtype,
                                     device="cuda")

    _run_megamoe_prepare(hidden_states, token_selected_experts,
                         token_final_scales)


@skip_pre_blackwell_unittest
def test_megamoe_prepare_allows_zero_tokens():
    k, top_k = 256, 3
    hidden_states = torch.empty((0, k), dtype=torch.bfloat16, device="cuda")
    token_selected_experts = torch.empty((0, top_k),
                                         dtype=torch.int32,
                                         device="cuda")
    token_final_scales = torch.empty((0, top_k),
                                     dtype=torch.float32,
                                     device="cuda")
    x_out = torch.empty((1, k), dtype=torch.float8_e4m3fn, device="cuda")
    x_sf_out = torch.empty((1, k // 128), dtype=torch.int32, device="cuda")
    topk_idx_out = torch.empty((1, top_k), dtype=torch.int64, device="cuda")
    topk_weights_out = torch.empty((1, top_k),
                                   dtype=torch.float32,
                                   device="cuda")

    torch.ops.trtllm.megamoe_prepare(hidden_states, token_selected_experts,
                                     token_final_scales, x_out, x_sf_out,
                                     topk_idx_out, topk_weights_out)
    torch.cuda.synchronize()


@pytest.mark.skipif(
    getSMVersion() != 103 and getSMVersion() != 90,
    reason="Only test on Blackwell and Hopper",
)
@pytest.mark.parametrize("k", [128, 256, 512])
@pytest.mark.parametrize("m", [4, 16, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fp8_quantize_ue8m0_vs_triton(dtype, m, k):
    """Test fp8_quantize_1x128 with use_ue8m0=True against Triton reference."""
    torch.random.manual_seed(42)

    # Ensure alignment requirements
    assert m % 4 == 0, "m must be divisible by 4"
    assert k % 128 == 0, "k must be divisible by 128"

    # Create input tensor
    input_tensor = torch.randn((m, k), device='cuda', dtype=dtype)

    # CUDA version with UE8M0
    from tensorrt_llm.quantization.utils import fp8_utils
    cuda_fp8, cuda_scale_float = fp8_utils.fp8_quantize_1x128_sf_transpose(
        input_tensor, use_ue8m0=True)

    # Triton reference with UE8M0
    triton_fp8, triton_scale_int32 = per_token_quant_and_transform(
        input_tensor.clone(), quant_group_size=128, scale_ue8m0=True)

    # Convert CUDA float32 scale to int32 packed format for comparison
    cuda_scale_int32 = fp8_utils.transform_sf_into_required_layout(
        sf=cuda_scale_float,
        mn=m,
        k=k,
        recipe=(1, 1, 128),
        num_groups=None,
        is_sfa=False)

    # Compare quantized FP8 values (should match within FP8 precision)
    torch.testing.assert_close(
        cuda_fp8.to(torch.float32),
        triton_fp8.to(torch.float32),
        atol=1.0,
        rtol=0.01,
        msg=f"FP8 quantized values mismatch for shape ({m}, {k})")

    # Compare packed int32 scales by decoding to float
    def decode_ue8m0_int32_to_float(int32_tensor):
        """Decode int32 packed UE8M0 scales to float32. Formula: value = 2^(exponent - 127)
        """
        # Extract 4 bytes from each int32
        b0 = (int32_tensor >> 0) & 0xFF
        b1 = (int32_tensor >> 8) & 0xFF
        b2 = (int32_tensor >> 16) & 0xFF
        b3 = (int32_tensor >> 24) & 0xFF

        # Decode: value = 2^(exponent - 127)
        s0 = torch.pow(2.0, b0.float() - 127.0)
        s1 = torch.pow(2.0, b1.float() - 127.0)
        s2 = torch.pow(2.0, b2.float() - 127.0)
        s3 = torch.pow(2.0, b3.float() - 127.0)

        return torch.stack([s0, s1, s2, s3], dim=-1)

    cuda_scales_decoded = decode_ue8m0_int32_to_float(cuda_scale_int32)
    triton_scales_decoded = decode_ue8m0_int32_to_float(triton_scale_int32)

    # Compare: values should match, or both be tiny (< 1e-10 means zero-block)
    torch.testing.assert_close(
        cuda_scales_decoded,
        triton_scales_decoded,
        atol=1e-10,
        rtol=0.01,
        msg=f"UE8M0 decoded scales mismatch for shape ({m}, {k})")


# ---------------------------------------------------------------------------
# Tests for fp8_quantize_1x128_packed_ue8m0 (SM100 fused quant+pack)
# ---------------------------------------------------------------------------


def _decode_packed_int32_ue8m0(packed_int32):
    """Decode int32 packed UE8M0 scales to (exponent, value) per byte."""
    b0 = (packed_int32 >> 0) & 0xFF
    b1 = (packed_int32 >> 8) & 0xFF
    b2 = (packed_int32 >> 16) & 0xFF
    b3 = (packed_int32 >> 24) & 0xFF
    return torch.stack([b0, b1, b2, b3], dim=-1)


@pytest.mark.skipif(not isSM100Family(),
                    reason="fp8_quantize_1x128_packed_ue8m0 is SM100 only.")
@pytest.mark.parametrize("m,k", [
    (1, 128),
    (3, 256),
    (4, 512),
    (7, 512),
    (16, 7168),
    (127, 4096),
    (1024, 7168),
])
def test_fp8_quantize_1x128_packed_ue8m0_matches_legacy(m, k):
    """The fused packed op should produce the same FP8 output and UE8M0
    scales as the legacy (fp8_quantize_1x128 → pack) two-kernel sequence.
    The legacy path is the canonical reference for what deep_gemm expects.
    """
    from tensorrt_llm.quantization.utils import fp8_utils

    torch.manual_seed(0)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)

    fused_fp8, fused_packed = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
        x)

    # Legacy: unpacked quant + manual pack
    ref_fp8, ref_scale = torch.ops.trtllm.fp8_quantize_1x128(x, use_ue8m0=True)
    # ref_scale shape from fp8_quantize_1x128: [num_n_blocks, m_padded] float
    # Convert to the same packed (m, num_packed_sf_k) int32 layout as fused.
    ref_packed = fp8_utils.get_col_major_tma_aligned_packed_tensor(
        ref_scale[:, :m].t().contiguous().to(torch.float32))

    # FP8 bytes must be bit-identical for the valid [0, m) region.
    assert torch.equal(fused_fp8.view(torch.uint8),
                       ref_fp8.view(torch.uint8)[:m]), \
        f"FP8 mismatch for ({m}, {k})"

    # Packed scales: compare element-by-element.
    fused_contig = fused_packed.contiguous().view(torch.int32)
    ref_contig = ref_packed.contiguous().view(torch.int32)
    assert fused_contig.shape == ref_contig.shape, \
        f"shape mismatch fused={fused_contig.shape} ref={ref_contig.shape}"
    assert torch.equal(
        fused_contig,
        ref_contig), (f"Packed UE8M0 mismatch for ({m}, {k}): "
                      f"fused[0,:]={fused_contig[0]} ref[0,:]={ref_contig[0]}")


@pytest.mark.skipif(not isSM100Family(),
                    reason="fp8_quantize_1x128_packed_ue8m0 is SM100 only.")
@pytest.mark.parametrize("m,k", [
    (1, 128),
    (3, 256),
    (7, 512),
    (13, 7168),
    (127, 4096),
])
def test_fp8_quantize_1x128_packed_ue8m0_padded_rows_are_zero(m, k):
    """Padded rows [m, m_aligned) of the physical packed scale buffer must be 0.
    The op returns a `(m, num_packed_sf_k)` strided view that hides the padded
    rows; allocate a sentinel buffer immediately before the call so any
    unwritten ints in the padded region surface as the sentinel.
    """
    if m % 4 == 0:
        pytest.skip("m is already TMA-aligned; no padded rows to check")

    torch.manual_seed(0)
    m_padded = ((m + 3) // 4) * 4
    num_packed_sf_k = ((k + 127) // 128 + 3) // 4
    total_int32 = num_packed_sf_k * m_padded

    # Poison the CUDA caching allocator: allocate-fill-free a buffer of the
    # exact size, then call the op. The op's empty_cuda is likely to land in
    # the same slab, so any unwritten ints surface as the sentinel.
    SENTINEL = 0x5A5A5A  # 5,921,370 — fits in int32 and is distinctive
    poison = torch.empty((num_packed_sf_k, m_padded),
                         dtype=torch.int32,
                         device="cuda")
    poison.fill_(SENTINEL)
    del poison
    torch.cuda.synchronize()

    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    _, packed = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x)

    # Physical layout: int32[num_packed_sf_k][m_padded] starting at packed.data_ptr().
    # The returned tensor's storage_size() reflects only the logical view, so
    # reach into it via the data_ptr + cudaMemcpyDeviceToDevice into a fresh
    # full-sized tensor.
    physical = torch.empty((num_packed_sf_k, m_padded),
                           dtype=torch.int32,
                           device="cuda")
    # Use cudart memcpy via torch's torch.cuda.memory functions
    src_ptr = packed.data_ptr()
    dst_ptr = physical.data_ptr()
    nbytes = total_int32 * 4
    # Build a 1-D byte view at src_ptr by creating a fresh tensor of the
    # appropriate size; this requires an untyped storage of full extent. The
    # storage created by `at::from_blob` is sized to the strided view's reach
    # (m, num_packed_sf_k) so we need an out-of-band copy. cudart via torch:
    torch.cuda.synchronize()
    import ctypes
    libcudart = ctypes.CDLL("libcudart.so")
    err = libcudart.cudaMemcpy(ctypes.c_void_p(dst_ptr),
                               ctypes.c_void_p(src_ptr),
                               ctypes.c_size_t(nbytes),
                               ctypes.c_int(3))  # cudaMemcpyDeviceToDevice
    assert err == 0, f"cudaMemcpy failed with err={err}"
    torch.cuda.synchronize()

    padded_tail = physical[:, m:m_padded]
    nonzero = int((padded_tail != 0).sum().item())
    assert nonzero == 0, (
        f"Padded rows [{m}, {m_padded}) must be zero; got {nonzero} non-zero "
        f"int32 in shape ({num_packed_sf_k}, {m_padded - m})")


# ---------------------------------------------------------------------------
# Tests for triton_fp8_quantize_1x128 (library Triton quant kernel)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not isSM100Family(),
    reason="Triton-vs-CUDA scale layout comparison requires SM100+. "
    "Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize("use_ue8m0", [True, False])
@pytest.mark.parametrize("k", [128, 256, 512, 576, 1024, 5120])
@pytest.mark.parametrize("m", [4, 16, 64, 256, 1024, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_triton_fp8_quantize_1x128_vs_cuda(dtype, m, k, use_ue8m0):
    """Validate triton_fp8_quantize_1x128 matches CUDA fp8_quantize_1x128.

    Both kernels perform 1x128 block-scale FP8 E4M3 quantization.  We compare:
      1. FP8 quantized values  (< 1% byte-level mismatch)
      2. Float32 scale tensors (match within FP tolerance)

    Tested with both use_ue8m0=True (power-of-2 scales) and False (raw scales).
    """
    from tensorrt_llm.quantization.utils.fp8_quantize import \
        triton_fp8_quantize_1x128

    torch.random.manual_seed(42)
    input_tensor = torch.randn((m, k), device='cuda', dtype=dtype)

    cuda_fp8, cuda_scale = torch.ops.trtllm.fp8_quantize_1x128(
        input_tensor, use_ue8m0=use_ue8m0)

    triton_fp8, triton_scale = triton_fp8_quantize_1x128(input_tensor,
                                                         use_ue8m0=use_ue8m0)

    # --- FP8 values ---
    cuda_u8 = cuda_fp8.view(torch.uint8)[:m, :k]
    triton_u8 = triton_fp8.view(torch.uint8)
    fp8_mismatch = torch.sum(cuda_u8 != triton_u8).item()
    fp8_total = cuda_u8.numel()
    fp8_mismatch_pct = fp8_mismatch / fp8_total * 100
    assert fp8_mismatch_pct < 1.0, (
        f"FP8 byte mismatch {fp8_mismatch_pct:.2f}% >= 1.0% "
        f"for shape ({m}, {k}) use_ue8m0={use_ue8m0}")

    # --- Scale values ---
    assert cuda_scale.shape == triton_scale.shape, (
        f"Scale shape mismatch: CUDA {cuda_scale.shape} vs "
        f"Triton {triton_scale.shape}")
    torch.testing.assert_close(
        triton_scale,
        cuda_scale,
        atol=1e-6,
        rtol=1e-3,
        msg=f"Scale mismatch for shape ({m}, {k}) use_ue8m0={use_ue8m0}")


@pytest.mark.skipif(
    not isSM100Family(),
    reason="Triton-vs-CUDA scale layout comparison requires SM100+. "
    "Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize("use_ue8m0", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_triton_fp8_quantize_1x128_large_m(dtype, use_ue8m0):
    """Stress-test with large M values typical of visual-gen / long-prefill.

    These are the shapes where the Triton kernel is expected to outperform
    CUDA by ~3x.  We still validate correctness here, not performance.
    """
    from tensorrt_llm.quantization.utils.fp8_quantize import \
        triton_fp8_quantize_1x128

    torch.random.manual_seed(42)

    for m, k in [(8192, 5120), (16384, 5120), (4096, 13824)]:
        input_tensor = torch.randn((m, k), device='cuda', dtype=dtype)

        cuda_fp8, cuda_scale = torch.ops.trtllm.fp8_quantize_1x128(
            input_tensor, use_ue8m0=use_ue8m0)
        triton_fp8, triton_scale = triton_fp8_quantize_1x128(
            input_tensor, use_ue8m0=use_ue8m0)

        cuda_u8 = cuda_fp8.view(torch.uint8)[:m, :k]
        triton_u8 = triton_fp8.view(torch.uint8)
        fp8_mismatch_pct = (torch.sum(cuda_u8 != triton_u8).item() /
                            cuda_u8.numel() * 100)
        assert fp8_mismatch_pct < 1.0, (
            f"FP8 mismatch {fp8_mismatch_pct:.2f}% for ({m}, {k}) "
            f"use_ue8m0={use_ue8m0}")

        assert cuda_scale.shape == triton_scale.shape
        torch.testing.assert_close(triton_scale,
                                   cuda_scale,
                                   atol=1e-6,
                                   rtol=1e-3,
                                   msg=f"Scale mismatch for shape ({m}, {k}) "
                                   f"use_ue8m0={use_ue8m0}")


# ---------------------------------------------------------------------------
# Round-trip tests for inverse_transform_sf / unpack_col_major_tma_aligned_packed_tensor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mn,k", [
    (256, 256),
    (384, 512),
    (1024, 5120),
    (768, 13824),
    (5120, 5120),
])
def test_unpack_col_major_tma_aligned_round_trip(mn, k):
    """Pack then unpack should recover the original float32 UE8M0 tensor."""
    from tensorrt_llm.quantization.utils.fp8_utils import (
        get_col_major_tma_aligned_packed_tensor,
        unpack_col_major_tma_aligned_packed_tensor)

    # Build a random UE8M0 scale tensor: 2^exp where exp in [-10, 10].
    exponents = torch.randint(-10, 11, (mn, k), device='cuda')
    original = torch.pow(2.0, exponents.float())

    packed = get_col_major_tma_aligned_packed_tensor(original)
    recovered = unpack_col_major_tma_aligned_packed_tensor(packed, mn, k)

    torch.testing.assert_close(recovered,
                               original,
                               atol=0.0,
                               rtol=0.0,
                               msg=f"Round-trip failed for ({mn}, {k})")


@pytest.mark.parametrize("out_features,in_features", [
    (256, 256),
    (384, 512),
    (1024, 5120),
    (768, 13824),
    (5120, 5120),
])
def test_inverse_transform_sf_round_trip(out_features, in_features):
    """transform_sf → inverse_transform_sf should recover block-scale grid."""
    from tensorrt_llm.quantization.utils.fp8_utils import (
        inverse_transform_sf, transform_sf_into_required_layout)

    block_size = 128
    nb_m = math.ceil(out_features / block_size)
    nb_k = math.ceil(in_features / block_size)

    exponents = torch.randint(-10, 11, (nb_m, nb_k), device='cuda')
    original_scale = torch.pow(2.0, exponents.float())

    packed = transform_sf_into_required_layout(
        original_scale.clone(),
        mn=out_features,
        k=in_features,
        recipe=(1, 128, 128),
        is_sfa=False,
    )

    recovered = inverse_transform_sf(
        packed,
        mn=out_features,
        k=in_features,
        block_size=block_size,
    )

    torch.testing.assert_close(
        recovered,
        original_scale,
        atol=0.0,
        rtol=0.0,
        msg=f"inverse_transform_sf failed for ({out_features}, {in_features})")

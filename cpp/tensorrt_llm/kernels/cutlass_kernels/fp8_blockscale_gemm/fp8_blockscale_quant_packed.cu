/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Fused 1x128 FP8 quantize + UE8M0 scale packing.
//
// Replaces the (scale_1x128_kernel + pack_fp32_into_ue8m0) two-kernel sequence
// used by SM100 deep_gemm fp8 block-scale GEMMs. Adapted from the SM120 MoE
// in-kernel packing pattern (`scale_1x128_kernel_sm120` in
// sm120_blockwise_gemm/sm120_fp8_moe_gemm_1d1d.cuh), specialised for the
// non-MoE case (single contiguous batch, no token offsets).

#include "fp8_blockscale_quant_packed.h"

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::fp8_blockscale_gemm
{

namespace
{

__device__ __forceinline__ float reciprocal_approximate_ftz_local(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

// Each warp consumes one row × 4 quantization blocks (4 × 128 = 512 K elems).
// 32 lanes split into 4 lane-groups of 8: each group covers 1 quant block
// (8 lanes × 16 BF16 elems = 128 elems). After per-block amax, lanes
// 0/8/16/24 each hold one UE8M0 scale byte; lane 0 packs them into a uint32
// and stores in the deep_gemm-expected MN-major layout.
template <int WarpsPerBlock>
__global__ void fp8_quantize_1x128_packed_kernel_impl(__nv_fp8_e4m3* __restrict__ fp8_output,
    int32_t* __restrict__ packed_scale_output, __nv_bfloat16 const* __restrict__ input, int const m, int const k,
    int const scale_leading_dim_uint32)
{
    int const packed_sf_k_idx = static_cast<int>(blockIdx.x);
    int const warp_id = static_cast<int>(threadIdx.x) >> 5;
    int const lane_id = static_cast<int>(threadIdx.x) & 31;
    int const m_idx = static_cast<int>(blockIdx.y) * WarpsPerBlock + warp_id;

    if (m_idx >= m)
    {
        return;
    }

    int const k_base = packed_sf_k_idx * 512 + lane_id * 16;

    // ---- 1. Load 16 BF16 elements per lane. ----
    auto const* in_ptr = reinterpret_cast<double4 const*>(input + static_cast<int64_t>(m_idx) * k + k_base);
    constexpr int kLoadNumElems = sizeof(double4) / sizeof(__nv_bfloat16); // 16

    union LoadTrick
    {
        double4 pack;
        __nv_bfloat16 v[kLoadNumElems];
    };

    LoadTrick load_trick;
    bool const k_in_range = (k_base < k);
    load_trick.pack = k_in_range ? in_ptr[0] : double4{};

    if (k_in_range && k_base + kLoadNumElems > k)
    {
        int const valid = k - k_base;
#pragma unroll
        for (int i = 0; i < kLoadNumElems; ++i)
        {
            if (i >= valid)
            {
                load_trick.v[i] = __nv_bfloat16(0.0f);
            }
        }
    }

    // ---- 2. Per-block amax (lanes 0..7 / 8..15 / 16..23 / 24..31 = 4 quant blocks). ----
    __nv_bfloat16 max_elem = __nv_bfloat16(0.0f);
#pragma unroll
    for (int i = 0; i < kLoadNumElems; ++i)
    {
        max_elem = __hmax(max_elem, __habs(load_trick.v[i]));
    }
    float amax = static_cast<float>(max_elem);
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 4, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1, 8));
    amax = fmaxf(amax, 1e-10f);

    // ---- 3. UE8M0 dequant scale. ----
    float const dequant_scale_raw = amax * reciprocal_approximate_ftz_local(448.0f);
    __nv_fp8_e8m0 ue8m0_scale;
    ue8m0_scale.__x = __nv_cvt_float_to_e8m0(dequant_scale_raw, __NV_SATFINITE, cudaRoundPosInf);

    // Recover quant_scale = 1 / 2^(exp - 127) for fp8 conversion.
    constexpr uint32_t FP32_EXPONENT_BIAS = 127u;
    float const quant_scale = (ue8m0_scale.__x == 0)
        ? 1.0f
        : exp2f(static_cast<float>(FP32_EXPONENT_BIAS) - static_cast<float>(ue8m0_scale.__x));

    // ---- 4. Quantize and store FP8 output. ----
    constexpr int kStoreNumElems = sizeof(float4) / sizeof(__nv_fp8_e4m3); // 16

    union StoreTrick
    {
        float4 pack;
        __nv_fp8_e4m3 v[kStoreNumElems];
    };

    StoreTrick store_trick;
    store_trick.pack = float4{};
#pragma unroll
    for (int i = 0; i < kStoreNumElems; ++i)
    {
        store_trick.v[i] = __nv_fp8_e4m3(static_cast<float>(load_trick.v[i]) * quant_scale);
    }
    auto* out_ptr = reinterpret_cast<float4*>(fp8_output + static_cast<int64_t>(m_idx) * k + k_base);
    if (k_in_range)
    {
        if (k_base + kStoreNumElems > k)
        {
            int const valid = k - k_base;
#pragma unroll
            for (int i = 0; i < kStoreNumElems; ++i)
            {
                if (i >= valid)
                {
                    store_trick.v[i] = __nv_fp8_e4m3(0.0f);
                }
            }
        }
        out_ptr[0] = store_trick.pack;
    }

    // ---- 5. Pack 4 UE8M0 scales (lanes 0/8/16/24) and store. ----
    uint32_t const s0 = __shfl_sync(0xFFFFFFFFu, static_cast<uint32_t>(ue8m0_scale.__x), 0);
    uint32_t const s1 = __shfl_sync(0xFFFFFFFFu, static_cast<uint32_t>(ue8m0_scale.__x), 8);
    uint32_t const s2 = __shfl_sync(0xFFFFFFFFu, static_cast<uint32_t>(ue8m0_scale.__x), 16);
    uint32_t const s3 = __shfl_sync(0xFFFFFFFFu, static_cast<uint32_t>(ue8m0_scale.__x), 24);
    if (lane_id == 0)
    {
        // Mask off scale bytes whose sf_k is past the actual K.
        int const num_sf_k = (k + 127) / 128;
        int const sf_k_base = packed_sf_k_idx * 4;
        uint32_t packed = 0u;
        if (sf_k_base + 0 < num_sf_k)
            packed |= s0;
        if (sf_k_base + 1 < num_sf_k)
            packed |= (s1 << 8);
        if (sf_k_base + 2 < num_sf_k)
            packed |= (s2 << 16);
        if (sf_k_base + 3 < num_sf_k)
            packed |= (s3 << 24);
        // Layout: packed_scale[packed_sf_k_idx, m_idx]
        packed_scale_output[static_cast<int64_t>(packed_sf_k_idx) * scale_leading_dim_uint32 + m_idx] = packed;
    }
}

} // namespace

void launch_fp8_quantize_1x128_packed_bf16_e4m3(__nv_fp8_e4m3* fp8_output, int32_t* packed_scale_output,
    __nv_bfloat16 const* input, int m, int k, int scale_leading_dim_uint32, cudaStream_t stream)
{
    constexpr int kWarpsPerBlock = 4;
    int const num_packed_sf_k = (((k + 127) / 128) + 3) / 4;
    int const m_blocks = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
    dim3 const grid(num_packed_sf_k, m_blocks, 1);
    dim3 const block(kWarpsPerBlock * 32, 1, 1);
    fp8_quantize_1x128_packed_kernel_impl<kWarpsPerBlock>
        <<<grid, block, 0, stream>>>(fp8_output, packed_scale_output, input, m, k, scale_leading_dim_uint32);
}

} // namespace kernels::fp8_blockscale_gemm

TRTLLM_NAMESPACE_END

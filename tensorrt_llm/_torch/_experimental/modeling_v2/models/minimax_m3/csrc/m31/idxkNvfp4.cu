/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// Writer (unit-scale recipe of vLLM's fused qknorm/rope/kv-insert kernel, for inputs exactly representable in bf16:
// bf16 rows, or E4M3 rows, which are a subset):
//   sf = E4M3_RN_satfinite(max(RN(amax * RN(1/6)), 2^-9)); q = RZ(x * rcp.approx.ftz(sf));
//   code = cvt.rn.satfinite.e2m1x2.f32(q); a zero magnitude is stored as +0.
// The recipe lives in idxkNvfp4Quant.cuh, which the horizontal producers' NVFP4 index-K mode shares: their pages equal
// this writer's on their bf16-rounded index-K rows.
// Gather: each element becomes E4M3_RN_satfinite(e2m1 * scale) (exact fp32 product, one rounding), the key the
// Q8KV4 decode indexer dequantizes to off SM107, so the prefill proxy scores the same keys vLLM's prefill does.

#include "minimax_m3/csrc/m31/idxkNvfp4.h"
#include "minimax_m3/csrc/m31/idxkNvfp4Quant.cuh"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::minimaxM31
{

namespace
{

constexpr int kPage = 128;
constexpr int kDim = 128;
constexpr int kDataBytes = kPage * kDim / 2;                         // 8192
constexpr int kScaleBytesPerToken = kDim / 16;                       // 8
constexpr int kPageBytes = kDataBytes + kPage * kScaleBytesPerToken; // 9216
static_assert(kPageBytes == kIdxkNvfp4PageBytes, "one NVFP4 index-K page layout");

__device__ __forceinline__ float loadElem(__nv_bfloat16 const* p)
{
    return __bfloat162float(*p);
}

__device__ __forceinline__ float loadElem(__nv_fp8_e4m3 const* p)
{
    return float(*p);
}

// One warp per token: lane l holds elements [4l, 4l + 4) of group l / 4.
template <typename T>
__global__ void writeKernel(T const* __restrict__ rows, int64_t rowStride, int64_t const* __restrict__ loc64,
    int32_t const* __restrict__ loc32, int numTokens, uint8_t* __restrict__ pages, int64_t pageStride,
    int64_t rowsPageStride, bool rowsBySlot)
{
    int const warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int const lane = threadIdx.x & 31;
    if (warp >= numTokens)
    {
        return;
    }
    int64_t const slot = loc64 != nullptr ? loc64[warp] : static_cast<int64_t>(loc32[warp]);
    if (slot < 0)
    {
        return;
    }
    // rowsBySlot: the rows are an FP8 index-K pool [pages, 1, 128, 128] (the producer's sink) read at the token's slot.
    T const* src = rowsBySlot ? rows + (slot / kPage) * rowsPageStride + (slot % kPage) * rowStride + lane * 4
                              : rows + warp * rowStride + lane * 4;
    float x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        x[i] = loadElem(src + i);
    }
    int64_t const page = slot / kPage;
    int const t = static_cast<int>(slot % kPage);
    idxkNvfp4QuantStore(x, lane, pages + page * pageStride, t);
}

// E2M1 code -> value, arithmetically (no table): magnitude m = code & 7 is 0.5 * m for m < 2, else
// (1 + (m & 1) / 2) * 2^((m >> 1) - 1); bit 3 is the sign, applied by a multiply so code 8 stays -0.
__device__ __forceinline__ float e2m1ToFloat(uint32_t code)
{
    uint32_t const m = code & 7u;
    float const normal = __uint_as_float((((m >> 1) + 126u) << 23) | ((m & 1u) << 22));
    float const mag = m < 2u ? 0.5f * static_cast<float>(m) : normal;
    return mag * ((code & 8u) ? -1.f : 1.f);
}

// Two E2M1 codes (low nibble = even element) times an E4M3 scale -> two E4M3 bytes (even element low): the exact fp32
// product, clamped to +-448, rounded once.
__device__ __forceinline__ uint32_t dequantPair(uint32_t byte, float scale)
{
    float lo = e2m1ToFloat(byte & 0xfu) * scale;
    float hi = e2m1ToFloat(byte >> 4) * scale;
    lo = fminf(fmaxf(lo, -448.f), 448.f);
    hi = fminf(fmaxf(hi, -448.f), 448.f);
    uint16_t out;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(out) : "f"(hi), "f"(lo));
    return out;
}

// One block per gathered page, 512 threads: thread i takes packed bytes [16 i, 16 i + 16) = elements
// [32 (i % 4), 32 (i % 4) + 32) of token i / 4 (two 16-element groups) and stores 32 E4M3 bytes.
__global__ void gatherKernel(uint8_t const* __restrict__ pages, int64_t pageStride, int32_t const* __restrict__ pageIds,
    __nv_fp8_e4m3* __restrict__ out)
{
    int const dst = blockIdx.x;
    int const tid = threadIdx.x;
    int const token = tid >> 2;
    int const quarter = tid & 3;
    uint4* o = reinterpret_cast<uint4*>(out + static_cast<int64_t>(dst) * kPage * kDim + token * kDim + quarter * 32);
    int const page = pageIds[dst];
    if (page < 0)
    {
        o[0] = make_uint4(0, 0, 0, 0);
        o[1] = make_uint4(0, 0, 0, 0);
        return;
    }
    uint8_t const* base = pages + static_cast<int64_t>(page) * pageStride;
    uint4 const d = *reinterpret_cast<uint4 const*>(base + tid * 16);
    uint16_t const sc
        = *reinterpret_cast<uint16_t const*>(base + kDataBytes + token * kScaleBytesPerToken + quarter * 2);
    __nv_fp8_e4m3 s0, s1;
    s0.__x = static_cast<uint8_t>(sc & 0xffu);
    s1.__x = static_cast<uint8_t>(sc >> 8);
    float const scale0 = float(s0);
    float const scale1 = float(s1);
    uint32_t const in[4] = {d.x, d.y, d.z, d.w};
    uint32_t res[8];
#pragma unroll
    for (int w = 0; w < 4; ++w)
    {
        // Word w holds elements [8 w, 8 w + 8): group 0 for w < 2, group 1 after.
        float const scale = w < 2 ? scale0 : scale1;
        uint32_t const v = in[w];
        res[2 * w] = dequantPair(v & 0xffu, scale) | (dequantPair((v >> 8) & 0xffu, scale) << 16);
        res[2 * w + 1] = dequantPair((v >> 16) & 0xffu, scale) | (dequantPair(v >> 24, scale) << 16);
    }
    o[0] = make_uint4(res[0], res[1], res[2], res[3]);
    o[1] = make_uint4(res[4], res[5], res[6], res[7]);
}

} // namespace

cudaError_t invokeNvfp4IdxkWrite(void const* rows, bool rowsFp8, int64_t rowStride, int64_t rowsPageStride,
    bool rowsBySlot, int64_t const* loc64, int32_t const* loc32, int numTokens, uint8_t* pages, int64_t pageStride,
    cudaStream_t stream)
{
    if (numTokens == 0)
    {
        return cudaSuccess;
    }
    constexpr int kThreads = 256;
    int const blocks = (numTokens * 32 + kThreads - 1) / kThreads;
    if (rowsFp8)
    {
        writeKernel<__nv_fp8_e4m3><<<blocks, kThreads, 0, stream>>>(static_cast<__nv_fp8_e4m3 const*>(rows), rowStride,
            loc64, loc32, numTokens, pages, pageStride, rowsPageStride, rowsBySlot);
    }
    else
    {
        writeKernel<__nv_bfloat16><<<blocks, kThreads, 0, stream>>>(static_cast<__nv_bfloat16 const*>(rows), rowStride,
            loc64, loc32, numTokens, pages, pageStride, rowsPageStride, rowsBySlot);
    }
    return cudaGetLastError();
}

cudaError_t invokeNvfp4IdxkGatherFp8(
    uint8_t const* pages, int64_t pageStride, int32_t const* pageIds, int numPages, uint8_t* out, cudaStream_t stream)
{
    if (numPages == 0)
    {
        return cudaSuccess;
    }
    gatherKernel<<<numPages, 512, 0, stream>>>(pages, pageStride, pageIds, reinterpret_cast<__nv_fp8_e4m3*>(out));
    return cudaGetLastError();
}

} // namespace kernels::minimaxM31

TRTLLM_MV2_NAMESPACE_END

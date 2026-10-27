/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// TP2CTX_SP_ADDNORM: the SP boundary's owned-row peer add (tp2ctxPeerAddKernel) and the residual RMSNorm + MXFP8
// (+ NVFP4, + FP32 router copy) quantize (ctxResidualRmsNormMxfp8Kernel) in one kernel. Each owned row is the
// two-rank sum of this rank's slot row and the peer's slot row (read over NVLink), rounded to BF16 exactly as the
// peer add rounds it; from there the arithmetic is the NORMQ kernel's, statement for statement, so every output is
// bit for bit the two-kernel sequence's. The summed row never round-trips through HBM.

#include "tensorrt_llm/kernels/minimaxM31/spAddNormqSm107.h"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <algorithm>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::minimaxM31Sm107
{

namespace
{

constexpr int kPackedSize = 8;
constexpr int kWarpSize = 32;

union PackedBf16
{
    int4 packed;
    __nv_bfloat162 unpacked[4];
};

// ctxResidualRmsNormMxfp8Sm103.cu's helpers, unchanged.
__device__ __forceinline__ float warpReduceSum(float val)
{
    val += __shfl_xor_sync(~0, val, 16);
    val += __shfl_xor_sync(~0, val, 8);
    val += __shfl_xor_sync(~0, val, 4);
    val += __shfl_xor_sync(~0, val, 2);
    val += __shfl_xor_sync(~0, val, 1);
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val)
{
    __shared__ float smem[kWarpSize];
    int const laneId = threadIdx.x % kWarpSize;
    int const warpId = threadIdx.x / kWarpSize;
    int const warpNum = blockDim.x / kWarpSize;
    val = warpReduceSum(val);
    if (laneId == 0)
    {
        smem[warpId] = val;
    }
    __syncthreads();
    val = laneId < warpNum ? smem[laneId] : 0.f;
    val = warpReduceSum(val);
    return val;
}

__device__ __forceinline__ int4 add128b(PackedBf16& a, PackedBf16& b)
{
    PackedBf16 c;
    c.unpacked[0] = a.unpacked[0] + b.unpacked[0];
    c.unpacked[1] = a.unpacked[1] + b.unpacked[1];
    c.unpacked[2] = a.unpacked[2] + b.unpacked[2];
    c.unpacked[3] = a.unpacked[3] + b.unpacked[3];
    return c.packed;
}

__device__ __forceinline__ float accumulate(float acc, PackedBf16& vec)
{
#pragma unroll
    for (int i = 0; i < kPackedSize; ++i)
    {
        float v = static_cast<float>(reinterpret_cast<__nv_bfloat16*>(vec.unpacked)[i]);
        acc += v * v;
    }
    return acc;
}

__device__ __forceinline__ int4 rmsNorm(float denom, PackedBf16& vec, PackedBf16& weight)
{
    PackedBf16 ret;
#pragma unroll
    for (int i = 0; i < kPackedSize; ++i)
    {
        float v1 = static_cast<float>(reinterpret_cast<__nv_bfloat16*>(vec.unpacked)[i]);
        float v2 = static_cast<float>(reinterpret_cast<__nv_bfloat16*>(weight.unpacked)[i]);
        reinterpret_cast<__nv_bfloat16*>(ret.unpacked)[i] = static_cast<__nv_bfloat16>(v1 * denom * v2);
    }
    return ret.packed;
}

// tp2ctxPeerAddKernel's per-16-byte sum: both operands widened to FP32, added, rounded to BF16 (RN).
__device__ __forceinline__ int4 peerSum(int4 const x, int4 const y)
{
    int4 r;
    auto const* xa = reinterpret_cast<__nv_bfloat162 const*>(&x);
    auto const* ya = reinterpret_cast<__nv_bfloat162 const*>(&y);
    auto* ra = reinterpret_cast<__nv_bfloat162*>(&r);
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        float2 const fx = __bfloat1622float2(xa[j]);
        float2 const fy = __bfloat1622float2(ya[j]);
        ra[j] = __floats2bfloat162_rn(fx.x + fy.x, fx.y + fy.y);
    }
    return r;
}

//! ctxResidualRmsNormMxfp8Kernel<kWriteNorm, kFp4, kF32> with inter = peerSum(local row, peer row); interOut gets
//! inter + residual (the NORMQ kernel's in-place inter update).
template <bool kWriteNorm, bool kFp4, bool kF32>
__global__ void spAddNormqKernel(__nv_bfloat16 const* local, __nv_bfloat16 const* peer, __nv_bfloat16* interOut,
    __nv_bfloat16 const* residual, __nv_bfloat16 const* weight, __nv_bfloat16* normOut, uint64_t* qOut, uint32_t* sfOut,
    uint32_t* fp4Out, uint32_t* fp4SfOut, float const* fp4GlobalScale, float* f32Out, int rows, int hidden, float eps)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static constexpr int kThreadsPerSf = 32 / CVT_ELTS_PER_THREAD;
    int const colIdx = threadIdx.x;
    int const offset = colIdx * kPackedSize;
    int const colThreads = hidden / kPackedSize;
    std::optional<int> const batchIdx = 0;
    std::optional<int> const numRows = rows;

    cudaGridDependencySynchronize();

    float const fp4SfScale = kFp4 ? fp4GlobalScale[0] : 1.0f;
    PackedBf16 weightVec;
    weightVec.packed = *reinterpret_cast<int4 const*>(weight + offset);
    int4 localVec, peerVec;
    PackedBf16 interVec, residualVec;
    int row = blockIdx.x;
    if (row < rows)
    {
        int64_t const o = static_cast<int64_t>(row) * hidden + offset;
        localVec = *reinterpret_cast<int4 const*>(local + o);
        peerVec = *reinterpret_cast<int4 const*>(peer + o);
        residualVec.packed = *reinterpret_cast<int4 const*>(residual + o);
    }
    for (; row < rows; row += gridDim.x)
    {
        int64_t const rowOffset = static_cast<int64_t>(row) * hidden + offset;
        int const next = row + gridDim.x;
        int4 nextLocalVec, nextPeerVec;
        PackedBf16 nextResidualVec;
        if (next < rows)
        {
            int64_t const o = static_cast<int64_t>(next) * hidden + offset;
            nextLocalVec = *reinterpret_cast<int4 const*>(local + o);
            nextPeerVec = *reinterpret_cast<int4 const*>(peer + o);
            nextResidualVec.packed = *reinterpret_cast<int4 const*>(residual + o);
        }
        interVec.packed = peerSum(localVec, peerVec);
        interVec.packed = add128b(interVec, residualVec);
        *reinterpret_cast<int4*>(interOut + rowOffset) = interVec.packed;
        float acc = accumulate(0.f, interVec);
        acc = blockReduceSum(acc);
        float const denom = rsqrtf(acc / hidden + eps);
        PackedBf16 outVec;
        outVec.packed = rmsNorm(denom, interVec, weightVec);
        if constexpr (kWriteNorm)
        {
            *reinterpret_cast<int4*>(normOut + rowOffset) = outVec.packed;
        }
        if constexpr (kF32)
        {
            float2 const f0 = __bfloat1622float2(outVec.unpacked[0]);
            float2 const f1 = __bfloat1622float2(outVec.unpacked[1]);
            float2 const f2 = __bfloat1622float2(outVec.unpacked[2]);
            float2 const f3 = __bfloat1622float2(outVec.unpacked[3]);
            reinterpret_cast<float4*>(f32Out + rowOffset)[0] = make_float4(f0.x, f0.y, f1.x, f1.y);
            reinterpret_cast<float4*>(f32Out + rowOffset)[1] = make_float4(f2.x, f2.y, f3.x, f3.y);
        }
        PackedVec<__nv_bfloat16> quantIn;
        *reinterpret_cast<int4*>(&quantIn) = outVec.packed;
        auto sf = cvt_quant_get_sf_out_offset<uint32_t, kThreadsPerSf>(
            batchIdx, row, colIdx, numRows, hidden / 32, sfOut, QuantizationSFLayout::SWIZZLED);
        qOut[static_cast<int64_t>(row) * colThreads + colIdx] = cvt_warp_fp16_to_mxfp8<__nv_bfloat16, 32>(quantIn, sf);
        if constexpr (kFp4)
        {
            PackedVec<__nv_bfloat16> fp4In;
            *reinterpret_cast<int4*>(&fp4In) = outVec.packed;
            auto sf4 = cvt_quant_get_sf_out_offset<uint32_t, 16 / CVT_ELTS_PER_THREAD>(
                batchIdx, row, colIdx, numRows, hidden / 16, fp4SfOut, QuantizationSFLayout::LINEAR);
            fp4Out[static_cast<int64_t>(row) * colThreads + colIdx]
                = cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, false>(fp4In, fp4SfScale, sf4);
        }
        __syncthreads();
        localVec = nextLocalVec;
        peerVec = nextPeerVec;
        residualVec.packed = nextResidualVec.packed;
    }
    int const paddedRows = (rows + 127) / 128 * 128;
    for (int padRow = rows + blockIdx.x; padRow < paddedRows; padRow += gridDim.x)
    {
        auto sf = cvt_quant_get_sf_out_offset<uint32_t, kThreadsPerSf>(
            batchIdx, padRow, colIdx, numRows, hidden / 32, sfOut, QuantizationSFLayout::SWIZZLED);
        cvt_store_replicated_sf<1>(sf, 0x00);
    }
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

} // namespace

cudaError_t invokeSpAddNormq(__nv_bfloat16 const* local, __nv_bfloat16 const* peer, __nv_bfloat16* interOut,
    __nv_bfloat16 const* residual, __nv_bfloat16 const* weight, __nv_bfloat16* normOut, uint64_t* qOut, uint32_t* sfOut,
    uint32_t* fp4Out, uint32_t* fp4SfOut, float const* fp4GlobalScale, float* f32Out, int rows, int hidden, float eps,
    int numSms, int ctasPerSm, bool enablePdl, cudaStream_t stream)
{
    cudaLaunchConfig_t config = {0};
    config.gridDim = std::max(1, std::min(rows, numSms * ctasPerSm));
    config.blockDim = hidden / kPackedSize;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = enablePdl;
    config.attrs = attribute;
    config.numAttrs = 1;
    auto launch = [&](auto kernel)
    {
        return cudaLaunchKernelEx(&config, kernel, local, peer, interOut, residual, weight, normOut, qOut, sfOut,
            fp4Out, fp4SfOut, fp4GlobalScale, f32Out, rows, hidden, eps);
    };
    bool const fp4 = fp4GlobalScale != nullptr;
    if (f32Out != nullptr)
    {
        return fp4 ? launch(spAddNormqKernel<true, true, true>) : launch(spAddNormqKernel<true, false, true>);
    }
    if (normOut != nullptr)
    {
        return fp4 ? launch(spAddNormqKernel<true, true, false>) : launch(spAddNormqKernel<true, false, false>);
    }
    return fp4 ? launch(spAddNormqKernel<false, true, false>) : launch(spAddNormqKernel<false, false, false>);
}

} // namespace kernels::minimaxM31Sm107

TRTLLM_NAMESPACE_END

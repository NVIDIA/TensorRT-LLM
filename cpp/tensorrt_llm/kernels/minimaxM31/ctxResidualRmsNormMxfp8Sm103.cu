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

#include "tensorrt_llm/kernels/minimaxM31/ctxResidualRmsNormMxfp8Sm103.h"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <algorithm>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::minimaxM31Sm103
{
namespace
{

constexpr int kPackedSize = 8; // BF16 elements per 16-byte access, = CVT_ELTS_PER_THREAD
constexpr int kWarpSize = 32;

union PackedBf16
{
    int4 packed;
    __nv_bfloat162 unpacked[4];
};

// The helpers below are customAllReduceKernels.cu's (reduce_fusion::warp_reduce_sum / block_reduce_sum,
// accumulate, rms_norm<T, Affine = true>, add128b): the same arithmetic in the same order, so the norm is bitwise
// rms_norm_rows_kernel's.
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

//! rms_norm_rows_kernel<bf16, Bias = false, Residual = true, Affine = true> (persistent rows, next row prefetched,
//! one 8-element chunk per thread) whose per-row output also goes through quantize_mxfp8_rows' per-element math
//! (cvt_warp_fp16_to_mxfp8<bf16, 32>, swizzled K32 scales) while it is still in registers. The scale rows that pad
//! the row count to a multiple of 128 are zero-filled, as quantize_mxfp8_rows does.
template <bool kWriteNorm, bool kFp4>
__global__ void ctxResidualRmsNormMxfp8Kernel(__nv_bfloat16* inter, __nv_bfloat16 const* residual,
    __nv_bfloat16 const* weight, __nv_bfloat16* normOut, uint64_t* qOut, uint32_t* sfOut, uint32_t* fp4Out,
    uint32_t* fp4SfOut, float const* fp4GlobalScale, int rows, int hidden, float eps)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static constexpr int kThreadsPerSf = 32 / CVT_ELTS_PER_THREAD;
    int const colIdx = threadIdx.x;
    int const offset = colIdx * kPackedSize;
    int const colThreads = hidden / kPackedSize;
    std::optional<int> const batchIdx = 0;
    std::optional<int> const numRows = rows;

    cudaGridDependencySynchronize();

    // quantize_with_block_size's per-tensor scale (SFScaleVal) for the NVFP4 copy.
    float const fp4SfScale = kFp4 ? fp4GlobalScale[0] : 1.0f;
    PackedBf16 weightVec;
    weightVec.packed = *reinterpret_cast<int4 const*>(weight + offset);
    PackedBf16 interVec, residualVec;
    int row = blockIdx.x;
    if (row < rows)
    {
        interVec.packed = *reinterpret_cast<int4 const*>(inter + static_cast<int64_t>(row) * hidden + offset);
        residualVec.packed = *reinterpret_cast<int4 const*>(residual + static_cast<int64_t>(row) * hidden + offset);
    }
    for (; row < rows; row += gridDim.x)
    {
        int64_t const rowOffset = static_cast<int64_t>(row) * hidden + offset;
        int const next = row + gridDim.x;
        PackedBf16 nextInterVec, nextResidualVec;
        if (next < rows)
        {
            nextInterVec.packed = *reinterpret_cast<int4 const*>(inter + static_cast<int64_t>(next) * hidden + offset);
            nextResidualVec.packed
                = *reinterpret_cast<int4 const*>(residual + static_cast<int64_t>(next) * hidden + offset);
        }
        interVec.packed = add128b(interVec, residualVec);
        *reinterpret_cast<int4*>(inter + rowOffset) = interVec.packed;
        float acc = accumulate(0.f, interVec);
        acc = blockReduceSum(acc);
        float const denom = rsqrtf(acc / hidden + eps);
        PackedBf16 outVec;
        outVec.packed = rmsNorm(denom, interVec, weightVec);
        if constexpr (kWriteNorm)
        {
            *reinterpret_cast<int4*>(normOut + rowOffset) = outVec.packed;
        }
        PackedVec<__nv_bfloat16> quantIn;
        *reinterpret_cast<int4*>(&quantIn) = outVec.packed;
        auto sf = cvt_quant_get_sf_out_offset<uint32_t, kThreadsPerSf>(
            batchIdx, row, colIdx, numRows, hidden / 32, sfOut, QuantizationSFLayout::SWIZZLED);
        qOut[static_cast<int64_t>(row) * colThreads + colIdx] = cvt_warp_fp16_to_mxfp8<__nv_bfloat16, 32>(quantIn, sf);
        if constexpr (kFp4)
        {
            // fp4_quantize(norm, globalScale, 16, ue8m0 = false, swizzled = false): E2M1 x 8 per thread, E4M3 scale per
            // 16 elements in the linear layout (no row padding).
            PackedVec<__nv_bfloat16> fp4In;
            *reinterpret_cast<int4*>(&fp4In) = outVec.packed;
            auto sf4 = cvt_quant_get_sf_out_offset<uint32_t, 16 / CVT_ELTS_PER_THREAD>(
                batchIdx, row, colIdx, numRows, hidden / 16, fp4SfOut, QuantizationSFLayout::LINEAR);
            fp4Out[static_cast<int64_t>(row) * colThreads + colIdx]
                = cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, false>(fp4In, fp4SfScale, sf4);
        }
        // blockReduceSum's shared memory is written again by the next row.
        __syncthreads();
        interVec.packed = nextInterVec.packed;
        residualVec.packed = nextResidualVec.packed;
    }
    int const paddedRows = (rows + 127) / 128 * 128;
    for (int padRow = rows + blockIdx.x; padRow < paddedRows; padRow += gridDim.x)
    {
        auto sf = cvt_quant_get_sf_out_offset<uint32_t, kThreadsPerSf>(
            batchIdx, padRow, colIdx, numRows, hidden / 32, sfOut, QuantizationSFLayout::SWIZZLED);
        cvt_store_replicated_sf<1>(sf, 0x00);
    }
    // Every thread's stores are visible before any thread reports PDL completion (as in quantize_mxfp8_rows).
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

} // namespace

void invokeCtxResidualRmsNormMxfp8(__nv_bfloat16* inter, __nv_bfloat16 const* residual, __nv_bfloat16 const* weight,
    __nv_bfloat16* normOut, uint64_t* qOut, uint32_t* sfOut, uint32_t* fp4Out, uint32_t* fp4SfOut,
    float const* fp4GlobalScale, int rows, int hidden, float eps, bool enablePdl, cudaStream_t stream)
{
    static int const multiProcessorCount = []
    {
        int device = 0;
        int count = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
        return count;
    }();
    cudaLaunchConfig_t config = {0};
    config.gridDim = std::max(1, std::min(rows, multiProcessorCount));
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
        cudaLaunchKernelEx(&config, kernel, inter, residual, weight, normOut, qOut, sfOut, fp4Out, fp4SfOut,
            fp4GlobalScale, rows, hidden, eps);
    };
    bool const fp4 = fp4Out != nullptr;
    if (normOut != nullptr)
    {
        fp4 ? launch(ctxResidualRmsNormMxfp8Kernel<true, true>) : launch(ctxResidualRmsNormMxfp8Kernel<true, false>);
    }
    else
    {
        fp4 ? launch(ctxResidualRmsNormMxfp8Kernel<false, true>) : launch(ctxResidualRmsNormMxfp8Kernel<false, false>);
    }
}

} // namespace kernels::minimaxM31Sm103

TRTLLM_NAMESPACE_END

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

// The target's copy of TensorRT-LLM's cuda-core NVFP4 GEMM (kernels/weightOnlyBatchedGemv/
// cudaCoreGemmNVFP4.cu), specialized to the calls it makes: BF16 output, no bias, activation scales
// in the 128x4 swizzled layout read in place, weights streamed with L1 no-allocate loads, and the
// first K step's weight loads issued before the PDL wait. The per-element math and the reduction
// order are the stock kernel's.

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/cudaCoreNvfp4Gemm.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int kBlockSize = 128;
constexpr int kTileN = 2;
constexpr int kSfVec = 16;
// 16-byte loads of packed E2M1: 32 elements per thread per K step.
constexpr int kStepK = 32;
constexpr int kStepKScale = kStepK / kSfVec;
constexpr int kTileK = kStepK * kBlockSize;

// Offset of scale-factor column colIdx of row rowIdx in the 128x4 swizzled SF layout (the math of
// get_sf_out_offset_128x4 without batching). colIdx is even here, so the SF pair (colIdx, colIdx + 1)
// is contiguous and is read with one 2-byte load.
__device__ __forceinline__ int swizzledSfOffset(int rowIdx, int colIdx, int numSfTilesK)
{
    int const tileOffset = ((rowIdx / 128) * numSfTilesK + colIdx / 4) * 512;
    return tileOffset + (rowIdx % 32) * 16 + ((rowIdx % 128) / 32) * 4 + colIdx % 4;
}

// 16-byte weight load that does not allocate in L1. The weights are read exactly once, so allocating
// them in L1 only evicts the activation and scale lines every CTA on the SM re-reads. Same bits as a
// plain load.
__device__ __forceinline__ int4 loadWeightNoL1(void const* ptr)
{
    int4 r;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(ptr));
    return r;
}

__device__ __forceinline__ void sfPairToFloat(__nv_fp8x2_e4m3 v, float* out)
{
    char2 const tmp = reinterpret_cast<char2 const&>(v);
    out[0] = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.x));
    out[1] = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.y));
}

// PRE_PDL_WEIGHT_LOADS: issue the first K step's weight and weight-SF loads (constant data) before
// the PDL wait, with that step peeled. Same FMAs in the same order either way, so the output is
// bitwise identical.
template <int TILE_M, bool PRE_PDL_WEIGHT_LOADS>
__global__ void __launch_bounds__(kBlockSize)
    cudaCoreNvfp4GemmKernel(uint8_t const* __restrict__ act, uint8_t const* __restrict__ weight,
        __nv_fp8_e4m3 const* __restrict__ scaleA, __nv_fp8_e4m3 const* __restrict__ scaleW,
        float const* __restrict__ alphaPtr, __nv_bfloat16* __restrict__ output, int n, int k)
{
    using Converter = cutlass::NumericArrayConverter<float, cutlass::float_e2m1_t, 8>;
    using CvtSrcType = typename Converter::source_type;
    using CvtResType = typename Converter::result_type;
    constexpr int kCvtCount = static_cast<int>(sizeof(int4) / sizeof(CvtSrcType));
    static_assert(kCvtCount * 8 == kStepK);

    float const alpha = alphaPtr[0];
    int const tileIdM = static_cast<int>(blockIdx.y) * TILE_M;
    int const tileIdN = static_cast<int>(blockIdx.x) * kTileN;
    int const tid = static_cast<int>(threadIdx.x);
    float tileA[kStepK];
    float tileW[kTileN * kStepK];
    float tileAScale[kStepKScale];
    float tileWScale[kTileN * kStepKScale];
    float acc[TILE_M * kTileN];
#pragma unroll
    for (int i = 0; i < TILE_M * kTileN; ++i)
    {
        acc[i] = 0.f;
    }
    act += static_cast<int64_t>(tileIdM) * k / 2;
    weight += static_cast<int64_t>(tileIdN) * k / 2;
    output += static_cast<int64_t>(tileIdM) * n + tileIdN;
    int const numSfTilesK = (k / kSfVec + 3) / 4;

    int4 tileWQuantized[kTileN];
    __nv_fp8x2_e4m3 tileWScaleFp8x2[kTileN];
    auto loadWeightStep = [&](int idxK)
    {
#pragma unroll
        for (int j = 0; j < kTileN; ++j)
        {
            tileWQuantized[j] = loadWeightNoL1(weight + (static_cast<int64_t>(j) * k + idxK) / 2);
        }
#pragma unroll
        for (int j = 0; j < kTileN; ++j)
        {
            int const idx = swizzledSfOffset(tileIdN + j, idxK / kSfVec, numSfTilesK);
            tileWScaleFp8x2[j] = reinterpret_cast<__nv_fp8x2_e4m3 const*>(scaleW + idx)[0];
        }
    };
    auto convertWeightStep = [&]()
    {
#pragma unroll
        for (int j = 0; j < kTileN; ++j)
        {
#pragma unroll
            for (int c = 0; c < kCvtCount; ++c)
            {
                reinterpret_cast<CvtResType*>(tileW)[j * kCvtCount + c]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tileWQuantized[j])[c]);
            }
            sfPairToFloat(tileWScaleFp8x2[j], tileWScale + j * kStepKScale);
        }
    };
    auto actFmaStep = [&](int idxK)
    {
#pragma unroll
        for (int i = 0; i < TILE_M; ++i)
        {
            int4 const tileAQuantized
                = reinterpret_cast<int4 const*>(act + (static_cast<int64_t>(i) * k + idxK) / 2)[0];
#pragma unroll
            for (int c = 0; c < kCvtCount; ++c)
            {
                reinterpret_cast<CvtResType*>(tileA)[c]
                    = Converter::convert(reinterpret_cast<CvtSrcType const*>(&tileAQuantized)[c]);
            }
            int const aSfIdx = swizzledSfOffset(tileIdM + i, idxK / kSfVec, numSfTilesK);
            sfPairToFloat(reinterpret_cast<__nv_fp8x2_e4m3 const*>(scaleA + aSfIdx)[0], tileAScale);
#pragma unroll
            for (int j = 0; j < kTileN; ++j)
            {
#pragma unroll
                for (int l = 0; l < kStepK; ++l)
                {
                    acc[i * kTileN + j] = fma(alpha * tileA[l] * tileAScale[l / kSfVec],
                        tileW[j * kStepK + l] * tileWScale[j * kStepKScale + l / kSfVec], acc[i * kTileN + j]);
                }
            }
        }
    };

    int const firstIdxK = tid * kStepK;
    if constexpr (!PRE_PDL_WEIGHT_LOADS)
    {
        cudaGridDependencySynchronize();
        for (int idxK = firstIdxK; idxK < k; idxK += kTileK)
        {
            loadWeightStep(idxK);
            convertWeightStep();
            actFmaStep(idxK);
        }
    }
    else
    {
        if (firstIdxK < k)
        {
            loadWeightStep(firstIdxK);
        }
        cudaGridDependencySynchronize();
        if (firstIdxK < k)
        {
            convertWeightStep();
            actFmaStep(firstIdxK);
        }
        for (int idxK = firstIdxK + kTileK; idxK < k; idxK += kTileK)
        {
            loadWeightStep(idxK);
            convertWeightStep();
            actFmaStep(idxK);
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    constexpr int kWarpNum = kBlockSize / 32;
    int const warpId = tid / 32;
    int const laneId = tid % 32;
    __shared__ float shmem[TILE_M * kTileN * kWarpNum];
    __shared__ typename WarpReduce::TempStorage tempStorage[kWarpNum];
#pragma unroll
    for (int mi = 0; mi < TILE_M; ++mi)
    {
#pragma unroll
        for (int ni = 0; ni < kTileN; ++ni)
        {
            float const val = WarpReduce(tempStorage[warpId]).Sum(acc[mi * kTileN + ni]);
            if (laneId == 0)
            {
                shmem[mi * kTileN + ni + warpId * TILE_M * kTileN] = val;
            }
        }
    }
    __syncthreads();
    for (int ii = tid; ii < TILE_M * kTileN; ii += kBlockSize)
    {
        int const mid = ii / kTileN;
        int const nid = ii % kTileN;
        float val = 0.f;
#pragma unroll
        for (int jj = 0; jj < kWarpNum; ++jj)
        {
            val += shmem[jj * TILE_M * kTileN + ii];
        }
        output[static_cast<int64_t>(mid) * n + nid] = static_cast<__nv_bfloat16>(val);
    }
    cudaTriggerProgrammaticLaunchCompletion();
}

template <int TILE_M>
void launch(void const* a, void const* w, void const* sfA, void const* sfW, float const* alpha, void* out, int n, int k,
    bool enablePdl, cudaStream_t stream)
{
    // TILE_M >= 2 always loads the first step's weights before the PDL wait; one row only when every
    // thread runs >= 4 K steps, where the later steps' weight stream overlaps the producer's tail.
    auto kernel = (TILE_M >= 2 || k >= 4 * kTileK) ? &cudaCoreNvfp4GemmKernel<TILE_M, true>
                                                   : &cudaCoreNvfp4GemmKernel<TILE_M, false>;
    cudaLaunchConfig_t config = {};
    // N rides grid.x: the quantized LM head is far wider than grid.y allows.
    config.gridDim = dim3(n / kTileN, 1);
    config.blockDim = dim3(kBlockSize);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = enablePdl ? 1 : 0;
    config.attrs = attribute;
    config.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, static_cast<uint8_t const*>(a), static_cast<uint8_t const*>(w),
        static_cast<__nv_fp8_e4m3 const*>(sfA), static_cast<__nv_fp8_e4m3 const*>(sfW), alpha,
        static_cast<__nv_bfloat16*>(out), n, k));
}

template <int TILE_M>
bool dispatchRows(void const* a, void const* w, void const* sfA, void const* sfW, float const* alpha, void* out, int m,
    int n, int k, bool enablePdl, cudaStream_t stream)
{
    if (m == TILE_M)
    {
        launch<TILE_M>(a, w, sfA, sfW, alpha, out, n, k, enablePdl, stream);
        return true;
    }
    if constexpr (TILE_M < kCudaCoreNvfp4GemmMaxRows)
    {
        return dispatchRows<TILE_M + 1>(a, w, sfA, sfW, alpha, out, m, n, k, enablePdl, stream);
    }
    return false;
}

} // namespace

bool cudaCoreNvfp4Gemm(void const* a, void const* w, void const* sfA, void const* sfW, float const* alpha, void* out,
    int m, int n, int k, bool enablePdl, cudaStream_t stream)
{
    if (m < 1 || n % kTileN != 0 || k % kStepK != 0)
    {
        return false;
    }
    return dispatchRows<1>(a, w, sfA, sfW, alpha, out, m, n, k, enablePdl, stream);
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END

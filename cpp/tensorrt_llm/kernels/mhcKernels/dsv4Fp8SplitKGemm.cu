/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "dsv4Fp8SplitKGemm.cuh"
#include "mhcKernels.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::mhc
{
namespace
{

constexpr int kDsv4ProN = 7168;
constexpr int kDsv4ProK = 16384;
constexpr int kBlockN = 128;
constexpr int kBlockK = 128;
constexpr int kClusterSize = 2;
constexpr int kNumSmsForSwizzle = 148;
constexpr int kNumThreads = 256;
constexpr int kSmemCapacity = 232448;
constexpr int kMaxPipelineStages = 32;

CUtensorMap makeTma2D(void const* base, CUtensorMapDataType dtype, uint64_t gmemInner, uint64_t gmemOuter,
    uint32_t smemInner, uint32_t smemOuter, uint64_t gmemOuterStrideBytes, CUtensorMapSwizzle swizzle)
{
    if (swizzle != CU_TENSOR_MAP_SWIZZLE_NONE)
    {
        uint32_t const elementBytes = dtype == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 ? 2u
            : dtype == CU_TENSOR_MAP_DATA_TYPE_INT32                            ? 4u
                                                                                : 1u;
        smemInner = 128u / elementBytes;
    }

    CUtensorMap tensorMap;
    uint64_t const gmemDims[2] = {gmemInner, gmemOuter};
    uint64_t const gmemStrides[1] = {gmemOuterStrideBytes};
    uint32_t const smemDims[2] = {smemInner, smemOuter};
    uint32_t const elementStrides[2] = {1, 1};
    CUresult const result = cuTensorMapEncodeTiled(&tensorMap, dtype, 2, const_cast<void*>(base), gmemDims, gmemStrides,
        smemDims, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TLLM_CHECK_WITH_INFO(
        result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for DSV4 O_b split-K (%d)", static_cast<int>(result));
    return tensorMap;
}

template <int BlockM>
constexpr int numPipelineStages()
{
    constexpr int smemCd = 2 * 16 * kBlockN * static_cast<int>(sizeof(__nv_bfloat16));
    constexpr int smemBarrierReserve = (kMaxPipelineStages * 3 + 4) * 8;
    constexpr int smemTmemPointer = 4;
    constexpr int smemPerStage
        = (BlockM / kClusterSize) * kBlockK + kBlockN * kBlockK + 2 * 128 * static_cast<int>(sizeof(uint32_t));
    constexpr int stages = (kSmemCapacity - smemCd - smemBarrierReserve - smemTmemPointer) / smemPerStage;
    return stages < kMaxPipelineStages ? stages : kMaxPipelineStages;
}

template <int BlockM>
constexpr int dynamicSmemSize()
{
    constexpr int smemCd = 2 * 16 * kBlockN * static_cast<int>(sizeof(__nv_bfloat16));
    constexpr int smemBarrierReserve = (kMaxPipelineStages * 3 + 4) * 8;
    constexpr int smemTmemPointer = 4;
    constexpr int smemPerStage
        = (BlockM / kClusterSize) * kBlockK + kBlockN * kBlockK + 2 * 128 * static_cast<int>(sizeof(uint32_t));
    return smemCd + numPipelineStages<BlockM>() * smemPerStage + smemBarrierReserve + smemTmemPointer;
}

template <int BlockM, int SplitK>
void launchInstance(CUtensorMap tensorMapA, CUtensorMap tensorMapB, CUtensorMap tensorMapSfa, CUtensorMap tensorMapSfb,
    CUtensorMap tensorMapD, __nv_bfloat16* partials, int M, int numSms, cudaStream_t stream)
{
    using namespace deep_gemm;
    constexpr int kNumStages = numPipelineStages<BlockM>();
    constexpr int kDynamicSmem = dynamicSmemSize<BlockM>();
    static_assert(kNumStages > 0, "DSV4 O_b split-K pipeline has no shared-memory stages");
    static_assert(kDynamicSmem <= kSmemCapacity, "DSV4 O_b split-K shared-memory budget exceeded");

    auto kernel = &dsv4_splitk::dsv4Fp8SplitKGemmKernel<cute::UMMA::Major::K, cute::UMMA::Major::K, 128, 128, 0,
        kDsv4ProN, kDsv4ProK, BlockM, kBlockN, kBlockK, 1, 128, 128, 128, kNumStages, 128, 128, kClusterSize, true,
        kNumSmsForSwizzle, true, GemmType::Normal, false, cutlass::float_e4m3_t, cutlass::float_e4m3_t,
        cutlass::bfloat16_t, epilogue::transform::EpilogueIdentity, SplitK>;

    TLLM_CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<void const*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmem));

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(numSms, 1, 1);
    config.blockDim = dim3(kNumThreads, 1, 1);
    config.dynamicSmemBytes = kDynamicSmem;
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = kClusterSize;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;

    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, static_cast<int*>(nullptr), static_cast<uint32_t>(M),
        static_cast<uint32_t>(kDsv4ProN), static_cast<uint32_t>(kDsv4ProK), tensorMapA, tensorMapB, tensorMapSfa,
        tensorMapSfb, tensorMapD, static_cast<void*>(partials)));
}

template <int SplitK>
void dispatchBlockM(int blockM, CUtensorMap tensorMapA, CUtensorMap tensorMapB, CUtensorMap tensorMapSfa,
    CUtensorMap tensorMapSfb, CUtensorMap tensorMapD, __nv_bfloat16* partials, int M, int numSms, cudaStream_t stream)
{
#define TRTLLM_DSV4_SPLITK_BLOCK_M(BLOCK_M)                                                                            \
    case BLOCK_M:                                                                                                      \
        launchInstance<BLOCK_M, SplitK>(                                                                               \
            tensorMapA, tensorMapB, tensorMapSfa, tensorMapSfb, tensorMapD, partials, M, numSms, stream);              \
        return
    switch (blockM)
    {
        TRTLLM_DSV4_SPLITK_BLOCK_M(16);
        TRTLLM_DSV4_SPLITK_BLOCK_M(32);
        TRTLLM_DSV4_SPLITK_BLOCK_M(48);
        TRTLLM_DSV4_SPLITK_BLOCK_M(64);
        TRTLLM_DSV4_SPLITK_BLOCK_M(80);
        TRTLLM_DSV4_SPLITK_BLOCK_M(96);
        TRTLLM_DSV4_SPLITK_BLOCK_M(112);
        TRTLLM_DSV4_SPLITK_BLOCK_M(128);
    default: TLLM_CHECK_WITH_INFO(false, "unsupported DSV4 O_b split-K block M: %d", blockM);
    }
#undef TRTLLM_DSV4_SPLITK_BLOCK_M
}

} // namespace

void dsv4Fp8SplitKGemmLaunch(void const* a, int32_t const* sfa, void const* b, int32_t const* sfb,
    __nv_bfloat16* partials, int M, int N, int K, int sfaKStride, int sfbKStride, int numSplits, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(a != nullptr && sfa != nullptr && b != nullptr && sfb != nullptr && partials != nullptr,
        "DSV4 O_b split-K received a null pointer");
    TLLM_CHECK_WITH_INFO(M > 0, "DSV4 O_b split-K requires a positive M, got %d", M);
    TLLM_CHECK_WITH_INFO(N == kDsv4ProN && K == kDsv4ProK, "DSV4 O_b split-K supports N=%d and K=%d, got N=%d K=%d",
        kDsv4ProN, kDsv4ProK, N, K);
    TLLM_CHECK_WITH_INFO(
        numSplits == 2 || numSplits == 4, "DSV4 O_b split-K requires 2 or 4 splits, got %d", numSplits);
    TLLM_CHECK_WITH_INFO(sfaKStride >= M && sfaKStride % 4 == 0,
        "DSV4 O_b split-K activation scale stride must be a 4-aligned value >= M, got %d", sfaKStride);
    TLLM_CHECK_WITH_INFO(
        sfbKStride == N, "DSV4 O_b split-K weight scale stride must equal N=%d, got %d", N, sfbKStride);

    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int major = 0;
    int minor = 0;
    int numSms = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device));
    TLLM_CHECK_WITH_INFO(major == 10, "DSV4 O_b split-K requires the SM100 family, got SM%d%d", major, minor);
    TLLM_CHECK_WITH_INFO(
        numSms % kClusterSize == 0, "DSV4 O_b split-K requires an even SM count for 2-CTA clusters, got %d", numSms);

    int const blockM = std::min(((M + 15) / 16) * 16, 128);
    CUtensorMap const tensorMapA = makeTma2D(a, CU_TENSOR_MAP_DATA_TYPE_UINT8, K, M, kBlockK, blockM / kClusterSize,
        static_cast<uint64_t>(K), CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap const tensorMapB = makeTma2D(
        b, CU_TENSOR_MAP_DATA_TYPE_UINT8, K, N, kBlockK, kBlockN, static_cast<uint64_t>(K), CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap const tensorMapSfa = makeTma2D(sfa, CU_TENSOR_MAP_DATA_TYPE_INT32, sfaKStride, K / 512, blockM, 1,
        static_cast<uint64_t>(sfaKStride) * sizeof(int32_t), CU_TENSOR_MAP_SWIZZLE_NONE);
    CUtensorMap const tensorMapSfb = makeTma2D(sfb, CU_TENSOR_MAP_DATA_TYPE_INT32, sfbKStride, K / 512, kBlockN, 1,
        static_cast<uint64_t>(sfbKStride) * sizeof(int32_t), CU_TENSOR_MAP_SWIZZLE_NONE);
    CUtensorMap const tensorMapD = makeTma2D(partials, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, N, M, kBlockN, 16,
        static_cast<uint64_t>(N) * sizeof(__nv_bfloat16), CU_TENSOR_MAP_SWIZZLE_128B);

    if (numSplits == 2)
    {
        dispatchBlockM<2>(
            blockM, tensorMapA, tensorMapB, tensorMapSfa, tensorMapSfb, tensorMapD, partials, M, numSms, stream);
    }
    else
    {
        dispatchBlockM<4>(
            blockM, tensorMapA, tensorMapB, tensorMapSfa, tensorMapSfb, tensorMapD, partials, M, numSms, stream);
    }
}

} // namespace kernels::mhc

TRTLLM_NAMESPACE_END

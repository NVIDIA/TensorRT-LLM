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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/csa2SparseContext/csa2SparseContext.h"
#include "tensorrt_llm/kernels/csa2SparseContext/csa2SparseContextKernel.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cuda.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::csa2SparseContext
{

namespace
{

constexpr int64_t kMaxTmaRows = int64_t{1} << 31;
constexpr int kMaxDevices = 64;

CUtensorMap encodeTensorMap(int rank, uint64_t const* dims, uint64_t const* strideBytes, uint32_t const* box,
    void const* base, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion)
{
    CUtensorMap tensorMap{};
    uint32_t const elementStrides[3] = {1, 1, 1};
    auto const driver = common::CUDADriverWrapper::getInstance();
    TLLM_CU_CHECK(driver->cuTensorMapEncodeTiled(&tensorMap, dataType, rank, const_cast<void*>(base), dims, strideBytes,
        box, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensorMap;
}

// Bank rows of 512 E4M3 channels, read 128 channels x boxRows rows at a time (1 row for gather4, 128 for windows).
CUtensorMap makeBankMap(void const* bank, int64_t rows, uint32_t boxRows)
{
    uint64_t const dims[2] = {kHeadDim, static_cast<uint64_t>(std::max<int64_t>(rows, 1))};
    uint64_t const strides[1] = {kHeadDim};
    uint32_t const box[2] = {128, boxRows};
    return encodeTensorMap(2, dims, strides, box, bank, CU_TENSOR_MAP_DATA_TYPE_UINT8, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
}

// [tokens, 64, 512] with strides in elements of elementBytes.
CUtensorMap makeTokenMap(void const* base, int64_t tokens, int64_t tokenStride, int64_t headStride, int elementBytes,
    uint32_t boxChannels, uint32_t boxHeads, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion)
{
    uint64_t const dims[3] = {kHeadDim, kNumHeads, static_cast<uint64_t>(tokens)};
    uint64_t const strides[2]
        = {static_cast<uint64_t>(headStride * elementBytes), static_cast<uint64_t>(tokenStride * elementBytes)};
    uint32_t const box[3] = {boxChannels, boxHeads, 1};
    return encodeTensorMap(3, dims, strides, box, base, dataType, swizzle, l2Promotion);
}

bool isAligned(void const* pointer, size_t alignment)
{
    return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

void checkParams(Csa2SparseContextParams const& p)
{
    int64_t const qBytes = p.qBf16 ? 2 : 1;
    TLLM_CHECK_WITH_INFO(p.q != nullptr && isAligned(p.q, 16) && (p.qTokenStride * qBytes) % 16 == 0
            && (p.qHeadStride * qBytes) % 16 == 0,
        "CSA2 sparse context Q must be 16-B aligned with 16-B multiple strides");
    TLLM_CHECK_WITH_INFO(!p.qBf16 || p.qScale != nullptr, "CSA2 sparse context BF16 Q needs its quant scale");
    TLLM_CHECK_WITH_INFO(p.bank != nullptr && isAligned(p.bank, 16) && p.bankRows > 0 && p.bankRows < kMaxTmaRows,
        "CSA2 sparse context bank must be 16-B aligned with 1 to 2^31 - 1 rows");
    TLLM_CHECK_WITH_INFO(p.indices != nullptr && isAligned(p.indices, 16) && p.indexStride % 4 == 0 && p.maxKeys > 0
            && p.maxKeys <= p.indexStride,
        "CSA2 sparse context indices must be 16-B aligned rows with a stride that is a multiple of 4");
    TLLM_CHECK_WITH_INFO(p.lens != nullptr && p.bmm1Scale != nullptr && p.bmm2Scale != nullptr,
        "CSA2 sparse context needs key counts and BMM scales");
    if (p.fp8Out != nullptr)
    {
        TLLM_CHECK_WITH_INFO(isAligned(p.fp8Out, 32) && p.fp8GroupStride % 32 == 0 && p.fp8TokenStride % 32 == 0
                && p.headsPerGroup > 0 && kNumHeads % p.headsPerGroup == 0,
            "CSA2 sparse context FP8 output must be 32-B aligned with 32-B multiple strides and whole head groups");
        TLLM_CHECK_WITH_INFO(p.fp8Scale != nullptr && p.fp8ScaleRows == (p.numTokens + 3) / 4 * 4
                && p.positions != nullptr && p.ropeCosSin != nullptr && isAligned(p.ropeCosSin, 16),
            "CSA2 sparse context FP8 output needs scales for the token count rounded up to 4, positions and RoPE "
            "cos/sin");
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            p.out != nullptr && isAligned(p.out, 16) && p.outTokenStride % 8 == 0 && p.outHeadStride % 8 == 0,
            "CSA2 sparse context output must be 16-B aligned with strides that are multiples of 8 elements");
    }
}

// The kernel needs more than the default 48 KB of dynamic shared memory; the attribute is set once per device.
template <bool kBf16Q, bool kFp8Out>
void ensureSharedMemoryAttribute()
{
    static std::array<std::atomic<bool>, kMaxDevices> configured{};
    int const device = common::getDevice();
    if (device < kMaxDevices && configured[device].load(std::memory_order_acquire))
    {
        return;
    }
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(csa2SparseContextKernel<kBf16Q, kFp8Out>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(detail::kSmemBytes)));
    if (device < kMaxDevices)
    {
        configured[device].store(true, std::memory_order_release);
    }
}

template <bool kBf16Q, bool kFp8Out>
void launch(Csa2SparseContextParams const& p, CUtensorMap const& tmapQ, cudaStream_t stream)
{
    ensureSharedMemoryAttribute<kBf16Q, kFp8Out>();
    CUtensorMap const tmapBank = makeBankMap(p.bank, p.bankRows, 1);
    CUtensorMap const tmapTile = makeBankMap(p.bank, p.bankRows, detail::kBlockKeys);
    // The FP8 output leaves through plain stores; its launches pass the bank map in the unused output slot.
    CUtensorMap const tmapOut = kFp8Out
        ? tmapBank
        : makeTokenMap(p.out, p.numTokens, p.outTokenStride, p.outHeadStride, 2, 32, 32,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE);
    KernelParams kp{};
    kp.indices = p.indices;
    kp.indexStride = p.indexStride;
    kp.maxKeys = p.maxKeys;
    kp.lens = p.lens;
    kp.bmm1Scale = p.bmm1Scale;
    kp.bmm2Scale = p.bmm2Scale;
    kp.qScale = p.qScale;
    kp.attnSinks = p.attnSinks;
    kp.numTokens = p.numTokens;
    kp.bankRows = static_cast<int>(p.bankRows);
    kp.fp8Out = static_cast<uint8_t*>(p.fp8Out);
    kp.fp8GroupStride = p.fp8GroupStride;
    kp.fp8TokenStride = p.fp8TokenStride;
    kp.fp8Scale = p.fp8Scale;
    kp.fp8ScaleGroupStride = p.fp8ScaleGroupStride;
    kp.fp8ScaleBlockStride = p.fp8ScaleBlockStride;
    kp.fp8ScaleRows = p.fp8ScaleRows;
    kp.headsPerGroup = p.headsPerGroup;
    kp.positions = p.positions;
    kp.positions64 = p.positions64;
    kp.ropeCosSin = p.ropeCosSin;
    int const grid = std::min(p.numTokens, common::getMultiProcessorCount());
    csa2SparseContextKernel<kBf16Q, kFp8Out>
        <<<grid, detail::kNumThreads, detail::kSmemBytes, stream>>>(tmapBank, tmapQ, tmapTile, tmapOut, kp);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

template <bool kBf16Q>
void launchOutput(Csa2SparseContextParams const& p, CUtensorMap const& tmapQ, cudaStream_t stream)
{
    if (p.fp8Out != nullptr)
    {
        launch<kBf16Q, true>(p, tmapQ, stream);
    }
    else
    {
        launch<kBf16Q, false>(p, tmapQ, stream);
    }
}

} // namespace

bool isSupported()
{
#if defined(PLACEHOLDER_KERNELS)
    return false;
#else
    if (!common::isSM100Family())
    {
        return false;
    }
    // The family builds carry no image for some members; ask the runtime for one.
    cudaFuncAttributes attributes{};
    cudaError_t const status = cudaFuncGetAttributes(&attributes, csa2SparseContextKernel<true, false>);
    if (status != cudaSuccess)
    {
        static_cast<void>(cudaGetLastError());
        return false;
    }
    return true;
#endif
}

void run(Csa2SparseContextParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.numTokens >= 0, "CSA2 sparse context needs a non-negative token count");
    if (params.numTokens == 0)
    {
        return;
    }
    checkParams(params);
    if (params.qBf16)
    {
        // Q quantization reads 64 BF16 channels (128 B) per box; the converted E4M3 chunk replaces the first box.
        CUtensorMap const tmapQ
            = makeTokenMap(params.q, params.numTokens, params.qTokenStride, params.qHeadStride, 2, 64, kNumHeads,
                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        launchOutput<true>(params, tmapQ, stream);
    }
    else
    {
        CUtensorMap const tmapQ
            = makeTokenMap(params.q, params.numTokens, params.qTokenStride, params.qHeadStride, 1, 128, kNumHeads,
                CU_TENSOR_MAP_DATA_TYPE_UINT8, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        launchOutput<false>(params, tmapQ, stream);
    }
}

} // namespace kernels::csa2SparseContext

TRTLLM_NAMESPACE_END

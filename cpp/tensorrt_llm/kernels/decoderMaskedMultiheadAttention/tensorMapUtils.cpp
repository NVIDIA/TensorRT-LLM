/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/tensorMapUtils.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"

#include <cstdint>
#include <type_traits>

namespace tensorrt_llm::kernels
{

namespace
{

using tensorrt_llm::common::CUDADriverWrapper;

uint32_t getElemBytes(CUtensorMapDataType_enum dataType)
{
    switch (dataType)
    {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8: return 1;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16: return 2;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32: return 4;
    case CU_TENSOR_MAP_DATA_TYPE_INT32: return 4;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64: return 8;
    case CU_TENSOR_MAP_DATA_TYPE_INT64: return 8;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16: return 2;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32: return 4;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64: return 8;
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: return 2;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ: return 4;
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32: return 4;
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ: return 4;
    case CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B: return 8;
    case CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B: return 16;
    case CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B: return 16;
    }
    throw std::runtime_error("unsupported data type");
}

CUtensorMap makeTensorMapForPagedKVCache(std::shared_ptr<CUDADriverWrapper> const& driver, void const* addr,
    CUtensorMapDataType_enum dataType, uint32_t headElems, uint32_t nbKHeads, uint32_t tokensPerPage,
    uint32_t nbTokensPerTile = 64)
{
    CUtensorMap tensorMap{};
    uint32_t elemBytes = getElemBytes(dataType);
    uint64_t const globalDims[] = {headElems, tokensPerPage, nbKHeads, 1U << 31};
    uint32_t const headBytes = elemBytes * headElems;
    uint64_t const globalStrides[] = {headBytes, headBytes * tokensPerPage, headBytes * tokensPerPage * nbKHeads};
    TLLM_CHECK(headElems <= 256);
    uint32_t const paddedHeadElems = headElems <= 64 ? 64 : (headElems <= 128 ? 128 : 256);
    uint32_t const partElems = std::min(elemBytes * paddedHeadElems, 128U) / elemBytes;
    uint32_t const boxDims[] = {partElems, std::min(tokensPerPage, nbTokensPerTile), 1, 1};
    uint32_t const elemStrides[] = {1, 1, 1, 1};

    auto const swizzle = [&]
    {
        switch (partElems)
        {
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
        default: TLLM_THROW("unsupported cache head size");
        }
    }();

    TLLM_CU_CHECK(driver->cuTensorMapEncodeTiled(&tensorMap, dataType, 4, const_cast<void*>(addr), globalDims,
        globalStrides, boxDims, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensorMap;
}

CUtensorMap makeTensorMapForContiguousKVCache(std::shared_ptr<CUDADriverWrapper> const& driver, void const* addr,
    CUtensorMapDataType_enum dataType, uint32_t headElems, uint32_t nbKHeads, uint32_t maxCacheLen, uint32_t beamWidth,
    uint32_t batchSize, uint32_t nbTokensPerTile = 64)
{
    CUtensorMap tensorMap{};
    uint64_t const globalDims[] = {headElems, maxCacheLen, nbKHeads, 2 * beamWidth * batchSize};
    uint32_t elemBytes = getElemBytes(dataType);
    uint32_t const headBytes = elemBytes * headElems;
    uint64_t const globalStrides[] = {headBytes, headBytes * maxCacheLen, headBytes * maxCacheLen * nbKHeads};
    TLLM_CHECK(headElems <= 256);
    uint32_t const paddedHeadElems = headElems <= 64 ? 64 : (headElems <= 128 ? 128 : 256);
    uint32_t const partElems = std::min(elemBytes * paddedHeadElems, 128U) / elemBytes;
    uint32_t const boxDims[] = {partElems, nbTokensPerTile, 1, 1};
    uint32_t const elemStrides[] = {1, 1, 1, 1};

    auto const swizzle = [&]
    {
        switch (partElems)
        {
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
        default: TLLM_THROW("unsupported cache head size");
        }
    }();

    TLLM_CU_CHECK(driver->cuTensorMapEncodeTiled(&tensorMap, dataType, 4, const_cast<void*>(addr), globalDims,
        globalStrides, boxDims, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensorMap;
}

} // namespace

template <typename KVCacheBuffer>
CUtensorMap makeTensorMapForKVCache(
    std::shared_ptr<CUDADriverWrapper> const& driver, XQAParams const& xqaParams, KVCacheBuffer const& kv_cache_buffer)
{
    if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
    {
        return makeTensorMapForPagedKVCache(driver, kv_cache_buffer.mPrimaryPoolPtr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            xqaParams.head_size, xqaParams.num_kv_heads, xqaParams.tokens_per_block);
    }
    else
    {
        static_assert(std::is_same_v<KVCacheBuffer, KVLinearBuffer>);
        return makeTensorMapForContiguousKVCache(driver, kv_cache_buffer.data, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            xqaParams.head_size, xqaParams.num_kv_heads, xqaParams.max_attention_window_size, xqaParams.beam_width,
            xqaParams.batch_size);
    }
}

template CUtensorMap makeTensorMapForKVCache(
    std::shared_ptr<CUDADriverWrapper> const&, XQAParams const&, KVBlockArray const&);
template CUtensorMap makeTensorMapForKVCache(
    std::shared_ptr<CUDADriverWrapper> const&, XQAParams const&, KVLinearBuffer const&);

} // namespace tensorrt_llm::kernels

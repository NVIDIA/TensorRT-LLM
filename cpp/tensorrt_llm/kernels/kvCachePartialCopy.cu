/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/kvCachePartialCopy.h"
#include <cstdint>
#include <cuda_runtime_api.h>

namespace tensorrt_llm
{
namespace kernels
{
namespace
{
template <typename T>
__global__ void cuKVCacheBlockPartialCopy(T* dst, T const* src, unsigned int numLayers, unsigned int numHeads,
    unsigned int tokensPerBlock, unsigned int numHidden, unsigned int numTokensToCopy)
{
    if (numTokensToCopy <= 0)
    {
        numTokensToCopy = tokensPerBlock;
    }

    int threadHead = blockIdx.z * blockDim.z + threadIdx.z;
    int threadToken = blockIdx.y * blockDim.y + threadIdx.y;
    int threadHidden = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadHead < numHeads && threadToken < numTokensToCopy && threadHidden < numHidden)
    {
        int64_t strideLayer = static_cast<int64_t>(numHeads) * tokensPerBlock * numHidden;
        for (int iLayer = 0; iLayer < numLayers; ++iLayer)
        {
            int flatIndex = (threadHead * tokensPerBlock + threadToken) * numHidden + threadHidden;
            int64_t globalIndex = strideLayer * iLayer + flatIndex;
            dst[globalIndex] = src[globalIndex];
        }
    }
}

unsigned int ipow2(unsigned int v)
{
    unsigned int cnt = 0;
    for (v = std::max(v, 1u) - 1u; v != 0u; v = v >> 1)
        ++cnt;
    return 1u << cnt;
}

template <typename T>
void hostKVCacheBlockPartialCopy(IBuffer& dst, IBuffer const& src, unsigned int numLayers, unsigned int numHeads,
    unsigned int tokensPerBlock, unsigned int numHidden, unsigned int numTokensToCopy, int kvFactor,
    cudaStream_t stream)
{
    unsigned int blockX = ipow2(numHidden);         // ensure block shape is a power of 2
    blockX = std::min(blockX, 32u);                 // blockX should not exceed warp size
    blockX = std::max(blockX, 2u);                  // blockX must be at least 2
    unsigned int blockY = 128u / blockX;            // blockX * blockY should be 128
    blockY = std::min(blockY, numTokensToCopy);     // blockY should not exceed numTokensToCopy
    blockY = ipow2(blockY);                         // blockY must be power of 2
    unsigned int blockZ = 128u / (blockY * blockX); // blockX * blockY * blockZ should be 128
    blockZ = std::min(blockZ, numHeads);            // blockZ should not exceed numHeads
    blockZ = ipow2(blockZ);                         // blockZ must be power of 2
    dim3 block = {blockX, blockY, blockZ};
    dim3 grid = {(numHidden + block.x - 1) / block.x, (numTokensToCopy + block.y - 1) / block.y,
        (numHeads + block.z - 1) / block.z};
    auto srcData = bufferCast<T>(src);
    auto dstData = bufferCast<T>(dst);
    cuKVCacheBlockPartialCopy<<<grid, block, 0, stream>>>(
        dstData, srcData, numLayers * kvFactor, numHeads, tokensPerBlock, numHidden, numTokensToCopy);
}
} // namespace

void kvCacheBlockPartialCopy(IBuffer& dst, IBuffer const& src, unsigned int numLayers, unsigned int numHeads,
    unsigned int tokensPerBlock, unsigned int numHidden, unsigned int numTokensToCopy, int kvFactor,
    cudaStream_t stream)
{
    auto dataType = src.getDataType();
    TLLM_CHECK_WITH_INFO(dataType == dst.getDataType(), "src and dst dataType does not match");
    switch (dataType)
    {
    case nvinfer1::DataType::kINT64:
        hostKVCacheBlockPartialCopy<SizeType64>(
            dst, src, numLayers, numHeads, tokensPerBlock, numHidden, numTokensToCopy, kvFactor, stream);
        break;
    case nvinfer1::DataType::kINT32:
        hostKVCacheBlockPartialCopy<std::int32_t>(
            dst, src, numLayers, numHeads, tokensPerBlock, numHidden, numTokensToCopy, kvFactor, stream);
        break;
    case nvinfer1::DataType::kFLOAT:
        hostKVCacheBlockPartialCopy<float>(
            dst, src, numLayers, numHeads, tokensPerBlock, numHidden, numTokensToCopy, kvFactor, stream);
        break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
        hostKVCacheBlockPartialCopy<__nv_bfloat16>(
            dst, src, numLayers, numHeads, tokensPerBlock, numHidden, numTokensToCopy, kvFactor, stream);
        break;
#endif
    case nvinfer1::DataType::kHALF:
        hostKVCacheBlockPartialCopy<half>(
            dst, src, numLayers, numHeads, tokensPerBlock, numHidden, numTokensToCopy, kvFactor, stream);
        break;
    case nvinfer1::DataType::kBOOL:
        hostKVCacheBlockPartialCopy<bool>(
            dst, src, numLayers, numHeads, tokensPerBlock, numHidden, numTokensToCopy, kvFactor, stream);
        break;
    case nvinfer1::DataType::kUINT8:
        hostKVCacheBlockPartialCopy<std::uint8_t>(
            dst, src, numLayers, numHeads, tokensPerBlock, numHidden, numTokensToCopy, kvFactor, stream);
        break;
    case nvinfer1::DataType::kINT8:
        hostKVCacheBlockPartialCopy<std::int8_t>(
            dst, src, numLayers, numHeads, tokensPerBlock, numHidden, numTokensToCopy, kvFactor, stream);
        break;
#ifdef ENABLE_FP8
    case nvinfer1::DataType::kFP8:
        hostKVCacheBlockPartialCopy<__nv_fp8_e4m3>(
            dst, src, numLayers, numHeads, tokensPerBlock, numHidden, numTokensToCopy, kvFactor, stream);
        break;
#endif
    default: TLLM_THROW("Unknown data type");
    }
}

} // namespace kernels
} // namespace tensorrt_llm

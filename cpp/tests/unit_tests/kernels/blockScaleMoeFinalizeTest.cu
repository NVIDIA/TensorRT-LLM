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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <cuda.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace tensorrt_llm::tests::kernels::blockscalemoe
{

namespace tc = tensorrt_llm::common;
namespace tg = batchedGemm::trtllm::gen;

using tensorrt_llm::runtime::BufferManager;
using tensorrt_llm::runtime::CudaStream;
using tensorrt_llm::runtime::ITensor;
using tensorrt_llm::runtime::MemoryType;
using tensorrt_llm::runtime::bufferCast;

namespace
{

struct CapturedKernelLaunch
{
    dim3 gridDim;
    dim3 blockDim;
};

CapturedKernelLaunch captureFinalizeNode(moe::dev::finalize::Data const& data, cudaStream_t stream)
{
    cudaGraph_t graph{};
    TLLM_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    moe::dev::finalize::run(data, stream);
    TLLM_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    std::size_t numNodes = 0;
    TLLM_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &numNodes));
    TLLM_CHECK_WITH_INFO(numNodes == 1, "Expected one finalize kernel node, got %zu", numNodes);
    cudaGraphNode_t node{};
    TLLM_CUDA_CHECK(cudaGraphGetNodes(graph, &node, &numNodes));
    cudaGraphNodeType nodeType{};
    TLLM_CUDA_CHECK(cudaGraphNodeGetType(node, &nodeType));
    TLLM_CHECK_WITH_INFO(nodeType == cudaGraphNodeTypeKernel, "Finalize graph node is not a kernel");

    // Extended launches may capture a kernel-object-backed node, which the
    // runtime parameter structure cannot represent.
    CUDA_KERNEL_NODE_PARAMS params{};
    CUresult const result = cuGraphKernelNodeGetParams(node, &params);
    TLLM_CHECK_WITH_INFO(result == CUDA_SUCCESS, "Could not read finalize kernel node parameters: %d", result);
    TLLM_CUDA_CHECK(cudaGraphDestroy(graph));
    return {
        {params.gridDimX, params.gridDimY, params.gridDimZ}, {params.blockDimX, params.blockDimY, params.blockDimZ}};
}

uint16_t bf16Bits(__nv_bfloat16 value)
{
    uint16_t bits{};
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

} // namespace

TEST(BlockScaleMoeFinalizeTest, Bf16SinglePassMatchesReferenceAndGuardsDispatch)
{
    if (tc::getSMVersion() < 90)
    {
        GTEST_SKIP() << "The trtllm-gen block-scale MoE finalize kernels target SM90+.";
    }

    constexpr int32_t kNumTokens = 119;
    constexpr int32_t kTopK = 10;
    constexpr int32_t kHiddenDim = 2560;
    constexpr int32_t kNumRows = kNumTokens * kTopK;

    auto stream = std::make_shared<CudaStream>();
    BufferManager manager(stream);

    std::vector<__nv_bfloat16> hostInput(static_cast<int64_t>(kNumRows) * kHiddenDim);
    for (int32_t row = 0; row < kNumRows; ++row)
    {
        for (int32_t hidden = 0; hidden < kHiddenDim; ++hidden)
        {
            float const value = static_cast<float>((row + hidden) % 17 - 8) / 64.F;
            hostInput[static_cast<int64_t>(row) * kHiddenDim + hidden] = __float2bfloat16(value);
        }
    }
    std::vector<__nv_bfloat16> hostWeights(kNumRows);
    for (int32_t token = 0; token < kNumTokens; ++token)
    {
        for (int32_t k = 0; k < kTopK; ++k)
        {
            hostWeights[token * kTopK + k] = __float2bfloat16(static_cast<float>(k + 1) / 32.F);
        }
    }
    std::vector<int32_t> hostMap(kNumRows);
    for (int32_t row = 0; row < kNumRows; ++row)
    {
        hostMap[row] = row;
    }

    auto input
        = manager.copyFrom(hostInput, ITensor::makeShape({static_cast<int64_t>(hostInput.size())}), MemoryType::kGPU);
    auto weights = manager.copyFrom(
        hostWeights, ITensor::makeShape({static_cast<int64_t>(hostWeights.size())}), MemoryType::kGPU);
    auto map = manager.copyFrom(hostMap, ITensor::makeShape({static_cast<int64_t>(hostMap.size())}), MemoryType::kGPU);
    auto totalRows = manager.copyFrom(std::vector<int32_t>{kNumRows}, ITensor::makeShape({1}), MemoryType::kGPU);
    auto output = manager.gpu(ITensor::makeShape({static_cast<int64_t>(kNumTokens) * kHiddenDim}), DataType::kBF16);
    stream->synchronize();

    moe::dev::finalize::Data data;
    data.mDtypeElt = tg::Dtype::Bfloat16;
    data.mDtypeExpW = tg::Dtype::Bfloat16;
    data.mUsePdl = false;
    data.mUseDeepSeekFp8 = false;
    data.inPtr = bufferCast<__nv_bfloat16>(*input);
    data.outPtr = bufferCast<__nv_bfloat16>(*output);
    data.expertWeightsPtr = bufferCast<__nv_bfloat16>(*weights);
    data.expandedIdxToPermutedIdx = bufferCast<int32_t>(*map);
    data.numTokens = kNumTokens;
    data.numExperts = kNumRows;
    data.topK = kTopK;
    data.hiddenDim = kHiddenDim;
    data.hiddenDimPadded = kHiddenDim;
    data.totalNumPaddedTokens = bufferCast<int32_t>(*totalRows);

    auto const targetNode = captureFinalizeNode(data, stream->get());
    // One thread per 128-bit vector of a row, not a hard-coded 320.
    constexpr unsigned int kVecPerRow = kHiddenDim * 16 / 128;
    EXPECT_EQ(targetNode.blockDim.x, kVecPerRow);
    EXPECT_EQ(targetNode.gridDim.x, static_cast<unsigned int>(kNumTokens));
    EXPECT_EQ(targetNode.gridDim.y, 1U);

    // Above 1.5x the default block the wider block loses more to occupancy than
    // the idle tail costs, so a wide hidden dim stays on the generic geometry.
    auto fallbackWide = data;
    fallbackWide.hiddenDim = 4096;
    fallbackWide.hiddenDimPadded = 4096;
    auto const wideNode = captureFinalizeNode(fallbackWide, stream->get());
    EXPECT_EQ(wideNode.blockDim.x, 256U);
    EXPECT_EQ(wideNode.gridDim.x, static_cast<unsigned int>(kNumTokens));

    auto fallbackAlignment = data;
    fallbackAlignment.outPtr = static_cast<std::byte*>(data.outPtr) + sizeof(__nv_bfloat16);
    auto const alignmentNode = captureFinalizeNode(fallbackAlignment, stream->get());
    EXPECT_EQ(alignmentNode.blockDim.x, 256U);
    EXPECT_EQ(alignmentNode.gridDim.x, 10U);
    EXPECT_EQ(alignmentNode.gridDim.y, static_cast<unsigned int>(kNumTokens));

    moe::dev::finalize::run(data, stream->get());
    TLLM_CUDA_CHECK(cudaGetLastError());
    std::vector<__nv_bfloat16> actual(static_cast<int64_t>(kNumTokens) * kHiddenDim);
    manager.copy(*output, actual.data());
    stream->synchronize();

    for (int32_t token = 0; token < kNumTokens; ++token)
    {
        for (int32_t hidden = 0; hidden < kHiddenDim; ++hidden)
        {
            float expected = 0.F;
            for (int32_t k = 0; k < kTopK; ++k)
            {
                int32_t const row = token * kTopK + k;
                expected += __bfloat162float(hostWeights[row])
                    * __bfloat162float(hostInput[static_cast<int64_t>(row) * kHiddenDim + hidden]);
            }
            auto const expectedBf16 = __float2bfloat16(expected);
            auto const idx = static_cast<int64_t>(token) * kHiddenDim + hidden;
            ASSERT_EQ(bf16Bits(actual[idx]), bf16Bits(expectedBf16)) << "token=" << token << " hidden=" << hidden;
        }
    }
}

} // namespace tensorrt_llm::tests::kernels::blockscalemoe

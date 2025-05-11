/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/kernels/lora/lora.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/groupGemm.h"
#include "tensorrt_llm/kernels/splitkGroupGemm.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <algorithm>
#include <utility>

namespace tensorrt_llm::kernels
{

// TODO should reuse the function in gemmPlugin
void _getProblemParams(cublasOperation_t& transa, cublasOperation_t& transb, int& m, int& n, int& k, int& lda, int& ldb,
    int& ldc, bool transA, bool transB, int M, int N, int K)
{
    transa = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    transb = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    m = N;
    n = M;
    k = K;
    lda = transB ? K : N;
    ldb = transA ? M : K;
    ldc = N;
}

// TODO should reuse the function in gemmPlugin
void _runGemm(int const M, int const N, int const K, bool const transA, bool const transB,
    nvinfer1::DataType const type, CublasGemmWrapperPtr const& cublasWrapperPtr, void const* act, void const* weight,
    void* output, std::optional<cublasLtMatmulHeuristicResult_t> const& heuristic, void* workspace, cudaStream_t stream)
{
    cublasWrapperPtr->setStream(stream);
    cublasWrapperPtr->setWorkspace(workspace);

    cublasOperation_t transa, transb;
    int m, n, k;
    int lda, ldb, ldc;
    _getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, transA, transB, M, N, K);

    cublasWrapperPtr->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
    cublasWrapperPtr->Gemm(transa, transb, m, n, k, weight, lda, act, ldb, output, ldc, heuristic);
    cublasWrapperPtr->destroyDescriptors();
}

LoraImpl::LoraImpl(int in_hidden_size, std::vector<int> out_hidden_sizes, bool transA, bool transB,
    int num_lora_modules, nvinfer1::DataType type, int max_low_rank, std::shared_ptr<CublasGemmWrapper> cublasWrapper)
    : mInHiddenSize(in_hidden_size)
    , mTransA(transA)
    , mTransB(transB)
    , mNumLoraModules(num_lora_modules)
    , mType(type)
    , mMaxLowRank(max_low_rank)
    , mCublasWrapper(std::move(cublasWrapper))
{
    mOutHiddenSizes.resize(mNumLoraModules);
    mOutHiddenSizes.assign(out_hidden_sizes.begin(), out_hidden_sizes.end());
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
}

void LoraImpl::setGemmConfig()
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    if (mType == nvinfer1::DataType::kHALF)
    {
        mCublasWrapper->setFP16GemmConfig();
    }
    else if (mType == nvinfer1::DataType::kFLOAT)
    {
        mCublasWrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (mType == nvinfer1::DataType::kBF16)
    {
        mCublasWrapper->setBF16GemmConfig();
    }
#endif
}

int64_t getLowRankWorkSpaceSize(int64_t numTokens, int64_t maxLoraModuleNum, int64_t maxLowRank, int64_t typeSize)
{
    return common::divUp(numTokens * maxLoraModuleNum * maxLowRank * typeSize, 16) * 16;
}

int64_t getGemmParamsWorkSpaceSize(int64_t nbReq)
{
    return std::max(getSplitkGroupedGemmParamsWorkSpaceSize(nbReq), getGroupedGemmParamsWorkSpaceSize(nbReq));
}

int64_t getSplitkGroupedGemmWorkSpaceSize(
    int64_t numTokens, int64_t maxLoraModuleNum, int64_t maxLowRank, int64_t splitKSlices)
{
    return common::divUp(numTokens * maxLoraModuleNum * maxLowRank * sizeof(float) * splitKSlices, 16) * 16;
}

int64_t getGemmWorkSpaceSize(int64_t numTokens, int64_t maxLoraModuleNum, int64_t maxLowRank, int64_t splitKSlices)
{
    return std::max((int64_t) CUBLAS_WORKSPACE_SIZE,
        getSplitkGroupedGemmWorkSpaceSize(numTokens, maxLoraModuleNum, maxLowRank, splitKSlices));
}

size_t LoraImpl::getWorkspaceSize(
    int64_t const numTokens, int64_t const numReqs, nvinfer1::DataType const type) const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    auto const typeSize = tensorrt_llm::common::getDTypeSize(type);

    return (size_t) getGemmWorkSpaceSize(numTokens, mNumLoraModules, mMaxLowRank, mSplitKSlices)
        + getLowRankWorkSpaceSize(numTokens, mNumLoraModules, mMaxLowRank, typeSize)
        + getGemmParamsWorkSpaceSize(std::min(numReqs, numTokens) * mNumLoraModules);
}

void LoraImpl::setBestTactic(std::optional<Config> config)
{
    mBestConfig = config;
}

int LoraImpl::run(int64_t numTokens, int64_t numReqs, void const* input, int32_t const* loraRanks,
    void const* const* loraWeightsPtr, int weightIndex, void* const* outputs, void* workspace, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // inputs
    //     numTokens
    //     numReqs
    //     input [numTokens, K] (view as 2D)
    //     loraRanks [mNumLoraModules, numTokens] on cpu
    //     loraWeightsPtr [mNumLoraModules, numTokens, 2] on cpu
    // outputs
    //     output [-1, N] (view as 2D)
    //     ... (there are mNumLoraModules outputs)

    if (numTokens == 0)
    {
        return 0;
    }

    auto const typeSize = tensorrt_llm::runtime::BufferDataType(mType).getSize();
    setGemmConfig();

    int64_t GemmWorkSpaceSize = getGemmWorkSpaceSize(numTokens, mNumLoraModules, mMaxLowRank, mSplitKSlices);
    int64_t groupGemmParamsWorkSpaceSize = getGemmParamsWorkSpaceSize(std::min(numReqs, numTokens) * mNumLoraModules);
    void* gemmWorkSpace = workspace; // [gemmWorkSpace, lowrankWorkSpace, groupGemmParamsWorkSpace]
    void* lowRankWorkSpace = static_cast<char*>(gemmWorkSpace) + GemmWorkSpaceSize;
    void* groupGemmParamsWorkSpace = static_cast<char*>(lowRankWorkSpace)
        + getLowRankWorkSpaceSize(numTokens, mNumLoraModules, mMaxLowRank, typeSize);

    for (int loraModuleIdx = 0; loraModuleIdx < mNumLoraModules; loraModuleIdx++)
    {
        size_t size = numTokens * mOutHiddenSizes[loraModuleIdx];
        cudaMemsetAsync(outputs[loraModuleIdx], 0, size * typeSize, stream);
    }

    char* useUnifiedGemmChar = std::getenv("LORA_USE_UNIFIED_GEMM");
    bool useUnifiedGemm = (useUnifiedGemmChar == nullptr || std::string(useUnifiedGemmChar) != "OFF");

    for (int loraModuleIdx = 0; loraModuleIdx < mNumLoraModules; loraModuleIdx++)
    {
        auto const loraRankModule = loraRanks[loraModuleIdx * numTokens];
        void const* const* loraWeightsPtrModule = &loraWeightsPtr[loraModuleIdx * numTokens * 2];
        for (int rowId = 0; rowId < numTokens; rowId++)
        {
            if (loraWeightsPtrModule[rowId * 2] != loraWeightsPtrModule[0]
                || loraWeightsPtrModule[rowId * 2 + 1] != loraWeightsPtrModule[1]
                || loraRanks[loraModuleIdx * numTokens + rowId] != loraRankModule)
            {
                useUnifiedGemm = false;
            }
        }
    }

    // TODO can add batch_size == 1 case
    if (useUnifiedGemm)
    {
        for (int loraModuleIdx = 0; loraModuleIdx < mNumLoraModules; loraModuleIdx++)
        {
            auto const* loraWeightsPtrModule
                = reinterpret_cast<int64_t const*>(&loraWeightsPtr[loraModuleIdx * numTokens * 2]);

            int const M = numTokens;

            auto const lora_rank = loraRanks[loraModuleIdx * numTokens];

            auto const N = lora_rank;

            if (N > 0)
            {
                TLLM_CHECK_WITH_INFO(N <= mMaxLowRank,
                    "Invalid low_rank (%d). low_rank must be smaller than mMaxLowRank (%d)", N, mMaxLowRank);
                // size
                auto const K = mInHiddenSize;
                auto const N2 = mOutHiddenSizes[loraModuleIdx];
                // [M, K] * [K, N] -> [M, N]
                // [M, N] * [N, N2] -> [M, N2]

                void* lora_in_weight
                    = reinterpret_cast<void*>(loraWeightsPtrModule[0] + K * N * typeSize * weightIndex);
                void* lora_out_weight
                    = reinterpret_cast<void*>(loraWeightsPtrModule[1] + N2 * N * typeSize * weightIndex);
                void* output = outputs[loraModuleIdx];

                _runGemm(M, N, K, mTransA, mTransB, mType, mCublasWrapper, input, lora_in_weight, lowRankWorkSpace,
                    mBestConfig, gemmWorkSpace, stream);

                _runGemm(M, N2, N, false, mTransB, mType, mCublasWrapper, lowRankWorkSpace, lora_out_weight, output,
                    mBestConfig, gemmWorkSpace, stream);
            }
        }
    }
    else
    {
        std::vector<cutlass::gemm::GemmCoord> problem_sizes;
        problem_sizes.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrA;
        ptrA.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrB;
        ptrB.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrC;
        ptrC.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrD;
        ptrD.reserve(numTokens * mNumLoraModules);

        std::vector<cutlass::gemm::GemmCoord> problem_sizes_2;
        problem_sizes_2.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrA_2;
        ptrA_2.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrB_2;
        ptrB_2.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrC_2;
        ptrC_2.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrD_2;
        ptrD_2.reserve(numTokens * mNumLoraModules);

        int minKN = mInHiddenSize; // Used to determine the alignment size
        for (int loraModuleIdx = 0; loraModuleIdx < mNumLoraModules; loraModuleIdx++)
        {
            auto const* loraWeightsPtrModule
                = reinterpret_cast<int64_t const*>(&loraWeightsPtr[loraModuleIdx * numTokens * 2]);
            int32_t const* loraRanksModule = &loraRanks[loraModuleIdx * numTokens];

            // The following loop aggregates the contiguous requests that use the same LoRA weights to reduce
            // the problem_size of grouped GEMMs and increase the M dimension of those GEMMs.
            int rowId = 0;
            int handled_token_num = 0;
            while (rowId < numTokens)
            {
                auto const lora_rank = loraRanksModule[rowId];
                auto const N = lora_rank;
                int count = 0;
                size_t M = 0;
                while (rowId + count < numTokens && lora_rank == loraRanksModule[rowId + count]
                    && loraWeightsPtrModule[rowId * 2] == loraWeightsPtrModule[(rowId + count) * 2]
                    && loraWeightsPtrModule[rowId * 2 + 1] == loraWeightsPtrModule[(rowId + count) * 2 + 1])
                {
                    M += 1;
                    count++;
                }

                if (N > 0)
                {
                    TLLM_CHECK_WITH_INFO(N <= mMaxLowRank,

                        "Invalid low_rank (%d). low_rank must be smaller than mMaxLowRank (%d)", N, mMaxLowRank);

                    auto const K = mInHiddenSize;
                    minKN = std::min(minKN, N);
                    minKN = std::min(minKN, K);

                    cutlass::gemm::GemmCoord problem(M, N, K);
                    problem_sizes.push_back(problem);

                    ptrA.push_back(static_cast<void*>(
                        static_cast<char*>(const_cast<void*>(input)) + handled_token_num * K * typeSize));
                    ptrB.push_back(
                        reinterpret_cast<void*>(loraWeightsPtrModule[rowId * 2] + K * N * typeSize * weightIndex));
                    ptrC.push_back(static_cast<void*>(static_cast<char*>(lowRankWorkSpace)
                        + (loraModuleIdx * numTokens * mMaxLowRank + handled_token_num * mMaxLowRank) * typeSize));
                    ptrD.push_back(static_cast<void*>(static_cast<char*>(lowRankWorkSpace)
                        + (loraModuleIdx * numTokens * mMaxLowRank + handled_token_num * mMaxLowRank) * typeSize));

                    auto const N2 = mOutHiddenSizes[loraModuleIdx];
                    cutlass::gemm::GemmCoord problem_2(M, N2, N);
                    problem_sizes_2.push_back(problem_2);
                    ptrA_2.push_back(static_cast<void*>(static_cast<char*>(lowRankWorkSpace)
                        + (loraModuleIdx * numTokens * mMaxLowRank + handled_token_num * mMaxLowRank) * typeSize));
                    ptrB_2.push_back(
                        reinterpret_cast<void*>(loraWeightsPtrModule[rowId * 2 + 1] + N2 * N * typeSize * weightIndex));
                    ptrC_2.push_back(static_cast<void*>(
                        static_cast<char*>(outputs[loraModuleIdx]) + handled_token_num * N2 * typeSize));
                    ptrD_2.push_back(static_cast<void*>(
                        static_cast<char*>(outputs[loraModuleIdx]) + handled_token_num * N2 * typeSize));
                }
                handled_token_num += M;
                rowId += count;
            }
            TLLM_CHECK(handled_token_num == numTokens);
        }
        if (problem_sizes.size() > 0)
        {
            TLLM_CHECK_WITH_INFO(mTransA == false && mTransB == true,
                "Invalid transA (%d) transB (%d). transA must be false, transB must be true", int(mTransA),
                int(mTransB));
            // For the first GEMM, K is the "hidden size" and N is the "lora rank". So, K is often much larger than N.
            // To improve the GPU utilization, we use splitK to handle the K dimension in multiple blocks in parallel.
            splitkGroupedGemm(problem_sizes, ptrA, ptrB, ptrC, ptrD, groupGemmParamsWorkSpace,
                groupGemmParamsWorkSpaceSize, gemmWorkSpace, GemmWorkSpaceSize, true, mType, mSplitKSlices, minKN,
                stream);
            sync_check_cuda_error(stream);
            groupedGemm(problem_sizes_2, ptrA_2, ptrB_2, ptrC_2, ptrD_2, groupGemmParamsWorkSpace,
                groupGemmParamsWorkSpaceSize, gemmWorkSpace, GemmWorkSpaceSize, false, mType, minKN, stream);
            sync_check_cuda_error(stream);
        }
    }

    return 0;
}

int Lora_run(LoraImpl* impl, int64_t numTokens, int64_t numReqs, void const* input, int32_t const* loraRanks,
    void const* const* loraWeightsPtr, int weightIndex, void* const* outputs, void* workspace, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(impl != nullptr, "Attempt to run an empty LoraImpl");
    return impl->run(numTokens, numReqs, input, loraRanks, loraWeightsPtr, weightIndex, outputs, workspace, stream);
}

} // namespace tensorrt_llm::kernels

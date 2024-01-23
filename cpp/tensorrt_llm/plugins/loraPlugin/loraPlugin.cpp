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
#include "loraPlugin.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/groupGemm.h"
#include "tensorrt_llm/kernels/splitkGroupGemm.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/cublasVersionCheck.h"
#include <algorithm>

namespace tk = tensorrt_llm::kernels;
using namespace nvinfer1;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::LoraPluginCreator;
using tensorrt_llm::plugins::LoraPlugin;
using tensorrt_llm::plugins::CublasGemmWrapperPtr;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;

static const char* LORA_PLUGIN_VERSION{"1"};
static const char* LORA_PLUGIN_NAME{"Lora"};
PluginFieldCollection LoraPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> LoraPluginCreator::mPluginAttributes;

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
void _runGemm(const int M, const int N, const int K, const bool transA, const bool transB,
    const nvinfer1::DataType type, const CublasGemmWrapperPtr& cublasWrapperPtr, const void* act, const void* weight,
    void* output, const std::optional<cublasLtMatmulHeuristicResult_t>& heuristic, void* workspace, cudaStream_t stream)
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

LoraPlugin::LoraPlugin(int in_hidden_size, std::vector<int> out_hidden_sizes, int transA, int transB,
    int num_lora_modules, nvinfer1::DataType type, const LoraPlugin::PluginProfilerPtr& pluginProfiler,
    bool remove_input_padding, int max_context_length, int max_low_rank)
    : mInHiddenSize(in_hidden_size)
    , mTransA(transA)
    , mTransB(transB)
    , mNumLoraModules(num_lora_modules)
    , mType(type)
    , mPluginProfiler(pluginProfiler)
    , mRemoveInputPadding(remove_input_padding)
    , mMaxContextLength(max_context_length)
    , mMaxLowRank(max_low_rank)
{
    mOutHiddenSizes.resize(mNumLoraModules);
    mOutHiddenSizes.assign(out_hidden_sizes.begin(), out_hidden_sizes.end());
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    init();
}

// Parameterized constructor
LoraPlugin::LoraPlugin(const void* data, size_t length, const LoraPlugin::PluginProfilerPtr& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mInHiddenSize);
    read(d, mTransA);
    read(d, mTransB);
    read(d, mNumLoraModules);
    read(d, mType);
    read(d, mRemoveInputPadding);
    read(d, mMaxContextLength);
    read(d, mMaxLowRank);
    mOutHiddenSizes.resize(mNumLoraModules);
    for (int i = 0; i < mNumLoraModules; i++)
    {
        read(d, mOutHiddenSizes[i]);
    }

    init();

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void LoraPlugin::init()
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    auto cublasHandle = getCublasHandle();
    auto cublasLtHandle = getCublasLtHandle();
    mCublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);

    mPluginProfiler->setTranspose(mTransA, mTransB);

    mGemmId = GemmIdCublas(mDims.n, mDims.k, mType, mTransA, mTransB);
}

void LoraPlugin::setGemmConfig()
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    if (mType == DataType::kHALF)
    {
        mCublasWrapper->setFP16GemmConfig();
    }
    else if (mType == DataType::kFLOAT)
    {
        mCublasWrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        mCublasWrapper->setBF16GemmConfig();
    }
#endif
}

void LoraPlugin::configGemm()
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    if (!mDims.isInitialized())
    {
        return;
    }

    setGemmConfig();

    mPluginProfiler->profileTactics(mCublasWrapper, mType, mDims, mGemmId);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* LoraPlugin::clone() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    auto* plugin = new LoraPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs LoraPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    try
    {
        TLLM_CHECK(outputIndex < mNumLoraModules);
        const int nbDimsA = inputs[getInputTensorIdx()].nbDims;
        DimsExprs ret;
        ret.nbDims = nbDimsA;

        for (int i = 0; i < ret.nbDims; ++i)
        {
            ret.d[0] = 0;
        }

        if (mTransA)
        {
            for (int i = 1; i < nbDimsA; ++i)
            {
                ret.d[i - 1] = inputs[getInputTensorIdx()].d[i];
            }
        }
        else
        {
            for (int i = 0; i < nbDimsA - 1; ++i)
            {
                ret.d[i] = inputs[getInputTensorIdx()].d[i];
            }
        }

        auto const* outHiddenSize = exprBuilder.constant(mOutHiddenSizes.at(outputIndex));
        TLLM_CHECK(outHiddenSize != nullptr);
        ret.d[ret.nbDims - 1] = outHiddenSize;
        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool LoraPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    if (pos == getHostRequestTypesIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (pos >= getLoraRanksIdx() && pos < getLoraRanksIdx() + mNumLoraModules)
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (pos >= getLoraWeightsPtrsIdx() && pos < getLoraWeightsPtrsIdx() + mNumLoraModules)
    {
        return inOut[pos].type == nvinfer1::DataType::kINT64;
    }
    else if (mRemoveInputPadding && pos == getHostContextLengthsIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

int32_t _computeMDimension(bool transA, const int32_t nbDims, const int32_t* dims)
{
    int32_t M = 1;
    if (transA)
    {
        for (int i = nbDims - 1; i > 0; --i)
        {
            M *= dims[i];
        }
    }
    else
    {
        for (int i = 0; i < nbDims - 1; ++i)
        {
            M *= dims[i];
        }
    }
    return M;
}

int32_t _computeNDimension(bool transB, const int32_t nbDims, const int32_t* dims)
{
    int32_t N = 1;
    if (transB)
    {
        for (int i = 0; i < nbDims - 1; ++i)
        {
            N *= dims[i];
        }
    }
    else
    {
        for (int i = nbDims - 1; i > 0; --i)
        {
            N *= dims[i];
        }
    }
    return N;
}

void LoraPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    const int nbDimsA = in[0].max.nbDims;
    const int nbDimsB = in[1].max.nbDims;

    const auto minM = _computeMDimension(mTransA, nbDimsA, in[0].min.d);
    const auto maxM = _computeMDimension(mTransA, nbDimsA, in[0].max.d);
    const auto N = _computeNDimension(mTransB, nbDimsB, in[1].max.d);
    const auto K = mTransA ? in[0].max.d[0] : in[0].max.d[nbDimsA - 1];

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, N, K};
    }
    mGemmId.n = N;
    mGemmId.k = K;
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

int64_t getLowRankWorkSpaceSize(
    int64_t nbReq, int64_t maxContextLength, int64_t maxLoraModuleNum, int64_t maxLowRank, int64_t typeSize)
{
    return divUp(nbReq * maxContextLength * maxLoraModuleNum * maxLowRank * typeSize, 16) * 16;
}

int64_t getGroupedGemmParamsWorkSpaceSize(int64_t nbReq)
{
    return std::max(tk::getSplitkGroupedGemmParamsWorkSpaceSize(nbReq), tk::getGroupedGemmParamsWorkSpaceSize(nbReq));
}

int64_t getSplitkGroupedGemmWorkSpaceSize(
    int64_t nbReq, int64_t maxContextLength, int64_t maxLoraModuleNum, int64_t maxLowRank, int64_t splitKSlices)
{
    return divUp(nbReq * maxContextLength * maxLoraModuleNum * maxLowRank * sizeof(float) * splitKSlices, 16) * 16;
}

int64_t getGemmWorkSpaceSize(
    int64_t nbReq, int64_t maxContextLength, int64_t maxLoraModuleNum, int64_t maxLowRank, int64_t splitKSlices)
{
    return std::max((int64_t) CUBLAS_WORKSPACE_SIZE,
        getSplitkGroupedGemmWorkSpaceSize(nbReq, maxContextLength, maxLoraModuleNum, maxLowRank, splitKSlices));
}

size_t LoraPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    const int nbReq = inputs[getLoraRanksIdx()].dims.d[0];
    auto const type = inputs[getInputTensorIdx()].type;
    auto const typeSize = tensorrt_llm::runtime::BufferDataType(type).getSize();

    return (size_t) getGemmWorkSpaceSize(nbReq, mMaxContextLength, mNumLoraModules, mMaxLowRank, mSplitKSlices)
        + getLowRankWorkSpaceSize(nbReq, mMaxContextLength, mNumLoraModules, mMaxLowRank, typeSize)
        + getGroupedGemmParamsWorkSpaceSize(nbReq * mNumLoraModules);
}

void runCublasGemmEx(const int M, const int N, const int K, const bool transA, const bool transB, const void* act,
    const void* weight, void* output, cublasHandle_t cublas_handle)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    float a = 1.0f;
    float b = 0.0f;
    void* alpha = &a;
    void* beta = &b;
    cublasOperation_t transa, transb;
    int m, n, k;
    int lda, ldb, ldc;
    _getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, transA, transB, M, N, K);

    tensorrt_llm::common::check_cuda_error(cublasGemmEx(cublas_handle, transa, transb, m, n, k, alpha, weight,
        CUDA_R_16F, lda, act, CUDA_R_16F, ldb, beta, output, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

int LoraPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    // inputs
    //     input [-1, K] (view as 2D)
    //     host_request_type [batch_size] on cpu
    //     lora_ranks [batch_size] on cpu
    //     ... (there are mNumLoraModules lora_ranks inputs)
    //     lora_weights_ptr [batch_size, 2] on cpu
    //     ... (there are mNumLoraModules lora_weights_ptr inputs)
    //     host_context_lengths [batch_size] on cpu
    // outputs
    //     output [-1, N] (view as 2D)
    //     ... (there are mNumLoraModules outputs)

    auto const typeSize = tensorrt_llm::runtime::BufferDataType(mType).getSize();
    setGemmConfig();
    auto const batch_size = inputDesc[getLoraRanksIdx()].dims.d[0];
    auto const host_context_lengths
        = mRemoveInputPadding ? static_cast<int32_t const*>(inputs[getHostContextLengthsIdx()]) : nullptr;
    RequestType const* reqTypes = static_cast<RequestType const*>(inputs[getHostRequestTypesIdx()]);

    int64_t GemmWorkSpaceSize
        = getGemmWorkSpaceSize(batch_size, mMaxContextLength, mNumLoraModules, mMaxLowRank, mSplitKSlices);
    int64_t groupGemmParamsWorkSpaceSize = getGroupedGemmParamsWorkSpaceSize(batch_size * mNumLoraModules);
    void* gemmWorkSpace = workspace; // [gemmWorkSpace, lowrankWorkSpace, groupGemmParamsWorkSpace]
    void* lowRankWorkSpace = static_cast<char*>(gemmWorkSpace) + GemmWorkSpaceSize;
    void* groupGemmParamsWorkSpace = static_cast<char*>(lowRankWorkSpace)
        + getLowRankWorkSpaceSize(batch_size, mMaxContextLength, mNumLoraModules, mMaxLowRank, typeSize);

    const int nbDimsA = inputDesc[0].dims.nbDims;
    for (int loraModuleIdx = 0; loraModuleIdx < mNumLoraModules; loraModuleIdx++)
    {
        size_t size = 1;
        for (int i = 0; i < outputDesc[loraModuleIdx].dims.nbDims; i++)
        {
            size *= outputDesc[loraModuleIdx].dims.d[i];
        }
        cudaMemsetAsync(outputs[loraModuleIdx], 0, size * typeSize, stream);
    }

    char* useUnifiedGemmChar = std::getenv("LORA_USE_UNIFIED_GEMM");
    bool useUnifiedGemm = (useUnifiedGemmChar != nullptr && std::string(useUnifiedGemmChar) == "ON");
    for (int batchIdx = 0; batchIdx < batch_size; batchIdx++)
    {
        for (int loraModuleIdx = 0; loraModuleIdx < mNumLoraModules; loraModuleIdx++)
        {
            auto const lora_ranks = static_cast<int32_t const*>(inputs[getLoraRanksIdx() + loraModuleIdx]);
            auto const lora_weights_ptr = static_cast<int64_t const*>(inputs[getLoraWeightsPtrsIdx() + loraModuleIdx]);
            if (lora_weights_ptr[batchIdx * 2] != lora_weights_ptr[0]
                || lora_weights_ptr[batchIdx * 2 + 1] != lora_weights_ptr[1] || lora_ranks[batchIdx] == 0)
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
            auto const lora_ranks = static_cast<int32_t const*>(inputs[getLoraRanksIdx() + loraModuleIdx]);
            auto const lora_weights_ptr = static_cast<int64_t const*>(inputs[getLoraWeightsPtrsIdx() + loraModuleIdx]);

            const RequestType reqType = reqTypes[0];
            int M = 0;
            for (int batchIdx = 0; batchIdx < batch_size; batchIdx++)
            {
                M += (reqType != RequestType::kCONTEXT)
                    ? 1
                    : (mRemoveInputPadding ? host_context_lengths[batchIdx] : inputDesc[0].dims.d[1]);
            }

            const auto lora_rank = lora_ranks[0];
            auto bestTactic = mPluginProfiler->getBestConfig(M, mGemmId);

            const auto N = lora_rank;

            if (N > 0)
            {
                TLLM_CHECK_WITH_INFO(N <= mMaxLowRank,
                    fmtstr("Invalid low_rank (%d). low_rank must be smaller than mMaxLowRank (%d)", N, mMaxLowRank));
                const auto K = mTransA ? inputDesc[0].dims.d[0] : inputDesc[0].dims.d[nbDimsA - 1]; // input hidden size
                const auto N2 = outputDesc[loraModuleIdx].dims.d[nbDimsA - 1];
                // [M, K] -> [M, N] -> [M, N2]

                void* lora_in_weight = reinterpret_cast<void*>(lora_weights_ptr[0]);
                void* lora_out_weight = reinterpret_cast<void*>(lora_weights_ptr[1]);
                const void* input = inputs[0];
                void* output = outputs[loraModuleIdx];

                _runGemm(M, N, K, mTransA, mTransB, mType, mCublasWrapper, input, lora_in_weight, lowRankWorkSpace,
                    bestTactic, gemmWorkSpace, stream);

                _runGemm(M, N2, N, mTransA, mTransB, mType, mCublasWrapper, lowRankWorkSpace, lora_out_weight, output,
                    bestTactic, gemmWorkSpace, stream);
            }
        }
    }
    else
    {
        std::vector<cutlass::gemm::GemmCoord> problem_sizes;
        problem_sizes.reserve(batch_size * mNumLoraModules);
        std::vector<void*> ptrA;
        ptrA.reserve(batch_size * mNumLoraModules);
        std::vector<void*> ptrB;
        ptrB.reserve(batch_size * mNumLoraModules);
        std::vector<void*> ptrC;
        ptrC.reserve(batch_size * mNumLoraModules);
        std::vector<void*> ptrD;
        ptrD.reserve(batch_size * mNumLoraModules);

        std::vector<cutlass::gemm::GemmCoord> problem_sizes_2;
        problem_sizes_2.reserve(batch_size * mNumLoraModules);
        std::vector<void*> ptrA_2;
        ptrA_2.reserve(batch_size * mNumLoraModules);
        std::vector<void*> ptrB_2;
        ptrB_2.reserve(batch_size * mNumLoraModules);
        std::vector<void*> ptrC_2;
        ptrC_2.reserve(batch_size * mNumLoraModules);
        std::vector<void*> ptrD_2;
        ptrD_2.reserve(batch_size * mNumLoraModules);

        std::vector<int64_t> splitkBufferOffsets;
        splitkBufferOffsets.push_back(0);
        for (int loraModuleIdx = 0; loraModuleIdx < mNumLoraModules; loraModuleIdx++)
        {
            auto const lora_ranks = static_cast<int32_t const*>(inputs[getLoraRanksIdx() + loraModuleIdx]);
            auto const lora_weights_ptr = static_cast<int64_t const*>(inputs[getLoraWeightsPtrsIdx() + loraModuleIdx]);

            // The following loop aggregates the contiguous requests that use the same LoRA weights to reduce
            // the problem_size of grouped GEMMs and increase the M dimension of those GEMMs.
            int batchIdx = 0;
            int handled_token_num = 0;
            while (batchIdx < batch_size)
            {
                const auto lora_rank = lora_ranks[batchIdx];
                const auto N = lora_rank;
                int count = 0;
                size_t M = 0;
                while (batchIdx + count < batch_size && lora_rank == lora_ranks[batchIdx + count]
                    && lora_weights_ptr[batchIdx * 2] == lora_weights_ptr[(batchIdx + count) * 2]
                    && lora_weights_ptr[batchIdx * 2 + 1] == lora_weights_ptr[(batchIdx + count) * 2 + 1])
                {
                    const RequestType reqType = reqTypes[batchIdx + count];
                    M += (reqType != RequestType::kCONTEXT)
                        ? 1
                        : (mRemoveInputPadding ? host_context_lengths[batchIdx + count] : inputDesc[0].dims.d[1]);
                    count++;
                }

                if (N > 0)
                {
                    TLLM_CHECK_WITH_INFO(N <= mMaxLowRank,
                        fmtstr(
                            "Invalid low_rank (%d). low_rank must be smaller than mMaxLowRank (%d)", N, mMaxLowRank));
                    const auto K
                        = mTransA ? inputDesc[0].dims.d[0] : inputDesc[0].dims.d[nbDimsA - 1]; // input hidden size

                    cutlass::gemm::GemmCoord problem(M, N, K);
                    problem_sizes.push_back(problem);

                    ptrA.push_back(static_cast<void*>(
                        static_cast<char*>(const_cast<void*>(inputs[0])) + handled_token_num * K * typeSize));
                    ptrB.push_back(reinterpret_cast<void*>(lora_weights_ptr[batchIdx * 2]));
                    ptrC.push_back(static_cast<void*>(static_cast<char*>(lowRankWorkSpace)
                        + (loraModuleIdx * batch_size * mMaxContextLength * mMaxLowRank
                              + handled_token_num * mMaxLowRank)
                            * typeSize));
                    ptrD.push_back(static_cast<void*>(static_cast<char*>(lowRankWorkSpace)
                        + (loraModuleIdx * batch_size * mMaxContextLength * mMaxLowRank
                              + handled_token_num * mMaxLowRank)
                            * typeSize));

                    const auto N2 = outputDesc[loraModuleIdx].dims.d[nbDimsA - 1];
                    cutlass::gemm::GemmCoord problem_2(M, N2, N);
                    problem_sizes_2.push_back(problem_2);
                    ptrA_2.push_back(static_cast<void*>(static_cast<char*>(lowRankWorkSpace)
                        + (loraModuleIdx * batch_size * mMaxContextLength * mMaxLowRank
                              + handled_token_num * mMaxLowRank)
                            * typeSize));
                    ptrB_2.push_back(reinterpret_cast<void*>(lora_weights_ptr[batchIdx * 2 + 1]));
                    ptrC_2.push_back(static_cast<void*>(
                        static_cast<char*>(outputs[loraModuleIdx]) + handled_token_num * N2 * typeSize));
                    ptrD_2.push_back(static_cast<void*>(
                        static_cast<char*>(outputs[loraModuleIdx]) + handled_token_num * N2 * typeSize));
                }
                handled_token_num += M;
                batchIdx += count;
                splitkBufferOffsets.push_back(splitkBufferOffsets.at(splitkBufferOffsets.size() - 1) + M * N);
            }
        }
        if (problem_sizes.size() > 0)
        {
            tk::splitkGroupedGemm(problem_sizes, ptrA, ptrB, ptrC, ptrD, groupGemmParamsWorkSpace,
                groupGemmParamsWorkSpaceSize, gemmWorkSpace, GemmWorkSpaceSize, splitkBufferOffsets, true, mType,
                mSplitKSlices, stream);
            sync_check_cuda_error();
            tk::groupedGemm(problem_sizes_2, ptrA_2, ptrB_2, ptrC_2, ptrD_2, groupGemmParamsWorkSpace,
                groupGemmParamsWorkSpaceSize, gemmWorkSpace, GemmWorkSpaceSize, false, mType, stream);
            sync_check_cuda_error();
        }
    }

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType LoraPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    TLLM_CHECK(index < mNumLoraModules);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* LoraPlugin::getPluginType() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return LORA_PLUGIN_NAME;
}

const char* LoraPlugin::getPluginVersion() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return LORA_PLUGIN_VERSION;
}

int LoraPlugin::getNbOutputs() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return mNumLoraModules;
}

int LoraPlugin::initialize() noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    configGemm();
    return 0;
}

void LoraPlugin::destroy() noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    delete this;
}

size_t LoraPlugin::getSerializationSize() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return sizeof(mInHiddenSize) + sizeof(mTransA) + sizeof(mTransB) + sizeof(mNumLoraModules) + sizeof(mType)
        + mPluginProfiler->getSerializationSize(mGemmId) + sizeof(mRemoveInputPadding) + sizeof(mMaxContextLength)
        + sizeof(mMaxLowRank) + sizeof(int) * mNumLoraModules; // selected tactics container size
}

void LoraPlugin::serialize(void* buffer) const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mInHiddenSize);
    write(d, mTransA);
    write(d, mTransB);
    write(d, mNumLoraModules);
    write(d, mType);
    write(d, mRemoveInputPadding);
    write(d, mMaxContextLength);
    write(d, mMaxLowRank);
    for (int i = 0; i < mNumLoraModules; i++)
    {
        write(d, mOutHiddenSizes.at(i));
    }
    mPluginProfiler->serialize(d, mGemmId);

    assert(d == a + getSerializationSize());
}

void LoraPlugin::terminate() noexcept {}

///////////////

LoraPluginCreator::LoraPluginCreator()
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("transA", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("transB", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("num_lora_modules", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LoraPluginCreator::getPluginName() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return LORA_PLUGIN_NAME;
}

const char* LoraPluginCreator::getPluginVersion() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return LORA_PLUGIN_VERSION;
}

const PluginFieldCollection* LoraPluginCreator::getFieldNames() noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return &mFC;
}

IPluginV2* LoraPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);

    const PluginField* fields = fc->fields;
    nvinfer1::DataType type;
    int num_lora_modules;
    int in_hidden_size, transA, transB;
    bool remove_input_padding;
    int max_context_length;
    int max_low_rank;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "in_hidden_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            in_hidden_size = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "transa"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            transA = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "transb"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            transB = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "remove_input_padding"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            remove_input_padding = static_cast<bool>(*(static_cast<const int8_t*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "max_context_length"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            max_context_length = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "max_low_rank"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            max_low_rank = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "num_lora_modules"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            num_lora_modules = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
    }
    std::vector<int> out_hidden_sizes;
    out_hidden_sizes.resize(num_lora_modules);
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        for (int j = 0; j < num_lora_modules; j++)
        {
            if (!strcmp(attrName, fmtstr("out_hidden_size_%d", j).c_str()))
            {
                TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
                out_hidden_sizes.at(j) = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
        }
    }
    try
    {
        // LoraPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        // FIXME enable tactic profiler
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ false, /* skip */ true);
        auto* obj = new LoraPlugin(in_hidden_size, out_hidden_sizes, transA, transB, num_lora_modules, type,
            pluginProfiler, remove_input_padding, max_context_length, max_low_rank);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* LoraPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    // This object will be deleted when the network is destroyed, which will
    // call LoraPlugin::destroy()
    try
    {
        // LoraPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        // FIXME enable tactic profiler
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ true, /* skip */ true);
        auto* obj = new LoraPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

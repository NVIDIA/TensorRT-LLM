/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION &
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

/**
 * @file trtllmGenFmhaOp.cpp
 * @brief PyTorch bindings for TRTLLM-Gen FMHA Runner.
 *
 * This file exports the TllmGenFmhaRunner class to Python
 * as torch.classes.trtllm.TrtllmGenFmhaRunner.
 *
 * The design philosophy is to keep the C++ bindings thin and delegate all
 * selection logic to Python. The runner provides:
 * - run for executing the FMHA kernel
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunnerParams.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

using tensorrt_llm::kernels::Data_type;
using tensorrt_llm::kernels::FmhaKernelType;
using tensorrt_llm::kernels::QkvLayout;
using tensorrt_llm::kernels::TileScheduler;
using tensorrt_llm::kernels::TllmGenFmhaRunner;
using tensorrt_llm::kernels::TllmGenFmhaRunnerParams;
using tensorrt_llm::kernels::TrtllmGenAttentionMaskType;

namespace
{

/**
 * @brief Convert PyTorch scalar type to internal Data_type.
 */
Data_type scalarTypeToDataType(at::ScalarType scalarType)
{
    switch (scalarType)
    {
    case at::ScalarType::Half: return Data_type::DATA_TYPE_FP16;
    case at::ScalarType::BFloat16: return Data_type::DATA_TYPE_BF16;
    case at::ScalarType::Float: return Data_type::DATA_TYPE_FP32;
    case at::ScalarType::Float8_e4m3fn: return Data_type::DATA_TYPE_E4M3;
    default: TLLM_THROW("Unsupported scalar type for TRTLLM-Gen FMHA: %s", toString(scalarType));
    }
}

/**
 * @brief Convert integer to QkvLayout enum.
 */
QkvLayout intToQkvLayout(int64_t layout)
{
    switch (layout)
    {
    case 0: return QkvLayout::SeparateQkv;
    case 1: return QkvLayout::PackedQkv;
    case 2: return QkvLayout::PagedKv;
    case 3: return QkvLayout::ContiguousKv;
    default: TLLM_THROW("Invalid QkvLayout value: %ld", layout);
    }
}

/**
 * @brief Convert integer to TrtllmGenAttentionMaskType enum.
 */
TrtllmGenAttentionMaskType intToMaskType(int64_t maskType)
{
    switch (maskType)
    {
    case 0: return TrtllmGenAttentionMaskType::Dense;
    case 1: return TrtllmGenAttentionMaskType::Causal;
    case 2: return TrtllmGenAttentionMaskType::SlidingOrChunkedCausal;
    case 3: return TrtllmGenAttentionMaskType::Custom;
    default: TLLM_THROW("Invalid TrtllmGenAttentionMaskType value: %ld", maskType);
    }
}

/**
 * @brief Convert integer to FmhaKernelType enum.
 */
FmhaKernelType intToKernelType(int64_t kernelType)
{
    switch (kernelType)
    {
    case 0: return FmhaKernelType::Context;
    case 1: return FmhaKernelType::Generation;
    case 2: return FmhaKernelType::SwapsMmaAbForGeneration;
    case 3: return FmhaKernelType::KeepsMmaAbForGeneration;
    case 4: return FmhaKernelType::SpecDecodingGeneration;
    default: TLLM_THROW("Invalid FmhaKernelType value: %ld", kernelType);
    }
}

/**
 * @brief Convert integer to TileScheduler enum.
 */
TileScheduler intToTileScheduler(int64_t scheduler)
{
    switch (scheduler)
    {
    case 0: return TileScheduler::Static;
    case 1: return TileScheduler::Persistent;
    default: TLLM_THROW("Invalid TileScheduler value: %ld", scheduler);
    }
}

/**
 * @brief Helper function to build TllmGenFmhaRunnerParams from unpacked arguments.
 */
TllmGenFmhaRunnerParams buildParams(
    // Enum fields
    int64_t qkvLayout, int64_t maskType, int64_t kernelType, int64_t tileScheduler, bool multiCtasKvMode,
    bool useBlockSparseAttention,
    // Pointer fields (tensors)
    std::optional<torch::Tensor> const& qPtr, std::optional<torch::Tensor> const& kPtr,
    std::optional<torch::Tensor> const& vPtr, std::optional<torch::Tensor> const& kvPtr,
    std::optional<torch::Tensor> const& kvSfPtr, std::optional<torch::Tensor> const& qkvPtr,
    std::optional<torch::Tensor> const& attentionSinksPtr, std::optional<torch::Tensor> const& customMaskPtr,
    std::optional<torch::Tensor> const& customMaskOffsetsPtr,
    std::optional<torch::Tensor> const& firstSparseMaskOffsetsKvPtr,
    std::optional<torch::Tensor> const& multiCtasKvCounterPtr, std::optional<torch::Tensor> const& seqLensKvPtr,
    std::optional<torch::Tensor> const& cumSeqLensQPtr, std::optional<torch::Tensor> const& cumSeqLensKvPtr,
    std::optional<torch::Tensor> const& kvPageIdxPtr, std::optional<torch::Tensor> const& outputScalePtr,
    std::optional<torch::Tensor> const& scaleSoftmaxLog2Ptr, std::optional<torch::Tensor> const& kvSfScalePtr,
    std::optional<torch::Tensor> const& oSfScalePtr, std::optional<torch::Tensor> const& multiCtasKvScratchPtr,
    std::optional<torch::Tensor> const& softmaxStatsPtr, std::optional<torch::Tensor> const& oPtr,
    std::optional<torch::Tensor> const& oSfPtr, std::optional<torch::Tensor> const& seqlensQPtr,
    // Scalar fields
    int64_t headDimQk, int64_t headDimV, int64_t headDimQkNope, int64_t numHeadsQ, int64_t numHeadsKv,
    int64_t numHeadsQPerKv, int64_t batchSize, int64_t maxSeqLenCacheKv, int64_t maxSeqLenQ, int64_t maxSeqLenKv,
    int64_t attentionWindowSize, int64_t chunkedAttentionSize, int64_t sumOfSeqLensQ, int64_t sumOfSeqLensKv,
    int64_t maxNumPagesPerSeqKv, int64_t numTokensPerPage, int64_t numPagesInMemPool, int64_t multiProcessorCount,
    double scaleQ, int64_t sfStartTokenIdx, double skipSoftmaxThresholdScaleFactor, bool sparseMla,
    int64_t sparseMlaTopK, int64_t layerIdx, bool isSpecDecTree)
{
    TllmGenFmhaRunnerParams params{};

    // Set enum fields
    params.mQkvLayout = intToQkvLayout(qkvLayout);
    params.mMaskType = intToMaskType(maskType);
    params.mKernelType = intToKernelType(kernelType);
    params.mTileScheduler = intToTileScheduler(tileScheduler);
    params.mMultiCtasKvMode = multiCtasKvMode;
    params.mUseBlockSparseAttention = useBlockSparseAttention;

    // Set pointer fields
    params.qPtr = qPtr.has_value() ? qPtr->data_ptr() : nullptr;
    params.kPtr = kPtr.has_value() ? kPtr->data_ptr() : nullptr;
    params.vPtr = vPtr.has_value() ? vPtr->data_ptr() : nullptr;
    params.kvPtr = kvPtr.has_value() ? kvPtr->data_ptr() : nullptr;
    params.kvSfPtr = kvSfPtr.has_value() ? kvSfPtr->data_ptr() : nullptr;
    params.qkvPtr = qkvPtr.has_value() ? qkvPtr->data_ptr() : nullptr;
    params.attentionSinksPtr = attentionSinksPtr.has_value() ? attentionSinksPtr->data_ptr<float>() : nullptr;
    params.customMaskPtr = customMaskPtr.has_value() ? customMaskPtr->data_ptr<uint32_t>() : nullptr;
    params.customMaskOffsetsPtr
        = customMaskOffsetsPtr.has_value() ? customMaskOffsetsPtr->data_ptr<int64_t>() : nullptr;
    params.firstSparseMaskOffsetsKvPtr
        = firstSparseMaskOffsetsKvPtr.has_value() ? firstSparseMaskOffsetsKvPtr->data_ptr<int32_t>() : nullptr;
    params.multiCtasKvCounterPtr
        = multiCtasKvCounterPtr.has_value() ? multiCtasKvCounterPtr->data_ptr<int32_t>() : nullptr;
    params.seqLensKvPtr = seqLensKvPtr.has_value() ? seqLensKvPtr->data_ptr<int>() : nullptr;
    params.cumSeqLensQPtr = cumSeqLensQPtr.has_value() ? cumSeqLensQPtr->data_ptr<int>() : nullptr;
    params.cumSeqLensKvPtr = cumSeqLensKvPtr.has_value() ? cumSeqLensKvPtr->data_ptr<int>() : nullptr;
    params.kvPageIdxPtr = kvPageIdxPtr.has_value() ? kvPageIdxPtr->data_ptr<int>() : nullptr;
    params.outputScalePtr = outputScalePtr.has_value() ? outputScalePtr->data_ptr<float>() : nullptr;
    params.scaleSoftmaxLog2Ptr = scaleSoftmaxLog2Ptr.has_value() ? scaleSoftmaxLog2Ptr->data_ptr<float>() : nullptr;
    params.kvSfScalePtr = kvSfScalePtr.has_value() ? kvSfScalePtr->data_ptr<float>() : nullptr;
    params.oSfScalePtr = oSfScalePtr.has_value() ? oSfScalePtr->data_ptr<float>() : nullptr;
    params.multiCtasKvScratchPtr = multiCtasKvScratchPtr.has_value() ? multiCtasKvScratchPtr->data_ptr() : nullptr;
    params.softmaxStatsPtr
        = softmaxStatsPtr.has_value() ? reinterpret_cast<float2*>(softmaxStatsPtr->data_ptr()) : nullptr;
    params.oPtr = oPtr.has_value() ? oPtr->data_ptr() : nullptr;
    params.oSfPtr = oSfPtr.has_value() ? oSfPtr->data_ptr() : nullptr;
    params.seqlensQPtr = seqlensQPtr.has_value() ? seqlensQPtr->data_ptr<int>() : nullptr;

    // Set scalar fields
    params.mHeadDimQk = static_cast<int>(headDimQk);
    params.mHeadDimV = static_cast<int>(headDimV);
    params.mHeadDimQkNope = static_cast<int>(headDimQkNope);
    params.mNumHeadsQ = static_cast<int>(numHeadsQ);
    params.mNumHeadsKv = static_cast<int>(numHeadsKv);
    params.mNumHeadsQPerKv = static_cast<int>(numHeadsQPerKv);
    params.mBatchSize = static_cast<int>(batchSize);
    params.mMaxSeqLenCacheKv = static_cast<int>(maxSeqLenCacheKv);
    params.mMaxSeqLenQ = static_cast<int>(maxSeqLenQ);
    params.mMaxSeqLenKv = static_cast<int>(maxSeqLenKv);
    params.mAttentionWindowSize = static_cast<int>(attentionWindowSize);
    params.mChunkedAttentionSize = static_cast<int>(chunkedAttentionSize);
    params.mSumOfSeqLensQ = static_cast<int>(sumOfSeqLensQ);
    params.mSumOfSeqLensKv = static_cast<int>(sumOfSeqLensKv);
    params.mMaxNumPagesPerSeqKv = static_cast<int>(maxNumPagesPerSeqKv);
    params.mNumTokensPerPage = static_cast<int>(numTokensPerPage);
    params.mNumPagesInMemPool = static_cast<int>(numPagesInMemPool);
    params.mMultiProcessorCount = static_cast<int>(multiProcessorCount);
    params.mScaleQ = static_cast<float>(scaleQ);
    params.mSfStartTokenIdx = static_cast<int>(sfStartTokenIdx);
    params.mSkipSoftmaxThresholdScaleFactor = static_cast<float>(skipSoftmaxThresholdScaleFactor);
    params.mSparseMla = sparseMla;
    params.mSparseMlaTopK = static_cast<int>(sparseMlaTopK);
    params.mLayerIdx = static_cast<int32_t>(layerIdx);
    params.mIsSpecDecTree = isSpecDecTree;

    // Set stream
    params.stream = at::cuda::getCurrentCUDAStream().stream();

    return params;
}

} // anonymous namespace

/**
 * @class TrtllmGenFmhaRunnerWrapper
 * @brief A wrapper class for TllmGenFmhaRunner that can be exported to Python.
 *
 * This class provides a thin interface to the underlying runner:
 * - run: Execute the FMHA kernel
 */
class TrtllmGenFmhaRunnerWrapper : public torch::CustomClassHolder
{
public:
    /**
     * @brief Construct a TrtllmGenFmhaRunner with the specified data types.
     * @param dtypeQ Data type for Q (0=FP16, 1=BF16, 2=FP32, 3=E4M3)
     * @param dtypeKv Data type for KV (0=FP16, 1=BF16, 2=FP32, 3=E4M3, 4=E2M1)
     * @param dtypeOut Data type for output (0=FP16, 1=BF16, 2=FP32, 3=E4M3, 4=E2M1)
     */
    TrtllmGenFmhaRunnerWrapper(int64_t dtypeQ, int64_t dtypeKv, int64_t dtypeOut)
    {
        auto const qType = intToDataType(dtypeQ);
        auto const kvType = intToDataType(dtypeKv);
        auto const outType = intToDataType(dtypeOut);
        mRunner = std::make_unique<TllmGenFmhaRunner>(qType, kvType, outType);
    }

    /**
     * @brief Run the FMHA kernel with the given parameters.
     */
    void run(
        // Enum fields
        int64_t qkvLayout, int64_t maskType, int64_t kernelType, int64_t tileScheduler, bool multiCtasKvMode,
        bool useBlockSparseAttention,
        // Pointer fields (tensors)
        std::optional<torch::Tensor> const& qPtr, std::optional<torch::Tensor> const& kPtr,
        std::optional<torch::Tensor> const& vPtr, std::optional<torch::Tensor> const& kvPtr,
        std::optional<torch::Tensor> const& kvSfPtr, std::optional<torch::Tensor> const& qkvPtr,
        std::optional<torch::Tensor> const& attentionSinksPtr, std::optional<torch::Tensor> const& customMaskPtr,
        std::optional<torch::Tensor> const& customMaskOffsetsPtr,
        std::optional<torch::Tensor> const& firstSparseMaskOffsetsKvPtr,
        std::optional<torch::Tensor> const& multiCtasKvCounterPtr, std::optional<torch::Tensor> const& seqLensKvPtr,
        std::optional<torch::Tensor> const& cumSeqLensQPtr, std::optional<torch::Tensor> const& cumSeqLensKvPtr,
        std::optional<torch::Tensor> const& kvPageIdxPtr, std::optional<torch::Tensor> const& outputScalePtr,
        std::optional<torch::Tensor> const& scaleSoftmaxLog2Ptr, std::optional<torch::Tensor> const& kvSfScalePtr,
        std::optional<torch::Tensor> const& oSfScalePtr, std::optional<torch::Tensor> const& multiCtasKvScratchPtr,
        std::optional<torch::Tensor> const& softmaxStatsPtr, std::optional<torch::Tensor> const& oPtr,
        std::optional<torch::Tensor> const& oSfPtr, std::optional<torch::Tensor> const& seqlensQPtr,
        // Scalar fields
        int64_t headDimQk, int64_t headDimV, int64_t headDimQkNope, int64_t numHeadsQ, int64_t numHeadsKv,
        int64_t numHeadsQPerKv, int64_t batchSize, int64_t maxSeqLenCacheKv, int64_t maxSeqLenQ, int64_t maxSeqLenKv,
        int64_t attentionWindowSize, int64_t chunkedAttentionSize, int64_t sumOfSeqLensQ, int64_t sumOfSeqLensKv,
        int64_t maxNumPagesPerSeqKv, int64_t numTokensPerPage, int64_t numPagesInMemPool, int64_t multiProcessorCount,
        double scaleQ, int64_t sfStartTokenIdx, double skipSoftmaxThresholdScaleFactor, bool sparseMla,
        int64_t sparseMlaTopK, int64_t layerIdx, bool isSpecDecTree) const
    {
        auto params = buildParams(qkvLayout, maskType, kernelType, tileScheduler, multiCtasKvMode,
            useBlockSparseAttention, qPtr, kPtr, vPtr, kvPtr, kvSfPtr, qkvPtr, attentionSinksPtr, customMaskPtr,
            customMaskOffsetsPtr, firstSparseMaskOffsetsKvPtr, multiCtasKvCounterPtr, seqLensKvPtr, cumSeqLensQPtr,
            cumSeqLensKvPtr, kvPageIdxPtr, outputScalePtr, scaleSoftmaxLog2Ptr, kvSfScalePtr, oSfScalePtr,
            multiCtasKvScratchPtr, softmaxStatsPtr, oPtr, oSfPtr, seqlensQPtr, headDimQk, headDimV, headDimQkNope,
            numHeadsQ, numHeadsKv, numHeadsQPerKv, batchSize, maxSeqLenCacheKv, maxSeqLenQ, maxSeqLenKv,
            attentionWindowSize, chunkedAttentionSize, sumOfSeqLensQ, sumOfSeqLensKv, maxNumPagesPerSeqKv,
            numTokensPerPage, numPagesInMemPool, multiProcessorCount, scaleQ, sfStartTokenIdx,
            skipSoftmaxThresholdScaleFactor, sparseMla, sparseMlaTopK, layerIdx, isSpecDecTree);
        mRunner->run(params);
    }

private:
    static Data_type intToDataType(int64_t dtype)
    {
        switch (dtype)
        {
        case 0: return Data_type::DATA_TYPE_FP16;
        case 1: return Data_type::DATA_TYPE_BF16;
        case 2: return Data_type::DATA_TYPE_FP32;
        case 3: return Data_type::DATA_TYPE_E4M3;
        case 4: return Data_type::DATA_TYPE_E2M1;
        default: TLLM_THROW("Invalid data type value: %ld", dtype);
        }
    }

    std::unique_ptr<TllmGenFmhaRunner> mRunner;
};

} // namespace torch_ext

TRTLLM_NAMESPACE_END

// PyTorch bindings
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    // Register the runner wrapper class
    m.class_<tensorrt_llm::torch_ext::TrtllmGenFmhaRunnerWrapper>("TrtllmGenFmhaRunner")
        .def(torch::init<int64_t, int64_t, int64_t>())
        .def("run", &tensorrt_llm::torch_ext::TrtllmGenFmhaRunnerWrapper::run);
}

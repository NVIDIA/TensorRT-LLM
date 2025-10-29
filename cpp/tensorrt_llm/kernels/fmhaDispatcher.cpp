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

#include "fmhaDispatcher.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

QkvLayout AttentionInputLayoutToQkvLayout(AttentionInputLayout layout)
{
    if (layout == AttentionInputLayout::PACKED_QKV)
    {
        return QkvLayout::PackedQkv;
    }
    else if (layout == AttentionInputLayout::Q_CONTIGUOUS_KV)
    {
        return QkvLayout::ContiguousKv;
    }
    else if (layout == AttentionInputLayout::Q_PAGED_KV)
    {
        return QkvLayout::PagedKv;
    }
    else if (layout == AttentionInputLayout::SEPARATE_Q_K_V)
    {
        return QkvLayout::SeparateQkv;
    }
    TLLM_CHECK_WITH_INFO(false, "Unexpected AttentionInputLayout");
    return QkvLayout::SeparateQkv;
}

FmhaDispatcher::FmhaDispatcher(MHARunnerFixedParams fixedParams)
    : mFixedParams(fixedParams)
    // TRTLLM-GEN only supports power of 2 head sizes.
    // The exception will fall back to fmha v2.
    // Please update fmha_v2/setup.py if you want to add more supported head sizes.
    , mUseTllmGen(tensorrt_llm::common::isSM100Family() && fixedParams.headSize != 80)
{
    if (mUseTllmGen)
    {
        mTllmGenFMHARunner.reset(
            new TllmGenFmhaRunner(mFixedParams.dataType, mFixedParams.dataTypeKv, mFixedParams.dataTypeOut));
        if (!isSupported())
        {
            TLLM_LOG_WARNING("TRTLLM-GEN does not support the requested kernels.");
        }
    }
    else
    {
        TLLM_CHECK_WITH_INFO(mFixedParams.dataType == mFixedParams.dataTypeKv,
            "KV cache data type %s is not the same as input data type %s.",
            data_type_to_string(mFixedParams.dataTypeKv).c_str(), data_type_to_string(mFixedParams.dataType).c_str());

        // For FP8 MLA generation, the output type is BF16, which could be different from the input type.
        // So we shouldn't do this check anymore.
        // TLLM_CHECK_WITH_INFO(mFixedParams.dataType == mFixedParams.dataTypeOut,
        //     "Output data type should be the same as input data type.");

        mFMHARunner.reset(new FusedMHARunnerV2(fixedParams));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool FmhaDispatcher::isSupported()
{
    bool foundKernels = false;
    if (mUseTllmGen)
    {
        if (mFixedParams.attentionMaskType == ContextAttentionMaskType::CUSTOM_MASK)
        {
            TLLM_LOG_WARNING("TRTLLM-GEN does not support custom mask.");
            return false;
        }
        if (mFixedParams.hasAlibi)
        {
            TLLM_LOG_WARNING("TRTLLM-GEN does not support ALiBi.");
            return false;
        }
        if (mFixedParams.isSPadded)
        {
            TLLM_LOG_WARNING("TRTLLM-GEN does not support padded inputs.");
            return false;
        }

        auto qkvLayout = AttentionInputLayoutToQkvLayout(mFixedParams.attentionInputLayout);
        // Create TllmGenFmhaRunnerParams based on MHARunnerFixedParams. Only fill necessary
        // attributes for kernel selection.
        TllmGenFmhaRunnerParams tllmRunnerParams;
        memset(&tllmRunnerParams, 0, sizeof(tllmRunnerParams));
        tllmRunnerParams.mQkvLayout = qkvLayout;
        tllmRunnerParams.setAttentionMaskType(static_cast<std::int8_t>(mFixedParams.attentionMaskType));
        tllmRunnerParams.mKernelType = FmhaKernelType::Context;
        tllmRunnerParams.mTileScheduler = TileScheduler::Persistent;
        tllmRunnerParams.mMultiCtasKvMode = false;
        // Assume same headDim for Qk and V here.
        tllmRunnerParams.mHeadDimQk = mFixedParams.headSize;
        tllmRunnerParams.mHeadDimV = mFixedParams.headSizeV;
        tllmRunnerParams.mNumTokensPerPage = mFixedParams.numTokensPerBlock;
        tllmRunnerParams.mNumHeadsQPerKv = mFixedParams.numQHeads / mFixedParams.numKvHeads;
        // Set the chunked attention size and sliding window size to INT_MAX to disable them when checking if
        // the kernel is supported.
        tllmRunnerParams.mChunkedAttentionSize = INT_MAX;
        tllmRunnerParams.mAttentionWindowSize = INT_MAX;

        foundKernels = mTllmGenFMHARunner->isSupported(tllmRunnerParams);
    }
    else
    {
        foundKernels = mFMHARunner->isFmhaSupported();
    }
    if (!foundKernels)
    {
        TLLM_LOG_WARNING("Fall back to unfused MHA for %s in sm_%d.", mFixedParams.convertToStrOutput().c_str(),
            tensorrt_llm::common::getSMVersion());
    }
    return foundKernels;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FmhaDispatcher::run(MHARunnerParams runnerParams)
{
    if (mUseTllmGen)
    {
        TLLM_LOG_DEBUG("Running TRTLLM-GEN context FMHA kernel.");
        TLLM_CHECK_WITH_INFO(mTllmGenFMHARunner.get(), "mTllmGenFMHARunner not initialized.");
        // Convert from MHAFixedParams + MHARunnerParams to TllmGenFmhaRunnerParams
        void const* kvPoolPtr = nullptr;
        void const* kvSfPoolPtr = nullptr;
        void const* kvPageIdxPtr = nullptr;
        auto qkvLayout = kernels::QkvLayout::PackedQkv;
        int32_t maxBlocksPerSeq = 0;
        int32_t numTokensPerBlock = 0;
        if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_PAGED_KV)
        {
            qkvLayout = kernels::QkvLayout::PagedKv;
            auto pagedKvCache = runnerParams.pagedKvCache.copyKVBlockArrayForContextFMHA();
            kvPoolPtr = pagedKvCache.mPrimaryPoolPtr;
            kvSfPoolPtr = runnerParams.pagedKvSfCache.mPrimaryPoolPtr;
            kvPageIdxPtr = reinterpret_cast<int const*>(pagedKvCache.data);
            maxBlocksPerSeq = pagedKvCache.mMaxBlocksPerSeq;
            numTokensPerBlock = pagedKvCache.mTokensPerBlock;
        }
        else if (mFixedParams.attentionInputLayout == AttentionInputLayout::SEPARATE_Q_K_V)
        {
            qkvLayout = kernels::QkvLayout::SeparateQkv;
        }

        TllmGenFmhaRunnerParams tllmRunnerParams;
        memset(&tllmRunnerParams, 0, sizeof(tllmRunnerParams));

        // Parameters to select kernels.
        tllmRunnerParams.mQkvLayout = qkvLayout;
        tllmRunnerParams.setAttentionMaskType(static_cast<std::int8_t>(mFixedParams.attentionMaskType));
        tllmRunnerParams.mKernelType = FmhaKernelType::Context;
        // Always use persistent scheduler for better performance.
        tllmRunnerParams.mTileScheduler = TileScheduler::Persistent;
        tllmRunnerParams.mMultiCtasKvMode = false;

        tllmRunnerParams.qPtr = runnerParams.qPtr;
        tllmRunnerParams.kPtr = runnerParams.kPtr;
        tllmRunnerParams.vPtr = runnerParams.vPtr;
        tllmRunnerParams.kvPtr = kvPoolPtr;
        tllmRunnerParams.kvSfPtr = kvSfPoolPtr;
        tllmRunnerParams.qkvPtr = runnerParams.qkvPtr;
        tllmRunnerParams.attentionSinksPtr = runnerParams.attentionSinksPtr;
        tllmRunnerParams.cumSeqLensQPtr = reinterpret_cast<int const*>(runnerParams.cuQSeqLenPtr);
        tllmRunnerParams.cumSeqLensKvPtr = reinterpret_cast<int const*>(runnerParams.cuKvSeqLenPtr);
        tllmRunnerParams.outputScalePtr = reinterpret_cast<float const*>(runnerParams.scaleBmm2Ptr);
        // TRTLLM-GEN kernels always use the Log2 scale
        tllmRunnerParams.scaleSoftmaxLog2Ptr
            = reinterpret_cast<float const*>(runnerParams.scaleBmm1Ptr + kIdxScaleSoftmaxLog2Ptr);
        tllmRunnerParams.kvPageIdxPtr = reinterpret_cast<int const*>(kvPageIdxPtr);
        tllmRunnerParams.oSfScalePtr = runnerParams.oSfScalePtr;
        tllmRunnerParams.oPtr = runnerParams.outputPtr;
        tllmRunnerParams.oSfPtr = runnerParams.outputSfPtr;
        // The sequence lengths for K/V.
        tllmRunnerParams.seqLensKvPtr = reinterpret_cast<int const*>(runnerParams.kvSeqLenPtr);
        // Assume same headDim for Qk and V here.
        tllmRunnerParams.mHeadDimQk = mFixedParams.headSize;
        tllmRunnerParams.mHeadDimV = mFixedParams.headSizeV;
        tllmRunnerParams.mHeadDimQkNope = mFixedParams.headSizeQkNope;
        tllmRunnerParams.mNumHeadsQ = mFixedParams.numQHeads;
        tllmRunnerParams.mNumHeadsKv = mFixedParams.numKvHeads;
        tllmRunnerParams.mNumHeadsQPerKv = tllmRunnerParams.mNumHeadsQ / tllmRunnerParams.mNumHeadsKv;
        tllmRunnerParams.mBatchSize = runnerParams.b;
        // It is used to construct contiguous kv cache TMA descriptors.
        tllmRunnerParams.mMaxSeqLenCacheKv = runnerParams.slidingWindowSize;
        tllmRunnerParams.mMaxSeqLenQ = runnerParams.qSeqLen;
        tllmRunnerParams.mMaxSeqLenKv = runnerParams.kvSeqLen;
        tllmRunnerParams.mAttentionWindowSize = runnerParams.slidingWindowSize;
        tllmRunnerParams.mChunkedAttentionSize = runnerParams.chunkedAttentionSize;
        tllmRunnerParams.mSumOfSeqLensQ = runnerParams.totalQSeqLen;
        tllmRunnerParams.mSumOfSeqLensKv = runnerParams.totalKvSeqLen;
        tllmRunnerParams.mMaxNumPagesPerSeqKv = maxBlocksPerSeq;
        tllmRunnerParams.mNumTokensPerPage = numTokensPerBlock;
        tllmRunnerParams.mScaleQ = mFixedParams.qScaling;
        // Set it to INT_MAX as the kv cache pageOffsets will ensure that there is no out-of-bounds access.
        tllmRunnerParams.mNumPagesInMemPool = INT_MAX;
        tllmRunnerParams.mSfStartTokenIdx = 0;
        // For mla chunked prefill
        tllmRunnerParams.softmaxStatsPtr = reinterpret_cast<float2*>(runnerParams.softmaxStatsPtr);
        tllmRunnerParams.stream = runnerParams.stream;

        mTllmGenFMHARunner->run(tllmRunnerParams);
    }
    else
    {
        mFMHARunner->run(runnerParams);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace tensorrt_llm::kernels

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/model.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <memory>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::batch_manager
{
enum class TrtGptModelType
{
    InflightBatching,
    InflightFusedBatching
};

class LlmRequest;

namespace kv_cache_manager
{
class BaseKVCacheManager;
} // namespace kv_cache_manager

class TrtGptModel : public executor::Model
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    TrtGptModel(runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        executor::ExecutorConfig const& executorConfig)
        : mMaxBatchSize{executorConfig.getMaxBatchSize().value_or(modelConfig.getMaxBatchSize())}
        , mMaxBeamWidth{executorConfig.getMaxBeamWidth()}
        , mMaxSequenceLen{modelConfig.getMaxSequenceLen()}
        , mMaxDraftLen{modelConfig.getMaxDecodingDraftTokens()}
        , mVocabSizePadded{modelConfig.getVocabSizePadded(worldConfig.getSize())}
        , mNormalizeLogProbs{executorConfig.getNormalizeLogProbs()}
        , mEnableTrtOverlap{executorConfig.getEnableTrtOverlap()}
        , mCudaGraphMode{executorConfig.getExtendedRuntimePerfKnobConfig().getCudaGraphMode()}
    {
        TLLM_CHECK_WITH_INFO(mMaxBeamWidth <= modelConfig.getMaxBeamWidth(),
            "Runtime configured max beam width (%d) must not exceed engine max beam width (%d)", mMaxBeamWidth,
            modelConfig.getMaxBeamWidth());
        TLLM_CHECK_WITH_INFO(mMaxBatchSize <= modelConfig.getMaxBatchSize(),
            "Runtime configured max batch size (%d) must not exceed engine max batch size (%d)", mMaxBatchSize,
            modelConfig.getMaxBatchSize());
        if (executorConfig.getEnableTrtOverlap())
        {
            if (mMaxBeamWidth > 1)
            {
                mEnableTrtOverlap = false;
                TLLM_LOG_WARNING(
                    "TRT overlap is not supported with beam search (maxBeamWidth is set to %d) and will be disabled.",
                    mMaxBeamWidth);
            }
            if (!modelConfig.getSpeculativeDecodingMode().isNone())
            {
                mEnableTrtOverlap = false;
                TLLM_LOG_WARNING("TRT overlap is not supported with speculative decoding and will be disabled.");
            }
        }

        mMaxAttentionWindow = 0;
        if (executorConfig.getKvCacheConfig().getMaxAttentionWindowVec().has_value())
        {
            bool warning = false;
            auto const& maxAttentionWindowVec = executorConfig.getKvCacheConfig().getMaxAttentionWindowVec();
            for (int maxAttenWin : maxAttentionWindowVec.value())
            {
                mMaxAttentionWindowVec.push_back(std::min(maxAttenWin, mMaxSequenceLen));
                mMaxAttentionWindow = std::max(mMaxAttentionWindow, mMaxAttentionWindowVec.back());
                if (maxAttenWin > mMaxSequenceLen)
                {
                    warning = true;
                }
                TLLM_CHECK_WITH_INFO(mMaxAttentionWindowVec.back() > 0,
                    "Attention window sizes (elements in maxAttentionWindowVec) must be > 0");
            }
            if (warning)
            {
                TLLM_LOG_WARNING(
                    "The value of maxAttentionWindow cannot exceed mMaxSequenceLen. "
                    "Therefore, it has been adjusted to match the value of mMaxSequenceLen.");
            }
        }
        else
        {
            mMaxAttentionWindowVec.push_back(mMaxSequenceLen);
            mMaxAttentionWindow = mMaxSequenceLen;
        }

        mSinkTokenLen = executorConfig.getKvCacheConfig().getSinkTokenLength().has_value()
            ? executorConfig.getKvCacheConfig().getSinkTokenLength().value()
            : 0;

        mMaxNumSequences = mMaxBatchSize * worldConfig.getPipelineParallelism();

        auto const numTotalAttenLayers = modelConfig.getNbAttentionLayers();
        auto const numRepeatsAttenWindow = numTotalAttenLayers / mMaxAttentionWindowVec.size();
        auto const numRemainsAttenWindow = numTotalAttenLayers % mMaxAttentionWindowVec.size();
        std::string attenWindowRemainInfo = numRemainsAttenWindow > 0
            ? " + " + tc::arr2str(mMaxAttentionWindowVec.data(), numRemainsAttenWindow)
            : "";

        TLLM_LOG_INFO("TRTGptModel maxNumSequences: %d", mMaxNumSequences);
        TLLM_LOG_INFO("TRTGptModel maxBatchSize: %d", mMaxBatchSize);
        TLLM_LOG_INFO("TRTGptModel maxBeamWidth: %d", mMaxBeamWidth);
        TLLM_LOG_INFO("TRTGptModel maxSequenceLen: %d", mMaxSequenceLen);
        TLLM_LOG_INFO("TRTGptModel maxDraftLen: %d", mMaxDraftLen);
        TLLM_LOG_INFO("TRTGptModel mMaxAttentionWindowSize: %s * %d%s", tc::vec2str(mMaxAttentionWindowVec).c_str(),
            numRepeatsAttenWindow, attenWindowRemainInfo.c_str());
        TLLM_LOG_INFO("TRTGptModel enableTrtOverlap: %d", mEnableTrtOverlap);
        TLLM_LOG_INFO("TRTGptModel normalizeLogProbs: %d", mNormalizeLogProbs);

        mMaxNumTokens = modelConfig.getMaxNumTokens();
        if (executorConfig.getMaxNumTokens().has_value() && mMaxNumTokens)
        {
            if (executorConfig.getMaxNumTokens().value() > mMaxNumTokens.value())
            {
                TLLM_LOG_WARNING(
                    "Runtime configured max num tokens (%d) is larger than model max num tokens (%d) and will be "
                    "ignored.",
                    executorConfig.getMaxNumTokens().value(), mMaxNumTokens.value());
            }
            else
            {
                mMaxNumTokens = executorConfig.getMaxNumTokens();
            }
        }
        if (mMaxNumTokens)
        {
            TLLM_LOG_INFO("TRTGptModel maxNumTokens: %d", mMaxNumTokens.value());
        }

        if (executorConfig.getEnableChunkedContext())
        {
            mMaxInputLen = mMaxSequenceLen - 1;
            TLLM_LOG_INFO(
                "TRTGptModel maxInputLen: %d  = maxSequenceLen - 1 since chunked context is enabled", mMaxInputLen);
            TLLM_LOG_INFO(
                "TRTGptModel If model type is encoder, maxInputLen would be reset in trtEncoderModel to maxInputLen: "
                "%d = maxSequenceLen.",
                mMaxSequenceLen);
        }
        else if (modelConfig.getContextFMHA() && modelConfig.usePackedInput())
        {
            TLLM_CHECK_WITH_INFO(
                mMaxNumTokens, "Max number of tokens has to be set for context FMHA and usePackedInput case.");
            mMaxInputLen = std::min(mMaxSequenceLen - 1, mMaxNumTokens.value());
            TLLM_LOG_INFO(
                "TRTGptModel maxInputLen: %d = min(maxSequenceLen - 1, maxNumTokens) since context FMHA "
                "and usePackedInput are enabled",
                mMaxInputLen);
            TLLM_LOG_INFO(
                "TRTGptModel If model type is encoder, maxInputLen would be reset in trtEncoderModel to maxInputLen: "
                "min(maxSequenceLen, maxNumTokens).");
        }
        else
        {
            mMaxInputLen = modelConfig.getMaxInputLen();
            TLLM_LOG_INFO("TRTGptModel maxInputLen: %d = max_input_len (in trtllm-build args)", mMaxInputLen);
        }

        // TODO: remove this when XQA JIT can be enabled for fp8 RNN models
        if ((modelConfig.getQuantMode().hasFp8Qdq() || modelConfig.getQuantMode().hasFp8RowWise())
            && modelConfig.isRnnBased())
        {
#if defined(_WIN32)
            if (getenv("TRTLLM_ENABLE_XQA_JIT") == nullptr)
                _putenv_s("TRTLLM_ENABLE_XQA_JIT", "0");
#else
            setenv("TRTLLM_ENABLE_XQA_JIT", "0", 0);
#endif //_WIN32
        }

        using tensorrt_llm::common::stl_utils::toString;

        TLLM_LOG_INFO("Capacity Scheduler Policy: %s",
            toString(executorConfig.getSchedulerConfig().getCapacitySchedulerPolicy()).c_str());
        TLLM_LOG_INFO("Context Chunking Scheduler Policy: %s",
            toString(executorConfig.getSchedulerConfig().getContextChunkingPolicy()).c_str());
    }

    [[nodiscard]] std::optional<SizeType32> getMaxNumTokens() const
    {
        return mMaxNumTokens;
    }

    [[nodiscard]] SizeType32 getMaxNumSequences() const override
    {
        return mMaxNumSequences;
    }

    [[nodiscard]] SizeType32 getMaxBatchSize() const
    {
        return mMaxBatchSize;
    }

    [[nodiscard]] SizeType32 getMaxInputLen() const override
    {
        return mMaxInputLen;
    }

    [[nodiscard]] SizeType32 getHiddenSize() const override
    {
        return getModelConfig().getHiddenSize();
    };

    [[nodiscard]] SizeType32 getMaxSequenceLen() const override
    {
        return mMaxSequenceLen;
    }

    [[nodiscard]] virtual TrtGptModelType getModelType() const = 0;

    [[nodiscard]] SizeType32 getVocabSizePadded() const override
    {
        return mVocabSizePadded;
    }

    [[nodiscard]] SizeType32 getMaxDraftLen() const override
    {
        return mMaxDraftLen;
    }

    [[nodiscard]] SizeType32 getOperatingBeamWidth() const override
    {
        return mMaxBeamWidth;
    }

    [[nodiscard]] bool hasSpeculativeDecodingFastLogits() const noexcept override
    {
        return false;
    }

    [[nodiscard]] bool hasGuidedDecoder() const noexcept override
    {
        return false;
    }

    virtual void setLayerProfiler() = 0;
    [[nodiscard]] virtual std::string getLayerProfileInfo() const = 0;

    [[nodiscard]] bool hasKVCacheManager() const
    {
        return getKVCacheManager() != nullptr;
    }

protected:
    [[nodiscard]] SizeType32 getMaxBeamWidth() const
    {
        return mMaxBeamWidth;
    }

    [[nodiscard]] std::vector<SizeType32> getMaxAttentionWindowVec() const
    {
        return mMaxAttentionWindowVec;
    }

    [[nodiscard]] SizeType32 getMaxAttentionWindow() const
    {
        return mMaxAttentionWindow;
    }

    [[nodiscard]] SizeType32 getSinkTokenLen() const
    {
        return mSinkTokenLen;
    }

    [[nodiscard]] bool isNormalizeLogProbs() const
    {
        return mNormalizeLogProbs;
    }

    [[nodiscard]] bool isTrtOverlap() const
    {
        return mEnableTrtOverlap;
    }

    [[nodiscard]] bool isCudaGraphMode() const
    {
        return mCudaGraphMode;
    }

    void setMaxAttentionWindowVec(std::vector<SizeType32> const& maxAttentionWindowVec)
    {
        TLLM_CHECK_WITH_INFO(maxAttentionWindowVec.size() == mMaxAttentionWindowVec.size(),
            "The size of maxAttentionWindowVec must match the size of mMaxAttentionWindowVec");
        mMaxAttentionWindowVec = maxAttentionWindowVec;
        mMaxAttentionWindow = *std::max_element(std::begin(mMaxAttentionWindowVec), std::end(mMaxAttentionWindowVec));
    }

    void setMaxSequenceLen(SizeType32 maxSequenceLen)
    {
        mMaxSequenceLen = maxSequenceLen;
    }

    void setMaxInputLen(SizeType32 maxInputLen)
    {
        mMaxInputLen = maxInputLen;
    }

    [[nodiscard]] std::shared_ptr<kv_cache_manager::BaseKVCacheManager> getKVCacheManager() override = 0;
    [[nodiscard]] std::shared_ptr<kv_cache_manager::BaseKVCacheManager const> getKVCacheManager() const override = 0;

    [[nodiscard]] virtual std::shared_ptr<BasePeftCacheManager> getPeftCacheManager() = 0;
    [[nodiscard]] virtual std::shared_ptr<BasePeftCacheManager const> getPeftCacheManager() const = 0;

private:
    std::optional<SizeType32> mMaxNumTokens;
    SizeType32 mMaxNumSequences;
    SizeType32 mMaxBatchSize;
    SizeType32 mMaxBeamWidth;
    SizeType32 mMaxInputLen;
    SizeType32 mMaxSequenceLen;
    SizeType32 mMaxDraftLen;

    SizeType32 mVocabSizePadded;
    std::vector<SizeType32> mMaxAttentionWindowVec;
    SizeType32 mMaxAttentionWindow;
    SizeType32 mSinkTokenLen;

    bool mNormalizeLogProbs;
    bool mEnableTrtOverlap;
    bool mCudaGraphMode;
};

} // namespace tensorrt_llm::batch_manager
